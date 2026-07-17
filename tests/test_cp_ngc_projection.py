import numpy as np

from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import default_config
from oldcode.ccop_validation.cp_ngc import (
    cp_ngc_clock_vector,
    cp_ngc_geometry,
    cp_ngc_geometry_jacobian,
    cp_ngc_statistic,
    regularize_cp_ngc_covariance,
)
from oldcode.ccop_validation.cp_ngc_covariance import (
    apply_empirical_cp_ngc_calibration,
    fit_empirical_cp_ngc_calibration,
    linearized_stage1_covariance,
    ris_training_fold,
)
from oldcode.ccop_validation.ccop_recovery import (
    estimate_for_assignment,
    rdc_clock_interval,
    top_l_assignment_hypotheses,
)
from src.tensor_utils import hankelize_frequency


def _scene():
    config = default_config()
    config.update(
        {
            "seed": 8801,
            "K": 3,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 7,
            "P": 4,
            "T": 6,
            "p_u_true": np.array([1.25, 0.55, 0.75]),
            "ris_centers": np.array(
                [
                    [4.20, -2.20, 1.05],
                    [5.10, 2.10, 1.15],
                    [4.80, 0.00, 1.25],
                ]
            ),
        }
    )
    return generate_scene(config, np.random.default_rng(config["seed"]))


def _mixed_covariance(scene):
    dimension = 4 * int(scene["K"])
    standard_deviation = np.concatenate(
        [
            np.full(int(scene["K"]), 0.25e-9),
            np.tile(np.array([0.06, 0.008, 0.010]), int(scene["K"])),
        ]
    )
    indices = np.arange(dimension)
    correlation = 0.15 ** np.abs(indices[:, None] - indices[None, :])
    return np.diag(standard_deviation) @ correlation @ np.diag(standard_deviation)


def test_cp_ngc_clock_projection_whitening_and_joint_rank_three():
    scene = _scene()
    covariance = _mixed_covariance(scene)
    geometry = cp_ngc_geometry(scene["p_u_true"], scene)
    clock = cp_ngc_clock_vector(scene)
    perturbation = np.linspace(-0.4, 0.4, geometry.size) * np.sqrt(np.diag(covariance))
    z_hat = geometry + clock * float(scene["delta_t_true"]) + perturbation
    diagnostic = cp_ngc_statistic(z_hat, scene["p_u_true"], covariance, scene)

    projector = diagnostic["projector"]
    null_relative = np.linalg.norm(projector @ clock) / max(
        np.linalg.norm(projector) * np.linalg.norm(clock), 1.0e-300
    )
    assert null_relative < 1.0e-11
    whitened = diagnostic["whitened_projector"]
    np.testing.assert_allclose(whitened, whitened.T, rtol=0.0, atol=1.0e-13)
    assert np.linalg.norm(whitened @ whitened - whitened) < 1.0e-12
    assert diagnostic["projected_geometry_rank"] == 3
    assert diagnostic["full_3d_certificate"]
    assert diagnostic["uncertifiable_position_directions"].shape == (3, 0)
    assert np.min(diagnostic["cert_information_eigenvalues"]) > 0.0

    jacobian_delay = cp_ngc_geometry_jacobian(scene["p_u_true"], scene)[: scene["K"]]
    covariance_delay = covariance[: scene["K"], : scene["K"]]
    factor = np.linalg.cholesky(covariance_delay)
    whitened_clock = np.linalg.solve(factor, np.ones(int(scene["K"])))
    delay_projector = np.eye(int(scene["K"])) - np.outer(
        whitened_clock, whitened_clock
    ) / np.vdot(whitened_clock, whitened_clock).real
    projected_delay_jacobian = delay_projector @ np.linalg.solve(
        factor, jacobian_delay
    )
    assert np.linalg.matrix_rank(projected_delay_jacobian) <= 2


def test_covariance_regularization_is_positive_definite_in_mixed_units():
    scene = _scene()
    covariance = _mixed_covariance(scene)
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    correlation[0, 1] = correlation[1, 0] = 1.02
    indefinite = scales[:, None] * correlation * scales[None, :]
    regularized, diagnostics = regularize_cp_ngc_covariance(
        indefinite,
        shrinkage=0.02,
        eigenvalue_floor_relative=1.0e-6,
    )
    np.linalg.cholesky(regularized)
    assert diagnostics["positive_definite"]
    assert diagnostics["correlation_min_eigenvalue_after"] > 0.0
    assert diagnostics["correlation_condition_number_after"] < 1.0e7

    geometry = cp_ngc_geometry(scene["p_u_true"], scene)
    z_hat = geometry + cp_ngc_clock_vector(scene) * float(scene["delta_t_true"])
    result = cp_ngc_statistic(
        z_hat,
        scene["p_u_true"],
        indefinite,
        scene,
        covariance_regularization={
            "shrinkage": 0.02,
            "eigenvalue_floor_relative": 1.0e-6,
        },
    )
    assert result["covariance_regularization"]["positive_definite"]
    assert result["statistic"] < 1.0e-12


def test_rank_deficiency_reports_uncertifiable_position_direction():
    scene = _scene()
    k_paths = int(scene["K"])
    standard_deviation = np.concatenate(
        [np.full(k_paths, 0.2e-9), np.full(3 * k_paths, 1.0e10)]
    )
    covariance = np.diag(standard_deviation**2)
    geometry = cp_ngc_geometry(scene["p_u_true"], scene)
    z_hat = geometry + cp_ngc_clock_vector(scene) * float(scene["delta_t_true"])
    result = cp_ngc_statistic(
        z_hat,
        scene["p_u_true"],
        covariance,
        scene,
        rank_relative_tolerance=1.0e-6,
    )
    assert result["projected_geometry_rank"] <= 2
    assert not result["full_3d_certificate"]
    assert result["uncertifiable_position_directions"].shape[1] >= 1


def test_c1_linearized_covariance_has_joint_blocks_and_is_positive_definite():
    scene = _scene()
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    stage1 = {
        "A": components["a_EVS"].T.copy(),
        "C": components["c"].T.copy(),
        "poles": components["poles"].copy(),
        "ris_eta": np.column_stack(
            [components["ranges"], components["elevations"], components["azimuths"]]
        ),
        "beta_z": np.asarray(scene["beta_true"], dtype=complex).copy(),
        "Z_hat": hankelize_frequency(y_true, int(scene["P"])),
        "columns_are_panel_ordered": True,
    }
    covariance = linearized_stage1_covariance(
        y_true,
        stage1,
        scene,
        noise_variance=max(float(np.mean(np.abs(y_true) ** 2)) * 0.1, 1.0e-12),
    )
    dimension = 4 * int(scene["K"])
    assert covariance["covariance_z"].shape == (dimension, dimension)
    assert covariance["C_tau"].shape == (scene["K"], scene["K"])
    assert covariance["C_xi"].shape == (3 * scene["K"], 3 * scene["K"])
    assert covariance["C_tau_xi"].shape == (scene["K"], 3 * scene["K"])
    np.linalg.cholesky(covariance["covariance_z"])
    assert covariance["delay_uses_single_mother_factor"]
    assert covariance["ris_uses_compressed_exact_geometry"]


def test_c3_empirical_thresholds_use_only_correct_validation_records():
    records = [
        {"statistic": float(value), "correct_basin": True, "stratum": "high"}
        for value in range(1, 31)
    ]
    records += [
        {"statistic": 0.01, "correct_basin": False, "stratum": "high"},
        {"statistic": 1.0e6, "correct_basin": False, "stratum": "high"},
    ]
    calibration = fit_empirical_cp_ngc_calibration(
        records, minimum_stratum_size=20
    )
    thresholds = calibration["strata"]["high"]
    assert thresholds["green_max"] == 28.0
    assert thresholds["red_min"] == 30.0
    assert apply_empirical_cp_ngc_calibration(27.0, "high", calibration) == "green"
    assert apply_empirical_cp_ngc_calibration(29.0, "high", calibration) == "gray"
    assert apply_empirical_cp_ngc_calibration(30.0, "high", calibration) == "red"


def test_ris_training_fold_keeps_full_bandwidth_and_disjoint_blocks():
    scene = _scene()
    rng = np.random.default_rng(99)
    y = rng.standard_normal((scene["I"], scene["N"], scene["T"])) + 1j * rng.standard_normal(
        (scene["I"], scene["N"], scene["T"])
    )
    data = {
        "scene": scene,
        "Y_noisy": y,
        "Y_true": y,
        "noise_variance": 1.0,
    }
    config = default_config()
    fold_a, _ = ris_training_fold(data, config, np.arange(0, scene["T"], 2))
    fold_b, _ = ris_training_fold(data, config, np.arange(1, scene["T"], 2))
    assert fold_a["Y_noisy"].shape[1] == scene["N"]
    assert fold_b["Y_noisy"].shape[1] == scene["N"]
    assert np.intersect1d(fold_a["training_indices"], fold_b["training_indices"]).size == 0
    assert fold_a["scene"]["Omega"].shape[1] == fold_a["scene"]["T"]
    assert fold_b["scene"]["Omega"].shape[1] == fold_b["scene"]["T"]


def test_top_l_assignment_rebuilds_raw_columns_without_annealing_or_mixture():
    # Raw columns [10,20,30] are currently stored in physical panel order
    # according to panel_to_column [2,0,1] -> [30,10,20].
    physical = np.array([[30.0, 10.0, 20.0]])
    stage1 = {
        "A": physical.astype(complex),
        "B": physical.astype(complex),
        "Q": physical.astype(complex),
        "C": physical.astype(complex),
        "poles": np.array([30.0, 10.0, 20.0], dtype=complex),
        "beta_z": np.array([30.0, 10.0, 20.0], dtype=complex),
        "gamma": np.array([30.0, 10.0, 20.0]),
        "eta_pol": np.array([30.0, 10.0, 20.0]),
        "ris_eta": np.column_stack(
            [np.array([30.0, 10.0, 20.0]), np.zeros(3), np.zeros(3)]
        ),
        "assignment": [1, 2, 0],
        "panel_to_column_assignment": [2, 0, 1],
        "columns_are_panel_ordered": True,
        "all_assignment_scores": [
            {"assignment": [1, 2, 0], "score": 1.0},
            {"assignment": [0, 1, 2], "score": 1.1},
        ],
    }
    hypotheses = top_l_assignment_hypotheses(stage1, top_l=2)
    assert len(hypotheses) == 2
    identity = next(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis["column_to_panel"] == (0, 1, 2)
    )
    rebuilt = estimate_for_assignment(stage1, identity)
    np.testing.assert_array_equal(rebuilt["A"], np.array([[10.0, 20.0, 30.0]]))
    np.testing.assert_array_equal(rebuilt["poles"], np.array([10.0, 20.0, 30.0]))
    assert rebuilt["columns_are_panel_ordered"]


def test_rdc_only_generates_a_clock_interval_and_retains_full_fallback():
    scene = _scene()
    geometry = cp_ngc_geometry(scene["p_u_true"], scene)
    k_paths = int(scene["K"])
    tau = geometry[:k_paths] + float(scene["delta_t_true"])
    stage1 = {
        "poles": np.exp(-1j * 2.0 * np.pi * scene["delta_f"] * tau),
        "ris_eta": geometry[k_paths:].reshape(k_paths, 3),
    }
    config = default_config()
    result = rdc_clock_interval(stage1, scene, config)
    assert result["available"]
    assert not result["is_final_estimator"]
    assert result["full_interval_fallback_retained"]
    assert result["interval_s"][0] <= scene["delta_t_true"] <= result["interval_s"][1]
