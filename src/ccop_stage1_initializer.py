"""Independent physics-subspace Stage-I initializer for the CCOP route."""

from __future__ import annotations

import copy
import time

import numpy as np

from .estimators import initialize_from_hankel
from .robust_jnpp import robust_jnpp_basin_recovery


CCOP_STAGE1_PRESETS = {
    "fast": {
        "initializer": "evs_sufficient_delay_subspace",
        "status": "diagnostic_only_after_development_introduced_outliers",
        "ccop_stage1_evs_subspace_rtol": 1.0e-12,
    },
    "balanced": {
        "initializer": "evs_sufficient_joint_position_geometry",
        "status": "validation100_passed_heldout_pending",
        "ccop_stage1_evs_subspace_rtol": 1.0e-12,
        "ccop_stage1_refresh_jones_anchor": True,
        "ccop_stage1_joint_geometry": {
            "num_starts": 4,
            "max_iter": 30,
            "use_leave_one_out": False,
            "enable_z_starts": True,
            "z_start_grid_size": 7,
            "use_coarse_grid": False,
        },
    },
    "accuracy": {
        "initializer": "evs_sufficient_joint_position_geometry",
        "status": "development20_no_gain_over_balanced",
        "ccop_stage1_evs_subspace_rtol": 1.0e-12,
        "ccop_stage1_refresh_jones_anchor": True,
        "ccop_stage1_joint_geometry": {
            "num_starts": 7,
            "max_iter": 30,
            "use_leave_one_out": False,
            "enable_z_starts": True,
            "z_start_grid_size": 7,
            "use_coarse_grid": False,
        },
    },
}


def apply_ccop_stage1_preset(config: dict, name: str = "balanced") -> dict:
    """Return a copy of ``config`` with one independent CCOP Stage-I preset."""
    if name not in CCOP_STAGE1_PRESETS:
        raise ValueError(f"unknown CCOP Stage-I preset {name!r}")
    resolved = copy.deepcopy(config)
    selected = copy.deepcopy(CCOP_STAGE1_PRESETS[name])
    resolved["ccop_stage1_preset"] = str(name)
    resolved["ccop_stage1_preset_status"] = selected.pop("status")
    resolved["ccop_stage1_initializer"] = selected.pop("initializer")
    resolved.update(selected)
    return resolved


def known_evs_union_basis(
    scene: dict,
    *,
    relative_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Return an orthonormal basis for the known union of EVS path subspaces.

    For physical panel ``k``, every admissible EVS factor is in
    ``range(diag(mask) kron(v_B[k], Theta[k]))``.  The union basis is independent
    of UE position, clock, path gain, and polarization coefficients.
    """
    i_dim = int(scene["I"])
    mask = np.asarray(
        scene.get("evs_observation_mask", np.ones(i_dim, dtype=bool)),
        dtype=float,
    ).reshape(-1)
    if mask.size != i_dim:
        raise ValueError("EVS observation mask has incompatible size")

    blocks = []
    for path in range(int(scene["K"])):
        v_b = np.asarray(scene["v_B"][path], dtype=complex)
        theta = np.asarray(scene["Theta"][path], dtype=complex)
        if theta.shape != (6, 2):
            raise ValueError("each Maxwell matrix must have shape 6 x 2")
        block = np.column_stack(
            [np.kron(v_b, theta[:, component]) for component in range(2)]
        )
        blocks.append(mask[:, None] * block)

    union = np.column_stack(blocks)
    left, singular_values, _ = np.linalg.svd(union, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("known EVS union subspace is empty")
    threshold = float(relative_tolerance) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank < 1:
        raise ValueError("known EVS union subspace is numerically rank deficient")
    basis = left[:, :rank]
    diagnostics = {
        "stage1_evs_union_rank": rank,
        "stage1_evs_union_nominal_columns": int(union.shape[1]),
        "stage1_evs_union_singular_values": singular_values,
        "stage1_evs_union_relative_tolerance": float(relative_tolerance),
        "stage1_evs_union_orthogonality_error": float(
            np.linalg.norm(basis.conj().T @ basis - np.eye(rank))
        ),
    }
    return basis, diagnostics


def compress_hankel_evs_observation(
    z_tensor: np.ndarray,
    scene: dict,
    *,
    relative_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Project the EVS mode of a Hankel tensor onto its known signal subspace."""
    z_tensor = np.asarray(z_tensor, dtype=complex)
    expected = (scene["I"], scene["P"], scene["L"], scene["T"])
    if z_tensor.shape != expected:
        raise ValueError(f"expected Hankel tensor shape {expected}, got {z_tensor.shape}")

    basis, diagnostics = known_evs_union_basis(
        scene,
        relative_tolerance=relative_tolerance,
    )
    # Contract the EVS mode as GEMMs rather than a 4-D einsum: the EVS axis
    # leads a C-contiguous tensor, so the reshape is a view and BLAS gets an
    # (r x I) by (I x m) product per block.
    #
    # The pass is blocked because the orthogonal energy has to come from an
    # explicit residual.  Reading it off the isometry identity
    # ||Z||^2 = ||B^H Z||^2 + ||Z - P Z||^2 would be cheaper still, but it
    # cancels catastrophically exactly where the diagnostic matters most: when
    # the subspace is matched the true fraction underflows to ~1e-31, while a
    # difference of two near-equal numbers can only resolve ~1e-16.  Blocking
    # keeps the honest difference while replacing the two full-size temporaries
    # (the reconstruction ``P Z`` and ``Z - P Z``) with one small buffer.
    i_dim = int(z_tensor.shape[0])
    r_dim = int(basis.shape[1])
    basis_h = basis.conj().T
    z_flat = z_tensor.reshape(i_dim, -1)
    num_columns = int(z_flat.shape[1])
    compressed_flat = np.empty((r_dim, num_columns), dtype=complex)

    orthogonal_sq = 0.0
    block = 8192
    for start in range(0, num_columns, block):
        stop = min(start + block, num_columns)
        z_block = z_flat[:, start:stop]
        c_block = basis_h @ z_block
        compressed_flat[:, start:stop] = c_block
        orthogonal_sq += float(
            np.linalg.norm(z_block - basis @ c_block) ** 2
        )
    compressed = compressed_flat.reshape((r_dim,) + z_tensor.shape[1:])

    norm_sq = float(np.linalg.norm(z_tensor) ** 2)
    denominator = max(norm_sq, np.finfo(float).tiny)
    diagnostics.update(
        {
            "stage1_evs_retained_energy_fraction": float(
                np.linalg.norm(compressed) ** 2 / denominator
            ),
            "stage1_evs_orthogonal_energy_fraction": float(
                orthogonal_sq / denominator
            ),
            "stage1_evs_original_dimension": int(z_tensor.shape[0]),
            "stage1_evs_compressed_dimension": int(compressed.shape[0]),
        }
    )
    return compressed, diagnostics


def initialize_ccop_stage1(
    z_tensor: np.ndarray,
    scene: dict,
    config: dict,
) -> dict:
    """Run Stage-I with sufficient EVS compression only in delay estimation."""
    start = time.perf_counter()
    tolerance = float(config.get("ccop_stage1_evs_subspace_rtol", 1.0e-12))
    delay_z, diagnostics = compress_hankel_evs_observation(
        z_tensor,
        scene,
        relative_tolerance=tolerance,
    )
    projection_time = time.perf_counter() - start
    estimate = initialize_from_hankel(
        z_tensor,
        scene,
        config,
        delay_z_tensor=delay_z,
    )
    estimate.update(diagnostics)
    estimate.update(
        {
            "stage1_initializer_route": "evs_sufficient_delay_subspace",
            "stage1_time_evs_sufficient_projection": float(projection_time),
        }
    )
    return estimate


def initialize_ccop_stage1_joint_geometry(
    z_tensor: np.ndarray,
    scene: dict,
    config: dict,
    *,
    y_raw: np.ndarray | None = None,
) -> dict:
    """Add an all-panel common-position near-field projection to Stage-I.

    This balanced variant first performs sufficient EVS delay compression and
    then profiles the per-panel complex RIS scales while constraining every
    compressed response to one common global UE position.  Delay-derived clock
    replicas are used only to select among the resulting local geometry
    candidates.  The frozen Stage-II route and selector are not invoked.
    """
    estimate = initialize_ccop_stage1(z_tensor, scene, config)
    refined = refine_ccop_stage1_joint_geometry(estimate, scene, config)
    if bool(config.get("ccop_stage1_refresh_jones_anchor", False)):
        if y_raw is None:
            raise ValueError(
                "ccop_stage1_refresh_jones_anchor requires the raw noisy tensor"
            )
        refined = refresh_ccop_stage1_jones_anchor(y_raw, refined, scene, config)
    return refined


def refresh_ccop_stage1_jones_anchor(
    y_raw: np.ndarray,
    estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Refresh Jones directions after common-position geometry initialization.

    At the fixed Stage-I common-position candidate, this performs one
    unregularized common-clock profile and one six-dimensional Jones least-
    squares solve.  Only Jones direction anchors and the clock initializer are
    refreshed; position, delay poles, RIS factors, and physics are unchanged.
    """
    # Keep the base initializer independent from CCOP unless this optional
    # raw-domain ablation is explicitly enabled.
    from .ccop_jvp import CommonClockJonesProfiler

    local_config = copy.deepcopy(config)
    local_config["global_vp"] = dict(local_config.get("global_vp", {}))
    local_config["global_vp"].update(
        {
            "mode": "jones_free",
            "jones_diagonal_loading": 0.0,
            "use_delay_prior": False,
            "use_weight": False,
        }
    )
    position = np.asarray(estimate["p_u"], dtype=float).reshape(3)
    start = time.perf_counter()
    profiler = CommonClockJonesProfiler(y_raw, estimate, scene, local_config)
    point = profiler.profile_clock(position)
    elapsed = time.perf_counter() - start

    coefficients = np.asarray(point["x_hat"], dtype=complex).reshape(-1)
    expected = 2 * int(scene["K"])
    if coefficients.size != expected:
        raise ValueError(
            f"Jones anchor refresh expected {expected} coefficients, "
            f"got {coefficients.size}"
        )
    a_evs = np.empty((int(scene["I"]), int(scene["K"])), dtype=complex)
    for path in range(int(scene["K"])):
        theta = np.asarray(scene["Theta"][path], dtype=complex)
        v_b = np.asarray(scene["v_B"][path], dtype=complex)
        basis = np.column_stack(
            [np.kron(v_b, theta[:, component]) for component in range(2)]
        )
        a_evs[:, path] = basis @ coefficients[2 * path : 2 * path + 2]

    refreshed = copy.deepcopy(estimate)
    refreshed["A"] = a_evs
    refreshed["a_evs"] = a_evs.copy()
    refreshed["delta_t"] = float(point["delta_t"])
    refreshed["Delta_t_init"] = float(point["delta_t"])
    refreshed.update(
        {
            "stage1_jones_anchor_refresh": "fixed_position_free_jones_ccop",
            "stage1_jones_anchor_refresh_runtime_s": float(elapsed),
            "stage1_jones_anchor_refresh_clock_certified": bool(
                point.get("clock_certified", False)
            ),
            "stage1_jones_anchor_refresh_raw_objective": float(
                point["raw_objective"]
            ),
            "stage1_jones_anchor_refresh_coefficients": coefficients.copy(),
        }
    )
    return refreshed


def refine_ccop_stage1_joint_geometry(
    estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Apply the common-position projection to an existing CCOP Stage-I result."""
    local_config = copy.deepcopy(config)
    options = dict(config.get("ccop_stage1_joint_geometry", {}))
    local_config.update(
        {
            "jnpp_num_starts": int(options.get("num_starts", 7)),
            "jnpp_max_iter": int(options.get("max_iter", 30)),
            "jnpp_use_leave_one_out": bool(options.get("use_leave_one_out", False)),
            "jnpp_enable_z_starts": bool(options.get("enable_z_starts", True)),
            "jnpp_z_start_grid_size": int(options.get("z_start_grid_size", 7)),
            "jnpp_use_coarse_grid": bool(options.get("use_coarse_grid", False)),
        }
    )
    start = time.perf_counter()
    refined, diagnostics = robust_jnpp_basin_recovery(
        estimate,
        scene,
        local_config,
    )
    elapsed = time.perf_counter() - start
    refined.update(
        {
            "stage1_initializer_route": "evs_sufficient_joint_position_geometry",
            "stage1_time_joint_position_geometry": float(elapsed),
            "stage1_joint_geometry_num_starts": int(
                diagnostics.get("jnpp_num_starts", 0)
            ),
            "stage1_joint_geometry_num_candidates": int(
                diagnostics.get("jnpp_num_candidates", 0)
            ),
            "stage1_joint_geometry_objective_evaluations": int(
                diagnostics.get("jnpp_objective_eval_count", 0)
            ),
            "stage1_joint_geometry_gradient_evaluations": int(
                diagnostics.get("jnpp_gradient_eval_count", 0)
            ),
            "stage1_joint_geometry_clock_std_ns": float(
                diagnostics.get("jnpp_best_clock_std_ns", np.nan)
            ),
            "stage1_joint_geometry_objective": float(
                diagnostics.get("jnpp_best_objective", np.nan)
            ),
            "stage1_joint_geometry_position": np.asarray(
                diagnostics.get("jnpp_best_position", refined["p_u"]), dtype=float
            ).copy(),
            "stage1_joint_geometry_all_panels": not bool(
                local_config["jnpp_use_leave_one_out"]
            ),
        }
    )
    return refined
