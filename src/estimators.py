"""Readable proposed-method estimator for one small RIS-EVS-OFDM run."""

from __future__ import annotations

import itertools
import numpy as np

from .channel_model import channel_components, synthesize_raw_tensor
from .geometry import position_from_local_geometry
from .projections_delay import (
    bq_from_poles,
    delay_matrix_from_poles,
    estimate_poles_esprit_from_hankel,
    structured_delay_mother_pgd,
    tau_from_pole,
)
from .projections_evs import project_evs_factor
from .projections_ris import local_ris_search_config, project_ris_factor, scaled_residual
from .tensor_utils import dehankelize_frequency, reconstruct_z, z_design_column
from .utils import bounded_coordinate_search, check_finite, solve_lstsq, scipy_is_available


def _relative_change(new_value: np.ndarray, old_value: np.ndarray, eps: float) -> float:
    """Return a safe relative change between two arrays."""
    return float(np.linalg.norm(new_value - old_value) / (np.linalg.norm(old_value) + eps))


def _count_nonfinite(array: np.ndarray) -> int:
    """Count NaN or Inf entries in an array."""
    return int(array.size - np.count_nonzero(np.isfinite(array)))


def _raw_design_matrix_from_factors(
    a_mat: np.ndarray, d_mat: np.ndarray, c_mat: np.ndarray
) -> np.ndarray:
    """Build raw-domain dictionary, shape (I*N*T) x K."""
    i_dim, k_paths = a_mat.shape
    n_dim = d_mat.shape[0]
    t_dim = c_mat.shape[0]
    design = np.empty((i_dim * n_dim * t_dim, k_paths), dtype=complex)
    for k in range(k_paths):
        design[:, k] = (
            a_mat[:, k, None, None]
            * d_mat[None, :, k, None]
            * c_mat[None, None, :, k]
        ).reshape(-1)
    return design


def _estimate_weights_raw(y: np.ndarray, a_mat: np.ndarray, d_mat: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """Estimate complex path gains by raw-domain variable projection."""
    design = _raw_design_matrix_from_factors(a_mat, d_mat, c_mat)
    return solve_lstsq(design, y.reshape(-1), reg=1e-12)


def _estimate_weights_z(
    z_tensor: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
) -> np.ndarray:
    """Estimate complex CP weights in the Hankelized tensor domain."""
    k_paths = a_mat.shape[1]
    design = np.column_stack(
        [
            z_design_column(a_mat[:, k], b_mat[:, k], q_mat[:, k], c_mat[:, k])
            for k in range(k_paths)
        ]
    )
    return solve_lstsq(design, z_tensor.reshape(-1), reg=1e-12)


def _z_fit_error(
    z_tensor: np.ndarray,
    a_mat: np.ndarray,
    poles: np.ndarray,
    c_mat: np.ndarray,
    p_dim: int,
    l_dim: int,
) -> float:
    """Return normalized Z-domain LS fitting error for a set of delay poles."""
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    beta = _estimate_weights_z(z_tensor, a_mat, b_mat, q_mat, c_mat)
    z_hat = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)
    return float(np.linalg.norm(z_hat - z_tensor) ** 2 / (np.linalg.norm(z_tensor) ** 2 + 1e-12))


def _fit_z_model(
    z_tensor: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate Z-domain weights and return reconstruction plus squared residual."""
    beta = _estimate_weights_z(z_tensor, a_mat, b_mat, q_mat, c_mat)
    z_hat = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)
    residual_sse = float(np.linalg.norm(z_hat - z_tensor) ** 2)
    return beta, z_hat, residual_sse


def reconstruct_raw_from_structured_estimate(estimate: dict, scene: dict) -> np.ndarray:
    """Reconstruct raw Y from current structured CPD factors."""
    a_mat = estimate["A"]
    c_mat = estimate["C"]
    beta = estimate["beta_z"]
    d_mat = delay_matrix_from_poles(estimate["poles"], scene["N"])
    return _raw_design_matrix_from_factors(a_mat, d_mat, c_mat) @ beta.reshape(-1)


def reconstruct_raw_tensor_from_structured_estimate(estimate: dict, scene: dict) -> np.ndarray:
    """Reconstruct raw Y tensor from current structured CPD factors."""
    y_vec = reconstruct_raw_from_structured_estimate(estimate, scene)
    return y_vec.reshape(scene["I"], scene["N"], scene["T"])


def _rank_one_snapshot_initialization(
    z_tensor: np.ndarray, poles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize EVS and compressed RIS factors after delay ESPRIT."""
    n_dim = z_tensor.shape[1] + z_tensor.shape[2] - 1
    y_like = dehankelize_frequency(z_tensor, n_dim)
    i_dim, _, t_dim = y_like.shape
    k_paths = poles.size
    d_mat = delay_matrix_from_poles(poles, n_dim)
    y_freq = np.transpose(y_like, (1, 0, 2)).reshape(n_dim, i_dim * t_dim)
    snapshots = solve_lstsq(d_mat, y_freq, reg=1e-10)

    a_proxy = np.empty((i_dim, k_paths), dtype=complex)
    c_proxy = np.empty((t_dim, k_paths), dtype=complex)
    for k in range(k_paths):
        snapshot_matrix = snapshots[k, :].reshape(i_dim, t_dim)
        u, s, vh = np.linalg.svd(snapshot_matrix, full_matrices=False)
        a_proxy[:, k] = u[:, 0] * np.sqrt(s[0])
        c_proxy[:, k] = vh[0, :] * np.sqrt(s[0])
    return a_proxy, c_proxy


def _assignment_by_projection(
    a_proxy: np.ndarray,
    c_proxy: np.ndarray,
    poles_raw: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[list[int], list[dict], list[dict]]:
    """Align CP columns to RIS panels using projection and clock consistency."""
    k_paths = scene["K"]
    scores = np.zeros((k_paths, k_paths), dtype=float)
    evs_cache: list[list[dict]] = [[{} for _ in range(k_paths)] for _ in range(k_paths)]
    ris_cache: list[list[dict]] = [[{} for _ in range(k_paths)] for _ in range(k_paths)]
    eps = config["eps"]
    implied_clock_offsets = np.zeros((k_paths, k_paths), dtype=float)

    for col in range(k_paths):
        for ris in range(k_paths):
            evs_proj = project_evs_factor(
                a_proxy[:, col], scene["v_B"][ris], scene["Theta"][ris], eps
            )
            ris_search = local_ris_search_config(scene, config, ris)
            ris_search["projection_mode"] = "exact"
            ris_proj = project_ris_factor(
                c_proxy[:, col],
                scene["Omega"][ris],
                scene["a_RB"][ris],
                scene["ris_grid"],
                scene["wavelength"],
                ris_search,
                eps,
            )
            evs_cache[col][ris] = evs_proj
            ris_cache[col][ris] = ris_proj
            scores[col, ris] = evs_proj["residual"] + ris_proj["relative_residual"]
            tau_hat = tau_from_pole(poles_raw[col], scene["delta_f"])
            range_hat = ris_proj["eta_local"][0]
            implied_clock_offsets[col, ris] = (
                tau_hat - (range_hat + scene["d_RB"][ris]) / scene["c0"]
            )

    best_perm = None
    best_score = np.inf
    for perm in itertools.permutations(range(k_paths)):
        projection_score = sum(scores[col, ris] for col, ris in enumerate(perm))
        clock_offsets = np.array(
            [implied_clock_offsets[col, ris] for col, ris in enumerate(perm)]
        )
        clock_spread = np.std(clock_offsets) / config.get("assignment_clock_scale_s", 1e-9)
        lower_dt, upper_dt = config["delta_t_bounds"]
        bound_violation = np.mean(
            np.maximum(lower_dt - clock_offsets, 0.0)
            + np.maximum(clock_offsets - upper_dt, 0.0)
        ) / config.get("assignment_clock_scale_s", 1e-9)
        score = projection_score + config.get("assignment_clock_weight", 0.0) * (
            clock_spread + bound_violation
        )
        if score < best_score:
            best_score = score
            best_perm = list(perm)

    assert best_perm is not None, "failed to find a column association"
    evs_selected = [evs_cache[col][ris] for col, ris in enumerate(best_perm)]
    ris_selected = [ris_cache[col][ris] for col, ris in enumerate(best_perm)]
    return best_perm, evs_selected, ris_selected


def _mode4_assignment_from_proxy(
    c_proxy: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[list[int], np.ndarray]:
    """Return physical-panel column order from the exact compressed RIS score."""
    k_paths = scene["K"]
    costs = np.empty((k_paths, k_paths), dtype=float)
    for col in range(k_paths):
        for ris in range(k_paths):
            ris_search = local_ris_search_config(scene, config, ris)
            ris_search["projection_mode"] = "exact"
            projection = project_ris_factor(
                c_proxy[:, col],
                scene["Omega"][ris],
                scene["a_RB"][ris],
                scene["ris_grid"],
                scene["wavelength"],
                ris_search,
                config["eps"],
            )
            costs[col, ris] = projection["relative_residual"] ** 2

    best_order = None
    best_score = np.inf
    for order in itertools.permutations(range(k_paths)):
        score = sum(costs[order[ris], ris] for ris in range(k_paths))
        if score < best_score:
            best_score = score
            best_order = list(order)
    assert best_order is not None, "failed to solve RIS column assignment"
    return best_order, costs


def _apply_physical_order(
    order: list[int],
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
    poles: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    eta_pol: np.ndarray,
    ris_eta: np.ndarray,
) -> tuple:
    """Reorder all Stage-II columns so column k corresponds to physical RIS k."""
    order_array = np.asarray(order, dtype=int)
    return (
        a_mat[:, order_array],
        b_mat[:, order_array],
        q_mat[:, order_array],
        c_mat[:, order_array],
        poles[order_array],
        beta[order_array],
        gamma[order_array],
        eta_pol[order_array],
        ris_eta[order_array],
    )


def initialize_from_hankel(z_tensor: np.ndarray, scene: dict, config: dict) -> dict:
    """Stage 1: HOSVD/ESPRIT-style initialization from the Hankelized tensor."""
    assert z_tensor.shape == (scene["I"], scene["P"], scene["L"], scene["T"])
    poles_raw = estimate_poles_esprit_from_hankel(z_tensor, scene["K"])
    a_proxy, c_proxy = _rank_one_snapshot_initialization(z_tensor, poles_raw)
    assignment, evs_selected, ris_selected = _assignment_by_projection(
        a_proxy, c_proxy, poles_raw, scene, config
    )

    k_paths = scene["K"]
    poles = np.empty(k_paths, dtype=complex)
    a_mat = np.empty((scene["I"], k_paths), dtype=complex)
    c_mat = np.empty((scene["T"], k_paths), dtype=complex)
    ris_eta = np.empty((k_paths, 3), dtype=float)
    gamma = np.empty(k_paths, dtype=float)
    eta_pol = np.empty(k_paths, dtype=float)

    # Store columns in physical RIS-panel order.
    for col, ris in enumerate(assignment):
        poles[ris] = poles_raw[col]
        a_mat[:, ris] = evs_selected[col]["a"]
        c_mat[:, ris] = ris_selected[col]["c"]
        ris_eta[ris] = ris_selected[col]["eta_local"]
        gamma[ris] = evs_selected[col]["gamma"]
        eta_pol[ris] = evs_selected[col]["eta"]

    b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
    beta_z = _estimate_weights_z(z_tensor, a_mat, b_mat, q_mat, c_mat)
    z_hat = reconstruct_z(beta_z, a_mat, b_mat, q_mat, c_mat)
    initial_residual = float(np.linalg.norm(z_hat - z_tensor) / np.sqrt(z_tensor.size))

    return {
        "poles": poles,
        "A": a_mat,
        "B": b_mat,
        "Q": q_mat,
        "C": c_mat,
        "beta_z": beta_z,
        "gamma": gamma,
        "eta_pol": eta_pol,
        "ris_eta": ris_eta,
        "assignment": assignment,
        "initial_z_residual": initial_residual,
        "Z_hat": z_hat,
    }


def _update_a_from_z(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
) -> np.ndarray:
    """Least-squares EVS-mode update before Maxwell-Kronecker projection."""
    i_dim = z_tensor.shape[0]
    k_paths = beta.size
    design = np.empty((b_mat.shape[0] * q_mat.shape[0] * c_mat.shape[0], k_paths), dtype=complex)
    for k in range(k_paths):
        design[:, k] = (
            beta[k]
            * b_mat[:, k, None, None]
            * q_mat[None, :, k, None]
            * c_mat[None, None, :, k]
        ).reshape(-1)

    target = z_tensor.reshape(i_dim, -1).T
    solution = solve_lstsq(design, target, reg=1e-10)
    return solution.T


def _update_c_from_z(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
) -> np.ndarray:
    """Least-squares RIS-training-mode update before compressed projection."""
    t_dim = z_tensor.shape[3]
    k_paths = beta.size
    design = np.empty((a_mat.shape[0] * b_mat.shape[0] * q_mat.shape[0], k_paths), dtype=complex)
    for k in range(k_paths):
        design[:, k] = (
            beta[k]
            * a_mat[:, k, None, None]
            * b_mat[None, :, k, None]
            * q_mat[None, None, :, k]
        ).reshape(-1)

    target = np.moveaxis(z_tensor, 3, 0).reshape(t_dim, -1).T
    solution = solve_lstsq(design, target, reg=1e-10)
    return solution.T


def _update_delay_poles_from_z(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    a_mat: np.ndarray,
    c_mat: np.ndarray,
    poles_old: np.ndarray,
    ris_eta: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Structured mother-delay LS update with Hankel rank-one projection."""
    r_dim = max(scene["P"], scene["L"])
    update = structured_delay_mother_pgd(
        z_tensor=z_tensor,
        beta=beta,
        a_mat=a_mat,
        c_mat=c_mat,
        poles_old=poles_old,
        p_dim=scene["P"],
        l_dim=scene["L"],
        r_dim=r_dim,
        lambda_d=float(config.get("delay_lambda", 1.0e-2)),
        num_steps=int(config.get("delay_num_pgd_steps", 10)),
        step_scale=float(config.get("delay_step_scale", 0.8)),
        damping=float(config.get("delay_damping", 1.0)),
        eps=config["eps"],
        phase_refine_span=config.get("delay_refine_phase_span", 0.0),
        phase_refine_grid=config.get("delay_refine_phase_grid", 0),
    )
    poles = update["poles"].copy()
    update["geometry_correction_accepted"] = False

    rho_g = float(config.get("delay_geometry_rho", 0.0))
    if rho_g > 0.0:
        rho_g = float(np.clip(rho_g, 0.0, 1.0))
        tau_est = np.array([tau_from_pole(pole, scene["delta_f"]) for pole in poles])
        ranges = ris_eta[:, 0]
        weights = np.ones_like(tau_est)
        delta_t = np.sum(weights * (tau_est - (ranges + scene["d_RB"]) / scene["c0"])) / (
            np.sum(weights) + config["eps"]
        )
        tau_geo = (ranges + scene["d_RB"]) / scene["c0"] + delta_t
        tau_corrected = (1.0 - rho_g) * tau_est + rho_g * tau_geo
        corrected = np.exp(-1j * 2.0 * np.pi * scene["delta_f"] * tau_corrected)
        error_before = _z_fit_error(z_tensor, a_mat, poles, c_mat, scene["P"], scene["L"])
        error_after = _z_fit_error(
            z_tensor, a_mat, corrected, c_mat, scene["P"], scene["L"]
        )
        update["geometry_correction"] = {
            "rho": rho_g,
            "delta_t": float(delta_t),
            "z_error_before": float(error_before),
            "z_error_after": float(error_after),
        }
        if error_after <= error_before + float(config.get("stage2_accept_tol", 1e-9)):
            poles = corrected
            update["geometry_correction_accepted"] = True

    return poles, update


def structured_refinement(z_tensor: np.ndarray, scene: dict, config: dict, estimate: dict) -> tuple[dict, dict]:
    """Stage 2: HP-R1P-CPD-style structured refinement in the Z domain."""
    a_mat = estimate["A"].copy()
    c_mat = estimate["C"].copy()
    poles = estimate["poles"].copy()
    gamma = estimate["gamma"].copy()
    eta_pol = estimate["eta_pol"].copy()
    ris_eta = estimate["ris_eta"].copy()
    diagnostics = {
        "z_hat_history": [],
        "residuals_noisy_rmse": [],
        "updates": [],
    }
    b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
    beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
    safeguard = bool(config.get("stage2_global_safeguard", True))
    accept_tol = float(config.get("stage2_accept_tol", 1e-9))
    stop_tol = float(config.get("stage2_tol", 0.0))

    for _ in range(config["num_structured_iters"]):
        b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
        beta_old, z_hat_old, iter_start_sse = _fit_z_model(
            z_tensor, a_mat, b_mat, q_mat, c_mat
        )
        a_old = a_mat.copy()
        b_old = b_mat.copy()
        q_old = q_mat.copy()
        c_old = c_mat.copy()
        poles_old = poles.copy()
        gamma_old = gamma.copy()
        eta_pol_old = eta_pol.copy()
        ris_eta_old = ris_eta.copy()

        a_proxy = _update_a_from_z(z_tensor, beta_old, b_mat, q_mat, c_mat)
        evs_projection_details = []
        for k in range(scene["K"]):
            a_before = a_mat[:, k].copy()
            evs_proj = project_evs_factor(
                a_proxy[:, k], scene["v_B"][k], scene["Theta"][k], config["eps"]
            )
            a_mat[:, k] = evs_proj["a"]
            gamma[k] = evs_proj["gamma"]
            eta_pol[k] = evs_proj["eta"]
            evs_projection_details.append(
                {
                    "path": k,
                    "accepted": bool(np.all(np.isfinite(a_mat[:, k]))),
                    "relative_change": _relative_change(a_mat[:, k], a_before, config["eps"]),
                    "projection_residual": evs_proj["residual"],
                }
            )

        beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
        poles_candidate, delay_projection_details = _update_delay_poles_from_z(
            z_tensor, beta_z, a_mat, c_mat, poles, ris_eta, scene, config
        )
        b_candidate, q_candidate = bq_from_poles(poles_candidate, scene["P"], scene["L"])
        beta_candidate, z_hat_candidate, delay_sse_after = _fit_z_model(
            z_tensor, a_mat, b_candidate, q_candidate, c_mat
        )
        delay_accepted = delay_sse_after <= current_sse + accept_tol
        delay_projection_details.update(
            {
                "accepted": bool(delay_accepted),
                "global_sse_before": float(current_sse),
                "global_sse_after": float(delay_sse_after),
            }
        )
        if delay_accepted:
            poles = poles_candidate
            b_mat = b_candidate
            q_mat = q_candidate
            beta_z = beta_candidate
            z_hat = z_hat_candidate
            current_sse = delay_sse_after

        c_proxy = _update_c_from_z(z_tensor, beta_z, a_mat, b_mat, q_mat)
        assignment_order, assignment_costs = _mode4_assignment_from_proxy(
            c_proxy, scene, config
        )
        (
            a_mat,
            b_mat,
            q_mat,
            c_mat,
            poles,
            beta_z,
            gamma,
            eta_pol,
            ris_eta,
        ) = _apply_physical_order(
            assignment_order,
            a_mat,
            b_mat,
            q_mat,
            c_mat,
            poles,
            beta_z,
            gamma,
            eta_pol,
            ris_eta,
        )
        c_proxy = c_proxy[:, assignment_order]
        beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
        ris_projection_details = []
        for k in range(scene["K"]):
            c_before = c_mat[:, k].copy()
            before_value, _ = scaled_residual(c_proxy[:, k], c_before, config["eps"])
            beta_before_c, z_hat_before_c, sse_before_c = _fit_z_model(
                z_tensor, a_mat, b_mat, q_mat, c_mat
            )
            ris_proj = project_ris_factor(
                c_proxy[:, k],
                scene["Omega"][k],
                scene["a_RB"][k],
                scene["ris_grid"],
                scene["wavelength"],
                local_ris_search_config(scene, config, k),
                config["eps"],
                current_eta=ris_eta[k],
            )

            trial_c = c_mat.copy()
            trial_c[:, k] = ris_proj["c"]
            beta_trial, z_hat_trial, sse_trial = _fit_z_model(
                z_tensor, a_mat, b_mat, q_mat, trial_c
            )
            accepted = sse_trial <= sse_before_c + accept_tol
            if accepted:
                c_mat = trial_c
                ris_eta[k] = ris_proj["eta_local"]
                beta_z = beta_trial
                z_hat = z_hat_trial
                current_sse = sse_trial
            else:
                beta_z = beta_before_c
                z_hat = z_hat_before_c
                current_sse = sse_before_c

            after_value, _ = scaled_residual(c_proxy[:, k], c_mat[:, k], config["eps"])
            c_change = _relative_change(c_mat[:, k], c_before, config["eps"])
            ris_projection_details.append(
                {
                    "path": k,
                    "residual_before": float(
                        np.sqrt(before_value / (np.linalg.norm(c_proxy[:, k]) ** 2 + config["eps"]))
                    ),
                    "residual_after": float(
                        np.sqrt(after_value / (np.linalg.norm(c_proxy[:, k]) ** 2 + config["eps"]))
                    ),
                    "selected_eta": ris_eta[k].copy(),
                    "c_relative_change": c_change,
                    "accepted": accepted,
                    "selected_model": ris_proj["selected_model"] if accepted else "current",
                    "global_sse_before": float(sse_before_c),
                    "global_sse_after": float(current_sse),
                    "exact_relative_residual": ris_proj.get("exact_relative_residual"),
                    "lifted_used": bool(ris_proj.get("lifted_used", False) and accepted),
                    "lifted_relative_residual": ris_proj.get("lifted_relative_residual"),
                    "optimizer_message": ris_proj["optimizer_message"],
                }
            )

        proposed_sse = current_sse
        iteration_accepted = (not safeguard) or (proposed_sse <= iter_start_sse + accept_tol)
        if not iteration_accepted:
            a_mat = a_old
            b_mat = b_old
            q_mat = q_old
            c_mat = c_old
            poles = poles_old
            gamma = gamma_old
            eta_pol = eta_pol_old
            ris_eta = ris_eta_old
            beta_z = beta_old
            z_hat = z_hat_old
            current_sse = iter_start_sse

        diagnostics["z_hat_history"].append(z_hat)
        diagnostics["residuals_noisy_rmse"].append(
            float(np.linalg.norm(z_hat - z_tensor) / np.sqrt(z_tensor.size))
        )
        relative_residual_change = abs(iter_start_sse - current_sse) / max(
            iter_start_sse, config["eps"]
        )
        diagnostics["updates"].append(
            {
                "delta_A": _relative_change(a_mat, a_old, config["eps"]),
                "delta_B": _relative_change(b_mat, b_old, config["eps"]),
                "delta_Q": _relative_change(q_mat, q_old, config["eps"]),
                "delta_C": _relative_change(c_mat, c_old, config["eps"]),
                "delta_beta": _relative_change(beta_z, beta_old, config["eps"]),
                "nonfinite_A": _count_nonfinite(a_mat),
                "nonfinite_B": _count_nonfinite(b_mat),
                "nonfinite_Q": _count_nonfinite(q_mat),
                "nonfinite_C": _count_nonfinite(c_mat),
                "nonfinite_beta": _count_nonfinite(beta_z),
                "evs_projection_details": evs_projection_details,
                "mode4_assignment_order": assignment_order,
                "mode4_assignment_costs": assignment_costs,
                "ris_projection_details": ris_projection_details,
                "delay_projection_details": delay_projection_details,
                "iteration_accepted": bool(iteration_accepted),
                "iteration_sse_before": float(iter_start_sse),
                "iteration_sse_proposed": float(proposed_sse),
                "iteration_sse_after": float(current_sse),
                "relative_residual_change": float(relative_residual_change),
            }
        )
        if not iteration_accepted:
            break
        if stop_tol > 0.0 and relative_residual_change < stop_tol:
            break

    estimate.update(
        {
            "poles": poles,
            "A": a_mat,
            "B": b_mat,
            "Q": q_mat,
            "C": c_mat,
            "beta_z": beta_z,
            "gamma": gamma,
            "eta_pol": eta_pol,
            "ris_eta": ris_eta,
        }
    )
    check_finite("structured A", a_mat)
    check_finite("structured C", c_mat)
    return estimate, diagnostics


def estimate_position_from_local_ris(scene: dict, estimate: dict, config: dict) -> np.ndarray:
    """Estimate UE position by averaging RIS-local geometry estimates."""
    return _initial_global_parameters(scene, estimate, config)[:3]


def _initial_global_parameters(scene: dict, estimate: dict, config: dict) -> np.ndarray:
    """Build p_u and Delta_t initial values from local RIS estimates."""
    positions = []
    for k in range(scene["K"]):
        positions.append(
            position_from_local_geometry(
                scene["ris_centers"][k],
                scene["rotations"][k],
                estimate["ris_eta"][k, 0],
                estimate["ris_eta"][k, 1],
                estimate["ris_eta"][k, 2],
            )
        )
    p_init = np.mean(np.asarray(positions), axis=0)
    p_init = np.clip(p_init, config["ue_bounds"][:, 0], config["ue_bounds"][:, 1])

    dt_values = []
    for k in range(scene["K"]):
        tau_hat = tau_from_pole(estimate["poles"][k], scene["delta_f"])
        range_hat = estimate["ris_eta"][k, 0]
        dt_values.append(tau_hat - (range_hat + scene["d_RB"][k]) / scene["c0"])
    dt_init = float(np.median(dt_values))
    dt_init = float(np.clip(dt_init, *config["delta_t_bounds"]))

    return np.concatenate(
        [
            p_init,
            np.array([dt_init]),
            estimate["gamma"],
            estimate["eta_pol"],
        ]
    )


def _bounds_global(scene: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for global VP-WNLS variables."""
    k_paths = scene["K"]
    lower = np.concatenate(
        [
            config["ue_bounds"][:, 0],
            np.array([config["delta_t_bounds"][0]]),
            np.full(k_paths, 0.05),
            np.full(k_paths, -np.pi),
        ]
    )
    upper = np.concatenate(
        [
            config["ue_bounds"][:, 1],
            np.array([config["delta_t_bounds"][1]]),
            np.full(k_paths, 1.50),
            np.full(k_paths, np.pi),
        ]
    )
    return lower, upper


def _dictionary_from_global_x(scene: dict, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build raw-domain VP dictionary for p_u, Delta_t, gamma, eta."""
    k_paths = scene["K"]
    p_u = x[:3]
    delta_t = float(x[3])
    gamma = x[4 : 4 + k_paths]
    eta_pol = x[4 + k_paths : 4 + 2 * k_paths]
    components = channel_components(scene, p_u, delta_t, gamma, eta_pol)
    a_mat = components["a_EVS"].T
    d_mat = components["d"].T
    c_mat = components["c"].T
    dictionary = _raw_design_matrix_from_factors(a_mat, d_mat, c_mat)
    return dictionary, components


def refine_global_raw(y_noisy: np.ndarray, scene: dict, config: dict, estimate: dict) -> dict:
    """Stage 3: raw-domain global VP-WNLS refinement."""
    x0 = _initial_global_parameters(scene, estimate, config)
    lower, upper = _bounds_global(scene, config)
    x0 = np.clip(x0, lower, upper)
    y_vec = y_noisy.reshape(-1)

    def unpack_scaled(x_scaled: np.ndarray) -> np.ndarray:
        return lower + np.clip(x_scaled, 0.0, 1.0) * (upper - lower)

    def residual_complex_from_scaled(x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        x = unpack_scaled(x_scaled)
        dictionary, components = _dictionary_from_global_x(scene, x)
        beta = solve_lstsq(dictionary, y_vec, reg=1e-12)
        residual = dictionary @ beta - y_vec
        return residual, beta, components

    x0_scaled = (x0 - lower) / (upper - lower)

    if scipy_is_available():
        from scipy.optimize import least_squares

        def residual_real(x_scaled: np.ndarray) -> np.ndarray:
            residual, _, _ = residual_complex_from_scaled(x_scaled)
            return np.concatenate([residual.real, residual.imag])

        result = least_squares(
            residual_real,
            x0_scaled,
            bounds=(np.zeros_like(x0_scaled), np.ones_like(x0_scaled)),
            max_nfev=120,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        x_scaled_best = result.x
        optimizer_info = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": result.message,
            "n_eval": int(result.nfev),
            "method": "scipy.optimize.least_squares",
        }
    else:

        def objective(x_scaled: np.ndarray) -> float:
            residual, _, _ = residual_complex_from_scaled(x_scaled)
            return float(np.vdot(residual, residual).real / y_vec.size)

        x_scaled_best, _, info = bounded_coordinate_search(
            objective,
            x0_scaled,
            np.zeros_like(x0_scaled),
            np.ones_like(x0_scaled),
            step0=0.06,
            max_iter=65,
            tol=8e-5,
        )
        optimizer_info = {
            "success": info["success"],
            "status": 1 if info["success"] else 0,
            "message": info["message"],
            "n_eval": info["n_eval"],
            "method": "bounded coordinate search",
        }

    x_best = unpack_scaled(x_scaled_best)
    residual, beta_hat, components_hat = residual_complex_from_scaled(x_scaled_best)
    y_hat_noiseless_model = synthesize_raw_tensor(components_hat, beta_hat)
    residual_rmse_noisy = float(np.linalg.norm(residual) / np.sqrt(y_vec.size))

    return {
        "x": x_best,
        "p_u": x_best[:3],
        "delta_t": float(x_best[3]),
        "gamma": x_best[4 : 4 + scene["K"]],
        "eta_pol": x_best[4 + scene["K"] : 4 + 2 * scene["K"]],
        "beta": beta_hat,
        "components": components_hat,
        "Y_hat": y_hat_noiseless_model,
        "raw_residual_rmse_noisy": residual_rmse_noisy,
        "optimizer": optimizer_info,
    }
