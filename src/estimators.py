"""Readable proposed-method estimator for one small RIS-EVS-OFDM run."""

from __future__ import annotations

import copy
import itertools
import time
import numpy as np

from .channel_model import channel_components, synthesize_raw_tensor
from .global_vp import global_exact_spherical_vp_refinement
from .geometry import position_from_local_geometry
from .projections_delay import (
    bq_from_poles,
    delay_matrix_from_poles,
    estimate_poles_aimdf_asym_tls_from_hankel,
    estimate_poles_aimdf_tls_from_hankel,
    estimate_poles_aimdf_tls_from_hankel_with_diagnostics,
    estimate_poles_esprit_from_hankel,
    structured_delay_mother_pgd,
    tau_from_pole,
)
from .projections_evs import project_evs_factor
from .projections_ris import (
    compressed_exact_response,
    local_ris_search_config,
    project_ris_factor,
    scaled_residual,
)
from .tensor_utils import dehankelize_frequency, reconstruct_z, z_design_column
from .utils import bounded_coordinate_search, check_finite, solve_lstsq, scipy_is_available


def _relative_change(new_value: np.ndarray, old_value: np.ndarray, eps: float) -> float:
    """Return a safe relative change between two arrays."""
    return float(np.linalg.norm(new_value - old_value) / (np.linalg.norm(old_value) + eps))


def _relative_scaled_residual(
    target: np.ndarray, model: np.ndarray, eps: float
) -> float:
    """Return min_alpha ||target - alpha model|| / ||target||."""
    scale = np.vdot(model, target) / (np.vdot(model, model) + eps)
    return float(np.linalg.norm(target - scale * model) / (np.linalg.norm(target) + eps))


def _inverse_column_to_panel_assignment(column_to_panel: list[int]) -> list[int]:
    """Return panel-to-column inverse of a column-to-panel assignment."""
    panel_to_column = [-1] * len(column_to_panel)
    for column, panel in enumerate(column_to_panel):
        panel_to_column[int(panel)] = int(column)
    return panel_to_column


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
    _, _, residual_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
    return float(residual_sse / (np.linalg.norm(z_tensor) ** 2 + 1e-12))


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


def _accept_strict_sse(new_sse: float, old_sse: float, abs_tol: float, rel_tol: float) -> bool:
    threshold = old_sse - abs_tol - rel_tol * max(1.0, old_sse)
    return bool(new_sse <= threshold)


def _normalize_column(x: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm <= eps:
        return x.copy()
    return x / norm


def _choose_damped_column_update(
    z_tensor: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
    factor_name: str,
    col: int,
    projected_col: np.ndarray,
    damping_grid: tuple,
    old_sse: float,
    eps: float,
    accept_tol: float,
    strict_accept_rel: float,
    guarded: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool, float]:
    """
    Try damped replacement for one column of A or C.

    factor_name must be "A" or "C". Returns:
        new_A, new_C, beta_new, new_sse, best_rho, accepted, best_sse
    """
    if factor_name not in ("A", "C"):
        raise ValueError("factor_name must be 'A' or 'C'")

    beta_old, _, fitted_old_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
    old_sse = float(old_sse)
    if not np.isfinite(old_sse):
        old_sse = float(fitted_old_sse)

    best_a = a_mat
    best_c = c_mat
    best_beta = beta_old
    best_sse = np.inf
    best_rho = float("nan")

    for rho_raw in damping_grid:
        rho = float(rho_raw)
        if factor_name == "A":
            trial_a = a_mat.copy()
            trial_a[:, col] = _normalize_column(
                (1.0 - rho) * a_mat[:, col] + rho * projected_col, eps
            )
            trial_c = c_mat
        else:
            trial_c = c_mat.copy()
            trial_c[:, col] = _normalize_column(
                (1.0 - rho) * c_mat[:, col] + rho * projected_col, eps
            )
            trial_a = a_mat

        beta_trial, _, sse_trial = _fit_z_model(z_tensor, trial_a, b_mat, q_mat, trial_c)
        if sse_trial < best_sse:
            best_a = trial_a
            best_c = trial_c
            best_beta = beta_trial
            best_sse = float(sse_trial)
            best_rho = rho

    if guarded:
        accepted = _accept_strict_sse(best_sse, old_sse, accept_tol, strict_accept_rel)
    else:
        accepted = bool(best_sse <= old_sse + accept_tol)

    if accepted:
        return best_a, best_c, best_beta, best_sse, best_rho, True, best_sse
    return a_mat, c_mat, beta_old, old_sse, best_rho, False, best_sse


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


def _coupled_hankel_factor_initialization(
    z_tensor: np.ndarray,
    poles: np.ndarray,
    reg: float | None = None,
    config: dict | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """Initialize A/C proxies by LS with coupled Hankel B/Q delay factors."""
    assert z_tensor.ndim == 4, "Z must have shape I x P x L x T"
    total_start = time.perf_counter()
    i_dim, p_dim, l_dim, t_dim = z_tensor.shape
    k_paths = poles.size
    eps = 1.0e-12 if config is None else float(config.get("eps", 1.0e-12))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z14_23 = np.transpose(z_tensor, (0, 3, 1, 2)).reshape(i_dim * t_dim, p_dim * l_dim)
    d_delay = np.empty((p_dim * l_dim, k_paths), dtype=complex)
    for k in range(k_paths):
        d_delay[:, k] = (b_mat[:, k, None] * q_mat[None, :, k]).reshape(-1)

    gram0 = d_delay.T @ d_delay.conj()
    if config is not None:
        reg_mode = str(config.get("stage1_factor_reg_mode", "absolute")).lower()
        if reg_mode == "relative":
            reg_abs = (
                float(config.get("stage1_factor_reg_rel", 1.0e-6))
                * float(np.trace(gram0).real)
                / max(k_paths, 1)
                + float(config.get("stage1_factor_reg_floor", 1.0e-12))
            )
        elif reg_mode == "absolute":
            reg_abs = float(config.get("stage1_factor_reg", 1.0e-10))
        else:
            raise ValueError(f"unknown stage1_factor_reg_mode {reg_mode!r}")
    else:
        reg_abs = float(1.0e-10 if reg is None else reg)
        reg_mode = "absolute"
    gram = gram0 + reg_abs * np.eye(k_paths, dtype=complex)
    rhs = z14_23 @ d_delay.conj()
    try:
        h_mat = np.linalg.solve(gram.T, rhs.T).T
    except np.linalg.LinAlgError:
        h_mat = rhs @ np.linalg.pinv(gram)
    split_start = time.perf_counter()

    a_proxy = np.empty((i_dim, k_paths), dtype=complex)
    c_proxy = np.empty((t_dim, k_paths), dtype=complex)
    snapshot_singular_values = []
    rank1_ratios = np.empty(k_paths, dtype=float)
    for k in range(k_paths):
        snapshot = h_mat[:, k].reshape(i_dim, t_dim)
        u_mat, s_val, vh = np.linalg.svd(snapshot, full_matrices=False)
        snapshot_singular_values.append(s_val.copy())
        rank1_ratios[k] = float(s_val[1] / (s_val[0] + eps)) if len(s_val) > 1 else 0.0
        scale = np.sqrt(s_val[0])
        a_proxy[:, k] = u_mat[:, 0] * scale
        c_proxy[:, k] = vh[0, :] * scale
    split_time = time.perf_counter() - split_start
    total_time = time.perf_counter() - total_start
    diagnostics = {
        "stage1_factor_reg_mode": reg_mode,
        "stage1_factor_reg_abs": float(reg_abs),
        "stage1_factor_gram_condition_number": float(np.linalg.cond(gram)),
        "stage1_snapshot_singular_values": snapshot_singular_values,
        "stage1_rank1_ratios": rank1_ratios,
        "stage1_max_rank1_ratio": float(np.max(rank1_ratios)) if rank1_ratios.size else 0.0,
        "stage1_time_coupled_ls": float(max(total_time - split_time, 0.0)),
        "stage1_time_rank1_svd_split": float(split_time),
    }
    if return_diagnostics:
        return a_proxy, c_proxy, diagnostics
    return a_proxy, c_proxy


def _min_unit_pole_phase_separation(poles: np.ndarray) -> float:
    """Return the minimum circular phase separation between delay poles."""
    if poles.size < 2:
        return float("inf")
    phases = np.sort(np.angle(poles))
    diffs = np.diff(np.concatenate([phases, phases[:1] + 2.0 * np.pi]))
    return float(np.min(np.abs(diffs)))


def _coarse_ris_factor_projection(
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    eps: float,
) -> dict:
    """Coarse exact-spherical RIS codebook match without local WESVP refinement."""
    start = time.perf_counter()
    r_grid = np.linspace(*search_config["range_bounds"], int(search_config["num_range"]))
    e_grid = np.linspace(*search_config["elev_bounds"], int(search_config["num_elev"]))
    a_grid = np.linspace(*search_config["az_bounds"], int(search_config["num_az"]))
    c_norm_sq = np.linalg.norm(c_tilde) ** 2 + eps
    best = None
    for range_m in r_grid:
        for elevation in e_grid:
            for azimuth in a_grid:
                eta_local = np.array([range_m, elevation, azimuth], dtype=float)
                h_model = compressed_exact_response(
                    eta_local, omega, a_rb, ris_grid, wavelength
                )
                value, alpha = scaled_residual(c_tilde, h_model, eps)
                if best is None or value < best[0]:
                    best = (float(value), eta_local, alpha, h_model)
    assert best is not None, "empty RIS coarse codebook"
    value, eta_local, alpha, h_model = best
    c_projected = alpha * h_model
    return {
        "c": c_projected,
        "eta_local": eta_local,
        "alpha": alpha,
        "data_residual": value,
        "relative_residual": float(np.sqrt(value / c_norm_sq)),
        "selected_model": "coarse_correlation",
        "stage1_time_ris_codebook_build": float(time.perf_counter() - start),
        "stage1_time_ris_projection_refine": 0.0,
    }


def _assignment_by_projection(
    a_proxy: np.ndarray,
    c_proxy: np.ndarray,
    poles_raw: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[list[int], list[dict], list[dict], dict]:
    """Align CP columns to RIS panels using projection and clock consistency."""
    k_paths = scene["K"]
    scores = np.zeros((k_paths, k_paths), dtype=float)
    evs_cache: list[list[dict]] = [[{} for _ in range(k_paths)] for _ in range(k_paths)]
    ris_cache: list[list[dict]] = [[{} for _ in range(k_paths)] for _ in range(k_paths)]
    eps = config["eps"]
    implied_clock_offsets = np.zeros((k_paths, k_paths), dtype=float)
    geometry_mode = str(config.get("stage1_ris_geometry_mode", "coarse_correlation")).lower()
    timing = {
        "stage1_time_assignment_evs": 0.0,
        "stage1_time_assignment_ris": 0.0,
        "stage1_time_ris_codebook_build": 0.0,
        "stage1_time_ris_projection_refine": 0.0,
    }
    assignment_start = time.perf_counter()

    for col in range(k_paths):
        for ris in range(k_paths):
            evs_start = time.perf_counter()
            evs_proj = project_evs_factor(
                a_proxy[:, col], scene["v_B"][ris], scene["Theta"][ris], eps
            )
            timing["stage1_time_assignment_evs"] += time.perf_counter() - evs_start
            ris_search = local_ris_search_config(scene, config, ris)
            if geometry_mode == "coarse_correlation":
                ris_start = time.perf_counter()
                ris_proj = _coarse_ris_factor_projection(
                    c_proxy[:, col],
                    scene["Omega"][ris],
                    scene["a_RB"][ris],
                    scene["ris_grid"],
                    scene["wavelength"],
                    ris_search,
                    eps,
                )
                ris_elapsed = time.perf_counter() - ris_start
                timing["stage1_time_assignment_ris"] += ris_elapsed
                timing["stage1_time_ris_codebook_build"] += float(
                    ris_proj.get("stage1_time_ris_codebook_build", ris_elapsed)
                )
            elif geometry_mode in ("exact_projection", "legacy_fast_projection"):
                ris_search["projection_mode"] = "exact"
                ris_start = time.perf_counter()
                ris_proj = project_ris_factor(
                    c_proxy[:, col],
                    scene["Omega"][ris],
                    scene["a_RB"][ris],
                    scene["ris_grid"],
                    scene["wavelength"],
                    ris_search,
                    eps,
                )
                ris_elapsed = time.perf_counter() - ris_start
                timing["stage1_time_assignment_ris"] += ris_elapsed
                timing["stage1_time_ris_projection_refine"] += ris_elapsed
            else:
                raise ValueError(f"unknown stage1_ris_geometry_mode {geometry_mode!r}")
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
    all_assignment_scores = []
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
        all_assignment_scores.append(
            {
                "assignment": list(perm),
                "score": float(score),
                "projection_score": float(projection_score),
                "clock_spread": float(clock_spread),
                "clock_bound_violation": float(bound_violation),
            }
        )
        if score < best_score:
            best_score = score
            best_perm = list(perm)

    assert best_perm is not None, "failed to find a column association"
    evs_selected = [evs_cache[col][ris] for col, ris in enumerate(best_perm)]
    ris_selected = [ris_cache[col][ris] for col, ris in enumerate(best_perm)]
    all_assignment_scores.sort(key=lambda item: item["score"])
    second_score = (
        all_assignment_scores[1]["score"] if len(all_assignment_scores) > 1 else float("inf")
    )
    selected_clock_offsets = np.array(
        [implied_clock_offsets[col, ris] for col, ris in enumerate(best_perm)]
    )
    diagnostics = {
        "assignment_costs_col_by_panel": scores,
        "best_assignment_score": float(best_score),
        "second_assignment_score": float(second_score),
        "assignment_margin": float((second_score - best_score) / (best_score + eps)),
        "selected_clock_offsets": selected_clock_offsets,
        "selected_clock_mean": float(np.mean(selected_clock_offsets)),
        "selected_clock_std": float(np.std(selected_clock_offsets)),
        "all_assignment_scores": all_assignment_scores,
        "stage1_time_assignment_total": float(time.perf_counter() - assignment_start),
        **{key: float(value) for key, value in timing.items()},
    }
    return best_perm, evs_selected, ris_selected, diagnostics


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
    """A-IMDF-inspired RIS-EVS initializer.

    This is an A-IMDF-inspired RIS-EVS initializer. It follows the A-IMDF
    delay-first principle but is not a direct copy of the EVS-only A-IMDF
    algorithm. The RIS-EVS model requires coupled LS recovery of joint EVS-RIS
    factors and rank-one SVD splitting. Heavy exact RIS near-field projection is
    deliberately excluded from default Stage-I and reserved for
    reliability-gated Stage-II basin recovery.
    """
    assert z_tensor.shape == (scene["I"], scene["P"], scene["L"], scene["T"])
    stage1_total_start = time.perf_counter()
    stage1_timing: dict[str, float] = {
        "stage1_time_delay_estimation": 0.0,
        "stage1_time_vandermonde_reconstruction": 0.0,
        "stage1_time_coupled_ls": 0.0,
        "stage1_time_rank1_svd_split": 0.0,
        "stage1_time_assignment_total": 0.0,
        "stage1_time_assignment_evs": 0.0,
        "stage1_time_assignment_ris": 0.0,
        "stage1_time_ris_codebook_build": 0.0,
        "stage1_time_ris_projection_refine": 0.0,
        "stage1_time_reliability_diagnostics": 0.0,
        "stage1_time_other": 0.0,
    }
    delay_method = str(config.get("stage1_delay_method", "aimdf_fullfreq_tls"))
    if delay_method == "aimdf_tls":
        delay_method = "aimdf_fullfreq_tls"
    forward_backward = bool(config.get("stage1_forward_backward", True))
    tls = bool(config.get("stage1_tls", True))
    delay_start = time.perf_counter()
    if delay_method == "aimdf_fullfreq_tls":
        poles_raw, delay_diagnostics = estimate_poles_aimdf_tls_from_hankel_with_diagnostics(
            z_tensor,
            scene["K"],
            forward_backward=forward_backward,
            tls=tls,
            eps=config["eps"],
        )
    elif delay_method == "aimdf_asym_tls":
        poles_raw, delay_diagnostics = estimate_poles_aimdf_asym_tls_from_hankel(
            z_tensor,
            scene["K"],
            forward_backward=forward_backward,
            tls=tls,
            snapshot_sketch_dim=config.get("stage1_snapshot_sketch_dim"),
            sketch_seed=int(config.get("seed", 0)),
            eps=config["eps"],
        )
    elif delay_method == "esprit_ls":
        poles_raw = estimate_poles_esprit_from_hankel(z_tensor, scene["K"])
        delay_diagnostics = {
            "delay_method": "esprit_ls",
            "singular_values": np.array([], dtype=float),
            "pole_magnitudes_before_unit_circle": np.abs(poles_raw),
            "forward_backward": False,
            "tls": False,
            "snapshot_sketch_dim": None,
        }
    else:
        raise ValueError(f"unknown stage1_delay_method {delay_method!r}")
    stage1_timing["stage1_time_delay_estimation"] = time.perf_counter() - delay_start

    factor_init = str(config.get("stage1_factor_init", "hankel_coupled_ls"))
    if factor_init == "hankel_coupled_ls":
        a_proxy, c_proxy, factor_diagnostics = _coupled_hankel_factor_initialization(
            z_tensor,
            poles_raw,
            config.get("stage1_factor_reg", 1e-10),
            config=config,
            return_diagnostics=True,
        )
        stage1_timing["stage1_time_coupled_ls"] = float(
            factor_diagnostics.get("stage1_time_coupled_ls", 0.0)
        )
        stage1_timing["stage1_time_rank1_svd_split"] = float(
            factor_diagnostics.get("stage1_time_rank1_svd_split", 0.0)
        )
    elif factor_init in ("raw_snapshot", "snapshot_ls"):
        factor_start = time.perf_counter()
        a_proxy, c_proxy = _rank_one_snapshot_initialization(z_tensor, poles_raw)
        stage1_timing["stage1_time_rank1_svd_split"] = time.perf_counter() - factor_start
        factor_diagnostics = {}
    else:
        raise ValueError(f"unknown stage1_factor_init {factor_init!r}")
    assignment, evs_selected, ris_selected, assignment_diagnostics = _assignment_by_projection(
        a_proxy, c_proxy, poles_raw, scene, config
    )
    for key in (
        "stage1_time_assignment_total",
        "stage1_time_assignment_evs",
        "stage1_time_assignment_ris",
        "stage1_time_ris_codebook_build",
        "stage1_time_ris_projection_refine",
    ):
        stage1_timing[key] = float(assignment_diagnostics.get(key, 0.0))

    k_paths = scene["K"]
    reliability_diag_start = time.perf_counter()
    poles = np.empty(k_paths, dtype=complex)
    a_mat = np.empty((scene["I"], k_paths), dtype=complex)
    c_mat = np.empty((scene["T"], k_paths), dtype=complex)
    ris_eta = np.empty((k_paths, 3), dtype=float)
    gamma = np.empty(k_paths, dtype=float)
    eta_pol = np.empty(k_paths, dtype=float)
    stage1_ris_residuals = np.empty(k_paths, dtype=float)

    # Store columns in physical RIS-panel order.
    for col, ris in enumerate(assignment):
        poles[ris] = poles_raw[col]
        a_mat[:, ris] = evs_selected[col]["a"]
        c_mat[:, ris] = ris_selected[col]["c"]
        ris_eta[ris] = ris_selected[col]["eta_local"]
        gamma[ris] = evs_selected[col]["gamma"]
        eta_pol[ris] = evs_selected[col]["eta"]
        stage1_ris_residuals[ris] = float(
            ris_selected[col].get("relative_residual", np.nan)
        )
    stage1_timing["stage1_time_reliability_diagnostics"] = (
        time.perf_counter() - reliability_diag_start
    )

    vandermonde_start = time.perf_counter()
    b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
    stage1_timing["stage1_time_vandermonde_reconstruction"] = (
        time.perf_counter() - vandermonde_start
    )
    beta_z = _estimate_weights_z(z_tensor, a_mat, b_mat, q_mat, c_mat)
    z_hat = reconstruct_z(beta_z, a_mat, b_mat, q_mat, c_mat)
    initial_residual = float(
        np.linalg.norm(z_hat - z_tensor) ** 2
        / (np.linalg.norm(z_tensor) ** 2 + config["eps"])
    )

    accounted_time = sum(stage1_timing.values())
    stage1_total_time = time.perf_counter() - stage1_total_start
    stage1_timing["stage1_time_other"] = float(
        max(stage1_total_time - accounted_time, 0.0)
    )
    residual_type = str(
        config.get(
            "stage1_ris_residual_type",
            "coarse_proxy"
            if str(config.get("stage1_ris_geometry_mode", "coarse_correlation"))
            == "coarse_correlation"
            else "legacy_fast_projection",
        )
    )

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
        "column_to_panel_assignment": assignment,
        "panel_to_column_assignment": _inverse_column_to_panel_assignment(assignment),
        "columns_are_panel_ordered": True,
        "assignment_costs": assignment_diagnostics["assignment_costs_col_by_panel"],
        "assignment_costs_col_by_panel": assignment_diagnostics[
            "assignment_costs_col_by_panel"
        ],
        "best_assignment_score": assignment_diagnostics["best_assignment_score"],
        "second_assignment_score": assignment_diagnostics["second_assignment_score"],
        "assignment_margin": assignment_diagnostics["assignment_margin"],
        "stage1_assignment_margin": assignment_diagnostics["assignment_margin"],
        "selected_clock_offsets": assignment_diagnostics["selected_clock_offsets"],
        "selected_clock_mean": assignment_diagnostics["selected_clock_mean"],
        "selected_clock_std": assignment_diagnostics["selected_clock_std"],
        "all_assignment_scores": assignment_diagnostics["all_assignment_scores"],
        "stage1_ris_residuals": stage1_ris_residuals,
        "stage1_max_ris_residual": float(np.nanmax(stage1_ris_residuals))
        if stage1_ris_residuals.size
        else float("nan"),
        "stage1_ris_residual_type": residual_type,
        "initial_z_residual": initial_residual,
        "Z_hat": z_hat,
        "stage1_delay_method": delay_method,
        "stage1_delay_singular_values": delay_diagnostics["singular_values"],
        "stage1_pole_magnitudes_before_unit_circle": delay_diagnostics[
            "pole_magnitudes_before_unit_circle"
        ],
        "stage1_min_delay_pole_phase_sep": _min_unit_pole_phase_separation(poles_raw),
        "stage1_factor_init": factor_init,
        "stage1_forward_backward": forward_backward,
        "stage1_tls": tls,
        "stage1_snapshot_sketch_dim": delay_diagnostics.get("snapshot_sketch_dim"),
        "stage1_ris_geometry_mode": config.get(
            "stage1_ris_geometry_mode", "coarse_correlation"
        ),
        **stage1_timing,
        **factor_diagnostics,
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


def _c_update_design_and_target(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common C-mode LS design and T-column target."""
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
    return design, target


def _ris_projection_weight_from_c_residual(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_proxy: np.ndarray,
    config: dict,
) -> tuple[np.ndarray | None, dict]:
    """Estimate diagonal WESVP sample weights from C-mode LS residuals."""
    mode = str(config.get("stage2_ris_weight_mode", "residual_diag")).lower()
    if mode in ("none", "identity", "unit"):
        return None, {
            "mode": mode,
            "enabled": False,
            "min": 1.0,
            "max": 1.0,
            "mean": 1.0,
            "std": 0.0,
        }
    if mode != "residual_diag":
        raise ValueError(f"unknown stage2_ris_weight_mode {mode!r}")

    design, target = _c_update_design_and_target(z_tensor, beta, a_mat, b_mat, q_mat)
    residual = target - design @ c_proxy.T
    residual_power = np.mean(np.abs(residual) ** 2, axis=0)
    finite = np.isfinite(residual_power)
    if not np.any(finite):
        weights = np.ones(z_tensor.shape[3], dtype=float)
    else:
        positive = residual_power[finite & (residual_power > config["eps"])]
        reference = float(np.median(positive)) if positive.size else 0.0
        if reference <= config["eps"]:
            weights = np.ones_like(residual_power, dtype=float)
        else:
            floor = max(
                config["eps"],
                float(config.get("stage2_ris_weight_floor_rel", 5.0e-2)) * reference,
            )
            weights = reference / (residual_power + floor)
            weights[~finite] = 1.0

    clip_min, clip_max = config.get("stage2_ris_weight_clip", (0.25, 4.0))
    weights = np.clip(weights, float(clip_min), float(clip_max))
    if bool(config.get("stage2_ris_weight_normalize", True)):
        weights = weights / max(float(np.mean(weights)), config["eps"])
    weights = np.asarray(weights, dtype=float)
    return weights, {
        "mode": mode,
        "enabled": True,
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
    }


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
    enable_evs = bool(config.get("stage2_enable_evs", True))
    enable_delay = bool(config.get("stage2_enable_delay", True))
    enable_ris = bool(config.get("stage2_enable_ris", True))
    guarded = bool(config.get("stage2_guarded", False))
    strict_accept_rel = float(config.get("stage2_strict_accept_rel", 0.0))
    ris_min_rel_improvement = float(config.get("ris_min_relative_improvement", 0.0))
    damping_grid = tuple(config.get("stage2_damping_grid", (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)))
    if not damping_grid:
        damping_grid = (1.0,)
    diagnostics = {
        "z_hat_history": [],
        "residuals_noisy_rmse": [],
        "updates": [],
        "ris_projection_total_s": 0.0,
    }
    ris_projection_time_total = 0.0
    b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
    beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
    safeguard = bool(config.get("stage2_global_safeguard", True))
    accept_tol = float(config.get("stage2_accept_tol", 1e-9))
    stop_tol = float(config.get("stage2_tol", 0.0))

    if not (enable_evs or enable_delay or enable_ris):
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
                "Z_hat": z_hat,
            }
        )
        check_finite("structured A", a_mat)
        check_finite("structured C", c_mat)
        return estimate, diagnostics

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

        evs_projection_details = []
        if enable_evs:
            a_proxy = _update_a_from_z(z_tensor, beta_old, b_mat, q_mat, c_mat)
            if guarded:
                current_sse = iter_start_sse
                beta_z = beta_old
                z_hat = z_hat_old
                for k in range(scene["K"]):
                    a_before = a_mat[:, k].copy()
                    gamma_before = float(gamma[k])
                    eta_pol_before = float(eta_pol[k])
                    evs_proj = project_evs_factor(
                        a_proxy[:, k], scene["v_B"][k], scene["Theta"][k], config["eps"]
                    )
                    local_res_before = _relative_scaled_residual(
                        a_proxy[:, k], a_before, config["eps"]
                    )
                    sse_before_a = current_sse
                    (
                        a_candidate,
                        c_candidate,
                        beta_candidate,
                        sse_after_a,
                        best_rho,
                        accepted,
                        best_sse,
                    ) = _choose_damped_column_update(
                        z_tensor,
                        a_mat,
                        b_mat,
                        q_mat,
                        c_mat,
                        "A",
                        k,
                        evs_proj["a"],
                        damping_grid,
                        sse_before_a,
                        config["eps"],
                        accept_tol,
                        strict_accept_rel,
                        guarded,
                    )
                    if accepted:
                        a_mat = a_candidate
                        c_mat = c_candidate
                        beta_z = beta_candidate
                        current_sse = sse_after_a
                        z_hat = reconstruct_z(beta_z, a_mat, b_mat, q_mat, c_mat)
                        gamma[k] = evs_proj["gamma"]
                        eta_pol[k] = evs_proj["eta"]
                    evs_projection_details.append(
                        {
                            "path": k,
                            "accepted": bool(accepted),
                            "skipped": False,
                            "guarded": True,
                            "reason": "accepted_guarded_damped_sse"
                            if accepted
                            else "rejected_guarded_global_sse",
                            "best_rho": float(best_rho),
                            "best_sse": float(best_sse),
                            "global_sse_before": float(sse_before_a),
                            "global_sse_after": float(current_sse),
                            "local_res_before": float(local_res_before),
                            "local_res_after": float(
                                _relative_scaled_residual(
                                    a_proxy[:, k], a_mat[:, k], config["eps"]
                                )
                            ),
                            "candidate_local_residual": float(evs_proj["residual"]),
                            "relative_improvement": float(
                                (local_res_before - _relative_scaled_residual(
                                    a_proxy[:, k], a_mat[:, k], config["eps"]
                                ))
                                / max(local_res_before, config["eps"])
                            ),
                            "relative_change": _relative_change(a_mat[:, k], a_before, config["eps"]),
                            "gamma_before": gamma_before,
                            "gamma_after": float(gamma[k]),
                            "eta_pol_before": eta_pol_before,
                            "eta_pol_after": float(eta_pol[k]),
                            "projection_residual": evs_proj["residual"],
                        }
                    )
            else:
                for k in range(scene["K"]):
                    a_before = a_mat[:, k].copy()
                    gamma_before = float(gamma[k])
                    eta_pol_before = float(eta_pol[k])
                    evs_proj = project_evs_factor(
                        a_proxy[:, k], scene["v_B"][k], scene["Theta"][k], config["eps"]
                    )
                    local_res_before = _relative_scaled_residual(
                        a_proxy[:, k], a_before, config["eps"]
                    )
                    a_mat[:, k] = evs_proj["a"]
                    gamma[k] = evs_proj["gamma"]
                    eta_pol[k] = evs_proj["eta"]
                    local_res_after = _relative_scaled_residual(
                        a_proxy[:, k], a_mat[:, k], config["eps"]
                    )
                    evs_projection_details.append(
                        {
                            "path": k,
                            "accepted": bool(np.all(np.isfinite(a_mat[:, k]))),
                            "skipped": False,
                            "guarded": False,
                            "reason": "accepted_projection"
                            if np.all(np.isfinite(a_mat[:, k]))
                            else "rejected_nonfinite",
                            "best_rho": 1.0,
                            "global_sse_before": "",
                            "global_sse_after": "",
                            "local_res_before": float(local_res_before),
                            "local_res_after": float(local_res_after),
                            "candidate_local_residual": float(evs_proj["residual"]),
                            "relative_improvement": float(
                                (local_res_before - local_res_after)
                                / max(local_res_before, config["eps"])
                            ),
                            "relative_change": _relative_change(a_mat[:, k], a_before, config["eps"]),
                            "gamma_before": gamma_before,
                            "gamma_after": float(gamma[k]),
                            "eta_pol_before": eta_pol_before,
                            "eta_pol_after": float(eta_pol[k]),
                            "projection_residual": evs_proj["residual"],
                        }
                    )
                beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
        else:
            for k in range(scene["K"]):
                evs_projection_details.append(
                    {
                        "path": k,
                        "accepted": False,
                        "skipped": True,
                        "guarded": guarded,
                        "reason": "skipped_stage2_evs_disabled",
                        "best_rho": float("nan"),
                        "global_sse_before": float(current_sse),
                        "global_sse_after": float(current_sse),
                        "local_res_before": float("nan"),
                        "local_res_after": float("nan"),
                        "candidate_local_residual": float("nan"),
                        "relative_improvement": float("nan"),
                        "relative_change": 0.0,
                        "gamma_before": float(gamma[k]),
                        "gamma_after": float(gamma[k]),
                        "eta_pol_before": float(eta_pol[k]),
                        "eta_pol_after": float(eta_pol[k]),
                        "projection_residual": float("nan"),
                    }
                )

        if enable_delay:
            tau_before_delay = np.array(
                [tau_from_pole(pole, scene["delta_f"]) for pole in poles]
            )
            poles_candidate, delay_projection_details = _update_delay_poles_from_z(
                z_tensor, beta_z, a_mat, c_mat, poles, ris_eta, scene, config
            )
            b_candidate, q_candidate = bq_from_poles(poles_candidate, scene["P"], scene["L"])
            beta_candidate, z_hat_candidate, delay_sse_after = _fit_z_model(
                z_tensor, a_mat, b_candidate, q_candidate, c_mat
            )
            if guarded:
                delay_accepted = _accept_strict_sse(
                    delay_sse_after, current_sse, accept_tol, strict_accept_rel
                )
            else:
                delay_accepted = delay_sse_after <= current_sse + accept_tol
            delay_projection_details.update(
                {
                    "accepted": bool(delay_accepted),
                    "skipped": False,
                    "guarded": bool(guarded),
                    "reason": "accepted_global_sse"
                    if delay_accepted
                    else "rejected_global_sse",
                    "global_sse_before": float(current_sse),
                    "global_sse_after": float(delay_sse_after),
                    "tau_before": tau_before_delay,
                    "tau_candidate": np.array(
                        [tau_from_pole(pole, scene["delta_f"]) for pole in poles_candidate]
                    ),
                }
            )
            if delay_accepted:
                poles = poles_candidate
                b_mat = b_candidate
                q_mat = q_candidate
                beta_z = beta_candidate
                z_hat = z_hat_candidate
                current_sse = delay_sse_after
            delay_projection_details["tau_after"] = np.array(
                [tau_from_pole(pole, scene["delta_f"]) for pole in poles]
            )
            delay_projection_details["local_res_before"] = float(
                delay_projection_details.get("initial_objective", np.nan)
            )
            delay_projection_details["local_res_after"] = float(
                delay_projection_details.get("final_objective", np.nan)
            )
            delay_projection_details["relative_improvement"] = float(
                (
                    delay_projection_details["local_res_before"]
                    - delay_projection_details["local_res_after"]
                )
                / max(delay_projection_details["local_res_before"], config["eps"])
            )
        else:
            tau_current = np.array([tau_from_pole(pole, scene["delta_f"]) for pole in poles])
            delay_projection_details = {
                "accepted": False,
                "skipped": True,
                "guarded": bool(guarded),
                "reason": "skipped_stage2_delay_disabled",
                "global_sse_before": float(current_sse),
                "global_sse_after": float(current_sse),
                "local_res_before": float("nan"),
                "local_res_after": float("nan"),
                "relative_improvement": float("nan"),
                "tau_before": tau_current,
                "tau_candidate": tau_current,
                "tau_after": tau_current,
            }

        assignment_order = list(range(scene["K"]))
        assignment_costs = np.full((scene["K"], scene["K"]), np.nan)
        ris_projection_details = []
        if enable_ris:
            c_proxy = _update_c_from_z(z_tensor, beta_z, a_mat, b_mat, q_mat)
            assignment_t0 = time.perf_counter()
            assignment_order, assignment_costs = _mode4_assignment_from_proxy(
                c_proxy, scene, config
            )
            assignment_time_s = time.perf_counter() - assignment_t0
            ris_projection_time_total += assignment_time_s
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
            ris_weight, ris_weight_diag = _ris_projection_weight_from_c_residual(
                z_tensor, beta_z, a_mat, b_mat, q_mat, c_proxy, config
            )
            beta_z, z_hat, current_sse = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
            for k in range(scene["K"]):
                c_before = c_mat[:, k].copy()
                eta_before = ris_eta[k].copy()
                before_value, _ = scaled_residual(c_proxy[:, k], c_before, config["eps"])
                beta_before_c, z_hat_before_c, sse_before_c = _fit_z_model(
                    z_tensor, a_mat, b_mat, q_mat, c_mat
                )
                projection_t0 = time.perf_counter()
                ris_proj = project_ris_factor(
                    c_proxy[:, k],
                    scene["Omega"][k],
                    scene["a_RB"][k],
                    scene["ris_grid"],
                    scene["wavelength"],
                    local_ris_search_config(scene, config, k),
                    config["eps"],
                    current_eta=ris_eta[k]
                    if bool(config.get("stage2_ris_use_current_eta", True))
                    else None,
                    weight=ris_weight,
                )
                projection_time_s = time.perf_counter() - projection_t0
                ris_projection_time_total += projection_time_s

                candidate_value, _ = scaled_residual(c_proxy[:, k], ris_proj["c"], config["eps"])
                local_relative_improvement = float(
                    (before_value - candidate_value) / max(before_value, config["eps"])
                )
                best_rho = 1.0
                best_sse = float("nan")
                reason = "accepted_global_sse"
                if guarded:
                    if local_relative_improvement < ris_min_rel_improvement:
                        accepted = False
                        beta_z = beta_before_c
                        z_hat = z_hat_before_c
                        current_sse = sse_before_c
                        best_rho = float("nan")
                        reason = "rejected_min_local_improvement"
                    else:
                        (
                            a_candidate,
                            c_candidate,
                            beta_trial,
                            sse_trial,
                            best_rho,
                            accepted,
                            best_sse,
                        ) = _choose_damped_column_update(
                            z_tensor,
                            a_mat,
                            b_mat,
                            q_mat,
                            c_mat,
                            "C",
                            k,
                            ris_proj["c"],
                            damping_grid,
                            sse_before_c,
                            config["eps"],
                            accept_tol,
                            strict_accept_rel,
                            guarded,
                        )
                        if accepted:
                            a_mat = a_candidate
                            c_mat = c_candidate
                            ris_eta[k] = ris_proj["eta_local"]
                            beta_z = beta_trial
                            z_hat = reconstruct_z(beta_z, a_mat, b_mat, q_mat, c_mat)
                            current_sse = sse_trial
                            reason = "accepted_guarded_damped_sse"
                        else:
                            beta_z = beta_trial
                            z_hat = reconstruct_z(beta_z, a_mat, b_mat, q_mat, c_mat)
                            current_sse = sse_trial
                            reason = "rejected_guarded_global_sse"
                else:
                    trial_c = c_mat.copy()
                    trial_c[:, k] = ris_proj["c"]
                    beta_trial, z_hat_trial, sse_trial = _fit_z_model(
                        z_tensor, a_mat, b_mat, q_mat, trial_c
                    )
                    best_sse = float(sse_trial)
                    accepted = sse_trial <= sse_before_c + accept_tol
                    if accepted:
                        c_mat = trial_c
                        ris_eta[k] = ris_proj["eta_local"]
                        beta_z = beta_trial
                        z_hat = z_hat_trial
                        current_sse = sse_trial
                        reason = "accepted_global_sse"
                    else:
                        beta_z = beta_before_c
                        z_hat = z_hat_before_c
                        current_sse = sse_before_c
                        best_rho = 0.0
                        reason = "rejected_global_sse"

                after_value, _ = scaled_residual(c_proxy[:, k], c_mat[:, k], config["eps"])
                c_change = _relative_change(c_mat[:, k], c_before, config["eps"])
                ris_projection_details.append(
                    {
                        "path": k,
                        "skipped": False,
                        "guarded": bool(guarded),
                        "local_relative_improvement": local_relative_improvement,
                        "relative_improvement": local_relative_improvement,
                        "best_rho": float(best_rho),
                        "best_sse": float(best_sse),
                        "residual_before": float(
                            np.sqrt(before_value / (np.linalg.norm(c_proxy[:, k]) ** 2 + config["eps"]))
                        ),
                        "residual_after": float(
                            np.sqrt(after_value / (np.linalg.norm(c_proxy[:, k]) ** 2 + config["eps"]))
                        ),
                        "selected_eta": ris_eta[k].copy(),
                        "eta_before": eta_before,
                        "candidate_eta": ris_proj["eta_local"].copy(),
                        "c_relative_change": c_change,
                        "accepted": bool(accepted),
                        "reason": reason,
                        "selected_model": ris_proj["selected_model"] if accepted else "current",
                        "global_sse_before": float(sse_before_c),
                        "global_sse_after": float(current_sse),
                        "projection_time_s": float(projection_time_s),
                        "candidate_ranking": ris_proj.get("candidate_ranking", []),
                        "exact_relative_residual": ris_proj.get("exact_relative_residual"),
                        "lifted_used": bool(ris_proj.get("lifted_used", False) and accepted),
                        "lifted_relative_residual": ris_proj.get("lifted_relative_residual"),
                        "optimizer_message": ris_proj["optimizer_message"],
                        "weight_mode": ris_weight_diag["mode"],
                        "weight_enabled": bool(ris_weight_diag["enabled"]),
                        "weight_min": float(ris_weight_diag["min"]),
                        "weight_max": float(ris_weight_diag["max"]),
                        "weight_mean": float(ris_weight_diag["mean"]),
                        "weight_std": float(ris_weight_diag["std"]),
                    }
                )
        else:
            for k in range(scene["K"]):
                ris_projection_details.append(
                    {
                        "path": k,
                        "skipped": True,
                        "guarded": bool(guarded),
                        "local_relative_improvement": float("nan"),
                        "best_rho": float("nan"),
                        "residual_before": float("nan"),
                        "residual_after": float("nan"),
                        "selected_eta": ris_eta[k].copy(),
                        "eta_before": ris_eta[k].copy(),
                        "candidate_eta": ris_eta[k].copy(),
                        "c_relative_change": 0.0,
                        "accepted": False,
                        "reason": "skipped_stage2_ris_disabled",
                        "selected_model": "skipped",
                        "global_sse_before": float(current_sse),
                        "global_sse_after": float(current_sse),
                        "projection_time_s": 0.0,
                        "candidate_ranking": [],
                    }
                )
            assignment_time_s = 0.0

        proposed_sse = current_sse
        if not safeguard:
            iteration_accepted = True
        elif guarded:
            iteration_accepted = _accept_strict_sse(
                proposed_sse, iter_start_sse, accept_tol, strict_accept_rel
            )
        else:
            iteration_accepted = proposed_sse <= iter_start_sse + accept_tol
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
                "mode4_assignment_time_s": float(assignment_time_s),
                "relative_residual_change": float(relative_residual_change),
            }
        )
        diagnostics["ris_projection_total_s"] = float(ris_projection_time_total)
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
            "Z_hat": z_hat,
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
    residual_initial, _, _ = residual_complex_from_scaled(x0_scaled)
    raw_objective_initial = float(np.vdot(residual_initial, residual_initial).real / y_vec.size)

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
            "solver_backend": "scipy.optimize",
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
            "solver_backend": "fallback",
        }

    x_best = unpack_scaled(x_scaled_best)
    residual, beta_hat, components_hat = residual_complex_from_scaled(x_scaled_best)
    raw_objective_final = float(np.vdot(residual, residual).real / y_vec.size)
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
        "raw_objective_initial": raw_objective_initial,
        "raw_objective_final": raw_objective_final,
        "optimizer": optimizer_info,
    }


def run_proposed_estimator(
    y_raw: np.ndarray,
    z_tensor: np.ndarray,
    scene: dict,
    config: dict,
) -> dict:
    """Run the proposed default architecture and optional ablation paths.

    Default:
      Stage I: RIS-aware A-IMDF/TLS tensor initialization.
      Stage II: none.
      Final: initialization-aided global exact-spherical variable projection.

    Legacy factor-domain EVS/delay/RIS projections remain available only via
    ``stage2_mode="full_legacy"`` for ablation.
    """
    stage1 = initialize_from_hankel(z_tensor, scene, config)
    stage2_mode = str(config.get("stage2_mode", "none")).lower()
    if stage2_mode == "full_legacy":
        stage2, structured_diag = structured_refinement(
            z_tensor, scene, config, copy.deepcopy(stage1)
        )
    elif stage2_mode == "ris_only":
        raise NotImplementedError("stage2_mode='ris_only' is not a standalone pipeline")
    elif stage2_mode == "none":
        stage2 = copy.deepcopy(stage1)
        structured_diag = {
            "z_hat_history": [],
            "residuals_noisy_rmse": [],
            "updates": [],
            "ris_projection_total_s": 0.0,
        }
    else:
        raise ValueError(f"unknown stage2_mode {stage2_mode!r}")

    final_method = str(
        config.get("final_refinement_method", "global_exact_spherical_vp")
    ).lower()
    if final_method == "global_exact_spherical_vp":
        final = global_exact_spherical_vp_refinement(y_raw, stage2, scene, config)
        final["stage2_mode"] = stage2_mode
    elif final_method == "legacy_raw_vp":
        final = refine_global_raw(y_raw, scene, config, stage2)
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "legacy_raw_vp"
    elif final_method == "none":
        final = copy.deepcopy(stage2)
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "none"
    else:
        raise ValueError(f"unknown final_refinement_method {final_method!r}")

    return {
        "stage1": stage1,
        "stage2": stage2,
        "structured_diag": structured_diag,
        "final": final,
    }
