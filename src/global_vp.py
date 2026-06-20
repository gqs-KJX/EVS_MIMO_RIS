"""Raw-domain global exact-spherical variable projection refinement.

The default proposed estimator uses Stage-I for path assignment and nonlinear
initialization, then runs the legacy high-performing least_squares VP-WNLS
solver in the raw OFDM tensor. That solver refines UE position, common clock
offset, and EVS polarization angles while eliminating path gains by variable
projection. The reduced L-BFGS-B scalar VP path is retained only as an
experimental ablation. Legacy factor-domain EVS/delay/RIS projections remain
separate ablation baselines.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .geometry import elev_az_from_unit_vector, polarization_vector, unit_vector_from_elev_az
from .projections_delay import tau_from_pole
from .utils import bounded_coordinate_search, scipy_is_available, solve_lstsq


def _global_vp_config(config: dict) -> dict:
    """Return global-VP options with conservative defaults."""
    defaults = {
        "solver": "least_squares",
        "mode": "adaptive_jones",
        "max_iter": 80,
        "ftol": 1.0e-12,
        "gtol": 1.0e-8,
        "beta_reg": 0.0,
        "evs_mode": "legacy_or_full_polarization",
        "jones_regularization_scaling": "gram",
        "jones_lambda0": 1.0,
        "jones_lambda_min": 1.0e-4,
        "jones_lambda_max": 1.0e8,
        "jones_snr_eps": 1.0e-12,
        "run_fixed_pol_anchor": True,
        "jones_leakage_threshold": 0.25,
        "jones_min_rel_improvement": 1.0e-3,
        "jones_tau": 0.25,
        "jones_tau_min": 1.0e-3,
        "jones_tau_max": 10.0,
        "jones_diagonal_loading": 1.0e-10,
        "gof_pfa": 0.05,
        "efim_lambda_min_threshold": 1.0e-8,
        "efim_cond_threshold": 1.0e12,
        "use_data_only_efim_gate": True,
        "use_delay_prior": False,
        "delay_prior_weight": 1.0,
        "delay_prior_sigma_s": 2.0e-11,
        "use_weight": False,
        "weight": None,
        "use_multistart": False,
        "num_perturb_starts": 0,
        "position_perturb_std_m": 0.05,
        "clock_perturb_std_s": 1.0e-10,
        "use_trust_region": False,
        "position_trust_radius_m": 0.3,
        "clock_trust_radius_s": 3.0e-10,
        "objective_rollback_tolerance": 1.0e-12,
        "overwrite_factor_keys": False,
        "finite_difference_check": False,
        "use_analytic_jacobian": True,
        "matrix_free_beta": False,
        "enable_z_rescue_multistart": True,
        "z_rescue_num_starts": 7,
        "z_rescue_trigger": "boundary_or_unreliable",
        "z_rescue_keep_xy": True,
        "z_rescue_margin_m": 0.02,
        "boundary_tol_m": 0.02,
        "boundary_accept_rel_tol": 1.0e-3,
    }
    options = dict(defaults)
    options.update(dict(config.get("global_vp", {})))
    return options


def distance_to_box_boundary(
    p: np.ndarray,
    bounds: np.ndarray,
    boundary_tol_m: float = 0.02,
) -> dict:
    """Return distances from a position to each face of a 3-D box."""
    position = np.asarray(p, dtype=float).reshape(3)
    box = np.asarray(bounds, dtype=float)
    if box.shape != (3, 2):
        raise ValueError("position bounds must have shape (3, 2)")
    per_axis = np.minimum(position - box[:, 0], box[:, 1] - position)
    labels = ("x", "y", "z")
    hit_axes = [labels[idx] for idx, value in enumerate(per_axis) if value <= boundary_tol_m]
    return {
        "boundary_hit": bool(hit_axes),
        "boundary_hit_axis": hit_axes[0] if len(hit_axes) == 1 else hit_axes,
        "distance_to_position_box_boundary_m": float(np.min(per_axis)),
        "min_distance_per_axis": {
            label: float(per_axis[idx]) for idx, label in enumerate(labels)
        },
        "boundary_tol_m": float(boundary_tol_m),
    }


def z_rescue_starts(
    current_start: np.ndarray,
    bounds: np.ndarray,
    num_starts: int = 7,
    margin_m: float = 0.02,
) -> list[np.ndarray]:
    """Build deterministic starts spanning the interior of the UE z box."""
    start = np.asarray(current_start, dtype=float).reshape(3)
    box = np.asarray(bounds, dtype=float)
    if box.shape != (3, 2):
        raise ValueError("position bounds must have shape (3, 2)")
    count = max(1, int(num_starts))
    z_low = float(box[2, 0] + margin_m)
    z_high = float(box[2, 1] - margin_m)
    if z_low > z_high:
        z_low = z_high = float(np.mean(box[2]))
    starts = []
    for z_value in np.linspace(z_low, z_high, count):
        candidate = np.clip(start.copy(), box[:, 0], box[:, 1])
        candidate[2] = z_value
        starts.append(candidate)
    return starts


def select_z_rescue_candidate(
    candidates: list[dict],
    bounds: np.ndarray,
    *,
    boundary_tol_m: float = 0.02,
    boundary_accept_rel_tol: float = 1.0e-3,
) -> tuple[dict, str]:
    """Select by raw objective, using interior status only as a close-score tie break."""
    finite = [
        candidate
        for candidate in candidates
        if np.isfinite(float(candidate.get("raw_objective_final", np.nan)))
    ]
    if not finite:
        return candidates[0], "no_finite_candidate_score"
    best = min(finite, key=lambda candidate: float(candidate["raw_objective_final"]))
    best_score = float(best["raw_objective_final"])
    close_limit = best_score * (1.0 + float(boundary_accept_rel_tol)) + 1.0e-15
    close_interior = [
        candidate
        for candidate in finite
        if float(candidate["raw_objective_final"]) <= close_limit
        and not distance_to_box_boundary(
            candidate["p_u"], bounds, boundary_tol_m
        )["boundary_hit"]
    ]
    if close_interior:
        selected = min(
            close_interior, key=lambda candidate: float(candidate["raw_objective_final"])
        )
        if selected is not best:
            return selected, "interior_within_boundary_accept_rel_tol"
    return best, "lowest_raw_objective"


def _global_vp_mode(config: dict) -> str:
    """Return the Stage-III VP mode, accepting legacy evs_mode aliases."""
    options = _global_vp_config(config)
    mode = str(options.get("mode", "adaptive_jones"))
    if mode not in {"adaptive_jones", "jones_regularized", "jones_free", "fixed_pol"}:
        # Backward compatibility for old experiments that selected the basis
        # through evs_mode before the paper algorithm exposed vp mode directly.
        evs_mode = str(options.get("evs_mode", "legacy_or_full_polarization"))
        if evs_mode == "linear_polarization_basis":
            mode = "jones_free"
        elif evs_mode in {"legacy_or_full_polarization", "fixed_stage1_A"}:
            mode = "fixed_pol"
        else:
            raise ValueError(f"unknown global_vp mode {mode!r}")
    return mode


def _build_global_vp_cache(scene: dict) -> dict:
    """Build reusable static arrays for global VP diagnostics and future fast paths."""
    kappa = 2.0 * np.pi / scene["wavelength"]
    omega_arb = [
        np.asarray(scene["Omega"][k], dtype=complex)
        * np.asarray(scene["a_RB"][k], dtype=complex)[None, :]
        for k in range(scene["K"])
    ]
    return {
        "rho": np.asarray(scene["ris_grid"], dtype=float),
        "n_idx": np.arange(scene["N"], dtype=float),
        "Omega_aRB": omega_arb,
        "R_GR": [np.asarray(scene["rotations"][k], dtype=float) for k in range(scene["K"])],
        "ris_centers": np.asarray(scene["ris_centers"], dtype=float),
        "d_RB": np.asarray(scene["d_RB"], dtype=float),
        "wavelength": float(scene["wavelength"]),
        "kappa": float(kappa),
    }


def _inverse_assignment(column_to_panel: np.ndarray, k_paths: int) -> np.ndarray:
    """Return panel-to-column inverse for a column-to-panel assignment."""
    panel_to_column = np.empty(k_paths, dtype=int)
    for column, panel in enumerate(np.asarray(column_to_panel, dtype=int).reshape(-1)):
        if column >= k_paths:
            break
        panel_to_column[int(panel)] = int(column)
    return panel_to_column


def _get_panel_ordered_stage1_factors(init_estimate: dict, scene: dict) -> dict:
    """Return Stage-I factors in physical RIS-panel order.

    Stage-I may carry raw CP columns plus an assignment. The global raw-domain
    VP dictionary is physical: path k means RIS panel k. If a panel-to-column
    mapping is supplied and columns are not explicitly marked as already panel
    ordered, this helper reorders all per-path Stage-I factors into that
    physical convention.
    """
    k_paths = int(scene["K"])
    a_raw = np.asarray(init_estimate["A"], dtype=complex)
    poles_raw = np.asarray(init_estimate["poles"], dtype=complex).reshape(-1)
    ris_eta_raw = np.asarray(init_estimate["ris_eta"], dtype=float)
    c_raw = init_estimate.get("C")
    c_raw_arr = None if c_raw is None else np.asarray(c_raw, dtype=complex)
    gamma_raw = init_estimate.get("gamma")
    gamma_raw_arr = None if gamma_raw is None else np.asarray(gamma_raw, dtype=float)
    eta_pol_raw = init_estimate.get("eta_pol")
    eta_pol_raw_arr = (
        None if eta_pol_raw is None else np.asarray(eta_pol_raw, dtype=float)
    )

    columns_are_panel_ordered = bool(
        init_estimate.get("columns_are_panel_ordered", False)
    )
    panel_to_column = init_estimate.get("panel_to_column_assignment")
    if panel_to_column is None:
        panel_to_column = init_estimate.get("panel_to_column")
    reported_panel_to_column = None
    if panel_to_column is None and "assignment" in init_estimate:
        column_to_panel = np.asarray(init_estimate["assignment"], dtype=int)
        if column_to_panel.size == k_paths:
            reported_panel_to_column = _inverse_assignment(column_to_panel, k_paths)
            if columns_are_panel_ordered:
                panel_to_column = reported_panel_to_column

    used_panel_to_column = False
    if panel_to_column is not None and not columns_are_panel_ordered:
        order = np.asarray(panel_to_column, dtype=int).reshape(k_paths)
        used_panel_to_column = True
    else:
        order = np.arange(k_paths, dtype=int)
    if reported_panel_to_column is None:
        reported_panel_to_column = (
            np.asarray(panel_to_column, dtype=int).reshape(k_paths)
            if panel_to_column is not None
            else order
        )

    a_phys = a_raw[:, order]
    poles_phys = poles_raw[order]
    ris_eta_phys = ris_eta_raw[order]
    c_phys = None if c_raw_arr is None else c_raw_arr[:, order]
    gamma_phys = None if gamma_raw_arr is None else gamma_raw_arr[order]
    eta_pol_phys = None if eta_pol_raw_arr is None else eta_pol_raw_arr[order]
    tau_phys = np.array(
        [tau_from_pole(pole, scene["delta_f"]) for pole in poles_phys],
        dtype=float,
    )

    return {
        "A_phys": a_phys,
        "poles_phys": poles_phys,
        "tau_phys": tau_phys,
        "ris_eta_phys": ris_eta_phys,
        "C_phys": c_phys,
        "gamma_phys": gamma_phys,
        "eta_pol_phys": eta_pol_phys,
        "global_vp_used_panel_to_column": bool(used_panel_to_column),
        "global_vp_panel_to_column": np.asarray(
            reported_panel_to_column, dtype=int
        ).tolist(),
        "global_vp_columns_are_panel_ordered": bool(columns_are_panel_ordered),
    }


def _score_initial_position_candidate(
    p_candidate: np.ndarray,
    panel_positions: np.ndarray,
    tau_stage1: np.ndarray,
    scene: dict,
) -> tuple[float, float]:
    """Score a UE-position candidate by clock consistency and geometry spread."""
    ranges = np.empty(scene["K"], dtype=float)
    for k in range(scene["K"]):
        q_vec = scene["rotations"][k] @ (p_candidate - scene["ris_centers"][k])
        ranges[k] = np.linalg.norm(q_vec)
    delta_t_values = tau_stage1 - (ranges + scene["d_RB"]) / scene["c0"]
    clock_mad = float(np.median(np.abs(delta_t_values - np.median(delta_t_values))))
    geometry_residual_s = float(
        np.median(np.linalg.norm(panel_positions - p_candidate[None, :], axis=1))
        / scene["c0"]
    )
    return clock_mad + geometry_residual_s, float(np.median(delta_t_values))


def _initial_xi_from_stage1_with_diagnostics(
    init_estimate: dict,
    scene: dict,
    config: dict,
    stage1_factors: dict,
) -> tuple[np.ndarray, dict]:
    """Build xi0 = [p_x, p_y, p_z, Delta_t] using robust Stage-I fusion."""
    panel_positions = []
    ris_eta = np.asarray(stage1_factors["ris_eta_phys"], dtype=float)
    for k in range(scene["K"]):
        range_m, elevation, azimuth = ris_eta[k]
        direction_local = unit_vector_from_elev_az(elevation, azimuth)
        q_local = range_m * direction_local
        p_candidate = scene["ris_centers"][k] + scene["rotations"][k].T @ q_local
        panel_positions.append(p_candidate)
    panel_positions = np.asarray(panel_positions, dtype=float)
    tau_stage1 = np.asarray(stage1_factors["tau_phys"], dtype=float)

    candidates: list[tuple[str, np.ndarray, float | None]] = [
        ("all_panel_mean", np.mean(panel_positions, axis=0), None),
        ("all_panel_median", np.median(panel_positions, axis=0), None),
    ]
    if scene["K"] > 1:
        for leave_out in range(scene["K"]):
            keep = [idx for idx in range(scene["K"]) if idx != leave_out]
            candidates.append(
                (
                    f"leave_one_out_mean_without_panel_{leave_out}",
                    np.mean(panel_positions[keep], axis=0),
                    None,
                )
            )
    if "p_u" in init_estimate and "delta_t" in init_estimate:
        candidates.append(
            (
                "provided_p_u_delta_t",
                np.asarray(init_estimate["p_u"], dtype=float).reshape(3),
                float(init_estimate["delta_t"]),
            )
        )

    scored_candidates = []
    for name, p_candidate, forced_dt in candidates:
        score, dt_candidate = _score_initial_position_candidate(
            p_candidate, panel_positions, tau_stage1, scene
        )
        if forced_dt is not None:
            dt_candidate = float(forced_dt)
        scored_candidates.append(
            {
                "name": name,
                "score": float(score),
                "delta_t_s": float(dt_candidate),
                "p_u": np.asarray(p_candidate, dtype=float).copy(),
            }
        )
    selected = min(scored_candidates, key=lambda item: item["score"])
    p_init = np.asarray(selected["p_u"], dtype=float)
    dt_init = float(selected["delta_t_s"])
    if "_global_vp_initial_p_u" in init_estimate:
        p_init = np.asarray(
            init_estimate["_global_vp_initial_p_u"], dtype=float
        ).reshape(3)
        selected = {
            "name": "forced_non_oracle_multistart",
            "score": float("nan"),
            "delta_t_s": float(
                init_estimate.get("_global_vp_initial_delta_t", dt_init)
            ),
            "p_u": p_init.copy(),
        }
    if "_global_vp_initial_delta_t" in init_estimate:
        dt_init = float(init_estimate["_global_vp_initial_delta_t"])
    diagnostics = {
        "global_vp_init_method": "robust_stage1_ris_eta_clock_consistency",
        "global_vp_init_candidate_scores": [
            {
                "name": item["name"],
                "score": item["score"],
                "delta_t_s": item["delta_t_s"],
                "p_u": item["p_u"].copy(),
            }
            for item in scored_candidates
        ],
        "global_vp_init_selected_candidate": selected["name"],
    }

    ue_bounds = np.asarray(config["ue_bounds"], dtype=float)
    dt_bounds = np.asarray(config["delta_t_bounds"], dtype=float)
    p_init = np.clip(p_init, ue_bounds[:, 0], ue_bounds[:, 1])
    dt_init = float(np.clip(dt_init, dt_bounds[0], dt_bounds[1]))
    return np.concatenate([p_init, np.array([dt_init])]), diagnostics


def _initial_xi_from_stage1(init_estimate: dict, scene: dict, config: dict) -> np.ndarray:
    """Build xi0 = [p_x, p_y, p_z, Delta_t] from Stage-I geometry."""
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
    xi0, _ = _initial_xi_from_stage1_with_diagnostics(
        init_estimate, scene, config, stage1_factors
    )
    return xi0


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


def _evs_linear_polarization_basis(scene: dict, path: int) -> np.ndarray:
    """Return EVS basis columns for the two linear polarization components."""
    theta = np.asarray(scene["Theta"][path], dtype=complex)
    basis_1 = np.kron(scene["v_B"][path], theta[:, 0])
    basis_2 = np.kron(scene["v_B"][path], theta[:, 1])
    return np.column_stack([basis_1, basis_2])


def _evs_legacy_full_polarization_atom(stage1_factors: dict, scene: dict, path: int) -> np.ndarray:
    """Return the old full-polarization EVS atom for one path."""
    gamma = stage1_factors.get("gamma_phys")
    eta_pol = stage1_factors.get("eta_pol_phys")
    if gamma is None or eta_pol is None:
        return stage1_factors["A_phys"][:, path : path + 1]
    pol = scene["Theta"][path] @ polarization_vector(float(gamma[path]), float(eta_pol[path]))
    return np.kron(scene["v_B"][path], pol)[:, None]


def _evs_atom_bases(init_estimate: dict, scene: dict, config: dict) -> tuple[list[np.ndarray], str]:
    """Return per-path EVS basis matrices for the configured global VP mode."""
    options = _global_vp_config(config)
    vp_mode = _global_vp_mode(config)
    if vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"}:
        return [_evs_linear_polarization_basis(scene, k) for k in range(scene["K"])], vp_mode
    if vp_mode == "fixed_pol":
        return [
            _evs_legacy_full_polarization_atom(
                _get_panel_ordered_stage1_factors(init_estimate, scene), scene, k
            )
            for k in range(scene["K"])
        ], vp_mode
    evs_mode = str(options.get("evs_mode", "legacy_or_full_polarization"))
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
    if evs_mode == "legacy_or_full_polarization":
        return [
            _evs_legacy_full_polarization_atom(stage1_factors, scene, k)
            for k in range(scene["K"])
        ], evs_mode
    if evs_mode == "fixed_stage1_A":
        a_phys = stage1_factors["A_phys"]
        return [a_phys[:, k : k + 1] for k in range(scene["K"])], evs_mode
    if evs_mode == "linear_polarization_basis":
        return [_evs_linear_polarization_basis(scene, k) for k in range(scene["K"])], evs_mode
    raise ValueError(f"unknown global_vp evs_mode {evs_mode!r}")


def _build_global_dictionary(
    xi: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
    need_jacobian: bool = False,
) -> tuple[np.ndarray, dict]:
    """Build exact-spherical raw dictionary Phi(xi) with optional Jacobian."""
    xi = np.asarray(xi, dtype=float).reshape(4)
    p_u = xi[:3]
    delta_t = float(xi[3])
    evs_bases, evs_mode = _evs_atom_bases(init_estimate, scene, config)
    k_paths = scene["K"]
    i_dim = scene["I"]
    n_dim = scene["N"]
    t_dim = scene["T"]
    kappa = 2.0 * np.pi / scene["wavelength"]
    n_idx = np.arange(n_dim, dtype=float)
    num_atoms = int(sum(basis.shape[1] for basis in evs_bases))

    phi = np.empty((i_dim * n_dim * t_dim, num_atoms), dtype=complex)
    d_mat = np.empty((n_dim, k_paths), dtype=complex)
    c_mat = np.empty((t_dim, k_paths), dtype=complex)
    poles = np.empty(k_paths, dtype=complex)
    ranges = np.empty(k_paths, dtype=float)
    elevations = np.empty(k_paths, dtype=float)
    azimuths = np.empty(k_paths, dtype=float)
    tau = np.empty(k_paths, dtype=float)
    q_local = np.empty((k_paths, 3), dtype=float)
    atoms = []
    path_for_atom = np.empty(num_atoms, dtype=int)
    basis_for_atom = np.empty(num_atoms, dtype=int)
    dtau_dx_all = np.empty((k_paths, 4), dtype=float) if need_jacobian else None
    dphi_dx = [
        np.empty((i_dim * n_dim * t_dim, num_atoms), dtype=complex)
        for _ in range(4)
    ] if need_jacobian else None

    atom_col = 0
    for k in range(k_paths):
        rotation = scene["rotations"][k]
        q_vec = rotation @ (p_u - scene["ris_centers"][k])
        range_m = float(np.linalg.norm(q_vec))
        if range_m <= config.get("eps", 1.0e-10):
            range_m = config.get("eps", 1.0e-10)
        q_local[k] = q_vec
        ranges[k] = range_m
        elev, az = elev_az_from_unit_vector(q_vec / range_m)
        elevations[k] = elev
        azimuths[k] = az

        tau_k = (range_m + scene["d_RB"][k]) / scene["c0"] + delta_t
        tau[k] = tau_k
        pole = np.exp(-1j * 2.0 * np.pi * scene["delta_f"] * tau_k)
        poles[k] = pole
        d_vec = pole ** np.arange(n_dim)
        d_mat[:, k] = d_vec

        rho = scene["ris_grid"]
        diff = q_vec[None, :] - rho
        dist_elem = np.linalg.norm(diff, axis=1)
        safe_dist = np.maximum(dist_elem, config.get("eps", 1.0e-10))
        delta = dist_elem - range_m
        u_vec = np.exp(-1j * kappa * delta)
        c_vec = scene["Omega"][k] @ (scene["a_RB"][k] * u_vec)
        c_mat[:, k] = c_vec

        evs_basis = evs_bases[k]
        for basis_idx in range(evs_basis.shape[1]):
            a_vec = evs_basis[:, basis_idx]
            atom_tensor = (
                a_vec[:, None, None]
                * d_vec[None, :, None]
                * c_vec[None, None, :]
            )
            phi[:, atom_col] = atom_tensor.reshape(-1)
            atoms.append(phi[:, atom_col])
            path_for_atom[atom_col] = k
            basis_for_atom[atom_col] = basis_idx
            atom_col += 1

        if need_jacobian and dphi_dx is not None:
            dr_dp = (q_vec / range_m) @ rotation / scene["c0"]
            dtau_dx = np.concatenate([dr_dp, np.array([1.0])])
            if dtau_dx_all is not None:
                dtau_dx_all[k] = dtau_dx
            geom_grad = diff / safe_dist[:, None] - q_vec[None, :] / range_m
            ddelta_dp = geom_grad @ rotation
            du_dp = -1j * kappa * u_vec[:, None] * ddelta_dp
            dc_dp = np.empty((3, t_dim), dtype=complex)
            for dim in range(3):
                dc_dp[dim] = scene["Omega"][k] @ (scene["a_RB"][k] * du_dp[:, dim])

            first_atom_for_path = atom_col - evs_basis.shape[1]
            for basis_idx in range(evs_basis.shape[1]):
                a_vec = evs_basis[:, basis_idx]
                col = first_atom_for_path + basis_idx
                for dim in range(4):
                    dd_dx = (
                        -1j
                        * 2.0
                        * np.pi
                        * scene["delta_f"]
                        * n_idx
                        * d_vec
                        * dtau_dx[dim]
                    )
                    if dim < 3:
                        d_atom = (
                            a_vec[:, None, None]
                            * dd_dx[None, :, None]
                            * c_vec[None, None, :]
                        ) + (
                            a_vec[:, None, None]
                            * d_vec[None, :, None]
                            * dc_dp[dim][None, None, :]
                        )
                    else:
                        d_atom = (
                            a_vec[:, None, None]
                            * dd_dx[None, :, None]
                            * c_vec[None, None, :]
                        )
                    dphi_dx[dim][:, col] = d_atom.reshape(-1)

    aux = {
        "q_local": q_local,
        "ranges": ranges,
        "elevations": elevations,
        "azimuths": azimuths,
        "tau": tau,
        "D": d_mat,
        "C": c_mat,
        "poles": poles,
        "atoms": atoms,
        "path_for_atom": path_for_atom,
        "basis_for_atom": basis_for_atom,
        "evs_mode": evs_mode,
        "evs_bases": evs_bases,
    }
    if need_jacobian:
        aux["dPhi_dx"] = dphi_dx
        aux["dtau_dx"] = dtau_dx_all
    return phi, aux


def build_jones_vp_dictionary(
    p_u: np.ndarray,
    delta_t: float,
    scene: dict,
    config: dict,
) -> np.ndarray:
    """Build Psi(p_u, Delta_t) for Stage-I-regularized Jones-VP.

    Columns are ordered as ``[path 0 basis 0, path 0 basis 1, path 1 basis 0, ...]``.
    The tensor vectorization is exactly the raw ``I x N x T`` order used by
    ``synthesize_raw_tensor`` and the legacy raw VP dictionary.
    """
    k_paths = int(scene["K"])
    dummy = {
        "A": np.zeros((scene["I"], k_paths), dtype=complex),
        "poles": np.ones(k_paths, dtype=complex),
        "ris_eta": np.zeros((k_paths, 3), dtype=float),
        "gamma": np.zeros(k_paths, dtype=float),
        "eta_pol": np.zeros(k_paths, dtype=float),
        "assignment": list(range(k_paths)),
        "panel_to_column_assignment": list(range(k_paths)),
        "columns_are_panel_ordered": True,
    }
    mode_config = copy.deepcopy(config)
    mode_config["global_vp"] = dict(mode_config.get("global_vp", {}))
    mode_config["global_vp"]["mode"] = "jones_free"
    psi, _ = _build_global_dictionary(
        np.r_[np.asarray(p_u, dtype=float).reshape(3), float(delta_t)],
        dummy,
        scene,
        mode_config,
    )
    return psi


def extract_stage1_jones_directions(
    stage1_estimate: dict,
    scene: dict,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Extract soft Stage-I Jones direction priors, shape K x 2.

    Stage-I estimates the EVS factor up to CPD scaling and noise. This helper
    projects that factor onto the known ``kron(v_B,k, Theta_k e_k)`` subspace
    and returns only a unit-norm direction prior, not a known true polarization.
    """
    k_paths = int(scene["K"])
    status = ["low_confidence_fallback"] * k_paths
    source = None
    for key in ("a_evs", "evs_factors", "A", "evs"):
        if key in stage1_estimate:
            arr = np.asarray(stage1_estimate[key], dtype=complex)
            if arr.shape == (scene["I"], k_paths):
                source = arr
                break
            if arr.shape == (k_paths, scene["I"]):
                source = arr.T
                break
    e0 = np.zeros((k_paths, 2), dtype=complex)
    if source is not None:
        if not bool(stage1_estimate.get("columns_are_panel_ordered", False)):
            panel_to_column = stage1_estimate.get("panel_to_column_assignment")
            if panel_to_column is not None:
                order = np.asarray(panel_to_column, dtype=int).reshape(k_paths)
                source = source[:, order]
        for k in range(k_paths):
            try:
                # a_EVS = kron(v_B, s), s = Theta e.  With NumPy's kron order,
                # reshape(M_A, 6).T gives columns proportional to s*v_B[m].
                a_matrix = source[:, k].reshape(scene["M_A"], 6).T
                s_hat = a_matrix @ np.conj(scene["v_B"][k]) / (
                    np.vdot(scene["v_B"][k], scene["v_B"][k]).real + eps
                )
                x0 = np.linalg.pinv(scene["Theta"][k], rcond=eps) @ s_hat
                norm = np.linalg.norm(x0)
                if np.isfinite(norm) and norm > eps:
                    e0[k] = x0 / norm
                    status[k] = "stage1_evs_factor"
                    continue
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
            gamma = stage1_estimate.get("gamma")
            eta_pol = stage1_estimate.get("eta_pol")
            if gamma is not None and eta_pol is not None:
                try:
                    e_fallback = polarization_vector(float(gamma[k]), float(eta_pol[k]))
                    e0[k] = e_fallback / max(np.linalg.norm(e_fallback), eps)
                    status[k] = "stage1_gamma_eta_fallback"
                    continue
                except (IndexError, TypeError, ValueError):
                    pass
            e0[k] = np.array([1.0, 0.0], dtype=complex)
    else:
        for k in range(k_paths):
            gamma = stage1_estimate.get("gamma")
            eta_pol = stage1_estimate.get("eta_pol")
            if gamma is not None and eta_pol is not None:
                try:
                    e_fallback = polarization_vector(float(gamma[k]), float(eta_pol[k]))
                    e0[k] = e_fallback / max(np.linalg.norm(e_fallback), eps)
                    status[k] = "stage1_gamma_eta_fallback"
                    continue
                except (IndexError, TypeError, ValueError):
                    pass
            e0[k] = np.array([1.0, 0.0], dtype=complex)
    extract_stage1_jones_directions.last_status = status
    return e0


extract_stage1_jones_directions.last_status = []


def _as_path_vector(value, k_paths, name="path_value", default=0.0):
    """Return a float vector with one value per path."""
    if value is None:
        value = default
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        arr = np.asarray(default, dtype=float)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return np.full(k_paths, float(flat[0]), dtype=float)
    if flat.size == k_paths:
        return flat.astype(float, copy=True)
    raise ValueError(
        f"{name} must be scalar or contain exactly {k_paths} path values; "
        f"got shape {arr.shape} with size {arr.size}"
    )


def _jones_regularizer(
    init_estimate: dict,
    scene: dict,
    config: dict,
    y_vec: np.ndarray | None = None,
    phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Return block-diagonal Jones prior Lambda and per-path rho values."""
    options = _global_vp_config(config)
    k_paths = int(scene["K"])
    e0 = extract_stage1_jones_directions(init_estimate, scene)
    status = list(getattr(extract_stage1_jones_directions, "last_status", []))
    if len(status) != k_paths:
        status = ["unknown"] * k_paths
    mode = _global_vp_mode(config)
    lambda_path = np.zeros(k_paths, dtype=float)
    if mode in {"adaptive_jones", "jones_regularized"}:
        if "jones_lambda_per_path" in init_estimate:
            lambda_path = _as_path_vector(
                init_estimate["jones_lambda_per_path"],
                k_paths,
                name="jones_lambda_per_path",
            )
        else:
            lambda_path = _as_path_vector(
                options.get("jones_lambda0", 1.0),
                k_paths,
                name="jones_lambda0",
                default=1.0,
            )
    if mode == "jones_regularized" and "jones_lambda_per_path" not in init_estimate:
        tau = _as_path_vector(
            init_estimate.get("stage1_jones_tau", None),
            k_paths,
            name="stage1_jones_tau",
            default=options.get("jones_tau", 0.25),
        )
        tau = np.clip(
            tau,
            float(options.get("jones_tau_min", 1.0e-3)),
            float(options.get("jones_tau_max", 10.0)),
        )
        tau_ref = float(options.get("jones_tau", 0.25))
        lambda_path *= (tau_ref**2) / (tau**2 + float(options.get("jones_snr_eps", 1.0e-12)))
    elif mode == "jones_free":
        lambda_path[:] = 0.0

    lambda_path = np.clip(
        lambda_path,
        float(options.get("jones_lambda_min", 1.0e-4)),
        float(options.get("jones_lambda_max", 1.0e8)),
    )
    if mode == "jones_free":
        lambda_path[:] = 0.0

    gram_scale = np.ones(k_paths, dtype=float)
    if str(options.get("jones_regularization_scaling", "gram")) == "gram" and phi is not None:
        for k in range(k_paths):
            psi_k = phi[:, 2 * k : 2 * k + 2]
            gram_scale[k] = 0.5 * float(np.trace(psi_k.conj().T @ psi_k).real)
    rho = lambda_path * gram_scale

    lam = np.zeros((2 * k_paths, 2 * k_paths), dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for k in range(k_paths):
        e = e0[k].reshape(2, 1)
        denom = np.vdot(e[:, 0], e[:, 0]).real
        if denom <= 0.0:
            p_perp = np.diag([0.0, 1.0]).astype(complex)
        else:
            p_perp = eye2 - (e @ e.conj().T) / denom
        lam[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = rho[k] * p_perp
    loading = float(options.get("jones_diagonal_loading", 1.0e-10))
    return lam, rho, lambda_path, status, loading


def _solve_linear_vp_regularized(
    phi: np.ndarray,
    y_vec: np.ndarray,
    regularizer: np.ndarray | None,
    diagonal_loading: float,
) -> tuple[np.ndarray, dict]:
    """Solve closed-form regularized LS for the current nonlinear state."""
    gram_data = phi.conj().T @ phi
    rhs = phi.conj().T @ y_vec
    gram = gram_data.copy()
    if regularizer is not None:
        gram += np.asarray(regularizer, dtype=complex)
    if diagonal_loading > 0.0:
        gram += float(diagonal_loading) * np.eye(phi.shape[1], dtype=complex)
    try:
        coeff = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.pinv(gram) @ rhs
    try:
        trace_h = float(np.trace(np.linalg.solve(gram, gram_data)).real)
    except np.linalg.LinAlgError:
        trace_h = float(np.trace(np.linalg.pinv(gram) @ gram_data).real)
    singular = np.linalg.svd(gram_data, compute_uv=False)
    rank = int(np.sum(singular > max(gram_data.shape) * np.finfo(float).eps * singular[0])) if singular.size else 0
    if singular.size and singular[-1] > 0.0:
        cond = float(singular[0] / singular[-1])
    else:
        cond = float("inf")
    return coeff, {
        "condition_number_gram": cond,
        "rank_gram": rank,
        "trace_H": trace_h,
        "gram_singular_values": singular,
    }


def data_only_efim_diagnostic(
    y_raw: np.ndarray,
    p_u: np.ndarray,
    delta_t: float,
    init_estimate: dict,
    scene: dict,
    config: dict,
    sigma2: float | None = None,
) -> dict:
    """Compute the data-only projected-Jacobian EFIM around a VP estimate."""
    y_vec = np.asarray(y_raw, dtype=complex).reshape(-1)
    efim_config = copy.deepcopy(config)
    efim_config["global_vp"] = dict(efim_config.get("global_vp", {}))
    efim_config["global_vp"]["mode"] = "jones_free"
    xi = np.r_[np.asarray(p_u, dtype=float).reshape(3), float(delta_t)]
    phi, aux = _build_global_dictionary(
        xi, init_estimate, scene, efim_config, need_jacobian=True
    )
    coeff, linear_diag = _solve_linear_vp_regularized(phi, y_vec, None, 0.0)
    d_model = np.column_stack([dphi @ coeff for dphi in aux["dPhi_dx"]])
    try:
        nuisance_fit = phi @ np.linalg.lstsq(phi, d_model, rcond=None)[0]
    except np.linalg.LinAlgError:
        nuisance_fit = phi @ (np.linalg.pinv(phi) @ d_model)
    projected = d_model - nuisance_fit
    if sigma2 is None or not np.isfinite(float(sigma2)) or float(sigma2) <= 0.0:
        residual = y_vec - phi @ coeff
        sigma2 = float(np.vdot(residual, residual).real / max(y_vec.size, 1))
    j_eq = (2.0 / max(float(sigma2), config.get("eps", 1.0e-10))) * np.real(
        projected.conj().T @ projected
    )
    scale = np.diag([1.0, 1.0, 1.0, 1.0 / float(scene["c0"])])
    j_eq_scaled = scale.T @ j_eq @ scale
    eigvals = np.linalg.eigvalsh((j_eq + j_eq.T) * 0.5)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals_scaled = np.maximum(
        np.linalg.eigvalsh((j_eq_scaled + j_eq_scaled.T) * 0.5), 0.0
    )
    positive = eigvals[eigvals > 0.0]
    positive_scaled = eigvals_scaled[eigvals_scaled > 0.0]
    lambda_min = float(eigvals[0]) if eigvals.size else float("nan")
    lambda_min_scaled = float(eigvals_scaled[0]) if eigvals_scaled.size else float("nan")
    condition = (
        float(eigvals[-1] / positive[0])
        if eigvals.size and positive.size
        else float("inf")
    )
    condition_scaled = (
        float(eigvals_scaled[-1] / positive_scaled[0])
        if eigvals_scaled.size and positive_scaled.size
        else float("inf")
    )
    return {
        "data_only_efim": j_eq,
        "data_only_efim_parameter_order": [
            "p_x_m",
            "p_y_m",
            "p_z_m",
            "delta_t_s",
        ],
        "data_only_efim_clock_eliminated": False,
        "data_only_efim_eigvals": eigvals,
        "data_only_efim_lambda_min": lambda_min,
        "data_only_efim_condition_number": condition,
        "data_only_scaled_efim": j_eq_scaled,
        "data_only_scaled_efim_parameter_order": [
            "p_x_m",
            "p_y_m",
            "p_z_m",
            "c_delta_t_m",
        ],
        "data_only_scaled_efim_clock_eliminated": False,
        "data_only_scaled_efim_eigvals": eigvals_scaled,
        "data_only_scaled_efim_lambda_min": lambda_min_scaled,
        "data_only_scaled_efim_condition_number": condition_scaled,
        "data_only_linear_condition_number_gram": linear_diag["condition_number_gram"],
        "data_only_rank_gram": linear_diag["rank_gram"],
    }


def _apply_weight(vec: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    """Apply identity, diagonal, or dense sample weight."""
    if weight is None:
        return vec
    weight_arr = np.asarray(weight)
    if weight_arr.ndim == 1:
        if weight_arr.shape[0] != vec.shape[0]:
            raise ValueError("weight vector length must match y_vec length")
        if vec.ndim == 1:
            return weight_arr * vec
        return weight_arr[:, None] * vec
    if weight_arr.ndim == 2:
        if weight_arr.shape != (vec.shape[0], vec.shape[0]):
            raise ValueError("weight matrix must have shape M x M")
        return weight_arr @ vec
    raise ValueError("weight must be None, vector, or matrix")


def _weighted_inner(x_vec: np.ndarray, y_vec: np.ndarray, weight: np.ndarray | None) -> complex:
    """Return x^H W y."""
    return np.vdot(x_vec, _apply_weight(y_vec, weight))


def _solve_beta_vp(
    phi: np.ndarray,
    y_vec: np.ndarray,
    weight: np.ndarray | None,
    beta_reg: float,
    objective_scale: float = 1.0,
) -> np.ndarray:
    """Solve the weighted variable-projection path-gain subproblem."""
    k_paths = phi.shape[1]
    weighted_phi = _apply_weight(phi, weight)
    scale = float(objective_scale)
    gram = (
        scale * (phi.conj().T @ weighted_phi)
        + float(beta_reg) * np.eye(k_paths, dtype=complex)
    )
    rhs = scale * (phi.conj().T @ _apply_weight(y_vec, weight))
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _objective_weight_from_config(config: dict, y_size: int) -> np.ndarray | None:
    options = _global_vp_config(config)
    if not bool(options.get("use_weight", False)):
        return None
    weight = options.get("weight")
    if weight is None:
        return None
    weight_arr = np.asarray(weight, dtype=float)
    if weight_arr.ndim == 1 and weight_arr.size != y_size:
        raise ValueError("global_vp weight vector length does not match raw tensor size")
    return weight_arr


def _delay_prior_enabled(config: dict) -> bool:
    options = _global_vp_config(config)
    return bool(options.get("use_delay_prior", True))


def _vp_objective_parts_and_grad(
    xi: np.ndarray,
    y_vec: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> tuple[dict, np.ndarray]:
    """Return weighted VP objective parts and analytic total gradient."""
    options = _global_vp_config(config)
    vp_mode = _global_vp_mode(config)
    beta_reg = float(options.get("beta_reg", 0.0))
    weight = _objective_weight_from_config(config, y_vec.size)
    objective_scale = 1.0 / float(y_vec.size)
    phi, aux = _build_global_dictionary(
        xi, init_estimate, scene, config, need_jacobian=True
    )
    regularizer = None
    jones_rho = np.zeros(scene["K"], dtype=float)
    jones_prior_status: list[str] = []
    diagonal_loading = 0.0
    linear_diag = {}
    lambda_jones = np.zeros(scene["K"], dtype=float)
    if vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"}:
        regularizer, jones_rho, lambda_jones, jones_prior_status, diagonal_loading = _jones_regularizer(
            init_estimate, scene, config, y_vec, phi
        )
        beta, linear_diag = _solve_linear_vp_regularized(
            phi, y_vec, regularizer, diagonal_loading
        )
    else:
        beta = _solve_beta_vp(phi, y_vec, weight, beta_reg, objective_scale)
    residual = y_vec - phi @ beta
    raw_objective = float(
        objective_scale * np.real(_weighted_inner(residual, residual, weight))
    )
    beta_reg_objective = 0.0
    if beta_reg > 0.0:
        beta_reg_objective = float(beta_reg * np.real(np.vdot(beta, beta)))
    jones_regularizer_objective = 0.0
    if regularizer is not None:
        jones_regularizer_objective = float(
            objective_scale * np.real(np.vdot(beta, regularizer @ beta))
        )

    grad = np.empty(4, dtype=float)
    for dim, dphi in enumerate(aux["dPhi_dx"]):
        d_model = dphi @ beta
        grad[dim] = -2.0 * objective_scale * float(
            np.real(_weighted_inner(residual, d_model, weight))
        )

    delay_prior_objective = 0.0
    if _delay_prior_enabled(config):
        stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
        tau_stage1 = np.asarray(stage1_factors["tau_phys"], dtype=float)
        sigma_tau = float(options.get("delay_prior_sigma_s", 2.0e-11))
        lambda_tau = float(options.get("delay_prior_weight", 1.0))
        tau_err = np.asarray(aux["tau"], dtype=float) - tau_stage1
        delay_prior_objective = float(
            lambda_tau * np.sum((tau_err / sigma_tau) ** 2)
        )
        dtau_dx = np.asarray(aux["dtau_dx"], dtype=float)
        grad += 2.0 * lambda_tau * (
            (tau_err / (sigma_tau**2))[:, None] * dtau_dx
        ).sum(axis=0)

    total_objective = (
        raw_objective
        + beta_reg_objective
        + delay_prior_objective
        + jones_regularizer_objective
    )
    parts = {
        "raw_objective": raw_objective,
        "beta_reg_objective": beta_reg_objective,
        "jones_regularizer_objective": jones_regularizer_objective,
        "delay_prior_objective": delay_prior_objective,
        "total_objective": float(total_objective),
        "beta": beta,
        "residual": residual,
        "aux": aux,
        "vp_mode": vp_mode,
        "jones_rho": jones_rho,
        "lambda_jones_per_path": lambda_jones,
        "jones_prior_status": jones_prior_status,
        "linear_diagnostics": linear_diag,
    }
    return parts, grad


def _vp_objective_parts(
    xi: np.ndarray,
    y_vec: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    parts, _ = _vp_objective_parts_and_grad(xi, y_vec, init_estimate, scene, config)
    return parts


def _vp_objective_and_grad(
    xi: np.ndarray,
    y_vec: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> tuple[float, np.ndarray]:
    """Return total weighted VP objective and analytic gradient."""
    parts, grad = _vp_objective_parts_and_grad(xi, y_vec, init_estimate, scene, config)
    return float(parts["total_objective"]), grad


def _vp_objective_only(
    xi: np.ndarray,
    y_vec: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> float:
    return _vp_objective_and_grad(xi, y_vec, init_estimate, scene, config)[0]


def _candidate_starts(xi0: np.ndarray, lower: np.ndarray, upper: np.ndarray, config: dict) -> list[np.ndarray]:
    """Return xi starts; optional perturb starts are deterministic."""
    options = _global_vp_config(config)
    starts = [np.clip(np.asarray(xi0, dtype=float), lower, upper)]
    if not bool(options.get("use_multistart", False)):
        return starts
    num_starts = int(options.get("num_perturb_starts", 0))
    if num_starts <= 0:
        return starts
    rng = np.random.default_rng(int(config.get("seed", 0)) + 9173)
    pos_std = float(options.get("position_perturb_std_m", 0.05))
    clock_std = float(options.get("clock_perturb_std_s", 1.0e-10))
    for _ in range(num_starts):
        perturb = np.concatenate(
            [rng.normal(scale=pos_std, size=3), np.array([rng.normal(scale=clock_std)])]
        )
        starts.append(np.clip(xi0 + perturb, lower, upper))
    return starts


def _trust_region_enabled(config: dict, options: dict) -> bool:
    value = options.get("use_trust_region", True)
    if value is None or value == "auto":
        if "SNR_dB" in config:
            return bool(float(config["SNR_dB"]) <= 5.0)
        return True
    return bool(value)


def _intersect_trust_region_bounds(
    xi0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: dict,
    options: dict,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Intersect global bounds with optional local trust-region bounds."""
    if not _trust_region_enabled(config, options):
        return lower, upper, False
    position_radius = float(options.get("position_trust_radius_m", 0.3))
    clock_radius = float(options.get("clock_trust_radius_s", 3.0e-10))
    radius = np.array([position_radius, position_radius, position_radius, clock_radius])
    trust_lower = xi0 - radius
    trust_upper = xi0 + radius
    return np.maximum(lower, trust_lower), np.minimum(upper, trust_upper), True


def _global_exact_spherical_vp_refinement_lbfgsb_reduced(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Experimental reduced scalar VP solved by L-BFGS-B.

    This is retained only for ablation. The default numerical solver is the old
    least_squares VP-WNLS implementation.
    """
    assert y_raw.shape == (scene["I"], scene["N"], scene["T"])
    options = _global_vp_config(config)
    beta_reg = float(options.get("beta_reg", 0.0))
    y_vec = y_raw.reshape(-1)
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
    xi0, init_diagnostics = _initial_xi_from_stage1_with_diagnostics(
        init_estimate, scene, config, stage1_factors
    )
    global_lower = np.concatenate([config["ue_bounds"][:, 0], [config["delta_t_bounds"][0]]])
    global_upper = np.concatenate([config["ue_bounds"][:, 1], [config["delta_t_bounds"][1]]])
    xi0 = np.clip(xi0, global_lower, global_upper)
    lower, upper, trust_region_used = _intersect_trust_region_bounds(
        xi0, global_lower, global_upper, config, options
    )
    xi0 = np.clip(xi0, lower, upper)

    objective_history: list[float] = []

    def fun(xi: np.ndarray) -> float:
        value, _ = _vp_objective_and_grad(xi, y_vec, init_estimate, scene, config)
        objective_history.append(float(value))
        return float(value)

    def jac(xi: np.ndarray) -> np.ndarray:
        _, grad = _vp_objective_and_grad(xi, y_vec, init_estimate, scene, config)
        return grad

    best_x = xi0.copy()
    best_value = fun(best_x)
    initial_parts = _vp_objective_parts(best_x, y_vec, init_estimate, scene, config)
    initial_objective = float(initial_parts["total_objective"])
    success = True
    message = "initial point only"
    num_iter = 0
    solver_method = "initial_point_only"
    solver_backend = "none"

    if scipy_is_available():
        from scipy.optimize import minimize

        success = False
        solver_method = "scipy.optimize.minimize:L-BFGS-B"
        solver_backend = "scipy.optimize"
        for start in _candidate_starts(xi0, lower, upper, config):
            result = minimize(
                fun,
                start,
                jac=jac,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={
                    "maxiter": int(options.get("max_iter", 80)),
                    "ftol": float(options.get("ftol", 1.0e-12)),
                    "gtol": float(options.get("gtol", 1.0e-8)),
                },
            )
            value = float(result.fun)
            if value <= best_value:
                best_value = value
                best_x = np.asarray(result.x, dtype=float)
                success = bool(result.success)
                message = str(result.message)
                num_iter = int(result.nit)
    else:
        span = np.maximum(upper - lower, config.get("eps", 1.0e-10))
        x0_scaled = (xi0 - lower) / span

        def scaled_objective(x_scaled: np.ndarray) -> float:
            xi = lower + np.clip(x_scaled, 0.0, 1.0) * span
            return _vp_objective_only(xi, y_vec, init_estimate, scene, config)

        x_best_scaled, best_value, info = bounded_coordinate_search(
            scaled_objective,
            x0_scaled,
            np.zeros(4),
            np.ones(4),
            step0=0.08,
            max_iter=int(options.get("max_iter", 80)),
            tol=1.0e-4,
        )
        best_x = lower + x_best_scaled * span
        success = bool(info["success"])
        message = str(info["message"])
        num_iter = int(info["iterations"])
        solver_method = "bounded_coordinate_search"
        solver_backend = "fallback"

    initial_parts = _vp_objective_parts(xi0, y_vec, init_estimate, scene, config)
    beta0 = np.asarray(initial_parts["beta"], dtype=complex)
    phi0 = _build_global_dictionary(xi0, init_estimate, scene, config)[0]
    initial_residual_vec = y_vec - phi0 @ beta0
    initial_residual = float(np.linalg.norm(initial_residual_vec) / np.sqrt(y_vec.size))

    final_parts = _vp_objective_parts(best_x, y_vec, init_estimate, scene, config)
    phi_final, aux = _build_global_dictionary(best_x, init_estimate, scene, config)
    beta_final = np.asarray(final_parts["beta"], dtype=complex)
    final_residual_vec = y_vec - phi_final @ beta_final
    final_residual = float(np.linalg.norm(final_residual_vec) / np.sqrt(y_vec.size))
    final_objective = float(final_parts["total_objective"])
    rollback_tolerance = float(options.get("objective_rollback_tolerance", 1.0e-12))
    if final_objective > initial_objective + rollback_tolerance:
        best_x = xi0.copy()
        final_parts = _vp_objective_parts(best_x, y_vec, init_estimate, scene, config)
        phi_final, aux = _build_global_dictionary(best_x, init_estimate, scene, config)
        beta_final = np.asarray(final_parts["beta"], dtype=complex)
        final_residual_vec = y_vec - phi_final @ beta_final
        final_residual = float(np.linalg.norm(final_residual_vec) / np.sqrt(y_vec.size))
        final_objective = float(final_parts["total_objective"])
        success = False
        message = "rollback_objective_increased"
    y_hat = (phi_final @ beta_final).reshape(scene["I"], scene["N"], scene["T"])
    linear_dim = int(phi_final.shape[1])
    vp_mode = str(final_parts.get("vp_mode", _global_vp_mode(config)))
    gamma_hat = None
    eta_hat = None
    beta_path_hat = beta_final.copy()
    if vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"}:
        x_blocks = beta_final.reshape(scene["K"], 2)
        beta_path_hat = np.linalg.norm(x_blocks, axis=1)
        e0_diag = extract_stage1_jones_directions(init_estimate, scene)
        leakage = np.empty(scene["K"], dtype=float)
        gamma_hat = np.empty(scene["K"], dtype=float)
        eta_hat = np.empty(scene["K"], dtype=float)
        for k in range(scene["K"]):
            x1, x2 = x_blocks[k]
            gain = np.linalg.norm(x_blocks[k])
            e = e0_diag[k]
            p_perp_x = x_blocks[k] - e * (np.vdot(e, x_blocks[k]) / (np.vdot(e, e).real + config.get("eps", 1.0e-10)))
            leakage[k] = float(np.vdot(p_perp_x, p_perp_x).real / (gain**2 + config.get("eps", 1.0e-10)))
            if gain <= config.get("eps", 1.0e-10):
                gamma_hat[k] = 0.0
                eta_hat[k] = 0.0
            else:
                gamma_hat[k] = float(np.arctan2(abs(x1), abs(x2)))
                eta_hat[k] = float(np.angle(np.exp(1j * (np.angle(x1) - np.angle(x2)))))
        a_evs_hat = np.empty((scene["K"], scene["I"]), dtype=complex)
        for k in range(scene["K"]):
            e = x_blocks[k] / (np.linalg.norm(x_blocks[k]) + config.get("eps", 1.0e-10))
            a_evs_hat[k] = np.kron(scene["v_B"][k], scene["Theta"][k] @ e)
    else:
        a_evs_hat = np.asarray(stage1_factors["A_phys"], dtype=complex).T.copy()
        leakage = np.array([], dtype=float)

    estimate = copy.deepcopy(init_estimate)
    estimate.update(
        {
            "p_u": best_x[:3].copy(),
            "delta_t": float(best_x[3]),
            "tau": aux["tau"].copy(),
            "poles": aux["poles"].copy(),
            "D": aux["D"].copy(),
            "C_global": aux["C"].copy(),
            "beta_raw": beta_final.copy(),
            "Y_hat": y_hat,
            "components": {
                "taus": aux["tau"].copy(),
                "ranges": aux["ranges"].copy(),
                "elevations": aux["elevations"].copy(),
                "azimuths": aux["azimuths"].copy(),
                "d": aux["D"].T.copy(),
                "c": aux["C"].T.copy(),
                "a_EVS": a_evs_hat.copy(),
            },
            "raw_residual_rmse_noisy": final_residual,
            "raw_residual_initial": initial_residual,
            "raw_residual_final": final_residual,
            "raw_residual": final_residual,
            "raw_objective_initial": float(initial_parts["raw_objective"]),
            "raw_objective_final": float(final_parts["raw_objective"]),
            "raw_objective": float(final_parts["raw_objective"]),
            "delay_prior_objective_initial": float(
                initial_parts["delay_prior_objective"]
            ),
            "delay_prior_objective_final": float(final_parts["delay_prior_objective"]),
            "delay_prior_objective": float(final_parts["delay_prior_objective"]),
            "total_objective_initial": initial_objective,
            "total_objective_final": final_objective,
            "total_objective": final_objective,
            "beta_reg_objective_initial": float(initial_parts["beta_reg_objective"]),
            "beta_reg_objective_final": float(final_parts["beta_reg_objective"]),
            "jones_regularizer_objective_initial": float(
                initial_parts.get("jones_regularizer_objective", 0.0)
            ),
            "jones_regularizer_objective_final": float(
                final_parts.get("jones_regularizer_objective", 0.0)
            ),
            "tau_stage1": np.asarray(stage1_factors["tau_phys"], dtype=float).copy(),
            "tau_after_global_vp": aux["tau"].copy(),
            "global_vp_solver": "lbfgsb_reduced",
            "global_vp_mode": vp_mode,
            "vp_mode": vp_mode,
            "global_vp_evs_mode": str(aux.get("evs_mode", options.get("evs_mode", "legacy_or_full_polarization"))),
            "nonlinear_dim": 4,
            "linear_nuisance_dim": linear_dim,
            "jones_rho": np.asarray(final_parts.get("jones_rho", []), dtype=float).copy(),
            "lambda_jones_per_path": np.asarray(final_parts.get("lambda_jones_per_path", []), dtype=float).copy(),
            "jones_leakage_per_path": leakage.copy(),
            "jones_prior_status": list(final_parts.get("jones_prior_status", [])),
            "x_hat": beta_final.copy() if vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"} else None,
            "condition_number_gram": float(
                final_parts.get("linear_diagnostics", {}).get("condition_number_gram", np.nan)
            ),
            "rank_gram": int(final_parts.get("linear_diagnostics", {}).get("rank_gram", 0)),
            "trace_H": float(final_parts.get("linear_diagnostics", {}).get("trace_H", np.nan)),
            "raw_residual_norm": float(np.linalg.norm(final_residual_vec)),
            "global_vp_use_delay_prior": _delay_prior_enabled(config),
            "global_vp_trust_region_used": bool(trust_region_used),
            "global_vp_bounds_lower": lower.copy(),
            "global_vp_bounds_upper": upper.copy(),
            "global_vp_xi0": xi0.copy(),
            "global_vp_used_panel_to_column": stage1_factors[
                "global_vp_used_panel_to_column"
            ],
            "global_vp_panel_to_column": stage1_factors["global_vp_panel_to_column"],
            "global_vp_columns_are_panel_ordered": stage1_factors[
                "global_vp_columns_are_panel_ordered"
            ],
            **init_diagnostics,
            "global_vp_success": bool(success),
            "global_vp_message": message,
            "global_vp_num_iter": int(num_iter),
            "global_vp_objective_history": objective_history,
            "optimizer": {
                "success": bool(success),
                "status": 1 if success else 0,
                "message": message,
                "n_eval": len(objective_history),
                "method": solver_method,
                "solver_backend": solver_backend,
                "n_iter": int(num_iter),
                "objective": (
                    "regularized_weighted_vp_raw_plus_delay_prior"
                    if beta_reg > 0.0
                    else "weighted_vp_raw_reduced"
                ),
                "solver": "lbfgsb_reduced",
            },
            "vp_enabled": True,
            "stage2_mode": "none",
            "final_refinement_method": "global_exact_spherical_vp",
        }
    )
    if gamma_hat is not None and eta_hat is not None:
        estimate["gamma"] = gamma_hat
        estimate["eta_pol"] = eta_hat
        estimate["beta"] = beta_path_hat
    if bool(options.get("overwrite_factor_keys", False)):
        estimate["C"] = aux["C"].copy()
        estimate["D"] = aux["D"].copy()
    return estimate


def _panel_ordered_estimate_for_legacy_solver(init_estimate: dict, scene: dict) -> dict:
    """Return an estimate copy in physical panel order for the legacy VP solver."""
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
    estimate = copy.deepcopy(init_estimate)
    estimate["A"] = stage1_factors["A_phys"].copy()
    estimate["poles"] = stage1_factors["poles_phys"].copy()
    estimate["ris_eta"] = stage1_factors["ris_eta_phys"].copy()
    if stage1_factors.get("C_phys") is not None:
        estimate["C"] = stage1_factors["C_phys"].copy()
    if stage1_factors.get("gamma_phys") is not None:
        estimate["gamma"] = stage1_factors["gamma_phys"].copy()
    if stage1_factors.get("eta_pol_phys") is not None:
        estimate["eta_pol"] = stage1_factors["eta_pol_phys"].copy()
    estimate["columns_are_panel_ordered"] = True
    estimate["global_vp_used_panel_to_column"] = stage1_factors[
        "global_vp_used_panel_to_column"
    ]
    estimate["global_vp_panel_to_column"] = stage1_factors["global_vp_panel_to_column"]
    estimate["global_vp_columns_are_panel_ordered"] = stage1_factors[
        "global_vp_columns_are_panel_ordered"
    ]
    return estimate


def _legacy_vp_initial_result(y_raw: np.ndarray, estimate: dict, scene: dict, config: dict) -> dict:
    """Evaluate the old VP-WNLS model at its initial point for rollback."""
    from .channel_model import synthesize_raw_tensor
    from .estimators import _bounds_global, _dictionary_from_global_x, _initial_global_parameters

    x0 = _initial_global_parameters(scene, estimate, config)
    lower, upper = _bounds_global(scene, config)
    x0 = np.clip(x0, lower, upper)
    y_vec = y_raw.reshape(-1)
    dictionary, components = _dictionary_from_global_x(scene, x0)
    beta = solve_lstsq(dictionary, y_vec, reg=1.0e-12)
    residual = dictionary @ beta - y_vec
    raw_objective = float(np.vdot(residual, residual).real / y_vec.size)
    return {
        "x": x0,
        "p_u": x0[:3],
        "delta_t": float(x0[3]),
        "gamma": x0[4 : 4 + scene["K"]],
        "eta_pol": x0[4 + scene["K"] : 4 + 2 * scene["K"]],
        "beta": beta,
        "components": components,
        "Y_hat": synthesize_raw_tensor(components, beta),
        "raw_residual_rmse_noisy": float(np.sqrt(raw_objective)),
        "raw_objective_initial": raw_objective,
        "raw_objective_final": raw_objective,
        "optimizer": {
            "success": False,
            "status": 0,
            "message": "rollback_objective_increased",
            "n_eval": 0,
            "method": "scipy.optimize.least_squares",
            "solver_backend": "scipy.optimize" if scipy_is_available() else "fallback",
        },
    }


def _augment_legacy_vp_result(
    legacy_result: dict,
    init_estimate: dict,
    scene: dict,
    options: dict,
    *,
    rolled_back: bool,
) -> dict:
    """Add global-VP compatibility diagnostics to old VP-WNLS output."""
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
    components = legacy_result["components"]
    raw_initial = float(legacy_result["raw_objective_initial"])
    raw_final = float(legacy_result["raw_objective_final"])
    final_rmse = float(legacy_result["raw_residual_rmse_noisy"])
    optimizer = dict(legacy_result.get("optimizer", {}))
    if rolled_back:
        optimizer["success"] = False
        optimizer["message"] = "rollback_objective_increased"

    estimate = copy.deepcopy(init_estimate)
    estimate.update(legacy_result)
    estimate.update(
        {
            "tau": components["taus"].copy(),
            "poles": components["poles"].copy(),
            "D": components["d"].T.copy(),
            "C_global": components["c"].T.copy(),
            "beta_raw": legacy_result["beta"].copy(),
            "raw_residual_initial": float(np.sqrt(raw_initial)),
            "raw_residual_final": final_rmse,
            "raw_residual": final_rmse,
            "raw_objective": raw_final,
            "delay_prior_objective_initial": 0.0,
            "delay_prior_objective_final": 0.0,
            "delay_prior_objective": 0.0,
            "total_objective_initial": raw_initial,
            "total_objective_final": raw_final,
            "total_objective": raw_final,
            "beta_reg_objective_initial": 0.0,
            "beta_reg_objective_final": 0.0,
            "tau_stage1": np.asarray(stage1_factors["tau_phys"], dtype=float).copy(),
            "tau_after_global_vp": components["taus"].copy(),
            "global_vp_solver": "least_squares",
            "global_vp_mode": "fixed_pol",
            "vp_mode": "fixed_pol",
            "global_vp_evs_mode": "legacy_or_full_polarization",
            "nonlinear_dim": 4 + 2 * int(scene["K"]),
            "linear_nuisance_dim": int(scene["K"]),
            "jones_rho": np.array([], dtype=float),
            "jones_prior_status": [],
            "x_hat": None,
            "raw_residual_norm": float(final_rmse * np.sqrt(np.prod(legacy_result["Y_hat"].shape))),
            "condition_number_gram": float("nan"),
            "rank_gram": 0,
            "global_vp_use_delay_prior": False,
            "global_vp_trust_region_used": False,
            "global_vp_used_panel_to_column": stage1_factors[
                "global_vp_used_panel_to_column"
            ],
            "global_vp_panel_to_column": stage1_factors["global_vp_panel_to_column"],
            "global_vp_columns_are_panel_ordered": stage1_factors[
                "global_vp_columns_are_panel_ordered"
            ],
            "global_vp_success": bool(optimizer.get("success", False)) and not rolled_back,
            "global_vp_message": str(optimizer.get("message", "")),
            "global_vp_num_iter": int(optimizer.get("n_eval", 0)),
            "global_vp_objective_history": [],
            "optimizer": {
                **optimizer,
                "objective": "least_squares_vp_wnls_raw_mse",
                "solver": "least_squares",
            },
            "vp_enabled": True,
            "stage2_mode": "none",
            "final_refinement_method": "global_exact_spherical_vp",
        }
    )
    if bool(options.get("overwrite_factor_keys", False)):
        estimate["C"] = components["c"].T.copy()
        estimate["D"] = components["d"].T.copy()
    return estimate


def _global_exact_spherical_vp_refinement_least_squares(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Wrap the old high-performing least_squares VP-WNLS implementation."""
    from .estimators import refine_global_raw

    options = _global_vp_config(config)
    cache_start = time.perf_counter()
    _ = _build_global_vp_cache(scene)
    cache_build_time = time.perf_counter() - cache_start
    panel_ordered_estimate = _panel_ordered_estimate_for_legacy_solver(
        init_estimate, scene
    )
    solver_start = time.perf_counter()
    legacy_result = refine_global_raw(y_raw, scene, config, panel_ordered_estimate)
    solver_time = time.perf_counter() - solver_start
    tolerance = float(options.get("objective_rollback_tolerance", 1.0e-12))
    rolled_back = bool(
        legacy_result["raw_objective_final"]
        > legacy_result["raw_objective_initial"] + tolerance
    )
    if rolled_back:
        legacy_result = _legacy_vp_initial_result(
            y_raw, panel_ordered_estimate, scene, config
        )
    result = _augment_legacy_vp_result(
        legacy_result,
        panel_ordered_estimate,
        scene,
        options,
        rolled_back=rolled_back,
    )
    num_calls = int(result.get("optimizer", {}).get("n_eval", 0))
    result["global_vp_cache_build_time"] = float(cache_build_time)
    result["global_vp_num_residual_calls"] = int(num_calls)
    result["global_vp_residual_eval_time_mean"] = float(
        solver_time / max(num_calls, 1)
    )
    result["global_vp_jacobian_mode"] = (
        "legacy_least_squares_numerical"
        if bool(options.get("use_analytic_jacobian", True))
        else "numerical"
    )
    result["global_vp_matrix_free_beta"] = bool(options.get("matrix_free_beta", False))
    return result


def _vp_family_score(raw_objective: float, sigma2_hat: float, d_eff: float, y_size: int) -> float:
    return float(
        2.0 * float(raw_objective) * float(y_size) / max(float(sigma2_hat), 1.0e-300)
        + float(d_eff) * np.log(2.0 * float(y_size))
    )


def _adaptive_jones_lambdas(
    fixed_result: dict,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-path effective SNR and adaptive dimensionless lambda_k."""
    options = _global_vp_config(config)
    k_paths = int(scene["K"])
    e0 = extract_stage1_jones_directions(init_estimate, scene)
    psi = build_jones_vp_dictionary(fixed_result["p_u"], fixed_result["delta_t"], scene, config)
    beta_fix = np.asarray(
        fixed_result.get("beta_raw", fixed_result.get("beta", np.ones(k_paths, dtype=complex))),
        dtype=complex,
    ).reshape(-1)[:k_paths]
    sigma2_hat = float(
        init_estimate.get(
            "noise_variance",
            config.get("noise_variance", fixed_result.get("raw_objective_final", 1.0)),
        )
    )
    m_samples = int(scene["I"]) * int(scene["N"]) * int(scene["T"])
    snr_eff = np.empty(k_paths, dtype=float)
    for k in range(k_paths):
        psi_e0 = psi[:, 2 * k : 2 * k + 2] @ e0[k]
        snr_eff[k] = (
            abs(beta_fix[k]) ** 2
            * float(np.vdot(psi_e0, psi_e0).real)
            / (float(m_samples) * max(sigma2_hat, config.get("eps", 1.0e-10)))
        )
    lambda0_path = _as_path_vector(
        options.get("jones_lambda0", 1.0),
        k_paths,
        name="jones_lambda0",
        default=1.0,
    )
    lambda_path = lambda0_path / (
        snr_eff + float(options.get("jones_snr_eps", 1.0e-12))
    )
    if "stage1_jones_tau" in init_estimate:
        tau = _as_path_vector(
            init_estimate["stage1_jones_tau"],
            k_paths,
            name="stage1_jones_tau",
        )
        tau_ref = float(options.get("jones_tau", 0.25))
        lambda_path *= (tau_ref**2) / (
            tau**2 + float(options.get("jones_snr_eps", 1.0e-12))
        )
    lambda_path = np.clip(
        lambda_path,
        float(options.get("jones_lambda_min", 1.0e-4)),
        float(options.get("jones_lambda_max", 1.0e8)),
    )
    return snr_eff, lambda_path


def _global_exact_spherical_vp_refinement_adaptive_jones(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Run fixed-pol anchor, adaptive Gram-scaled Jones-VP, then select by score."""
    options = _global_vp_config(config)
    fixed_config = copy.deepcopy(config)
    fixed_config["global_vp"] = dict(fixed_config.get("global_vp", {}))
    fixed_config["global_vp"]["mode"] = "fixed_pol"
    fixed_result = _global_exact_spherical_vp_refinement_least_squares(
        y_raw, init_estimate, scene, fixed_config
    )

    snr_eff, lambda_path = _adaptive_jones_lambdas(
        fixed_result, init_estimate, scene, config
    )
    jones_init = copy.deepcopy(init_estimate)
    jones_init["p_u"] = np.asarray(fixed_result["p_u"], dtype=float).copy()
    jones_init["delta_t"] = float(fixed_result["delta_t"])
    jones_init["jones_lambda_per_path"] = lambda_path.copy()
    jones_config = copy.deepcopy(config)
    jones_config["global_vp"] = dict(jones_config.get("global_vp", {}))
    jones_config["global_vp"]["mode"] = "adaptive_jones"
    jones_result = _global_exact_spherical_vp_refinement_lbfgsb_reduced(
        y_raw, jones_init, scene, jones_config
    )

    m_samples = int(y_raw.size)
    sigma2_hat = float(
        init_estimate.get(
            "noise_variance",
            config.get("noise_variance", fixed_result.get("raw_objective_final", 1.0)),
        )
    )
    d_eff_fixed = float(4 + 2 * int(scene["K"]))
    trace_h = float(jones_result.get("trace_H", 2 * int(scene["K"])))
    d_eff_jones = float(4.0 + 2.0 * trace_h)
    fixed_score = _vp_family_score(
        fixed_result["raw_objective_final"], sigma2_hat, d_eff_fixed, m_samples
    )
    jones_score = _vp_family_score(
        jones_result["raw_objective_final"], sigma2_hat, d_eff_jones, m_samples
    )
    leakage = np.asarray(jones_result.get("jones_leakage_per_path", []), dtype=float)
    fixed_raw = float(fixed_result["raw_objective_final"])
    jones_raw = float(jones_result["raw_objective_final"])
    rel_improvement = (fixed_raw - jones_raw) / max(fixed_raw, config.get("eps", 1.0e-10))
    leakage_guard = bool(
        leakage.size
        and np.nanmax(leakage) > float(options.get("jones_leakage_threshold", 0.25))
        and rel_improvement < float(options.get("jones_min_rel_improvement", 1.0e-3))
    )
    choose_jones = bool(jones_score < fixed_score and not leakage_guard)
    selected = copy.deepcopy(jones_result if choose_jones else fixed_result)
    selected_branch = "adaptive_jones" if choose_jones else "fixed_pol_anchor"

    diagnostics = {
        "global_vp_mode": "adaptive_jones",
        "vp_mode": "adaptive_jones",
        "selected_vp_family_branch": selected_branch,
        "fixed_pol_score": fixed_score,
        "jones_score": jones_score,
        "d_eff_fixed": d_eff_fixed,
        "d_eff_jones": d_eff_jones,
        "snr_eff_per_path": snr_eff.copy(),
        "lambda_jones_per_path": lambda_path.copy(),
        "jones_leakage_per_path": leakage.copy(),
        "jones_leakage_guard_triggered": leakage_guard,
        "jones_relative_residual_improvement": float(rel_improvement),
        "fixed_pol_anchor_raw_objective": fixed_raw,
        "adaptive_jones_raw_objective": jones_raw,
    }
    selected.update(diagnostics)
    selected["nonlinear_dim"] = 4
    if not choose_jones:
        selected["linear_nuisance_dim"] = int(scene["K"])
    return selected


def _global_exact_spherical_vp_refinement_once(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Initialization-aided unified raw-domain exact-spherical VP.

    The default numerical solver is the legacy high-performing
    least_squares VP-WNLS path, which optimizes UE position, clock offset, and
    EVS polarization angles while eliminating path gains. The reduced L-BFGS-B
    implementation is retained only as an experimental ablation.
    """
    options = _global_vp_config(config)
    mode = _global_vp_mode(config)
    solver = str(options.get("solver", "least_squares"))
    if mode == "adaptive_jones":
        return _global_exact_spherical_vp_refinement_adaptive_jones(
            y_raw, init_estimate, scene, config
        )
    if mode in {"jones_regularized", "jones_free"}:
        return _global_exact_spherical_vp_refinement_lbfgsb_reduced(
            y_raw, init_estimate, scene, config
        )
    if solver == "least_squares":
        return _global_exact_spherical_vp_refinement_least_squares(
            y_raw, init_estimate, scene, config
        )
    if solver == "lbfgsb_reduced":
        return _global_exact_spherical_vp_refinement_lbfgsb_reduced(
            y_raw, init_estimate, scene, config
        )
    raise ValueError(f"unknown global_vp solver {solver!r}")


def global_exact_spherical_vp_refinement(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Run global VP and, when suspicious, deterministic z-directed restarts."""
    options = _global_vp_config(config)
    normal = _global_exact_spherical_vp_refinement_once(
        y_raw, init_estimate, scene, config
    )
    bounds = np.asarray(config["ue_bounds"], dtype=float)
    boundary_tol = float(options.get("boundary_tol_m", 0.02))
    before = distance_to_box_boundary(normal["p_u"], bounds, boundary_tol)
    normal.update(before)
    normal.setdefault("z_rescue_triggered", False)
    normal.setdefault("z_rescue_num_starts", 0)
    normal.setdefault("z_rescue_best_z", float(normal["p_u"][2]))
    normal.setdefault(
        "z_rescue_best_score",
        float(normal.get("raw_objective_final", normal.get("raw_objective", np.nan))),
    )
    normal.setdefault("z_rescue_candidate_scores", [])
    normal.setdefault("z_rescue_selected_reason", "not_triggered")
    normal["boundary_hit_before_rescue"] = bool(before["boundary_hit"])
    normal["boundary_hit_after_rescue"] = bool(before["boundary_hit"])

    if not bool(options.get("enable_z_rescue_multistart", True)):
        normal["z_rescue_selected_reason"] = "disabled"
        return normal
    if bool(config.get("_global_vp_z_rescue_active", False)):
        return normal

    force_rescue = bool(config.get("_global_vp_force_z_rescue", False))
    unreliable = not bool(
        normal.get("global_vp_success", normal.get("optimizer", {}).get("success", False))
    )
    suspicious = bool(before["boundary_hit"] or force_rescue or unreliable)
    if not suspicious:
        return normal

    current = np.asarray(normal["p_u"], dtype=float)
    if not bool(options.get("z_rescue_keep_xy", True)):
        current[:2] = np.mean(bounds[:2], axis=1)
    starts = z_rescue_starts(
        current,
        bounds,
        int(options.get("z_rescue_num_starts", 7)),
        float(options.get("z_rescue_margin_m", 0.02)),
    )
    candidates = [normal]
    candidate_scores = [
        {
            "z_start": float(current[2]),
            "z_final": float(current[2]),
            "raw_objective_final": float(
                normal.get("raw_objective_final", normal.get("raw_objective", np.nan))
            ),
            "boundary_hit": bool(before["boundary_hit"]),
            "kind": "normal",
        }
    ]
    for start in starts:
        rescue_init = copy.deepcopy(init_estimate)
        rescue_init["_global_vp_initial_p_u"] = start.copy()
        rescue_init["_global_vp_initial_delta_t"] = float(normal["delta_t"])
        rescue_config = copy.deepcopy(config)
        rescue_config["_global_vp_z_rescue_active"] = True
        rescue = _global_exact_spherical_vp_refinement_once(
            y_raw, rescue_init, scene, rescue_config
        )
        score = float(
            rescue.get("raw_objective_final", rescue.get("raw_objective", np.nan))
        )
        rescue["raw_objective_final"] = score
        rescue_boundary = distance_to_box_boundary(
            rescue["p_u"], bounds, boundary_tol
        )
        rescue.update(rescue_boundary)
        candidates.append(rescue)
        candidate_scores.append(
            {
                "z_start": float(start[2]),
                "z_final": float(rescue["p_u"][2]),
                "raw_objective_final": score,
                "boundary_hit": bool(rescue_boundary["boundary_hit"]),
                "kind": "z_rescue",
            }
        )

    selected, reason = select_z_rescue_candidate(
        candidates,
        bounds,
        boundary_tol_m=boundary_tol,
        boundary_accept_rel_tol=float(options.get("boundary_accept_rel_tol", 1.0e-3)),
    )
    selected = copy.deepcopy(selected)
    after = distance_to_box_boundary(selected["p_u"], bounds, boundary_tol)
    selected.update(after)
    selected.update(
        {
            "z_rescue_triggered": True,
            "z_rescue_num_starts": int(len(starts)),
            "z_rescue_best_z": float(selected["p_u"][2]),
            "z_rescue_best_score": float(
                selected.get("raw_objective_final", selected.get("raw_objective", np.nan))
            ),
            "z_rescue_candidate_scores": candidate_scores,
            "z_rescue_selected_reason": reason,
            "boundary_hit_before_rescue": bool(before["boundary_hit"]),
            "boundary_hit_after_rescue": bool(after["boundary_hit"]),
        }
    )
    return selected
