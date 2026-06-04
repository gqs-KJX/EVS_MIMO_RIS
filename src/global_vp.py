"""Raw-domain global exact-spherical variable projection refinement.

The default proposed estimator uses Stage-I only for path assignment and fixed
EVS factors, then refines the UE position and common clock offset directly in
the raw OFDM tensor. Path gains are eliminated by variable projection. Legacy
factor-domain EVS/delay/RIS projections remain separate ablation baselines.
"""

from __future__ import annotations

import copy

import numpy as np

from .geometry import elev_az_from_unit_vector, unit_vector_from_elev_az
from .projections_delay import tau_from_pole
from .utils import bounded_coordinate_search, scipy_is_available


def _global_vp_config(config: dict) -> dict:
    """Return global-VP options with conservative defaults."""
    defaults = {
        "max_iter": 80,
        "ftol": 1.0e-12,
        "gtol": 1.0e-8,
        "beta_reg": 0.0,
        "evs_mode": "linear_polarization_basis",
        "use_delay_prior": True,
        "delay_prior_weight": 1.0,
        "delay_prior_sigma_s": 2.0e-11,
        "use_weight": False,
        "weight": None,
        "use_multistart": False,
        "num_perturb_starts": 0,
        "position_perturb_std_m": 0.05,
        "clock_perturb_std_s": 1.0e-10,
        "use_trust_region": True,
        "position_trust_radius_m": 0.3,
        "clock_trust_radius_s": 3.0e-10,
        "overwrite_factor_keys": False,
        "finite_difference_check": False,
    }
    options = dict(defaults)
    options.update(dict(config.get("global_vp", {})))
    return options


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


def _evs_atom_bases(init_estimate: dict, scene: dict, config: dict) -> tuple[list[np.ndarray], str]:
    """Return per-path EVS basis matrices for the configured global VP mode."""
    options = _global_vp_config(config)
    evs_mode = str(options.get("evs_mode", "linear_polarization_basis"))
    stage1_factors = _get_panel_ordered_stage1_factors(init_estimate, scene)
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
    beta_reg = float(options.get("beta_reg", 0.0))
    weight = _objective_weight_from_config(config, y_vec.size)
    objective_scale = 1.0 / float(y_vec.size)
    phi, aux = _build_global_dictionary(
        xi, init_estimate, scene, config, need_jacobian=True
    )
    beta = _solve_beta_vp(phi, y_vec, weight, beta_reg, objective_scale)
    residual = y_vec - phi @ beta
    raw_objective = float(
        objective_scale * np.real(_weighted_inner(residual, residual, weight))
    )
    beta_reg_objective = 0.0
    if beta_reg > 0.0:
        beta_reg_objective = float(beta_reg * np.real(np.vdot(beta, beta)))

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

    total_objective = raw_objective + beta_reg_objective + delay_prior_objective
    parts = {
        "raw_objective": raw_objective,
        "beta_reg_objective": beta_reg_objective,
        "delay_prior_objective": delay_prior_objective,
        "total_objective": float(total_objective),
        "beta": beta,
        "residual": residual,
        "aux": aux,
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


def global_exact_spherical_vp_refinement(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Initialization-aided global exact-spherical VP refinement.

    Stage-I provides path assignment and fixed EVS factors. This refinement
    optimizes only UE position and common clock offset by default; path gains are
    eliminated by VP, and RIS near-field plus delay/clock coupling are embedded
    directly in each raw-domain atom.
    """
    assert y_raw.shape == (scene["I"], scene["N"], scene["T"])
    options = _global_vp_config(config)
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

    phi0, _ = _build_global_dictionary(xi0, init_estimate, scene, config)
    weight = _objective_weight_from_config(config, y_vec.size)
    beta_reg = float(options.get("beta_reg", 0.0))
    objective_scale = 1.0 / float(y_vec.size)
    beta0 = _solve_beta_vp(phi0, y_vec, weight, beta_reg, objective_scale)
    initial_residual_vec = y_vec - phi0 @ beta0
    initial_residual = float(np.linalg.norm(initial_residual_vec) / np.sqrt(y_vec.size))

    phi_final, aux = _build_global_dictionary(best_x, init_estimate, scene, config)
    beta_final = _solve_beta_vp(phi_final, y_vec, weight, beta_reg, objective_scale)
    final_residual_vec = y_vec - phi_final @ beta_final
    final_residual = float(np.linalg.norm(final_residual_vec) / np.sqrt(y_vec.size))
    final_parts = _vp_objective_parts(best_x, y_vec, init_estimate, scene, config)
    final_objective = float(final_parts["total_objective"])
    y_hat = (phi_final @ beta_final).reshape(scene["I"], scene["N"], scene["T"])

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
                "a_EVS": np.asarray(stage1_factors["A_phys"], dtype=complex).T.copy(),
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
            "tau_stage1": np.asarray(stage1_factors["tau_phys"], dtype=float).copy(),
            "tau_after_global_vp": aux["tau"].copy(),
            "global_vp_evs_mode": str(options.get("evs_mode", "linear_polarization_basis")),
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
                    else "weighted_vp_raw_plus_delay_prior"
                ),
            },
            "vp_enabled": True,
            "stage2_mode": "none",
            "final_refinement_method": "global_exact_spherical_vp",
        }
    )
    if bool(options.get("overwrite_factor_keys", False)):
        estimate["C"] = aux["C"].copy()
        estimate["D"] = aux["D"].copy()
    return estimate
