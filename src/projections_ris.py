"""Compressed near-field RIS projection with dechirped rank-one lifting."""

from __future__ import annotations

import numpy as np

from .geometry import (
    elev_az_from_unit_vector,
    near_field_spherical_response,
    unit_vector_from_elev_az,
)
from .utils import bounded_coordinate_search, scipy_is_available


def compressed_exact_response(
    eta_local: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Return h_ex(eta) = Omega @ (a_RB * a_UR^NF_exact(eta))."""
    range_m, elevation, azimuth = eta_local
    a_ur = near_field_spherical_response(range_m, elevation, azimuth, ris_grid, wavelength)
    g_elem = a_rb * a_ur
    return omega @ g_elem


def local_ris_search_config(scene: dict, config: dict, path: int) -> dict:
    """Build RIS-specific geometry-search bounds from UE position bounds."""
    base = dict(config["ris_search"])
    ue_bounds = np.asarray(config["ue_bounds"], dtype=float)
    corners = np.array(
        [
            [x, y, z]
            for x in ue_bounds[0]
            for y in ue_bounds[1]
            for z in ue_bounds[2]
        ],
        dtype=float,
    )
    ranges = []
    elevations = []
    azimuths = []
    for corner in corners:
        q_local = scene["rotations"][path] @ (corner - scene["ris_centers"][path])
        range_m = np.linalg.norm(q_local)
        if range_m <= 0.0:
            continue
        elev, az = elev_az_from_unit_vector(q_local / range_m)
        ranges.append(range_m)
        elevations.append(elev)
        azimuths.append(az)

    range_margin = float(base.get("local_range_margin", 0.35))
    angle_margin = float(base.get("local_angle_margin", 0.10))
    global_r_min, global_r_max = base["range_bounds"]
    global_e_min, global_e_max = base["elev_bounds"]
    base["range_bounds"] = (
        max(global_r_min, float(np.min(ranges) - range_margin)),
        min(global_r_max, float(np.max(ranges) + range_margin)),
    )
    base["elev_bounds"] = (
        max(global_e_min, float(np.min(elevations) - angle_margin)),
        min(global_e_max, float(np.max(elevations) + angle_margin)),
    )

    azimuths = np.asarray(azimuths)
    center = np.angle(np.mean(np.exp(1j * azimuths)))
    diffs = np.angle(np.exp(1j * (azimuths - center)))
    az_min = center + float(np.min(diffs) - angle_margin)
    az_max = center + float(np.max(diffs) + angle_margin)
    base["az_bounds"] = (az_min, az_max)
    return base


def scaled_residual(c_tilde: np.ndarray, h_model: np.ndarray, eps: float) -> tuple[float, complex]:
    """Return min_alpha ||c_tilde - alpha h_model||^2 and alpha."""
    denom = np.vdot(h_model, h_model) + eps
    alpha = np.vdot(h_model, c_tilde) / denom
    residual = np.linalg.norm(c_tilde - alpha * h_model) ** 2
    return float(residual), alpha


def _infer_ris_shape(ris_grid: np.ndarray) -> tuple[int, int]:
    """Infer rectangular RIS dimensions from the local element coordinates."""
    x_values = np.unique(np.round(ris_grid[:, 0], decimals=14))
    y_values = np.unique(np.round(ris_grid[:, 1], decimals=14))
    mx, my = len(x_values), len(y_values)
    assert mx * my == ris_grid.shape[0], "RIS grid is not a rectangular Mx x My grid"
    return mx, my


def _hankel_window(length: int) -> tuple[int, int]:
    """ES-CPD balanced Hankel window: P + L - 1 = length."""
    if length <= 0:
        raise ValueError("length must be positive")
    p_dim = (length + 1) // 2
    l_dim = length + 1 - p_dim
    return p_dim, l_dim


def _hankel_counts_1d(length: int) -> np.ndarray:
    p_dim, l_dim = _hankel_window(length)
    counts = np.zeros(length, dtype=float)
    for p_idx in range(p_dim):
        for l_idx in range(l_dim):
            counts[p_idx + l_idx] += 1.0
    return counts


def _block_hankel_counts_2d(
    mx: int, my: int, px: int, lx: int, py: int, ly: int
) -> np.ndarray:
    counts = np.zeros((mx, my), dtype=float)
    for ix in range(px):
        for iy in range(py):
            for jx in range(lx):
                for jy in range(ly):
                    counts[ix + jx, iy + jy] += 1.0
    return counts


def _block_hankel_inverse_weights_2d(
    mx: int, my: int, px: int, lx: int, py: int, ly: int
) -> np.ndarray:
    counts = _block_hankel_counts_2d(mx, my, px, lx, py, ly)
    return 1.0 / np.maximum(counts, 1.0)


def _block_hankel_2d(matrix: np.ndarray, px: int, lx: int, py: int, ly: int) -> np.ndarray:
    """2-D block-Hankel lifting H_2D(X), shape (Px*Py) x (Lx*Ly)."""
    mx, my = matrix.shape
    assert px + lx - 1 == mx, "invalid x Hankel windows"
    assert py + ly - 1 == my, "invalid y Hankel windows"
    lifted = np.empty((px * py, lx * ly), dtype=matrix.dtype)
    for ix in range(px):
        for iy in range(py):
            row = ix * py + iy
            for jx in range(lx):
                for jy in range(ly):
                    col = jx * ly + jy
                    lifted[row, col] = matrix[ix + jx, iy + jy]
    return lifted


def _block_dehankel_2d(lifted: np.ndarray, mx: int, my: int, px: int, lx: int, py: int, ly: int) -> np.ndarray:
    """Inverse 2-D block-Hankel lifting by anti-diagonal averaging."""
    matrix = np.zeros((mx, my), dtype=lifted.dtype)
    counts = np.zeros((mx, my), dtype=float)
    for ix in range(px):
        for iy in range(py):
            row = ix * py + iy
            for jx in range(lx):
                for jy in range(ly):
                    col = jx * ly + jy
                    matrix[ix + jx, iy + jy] += lifted[row, col]
                    counts[ix + jx, iy + jy] += 1.0
    return matrix / np.maximum(counts, 1.0)


def _block_dehankel_adjoint_2d(
    matrix: np.ndarray, mx: int, my: int, px: int, lx: int, py: int, ly: int
) -> np.ndarray:
    """Adjoint of anti-diagonal averaging used by _block_dehankel_2d."""
    counts = np.zeros((mx, my), dtype=float)
    for ix in range(px):
        for iy in range(py):
            for jx in range(lx):
                for jy in range(ly):
                    counts[ix + jx, iy + jy] += 1.0

    lifted = np.empty((px * py, lx * ly), dtype=matrix.dtype)
    for ix in range(px):
        for iy in range(py):
            row = ix * py + iy
            for jx in range(lx):
                for jy in range(ly):
                    col = jx * ly + jy
                    lifted[row, col] = matrix[ix + jx, iy + jy] / counts[ix + jx, iy + jy]
    return lifted


def _block_hankel_adjoint_sum_2d(
    lifted: np.ndarray, mx: int, my: int, px: int, lx: int, py: int, ly: int
) -> np.ndarray:
    """Adjoint of the 2D block-Hankel lifting H_2D, using summation over repeats."""
    matrix = np.zeros((mx, my), dtype=lifted.dtype)
    for ix in range(px):
        for iy in range(py):
            row = ix * py + iy
            for jx in range(lx):
                for jy in range(ly):
                    col = jx * ly + jy
                    matrix[ix + jx, iy + jy] += lifted[row, col]
    return matrix


def _rank_one_projection(matrix: np.ndarray) -> np.ndarray:
    """Best Frobenius-norm rank-one projection by truncated SVD."""
    u_vec, s_val, vh = np.linalg.svd(matrix, full_matrices=False)
    return s_val[0] * np.outer(u_vec[:, 0], vh[0, :])


def _fresnel_response_matrix(
    eta_local: np.ndarray, ris_grid: np.ndarray, wavelength: float
) -> np.ndarray:
    """Second-order Fresnel near-field response on the rectangular RIS grid."""
    range_m, elevation, azimuth = eta_local
    unit_vec = unit_vector_from_elev_az(elevation, azimuth)
    rho_dot_u = ris_grid @ unit_vec
    rho_norm_sq = np.sum(ris_grid**2, axis=1)
    delta_fresnel = -rho_dot_u + (rho_norm_sq - rho_dot_u**2) / (2.0 * range_m)
    response = np.exp(-1j * (2.0 * np.pi / wavelength) * delta_fresnel)
    mx, my = _infer_ris_shape(ris_grid)
    return response.reshape(mx, my)


def _dechirp_kernel(
    eta_local: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    px: int,
    lx: int,
    py: int,
    ly: int,
) -> np.ndarray:
    """Curvature-dependent dechirping kernel D_C for the 2-D lifting."""
    range_m, elevation, azimuth = eta_local
    unit_vec = unit_vector_from_elev_az(elevation, azimuth)
    projector = np.eye(3) - np.outer(unit_vec, unit_vec)
    kappa = 2.0 * np.pi / wavelength
    mx, my = _infer_ris_shape(ris_grid)
    grid = ris_grid.reshape(mx, my, 3)

    # Decompose the coordinate of element (ix+jx, iy+jy) into row and shift parts.
    # Constant offsets only change row/column phases and are absorbed by the rank-one factors.
    row_coords = np.empty((px * py, 3), dtype=float)
    col_shifts = np.empty((lx * ly, 3), dtype=float)
    for ix in range(px):
        for iy in range(py):
            row_coords[ix * py + iy] = grid[ix, iy]
    for jx in range(lx):
        for jy in range(ly):
            col_shifts[jx * ly + jy] = grid[jx, jy] - grid[0, 0]

    cross = row_coords @ projector @ col_shifts.T
    return np.exp(1j * kappa * cross / range_m)


def _lifted_forward(
    x_lift: np.ndarray,
    dechirp: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    shape_info: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    """Apply T_eta(X) = Omega diag(a_RB) H_2D^dagger(D_C^* X)."""
    mx, my, px, lx, py, ly = shape_info
    restored = np.conj(dechirp) * x_lift
    element_matrix = _block_dehankel_2d(restored, mx, my, px, lx, py, ly)
    return omega @ (a_rb * element_matrix.reshape(-1))


def _lifted_adjoint(
    residual: np.ndarray,
    dechirp: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    shape_info: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    """Adjoint of _lifted_forward for projected-gradient RIS updates."""
    mx, my, px, lx, py, ly = shape_info
    element_vec = np.conj(a_rb) * (omega.conj().T @ residual)
    element_matrix = element_vec.reshape(mx, my)
    lifted = _block_dehankel_adjoint_2d(element_matrix, mx, my, px, lx, py, ly)
    return dechirp * lifted


def _physical_lifted_matrix(
    eta_local: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    shape_info: tuple[int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return X_phys and dechirp kernel for one candidate geometry."""
    mx, my, px, lx, py, ly = shape_info
    fresnel = _fresnel_response_matrix(eta_local, ris_grid, wavelength)
    dechirp = _dechirp_kernel(eta_local, ris_grid, wavelength, px, lx, py, ly)
    x_phys = dechirp * _block_hankel_2d(fresnel, px, lx, py, ly)
    return _rank_one_projection(x_phys), dechirp


def _compressed_lifted_candidate(
    c_tilde: np.ndarray,
    eta_local: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    eps: float,
    num_steps: int,
    lambda_phys: float,
    pgd_step_scale: float = 0.5,
) -> dict:
    """Solve the fixed-geometry compressed dechirped rank-one subproblem."""
    mx, my = _infer_ris_shape(ris_grid)
    px, lx = _hankel_window(mx)
    py, ly = _hankel_window(my)
    shape_info = (mx, my, px, lx, py, ly)
    x_phys, dechirp = _physical_lifted_matrix(eta_local, ris_grid, wavelength, shape_info)
    u_elem = _block_dehankel_2d(
        np.conj(dechirp) * x_phys, mx, my, px, lx, py, ly
    )

    def lift_from_element(u_matrix: np.ndarray) -> np.ndarray:
        return dechirp * _block_hankel_2d(u_matrix, px, lx, py, ly)

    def compressed_from_element(u_matrix: np.ndarray) -> np.ndarray:
        return omega @ (a_rb * u_matrix.reshape(-1))

    def pgd_objective_unscaled(u_matrix: np.ndarray) -> float:
        model = compressed_from_element(u_matrix)
        data = np.linalg.norm(model - c_tilde) ** 2
        lifted = lift_from_element(u_matrix)
        regularizer = lambda_phys * np.linalg.norm(lifted - x_phys) ** 2
        return float(data + regularizer)

    weights_2d = _block_hankel_inverse_weights_2d(mx, my, px, lx, py, ly)
    counts_2d = _block_hankel_counts_2d(mx, my, px, lx, py, ly)
    omega_eff = omega * a_rb[None, :]
    omega_norm = np.linalg.norm(omega_eff, 2)
    max_count = float(np.max(counts_2d))
    lambda_phys = max(float(lambda_phys), 0.0)
    step_scale = max(float(pgd_step_scale), eps)
    step = step_scale / (omega_norm**2 + lambda_phys * max_count + eps)

    current_obj = pgd_objective_unscaled(u_elem)
    accepted_steps = 0
    num_steps = max(int(num_steps), 0)

    for _ in range(num_steps):
        trial_step = step
        accepted = False
        for _ in range(3):
            model = compressed_from_element(u_elem)
            residual = model - c_tilde
            grad_vec = np.conj(a_rb) * (omega.conj().T @ residual)
            grad_elem = grad_vec.reshape(mx, my)

            lifted_current = lift_from_element(u_elem)
            lifted_anchor_residual = lifted_current - x_phys
            anchor_grad_elem = _block_hankel_adjoint_sum_2d(
                np.conj(dechirp) * lifted_anchor_residual,
                mx,
                my,
                px,
                lx,
                py,
                ly,
            )
            grad_elem = grad_elem + lambda_phys * anchor_grad_elem

            u_trial = u_elem - trial_step * weights_2d * grad_elem
            z_lift = lift_from_element(u_trial)
            z_rank1 = _rank_one_projection(z_lift)
            u_next = _block_dehankel_2d(
                np.conj(dechirp) * z_rank1, mx, my, px, lx, py, ly
            )
            next_obj = pgd_objective_unscaled(u_next)
            accept_tol = 1.0e-10 * max(1.0, current_obj)
            if next_obj <= current_obj + accept_tol:
                u_elem = u_next
                current_obj = next_obj
                accepted_steps += 1
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            break

    c_lifted = compressed_from_element(u_elem)
    data_residual, alpha = scaled_residual(c_tilde, c_lifted, eps)
    lifted_final = lift_from_element(u_elem)
    regularizer = lambda_phys * np.linalg.norm(lifted_final - x_phys) ** 2
    objective = data_residual + float(regularizer)
    return {
        "c_lifted": c_lifted,
        "eta_local": np.asarray(eta_local, dtype=float),
        "objective": float(objective),
        "data_residual": float(data_residual),
        "alpha": alpha,
        "pgd_unscaled_objective": float(current_obj),
        "pgd_accepted_steps": int(accepted_steps),
        "pgd_step": float(step),
    }


def project_ris_factor(
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    eps: float = 1e-10,
    current_eta: np.ndarray | None = None,
) -> dict:
    """Project a compressed RIS factor according to the paper's Mode-4 rule."""
    assert c_tilde.ndim == 1, "c_tilde must be a vector"
    assert omega.shape[0] == c_tilde.size, "Omega rows must match c_tilde length"
    assert omega.shape[1] == a_rb.size, "Omega columns must match RIS response length"

    lower = np.array(
        [
            search_config["range_bounds"][0],
            search_config["elev_bounds"][0],
            search_config["az_bounds"][0],
        ],
        dtype=float,
    )
    upper = np.array(
        [
            search_config["range_bounds"][1],
            search_config["elev_bounds"][1],
            search_config["az_bounds"][1],
        ],
        dtype=float,
    )

    projection_mode = str(search_config.get("projection_mode", "paper")).lower()
    use_local_grid = current_eta is not None and projection_mode != "exact"
    if use_local_grid:
        center = np.clip(np.asarray(current_eta, dtype=float), lower, upper)
        range_span = float(search_config.get("stage2_range_span", 0.45))
        angle_span = float(search_config.get("stage2_angle_span", 0.12))
        local_lower = np.maximum(
            lower, center - np.array([range_span, angle_span, angle_span])
        )
        local_upper = np.minimum(
            upper, center + np.array([range_span, angle_span, angle_span])
        )
        r_grid = np.linspace(
            local_lower[0], local_upper[0], int(search_config.get("stage2_num_range", 5))
        )
        e_grid = np.linspace(
            local_lower[1], local_upper[1], int(search_config.get("stage2_num_elev", 5))
        )
        a_grid = np.linspace(
            local_lower[2], local_upper[2], int(search_config.get("stage2_num_az", 7))
        )
        refine_lower = local_lower
        refine_upper = local_upper
    else:
        r_grid = np.linspace(*search_config["range_bounds"], search_config["num_range"])
        e_grid = np.linspace(*search_config["elev_bounds"], search_config["num_elev"])
        a_grid = np.linspace(*search_config["az_bounds"], search_config["num_az"])
        refine_lower = lower
        refine_upper = upper

    grid_candidates = [
        np.array([range_m, elevation, azimuth], dtype=float)
        for range_m in r_grid
        for elevation in e_grid
        for azimuth in a_grid
    ]

    coarse_candidates = []
    for eta_local in grid_candidates:
        h_model = compressed_exact_response(eta_local, omega, a_rb, ris_grid, wavelength)
        value, alpha = scaled_residual(c_tilde, h_model, eps)
        coarse_candidates.append((float(value), eta_local, alpha))
    coarse_candidates.sort(key=lambda item: item[0])
    best_value, best_eta, _ = coarse_candidates[0]

    def exact_objective(eta_local: np.ndarray) -> float:
        h_model = compressed_exact_response(eta_local, omega, a_rb, ris_grid, wavelength)
        value, _ = scaled_residual(c_tilde, h_model, eps)
        return value / (np.linalg.norm(c_tilde) ** 2 + eps)

    num_lift_candidates = int(search_config.get("num_lift_candidates", 4))
    num_lift_steps = int(search_config.get("num_lift_steps", 3))
    lambda_phys = float(search_config.get("lambda_phys", 1.0e-2))
    ris_pgd_step_scale = float(search_config.get("ris_pgd_step_scale", 0.5))
    lifted_best = None
    lifted_used_for_start = False

    if projection_mode != "exact":
        lift_candidates = grid_candidates if use_local_grid else [
            eta for _, eta, _ in coarse_candidates[:num_lift_candidates]
        ]
        for eta_candidate in lift_candidates:
            lifted = _compressed_lifted_candidate(
                c_tilde,
                eta_candidate,
                omega,
                a_rb,
                ris_grid,
                wavelength,
                eps,
                num_lift_steps,
                lambda_phys,
                ris_pgd_step_scale,
            )
            if lifted_best is None or lifted["objective"] < lifted_best["objective"]:
                lifted_best = lifted
        if (
            lifted_best is not None
            and lifted_best["data_residual"] <= best_value * (1.0 + 1.0e-8) + eps
        ):
            best_eta = lifted_best["eta_local"]
            lifted_used_for_start = True

    optimizer_message = (
        "physically anchored Fresnel dechirped rank-one candidate"
        if lifted_best is not None
        else "compressed exact spherical matching"
    )

    refine_starts = [coarse_candidates[0][1]]
    if lifted_best is not None:
        refine_starts.append(lifted_best["eta_local"])
    if current_eta is not None:
        refine_starts.append(current_eta)
    for _, eta_candidate, _ in coarse_candidates[
        : int(search_config.get("num_exact_refine_starts", 6))
    ]:
        refine_starts.append(eta_candidate)

    unique_starts = []
    for eta_start in refine_starts:
        eta_clipped = np.clip(np.asarray(eta_start, dtype=float), refine_lower, refine_upper)
        if not any(np.linalg.norm(eta_clipped - old) < 1e-9 for old in unique_starts):
            unique_starts.append(eta_clipped)

    best_exact_value = exact_objective(best_eta)
    best_exact_success = False
    if scipy_is_available():
        from scipy.optimize import minimize

        for eta_start in unique_starts:
            result = minimize(
                exact_objective,
                eta_start,
                method="L-BFGS-B",
                bounds=list(zip(refine_lower, refine_upper)),
                options={"maxiter": 100, "ftol": 1e-12},
            )
            if result.fun <= best_exact_value:
                best_eta = np.asarray(result.x, dtype=float)
                best_exact_value = float(result.fun)
                best_exact_success = bool(result.success)
        optimizer_message += f" + exact spherical L-BFGS-B success={best_exact_success}"
    else:
        best_info_message = ""
        for eta_start in unique_starts:
            span = np.maximum(refine_upper - refine_lower, eps)
            x0_scaled = (eta_start - refine_lower) / span

            def scaled_objective(x_scaled: np.ndarray) -> float:
                eta_local = refine_lower + np.clip(x_scaled, 0.0, 1.0) * span
                return exact_objective(eta_local)

            x_best, value, info = bounded_coordinate_search(
                scaled_objective,
                x0_scaled,
                np.zeros(3),
                np.ones(3),
                step0=0.10,
                max_iter=45,
                tol=1e-4,
            )
            if value <= best_exact_value:
                best_eta = refine_lower + x_best * span
                best_exact_value = float(value)
                best_info_message = info["message"]
        optimizer_message += f" + exact spherical {best_info_message}"

    c_norm_sq = np.linalg.norm(c_tilde) ** 2 + eps
    h_best = compressed_exact_response(best_eta, omega, a_rb, ris_grid, wavelength)
    exact_value, exact_alpha = scaled_residual(c_tilde, h_best, eps)
    c_projected = exact_alpha * h_best
    c_projected_norm = np.linalg.norm(c_projected)
    if c_projected_norm > eps:
        c_projected = c_projected / c_projected_norm
    else:
        h_norm = np.linalg.norm(h_best)
        c_projected = h_best / (h_norm + eps)

    final_value, final_alpha = scaled_residual(c_tilde, c_projected, eps)
    final_relative = float(np.sqrt(final_value / c_norm_sq))
    candidates = {
        "paper": {
            "c": c_projected,
            "eta_local": best_eta,
            "alpha": final_alpha,
            "data_residual": float(final_value),
            "relative_residual": final_relative,
        }
    }
    return {
        "c": c_projected,
        "eta_local": best_eta,
        "alpha": final_alpha,
        "relative_residual": final_relative,
        "selected_model": "exact_refined_from_lifted"
        if lifted_best is not None
        else "exact",
        "candidates": candidates,
        "coarse_eta_local": coarse_candidates[0][1],
        "coarse_relative_residual": float(
            np.sqrt(best_value / c_norm_sq)
        ),
        "exact_relative_residual": float(np.sqrt(exact_value / c_norm_sq)),
        "lifted_available": lifted_best is not None,
        "lifted_used": lifted_best is not None,
        "lifted_used_for_start": bool(lifted_used_for_start),
        "lifted_relative_residual": None
        if lifted_best is None
        else float(np.sqrt(lifted_best["data_residual"] / c_norm_sq)),
        "lifted_objective": None if lifted_best is None else lifted_best["objective"],
        "ris_pgd_accepted_steps": None
        if lifted_best is None
        else lifted_best.get("pgd_accepted_steps"),
        "ris_pgd_unscaled_objective": None
        if lifted_best is None
        else lifted_best.get("pgd_unscaled_objective"),
        "optimizer_message": optimizer_message,
    }
