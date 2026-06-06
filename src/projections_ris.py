"""Compressed near-field RIS projection helpers.

The default Stage-II RIS projection is weighted exact spherical variable
projection with multi-start refinement. The current Stage-I/previous-iteration
geometry is the primary start. Quadratic-distance and Fresnel/dechirped
rank-one lifting are optional auxiliary starts only; the returned RIS factor is
always reconstructed from the exact compressed spherical-wave response.
"""

from __future__ import annotations

import time

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


def exact_spherical_response_and_jacobian(
    eta_local: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact compressed spherical response and eta Jacobian.

    The Jacobian columns are derivatives with respect to
    ``[range, elevation, azimuth]`` under the phase-dominant spherical model.
    No amplitude factor is used, matching ``near_field_spherical_response``.
    """
    assert omega.ndim == 2, "omega must have shape T x M_R"
    assert a_rb.ndim == 1, "a_rb must have shape M_R"
    assert ris_grid.ndim == 2 and ris_grid.shape[1] == 3, "ris_grid must be M_R x 3"
    assert omega.shape[1] == a_rb.size == ris_grid.shape[0]
    range_m, elevation, azimuth = np.asarray(eta_local, dtype=float)
    kappa = 2.0 * np.pi / wavelength

    ce = np.cos(elevation)
    se = np.sin(elevation)
    ca = np.cos(azimuth)
    sa = np.sin(azimuth)
    unit_vec = np.array([ce * ca, ce * sa, se], dtype=float)
    ds_delev = np.array([-se * ca, -se * sa, ce], dtype=float)
    ds_daz = np.array([-ce * sa, ce * ca, 0.0], dtype=float)

    q_local = range_m * unit_vec
    diff = q_local[None, :] - ris_grid
    distances = np.linalg.norm(diff, axis=1)
    safe_distances = np.maximum(distances, eps)
    safe_range = max(float(range_m), eps)
    delta = distances - range_m
    u_vec = np.exp(-1j * kappa * delta)

    geom_grad = diff / safe_distances[:, None] - q_local[None, :] / safe_range
    dq = np.column_stack(
        [
            unit_vec,
            range_m * ds_delev,
            range_m * ds_daz,
        ]
    )
    ddelta = geom_grad @ dq
    du = -1j * kappa * u_vec[:, None] * ddelta

    a_eff = omega * a_rb[None, :]
    h_vec = a_eff @ u_vec
    jac = a_eff @ du
    return h_vec, jac


def local_ris_search_config(scene: dict, config: dict, path: int) -> dict:
    """Build RIS-specific geometry-search bounds from UE position bounds."""
    base = dict(config["ris_search"])
    base["panel_index"] = int(path)
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


def _ris_cache_key(
    search_config: dict,
    current_eta: np.ndarray | None,
    use_local_grid: bool,
    wavelength: float,
    ris_grid: np.ndarray,
    omega: np.ndarray,
) -> tuple:
    """Return a deterministic key for a panel/local RIS codebook."""
    if use_local_grid and current_eta is not None:
        center = tuple(np.round(np.asarray(current_eta, dtype=float), 12))
        span = (
            float(search_config.get("stage2_range_span", 0.45)),
            float(search_config.get("stage2_angle_span", 0.12)),
        )
        grid_size = (
            int(search_config.get("stage2_num_range", 5)),
            int(search_config.get("stage2_num_elev", 5)),
            int(search_config.get("stage2_num_az", 7)),
        )
    else:
        center = (
            tuple(np.round(search_config["range_bounds"], 12)),
            tuple(np.round(search_config["elev_bounds"], 12)),
            tuple(np.round(search_config["az_bounds"], 12)),
        )
        span = ("global",)
        grid_size = (
            int(search_config.get("num_range", 5)),
            int(search_config.get("num_elev", 5)),
            int(search_config.get("num_az", 7)),
        )
    return (
        int(search_config.get("panel_index", -1)),
        center,
        span,
        grid_size,
        float(wavelength),
        tuple(ris_grid.shape),
        tuple(omega.shape),
    )


def _cached_compressed_responses(
    grid_candidates: list[np.ndarray],
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    current_eta: np.ndarray | None,
    use_local_grid: bool,
) -> tuple[list[np.ndarray], float]:
    """Return compressed steering vectors for a grid, using an optional call-local cache."""
    cache = search_config.get("_ris_projection_cache")
    key = _ris_cache_key(
        search_config, current_eta, use_local_grid, wavelength, ris_grid, omega
    )
    if isinstance(cache, dict) and key in cache:
        return cache[key], 0.0
    start = time.perf_counter()
    responses = [
        compressed_exact_response(eta, omega, a_rb, ris_grid, wavelength)
        for eta in grid_candidates
    ]
    elapsed = time.perf_counter() - start
    if isinstance(cache, dict):
        cache[key] = responses
    return responses, elapsed


def scaled_residual(c_tilde: np.ndarray, h_model: np.ndarray, eps: float) -> tuple[float, complex]:
    """Return min_alpha ||c_tilde - alpha h_model||^2 and alpha."""
    denom = np.vdot(h_model, h_model) + eps
    alpha = np.vdot(h_model, c_tilde) / denom
    residual = np.linalg.norm(c_tilde - alpha * h_model) ** 2
    return float(residual), alpha


def _apply_weight(vec: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    """Apply identity, diagonal, or full Hermitian sample weight."""
    vec = np.asarray(vec)
    if weight is None:
        return vec
    weight_array = np.asarray(weight)
    if weight_array.ndim == 1:
        if weight_array.shape[0] != vec.shape[0]:
            raise ValueError("weight vector length must match response length")
        return weight_array * vec
    if weight_array.ndim == 2:
        if weight_array.shape != (vec.shape[0], vec.shape[0]):
            raise ValueError("weight matrix must have shape T x T")
        return weight_array @ vec
    raise ValueError("weight must be None, a vector, or a matrix")


def _weighted_inner(x_vec: np.ndarray, y_vec: np.ndarray, weight: np.ndarray | None) -> complex:
    """Return x^H W y for identity, diagonal, or full sample weight."""
    return np.vdot(x_vec, _apply_weight(y_vec, weight))


def _weighted_norm_sq(x_vec: np.ndarray, weight: np.ndarray | None) -> float:
    """Return real part of x^H W x."""
    return float(np.real(_weighted_inner(x_vec, x_vec, weight)))


def _vp_alpha(
    c_tilde: np.ndarray,
    h_model: np.ndarray,
    weight: np.ndarray | None,
    eps: float,
) -> complex:
    """Return weighted variable-projection gain for a fixed response."""
    return _weighted_inner(h_model, c_tilde, weight) / (
        _weighted_inner(h_model, h_model, weight) + eps
    )


def _wesvp_objective_and_grad(
    eta_local: np.ndarray,
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    lower: np.ndarray,
    upper: np.ndarray,
    weight: np.ndarray | None = None,
    eps: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Weighted exact spherical VP objective and analytic gradient."""
    eta_local = np.clip(np.asarray(eta_local, dtype=float), lower, upper)
    h_vec, jac = exact_spherical_response_and_jacobian(
        eta_local, omega, a_rb, ris_grid, wavelength, eps
    )
    s_val = _weighted_inner(h_vec, c_tilde, weight)
    n_val = float(np.real(_weighted_inner(h_vec, h_vec, weight))) + eps
    const = _weighted_norm_sq(c_tilde, weight)
    objective = const - (abs(s_val) ** 2) / n_val

    grad = np.empty(3, dtype=float)
    for dim in range(3):
        h_x = jac[:, dim]
        s_x = _weighted_inner(h_x, c_tilde, weight)
        n_x = 2.0 * float(np.real(_weighted_inner(h_x, h_vec, weight)))
        numerator = 2.0 * float(np.real(np.conj(s_val) * s_x)) * n_val
        numerator -= (abs(s_val) ** 2) * n_x
        grad[dim] = -numerator / (n_val**2)
    return float(np.real(objective)), grad


def _element_domain_proxy(
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    weight: np.ndarray | None = None,
    reg: float = 1e-6,
    eps: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Estimate an element-domain phase proxy by weighted ridge LS."""
    a_eff = omega * a_rb[None, :]
    reg = max(float(reg), eps)
    t_dim, m_dim = a_eff.shape

    if weight is None:
        system = a_eff @ a_eff.conj().T + reg * np.eye(t_dim, dtype=complex)
        try:
            y_vec = np.linalg.solve(system, c_tilde)
        except np.linalg.LinAlgError:
            y_vec = np.linalg.pinv(system) @ c_tilde
        u_proxy = a_eff.conj().T @ y_vec
    elif np.asarray(weight).ndim == 1:
        weight_vec = np.asarray(weight)
        if weight_vec.shape[0] != t_dim:
            raise ValueError("weight vector length must match response length")
        sqrt_weight = np.sqrt(np.maximum(np.real(weight_vec), 0.0))
        a_weighted = sqrt_weight[:, None] * a_eff
        c_weighted = sqrt_weight * c_tilde
        system = a_weighted @ a_weighted.conj().T + reg * np.eye(t_dim, dtype=complex)
        try:
            y_vec = np.linalg.solve(system, c_weighted)
        except np.linalg.LinAlgError:
            y_vec = np.linalg.pinv(system) @ c_weighted
        u_proxy = a_weighted.conj().T @ y_vec
    else:
        weight_mat = np.asarray(weight)
        if weight_mat.shape != (t_dim, t_dim):
            raise ValueError("weight matrix must have shape T x T")
        if m_dim <= 512:
            gram = a_eff.conj().T @ (weight_mat @ a_eff)
            rhs = a_eff.conj().T @ (weight_mat @ c_tilde)
            system = gram + reg * np.eye(m_dim, dtype=complex)
            try:
                u_proxy = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                u_proxy = np.linalg.pinv(system) @ rhs
        else:
            try:
                weight_inv = np.linalg.inv(weight_mat)
            except np.linalg.LinAlgError:
                weight_inv = np.linalg.pinv(weight_mat)
            system = a_eff @ a_eff.conj().T + reg * weight_inv
            try:
                y_vec = np.linalg.solve(system, c_tilde)
            except np.linalg.LinAlgError:
                y_vec = np.linalg.pinv(system) @ c_tilde
            u_proxy = a_eff.conj().T @ y_vec

    residual = c_tilde - a_eff @ u_proxy
    denom = max(_weighted_norm_sq(c_tilde, weight), eps)
    rel_residual = float(np.sqrt(max(_weighted_norm_sq(residual, weight), 0.0) / denom))
    return u_proxy, rel_residual


def _azimuth_in_bounds(azimuth: float, bounds: tuple[float, float]) -> float | None:
    """Return an equivalent azimuth inside possibly unwrapped bounds."""
    lower, upper = bounds
    for shift in range(-2, 3):
        candidate = float(azimuth + 2.0 * np.pi * shift)
        if lower <= candidate <= upper:
            return candidate
    return None


def _qd_initializer_from_element_proxy(
    u_proxy: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    eps: float = 1e-10,
) -> dict | None:
    """Quadratic-distance geometry warm start from an element-domain proxy.

    The result is only a warm start for exact spherical VP; it is never returned
    as the final RIS projection.
    """
    mx, my = _infer_ris_shape(ris_grid)
    if u_proxy.size != mx * my:
        raise ValueError("u_proxy length must match RIS grid size")
    grid = ris_grid.reshape(mx, my, 3)
    radii = np.linalg.norm(ris_grid, axis=1)
    ref_index = int(np.argmin(radii))
    ref_is_center = bool(radii[ref_index] <= eps)
    ref_value = u_proxy[ref_index]
    if abs(ref_value) <= eps:
        return None

    relative_phase = np.angle(u_proxy * np.conj(ref_value))
    phase_grid = relative_phase.reshape(mx, my)
    unwrapped = np.unwrap(np.unwrap(phase_grid, axis=0), axis=1)
    delta = -unwrapped.reshape(-1) / (2.0 * np.pi / wavelength)

    x_coord = ris_grid[:, 0]
    y_coord = ris_grid[:, 1]
    design = np.column_stack([2.0 * x_coord, 2.0 * y_coord])
    range_min, range_max = search_config["range_bounds"]
    elev_bounds = search_config["elev_bounds"]
    az_bounds = search_config["az_bounds"]
    qd_num_range = max(int(search_config.get("qd_num_range", 41)), 3)
    max_ls_rel = float(search_config.get("qd_max_ls_relative_residual", 1.0))

    best = None
    for range_m in np.linspace(range_min, range_max, qd_num_range):
        if range_m <= eps:
            continue
        rhs = x_coord**2 + y_coord**2 - 2.0 * range_m * delta - delta**2
        try:
            q_xy = np.linalg.lstsq(design, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        qx, qy = float(q_xy[0]), float(q_xy[1])
        radial_sq = qx**2 + qy**2
        if radial_sq >= range_m**2:
            continue
        predicted = design @ q_xy
        ls_rel = float(
            np.linalg.norm(predicted - rhs)
            / max(np.linalg.norm(rhs), eps)
        )
        if not np.isfinite(ls_rel) or ls_rel > max_ls_rel:
            continue

        qz_abs = float(np.sqrt(max(range_m**2 - radial_sq, 0.0)))
        for qz in (qz_abs, -qz_abs):
            elevation = float(np.arcsin(np.clip(qz / range_m, -1.0, 1.0)))
            if not (elev_bounds[0] <= elevation <= elev_bounds[1]):
                continue
            azimuth = _azimuth_in_bounds(np.arctan2(qy, qx), az_bounds)
            if azimuth is None:
                continue
            eta_local = np.array([range_m, elevation, azimuth], dtype=float)
            if best is None or ls_rel < best["qd_ls_relative_residual"]:
                best = {
                    "eta_local": eta_local,
                    "qd_ls_relative_residual": ls_rel,
                    "reference_index": ref_index,
                    "reference_is_center": ref_is_center,
                    "warning": ""
                    if ref_is_center
                    else "QD reference is nearest element; global phase is absorbed by alpha",
                }
    return best


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


def _ris_search_bounds(search_config: dict) -> tuple[np.ndarray, np.ndarray]:
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
    return lower, upper


def _ris_grid_candidates(
    search_config: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    current_eta: np.ndarray | None,
    use_local_grid: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if use_local_grid and current_eta is not None:
        center = np.clip(np.asarray(current_eta, dtype=float), lower, upper)
        range_span = float(search_config.get("stage2_range_span", 0.45))
        angle_span = float(search_config.get("stage2_angle_span", 0.12))
        refine_lower = np.maximum(
            lower, center - np.array([range_span, angle_span, angle_span])
        )
        refine_upper = np.minimum(
            upper, center + np.array([range_span, angle_span, angle_span])
        )
        r_grid = np.linspace(
            refine_lower[0], refine_upper[0], int(search_config.get("stage2_num_range", 5))
        )
        e_grid = np.linspace(
            refine_lower[1], refine_upper[1], int(search_config.get("stage2_num_elev", 5))
        )
        a_grid = np.linspace(
            refine_lower[2], refine_upper[2], int(search_config.get("stage2_num_az", 7))
        )
    else:
        r_grid = np.linspace(*search_config["range_bounds"], int(search_config["num_range"]))
        e_grid = np.linspace(*search_config["elev_bounds"], int(search_config["num_elev"]))
        a_grid = np.linspace(*search_config["az_bounds"], int(search_config["num_az"]))
        refine_lower = lower
        refine_upper = upper

    candidates = [
        np.array([range_m, elevation, azimuth], dtype=float)
        for range_m in r_grid
        for elevation in e_grid
        for azimuth in a_grid
    ]
    return candidates, refine_lower, refine_upper


def _ris_candidate_record(
    model: str,
    eta_local: np.ndarray,
    local_residual: float,
    exact_refined: bool,
    selected: bool = False,
) -> dict:
    """Build a compact RIS candidate-ranking diagnostic row."""
    eta = np.asarray(eta_local, dtype=float)
    return {
        "_eta_local": eta,
        "model": str(model),
        "range_m": float(eta[0]),
        "elev_deg": float(np.rad2deg(eta[1])),
        "az_deg": float(np.rad2deg(eta[2])),
        "local_residual": float(local_residual),
        "exact_refined": bool(exact_refined),
        "selected": bool(selected),
    }


def _finalize_ris_candidate_ranking(records: list[dict], top_k: int = 3) -> list[dict]:
    """Sort and de-duplicate RIS candidate diagnostics without changing selection."""
    sorted_records = sorted(
        records,
        key=lambda item: (
            not np.isfinite(item["local_residual"]),
            item["local_residual"],
            not item["selected"],
        ),
    )
    unique_records: list[dict] = []
    for record in sorted_records:
        eta = record["_eta_local"]
        duplicate = None
        for existing in unique_records:
            if np.linalg.norm(eta - existing["_eta_local"]) <= 1.0e-8:
                duplicate = existing
                break
        if duplicate is None:
            unique_records.append(dict(record))
        else:
            duplicate["selected"] = bool(duplicate["selected"] or record["selected"])
            duplicate["exact_refined"] = bool(
                duplicate["exact_refined"] or record["exact_refined"]
            )
            if record["selected"]:
                duplicate["model"] = record["model"]

    ranking = []
    for rank, record in enumerate(unique_records[:top_k], start=1):
        item = dict(record)
        item.pop("_eta_local", None)
        item["rank"] = rank
        ranking.append(item)
    return ranking


def _eta_within_bounds(eta_local: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    """Return whether eta is finite and inside closed box bounds."""
    eta = np.asarray(eta_local, dtype=float)
    return bool(
        eta.shape == (3,)
        and np.all(np.isfinite(eta))
        and np.all(eta >= lower)
        and np.all(eta <= upper)
    )


def _project_ris_factor_wesvp_ms(
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    eps: float = 1e-10,
    current_eta: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> dict:
    """Weighted exact spherical variable projection with multi-start refinement."""
    timing = {
        "stage2_time_ris_codebook_build": 0.0,
        "stage2_time_ris_correlation": 0.0,
        "stage2_time_ris_refine": 0.0,
    }
    lower, upper = _ris_search_bounds(search_config)
    projection_mode = str(search_config.get("projection_mode", "wesvp_ms")).lower()
    use_local_grid = current_eta is not None
    grid_candidates, refine_lower, refine_upper = _ris_grid_candidates(
        search_config, lower, upper, current_eta, use_local_grid
    )
    c_norm_sq = max(_weighted_norm_sq(c_tilde, weight), eps)

    grid_responses, build_time = _cached_compressed_responses(
        grid_candidates,
        omega,
        a_rb,
        ris_grid,
        wavelength,
        search_config,
        current_eta,
        use_local_grid,
    )
    timing["stage2_time_ris_codebook_build"] += float(build_time)
    coarse_candidates = []
    corr_start = time.perf_counter()
    const_norm = _weighted_norm_sq(c_tilde, weight)
    for eta_local, h_vec in zip(grid_candidates, grid_responses):
        alpha = _vp_alpha(c_tilde, h_vec, weight, eps)
        value = const_norm - abs(_weighted_inner(h_vec, c_tilde, weight)) ** 2 / (
            _weighted_inner(h_vec, h_vec, weight).real + eps
        )
        coarse_candidates.append((float(value), eta_local))
    timing["stage2_time_ris_correlation"] += time.perf_counter() - corr_start
    coarse_candidates.sort(key=lambda item: item[0])
    best_coarse_value, best_coarse_eta = coarse_candidates[0]
    starts: list[tuple[np.ndarray, str]] = []
    primary_start_source = "stage1"
    j_current_eta_before_refine = None
    if current_eta is not None and _eta_within_bounds(
        np.asarray(current_eta, dtype=float), refine_lower, refine_upper
    ):
        starts.append((np.asarray(current_eta, dtype=float), "current_eta"))
        primary_start_source = "current_eta"
        j_current_eta_before_refine, _ = _wesvp_objective_and_grad(
            np.asarray(current_eta, dtype=float),
            c_tilde,
            omega,
            a_rb,
            ris_grid,
            wavelength,
            refine_lower,
            refine_upper,
            weight,
            eps,
        )
        j_current_eta_before_refine = float(j_current_eta_before_refine)

    starts.append((best_coarse_eta, "exact_grid"))
    j_grid_before_refine = float(best_coarse_value)

    use_qd = bool(search_config.get("use_qd_init", False)) and projection_mode == "wesvp_ms"
    qd_attempted = bool(use_qd)
    qd_available = False
    qd_used_as_start = False
    qd_proxy_relative_residual = float("nan")
    qd_rejected_reason = "disabled"
    j_qd_before_refine = None
    qd_diag = None
    if use_qd:
        u_proxy, qd_proxy_relative_residual = _element_domain_proxy(
            c_tilde,
            omega,
            a_rb,
            weight=weight,
            reg=float(search_config.get("qd_proxy_reg", 1.0e-6)),
            eps=eps,
        )
        if qd_proxy_relative_residual <= float(
            search_config.get("qd_proxy_max_rel_residual", 0.5)
        ):
            qd_diag = _qd_initializer_from_element_proxy(
                u_proxy, ris_grid, wavelength, search_config, eps
            )
            if qd_diag is not None:
                if _eta_within_bounds(qd_diag["eta_local"], refine_lower, refine_upper):
                    qd_available = True
                    starts.append((qd_diag["eta_local"], "qd"))
                    qd_used_as_start = True
                    qd_rejected_reason = ""
                    j_qd_before_refine, _ = _wesvp_objective_and_grad(
                        qd_diag["eta_local"],
                        c_tilde,
                        omega,
                        a_rb,
                        ris_grid,
                        wavelength,
                        refine_lower,
                        refine_upper,
                        weight,
                        eps,
                    )
                    j_qd_before_refine = float(j_qd_before_refine)
                else:
                    qd_rejected_reason = "eta_out_of_bounds"
            else:
                qd_rejected_reason = "initializer_failed"
        else:
            qd_rejected_reason = "proxy_residual_above_threshold"

    use_fresnel = bool(search_config.get("use_fresnel_warm_start", True))
    use_fresnel = use_fresnel and projection_mode == "wesvp_ms"
    lifted_best = None
    fresnel_used_as_start = False
    if use_fresnel:
        num_lift_candidates = int(search_config.get("num_lift_candidates", 4))
        num_lift_steps = int(search_config.get("num_lift_steps", 3))
        lambda_phys = float(search_config.get("lambda_phys", 1.0e-2))
        ris_pgd_step_scale = float(search_config.get("ris_pgd_step_scale", 0.5))
        for _, eta_candidate in coarse_candidates[:num_lift_candidates]:
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
        if lifted_best is not None:
            if _eta_within_bounds(lifted_best["eta_local"], refine_lower, refine_upper):
                starts.append((lifted_best["eta_local"], "fresnel"))
                fresnel_used_as_start = True

    unique_starts: list[tuple[np.ndarray, str]] = []
    for eta_start, source in starts:
        eta_start = np.asarray(eta_start, dtype=float)
        if not _eta_within_bounds(eta_start, refine_lower, refine_upper):
            continue
        if not any(np.linalg.norm(eta_start - old_eta) < 1e-9 for old_eta, _ in unique_starts):
            unique_starts.append((eta_start, source))

    def objective(eta_local: np.ndarray) -> float:
        value, _ = _wesvp_objective_and_grad(
            eta_local,
            c_tilde,
            omega,
            a_rb,
            ris_grid,
            wavelength,
            refine_lower,
            refine_upper,
            weight,
            eps,
        )
        return value

    def objective_grad(eta_local: np.ndarray) -> tuple[float, np.ndarray]:
        return _wesvp_objective_and_grad(
            eta_local,
            c_tilde,
            omega,
            a_rb,
            ris_grid,
            wavelength,
            refine_lower,
            refine_upper,
            weight,
            eps,
        )

    candidate_sources = [source for _, source in unique_starts]
    best_eta = None
    best_value = float("inf")
    best_eta_source = ""
    optimizer_messages = []
    refined_candidates = []
    analytic_jacobian_used = False
    if scipy_is_available():
        from scipy.optimize import minimize

        analytic_jacobian_used = True
        refine_start = time.perf_counter()
        for eta_start, source in unique_starts:
            result = minimize(
                lambda eta: objective_grad(eta)[0],
                eta_start,
                jac=lambda eta: objective_grad(eta)[1],
                method="L-BFGS-B",
                bounds=list(zip(refine_lower, refine_upper)),
                options={
                    "maxiter": int(search_config.get("wesvp_max_iter", 100)),
                    "ftol": float(search_config.get("wesvp_ftol", 1.0e-12)),
                    "gtol": float(search_config.get("wesvp_gtol", 1.0e-8)),
                },
            )
            optimizer_messages.append(f"{source}: L-BFGS-B success={bool(result.success)}")
            refined_candidates.append(
                {
                    "source": source,
                    "eta_local": np.asarray(result.x, dtype=float),
                    "objective": float(result.fun),
                    "success": bool(result.success),
                }
            )
            if float(result.fun) < best_value:
                best_eta = np.asarray(result.x, dtype=float)
                best_value = float(result.fun)
                best_eta_source = source
        timing["stage2_time_ris_refine"] += time.perf_counter() - refine_start
    else:
        refine_start = time.perf_counter()
        for eta_start, source in unique_starts:
            span = np.maximum(refine_upper - refine_lower, eps)
            x0_scaled = (eta_start - refine_lower) / span

            def scaled_objective(x_scaled: np.ndarray) -> float:
                eta_local = refine_lower + np.clip(x_scaled, 0.0, 1.0) * span
                return objective(eta_local)

            x_best, value, info = bounded_coordinate_search(
                scaled_objective,
                x0_scaled,
                np.zeros(3),
                np.ones(3),
                step0=0.10,
                max_iter=45,
                tol=1e-4,
            )
            optimizer_messages.append(f"{source}: {info['message']}")
            eta_best_candidate = refine_lower + x_best * span
            refined_candidates.append(
                {
                    "source": source,
                    "eta_local": eta_best_candidate,
                    "objective": float(value),
                    "success": bool(info["success"]),
                }
            )
            if float(value) < best_value:
                best_eta = eta_best_candidate
                best_value = float(value)
                best_eta_source = source
        timing["stage2_time_ris_refine"] += time.perf_counter() - refine_start

    assert best_eta is not None, "WESVP-MS requires at least one valid start"
    h_best = compressed_exact_response(best_eta, omega, a_rb, ris_grid, wavelength)
    alpha_raw = _vp_alpha(c_tilde, h_best, weight, eps)
    c_raw = alpha_raw * h_best
    raw_norm = float(np.linalg.norm(c_raw))
    if raw_norm > eps:
        c_projected = c_raw / raw_norm
        absorbed_norm = raw_norm
    else:
        h_norm = np.linalg.norm(h_best)
        c_projected = h_best / (h_norm + eps)
        absorbed_norm = 1.0

    final_value = _weighted_norm_sq(c_tilde - _vp_alpha(c_tilde, c_projected, weight, eps) * c_projected, weight)
    alpha_final = _vp_alpha(c_tilde, c_projected, weight, eps)
    final_relative = float(np.sqrt(max(final_value, 0.0) / c_norm_sq))
    coarse_relative = float(np.sqrt(max(best_coarse_value, 0.0) / c_norm_sq))
    exact_relative = float(np.sqrt(max(best_value, 0.0) / c_norm_sq))
    candidates = {
        "wesvp_ms": {
            "c": c_projected,
            "eta_local": best_eta,
            "alpha": alpha_final,
            "data_residual": float(final_value),
            "relative_residual": final_relative,
        }
    }
    if qd_used_as_start:
        candidates["qd"] = qd_diag
    if lifted_best is not None:
        candidates["fresnel_warm_start"] = lifted_best

    candidate_records = [
        _ris_candidate_record(
            "wesvp_ms",
            best_eta,
            final_relative,
            exact_refined=True,
            selected=True,
        )
    ]
    for rank, (value, eta_candidate) in enumerate(coarse_candidates[:3], start=1):
        candidate_records.append(
            _ris_candidate_record(
                f"exact_grid_{rank}",
                eta_candidate,
                float(np.sqrt(max(value, 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    if current_eta is not None:
        candidate_records.append(
            _ris_candidate_record(
                "current_eta",
                np.asarray(current_eta, dtype=float),
                float(np.sqrt(max(objective(current_eta), 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    if qd_used_as_start:
        candidate_records.append(
            _ris_candidate_record(
                "qd",
                qd_diag["eta_local"],
                float(np.sqrt(max(objective(qd_diag["eta_local"]), 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    if lifted_best is not None:
        candidate_records.append(
            _ris_candidate_record(
                "fresnel_warm_start",
                lifted_best["eta_local"],
                float(np.sqrt(max(lifted_best["data_residual"], 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    for refined in refined_candidates:
        candidate_records.append(
            _ris_candidate_record(
                f"exact_refined_{refined['source']}",
                refined["eta_local"],
                float(np.sqrt(max(refined["objective"], 0.0) / c_norm_sq)),
                exact_refined=True,
            )
        )
    candidate_ranking = _finalize_ris_candidate_ranking(candidate_records)

    optimizer_message = "; ".join(optimizer_messages) if optimizer_messages else "grid only"
    return {
        "c": c_projected,
        "eta_local": best_eta,
        "alpha": alpha_final,
        "relative_residual": final_relative,
        "selected_model": "wesvp_ms",
        "candidates": candidates,
        "candidate_ranking": candidate_ranking,
        "coarse_eta_local": best_coarse_eta,
        "coarse_relative_residual": coarse_relative,
        "exact_relative_residual": exact_relative,
        "optimizer_message": optimizer_message,
        "raw_alpha": alpha_raw,
        "raw_norm": raw_norm,
        "absorbed_norm": absorbed_norm,
        "wesvp_objective": float(best_value),
        "wesvp_relative_residual": exact_relative,
        "best_eta_source": best_eta_source,
        "primary_start_source": primary_start_source,
        "candidate_sources": candidate_sources,
        "selected_start_source": best_eta_source,
        "selected_after_refinement_source": f"exact_refined_{best_eta_source}",
        "qd_attempted": bool(qd_attempted),
        "qd_available": bool(qd_available),
        "qd_used_as_start": bool(qd_used_as_start),
        "qd_rejected_reason": qd_rejected_reason,
        "qd_proxy_relative_residual": float(qd_proxy_relative_residual),
        "J_current_eta_before_refine": j_current_eta_before_refine,
        "J_grid_before_refine": j_grid_before_refine,
        "J_qd_before_refine": j_qd_before_refine,
        "J_selected_after_refine": float(best_value),
        "fresnel_used_as_start": bool(fresnel_used_as_start),
        "analytic_jacobian_used": bool(analytic_jacobian_used),
        "lifted_available": lifted_best is not None,
        "lifted_used": False,
        "lifted_used_for_start": bool(fresnel_used_as_start),
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
        **timing,
    }


def _project_ris_factor_legacy(
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
    timing = {
        "stage2_time_ris_codebook_build": 0.0,
        "stage2_time_ris_correlation": 0.0,
        "stage2_time_ris_refine": 0.0,
    }
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
    c_norm_sq = np.linalg.norm(c_tilde) ** 2 + eps

    def exact_objective(eta_local: np.ndarray) -> float:
        h_model = compressed_exact_response(eta_local, omega, a_rb, ris_grid, wavelength)
        value, _ = scaled_residual(c_tilde, h_model, eps)
        return value / c_norm_sq

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
    refined_candidates = []
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
            refined_candidates.append(
                {
                    "source": "exact_refine",
                    "eta_local": np.asarray(result.x, dtype=float),
                    "objective": float(result.fun),
                    "success": bool(result.success),
                }
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
            eta_best_candidate = refine_lower + x_best * span
            refined_candidates.append(
                {
                    "source": "exact_refine",
                    "eta_local": eta_best_candidate,
                    "objective": float(value),
                    "success": bool(info["success"]),
                }
            )
            if value <= best_exact_value:
                best_eta = eta_best_candidate
                best_exact_value = float(value)
                best_info_message = info["message"]
        optimizer_message += f" + exact spherical {best_info_message}"

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
    selected_model = "exact_refined_from_lifted" if lifted_best is not None else "exact"
    candidate_records = [
        _ris_candidate_record(
            selected_model,
            best_eta,
            final_relative,
            exact_refined=True,
            selected=True,
        )
    ]
    for rank, (value, eta_candidate, _) in enumerate(coarse_candidates[:3], start=1):
        candidate_records.append(
            _ris_candidate_record(
                f"exact_grid_{rank}",
                eta_candidate,
                float(np.sqrt(max(value, 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    if current_eta is not None:
        candidate_records.append(
            _ris_candidate_record(
                "current_eta",
                np.asarray(current_eta, dtype=float),
                float(np.sqrt(max(exact_objective(current_eta), 0.0))),
                exact_refined=False,
            )
        )
    if lifted_best is not None:
        candidate_records.append(
            _ris_candidate_record(
                "fresnel_warm_start",
                lifted_best["eta_local"],
                float(np.sqrt(max(lifted_best["data_residual"], 0.0) / c_norm_sq)),
                exact_refined=False,
            )
        )
    for refined in refined_candidates:
        candidate_records.append(
            _ris_candidate_record(
                f"exact_refined_{refined['source']}",
                refined["eta_local"],
                float(np.sqrt(max(refined["objective"], 0.0))),
                exact_refined=True,
            )
        )
    candidate_ranking = _finalize_ris_candidate_ranking(candidate_records)
    return {
        "c": c_projected,
        "eta_local": best_eta,
        "alpha": final_alpha,
        "relative_residual": final_relative,
        "selected_model": selected_model,
        "candidates": candidates,
        "candidate_ranking": candidate_ranking,
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
        **timing,
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
    weight: np.ndarray | None = None,
) -> dict:
    """Project a compressed RIS factor onto a spherical-wave manifold.

    The default mode is WESVP-MS: weighted exact spherical variable projection
    with current-eta-first multi-start refinement and optional QD auxiliary
    start. The legacy ``paper`` mode keeps the previous Fresnel/dechirped
    rank-one baseline, while ``exact`` keeps the previous exact-only baseline.
    """
    assert c_tilde.ndim == 1, "c_tilde must be a vector"
    assert omega.shape[0] == c_tilde.size, "Omega rows must match c_tilde length"
    assert omega.shape[1] == a_rb.size, "Omega columns must match RIS response length"

    projection_mode = str(search_config.get("projection_mode", "wesvp_ms")).lower()
    if projection_mode in ("wesvp_ms", "exact_vp", "spherical_vp"):
        local_config = dict(search_config)
        if projection_mode in ("exact_vp", "spherical_vp"):
            local_config["use_qd_init"] = False
            local_config["use_fresnel_warm_start"] = False
        return _project_ris_factor_wesvp_ms(
            c_tilde,
            omega,
            a_rb,
            ris_grid,
            wavelength,
            local_config,
            eps=eps,
            current_eta=current_eta,
            weight=weight,
        )
    if projection_mode in ("paper", "exact"):
        return _project_ris_factor_legacy(
            c_tilde,
            omega,
            a_rb,
            ris_grid,
            wavelength,
            search_config,
            eps=eps,
            current_eta=current_eta,
        )
    raise ValueError(f"unknown RIS projection_mode {projection_mode!r}")
