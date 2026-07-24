"""Geometry and array-response helpers for the RIS-EVS-OFDM model."""

from __future__ import annotations

import numpy as np


def ue_box_corners(ue_bounds: np.ndarray) -> np.ndarray:
    """Return the eight corners of an axis-aligned 3-D UE box."""
    bounds = np.asarray(ue_bounds, dtype=float)
    if bounds.shape != (3, 2) or np.any(~np.isfinite(bounds)):
        raise ValueError("ue_bounds must have finite shape (3, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("ue_bounds lower limits must be smaller than upper limits")
    return np.asarray(
        [
            [x_coord, y_coord, z_coord]
            for x_coord in bounds[0]
            for y_coord in bounds[1]
            for z_coord in bounds[2]
        ],
        dtype=float,
    )


def rotation_from_panel_normal(normal_global: np.ndarray) -> np.ndarray:
    """Build the deterministic wall-like global-to-RIS rotation for one panel."""
    normal = np.asarray(normal_global, dtype=float).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError("panel normal must be finite and nonzero")
    normal = normal / normal_norm
    vertical = np.array([0.0, 0.0, 1.0])
    tangent_vertical = vertical - float(vertical @ normal) * normal
    if np.linalg.norm(tangent_vertical) <= 1.0e-12:
        vertical = np.array([0.0, 1.0, 0.0])
        tangent_vertical = vertical - float(vertical @ normal) * normal
    tangent_vertical /= np.linalg.norm(tangent_vertical)
    tangent_horizontal = np.cross(tangent_vertical, normal)
    rotation = np.vstack([tangent_horizontal, tangent_vertical, normal])
    return rotation


def solve_ue_box_bs_maximin_rotation(
    ris_center: np.ndarray,
    bs_position: np.ndarray,
    ue_bounds: np.ndarray,
    *,
    initial_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve the deterministic BS/UE-box spherical-cap maximin orientation."""
    from scipy.optimize import minimize

    center = np.asarray(ris_center, dtype=float).reshape(3)
    directions = np.vstack(
        [
            np.asarray(bs_position, dtype=float).reshape(3) - center,
            ue_box_corners(ue_bounds) - center[None, :],
        ]
    )
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("BS and UE-box corners must not coincide with a RIS center")
    directions = directions / norms[:, None]
    if initial_normal is None:
        normal0 = np.sum(directions, axis=0)
    else:
        normal0 = np.asarray(initial_normal, dtype=float).reshape(3)
    normal0 /= np.linalg.norm(normal0)
    x0 = np.concatenate([normal0, [float(np.min(directions @ normal0))]])
    constraints = (
        {
            "type": "eq",
            "fun": lambda value: float(value[:3] @ value[:3] - 1.0),
        },
        {
            "type": "ineq",
            "fun": lambda value: directions @ value[:3] - value[3],
        },
    )
    result = minimize(
        lambda value: -float(value[3]),
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"maximin RIS orientation solve failed: {result.message}")
    normal = np.asarray(result.x[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    objective = float(np.min(directions @ normal))
    rotation = rotation_from_panel_normal(normal)
    return rotation, {
        "normal_global": normal,
        "minimum_normal_cosine": objective,
        "worst_angle_deg": float(
            np.degrees(np.arccos(np.clip(objective, -1.0, 1.0)))
        ),
        "optimizer": "SLSQP spherical-cap maximin over BS and UE-box extreme rays",
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nit),
    }


def validate_ris_rotations(
    rotations: np.ndarray,
    expected_count: int | None = None,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Validate and return calibrated right-handed global-to-RIS rotations."""
    array = np.asarray(rotations, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError("ris_rotations must have shape (num_ris, 3, 3)")
    if expected_count is not None and array.shape[0] < int(expected_count):
        raise ValueError(
            f"ris_rotations contains {array.shape[0]} panels, expected at least "
            f"{int(expected_count)}"
        )
    if np.any(~np.isfinite(array)):
        raise ValueError("ris_rotations must be finite")
    identity = np.eye(3)
    for panel, rotation in enumerate(array):
        orthogonality_error = float(np.max(np.abs(rotation @ rotation.T - identity)))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > tolerance:
            raise ValueError(
                f"ris_rotations[{panel}] is not orthogonal "
                f"(max error {orthogonality_error:.3e})"
            )
        if abs(determinant - 1.0) > tolerance:
            raise ValueError(
                f"ris_rotations[{panel}] must be right-handed "
                f"(det={determinant:.12g})"
            )
    return array


def _line_intersects_box(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: np.ndarray,
    *,
    positive_only: bool,
) -> bool:
    """Return whether an infinite/positive ray intersects an axis-aligned box."""
    lower = 0.0 if positive_only else -np.inf
    upper = np.inf
    for dim in range(3):
        component = float(direction[dim])
        if abs(component) <= 1.0e-15:
            if not bounds[dim, 0] <= origin[dim] <= bounds[dim, 1]:
                return False
            continue
        crossings = (bounds[dim] - origin[dim]) / component
        dim_lower, dim_upper = np.sort(crossings)
        lower = max(lower, float(dim_lower))
        upper = min(upper, float(dim_upper))
        if upper < lower:
            return False
    return bool(upper >= lower)


def _direction_cosine_extrema_over_box(
    ris_center: np.ndarray,
    normal_global: np.ndarray,
    ue_bounds: np.ndarray,
) -> tuple[float, float]:
    """Deterministically optimize the local-normal direction cosine on a box."""
    from scipy.optimize import minimize

    center = np.asarray(ris_center, dtype=float).reshape(3)
    normal = np.asarray(normal_global, dtype=float).reshape(3)
    bounds = np.asarray(ue_bounds, dtype=float)
    scipy_bounds = [tuple(item) for item in bounds]
    axes = [np.linspace(item[0], item[1], 3) for item in bounds]
    starts = np.asarray(
        [
            [x_coord, y_coord, z_coord]
            for x_coord in axes[0]
            for y_coord in axes[1]
            for z_coord in axes[2]
        ],
        dtype=float,
    )

    def direction_cosine(position: np.ndarray) -> float:
        offset = np.asarray(position, dtype=float) - center
        distance = float(np.linalg.norm(offset))
        if distance <= 0.0:
            raise ValueError("RIS center must lie outside the UE search box")
        return float((normal @ offset) / distance)

    minimum = min(direction_cosine(start) for start in starts)
    maximum = max(direction_cosine(start) for start in starts)
    for sign in (1.0, -1.0):
        for start in starts:
            result = minimize(
                lambda position, sign=sign: sign * direction_cosine(position),
                start,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 200},
            )
            value = direction_cosine(result.x)
            minimum = min(minimum, value)
            maximum = max(maximum, value)
    if _line_intersects_box(center, normal, bounds, positive_only=True):
        maximum = 1.0
    if _line_intersects_box(center, -normal, bounds, positive_only=True):
        minimum = -1.0
    return float(np.clip(minimum, -1.0, 1.0)), float(
        np.clip(maximum, -1.0, 1.0)
    )


def induced_local_geometry_bounds(
    ris_center: np.ndarray,
    rotation_global_to_ris: np.ndarray,
    ue_bounds: np.ndarray,
    *,
    range_guard_m: float = 0.0,
    angle_guard_rad: float = 0.0,
) -> dict:
    """Derive exact/guarded local range and circular-angle bounds from a UE box."""
    center = np.asarray(ris_center, dtype=float).reshape(3)
    rotation = validate_ris_rotations(
        np.asarray(rotation_global_to_ris, dtype=float)[None, :, :],
        expected_count=1,
    )[0]
    bounds = np.asarray(ue_bounds, dtype=float)
    corners = ue_box_corners(bounds)
    range_guard_m = max(float(range_guard_m), 0.0)
    angle_guard_rad = max(float(angle_guard_rad), 0.0)

    nearest_point = np.clip(center, bounds[:, 0], bounds[:, 1])
    range_min_exact = float(np.linalg.norm(center - nearest_point))
    corner_ranges = np.linalg.norm(corners - center[None, :], axis=1)
    range_max_exact = float(np.max(corner_ranges))
    if range_min_exact <= 0.0:
        raise ValueError("RIS center must lie outside the UE search box")

    cosine_min, cosine_max = _direction_cosine_extrema_over_box(
        center, rotation[2], bounds
    )
    elev_min_exact = float(np.arcsin(cosine_min))
    elev_max_exact = float(np.arcsin(cosine_max))

    local_corners = (corners - center[None, :]) @ rotation.T
    if _line_intersects_box(center, rotation[2], bounds, positive_only=False):
        az_min_exact, az_max_exact = -np.pi, np.pi
        azimuth_full_circle = True
    else:
        corner_azimuths = np.mod(
            np.arctan2(local_corners[:, 1], local_corners[:, 0]), 2.0 * np.pi
        )
        ordered = np.sort(corner_azimuths)
        gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * np.pi]))
        gap_index = int(np.argmax(gaps))
        az_min_exact = float(ordered[(gap_index + 1) % ordered.size])
        az_max_exact = float(ordered[gap_index])
        if az_max_exact < az_min_exact:
            az_max_exact += 2.0 * np.pi
        center_angle = 0.5 * (az_min_exact + az_max_exact)
        shift = 2.0 * np.pi * np.floor((center_angle + np.pi) / (2.0 * np.pi))
        az_min_exact -= shift
        az_max_exact -= shift
        azimuth_full_circle = False

    range_bounds_exact = (range_min_exact, range_max_exact)
    range_bounds = (
        max(np.finfo(float).eps, range_min_exact - range_guard_m),
        range_max_exact + range_guard_m,
    )
    elev_bounds_exact = (elev_min_exact, elev_max_exact)
    elev_bounds = (
        max(-0.5 * np.pi, elev_min_exact - angle_guard_rad),
        min(0.5 * np.pi, elev_max_exact + angle_guard_rad),
    )
    az_bounds_exact = (az_min_exact, az_max_exact)
    if azimuth_full_circle or (
        az_max_exact - az_min_exact + 2.0 * angle_guard_rad >= 2.0 * np.pi
    ):
        az_bounds = (-np.pi, np.pi)
        azimuth_full_circle = True
    else:
        az_bounds = (
            az_min_exact - angle_guard_rad,
            az_max_exact + angle_guard_rad,
        )
    return {
        "range_bounds_exact": range_bounds_exact,
        "range_bounds": range_bounds,
        "elev_bounds_exact": elev_bounds_exact,
        "elev_bounds": elev_bounds,
        "az_bounds_exact": az_bounds_exact,
        "az_bounds": az_bounds,
        "azimuth_full_circle": bool(azimuth_full_circle),
        "range_guard_m": range_guard_m,
        "angle_guard_rad": angle_guard_rad,
    }


def unit_vector_from_elev_az(elevation: float, azimuth: float) -> np.ndarray:
    """Convert elevation/azimuth angles to a 3-D unit vector."""
    ce = np.cos(elevation)
    return np.array([ce * np.cos(azimuth), ce * np.sin(azimuth), np.sin(elevation)])


def elev_az_from_unit_vector(unit_vector: np.ndarray) -> tuple[float, float]:
    """Convert a 3-D unit vector to elevation/azimuth angles."""
    u = np.asarray(unit_vector, dtype=float)
    norm_u = np.linalg.norm(u)
    if norm_u <= 0.0:
        raise ValueError("unit_vector has zero norm")
    u = u / norm_u
    elevation = np.arcsin(np.clip(u[2], -1.0, 1.0))
    azimuth = np.arctan2(u[1], u[0])
    return float(elevation), float(azimuth)


def make_ris_grid(mx: int, my: int, dx: float, dy: float) -> np.ndarray:
    """Return RIS element coordinates, shape (M_R, 3), centered at the panel."""
    assert mx >= 2 and my >= 2, "RIS grid must have at least 2 elements per axis"
    x_axis = (np.arange(mx) - (mx - 1) / 2.0) * dx
    y_axis = (np.arange(my) - (my - 1) / 2.0) * dy
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    coords = np.zeros((mx * my, 3), dtype=float)
    coords[:, 0] = grid_x.reshape(-1)
    coords[:, 1] = grid_y.reshape(-1)
    return coords


def local_geometry_from_position(
    p_u: np.ndarray, ris_center: np.ndarray, rotation_global_to_ris: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Return range, elevation, azimuth, and local unit direction for one RIS."""
    q_local = rotation_global_to_ris @ (p_u - ris_center)
    range_m = float(np.linalg.norm(q_local))
    if range_m <= 0.0:
        raise ValueError("UE and RIS center coincide")
    unit_local = q_local / range_m
    elevation, azimuth = elev_az_from_unit_vector(unit_local)
    return range_m, elevation, azimuth, unit_local


def position_from_local_geometry(
    ris_center: np.ndarray,
    rotation_global_to_ris: np.ndarray,
    range_m: float,
    elevation: float,
    azimuth: float,
) -> np.ndarray:
    """Map RIS-local spherical geometry back to a global UE position."""
    unit_local = unit_vector_from_elev_az(elevation, azimuth)
    return ris_center + rotation_global_to_ris.T @ (range_m * unit_local)


def near_field_spherical_response(
    range_m: float,
    elevation: float,
    azimuth: float,
    ris_grid: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Exact phase-dominant UE-RIS spherical response, shape (M_R,)."""
    assert ris_grid.ndim == 2 and ris_grid.shape[1] == 3, "ris_grid must be M_R x 3"
    unit_local = unit_vector_from_elev_az(elevation, azimuth)
    q_local = range_m * unit_local
    distance_offsets = np.linalg.norm(q_local[None, :] - ris_grid, axis=1) - range_m
    wavenumber = 2.0 * np.pi / wavelength
    return np.exp(-1j * wavenumber * distance_offsets)


def far_field_ris_response(
    ris_center: np.ndarray,
    target_position: np.ndarray,
    rotation_global_to_ris: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Known RIS-BS far-field element response, shape (M_R,)."""
    direction_local = rotation_global_to_ris @ (target_position - ris_center)
    direction_local = direction_local / np.linalg.norm(direction_local)
    wavenumber = 2.0 * np.pi / wavelength
    return np.exp(-1j * wavenumber * (ris_grid @ direction_local))


def ula_steering(
    num_sensors: int,
    spacing: float,
    wavelength: float,
    arrival_direction_global: np.ndarray,
) -> np.ndarray:
    """ULA steering vector for a BS array aligned with the global x-axis."""
    assert num_sensors >= 1, "num_sensors must be positive"
    direction = np.asarray(arrival_direction_global, dtype=float)
    direction = direction / np.linalg.norm(direction)
    x_positions = (np.arange(num_sensors) - (num_sensors - 1) / 2.0) * spacing
    wavenumber = 2.0 * np.pi / wavelength
    return np.exp(-1j * wavenumber * x_positions * direction[0])


def maxwell_matrix(propagation_direction_global: np.ndarray) -> np.ndarray:
    """Return a simple Maxwell-consistent EVS matrix, shape (6, 2).

    The two columns are transverse electric-field bases stacked with their
    corresponding magnetic-field directions.
    """
    u = np.asarray(propagation_direction_global, dtype=float)
    u = u / np.linalg.norm(u)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(u, reference)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 = e2 / np.linalg.norm(e2)
    h1 = np.cross(u, e1)
    h2 = np.cross(u, e2)
    theta = np.column_stack([np.concatenate([e1, h1]), np.concatenate([e2, h2])])
    assert theta.shape == (6, 2), "Maxwell matrix must be 6 x 2"
    return theta.astype(complex)


def polarization_vector(gamma: float, eta: float) -> np.ndarray:
    """Two-component polarization vector with the paper's phase convention."""
    return np.array([np.sin(gamma) * np.exp(1j * eta), np.cos(gamma)], dtype=complex)
