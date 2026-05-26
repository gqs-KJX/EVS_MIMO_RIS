"""Geometry and array-response helpers for the RIS-EVS-OFDM model."""

from __future__ import annotations

import numpy as np


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
    coords = []
    for ix in range(mx):
        for iy in range(my):
            x = (ix - (mx - 1) / 2.0) * dx
            y = (iy - (my - 1) / 2.0) * dy
            coords.append([x, y, 0.0])
    return np.asarray(coords, dtype=float)


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

