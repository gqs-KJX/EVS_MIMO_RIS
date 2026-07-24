"""Exact factorized scoring for Kronecker-structured baseline atoms.

The raw baseline columns have the form ``a \u2297 d \u2297 c``.  This module
computes their correlations and Gram matrices from the three small factors,
without materializing a full ``I*N*T`` column for every grid candidate.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .backend import BackendConfig, get_backend
from .common import (
    build_jones_basis_evs_atoms,
    delay_response,
    training_response_from_direction,
    training_response_from_position,
)
from ..geometry import local_geometry_from_position


def _normalized(value: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(array))
    valid = bool(np.isfinite(norm) and norm > 0.0)
    return (array / norm if valid else array), valid


def _training_key(group: dict[str, Any]) -> tuple[Any, ...]:
    panel = int(group.get("panel", 0))
    if "position" in group:
        return (
            "position",
            panel,
            bool(group.get("near_field", True)),
            tuple(np.asarray(group["position"], dtype=float).reshape(-1)),
        )
    if "direction" in group:
        return (
            "direction",
            panel,
            tuple(np.asarray(group["direction"], dtype=float).reshape(-1)),
        )
    return ("ones", panel)


def factorized_fit_supports(
    scene: dict,
    config: dict,
    supports: list[dict[str, Any]],
    y_vec: np.ndarray,
    *,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalized Kronecker columns and materialize only the fitted tensor."""
    y = np.asarray(y_vec, dtype=complex).reshape(-1)
    shape = (int(scene["I"]), int(scene["N"]), int(scene["T"]))
    if not supports:
        y_hat = np.zeros_like(y)
        return np.zeros(0, dtype=complex), y_hat, y.copy()
    q = len(supports)
    evs = np.empty((shape[0], q), dtype=complex)
    delays = np.empty((shape[1], q), dtype=complex)
    trainings = np.empty((shape[2], q), dtype=complex)
    for column, support in enumerate(supports):
        panel = int(support.get("panel", 0))
        pol_index = int(support.get("pol_index", 0))
        evs_atoms, _ = build_jones_basis_evs_atoms(
            scene, config, panel_index=panel
        )
        evs[:, column], evs_valid = _normalized(
            evs_atoms[min(pol_index, len(evs_atoms) - 1)]
        )
        delays[:, column], delay_valid = _normalized(
            delay_response(scene, float(support.get("tau", 0.0)))
        )
        if "position" in support:
            response = training_response_from_position(
                scene,
                panel,
                np.asarray(support["position"], dtype=float),
                near_field=bool(support.get("near_field", True)),
            )
        elif "direction" in support:
            response = training_response_from_direction(
                scene,
                panel,
                np.asarray(support["direction"], dtype=float),
            )
        else:
            response = np.ones(shape[2], dtype=complex)
        trainings[:, column], training_valid = _normalized(response)
        if not (evs_valid and delay_valid and training_valid):
            raise ValueError("invalid Kronecker factor in selected support")

    gram = (
        (evs.conj().T @ evs)
        * (delays.conj().T @ delays)
        * (trainings.conj().T @ trainings)
    )
    y_tensor = y.reshape(shape)
    evs_projection = np.einsum(
        "iq,int->qnt", evs.conj(), y_tensor, optimize=True
    )
    rhs = np.einsum(
        "nq,tq,qnt->q",
        delays.conj(),
        trainings.conj(),
        evs_projection,
        optimize=True,
    )
    regularized_gram = gram + float(ridge) * np.eye(q, dtype=complex)
    try:
        coeffs = np.linalg.solve(regularized_gram, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(regularized_gram) @ rhs
    y_hat_tensor = np.einsum(
        "q,iq,nq,tq->int",
        coeffs,
        evs,
        delays,
        trainings,
        optimize=True,
    )
    y_hat = y_hat_tensor.reshape(-1)
    return coeffs, y_hat, y - y_hat


def build_group_factor_context(
    scene: dict,
    config: dict,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build noise-independent factor banks for two-column Jones groups."""
    k_paths = int(scene["K"])
    evs_bank = np.empty((k_paths, int(scene["I"]), 2), dtype=complex)
    valid_panel = np.ones(k_paths, dtype=bool)
    for panel in range(k_paths):
        evs_atoms, _ = build_jones_basis_evs_atoms(
            scene, config, panel_index=panel
        )
        for pol_index in range(2):
            atom, valid = _normalized(evs_atoms[pol_index])
            evs_bank[panel, :, pol_index] = atom
            valid_panel[panel] &= valid

    delay_lookup: dict[float, int] = {}
    delay_values: list[np.ndarray] = []
    delay_valid: list[bool] = []
    training_lookup: dict[tuple[Any, ...], int] = {}
    training_values: list[np.ndarray] = []
    training_valid: list[bool] = []
    group_panels = np.empty(len(groups), dtype=np.int32)
    group_delays = np.empty(len(groups), dtype=np.int32)
    group_trainings = np.empty(len(groups), dtype=np.int32)

    for group_index, group in enumerate(groups):
        panel = int(group.get("panel", 0))
        group_panels[group_index] = panel

        tau = float(group.get("tau", 0.0))
        delay_index = delay_lookup.get(tau)
        if delay_index is None:
            delay_index = len(delay_values)
            delay_lookup[tau] = delay_index
            factor, valid = _normalized(delay_response(scene, tau))
            delay_values.append(factor)
            delay_valid.append(valid)
        group_delays[group_index] = delay_index

        key = _training_key(group)
        training_index = training_lookup.get(key)
        if training_index is None:
            training_index = len(training_values)
            training_lookup[key] = training_index
            if "position" in group:
                response = training_response_from_position(
                    scene,
                    panel,
                    np.asarray(group["position"], dtype=float),
                    near_field=bool(group.get("near_field", True)),
                )
            elif "direction" in group:
                response = training_response_from_direction(
                    scene,
                    panel,
                    np.asarray(group["direction"], dtype=float),
                )
            else:
                response = np.ones(int(scene["T"]), dtype=complex)
            factor, valid = _normalized(response)
            training_values.append(factor)
            training_valid.append(valid)
        group_trainings[group_index] = training_index

    delays = (
        np.stack(delay_values, axis=0)
        if delay_values
        else np.empty((0, int(scene["N"])), dtype=complex)
    )
    trainings = (
        np.stack(training_values, axis=0)
        if training_values
        else np.empty((0, int(scene["T"])), dtype=complex)
    )
    valid = (
        valid_panel[group_panels]
        & np.asarray(delay_valid, dtype=bool)[group_delays]
        & np.asarray(training_valid, dtype=bool)[group_trainings]
    )

    panel_training_indices: list[np.ndarray] = []
    group_training_local = np.empty(len(groups), dtype=np.int32)
    for panel in range(k_paths):
        group_indices = np.flatnonzero(group_panels == panel)
        unique_training = np.unique(group_trainings[group_indices])
        panel_training_indices.append(unique_training.astype(np.int32, copy=False))
        local_lookup = {
            int(global_index): local_index
            for local_index, global_index in enumerate(unique_training)
        }
        group_training_local[group_indices] = np.asarray(
            [local_lookup[int(index)] for index in group_trainings[group_indices]],
            dtype=np.int32,
        )

    rho = np.einsum(
        "ki,ki->k",
        evs_bank[:, :, 0].conj(),
        evs_bank[:, :, 1],
        optimize=True,
    )
    return {
        "evs_bank": evs_bank,
        "delay_bank": delays,
        "training_bank": trainings,
        "group_panels": group_panels,
        "group_delays": group_delays,
        "group_training_local": group_training_local,
        "panel_training_indices": panel_training_indices,
        "valid_groups": valid,
        "rho": rho,
        "shape": (int(scene["I"]), int(scene["N"]), int(scene["T"])),
    }


class FactorizedGroupScorer:
    """Score all two-column groups with separable contractions."""

    def __init__(
        self,
        context: dict[str, Any],
        backend_config: BackendConfig | dict[str, Any] | None = None,
    ) -> None:
        self.backend = get_backend(backend_config)
        self.xp = self.backend.xp
        xp = self.xp
        self.shape = tuple(int(value) for value in context["shape"])
        self.evs = self.backend.asarray(context["evs_bank"], dtype=xp.complex128)
        self.delays = self.backend.asarray(context["delay_bank"], dtype=xp.complex128)
        self.trainings = self.backend.asarray(context["training_bank"], dtype=xp.complex128)
        self.panels = self.backend.asarray(context["group_panels"], dtype=xp.int32)
        self.delay_indices = self.backend.asarray(context["group_delays"], dtype=xp.int32)
        self.training_local = self.backend.asarray(
            context["group_training_local"], dtype=xp.int32
        )
        self.panel_training_indices = [
            self.backend.asarray(indices, dtype=xp.int32)
            for indices in context["panel_training_indices"]
        ]
        self.valid = self.backend.asarray(context["valid_groups"], dtype=xp.bool_)
        self.rho = self.backend.asarray(context["rho"], dtype=xp.complex128)
        self.excluded = xp.zeros(self.panels.shape, dtype=xp.bool_)

    def exclude(self, indices: Iterable[int]) -> None:
        host_indices = np.asarray(list(indices), dtype=np.int64)
        if host_indices.size:
            self.excluded[self.backend.asarray(host_indices)] = True

    def scores(self, residual: np.ndarray, *, rank_tol: float = 1.0e-10) -> Any:
        xp = self.xp
        residual_device = self.backend.asarray(
            np.asarray(residual, dtype=complex).reshape(self.shape),
            dtype=xp.complex128,
        )
        projected = xp.einsum(
            "kip,int->kpnt", self.evs.conj(), residual_device, optimize=True
        )
        b = xp.zeros((self.panels.size, 2), dtype=xp.complex128)
        for panel in range(int(self.evs.shape[0])):
            group_indices = xp.flatnonzero(self.panels == panel)
            if int(group_indices.size) == 0:
                continue
            training_indices = self.panel_training_indices[panel]
            panel_trainings = self.trainings[training_indices]
            delay_contraction = xp.einsum(
                "dn,pnt->pdt",
                self.delays.conj(),
                projected[panel],
                optimize=True,
            )
            correlations = xp.einsum(
                "pdt,ct->pdc",
                delay_contraction,
                panel_trainings.conj(),
                optimize=True,
            )
            delay_index = self.delay_indices[group_indices]
            training_index = self.training_local[group_indices]
            b[group_indices] = correlations[:, delay_index, training_index].T

        group_rho = self.rho[self.panels]
        second_norm_sq = xp.maximum(1.0 - xp.abs(group_rho) ** 2, 0.0)
        scores = xp.abs(b[:, 0]) ** 2
        independent = xp.sqrt(second_norm_sq) > float(rank_tol)
        second_projection = b[:, 1] - group_rho.conj() * b[:, 0]
        scores = scores + xp.where(
            independent,
            xp.abs(second_projection) ** 2
            / xp.where(second_norm_sq > 0.0, second_norm_sq, 1.0),
            0.0,
        )
        scores = xp.where(self.valid & ~self.excluded, scores.real, -xp.inf)
        return scores

    def best(self, residual: np.ndarray) -> tuple[int, float]:
        scores = self.scores(residual)
        best_index = int(self.backend.to_host(self.xp.argmax(scores)))
        best_score = float(self.backend.to_host(scores[best_index]))
        return best_index, best_score


class FactorizedPositionClockScorer:
    """Batched exact small-Gram scorer for joint position/clock candidates."""

    def __init__(
        self,
        scene: dict,
        config: dict,
        y_vec: np.ndarray,
        backend_config: BackendConfig | dict[str, Any] | None = None,
        *,
        ridge: float = 1.0e-10,
    ) -> None:
        self.scene = scene
        self.config = config
        self.backend = get_backend(backend_config)
        self.xp = self.backend.xp
        self.ridge = float(ridge)
        self.k_paths = int(scene["K"])
        self.q = 2 * self.k_paths
        self.panel_for_column = np.repeat(np.arange(self.k_paths), 2)
        self._training_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

        evs = np.empty((self.k_paths, 2, int(scene["I"])), dtype=complex)
        for panel in range(self.k_paths):
            atoms, _ = build_jones_basis_evs_atoms(scene, config, panel_index=panel)
            for pol_index in range(2):
                evs[panel, pol_index], valid = _normalized(atoms[pol_index])
                if not valid:
                    raise ValueError("invalid EVS Jones factor in factorized scorer")
        evs_flat = evs.reshape(self.q, int(scene["I"]))
        y_tensor = np.asarray(y_vec, dtype=complex).reshape(
            int(scene["I"]), int(scene["N"]), int(scene["T"])
        )
        projected = np.einsum(
            "qi,int->qnt", evs_flat.conj(), y_tensor, optimize=True
        ).reshape(self.k_paths, 2, int(scene["N"]), int(scene["T"]))
        evs_gram = evs_flat.conj() @ evs_flat.T
        xp = self.xp
        self.projected = self.backend.asarray(projected, dtype=xp.complex128)
        self.evs_gram = self.backend.asarray(evs_gram, dtype=xp.complex128)
        self.panel_for_column_device = self.backend.asarray(
            self.panel_for_column, dtype=xp.int32
        )
        self.y_norm_sq = float(np.linalg.norm(y_tensor.reshape(-1)) ** 2)
        self.y_energy = self.y_norm_sq + 1.0e-12

    def _training_for_position(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(position, dtype=float).reshape(3)
        key = tuple(float(value) for value in position)
        cached = self._training_cache.get(key)
        if cached is not None:
            return cached
        responses = np.empty(
            (self.k_paths, int(self.scene["T"])), dtype=complex
        )
        ranges = np.empty(self.k_paths, dtype=float)
        for panel in range(self.k_paths):
            ranges[panel], _, _, _ = local_geometry_from_position(
                position,
                np.asarray(self.scene["ris_centers"][panel], dtype=float),
                np.asarray(self.scene["rotations"][panel], dtype=float),
            )
            response = training_response_from_position(
                self.scene, panel, position, near_field=True
            )
            responses[panel], valid = _normalized(response)
            if not valid:
                raise ValueError("invalid RIS training factor in factorized scorer")
        cached = (responses, ranges)
        self._training_cache[key] = cached
        return cached

    def score_candidates(
        self,
        positions: np.ndarray,
        clocks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        clocks = np.asarray(clocks, dtype=float).reshape(-1)
        if positions.shape[0] != clocks.size:
            raise ValueError("positions and clocks must have the same batch size")
        batch_size = clocks.size
        if batch_size == 0:
            return (
                np.empty(0, dtype=float),
                np.empty((0, self.q), dtype=complex),
                np.empty(0, dtype=float),
            )
        trainings = np.empty(
            (batch_size, self.k_paths, int(self.scene["T"])), dtype=complex
        )
        ranges = np.empty((batch_size, self.k_paths), dtype=float)
        for index, position in enumerate(positions):
            trainings[index], ranges[index] = self._training_for_position(position)

        taus = (
            ranges + np.asarray(self.scene["d_RB"], dtype=float)[None, :]
        ) / float(self.scene["c0"]) + clocks[:, None]
        n_index = np.asarray(
            self.scene.get(
                "subcarrier_indices", np.arange(int(self.scene["N"]))
            ),
            dtype=float,
        )
        delays = np.exp(
            -1j
            * 2.0
            * np.pi
            * float(self.scene["delta_f"])
            * taus[:, :, None]
            * n_index[None, None, :]
        )
        delays /= np.linalg.norm(delays, axis=2, keepdims=True)

        xp = self.xp
        delays_device = self.backend.asarray(delays, dtype=xp.complex128)
        trainings_device = self.backend.asarray(trainings, dtype=xp.complex128)
        rhs_panel = xp.einsum(
            "bkn,bkt,kpnt->bkp",
            delays_device.conj(),
            trainings_device.conj(),
            self.projected,
            optimize=True,
        )
        rhs = rhs_panel.reshape(batch_size, self.q)
        delay_gram = xp.einsum(
            "bkn,bln->bkl", delays_device.conj(), delays_device, optimize=True
        )
        training_gram = xp.einsum(
            "bkt,blt->bkl",
            trainings_device.conj(),
            trainings_device,
            optimize=True,
        )
        panels = self.panel_for_column_device
        gram = (
            self.evs_gram[None, :, :]
            * delay_gram[:, panels[:, None], panels[None, :]]
            * training_gram[:, panels[:, None], panels[None, :]]
        )
        eye = xp.eye(self.q, dtype=xp.complex128)
        try:
            coeffs = xp.linalg.solve(gram + self.ridge * eye[None, :, :], rhs)
        except Exception:
            coeffs = xp.einsum(
                "bij,bj->bi",
                xp.linalg.pinv(gram + self.ridge * eye[None, :, :]),
                rhs,
                optimize=True,
            )
        fit_energy = xp.real(
            xp.einsum(
                "bi,bij,bj->b", coeffs.conj(), gram, coeffs, optimize=True
            )
        )
        cross = xp.real(xp.einsum("bi,bi->b", coeffs.conj(), rhs, optimize=True))
        residual_sq = xp.maximum(self.y_norm_sq - 2.0 * cross + fit_energy, 0.0)
        scores = fit_energy / self.y_energy
        return (
            np.asarray(self.backend.to_host(scores), dtype=float),
            np.asarray(self.backend.to_host(coeffs), dtype=complex),
            np.asarray(self.backend.to_host(residual_sq), dtype=float),
        )
