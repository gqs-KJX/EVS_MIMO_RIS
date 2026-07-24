"""Experimental common-clock orbit-profiled Jones variable projection.

This module is intentionally separate from the frozen global-VP path.  It
reuses the exact matrix-free sufficient statistics from :mod:`src.global_vp`,
but profiles the common OFDM clock on its one-dimensional unitary orbit before
optimizing the three UE-position coordinates.

The strict-equivalence experiment currently requires zero Jones diagonal
loading.  The frozen VP code uses that loading in the normal-equation solve but
does not include it in the reported objective.  Setting it to zero makes the
linear solve and the stated regularized objective exactly the same problem;
production integration should either retain this convention or explicitly add
the loading penalty to both old and new objectives.
"""

from __future__ import annotations

import copy
import heapq
import math
from typing import Any

import numpy as np

from .global_vp import (
    _build_vp_matrix_free_cache,
    _dynamic_vp_factors,
    _global_vp_config,
    _global_vp_mode,
    _initial_xi_from_stage1,
    _jones_regularizer_from_gram,
    _raw_residual_from_stats,
    _solve_linear_vp_regularized_from_stats,
    build_vp_sufficient_statistics_matrix_free,
    extract_stage1_jones_directions,
)
from .utils import scipy_is_available


def _relative_error(left: float, right: float) -> float:
    return float(abs(left - right) / max(abs(left), abs(right), 1.0e-300))


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return 0.5 * (arr + arr.conj().T)


def common_clock_frequency_phase(scene: dict, delta_t_s: float) -> np.ndarray:
    """Return the unit-modulus OFDM phase defining the common-clock orbit."""
    n_index = np.asarray(
        scene.get("subcarrier_indices", np.arange(int(scene["N"]))),
        dtype=float,
    )
    return np.exp(
        -1j
        * 2.0
        * np.pi
        * float(scene["delta_f"])
        * float(delta_t_s)
        * n_index
    )


def apply_common_clock_unitary(
    values: np.ndarray,
    scene: dict,
    delta_t_s: float,
    *,
    adjoint: bool = False,
) -> np.ndarray:
    """Apply ``U(delta_t)`` to vectorized raw tensors or dictionaries.

    ``values`` may have shape ``(I*N*T,)`` or ``(I*N*T, q)``.  The returned
    array has the same shape and uses the repository's exact ``I,N,T``
    vectorization order.
    """
    array = np.asarray(values, dtype=complex)
    rows = int(scene["I"]) * int(scene["N"]) * int(scene["T"])
    if array.ndim not in (1, 2) or array.shape[0] != rows:
        raise ValueError(f"values must have {rows} rows")
    phase = common_clock_frequency_phase(scene, delta_t_s)
    if adjoint:
        phase = phase.conj()
    if array.ndim == 1:
        tensor = array.reshape(int(scene["I"]), int(scene["N"]), int(scene["T"]))
        return (tensor * phase[None, :, None]).reshape(array.shape)
    tensor = array.reshape(
        int(scene["I"]), int(scene["N"]), int(scene["T"]), array.shape[1]
    )
    return (tensor * phase[None, :, None, None]).reshape(array.shape)


class CommonClockJonesProfiler:
    """Profile the common clock for the repository's exact Jones-VP objective."""

    def __init__(
        self,
        y_raw: np.ndarray,
        init_estimate: dict,
        scene: dict,
        config: dict,
    ) -> None:
        expected = (int(scene["I"]), int(scene["N"]), int(scene["T"]))
        if tuple(np.asarray(y_raw).shape) != expected:
            raise ValueError(f"y_raw must have shape {expected}")
        self.y_raw = np.asarray(y_raw, dtype=complex)
        self.y_vec = self.y_raw.reshape(-1)
        self.init_estimate = init_estimate
        self.scene = scene
        self.config = copy.deepcopy(config)
        self.options = _global_vp_config(self.config)
        self.ccop_options = {
            "clock_fft_size": 4096,
            "clock_abs_tol_objective": 1.0e-12,
            "clock_rel_tol": 1.0e-10,
            "clock_max_intervals": 20000,
            "clock_newton_max_iter": 12,
            "clock_peak_separation_bins": 3,
            "clock_branch_switch_abs_gap_objective": 1.0e-10,
            "clock_branch_switch_rel_gap": 1.0e-8,
            "outer_max_iter": 20,
            "outer_ftol": 1.0e-12,
            "outer_gtol": 1.0e-8,
            "outer_safeguard_max_iter": 40,
        }
        self.ccop_options.update(dict(self.config.get("ccop_jvp", {})))
        self._validate_assumptions()
        self.cache = _build_vp_matrix_free_cache(
            self.y_vec, self.init_estimate, self.scene, self.config
        )
        self.omega0 = 2.0 * np.pi * float(self.scene["delta_f"])
        clock_bounds = np.asarray(self.config["delta_t_bounds"], dtype=float)
        self.clock_bounds_s = (float(clock_bounds[0]), float(clock_bounds[1]))
        self.theta_bounds = (
            self.omega0 * self.clock_bounds_s[0],
            self.omega0 * self.clock_bounds_s[1],
        )
        self._last_position: np.ndarray | None = None
        self._last_orbit: dict | None = None
        self.position_evaluations = 0
        self.clock_interval_evaluations = 0
        self.clock_profile_evaluations = 0
        self.clock_profile_certified_count = 0
        self.clock_profile_max_certificate_gap_objective = 0.0
        self.clock_profile_max_certificate_gap_ratio = 0.0

    def _validate_assumptions(self) -> None:
        mode = _global_vp_mode(self.config)
        if mode not in {
            "adaptive_jones",
            "jones_regularized",
            "jones_free",
            "fixed_pol",
        }:
            raise ValueError("CCOP profiling requires a supported linear VP mode")
        if bool(self.options.get("use_weight", False)):
            raise ValueError(
                "strict CCOP-JVP experiment currently requires an unweighted raw objective"
            )
        if bool(self.options.get("use_delay_prior", False)):
            raise ValueError(
                "strict common-clock orbit profiling excludes the Stage-I delay prior"
            )
        loading = float(self.options.get("jones_diagonal_loading", 1.0e-10))
        if mode != "fixed_pol" and loading != 0.0:
            raise ValueError(
                "strict equality with the frozen Jones-VP objective requires "
                "global_vp.jones_diagonal_loading=0"
            )
        if mode == "fixed_pol" and float(self.options.get("beta_reg", 0.0)) != 0.0:
            raise ValueError("strict fixed-pol CCOP equality requires global_vp.beta_reg=0")
        bounds = np.asarray(self.config["delta_t_bounds"], dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
            raise ValueError("delta_t_bounds must be a finite increasing pair")
        ambiguity_period = 1.0 / float(self.scene["delta_f"])
        if float(bounds[1] - bounds[0]) >= ambiguity_period:
            raise ValueError(
                "delta_t_bounds must be strictly shorter than one OFDM clock period"
            )

    def _clock_matched_coefficients(self, dynamic: dict) -> np.ndarray:
        """Return u[n] such that b(clock)=sum_n u[n] exp(+j*n*theta)."""
        n_dim = int(self.scene["N"])
        num_atoms = int(self.cache["num_atoms"])
        u_coeff = np.zeros((n_dim, num_atoms), dtype=complex)
        d_mat = np.asarray(dynamic["D"], dtype=complex)
        c_mat = np.asarray(dynamic["C"], dtype=complex)
        for path, column_slice in enumerate(self.cache["column_slices"]):
            # matched[b,n] = sum_t conj(c[t]) y_evs[b,n,t]
            matched = np.einsum(
                "t,bnt->bn",
                c_mat[:, path].conj(),
                self.cache["y_evs"][path],
                optimize=True,
            )
            u_coeff[:, column_slice] = (
                matched * d_mat[:, path].conj()[None, :]
            ).T
        return u_coeff

    def _position_orbit(self, p_u: np.ndarray) -> dict:
        position = np.asarray(p_u, dtype=float).reshape(3)
        if self._last_position is not None and np.array_equal(position, self._last_position):
            assert self._last_orbit is not None
            return self._last_orbit

        xi_zero_clock = np.r_[position, 0.0]
        stats = build_vp_sufficient_statistics_matrix_free(
            xi_zero_clock,
            self.y_vec,
            self.init_estimate,
            self.scene,
            self.config,
            cache=self.cache,
        )
        gram = _hermitian(stats["G"])
        mode = _global_vp_mode(self.config)
        if mode == "fixed_pol":
            regularizer = np.zeros_like(gram)
            rho = np.zeros(int(self.scene["K"]), dtype=float)
            lambda_path = np.zeros(int(self.scene["K"]), dtype=float)
            status = ["fixed_pol"] * int(self.scene["K"])
            loading = 0.0
        else:
            regularizer, rho, lambda_path, status, loading = (
                _jones_regularizer_from_gram(
                    self.init_estimate, self.scene, self.config, gram
                )
            )
        if loading != 0.0:
            raise RuntimeError("CCOP strict-equivalence loading changed after validation")
        normal = _hermitian(gram + regularizer)
        identity = np.eye(normal.shape[0], dtype=complex)
        inverse_backend = "numpy.linalg.solve"
        try:
            normal_inverse = np.linalg.solve(normal, identity)
        except np.linalg.LinAlgError:
            normal_inverse = np.linalg.pinv(normal)
            inverse_backend = "numpy.linalg.pinv"
        normal_inverse = _hermitian(normal_inverse)

        u_coeff = self._clock_matched_coefficients(stats["aux"])
        b_zero_from_orbit = np.sum(u_coeff, axis=0)
        b_zero_reference = np.asarray(stats["b"], dtype=complex)
        b_zero_rel_error = float(
            np.linalg.norm(b_zero_from_orbit - b_zero_reference)
            / max(np.linalg.norm(b_zero_reference), 1.0e-300)
        )

        n_dim = int(self.scene["N"])
        trig_coeff = np.empty(n_dim, dtype=complex)
        for lag in range(n_dim):
            value = 0.0j
            for left in range(n_dim - lag):
                value += np.vdot(
                    u_coeff[left], normal_inverse @ u_coeff[left + lag]
                )
            trig_coeff[lag] = value

        eigvals, eigvecs = np.linalg.eigh(normal_inverse)
        eig_scale = max(float(np.max(np.abs(eigvals))), 1.0)
        if float(np.min(eigvals)) < -1.0e-10 * eig_scale:
            raise np.linalg.LinAlgError("Jones normal inverse is not positive semidefinite")
        eigvals = np.maximum(eigvals, 0.0)
        whitened_u = (
            np.sqrt(eigvals)[:, None]
            * (eigvecs.conj().T @ u_coeff.T)
        )

        orbit = {
            "p_u": position.copy(),
            "G": gram,
            "regularizer": regularizer,
            "normal": normal,
            "normal_inverse": normal_inverse,
            "inverse_backend": inverse_backend,
            "u": u_coeff,
            "whitened_u": whitened_u,
            "trig_coeff": trig_coeff,
            "y_norm": float(stats["y_norm"]),
            "y_size": int(stats["y_size"]),
            "jones_rho": np.asarray(rho, dtype=float),
            "lambda_jones_per_path": np.asarray(lambda_path, dtype=float),
            "jones_prior_status": list(status),
            "b_zero_relative_error": b_zero_rel_error,
            "G_zero_clock": gram.copy(),
        }
        self.position_evaluations += 1
        self._last_position = position.copy()
        self._last_orbit = orbit
        return orbit

    @staticmethod
    def _score_and_derivatives(theta: float, trig_coeff: np.ndarray) -> tuple[float, float, float]:
        coeff = np.asarray(trig_coeff, dtype=complex)
        if coeff.size == 1:
            return float(coeff[0].real), 0.0, 0.0
        lag = np.arange(1, coeff.size, dtype=float)
        phase = np.exp(1j * lag * float(theta))
        positive = coeff[1:] * phase
        score = float(coeff[0].real + 2.0 * np.real(np.sum(positive)))
        first = float(2.0 * np.real(np.sum(1j * lag * positive)))
        second = float(2.0 * np.real(np.sum(-(lag**2) * positive)))
        return score, first, second

    def evaluate_clock(
        self,
        p_u: np.ndarray,
        delta_t_s: float,
        *,
        orbit: dict | None = None,
    ) -> dict:
        """Evaluate the orbit-reduced objective at one fixed position and clock."""
        orbit = self._position_orbit(p_u) if orbit is None else orbit
        theta = self.omega0 * float(delta_t_s)
        n_power = np.asarray(
            self.scene.get(
                "subcarrier_indices", np.arange(int(self.scene["N"]))
            ),
            dtype=float,
        )
        phase = np.exp(1j * n_power * theta)
        rhs = phase @ orbit["u"]
        try:
            coeff = np.linalg.solve(orbit["normal"], rhs)
        except np.linalg.LinAlgError:
            coeff = orbit["normal_inverse"] @ rhs
        raw_residual = _raw_residual_from_stats(
            orbit["y_norm"], orbit["G"], rhs, coeff
        )
        jones_unscaled = float(
            np.real(np.vdot(coeff, orbit["regularizer"] @ coeff))
        )
        objective = float(
            (raw_residual + jones_unscaled) / float(orbit["y_size"])
        )
        score, score_first_theta, score_second_theta = self._score_and_derivatives(
            theta, orbit["trig_coeff"]
        )
        objective_from_score = float(
            (orbit["y_norm"] - score) / float(orbit["y_size"])
        )
        return {
            "p_u": np.asarray(p_u, dtype=float).reshape(3).copy(),
            "delta_t": float(delta_t_s),
            "theta": float(theta),
            "b": rhs,
            "x_hat": coeff,
            "raw_residual_unscaled": raw_residual,
            "raw_objective": float(raw_residual / float(orbit["y_size"])),
            "jones_regularizer_objective": float(
                jones_unscaled / float(orbit["y_size"])
            ),
            "total_objective": objective,
            "score": score,
            "score_first_theta": score_first_theta,
            "score_second_theta": score_second_theta,
            "objective_from_trig_score": objective_from_score,
            "objective_trig_abs_error": float(abs(objective - objective_from_score)),
        }

    def _fft_lower_bound_candidates(self, orbit: dict) -> list[tuple[float, float]]:
        requested = int(self.ccop_options["clock_fft_size"])
        fft_size = 1 << max(1, int(math.ceil(math.log2(max(requested, int(self.scene["N"]))))))
        transformed = np.fft.ifft(
            orbit["whitened_u"], n=fft_size, axis=1
        ) * fft_size
        grid_scores = np.sum(np.abs(transformed) ** 2, axis=0).real
        theta_low, theta_high = self.theta_bounds
        step = 2.0 * np.pi / fft_size
        candidates: list[tuple[float, float]] = []
        for index, score in enumerate(grid_scores):
            base_theta = index * step
            shift_min = int(math.ceil((theta_low - base_theta) / (2.0 * np.pi)))
            shift_max = int(math.floor((theta_high - base_theta) / (2.0 * np.pi)))
            for shift in range(shift_min, shift_max + 1):
                theta = base_theta + shift * 2.0 * np.pi
                if theta_low <= theta <= theta_high:
                    candidates.append((float(score), float(theta)))
        for theta in (theta_low, theta_high):
            score = self._score_and_derivatives(theta, orbit["trig_coeff"])[0]
            candidates.append((score, theta))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _safeguarded_newton(
        self,
        theta_start: float,
        orbit: dict,
        half_width: float,
    ) -> tuple[float, float]:
        theta_low = max(self.theta_bounds[0], float(theta_start) - half_width)
        theta_high = min(self.theta_bounds[1], float(theta_start) + half_width)
        theta = float(np.clip(theta_start, theta_low, theta_high))
        score = self._score_and_derivatives(theta, orbit["trig_coeff"])[0]
        for _ in range(int(self.ccop_options["clock_newton_max_iter"])):
            theta_before = theta
            value, first, second = self._score_and_derivatives(
                theta, orbit["trig_coeff"]
            )
            if not np.isfinite(second) or second >= 0.0 or abs(second) <= 1.0e-18:
                break
            step = -first / second
            trial = float(np.clip(theta + step, theta_low, theta_high))
            accepted = False
            for _ in range(20):
                trial_score = self._score_and_derivatives(
                    trial, orbit["trig_coeff"]
                )[0]
                if trial_score >= value:
                    theta = trial
                    score = trial_score
                    accepted = True
                    break
                trial = 0.5 * (trial + theta)
            if not accepted or abs(theta - theta_before) <= 1.0e-14:
                break
        return score, theta

    def profile_clock(
        self,
        p_u: np.ndarray,
        *,
        incumbent_clock_s: float | None = None,
    ) -> dict:
        """Return an epsilon-global common-clock maximum and certificate."""
        orbit = self._position_orbit(p_u)
        trig_coeff = orbit["trig_coeff"]
        fft_candidates = self._fft_lower_bound_candidates(orbit)
        best_score, best_theta = fft_candidates[0]
        incumbent_clock_used = False
        if incumbent_clock_s is not None:
            incumbent_clock = float(incumbent_clock_s)
            if self.clock_bounds_s[0] <= incumbent_clock <= self.clock_bounds_s[1]:
                incumbent_theta = self.omega0 * incumbent_clock
                incumbent_score = self._score_and_derivatives(
                    incumbent_theta, trig_coeff
                )[0]
                if incumbent_score > best_score:
                    best_score, best_theta = incumbent_score, incumbent_theta
                incumbent_clock_used = True

        fft_size = 1 << max(
            1,
            int(
                math.ceil(
                    math.log2(
                        max(
                            int(self.ccop_options["clock_fft_size"]),
                            int(self.scene["N"]),
                        )
                    )
                )
            ),
        )
        grid_step = 2.0 * np.pi / fft_size
        newton_score, newton_theta = self._safeguarded_newton(
            best_theta, orbit, 2.0 * grid_step
        )
        if newton_score > best_score:
            best_score, best_theta = newton_score, newton_theta

        separated: list[tuple[float, float]] = []
        minimum_separation = (
            int(self.ccop_options["clock_peak_separation_bins"]) * grid_step
        )
        for score, theta in fft_candidates:
            if all(abs(theta - kept_theta) >= minimum_separation for _, kept_theta in separated):
                separated.append((score, theta))
            if len(separated) == 2:
                break
        second_fft_score = separated[1][0] if len(separated) > 1 else float("-inf")
        second_fft_theta = separated[1][1] if len(separated) > 1 else float("nan")

        lag = np.arange(1, trig_coeff.size, dtype=float)
        second_derivative_bound = float(
            2.0 * np.sum((lag**2) * np.abs(trig_coeff[1:]))
        )
        theta_low, theta_high = self.theta_bounds
        num_initial = max(1, int(math.ceil((theta_high - theta_low) / grid_step)))
        edges = np.linspace(theta_low, theta_high, num_initial + 1)
        heap: list[tuple[float, int, float, float]] = []
        serial = 0
        max_pruned_upper = float(best_score)

        def interval_upper(left: float, right: float) -> tuple[float, float, float]:
            center = 0.5 * (left + right)
            radius = 0.5 * (right - left)
            center_score, first, _ = self._score_and_derivatives(center, trig_coeff)
            rounding = 64.0 * np.finfo(float).eps * (
                abs(center_score) + abs(first) * radius + second_derivative_bound * radius**2 + 1.0
            )
            upper = (
                center_score
                + abs(first) * radius
                + 0.5 * second_derivative_bound * radius**2
                + rounding
            )
            return float(upper), float(center_score), float(center)

        for left, right in zip(edges[:-1], edges[1:]):
            upper, center_score, center = interval_upper(float(left), float(right))
            self.clock_interval_evaluations += 1
            if center_score > best_score:
                best_score, best_theta = center_score, center
            heapq.heappush(heap, (-upper, serial, float(left), float(right)))
            serial += 1

        abs_tol_objective = float(self.ccop_options["clock_abs_tol_objective"])
        rel_tol = float(self.ccop_options["clock_rel_tol"])
        score_tolerance = (
            float(orbit["y_size"]) * abs_tol_objective
            + rel_tol * max(float(orbit["y_norm"]), abs(best_score), 1.0e-300)
        )
        max_intervals = int(self.ccop_options["clock_max_intervals"])
        split_count = 0
        while heap:
            global_upper = -heap[0][0]
            if global_upper - best_score <= score_tolerance:
                break
            if split_count >= max_intervals:
                break
            neg_upper, _, left, right = heapq.heappop(heap)
            if -neg_upper <= best_score + score_tolerance:
                max_pruned_upper = max(max_pruned_upper, float(-neg_upper))
                continue
            center = 0.5 * (left + right)
            for child_left, child_right in ((left, center), (center, right)):
                upper, center_score, child_center = interval_upper(
                    child_left, child_right
                )
                self.clock_interval_evaluations += 1
                if center_score > best_score:
                    best_score, best_theta = center_score, child_center
                    newton_score, newton_theta = self._safeguarded_newton(
                        child_center,
                        orbit,
                        0.5 * (child_right - child_left),
                    )
                    if newton_score > best_score:
                        best_score, best_theta = newton_score, newton_theta
                if upper > best_score + score_tolerance:
                    heapq.heappush(
                        heap,
                        (-upper, serial, child_left, child_right),
                    )
                    serial += 1
                else:
                    max_pruned_upper = max(max_pruned_upper, float(upper))
            split_count += 1

        global_upper = max(
            best_score,
            max_pruned_upper,
            -heap[0][0] if heap else best_score,
        )
        score_gap = float(max(0.0, global_upper - best_score))
        objective_gap = float(score_gap / float(orbit["y_size"]))
        certified = bool(score_gap <= score_tolerance)
        delta_t = float(best_theta / self.omega0)
        point = self.evaluate_clock(p_u, delta_t, orbit=orbit)
        point.update(
            {
                "clock_certified": certified,
                "clock_certificate_gap_objective": objective_gap,
                "clock_certificate_gap_score": score_gap,
                "clock_certificate_tolerance_score": float(score_tolerance),
                "clock_global_upper_score": float(global_upper),
                "clock_fft_lower_score": float(fft_candidates[0][0]),
                "clock_fft_size": int(fft_size),
                "clock_bnb_splits": int(split_count),
                "clock_bnb_active_intervals": int(len(heap)),
                "clock_second_fft_score": float(second_fft_score),
                "clock_second_fft_delta_t": float(second_fft_theta / self.omega0)
                if np.isfinite(second_fft_theta)
                else float("nan"),
                "clock_fft_peak_gap_objective": float(
                    (best_score - second_fft_score) / float(orbit["y_size"])
                )
                if np.isfinite(second_fft_score)
                else float("inf"),
                "clock_incumbent_used": incumbent_clock_used,
            }
        )
        fft_peak_gap_objective = float(point["clock_fft_peak_gap_objective"])
        branch_gap_tolerance = float(
            self.ccop_options["clock_branch_switch_abs_gap_objective"]
            + self.ccop_options["clock_branch_switch_rel_gap"]
            * max(abs(best_score) / float(orbit["y_size"]), 1.0e-300)
        )
        point["clock_branch_gap_tolerance_objective"] = branch_gap_tolerance
        point["clock_branch_ambiguous"] = bool(
            np.isfinite(fft_peak_gap_objective)
            and fft_peak_gap_objective <= branch_gap_tolerance
        )
        point["gradient_reliable"] = bool(
            point["clock_certified"] and not point["clock_branch_ambiguous"]
        )
        certificate_tolerance_objective = float(
            score_tolerance / float(orbit["y_size"])
        )
        certificate_gap_ratio = float(
            objective_gap / certificate_tolerance_objective
        )
        self.clock_profile_evaluations += 1
        self.clock_profile_certified_count += int(certified)
        self.clock_profile_max_certificate_gap_objective = max(
            self.clock_profile_max_certificate_gap_objective,
            objective_gap,
        )
        self.clock_profile_max_certificate_gap_ratio = max(
            self.clock_profile_max_certificate_gap_ratio,
            certificate_gap_ratio,
        )
        point["gradient_p"] = self.envelope_gradient(point, orbit=orbit)
        return point

    def envelope_gradient(self, point: dict, *, orbit: dict | None = None) -> np.ndarray:
        """Return the exact position gradient of the profiled Jones objective."""
        position = np.asarray(point["p_u"], dtype=float).reshape(3)
        delta_t = float(point["delta_t"])
        orbit = self._position_orbit(position) if orbit is None else orbit
        dynamic = _dynamic_vp_factors(
            np.r_[position, delta_t], self.scene, self.config
        )
        d_mat = np.asarray(dynamic["D"], dtype=complex)
        c_mat = np.asarray(dynamic["C"], dtype=complex)
        dd_dx = np.asarray(dynamic["dD_dx"], dtype=complex)
        dc_dx = np.asarray(dynamic["dC_dx"], dtype=complex)
        slices = self.cache["column_slices"]
        z_hat = np.asarray(point["x_hat"], dtype=complex)
        gradient = np.empty(3, dtype=float)
        mode = _global_vp_mode(self.config)
        e0 = (
            extract_stage1_jones_directions(self.init_estimate, self.scene)
            if mode != "fixed_pol"
            else np.empty((0, 2), dtype=complex)
        )
        identity2 = np.eye(2, dtype=complex)

        for dim in range(3):
            db = np.empty_like(z_hat)
            a_h_da = np.empty((z_hat.size, z_hat.size), dtype=complex)
            for path, sl_path in enumerate(slices):
                d_path = d_mat[:, path]
                c_path = c_mat[:, path]
                dd_path = dd_dx[dim, :, path]
                dc_path = dc_dx[dim, :, path]
                db[sl_path] = (
                    np.einsum(
                        "n,t,bnt->b",
                        dd_path.conj(),
                        c_path.conj(),
                        self.cache["y_evs"][path],
                        optimize=True,
                    )
                    + np.einsum(
                        "n,t,bnt->b",
                        d_path.conj(),
                        dc_path.conj(),
                        self.cache["y_evs"][path],
                        optimize=True,
                    )
                )
                for other, sl_other in enumerate(slices):
                    a_h_da[sl_path, sl_other] = self.cache["evs_gram"][path][other] * (
                        np.vdot(d_path, dd_dx[dim, :, other])
                        * np.vdot(c_path, c_mat[:, other])
                        + np.vdot(d_path, d_mat[:, other])
                        * np.vdot(c_path, dc_dx[dim, :, other])
                    )
            d_gram = _hermitian(a_h_da + a_h_da.conj().T)
            d_regularizer = np.zeros_like(d_gram)
            if mode != "fixed_pol":
                for path, sl_path in enumerate(slices):
                    if sl_path.stop - sl_path.start != 2:
                        raise ValueError("CCOP Jones regularizer expects two columns per path")
                    direction = np.asarray(e0[path], dtype=complex).reshape(2, 1)
                    denom = float(np.vdot(direction[:, 0], direction[:, 0]).real)
                    projector = (
                        identity2 - direction @ direction.conj().T / denom
                        if denom > 0.0
                        else np.diag([0.0, 1.0]).astype(complex)
                    )
                    d_scale = 0.5 * float(np.trace(d_gram[sl_path, sl_path]).real)
                    d_regularizer[sl_path, sl_path] = (
                        orbit["lambda_jones_per_path"][path] * d_scale * projector
                    )
            d_normal = _hermitian(d_gram + d_regularizer)
            d_score = 2.0 * float(np.real(np.vdot(db, z_hat))) - float(
                np.real(np.vdot(z_hat, d_normal @ z_hat))
            )
            gradient[dim] = -d_score / float(orbit["y_size"])
        return gradient


def refine_ccop_jvp(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
    *,
    incumbent: dict | None = None,
) -> dict:
    """Run experimental 3-D position refinement with certified clock profiling."""
    profiler = CommonClockJonesProfiler(y_raw, init_estimate, scene, config)
    xi0 = _initial_xi_from_stage1(init_estimate, scene, config)
    position_bounds = np.asarray(config["ue_bounds"], dtype=float)
    lower = position_bounds[:, 0]
    upper = position_bounds[:, 1]
    p0 = np.clip(np.asarray(xi0[:3], dtype=float), lower, upper)
    history: list[float] = []
    cache: dict[str, Any] = {
        "p": None,
        "profile": None,
        "branch_ambiguous_seen": False,
    }

    def evaluate(position: np.ndarray) -> dict:
        p_arr = np.asarray(position, dtype=float).reshape(3)
        if cache["p"] is None or not np.array_equal(p_arr, cache["p"]):
            cache["p"] = p_arr.copy()
            cache["profile"] = profiler.profile_clock(p_arr)
            cache["branch_ambiguous_seen"] = bool(
                cache["branch_ambiguous_seen"]
                or cache["profile"].get("clock_branch_ambiguous", False)
            )
        return cache["profile"]

    def objective(position: np.ndarray) -> float:
        profile = evaluate(position)
        value = float(profile["total_objective"])
        history.append(value)
        return value

    def gradient(position: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate(position)["gradient_p"], dtype=float)

    initial_profile = evaluate(p0)
    candidates: list[tuple[str, dict]] = [("stage1_start", copy.deepcopy(initial_profile))]
    optimizer_success = True
    optimizer_message = "initial point only"
    optimizer_nit = 0
    optimizer_nfev = 1
    if scipy_is_available():
        from scipy.optimize import minimize

        use_safeguard = not bool(initial_profile.get("gradient_reliable", True))
        if not use_safeguard:
            result = minimize(
                objective,
                p0,
                jac=gradient,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={
                    "maxiter": int(profiler.ccop_options["outer_max_iter"]),
                    "ftol": float(profiler.ccop_options["outer_ftol"]),
                    "gtol": float(profiler.ccop_options["outer_gtol"]),
                },
            )
            candidates.append(("ccop_lbfgsb", copy.deepcopy(evaluate(result.x))))
            optimizer_success = bool(result.success)
            optimizer_message = str(result.message)
            optimizer_nit = int(result.nit)
            optimizer_nfev = int(result.nfev)
            use_safeguard = bool(cache["branch_ambiguous_seen"])
            safeguard_start = np.asarray(result.x, dtype=float)
        else:
            safeguard_start = p0
        if use_safeguard:
            safeguard = minimize(
                objective,
                safeguard_start,
                method="Powell",
                bounds=list(zip(lower, upper)),
                options={
                    "maxiter": int(
                        profiler.ccop_options["outer_safeguard_max_iter"]
                    ),
                    "ftol": float(profiler.ccop_options["outer_ftol"]),
                    "xtol": 1.0e-6,
                },
            )
            candidates.append(
                ("ccop_powell_branch_safeguard", copy.deepcopy(evaluate(safeguard.x)))
            )
            optimizer_success = bool(safeguard.success)
            optimizer_message = f"branch safeguard: {safeguard.message}"
            optimizer_nit += int(safeguard.nit)
            optimizer_nfev += int(safeguard.nfev)

    incumbent_old_objective = float("nan")
    incumbent_profile: dict | None = None
    if incumbent is not None and "p_u" in incumbent:
        incumbent_profile = profiler.profile_clock(
            incumbent["p_u"], incumbent_clock_s=float(incumbent["delta_t"])
        )
        candidates.append(("old_position_profiled", copy.deepcopy(incumbent_profile)))
        incumbent_old_objective = float(
            incumbent.get(
                "total_objective_final",
                incumbent.get("total_objective", incumbent.get("raw_objective_final", np.nan)),
            )
        )

    selected_name, selected = min(
        candidates, key=lambda item: float(item[1]["total_objective"])
    )
    result = copy.deepcopy(selected)
    result.update(
        {
            "method": "CCOP-JVP experimental branch",
            "global_vp_solver": "ccop_jvp_profiled_3d",
            "nonlinear_dim": 3,
            "linear_nuisance_dim": int(profiler.cache["num_atoms"]),
            "selected_candidate": selected_name,
            "candidate_objectives": {
                name: float(candidate["total_objective"])
                for name, candidate in candidates
            },
            "initial_total_objective": float(initial_profile["total_objective"]),
            "total_objective_final": float(selected["total_objective"]),
            "raw_objective_final": float(selected["raw_objective"]),
            "incumbent_old_objective": incumbent_old_objective,
            "incumbent_profiled_objective": float(
                incumbent_profile["total_objective"]
            )
            if incumbent_profile is not None
            else float("nan"),
            "incumbent_non_degradation": bool(
                incumbent_profile is not None
                and np.isfinite(incumbent_old_objective)
                and float(selected["total_objective"])
                <= incumbent_old_objective
                + 1.0e-12
            ),
            "optimizer": {
                "success": optimizer_success,
                "message": optimizer_message,
                "n_iter": optimizer_nit,
                "n_eval": optimizer_nfev,
                "method": (
                    "scipy.optimize.minimize:Powell:branch_safeguard"
                    if any(
                        name == "ccop_powell_branch_safeguard"
                        for name, _ in candidates
                    )
                    else "scipy.optimize.minimize:L-BFGS-B:profiled_3d"
                )
                if scipy_is_available()
                else "initial_point_only",
            },
            "objective_history": history,
            "ccop_position_evaluations": int(profiler.position_evaluations),
            "ccop_clock_interval_evaluations": int(
                profiler.clock_interval_evaluations
            ),
            "ccop_clock_profile_evaluations": int(
                profiler.clock_profile_evaluations
            ),
            "ccop_clock_profile_certified_count": int(
                profiler.clock_profile_certified_count
            ),
            "ccop_clock_profiles_all_certified": bool(
                profiler.clock_profile_certified_count
                == profiler.clock_profile_evaluations
            ),
            "ccop_clock_profile_max_certificate_gap_objective": float(
                profiler.clock_profile_max_certificate_gap_objective
            ),
            "ccop_clock_profile_max_certificate_gap_ratio": float(
                profiler.clock_profile_max_certificate_gap_ratio
            ),
            "clock_branch_ambiguous_seen": bool(cache["branch_ambiguous_seen"]),
            "outer_branch_safeguard_used": bool(
                any(name == "ccop_powell_branch_safeguard" for name, _ in candidates)
            ),
        }
    )
    if incumbent_profile is not None:
        result["incumbent_profile_gain"] = float(
            incumbent_old_objective - incumbent_profile["total_objective"]
        )
        result["incumbent_profile_relative_error_if_same_clock"] = _relative_error(
            incumbent_old_objective,
            profiler.evaluate_clock(
                incumbent["p_u"], incumbent["delta_t"]
            )["total_objective"],
        )
    return result


def refine_four_dimensional_jvp_experimental(
    y_raw: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
    *,
    clock_coordinate: str = "distance_m",
    max_iter: int | None = None,
    max_evaluations: int | None = None,
) -> dict:
    """Run an isolated 4-D VP route with an explicit clock coordinate.

    This helper exists only for the R2/R4 validation routes.  It evaluates the
    same Jones-eliminated objective as :class:`CommonClockJonesProfiler`, but
    does *not* profile the clock.  ``clock_coordinate='distance_m'`` uses
    ``c0 * delta_t`` and therefore isolates numerical parameter scaling;
    ``'nanoseconds'`` uses ``1e9 * delta_t``; and ``'seconds'`` preserves
    the raw clock coordinate for evaluation-budget comparisons.  None of
    these choices changes the physical dictionary.
    """
    if clock_coordinate not in {"seconds", "nanoseconds", "distance_m"}:
        raise ValueError(
            "clock_coordinate must be 'seconds', 'nanoseconds', or "
            "'distance_m'"
        )
    profiler = CommonClockJonesProfiler(y_raw, init_estimate, scene, config)
    xi0 = _initial_xi_from_stage1(init_estimate, scene, config)
    p_bounds = np.asarray(config["ue_bounds"], dtype=float)
    t_bounds = np.asarray(config["delta_t_bounds"], dtype=float)
    clock_scale = {
        "seconds": 1.0,
        "nanoseconds": 1.0e9,
        "distance_m": float(scene["c0"]),
    }[clock_coordinate]
    lower = np.r_[p_bounds[:, 0], t_bounds[0] * clock_scale]
    upper = np.r_[p_bounds[:, 1], t_bounds[1] * clock_scale]
    start = np.clip(np.r_[xi0[:3], xi0[3] * clock_scale], lower, upper)
    history: list[float] = []
    cache: dict[str, Any] = {"coordinate": None, "point": None, "orbit": None}

    def evaluate(coordinate: np.ndarray) -> tuple[dict, dict]:
        value = np.asarray(coordinate, dtype=float).reshape(4)
        if cache["coordinate"] is None or not np.array_equal(
            value, cache["coordinate"]
        ):
            orbit = profiler._position_orbit(value[:3])
            point = profiler.evaluate_clock(
                value[:3], float(value[3] / clock_scale), orbit=orbit
            )
            cache.update(
                {"coordinate": value.copy(), "point": point, "orbit": orbit}
            )
        return cache["point"], cache["orbit"]

    def objective(coordinate: np.ndarray) -> float:
        point, _ = evaluate(coordinate)
        value = float(point["total_objective"])
        history.append(value)
        return value

    def gradient(coordinate: np.ndarray) -> np.ndarray:
        point, orbit = evaluate(coordinate)
        grad_p = profiler.envelope_gradient(point, orbit=orbit)
        grad_clock_s = (
            -float(point["score_first_theta"])
            * float(profiler.omega0)
            / float(orbit["y_size"])
        )
        return np.r_[grad_p, grad_clock_s / clock_scale]

    initial_point, _ = evaluate(start)
    best_coordinate = start.copy()
    best_point = copy.deepcopy(initial_point)
    optimizer_success = True
    optimizer_message = "initial point only"
    optimizer_nit = 0
    optimizer_nfev = 1
    requested_max_iter = int(
        max_iter if max_iter is not None else profiler.options.get("max_iter", 80)
    )
    if scipy_is_available() and requested_max_iter > 0:
        from scipy.optimize import minimize

        options = {
            "maxiter": requested_max_iter,
            "ftol": float(profiler.options.get("ftol", 1.0e-12)),
            "gtol": float(profiler.options.get("gtol", 1.0e-8)),
        }
        if max_evaluations is not None:
            options["maxfun"] = int(max_evaluations)
        optimize_result = minimize(
            objective,
            start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options=options,
        )
        candidate, _ = evaluate(optimize_result.x)
        if float(candidate["total_objective"]) <= float(
            best_point["total_objective"]
        ):
            best_coordinate = np.asarray(optimize_result.x, dtype=float).copy()
            best_point = copy.deepcopy(candidate)
        optimizer_success = bool(optimize_result.success)
        optimizer_message = str(optimize_result.message)
        optimizer_nit = int(optimize_result.nit)
        optimizer_nfev = int(optimize_result.nfev)

    result = copy.deepcopy(best_point)
    result.update(
        {
            "method": "experimental explicit 4-D Jones-VP scaling route",
            "global_vp_solver": "experimental_lbfgsb_4d_coordinate",
            "global_vp_backend": "numpy_cpu",
            "nonlinear_dim": 4,
            "linear_nuisance_dim": int(profiler.cache["num_atoms"]),
            "clock_coordinate": clock_coordinate,
            "clock_coordinate_scale_per_second": float(clock_scale),
            "optimized_coordinate": best_coordinate,
            "initial_total_objective": float(initial_point["total_objective"]),
            "total_objective_final": float(best_point["total_objective"]),
            "raw_objective_final": float(best_point["raw_objective"]),
            "objective_history": history,
            "vp_position_evaluations": int(profiler.position_evaluations),
            "optimizer": {
                "success": optimizer_success,
                "message": optimizer_message,
                "n_iter": optimizer_nit,
                "n_eval": optimizer_nfev,
                "method": "scipy.optimize.minimize:L-BFGS-B:explicit_4d",
            },
        }
    )
    return result
