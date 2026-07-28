"""Domain of the Stage-I coupled factor recovery.

``stage1_factor_domain`` selects the EVS dimension the coupled Hankel LS and the
rank-one split operate in:

``"raw_evs"``        the full EVS dimension ``I`` (the published behaviour,
                     retained as an ablation);
``"compressed_evs"`` the known union subspace ``range(B)`` of dimension
                     ``r = 2K``, with the EVS factor lifted back through ``B``
                     (the default since 2026-07-28).

The switch is justified by an exact identity, not by an approximation argument.
Every admissible EVS factor lies in ``range(B)`` because ``Theta_k`` depends only
on the known RIS->BS direction, so ``B`` discards no signal; the coupled LS is
linear in the EVS mode and ``B`` is an isometry, so the rank-one split commutes
with the lift.  These tests pin both halves of that statement, since the whole
complexity argument rests on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ccop_stage1_initializer import known_evs_union_basis
from src.channel_model import (
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.config import default_config
from src.estimators import (
    _coupled_hankel_factor_initialization,
    _factor_recovery_observation,
)
from src.tensor_utils import hankelize_frequency


def _scene(snr_db: float = 0.0, seed: int = 20260728):
    config = default_config()
    config["K"] = 3
    config["seed"] = seed
    config["SNR_dB"] = snr_db
    rng = np.random.default_rng(int(config["seed"]))
    scene = generate_scene(config, rng)
    truth = channel_components(
        scene, scene["p_u_true"], scene["delta_t_true"],
        scene["gamma_true"], scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(truth, scene["beta_true"])
    y_noisy, _ = add_awgn(
        y_true, snr_db, np.random.default_rng(7),
        active_mask=scene.get("evs_observation_mask"),
    )
    return scene, config, truth, hankelize_frequency(y_noisy, scene["P"])


def _rank_one_signature(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Phase-invariant signature of the split.

    The split stores ``a = u0*sqrt(s0)`` and ``c = vh[0]*sqrt(s0)``, so ``c``
    already carries the conjugate and ``a c^T`` reproduces ``s0 u0 v0^H``.  That
    product is invariant under the SVD's residual unit phase; ``a`` and ``c``
    individually are not.
    """
    return np.stack([np.outer(a[:, k], c[:, k]) for k in range(a.shape[1])])


def test_default_is_the_compressed_union_domain():
    """Settled 2026-07-28 on 256 paired trials per SNR.

    Against ``raw_evs`` the compressed domain is outlier-neutral (exact McNemar
    p >= 0.73 at -20/-15/-10 dB), gives a significant per-trial error reduction
    at -20 dB (sign test 94/152, p = 0.004) and a 25% conditional-RMSE
    reduction at -15 dB, is indistinguishable at -10 dB, and is 11x cheaper
    on the coupled-LS block.  The default therefore moved off ``raw_evs``,
    which stays reachable for the published-behaviour ablation.
    """
    assert default_config()["stage1_factor_domain"] == "compressed_evs"


def test_unknown_domain_is_rejected():
    scene, config, _, z = _scene()
    with pytest.raises(ValueError, match="stage1_factor_domain"):
        _factor_recovery_observation(z, scene, dict(config, stage1_factor_domain="svd"))


def test_raw_domain_passes_the_tensor_through_untouched():
    scene, config, _, z = _scene()
    obs, basis, domain = _factor_recovery_observation(
        z, scene, dict(config, stage1_factor_domain="raw_evs")
    )
    assert obs is z
    assert basis is None
    assert domain == "raw_evs"


def test_compressed_domain_reduces_the_evs_mode_to_the_union_rank():
    scene, config, _, z = _scene()
    obs, basis, domain = _factor_recovery_observation(
        z, scene, dict(config, stage1_factor_domain="compressed_evs")
    )
    assert domain == "compressed_evs"
    assert obs.shape == (2 * int(scene["K"]), z.shape[1], z.shape[2], z.shape[3])
    assert basis.shape == (int(scene["I"]), 2 * int(scene["K"]))
    # 96 -> 6 in the reference configuration.
    assert z.shape[0] // obs.shape[0] == 16


def test_true_evs_factors_lie_in_the_union_subspace():
    """The premise the whole reduction rests on: B discards no signal."""
    scene, _, truth, _ = _scene()
    basis, _ = known_evs_union_basis(scene)
    a_true = np.asarray(truth["a_EVS"], dtype=complex).T
    leakage = np.linalg.norm(a_true - basis @ (basis.conj().T @ a_true))
    assert leakage / np.linalg.norm(a_true) < 1.0e-12


@pytest.mark.parametrize("snr_db", [-20.0, 0.0, 20.0])
def test_compressed_recovery_equals_the_raw_recovery_on_projected_data(snr_db):
    """``B @ F_r(B^H Z) == F(B B^H Z)`` -- the identity that makes the switch
    a reformulation rather than an approximation."""
    scene, config, truth, z = _scene(snr_db=snr_db)
    poles = np.asarray(truth["poles"], dtype=complex)
    basis, _ = known_evs_union_basis(scene)

    z_compressed = np.einsum("ir,iplt->rplt", basis.conj(), z, optimize=True)
    z_projected = np.einsum("ir,rplt->iplt", basis, z_compressed, optimize=True)

    a_small, c_small = _coupled_hankel_factor_initialization(
        z_compressed, poles, config=config
    )
    a_projected, c_projected = _coupled_hankel_factor_initialization(
        z_projected, poles, config=config
    )
    a_lifted = basis @ a_small

    assert a_lifted.shape == a_projected.shape
    np.testing.assert_allclose(
        _rank_one_signature(a_lifted, c_small),
        _rank_one_signature(a_projected, c_projected),
        rtol=1.0e-10, atol=1.0e-10,
    )


def test_compression_actually_changes_the_estimate_at_low_snr():
    """It is a denoiser, not a no-op: it must move the raw-domain estimate, and
    it must move it less as the SNR rises."""
    scene, config, truth, z_low = _scene(snr_db=-20.0)
    poles = np.asarray(truth["poles"], dtype=complex)
    basis, _ = known_evs_union_basis(scene)

    def relative_shift(z):
        z_c = np.einsum("ir,iplt->rplt", basis.conj(), z, optimize=True)
        a_raw, c_raw = _coupled_hankel_factor_initialization(z, poles, config=config)
        a_small, c_small = _coupled_hankel_factor_initialization(
            z_c, poles, config=config
        )
        raw = _rank_one_signature(a_raw, c_raw)
        compressed = _rank_one_signature(basis @ a_small, c_small)
        return float(np.linalg.norm(raw - compressed) / np.linalg.norm(raw))

    _, _, _, z_high = _scene(snr_db=20.0)
    shift_low = relative_shift(z_low)
    shift_high = relative_shift(z_high)

    assert shift_low > 1.0e-2
    assert shift_high < shift_low / 10.0


def test_precompressed_tensor_is_reused_when_the_rank_matches():
    """The delay path already forms ``B^H Z``; the factor path must not pay twice."""
    scene, config, _, z = _scene()
    basis, _ = known_evs_union_basis(scene)
    precompressed = np.einsum("ir,iplt->rplt", basis.conj(), z, optimize=True)

    obs, returned_basis, _ = _factor_recovery_observation(
        z, scene, dict(config, stage1_factor_domain="compressed_evs"),
        precompressed=precompressed,
    )
    np.testing.assert_allclose(obs, precompressed)
    np.testing.assert_allclose(returned_basis, basis)


def test_precompressed_tensor_is_ignored_when_the_rank_disagrees():
    scene, config, _, z = _scene()
    wrong = np.zeros((3, z.shape[1], z.shape[2], z.shape[3]), dtype=complex)
    obs, basis, _ = _factor_recovery_observation(
        z, scene, dict(config, stage1_factor_domain="compressed_evs"),
        precompressed=wrong,
    )
    assert obs.shape[0] == basis.shape[1] == 2 * int(scene["K"])
    assert np.linalg.norm(obs) > 0.0
