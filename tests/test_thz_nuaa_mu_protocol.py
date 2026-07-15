"""Regression tests for the paper Table 4 THz protocol."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp_streaming_thz_nuaa_mu import (
    gen_thz_train_batch,
    make_thz_period,
    recover_symbols,
)
from nuaa.config import SystemConfig
from nuaa.streaming import StreamingNUAAController


class _DummyModel:
    def eval(self):
        return self

    def forward_chunk(
        self,
        _tok,
        _dt_norm,
        A,
        _Y,
        **_kwargs,
    ):
        batch, events, n_cand = A.shape
        xhat = torch.zeros((batch, n_cand), dtype=torch.complex64)
        # Put mass on the first K candidates so pretrained-only top-K is nontrivial.
        useful = torch.zeros((batch, n_cand))
        useful[:, :2] = torch.tensor([0.9, 0.8])
        aux = {
            "useful_prior": useful,
            "event_weight": torch.ones((batch, events)),
            "jammer_logits": torch.zeros((batch, n_cand)),
            "burst_logits": torch.zeros((batch, events)),
            "burst_complex": torch.zeros((batch, events), dtype=torch.complex64),
            "support_logits": torch.logit(useful.clamp(1e-4, 1 - 1e-4)),
        }
        return xhat, aux, {"n_events": events}


def test_obs_and_clean_target_are_separated() -> None:
    """Measurements come from impaired spectrum; target is clean spectrum."""
    cfg = SystemConfig(N0=5000)
    rng = np.random.default_rng(3)
    x_obs, x_tgt, support, sym_clean, sym_obs, _ = make_thz_period(
        cfg, center=1500, evm_pct=18.0, rng=rng, n_sym=32)
    assert support.size == 7
    assert not np.allclose(sym_clean, sym_obs)
    assert not np.allclose(x_obs[support], x_tgt[support])
    # Shared carrier direction: after removing scales, phases align.
    a = x_obs[support] / (np.linalg.norm(x_obs[support]) + 1e-12)
    b = x_tgt[support] / (np.linalg.norm(x_tgt[support]) + 1e-12)
    assert abs(np.vdot(a, b)) > 0.99
    # Scale tracks symbol RMS (clean vs impaired).
    assert np.isclose(
        np.linalg.norm(x_tgt[support]) / (np.linalg.norm(x_obs[support]) + 1e-12),
        np.linalg.norm(sym_clean) / (np.linalg.norm(sym_obs) + 1e-12),
        rtol=1e-5,
    )


def test_thz_training_target_is_clean_spectrum() -> None:
    cfg = SystemConfig(N0=5000)
    batch = gen_thz_train_batch(
        cfg,
        batch=3,
        nC=48,
        K=7,
        steps=4,
        snr_db=5.0,
        rng=np.random.default_rng(12),
        evm_lo=10.0,
        evm_hi=20.0,
    )
    _, _, _, _, x_cand, cand, _, _, support, _, soft_prior = batch

    assert cand.shape == (3, 48, 3)
    assert soft_prior.shape == (3, 48)
    assert support.shape == (3, 7)
    for row in range(3):
        assert len(np.unique(support[row])) == 7
        assert torch.count_nonzero(x_cand[row]).item() == 7
        assert torch.all(x_cand[row, torch.as_tensor(support[row])].abs() > 0)


def test_clean_prior_lock_does_not_remove_fake_jammer() -> None:
    cfg = SystemConfig(N0=128)
    ctrl = StreamingNUAAController(cfg, K=2, seed=0)
    rng = np.random.default_rng(4)
    candidates = np.arange(20, 25)
    support = np.array([1, 3])
    x = np.zeros(5, dtype=np.complex128)
    x[support] = [0.8 + 0.2j, -0.3 + 0.7j]
    A = (
        rng.standard_normal((40, 5)) + 1j * rng.standard_normal((40, 5))
    ) / np.sqrt(80)
    y = A @ x
    prior = np.zeros(5)
    prior[support] = [0.99, 1.0]

    rec = ctrl.reconstruct_with_model(
        _DummyModel(),
        np.zeros((40, 10), dtype=np.float32),
        np.zeros(40, dtype=np.float32),
        A.astype(np.complex64),
        y.astype(np.complex64),
        candidates,
        prior_override=prior,
        allow_prior_lock=True,
        accumulate_coefficients=False,
        detect_jammer=False,
        truth_X=np.pad(x, (20, cfg.N0 - 25)),
        truth_support=candidates[support],
    )

    assert rec.jammer_index is None
    assert set(rec.support.tolist()) == set(candidates[support].tolist())
    assert np.allclose(rec.Xhat[candidates[support], 0], x[support], atol=1e-5)


def test_pretrained_only_path_uses_model_prior_without_scene_override() -> None:
    """NUAA-MU THz path: support from model useful_prior, no scene prior lock."""
    cfg = SystemConfig(N0=128)
    ctrl = StreamingNUAAController(cfg, K=2, seed=1)
    rng = np.random.default_rng(5)
    candidates = np.arange(10, 15)
    support = np.array([0, 1])  # model dummy prior peaks on first two
    x = np.zeros(5, dtype=np.complex128)
    x[support] = [1.0, -0.5j]
    A = (
        rng.standard_normal((50, 5)) + 1j * rng.standard_normal((50, 5))
    ) / np.sqrt(100)
    y = A @ x

    rec = ctrl.reconstruct_with_model(
        _DummyModel(),
        np.zeros((50, 10), dtype=np.float32),
        np.zeros(50, dtype=np.float32),
        A.astype(np.complex64),
        y.astype(np.complex64),
        candidates,
        cand_feat=np.zeros((5, 3), dtype=np.float32),
        prior_override=None,
        allow_prior_lock=False,
        accumulate_coefficients=True,
        detect_jammer=False,
        clean_ridge_rel=0.003,
        truth_X=np.pad(x, (10, cfg.N0 - 15)),
        truth_support=candidates[support],
    )

    assert set(rec.support.tolist()) == set(candidates[support].tolist())
    assert rec.jammer_index is None
    assert np.allclose(rec.Xhat[candidates[support], 0], x[support], atol=5e-3)


def test_constellation_requires_support_lock() -> None:
    symbols = np.array([1 + 1j, -1 + 1j], dtype=np.complex128)
    support = np.array([10, 11])
    xhat = np.zeros(32, dtype=np.complex128)
    xhat[support] = [0.5 + 0.2j, 0.7 - 0.1j]

    ok, hit_ok = recover_symbols(symbols, xhat, support, est_support=support)
    assert not np.allclose(ok, 0)
    assert hit_ok == 1.0

    # Wrong support must not unlock the observed symbol frame.
    bad, hit_bad = recover_symbols(symbols, xhat, support, est_support=np.array([0, 1]))
    assert np.allclose(bad, 0)
    assert hit_bad == 0.0

    # Partial recovery reports fractional hit and still returns a scaled frame.
    partial, hit_p = recover_symbols(
        symbols, xhat, support, est_support=np.array([10]))
    assert hit_p == 0.5
    assert not np.allclose(partial, 0)
