"""Deterministic checks for the optimized §4.2 NUAA-MU experiment."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp_e5_radar_complex import _costas_perm, _phase_code
from experiments.exp_streaming_radar_complex import make_candidates, make_true_seed
from experiments.exp_streaming_radar_complex_nuaa_mu import (
    build_candidate_matrix,
    pulse_sample_times,
)
from nuaa.config import cfg_main


def _args(family: str) -> SimpleNamespace:
    return SimpleNamespace(
        family=family,
        f_lo_ghz=20.0,
        f_hi_ghz=80.0,
        bw_ghz=20.0,
        f0_amp_ghz=6.0,
        f_jitter_ghz=6.0,
        codebook_size=8,
        n_chips=64,
        nC=48,
    )


def test_window_materializes_200_points_over_pulse_segments() -> None:
    cfg = cfg_main()
    cosets = np.array([10, 120, 300, 650, 990], dtype=np.int64)
    waveform_t, event_cosets, acquisition_t = pulse_sample_times(
        cosets, cfg, window_periods=40, pulse_width_ns=16.0
    )
    assert waveform_t.shape == (200,)
    assert event_cosets.shape == (200,)
    assert acquisition_t.shape == (200,)
    assert np.array_equal(np.unique(waveform_t // cfg.N0), np.arange(4))
    assert np.all(np.diff(acquisition_t) >= 0)


def test_candidate_matrix_builds_only_requested_columns() -> None:
    cfg = cfg_main()
    rng = np.random.default_rng(5)
    args = _args("frank")
    true_seed = make_true_seed(args, rng)
    candidates, true_loc = make_candidates(true_seed, args, rng)
    waveform_t, _, _ = pulse_sample_times(
        np.array([20, 180, 360, 720, 980]),
        cfg,
        window_periods=40,
        pulse_width_ns=16.0,
    )
    matrix = build_candidate_matrix(
        candidates,
        pulse_idx=20,
        waveform_times=waveform_t,
        n_pulse=16_000,
        codebook_size=args.codebook_size,
    )
    assert true_loc is not None
    assert matrix.shape == (200, 48)
    assert matrix.dtype == np.complex64
    assert np.allclose(np.linalg.norm(matrix, axis=0), 1.0, atol=1e-5)


def test_deterministic_codes_are_cached() -> None:
    _phase_code.cache_clear()
    _costas_perm.cache_clear()
    first_phase = _phase_code("frank", 3, 64)
    second_phase = _phase_code("frank", 3, 64)
    first_costas = _costas_perm(2, 64)
    second_costas = _costas_perm(2, 64)
    assert first_phase is second_phase
    assert first_costas is second_costas
    assert _phase_code.cache_info().hits == 1
    assert _costas_perm.cache_info().hits == 1
