"""线性化前向算子 A(τ) 与测量矩阵（对应 计算光采样.md §3）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from scipy import signal

from .config import SystemConfig
from . import layout as _layout
from .kernels import pd_tha_kernel


@dataclass
class LinearOperator:
    matvec: Callable[[np.ndarray], np.ndarray]
    rmatvec: Callable[[np.ndarray], np.ndarray]
    M: int
    N_ref: int


def build_phi(cosets: np.ndarray, N0: int) -> np.ndarray:
    cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
    k = np.arange(N0)
    return np.exp(2j * np.pi * np.outer(cosets, k) / N0)


def build_A_spec(cosets: np.ndarray, N0: int) -> np.ndarray:
    return build_phi(cosets, N0) / N0


def stack_phi(coset_sets: Sequence[np.ndarray], N0: int):
    all_cosets = np.concatenate([np.asarray(c, np.int64).reshape(-1) for c in coset_sets])
    uniq = np.unique(all_cosets)
    return build_A_spec(uniq, N0), uniq


def measure_spectrum(
    X: np.ndarray,
    cosets: np.ndarray,
    N0: int,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    A = build_A_spec(cosets, N0)
    Y = A @ X
    if noise_std > 0:
        rng = rng or np.random.default_rng()
        Y = Y + (noise_std / np.sqrt(2)) * (
            rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)
        )
    return Y


def _scatter(vals: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.result_type(vals.dtype, np.float64))
    np.add.at(out, idx, vals)
    return out


def build_forward_operator(layout: _layout.Layout, cfg: SystemConfig, cal) -> LinearOperator:
    """A = S_smp · H_PD·H_THA · scatter(MZM · S_opt)。"""
    assert layout.t_opt is not None and layout.sample_idx is not None
    idx_opt = np.round(layout.t_opt / cfg.eta_ps).astype(np.int64)
    idx_smp = np.asarray(layout.sample_idx, dtype=np.int64)
    branch = np.asarray(layout.branch, dtype=np.int64)
    cal = cal.ensure(cfg.L, cfg)
    N_ref = int(max(idx_opt.max(), idx_smp.max())) + len(
        pd_tha_kernel(cfg.pd_bw_GHz, cfg.tha_bw_GHz, 256, cfg.eta_ps)
    )
    h = pd_tha_kernel(cfg.pd_bw_GHz, cfg.tha_bw_GHz, N_ref, cfg.eta_ps)
    g = cal.g[branch] * cal.mzm_slope

    def matvec(x_ref: np.ndarray) -> np.ndarray:
        x_ref = np.asarray(x_ref).reshape(-1)
        s_evt = x_ref[np.clip(idx_opt, 0, x_ref.size - 1)]
        p_evt = g * s_evt
        grid = _scatter(p_evt, np.clip(idx_opt, 0, N_ref - 1), N_ref)
        i_g = signal.fftconvolve(grid, h, mode="full")[:N_ref]
        return i_g[np.clip(idx_smp, 0, N_ref - 1)]

    def rmatvec(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        s = _scatter(y, np.clip(idx_smp, 0, N_ref - 1), N_ref)
        z = signal.fftconvolve(s, h[::-1], mode="full")[:N_ref]
        z_opt = z[np.clip(idx_opt, 0, N_ref - 1)]
        return _scatter(np.conj(g) * z_opt, np.clip(idx_opt, 0, N_ref - 1), N_ref)

    return LinearOperator(matvec=matvec, rmatvec=rmatvec, M=idx_smp.size, N_ref=N_ref)


def _bang_bang_inc(target: np.ndarray, prev: np.ndarray, max_step: int) -> np.ndarray:
    if max_step <= 0:
        return prev.copy()
    diff = target - prev
    step = np.minimum(np.abs(diff), int(max_step)).astype(np.int64)
    out = prev.copy()
    move = np.abs(diff) > 0
    out[move] = prev[move] + np.sign(diff[move]).astype(np.int64) * step[move]
    return out


def matrix_to_delays(
    desired_cosets: np.ndarray,
    cfg: SystemConfig,
    prev_tau_ps: Optional[np.ndarray] = None,
    dt_s: Optional[float] = None,
) -> np.ndarray:
    """期望陪集 -> 可部署 τ（前缀和投影 + 时隙互异 + bang-bang）。

    When ``prev_tau_ps`` and ``dt_s`` are supplied, the returned delay respects
    the EDL slew rate. Sub-ps moves are not rounded up: for time slices shorter
    than ``eta / v`` the actuator may legitimately stay frozen.
    """
    desired = np.asarray(desired_cosets, dtype=np.float64).reshape(-1)
    assert desired.size == cfg.L
    inc = np.empty(cfg.L, dtype=np.int64)
    inc[0] = int(np.clip(np.round(desired[0]), 0, cfg.dmax_slots))
    for l in range(1, cfg.L):
        inc[l] = int(
            np.clip(np.round(desired[l] - desired[l - 1]), cfg.min_slot_gap, cfg.dmax_slots)
        )
    inc = _layout.enforce_distinct_slots(inc, cfg)
    if prev_tau_ps is not None and dt_s is not None:
        prev_inc = _layout.tau_to_increments(prev_tau_ps, cfg)
        max_step = int(np.floor(cfg.v_ramp_ps_per_s * float(dt_s) / cfg.eta_ps + 1e-12))
        inc = _bang_bang_inc(inc, prev_inc, max_step)
        inc = _layout.enforce_distinct_slots(inc, cfg)
    return _layout.increments_to_tau(inc, cfg)


def check_phi_time_consistency(
    cosets: np.ndarray,
    N0: int,
    rng: Optional[np.random.Generator] = None,
    atol: float = 1e-8,
) -> float:
    rng = rng or np.random.default_rng(0)
    X = rng.standard_normal(N0) + 1j * rng.standard_normal(N0)
    y_spec = np.real(build_A_spec(cosets, N0) @ X)
    x_time = np.real(np.fft.ifft(X))
    y_time = x_time[np.asarray(cosets, np.int64)]
    return float(np.max(np.abs(y_spec - y_time)))
