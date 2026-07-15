"""PD/THA 脉冲叠加核与条件数（对应 计算光采样.md §物理观测模型 / §有限转换窗）。"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import SystemConfig


def _tau_ps(bw_ghz: float) -> float:
    return 1e3 / (2.0 * np.pi * bw_ghz)  # ps


def pd_impulse(bw_ghz: float, n: int, eta_ps: float) -> np.ndarray:
    """单极点 PD 冲激响应（归一化峰值=1），长度 n 个 1ps 栅格。"""
    t = np.arange(n, dtype=np.float64) * eta_ps
    tau = _tau_ps(bw_ghz)
    h = (t / tau) * np.exp(1.0 - t / tau)
    h[t < 0] = 0.0
    peak = float(h.max()) if h.size else 1.0
    return h / max(peak, 1e-12)


def pd_tha_kernel(pd_bw_GHz: float, tha_bw_GHz: float, n: int, eta_ps: float) -> np.ndarray:
    """H_PD 与 H_THA 串联（均为单极点），返回离散核。"""
    h_pd = pd_impulse(pd_bw_GHz, n, eta_ps)
    h_tha = pd_impulse(tha_bw_GHz, n, eta_ps)
    return signal.fftconvolve(h_pd, h_tha, mode="full")[:n]


def _event_indices(t_ps: np.ndarray, eta_ps: float, n_grid: int) -> np.ndarray:
    idx = np.round(np.asarray(t_ps, dtype=np.float64) / eta_ps).astype(np.int64)
    return np.clip(idx, 0, n_grid - 1)


def build_event_kernel(t_opt_ps: np.ndarray, t_sample_ps: np.ndarray,
                       cfg: SystemConfig, n_grid: int | None = None) -> np.ndarray:
    """光事件时刻 -> THA 读出时刻 的叠加核 H[i,j] = h(t_sample_i - t_opt_j)。"""
    t_opt = np.asarray(t_opt_ps, dtype=np.float64).reshape(-1)
    t_smp = np.asarray(t_sample_ps, dtype=np.float64).reshape(-1)
    M = t_opt.size
    if n_grid is None:
        n_grid = int(max(t_smp.max(), t_opt.max()) / cfg.eta_ps) + 1
    h = pd_tha_kernel(cfg.pd_bw_GHz, cfg.tha_bw_GHz, n_grid, cfg.eta_ps)
    H = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        dt_slots = np.round((t_smp[i] - t_opt) / cfg.eta_ps).astype(np.int64)
        for j, d in enumerate(dt_slots):
            if 0 <= d < h.size:
                H[i, j] = h[d]
    return H


def cond_number(t_opt_ps: np.ndarray, t_sample_ps: np.ndarray, cfg: SystemConfig) -> float:
    H = build_event_kernel(t_opt_ps, t_sample_ps, cfg)
    if H.size == 0:
        return 1.0
    s = np.linalg.svd(H, compute_uv=False)
    if s[-1] < 1e-12:
        return float("inf")
    return float(s[0] / s[-1])
