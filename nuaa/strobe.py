"""THA strobe 预测与逐事件转换窗（对应 计算光采样.md §有限转换窗 / 自校准）。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SystemConfig
from .kernels import pd_tha_kernel


@dataclass
class Calibration:
    g: np.ndarray | None = None
    sigma: np.ndarray | None = None
    delta_ctrl_ps: float = 0.0
    Vpi: float = 1.0
    R: float = 0.8
    P_opt: float = 1.0e-3
    RL: float = 50.0
    T_K: float = 300.0
    full_scale: float = 1.0
    sigma_q0: float = 1e-3

    @property
    def mzm_slope(self) -> float:
        return 0.5 * np.pi / self.Vpi

    def ensure(self, L: int, cfg: SystemConfig):
        if self.g is None:
            self.g = np.ones(L)
        if self.sigma is None:
            self.sigma = np.full(L, cfg.tau_reg_ps)
        return self


@dataclass
class StrobeSchedule:
    """Predicted event-wise THA strobe schedule for a streaming segment."""

    t_opt_ps: np.ndarray
    t_sample_ps: np.ndarray
    t_cnv_ps: np.ndarray
    branch: np.ndarray
    period: np.ndarray
    sample_idx: np.ndarray

    def ledger(self, cfg: SystemConfig) -> dict:
        t_cnv = np.asarray(self.t_cnv_ps, dtype=np.float64)
        return dict(
            n_events=int(t_cnv.size),
            t_cnv_min_ps=float(np.min(t_cnv)) if t_cnv.size else 0.0,
            t_cnv_med_ps=float(np.median(t_cnv)) if t_cnv.size else 0.0,
            strobe_feasible_rate=float(np.mean(t_cnv >= cfg.t_cnv_min_ps)) if t_cnv.size else 1.0,
        )


def peak_time(t_opt_ps: np.ndarray, cfg: SystemConfig, search_ps: float = 800.0) -> np.ndarray:
    """在叠加 PD/THA 响应上逐事件搜索局部峰值偏移（相对 t_opt）。

    按时间顺序贪心分配峰值，避免相邻事件落到同一栅格峰。
    """
    t_opt = np.asarray(t_opt_ps, dtype=np.float64).reshape(-1)
    M = t_opt.size
    if M == 0:
        return np.zeros(0, dtype=np.float64)
    n_grid = int(np.ceil((t_opt.max() + search_ps) / cfg.eta_ps)) + 64
    h = pd_tha_kernel(cfg.pd_bw_GHz, cfg.tha_bw_GHz, n_grid, cfg.eta_ps)
    grid = np.zeros(n_grid, dtype=np.float64)
    idx_opt = np.round(t_opt / cfg.eta_ps).astype(np.int64)
    np.add.at(grid, np.clip(idx_opt, 0, n_grid - 1), 1.0)
    resp = np.convolve(grid, h, mode="full")[:n_grid]
    half = int(np.ceil(search_ps / cfg.eta_ps))
    min_gap_slots = max(1, int(np.ceil(cfg.t_acq_ps / cfg.eta_ps)))
    offsets = np.zeros(M, dtype=np.float64)
    last_pk = -min_gap_slots
    for i, ti in enumerate(t_opt):
        c = int(np.round(ti / cfg.eta_ps))
        lo = max(0, c - half, last_pk + min_gap_slots)
        hi = min(n_grid, c + half + 1)
        if lo >= hi:
            pk = max(lo, min(c, n_grid - 1))
        else:
            pk = lo + int(np.argmax(resp[lo:hi]))
        offsets[i] = (pk - c) * cfg.eta_ps
        last_pk = pk
    return offsets


def predict_strobe_from_times(t_opt_ps: np.ndarray, cfg: SystemConfig, cal: Calibration) -> np.ndarray:
    t_opt = np.asarray(t_opt_ps, dtype=np.float64).reshape(-1)
    return t_opt + peak_time(t_opt, cfg) + float(cal.delta_ctrl_ps)


def schedule_from_layout_times(
    t_opt_ps: np.ndarray,
    branch: np.ndarray,
    period: np.ndarray,
    cfg: SystemConfig,
    cal: Calibration,
) -> StrobeSchedule:
    """Build a strobe schedule from absolute optical event times."""
    t_opt = np.asarray(t_opt_ps, dtype=np.float64).reshape(-1)
    order = np.argsort(t_opt, kind="stable")
    t_opt = t_opt[order]
    branch = np.asarray(branch, dtype=np.int64).reshape(-1)[order]
    period = np.asarray(period, dtype=np.int64).reshape(-1)[order]
    # 峰值搜索只依赖周期内相对间距；用周期起点作参考避免卷积栅格随绝对时间增长。
    t_ref = float(np.floor(t_opt.min() / cfg.T0_ps) * cfg.T0_ps) if t_opt.size else 0.0
    t_sample = t_ref + predict_strobe_from_times(t_opt - t_ref, cfg, cal)
    t_next = np.roll(t_sample, -1)
    if t_next.size:
        t_next[-1] += cfg.T0_ps
    t_cnv = t_next - t_sample - cfg.t_acq_ps
    sample_idx = np.round(t_sample / cfg.eta_ps).astype(np.int64)
    return StrobeSchedule(
        t_opt_ps=t_opt,
        t_sample_ps=t_sample,
        t_cnv_ps=t_cnv,
        branch=branch,
        period=period,
        sample_idx=sample_idx,
    )


def sigma_cnv2(t_cnv_ps: np.ndarray, cfg: SystemConfig, cal: Calibration) -> np.ndarray:
    t_cnv = np.asarray(t_cnv_ps, dtype=np.float64)
    reg = cfg.tau_reg_ps
    extra = np.maximum(0.0, cfg.t_cnv_min_ps - t_cnv) / max(reg, 1e-6)
    return cal.sigma_q0 ** 2 + (cfg.tau_reg_ps * extra) ** 2
