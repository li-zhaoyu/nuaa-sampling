"""脉冲布局：τ ⇄ 陪集 / 掩码 / 事件时刻 / strobe（对应 计算光采样.md §3 正向）。

级联前缀和：inc[l]=round(τ_l/η)，cosets[l]=Σ inc[0:l+1]；硬约束为 1ps 时隙互异。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import SystemConfig
from .strobe import predict_strobe_from_times


@dataclass
class Layout:
    cosets: np.ndarray
    tau_ps: np.ndarray
    cfg: SystemConfig
    t_opt: Optional[np.ndarray] = None
    branch: Optional[np.ndarray] = None
    period: Optional[np.ndarray] = None
    t_sample: Optional[np.ndarray] = None
    t_cnv: Optional[np.ndarray] = None
    sample_idx: Optional[np.ndarray] = None

    def mask(self) -> np.ndarray:
        return cosets_to_mask(self.cosets, self.cfg)


# --------------------------------------------------------------------------
# τ ⇄ 陪集（前缀和）
# --------------------------------------------------------------------------
def tau_to_increments(tau_ps: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    tau_ps = np.asarray(tau_ps, dtype=np.float64).reshape(-1)
    assert tau_ps.shape == (cfg.L,)
    inc = np.round(tau_ps / cfg.eta_ps).astype(np.int64)
    inc = np.clip(inc, 0, cfg.dmax_slots)
    return inc


def increments_to_tau(inc: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    inc = np.asarray(inc, dtype=np.int64).reshape(-1)
    return inc.astype(np.float64) * cfg.eta_ps


def increments_to_cosets(inc: np.ndarray) -> np.ndarray:
    inc = np.asarray(inc, dtype=np.int64).reshape(-1)
    return np.cumsum(inc)


def tau_to_cosets(tau_ps: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    return increments_to_cosets(tau_to_increments(tau_ps, cfg))


def cosets_to_increments(cosets: np.ndarray) -> np.ndarray:
    cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
    inc = np.empty_like(cosets)
    inc[0] = cosets[0]
    inc[1:] = np.diff(cosets)
    return inc


def cosets_to_tau(cosets: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    return increments_to_tau(cosets_to_increments(cosets), cfg)


def cosets_to_mask(cosets: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    mask = np.zeros(cfg.N0, dtype=np.int8)
    mask[np.asarray(cosets, dtype=np.int64)] = 1
    return mask


def enforce_distinct_slots(inc: np.ndarray, cfg: SystemConfig, wrap: bool = True) -> np.ndarray:
    """保证 inc[l]≥min_slot_gap (l≥1)、Σ inc < N0，且跨周期间隔≥min_slot_gap。"""
    inc = np.asarray(inc, dtype=np.int64).copy().reshape(-1)
    L = inc.size
    gmin = cfg.min_slot_gap
    dmax = cfg.dmax_slots
    inc = np.clip(inc, 0, dmax)
    for _ in range(8 * L):
        changed = False
        for l in range(1, L):
            if inc[l] < gmin:
                inc[l] = gmin
                changed = True
        if int(inc.sum()) >= cfg.N0:
            overflow = int(inc.sum()) - cfg.N0 + 1
            for l in range(L - 1, 0, -1):
                take = min(overflow, max(0, inc[l] - gmin))
                if take:
                    inc[l] -= take
                    overflow -= take
                if overflow <= 0:
                    break
            changed = True
        if wrap:
            cosets = increments_to_cosets(inc)
            wrap_gap = (cfg.N0 - cosets[-1]) + cosets[0]
            if wrap_gap < gmin:
                need = gmin - wrap_gap
                if inc[0] + need <= dmax:
                    inc[0] += need
                else:
                    inc[0] = dmax
                changed = True
        if not changed:
            break
    return inc


def validate_layout(cosets: np.ndarray, cfg: SystemConfig, check_gap: bool = True) -> bool:
    cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
    if cosets.size != cfg.L:
        return False
    if not np.all(np.diff(cosets) >= cfg.min_slot_gap):
        return False
    if cosets[-1] >= cfg.N0:
        return False
    if check_gap:
        wrap = (cfg.N0 - cosets[-1]) + cosets[0]
        if wrap < cfg.min_slot_gap:
            return False
    inc = cosets_to_increments(cosets)
    if np.any(inc < 0) or np.any(inc > cfg.dmax_slots):
        return False
    return True


def make_layout(tau_ps: np.ndarray, cfg: SystemConfig) -> Layout:
    inc = enforce_distinct_slots(tau_to_increments(tau_ps, cfg), cfg)
    cosets = increments_to_cosets(inc)
    tau = increments_to_tau(inc, cfg)
    return Layout(cosets=cosets, tau_ps=tau, cfg=cfg)


def delays_to_layout(
    tau_ps: np.ndarray,
    n_periods: int,
    cfg: SystemConfig,
    cal=None,
) -> Layout:
    """τ -> 陪集/掩码/多周期 t_opt；若提供 cal 则预测 THA strobe 与转换窗。"""
    lay = make_layout(tau_ps, cfg)
    L, P = cfg.L, int(n_periods)
    t_opt = []
    branch = []
    period = []
    for k in range(P):
        for l in range(L):
            t_opt.append(k * cfg.T0_ps + lay.cosets[l] * cfg.eta_ps)
            branch.append(l)
            period.append(k)
    t_opt = np.asarray(t_opt, dtype=np.float64)
    branch = np.asarray(branch, dtype=np.int64)
    period = np.asarray(period, dtype=np.int64)
    order = np.argsort(t_opt, kind="stable")
    t_opt = t_opt[order]
    branch = branch[order]
    period = period[order]

    t_sample = t_cnv = sample_idx = None
    if cal is not None:
        t_sample = predict_strobe_from_times(t_opt, cfg, cal)
        t_next = np.roll(t_sample, -1)
        t_next[-1] += cfg.T0_ps
        t_cnv = t_next - t_sample - cfg.t_acq_ps
        sample_idx = np.round(t_sample / cfg.eta_ps).astype(np.int64)

    return Layout(
        cosets=lay.cosets,
        tau_ps=lay.tau_ps,
        cfg=cfg,
        t_opt=t_opt,
        branch=branch,
        period=period,
        t_sample=t_sample,
        t_cnv=t_cnv,
        sample_idx=sample_idx,
    )


def event_times(cosets: np.ndarray, n_periods: int, cfg: SystemConfig):
    """多周期事件（升序）。返回 times_ps, branch, period, coset。"""
    cosets = np.asarray(cosets, dtype=np.int64)
    L = cfg.L
    k = np.repeat(np.arange(n_periods), L)
    l = np.tile(np.arange(L), n_periods)
    c = np.tile(cosets, n_periods)
    times = k * cfg.T0_ps + c.astype(np.float64) * cfg.eta_ps
    order = np.argsort(times, kind="stable")
    return times[order], l[order], k[order], c[order]


# --------------------------------------------------------------------------
# 布局生成器
# --------------------------------------------------------------------------
def _from_increments(inc: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    inc = enforce_distinct_slots(inc, cfg)
    return increments_to_cosets(inc)


def gen_fixed_fiber(cfg: SystemConfig) -> np.ndarray:
    """等间隔前缀和：各级增量近似均匀。"""
    target = cfg.N0 // (cfg.L + 1)
    inc = np.full(cfg.L, max(cfg.min_slot_gap, target), dtype=np.int64)
    return _from_increments(inc, cfg)


def gen_uniform_microstep(cfg: SystemConfig, frac: float = 0.5) -> np.ndarray:
    base = max(cfg.min_slot_gap, int(frac * cfg.N0 / cfg.L))
    inc = np.full(cfg.L, base, dtype=np.int64)
    return _from_increments(inc, cfg)


def gen_fixed_random(cfg: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    inc = np.empty(cfg.L, dtype=np.int64)
    inc[0] = rng.integers(1, max(2, cfg.dmax_slots // 2))
    for l in range(1, cfg.L):
        inc[l] = rng.integers(cfg.min_slot_gap, max(cfg.min_slot_gap + 1, cfg.dmax_slots))
    return _from_increments(inc, cfg)


def gen_poisson_gap(cfg: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    inc = np.empty(cfg.L, dtype=np.int64)
    inc[0] = max(cfg.min_slot_gap, int(rng.exponential(cfg.N0 / (4 * cfg.L))))
    for l in range(1, cfg.L):
        inc[l] = max(cfg.min_slot_gap, int(rng.exponential(cfg.N0 / (6 * cfg.L))))
    return _from_increments(inc, cfg)


LAYOUT_GENERATORS = {
    "fixed_fiber": lambda cfg, rng: gen_fixed_fiber(cfg),
    "uniform": lambda cfg, rng: gen_uniform_microstep(cfg),
    "fixed_random": lambda cfg, rng: gen_fixed_random(cfg, rng),
    "poisson_gap": lambda cfg, rng: gen_poisson_gap(cfg, rng),
}


def acquisition_ledger(coset_sets, cfg: SystemConfig, T_acq_ps: float) -> dict:
    coset_sets = [np.asarray(c, dtype=np.int64) for c in coset_sets]
    n_steps = len(coset_sets)
    delta_drift = 0.0
    for a, b in zip(coset_sets[:-1], coset_sets[1:]):
        delta_drift += float(np.abs(b - a).sum())
    n_distinct = len({tuple(c.tolist()) for c in coset_sets})
    return dict(
        T_acq_ps=T_acq_ps,
        n_steps=n_steps,
        n_distinct_cosets=n_distinct,
        delta_drift_slots=delta_drift,
    )
