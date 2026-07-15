"""NUAA 延迟策略：布局评分与慢扫描轨迹（对应 计算光采样.md §6）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np

from .config import SystemConfig
from . import layout as _layout
from . import measurement as _measurement
from .kernels import cond_number
from .strobe import Calibration, predict_strobe_from_times


def restricted_coherence(cosets: np.ndarray, support: Sequence[int], N0: int) -> float:
    support = np.asarray(list(support), dtype=np.int64)
    cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
    if support.size < 2 or cosets.size == 0:
        return 0.0
    M = np.exp(2j * np.pi * np.outer(cosets, support) / N0)
    M = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)
    G = np.abs(M.conj().T @ M)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def score_layout(
    tau_ps: np.ndarray,
    cfg: SystemConfig,
    cal: Calibration,
    lambda_cnv: float = 1.0,
    lambda_kappa: float = 1.0,
    delta_nu_eff: float = 0.0,
) -> float:
    """J(τ) ≈ λ_κ log κ(H_PD) + λ_cnv Σ hinge(t_cnv)。

    κ(H_PD) 对密簇布局可达 inf（病态/退化）；钳位到 1e12 以保证 J 有限，使其在策略中
    被强烈惩罚（高但有限代价）而非产生 inf/NaN。
    """
    lay = _layout.delays_to_layout(tau_ps, n_periods=1, cfg=cfg, cal=cal)
    t_sample = predict_strobe_from_times(lay.t_opt, cfg, cal)
    kappa = cond_number(lay.t_opt, t_sample, cfg)
    kappa = min(max(float(kappa), 1.0), 1e12)              # 钳位，避免 inf
    cnv_pen = float(np.sum(np.maximum(0.0, cfg.t_cnv_min_ps - lay.t_cnv)))
    relax = 1.0 + 0.5 * max(0.0, delta_nu_eff)
    return lambda_kappa * relax * float(np.log(kappa)) + lambda_cnv * cnv_pen


def select_next_coset_set(
    accumulated: Sequence[np.ndarray],
    support_belief: Sequence[int],
    cfg: SystemConfig,
    rng: np.random.Generator,
    cal: Optional[Calibration] = None,
    n_cand: int = 64,
    n_passes: int = 2,
) -> np.ndarray:
    cal = cal or Calibration().ensure(cfg.L, cfg)
    acc = (
        np.concatenate([np.asarray(c, np.int64).reshape(-1) for c in accumulated])
        if accumulated
        else np.zeros(0, dtype=np.int64)
    )
    new = _layout.gen_fixed_random(cfg, rng)
    if len(support_belief) < 2:
        return new
    for _ in range(n_passes):
        for l in range(cfg.L):
            best_mu, best_c = np.inf, new[l]
            for _ in range(n_cand):
                trial = new.copy()
                trial[l] = rng.integers(1, cfg.N0 // cfg.L)
                if not _layout.validate_layout(trial, cfg):
                    continue
                mu = restricted_coherence(np.concatenate([acc, trial]), support_belief, cfg.N0)
                if mu < best_mu:
                    best_mu, best_c = mu, trial[l]
            new[l] = best_c
        inc = _layout.cosets_to_increments(new)
        inc = _layout.enforce_distinct_slots(inc, cfg)
        new = _layout.increments_to_cosets(inc)
    return new


def offline_learned_layout(
    cfg: SystemConfig,
    candidate_support: Sequence[int],
    rng: np.random.Generator,
    cal: Optional[Calibration] = None,
    n_restarts: int = 8,
    n_cand: int = 64,
) -> np.ndarray:
    cal = cal or Calibration().ensure(cfg.L, cfg)
    best_mu, best = np.inf, None
    for _ in range(n_restarts):
        cs = select_next_coset_set([], candidate_support, cfg, rng, cal=cal, n_cand=n_cand)
        mu = restricted_coherence(cs, candidate_support, cfg.N0)
        if mu < best_mu:
            best_mu, best = mu, cs
    return best


def nuaa_scan_trajectory(
    cfg: SystemConfig,
    support_belief: Sequence[int],
    n_steps: int,
    rng: np.random.Generator,
    cal: Optional[Calibration] = None,
    n_cand: int = 64,
) -> List[np.ndarray]:
    cal = cal or Calibration().ensure(cfg.L, cfg)
    acc: List[np.ndarray] = []
    for _ in range(n_steps):
        nxt = select_next_coset_set(acc, support_belief, cfg, rng, cal=cal, n_cand=n_cand)
        acc.append(nxt)
    return acc


def random_scan_trajectory(cfg: SystemConfig, n_steps: int,
                           rng: np.random.Generator) -> List[np.ndarray]:
    return [_layout.gen_fixed_random(cfg, rng) for _ in range(n_steps)]


def static_hold_trajectory(coset_set: np.ndarray, n_steps: int) -> List[np.ndarray]:
    return [np.asarray(coset_set, np.int64).copy() for _ in range(n_steps)]


# --------------------------------------------------------------------------
# 动态衰减先验（在线积累 + 遗忘）：对应 计算光采样.md §在线先验积累的时间增益
# --------------------------------------------------------------------------
class BeliefPrior:
    """1 ps 等效栅格（N0 个频率箱，η=T0/N0=1 ps@N0=5000）上的支撑信念 π。

    随每步采样用重构后验做**乘性贝叶斯更新**，并在每步更新前以遗忘因子 γ 向带内均匀分布
    **泄漏（衰减）**——既让先验随采样过程动态收紧，又避免早期（强干扰折叠下）的错误信念被
    正反馈锁死（对应文档「安全回退」与「错误先验若被正反馈强化会发散」的诚实边界）。

    - decay(): π ← γ·π + (1-γ)·u_band      （γ<1 遗忘；γ=1 不遗忘，作消融）
    - update(evidence): π ← π·(evidence+floor) 归一化  （动态更新）
    - candidates(m): 取信念最高的 m 个箱作动态候选集（替代静态候选集 C）
    """

    def __init__(self, N0: int, gamma: float = 0.85, floor: float = 1e-4,
                 guard: Optional[int] = None):
        self.N0 = int(N0)
        self.gamma = float(gamma)
        self.floor = float(floor)
        self.guard = int(guard if guard is not None else max(1, N0 // 50))
        lo, hi = self.guard, self.N0 - self.guard
        self._u = np.zeros(self.N0)
        self._u[lo:hi] = 1.0 / max(1, hi - lo)         # 带内均匀（盲冷启动）
        self.pi = self._u.copy()

    def _norm(self):
        s = float(self.pi.sum())
        if s <= 0 or not np.isfinite(s):
            self.pi = self._u.copy()
        else:
            self.pi = self.pi / s

    def decay(self):
        """向带内均匀分布泄漏：遗忘旧证据，保证先验不被早期错误锁死。"""
        self.pi = self.gamma * self.pi + (1.0 - self.gamma) * self._u
        self._norm()

    def update(self, evidence: np.ndarray, strength: float = 1.0):
        """乘性更新：evidence 为各箱的归一化似然（如后验相关能量）。"""
        ev = np.asarray(evidence, float).reshape(-1)
        ev = np.clip(ev / (ev.max() + 1e-12), 0.0, 1.0)
        self.pi = self.pi * np.power(ev + self.floor, float(strength))
        self._norm()

    def candidates(self, m: int) -> np.ndarray:
        """信念最高的 m 个箱（升序索引），作为本步重构/布局的动态候选集。"""
        m = int(min(max(1, m), self.N0 - 2 * self.guard))
        idx = np.argpartition(self.pi, -m)[-m:]
        return np.sort(idx.astype(np.int64))

    def entropy(self) -> float:
        p = self.pi[self.pi > 0]
        return float(-np.sum(p * np.log(p)))

    def topm_mass(self, m: int) -> float:
        """信念最高 m 个箱占的总概率质量∈[0,1]：作为**可靠性/集中度**度量。

        均匀（盲）时 ≈ m/带宽（很小）；信念收紧到真支撑邻域后趋于 1。用于门控
        「仅在信念可靠后才启用布局自适应」——早期（强干扰折叠下信念不可靠）保持随机
        布局以维持宽带 RIP，集中度达标后再做信念驱动的有针对性精炼。
        """
        m = int(min(max(1, m), self.N0))
        return float(np.sort(self.pi)[-m:].sum())


def select_next_coset_belief(
    accumulated: Sequence[np.ndarray],
    belief: BeliefPrior,
    cfg: SystemConfig,
    rng: np.random.Generator,
    top_m: int = 40,
    cal: Optional[Calibration] = None,
    n_cand: int = 48,
    n_passes: int = 2,
) -> np.ndarray:
    """据当前动态信念选下一步布局：把信念最高的 top_m 个箱作为待判别支撑，
    最小化（已累积 ∪ 新布局）在这些混淆箱上的受限互相干——即把测量多样性
    聚焦到「当前最该被区分」的频率，而非已确定支撑或全频带盲随机。
    """
    return select_next_coset_set(
        accumulated, belief.candidates(top_m), cfg, rng,
        cal=cal, n_cand=n_cand, n_passes=n_passes)


@dataclass
class DelayCommand:
    """One slow EDL command produced by the streaming controller."""

    tau_ps: np.ndarray
    cosets: np.ndarray
    velocity_ps_per_s: np.ndarray
    desired_cosets: np.ndarray
    adaptive_enabled: bool
    movement_ps: float
    dt_s: float
    reliability: float


class BangBangDelayPlanner:
    """Slow trajectory planner for EDL-limited streaming NUAA.

    The planner separates a fast belief/reconstruction loop from the slow
    mechanical loop. It proposes an information-oriented target layout, then
    projects it through the bang-bang slew constraint so sub-ms control slices
    may legitimately keep the optical delays frozen.
    """

    def __init__(
        self,
        cfg: SystemConfig,
        cal: Optional[Calibration] = None,
        top_m: int = 40,
        conf_thr: float = 0.02,
        n_cand: int = 48,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg
        self.cal = cal or Calibration().ensure(cfg.L, cfg)
        self.top_m = int(top_m)
        self.conf_thr = float(conf_thr)
        self.n_cand = int(n_cand)
        self.rng = rng or np.random.default_rng()

    def choose_target(
        self,
        accumulated: Sequence[np.ndarray],
        belief: Optional[BeliefPrior],
        mode: str = "belief_bangbang",
    ) -> tuple[np.ndarray, bool, float]:
        if mode == "static_hold" and accumulated:
            return np.asarray(accumulated[-1], dtype=np.int64), False, 0.0
        if mode == "oracle_bangbang" and belief is not None:
            cs = select_next_coset_belief(
                accumulated, belief, self.cfg, self.rng,
                top_m=self.top_m, cal=self.cal, n_cand=self.n_cand)
            return cs, True, 1.0
        if mode == "belief_bangbang" and belief is not None:
            reliability = belief.topm_mass(self.top_m)
            if reliability >= self.conf_thr and accumulated:
                cs = select_next_coset_belief(
                    accumulated, belief, self.cfg, self.rng,
                    top_m=self.top_m, cal=self.cal, n_cand=self.n_cand)
                return cs, True, reliability
            return _layout.gen_fixed_random(self.cfg, self.rng), False, reliability
        return _layout.gen_fixed_random(self.cfg, self.rng), False, 0.0

    def step(
        self,
        prev_tau_ps: np.ndarray,
        dt_s: float,
        accumulated: Sequence[np.ndarray],
        belief: Optional[BeliefPrior],
        mode: str = "belief_bangbang",
    ) -> DelayCommand:
        desired, adaptive, reliability = self.choose_target(accumulated, belief, mode=mode)
        tau = _measurement.matrix_to_delays(
            desired, self.cfg, prev_tau_ps=prev_tau_ps, dt_s=dt_s)
        cosets = _layout.tau_to_cosets(tau, self.cfg)
        delta = tau - np.asarray(prev_tau_ps, dtype=np.float64)
        velocity = delta / max(float(dt_s), 1e-12)
        velocity = np.clip(velocity, -self.cfg.v_ramp_ps_per_s, self.cfg.v_ramp_ps_per_s)
        return DelayCommand(
            tau_ps=tau,
            cosets=cosets,
            velocity_ps_per_s=velocity,
            desired_cosets=np.asarray(desired, dtype=np.int64),
            adaptive_enabled=bool(adaptive),
            movement_ps=float(np.sum(np.abs(delta))),
            dt_s=float(dt_s),
            reliability=float(reliability),
        )
