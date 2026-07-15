"""系统配置（对应 计算光采样.md §1 / §9）。

约定（全代码统一）：
- 时间单位 ps；主周期 T0=5000 ps（5 ns）；支路数 L=5。
- 等效时隙宽度 eta_ps = T0_ps / N0（N0=5000 -> 1 ps；N0=1000 -> 5 ps）。
- 级联前缀和布局：τ_l ∈ [0, dmax_ps) ps，陪集 c_l = prefix_sum(inc)/eta。
- 硬约束：事件占不同 1ps 时隙（min_slot_gap=1），无 ns 级 τ_min。
- 读出：PD → THA → 纯量化器（tha_bw_GHz + t_acq_ps）。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    T0_ps: float = 5000.0
    L: int = 5
    N0: int = 5000
    dmax_ps: float = 1000.0          # 每级可编程延迟上界（ps）
    min_slot_gap: int = 1            # 相邻事件至少相隔的时隙数
    f_avg_GHz: float = 1.0
    pd_bw_GHz: float = 1.0
    tha_bw_GHz: float = 1.0
    enob: float = 8.0
    nbits: int = 10
    v_ramp_ps_per_s: float = 1000.0  # EDL bang-bang 满速
    t_acq_ps: float = 50.0           # THA 重采集时间
    tau_reg_ps: float = 0.05         # σ_cnv 正则（ps）
    t_cnv_min_ps: float = 200.0      # 转换窗软约束参考下限
    mzm_slope: float = 0.5           # 小信号 MZM 斜率

    @property
    def eta_ps(self) -> float:
        return self.T0_ps / self.N0

    @property
    def dmax_slots(self) -> int:
        return int(np.floor(self.dmax_ps / self.eta_ps))

    def f_avg_actual_GHz(self) -> float:
        return self.L / self.T0_ps * 1e3

    def summary(self) -> dict:
        return dict(
            T0_ps=self.T0_ps,
            L=self.L,
            N0=self.N0,
            eta_ps=self.eta_ps,
            dmax_ps=self.dmax_ps,
            min_slot_gap=self.min_slot_gap,
            pd_bw_GHz=self.pd_bw_GHz,
            tha_bw_GHz=self.tha_bw_GHz,
            t_acq_ps=self.t_acq_ps,
            f_avg_GHz=self.f_avg_actual_GHz(),
            enob=self.enob,
            nbits=self.nbits,
            v_ramp_ps_per_s=self.v_ramp_ps_per_s,
        )


def cfg_sanity() -> SystemConfig:
    return SystemConfig(N0=1000)


def cfg_main() -> SystemConfig:
    return SystemConfig(N0=5000)
