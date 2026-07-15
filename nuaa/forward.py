"""物理前向观测模型（对应 计算光采样.md §4）。

链路：S_opt 光事件门控 -> MZM -> PD/THA 栅格卷积 -> S_smp strobe 读出 -> 纯量化器。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import SystemConfig
from . import layout as _layout
from . import measurement as _meas
from .strobe import Calibration, sigma_cnv2

Q_E = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class Events:
    y: np.ndarray
    t_opt: np.ndarray
    t_sample: np.ndarray
    t_cnv: np.ndarray
    branch: np.ndarray
    period: np.ndarray
    coset: np.ndarray
    sigma_q2: np.ndarray
    cfg: SystemConfig


def _shot_thermal_noise(i_signal: np.ndarray, cfg: SystemConfig, cal: Calibration,
                        rng: np.random.Generator) -> np.ndarray:
    bw = cfg.pd_bw_GHz * 1e9
    shot_var = 2 * Q_E * np.abs(i_signal) * bw
    thermal_var = 4 * K_B * cal.T_K * bw / cal.RL
    return np.sqrt(shot_var + thermal_var) * rng.standard_normal(i_signal.shape)


def quantize_events(y: np.ndarray, sigma_q2: np.ndarray, cfg: SystemConfig,
                    cal: Calibration, rng: np.random.Generator) -> np.ndarray:
    fs = cal.full_scale
    enob_step = 2 * fs / (2 ** cfg.enob)
    y = y + np.sqrt(sigma_q2 + (enob_step / np.sqrt(12)) ** 2) * rng.standard_normal(y.shape)
    q = 2 * fs / (2 ** cfg.nbits)
    return np.round(np.clip(y, -fs, fs) / q) * q


def forward_observe(
    x_ref: np.ndarray,
    tau_ps: np.ndarray,
    cfg: SystemConfig,
    n_periods: int,
    cal: Optional[Calibration] = None,
    rng: Optional[np.random.Generator] = None,
    add_noise: bool = True,
    normalize_drive: bool = True,
) -> Events:
    rng = rng or np.random.default_rng()
    cal = (cal or Calibration()).ensure(cfg.L, cfg)
    lay = _layout.delays_to_layout(tau_ps, n_periods, cfg, cal=cal)
    op = _meas.build_forward_operator(lay, cfg, cal)
    x_ref = np.asarray(x_ref).reshape(-1)
    if np.iscomplexobj(x_ref):
        x_ref = np.real(x_ref)

    s = x_ref[: op.N_ref] if x_ref.size >= op.N_ref else np.pad(x_ref, (0, op.N_ref - x_ref.size))
    if normalize_drive:
        s = s / (np.max(np.abs(s)) + 1e-12)

    i_smp = op.matvec(s.astype(np.float64))
    if add_noise:
        i_smp = i_smp + _shot_thermal_noise(i_smp, cfg, cal, rng)

    sig2 = sigma_cnv2(lay.t_cnv, cfg, cal)
    y = quantize_events(i_smp, sig2, cfg, cal, rng) if add_noise else i_smp

    coset = np.round(lay.t_opt / cfg.eta_ps).astype(np.int64) % cfg.N0
    return Events(
        y=y,
        t_opt=lay.t_opt,
        t_sample=lay.t_sample,
        t_cnv=lay.t_cnv,
        branch=lay.branch,
        period=lay.period,
        coset=coset,
        sigma_q2=sig2,
        cfg=cfg,
    )


def events_to_Y(ev: Events, cfg: SystemConfig) -> np.ndarray:
    P = int(ev.period.max()) + 1
    Y = np.zeros((cfg.L, P), dtype=ev.y.dtype)
    Y[ev.branch, ev.period] = ev.y
    return Y
