"""THz digital communication waveform + deployment impairments.

Maps 计算光采样.md §太赫兹数字通信 deployment table onto the 1 THz / N0
periodic spectrum grid used by streaming experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from .config import SystemConfig

# mod id in gen_evolving_sequence(thz) -> constellation order
_MOD_ORDER = {0: 2, 1: 4, 2: 16, 3: 64}
_THZ_SCHEDULE = np.array([0, 1, 2, 3, 2, 1], dtype=np.int64)


@dataclass(frozen=True)
class ThzDeployProfile:
    """Deployment impairment bundle (defaults = clean AWGN-only baseline)."""
    evm_pct: float = 0.0
    phase_noise_rad: float = 0.0
    cfo_khz: float = 0.0
    multipath: bool = False
    tau_rms_frac: float = 0.1
    snr_db: float = 5.0
    n_sym: int = 32
    roll_off: float = 0.35

    def label(self) -> str:
        parts = []
        if self.evm_pct > 0:
            parts.append(f"evm{self.evm_pct:g}")
        if self.phase_noise_rad > 0:
            parts.append("pn")
        if self.cfo_khz > 0:
            parts.append(f"cfo{self.cfo_khz:g}k")
        if self.multipath:
            parts.append("mp")
        if self.snr_db != 5.0:
            parts.append(f"snr{self.snr_db:g}")
        return "_".join(parts) if parts else "clean"


DEPLOY_PRESETS: dict[str, ThzDeployProfile] = {
    "clean": ThzDeployProfile(),
    "evm_12": ThzDeployProfile(evm_pct=12.0),
    "evm_25": ThzDeployProfile(evm_pct=25.0),
    "phase_noise": ThzDeployProfile(phase_noise_rad=0.10),
    "cfo_30k": ThzDeployProfile(cfo_khz=30.0),
    "cfo_300k": ThzDeployProfile(cfo_khz=300.0),
    "multipath": ThzDeployProfile(multipath=True, tau_rms_frac=0.1),
    "snr_m10": ThzDeployProfile(snr_db=-10.0),
    "snr_m25": ThzDeployProfile(snr_db=-25.0),
    "full_deploy": ThzDeployProfile(
        evm_pct=18.0,
        phase_noise_rad=0.10,
        cfo_khz=100.0,
        multipath=True,
        tau_rms_frac=0.1,
        snr_db=-25.0,
    ),
}


def qam_symbols(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    m = int(np.sqrt(order))
    levels = np.arange(-m + 1, m, 2, dtype=np.float64)
    if order == 2:
        i = rng.integers(0, 2, size=n)
        return (2 * i - 1).astype(np.complex128)
    i = rng.integers(0, m, size=n)
    q = rng.integers(0, m, size=n)
    sym = levels[i] + 1j * levels[q]
    return (sym / np.sqrt(np.mean(np.abs(sym) ** 2) + 1e-12)).astype(np.complex128)


def apply_evm(symbols: np.ndarray, evm_pct: float,
              rng: np.random.Generator) -> np.ndarray:
    """Inject constellation error to approach target RMS EVM (%)."""
    if evm_pct <= 0:
        return symbols
    target = float(evm_pct) / 100.0
    noise = (rng.standard_normal(symbols.size)
             + 1j * rng.standard_normal(symbols.size)) / np.sqrt(2)
    scale = target * np.linalg.norm(symbols) / (np.linalg.norm(noise) + 1e-12)
    out = symbols + scale * noise
    alpha = np.vdot(out, symbols) / (np.vdot(out, out) + 1e-12)
    err = np.linalg.norm(symbols - alpha * out)
    cur = err / (np.linalg.norm(symbols) + 1e-12)
    if cur > 1e-12:
        out = symbols + (target / cur) * (out - symbols)
    return out.astype(np.complex128)


def _cfo_bins(cfg: SystemConfig, cfo_khz: float) -> int:
    df_hz = 1.0 / (cfg.T0_ps * 1e-12)
    return int(round(float(cfo_khz) * 1e3 / df_hz))


def _multipath_tau_slots(cfg: SystemConfig, profile: ThzDeployProfile) -> float:
    t_sym_ps = cfg.T0_ps / max(1, profile.n_sym)
    tau_ps = profile.tau_rms_frac * t_sym_ps
    return float(tau_ps / cfg.eta_ps)


def apply_thz_impairments(
    x_base: np.ndarray,
    support_base: np.ndarray,
    mod_id: int,
    profile: ThzDeployProfile,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deployment impairments on top of an existing period spectrum."""
    order = _MOD_ORDER.get(int(mod_id), 4)
    sym = qam_symbols(order, profile.n_sym, rng)
    sym = apply_evm(sym, profile.evm_pct, rng)

    x = np.asarray(x_base, dtype=np.complex128).copy()
    sup = np.asarray(support_base, dtype=np.int64).reshape(-1)
    if profile.evm_pct > 0 and sup.size:
        jitter = 1.0 + (profile.evm_pct / 100.0) * (
            rng.standard_normal(sup.size) + 1j * rng.standard_normal(sup.size)
        ) / np.sqrt(2)
        x[sup] *= jitter

    if profile.phase_noise_rad > 0:
        x *= np.exp(1j * float(profile.phase_noise_rad * rng.standard_normal()))

    shift = _cfo_bins(cfg, profile.cfo_khz)
    if shift:
        x = np.roll(x, shift)
        sup = (sup + shift) % cfg.N0

    if profile.multipath:
        tau = _multipath_tau_slots(cfg, profile)
        h0 = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        h1 = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        norm = np.sqrt(abs(h0) ** 2 + abs(h1) ** 2) + 1e-12
        h0, h1 = h0 / norm, h1 / norm
        k = np.arange(cfg.N0, dtype=np.float64)
        H = h0 + h1 * np.exp(-2j * np.pi * k * tau / cfg.N0)
        x = np.fft.ifft(np.fft.fft(x) * H)
        mag = np.abs(x)
        thr = 1e-3 * (mag.max() + 1e-12)
        sup = np.where(mag > thr)[0].astype(np.int64)
        if sup.size == 0:
            sup = np.asarray(support_base, dtype=np.int64)

    return x, sup, sym


def synth_thz_period_spectrum(
    cfg: SystemConfig,
    center: int,
    mod_id: int,
    profile: ThzDeployProfile,
    rng: np.random.Generator,
    *,
    support_width: Optional[int] = None,
    symbol_sequence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-period THz comm spectrum with optional impairments.

    If ``symbol_sequence`` is set, each symbol is placed on one support bin
  (periodic spectrum embedding for constellation-visible reconstruction).

    Returns (x_full, support, tx_symbols).
    """
    n0 = cfg.N0
    order = _MOD_ORDER.get(int(mod_id), 4)
    sym = qam_symbols(order, profile.n_sym, rng)
    sym = apply_evm(sym, profile.evm_pct, rng)
    if symbol_sequence is not None:
        sym = np.asarray(symbol_sequence, dtype=np.complex128).reshape(-1)
        n_sym = min(sym.size, profile.n_sym)
        sym = sym[:n_sym]

    if symbol_sequence is not None:
        width = int(symbol_sequence.size)
    else:
        width = support_width or max(1, int(np.ceil(np.sqrt(order))))
    c = int(np.clip(round(center), 1, n0 - 2))
    lo, hi = max(0, c - width // 2), min(n0, c + width // 2 + 1)
    sup = np.arange(lo, hi, dtype=np.int64)
    x = np.zeros(n0, dtype=np.complex128)
    if symbol_sequence is not None:
        n_place = min(sym.size, sup.size)
        x[sup[:n_place]] = sym[:n_place]
    else:
        blob = np.mean(sym)
        x[sup] = blob * (rng.standard_normal(sup.size)
                         + 1j * rng.standard_normal(sup.size)) / np.sqrt(2)

    if profile.phase_noise_rad > 0:
        phi = float(profile.phase_noise_rad * rng.standard_normal())
        x *= np.exp(1j * phi)

    shift = _cfo_bins(cfg, profile.cfo_khz)
    if shift:
        x = np.roll(x, shift)

    if profile.multipath:
        tau = _multipath_tau_slots(cfg, profile)
        h0 = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        h1 = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        norm = np.sqrt(abs(h0) ** 2 + abs(h1) ** 2) + 1e-12
        h0, h1 = h0 / norm, h1 / norm
        k = np.arange(n0, dtype=np.float64)
        H = h0 + h1 * np.exp(-2j * np.pi * k * tau / n0)
        x = np.fft.ifft(np.fft.fft(x) * H)

    mag = np.abs(x)
    thr = 1e-3 * (mag.max() + 1e-12)
    spread = np.where(mag > thr)[0]
    if spread.size == 0:
        spread = sup
    return x, spread.astype(np.int64), sym


def predict_mod_schedule(true_mod: np.ndarray, r: int, warmup: int) -> int:
    if r < warmup:
        return int(true_mod[r])
    phase0 = None
    for p in range(len(_THZ_SCHEDULE)):
        if all(int(true_mod[i]) == int(_THZ_SCHEDULE[(p + i) % len(_THZ_SCHEDULE)])
               for i in range(warmup)):
            phase0 = p
            break
    if phase0 is None:
        return int(true_mod[r - 1])
    return int(_THZ_SCHEDULE[(phase0 + r) % len(_THZ_SCHEDULE)])


def profile_summary(profile: ThzDeployProfile, cfg: SystemConfig) -> dict:
    d = asdict(profile)
    d["cfo_bins"] = _cfo_bins(cfg, profile.cfo_khz)
    d["multipath_tau_slots"] = _multipath_tau_slots(cfg, profile) if profile.multipath else 0.0
    d["df_mhz"] = 1.0 / (cfg.T0_ps * 1e-12) / 1e6
    return d


def nominal_multipath_tau_slots(cfg: SystemConfig, profile: ThzDeployProfile) -> float:
    """Deployment-table nominal delay (slots) for tap dictionary."""
    if not profile.multipath:
        return 0.0
    return _multipath_tau_slots(cfg, profile)


def build_spec_measurement_columns(
    cosets: np.ndarray,
    atom_bins: np.ndarray,
    N0: int,
) -> np.ndarray:
    """Φ(coset_i, bin_j) rows without forming full N0×N0 spectrum matmul."""
    c = np.asarray(cosets, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(atom_bins, dtype=np.float64).reshape(1, -1) % N0
    return np.exp(2j * np.pi * c @ b / N0) / N0


def tap_coef_to_spectrum(
    coef: np.ndarray,
    meta: list[tuple[int, int]],
    N0: int,
) -> np.ndarray:
    """Scatter tap coefficients onto frequency bins."""
    xhat = np.zeros(N0, dtype=np.complex128)
    for val, (ci, d) in zip(np.asarray(coef).reshape(-1), meta):
        if abs(val) > 0:
            xhat[(int(ci) + int(d)) % N0] += val
    return xhat


def build_multipath_channel_atoms(
    cfg: SystemConfig,
    center_bins: np.ndarray,
    tau_slots: float,
    *,
    blob_width: int = 6,
) -> np.ndarray:
    """Nominal dual-path channel templates: narrow blob at c passed through H(k)."""
    n0 = cfg.N0
    tau = float(tau_slots)
    k = np.arange(n0, dtype=np.float64)
    h0 = h1 = 1.0 / np.sqrt(2.0)
    H = h0 + h1 * np.exp(-2j * np.pi * k * tau / n0)
    half = max(1, int(blob_width) // 2)
    cols = []
    for c in np.asarray(center_bins, dtype=np.int64).reshape(-1):
        e = np.zeros(n0, dtype=np.complex128)
        ci = int(c) % n0
        lo, hi = max(0, ci - half), min(n0, ci + half + 1)
        e[lo:hi] = 1.0 / np.sqrt(max(1, hi - lo))
        cols.append(np.fft.ifft(np.fft.fft(e) * H))
    if not cols:
        return np.zeros((n0, 0), dtype=np.complex128)
    return np.stack(cols, axis=1)


def build_tap_atom_columns(
    cfg: SystemConfig,
    center_bins: np.ndarray,
    tau_slots: float,
    *,
    include_opposite: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Frequency-domain tap atoms: unit pulse at center ± {0, ±tau} delays."""
    cols, meta = [], []
    tau_i = int(round(float(tau_slots)))
    delays = [0]
    if tau_i > 0:
        delays.extend([tau_i, -tau_i] if include_opposite else [tau_i])
    for c in np.asarray(center_bins, dtype=np.int64).reshape(-1):
        e = np.zeros(cfg.N0, dtype=np.complex128)
        ci = int(c) % cfg.N0
        e[ci] = 1.0
        for d in delays:
            cols.append(np.roll(e, d))
            meta.append((ci, d))
    if not cols:
        return np.zeros((cfg.N0, 0), dtype=np.complex128), meta
    return np.stack(cols, axis=1), meta


def effective_period_slice(arr: dict, n_periods: int) -> dict:
    periods = np.asarray(arr.get("period", []), dtype=np.int64)
    if periods.size == 0:
        return arr
    lo = max(0, int(periods.max()) - int(n_periods) + 1)
    keep = periods >= lo
    return {k: np.asarray(v)[keep] for k, v in arr.items()}
