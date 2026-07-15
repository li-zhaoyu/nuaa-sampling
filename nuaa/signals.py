"""待采样电磁信号生成（对应 计算光采样.md §8）。

主打 E1：稀疏多频带 + 带内强干扰（在频域/多陪集模型下最能体现「布局自适应」优势）。
另含 E2 LFM、E3 跳频 波形（时域），供波形重构与 NUAA-MU 演示。

实/复约定：E1 采用复多陪集模型（标准多陪集/MWC 文献口径，等价理想 I/Q 前端），
以隔离「布局」对可分辨性的独立影响（见 §8 与 §0 第 2 条说明）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import SystemConfig


# --------------------------------------------------------------------------
# E1：稀疏多频带 + 带内强干扰（频域多陪集模型）
# --------------------------------------------------------------------------
@dataclass
class MultibandSpectrum:
    X: np.ndarray             # (N0, J) 复谱（含干扰），多快照（MMV）
    support: np.ndarray       # (K,) 活跃信号子带（不含干扰）
    jammer_idx: int           # 干扰所在子带（-1 表示无）
    cfg: SystemConfig
    sir_db: float


def _choose_support(N0: int, K: int, rng: np.random.Generator,
                    min_sep: int, guard: int) -> np.ndarray:
    """在 [guard, N0-guard) 内取 K 个间隔 ≥ min_sep 的子带索引。"""
    for _ in range(1000):
        cand = np.sort(rng.choice(np.arange(guard, N0 - guard), size=K, replace=False))
        if K == 1 or np.all(np.diff(cand) >= min_sep):
            return cand
    return cand  # 退化时直接返回


def gen_multiband_spectrum(cfg: SystemConfig, K: int = 3, J: int = 8,
                           rng: Optional[np.random.Generator] = None,
                           jammer: bool = True, sir_db: float = -40.0,
                           min_sep: Optional[int] = None,
                           jammer_adjacent: bool = True) -> MultibandSpectrum:
    """生成 K 子带多频带谱 + 1 个 −SIR dB 强干扰。

    - 信号子带功率归一为 1；干扰功率 = 10^(-sir_db/10)（SIR=-40 -> 干扰强 1e4 倍）。
    - jammer_adjacent=True 时把干扰放在某信号子带邻近，制造最难的折叠/混叠。
    """
    rng = rng or np.random.default_rng()
    N0 = cfg.N0
    min_sep = min_sep or max(2, N0 // (4 * max(K, 1)))
    guard = max(1, N0 // 50)
    support = _choose_support(N0, K, rng, min_sep, guard)

    X = np.zeros((N0, J), dtype=np.complex128)
    # 信号子带：单位功率复高斯（每快照独立幅度 -> 多频带随机过程）
    X[support, :] = (rng.standard_normal((K, J)) + 1j * rng.standard_normal((K, J))) / np.sqrt(2)

    jammer_idx = -1
    if jammer:
        jam_pow = 10.0 ** (-sir_db / 10.0)         # SIR=-40 -> 1e4
        if jammer_adjacent:
            base = int(rng.choice(support))
            off = int(rng.choice([-1, 1])) * max(1, min_sep // 3)
            jammer_idx = int(np.clip(base + off, guard, N0 - guard - 1))
            while jammer_idx in support:
                jammer_idx = int(np.clip(jammer_idx + 1, guard, N0 - guard - 1))
        else:
            jammer_idx = int(rng.choice([i for i in range(guard, N0 - guard) if i not in support]))
        X[jammer_idx, :] = np.sqrt(jam_pow) * (
            rng.standard_normal(J) + 1j * rng.standard_normal(J)) / np.sqrt(2)

    return MultibandSpectrum(X=X, support=support, jammer_idx=jammer_idx, cfg=cfg, sir_db=sir_db)


def add_measurement_noise(Y_clean: np.ndarray, snr_db: float,
                          ref_power: float, rng: np.random.Generator) -> np.ndarray:
    """按相对信号参考功率 ref_power 的 SNR 加复高斯测量噪声。"""
    noise_pow = ref_power * 10.0 ** (-snr_db / 10.0)
    std = np.sqrt(noise_pow)
    n = (std / np.sqrt(2)) * (rng.standard_normal(Y_clean.shape) + 1j * rng.standard_normal(Y_clean.shape))
    return Y_clean + n


# --------------------------------------------------------------------------
# E2：LFM 雷达（时域复基带波形，1-ps 等效栅格）
# --------------------------------------------------------------------------
def gen_lfm_waveform(cfg: SystemConfig, n_periods: int,
                     f0_norm: float = 0.02, k_norm: Optional[float] = None,
                     rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, dict]:
    """生成复基带 LFM chirp： x[n]=exp(j 2π (f0 n + 0.5 k n^2))，归一化频率（cycles/slot）。

    f0_norm, k_norm 为相对等效栅格（1 THz）的归一化频率/调频率。
    返回 x_ref(C^{N_ref}) 与参数字典（供 (f0,k) 误差评估）。
    """
    rng = rng or np.random.default_rng()
    N_ref = cfg.N0 * n_periods
    n = np.arange(N_ref)
    if k_norm is None:
        k_norm = 0.6 * f0_norm / N_ref            # 全窗扫过 ~0.6*f0
    phase = 2 * np.pi * (f0_norm * n + 0.5 * k_norm * n ** 2)
    x = np.exp(1j * phase)
    return x.astype(np.complex128), dict(f0_norm=f0_norm, k_norm=k_norm, N_ref=N_ref)


# --------------------------------------------------------------------------
# E5：宽带 LFM 雷达（20–40 GHz，1 µs 脉宽，映射至 1 THz 等效栅格）
# --------------------------------------------------------------------------
# 物理口径：f_norm = f_GHz × η_ps（η=1 ps → 20 GHz↦0.02）；BT≈2×10^4≈43 dB 处理增益。
RADAR_F_LO_GHZ = 20.0
RADAR_F_HI_GHZ = 40.0
RADAR_PULSE_US = 1.0
RADAR_BT_DB = 10.0 * np.log10((RADAR_F_HI_GHZ - RADAR_F_LO_GHZ) * 1e9 * RADAR_PULSE_US * 1e-6)


def ghz_to_norm(f_ghz: float) -> float:
    """物理频率 (GHz) → 归一化频率 (cycles/slot)，η=1 ps。"""
    return float(f_ghz) * 1e-3


def norm_to_ghz(f_norm: float) -> float:
    return float(f_norm) * 1e3


def radar_n_periods(cfg: SystemConfig, pulse_us: float = RADAR_PULSE_US) -> int:
    """1 µs 脉宽对应的主周期数（T0=5 ns）。"""
    return max(1, int(round(pulse_us * 1e-6 / (cfg.T0_ps * 1e-12))))


@dataclass
class RadarLFMPulse:
    """宽带 LFM 雷达脉冲（时域 + chirplet 稀疏表示）。"""
    x: np.ndarray              # (N_ref,) 时域 chirp
    atom_idx: int              # 在 chirplet 字典中的真值列索引
    f0_norm: float
    k_norm: float
    f0_ghz: float
    k_ghz_per_s: float
    C: np.ndarray              # 候选列索引 (nC,)
    f0_tab: np.ndarray         # (n_atoms,) 字典各列 f0_norm
    k_tab: np.ndarray          # (n_atoms,) 字典各列 k_norm
    params: dict
    cfg: SystemConfig


def chirplet_atom(N: int, f0_norm: float, k_norm: float) -> np.ndarray:
    """单位幅度 chirplet：exp(j2π(f0·n + ½k·n²))，n=0..N-1。"""
    n = np.arange(N, dtype=np.float64)
    return np.exp(1j * 2 * np.pi * (f0_norm * n + 0.5 * k_norm * n ** 2)).astype(np.complex128)


def chirplet_at_times(time_idx: np.ndarray, f0_norm: float, k_norm: float) -> np.ndarray:
    """在指定时刻索引上求 chirplet 值（避免物化全长字典）。"""
    t = np.asarray(time_idx, dtype=np.float64).reshape(-1)
    return np.exp(1j * 2 * np.pi * (f0_norm * t + 0.5 * k_norm * t ** 2)).astype(np.complex128)


def build_chirplet_grids(
    f_lo_ghz: float, f_hi_ghz: float, n_f0: int, n_k: int,
    k_ref_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (f0_norm_grid, k_norm_grid) 用于字典铺格。"""
    margin = ghz_to_norm(2.0)
    f0_grid = np.linspace(ghz_to_norm(f_lo_ghz) - margin,
                          ghz_to_norm(f_hi_ghz) + margin, n_f0)
    k_span = 1.5 * abs(k_ref_norm)
    k_grid = np.linspace(max(k_ref_norm - k_span, 0), k_ref_norm + k_span, n_k)
    return f0_grid, k_grid


def build_chirplet_dictionary(
    N_ref: int,
    f0_norm_grid: np.ndarray,
    k_norm_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 chirplet 字典 Ψ:(N_ref, n_atoms) 及 (f0,k) 参数表。

    返回 (Psi, f0_table, k_table)，f0_table/k_table 长度 n_atoms。
    注：N_ref 较大时优先用 chirplet_at_times 按需取样。
    """
    f0g = np.asarray(f0_norm_grid, dtype=np.float64).reshape(-1)
    kg = np.asarray(k_norm_grid, dtype=np.float64).reshape(-1)
    cols, f0s, ks = [], [], []
    for f0 in f0g:
        for k in kg:
            cols.append(chirplet_atom(N_ref, f0, k))
            f0s.append(f0)
            ks.append(k)
    Psi = np.column_stack(cols).astype(np.complex128)
    return Psi, np.asarray(f0s), np.asarray(ks)


def build_chirplet_meas_matrix(
    time_idx: np.ndarray,
    C: np.ndarray,
    f0_tab: np.ndarray,
    k_tab: np.ndarray,
) -> np.ndarray:
    """A:(M, nC) 在时刻 time_idx 上取候选 chirplet 列（按需生成，O(M·nC)）。"""
    M, nC = len(time_idx), len(C)
    A = np.empty((M, nC), dtype=np.complex128)
    for j, col in enumerate(C):
        A[:, j] = chirplet_at_times(time_idx, float(f0_tab[col]), float(k_tab[col]))
    return A


def gen_radar_lfm_pulse(
    cfg: SystemConfig,
    n_periods: Optional[int] = None,
    f_lo_ghz: float = RADAR_F_LO_GHZ,
    f_hi_ghz: float = RADAR_F_HI_GHZ,
    n_f0: int = 12,
    n_k: int = 8,
    nC: int = 48,
    rng: Optional[np.random.Generator] = None,
) -> RadarLFMPulse:
    """生成 20–40 GHz 宽带 LFM 脉冲 + chirplet 字典与先验候选集 C。

    - 真值 chirp 扫过 [f_lo, f_hi] GHz；字典在略宽邻域均匀铺格。
    - C 为 nC 个候选列（先验带宽），必含真值列。
    """
    rng = rng or np.random.default_rng()
    n_periods = n_periods or radar_n_periods(cfg)
    N_ref = cfg.N0 * n_periods
    f0_true = ghz_to_norm(rng.uniform(f_lo_ghz, f_hi_ghz - 0.5 * (f_hi_ghz - f_lo_ghz) / max(n_f0, 1)))
    f1_true = f0_true + ghz_to_norm(f_hi_ghz - f_lo_ghz)
    k_true = (f1_true - f0_true) / max(N_ref - 1, 1)

    margin = ghz_to_norm(2.0)
    f0_grid, k_grid = build_chirplet_grids(f_lo_ghz, f_hi_ghz, n_f0, n_k, k_true)
    f0_tab = np.repeat(f0_grid, len(k_grid))
    k_tab = np.tile(k_grid, len(f0_grid))
    n_atoms = f0_tab.size

    dist = (f0_tab - f0_true) ** 2 + ((k_tab - k_true) / (abs(k_true) + 1e-30)) ** 2
    atom_idx = int(np.argmin(dist))
    x = chirplet_atom(N_ref, float(f0_tab[atom_idx]), float(k_tab[atom_idx]))

    guard = max(2, n_atoms // 20)
    band_mask = (f0_tab >= ghz_to_norm(f_lo_ghz - margin)) & (f0_tab <= ghz_to_norm(f_hi_ghz + margin))
    band_idx = np.where(band_mask)[0]
    if len(band_idx) < nC:
        band_idx = np.arange(n_atoms)
    rest = np.setdiff1d(band_idx, [atom_idx])
    n_pick = min(nC - 1, len(rest))
    pick = rng.choice(rest, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
    C = np.sort(np.concatenate([[atom_idx], pick]))

    f0_ghz = norm_to_ghz(f0_true)
    k_ghz = k_true / (cfg.eta_ps * 1e-12) / 1e9                    # GHz/s
    params = dict(
        N_ref=N_ref, n_periods=n_periods, n_atoms=n_atoms,
        f_lo_ghz=f_lo_ghz, f_hi_ghz=f_hi_ghz, BT_db=float(RADAR_BT_DB),
        pulse_us=RADAR_PULSE_US,
    )
    return RadarLFMPulse(
        x=x, atom_idx=atom_idx, f0_norm=float(f0_true), k_norm=float(k_true),
        f0_ghz=f0_ghz, k_ghz_per_s=float(k_ghz), C=C,
        f0_tab=f0_tab, k_tab=k_tab,
        params=params, cfg=cfg,
    )


def matched_filter_peak_snr(x_hat: np.ndarray, x_ref: np.ndarray) -> float:
    """匹配滤波峰信噪比增益 (dB)：|⟨x̂,x⟩|² / ||x̂-αx||²，α 最优复标度。"""
    x_ref = x_ref.reshape(-1)
    x_hat = x_hat.reshape(-1)
    alpha = np.vdot(x_hat, x_ref) / (np.vdot(x_hat, x_hat) + 1e-30)
    target = alpha * x_ref
    noise = x_hat - target
    return float(10.0 * np.log10(
        (np.vdot(target, target).real + 1e-30) / (np.vdot(noise, noise).real + 1e-30)))


def estimate_chirp_params(x_hat: np.ndarray, Psi: np.ndarray,
                          f0_tab: np.ndarray, k_tab: np.ndarray) -> Tuple[float, float, int]:
    """在字典上取最大投影列，返回 (f0_ghz, k_ghz_per_s, atom_idx)。"""
    proj = np.abs(Psi.conj().T @ x_hat.reshape(-1))
    j = int(np.argmax(proj))
    eta = Psi.shape[0] // max(1, int(round(RADAR_PULSE_US * 1e-6 / (5e-9))))  # fallback
    # 用全局 N_ref 与 eta_ps 反推物理调频率
    cfg_eta = 1e-12                                           # 1 ps
    f0_ghz = norm_to_ghz(float(f0_tab[j]))
    k_ghz = float(k_tab[j]) / cfg_eta / 1e9
    return f0_ghz, k_ghz, j


# --------------------------------------------------------------------------
# E3：跳频通信（时域复基带波形）
# --------------------------------------------------------------------------
def gen_fh_waveform(cfg: SystemConfig, n_periods: int, n_hops: int = 8,
                    freqs_norm: Optional[np.ndarray] = None,
                    rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, dict]:
    """生成跳频复基带波形：均分为 n_hops 段，每段一个载频（归一化）+ BPSK 符号。"""
    rng = rng or np.random.default_rng()
    N_ref = cfg.N0 * n_periods
    if freqs_norm is None:
        freqs_norm = rng.uniform(0.01, 0.2, size=n_hops)
    seg = N_ref // n_hops
    x = np.zeros(N_ref, dtype=np.complex128)
    bounds = []
    for h in range(n_hops):
        a, b = h * seg, (h + 1) * seg if h < n_hops - 1 else N_ref
        n = np.arange(b - a)
        sym = rng.choice([-1.0, 1.0])             # 简化：整段 BPSK 符号
        x[a:b] = sym * np.exp(1j * 2 * np.pi * freqs_norm[h] * n)
        bounds.append(a)
    return x, dict(hop_bounds=np.array(bounds), freqs_norm=freqs_norm, n_hops=n_hops, N_ref=N_ref)


# --------------------------------------------------------------------------
# E4：准周期演化信号序列（对应 计算光采样.md §准周期演化信号族 / §两级先验 / §8 E4）
# --------------------------------------------------------------------------
@dataclass
class EvolvingSequence:
    """一段沿窗序列 r=0..W-1 缓慢演化的准周期信号。

    - X_seq[r]: (N0, J) 复谱快照（多陪集口径，单位功率活跃子带）。
    - support_seq[r]: 该窗活跃子带索引（不含干扰）。
    - center_seq: (W,) 主带中心索引轨迹（潜参数 θ_r，演化预测的目标）。
    - mod_seq: (W,) 调制阶数 id（THz 调度用；其余为 0）。
    - kind: 'lfm' | 'fh' | 'thz'。
    - predictability: 'pred'(可预测) | 'semi'(半随机) | 'random'(随机对照)。
    """
    X_seq: list
    support_seq: list
    center_seq: np.ndarray
    mod_seq: np.ndarray
    kind: str
    predictability: str
    cfg: SystemConfig


_PRED_SIGMA = {"pred": 0.0, "semi": 1.0, "random": None}   # None=完全随机
# THz 调制调度：阶数 -> 活跃子带数（BPSK/QPSK=1, 16QAM=2, 64QAM=3）
_THZ_SCHEDULE = np.array([0, 1, 2, 3, 2, 1])               # 一个调度周期内的 mod id
_THZ_K = {0: 1, 1: 1, 2: 2, 3: 3}
# 跳频「跳图样」为**类级共享常量**（标准化跳频码）：按场景预训练才能学到，
# 容量越大越能拟合其周期形状（与逐序列随机的载频/漂移区分开）。
_FH_PRN_PERIOD = 6
_FH_PRN_PATTERN = np.random.default_rng(20240601).standard_normal(_FH_PRN_PERIOD)
_FH_PRN_PATTERN -= _FH_PRN_PATTERN.mean()
# 长结构跳频码（period=24）：复杂度远高于 period-6，**小模型无法记忆其周期形状、大模型可以**
# —— 用于「复杂结构上发挥参数量优势」的容量扫描（gen_evolving_sequence kind='fh_long'）。
_FH_LONG_PERIOD = 24
_FH_LONG_PATTERN = np.random.default_rng(20240777).standard_normal(_FH_LONG_PERIOD)
_FH_LONG_PATTERN -= _FH_LONG_PATTERN.mean()
# lfm_nl 多谐波类级常量：3 个类级共享频率（逐序列仅随机幅度/相位/整体符号），
# 需要足够容量同时拟合多个频率分量（线性/二阶漂移则小模型即可，故容量在此分化）。
_NL_FREQS = 2.0 * np.pi * np.array([0.07, 0.19, 0.34])     # cycles/window 的类级频率
_NL_AMPW = np.array([0.13, 0.08, 0.045])                   # 相对 span 的谐波幅度权重


def _fill_bands(N0: int, J: int, support: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((N0, J), dtype=np.complex128)
    K = len(support)
    if K:
        X[support, :] = (rng.standard_normal((K, J)) + 1j * rng.standard_normal((K, J))) / np.sqrt(2)
    return X


def gen_evolving_sequence(cfg: SystemConfig, kind: str, W: int, J: int = 8,
                          predictability: str = "pred",
                          drift_scale: float = 1.0,
                          epsilon_frac: float = 0.0,
                          rng: Optional[np.random.Generator] = None) -> EvolvingSequence:
    """生成准周期演化信号序列（统一在多陪集谱口径下，便于复用 SOMP/U_jam）。

    三类演化律（窗间低维、可预测）：
      - 'lfm' : chirp-rate 递变 → 主带中心二阶漂移 c_r=c0+v r+0.5 a r^2。
      - 'fh'  : 跳频整体频点漂移 → 中心线性漂移 c_r=c0+drift r，叠加已知伪随机图样。
      - 'thz' : 调制格式调度切换 → 活跃子带数 K_r 按已知调度周期变化。
    predictability 控制演化轨迹叠加的随机扰动（'random' 退化为不可预测对照）。
    drift_scale 缩放每窗漂移幅度（用于漂移–候选宽度比扫描）。
    epsilon_frac 每窗不可约新息 ε_r 的标准差（占带宽 span 的比例，与可学规律无关）。
    """
    rng = rng or np.random.default_rng()
    N0 = cfg.N0
    guard = max(2, N0 // 50)
    lo, hi = guard, N0 - guard
    span = hi - lo
    sigma = _PRED_SIGMA.get(predictability, 0.0)

    def reflect(idx):
        """把 idx 反射到 [lo,hi)，避免绕回造成的模数不连续（利于演化外推）。"""
        p = span
        x = (int(round(idx)) - lo) % (2 * p)
        if x >= p:
            x = 2 * p - 1 - x
        return lo + x

    center = np.zeros(W, dtype=np.int64)
    mod_seq = np.zeros(W, dtype=np.int64)
    support_seq, X_seq = [], []
    rr = np.arange(W)
    unit = span / 100.0                       # 1% 带宽为单位
    # 关键尺度：每窗漂移幅度 ≫ 候选半宽（默认 ~3% 带宽）→ 持续性必然漏、可学外推能中
    base_drift = 6.0 * unit                   # ≈6% 带宽/窗

    if kind == "lfm":
        # chirp-rate 递变 → 增量随窗线性增长（位置二阶）：inc_r = v + a r
        c0 = rng.uniform(0.10 * span, 0.25 * span)
        sgn = rng.choice([-1, 1])
        v = sgn * (3.0 * unit + rng.uniform(0, 1.0) * unit)
        a = sgn * (5.0 * unit / max(W - 1, 1))                    # 增量从 ~3% 增到 ~8%/窗
        inc = v + a * rr
    elif kind == "fh":
        # 跳频整体频点漂移 → 线性漂移 + 类级共享跳图样（标准化跳频码，需更大容量学）
        c0 = rng.uniform(0.15 * span, 0.35 * span)
        drift = rng.choice([-1, 1]) * base_drift
        prn = 2.5 * unit * _FH_PRN_PATTERN[rr % _FH_PRN_PERIOD]
        inc = drift + prn
    elif kind == "thz":
        # 调制格式调度切换 + 载频线性漂移（中心线性，K 按调度周期变）
        c0 = rng.uniform(0.15 * span, 0.35 * span)
        drift = rng.choice([-1, 1]) * (5.0 * unit)
        phase0 = int(rng.integers(0, len(_THZ_SCHEDULE)))
        inc = np.full(W, drift, dtype=np.float64)
    elif kind == "lfm_nl":
        # 【复杂结构】非线性多谐波中心轨迹：c_r = base + Σ_k A_k·span·sin(ω_k r + φ_k)
        # 频率为类级共享常量、逐序列随机幅度/相位/符号 → 需较大容量同时拟合多分量。
        base = lo + rng.uniform(0.4 * span, 0.6 * span)
        sgn = rng.choice([-1, 1])
        amp = _NL_AMPW * span * (0.7 + 0.6 * rng.random(_NL_FREQS.size))
        ph = rng.uniform(0, 2 * np.pi, _NL_FREQS.size)
        harmonics = sum(amp[k] * np.sin(_NL_FREQS[k] * rr + ph[k]) for k in range(_NL_FREQS.size))
        path = base + sgn * float(drift_scale) * harmonics
        inc = None
    elif kind == "fh_long":
        # 【复杂结构】线性漂移 + period-24 长结构码（高复杂度，需大容量记忆周期形状）
        c0 = rng.uniform(0.15 * span, 0.35 * span)
        drift = rng.choice([-1, 1]) * base_drift
        prn = 3.0 * unit * _FH_LONG_PATTERN[rr % _FH_LONG_PERIOD]
        inc = drift + prn
    else:
        raise ValueError(f"unknown kind {kind}")

    if inc is not None:                       # 增量式：位置 = 初值 + 增量累积
        inc = inc * float(drift_scale)
        path = c0 + np.cumsum(inc)
    # 否则（lfm_nl）path 已直接给定为绝对轨迹

    for r in range(W):
        if predictability == "random":
            cen = lo + int(rng.integers(0, span))
        else:
            jit = sigma * rng.standard_normal() * unit
            eps = float(epsilon_frac) * span * rng.standard_normal()   # 不可约新息 ε_r
            cen = reflect(path[r] + jit + eps)
        center[r] = cen
        if kind == "thz":
            # 调制阶数按调度切换（记入 mod_seq 作可预测辅助标签）；
            # 重构以单一主载频（K=1）隔离「载频演化预测」收益，避免多子带分辨成为瓶颈。
            mid = int(_THZ_SCHEDULE[(phase0 + r) % len(_THZ_SCHEDULE)])
            if predictability == "random":
                mid = int(rng.integers(0, 4))
            mod_seq[r] = mid
        sup = np.array([cen], dtype=np.int64)
        support_seq.append(sup)
        X_seq.append(_fill_bands(N0, J, sup, rng))

    return EvolvingSequence(X_seq=X_seq, support_seq=support_seq, center_seq=center,
                            mod_seq=mod_seq, kind=kind, predictability=predictability, cfg=cfg)


# --------------------------------------------------------------------------
# 波形 <-> 谱（按周期）的辅助
# --------------------------------------------------------------------------
def waveform_to_period_spectra(x_ref: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """把 N_ref 波形按周期切成 (N0, P)，逐周期 DFT -> 谱快照（C^{N0×P}）。"""
    N0 = cfg.N0
    P = x_ref.size // N0
    Xt = x_ref[: N0 * P].reshape(P, N0).T          # (N0, P)
    return np.fft.fft(Xt, axis=0)
