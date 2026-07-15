"""Streaming NUAA-MU on complex WIDEBAND signals at N0=5000 (main protocol).

主验证：复杂宽带演化信号——脉内 LFM/chirp（默认 τ=16 ns，瞬时带宽 ~20 GHz，
频域非稀疏）；脉间慢时参量沿雷达 PRI 演化（默认 320 ms，PRI ≫ τ，与光采样主周期 T0 异步）。
叠加宽带 chirp 强干扰（SIR=-40 dB）与 AWGN。

稀疏性在参数化 chirplet 字典 (f0, k)：每条宽带分量本征自由度 O(1)。
NUAA-MU 直接在参数域重构 → 展示"直接高效处理复杂宽带信号"的优势；
频域 SOMP 基线（fourier_somp）在同一事件流上失效，作为宽带反例。

流式口径：每个 100 ms tick 含一个 1 us 观察窗（200 个 T0、1000 点），
其余时间用于 EDL v=1000 ps/s bang-bang 调节；每 tick 输出重构波形、
延迟线指令与 THA strobe 账本。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure
configure()

from nuaa.config import SystemConfig
from nuaa import metrics as Met
from nuaa import reconstruct as R
from nuaa import signals as S
from nuaa import streaming as St
from experiments.train_nuaa_mu import _event_tokens, make_multitask_targets
from experiments.exp_structured_nuaa_mu import (
    CAPS, LR_SCALE, TRAIN_SCALE, STEP_CODE, JAM_CODE, _reflect,
    _append_window_phase,
)
from models.nuaa_mu import NUAAMU, si_complex_nmse

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
METHODS = ["fourier_somp", "chirplet_somp", "prior_bangbang", "mt_bangbang"]


# --------------------------------------------------------------------------
# 宽带 chirplet 原子网格与可学习演化律
# --------------------------------------------------------------------------
def pulse_slots(cfg, pulse_width_ns: float) -> int:
    """脉宽 τ 对应的 1 ps 时隙数（与光采样主周期 T0 无关）。"""
    return max(1, int(round(pulse_width_ns * 1000.0 / cfg.eta_ps)))


def pulse_slot_offset(cfg, n_pulse: int, period_index: int) -> int:
    """当前主周期在脉内的时间起点（跨主周期相位连续，每 τ 循环一段）。"""
    n_per = max(1, int(np.ceil(n_pulse / cfg.N0)))
    return (period_index % n_per) * cfg.N0


def build_atom_grid(cfg, f_lo_ghz, f_hi_ghz, bw_lo_ghz, bw_hi_ghz, n_f0, n_k, n_pulse):
    """(f0, k) 铺格。k 由脉内 LFM 扫频带宽决定：k = B_norm / N_pulse（脉宽 τ）。"""
    f0_grid = np.linspace(S.ghz_to_norm(f_lo_ghz), S.ghz_to_norm(f_hi_ghz), n_f0)
    k_grid = np.linspace(S.ghz_to_norm(bw_lo_ghz), S.ghz_to_norm(bw_hi_ghz), n_k) / n_pulse
    f0_tab = np.repeat(f0_grid, n_k)
    k_tab = np.tile(k_grid, n_f0)
    return f0_tab, k_tab


def atom_bins(f0_tab, k_tab, cfg, n_pulse):
    """原子 → N0 域代表 bin（脉内中时刻瞬时频率）。"""
    f_mid = f0_tab + 0.5 * k_tab * n_pulse
    return (np.round(f_mid * cfg.N0).astype(np.int64)) % cfg.N0


def wideband_useful_atoms(n_f0, n_k, seq_id, win, K):
    """K 条宽带分量的 (f0_idx, k_idx) 沿多谐波 + 阶跃码可学习律演化。"""
    t = float(win)
    sid = float(seq_id % 17)
    f_base = 0.25 * n_f0 + 0.02 * (seq_id % 9) * n_f0
    fh = (
        0.16 * n_f0 * np.sin(0.17 * t + 0.31 * sid)
        + 0.09 * n_f0 * np.sin(0.41 * t + 0.13 * sid)
        + 0.012 * n_f0 * STEP_CODE[win % len(STEP_CODE)]
    )
    kh = 0.5 * n_k + 0.35 * n_k * np.sin(0.29 * t + 0.11 * sid)
    atoms = []
    for i in range(K):
        fi = _reflect(f_base + fh + i * 0.22 * n_f0, 0, n_f0 - 1)
        ki = _reflect(kh + i * 0.3 * n_k, 0, n_k - 1)
        atoms.append(int(fi) * n_k + int(ki))
    return np.array(atoms, dtype=np.int64)


def wideband_jammer_atom(n_f0, n_k, seq_id, win):
    """宽带 chirp blocker 的演化轨迹（与有用分量交叠但独立）。"""
    t = float(win)
    fj = _reflect(
        0.62 * n_f0
        + 0.20 * n_f0 * np.sin(0.23 * t + 0.19 * seq_id)
        + 0.015 * n_f0 * JAM_CODE[win % len(JAM_CODE)],
        0, n_f0 - 1)
    kj = _reflect(0.4 * n_k + 0.4 * n_k * np.sign(np.sin(0.11 * t + 0.5)), 0, n_k - 1)
    return int(fj) * n_k + int(kj)


def make_atom_candidates(n_atoms, required, nC, bins_all, rng):
    """nC 个候选原子（必含真值/干扰），且 N0 代表 bin 互异。"""
    required = np.unique(np.asarray(required, dtype=np.int64))
    chosen = list(required)
    used_bins = set(int(bins_all[a]) for a in chosen)
    pool = rng.permutation(np.setdiff1d(np.arange(n_atoms), required))
    for a in pool:
        if len(chosen) >= nC:
            break
        b = int(bins_all[a])
        if b in used_bins:
            continue
        chosen.append(int(a))
        used_bins.add(b)
    return np.sort(np.array(chosen, dtype=np.int64))


def make_wideband_state(cfg, atoms_meta, seq_id, win, K, nC, rng, sir_db):
    """当前窗的宽带场景：真值原子/干扰原子/候选集/系数。"""
    f0_tab, k_tab, bins_all, n_f0, n_k = atoms_meta[:5]
    useful = wideband_useful_atoms(n_f0, n_k, seq_id, win, K)
    jam = wideband_jammer_atom(n_f0, n_k, seq_id, win)
    useful_bins = set(int(bins_all[a]) for a in useful)
    while jam in set(int(a) for a in useful) or int(bins_all[jam]) in useful_bins:
        jam = (jam + n_k + 1) % (n_f0 * n_k)
    C = make_atom_candidates(n_f0 * n_k, np.concatenate([useful, [jam]]), nC, bins_all, rng)
    loc = np.array([int(np.where(C == a)[0][0]) for a in useful], dtype=np.int64)
    jam_loc = int(np.where(C == jam)[0][0])
    alpha = np.zeros(nC, np.complex128)
    gain = 1.0 + 0.25 * np.exp(1j * (0.37 * win + 0.11 * (seq_id % 13)))
    alpha[loc] = gain * (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    alpha[jam_loc] = np.sqrt(10.0 ** (-sir_db / 10.0)) * np.exp(1j * (0.29 * win + seq_id))
    return useful, jam, C, loc, jam_loc, alpha


def build_A_chirplet(cosets, C, atoms_meta, norm_events, slot0=0):
    """A:(M,nC) 事件时刻（1 ps 槽位，含脉内偏移 slot0）上的候选 chirplet 取样。"""
    f0_tab, k_tab, _, _, _, n_pulse = atoms_meta
    cos = np.asarray(cosets, dtype=np.float64).reshape(-1)
    slot_arr = np.asarray(slot0, dtype=np.float64).reshape(-1)
    if slot_arr.size == 1:
        t = slot_arr[0] + cos
    else:
        t = slot_arr + cos
    env = (t < n_pulse).astype(np.float64)
    M, nC = t.size, len(C)
    A = np.empty((M, nC), dtype=np.complex128)
    for j, a in enumerate(C):
        A[:, j] = env * np.exp(1j * 2 * np.pi * (f0_tab[a] * t + 0.5 * k_tab[a] * t ** 2))
    return A / np.sqrt(max(1, norm_events))


def synth_waveform(C, alpha, atoms_meta, cfg, slot0: int = 0):
    """由 chirplet 系数合成当前主周期内的脉内波形片段 (N0,)。"""
    f0_tab, k_tab, _, _, _, n_pulse = atoms_meta
    n = slot0 + np.arange(cfg.N0, dtype=np.float64)
    active = n < n_pulse
    x = np.zeros(cfg.N0, dtype=np.complex128)
    for j, a in enumerate(C):
        if abs(alpha[j]) > 1e-12:
            x += alpha[j] * active * np.exp(1j * 2 * np.pi * (f0_tab[a] * n + 0.5 * k_tab[a] * n ** 2))
    return x


def synth_period_y(cosets, C, alpha, loc, atoms_meta, rng, snr_db, norm_events,
                   burst_role, win, seq_id, cfg, slot0: int = 0):
    A = build_A_chirplet(cosets, C, atoms_meta, norm_events, slot0=slot0)
    y_clean = A @ alpha
    ref = float(np.mean(np.abs(A[:, loc] @ alpha[loc]) ** 2)) + 1e-30
    y = S.add_measurement_noise(y_clean, snr_db, ref, rng)
    if burst_role != "none":
        idx = np.arange(len(cosets))
        phase_hot = ((win + idx) % 7) == (seq_id % 7)
        if np.any(phase_hot):
            burst_pow = ref * 10.0 ** (35.0 / 10.0)
            envelope = 0.75 + 0.25 * np.sin(2 * np.pi * (np.asarray(cosets) / cfg.N0) + 0.23 * win)
            phase = 2 * np.pi * (0.37 * np.asarray(cosets) / cfg.N0 + 0.09 * win)
            y = y + phase_hot * np.sqrt(burst_pow) * envelope * np.exp(1j * phase)
    return y


def scene_prior(C, useful, atoms_meta, sigma_f=0.008, sigma_k=0.35):
    """离线演化场景先验：候选原子到真值原子的 (f0,k) 归一化高斯距离。"""
    f0_tab, k_tab, _, _, n_k, _ = atoms_meta
    kk = k_tab * 1e6
    p = np.zeros(len(C), dtype=np.float64)
    for a in useful:
        df = (f0_tab[C] - f0_tab[a]) / sigma_f
        dk = (kk[C] - kk[a]) / (sigma_k * (np.ptp(kk) / max(1, n_k - 1) + 1e-12) * n_k)
        p = np.maximum(p, np.exp(-0.5 * (df ** 2 + dk ** 2)))
    return (p / (p.max() + 1e-12)).astype(np.float32)


def candidate_features(C, atoms_meta, cfg, win, period, prior=None):
    """候选原子特征。若提供 scene prior，则追加为第 5 维供先验条件训练/部署。"""
    f0_tab, k_tab, _, _, _, n_pulse = atoms_meta
    ph = 2 * np.pi * (win % period) / float(period)
    n_feat = 5 if prior is not None else 4
    feat = np.zeros((len(C), n_feat), np.float32)
    feat[:, 0] = (f0_tab[C] * 10.0).astype(np.float32)
    feat[:, 1] = (k_tab[C] * n_pulse * 10.0).astype(np.float32)
    feat[:, 2] = np.sin(ph)
    feat[:, 3] = np.cos(ph)
    if prior is not None:
        feat[:, 4] = np.asarray(prior, dtype=np.float32).reshape(-1)
    return feat


def streaming_belief_soft_prior(ctrl, bins_C, peak: float = 0.85) -> np.ndarray:
    """将跨 tick 流式信念投影到当前候选集，作为软先验（峰值 <0.9，不触发硬锁支撑）。"""
    bel = np.asarray(ctrl.state.belief.pi[np.asarray(bins_C, dtype=np.int64)], dtype=np.float64)
    bel = bel / (bel.max() + 1e-12)
    return (float(peak) * bel).astype(np.float32)


# --------------------------------------------------------------------------
# 训练（宽带 chirplet 域多任务，与结构化实验同框架）
# --------------------------------------------------------------------------
def gen_wideband_batch(cfg, atoms_meta, B, nC, K, steps, snr_db, sir_db, rng,
                       W=32, period=32, burst_role="none", with_scene_prior=False):
    from nuaa import layout as L
    n_pulse = atoms_meta[5]
    Rmeas = cfg.L * steps
    n_feat = 5 if with_scene_prior else 4
    tok = np.zeros((B, Rmeas, 10), np.float32)
    dt = np.zeros((B, Rmeas), np.float32)
    A = np.zeros((B, Rmeas, nC), np.complex64)
    Y = np.zeros((B, Rmeas), np.complex64)
    Xc = np.zeros((B, nC), np.complex64)
    cand = np.zeros((B, nC, n_feat), np.float32)
    prior = np.zeros((B, nC), np.float32)
    supp_local = np.zeros((B, K), np.int64)
    jam_local = np.zeros(B, np.int64)
    for b in range(B):
        seq_id = int(rng.integers(0, 100000))
        win = int(rng.integers(0, W))
        useful, jam, C, loc, jam_loc, alpha = make_wideband_state(
            cfg, atoms_meta, seq_id, win, K, nC, rng, sir_db)
        cosets = np.sort(np.concatenate([L.gen_fixed_random(cfg, rng) for _ in range(steps)]))
        slot0 = pulse_slot_offset(cfg, n_pulse, int(rng.integers(0, max(1, int(np.ceil(n_pulse / cfg.N0))))))
        y = synth_period_y(cosets, C, alpha, loc, atoms_meta, rng, snr_db, Rmeas,
                           burst_role, win, seq_id, cfg, slot0=slot0)
        Ac = build_A_chirplet(cosets, C, atoms_meta, Rmeas, slot0=slot0)
        tok[b] = _append_window_phase(_event_tokens(y, cosets, cfg), win, period)
        d = np.diff(cosets, prepend=cosets[0]).astype(np.float64) / max(1, cfg.N0 // cfg.L)
        dt[b] = np.log1p(np.abs(d))
        A[b] = Ac.astype(np.complex64)
        Y[b] = y.astype(np.complex64)
        Xc[b] = alpha.astype(np.complex64)
        pri = scene_prior(C, useful, atoms_meta) if with_scene_prior else None
        if pri is not None:
            prior[b] = pri
        cand[b] = candidate_features(C, atoms_meta, cfg, win, period, prior=pri)
        supp_local[b] = loc
        jam_local[b] = jam_loc
    return (torch.tensor(tok), torch.tensor(dt), torch.tensor(A), torch.tensor(Y),
            torch.tensor(Xc), torch.tensor(cand), supp_local, jam_local,
            torch.tensor(prior))


def _curriculum_snr_sir(args, t: float, rng: np.random.Generator):
    """训练 SNR/SIR：默认线性课程退火到评估点；--train-fixed-eval 时固定为评估点。"""
    snr_end = float(args.eval_snr)
    sir_end = float(args.sir)
    if getattr(args, "train_fixed_eval", False):
        # 与在线评估严格对齐，不加入随机抖动。
        return snr_end, sir_end
    snr_start = max(snr_end, float(args.snr_hi))
    if getattr(args, "sir_lo", None) is not None:
        sir_start = float(args.sir_lo)
    else:
        sir_start = min(-10.0, sir_end + 30.0)
    if sir_start < sir_end:
        sir_start = sir_end
    snr = snr_start + t * (snr_end - snr_start) + float(rng.uniform(-1.0, 1.0))
    sir = sir_start + t * (sir_end - sir_start) + float(rng.uniform(-1.0, 1.0))
    return float(snr), float(sir)


def _train_steps_pool(args) -> list[int]:
    """训练观测深度覆盖评估观察窗长度。"""
    depth = max(1, int(args.window_periods))
    pool = set(int(s) for s in args.train_steps_list)
    pool.update({depth, max(1, depth // 2), max(1, (3 * depth) // 4)})
    return sorted(pool)


def train_wideband_model(cfg, atoms_meta, args):
    hp = CAPS[args.cap]
    train_prior = bool(getattr(args, "train_scene_prior", False))
    cand_dim = 5 if train_prior else 4
    model = NUAAMU(d_in=10, nC=args.nC, K_sparse=args.K, cand_dim=cand_dim, **hp)
    if getattr(args, "model_in", None):
        checkpoint = torch.load(args.model_in, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Infer cand_dim from checkpoint if present.
        ck_cand = None
        if "prior_head.0.weight" in state_dict:
            # prior_head input = 3 + d_model + cand_dim
            in_f = int(state_dict["prior_head.0.weight"].shape[1])
            d_model = int(hp["d_model"])
            ck_cand = in_f - 3 - d_model
        if ck_cand is not None and ck_cand != cand_dim:
            model = NUAAMU(d_in=10, nC=args.nC, K_sparse=args.K,
                           cand_dim=ck_cand, **hp)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"loaded model {args.model_in} cand_dim={model.cand_dim}", flush=True)
        return model
    if args.iters <= 0:
        model.eval()
        return model
    # 固定评估点 / 先验条件训练时提高学习率：大模型默认 LR_SCALE 过小
    lr_scale = 1.0 if (getattr(args, "train_fixed_eval", False) or train_prior) else LR_SCALE[args.cap]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr * lr_scale)
    rng = np.random.default_rng(args.seed + 123)
    n_iters = int(args.iters * TRAIN_SCALE[args.cap])
    steps_pool = _train_steps_pool(args)
    jam_pos_w = float(getattr(args, "jam_pos_weight", None) or max(8.0, args.nC - 1))
    supp_pos_w = float(max(1.0, (args.nC - args.K) / max(1, args.K)))
    fixed = bool(getattr(args, "train_fixed_eval", False)) or train_prior
    model.train()
    print(f"train mode={'scene_prior' if train_prior else 'autonomous'} "
          f"cand_dim={cand_dim} fixed_eval={int(fixed)}", flush=True)
    for it in range(n_iters):
        t = it / max(1, n_iters - 1)
        snr, sir = _curriculum_snr_sir(args, t, rng)
        steps = int(rng.choice(steps_pool))
        tok, dt, A, Y, Xc, cand, supp, jam, prior = gen_wideband_batch(
            cfg, atoms_meta, args.batch, args.nC, args.K, steps, snr, sir, rng,
            W=args.W, period=args.period, burst_role=args.burst_role,
            with_scene_prior=train_prior)

        # 先验条件训练：全程 refine，使先验锁定下的重构在预训练阶段饱和
        if train_prior:
            use_refine = True
        elif fixed:
            use_refine = t >= 0.4
        else:
            use_refine = False
        Xhat, aux = model(tok, dt, A, Y, refine=use_refine, return_aux=True,
                          cand_feat=cand, use_burst=False, jam_prob_thr=0.35)
        X_target, support_target, jammer_target, supp_t = make_multitask_targets(
            Xc, supp, jam, True)

        rec = si_complex_nmse(torch.gather(Xhat, 1, supp_t), torch.gather(Xc, 1, supp_t))
        rec = rec + 0.15 * si_complex_nmse(Xhat, X_target)
        supp_w = torch.ones_like(support_target) + (supp_pos_w - 1.0) * support_target
        supp_loss = F.binary_cross_entropy_with_logits(
            aux["support_logits"], support_target, weight=supp_w)
        jam_w = torch.ones_like(jammer_target) + (jam_pos_w - 1.0) * jammer_target
        jam_loss = F.binary_cross_entropy_with_logits(
            aux["jammer_logits"], jammer_target, weight=jam_w)

        prior_loss = torch.tensor(0.0)
        if train_prior:
            # 强制 useful_prior 贴合 scene prior，使在线硬锁前模型已“吃透”先验
            prior_loss = F.mse_loss(aux["useful_prior"], prior)
            # 先验可用时优先学支撑与重构；干扰识别仍保留
            w_rec, w_supp, w_jam, w_prior = (0.55 + 0.20 * t), 0.30, 0.20, (0.50 + 0.25 * t)
        elif fixed:
            w_rec, w_supp, w_jam, w_prior = (0.25 + 0.20 * t), 0.25, 0.55, 0.0
        else:
            w_rec, w_supp, w_jam, w_prior = 1.0, 0.16, (0.12 + 0.20 * t), 0.0

        # Teacher-forced：真值干扰置零后重建 token，再前向一次专攻弱信号重构
        if fixed and t >= (0.0 if train_prior else 0.25):
            A_n = A.clone()
            Y_n = Y.clone()
            tok_n = tok.clone()
            for b in range(A_n.shape[0]):
                j = int(jam[b])
                Y_n[b] = Y_n[b] - A_n[b, :, j] * Xc[b, j]
                A_n[b, :, j] = 0
                yb = Y_n[b].detach().cpu().numpy()
                mag = np.abs(yb)
                med = float(np.median(mag))
                mad = float(np.median(np.abs(mag - med))) + 1e-6
                y_scale = float(np.quantile(mag, 0.75)) + 1e-6
                tok_n[b, :, 0] = torch.as_tensor((yb.real / y_scale).astype(np.float32))
                tok_n[b, :, 1] = torch.as_tensor((yb.imag / y_scale).astype(np.float32))
                tok_n[b, :, 2] = torch.as_tensor(np.log1p(mag).astype(np.float32))
                tok_n[b, :, 3] = torch.as_tensor(
                    np.clip((mag - med) / mad, 0, 20).astype(np.float32))
            Xhat_n, aux_n = model(tok_n, dt, A_n, Y_n, refine=True, return_aux=True,
                                  cand_feat=cand, use_burst=False, jam_prob_thr=0.35)
            rec_n = si_complex_nmse(torch.gather(Xhat_n, 1, supp_t), torch.gather(Xc, 1, supp_t))
            rec_n = rec_n + 0.15 * si_complex_nmse(Xhat_n, X_target)
            # 先验锁定 LS 教师：在去干扰观测上对真值支撑做 LS，作为稳定重构目标
            if train_prior:
                with torch.no_grad():
                    X_ls = torch.zeros_like(Xc)
                    for b in range(A_n.shape[0]):
                        idx = supp_t[b]
                        As = A_n[b, :, idx]
                        coef = torch.linalg.lstsq(
                            As, Y_n[b].unsqueeze(-1)).solution.squeeze(-1)
                        X_ls[b, idx] = coef.to(X_ls.dtype)
                rec_n = rec_n + 0.75 * si_complex_nmse(
                    torch.gather(Xhat_n, 1, supp_t), torch.gather(X_ls, 1, supp_t))
                prior_loss = prior_loss + F.mse_loss(aux_n["useful_prior"], prior)
            rec = 0.25 * rec + 0.75 * rec_n
            supp_loss_n = F.binary_cross_entropy_with_logits(
                aux_n["support_logits"], support_target, weight=supp_w)
            supp_loss = 0.2 * supp_loss + 0.8 * supp_loss_n

        loss = w_rec * rec + w_supp * supp_loss + w_jam * jam_loss + w_prior * prior_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if args.progress_every and ((it + 1) % args.progress_every == 0 or (it + 1) == n_iters):
            with torch.no_grad():
                jam_prob = torch.sigmoid(aux["jammer_logits"])
                jam_acc = float((jam_prob.argmax(dim=1) == torch.as_tensor(jam)).float().mean())
                supp_topk = torch.topk(aux["support_logits"], k=args.K, dim=1).indices
                supp_true = torch.as_tensor(supp)
                supp_acc = float(
                    (supp_topk.unsqueeze(2) == supp_true.unsqueeze(1)).any(dim=2).float().mean())
                prior_mse = float(prior_loss.detach().cpu()) if train_prior else 0.0
            print(f"PROGRESS train cap={args.cap} iter={it+1}/{n_iters} "
                  f"snr={snr:.1f} sir={sir:.1f} steps={steps} "
                  f"loss={float(loss.detach().cpu()):.4f} "
                  f"rec={float(rec.detach().cpu()):.3f} "
                  f"supp={float(supp_loss.detach().cpu()):.3f} "
                  f"jam={float(jam_loss.detach().cpu()):.3f} "
                  f"prior={prior_mse:.3f} "
                  f"supp_recall={supp_acc:.2f} jam_acc={jam_acc:.2f} "
                  f"refine={int(use_refine)}", flush=True)
    model.eval()
    model_out = getattr(args, "model_out", None)
    if model_out is None:
        model_out = os.path.join(OUT_DIR, f"streaming_wideband_nuaa_{args.tag}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cap": args.cap, "nC": args.nC,
                "K": args.K, "cand_dim": cand_dim,
                "train_scene_prior": train_prior, "params": vars(args)}, model_out)
    print(f"saved model {model_out}", flush=True)
    return model


# --------------------------------------------------------------------------
# 评估基线
# --------------------------------------------------------------------------
def eval_fourier_somp(ctrl, x_wave_true, cfg, norm_events, K_f=12):
    """频域稀疏基线：宽带信号在傅里叶字典上非稀疏 → 展示失效。"""
    arr = ctrl.state.observation_window.arrays()
    y, cosets = arr["y"], arr["coset"]
    if len(y) < 3:
        return 0.0, 0.0, -60.0
    n = np.arange(cfg.N0)
    Af = np.exp(2j * np.pi * np.outer(cosets, n) / cfg.N0) / np.sqrt(max(1, norm_events))
    k_use = int(min(K_f, max(1, len(y) - 2)))
    sp, Xf = R.somp(Af, y.reshape(-1, 1), k_use)
    x_hat = np.zeros(cfg.N0, dtype=np.complex128)
    tt = np.arange(cfg.N0, dtype=np.float64)
    for b in sp:
        x_hat += Xf[b, 0] * np.exp(2j * np.pi * b * tt / cfg.N0) / np.sqrt(max(1, norm_events))
    nm = Met.nmse_db(x_hat, x_wave_true)
    mf = S.matched_filter_peak_snr(x_hat, x_wave_true)
    return 0.0, float(nm), float(mf)


def eval_chirplet_somp(ctrl, C, alpha, loc, useful, atoms_meta, cfg, K, norm_events,
                       x_wave_true, prior=None, beta=2.0, null_loc=None, slot0: int = 0):
    arr = ctrl.state.observation_window.arrays()
    y, cosets = arr["y"], arr["coset"]
    pulse_slots_buf = arr.get("pulse_slot0", np.zeros_like(cosets))
    if len(y) < K + 1:
        return 0.0, 0.0, -60.0, None
    A = build_A_chirplet(cosets, C, atoms_meta, norm_events, slot0=pulse_slots_buf)
    Y = y.reshape(-1, 1)
    if null_loc is not None:
        A_use, Y_use = R.null_jammer(A, Y, [int(null_loc)])
    else:
        A_use, Y_use = A, Y
    if prior is not None:
        sp, Xh = R.somp_prior(A_use, Y_use, K, prior=prior, beta=beta)
    else:
        sp, Xh = R.somp(A_use, Y_use, K)
    est_atoms = C[np.asarray(sp, dtype=np.int64)] if len(sp) else np.zeros(0, np.int64)
    _, _, f1 = Met.support_f1(est_atoms, useful)
    alpha_hat = np.zeros(len(C), np.complex128)
    if len(sp):
        alpha_hat[np.asarray(sp)] = Xh[np.asarray(sp), 0]
    x_hat = synth_waveform(C, alpha_hat, atoms_meta, cfg, slot0=slot0)
    nm = Met.nmse_db(x_hat, x_wave_true)
    mf = S.matched_filter_peak_snr(x_hat, x_wave_true)
    return float(f1), float(nm), float(mf), sp


def update_belief_from_window(ctrl, C, bins_C, atoms_meta, cfg, norm_events, null_loc=None):
    """用当前观察窗证据更新在线信念（供基线驱动慢环）。"""
    arr = ctrl.state.observation_window.arrays()
    A_full = build_A_chirplet(arr["coset"], C, atoms_meta, norm_events,
                              slot0=arr.get("pulse_slot0", 0))
    Y = arr["y"].reshape(-1, 1)
    if null_loc is not None:
        A_full, Y = R.null_jammer(A_full, Y, [int(null_loc)])
    ctrl.state.belief.decay()
    ev = np.abs((A_full / (np.linalg.norm(A_full, axis=0, keepdims=True) + 1e-12)
                 ).conj().T @ Y).reshape(-1)
    full_ev = np.full(cfg.N0, 1e-6)
    full_ev[bins_C] = ev
    ctrl.state.belief.update(full_ev)


def window_tensors(ctrl, C, atoms_meta, cfg, win, period, norm_events, prior=None):
    arr = ctrl.state.observation_window.arrays()
    y = arr["y"].astype(np.complex64)
    cosets = arr["coset"].astype(np.int64)
    slot0 = arr.get("pulse_slot0", np.zeros_like(cosets))
    A = build_A_chirplet(cosets, C, atoms_meta, norm_events, slot0=slot0).astype(np.complex64)
    tok = _append_window_phase(_event_tokens(y, cosets, cfg), win, period=max(8, period + 1))
    absolute_slots = arr["period"].astype(np.float64) * cfg.N0 + cosets
    d = np.diff(absolute_slots, prepend=absolute_slots[0])
    d = d.astype(np.float64) / max(1, cfg.N0 // cfg.L)
    dt = np.log1p(np.abs(d)).astype(np.float32)
    cand = candidate_features(
        C, atoms_meta, cfg, win, max(8, period + 1), prior=prior)
    return tok.astype(np.float32), dt, A, y, cand


def calibrate_scene_coefficient_prior(
    ctrl,
    C,
    alpha,
    loc,
    jam_loc,
    atoms_meta,
    cfg,
    args,
    win,
    seq_id,
    rng,
):
    """Build a saturated pre-deployment coefficient posterior.

    The scene support is supplied by the offline prior. Historical calibration
    windows estimate the random complex gains, which cannot be inferred from
    support-only neural pretraining. The returned posterior mean is frozen
    during deployment and receives only a small current-window update.
    """
    n_windows = max(0, int(args.prior_calibration_windows))
    if n_windows == 0:
        return None
    norm_events = int(args.window_periods * cfg.L)
    n_pulse = atoms_meta[5]
    A_rows = []
    y_rows = []
    for cal_tick in range(n_windows):
        first_period = cal_tick * args.window_periods
        for period_offset in range(args.window_periods):
            slot0 = pulse_slot_offset(
                cfg, n_pulse, first_period + period_offset)
            cosets = ctrl.current_cosets()
            A_rows.append(
                build_A_chirplet(
                    cosets, C, atoms_meta, norm_events, slot0=slot0))
            y_rows.append(
                synth_period_y(
                    cosets, C, alpha, loc, atoms_meta, rng, args.eval_snr,
                    norm_events, args.burst_role, win, seq_id, cfg,
                    slot0=slot0))
    A_cal = np.vstack(A_rows)
    y_cal = np.concatenate(y_rows)
    return ctrl._window_known_support_ls(A_cal, y_cal, loc, jam_loc)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def method_list(args):
    ms = list(args.methods) if args.methods else list(METHODS)
    if args.with_oracle_null:
        ms.append("nullsomp_oracle")
    return ms


def signal_slow_index(tick: int, trial_seed: int, signal_pri_ticks: int, W: int) -> int:
    """慢时索引：每 signal_pri_ticks 个流式步进一次（雷达 PRI 尺度，与 T0 异步）。"""
    seg = tick // max(1, signal_pri_ticks)
    return (trial_seed // 1000 + seg) % max(1, W)


def run_one_trial(cfg, atoms_meta, model, args, trial_seed):
    methods = method_list(args)
    rng = np.random.default_rng(trial_seed)
    seq_id = int(rng.integers(0, 100000))
    norm_events = args.window_periods * cfg.L
    bins_all = atoms_meta[2]
    ctrls = {m: St.StreamingNUAAController(cfg, K=args.K, window_events=norm_events,
                                           seed=trial_seed + 1 + i)
             for i, m in enumerate(methods)}
    curves = {
        m: {
            "f1": [], "nmse": [], "mf": [], "belief_mass": [],
            "evidence_events": [], "jammer_correct": [], "jammer_confidence": [],
        }
        for m in methods
    }
    hit = {m: None for m in methods}
    ledgers = {}
    seg_states: dict[int, tuple] = {}
    alpha_ema: dict[str, np.ndarray] = {}
    alpha_prior_init: dict[str, np.ndarray] = {}

    for tick in range(args.ticks):
        seg = tick // max(1, args.signal_pri_ticks)
        win = signal_slow_index(tick, trial_seed, args.signal_pri_ticks, args.W)
        n_pulse = atoms_meta[5]
        first_period = tick * args.window_periods
        # 固定脉内参考槽位评估，避免 tick 间脉内片段轮换造成的假性 NMSE 抖动
        report_slot0 = 0
        if args.hold_coeffs:
            # 静态场景捕获口径：同一演化段内场景系数保持，事件跨周期相干积累。
            if seg not in seg_states:
                seg_states[seg] = make_wideband_state(
                    cfg, atoms_meta, seq_id, win, args.K, args.nC, rng, args.sir)
            useful, jam, C, loc, jam_loc, alpha = seg_states[seg]
        else:
            useful, jam, C, loc, jam_loc, alpha = make_wideband_state(
                cfg, atoms_meta, seq_id, win, args.K, args.nC, rng, args.sir)
        alpha_useful = np.zeros_like(alpha)
        alpha_useful[loc] = alpha[loc]
        x_wave_true = synth_waveform(
            C, alpha_useful, atoms_meta, cfg, slot0=report_slot0)
        bins_C = bins_all[C]
        truth_X = np.zeros(cfg.N0, dtype=np.complex128)
        truth_X[bins_C] = alpha
        truth_support = bins_C[loc]

        for method, ctrl in ctrls.items():
            ctrl.begin_observation_window()
            for period_offset in range(args.window_periods):
                slot0 = pulse_slot_offset(
                    cfg, n_pulse, first_period + period_offset)
                cosets = ctrl.current_cosets()
                y = synth_period_y(
                    cosets, C, alpha, loc, atoms_meta, rng, args.eval_snr,
                    norm_events, args.burst_role, win, seq_id, cfg, slot0=slot0)
                ctrl.append_period_measurement(y, pulse_slot0=slot0)
            if method == "fourier_somp":
                rec = None
                f1, nm, mf = eval_fourier_somp(ctrl, x_wave_true, cfg, norm_events)
            elif method == "chirplet_somp":
                rec = None
                f1, nm, mf, _ = eval_chirplet_somp(
                    ctrl, C, alpha, loc, useful, atoms_meta, cfg, args.K,
                    norm_events, x_wave_true, slot0=report_slot0)
            elif method == "prior_bangbang":
                rec = None
                pri = ctrl.state.belief.pi[bins_C]
                pri = pri / (pri.max() + 1e-12)
                f1, nm, mf, sp = eval_chirplet_somp(
                    ctrl, C, alpha, loc, useful, atoms_meta, cfg, args.K,
                    norm_events, x_wave_true, prior=pri, slot0=report_slot0)
                if ctrl.state.observation_window.arrays()["y"].size >= args.K:
                    update_belief_from_window(
                        ctrl, C, bins_C, atoms_meta, cfg, norm_events)
            elif method == "nullsomp_oracle":
                rec = None
                # 诊断基线：oracle 给定干扰原子并 U_jam 置零，其余同 chirplet SOMP + bang-bang。
                f1, nm, mf, sp = eval_chirplet_somp(
                    ctrl, C, alpha, loc, useful, atoms_meta, cfg, args.K,
                    norm_events, x_wave_true, null_loc=jam_loc, slot0=report_slot0)
                if ctrl.state.observation_window.arrays()["y"].size >= args.K:
                    update_belief_from_window(
                        ctrl, C, bins_C, atoms_meta, cfg, norm_events,
                        null_loc=jam_loc)
            else:  # mt_bangbang
                if args.no_scene_prior:
                    # 自主路径：用跨 tick 流式信念作软先验（非 GT scene prior）
                    prior_ov = streaming_belief_soft_prior(ctrl, bins_C)
                    allow_lock = False
                    prior_feat = None
                else:
                    prior_ov = scene_prior(C, useful, atoms_meta)
                    allow_lock = True
                    # 有先验部署：把 scene prior 写入候选特征，供已训练的先验条件模型使用
                    prior_feat = prior_ov if getattr(model, "cand_dim", 4) >= 5 else None
                    if method not in alpha_prior_init:
                        cal_rng = np.random.default_rng(
                            trial_seed + 900001 + methods.index(method))
                        calibrated = calibrate_scene_coefficient_prior(
                            ctrl, C, alpha, loc, jam_loc, atoms_meta, cfg,
                            args, win, seq_id, cal_rng)
                        if calibrated is not None:
                            alpha_prior_init[method] = calibrated
                tok, dtn, A, Y, cand = window_tensors(
                    ctrl, C, atoms_meta, cfg, win, args.period, norm_events,
                    prior=prior_feat)
                rec = ctrl.reconstruct_with_model(
                    model, tok, dtn, A, Y, bins_C, cand_feat=cand,
                    prior_override=prior_ov, allow_prior_lock=allow_lock,
                    accumulate_coefficients=(
                        args.no_scene_prior
                        or not args.fixed_window_coefficients),
                    use_burst=False, truth_X=truth_X, truth_support=truth_support)
                alpha_hat = rec.Xhat[bins_C, 0]
                # 有先验路径：保留在线学习（递归状态/信念），仅用轻度 EMA 抑制
                # 单窗噪声；系数不跨窗累积，故饱和后无持续下降趋势。
                if allow_lock:
                    ema_key = method
                    beta = float(getattr(args, "prior_coef_ema", 0.0))
                    if ema_key not in alpha_ema or beta <= 0.0:
                        alpha_ema[ema_key] = alpha_hat.copy()
                    else:
                        alpha_ema[ema_key] = (
                            beta * alpha_ema[ema_key] + (1.0 - beta) * alpha_hat)
                    alpha_report = alpha_ema[ema_key]
                    if method in alpha_prior_init:
                        gain = float(np.clip(args.prior_online_gain, 0.0, 1.0))
                        alpha_report = (
                            (1.0 - gain) * alpha_prior_init[method]
                            + gain * alpha_report)
                else:
                    alpha_report = alpha_hat
                x_hat = synth_waveform(
                    C, alpha_report, atoms_meta, cfg, slot0=report_slot0)
                est_atoms = C[np.isin(bins_C, rec.support)]
                _, _, f1 = Met.support_f1(est_atoms, useful)
                nm = Met.nmse_db(x_hat, x_wave_true)
                mf = S.matched_filter_peak_snr(x_hat, x_wave_true)
                f1, nm, mf = float(f1), float(nm), float(mf)
            curves[method]["f1"].append(f1)
            curves[method]["nmse"].append(nm)
            curves[method]["mf"].append(mf)
            curves[method]["belief_mass"].append(ctrl.state.belief.topm_mass(args.K))
            curves[method]["evidence_events"].append(ctrl.state.evidence_events)
            if method == "mt_bangbang" and rec is not None and rec.jammer_index is not None:
                curves[method]["jammer_correct"].append(
                    float(rec.jammer_index == jam_loc))
                curves[method]["jammer_confidence"].append(
                    float(rec.jammer_confidence or 0.0))
            else:
                curves[method]["jammer_correct"].append(0.0)
                curves[method]["jammer_confidence"].append(0.0)
            if hit[method] is None and nm <= args.target_nmse_db:
                hit[method] = (tick + 1) * args.control_dt_ms * 1e-3
            if method in ("fourier_somp", "chirplet_somp"):
                mode = "static_hold"
            elif method == "mt_bangbang":
                mode = args.mt_mode
            else:
                mode = "belief_bangbang"
            ctrl.plan_next_period(dt_s=args.control_dt_ms * 1e-3, mode=mode)

    for method, ctrl in ctrls.items():
        led = ctrl.acquisition_ledger()
        led["wall_time_s"] = args.ticks * args.control_dt_ms * 1e-3
        led["observation_time_s"] = (
            args.ticks * args.window_periods * cfg.T0_ps * 1e-12)
        led["events_per_window"] = norm_events
        led["materialized_events"] = args.ticks * norm_events
        ledgers[method] = led
    return curves, hit, ledgers


def run(cfg, args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    n_pulse = pulse_slots(cfg, args.pulse_width_ns)
    f0_tab, k_tab = build_atom_grid(cfg, args.f_lo_ghz, args.f_hi_ghz,
                                    args.bw_lo_ghz, args.bw_hi_ghz, args.n_f0, args.n_k,
                                    n_pulse)
    atoms_meta = (f0_tab, k_tab, atom_bins(f0_tab, k_tab, cfg, n_pulse),
                  args.n_f0, args.n_k, n_pulse)
    model = train_wideband_model(cfg, atoms_meta, args)
    methods = method_list(args)
    n_par = int(sum(p.numel() for p in model.parameters()))
    results = {"config": cfg.summary(), "params": vars(args), "mt_n_params": n_par,
               "methods": {}}
    all_curves = {m: [] for m in methods}
    all_hits = {m: [] for m in methods}
    all_ledgers = {m: [] for m in methods}
    print(f"streaming WIDEBAND NUAA | N0={cfg.N0} eta={cfg.eta_ps:g}ps "
          f"band {args.f_lo_ghz:g}-{args.f_hi_ghz:g} GHz, intrapulse BW "
          f"{args.bw_lo_ghz:g}-{args.bw_hi_ghz:g} GHz | "
          f"tau={args.pulse_width_ns:g}ns ({n_pulse} slots) | "
          f"PRI≈{args.signal_pri_ticks * args.control_dt_ms:g}ms "
          f"({args.signal_pri_ticks} ticks×{args.control_dt_ms:g}ms) | "
          f"window={args.window_periods * cfg.T0_ps * 1e-6:g}us "
          f"({args.window_periods} periods, {args.window_periods * cfg.L} points) | "
          f"cap={args.cap} params={n_par/1e3:.1f}K ticks={args.ticks} trials={args.trials}", flush=True)
    for tr in range(args.trials):
        print(f"PROGRESS trial={tr+1}/{args.trials} start", flush=True)
        curves, hit, ledgers = run_one_trial(cfg, atoms_meta, model, args,
                                             args.seed + 1000 * tr)
        for m in methods:
            all_curves[m].append(curves[m])
            if hit[m] is not None:
                all_hits[m].append(hit[m])
            all_ledgers[m].append(ledgers[m])
        summary = " ".join(f"{m}:{curves[m]['nmse'][-1]:+.1f}dB" for m in methods)
        print(f"PROGRESS trial={tr+1}/{args.trials} done final_nmse {summary}", flush=True)
    for m in methods:
        final_f1 = [c["f1"][-1] for c in all_curves[m]]
        final_nmse = [c["nmse"][-1] for c in all_curves[m]]
        final_mf = [c["mf"][-1] for c in all_curves[m]]
        hits = all_hits[m]
        # 逐 tick 跨 trial 聚合曲线：波形质量随流式处理时间的演进。
        tick_curve = {}
        for key in (
            "f1", "nmse", "mf", "belief_mass", "evidence_events",
            "jammer_correct", "jammer_confidence",
        ):
            mat = np.array([c[key] for c in all_curves[m]], dtype=np.float64)
            tick_curve[f"{key}_med"] = np.median(mat, axis=0).tolist()
            tick_curve[f"{key}_p25"] = np.percentile(mat, 25, axis=0).tolist()
            tick_curve[f"{key}_p75"] = np.percentile(mat, 75, axis=0).tolist()
        # 累计最优 NMSE（到时刻 t 为止的最好波形质量，单调，体现持续提升）。
        nmse_mat = np.array([c["nmse"] for c in all_curves[m]], dtype=np.float64)
        best_so_far = np.minimum.accumulate(nmse_mat, axis=1)
        tick_curve["nmse_best_med"] = np.median(best_so_far, axis=0).tolist()
        results["methods"][m] = dict(
            final_f1_mean=float(np.mean(final_f1)),
            final_nmse_med_db=float(np.median(final_nmse)),
            final_mf_med_db=float(np.median(final_mf)),
            hit_rate=float(len(hits) / max(1, args.trials)),
            median_time_to_target_s=(float(np.median(hits)) if hits else None),
            tau_movement_ps_mean=float(np.mean([l["tau_movement_ps"] for l in all_ledgers[m]])),
            strobe_feasible_rate_mean=float(np.mean([l["strobe_feasible_rate"] for l in all_ledgers[m]])),
            tick_curve=tick_curve,
            example_curve=all_curves[m][0],
            example_ledger=all_ledgers[m][0],
        )
        r = results["methods"][m]
        print(f"  {m:14s} F1={r['final_f1_mean']:.3f} NMSE={r['final_nmse_med_db']:+.1f}dB "
              f"MF={r['final_mf_med_db']:+.1f}dB hit={r['hit_rate']:.0%} "
              f"Δtau={r['tau_movement_ps_mean']:.1f}ps", flush=True)
    out = os.path.join(OUT_DIR, f"streaming_wideband_nuaa_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out}")
    _plot_curves(results, args, os.path.join(OUT_DIR, f"streaming_wideband_nuaa_{args.tag}.png"))
    return results


def _plot_curves(results, args, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})")
        return
    t_ms = (np.arange(args.ticks) + 1) * args.control_dt_ms
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in results["methods"]:
        tc = results["methods"][m]["tick_curve"]
        style = dict(lw=2.2, marker="o", ms=4) if m == "mt_bangbang" else dict(lw=1.2, marker=".", ms=3, alpha=0.8)
        ax[0].plot(t_ms, tc["nmse_med"], label=m, **style)
        ax[0].fill_between(t_ms, tc["nmse_p25"], tc["nmse_p75"], alpha=0.12)
        ax[1].plot(t_ms, tc["nmse_best_med"], label=m, **style)
        ax[2].plot(t_ms, tc["f1_med"], label=m, **style)
    ax[0].set(xlabel="streaming time (ms)", ylabel="waveform NMSE (dB)",
              title="Per-tick waveform NMSE (median, IQR)")
    ax[1].set(xlabel="streaming time (ms)", ylabel="best-so-far NMSE (dB)",
              title="Cumulative-best waveform quality")
    ax[2].set(xlabel="streaming time (ms)", ylabel="atom F1 (median)",
              title="Support capture", ylim=(-0.05, 1.05))
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--K", type=int, default=2, help="宽带有用分量条数")
    ap.add_argument("--nC", type=int, default=48, help="chirplet 候选原子数")
    ap.add_argument("--n-f0", type=int, default=24)
    ap.add_argument("--n-k", type=int, default=8)
    ap.add_argument("--f-lo-ghz", type=float, default=20.0)
    ap.add_argument("--f-hi-ghz", type=float, default=120.0)
    ap.add_argument("--bw-lo-ghz", type=float, default=10.0,
                    help="脉内 LFM 扫频带宽下限（GHz）")
    ap.add_argument("--bw-hi-ghz", type=float, default=30.0,
                    help="脉内 LFM 扫频带宽上限（GHz）")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--period", type=int, default=32,
                    help="慢时相位特征周期（与光采样主周期无关）")
    ap.add_argument("--signal-pri-ticks", type=int, default=32,
                    help="脉冲重复间隔 PRI：每隔多少个流式步（control tick）慢时参量才演化一次；"
                         "PRI_ms ≈ signal_pri_ticks × control_dt_ms，须满足 PRI ≫ τ")
    ap.add_argument("--pulse-width-ns", type=float, default=16.0,
                    help="雷达脉宽 τ（ns）；chirp 率 k=B/τ，跨 tick 脉内相位连续")
    ap.add_argument("--sir", type=float, default=-40.0)
    ap.add_argument("--sir-lo", type=float, default=None,
                    help="训练 SIR 课程起点（默认 min(-10, sir+30)；越接近 0 越易）")
    ap.add_argument("--train-fixed-eval", action="store_true",
                    help="训练仅使用评估点 (eval_snr, sir)，不做 SNR/SIR 课程退火")
    ap.add_argument("--jam-pos-weight", type=float, default=None,
                    help="jammer BCE 正类权重（默认 nC-1）；SIR=-40 固定训练时必需，否则负类淹没")
    ap.add_argument("--eval-snr", type=float, default=3.0)
    ap.add_argument("--snr-lo", type=float, default=-6.0,
                    help="（兼容旧接口）课程终点仍对齐 --eval-snr；起点取 max(eval_snr, snr_hi)")
    ap.add_argument("--snr-hi", type=float, default=8.0,
                    help="训练 SNR 课程起点（退火到 --eval-snr）")
    ap.add_argument("--burst-role", choices=["none", "interference"], default="none")
    ap.add_argument("--cap", choices=sorted(CAPS), default="mid")
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--model-in", type=str, default=None,
                    help="加载已训练 checkpoint 并跳过训练")
    ap.add_argument("--model-out", type=str, default=None,
                    help="训练 checkpoint 输出路径（默认按 tag 写入 outputs/）")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--train-steps-list", type=int, nargs="+", default=[4, 8, 12, 16],
                    help="训练观测深度候选；实际还会并入 --window-periods 及其分数倍")
    ap.add_argument("--window-periods", type=int, default=200,
                    help="每个观察窗的主周期数；T0=5 ns 时 200 周期即 1 us/1000 点")
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--evo-hold-ticks", type=int, default=None,
                    help="（已弃用，由 --signal-pri-ticks 取代）同一慢时脉冲内保持场景系数")
    ap.add_argument("--hold-coeffs", action="store_true",
                    help="同一演化段内保持场景系数，事件跨周期相干积累（静态捕获口径，用于持续提升曲线）")
    ap.add_argument("--mt-mode", choices=["belief_bangbang", "static_hold", "random_slow_scan"],
                    default="belief_bangbang",
                    help="MT 方法的延迟线控制模式（消融：static_hold=冻结轨迹，剥离布局多样性贡献）")
    ap.add_argument("--no-scene-prior", action="store_true",
                    help="不注入离线 scene prior，仅用模型自身先验头 + jammer head 置零（容量扫描口径）")
    ap.add_argument("--train-scene-prior", action="store_true",
                    help="训练时注入 scene prior 特征并匹配 useful_prior，使有先验部署更稳定")
    ap.add_argument("--prior-coef-ema", type=float, default=0.0,
                    help="有先验在线部署时系数 EMA（0=关闭；0.3 轻度平滑；越大越滞后）")
    ap.add_argument("--prior-calibration-windows", type=int, default=0,
                    help="部署前用于初始化饱和系数后验的历史观察窗数")
    ap.add_argument("--prior-online-gain", type=float, default=0.1,
                    help="有先验部署时当前窗估计注入预校准系数后验的增益")
    ap.add_argument("--fixed-window-coefficients", action="store_true",
                    help="有先验路径使用固定窗系数；无先验路径仍在线累积充分统计量")
    ap.add_argument("--with-oracle-null", action="store_true",
                    help="附加诊断基线：oracle 干扰原子 U_jam 置零后的 chirplet SOMP + bang-bang")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=None,
                    help="只运行指定方法；先验积累对比建议 --methods mt_bangbang")
    ap.add_argument("--control-dt-ms", type=float, default=100.0)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--target-nmse-db", type=float, default=-10.0,
                    help="宽带波形 NMSE 命中阈值（time-to-target 判据）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n5000_wb")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.evo_hold_ticks is not None and args.evo_hold_ticks < args.signal_pri_ticks:
        # 兼容旧脚本：显式传入更小 evo_hold_ticks 时覆盖 PRI
        args.signal_pri_ticks = int(args.evo_hold_ticks)
    if args.quick:
        args.iters = 0
        args.ticks = 2
        args.trials = 2
        args.cap = "small"
        args.window_periods = min(args.window_periods, 4)
        args.tag = "quick" if args.tag == "n5000_wb" else args.tag
    cfg = SystemConfig(N0=args.n0)
    run(cfg, args)


if __name__ == "__main__":
    main()
