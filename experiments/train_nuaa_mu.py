"""训练 NUAA-MU 学习多陪集重构器，并与 SOMP 基线对比（对应 计算光采样.md §5C/§7）。

固定训练域（CPU 友好）：N0、候选集大小 nC、活跃子带 K、累积步数 n_steps(=R/L)、单快照。
学习目标：从多陪集测量 Y（含 −SIR dB 强干扰折叠）恢复 C 上复谱，重点弱信号子带。

用法：
  python experiments/train_nuaa_mu.py --quick
  python experiments/train_nuaa_mu.py --n0 5000 --nC 18 --K 3 --steps 3 --iters 400
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure
configure()

from nuaa.config import SystemConfig
from nuaa import measurement as M, reconstruct as R, metrics as Met, layout as L
from models.nuaa_mu import NUAAMU, si_complex_nmse

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _event_tokens(y, cosets, cfg):
    cn = cosets / cfg.N0
    mag = np.abs(y)
    med = np.median(mag)
    mad = np.median(np.abs(mag - med)) + 1e-6
    y_scale = np.quantile(mag, 0.75) + 1e-6
    tok = np.zeros((len(cosets), 8), np.float32)
    tok[:, 0] = (y.real / y_scale).astype(np.float32)
    tok[:, 1] = (y.imag / y_scale).astype(np.float32)
    tok[:, 2] = np.log1p(mag).astype(np.float32)
    tok[:, 3] = np.clip((mag - med) / mad, 0, 20).astype(np.float32)
    tok[:, 4] = cn
    tok[:, 5] = np.sin(2 * np.pi * cn)
    tok[:, 6] = np.cos(2 * np.pi * cn)
    tok[:, 7] = np.arange(len(cosets)) / max(1, len(cosets))
    return tok


def gen_batch(cfg, B, nC, K, n_steps, snr_db, sir_db, rng, layout="random", jammer=True,
              pre_null: bool = True, burst_jammer: bool = False,
              burst_prob: float = 0.15, burst_power_db: float = 35.0):
    """生成一个训练/评估批。返回 torch 张量与（用于 SOMP/F1 的）numpy 元信息。"""
    Rmeas = cfg.L * n_steps
    guard = max(1, cfg.N0 // 50)
    tok = np.zeros((B, Rmeas, 8), np.float32)
    dt = np.zeros((B, Rmeas), np.float32)
    A = np.zeros((B, Rmeas, nC), np.complex64)
    Y = np.zeros((B, Rmeas), np.complex64)
    Xc = np.zeros((B, nC), np.complex64)
    supp_local = np.zeros((B, K), np.int64)
    Cs = np.zeros((B, nC), np.int64)
    jam_local = np.zeros((B,), np.int64)
    for bsi in range(B):
        C = np.sort(rng.choice(np.arange(guard, cfg.N0 - guard), size=nC, replace=False))
        sel = rng.choice(nC, size=K + 1, replace=False)
        sup, jam = np.sort(sel[:K]), int(sel[K])
        x = np.zeros(nC, np.complex128)
        x[sup] = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
        if jammer:
            x[jam] = np.sqrt(10.0 ** (-sir_db / 10.0)) * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        # 布局：累积 n_steps 组（每带一个），随机或 NUAA（这里训练用随机以求泛化）
        cosets = np.concatenate([L.gen_fixed_random(cfg, rng) for _ in range(n_steps)])
        cosets = np.sort(cosets)
        # 列归一化算子 Φ/√R（|entry|=1/√R），使展开步 αA^H(AX-Y) 良态（避免 Φ/N0 的 1e-3 量级塌缩）
        # 直接构造候选列（exp(2πi·cosets⊗C/N0)），避免在 N0=5000 时先建全 (R,N0) 再切片（~200×浪费）
        Ac = np.exp(2j * np.pi * np.outer(cosets, C) / cfg.N0) / np.sqrt(Rmeas)   # (R,nC)
        yc = Ac @ x
        ref = float(np.mean(np.abs((Ac[:, sup] @ x[sup])) ** 2)) + 1e-30
        noise = np.sqrt(ref * 10.0 ** (-snr_db / 10.0) / 2) * (
            rng.standard_normal(Rmeas) + 1j * rng.standard_normal(Rmeas))
        y = yc + noise
        if burst_jammer:
            mask = rng.random(Rmeas) < burst_prob
            if np.any(mask):
                burst_ref = ref * 10.0 ** (burst_power_db / 10.0)
                burst = np.sqrt(burst_ref / 2) * (
                    rng.standard_normal(Rmeas) + 1j * rng.standard_normal(Rmeas))
                y = y + mask.astype(np.float64) * burst
        if pre_null:
            # 原论文公平口径：两法均先做 U_jam 预置零。
            Ac_p, y_p, _jc = R.detect_and_null_jammer(Ac, y)
            Ac, y = Ac_p, y_p.reshape(-1)
            sca = np.median(np.linalg.norm(Ac, axis=0)) + 1e-12
            Ac = Ac / sca; y = y / sca
        tok[bsi] = _event_tokens(y, cosets, cfg)
        d = np.diff(cosets, prepend=cosets[0]).astype(np.float64) / max(1, cfg.N0 // cfg.L)
        dt[bsi] = np.log1p(np.abs(d))
        A[bsi] = Ac.astype(np.complex64); Y[bsi] = y.astype(np.complex64)
        Xc[bsi] = x.astype(np.complex64); supp_local[bsi] = sup; Cs[bsi] = C; jam_local[bsi] = jam
    return (torch.tensor(tok), torch.tensor(dt), torch.tensor(A), torch.tensor(Y),
            torch.tensor(Xc), supp_local, Cs, jam_local)


def model_f1_nmse(Xhat, Xc, supp_local, K):
    """从模型输出直接取弱信号 top-K。返回 (F1, NMSE_dB)。"""
    B = Xhat.shape[0]
    f1s, nms = [], []
    Xh = Xhat.detach().cpu().numpy(); Xt = Xc.cpu().numpy()
    for b in range(B):
        mag = np.abs(Xh[b]).copy()
        est = np.argsort(mag)[-K:]
        _, _, f1 = Met.support_f1(est, supp_local[b])
        f1s.append(f1)
        nms.append(Met.nmse(Xh[b][supp_local[b]], Xt[b][supp_local[b]]))
    return float(np.mean(f1s)), float(np.median([10 * np.log10(x + 1e-12) for x in nms]))


def make_multitask_targets(Xc, supp, jam, jammer: bool):
    """构造弱信号重构目标 + useful/jammer 候选标签。

    U_jam 投影后干扰坐标已经被从观测中移除，因此它不应再进入全谱重构目标；
    否则模型会被迫拟合一个线性代数上不可恢复的分量。
    """
    X_target = Xc.clone()
    support_target = torch.zeros_like(Xc.real)
    jammer_target = torch.zeros_like(Xc.real)
    supp_t = torch.as_tensor(supp, dtype=torch.long, device=Xc.device)
    support_target.scatter_(1, supp_t, 1.0)
    if jammer:
        jam_t = torch.as_tensor(jam, dtype=torch.long, device=Xc.device)
        rows = torch.arange(Xc.shape[0], device=Xc.device)
        X_target[rows, jam_t] = 0
        jammer_target[rows, jam_t] = 1.0
    return X_target, support_target, jammer_target, supp_t


def nuaa_mu_multitask_loss(model, tok, dt, A, Y, Xc, supp, jam,
                           snr_db: float, jammer: bool,
                           jam_pos_weight: float = 1.0,
                           refine_loss_weight: float = 0.0,
                           jam_prob_thr: float = 0.5):
    Xhat, aux = model(tok, dt, A, Y, refine=False, return_aux=True)
    X_target, support_target, jammer_target, supp_t = make_multitask_targets(
        Xc, supp, jam, jammer)
    Xhat_s = torch.gather(Xhat, 1, supp_t)
    Xc_s = torch.gather(Xc, 1, supp_t)
    rec_loss = si_complex_nmse(Xhat_s, Xc_s) + 0.2 * si_complex_nmse(Xhat, X_target)
    support_loss = F.binary_cross_entropy_with_logits(
        aux["support_logits"], support_target)
    jam_w = torch.ones_like(jammer_target)
    if jam_pos_weight != 1.0:
        jam_w = jam_w + (float(jam_pos_weight) - 1.0) * jammer_target
    jammer_loss = F.binary_cross_entropy_with_logits(
        aux["jammer_logits"], jammer_target, weight=jam_w)
    noise_target = torch.full_like(aux["noise_logvar"], -snr_db / 10.0)
    noise_loss = F.mse_loss(aux["noise_logvar"], noise_target)
    loss = rec_loss + 0.12 * support_loss + 0.08 * jammer_loss + 0.01 * noise_loss
    if refine_loss_weight > 0.0:
        Xhat_r, _ = model(tok, dt, A, Y, refine=True, return_aux=True,
                          jam_prob_thr=jam_prob_thr)
        Xhat_rs = torch.gather(Xhat_r, 1, supp_t)
        refine_rec = si_complex_nmse(Xhat_rs, Xc_s) + 0.2 * si_complex_nmse(Xhat_r, X_target)
        loss = loss + float(refine_loss_weight) * refine_rec
    return Xhat, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000, help="N0=5000 -> eta=1ps（论文口径）")
    ap.add_argument("--nC", type=int, default=18)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3, help="累积步数 n_steps；R=L*steps")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--snr-lo", type=float, default=None,
                    help="训练 SNR 下界；与 --snr-hi 同时设置则随机混合 SNR")
    ap.add_argument("--snr-hi", type=float, default=None)
    ap.add_argument("--sir", type=float, default=-40.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-jammer", action="store_true", help="关闭强干扰（演示学习式多陪集求逆）")
    ap.add_argument("--raw", action="store_true",
                    help="不做 U_jam 预投影/白化，让 NUAA-MU 直接学习干扰消除")
    ap.add_argument("--burst-jammer", action="store_true", help="加入突发测量干扰")
    ap.add_argument("--big", action="store_true",
                    help="大容量 NUAA-MU（容纳更多先验）：d_model=192 层=4 state=32 unroll=8")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.big:
        args.d_model, args.n_layers, args.d_state, args.unroll = 192, 4, 32, 8
        if args.iters == 400:
            args.iters = 800
    if args.quick:
        args.iters, args.batch = 120, 24

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = SystemConfig(N0=args.n0)
    print(f"train NUAA-MU | N0={cfg.N0} nC={args.nC} K={args.K} R={cfg.L*args.steps} "
          f"snr={args.snr} sir={args.sir} iters={args.iters}")

    model = NUAAMU(d_in=8, nC=args.nC, d_model=args.d_model, n_layers=args.n_layers,
                   d_state=args.d_state, K_unroll=args.unroll, K_sparse=args.K)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NUAA-MU params: {n_params/1e3:.1f}K")

    model.train()
    snr_lo = args.snr_lo if args.snr_lo is not None else args.snr
    snr_hi = args.snr_hi if args.snr_hi is not None else args.snr
    for it in range(args.iters):
        snr_train = float(rng.uniform(snr_lo, snr_hi)) if snr_lo < snr_hi else snr_lo
        tok, dt, A, Y, Xc, supp, Cs, jam = gen_batch(
            cfg, args.batch, args.nC, args.K, args.steps, snr_train, args.sir, rng,
            jammer=not args.no_jammer, pre_null=not args.raw,
            burst_jammer=args.burst_jammer)
        Xhat, loss = nuaa_mu_multitask_loss(
            model, tok, dt, A, Y, Xc, supp, jam, snr_train, not args.no_jammer)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (it + 1) % max(1, args.iters // 8) == 0:
            print(f"  iter {it+1:4d}  loss={loss.item():.4f}", flush=True)

    # ---- 评估：held-out 批，NUAA-MU vs SOMP ----
    model.eval()
    rng_e = np.random.default_rng(args.seed + 999)
    tok, dt, A, Y, Xc, supp, Cs, jam = gen_batch(
        cfg, 200, args.nC, args.K, args.steps, args.snr, args.sir, rng_e,
        jammer=not args.no_jammer, pre_null=not args.raw,
        burst_jammer=args.burst_jammer)
    with torch.no_grad():
        Xhat = model(tok, dt, A, Y, refine=True)
    m_f1, m_nmse = model_f1_nmse(Xhat, Xc, supp, args.K)
    # SOMP 基线（带真值 NMSE）
    A_np = A.numpy(); Y_np = Y.numpy(); Xc_np = Xc.numpy()
    f1s, nms = [], []
    for b in range(A_np.shape[0]):
        # raw 模式下 SOMP 不获得 U_jam 预处理；对比学习式干扰消除的收益。
        sp, _ = R.somp(A_np[b], Y_np[b][:, None], args.K)
        _, _, f1 = Met.support_f1(sp, supp[b]); f1s.append(f1)
        Xh = np.zeros(args.nC, np.complex128)
        if len(sp):
            Xh[sp] = (np.linalg.pinv(A_np[b][:, sp]) @ Y_np[b][:, None])[:, 0]
        nms.append(Met.nmse(Xh[supp[b]], Xc_np[b][supp[b]]))
    s_f1 = float(np.mean(f1s)); s_nmse = float(np.median([10*np.log10(x+1e-12) for x in nms]))

    print("\n==== Eval (held-out 200) ====")
    print(f"  SOMP     : support F1={s_f1:.2f}  weak NMSE={s_nmse:.1f} dB")
    print(f"  NUAA-MU  : support F1={m_f1:.2f}  weak NMSE={m_nmse:.1f} dB")
    os.makedirs(OUT_DIR, exist_ok=True)
    res = dict(config=cfg.summary(),
               params=vars(args), n_params=int(n_params),
               somp=dict(f1=s_f1, nmse_db=s_nmse),
               nuaa_mu=dict(f1=m_f1, nmse_db=m_nmse))
    with open(os.path.join(OUT_DIR, "nuaa_mu_eval.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "nuaa_mu.pt"))
    print(f"saved {OUT_DIR}/nuaa_mu_eval.json , nuaa_mu.pt")


if __name__ == "__main__":
    main()
