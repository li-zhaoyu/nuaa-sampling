"""结构化可学习信号/干扰场景：验证 NUAA-MU 参数优势。

场景设计：
  - 有用信号：3 个宽带稀疏分量，中心频点按多谐波 + 不规则阶跃码非线性演化；
  - 干扰：强结构化 blocker，位置按不同周期码演化，功率高于信号 40 dB；
  - 突发分量：两种角色，interference=需检测并扣除，useful=需检测/重构；
  - 信道/噪声：慢变复增益 + AWGN。

对比：
  - raw SOMP：不白化、不 U_jam，仅靠本窗观测；
  - NUAA-MU：输入 raw 事件、候选频点位置和窗序相位，学习 useful/jammer/noise 先验并自投影。

用法：
  python experiments/exp_structured_nuaa_mu.py --quick
  python experiments/exp_structured_nuaa_mu.py --tag main
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
from nuaa import layout as L, reconstruct as R, metrics as Met
from models.nuaa_mu import NUAAMU, si_complex_nmse
from experiments.train_nuaa_mu import _event_tokens, make_multitask_targets

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

CAPS = {
    "small": dict(d_model=48, n_layers=2, d_state=16, K_unroll=2),
    "mid": dict(d_model=96, n_layers=2, d_state=24, K_unroll=4),
    "large": dict(d_model=192, n_layers=4, d_state=32, K_unroll=8),
}
TRAIN_SCALE = {"small": 1.0, "mid": 1.6, "large": 2.6}
LR_SCALE = {"small": 1.0, "mid": 0.8, "large": 0.35}

STEP_CODE = np.array([0, 4, -3, 7, -6, 2, 9, -8, 5, -2, 6, -5, 1, -7, 8, -4], dtype=np.float64)
JAM_CODE = np.array([5, -9, 3, 11, -6, 8, -12, 4, 10, -3, 7, -8], dtype=np.float64)


def _reflect(x: float, lo: int, hi: int) -> int:
    span = hi - lo
    z = (int(round(x)) - lo) % (2 * span)
    if z >= span:
        z = 2 * span - 1 - z
    return lo + z


def structured_centers(cfg: SystemConfig, seq_id: int, win: int, K: int):
    """多谐波 + 阶跃码的可学习非线性轨迹。"""
    guard = max(2, cfg.N0 // 50)
    lo, hi = guard, cfg.N0 - guard
    span = hi - lo
    t = float(win)
    sid = float(seq_id % 17)
    base = lo + (0.30 + 0.025 * (seq_id % 9)) * span
    h = (
        0.115 * span * np.sin(0.17 * t + 0.31 * sid)
        + 0.075 * span * np.sin(0.41 * t + 0.13 * sid)
        + 0.040 * span * np.sin(0.73 * t + 0.07 * sid)
        + 0.010 * span * STEP_CODE[win % len(STEP_CODE)]
    )
    c0 = _reflect(base + h, lo, hi)
    offsets = np.array([0, 0.055 * span, -0.043 * span], dtype=np.float64)
    return np.array([_reflect(c0 + offsets[i], lo, hi) for i in range(K)], dtype=np.int64)


def structured_jammer(cfg: SystemConfig, seq_id: int, win: int):
    guard = max(2, cfg.N0 // 50)
    lo, hi = guard, cfg.N0 - guard
    span = hi - lo
    t = float(win)
    base = lo + (0.62 + 0.017 * (seq_id % 11)) * span
    h = (
        0.18 * span * np.sin(0.23 * t + 0.19 * seq_id)
        + 0.07 * span * np.sign(np.sin(0.11 * t + 0.5))
        + 0.012 * span * JAM_CODE[win % len(JAM_CODE)]
    )
    return _reflect(base + h, lo, hi)


def make_candidate_set(cfg, support, jam, nC, rng):
    guard = max(2, cfg.N0 // 50)
    required = np.unique(np.concatenate([support, np.array([jam], dtype=np.int64)]))
    pool = np.setdiff1d(np.arange(guard, cfg.N0 - guard), required)
    extra = rng.choice(pool, size=max(0, nC - len(required)), replace=False)
    return np.sort(np.concatenate([required, extra]).astype(np.int64))


def candidate_features(C, cfg, win, period):
    ph = 2 * np.pi * (win % period) / float(period)
    feat = np.zeros((len(C), 3), np.float32)
    feat[:, 0] = C.astype(np.float32) / float(cfg.N0)
    feat[:, 1] = np.sin(ph)
    feat[:, 2] = np.cos(ph)
    return feat


def _append_window_phase(tok8, win, period):
    ph = 2 * np.pi * (win % period) / float(period)
    extra = np.zeros((tok8.shape[0], 2), np.float32)
    extra[:, 0] = np.sin(ph)
    extra[:, 1] = np.cos(ph)
    return np.concatenate([tok8, extra], axis=1)


def gen_structured_batch(cfg, B, nC, K, steps, snr_db, sir_db, rng,
                         W=32, period=32, burst_role="interference"):
    Rmeas = cfg.L * steps
    tok = np.zeros((B, Rmeas, 10), np.float32)
    dt = np.zeros((B, Rmeas), np.float32)
    A = np.zeros((B, Rmeas, nC), np.complex64)
    Y = np.zeros((B, Rmeas), np.complex64)
    Xc = np.zeros((B, nC), np.complex64)
    cand = np.zeros((B, nC, 3), np.float32)
    burst_mask = np.zeros((B, Rmeas), np.float32)
    burst_sig = np.zeros((B, Rmeas), np.complex64)
    supp_local = np.zeros((B, K), np.int64)
    jam_local = np.zeros(B, np.int64)
    for b in range(B):
        seq_id = int(rng.integers(0, 100000))
        win = int(rng.integers(0, W))
        support = structured_centers(cfg, seq_id, win, K)
        jam = structured_jammer(cfg, seq_id, win)
        while jam in set(int(s) for s in support):
            jam = int(np.clip(jam + cfg.N0 // 80, 1, cfg.N0 - 2))
        C = make_candidate_set(cfg, support, jam, nC, rng)
        loc = np.array([int(np.where(C == s)[0][0]) for s in support], dtype=np.int64)
        jam_loc = int(np.where(C == jam)[0][0])
        x = np.zeros(nC, np.complex128)
        # 慢变信道：复增益随窗相位规律变化，可由模型学习为场景先验的一部分。
        gain = 1.0 + 0.25 * np.exp(1j * (0.37 * win + 0.11 * (seq_id % 13)))
        x[loc] = gain * (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
        x[jam_loc] = np.sqrt(10.0 ** (-sir_db / 10.0)) * np.exp(1j * (0.29 * win + seq_id))
        cosets = np.sort(np.concatenate([L.gen_fixed_random(cfg, rng) for _ in range(steps)]))
        Ac = np.exp(2j * np.pi * np.outer(cosets, C) / cfg.N0) / np.sqrt(Rmeas)
        y_clean = Ac @ x
        ref = float(np.mean(np.abs(Ac[:, loc] @ x[loc]) ** 2)) + 1e-30
        noise = np.sqrt(ref * 10.0 ** (-snr_db / 10.0) / 2) * (
            rng.standard_normal(Rmeas) + 1j * rng.standard_normal(Rmeas))
        y = y_clean + noise
        if burst_role != "none":
            phase_hot = ((win + np.arange(Rmeas)) % 7) == (seq_id % 7)
            if np.any(phase_hot):
                burst_pow = ref * 10.0 ** (35.0 / 10.0)
                envelope = 0.75 + 0.25 * np.sin(2 * np.pi * (cosets / cfg.N0) + 0.23 * win)
                phase = 2 * np.pi * (0.37 * cosets / cfg.N0 + 0.09 * win + 0.03 * (seq_id % 23))
                b_sig = phase_hot.astype(np.float64) * np.sqrt(burst_pow) * envelope * np.exp(1j * phase)
                y = y + b_sig
                burst_mask[b] = phase_hot.astype(np.float32)
                burst_sig[b] = b_sig.astype(np.complex64)
        tok[b] = _append_window_phase(_event_tokens(y, cosets, cfg), win, period)
        d = np.diff(cosets, prepend=cosets[0]).astype(np.float64) / max(1, cfg.N0 // cfg.L)
        dt[b] = np.log1p(np.abs(d))
        A[b] = Ac.astype(np.complex64)
        Y[b] = y.astype(np.complex64)
        Xc[b] = x.astype(np.complex64)
        cand[b] = candidate_features(C, cfg, win, period)
        supp_local[b] = loc
        jam_local[b] = jam_loc
    return (torch.tensor(tok), torch.tensor(dt), torch.tensor(A), torch.tensor(Y),
            torch.tensor(Xc), torch.tensor(cand), torch.tensor(burst_mask),
            torch.tensor(burst_sig), supp_local, jam_local)


def train_model(cap, cfg, args, seed):
    hp = CAPS[cap]
    model = NUAAMU(d_in=10, nC=args.nC, K_sparse=args.K, cand_dim=3, **hp)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr * LR_SCALE[cap])
    rng = np.random.default_rng(seed)
    has_burst = args.burst_role != "none"
    model.train()
    n_iters = int(args.iters * TRAIN_SCALE[cap])
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    for it in range(n_iters):
        snr = float(rng.uniform(args.snr_lo, args.snr_hi))
        train_steps = int(rng.choice(args.steps_list)) if getattr(args, "steps_list", None) else args.steps
        tok, dt, A, Y, Xc, cand, bmask, bsig, supp, jam = gen_structured_batch(
            cfg, args.batch, args.nC, args.K, train_steps, snr, args.sir, rng,
            W=args.W, period=args.period, burst_role=args.burst_role)
        Xhat, aux = model(tok, dt, A, Y, refine=False, return_aux=True,
                          cand_feat=cand, use_burst=has_burst)
        X_target, support_target, jammer_target, supp_t = make_multitask_targets(Xc, supp, jam, True)
        rec = si_complex_nmse(torch.gather(Xhat, 1, supp_t), torch.gather(Xc, 1, supp_t))
        rec = rec + 0.15 * si_complex_nmse(Xhat, X_target)
        loss = rec
        loss = loss + 0.16 * F.binary_cross_entropy_with_logits(aux["support_logits"], support_target)
        loss = loss + 0.12 * F.binary_cross_entropy_with_logits(aux["jammer_logits"], jammer_target)
        if has_burst:
            burst_bce = F.binary_cross_entropy_with_logits(aux["burst_logits"], bmask)
            event_target = 1.0 - bmask
            event_bce = F.binary_cross_entropy_with_logits(aux["event_logits"], event_target)
            bprob = torch.sigmoid(aux["burst_logits"])
            bpred = bprob.to(bsig.dtype) * aux["burst_complex"]
            denom = bsig.abs().pow(2).sum(dim=1).clamp_min(1e-9)
            berr = ((bpred - bsig).abs().pow(2) * bmask).sum(dim=1) / denom
            if args.burst_role == "interference":
                loss = loss + 0.18 * burst_bce + 0.18 * event_bce + 0.03 * berr.mean()
            else:
                loss = loss + 0.10 * burst_bce + 0.05 * event_bce + 0.05 * berr.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if progress_every and ((it + 1) % progress_every == 0 or (it + 1) == n_iters):
            print(f"PROGRESS train cap={cap} iter={it+1}/{n_iters} loss={float(loss.detach().cpu()):.4f}", flush=True)
    model.eval()
    return model


def _somp_eval(A, Y, Xc, supp, K):
    sp, _ = R.somp(A, Y[:, None], K)
    _, _, f1 = Met.support_f1(sp, supp)
    Xs = np.zeros(Xc.shape[0], np.complex128)
    if len(sp):
        Xs[sp] = (np.linalg.pinv(A[:, sp]) @ Y[:, None])[:, 0]
    return f1, Met.nmse(Xs[supp], Xc[supp])


def _robust_somp_eval(A, Y, Xc, supp, K, tok):
    """简单非学习基线：删去异常度最高的事件后 SOMP。"""
    anomaly = np.asarray(tok[:, 3], dtype=np.float64)
    keep_n = max(K + 2, int(np.ceil(0.75 * len(anomaly))))
    keep = np.argsort(anomaly)[:keep_n]
    return _somp_eval(A[keep], Y[keep], Xc, supp, K)


def eval_model(model, cfg, args, seed, n=200, steps_override=None):
    rng = np.random.default_rng(seed)
    steps = int(steps_override if steps_override is not None else args.steps)
    has_burst = args.burst_role != "none"
    tok, dt, A, Y, Xc, cand, bmask, bsig, supp, _jam = gen_structured_batch(
        cfg, n, args.nC, args.K, steps, args.eval_snr, args.sir, rng,
        W=args.W, period=args.period, burst_role=args.burst_role)
    with torch.no_grad():
        Xhat, aux = model(tok, dt, A, Y, refine=True, return_aux=True,
                          cand_feat=cand, use_burst=has_burst)
    Xh = Xhat.detach().cpu().numpy()
    Xt = Xc.numpy()
    A_np, Y_np, tok_np = A.numpy(), Y.numpy(), tok.numpy()
    mt_f1, mt_nmse, somp_f1, somp_nmse = [], [], [], []
    robust_f1, robust_nmse = [], []
    burst_f1, burst_nmse = [], []
    if has_burst:
        bprob = torch.sigmoid(aux["burst_logits"]).cpu().numpy()
        brec = (torch.sigmoid(aux["burst_logits"]).to(aux["burst_complex"].dtype)
                * aux["burst_complex"]).detach().cpu().numpy()
        bm = bmask.numpy()
        bt = bsig.numpy()
    for b in range(n):
        est = np.argsort(np.abs(Xh[b]))[-args.K:]
        _, _, f1 = Met.support_f1(est, supp[b])
        mt_f1.append(f1)
        mt_nmse.append(Met.nmse(Xh[b][supp[b]], Xt[b][supp[b]]))
        f1s, nms = _somp_eval(A_np[b], Y_np[b], Xt[b], supp[b], args.K)
        somp_f1.append(f1s)
        somp_nmse.append(nms)
        f1r, nmr = _robust_somp_eval(A_np[b], Y_np[b], Xt[b], supp[b], args.K, tok_np[b])
        robust_f1.append(f1r)
        robust_nmse.append(nmr)
        if has_burst:
            est_b = np.where(bprob[b] >= 0.5)[0]
            true_b = np.where(bm[b] > 0.5)[0]
            _, _, bf = Met.support_f1(est_b, true_b)
            burst_f1.append(bf)
            if len(true_b):
                burst_nmse.append(Met.nmse(brec[b][true_b], bt[b][true_b]))
    return dict(
        mt_f1=float(np.mean(mt_f1)),
        mt_nmse=float(np.median([10 * np.log10(x + 1e-12) for x in mt_nmse])),
        somp_f1=float(np.mean(somp_f1)),
        somp_nmse=float(np.median([10 * np.log10(x + 1e-12) for x in somp_nmse])),
        robust_f1=float(np.mean(robust_f1)),
        robust_nmse=float(np.median([10 * np.log10(x + 1e-12) for x in robust_nmse])),
        burst_f1=float(np.mean(burst_f1)) if burst_f1 else float("nan"),
        burst_nmse=float(np.median([10 * np.log10(x + 1e-12) for x in burst_nmse])) if burst_nmse else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--nC", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--steps-list", type=int, nargs="+", default=None,
                    help="评估采样周期数列表；events=L*steps")
    ap.add_argument("--sir", type=float, default=-40.0)
    ap.add_argument("--snr-lo", type=float, default=-6.0)
    ap.add_argument("--snr-hi", type=float, default=8.0)
    ap.add_argument("--eval-snr", type=float, default=3.0)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--iters", type=int, default=220)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-n", type=int, default=240)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burst-role", choices=["none", "interference", "useful"],
                    default="interference")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", type=str, default="structured")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.batch, args.eval_n = 80, 24, 120
        args.nC = min(args.nC, 48)
        args.steps = max(args.steps, 4)
        if args.steps_list is None:
            args.steps_list = [2, 3, 5]
    if args.steps_list is None:
        args.steps_list = [args.steps]
    args.steps = max(args.steps, max(args.steps_list))
    cfg = SystemConfig(N0=args.n0)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"config": cfg.summary(), "params": vars(args), "capacity": {}}
    print(f"structured NUAA-MU | nC={args.nC} K={args.K} train_events={cfg.L*args.steps} "
          f"eval_steps={args.steps_list} SIR={args.sir} eval_snr={args.eval_snr} "
          f"burst_role={args.burst_role}", flush=True)
    for cap in CAPS:
        model = train_model(cap, cfg, args, args.seed + 17)
        n_params = int(sum(p.numel() for p in model.parameters()))
        sweep = {}
        print(f"[{cap:5s}] params={n_params/1e3:.1f}K", flush=True)
        for st in args.steps_list:
            ev = eval_model(model, cfg, args, args.seed + 100 + st, n=args.eval_n,
                            steps_override=st)
            sweep[str(st)] = dict(events=cfg.L * st, **ev)
            print(
                f"  events={cfg.L*st:2d} MT {ev['mt_f1']:.3f}/{ev['mt_nmse']:+.1f} "
                f"SOMP {ev['somp_f1']:.3f}/{ev['somp_nmse']:+.1f} "
                f"robust {ev['robust_f1']:.3f}/{ev['robust_nmse']:+.1f} "
                f"burst {ev['burst_f1']:.3f}/{ev['burst_nmse']:+.1f}",
                flush=True,
            )
        results["capacity"][cap] = dict(n_params=n_params, **CAPS[cap], sweep=sweep)
    out = os.path.join(OUT_DIR, f"structured_nuaa_mu_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
