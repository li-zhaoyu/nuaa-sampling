"""E4（新增主打）：准周期演化信号的「离线场景预训练 + 在线演化预测」验证。

复现 计算光采样.md §准周期演化信号族 / §两级先验 / §8 E4 的核心论证：
  在**可预测演化**信号上，离线预训练的演化预测器（θ_class）能预测下一窗占用，
  形成预测性支撑先验 → 等事件预算下重构更优、且增益随模型容量单调上升；
  在**随机对照**上，三档先验与各容量趋同（容量–增益关系是信号结构相关的）。

两部分：
  Part 1 演化预测质量：预训练中心预测器，验证 MAE vs 持续性基线，扫容量 × 可预测性。
  Part 2 下游重构：online-only(持续性) | pretrained-evo(预测) | oracle(已知) 在紧预算下的 F1/NMSE。

用法：
  python experiments/exp_e4_evolution.py --quick
  python experiments/exp_e4_evolution.py --kinds lfm fh thz --trials 40
  python experiments/exp_e4_evolution.py --paper --tag paper_n5000   # N0=5000 + 扫描
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure
configure()

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from nuaa.config import SystemConfig
from nuaa import measurement as M, signals as S, reconstruct as R, policy as P, metrics as Met
from models.evolution_prior import pretrain_evolution, predict_next_centers

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


# --------------------------------------------------------------------------
def _gen_centers(cfg, kind, W, n_seq, predictability, seed,
                 drift_scale=1.0, epsilon_frac=0.0):
    """批量生成中心轨迹 (n_seq, W)（仅取演化潜参数，供预测器训练/评估）。"""
    centers = np.zeros((n_seq, W), dtype=np.int64)
    for i in range(n_seq):
        rng = np.random.default_rng(seed + i)
        seq = S.gen_evolving_sequence(cfg, kind, W, J=1, predictability=predictability,
                                      drift_scale=drift_scale, epsilon_frac=epsilon_frac, rng=rng)
        centers[i] = seq.center_seq
    return centers


def measure_window(X, support, cfg, traj, snr_db, rng):
    A, uniq = M.stack_phi(traj, cfg.N0)
    Yc = A @ X
    Asig = M.build_A_spec(uniq, cfg.N0)[:, support]
    ref = float(np.mean(np.abs(Asig @ X[support]) ** 2)) + 1e-30
    Y = S.add_measurement_noise(Yc, snr_db, ref, rng)
    return uniq, Y


def recon_restricted(uniq, Y, X, support, C, K, cfg):
    """在候选集 C 上 SOMP 恢复 K 个子带，返回 (F1, NMSE_dB, est_support)。"""
    C = np.asarray(C, int)
    A_full = M.build_A_spec(uniq, cfg.N0)
    A_C = A_full[:, C]
    sp, _ = R.somp(A_C, Y, K)
    est = C[sp]
    _, _, f1 = Met.support_f1(est, support)
    Xhat = np.zeros_like(X)
    if len(est):
        As = A_C[:, sp]
        Xhat[est, :] = np.linalg.pinv(As) @ Y
    nm = Met.weak_signal_nmse(Xhat, X, support)
    return f1, 10 * np.log10(nm + 1e-12), est


def _cand_window(center, w, cfg):
    guard = max(2, cfg.N0 // 50)
    lo, hi = guard, cfg.N0 - guard
    if center is None:
        return np.arange(lo, hi)
    c = int(round(center))
    idx = (np.arange(c - w, c + w + 1) - lo) % (hi - lo) + lo   # 以 c 为中心、在 [lo,hi) 内环绕
    return np.unique(idx)


def _span_slots(cfg):
    guard = max(2, cfg.N0 // 50)
    span = cfg.N0 - 2 * guard
    return guard, span


def _cand_from_frac(cfg, frac):
    return max(3, int(frac * _span_slots(cfg)[1]))


def _mean_step_slots(center_seq) -> float:
    c = np.asarray(center_seq, dtype=np.float64).reshape(-1)
    if c.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(c))))


def _train_big_model(cfg, kind, W, n_train, seed):
    big = dict(hidden=96, layers=2, epochs=80)
    tr = _gen_centers(cfg, kind, W, n_train, "pred", seed=10_000)
    va = _gen_centers(cfg, kind, W, max(16, n_train // 8), "pred", seed=90_000)
    return pretrain_evolution(tr, va, cfg.N0, hidden=big["hidden"],
                              n_layers=big["layers"], epochs=big["epochs"], seed=seed).model


def _eval_part2_grid(cfg, kind, W, J, snr_db, steps, cand_w, trials, seed, model,
                     drift_scale, epsilon_frac, warmup=3):
    methods = ["online_only", "pretrained_evo", "oracle"]
    agg = {m: {"f1": [], "hit": []} for m in methods}
    for t in range(trials):
        rng = np.random.default_rng(seed + 9000 + t)
        seq = S.gen_evolving_sequence(cfg, kind, W, J=J, predictability="pred",
                                      drift_scale=drift_scale, epsilon_frac=epsilon_frac, rng=rng)
        true_c = seq.center_seq.astype(np.float64)
        for r in range(warmup, W):
            Xr, sup = seq.X_seq[r], seq.support_seq[r]
            K = len(sup)
            hist = true_c[None, :r]
            preds = {
                "oracle": true_c[r],
                "online_only": true_c[r - 1],
                "pretrained_evo": float(predict_next_centers(model, hist, cfg.N0)[0, -1]),
            }
            for m in methods:
                C = _cand_window(preds[m], cand_w, cfg)
                hit = 1.0 if int(round(true_c[r])) in set(C.tolist()) else 0.0
                traj = P.random_scan_trajectory(cfg, steps, rng)
                uniq, Y = measure_window(Xr, sup, cfg, traj, snr_db, rng)
                f1, _, _ = recon_restricted(uniq, Y, Xr, sup, C, K, cfg)
                agg[m]["f1"].append(f1); agg[m]["hit"].append(hit)
    return {m: dict(f1_mean=float(np.mean(agg[m]["f1"])),
                    hit_rate=float(np.mean(agg[m]["hit"]))) for m in methods}


def run_drift_cand_sweep(cfg, kind, W, J, snr_db, steps, trials, seed, n_train,
                         cand_fracs, drift_scales):
    model = _train_big_model(cfg, kind, W, n_train, seed)
    rows = []
    for ds in drift_scales:
        for cf in cand_fracs:
            cw = _cand_from_frac(cfg, cf)
            ev = _eval_part2_grid(cfg, kind, W, J, snr_db, steps, cw, trials, seed, model,
                                  drift_scale=ds, epsilon_frac=0.0)
            rng = np.random.default_rng(seed + int(ds * 1000) + int(cf * 10000))
            seq = S.gen_evolving_sequence(cfg, kind, W, predictability="pred",
                                            drift_scale=ds, rng=rng)
            step = _mean_step_slots(seq.center_seq)
            ratio = step / max(2 * cw, 1)
            row = dict(drift_scale=ds, cand_frac=cf, cand_w=cw,
                       mean_step_slots=step, drift_cand_ratio=ratio)
            for k, v in ev.items():
                row[f"{k}_f1"] = v["f1_mean"]; row[f"{k}_hit"] = v["hit_rate"]
            rows.append(row)
            print(f"[drift-cand {kind}] ds={ds:.2f} cf={cf:.3f} ratio={ratio:.2f} "
                  f"on={ev['online_only']['f1_mean']:.2f} pre={ev['pretrained_evo']['f1_mean']:.2f} "
                  f"ora={ev['oracle']['f1_mean']:.2f}", flush=True)
    return {"kind": kind, "rows": rows}


def run_epsilon_sweep(cfg, kind, W, J, snr_db, steps, trials, seed, n_train,
                      epsilon_fracs, cand_frac=0.03, drift_scale=1.0):
    model = _train_big_model(cfg, kind, W, n_train, seed)
    cw = _cand_from_frac(cfg, cand_frac)
    rows = []
    for ef in epsilon_fracs:
        ev = _eval_part2_grid(cfg, kind, W, J, snr_db, steps, cw, trials, seed, model,
                              drift_scale=drift_scale, epsilon_frac=ef)
        row = dict(epsilon_frac=ef, cand_w=cw, drift_scale=drift_scale)
        for k, v in ev.items():
            row[f"{k}_f1"] = v["f1_mean"]; row[f"{k}_hit"] = v["hit_rate"]
        rows.append(row)
        print(f"[epsilon {kind}] ε={ef:.3f} on={ev['online_only']['f1_mean']:.2f} "
              f"pre={ev['pretrained_evo']['f1_mean']:.2f} ora={ev['oracle']['f1_mean']:.2f}", flush=True)
    return {"kind": kind, "cand_frac": cand_frac, "rows": rows}


def run_estimated_history(cfg, kind, W, J, snr_db, steps, cand_w, trials, seed, n_train,
                          drift_scale=1.0, warmup=3):
    """W#1：用前窗 estimated center（来自实际重构输出，带误差传播）替代真值历史。

    对比 true-history（隔离上界）vs estimated-history（部署级闭环近似）：
      冷启动 warmup 窗用真值历史（假设正常预算可跟踪），之后**纯用上一窗重构得到的
      支撑中心**作为历史输入预测器，逐窗误差自累积。验证 §7.8 W#1：去掉 oracle 历史后
      演化先验优势是否保持。
    """
    model = _train_big_model(cfg, kind, W, n_train, seed)
    methods = ["online_only", "pretrained_evo"]
    out = {}
    for hist_mode in ["true", "estimated"]:
        agg = {m: [] for m in methods}
        for t in range(trials):
            rng = np.random.default_rng(seed + 9000 + t)
            seq = S.gen_evolving_sequence(cfg, kind, W, J=J, predictability="pred",
                                          drift_scale=drift_scale, epsilon_frac=0.0, rng=rng)
            true_c = seq.center_seq.astype(np.float64)
            for m in methods:
                est_hist = list(true_c[:warmup])
                for r in range(warmup, W):
                    Xr, sup = seq.X_seq[r], seq.support_seq[r]
                    K = len(sup)
                    hist_arr = (true_c[:r] if hist_mode == "true"
                                else np.asarray(est_hist, dtype=np.float64))
                    if m == "online_only":
                        pred = float(hist_arr[-1])
                    else:
                        pred = float(predict_next_centers(model, hist_arr[None, :], cfg.N0)[0, -1])
                    C = _cand_window(pred, cand_w, cfg)
                    traj = P.random_scan_trajectory(cfg, steps, rng)
                    uniq, Y = measure_window(Xr, sup, cfg, traj, snr_db, rng)
                    f1, _, est = recon_restricted(uniq, Y, Xr, sup, C, K, cfg)
                    agg[m].append(f1)
                    est_hist.append(float(np.median(est)) if len(est) else pred)
        out[hist_mode] = {m: float(np.mean(agg[m])) for m in methods}
        print(f"[esthist {kind} {hist_mode}] online={out[hist_mode]['online_only']:.2f} "
              f"pre={out[hist_mode]['pretrained_evo']:.2f}", flush=True)
    return {"kind": kind, "drift_scale": drift_scale, "cand_w": cand_w, **out}


def _plot_drift_cand(sweep_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})"); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for kind, data in sweep_res.items():
        rows = data["rows"]
        ratios = [r["drift_cand_ratio"] for r in rows]
        ax[0].scatter(ratios, [r["online_only_f1"] for r in rows], s=18, alpha=0.45, label=f"{kind} online")
        ax[0].scatter(ratios, [r["pretrained_evo_f1"] for r in rows], s=22, alpha=0.75, marker="^",
                      label=f"{kind} pre")
        ax[1].scatter(ratios, [r["pretrained_evo_f1"] - r["online_only_f1"] for r in rows],
                      s=25, alpha=0.75, label=kind)
    ax[0].set(xlabel="mean|Δc| / (2·cand_w)", ylabel="support F1", title="F1 vs drift–cand ratio")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set(xlabel="mean|Δc| / (2·cand_w)", ylabel="ΔF1 (pre − online)", title="Prior gain")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    for sp in ax: sp.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt2; plt2.close(fig)
    print(f"saved {path}")


def _plot_epsilon(eps_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})"); return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = [("online_only", "o-"), ("pretrained_evo", "^-"), ("oracle", "s--")]
    for kind, data in eps_res.items():
        rows = data["rows"]
        xs = [r["epsilon_frac"] for r in rows]
        for m, sty in styles:
            ax.plot(xs, [r[f"{m}_f1"] for r in rows], sty, ms=5, label=f"{kind} {m}")
    ax.set(xlabel="ε_r std (× span)", ylabel="support F1",
           title="Innovation noise ε_r → asymptotic floor", ylim=(0, 1.05))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt2; plt2.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------
def run_kind(cfg, kind, W, J, snr_db, steps, cand_w, n_train, n_val, trials,
             capacities, seed):
    """对单一场景类跑 Part1（容量×可预测性预测质量）+ Part2（下游重构）。"""
    res = {"kind": kind, "part1_prediction": {}, "part2_recon": {}}

    # ---- Part 1：容量 × 可预测性 的中心预测质量 ----
    for pred_lvl in ["pred", "semi", "random"]:
        tr = _gen_centers(cfg, kind, W, n_train, pred_lvl, seed=10_000)
        va = _gen_centers(cfg, kind, W, n_val, pred_lvl, seed=90_000)
        cap_rows = {}
        for cap in capacities:
            pr = pretrain_evolution(tr, va, cfg.N0, hidden=cap["hidden"],
                                    n_layers=cap["layers"], epochs=cap["epochs"], seed=seed)
            cap_rows[cap["name"]] = dict(
                n_params=pr.n_params, train_loss=pr.train_loss,
                pred_mae_slots=pr.val_pred_mae_slots,
                persistence_mae_slots=pr.persistence_mae_slots,
                gain_over_persistence=pr.persistence_mae_slots - pr.val_pred_mae_slots)
        res["part1_prediction"][pred_lvl] = cap_rows
        line = " | ".join(f"{n}:MAE={r['pred_mae_slots']:.1f}(pers={r['persistence_mae_slots']:.1f})"
                          for n, r in cap_rows.items())
        print(f"[{kind} P1 {pred_lvl:6s}] {line}", flush=True)

    # 取“大”容量模型做 Part2 的 pretrained-evo 先验（在 pred / random 上各训一个）
    big = capacities[-1]
    models = {}
    for pred_lvl in ["pred", "random"]:
        tr = _gen_centers(cfg, kind, W, n_train, pred_lvl, seed=10_000)
        va = _gen_centers(cfg, kind, W, max(8, n_val // 4), pred_lvl, seed=90_000)
        models[pred_lvl] = pretrain_evolution(tr, va, cfg.N0, hidden=big["hidden"],
                                              n_layers=big["layers"], epochs=big["epochs"],
                                              seed=seed).model

    # ---- Part 2：下游重构（紧预算）----
    # 设定：过去窗在正常工况下已采集/可跟踪（用真值中心作历史），仅对**即将到来**的窗
    # 以紧事件预算重构，比较先验中心来源：持续性 vs 演化预测 vs oracle。
    # 这隔离「演化预测→紧先验预置」的独立收益（理想化：历史可跟踪；见文档诚实边界）。
    methods = ["online_only", "pretrained_evo", "oracle"]
    warmup = 3
    for pred_lvl in ["pred", "random"]:
        agg = {m: {"f1": [], "nmse": [], "hit": []} for m in methods}
        model = models[pred_lvl]
        for t in range(trials):
            rng = np.random.default_rng(seed + 5000 + t)
            seq = S.gen_evolving_sequence(cfg, kind, W, J=J, predictability=pred_lvl, rng=rng)
            true_c = seq.center_seq.astype(np.float64)
            for r in range(warmup, W):
                Xr, sup = seq.X_seq[r], seq.support_seq[r]
                K = len(sup)
                hist = true_c[None, :r]                      # 可跟踪的历史（真值，到窗 r-1）
                preds = {
                    "oracle": true_c[r],
                    "online_only": true_c[r - 1],            # 持续性：沿用上一窗
                    "pretrained_evo": float(predict_next_centers(model, hist, cfg.N0)[0, -1]),
                }
                for m in methods:
                    C = _cand_window(preds[m], cand_w, cfg)
                    hit = 1.0 if int(round(true_c[r])) in set(C.tolist()) else 0.0
                    traj = P.random_scan_trajectory(cfg, steps, rng)
                    uniq, Y = measure_window(Xr, sup, cfg, traj, snr_db, rng)
                    f1, nm, _ = recon_restricted(uniq, Y, Xr, sup, C, K, cfg)
                    agg[m]["f1"].append(f1); agg[m]["nmse"].append(nm); agg[m]["hit"].append(hit)
        res["part2_recon"][pred_lvl] = {
            m: dict(events=cfg.L * steps,
                    f1_mean=float(np.mean(agg[m]["f1"])),
                    nmse_med=float(np.median(agg[m]["nmse"])),
                    center_hit_rate=float(np.mean(agg[m]["hit"]))) for m in methods}
        line = " | ".join(f"{m}:F1={res['part2_recon'][pred_lvl][m]['f1_mean']:.2f}"
                          f",hit={res['part2_recon'][pred_lvl][m]['center_hit_rate']:.2f}"
                          f",NMSE={res['part2_recon'][pred_lvl][m]['nmse_med']:.1f}dB" for m in methods)
        print(f"[{kind} P2 {pred_lvl:6s}] events={cfg.L*steps} {line}", flush=True)
    return res


# --------------------------------------------------------------------------
# 在线学习曲线（对应 §在线先验积累的时间增益）
# --------------------------------------------------------------------------
def _unwrap_slots(w: np.ndarray, span: int) -> np.ndarray:
    """对环绕（mod span）的中心序列解卷绕（假设真步长 < span/2）。"""
    u = np.empty_like(w, dtype=np.float64)
    u[0] = w[0]
    for i in range(1, len(w)):
        d = w[i] - w[i - 1]
        d -= span * round(d / span)
        u[i] = u[i - 1] + d
    return u


def _poly_extrap(hist_unwrapped: np.ndarray, order: int = 2) -> float:
    """对解卷绕历史做多项式 LS 拟合外推下一窗；历史越长拟合方差越小（在线积累）。"""
    n = len(hist_unwrapped)
    if n < 2:
        return float(hist_unwrapped[-1]) if n else 0.0
    o = min(order, n - 1)
    t = np.arange(n)
    coef = np.polyfit(t, hist_unwrapped, o)
    return float(np.polyval(coef, n))


def _gen_curve_centers(cfg, kind, n_seq, W, seed):
    """圆周演化中心轨迹（频率 mod span，大幅持续漂移无折返）：(n_seq, W) 解卷绕真值。"""
    guard = max(2, cfg.N0 // 50)
    span = cfg.N0 - 2 * guard
    unit = span / 100.0
    out = np.zeros((n_seq, W), dtype=np.float64)
    for i in range(n_seq):
        rng = np.random.default_rng(seed + i)
        c0 = rng.uniform(0.2 * span, 0.6 * span)
        if kind == "lfm":          # 二阶：增量线性增长
            v = rng.choice([-1, 1]) * (4.0 * unit + rng.uniform(0, 1) * unit)
            a = rng.choice([-1, 1]) * (3.0 * unit / max(W - 1, 1))
            inc = v + a * np.arange(W)
        elif kind == "fh":         # 线性 + 类级共享周期跳码
            drift = rng.choice([-1, 1]) * 6.0 * unit
            inc = drift + 2.0 * unit * S._FH_PRN_PATTERN[np.arange(W) % S._FH_PRN_PERIOD]
        else:                       # thz: 线性载频漂移
            inc = np.full(W, rng.choice([-1, 1]) * 5.0 * unit)
        out[i] = c0 + np.cumsum(inc)         # 解卷绕真值（可超出带宽，物理上 mod span）
    return out, guard, span


def run_curve(cfg, kind, W, J, snr_db, steps, cand_w, n_train, trials, seed,
              track_sigma_frac=0.012):
    """在线学习曲线：固定事件预算下逐窗 F1 vs 窗序 r（对应 §在线先验积累的时间增益）。

    方法（均用同一紧候选半宽，差别仅在"下一窗中心"来源）：
      - no_accum  : 窗受限"持续性"（仅看上一窗，不积累）→ 漂移>候选宽 → 恒定低（其他算法）。
      - online_accum: 在线积累（解卷绕历史多项式 LS 外推），无离线预训练 → 随 r 单调改善。
      - two_tier  : 离线预训练 GRU + 在线历史 → 起点更高/更快触底。
    历史含跟踪抖动 track_sigma；更多历史 → 拟合方差下降 → 预测变准（积累体现）。
    """
    big = dict(hidden=96, layers=2, epochs=80)
    tr, _, _ = _gen_curve_centers(cfg, kind, n_train, W, seed=10_000)
    va, _, _ = _gen_curve_centers(cfg, kind, 16, W, seed=90_000)
    model = pretrain_evolution(tr, va, cfg.N0, hidden=big["hidden"],
                               n_layers=big["layers"], epochs=big["epochs"], seed=seed).model
    track_sigma = track_sigma_frac * cfg.N0
    guard = max(2, cfg.N0 // 50)
    span = cfg.N0 - 2 * guard
    order = 2 if kind == "lfm" else 1
    methods = ["no_accum", "online_accum", "two_tier"]
    warmup = 3
    f1_curve = {m: np.zeros(W) for m in methods}
    cnt = np.zeros(W)
    cen_seqs, _, _ = _gen_curve_centers(cfg, kind, trials, W, seed=seed + 3000)
    for t in range(trials):
        rng = np.random.default_rng(seed + 4000 + t)
        true_u = cen_seqs[t]                                   # 解卷绕真值
        wrapped = guard + (true_u - guard) % span              # 物理环绕中心
        obs_u = _unwrap_slots(wrapped + track_sigma * rng.standard_normal(W), span)  # 含抖动的解卷绕观测
        for r in range(warmup, W):
            band = int(round(guard + (true_u[r] - guard) % span))
            band = int(np.clip(band, guard, cfg.N0 - guard - 1))
            Xr = np.zeros((cfg.N0, J), dtype=np.complex128)
            Xr[band, :] = (rng.standard_normal((1, J)) + 1j * rng.standard_normal((1, J))) / np.sqrt(2)
            sup = np.array([band])
            preds_u = {
                "no_accum": obs_u[r - 1],
                "online_accum": _poly_extrap(obs_u[:r], order=order),
                "two_tier": float(predict_next_centers(model, obs_u[None, :r], cfg.N0)[0, -1]),
            }
            for m in methods:
                cw = guard + (preds_u[m] - guard) % span        # 预测中心 wrap 回带内
                C = _cand_window(cw, cand_w, cfg)
                traj = P.random_scan_trajectory(cfg, steps, rng)
                uniq, Y = measure_window(Xr, sup, cfg, traj, snr_db, rng)
                f1, _, _ = recon_restricted(uniq, Y, Xr, sup, C, 1, cfg)
                f1_curve[m][r] += f1
            cnt[r] += 1
    sel = cnt > 0
    out = {m: dict(r=np.where(sel)[0].tolist(), f1=(f1_curve[m][sel] / cnt[sel]).tolist())
           for m in methods}
    rs = np.where(sel)[0]
    early = rs[: max(1, len(rs) // 3)]; late = rs[-max(1, len(rs) // 3):]
    summ = {m: dict(f1_early=float(np.mean(f1_curve[m][early] / cnt[early])),
                    f1_late=float(np.mean(f1_curve[m][late] / cnt[late]))) for m in methods}
    for m in methods:
        summ[m]["f1_improve"] = summ[m]["f1_late"] - summ[m]["f1_early"]
    print(f"[{kind} curve] events={cfg.L*steps} " + " | ".join(
        f"{m}:F1 {summ[m]['f1_early']:.2f}→{summ[m]['f1_late']:.2f}(Δ{summ[m]['f1_improve']:+.2f})"
        for m in methods), flush=True)
    return {"events": cfg.L * steps, "curve": out, "summary": summ}


def _plot_curve(curve_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})"); return
    kinds = list(curve_res.keys())
    fig, ax = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4.0), squeeze=False)
    colors = {"no_accum": "#c44", "online_accum": "#48c", "two_tier": "#4a4"}
    for j, kind in enumerate(kinds):
        cv = curve_res[kind]["curve"]
        for m, c in colors.items():
            ax[0][j].plot(cv[m]["r"], cv[m]["f1"], "-o", ms=3, color=c, label=m)
        ax[0][j].set(xlabel="deployment window index r", ylabel="support F1",
                     title=f"E4-curve {kind}: F1 vs time (fixed budget)", ylim=(0, 1.05))
        ax[0][j].legend(fontsize=8); ax[0][j].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"saved {path}")


def _plot(all_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})"); return
    kinds = list(all_res.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左：Part1 大容量在 pred vs random 的预测增益（相对持续性）
    for kind in kinds:
        caps = all_res[kind]["part1_prediction"]["pred"]
        names = list(caps.keys())
        params = [caps[n]["n_params"] for n in names]
        gains = [caps[n]["gain_over_persistence"] for n in names]
        ax[0].plot(params, gains, "-o", label=f"{kind} (pred)")
        capsr = all_res[kind]["part1_prediction"]["random"]
        ax[0].plot([capsr[n]["n_params"] for n in names],
                   [capsr[n]["gain_over_persistence"] for n in names], "--x", label=f"{kind} (random)")
    ax[0].axhline(0, color="k", lw=.7)
    ax[0].set(xscale="log", xlabel="predictor params (capacity)",
              ylabel="center-pred gain over persistence (slots)",
              title="E4 P1: capacity→gain is structure-dependent")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    # 右：Part2 pred 场景三方法 F1
    methods = ["online_only", "pretrained_evo", "oracle"]
    x = np.arange(len(kinds)); wq = 0.25
    for i, m in enumerate(methods):
        vals = [all_res[k]["part2_recon"]["pred"][m]["f1_mean"] for k in kinds]
        ax[1].bar(x + (i - 1) * wq, vals, wq, label=m)
    ax[1].set(xticks=x, ylabel="support F1 (mean)", ylim=(0, 1.05),
              title="E4 P2: prior on predictable signals")
    ax[1].set_xticklabels(kinds); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000, help="N0=5000 -> eta=1ps（论文口径）")
    ap.add_argument("--kinds", type=str, nargs="+", default=["lfm", "fh", "thz"])
    ap.add_argument("--W", type=int, default=24, help="窗序列长度")
    ap.add_argument("--J", type=int, default=8)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=2, help="每窗事件预算步数（events=L*steps）")
    ap.add_argument("--cand-w", type=int, default=30,
                    help="预测中心两侧候选半宽(slots)；应 < 每窗漂移幅度(~6%%带宽)以体现先验价值")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve-W", type=int, default=48, help="在线学习曲线的部署窗序长度")
    ap.add_argument("--no-curve", action="store_true", help="跳过在线学习曲线实验")
    ap.add_argument("--paper", action="store_true", help="论文口径 N0=5000 + 更多 trials")
    ap.add_argument("--sweep-drift-cand", action="store_true", help="扫漂移–候选宽度比")
    ap.add_argument("--sweep-epsilon", action="store_true", help="扫不可约新息 ε_r 渐近下界")
    ap.add_argument("--sweep-esthist", action="store_true", help="W#1：estimated-history vs true-history")
    ap.add_argument("--only-sweeps", action="store_true", help="仅跑扫描（跳过主实验/曲线）")
    ap.add_argument("--no-sweeps", action="store_true", help="跳过漂移/ε 扫描")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()

    if args.quick:
        args.n0, args.W, args.n_train, args.n_val, args.trials = 1000, 16, 96, 24, 12
        args.curve_W = 32
    elif args.paper:
        args.n0 = 5000
        args.W = 32
        args.trials = 60
        args.n_train = 384
        args.n_val = 96
        args.curve_W = 64

    capacities = [
        dict(name="small", hidden=8, layers=1, epochs=40),
        dict(name="mid", hidden=32, layers=1, epochs=60),
        dict(name="large", hidden=96, layers=2, epochs=80),
    ]
    if args.quick:
        for c in capacities:
            c["epochs"] = max(20, c["epochs"] // 2)
    elif args.paper:
        for c in capacities:
            c["epochs"] = int(c["epochs"] * 1.25)

    cfg = SystemConfig(N0=args.n0)
    if args.paper or args.cand_w == 30:          # 默认 cand_w 随 N0 缩放为 ~3% 带宽
        args.cand_w = _cand_from_frac(cfg, 0.03)

    cand_fracs = [0.01, 0.02, 0.03, 0.06, 0.12]
    drift_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    epsilon_fracs = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]

    print(f"E4 evolution | {cfg.summary()} | kinds={args.kinds} W={args.W} cand_w={args.cand_w}")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_res, curve_res, drift_sweep, eps_sweep, esthist = {}, {}, {}, {}, {}

    if not args.only_sweeps:
        for kind in args.kinds:
            all_res[kind] = run_kind(cfg, kind, args.W, args.J, args.snr, args.steps,
                                     args.cand_w, args.n_train, args.n_val, args.trials,
                                     capacities, args.seed)
        if not args.no_curve:
            for kind in args.kinds:
                curve_res[kind] = run_curve(cfg, kind, args.curve_W, args.J, args.snr, args.steps,
                                            args.cand_w, args.n_train, args.trials, args.seed)

    sweep_kinds = args.kinds if len(args.kinds) <= 2 else ["lfm", "fh"]
    if args.sweep_drift_cand or (args.paper and not args.only_sweeps and not args.no_sweeps):
        for kind in sweep_kinds:
            drift_sweep[kind] = run_drift_cand_sweep(
                cfg, kind, args.W, args.J, args.snr, args.steps, args.trials, args.seed,
                args.n_train, cand_fracs, drift_scales)

    if args.sweep_epsilon or (args.paper and not args.only_sweeps and not args.no_sweeps):
        for kind in sweep_kinds:
            eps_sweep[kind] = run_epsilon_sweep(
                cfg, kind, args.W, args.J, args.snr, args.steps, args.trials, args.seed,
                args.n_train, epsilon_fracs, cand_frac=0.03, drift_scale=1.0)

    if args.sweep_esthist:
        for kind in sweep_kinds:
            esthist[kind] = run_estimated_history(
                cfg, kind, args.W, args.J, args.snr, args.steps, args.cand_w,
                args.trials, args.seed, args.n_train, drift_scale=1.0)

    out = {"config": cfg.summary(),
           "params": vars(args), "capacities": capacities,
           "kinds": all_res, "online_curve": curve_res,
           "drift_cand_sweep": drift_sweep, "epsilon_sweep": eps_sweep,
           "estimated_history": esthist}
    out_json = os.path.join(OUT_DIR, f"e4_{args.tag}.json")
    if args.only_sweeps and os.path.isfile(out_json):
        with open(out_json) as f:
            prev = json.load(f)
        for key in ["kinds", "online_curve", "drift_cand_sweep", "epsilon_sweep",
                    "estimated_history"]:
            if not out.get(key):                         # 本次未产出则保留旧值
                out[key] = prev.get(key, out.get(key))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved {out_json}")
    if all_res:
        _plot(all_res, os.path.join(OUT_DIR, f"e4_{args.tag}.png"))
    if curve_res:
        _plot_curve(curve_res, os.path.join(OUT_DIR, f"e4curve_{args.tag}.png"))
    if drift_sweep:
        _plot_drift_cand(drift_sweep, os.path.join(OUT_DIR, f"e4drift_{args.tag}.png"))
    if eps_sweep:
        _plot_epsilon(eps_sweep, os.path.join(OUT_DIR, f"e4epsilon_{args.tag}.png"))


if __name__ == "__main__":
    main()
