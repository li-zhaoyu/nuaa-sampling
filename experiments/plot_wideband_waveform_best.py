"""Plot broadband chirplet useful-signal reconstruction vs ground truth and observed jam+noise.

Uses the current Fig. 3 / Table 2 scene-prior deployment protocol
(window_periods=40, control_dt=100 ms, calibrated prior + mild EMA).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure

configure()

from nuaa import signals as S
from nuaa.config import SystemConfig
from nuaa import metrics as Met
from nuaa import streaming as St
from experiments.exp_streaming_wideband_nuaa import (
    build_atom_grid,
    atom_bins,
    pulse_slots,
    pulse_slot_offset,
    make_wideband_state,
    synth_waveform,
    synth_period_y,
    scene_prior,
    train_wideband_model,
    window_tensors,
    calibrate_scene_coefficient_prior,
    signal_slow_index,
)
from experiments.exp_structured_nuaa_mu import CAPS


def stitch_pulse_waveform(C, alpha, atoms_meta, cfg, n_pulse: int) -> tuple[np.ndarray, np.ndarray]:
    """Stitch the useful waveform over full pulse width τ; return (t_ns, x)."""
    n_per = max(1, int(np.ceil(n_pulse / cfg.N0)))
    xs, ts = [], []
    for seg_i in range(n_per):
        slot0 = seg_i * cfg.N0
        x_seg = synth_waveform(C, alpha, atoms_meta, cfg, slot0=slot0)
        n = slot0 + np.arange(cfg.N0, dtype=np.float64)
        mask = n < n_pulse
        if not np.any(mask):
            continue
        t_ps = n[mask] * cfg.eta_ps
        xs.append(x_seg[mask])
        ts.append(t_ps)
    x = np.concatenate(xs)
    t_ns = np.concatenate(ts) / 1000.0
    return t_ns, x


def amp_series(x: np.ndarray) -> np.ndarray:
    return np.real(x)


def synth_observed_waveform(
    C, alpha_full, alpha_useful, atoms_meta, cfg, n_pulse, snr_db, rng,
) -> tuple[np.ndarray, np.ndarray]:
    t_ns, x_total = stitch_pulse_waveform(C, alpha_full, atoms_meta, cfg, n_pulse)
    _, x_use = stitch_pulse_waveform(C, alpha_useful, atoms_meta, cfg, n_pulse)
    ref = float(np.mean(np.abs(x_use) ** 2)) + 1e-30
    x_noisy = S.add_measurement_noise(x_total, snr_db, ref, rng)
    return t_ns, x_noisy


def align_complex(x_hat: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    denom = np.vdot(x_hat, x_hat)
    if abs(denom) < 1e-20:
        return x_hat.copy(), 0.0
    alpha = np.vdot(x_hat, x) / denom
    return alpha * x_hat, float(np.abs(alpha))


def capture_trial(cfg, atoms_meta, model, args, trial_seed: int):
    """Scene-prior NUAA-MU path matching Fig. 3 / Table 2 deployment."""
    rng = np.random.default_rng(trial_seed)
    seq_id = int(rng.integers(0, 100000))
    norm_events = int(args.window_periods) * cfg.L
    bins_all = atoms_meta[2]
    n_pulse = atoms_meta[5]
    ctrl = St.StreamingNUAAController(
        cfg, K=args.K, window_events=norm_events, seed=trial_seed + 1)
    seg_states: dict[int, tuple] = {}
    alpha_ema = None
    alpha_prior_init = None
    records = []

    for tick in range(args.ticks):
        seg = tick // max(1, args.signal_pri_ticks)
        win = signal_slow_index(tick, trial_seed, args.signal_pri_ticks, args.W)
        first_period = tick * args.window_periods
        report_slot0 = 0
        if seg not in seg_states:
            seg_states[seg] = make_wideband_state(
                cfg, atoms_meta, seq_id, win, args.K, args.nC, rng, args.sir)
        useful, jam, C, loc, jam_loc, alpha = seg_states[seg]
        alpha_useful = np.zeros_like(alpha)
        alpha_useful[loc] = alpha[loc]
        x_true_seg = synth_waveform(
            C, alpha_useful, atoms_meta, cfg, slot0=report_slot0)
        bins_C = bins_all[C]
        truth_X = np.zeros(cfg.N0, dtype=np.complex128)
        truth_X[bins_C] = alpha
        truth_support = bins_C[loc]

        ctrl.begin_observation_window()
        for period_offset in range(args.window_periods):
            slot0 = pulse_slot_offset(cfg, n_pulse, first_period + period_offset)
            cosets = ctrl.current_cosets()
            y = synth_period_y(
                cosets, C, alpha, loc, atoms_meta, rng, args.eval_snr,
                norm_events, args.burst_role, win, seq_id, cfg, slot0=slot0)
            ctrl.append_period_measurement(y, pulse_slot0=slot0)

        prior_ov = scene_prior(C, useful, atoms_meta)
        prior_feat = prior_ov if getattr(model, "cand_dim", 4) >= 5 else None
        if alpha_prior_init is None and args.prior_calibration_windows > 0:
            cal_rng = np.random.default_rng(trial_seed + 900001)
            alpha_prior_init = calibrate_scene_coefficient_prior(
                ctrl, C, alpha, loc, jam_loc, atoms_meta, cfg,
                args, win, seq_id, cal_rng)

        tok, dtn, A, Y, cand = window_tensors(
            ctrl, C, atoms_meta, cfg, win, args.period, norm_events,
            prior=prior_feat)
        rec = ctrl.reconstruct_with_model(
            model, tok, dtn, A, Y, bins_C, cand_feat=cand,
            prior_override=prior_ov, allow_prior_lock=True,
            accumulate_coefficients=not args.fixed_window_coefficients,
            use_burst=False, truth_X=truth_X, truth_support=truth_support)
        alpha_hat = rec.Xhat[bins_C, 0]
        beta = float(args.prior_coef_ema)
        if alpha_ema is None or beta <= 0.0:
            alpha_ema = alpha_hat.copy()
        else:
            alpha_ema = beta * alpha_ema + (1.0 - beta) * alpha_hat
        alpha_report = alpha_ema
        if alpha_prior_init is not None:
            gain = float(np.clip(args.prior_online_gain, 0.0, 1.0))
            alpha_report = (1.0 - gain) * alpha_prior_init + gain * alpha_report
        x_hat_seg = synth_waveform(
            C, alpha_report, atoms_meta, cfg, slot0=report_slot0)
        nm = Met.nmse_db(x_hat_seg, x_true_seg)
        records.append(dict(
            tick=tick, slot0=report_slot0, nmse_db=nm,
            C=C, alpha=alpha, alpha_useful=alpha_useful, alpha_hat=alpha_report,
            x_true_seg=x_true_seg, x_hat_seg=x_hat_seg,
        ))
        ctrl.plan_next_period(dt_s=args.control_dt_ms * 1e-3, mode=args.mt_mode)

    best = min(records, key=lambda r: r["nmse_db"])
    C = best["C"]
    t_ns, x_true_full = stitch_pulse_waveform(
        C, best["alpha_useful"], atoms_meta, cfg, n_pulse)
    _, x_hat_full = stitch_pulse_waveform(
        C, best["alpha_hat"], atoms_meta, cfg, n_pulse)
    x_hat_aligned, scale = align_complex(x_hat_full, x_true_full)
    obs_rng = np.random.default_rng(trial_seed + 17_000 + best["tick"])
    _, x_obs_full = synth_observed_waveform(
        C, best["alpha"], best["alpha_useful"], atoms_meta, cfg, n_pulse,
        args.eval_snr, obs_rng)
    nm_full = Met.nmse_db(x_hat_aligned, x_true_full)
    return dict(
        records=records,
        best=best,
        atoms_meta=atoms_meta,
        t_ns=t_ns,
        x_true=x_true_full,
        x_obs=x_obs_full,
        x_hat=x_hat_aligned,
        nmse_full_db=nm_full,
        scale=scale,
        n_pulse=n_pulse,
        eval_snr=args.eval_snr,
        sir=args.sir,
        obs_noise_seed=trial_seed + 17_000 + best["tick"],
    )


def eval_waveform_t(
    C, alpha, atoms_meta, n_pulse: int, t_ns: np.ndarray,
) -> np.ndarray:
    f0_tab, k_tab, _, _, _, _ = atoms_meta
    n = np.asarray(t_ns, dtype=np.float64) * 1000.0
    active = (n >= 0) & (n < n_pulse)
    x = np.zeros_like(n, dtype=np.complex128)
    for j, a in enumerate(C):
        if abs(alpha[j]) < 1e-12:
            continue
        ph = 2 * np.pi * (f0_tab[a] * n + 0.5 * k_tab[a] * n ** 2)
        x += alpha[j] * active * np.exp(1j * ph)
    return x


def zoom_window(t_ns: np.ndarray, x_ref: np.ndarray, n_cycles: float) -> tuple[float, float]:
    if len(t_ns) < 32:
        return float(t_ns[0]), float(t_ns[-1])
    amp = np.abs(x_ref)
    i_peak = int(np.argmax(amp))
    yr = np.real(x_ref)
    sgn = np.sign(yr)
    for i in range(1, len(sgn)):
        if sgn[i] == 0:
            sgn[i] = sgn[i - 1]
    zc = np.where(np.diff(sgn) != 0)[0]
    if zc.size < 4:
        dt_ns = float(np.median(np.diff(t_ns)))
        spec = np.abs(np.fft.rfft(x_ref * np.hanning(len(x_ref))))
        freqs = np.fft.rfftfreq(len(x_ref), d=dt_ns * 1e-9)
        k = int(np.argmax(spec[1:])) + 1
        f0 = float(freqs[k]) if freqs[k] > 0 else 1.0 / max(dt_ns * 1e-9, 1e-12)
        period_ns = 1e9 / f0
    else:
        half_period_ns = np.diff(t_ns[zc])
        half_period_ns = half_period_ns[(half_period_ns > 0) & np.isfinite(half_period_ns)]
        period_ns = float(2.0 * np.median(half_period_ns)) if half_period_ns.size else t_ns[-1] / 10.0
    win_ns = max(period_ns * n_cycles, 4.0 * period_ns)
    t0 = float(t_ns[i_peak]) - 0.5 * win_ns
    t1 = t0 + win_ns
    return t0, t1


def smooth_series(y: np.ndarray, win: int = 31, poly: int = 3) -> np.ndarray:
    if y.size < 7:
        return y
    from scipy.signal import savgol_filter

    w = int(win)
    if w >= y.size:
        w = y.size - 1 if y.size % 2 == 0 else y.size - 2
    if w < 5:
        return y
    if w % 2 == 0:
        w -= 1
    return savgol_filter(y, w, min(poly, w - 1))


def prepare_plot_series(
    data: dict,
    n_cycles: float,
    n_fine: int = 5000,
    smooth_win: int = 41,
) -> dict:
    best = data["best"]
    atoms_meta = data["atoms_meta"]
    n_pulse = data["n_pulse"]
    C = best["C"]
    t0, t1 = zoom_window(data["t_ns"], data["x_true"], n_cycles)
    t = np.linspace(t0, t1, int(n_fine))

    x_gt = eval_waveform_t(C, best["alpha_useful"], atoms_meta, n_pulse, t)
    x_hat = eval_waveform_t(C, best["alpha_hat"], atoms_meta, n_pulse, t)
    x_hat = align_complex(x_hat, x_gt)[0]
    x_tot = eval_waveform_t(C, best["alpha"], atoms_meta, n_pulse, t)
    ref = float(np.mean(np.abs(x_gt) ** 2)) + 1e-30
    rng = np.random.default_rng(data["obs_noise_seed"])
    x_obs = S.add_measurement_noise(x_tot, data["eval_snr"], ref, rng)

    y_gt = smooth_series(amp_series(x_gt), win=smooth_win)
    y_hat = smooth_series(amp_series(x_hat), win=smooth_win)
    y_obs = smooth_series(amp_series(x_obs), win=max(11, smooth_win // 3))
    return dict(t=t, y_gt=y_gt, y_hat=y_hat, y_obs=y_obs, t0=t0, t1=t1,
                seg_nmse=Met.nmse_db(x_hat, x_gt))


def plot_comparison(
    data: dict,
    out_path: str,
    n_cycles: float = 10.0,
    n_fine: int = 5000,
    smooth_win: int = 41,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "axes.spines.top": False,
    })

    series = prepare_plot_series(data, n_cycles, n_fine=n_fine, smooth_win=smooth_win)
    t = series["t"]
    y_gt, y_hat, y_obs = series["y_gt"], series["y_hat"], series["y_obs"]
    t0, t1 = series["t0"], series["t1"]
    sig_lim = max(float(np.max(np.abs(y_gt))), float(np.max(np.abs(y_hat))), 1e-9) * 1.18
    obs_lim = max(float(np.max(np.abs(y_obs))), 1e-9) * 1.08

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.11, right=0.88)

    ax.plot(t, y_gt, color="#1f4e79", lw=1.4, ls="--", label="Ground truth")
    ax.plot(t, y_hat, color="#c00000", lw=1.15, label="NUAA-MU reconstruction")
    ax.set_ylabel("Amplitude (useful)")
    ax.set_xlabel("Time (ns)")
    ax.set_xlim(t0, t1)
    ax.set_ylim(-sig_lim, sig_lim)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(t, y_obs, color="#7f7f7f", lw=0.9, alpha=0.7,
             label=rf"Observed (jam + noise, SNR $={data['eval_snr']:g}$ dB, SIR $={data['sir']:g}$ dB)")
    ax2.set_ylabel("Amplitude (observed)", color="#666666")
    ax2.tick_params(axis="y", labelcolor="#666666")
    ax2.set_ylim(-obs_lim, obs_lim)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
              fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.22)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    base, ext = os.path.splitext(out_path)
    ext = ext.lower() if ext else ".svg"
    if ext not in (".svg", ".png", ".pdf"):
        ext = ".svg"
    save_path = base + ext
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {save_path}")
    print(f"zoom_window_ns=[{t0:.4f}, {t1:.4f}] local_nmse={series['seg_nmse']:+.2f} dB")
    return series


def main():
    from nuaa.repo_paths import OUTPUT_DIR, FIGURE_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-snr", type=float, default=-10.0)
    ap.add_argument("--sir", type=float, default=-40.0)
    ap.add_argument("--pulse-width-ns", type=float, default=16.0)
    ap.add_argument("--signal-pri-ticks", type=int, default=20)
    ap.add_argument("--window-periods", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--cap", choices=sorted(CAPS), default="large")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--mt-mode", default="static_hold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trial", type=int, default=0, help="trial index (seed + 1000*trial)")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--nC", type=int, default=48)
    ap.add_argument("--n-f0", type=int, default=24)
    ap.add_argument("--n-k", type=int, default=8)
    ap.add_argument("--f-lo-ghz", type=float, default=20.0)
    ap.add_argument("--f-hi-ghz", type=float, default=120.0)
    ap.add_argument("--bw-lo-ghz", type=float, default=10.0)
    ap.add_argument("--bw-hi-ghz", type=float, default=30.0)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--control-dt-ms", type=float, default=100.0)
    ap.add_argument("--prior-calibration-windows", type=int, default=20)
    ap.add_argument("--prior-online-gain", type=float, default=0.1)
    ap.add_argument("--prior-coef-ema", type=float, default=0.3)
    ap.add_argument("--fixed-window-coefficients", action="store_true", default=True)
    ap.add_argument("--model-in", type=str,
                    default=str(
                        OUTPUT_DIR
                        / "streaming_wideband_nuaa_n5000_wb_tau16_wp40_withprior_sat.pt"
                    ))
    ap.add_argument("--n-cycles", type=float, default=10.0)
    ap.add_argument("--n-fine", type=int, default=5000)
    ap.add_argument("--smooth-win", type=int, default=41)
    ap.add_argument("--out", type=str,
                    default=str(FIGURE_DIR / "wideband_waveform_best.svg"))
    ap.add_argument(
        "--data-out",
        type=str,
        default=str(FIGURE_DIR.parent / "data" / "waveforms" / "fig4_wideband_chirp_waveforms.npz"),
        help="Save ground-truth / reconstructed / observed waveform arrays (npz).",
    )
    args = ap.parse_args()
    args.hold_coeffs = True
    args.burst_role = "none"
    args.batch = 24
    args.lr = 1.5e-3
    args.progress_every = 0
    args.train_steps_list = [4, 8, 12, 16]
    args.snr_lo = -6.0
    args.snr_hi = 8.0
    args.no_scene_prior = False
    args.target_nmse_db = -10.0

    torch.manual_seed(args.seed)
    cfg = SystemConfig(N0=5000)
    n_pulse = pulse_slots(cfg, args.pulse_width_ns)
    f0_tab, k_tab = build_atom_grid(
        cfg, args.f_lo_ghz, args.f_hi_ghz, args.bw_lo_ghz, args.bw_hi_ghz,
        args.n_f0, args.n_k, n_pulse)
    atoms_meta = (f0_tab, k_tab, atom_bins(f0_tab, k_tab, cfg, n_pulse),
                  args.n_f0, args.n_k, n_pulse)
    model = train_wideband_model(cfg, atoms_meta, args)

    trial_seed = args.seed + 1000 * args.trial
    data = capture_trial(cfg, atoms_meta, model, args, trial_seed)
    series = plot_comparison(
        data, args.out, n_cycles=args.n_cycles,
        n_fine=args.n_fine, smooth_win=args.smooth_win)
    if args.data_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.data_out)) or ".", exist_ok=True)
        np.savez_compressed(
            args.data_out,
            t_ns=np.asarray(series["t"], dtype=np.float64),
            amp_ground_truth=np.asarray(series["y_gt"], dtype=np.float64),
            amp_reconstructed=np.asarray(series["y_hat"], dtype=np.float64),
            amp_observed_jam_noise=np.asarray(series["y_obs"], dtype=np.float64),
            zoom_t0_ns=float(series["t0"]),
            zoom_t1_ns=float(series["t1"]),
            segment_nmse_db=float(series["seg_nmse"]),
            full_pulse_nmse_db=float(data["nmse_full_db"]),
            eval_snr_db=float(data["eval_snr"]),
            sir_db=float(data["sir"]),
        )
        print(f"saved waveform data {args.data_out}")
    print(f"best tick {data['best']['tick']} nmse_seg={data['best']['nmse_db']:.2f} dB "
          f"nmse_full={data['nmse_full_db']:.2f} dB scale={data['scale']:.3f} "
          f"local_nmse={series['seg_nmse']:+.2f} dB")


if __name__ == "__main__":
    main()
