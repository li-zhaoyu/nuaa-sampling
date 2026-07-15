"""Plot prior-free NMSE convergence toward the scene-prior reference."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load_mt_curve(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    mt = d["methods"]["mt_bangbang"]
    tc = mt["tick_curve"]
    dt_ms = float(d["params"].get("control_dt_ms", 10.0))
    n = len(tc["nmse_med"])
    t_ms = (np.arange(1, n + 1)) * dt_ms
    return dict(
        t_ms=t_ms,
        nmse_med=np.asarray(tc["nmse_med"], dtype=np.float64),
        nmse_p25=np.asarray(tc.get("nmse_p25", tc["nmse_med"]), dtype=np.float64),
        nmse_p75=np.asarray(tc.get("nmse_p75", tc["nmse_med"]), dtype=np.float64),
        nmse_best=np.asarray(tc["nmse_best_med"], dtype=np.float64),
        f1=np.asarray(tc.get("f1_med", np.full(n, np.nan)), dtype=np.float64),
        belief_mass=np.asarray(
            tc.get("belief_mass_med", np.full(n, np.nan)), dtype=np.float64),
        hit=float(mt.get("hit_rate", 0.0)),
        params=d.get("params", {}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-prior", required=True, help="JSON with scene-prior locking")
    ap.add_argument("--no-prior", required=True, help="JSON without scene prior (streaming belief)")
    ap.add_argument("--out", required=True, help="output SVG/PNG/PDF path (extension sets format)")
    ap.add_argument("--gap-threshold-db", type=float, default=2.0,
                    help="maximum no-prior NMSE deficit regarded as convergence")
    ap.add_argument("--metrics-out", default=None,
                    help="optional JSON sidecar with curves and convergence time")
    ap.add_argument(
        "--stationary-with-prior",
        action="store_true",
        help="pool the fixed-window prior reference over time; use when no online "
             "evidence or recurrent state is accumulated on that branch",
    )
    args = ap.parse_args()

    wp = _load_mt_curve(args.with_prior)
    np_ = _load_mt_curve(args.no_prior)
    n = min(len(wp["t_ms"]), len(np_["t_ms"]))
    for curve in (wp, np_):
        for key in ("t_ms", "nmse_med", "nmse_p25", "nmse_p75",
                    "nmse_best", "f1", "belief_mass"):
            curve[key] = curve[key][:n]
    stationary_prior_db = None
    if args.stationary_with_prior:
        # The prior branch is deliberately stationary: every tick has the same
        # support prior, layout, slot-0 target, and M-point coefficient budget.
        # Pool its repeated draws to estimate one reference distribution rather
        # than interpreting finite-trial per-tick median noise as a time trend.
        stationary_prior_db = float(np.nanmedian(wp["nmse_med"]))
        stationary_p25_db = float(np.nanmedian(wp["nmse_p25"]))
        stationary_p75_db = float(np.nanmedian(wp["nmse_p75"]))
        wp["nmse_med"] = np.full(n, stationary_prior_db, dtype=np.float64)
        wp["nmse_p25"] = np.full(n, stationary_p25_db, dtype=np.float64)
        wp["nmse_p75"] = np.full(n, stationary_p75_db, dtype=np.float64)
    deficit = np_["nmse_med"] - wp["nmse_med"]
    within = deficit <= args.gap_threshold_db
    first_close_idx = next((i for i in range(n) if bool(within[i])), None)
    sustained_idx = next(
        (i for i in range(n) if bool(np.all(within[i:]))),
        None,
    )
    success_idx = next(
        (i for i in range(n) if float(np_["nmse_med"][i]) <= -10.0),
        None,
    )
    matched_idx = next(
        (i for i in range(n) if abs(float(deficit[i])) <= 0.05),
        None,
    )
    events_per_window = int(np_["params"].get("window_periods", 0)) * 5
    evidence = (np.arange(1, n + 1)) * max(1, events_per_window)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=180)
    prior_label = (
        "With scene prior (pre-calibrated)"
        if int(wp["params"].get("prior_calibration_windows", 0)) > 0
        else "With scene prior"
    )
    ax.plot(
        wp["t_ms"], wp["nmse_med"], color="#0072B2", lw=2.0,
        marker=None if args.stationary_with_prior else "o", ms=3.5,
        label=("With scene prior (stationary reference)"
               if args.stationary_with_prior else prior_label))
    ax.fill_between(wp["t_ms"], wp["nmse_p25"], wp["nmse_p75"],
                    color="#0072B2", alpha=0.14, linewidth=0)
    ax.plot(
        np_["t_ms"], np_["nmse_med"], color="#D55E00", lw=2.0,
        marker="s", ms=3.5, label="No scene prior (accumulated evidence)")
    ax.fill_between(np_["t_ms"], np_["nmse_p25"], np_["nmse_p75"],
                    color="#D55E00", alpha=0.12, linewidth=0)
    ax.axhline(
        -10.0, color="#666666", ls=":", lw=1.0,
        label=r"Success threshold ($-10$ dB)")

    if success_idx is not None:
        x = float(np_["t_ms"][success_idx])
        y = float(np_["nmse_med"][success_idx])
        ax.scatter([x], [y], s=42, color="#555555", edgecolor="white",
                   linewidth=0.7, zorder=6)
        ax.annotate(
            rf"$t={x:g}$ ms"
            "\n"
            rf"NMSE$={y:+.1f}$ dB",
            xy=(x, y),
            xytext=(12, 18 if y > -18 else -28),
            textcoords="offset points",
            fontsize=7.5, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#555555", lw=0.6, alpha=0.92),
        )

    ax.set_xlabel("Control-loop time (ms)")
    ax.set_ylabel("Waveform NMSE (dB)")
    ax.set_xlim(left=0, right=float(max(wp["t_ms"][-1], np_["t_ms"][-1])))
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    ax.legend(loc="upper right", frameon=False)

    # Top axis: cumulative evidence count, to make the gradual process explicit.
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    tick_pos = list(np_["t_ms"][:: max(1, n // 5)])
    if float(np_["t_ms"][-1]) not in tick_pos:
        tick_pos.append(float(np_["t_ms"][-1]))
    tick_lbl = [
        f"{int(evidence[np.where(np_['t_ms'] == t)[0][0]]):d}"
        for t in tick_pos
    ]
    ax_top.set_xticks(tick_pos)
    ax_top.set_xticklabels(tick_lbl)
    ax_top.set_xlabel("Accumulated sampling points")
    ax_top.spines["top"].set_visible(True)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.86)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    convergence_ms = (
        float(np_["t_ms"][sustained_idx]) if sustained_idx is not None else None)
    first_close_ms = (
        float(np_["t_ms"][first_close_idx]) if first_close_idx is not None else None)
    success_ms = (
        float(np_["t_ms"][success_idx]) if success_idx is not None else None)
    matched_ms = (
        float(np_["t_ms"][matched_idx]) if matched_idx is not None else None)
    metrics = {
        "control_dt_ms": float(np_["params"].get("control_dt_ms", 100.0)),
        "window_periods": int(np_["params"].get("window_periods", 0)),
        "events_per_window": int(np_["params"].get("window_periods", 0)) * 5,
        "gap_threshold_db": args.gap_threshold_db,
        "with_prior_summary_mode": (
            "time-pooled stationary reference"
            if args.stationary_with_prior else "per-tick median"),
        "stationary_with_prior_nmse_db": stationary_prior_db,
        "prior_calibration_windows": int(
            wp["params"].get("prior_calibration_windows", 0)),
        "prior_online_gain": float(
            wp["params"].get("prior_online_gain", 0.0)),
        "prior_coefficient_ema": float(
            wp["params"].get("prior_coef_ema", 0.0)),
        "first_close_ms": first_close_ms,
        "matched_ms": matched_ms,
        "success_ms": success_ms,
        "sustained_convergence_ms": convergence_ms,
        "time_ms": np_["t_ms"].tolist(),
        "accumulated_points": evidence.tolist(),
        "no_prior_nmse_med_db": np_["nmse_med"].tolist(),
        "with_prior_nmse_med_db": wp["nmse_med"].tolist(),
        "nmse_deficit_db": deficit.tolist(),
        "no_prior_f1_med": np_["f1"].tolist(),
    }
    if args.metrics_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.metrics_out)) or ".",
            exist_ok=True,
        )
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"saved {args.metrics_out}")
    print(f"saved {args.out}")
    print(f"with-prior hit={wp['hit']:.0%} final_med={wp['nmse_med'][-1]:+.1f} best={wp['nmse_best'][-1]:+.1f}")
    print(f"no-prior   hit={np_['hit']:.0%} final_med={np_['nmse_med'][-1]:+.1f} best={np_['nmse_best'][-1]:+.1f}")
    if success_ms is not None:
        print(f"no-prior first success (<= -10 dB): {success_ms:g} ms")
    if first_close_ms is None:
        print(f"no first close within {args.gap_threshold_db:g} dB")
    else:
        print(
            f"first close within {args.gap_threshold_db:g} dB "
            f"at {first_close_ms:g} ms")
    if matched_ms is not None:
        print(f"first near-match (|Δ|<=0.05 dB): {matched_ms:g} ms")
    if convergence_ms is None:
        print(f"no sustained convergence within {args.gap_threshold_db:g} dB")
    else:
        print(
            f"sustained convergence within {args.gap_threshold_db:g} dB "
            f"from {convergence_ms:g} ms")


if __name__ == "__main__":
    main()
