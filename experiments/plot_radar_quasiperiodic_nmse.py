"""Render a 2x2 publication SVG for §4.2 quasi-periodic radar tracking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_STYLE = {
    "nlfm": ("NLFM", "#0072B2"),
    "phase_lfm": ("polyphase-LFM", "#D55E00"),
    "costas": ("Costas", "#009E73"),
    "frank": ("Frank", "#CC79A7"),
}


def shared_limits(by_family: dict, target_nmse_db: float) -> tuple[float, float]:
    values = [float(target_nmse_db)]
    for family in FAMILY_STYLE:
        data = by_family[family]
        values.extend(data["nmse_p25_db"])
        values.extend(data["nmse_p75_db"])
        values.extend(data["nmse_med_db"])
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    pad = max(1.0, 0.08 * (hi - lo))
    return np.floor(lo - pad), np.ceil(hi + pad)


def draw_panel(
    ax,
    family: str,
    data: dict,
    target_nmse_db: float,
    y_limits: tuple[float, float],
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    label, color = FAMILY_STYLE[family]
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    median = np.asarray(data["nmse_med_db"], dtype=np.float64)
    p25 = np.asarray(data["nmse_p25_db"], dtype=np.float64)
    p75 = np.asarray(data["nmse_p75_db"], dtype=np.float64)

    ax.fill_between(
        time_s, p25, p75, color=color, alpha=0.18, linewidth=0,
        label="IQR (25th–75th)",
    )
    ax.plot(
        time_s, median, color=color, linewidth=1.8,
        marker="o", markersize=2.8, label="Median",
    )
    ax.axhline(
        target_nmse_db, color="#555555", linewidth=1.0, linestyle=":",
        label=rf"Threshold ({target_nmse_db:g} dB)",
    )
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks(np.arange(0.0, 2.01, 0.5))
    ax.set_ylim(*y_limits)
    ax.set_title(label, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Streaming time (s)")
    if show_ylabel:
        ax.set_ylabel("Waveform NMSE (dB)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0)


def main() -> None:
    from nuaa.repo_paths import OUTPUT_DIR, FIGURE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "streaming_radar_complex_nuaa_mu_2s.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURE_DIR / "radar_quasiperiodic_nmse_2x2.svg",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        results = json.load(handle)
    by_family = results["by_family"]
    missing = [family for family in FAMILY_STYLE if family not in by_family]
    if missing:
        raise KeyError(f"missing families in result JSON: {missing}")

    matplotlib.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    })
    target = float(results["params"]["target_nmse_db"])
    limits = shared_limits(by_family, target)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    families = list(FAMILY_STYLE.keys())
    for i, family in enumerate(families):
        row, col = divmod(i, 2)
        draw_panel(
            axes[row, col],
            family,
            by_family[family],
            target,
            limits,
            show_xlabel=(row == 1),
            show_ylabel=(col == 0),
        )
    fig.tight_layout(w_pad=0.6, h_pad=0.7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
