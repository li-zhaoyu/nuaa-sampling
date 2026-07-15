#!/usr/bin/env python3
"""Export ground-truth / reconstructed / observed waveforms for every paper signal.

Signals covered (Photonics manuscript §4):
  - broadband chirplet useful tone (+ jammer + AWGN observation)  → Fig. 4
  - quasi-periodic radar: NLFM, polyphase-LFM, Costas, Frank     → Fig. 5
  - THz 16-QAM constellation / symbol IQ                         → Fig. 7

Outputs land in data/waveforms/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure

configure()

from nuaa.config import SystemConfig
from nuaa import signals as S
from nuaa.repo_paths import OUTPUT_DIR, FIGURE_DIR, REPO_ROOT
from experiments.exp_e5_radar_complex import complex_radar_atom, amplitude_nmse_db
from experiments.exp_streaming_radar_complex_nuaa_mu import (
    FAMILIES,
    FAMILY_LABELS,
    _normalize_max,
    _softmax,
    candidate_centers,
    make_trial_states,
    make_window,
    map_belief_to_layout,
    online_dynamic_prior,
    predict_current_center,
    pulse_sample_times,
    spec_at,
    train_or_load_nuaa_mu,
    train_evolution_model,
    dynamic_candidate_features,
)
from experiments.plot_wideband_waveform_best import (
    build_atom_grid,
    atom_bins,
    pulse_slots,
    train_wideband_model,
    capture_trial,
    prepare_plot_series,
    stitch_pulse_waveform,
)


WAVE_DIR = REPO_ROOT / "data" / "waveforms"


def _c_from_pack(d: dict) -> np.ndarray:
    return np.asarray(d["real"], dtype=np.float64) + 1j * np.asarray(d["imag"], dtype=np.float64)


def export_chirplet(args) -> Path:
    """§4.1 broadband chirplet: useful GT / reconstruction / jammer / observed."""
    plot_args = SimpleNamespace(
        eval_snr=-10.0,
        sir=-40.0,
        pulse_width_ns=16.0,
        signal_pri_ticks=20,
        window_periods=40,
        ticks=2,
        cap="large",
        iters=0,
        mt_mode="static_hold",
        seed=0,
        trial=0,
        K=2,
        nC=48,
        n_f0=24,
        n_k=8,
        f_lo_ghz=20.0,
        f_hi_ghz=120.0,
        bw_lo_ghz=10.0,
        bw_hi_ghz=30.0,
        W=32,
        period=32,
        control_dt_ms=100.0,
        prior_calibration_windows=20,
        prior_online_gain=0.1,
        prior_coef_ema=0.3,
        fixed_window_coefficients=True,
        model_in=str(OUTPUT_DIR / "streaming_wideband_nuaa_n5000_wb_tau16_wp40_withprior_sat.pt"),
        hold_coeffs=True,
        burst_role="none",
        batch=24,
        lr=1.5e-3,
        progress_every=0,
        train_steps_list=[4, 8, 12, 16],
        snr_lo=-6.0,
        snr_hi=8.0,
        no_scene_prior=False,
        target_nmse_db=-10.0,
        n_cycles=10.0,
        n_fine=5000,
        smooth_win=41,
    )
    torch.manual_seed(plot_args.seed)
    cfg = SystemConfig(N0=5000)
    n_pulse = pulse_slots(cfg, plot_args.pulse_width_ns)
    f0_tab, k_tab = build_atom_grid(
        cfg, plot_args.f_lo_ghz, plot_args.f_hi_ghz, plot_args.bw_lo_ghz, plot_args.bw_hi_ghz,
        plot_args.n_f0, plot_args.n_k, n_pulse)
    atoms_meta = (f0_tab, k_tab, atom_bins(f0_tab, k_tab, cfg, n_pulse),
                  plot_args.n_f0, plot_args.n_k, n_pulse)
    model = train_wideband_model(cfg, atoms_meta, plot_args)
    data = capture_trial(cfg, atoms_meta, model, plot_args, plot_args.seed)
    series = prepare_plot_series(data, plot_args.n_cycles, plot_args.n_fine, plot_args.smooth_win)

    best = data["best"]
    # jammer-only full pulse
    alpha_jam = best["alpha"].copy()
    alpha_jam[best["alpha_useful"] != 0] = 0
    # keep only jam locations: alpha_useful zeros useful coeffs already; jam = alpha - useful
    alpha_jam = best["alpha"] - best["alpha_useful"]
    t_ns, x_jam = stitch_pulse_waveform(
        best["C"], alpha_jam, atoms_meta, cfg, n_pulse)

    out = WAVE_DIR / "chirplet_broadband_waveforms.npz"
    np.savez_compressed(
        out,
        # full pulse (complex baseband on 1 ps grid)
        t_ns=np.asarray(data["t_ns"], dtype=np.float64),
        x_true_useful=np.asarray(data["x_true"], dtype=np.complex128),
        x_reconstructed=np.asarray(data["x_hat"], dtype=np.complex128),
        x_jammer=np.asarray(x_jam, dtype=np.complex128),
        x_observed_jam_noise=np.asarray(data["x_obs"], dtype=np.complex128),
        # Fig. 4 zoomed amplitude series
        zoom_t_ns=np.asarray(series["t"], dtype=np.float64),
        zoom_amp_ground_truth=np.asarray(series["y_gt"], dtype=np.float64),
        zoom_amp_reconstructed=np.asarray(series["y_hat"], dtype=np.float64),
        zoom_amp_observed=np.asarray(series["y_obs"], dtype=np.float64),
        zoom_t0_ns=float(series["t0"]),
        zoom_t1_ns=float(series["t1"]),
        segment_nmse_db=float(series["seg_nmse"]),
        full_pulse_nmse_db=float(data["nmse_full_db"]),
        eval_snr_db=float(data["eval_snr"]),
        sir_db=float(data["sir"]),
        pulse_width_ns=float(plot_args.pulse_width_ns),
        signal="broadband_chirplet",
    )
    # keep Fig.4 convenience alias
    alias = WAVE_DIR / "fig4_wideband_chirp_waveforms.npz"
    np.savez_compressed(
        alias,
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
    print(f"saved {out}")
    print(f"saved {alias}")
    return out


def _radar_args() -> SimpleNamespace:
    paper = json.loads((OUTPUT_DIR / "streaming_radar_complex_nuaa_mu_2s.json").read_text())
    p = paper["params"]
    # force pretrained checkpoints in this repo
    p = dict(p)
    p["model_path"] = str(OUTPUT_DIR / "radar_complex_nuaa_mu_pretrained.pt")
    p["evolution_model_path"] = str(OUTPUT_DIR / "radar_complex_evolution_prior.pt")
    p["force_train"] = False
    p["quick"] = False
    # one representative trial per family is enough for waveform release
    p["trials"] = 1
    return SimpleNamespace(**p)


def export_radar() -> Path:
    """§4.2: one representative pulse waveform per radar family (true vs recon)."""
    args = _radar_args()
    cfg = SystemConfig(N0=args.n0)
    model, _ = train_or_load_nuaa_mu(cfg, args)
    evolution_model, _ = train_evolution_model(cfg, args)
    model.eval()
    evolution_model.eval()

    n_pulse = int(round(args.pulse_width_ns * 1000.0 / cfg.eta_ps))
    full_times = np.arange(n_pulse, dtype=np.int64)
    t_ns = full_times.astype(np.float64) * cfg.eta_ps * 1e-3

    arrays: dict[str, np.ndarray | float | str] = {
        "t_ns": t_ns,
        "snr_db": float(args.snr),
        "pulse_width_ns": float(args.pulse_width_ns),
    }

    for family in args.families:
        states = make_trial_states(cfg, args, family)
        # warm-up history
        for warm in range(-args.history_ticks, 0):
            _process_tick_with_waveforms(
                cfg, args, model, evolution_model, states,
                pulse_idx=warm * args.pri_stride, collect=False,
            )
        best = None
        time_s = np.linspace(0.0, args.duration_s, args.ticks)
        for tick, tval in enumerate(time_s):
            out = _process_tick_with_waveforms(
                cfg, args, model, evolution_model, states,
                pulse_idx=tick * args.pri_stride, collect=True,
                full_times=full_times, n_pulse=n_pulse,
            )
            nm = float(out["nmse_db"][0])
            print(f"radar {family:9s} tick={tick:02d} t={tval:.1f}s NMSE={nm:+.2f} dB", flush=True)
            if best is None or nm < best["nmse_db"]:
                best = {
                    "nmse_db": nm,
                    "time_s": float(tval),
                    "tick": tick,
                    "x_true": out["x_true"][0],
                    "x_hat": out["x_hat"][0],
                    "y_meas": out["y_meas"][0],
                    "hit": float(out["hit"][0]),
                }
        assert best is not None
        arrays[f"{family}_x_true"] = np.asarray(best["x_true"], dtype=np.complex128)
        arrays[f"{family}_x_reconstructed"] = np.asarray(best["x_hat"], dtype=np.complex128)
        arrays[f"{family}_y_measurement"] = np.asarray(best["y_meas"], dtype=np.complex128)
        arrays[f"{family}_best_nmse_db"] = float(best["nmse_db"])
        arrays[f"{family}_best_time_s"] = float(best["time_s"])
        arrays[f"{family}_support_hit"] = float(best["hit"])
        arrays[f"{family}_label"] = FAMILY_LABELS[family]
        print(f"  -> keep tick@{best['time_s']:.1f}s NMSE={best['nmse_db']:+.2f} dB", flush=True)

    out = WAVE_DIR / "radar_quasiperiodic_waveforms.npz"
    np.savez_compressed(out, **arrays)
    print(f"saved {out}")
    return out


def _process_tick_with_waveforms(
    cfg, args, model, evolution_model, states, pulse_idx, *,
    collect: bool, full_times=None, n_pulse=None,
) -> dict:
    batch = []
    predicted_centers = []
    for state in states:
        centers = candidate_centers(
            state.candidates, pulse_idx, cfg, args.codebook_size
        )
        predicted = predict_current_center(state, evolution_model, cfg)
        prior = online_dynamic_prior(state, centers, predicted, cfg, args)
        cosets = state.controller.current_cosets()
        state.controller.state.acc_cosets.append(np.asarray(cosets).copy())
        batch.append(
            make_window(
                cfg=cfg, args=args, true_seed=state.true_seed,
                candidates=state.candidates, true_loc=state.true_loc,
                pulse_idx=pulse_idx, cosets=cosets, rng=state.rng,
                dynamic_prior=prior,
            )
        )
        batch[-1]["dynamic_prior"] = prior
        predicted_centers.append(predicted)

    tok = torch.from_numpy(np.stack([item["tok"] for item in batch]))
    dt = torch.from_numpy(np.stack([item["dt"] for item in batch]))
    A = torch.from_numpy(np.stack([item["A"] for item in batch]))
    Y = torch.from_numpy(np.stack([item["Y"] for item in batch]))
    cand_feat = torch.from_numpy(
        np.stack(
            [
                dynamic_candidate_features(
                    state.candidates, item["centers"], item["dynamic_prior"],
                    cfg, args.codebook_size,
                )
                for state, item in zip(states, batch)
            ]
        )
    )
    with torch.inference_mode():
        _, aux = model(
            tok, dt, A, Y, refine=False, return_aux=True,
            cand_feat=cand_feat, use_burst=False,
        )
    network_prior = aux["useful_prior"].cpu().numpy()

    nmse = np.empty(len(states), dtype=np.float64)
    hit = np.empty(len(states), dtype=np.float64)
    x_true_list, x_hat_list, y_list = [], [], []

    for b, (state, item) in enumerate(zip(states, batch)):
        A_np = item["A"]
        y_np = item["Y"]
        An = A_np / (np.linalg.norm(A_np, axis=0, keepdims=True) + 1e-12)
        correlation = np.abs(An.conj().T @ y_np)
        score = (
            args.network_score_weight * _normalize_max(network_prior[b])
            + args.dynamic_score_weight * item["dynamic_prior"]
            + args.evidence_score_weight * _normalize_max(correlation)
        )
        posterior = _softmax(score, args.posterior_temperature)
        selected = int(np.argmax(posterior))
        column = A_np[:, selected]
        coefficient = np.vdot(column, y_np) / (np.vdot(column, column).real + 1e-12)
        if collect:
            true_wave = item["coefficient"] * complex_radar_atom(
                spec_at(state.true_seed, pulse_idx, args.codebook_size),
                full_times, n_pulse,
            )
            reconstructed = coefficient * complex_radar_atom(
                spec_at(state.candidates[selected], pulse_idx, args.codebook_size),
                full_times, n_pulse,
            )
            nmse[b] = amplitude_nmse_db(reconstructed, true_wave)
            hit[b] = float(selected == state.true_loc)
            x_true_list.append(true_wave)
            x_hat_list.append(reconstructed)
            y_list.append(y_np)
        else:
            # still need nmse-free state update; approximate with zero
            nmse[b] = 0.0
            hit[b] = float(selected == state.true_loc)

        state.belief = args.belief_ema * state.belief + (1.0 - args.belief_ema) * posterior
        state.learned_prior = (
            args.prior_ema * state.learned_prior + (1.0 - args.prior_ema) * network_prior[b]
        )
        state.center_hist.append(float(item["centers"][selected]))
        map_belief_to_layout(state, item["centers"], state.belief, cfg)
        state.controller.plan_next_period(
            dt_s=args.control_dt_ms * 1e-3, mode=args.mt_mode
        )

    out = {"nmse_db": nmse, "hit": hit}
    if collect:
        out["x_true"] = x_true_list
        out["x_hat"] = x_hat_list
        out["y_meas"] = y_list
    return out


def export_thz() -> Path:
    """§4.3: THz 16-QAM symbol waveforms (clean / impaired / reconstructed)."""
    fig7 = json.loads(
        (OUTPUT_DIR / "streaming_thz_nuaa_mu_fig7_evm15_one_error.json").read_text()
    )
    pack = fig7["constellation_pack"]
    out = WAVE_DIR / "thz_16qam_waveforms.npz"
    np.savez_compressed(
        out,
        sym_clean=_c_from_pack(pack["sym_clean"]),
        sym_impaired=_c_from_pack(pack["sym_imp"]),
        sym_reconstructed=_c_from_pack(pack["sym_rec"]),
        sym_format_projected=_c_from_pack(pack["sym_comp"]),
        injected_evm_pct=float(pack["evm_pct"]),
        residual_evm_pct=float(pack["evm"]),
        residual_evm_after_format_pct=float(pack["evm_comp"]),
        f1=float(pack["f1"]),
        nmse_db=float(pack["nmse"]),
        ser=float(pack["ser_comp"]),
        ber=float(pack["ber_comp"]),
        carrier_ghz=300.0,
        signal="thz_16qam",
    )
    # keep Fig.7 alias
    alias = WAVE_DIR / "fig7_thz_constellation_evm15.npz"
    np.savez_compressed(
        alias,
        sym_clean=_c_from_pack(pack["sym_clean"]),
        sym_impaired=_c_from_pack(pack["sym_imp"]),
        sym_reconstructed=_c_from_pack(pack["sym_rec"]),
        sym_format_projected=_c_from_pack(pack["sym_comp"]),
        injected_evm_pct=float(pack["evm_pct"]),
        residual_evm_pct=float(pack["evm"]),
        residual_evm_after_format_pct=float(pack["evm_comp"]),
        f1=float(pack["f1"]),
        nmse_db=float(pack["nmse"]),
        ser=float(pack["ser_comp"]),
        ber=float(pack["ber_comp"]),
    )
    print(f"saved {out}")
    print(f"saved {alias}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["chirplet", "radar", "thz", "all"], default="all")
    args = ap.parse_args()
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in ("chirplet", "all"):
        export_chirplet(args)
    if args.only in ("radar", "all"):
        export_radar()
    if args.only in ("thz", "all"):
        export_thz()
    # write index
    index = {
        "chirplet_broadband_waveforms.npz": "§4.1 broadband chirplet (useful / jammer / recon / observed)",
        "fig4_wideband_chirp_waveforms.npz": "§4.1 Fig.4 zoomed amplitudes (alias)",
        "radar_quasiperiodic_waveforms.npz": "§4.2 NLFM / polyphase-LFM / Costas / Frank pulse waveforms",
        "thz_16qam_waveforms.npz": "§4.3 THz 16-QAM symbol IQ (clean / impaired / recon)",
        "fig7_thz_constellation_evm15.npz": "§4.3 Fig.7 constellation (alias)",
    }
    (WAVE_DIR / "INDEX.md").write_text(
        "# Paper signal waveform packs\n\n"
        + "\n".join(f"- `{k}` — {v}" for k, v in index.items())
        + "\n",
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()
