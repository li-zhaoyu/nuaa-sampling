"""Streaming complex radar pulses with regular inter-pulse evolution.

This is the streaming counterpart of ``exp_e5_radar_complex.py``.  Each optical
period is treated as one radar pulse whose intra-pulse modulation is complex
(phase-coded LFM by default), while the pulse parameters evolve regularly across
pulses.  Candidate columns are therefore *trajectory atoms*: for every buffered
event, the column value is generated from the candidate's predicted state at
that event's pulse index.
"""
from __future__ import annotations

# pylint: disable=import-error,wrong-import-position

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure
configure()

from nuaa.config import SystemConfig
from nuaa import reconstruct as R, signals as S, streaming as St
from experiments.exp_e5_radar_complex import (
    FAMILIES,
    FAMILY_SEED,
    PulseSpec,
    amplitude_nmse_db,
    complex_radar_atom,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
METHODS = ("lfm_static", "complex_static", "complex_bangbang")
METHOD_ALIASES = {
    "lfm_static": "lfm_chirplet",
    "complex_static": "complex_random",
    "complex_bangbang": "complex_adaptive",
}
STEP_CODE = np.array([0, 1, 3, 2, 5, 4, 6, 7, 5, 2, 1, 4], dtype=np.int64)


def slow_pulse_idx(tick: int, signal_pri_ticks: int) -> int:
    """慢时脉冲索引：与 §5.1 一致，PRI 尺度上保持场景系数（与 T0 异步）。"""
    return tick // max(1, signal_pri_ticks)


@dataclass(frozen=True)
class TrajectorySeed:
    family: str
    f0_base_ghz: float
    f0_amp_ghz: float
    phase: float
    bw_ghz: float
    code_base: int
    code_step: int
    n_chips: int


def spec_at(seed: TrajectorySeed, pulse_idx: int, codebook_size: int) -> PulseSpec:
    p = float(pulse_idx)
    f0 = (
        seed.f0_base_ghz
        + seed.f0_amp_ghz * np.sin(0.37 * p + seed.phase)
        + 0.35 * seed.f0_amp_ghz * np.sin(0.11 * p + 0.7 * seed.phase)
    )
    code = int(seed.code_base + seed.code_step * (pulse_idx // 2)
               + STEP_CODE[pulse_idx % len(STEP_CODE)]) % codebook_size
    return PulseSpec(seed.family, float(f0), seed.bw_ghz, code, seed.n_chips)


def trajectory_bin(seed: TrajectorySeed, pulse_idx: int, cfg: SystemConfig, codebook_size: int) -> int:
    spec = spec_at(seed, pulse_idx, codebook_size)
    f_mid = spec.f0_ghz + 0.5 * spec.bw_ghz
    return int(round(S.ghz_to_norm(f_mid) * cfg.N0)) % cfg.N0


def make_true_seed(args, rng: np.random.Generator) -> TrajectorySeed:
    base = float(rng.uniform(args.f_lo_ghz + args.f0_amp_ghz,
                             args.f_hi_ghz - args.bw_ghz - args.f0_amp_ghz))
    return TrajectorySeed(
        family=args.family,
        f0_base_ghz=base,
        f0_amp_ghz=args.f0_amp_ghz,
        phase=float(rng.uniform(0, 2 * np.pi)),
        bw_ghz=args.bw_ghz,
        code_base=int(rng.integers(0, args.codebook_size)),
        code_step=int(rng.choice([1, 3, 5])),
        n_chips=args.n_chips,
    )


def make_candidates(true_seed: TrajectorySeed, args, rng: np.random.Generator,
                    family_override: str | None = None) -> tuple[list[TrajectorySeed], int | None]:
    family = family_override or true_seed.family
    seeds = []
    for df in np.linspace(-args.f_jitter_ghz, args.f_jitter_ghz, 5):
        for da in (0.8, 1.0, 1.2):
            for dphase in np.linspace(-0.6, 0.6, 3):
                for dc in range(args.codebook_size):
                    seeds.append(TrajectorySeed(
                        family=family,
                        f0_base_ghz=true_seed.f0_base_ghz + float(df),
                        f0_amp_ghz=max(0.2, true_seed.f0_amp_ghz * float(da)),
                        phase=true_seed.phase + float(dphase),
                        bw_ghz=true_seed.bw_ghz,
                        code_base=dc,
                        code_step=true_seed.code_step,
                        n_chips=true_seed.n_chips,
                    ))
    true_loc = None
    if family_override is None:
        seeds.append(true_seed)
    # Stable de-duplication keeps the exact true seed when present.
    unique = list(dict.fromkeys(seeds))
    if family_override is None:
        true_loc = unique.index(true_seed)
    rest = [s for i, s in enumerate(unique) if i != true_loc]
    if len(rest) > args.nC - 1:
        rest = list(rng.choice(np.array(rest, dtype=object), size=args.nC - 1, replace=False))
    out = ([true_seed] if family_override is None else []) + rest
    rng.shuffle(out)
    loc = (out.index(true_seed) if family_override is None else None)
    return out[:args.nC], loc


def build_traj_matrix(cosets: np.ndarray, periods: np.ndarray, cands: list[TrajectorySeed],
                      cfg: SystemConfig, args) -> np.ndarray:
    cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    A = np.empty((cosets.size, len(cands)), dtype=np.complex128)
    for j, seed in enumerate(cands):
        col = np.empty(cosets.size, dtype=np.complex128)
        for p in np.unique(periods):
            mask = periods == p
            spec = spec_at(seed, int(p), args.codebook_size)
            col[mask] = complex_radar_atom(spec, cosets[mask], cfg.N0)
        A[:, j] = col
    return A


def synth_period_y(seed: TrajectorySeed, pulse_idx: int, cosets: np.ndarray,
                   cfg: SystemConfig, args, rng: np.random.Generator) -> np.ndarray:
    spec = spec_at(seed, pulse_idx, args.codebook_size)
    y_clean = complex_radar_atom(spec, cosets, cfg.N0)
    ref = float(np.mean(np.abs(y_clean) ** 2)) + 1e-30
    return S.add_measurement_noise(y_clean, args.snr, ref, rng)


def update_belief(ctrl: St.StreamingNUAAController, cands: list[TrajectorySeed],
                  corr: np.ndarray, pulse_idx: int, cfg: SystemConfig, args) -> None:
    pi = np.full(cfg.N0, ctrl.state.belief.floor, dtype=np.float64)
    score = np.asarray(corr, dtype=np.float64)
    score = score / (score.max() + 1e-12)
    for seed, val in zip(cands, score):
        b = trajectory_bin(seed, pulse_idx, cfg, args.codebook_size)
        pi[b] = max(pi[b], float(val) + ctrl.state.belief.floor)
    ctrl.state.belief.pi = pi / (pi.sum() + 1e-12)


def reconstruct_current(ctrl: St.StreamingNUAAController, cands: list[TrajectorySeed],
                        true_seed: TrajectorySeed, true_loc: int | None,
                        pulse_idx: int, cfg: SystemConfig, args,
                        use_prior: bool) -> dict:
    arr = ctrl.state.buffer.arrays()
    y = arr["y"]
    cosets = arr["coset"]
    periods = arr["period"]
    if y.size < 2 or not cands:
        return {"amp_nmse_db": 0.0, "mf_peak_db": -120.0, "atom_hit": False}
    A = build_traj_matrix(cosets, periods, cands, cfg, args)
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    corr = np.abs(An.conj().T @ y.reshape(-1, 1)).reshape(-1)
    if use_prior:
        bins = [trajectory_bin(s, pulse_idx, cfg, args.codebook_size) for s in cands]
        prior = ctrl.state.belief.pi[np.asarray(bins, dtype=np.int64)]
        prior = prior / (prior.max() + 1e-12)
        sp, Xh = R.somp_prior(A, y.reshape(-1, 1), 1, prior=prior, beta=2.0)
    else:
        sp, Xh = R.somp(A, y.reshape(-1, 1), 1)
    update_belief(ctrl, cands, corr, pulse_idx, cfg, args)
    true_x = complex_radar_atom(spec_at(true_seed, pulse_idx, args.codebook_size),
                                np.arange(cfg.N0), cfg.N0)
    x_hat = np.zeros_like(true_x)
    hit = False
    if len(sp):
        j = int(sp[0])
        hit = bool(true_loc is not None and j == int(true_loc))
        x_hat = Xh[j, 0] * complex_radar_atom(
            spec_at(cands[j], pulse_idx, args.codebook_size), np.arange(cfg.N0), cfg.N0)
    return {
        "amp_nmse_db": amplitude_nmse_db(x_hat, true_x),
        "mf_peak_db": S.matched_filter_peak_snr(x_hat, true_x),
        "atom_hit": hit,
    }


def run_one_trial(cfg: SystemConfig, args, trial_seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(trial_seed)
    true_seed = make_true_seed(args, rng)
    complex_cands, complex_loc = make_candidates(true_seed, args, rng)
    lfm_cands, _ = make_candidates(true_seed, args, rng, family_override="lfm")
    buffer_events = args.buffer_periods * cfg.L
    ctrls = {
        "lfm_static": St.StreamingNUAAController(cfg, K=1, buffer_events=buffer_events, seed=trial_seed + 1),
        "complex_static": St.StreamingNUAAController(cfg, K=1, buffer_events=buffer_events, seed=trial_seed + 2),
        "complex_bangbang": St.StreamingNUAAController(cfg, K=1, buffer_events=buffer_events, seed=trial_seed + 3),
    }
    curves = {m: {"amp_nmse": [], "mf": [], "hit": []} for m in METHODS}
    hit_time = {m: None for m in METHODS}
    best_nmse = {m: float("inf") for m in METHODS}
    process_hit = {m: False for m in METHODS}

    for tick in range(args.ticks):
        pulse_idx = slow_pulse_idx(tick, args.signal_pri_ticks)
        for method, ctrl in ctrls.items():
            cosets = ctrl.current_cosets()
            y = synth_period_y(true_seed, pulse_idx, cosets, cfg, args, rng)
            ctrl.append_period_measurement(y)
            if method == "lfm_static":
                rec = reconstruct_current(ctrl, lfm_cands, true_seed, None, pulse_idx, cfg, args, use_prior=False)
                mode = "static_hold"
            elif method == "complex_static":
                rec = reconstruct_current(ctrl, complex_cands, true_seed, complex_loc, pulse_idx, cfg, args, use_prior=False)
                mode = "static_hold"
            else:
                # Early trajectory belief can be wrong; use it for the slow EDL
                # planner only, while reconstruction remains evidence-driven.
                rec = reconstruct_current(ctrl, complex_cands, true_seed, complex_loc, pulse_idx, cfg, args, use_prior=False)
                mode = "belief_bangbang"
            curves[method]["amp_nmse"].append(float(rec["amp_nmse_db"]))
            curves[method]["mf"].append(float(rec["mf_peak_db"]))
            curves[method]["hit"].append(float(rec["atom_hit"]))
            best_nmse[method] = min(best_nmse[method], float(rec["amp_nmse_db"]))
            if rec["amp_nmse_db"] <= args.target_nmse_db:
                process_hit[method] = True
            if hit_time[method] is None and rec["amp_nmse_db"] <= args.target_nmse_db:
                hit_time[method] = (tick + 1) * args.control_dt_ms * 1e-3
            ctrl.plan_next_period(dt_s=args.control_dt_ms * 1e-3, mode=mode)

    ledgers = {}
    for method, ctrl in ctrls.items():
        led = ctrl.acquisition_ledger()
        led["physical_T_acq_s"] = args.ticks * args.control_dt_ms * 1e-3
        led["physical_events"] = led["physical_T_acq_s"] * cfg.f_avg_actual_GHz() * 1e9
        led["materialized_events"] = args.ticks * cfg.L
        ledgers[method] = led
    return curves, {"hit_time": hit_time, "ledgers": ledgers, "true_seed": asdict(true_seed),
                    "best_nmse": best_nmse, "process_hit": process_hit}


def aggregate(all_curves: list[dict], aux: list[dict]) -> dict:
    results = {}
    for method in METHODS:
        nmse_final = [c[method]["amp_nmse"][-1] for c in all_curves]
        hit_final = [c[method]["hit"][-1] for c in all_curves]
        hits = [a["hit_time"][method] for a in aux if a["hit_time"][method] is not None]
        best_nmse = [a["best_nmse"][method] for a in aux]
        process_hits = [float(a["process_hit"][method]) for a in aux]
        mat = np.array([c[method]["amp_nmse"] for c in all_curves], dtype=np.float64)
        hit_mat = np.array([c[method]["hit"] for c in all_curves], dtype=np.float64)
        results[method] = {
            "paper_alias": METHOD_ALIASES[method],
            "final_amp_nmse_med_db": float(np.median(nmse_final)),
            "best_amp_nmse_med_db": float(np.median(best_nmse)),
            "process_hit_rate": float(np.mean(process_hits)),
            "final_hit_rate": float(np.mean(hit_final)),
            "time_to_target_s": float(np.median(hits)) if hits else None,
            "time_to_target_ms": float(np.median(hits) * 1e3) if hits else None,
            "tau_movement_ps_mean": float(np.mean([a["ledgers"][method]["tau_movement_ps"] for a in aux])),
            "strobe_feasible_rate_mean": float(np.mean([a["ledgers"][method]["strobe_feasible_rate"] for a in aux])),
            "tick_curve": {
                "amp_nmse_med": np.median(mat, axis=0).tolist(),
                "hit_rate": np.mean(hit_mat, axis=0).tolist(),
            },
            "example_curve": all_curves[0][method],
            "example_ledger": aux[0]["ledgers"][method],
        }
    return results


def run_family(cfg: SystemConfig, args, family: str) -> tuple[list[dict], list[dict]]:
    args.family = family
    all_curves, aux = [], []
    for tr in range(args.trials):
        trial_seed = args.seed + 1000 * tr + 10_000 * FAMILY_SEED[family]
        curves, meta = run_one_trial(cfg, args, trial_seed)
        all_curves.append(curves)
        aux.append(meta)
        summary = " ".join(f"{m}:{curves[m]['amp_nmse'][-1]:+.1f}dB/{curves[m]['hit'][-1]:.0f}"
                           for m in METHODS)
        print(f"  [{family}] trial={tr+1}/{args.trials} final {summary}", flush=True)
    summary = aggregate(all_curves, aux)
    for m, r in summary.items():
        print(f"  [{family}] {METHOD_ALIASES[m]:18s} final={r['final_amp_nmse_med_db']:+.1f}dB "
              f"best={r['best_amp_nmse_med_db']:+.1f}dB proc_hit={r['process_hit_rate']:.0%} "
              f"t_hit={r['time_to_target_ms']}ms", flush=True)
    return all_curves, aux, summary


def run(cfg: SystemConfig, args) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    pri_ms = args.signal_pri_ticks * args.control_dt_ms
    results = {
        "config": cfg.summary(),
        "params": vars(args),
        "methods": list(METHODS),
        "method_aliases": METHOD_ALIASES,
        "description": "Streaming complex intra-pulse radar with regular inter-pulse evolution.",
        "by_family": {},
        "pooled": {},
    }
    print(f"streaming complex radar | N0={cfg.N0} families={','.join(args.families)} "
          f"ticks={args.ticks} ({args.ticks * args.control_dt_ms:g} ms) "
          f"tau={args.pulse_width_ns:g}ns PRI≈{pri_ms:g}ms "
          f"buffer={args.buffer_periods} SNR={args.snr:g}dB trials={args.trials}", flush=True)
    pooled_curves, pooled_aux = [], []
    for family in args.families:
        all_curves, aux, summary = run_family(cfg, args, family)
        results["by_family"][family] = {
            "methods_summary": summary,
            "trial_examples": [a["true_seed"] for a in aux[:3]],
        }
        pooled_curves.extend(all_curves)
        pooled_aux.extend(aux)
    results["pooled"] = aggregate(pooled_curves, pooled_aux)
    print("POOLED (all families):", flush=True)
    for m, r in results["pooled"].items():
        print(f"  {METHOD_ALIASES[m]:18s} final={r['final_amp_nmse_med_db']:+.1f}dB "
              f"best={r['best_amp_nmse_med_db']:+.1f}dB proc_hit={r['process_hit_rate']:.0%} "
              f"t_hit={r['time_to_target_ms']}ms Δtau={r['tau_movement_ps_mean']:.1f}ps", flush=True)
    out = os.path.join(OUT_DIR, f"streaming_radar_complex_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--family", choices=list(FAMILIES), default="phase_lfm",
                    help="(deprecated) single family; use --families")
    ap.add_argument("--families", nargs="+", default=None, choices=list(FAMILIES))
    ap.add_argument("--nC", type=int, default=48)
    ap.add_argument("--f-lo-ghz", type=float, default=20.0)
    ap.add_argument("--f-hi-ghz", type=float, default=80.0)
    ap.add_argument("--bw-ghz", type=float, default=20.0)
    ap.add_argument("--f0-amp-ghz", type=float, default=6.0)
    ap.add_argument("--f-jitter-ghz", type=float, default=6.0)
    ap.add_argument("--n-chips", type=int, default=64)
    ap.add_argument("--codebook-size", type=int, default=8)
    ap.add_argument("--snr", type=float, default=-10.0)
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--buffer-periods", type=int, default=16)
    ap.add_argument("--signal-pri-ticks", type=int, default=32,
                    help="PRI_ms ≈ signal_pri_ticks × control_dt_ms")
    ap.add_argument("--pulse-width-ns", type=float, default=16.0,
                    help="Radar pulse width τ (aligned with §5.1; logged in params)")
    ap.add_argument("--control-dt-ms", type=float, default=10.0)
    ap.add_argument("--target-nmse-db", type=float, default=-10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.families is None:
        args.families = [args.family]
    if args.quick:
        args.ticks = 8
        args.trials = 3
        args.nC = 32
        args.buffer_periods = 3
        args.tag = "quick" if args.tag == "run" else args.tag
    cfg = SystemConfig(N0=args.n0)
    run(cfg, args)


if __name__ == "__main__":
    main()
