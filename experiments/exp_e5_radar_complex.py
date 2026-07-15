"""E5b: complex intra-pulse radar modulation beyond simple LFM.

This experiment keeps the same non-uniform optical sampling interface as the
LFM radar test, but changes the radar pulse family to common complex
intra-pulse modulations:

  - NLFM: sinusoidally warped instantaneous frequency;
  - phase_lfm: LFM with per-chip BPSK/QPSK phase coding;
  - costas: stepped-frequency Costas-like chip code;
  - frank: Frank polyphase code.

The main contrast is deliberately simple:

  - lfm_chirplet: a plain LFM/chirplet dictionary, which is the old assumption;
  - complex_random: matched complex waveform candidates under random layouts;
  - complex_adaptive: the same matched candidates with NUAA-style adaptive layouts.

The output answers whether the current evidence covers radar waveforms whose
intra-pulse modulation is not just a linear chirp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure
configure()

from nuaa.config import SystemConfig
from nuaa import layout as L, metrics as Met, policy as P, reconstruct as R, signals as S

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
FAMILIES = ("nlfm", "phase_lfm", "costas", "frank")
METHODS = ("lfm_chirplet", "complex_random", "complex_adaptive")
FAMILY_SEED = {name: i + 1 for i, name in enumerate(FAMILIES)}


@dataclass(frozen=True)
class PulseSpec:
    family: str
    f0_ghz: float
    bw_ghz: float
    code_id: int
    n_chips: int


def event_time_indices(traj, cfg: SystemConfig) -> np.ndarray:
    idx = []
    for p, cs in enumerate(traj):
        for c in np.asarray(cs, dtype=np.int64).reshape(-1):
            idx.append(p * cfg.N0 + int(c))
    return np.asarray(idx, dtype=np.int64)


def _chip_indices(t: np.ndarray, N_ref: int, n_chips: int) -> np.ndarray:
    chip = np.floor(t * n_chips / max(1, N_ref)).astype(np.int64)
    return np.clip(chip, 0, n_chips - 1)


@lru_cache(maxsize=None)
def _phase_code(family: str, code_id: int, n_chips: int) -> np.ndarray:
    if family == "phase_lfm":
        rng = np.random.default_rng(10_000 + code_id)
        # Mixed BPSK/QPSK phase codes represent modern pulse-compression codes
        # without committing to one proprietary radar codebook.
        alphabet = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        return rng.choice(alphabet, size=n_chips)
    if family == "frank":
        m = int(round(np.sqrt(n_chips)))
        if m * m != n_chips:
            raise ValueError("Frank code requires n_chips to be a square number")
        p = np.arange(n_chips) // m
        q = np.arange(n_chips) % m
        return 2.0 * np.pi * ((p * q + code_id) % m) / m
    return np.zeros(n_chips, dtype=np.float64)


@lru_cache(maxsize=None)
def _costas_perm(code_id: int, n_chips: int) -> np.ndarray:
    rng = np.random.default_rng(20_000 + code_id)
    return rng.permutation(n_chips).astype(np.float64)


def complex_radar_atom(spec: PulseSpec, times: np.ndarray, N_ref: int) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    u = t / max(1.0, float(N_ref - 1))
    f0 = S.ghz_to_norm(spec.f0_ghz)
    bw = S.ghz_to_norm(spec.bw_ghz)

    if spec.family == "costas":
        chip = _chip_indices(t, N_ref, spec.n_chips)
        perm = _costas_perm(spec.code_id, spec.n_chips)
        f_chip = f0 + bw * perm[chip] / max(1, spec.n_chips - 1)
        # Continuous phase for stepped-frequency chips.
        if t.size == N_ref and np.all(np.diff(t) == 1):
            cycles = np.cumsum(f_chip)
        else:
            # Closed-form per-chip integral for arbitrary sample times.
            samples_per_chip = float(N_ref) / spec.n_chips
            chip_start = chip * samples_per_chip
            prefix = np.concatenate([[0.0], np.cumsum(f0 + bw * perm / max(1, spec.n_chips - 1))])
            cycles = samples_per_chip * prefix[chip] + (t - chip_start) * f_chip
        return np.exp(1j * 2.0 * np.pi * cycles).astype(np.complex128)

    k = bw / max(1.0, float(N_ref - 1))
    cycles = f0 * t + 0.5 * k * t ** 2
    if spec.family == "nlfm":
        # Smoothly warped instantaneous frequency: f(t)=f0+bw*(u+beta*sin(2piu)).
        beta = 0.11 + 0.015 * (spec.code_id % 5)
        cycles = f0 * t + bw * (N_ref - 1) * (
            0.5 * u ** 2 + beta * (1.0 - np.cos(2.0 * np.pi * u)) / (2.0 * np.pi)
        )
    elif spec.family in ("phase_lfm", "frank"):
        chip = _chip_indices(t, N_ref, spec.n_chips)
        cycles = cycles + _phase_code(spec.family, spec.code_id, spec.n_chips)[chip] / (2.0 * np.pi)
    elif spec.family != "lfm":
        raise ValueError(f"unknown radar family {spec.family}")
    return np.exp(1j * 2.0 * np.pi * cycles).astype(np.complex128)


def make_true_spec(args, rng: np.random.Generator, family: str) -> PulseSpec:
    f0 = float(rng.uniform(args.f_lo_ghz, args.f_hi_ghz - args.bw_ghz))
    code_id = int(rng.integers(0, args.codebook_size))
    return PulseSpec(family=family, f0_ghz=f0, bw_ghz=args.bw_ghz,
                     code_id=code_id, n_chips=args.n_chips)


def make_complex_candidates(true_spec: PulseSpec, args, rng: np.random.Generator) -> tuple[list[PulseSpec], int]:
    cands = [true_spec]
    f_offsets = np.linspace(-args.f_jitter_ghz, args.f_jitter_ghz, 5)
    bw_scales = np.linspace(0.85, 1.15, 3)
    for fam in FAMILIES:
        for cid in range(args.codebook_size):
            for df in f_offsets:
                for bs in bw_scales:
                    spec = PulseSpec(fam, true_spec.f0_ghz + float(df),
                                     true_spec.bw_ghz * float(bs), cid, args.n_chips)
                    if spec != true_spec:
                        cands.append(spec)
    # Keep the true waveform and a reproducible local prior cloud around it.
    rest = np.array(cands[1:], dtype=object)
    if len(rest) > args.nC - 1:
        rest = rng.choice(rest, size=args.nC - 1, replace=False)
    out = [true_spec] + list(rest)
    rng.shuffle(out)
    loc = int(next(i for i, s in enumerate(out) if s == true_spec))
    return out, loc


def make_lfm_candidates(true_spec: PulseSpec, args) -> tuple[list[PulseSpec], int]:
    # Old assumption: no phase/frequency coding, only a chirplet grid near the same band.
    f_grid = np.linspace(true_spec.f0_ghz - args.f_jitter_ghz,
                         true_spec.f0_ghz + args.f_jitter_ghz, args.n_f0)
    bw_grid = np.linspace(0.75 * true_spec.bw_ghz, 1.25 * true_spec.bw_ghz, args.n_k)
    cands = [PulseSpec("lfm", float(f), float(bw), 0, true_spec.n_chips)
             for f in f_grid for bw in bw_grid]
    return cands[:args.nC], 0


def build_matrix(cands: list[PulseSpec], time_idx: np.ndarray, N_ref: int) -> np.ndarray:
    A = np.empty((len(time_idx), len(cands)), dtype=np.complex128)
    for j, spec in enumerate(cands):
        A[:, j] = complex_radar_atom(spec, time_idx, N_ref)
    return A


def amplitude_nmse_db(x_hat: np.ndarray, x_ref: np.ndarray) -> float:
    err = np.linalg.norm(x_hat.reshape(-1) - x_ref.reshape(-1)) ** 2
    ref = np.linalg.norm(x_ref.reshape(-1)) ** 2 + 1e-20
    return float(10.0 * np.log10(err / ref + 1e-20))


def representative_bins(cands: list[PulseSpec], cfg: SystemConfig) -> list[int]:
    bins = []
    for spec in cands:
        if spec.family == "costas":
            f_mid = spec.f0_ghz + 0.5 * spec.bw_ghz
        else:
            f_mid = spec.f0_ghz + 0.5 * spec.bw_ghz
        bins.append(int(round(S.ghz_to_norm(f_mid) * cfg.N0)) % cfg.N0)
    return bins


def measure_and_reconstruct(cands: list[PulseSpec], true_loc: int | None, true_x: np.ndarray,
                            true_y_spec: PulseSpec, traj, cfg: SystemConfig, args,
                            rng: np.random.Generator) -> dict:
    N_ref = true_x.size
    tidx = event_time_indices(traj, cfg)
    A = build_matrix(cands, tidx, N_ref)
    y_clean = complex_radar_atom(true_y_spec, tidx, N_ref)
    ref = float(np.mean(np.abs(y_clean) ** 2)) + 1e-30
    y = S.add_measurement_noise(y_clean, args.snr, ref, rng)
    sp, _ = R.somp(A, y.reshape(-1, 1), 1)
    alpha = np.zeros(len(cands), dtype=np.complex128)
    if len(sp):
        alpha[sp] = (np.linalg.pinv(A[:, sp]) @ y.reshape(-1, 1))[:, 0]
    x_hat = np.zeros_like(true_x)
    if len(sp):
        j = int(sp[0])
        x_hat = alpha[j] * complex_radar_atom(cands[j], np.arange(N_ref), N_ref)
    nmse = Met.nmse_db(x_hat, true_x)
    amp_nmse = amplitude_nmse_db(x_hat, true_x)
    mf = S.matched_filter_peak_snr(x_hat, true_x)
    hit = bool(true_loc is not None and len(sp) and int(sp[0]) == int(true_loc))
    return {"si_nmse_db": float(nmse), "amp_nmse_db": float(amp_nmse),
            "mf_peak_db": float(mf), "atom_hit": hit}


def run_family(cfg: SystemConfig, args, family: str, n_steps: int, trial: int) -> dict:
    rng = np.random.default_rng(args.seed + 100_000 * n_steps + 1_000 * trial + FAMILY_SEED[family])
    N_ref = cfg.N0 * args.n_periods
    true_spec = make_true_spec(args, rng, family)
    true_x = complex_radar_atom(true_spec, np.arange(N_ref), N_ref)

    complex_cands, complex_loc = make_complex_candidates(true_spec, args, rng)
    lfm_cands, _ = make_lfm_candidates(true_spec, args)

    rand_traj = [L.gen_fixed_random(cfg, rng) for _ in range(n_steps)]
    pois_traj = [L.gen_poisson_gap(cfg, rng) for _ in range(n_steps)]
    acc = [L.gen_fixed_random(cfg, rng)]
    belief = representative_bins(complex_cands, cfg)
    for _ in range(n_steps - 1):
        acc.append(P.select_next_coset_set(acc, belief, cfg, rng, n_cand=args.nuaa_cand))

    return {
        "lfm_chirplet": measure_and_reconstruct(
            lfm_cands, None, true_x, true_spec, rand_traj, cfg, args, rng),
        "complex_random": measure_and_reconstruct(
            complex_cands, complex_loc, true_x, true_spec, pois_traj, cfg, args, rng),
        "complex_adaptive": measure_and_reconstruct(
            complex_cands, complex_loc, true_x, true_spec, acc, cfg, args, rng),
        "true_spec": asdict(true_spec),
    }


def summarize(items: list[dict]) -> dict:
    return {
        "amp_nmse_med_db": float(np.median([x["amp_nmse_db"] for x in items])),
        "amp_nmse_p90_db": float(np.percentile([x["amp_nmse_db"] for x in items], 90)),
        "si_nmse_med_db": float(np.median([x["si_nmse_db"] for x in items])),
        "mf_peak_med_db": float(np.median([x["mf_peak_db"] for x in items])),
        "atom_hit_rate": float(np.mean([x["atom_hit"] for x in items])),
    }


def run(cfg: SystemConfig, args) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "config": cfg.summary(),
        "params": vars(args),
        "families": list(args.families),
        "methods": list(METHODS),
        "budget_sweep": {},
        "note": "lfm_chirplet is the old simple-LFM dictionary; complex_* use matched complex intra-pulse waveform candidates.",
    }
    print(f"E5b complex radar modulation | {cfg.summary()} families={','.join(args.families)} "
          f"SNR={args.snr:g}dB n_periods={args.n_periods}", flush=True)

    for n_steps in args.steps:
        per_family = {}
        pooled = {m: [] for m in METHODS}
        for family in args.families:
            runs = {m: [] for m in METHODS}
            examples = []
            for tr in range(args.trials):
                out = run_family(cfg, args, family, n_steps, tr)
                examples.append(out["true_spec"])
                for m in METHODS:
                    runs[m].append(out[m])
                    pooled[m].append(out[m])
            per_family[family] = {
                "events": cfg.L * n_steps,
                "methods": {m: summarize(runs[m]) for m in METHODS},
                "example_true_specs": examples[:3],
            }
        results["budget_sweep"][str(n_steps)] = {
            "events": cfg.L * n_steps,
            "families": per_family,
            "pooled": {m: summarize(pooled[m]) for m in METHODS},
        }
        line = " | ".join(
            f"{m}: ampNMSE={results['budget_sweep'][str(n_steps)]['pooled'][m]['amp_nmse_med_db']:+.1f}dB "
            f"hit={results['budget_sweep'][str(n_steps)]['pooled'][m]['atom_hit_rate']:.0%}"
            for m in METHODS)
        print(f"[budget] events={cfg.L*n_steps:3d}  {line}", flush=True)

    out_json = os.path.join(OUT_DIR, f"e5_radar_complex_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out_json}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=5000)
    ap.add_argument("--n-periods", type=int, default=40)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--nC", type=int, default=48)
    ap.add_argument("--n-f0", type=int, default=8)
    ap.add_argument("--n-k", type=int, default=6)
    ap.add_argument("--f-lo-ghz", type=float, default=20.0)
    ap.add_argument("--f-hi-ghz", type=float, default=80.0)
    ap.add_argument("--bw-ghz", type=float, default=20.0)
    ap.add_argument("--f-jitter-ghz", type=float, default=5.0)
    ap.add_argument("--n-chips", type=int, default=64)
    ap.add_argument("--codebook-size", type=int, default=8)
    ap.add_argument("--snr", type=float, default=-15.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--nuaa-cand", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_periods = 4
        args.trials = 5
        args.steps = [4, 8, 16]
        args.nC = 24
        args.n_f0 = 6
        args.n_k = 4
        args.snr = 0.0
        args.tag = "quick" if args.tag == "run" else args.tag

    cfg = SystemConfig(N0=args.n0)
    run(cfg, args)


if __name__ == "__main__":
    main()
