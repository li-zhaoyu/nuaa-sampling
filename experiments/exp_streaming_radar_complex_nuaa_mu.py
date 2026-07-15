"""NUAA-MU tracking of quasi-periodically evolving intra-pulse radar signals.

This experiment implements the §4.2 deployment path used by the paper:

* one scene-class NUAA-MU model is pretrained across NLFM, polyphase-LFM,
  Costas, and Frank waveforms;
* an offline GRU evolution prior predicts the next carrier-center state;
* the predicted prior is fused with EMA-updated online candidate belief;
* every 100 ms slow-loop point reconstructs one M=200 observation window;
* the radar state advances every 10 ms PRI, so adjacent reconstruction points
  are separated by ten trajectory updates.

CPU efficiency is obtained by batching all trials for one family, building only
candidate columns (never an N0-wide Fourier matrix), caching deterministic
phase/Costas codes, and bypassing per-event Python record objects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.cpu_env import configure

configure()

from experiments.exp_e5_radar_complex import (  # noqa: E402
    FAMILIES,
    FAMILY_SEED,
    amplitude_nmse_db,
    complex_radar_atom,
)
from experiments.exp_streaming_radar_complex import (  # noqa: E402
    TrajectorySeed,
    make_candidates,
    make_true_seed,
    spec_at,
    trajectory_bin,
)
from experiments.train_nuaa_mu import _event_tokens  # noqa: E402
from models.evolution_prior import (  # noqa: E402
    EvolutionPredictor,
    predict_next_centers,
    pretrain_evolution,
)
from models.nuaa_mu import NUAAMU  # noqa: E402
from nuaa import layout as L, signals as S, streaming as St  # noqa: E402
from nuaa.config import SystemConfig  # noqa: E402


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FAMILY_LABELS = {
    "nlfm": "NLFM",
    "phase_lfm": "polyphase-LFM",
    "costas": "Costas",
    "frank": "Frank",
}
FAMILY_INDEX = {family: i for i, family in enumerate(FAMILIES)}


@dataclass
class TrialState:
    rng: np.random.Generator
    true_seed: TrajectorySeed
    candidates: list[TrajectorySeed]
    true_loc: int
    controller: St.StreamingNUAAController
    belief: np.ndarray
    learned_prior: np.ndarray
    center_hist: list[float]


def _family_args(args, family: str) -> SimpleNamespace:
    values = vars(args).copy()
    values["family"] = family
    return SimpleNamespace(**values)


def _circular_distance(values: np.ndarray, center: float, period: int) -> np.ndarray:
    diff = np.abs(np.asarray(values, dtype=np.float64) - float(center))
    return np.minimum(diff, float(period) - diff)


def _normalize_max(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float64), floor)
    return arr / (float(np.max(arr)) + 1e-12)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-3)
    z = z - float(np.max(z))
    out = np.exp(np.clip(z, -60.0, 0.0))
    return out / (float(np.sum(out)) + 1e-12)


def pulse_sample_times(
    cosets: np.ndarray,
    cfg: SystemConfig,
    window_periods: int,
    pulse_width_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return waveform-grid times, repeated cosets, and acquisition-grid times."""
    n_pulse = int(round(float(pulse_width_ns) * 1000.0 / cfg.eta_ps))
    n_segments = max(1, int(np.ceil(n_pulse / cfg.N0)))
    periods = np.arange(int(window_periods), dtype=np.int64)
    segment = periods % n_segments
    cos = np.sort(np.asarray(cosets, dtype=np.int64).reshape(-1))
    waveform_times = (
        segment[:, None] * cfg.N0 + cos[None, :]
    ).reshape(-1)
    acquisition_times = (
        periods[:, None] * cfg.N0 + cos[None, :]
    ).reshape(-1)
    event_cosets = np.broadcast_to(cos[None, :], (len(periods), len(cos))).reshape(-1)
    return waveform_times, event_cosets, acquisition_times


def build_candidate_matrix(
    candidates: list[TrajectorySeed],
    pulse_idx: int,
    waveform_times: np.ndarray,
    n_pulse: int,
    codebook_size: int,
) -> np.ndarray:
    """Build only the nC trajectory columns required by NUAA-MU."""
    scale = np.float32(1.0 / np.sqrt(max(1, len(waveform_times))))
    columns = [
        complex_radar_atom(
            spec_at(seed, int(pulse_idx), codebook_size),
            waveform_times,
            n_pulse,
        ).astype(np.complex64, copy=False)
        for seed in candidates
    ]
    return np.stack(columns, axis=1) * scale


def candidate_centers(
    candidates: list[TrajectorySeed],
    pulse_idx: int,
    cfg: SystemConfig,
    codebook_size: int,
) -> np.ndarray:
    return np.fromiter(
        (
            trajectory_bin(seed, int(pulse_idx), cfg, codebook_size)
            for seed in candidates
        ),
        dtype=np.float64,
        count=len(candidates),
    )


def append_window_phase(tokens: np.ndarray, pulse_idx: int, phase_period: int) -> np.ndarray:
    phase = 2.0 * np.pi * (int(pulse_idx) % max(1, int(phase_period))) / max(
        1, int(phase_period)
    )
    extra = np.empty((len(tokens), 2), dtype=np.float32)
    extra[:, 0] = np.sin(phase)
    extra[:, 1] = np.cos(phase)
    return np.concatenate([tokens, extra], axis=1)


def normalized_dt(acquisition_times: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    t = np.asarray(acquisition_times, dtype=np.float64)
    delta = np.diff(t, prepend=t[0])
    return np.log1p(np.abs(delta) / max(1.0, cfg.N0 / cfg.L)).astype(np.float32)


def dynamic_candidate_features(
    candidates: list[TrajectorySeed],
    centers: np.ndarray,
    dynamic_prior: np.ndarray,
    cfg: SystemConfig,
    codebook_size: int,
) -> np.ndarray:
    feat = np.zeros((len(candidates), 4), dtype=np.float32)
    feat[:, 0] = (centers / float(cfg.N0)).astype(np.float32)
    feat[:, 1] = np.asarray(
        [seed.code_base / max(1, codebook_size - 1) for seed in candidates],
        dtype=np.float32,
    )
    feat[:, 2] = np.asarray(
        [FAMILY_INDEX[seed.family] / max(1, len(FAMILIES) - 1) for seed in candidates],
        dtype=np.float32,
    )
    feat[:, 3] = np.asarray(dynamic_prior, dtype=np.float32)
    return feat


def make_window(
    *,
    cfg: SystemConfig,
    args,
    true_seed: TrajectorySeed,
    candidates: list[TrajectorySeed],
    true_loc: int,
    pulse_idx: int,
    cosets: np.ndarray,
    rng: np.random.Generator,
    dynamic_prior: np.ndarray,
) -> dict:
    waveform_times, event_cosets, acquisition_times = pulse_sample_times(
        cosets, cfg, args.window_periods, args.pulse_width_ns
    )
    n_pulse = int(round(args.pulse_width_ns * 1000.0 / cfg.eta_ps))
    A = build_candidate_matrix(
        candidates, pulse_idx, waveform_times, n_pulse, args.codebook_size
    )
    coefficient = np.complex64(
        (0.85 + 0.30 * rng.random()) * np.exp(1j * rng.uniform(-np.pi, np.pi))
    )
    y_clean = A[:, int(true_loc)] * coefficient
    ref = float(np.mean(np.abs(y_clean) ** 2)) + 1e-30
    y = S.add_measurement_noise(y_clean, args.snr, ref, rng).astype(
        np.complex64, copy=False
    )
    tokens = append_window_phase(
        _event_tokens(y, event_cosets, cfg), pulse_idx, args.phase_period_pris
    )
    centers = candidate_centers(
        candidates, pulse_idx, cfg, args.codebook_size
    )
    cand_feat = dynamic_candidate_features(
        candidates, centers, dynamic_prior, cfg, args.codebook_size
    )
    target = np.zeros(len(candidates), dtype=np.complex64)
    target[int(true_loc)] = coefficient
    return {
        "tok": tokens,
        "dt": normalized_dt(acquisition_times, cfg),
        "A": A,
        "Y": y,
        "X": target,
        "centers": centers,
        "coefficient": coefficient,
        "waveform_times": waveform_times,
    }


def _training_dynamic_prior(
    centers: np.ndarray,
    true_center: float,
    cfg: SystemConfig,
    args,
    rng: np.random.Generator,
) -> np.ndarray:
    predicted = float(true_center) + rng.normal(
        0.0, max(1.0, args.prior_train_noise_bins)
    )
    distance = _circular_distance(centers, predicted, cfg.N0)
    return _normalize_max(
        np.exp(-0.5 * (distance / max(1.0, args.prior_sigma_bins)) ** 2)
    )


def generate_training_batch(
    cfg: SystemConfig,
    args,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    tok = np.zeros((batch_size, args.window_periods * cfg.L, 10), dtype=np.float32)
    dt = np.zeros((batch_size, args.window_periods * cfg.L), dtype=np.float32)
    A = np.zeros(
        (batch_size, args.window_periods * cfg.L, args.nC), dtype=np.complex64
    )
    Y = np.zeros((batch_size, args.window_periods * cfg.L), dtype=np.complex64)
    X = np.zeros((batch_size, args.nC), dtype=np.complex64)
    cand_feat = np.zeros((batch_size, args.nC, 4), dtype=np.float32)
    support = np.zeros(batch_size, dtype=np.int64)
    prior_target = np.zeros((batch_size, args.nC), dtype=np.float32)

    max_pulse = int(round(args.duration_s * 1000.0 / args.pri_ms))
    for b in range(batch_size):
        family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
        local_args = _family_args(args, family)
        true_seed = make_true_seed(local_args, rng)
        candidates, true_loc = make_candidates(true_seed, local_args, rng)
        if true_loc is None:
            raise RuntimeError("training candidate set does not contain true trajectory")
        pulse_idx = int(rng.integers(-args.history_ticks * args.pri_stride, max_pulse + 1))
        cosets = L.gen_fixed_random(cfg, rng)
        centers = candidate_centers(candidates, pulse_idx, cfg, args.codebook_size)
        prior = _training_dynamic_prior(
            centers, centers[int(true_loc)], cfg, args, rng
        )
        window = make_window(
            cfg=cfg,
            args=args,
            true_seed=true_seed,
            candidates=candidates,
            true_loc=int(true_loc),
            pulse_idx=pulse_idx,
            cosets=cosets,
            rng=rng,
            dynamic_prior=prior,
        )
        tok[b] = window["tok"]
        dt[b] = window["dt"]
        A[b] = window["A"]
        Y[b] = window["Y"]
        X[b] = window["X"]
        cand_feat[b] = dynamic_candidate_features(
            candidates, centers, prior, cfg, args.codebook_size
        )
        support[b] = int(true_loc)
        prior_target[b] = prior.astype(np.float32)

    return (
        torch.from_numpy(tok),
        torch.from_numpy(dt),
        torch.from_numpy(A),
        torch.from_numpy(Y),
        torch.from_numpy(X),
        torch.from_numpy(cand_feat),
        torch.from_numpy(support),
        torch.from_numpy(prior_target),
    )


def make_nuaa_mu(args) -> NUAAMU:
    return NUAAMU(
        d_in=10,
        nC=args.nC,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        K_unroll=args.unroll,
        K_sparse=1,
        cand_dim=4,
    )


def train_or_load_nuaa_mu(cfg: SystemConfig, args) -> tuple[NUAAMU, dict]:
    model = make_nuaa_mu(args)
    model_path = Path(args.model_path)
    if model_path.exists() and not args.force_train:
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"loaded radar NUAA-MU: {model_path}", flush=True)
        return model, checkpoint.get("training", {})

    rng = np.random.default_rng(args.seed + 50_001)
    torch.manual_seed(args.seed + 50_002)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    support_pos_weight = torch.tensor(float(max(1, args.nC - 1)))
    start = time.perf_counter()
    last = {}
    model.train()
    for step in range(args.train_iters):
        tok, dt, A, Y, X, cand_feat, support, prior_target = generate_training_batch(
            cfg, args, rng, args.batch
        )
        Xhat, aux = model(
            tok,
            dt,
            A,
            Y,
            refine=False,
            return_aux=True,
            cand_feat=cand_feat,
            use_burst=False,
        )
        rows = torch.arange(args.batch)
        target_mask = F.one_hot(support, num_classes=args.nC).to(torch.float32)
        support_loss = F.binary_cross_entropy_with_logits(
            aux["support_logits"],
            target_mask,
            pos_weight=support_pos_weight,
        )
        denom = X.abs().pow(2).sum(dim=1) + 1e-9
        rec_loss = ((Xhat - X).abs().pow(2).sum(dim=1) / denom).mean()
        selected_loss = (
            (Xhat[rows, support] - X[rows, support]).abs().pow(2)
            / (X[rows, support].abs().pow(2) + 1e-9)
        ).mean()
        prior_loss = F.mse_loss(aux["useful_prior"], prior_target)
        jammer_loss = F.binary_cross_entropy_with_logits(
            aux["jammer_logits"], torch.zeros_like(aux["jammer_logits"])
        )
        loss = (
            0.15 * rec_loss
            + 0.20 * selected_loss
            + 0.90 * support_loss
            + 0.35 * prior_loss
            + 0.03 * jammer_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % args.progress_every == 0 or step + 1 == args.train_iters:
            with torch.no_grad():
                pred = aux["support_logits"].argmax(dim=1)
                accuracy = float((pred == support).float().mean())
            last = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "support_loss": float(support_loss.detach()),
                "support_accuracy": accuracy,
            }
            print(
                "PRETRAIN "
                f"step={step + 1}/{args.train_iters} "
                f"loss={last['loss']:.4f} "
                f"support={last['support_loss']:.4f} "
                f"acc={accuracy:.2f}",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    training = {
        **last,
        "elapsed_s": elapsed,
        "iters": args.train_iters,
        "batch": args.batch,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "training": training,
            "model_config": {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "d_state": args.d_state,
                "unroll": args.unroll,
                "nC": args.nC,
                "cand_dim": 4,
            },
        },
        model_path,
    )
    print(f"saved radar NUAA-MU: {model_path}", flush=True)
    model.eval()
    return model, training


def generate_center_dataset(
    cfg: SystemConfig,
    args,
    rng: np.random.Generator,
    n_sequences: int,
    n_steps: int,
) -> np.ndarray:
    local_args = _family_args(args, "nlfm")
    pulse_indices = (
        np.arange(n_steps, dtype=np.int64) - int(args.history_ticks)
    ) * int(args.pri_stride)
    centers = np.empty((n_sequences, n_steps), dtype=np.float64)
    for i in range(n_sequences):
        seed = make_true_seed(local_args, rng)
        centers[i] = [
            trajectory_bin(seed, int(p), cfg, args.codebook_size)
            for p in pulse_indices
        ]
    return centers


def train_evolution_model(
    cfg: SystemConfig, args
) -> tuple[EvolutionPredictor, dict]:
    path = Path(args.evolution_model_path)
    if path.exists() and not args.force_train:
        checkpoint = torch.load(path, map_location="cpu")
        model = EvolutionPredictor(
            hidden=args.evolution_hidden, n_layers=1
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"loaded evolution prior: {path}", flush=True)
        return model, checkpoint.get("training", {})

    rng = np.random.default_rng(args.seed + 60_001)
    n_steps = args.ticks + args.history_ticks + 1
    train = generate_center_dataset(
        cfg, args, rng, args.evolution_sequences, n_steps
    )
    val = generate_center_dataset(
        cfg, args, rng, max(32, args.evolution_sequences // 4), n_steps
    )
    start = time.perf_counter()
    result = pretrain_evolution(
        train,
        val,
        cfg.N0,
        hidden=args.evolution_hidden,
        n_layers=1,
        epochs=args.evolution_epochs,
        lr=args.evolution_lr,
        seed=args.seed + 60_002,
    )
    training = {
        "elapsed_s": time.perf_counter() - start,
        "n_params": result.n_params,
        "train_loss": result.train_loss,
        "val_pred_mae_slots": result.val_pred_mae_slots,
        "persistence_mae_slots": result.persistence_mae_slots,
        "pulse_stride": args.pri_stride,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": result.model.state_dict(), "training": training},
        path,
    )
    print(
        "EVOLUTION PRETRAIN "
        f"MAE={result.val_pred_mae_slots:.1f} slots "
        f"persistence={result.persistence_mae_slots:.1f} slots",
        flush=True,
    )
    return result.model.eval(), training


def make_trial_states(
    cfg: SystemConfig,
    args,
    family: str,
) -> list[TrialState]:
    states = []
    local_args = _family_args(args, family)
    for trial in range(args.trials):
        seed = (
            args.seed
            + 1000 * trial
            + 10_000 * FAMILY_SEED[family]
        )
        rng = np.random.default_rng(seed)
        true_seed = make_true_seed(local_args, rng)
        candidates, true_loc = make_candidates(true_seed, local_args, rng)
        if true_loc is None:
            raise RuntimeError("evaluation candidate set does not contain true trajectory")
        states.append(
            TrialState(
                rng=rng,
                true_seed=true_seed,
                candidates=candidates,
                true_loc=int(true_loc),
                controller=St.StreamingNUAAController(
                    cfg, K=1, window_events=args.window_periods * cfg.L, seed=seed + 1
                ),
                belief=np.full(args.nC, 1.0 / args.nC, dtype=np.float64),
                learned_prior=np.full(args.nC, 1.0 / args.nC, dtype=np.float64),
                center_hist=[],
            )
        )
    return states


def predict_current_center(
    state: TrialState,
    evolution_model: EvolutionPredictor,
    cfg: SystemConfig,
) -> float:
    if not state.center_hist:
        return float(cfg.N0) * 0.5
    if len(state.center_hist) < 2:
        return float(state.center_hist[-1])
    history = np.asarray(state.center_hist, dtype=np.float64)[None, :]
    return float(predict_next_centers(evolution_model, history, cfg.N0)[0, -1])


def online_dynamic_prior(
    state: TrialState,
    centers: np.ndarray,
    predicted_center: float,
    cfg: SystemConfig,
    args,
) -> np.ndarray:
    distance = _circular_distance(centers, predicted_center, cfg.N0)
    evolution = _normalize_max(
        np.exp(-0.5 * (distance / max(1.0, args.prior_sigma_bins)) ** 2)
    )
    belief = _normalize_max(state.belief)
    learned = _normalize_max(state.learned_prior)
    fused = (
        args.evolution_prior_weight * evolution
        + args.belief_prior_weight * belief
        + args.learned_prior_weight * learned
    )
    return _normalize_max(fused)


def map_belief_to_layout(
    state: TrialState,
    centers: np.ndarray,
    posterior: np.ndarray,
    cfg: SystemConfig,
) -> None:
    full = np.full(cfg.N0, 1e-12, dtype=np.float64)
    for center, probability in zip(centers, posterior):
        idx = int(round(center)) % cfg.N0
        full[idx] += float(probability)
    state.controller.state.belief.pi = full / (float(np.sum(full)) + 1e-15)


def process_family_tick(
    cfg: SystemConfig,
    args,
    model: NUAAMU,
    evolution_model: EvolutionPredictor,
    states: list[TrialState],
    pulse_idx: int,
) -> dict:
    batch = []
    predicted_centers = []
    for state in states:
        centers = candidate_centers(
            state.candidates, pulse_idx, cfg, args.codebook_size
        )
        predicted = predict_current_center(state, evolution_model, cfg)
        prior = online_dynamic_prior(
            state, centers, predicted, cfg, args
        )
        cosets = state.controller.current_cosets()
        state.controller.state.acc_cosets.append(np.asarray(cosets).copy())
        batch.append(
            make_window(
                cfg=cfg,
                args=args,
                true_seed=state.true_seed,
                candidates=state.candidates,
                true_loc=state.true_loc,
                pulse_idx=pulse_idx,
                cosets=cosets,
                rng=state.rng,
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
                    state.candidates,
                    item["centers"],
                    item["dynamic_prior"],
                    cfg,
                    args.codebook_size,
                )
                for state, item in zip(states, batch)
            ]
        )
    )

    with torch.inference_mode():
        _, aux = model(
            tok,
            dt,
            A,
            Y,
            refine=False,
            return_aux=True,
            cand_feat=cand_feat,
            use_burst=False,
        )
    network_prior = aux["useful_prior"].cpu().numpy()

    nmse = np.empty(len(states), dtype=np.float64)
    hit = np.empty(len(states), dtype=np.float64)
    center_error = np.empty(len(states), dtype=np.float64)
    selected_index = np.empty(len(states), dtype=np.int64)
    n_pulse = int(round(args.pulse_width_ns * 1000.0 / cfg.eta_ps))
    full_times = np.arange(n_pulse, dtype=np.int64)

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
        selected_index[b] = selected
        column = A_np[:, selected]
        coefficient = np.vdot(column, y_np) / (
            np.vdot(column, column).real + 1e-12
        )
        true_wave = item["coefficient"] * complex_radar_atom(
            spec_at(state.true_seed, pulse_idx, args.codebook_size),
            full_times,
            n_pulse,
        )
        reconstructed = coefficient * complex_radar_atom(
            spec_at(state.candidates[selected], pulse_idx, args.codebook_size),
            full_times,
            n_pulse,
        )
        nmse[b] = amplitude_nmse_db(reconstructed, true_wave)
        hit[b] = float(selected == state.true_loc)
        true_center = item["centers"][state.true_loc]
        selected_center = item["centers"][selected]
        center_error[b] = float(
            _circular_distance(
                np.asarray([selected_center]), true_center, cfg.N0
            )[0]
        )
        state.belief = (
            args.belief_ema * state.belief
            + (1.0 - args.belief_ema) * posterior
        )
        state.learned_prior = (
            args.prior_ema * state.learned_prior
            + (1.0 - args.prior_ema) * network_prior[b]
        )
        state.center_hist.append(float(selected_center))
        map_belief_to_layout(
            state, item["centers"], state.belief, cfg
        )
        state.controller.plan_next_period(
            dt_s=args.control_dt_ms * 1e-3, mode=args.mt_mode
        )

    return {
        "nmse_db": nmse,
        "hit": hit,
        "center_error_bins": center_error,
        "selected_index": selected_index,
        "predicted_center": np.asarray(predicted_centers, dtype=np.float64),
    }


def aggregate_family(
    family: str,
    time_s: np.ndarray,
    nmse: np.ndarray,
    hit: np.ndarray,
    center_error: np.ndarray,
    states: list[TrialState],
    args,
) -> dict:
    best = np.minimum.accumulate(nmse, axis=1)
    success = np.any(nmse <= args.target_nmse_db, axis=1)
    first_success = []
    for curve in nmse:
        idx = np.flatnonzero(curve <= args.target_nmse_db)
        if idx.size:
            first_success.append(float(time_s[int(idx[0])]))
    return {
        "label": FAMILY_LABELS[family],
        "time_s": time_s.tolist(),
        "nmse_med_db": np.median(nmse, axis=0).tolist(),
        "nmse_p25_db": np.percentile(nmse, 25, axis=0).tolist(),
        "nmse_p75_db": np.percentile(nmse, 75, axis=0).tolist(),
        "nmse_best_med_db": np.median(best, axis=0).tolist(),
        "support_hit_rate": np.mean(hit, axis=0).tolist(),
        "center_error_med_bins": np.median(center_error, axis=0).tolist(),
        "trial_nmse_db": nmse.tolist(),
        "trial_support_hit": hit.tolist(),
        "final_nmse_med_db": float(np.median(nmse[:, -1])),
        "best_nmse_med_db": float(np.median(best[:, -1])),
        "success_rate": float(np.mean(success)),
        "median_first_success_s": (
            float(np.median(first_success)) if first_success else None
        ),
        "tau_movement_ps_mean": float(
            np.mean(
                [
                    sum(cmd.movement_ps for cmd in state.controller.state.command_log)
                    for state in states
                ]
            )
        ),
        "trajectory_examples": [
            asdict(state.true_seed) for state in states[:3]
        ],
    }


def run_family(
    cfg: SystemConfig,
    args,
    family: str,
    model: NUAAMU,
    evolution_model: EvolutionPredictor,
) -> dict:
    states = make_trial_states(cfg, args, family)

    # Historical windows initialize the online prior but are not plotted.
    for warm in range(-args.history_ticks, 0):
        process_family_tick(
            cfg,
            args,
            model,
            evolution_model,
            states,
            pulse_idx=warm * args.pri_stride,
        )

    time_s = np.linspace(0.0, args.duration_s, args.ticks)
    nmse = np.empty((args.trials, args.ticks), dtype=np.float64)
    hit = np.empty_like(nmse)
    center_error = np.empty_like(nmse)
    for tick, _ in enumerate(time_s):
        out = process_family_tick(
            cfg,
            args,
            model,
            evolution_model,
            states,
            pulse_idx=tick * args.pri_stride,
        )
        nmse[:, tick] = out["nmse_db"]
        hit[:, tick] = out["hit"]
        center_error[:, tick] = out["center_error_bins"]
        print(
            f"TRACK family={family:9s} "
            f"tick={tick + 1:02d}/{args.ticks} "
            f"t={time_s[tick]:.1f}s "
            f"NMSE={np.median(nmse[:, tick]):+.2f}dB "
            f"hit={np.mean(hit[:, tick]):.0%}",
            flush=True,
        )
    return aggregate_family(
        family, time_s, nmse, hit, center_error, states, args
    )


def run(args) -> dict:
    cfg = SystemConfig(N0=args.n0)
    model, model_training = train_or_load_nuaa_mu(cfg, args)
    evolution_model, evolution_training = train_evolution_model(cfg, args)
    model.eval()
    evolution_model.eval()

    start = time.perf_counter()
    by_family = {}
    for family in args.families:
        by_family[family] = run_family(
            cfg, args, family, model, evolution_model
        )
    elapsed = time.perf_counter() - start
    results = {
        "description": (
            "Pretrained NUAA-MU + GRU evolution prior + EMA online belief "
            "for quasi-periodically evolving complex radar waveforms."
        ),
        "config": cfg.summary(),
        "params": vars(args),
        "model_training": model_training,
        "evolution_training": evolution_training,
        "runtime_s": elapsed,
        "by_family": by_family,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"streaming_radar_complex_nuaa_mu_{args.tag}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"saved {output_path}", flush=True)
    print(f"tracking runtime={elapsed:.1f}s", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    from nuaa.repo_paths import OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--n0", type=int, default=5000)
    parser.add_argument(
        "--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES)
    )
    parser.add_argument("--nC", type=int, default=48)
    parser.add_argument("--f-lo-ghz", type=float, default=20.0)
    parser.add_argument("--f-hi-ghz", type=float, default=80.0)
    parser.add_argument("--bw-ghz", type=float, default=20.0)
    parser.add_argument("--f0-amp-ghz", type=float, default=6.0)
    parser.add_argument("--f-jitter-ghz", type=float, default=6.0)
    parser.add_argument("--n-chips", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=8)
    parser.add_argument("--snr", type=float, default=-10.0)
    parser.add_argument("--pulse-width-ns", type=float, default=16.0)
    parser.add_argument("--pri-ms", type=float, default=10.0)
    parser.add_argument("--control-dt-ms", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--window-periods", type=int, default=40)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--history-ticks", type=int, default=4)
    parser.add_argument("--target-nmse-db", type=float, default=-10.0)
    parser.add_argument(
        "--mt-mode",
        choices=["belief_bangbang", "static_hold", "random_slow_scan"],
        default="belief_bangbang",
    )

    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=24)
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=240)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(OUTPUT_DIR / "radar_complex_nuaa_mu_pretrained.pt"),
    )

    parser.add_argument("--evolution-hidden", type=int, default=64)
    parser.add_argument("--evolution-epochs", type=int, default=80)
    parser.add_argument("--evolution-sequences", type=int, default=256)
    parser.add_argument("--evolution-lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--evolution-model-path",
        type=str,
        default=str(OUTPUT_DIR / "radar_complex_evolution_prior.pt"),
    )
    parser.add_argument("--prior-sigma-bins", type=float, default=120.0)
    parser.add_argument("--prior-train-noise-bins", type=float, default=45.0)
    parser.add_argument("--evolution-prior-weight", type=float, default=0.55)
    parser.add_argument("--belief-prior-weight", type=float, default=0.30)
    parser.add_argument("--learned-prior-weight", type=float, default=0.15)
    parser.add_argument("--network-score-weight", type=float, default=0.45)
    parser.add_argument("--dynamic-score-weight", type=float, default=0.40)
    parser.add_argument("--evidence-score-weight", type=float, default=0.15)
    parser.add_argument("--posterior-temperature", type=float, default=0.18)
    parser.add_argument("--belief-ema", type=float, default=0.70)
    parser.add_argument("--prior-ema", type=float, default=0.70)
    parser.add_argument("--phase-period-pris", type=int, default=57)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="2s")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ratio = args.control_dt_ms / args.pri_ms
    args.pri_stride = int(round(ratio))
    if not np.isclose(ratio, args.pri_stride):
        parser.error("--control-dt-ms must be an integer multiple of --pri-ms")
    args.ticks = int(round(args.duration_s * 1000.0 / args.control_dt_ms)) + 1
    if args.window_periods * 5 != 200:
        parser.error("paper §4.2 requires M=200, i.e. --window-periods 40")
    if args.quick:
        args.families = ["nlfm"]
        args.duration_s = 0.2
        args.ticks = 3
        args.trials = 2
        args.history_ticks = 2
        args.train_iters = 2
        args.batch = 2
        args.evolution_epochs = 2
        args.evolution_sequences = 16
        args.model_path = str(OUT_DIR / "radar_complex_nuaa_mu_quick.pt")
        args.evolution_model_path = str(
            OUT_DIR / "radar_complex_evolution_prior_quick.pt"
        )
        args.force_train = True
        args.tag = "quick" if args.tag == "2s" else args.tag
    return args


if __name__ == "__main__":
    run(parse_args())
