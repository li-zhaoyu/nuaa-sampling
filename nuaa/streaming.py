"""Streaming NUAA controller for EDL-limited optical sampling.

This module is the system-level counterpart of the fixed ``events=L*steps``
experiments. It stores the events acquired in the current finite observation
window and lets the EDL trajectory evolve only as fast as the physical slew
rate permits between windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from .config import SystemConfig
from .strobe import Calibration, StrobeSchedule, schedule_from_layout_times, sigma_cnv2
from . import layout as L
from . import measurement as M
from . import policy as P
from . import reconstruct as R
from . import signals as S


@dataclass
class EventRecord:
    y: complex
    t_opt_ps: float
    t_sample_ps: float
    t_cnv_ps: float
    tau_ps: np.ndarray
    coset: int
    branch: int
    period: int
    pulse_slot0: int = 0
    sigma_q2: float = 0.0


class ObservationWindow:
    """Events from one finite acquisition window, ordered by arrival time."""

    def __init__(self, max_events: Optional[int] = None):
        self.max_events = max_events
        self._events: list[EventRecord] = []

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()

    def append(self, event: EventRecord) -> None:
        if self.max_events is not None and len(self._events) >= self.max_events:
            raise RuntimeError(
                f"observation window capacity exceeded: "
                f"{len(self._events) + 1} > {self.max_events}"
            )
        self._events.append(event)

    def extend(
        self,
        y: np.ndarray,
        tau_ps: np.ndarray,
        cosets: np.ndarray,
        schedule: StrobeSchedule,
        sigma_q2: Optional[np.ndarray] = None,
        pulse_slot0: int = 0,
    ) -> None:
        y = np.asarray(y).reshape(-1)
        sigma = np.zeros(y.size, dtype=np.float64) if sigma_q2 is None else np.asarray(sigma_q2).reshape(-1)
        tau = np.asarray(tau_ps, dtype=np.float64).reshape(-1)
        cosets = np.asarray(cosets, dtype=np.int64).reshape(-1)
        for i in range(y.size):
            branch = int(schedule.branch[i])
            self.append(EventRecord(
                y=complex(y[i]),
                t_opt_ps=float(schedule.t_opt_ps[i]),
                t_sample_ps=float(schedule.t_sample_ps[i]),
                t_cnv_ps=float(schedule.t_cnv_ps[i]),
                tau_ps=tau.copy(),
                coset=int(cosets[branch]),
                branch=branch,
                period=int(schedule.period[i]),
                pulse_slot0=int(pulse_slot0),
                sigma_q2=float(sigma[i]),
            ))

    def arrays(self) -> dict:
        if not self._events:
            return dict(
                y=np.zeros(0, dtype=np.complex128),
                coset=np.zeros(0, dtype=np.int64),
                t_cnv_ps=np.zeros(0, dtype=np.float64),
                branch=np.zeros(0, dtype=np.int64),
                period=np.zeros(0, dtype=np.int64),
                pulse_slot0=np.zeros(0, dtype=np.int64),
            )
        return dict(
            y=np.asarray([e.y for e in self._events], dtype=np.complex128),
            coset=np.asarray([e.coset for e in self._events], dtype=np.int64),
            t_cnv_ps=np.asarray([e.t_cnv_ps for e in self._events], dtype=np.float64),
            branch=np.asarray([e.branch for e in self._events], dtype=np.int64),
            period=np.asarray([e.period for e in self._events], dtype=np.int64),
            pulse_slot0=np.asarray([e.pulse_slot0 for e in self._events], dtype=np.int64),
        )

    def strobe_ledger(self, cfg: SystemConfig) -> dict:
        t_cnv = self.arrays()["t_cnv_ps"]
        return dict(
            n_events=int(t_cnv.size),
            t_cnv_min_ps=float(np.min(t_cnv)) if t_cnv.size else 0.0,
            t_cnv_med_ps=float(np.median(t_cnv)) if t_cnv.size else 0.0,
            strobe_feasible_rate=float(np.mean(t_cnv >= cfg.t_cnv_min_ps)) if t_cnv.size else 1.0,
        )


@dataclass
class StreamingRecon:
    support: np.ndarray
    Xhat: np.ndarray
    f1: Optional[float]
    nmse_db: Optional[float]
    n_events: int
    belief_topk_mass: Optional[float] = None
    evidence_events: int = 0
    jammer_index: Optional[int] = None
    jammer_confidence: Optional[float] = None


@dataclass
class StreamingNUAAState:
    cfg: SystemConfig
    cal: Calibration
    tau_ps: np.ndarray
    belief: P.BeliefPrior
    observation_window: ObservationWindow
    period_index: int = 0
    acc_cosets: list[np.ndarray] = field(default_factory=list)
    command_log: list[P.DelayCommand] = field(default_factory=list)
    last_recon: Optional[StreamingRecon] = None
    model_state: Optional[dict] = None
    evidence_candidates: Optional[np.ndarray] = None
    evidence_gram: Optional[np.ndarray] = None
    evidence_rhs: Optional[np.ndarray] = None
    evidence_energy: float = 0.0
    evidence_events: int = 0

    @property
    def buffer(self) -> ObservationWindow:
        """Compatibility accessor for experiments not yet migrated."""
        return self.observation_window


class StreamingNUAAController:
    """Streaming acquisition, reconstruction, and slow EDL control."""

    def __init__(
        self,
        cfg: SystemConfig,
        cal: Optional[Calibration] = None,
        K: int = 3,
        beta: float = 2.0,
        top_m: int = 40,
        conf_thr: float = 0.02,
        window_events: int = 4096,
        buffer_events: Optional[int] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.cal = (cal or Calibration()).ensure(cfg.L, cfg)
        self.K = int(K)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        tau0 = L.cosets_to_tau(L.gen_fixed_random(cfg, self.rng), cfg)
        self.planner = P.BangBangDelayPlanner(
            cfg, self.cal, top_m=top_m, conf_thr=conf_thr, rng=self.rng)
        if buffer_events is not None:
            window_events = int(buffer_events)
        self.state = StreamingNUAAState(
            cfg=cfg,
            cal=self.cal,
            tau_ps=tau0,
            belief=P.BeliefPrior(cfg.N0),
            observation_window=ObservationWindow(window_events),
        )

    def current_cosets(self) -> np.ndarray:
        return L.tau_to_cosets(self.state.tau_ps, self.cfg)

    def strobe_schedule_for_current_period(self) -> StrobeSchedule:
        cosets = self.current_cosets()
        period = self.state.period_index
        t_opt = period * self.cfg.T0_ps + cosets.astype(np.float64) * self.cfg.eta_ps
        branch = np.arange(self.cfg.L, dtype=np.int64)
        periods = np.full(self.cfg.L, period, dtype=np.int64)
        return schedule_from_layout_times(t_opt, branch, periods, self.cfg, self.cal)

    def begin_observation_window(self) -> None:
        """Discard the preceding window while retaining cross-window state."""
        self.state.observation_window.clear()

    def append_period_measurement(self, y: np.ndarray, pulse_slot0: int = 0) -> StrobeSchedule:
        schedule = self.strobe_schedule_for_current_period()
        sigma = sigma_cnv2(schedule.t_cnv_ps, self.cfg, self.cal)
        cosets = self.current_cosets()
        self.state.observation_window.extend(
            y,
            self.state.tau_ps,
            cosets,
            schedule,
            sigma_q2=sigma,
            pulse_slot0=pulse_slot0,
        )
        self.state.acc_cosets.append(cosets.copy())
        self.state.period_index += 1
        return schedule

    def simulate_period_from_spectrum(
        self,
        X: np.ndarray,
        support: Optional[np.ndarray] = None,
        snr_db: float = 10.0,
    ) -> StrobeSchedule:
        cosets = self.current_cosets()
        A = M.build_A_spec(cosets, self.cfg.N0)
        Y_clean = A @ X
        if Y_clean.ndim == 2:
            y = Y_clean[:, 0]
        else:
            y = Y_clean.reshape(-1)
        if support is not None and len(support) > 0:
            ref = float(np.mean(np.abs(A[:, support] @ X[support].reshape(len(support), -1)) ** 2)) + 1e-30
        else:
            ref = float(np.mean(np.abs(y) ** 2)) + 1e-30
        y = S.add_measurement_noise(y, snr_db, ref, self.rng)
        return self.append_period_measurement(y)

    def _evidence(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        corr = np.sum(np.abs(An.conj().T @ Y.reshape(A.shape[0], -1)), axis=1)
        return corr

    def reconstruct(self, truth_X: Optional[np.ndarray] = None,
                    truth_support: Optional[np.ndarray] = None) -> StreamingRecon:
        arr = self.state.observation_window.arrays()
        y = arr["y"]
        cosets = arr["coset"]
        if y.size < self.K:
            recon = StreamingRecon(np.zeros(0, dtype=int), np.zeros((self.cfg.N0, 1), dtype=np.complex128),
                                   None, None, int(y.size))
            self.state.last_recon = recon
            return recon
        A = M.build_A_spec(cosets, self.cfg.N0)
        Y = y.reshape(-1, 1)
        self.state.belief.decay()
        self.state.belief.update(self._evidence(A, Y))
        support, Xhat = R.somp_prior(A, Y, self.K, prior=self.state.belief.pi, beta=self.beta)
        f1 = nmse_db = None
        if truth_support is not None:
            inter = len(set(map(int, support)) & set(map(int, truth_support)))
            prec = inter / max(1, len(support))
            rec = inter / max(1, len(truth_support))
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if truth_X is not None and truth_support is not None:
            denom = np.sum(np.abs(truth_X[truth_support]) ** 2) + 1e-12
            err = np.sum(np.abs(Xhat[truth_support] - truth_X[truth_support].reshape(-1, Xhat.shape[1])) ** 2)
            nmse_db = float(10 * np.log10(err / denom + 1e-12))
        recon = StreamingRecon(support=support, Xhat=Xhat, f1=f1, nmse_db=nmse_db, n_events=int(y.size))
        self.state.last_recon = recon
        return recon

    def reconstruct_with_model(
        self,
        model,
        tok: np.ndarray,
        dt_norm: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        candidates: np.ndarray,
        cand_feat: Optional[np.ndarray] = None,
        prior_override: Optional[np.ndarray] = None,
        allow_prior_lock: bool = True,
        accumulate_coefficients: bool = True,
        detect_jammer: bool = True,
        clean_ridge_rel: float = 0.0,
        use_burst: bool = False,
        truth_X: Optional[np.ndarray] = None,
        truth_support: Optional[np.ndarray] = None,
    ) -> StreamingRecon:
        """Run NUAA-MU on the current observation window and update belief.

        prior_override: optional per-candidate prior merged with the model useful head.
        allow_prior_lock: if True and the merged prior peaks ≥0.9, hard-lock support + LS
            (oracle scene-prior path). Soft streaming belief should set this False.
        accumulate_coefficients: if False, cumulative evidence is used only to identify
            prior-free support; coefficients are estimated from the current fixed-budget
            observation window. A fully injected scene prior then sits at a saturated
            noise floor with mild tick-to-tick fluctuation rather than a downward trend.
        detect_jammer: disable for clean scenarios so prior-locked LS does not project
            out an arbitrary non-support column as a nonexistent jammer.
        clean_ridge_rel: relative Tikhonov regularization for clean, prior-locked
            cumulative inversion; zero recovers ordinary least squares.
        """
        model.eval()
        tok_t = torch.tensor(tok[None, ...], dtype=torch.float32)
        dt_t = torch.tensor(dt_norm[None, ...], dtype=torch.float32)
        A_t = torch.tensor(A[None, ...], dtype=torch.complex64)
        Y_t = torch.tensor(Y[None, ...], dtype=torch.complex64)
        cand_t = None if cand_feat is None else torch.tensor(cand_feat[None, ...], dtype=torch.float32)
        with torch.no_grad():
            Xhat_t, aux, new_state = model.forward_chunk(
                tok_t, dt_t, A_t, Y_t, state=self.state.model_state,
                refine=True, return_aux=True, cand_feat=cand_t, use_burst=use_burst)
        self.state.model_state = new_state
        Xc_nn = Xhat_t[0].detach().cpu().numpy()
        cand = np.asarray(candidates, dtype=np.int64)
        prior = aux["useful_prior"][0].detach().cpu().numpy()
        if prior_override is not None:
            po = np.asarray(prior_override, dtype=np.float64).reshape(-1)
            if po.size == prior.size:
                prior = np.maximum(prior, po.astype(prior.dtype))
        event_w = aux["event_weight"][0].detach().cpu().numpy()
        y_base = np.asarray(Y, dtype=np.complex64).copy()
        if use_burst:
            bprob = torch.sigmoid(aux["burst_logits"][0]).detach().cpu().numpy()
            bcomp = aux["burst_complex"][0].detach().cpu().numpy()
            y_base = y_base - bprob.astype(np.complex64) * bcomp
        sqrt_w = np.sqrt(np.clip(event_w, 0.02, None)).astype(np.float32)
        A_w = np.asarray(A, dtype=np.complex64) * sqrt_w[:, None]
        Y_w = y_base * sqrt_w
        jammer_index = None
        jammer_confidence = None
        try:
            if allow_prior_lock and prior_override is not None and np.max(prior) >= 0.9:
                local_support = np.argsort(prior)[-self.K:]
                if detect_jammer:
                    A_det = np.asarray(A) if not accumulate_coefficients else A_w
                    Y_det = np.asarray(Y) if not accumulate_coefficients else Y_w
                    An = A_det / (np.linalg.norm(A_det, axis=0, keepdims=True) + 1e-12)
                    corr = np.abs(An.conj().T @ Y_det.reshape(-1, 1)).reshape(-1)
                    corr[local_support] = -np.inf
                    jam = int(np.argmax(corr))
                    jammer_index = jam
                    if accumulate_coefficients:
                        n_new = len(Y)
                        self._accumulate_candidate_evidence(
                            cand, np.asarray(A[-n_new:]), np.asarray(Y[-n_new:]))
                        Xc = self._online_known_support_ls(local_support, jam)
                    else:
                        Xc = self._window_known_support_ls(
                            A, Y, local_support, jam)
                else:
                    if accumulate_coefficients:
                        self._accumulate_candidate_evidence(
                            cand, np.asarray(A), np.asarray(Y))
                        G = self.state.evidence_gram
                        b = self.state.evidence_rhs
                        if G is None or b is None:
                            raise RuntimeError("candidate evidence is not initialized")
                        Gs = G[np.ix_(local_support, local_support)]
                        diag_scale = max(
                            float(np.mean(np.maximum(Gs.diagonal().real, 0.0))),
                            1e-12,
                        )
                        ridge = max(0.0, float(clean_ridge_rel)) * diag_scale
                        coef = np.linalg.solve(
                            Gs + ridge * np.eye(len(local_support)),
                            b[local_support],
                        )
                    else:
                        A_support = np.asarray(A)[:, local_support]
                        y_window = np.asarray(Y).reshape(-1)
                        Gs = A_support.conj().T @ A_support
                        bs = A_support.conj().T @ y_window
                        diag_scale = max(
                            float(np.mean(np.maximum(Gs.diagonal().real, 0.0))),
                            1e-12,
                        )
                        ridge = max(0.0, float(clean_ridge_rel)) * diag_scale
                        coef = np.linalg.solve(
                            Gs + ridge * np.eye(len(local_support)),
                            bs,
                        )
                    Xc = np.zeros(len(cand), dtype=np.complex128)
                    Xc[local_support] = coef
            else:
                # Model-learned prior path (no explicit scene prior lock).
                # Support comes from the pretrained useful_prior head; coefficients
                # optionally accumulate across windows via Tikhonov LS.
                A_eff, Y_eff = A_w, Y_w[:, None]
                jam_logits = aux.get("jammer_logits") if isinstance(aux, dict) else None
                jam = None
                if jam_logits is not None and detect_jammer:
                    jam_prob = torch.sigmoid(jam_logits[0]).detach().cpu().numpy().reshape(-1)
                    model_jam = int(np.argmax(jam_prob))
                    jam = model_jam
                    jammer_confidence = float(jam_prob[model_jam])
                if not allow_prior_lock and self.K == 2 and detect_jammer:
                    n_new = len(Y)
                    self._accumulate_candidate_evidence(
                        cand, np.asarray(A[-n_new:]), np.asarray(Y[-n_new:]))
                    # At SIR=-40 dB the blocker is directly identifiable as the
                    # strongest cumulative fitted atom. This remains prior-free
                    # and is more reliable than a single-window neural top-1.
                    evidence_jam, evidence_conf = self._strongest_candidate()
                    jam = evidence_jam
                    jammer_index = evidence_jam
                    jammer_confidence = max(float(jammer_confidence or 0.0), evidence_conf)
                    A_eff, Y_eff = R.null_jammer(A_w, Y_w[:, None], [jam])
                    local_support, Xc, posterior = self._online_pair_glrt(jam)
                    self._set_belief_from_candidate_posterior(cand, posterior)
                    if not accumulate_coefficients:
                        Xc = self._window_known_support_ls(
                            A, Y, local_support, jam)
                elif not allow_prior_lock:
                    # General-K pretrained path: learned useful_prior selects support.
                    # When the head is calibrated (peaky mass on few candidates), hard
                    # top-K + ridge-LS matches training; otherwise fall back to
                    # prior-weighted SOMP with a sharpened useful prior.
                    pri = np.asarray(prior, dtype=np.float64).reshape(-1)
                    pri = np.clip(pri, 0.0, None)
                    pri_max = float(np.max(pri)) + 1e-12
                    pri_n = pri / pri_max
                    # Prefer hard top-K from the learned useful head (calibrated under
                    # diverse M=200 training). Fall back to prior-weighted SOMP only
                    # when the head is essentially flat.
                    top_idx = np.argsort(pri_n)[-self.K:]
                    top_mass = float(pri_n[top_idx].sum() / (pri_n.sum() + 1e-12))
                    flat = bool(top_mass < 0.15 or pri_max < 0.05)
                    if not flat:
                        local_support = top_idx
                        Xc_ls = None
                    else:
                        pri_s = np.power(pri_n, 3.0)
                        local_support, Xc_ls = R.somp_prior(
                            A_eff, Y_eff, self.K, prior=pri_s,
                            beta=max(2.0, self.beta))
                    if accumulate_coefficients:
                        self._accumulate_candidate_evidence(
                            cand, np.asarray(A), np.asarray(Y))
                        G = self.state.evidence_gram
                        b = self.state.evidence_rhs
                        if G is None or b is None:
                            raise RuntimeError("candidate evidence is not initialized")
                        Gs = G[np.ix_(local_support, local_support)]
                        diag_scale = max(
                            float(np.mean(np.maximum(Gs.diagonal().real, 0.0))),
                            1e-12,
                        )
                        ridge = max(0.0, float(clean_ridge_rel)) * diag_scale
                        coef = np.linalg.solve(
                            Gs + ridge * np.eye(len(local_support)),
                            b[local_support],
                        )
                        Xc = np.zeros(len(cand), dtype=np.complex128)
                        Xc[local_support] = coef
                    else:
                        if Xc_ls is not None:
                            Xc = Xc_ls[:, 0]
                        else:
                            A_s = np.asarray(A_eff)[:, local_support]
                            y_s = np.asarray(Y_eff).reshape(-1)
                            Xc = np.zeros(len(cand), dtype=np.complex128)
                            Xc[local_support] = np.linalg.pinv(A_s) @ y_s
                    full_ev = np.full(self.cfg.N0, 1e-6, dtype=np.float64)
                    full_ev[cand] = np.maximum(pri_n, 1e-6)
                    self.state.belief.update(full_ev)
                else:
                    local_support, Xc_ls = R.somp_prior(
                        A_eff, Y_eff, self.K, prior=prior, beta=max(2.0, self.beta))
                    Xc = Xc_ls[:, 0]
        except np.linalg.LinAlgError:
            local_support = np.argsort(np.abs(Xc_nn))[-self.K:]
            Xc = Xc_nn
        full = np.zeros((self.cfg.N0, 1), dtype=np.complex128)
        full[cand, 0] = Xc
        if allow_prior_lock:
            # 场景先验路径仅用于布局状态；自主路径的 belief 已由独立新增证据更新，
            # 禁止将旧 belief 再乘回自身造成伪置信累积。
            full_ev = np.full(self.cfg.N0, 1e-6, dtype=np.float64)
            full_ev[cand] = np.maximum(prior.astype(np.float64), 1e-6)
            self.state.belief.update(full_ev)
        est_idx = np.asarray(local_support, dtype=np.int64)
        support = cand[est_idx]
        f1 = nmse_db = None
        if truth_support is not None:
            truth = set(map(int, truth_support))
            inter = len(set(map(int, support)) & truth)
            prec = inter / max(1, len(support))
            rec = inter / max(1, len(truth))
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if truth_X is not None and truth_support is not None:
            denom = np.sum(np.abs(truth_X[truth_support]) ** 2) + 1e-12
            err = np.sum(np.abs(full[truth_support, 0] - truth_X[truth_support].reshape(-1)) ** 2)
            nmse_db = float(10 * np.log10(err / denom + 1e-12))
        recon = StreamingRecon(
            support=support,
            Xhat=full,
            f1=f1,
            nmse_db=nmse_db,
            n_events=int(Y.size),
            belief_topk_mass=self.state.belief.topm_mass(self.K),
            evidence_events=self.state.evidence_events,
            jammer_index=jammer_index,
            jammer_confidence=jammer_confidence,
        )
        self.state.last_recon = recon
        return recon

    def _reset_candidate_evidence(self, candidates: np.ndarray) -> None:
        n_cand = int(len(candidates))
        self.state.evidence_candidates = np.asarray(candidates, dtype=np.int64).copy()
        self.state.evidence_gram = np.zeros((n_cand, n_cand), dtype=np.complex128)
        self.state.evidence_rhs = np.zeros(n_cand, dtype=np.complex128)
        self.state.evidence_energy = 0.0
        self.state.evidence_events = 0

    def _accumulate_candidate_evidence(
        self,
        candidates: np.ndarray,
        A_new: np.ndarray,
        y_new: np.ndarray,
    ) -> None:
        """Accumulate each new event exactly once via sufficient statistics."""
        cand = np.asarray(candidates, dtype=np.int64)
        if (
            self.state.evidence_candidates is None
            or not np.array_equal(cand, self.state.evidence_candidates)
        ):
            self._reset_candidate_evidence(cand)
        A64 = np.asarray(A_new, dtype=np.complex128)
        y64 = np.asarray(y_new, dtype=np.complex128).reshape(-1)
        self.state.evidence_gram += A64.conj().T @ A64
        self.state.evidence_rhs += A64.conj().T @ y64
        self.state.evidence_energy += float(np.vdot(y64, y64).real)
        self.state.evidence_events += int(y64.size)

    def _online_pair_glrt(
        self,
        jammer: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exhaustive K=2 support GLRT with the jammer fitted as a nuisance atom.

        The accumulated Gram matrix and right-hand side summarize all windows, so
        support confidence and coefficient variance improve without expanding
        the current inversion window.
        """
        G = self.state.evidence_gram
        b = self.state.evidence_rhs
        yy = max(self.state.evidence_energy, 1e-18)
        if G is None or b is None:
            raise RuntimeError("candidate evidence is not initialized")
        n_cand = int(b.size)
        jammer = int(jammer)
        choices = [i for i in range(n_cand) if i != jammer]
        diag_scale = max(float(np.mean(np.maximum(G.diagonal().real, 0.0))), 1e-12)
        ridge = 1e-7 * diag_scale

        gjj = max(float(G[jammer, jammer].real), ridge)
        explained_jam = float((abs(b[jammer]) ** 2) / gjj)
        residual_after_jam = max(yy - explained_jam, 1e-15)
        # Stable Schur-complement projection of the nuisance jammer. This avoids
        # subtracting two O(10^4) explained-energy terms to recover an O(1)
        # useful-signal increment at SIR=-40 dB.
        g_to_jam = G[:, jammer]
        G_res = G - np.outer(g_to_jam, G[jammer, :]) / gjj
        b_res = b - g_to_jam * b[jammer] / gjj

        pairs: list[tuple[int, int]] = []
        scores: list[float] = []
        coefs: list[np.ndarray] = []
        for ai, i in enumerate(choices[:-1]):
            for j in choices[ai + 1:]:
                idx = np.asarray([i, j], dtype=np.int64)
                Gs = G_res[np.ix_(idx, idx)] + ridge * np.eye(2)
                bs = b_res[idx]
                coef = np.linalg.solve(Gs, bs)
                explained = float(np.vdot(bs, coef).real)
                score = max(explained, 0.0) / residual_after_jam
                pairs.append((i, j))
                scores.append(score)
                coefs.append(coef)

        score_arr = np.asarray(scores, dtype=np.float64)
        best = int(np.argmax(score_arr))
        best_pair = np.asarray(pairs[best], dtype=np.int64)
        best_coef = coefs[best]
        Xc = np.zeros(n_cand, dtype=np.complex128)
        Xc[best_pair] = best_coef

        # Marginalize pair likelihoods into a candidate posterior. Temperature
        # decreases with independent tick count as evidence accumulates.
        ticks = max(1.0, self.state.evidence_events / max(1, self.cfg.L))
        temperature = max(0.004, 0.05 / np.sqrt(ticks))
        logits = (score_arr - score_arr.max()) / temperature
        pair_prob = np.exp(np.clip(logits, -60.0, 0.0))
        pair_prob /= pair_prob.sum() + 1e-15
        posterior = np.full(n_cand, 1e-8, dtype=np.float64)
        for prob, (i, j) in zip(pair_prob, pairs):
            posterior[i] += float(prob)
            posterior[j] += float(prob)
        posterior[jammer] = 1e-8
        posterior /= posterior.sum() + 1e-15
        return best_pair, Xc, posterior

    def _strongest_candidate(self) -> tuple[int, float]:
        """Return the strongest cumulative single-atom fit and its dominance."""
        G = self.state.evidence_gram
        b = self.state.evidence_rhs
        if G is None or b is None:
            raise RuntimeError("candidate evidence is not initialized")
        diag = np.maximum(np.asarray(G.diagonal().real), 1e-15)
        power = np.abs(np.asarray(b)) ** 2 / diag
        order = np.argsort(power)
        best = int(order[-1])
        second = float(power[order[-2]]) if len(order) > 1 else 0.0
        confidence = float(power[best] / (power[best] + second + 1e-15))
        return best, confidence

    def _online_known_support_ls(
        self,
        support: np.ndarray,
        jammer: int,
    ) -> np.ndarray:
        """Cumulative joint LS when a scene prior supplies the useful support."""
        G = self.state.evidence_gram
        b = self.state.evidence_rhs
        if G is None or b is None:
            raise RuntimeError("candidate evidence is not initialized")
        support = np.asarray(support, dtype=np.int64)
        jammer = int(jammer)
        diag_scale = max(float(np.mean(np.maximum(G.diagonal().real, 0.0))), 1e-12)
        ridge = 1e-7 * diag_scale
        gjj = max(float(G[jammer, jammer].real), ridge)
        g_to_jam = G[:, jammer]
        G_res = G - np.outer(g_to_jam, G[jammer, :]) / gjj
        b_res = b - g_to_jam * b[jammer] / gjj
        Gs = G_res[np.ix_(support, support)] + ridge * np.eye(len(support))
        coef = np.linalg.solve(Gs, b_res[support])
        Xc = np.zeros(len(b), dtype=np.complex128)
        Xc[support] = coef
        return Xc

    @staticmethod
    def _window_known_support_ls(
        A: np.ndarray,
        Y: np.ndarray,
        support: np.ndarray,
        jammer: int,
    ) -> np.ndarray:
        """Current-window joint LS with the blocker projected out.

        Cumulative sufficient statistics may determine ``support`` in the
        prior-free path, but they are deliberately excluded from coefficient
        estimation so both prior conditions use the same per-tick sample budget.
        """
        A64 = np.asarray(A, dtype=np.complex128)
        y64 = np.asarray(Y, dtype=np.complex128).reshape(-1)
        support = np.asarray(support, dtype=np.int64)
        jammer = int(jammer)
        G = A64.conj().T @ A64
        b = A64.conj().T @ y64
        diag_scale = max(float(np.mean(np.maximum(G.diagonal().real, 0.0))), 1e-12)
        ridge = 1e-7 * diag_scale
        gjj = max(float(G[jammer, jammer].real), ridge)
        g_to_jam = G[:, jammer]
        G_res = G - np.outer(g_to_jam, G[jammer, :]) / gjj
        b_res = b - g_to_jam * b[jammer] / gjj
        Gs = G_res[np.ix_(support, support)] + ridge * np.eye(len(support))
        coef = np.linalg.solve(Gs, b_res[support])
        Xc = np.zeros(A64.shape[1], dtype=np.complex128)
        Xc[support] = coef
        return Xc

    def _set_belief_from_candidate_posterior(
        self,
        candidates: np.ndarray,
        posterior: np.ndarray,
    ) -> None:
        """Map cumulative-evidence posterior to the full-grid layout belief."""
        # Keep a finite exploration floor without assigning O(1) total mass to
        # thousands of out-of-candidate bins.
        full = np.full(
            self.cfg.N0,
            self.state.belief.floor / max(1, self.cfg.N0),
            dtype=np.float64,
        )
        full[np.asarray(candidates, dtype=np.int64)] += np.asarray(posterior, dtype=np.float64)
        self.state.belief.pi = full / (full.sum() + 1e-15)

    def plan_next_period(self, dt_s: Optional[float] = None,
                         mode: str = "belief_bangbang") -> P.DelayCommand:
        if dt_s is None:
            dt_s = self.cfg.T0_ps * 1e-12
        cmd = self.planner.step(
            self.state.tau_ps, dt_s, self.state.acc_cosets,
            self.state.belief, mode=mode)
        self.state.tau_ps = cmd.tau_ps
        self.state.command_log.append(cmd)
        return cmd

    def acquisition_ledger(self) -> dict:
        movement = [cmd.movement_ps for cmd in self.state.command_log]
        strobe = self.state.observation_window.strobe_ledger(self.cfg)
        return dict(
            n_periods=int(self.state.period_index),
            n_events=len(self.state.observation_window),
            T_acq_s=float(self.state.period_index * self.cfg.T0_ps * 1e-12),
            tau_movement_ps=float(np.sum(movement)) if movement else 0.0,
            adaptive_rate=float(np.mean([cmd.adaptive_enabled for cmd in self.state.command_log]))
            if self.state.command_log else 0.0,
            strobe_t_cnv_min_ps=strobe["t_cnv_min_ps"],
            strobe_t_cnv_med_ps=strobe["t_cnv_med_ps"],
            strobe_feasible_rate=strobe["strobe_feasible_rate"],
        )
