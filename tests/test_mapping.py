"""§3 测量矩阵 ⇄ 光脉冲布局掩码 映射的单元测试。

可直接运行：  python tests/test_mapping.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuaa.config import cfg_sanity, cfg_main
from nuaa import layout as L
from nuaa import measurement as M
from nuaa import signals as S
from nuaa import streaming as St
from nuaa.strobe import Calibration


def test_distinct_slots():
    for cfg in (cfg_sanity(), cfg_main()):
        rng = np.random.default_rng(0)
        for _ in range(50):
            cosets = L.gen_fixed_random(cfg, rng)
            assert L.validate_layout(cosets, cfg)


def test_phi_time_consistency():
    for cfg in (cfg_sanity(), cfg_main()):
        rng = np.random.default_rng(1)
        cosets = L.gen_fixed_random(cfg, rng)
        err = M.check_phi_time_consistency(cosets, cfg.N0, rng=rng)
        assert err < 1e-7, f"一致性误差过大: {err:.2e}"


def test_roundtrip_tau_cosets():
    cfg = cfg_main()
    rng = np.random.default_rng(2)
    for _ in range(50):
        inc = rng.integers(1, 200, size=cfg.L)
        inc = L.enforce_distinct_slots(inc, cfg)
        tau = L.increments_to_tau(inc, cfg)
        cosets = L.tau_to_cosets(tau, cfg)
        tau2 = L.cosets_to_tau(cosets, cfg)
        assert np.allclose(tau, tau2, atol=cfg.eta_ps + 1e-9)


def test_matrix_to_delays_projection():
    cfg = cfg_main()
    rng = np.random.default_rng(3)
    for _ in range(50):
        desired = L.gen_poisson_gap(cfg, rng)
        tau = M.matrix_to_delays(desired, cfg)
        cosets = L.tau_to_cosets(tau, cfg)
        assert L.validate_layout(cosets, cfg)


def test_bang_bang_projection():
    cfg = cfg_main()
    rng = np.random.default_rng(4)
    prev = L.cosets_to_tau(L.gen_fixed_random(cfg, rng), cfg)
    desired = L.gen_poisson_gap(cfg, rng)

    tau_frozen = M.matrix_to_delays(desired, cfg, prev_tau_ps=prev, dt_s=1e-6)
    assert np.allclose(tau_frozen, prev), "µs control slices must not jump a full ps"

    dt_s = 1e-3
    max_step = cfg.v_ramp_ps_per_s * dt_s
    tau = M.matrix_to_delays(desired, cfg, prev_tau_ps=prev, dt_s=dt_s)
    steps = np.abs(tau - prev)
    assert np.all((steps < 1e-6) | (np.abs(steps - max_step) < cfg.eta_ps + 1e-6))

    tau_reached = M.matrix_to_delays(desired, cfg, prev_tau_ps=prev, dt_s=1.0)
    assert np.allclose(tau_reached, L.cosets_to_tau(desired, cfg), atol=cfg.eta_ps + 1e-9)


def test_event_times_count_and_order():
    cfg = cfg_sanity()
    rng = np.random.default_rng(5)
    cosets = L.gen_fixed_random(cfg, rng)
    P = 16
    times, branch, period, coset = L.event_times(cosets, P, cfg)
    assert times.size == cfg.L * P
    assert np.all(np.diff(times) >= 0)


def test_strobe_layout():
    cfg = cfg_sanity()
    cal = Calibration().ensure(cfg.L, cfg)
    tau = L.cosets_to_tau(L.gen_fixed_random(cfg, np.random.default_rng(6)), cfg)
    lay = L.delays_to_layout(tau, n_periods=4, cfg=cfg, cal=cal)
    assert lay.t_sample is not None and lay.t_cnv is not None
    assert lay.t_sample.size == cfg.L * 4
    assert np.all(np.diff(lay.t_sample) >= 0)


def test_streaming_controller_smoke():
    cfg = cfg_sanity()
    rng = np.random.default_rng(7)
    sig = S.gen_multiband_spectrum(cfg, K=2, J=1, rng=rng, jammer=False)
    ctrl = St.StreamingNUAAController(cfg, K=2, buffer_events=64, seed=7)
    ctrl.simulate_period_from_spectrum(sig.X, support=sig.support, snr_db=20.0)
    rec = ctrl.reconstruct(truth_X=sig.X, truth_support=sig.support)
    cmd = ctrl.plan_next_period(dt_s=5e-3, mode="random_slow_scan")
    ledger = ctrl.acquisition_ledger()
    assert rec.n_events == cfg.L
    assert cmd.tau_ps.shape == (cfg.L,)
    assert ledger["n_events"] == cfg.L
    assert ledger["strobe_feasible_rate"] >= 0.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
