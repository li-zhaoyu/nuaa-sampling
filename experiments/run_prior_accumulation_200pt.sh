#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${CODE_DIR}"

# Static-scene convergence diagnostic:
# one 100 ms control tick contains a fresh 0.2 us acquisition segment
# (40 master periods x 5 samples = 200 samples).
export WINDOW_PERIODS=40
export CONTROL_DT_MS=100
export TICKS="${TICKS:-20}"
export SIGNAL_PRI_TICKS="${SIGNAL_PRI_TICKS:-${TICKS}}"
export TRIALS="${TRIALS:-8}"
export GAP_THRESHOLD_DB="${GAP_THRESHOLD_DB:-2}"
export MT_MODE="${MT_MODE:-belief_bangbang}"
# Retrain a saturated scene-prior model, then evaluate with mild EMA.
# Fixed-window coefficients prevent a sustained downward trend from Gram
# accumulation; recurrent state is kept so the prior path still adapts online.
# Layout is held static on the prior path so EDL diversity does not masquerade
# as continued learning after the prior has saturated.
export TRAIN_PRIOR="${TRAIN_PRIOR:-1}"
export PRIOR_ITERS="${PRIOR_ITERS:-600}"
export PRIOR_COEF_EMA="${PRIOR_COEF_EMA:-0.3}"
export PRIOR_CALIBRATION_WINDOWS="${PRIOR_CALIBRATION_WINDOWS:-20}"
export PRIOR_ONLINE_GAIN="${PRIOR_ONLINE_GAIN:-0.1}"
export PRIOR_MT_MODE="${PRIOR_MT_MODE:-static_hold}"
export FIXED_WINDOW_COEFFICIENTS="${FIXED_WINDOW_COEFFICIENTS:-1}"
export SKIP_NO_PRIOR="${SKIP_NO_PRIOR:-1}"
export FIGURE_OUT="${FIGURE_OUT:-${CODE_DIR}/figures/prior_accumulation_nmse_200pt.svg}"
export METRICS_OUT="${METRICS_OUT:-${CODE_DIR}/outputs/prior_accumulation_200pt_metrics.json}"

exec "${CODE_DIR}/experiments/run_prior_accumulation_comparison.sh"
