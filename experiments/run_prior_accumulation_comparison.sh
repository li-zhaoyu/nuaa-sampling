#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${CODE_DIR}"
PYTHON="${PYTHON:-${CODE_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

TICKS="${TICKS:-4}"
TRIALS="${TRIALS:-8}"
WINDOW_PERIODS="${WINDOW_PERIODS:-200}"
CONTROL_DT_MS="${CONTROL_DT_MS:-100}"
SIGNAL_PRI_TICKS="${SIGNAL_PRI_TICKS:-32}"
MT_MODE="${MT_MODE:-belief_bangbang}"
PRIOR_MT_MODE="${PRIOR_MT_MODE:-static_hold}"
MODEL_IN="${MODEL_IN:-${CODE_DIR}/outputs/streaming_wideband_nuaa_n5000_wb_tau16_pri320_t20_b16_nosp_glrt.pt}"
MODEL_IN_PRIOR="${MODEL_IN_PRIOR:-}"
TRAIN_PRIOR="${TRAIN_PRIOR:-0}"
PRIOR_ITERS="${PRIOR_ITERS:-600}"
# Mild EMA damps single-window noise while preserving saturated online learning.
PRIOR_COEF_EMA="${PRIOR_COEF_EMA:-0.3}"
PRIOR_CALIBRATION_WINDOWS="${PRIOR_CALIBRATION_WINDOWS:-20}"
PRIOR_ONLINE_GAIN="${PRIOR_ONLINE_GAIN:-0.1}"
FIXED_WINDOW_COEFFICIENTS="${FIXED_WINDOW_COEFFICIENTS:-0}"
SKIP_NO_PRIOR="${SKIP_NO_PRIOR:-0}"
FIGURE_OUT="${FIGURE_OUT:-${CODE_DIR}/figures/prior_accumulation_nmse.svg}"
METRICS_OUT="${METRICS_OUT:-${CODE_DIR}/outputs/prior_accumulation_nmse_metrics.json}"
GAP_THRESHOLD_DB="${GAP_THRESHOLD_DB:-2}"
RUN_ID="wp${WINDOW_PERIODS}_dt${CONTROL_DT_MS}_t${TICKS}"
COEF_TAG=""
if [[ "${FIXED_WINDOW_COEFFICIENTS}" == "1" ]]; then
  COEF_TAG="_fwcoef"
fi
NO_PRIOR_TAG="n5000_wb_tau16_${RUN_ID}_nosp_accum"
WITH_PRIOR_TAG="n5000_wb_tau16_${RUN_ID}_withprior_cal${PRIOR_CALIBRATION_WINDOWS}_gain${PRIOR_ONLINE_GAIN}_ema${PRIOR_COEF_EMA}${COEF_TAG}"
PRIOR_CKPT_TAG="n5000_wb_tau16_wp${WINDOW_PERIODS}_withprior_sat"
PRIOR_CKPT="${CODE_DIR}/outputs/streaming_wideband_nuaa_${PRIOR_CKPT_TAG}.pt"

if [[ ! -f "${MODEL_IN}" ]]; then
  echo "Missing checkpoint: ${MODEL_IN}" >&2
  echo "Set MODEL_IN to a trained NUAA-MU checkpoint." >&2
  exit 2
fi

COMMON=(
  --eval-snr -10
  --sir -40
  --window-periods "${WINDOW_PERIODS}"
  --ticks "${TICKS}"
  --trials "${TRIALS}"
  --seed 0
  --hold-coeffs
  --signal-pri-ticks "${SIGNAL_PRI_TICKS}"
  --pulse-width-ns 16
  --control-dt-ms "${CONTROL_DT_MS}"
  --target-nmse-db -10
  --nC 48
  --K 2
  --n-f0 24
  --n-k 8
  --f-lo-ghz 20
  --f-hi-ghz 120
  --bw-lo-ghz 10
  --bw-hi-ghz 30
  --W 32
  --period 32
  --methods mt_bangbang
  --mt-mode "${MT_MODE}"
  --cap large
)
if [[ "${FIXED_WINDOW_COEFFICIENTS}" == "1" ]]; then
  COMMON+=(--fixed-window-coefficients)
fi

cd "${CODE_DIR}"

# Optionally train a scene-prior-conditioned model so online with-prior NMSE is stable.
if [[ -n "${MODEL_IN_PRIOR}" ]]; then
  PRIOR_CKPT="${MODEL_IN_PRIOR}"
elif [[ "${TRAIN_PRIOR}" == "1" || ! -f "${PRIOR_CKPT}" ]]; then
  echo "Training scene-prior-conditioned model -> ${PRIOR_CKPT}"
  "${PYTHON}" experiments/exp_streaming_wideband_nuaa.py \
    "${COMMON[@]}" \
    --train-scene-prior \
    --train-fixed-eval \
    --jam-pos-weight 47 \
    --train-steps-list 20 30 40 40 \
    --iters "${PRIOR_ITERS}" \
    --progress-every 50 \
    --ticks 1 \
    --trials 1 \
    --model-out "${PRIOR_CKPT}" \
    --tag "${PRIOR_CKPT_TAG}" \
    2>&1 | tee "outputs/streaming_wideband_nuaa_${PRIOR_CKPT_TAG}.log"
fi

NO_PRIOR_JSON="outputs/streaming_wideband_nuaa_${NO_PRIOR_TAG}.json"
if [[ "${SKIP_NO_PRIOR}" == "1" && -f "${NO_PRIOR_JSON}" ]]; then
  echo "Reusing no-prior result: ${NO_PRIOR_JSON}"
else
  "${PYTHON}" experiments/exp_streaming_wideband_nuaa.py \
    "${COMMON[@]}" \
    --iters 0 \
    --model-in "${MODEL_IN}" \
    --no-scene-prior \
    --tag "${NO_PRIOR_TAG}" \
    2>&1 | tee "outputs/streaming_wideband_nuaa_${NO_PRIOR_TAG}.log"
fi

"${PYTHON}" experiments/exp_streaming_wideband_nuaa.py \
  "${COMMON[@]}" \
  --iters 0 \
  --model-in "${PRIOR_CKPT}" \
  --mt-mode "${PRIOR_MT_MODE}" \
  --prior-coef-ema "${PRIOR_COEF_EMA}" \
  --prior-calibration-windows "${PRIOR_CALIBRATION_WINDOWS}" \
  --prior-online-gain "${PRIOR_ONLINE_GAIN}" \
  --tag "${WITH_PRIOR_TAG}" \
  2>&1 | tee "outputs/streaming_wideband_nuaa_${WITH_PRIOR_TAG}.log"

PLOT_EXTRA=()
"${PYTHON}" experiments/plot_prior_accumulation_nmse.py \
  --with-prior "outputs/streaming_wideband_nuaa_${WITH_PRIOR_TAG}.json" \
  --no-prior "outputs/streaming_wideband_nuaa_${NO_PRIOR_TAG}.json" \
  "${PLOT_EXTRA[@]}" \
  --gap-threshold-db "${GAP_THRESHOLD_DB}" \
  --metrics-out "${METRICS_OUT}" \
  --out "${FIGURE_OUT}"

echo "RESULT_NO_PRIOR=${CODE_DIR}/outputs/streaming_wideband_nuaa_${NO_PRIOR_TAG}.json"
echo "RESULT_WITH_PRIOR=${CODE_DIR}/outputs/streaming_wideband_nuaa_${WITH_PRIOR_TAG}.json"
echo "PRIOR_CKPT=${PRIOR_CKPT}"
echo "FIGURE=${FIGURE_OUT}"
echo "METRICS=${METRICS_OUT}"
