#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# D-only keeps the refined CMT recipe exactly as documented: D is corrected
# globally with tanh-q99, then allocated with the same direct bounded Gibbs
# solver.  The overlay changes only S_t from g_t + corrected(D_t) to
# corrected(D_t).  Infrastructure variables remain overrideable exactly as in
# train_cmt_b200.sh.
export RUN_NAME="${RUN_NAME:-cmt_d_only_$(date +%Y%m%d_%H%M%S_%N)}"
export CMT_CORRECTION_MODE="${CMT_CORRECTION_MODE:-tanh_q99}"
export CMT_CORRECTION_QUANTILE="${CMT_CORRECTION_QUANTILE:-0.99}"
export CMT_ALLOCATION_MODE="${CMT_ALLOCATION_MODE:-direct_bounded_gibbs}"
export CMT_WEIGHT_MIN="${CMT_WEIGHT_MIN:-0.5}"
export CMT_WEIGHT_MAX="${CMT_WEIGHT_MAX:-2.0}"
export CMT_FINAL_ALLOCATION_KL="${CMT_FINAL_ALLOCATION_KL:-0.02}"
export TOKEN_SCORE_INTERVAL="${TOKEN_SCORE_INTERVAL:-1}"
export CMT_TOKEN_AUDIT_ENABLED="${CMT_TOKEN_AUDIT_ENABLED:-true}"
export CMT_TOKEN_AUDIT_INTERVAL="${CMT_TOKEN_AUDIT_INTERVAL:-150}"
export CMT_TOKEN_AUDIT_TOP_K="${CMT_TOKEN_AUDIT_TOP_K:-50}"
export CMT_TOKEN_CONTEXT_RADIUS="${CMT_TOKEN_CONTEXT_RADIUS:-32}"

echo "CMT D-only run: ${RUN_NAME}"
echo "Score pipeline: D_raw -> ${CMT_CORRECTION_MODE} -> D -> ${CMT_ALLOCATION_MODE}"
exec bash "${SCRIPT_DIR}/train_cmt_b200.sh" \
  --overlay "${REPO_DIR}/configs/qwen3_b200_cmt_d_only.yaml" \
  "$@"
