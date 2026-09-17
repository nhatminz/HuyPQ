#!/usr/bin/env bash
set -euo pipefail

# Compare runs whose artifacts live in different repository trees.  The
# plotting Python code still runs from Bellman2, while each method's history
# and metrics are read from the explicitly constructed absolute paths below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINHPN19_OUTPUT_ROOT="${MINHPN19_OUTPUT_ROOT:-/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs}"
HUYPQ_OUTPUT_ROOT="${HUYPQ_OUTPUT_ROOT:-/workspace/storage-shared/nlp/huypq51/projects/HuyPQ-main/outputs}"
HUYPQ_RESULTS_ROOT="${HUYPQ_RESULTS_ROOT:-/workspace/storage-shared/nlp/huypq51/projects/HuyPQ-main/results}"
PLOT_MODE="${PLOT_MODE:-progress}"
PLOT_METHODS="${PLOT_METHODS:-opd ta cmt snig}"

case "${PLOT_MODE,,}" in
  progress|training|training-progress) PLOT_MODE=progress ;;
  final|aggregate|results) PLOT_MODE=final ;;
  *)
    echo "PLOT_MODE must be progress or final, got: ${PLOT_MODE}" >&2
    exit 2
    ;;
esac

RAW_METHODS="${PLOT_METHODS//,/ }"
read -r -a METHODS <<< "${RAW_METHODS}"
if (( ${#METHODS[@]} == 0 )); then
  echo "PLOT_METHODS must contain at least one method" >&2
  exit 2
fi

RUN_NAMES=()
require_run_name() {
  local method="$1" variable="$2"
  if [[ -z "${!variable:-}" ]]; then
    echo "${variable} is required when plotting ${method}" >&2
    exit 2
  fi
  RUN_NAMES+=("${!variable}")
}

for method in "${METHODS[@]}"; do
  case "${method,,}" in
    opd|pure-opd)
      require_run_name opd OPD_RUN_NAME
      export OPD_OUTPUT_DIR="${OPD_OUTPUT_DIR:-${MINHPN19_OUTPUT_ROOT}/${OPD_RUN_NAME}/opd}"
      ;;
    ta|ta-opd)
      require_run_name ta TA_RUN_NAME
      export TA_OUTPUT_DIR="${TA_OUTPUT_DIR:-${MINHPN19_OUTPUT_ROOT}/${TA_RUN_NAME}/ta_opd}"
      ;;
    cmt|cmt-opd)
      require_run_name cmt CMT_RUN_NAME
      export CMT_OUTPUT_DIR="${CMT_OUTPUT_DIR:-${MINHPN19_OUTPUT_ROOT}/${CMT_RUN_NAME}/cmt_opd}"
      ;;
    snig|snig-opd)
      require_run_name snig SNIG_RUN_NAME
      export SNIG_OUTPUT_DIR="${SNIG_OUTPUT_DIR:-${HUYPQ_OUTPUT_ROOT}/${SNIG_RUN_NAME}/snig_opd}"
      ;;
    *)
      echo "Unsupported cross-repository plot method: ${method}" >&2
      echo "Supported methods: opd ta cmt snig" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RESULTS_DIR:-}" ]]; then
  comparison="${RUN_NAMES[0]}"
  for run_name in "${RUN_NAMES[@]:1}"; do
    comparison+="_vs_${run_name}"
  done
  export RESULTS_DIR="${HUYPQ_RESULTS_ROOT}/${comparison}"
fi

echo "Cross-repository plotting"
echo "  methods: ${PLOT_METHODS}"
echo "  OPD:    ${OPD_OUTPUT_DIR:-<not selected>}"
echo "  TA:     ${TA_OUTPUT_DIR:-<not selected>}"
echo "  CMT:    ${CMT_OUTPUT_DIR:-<not selected>}"
echo "  SNIG:   ${SNIG_OUTPUT_DIR:-<not selected>}"
echo "  results: ${RESULTS_DIR}"

if [[ "${PLOT_MODE}" == progress ]]; then
  exec bash "${SCRIPT_DIR}/plot_training_progress.sh" "$@"
fi
exec bash "${SCRIPT_DIR}/plot_results.sh" "$@"
