#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_RUNNER="${LIFT_MECHANISM_SINGLE_RUNNER:-${SCRIPT_DIR}/validate_lift_mechanism_b200.sh}"

usage() {
  cat >&2 <<'EOF'
Usage:
  validate_lift_checkpoints_4gpu_b200.sh CHECKPOINT_ROOT \
    CHECKPOINT_1 CHECKPOINT_2 CHECKPOINT_3 CHECKPOINT_4 \
    [additional validate-lift-mechanism CLI arguments...]

Environment:
  LIFT_MECHANISM_GPUS       Four physical GPU IDs (default: 4,5,6,7)
  MECHANISM_OUTPUT_ROOT     Parent for four result directories
  LIFT_MECHANISM_SINGLE_RUNNER
                            Override the single-checkpoint launcher (tests/debug)

Example checkpoint names: checkpoint-000150 checkpoint-000300 checkpoint-000450 final
EOF
}

if [[ $# -lt 5 ]]; then
  usage
  exit 2
fi

CHECKPOINT_ROOT="$1"
shift
CHECKPOINT_NAMES=("$1" "$2" "$3" "$4")
shift 4
EXTRA_ARGS=("$@")

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root does not exist: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
CHECKPOINT_ROOT="$(cd "${CHECKPOINT_ROOT}" && pwd)"

GPU_LIST="${LIFT_MECHANISM_GPUS:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "LIFT_MECHANISM_GPUS must contain exactly four comma-separated GPU IDs" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID in LIFT_MECHANISM_GPUS: ${gpu}" >&2
    exit 2
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${MECHANISM_OUTPUT_ROOT:-${CHECKPOINT_ROOT}/lift_mechanism_4gpu_${timestamp}}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

CHECKPOINT_PATHS=()
OUTPUT_PATHS=()
LOG_PATHS=()

resolve_checkpoint() {
  local requested="$1"
  local direct="${CHECKPOINT_ROOT}/${requested}"
  if [[ -f "${direct}/config.json" ]]; then
    printf '%s\n' "${direct}"
    return 0
  fi

  local basename="${requested##*/}"
  local candidate
  local matches=()
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/config.json" ]]; then
      matches+=("${candidate}")
    fi
  done < <(
    find "${CHECKPOINT_ROOT}" \
      -mindepth 1 \
      -maxdepth 4 \
      -type d \
      -name "${basename}" \
      -print | sort
  )

  if [[ ${#matches[@]} -eq 1 ]]; then
    echo "[resolve] ${requested} -> ${matches[0]}" >&2
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Checkpoint not found under ${CHECKPOINT_ROOT}: ${requested}" >&2
    echo "Expected config.json at ${direct}/config.json or in a uniquely named nested directory." >&2
    return 1
  fi
  echo "Checkpoint name is ambiguous under ${CHECKPOINT_ROOT}: ${requested}" >&2
  printf '  %s\n' "${matches[@]}" >&2
  echo "Pass a relative path such as cmt_opd/${requested} to disambiguate." >&2
  return 1
}

for index in 0 1 2 3; do
  name="${CHECKPOINT_NAMES[$index]}"
  case "${name}" in
    ""|/*|*..*)
      echo "Checkpoint names must be non-empty relative paths without '..': ${name}" >&2
      exit 2
      ;;
  esac
  if ! checkpoint="$(resolve_checkpoint "${name}")"; then
    exit 2
  fi
  slug="${name//\//__}"
  output="${OUTPUT_ROOT}/${slug}"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing result directory: ${output}" >&2
    exit 2
  fi
  CHECKPOINT_PATHS+=("${checkpoint}")
  OUTPUT_PATHS+=("${output}")
  LOG_PATHS+=("${LOG_ROOT}/${slug}.log")
done

PIDS=()
for index in 0 1 2 3; do
  echo "[launch] checkpoint=${CHECKPOINT_NAMES[$index]} gpu=${GPUS[$index]}"
  echo "         output=${OUTPUT_PATHS[$index]}"
  echo "         log=${LOG_PATHS[$index]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" \
    bash "${SINGLE_RUNNER}" \
      "${CHECKPOINT_PATHS[$index]}" \
      "${OUTPUT_PATHS[$index]}" \
      "${EXTRA_ARGS[@]}" \
      >"${LOG_PATHS[$index]}" 2>&1 &
  PIDS+=("$!")
done

terminate_children() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

STATUSES=()
failed=0
for index in 0 1 2 3; do
  if wait "${PIDS[$index]}"; then
    status="completed"
    echo "[done] ${CHECKPOINT_NAMES[$index]} on GPU ${GPUS[$index]}"
  else
    status="failed"
    failed=1
    echo "[failed] ${CHECKPOINT_NAMES[$index]} on GPU ${GPUS[$index]}" >&2
    echo "         inspect ${LOG_PATHS[$index]}" >&2
  fi
  STATUSES+=("${status}")
done
trap - INT TERM

MANIFEST="${OUTPUT_ROOT}/runs.tsv"
{
  printf 'checkpoint_name\tgpu\tcheckpoint_path\toutput_path\tlog_path\tstatus\n'
  for index in 0 1 2 3; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${CHECKPOINT_NAMES[$index]}" \
      "${GPUS[$index]}" \
      "${CHECKPOINT_PATHS[$index]}" \
      "${OUTPUT_PATHS[$index]}" \
      "${LOG_PATHS[$index]}" \
      "${STATUSES[$index]}"
  done
} >"${MANIFEST}"

echo "Run manifest: ${MANIFEST}"
if [[ ${failed} -ne 0 ]]; then
  exit 1
fi
