#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [OUTPUT_DIR] [additional CLI arguments...]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift
OUTPUT_DIR="${1:-${REPO_DIR}/outputs/lift_mechanism_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_DIR}"
"${PYTHON_BIN}" -m b200_experiment.cli validate-lift-mechanism \
  --config "${REPO_DIR}/configs/qwen3_b200_lift_mechanism.yaml" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}" \
  "$@"
