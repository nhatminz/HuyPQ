#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Keep benchmark evaluation disabled even with a supplied training config.
exec "${PYTHON:-python}" -m b200_experiment.lift_mechanism "$@" \
  --set training_evaluation.enabled=false \
  --set training_evaluation.eval_at_start=false \
  --set training_evaluation.eval_at_end=false
