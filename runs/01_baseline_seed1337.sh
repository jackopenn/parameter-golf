#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export RUN_ID="${RUN_ID:-baseline_30m_h100_seed1337}"
export WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-baseline,single-h100-30m}"

export SEED="${SEED:-1337}"
export BRANCH_TAIL_LAYERS="${BRANCH_TAIL_LAYERS:-0}"

run_experiment
