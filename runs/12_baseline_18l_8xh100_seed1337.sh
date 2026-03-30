#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-baseline_18l_10m_8xh100_seed1337}"
export WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-baseline,18l,8xh100-10m}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-8xh100-10m}"

export SEED="${SEED:-1337}"
export SINGLE_H100_30M="${SINGLE_H100_30M:-0}"
export BRANCH_TAIL_LAYERS="${BRANCH_TAIL_LAYERS:-0}"
export NUM_LAYERS="${NUM_LAYERS:-18}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-0}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600.0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"

source "$(dirname "$0")/_common.sh"
run_experiment_torchrun "$@"
