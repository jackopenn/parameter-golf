#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-baseline_30m_1xh100_seed1337}"
export WANDB_NAME="${WANDB_NAME:-baseline_30m_1xh100_seed1337}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-matformer-two-scale-1xh100-30m}"
export SEED="${SEED:-1337}"
export SINGLE_H100_30M="${SINGLE_H100_30M:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800.0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-450}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_baseline "$@"
