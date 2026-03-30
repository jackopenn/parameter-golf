#!/usr/bin/env bash

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNS_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-parameter-golf}"
export WANDB_ENTITY="${WANDB_ENTITY:-jackpenn}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-matformer-two-scale-8xh100}"

export SINGLE_H100_30M="${SINGLE_H100_30M:-0}"
export USE_COMPILE="${USE_COMPILE:-1}"

export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-0}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600.0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"

run_baseline() {
  torchrun \
    --standalone \
    --nnodes="${NNODES:-1}" \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    train_gpt.py "$@"
}

run_matformer() {
  torchrun \
    --standalone \
    --nnodes="${NNODES:-1}" \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    train_gpt_matformer.py "$@"
}
