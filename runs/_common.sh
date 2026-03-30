#!/usr/bin/env bash

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNS_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-parameter-golf}"
export WANDB_ENTITY="${WANDB_ENTITY:-jackpenn}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-single-h100-30m}"

export SINGLE_H100_30M="${SINGLE_H100_30M:-1}"
export USE_COMPILE="${USE_COMPILE:-1}"

export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
export KD_WEIGHT_MAX="${KD_WEIGHT_MAX:-0.5}"
export PHASE_A_STEPS="${PHASE_A_STEPS:-400}"
export PHASE_B_STEPS="${PHASE_B_STEPS:-800}"
export PHASE_D_STEPS="${PHASE_D_STEPS:-800}"

run_experiment() {
  python3 train_gpt_branch_tail.py "$@"
}
