#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export RUN_ID="${RUN_ID:-branch_k2_t4_kd_consolidate_no_warmup_seed1337}"
export WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-branch-tail,kd,consolidate,no-warmup,k2,t4,single-h100-30m}"

export SEED="${SEED:-1337}"
export BRANCH_TAIL_LAYERS="${BRANCH_TAIL_LAYERS:-2}"
export TEACHER_MLP_MULT="${TEACHER_MLP_MULT:-4}"
export KD_WEIGHT_MAX="${KD_WEIGHT_MAX:-0.5}"
export PHASE_A_FRAC="${PHASE_A_FRAC:-0.0}"
export PHASE_B_FRAC="${PHASE_B_FRAC:-0.20}"
export PHASE_D_FRAC="${PHASE_D_FRAC:-0.20}"

run_experiment
