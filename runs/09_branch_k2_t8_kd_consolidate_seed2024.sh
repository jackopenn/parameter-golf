#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export RUN_ID="${RUN_ID:-branch_k2_t8_kd_consolidate_seed2024}"
export WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-branch-tail,kd,consolidate,k2,t8,seed2024,single-h100-30m}"

export SEED="${SEED:-2024}"
export BRANCH_TAIL_LAYERS="${BRANCH_TAIL_LAYERS:-2}"
export TEACHER_MLP_MULT="${TEACHER_MLP_MULT:-8}"
export KD_WEIGHT_MAX="${KD_WEIGHT_MAX:-0.5}"
export PHASE_A_STEPS="${PHASE_A_STEPS:-400}"
export PHASE_B_STEPS="${PHASE_B_STEPS:-800}"
export PHASE_D_STEPS="${PHASE_D_STEPS:-800}"

run_experiment
