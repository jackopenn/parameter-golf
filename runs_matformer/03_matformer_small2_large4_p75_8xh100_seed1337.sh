#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-matformer_small2_large4_p75_8xh100_seed1337}"
export WANDB_NAME="${WANDB_NAME:-matformer_small2_large4_p75_8xh100_seed1337}"
export SEED="${SEED:-1337}"
export MLP_MULT="${MLP_MULT:-2}"
export LARGE_MLP_MULT="${LARGE_MLP_MULT:-4}"
export LARGE_SAMPLE_PROB="${LARGE_SAMPLE_PROB:-0.75}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_matformer "$@"
