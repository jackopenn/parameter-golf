#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-baseline_8xh100_seed1337}"
export WANDB_NAME="${WANDB_NAME:-baseline_8xh100_seed1337}"
export SEED="${SEED:-1337}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_baseline "$@"
