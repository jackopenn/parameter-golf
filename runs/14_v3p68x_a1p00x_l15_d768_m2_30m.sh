#!/usr/bin/env bash
set -euo pipefail

export NUM_LAYERS=15
export MODEL_DIM=768
export NUM_HEADS=12
export NUM_KV_HEADS=6
export MLP_MULT=2
export HYPER_K=16
export HYPER_S=6
export LORA_RANK=12
export HYPER_BASIS_GROUP_SIZE=1

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_virtual_budget_h10030.sh" v3p68x_a1p00x_l15_d768_m2_30m
