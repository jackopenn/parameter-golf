#!/usr/bin/env bash
set -euo pipefail

export NUM_LAYERS=14
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export MLP_MULT=3
export HYPER_K=10
export HYPER_S=12
export LORA_RANK=2
export HYPER_BASIS_GROUP_SIZE=1

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_virtual_budget_h10030.sh" v3p07x_a1p00x_l14_d640_m3_30m
