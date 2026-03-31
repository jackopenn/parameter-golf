#!/usr/bin/env bash
set -euo pipefail

export NUM_LAYERS=12
export MODEL_DIM=896
export NUM_HEADS=14
export NUM_KV_HEADS=7
export MLP_MULT=2
export HYPER_K=14
export HYPER_S=8
export LORA_RANK=4
export HYPER_BASIS_GROUP_SIZE=1
export TRAIN_BATCH_TOKENS=81920

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_virtual_budget_h10030.sh" v4p01x_a1p01x_l12_d896_m2_30m
