#!/usr/bin/env bash
set -euo pipefail

export NUM_LAYERS=11
export MODEL_DIM=768
export NUM_HEADS=12
export NUM_KV_HEADS=6
export MLP_MULT=2
export HYPER_K=24
export HYPER_S=6
export LORA_RANK=4
export HYPER_BASIS_GROUP_SIZE=1
export TRAIN_BATCH_TOKENS=147456

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_virtual_budget_h10030.sh" v2p71x_a1p00x_l11_d768_m2_30m
