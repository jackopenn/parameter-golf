#!/usr/bin/env bash
set -euo pipefail

export NUM_LAYERS=13
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export MLP_MULT=2
export HYPER_K=24
export HYPER_S=6
export LORA_RANK=8
export HYPER_BASIS_GROUP_SIZE=1

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_virtual_budget_h10030.sh" v2p23x_a1p00x_l13_d640_m2_30m
