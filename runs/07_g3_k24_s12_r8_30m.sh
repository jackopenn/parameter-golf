#!/usr/bin/env bash
set -euo pipefail

export HYPER_K=24
export HYPER_S=12
export LORA_RANK=8
export HYPER_BASIS_GROUP_SIZE=3
export TRAIN_BATCH_TOKENS=393216

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_hyperlora_h10030.sh" g3_k24_s12_r8_30m
