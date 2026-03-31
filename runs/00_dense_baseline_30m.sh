#!/usr/bin/env bash
set -euo pipefail

export HYPER_MODE=dense
export EXPORT_MODE=dense_baseline
export TRAIN_BATCH_TOKENS=393216

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_launch_hyperlora_h10030.sh" dense_baseline_30m
