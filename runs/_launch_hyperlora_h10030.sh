#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN_NAME="$1"
shift || true

: "${PYTHON:=python3}"

timestamp="$(date +%Y%m%d_%H%M%S)"

export RUN_ID="${RUN_ID:-${RUN_NAME}_${timestamp}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export ITERATIONS="${ITERATIONS:-60000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3600}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"

export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"

export HYPER_MODE="${HYPER_MODE:-hyper_plus_lora}"
export EXPORT_MODE="${EXPORT_MODE:-factored}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-joint}"
export MATRIX_LR="${MATRIX_LR:-0.04}"
export SCALAR_LR="${SCALAR_LR:-0.04}"
export ALPHA_LR="${ALPHA_LR:-0.001}"
export LORA_LR="${LORA_LR:-0.04}"

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  export WANDB_GROUP="${WANDB_GROUP:-h10030_grouped_basis}"
  export WANDB_TAGS="${WANDB_TAGS:-1xh100,30m,grouped-basis}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_ID}"
fi

echo "Launching ${RUN_ID}"
echo "  HYPER_MODE=${HYPER_MODE} EXPORT_MODE=${EXPORT_MODE}"
echo "  NUM_LAYERS=${NUM_LAYERS} MODEL_DIM=${MODEL_DIM} NUM_HEADS=${NUM_HEADS} NUM_KV_HEADS=${NUM_KV_HEADS} MLP_MULT=${MLP_MULT}"
echo "  HYPER_K=${HYPER_K:-dense} HYPER_S=${HYPER_S:-dense} LORA_RANK=${LORA_RANK:-dense} HYPER_BASIS_GROUP_SIZE=${HYPER_BASIS_GROUP_SIZE:-dense}"
echo "  TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS} MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"

exec "$PYTHON" train_gpt_hyperlora.py "$@"
