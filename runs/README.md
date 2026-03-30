Run these from the repo root or directly via `bash runs/<script>.sh`.

The launchers all target:
- `WANDB_PROJECT=parameter-golf`
- `WANDB_ENTITY=jackpenn`
- `WANDB_RUN_GROUP=single-h100-30m`
- `SINGLE_H100_30M=1`
- `python3 train_gpt_branch_tail.py`

Shared defaults live in [runs/_common.sh](/Users/jack/.codex/worktrees/d14a/parameter-golf/runs/_common.sh).

Expected run order:
1. `01_baseline_seed1337.sh`
2. `02_branch_k2_t4_no_kd.sh`
3. `03_branch_k2_t4_kd.sh`
4. `04_branch_k2_t4_kd_consolidate.sh`
5. `05_branch_k2_t8_kd.sh`
6. `06_branch_k2_t8_kd_consolidate.sh`
7. `07_branch_k1_t8_kd_consolidate.sh`
8. `08_branch_k2_t4_kd_consolidate_no_warmup.sh`
9. `09_branch_k2_t8_kd_consolidate_seed2024.sh`
10. `10_baseline_18l_seed1337.sh`
11. `11_baseline_8xh100_seed1337.sh`
12. `12_baseline_18l_8xh100_seed1337.sh`

Useful overrides:
- `TRAIN_BATCH_TOKENS` to push H100 utilization.
- `GRAD_ACCUM_STEPS` if you want a different microbatching regime.
- `MAX_WALLCLOCK_SECONDS` to change the timed schedule budget. Branched-tail runs require this to stay positive.
- `PHASE_A_FRAC`, `PHASE_B_FRAC`, and `PHASE_D_FRAC` to control timed teacher warmup, onboarding, and final student consolidation as fractions of the run budget.
- `DATA_PATH`, `TOKENIZER_PATH`, and `VOCAB_SIZE` if your local dataset layout differs from repo defaults.
- `WANDB_ENABLE=0` if you want to test locally without online logging.

The `11_*` and `12_*` launchers target the original `10 minutes on 8xH100` regime:
- `WANDB_RUN_GROUP=8xh100-10m`
- `SINGLE_H100_30M=0`
- `MAX_WALLCLOCK_SECONDS=600`
- `WARMDOWN_ITERS=1200`
- `GRAD_ACCUM_STEPS=0`, so the trainer derives `8 / WORLD_SIZE` automatically
- `torchrun --standalone --nnodes=${NNODES:-1} --nproc_per_node=${NPROC_PER_NODE:-8}`
