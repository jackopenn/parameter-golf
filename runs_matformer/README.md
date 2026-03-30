# Two-Scale MatFormer Runs

These runs keep only two FFN sizes:

- small export model: `MLP_MULT=2`
- large training-only model: `LARGE_MLP_MULT=4`

The exported model is always the small one, which is intended to stay under the 16 MB packed limit.

Runs:

- `01_baseline_8xh100_seed1337.sh`: original baseline trainer for comparison
- `02_matformer_small2_large4_p50_8xh100_seed1337.sh`: balanced sampling between small and large FFNs
- `03_matformer_small2_large4_p75_8xh100_seed1337.sh`: large-biased sampling
- `11_baseline_30m_1xh100_seed1337.sh`: 30-minute single-H100 baseline
- `12_matformer_small2_large4_p50_30m_1xh100_seed1337.sh`: 30-minute single-H100 balanced sampling
- `13_matformer_small2_large4_p75_30m_1xh100_seed1337.sh`: 30-minute single-H100 large-biased sampling

The 30-minute single-H100 scripts use `WARMDOWN_ITERS=450` so the LR decay occupies roughly the same wall-clock fraction as the original `10m / 8xH100` baseline.

Usage:

```bash
bash runs_matformer/01_baseline_8xh100_seed1337.sh
bash runs_matformer/02_matformer_small2_large4_p50_8xh100_seed1337.sh
bash runs_matformer/03_matformer_small2_large4_p75_8xh100_seed1337.sh
bash runs_matformer/11_baseline_30m_1xh100_seed1337.sh
bash runs_matformer/12_matformer_small2_large4_p50_30m_1xh100_seed1337.sh
bash runs_matformer/13_matformer_small2_large4_p75_30m_1xh100_seed1337.sh
```

Useful overrides:

```bash
NNODES=1 NPROC_PER_NODE=8 bash runs_matformer/02_matformer_small2_large4_p50_8xh100_seed1337.sh
WANDB_ENABLE=0 bash runs_matformer/02_matformer_small2_large4_p50_8xh100_seed1337.sh
MAX_WALLCLOCK_SECONDS=1800 SINGLE_H100_30M=1 NPROC_PER_NODE=1 \
  GRAD_ACCUM_STEPS=8 bash runs_matformer/02_matformer_small2_large4_p50_8xh100_seed1337.sh
```
