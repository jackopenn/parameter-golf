# 1xH100 / 30-Minute HyperLoRA Grid

This directory contains a small experiment grid for the grouped-basis + direct-alpha + LoRA training setup in `train_gpt_hyperlora.py`.

## Constraint Summary

The current default export path is `EXPORT_MODE=factored`.

That means HyperLoRA runs export the shared representation directly:

- grouped shared basis factors
- per-layer alpha coefficients
- per-layer LoRA factors
- embeddings, skip weights, and block control tensors

At inference, each `HyperLoRALinear` rebuilds its dense weight from those shared factors inside the forward pass. We are no longer collapsing the checkpoint back to a dense baseline-shaped state dict by default.

So for HyperLoRA runs:

- exported parameter count is the same as the train-time HyperLoRA parameter count
- the old dense-baseline `~15.89MB` estimate no longer applies
- actual artifact bytes need to be measured from the `Serialized factored int8+zlib` line in the logs

The dense baseline reference is still useful:

- baseline dense export params: `17,059,912`
- baseline dense int8+zlib artifact: `15,815,847` bytes
  source: `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`

Lower factorized export param counts are the safest first bet for fitting 16MB. Increasing group sharing is the cheapest way to reduce exported size without changing the dense logical model shape.

## Parameter Formula

For the baseline 9-layer, 512-dim logical model, the grouped-basis HyperLoRA parameter count is:

```text
hyperlora_params =
    544,840
    + num_basis_groups * 6,656 * K * s
    + 9 * (6 * K)
    + 9 * (6,656 * r + 6)
```

where:

- `K` = number of basis components
- `s` = rank per basis component
- `r` = LoRA rank
- `num_basis_groups = ceil(9 / HYPER_BASIS_GROUP_SIZE)`

For HyperLoRA runs with `EXPORT_MODE=factored`:

- `exported_params = hyperlora_params`

The practical point is:

- increasing `HYPER_BASIS_GROUP_SIZE` lowers exported params a lot
- decreasing `HYPER_BASIS_GROUP_SIZE` buys capacity without changing the dense logical model shape
- increasing `K` and `s` increases both params and compute

So `group_size` is the cheapest dial for export size, while `K` and `s` are the main dials for added capacity.

## Recommended Grid

All HyperLoRA runs below keep the same dense logical model shape and export in factorized shared format by default. `00` is the dense baseline reference.

`basis_rank = K * s`

`compute_proxy = K * s + r`

| Priority | Script | Export style | Group size | Basis groups | K | s | r | basis_rank | compute_proxy | Export params | Export/Baseline |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `runs/00_dense_baseline_30m.sh` | dense | - | - | - | - | - | - | - | 17,059,912 | 1.00x |
| 1 | `runs/01_g1_k32_s8_r8_30m.sh` | factored | 1 | 9 | 32 | 8 | 8 | 256 | 264 | 16,361,278 | 0.96x |
| 2 | `runs/02_g1_k36_s8_r8_30m.sh` | factored | 1 | 9 | 36 | 8 | 8 | 288 | 296 | 18,278,422 | 1.07x |
| 3 | `runs/04_g1_k24_s12_r8_30m.sh` | factored | 1 | 9 | 24 | 12 | 8 | 288 | 296 | 18,277,774 | 1.07x |
| 4 | `runs/05_g1_k24_s12_r16_30m.sh` | factored | 1 | 9 | 24 | 12 | 16 | 288 | 304 | 18,757,006 | 1.10x |
| 5 | `runs/03_g1_k40_s8_r8_30m.sh` | factored | 1 | 9 | 40 | 8 | 8 | 320 | 328 | 20,195,566 | 1.18x |
| 6 | `runs/06_g2_k24_s12_r8_30m.sh` | factored | 2 | 5 | 24 | 12 | 8 | 288 | 296 | 10,610,062 | 0.62x |
| 7 | `runs/07_g3_k24_s12_r8_30m.sh` | factored | 3 | 3 | 24 | 12 | 8 | 288 | 296 | 6,776,206 | 0.40x |
| 8 | `runs/08_g9_k24_s12_r8_30m.sh` | factored | 9 | 1 | 24 | 12 | 8 | 288 | 296 | 2,942,350 | 0.17x |

## What To Learn From This Sweep

- `01` vs `02` vs `03`: does larger per-layer grouped-basis capacity help when export is still factorized?
- `02` vs `04`: same rough export size, different factorization shape (`36x8` vs `24x12`)
- `04` vs `05`: does more LoRA residual help once basis capacity is already reasonable?
- `04` vs `06` vs `07` vs `08`: how much sharing is too much?

## Runtime Choices

These launchers target:

- `1xH100`
- `30 minutes wallclock`
- baseline logical model dimensions
- moderate validation cadence

Common settings:

- `MAX_WALLCLOCK_SECONDS=1800`
- `ITERATIONS=60000`
- `WARMDOWN_ITERS=3600`
- `TRAIN_SEQ_LEN=1024`
- `VAL_LOSS_EVERY=4000`
- `TRAIN_LOG_EVERY=100`

Batch sizes:

- most runs use `TRAIN_BATCH_TOKENS=393216`
- the two heavier runs use `TRAIN_BATCH_TOKENS=262144`

If a run is clearly stable and fast on your H100, the first knob to raise is `TRAIN_BATCH_TOKENS`.

## Suggested Run Order

If you do not want to run everything, start with:

1. `runs/00_dense_baseline_30m.sh`
2. `runs/08_g9_k24_s12_r8_30m.sh`
3. `runs/07_g3_k24_s12_r8_30m.sh`
4. `runs/06_g2_k24_s12_r8_30m.sh`
5. `runs/01_g1_k32_s8_r8_30m.sh`

Then use:

- `runs/02_g1_k36_s8_r8_30m.sh` and `runs/04_g1_k24_s12_r8_30m.sh` for larger factored exports
- `runs/05_g1_k24_s12_r16_30m.sh` if the basis-heavy runs are stable and you want more local correction
- `runs/03_g1_k40_s8_r8_30m.sh` as the largest speculative run in this set
