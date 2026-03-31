# Same Actual / Larger Virtual Sweep

This sweep targets the hypothesis:

> keep stored/exported params roughly equal to the dense baseline, while increasing the dense logical model size.

These runs all use factorized HyperLoRA export by default, so:

- `actual params` = stored/exported factorized checkpoint params
- `virtual params` = dense-equivalent logical params if every layer matrix were materialized

## Baseline Reference

- dense baseline actual params: `17,059,912`
- dense baseline virtual params: `17,059,912`

All HyperLoRA runs below target roughly that same actual budget, within about `+/-1%`, while increasing virtual params from `2.23x` to `4.09x`.

## Formulas

For model shape:

- `L = NUM_LAYERS`
- `d = MODEL_DIM`
- `kv = NUM_KV_HEADS * (d / NUM_HEADS)`
- `hidden = MLP_MULT * d`
- `groups = ceil(L / HYPER_BASIS_GROUP_SIZE)`

the dense logical size is:

```text
virtual_params =
    vocab_size * d
    + skip_count * d
    + L * (2*d*d + 2*kv*d + 2*hidden*d + 4*d + NUM_HEADS)
```

and the stored factorized size is:

```text
sumdims = 8*d + 2*kv + 2*hidden

actual_params =
    vocab_size * d
    + skip_count * d
    + L * (4*d + NUM_HEADS)
    + groups * sumdims * HYPER_K * HYPER_S
    + L * (6 * HYPER_K)
    + L * (sumdims * LORA_RANK + 6)
```

## Recommended Runs

| Priority | Script | Shape | Heads / KV | K | s | r | Group size | Virtual params | Actual params | Virtual / Baseline | Actual / Baseline |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `runs/10_dense_ref_wandb_30m.sh` | `L9 D512 M2` | `8 / 4` | - | - | - | - | 17,059,912 | 17,059,912 | 1.00x | 1.00x |
| 1 | `runs/11_v2p23x_a1p00x_l13_d640_m2_30m.sh` | `L13 D640 M2` | `10 / 5` | 24 | 6 | 8 | 1 | 37,966,210 | 17,134,880 | 2.23x | 1.00x |
| 2 | `runs/12_v2p71x_a1p00x_l11_d768_m2_30m.sh` | `L11 D768 M2` | `12 / 6` | 24 | 6 | 4 | 1 | 46,240,644 | 17,079,798 | 2.71x | 1.00x |
| 3 | `runs/13_v3p07x_a1p00x_l14_d640_m3_30m.sh` | `L14 D640 M3` | `10 / 5` | 10 | 12 | 2 | 1 | 52,305,420 | 17,093,544 | 3.07x | 1.00x |
| 4 | `runs/14_v3p68x_a1p00x_l15_d768_m2_30m.sh` | `L15 D768 M2` | `12 / 6` | 16 | 6 | 12 | 1 | 62,769,588 | 17,013,678 | 3.68x | 1.00x |
| 5 | `runs/15_v4p01x_a1p01x_l12_d896_m2_30m.sh` | `L12 D896 M2` | `14 / 7` | 14 | 8 | 4 | 1 | 68,402,600 | 17,181,152 | 4.01x | 1.01x |
| 6 | `runs/16_v4p09x_a1p00x_l13_d768_m3_30m.sh` | `L13 D768 M3` | `12 / 6` | 16 | 6 | 12 | 1 | 69,840,540 | 17,006,538 | 4.09x | 1.00x |

## What Each Run Tests

- `11`: smallest clean step above baseline, mostly depth + width.
- `12`: width-first move to `D768` while holding MLP at `2x`.
- `13`: same-width family, but spends the budget on `MLP_MULT=3`.
- `14`: deeper `D768` model with actual params still on budget.
- `15`: widest `2x` MLP run in the set.
- `16`: aggressive `D768` + `MLP_MULT=3` stretch run while still staying on the actual budget.

## Runtime Notes

These runs now keep the same batch size as the dense baseline:

- `TRAIN_BATCH_TOKENS=393216`

So the comparison is cleaner: the main change is model shape and factorized parameterization, not token budget per step.

The tradeoff is that the larger virtual models may run slower or hit memory limits sooner than the earlier reduced-batch versions.

## W&B

The dedicated launcher enables W&B by default with:

- `WANDB_PROJECT=parameter-golf`
- `WANDB_ENTITY=jackpenn`
- `WANDB_GROUP=h10030_virtual_budget`

You can still override any of these per run.
