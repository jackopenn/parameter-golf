# Recursive Transformer

A transformer that matches the baseline at 1 iteration, then gains extra effective depth by iterating through the same blocks again with per-iteration low-rank weight perturbations.

## Core Idea

Start with **9 physical blocks** (same as baseline). At iteration 0 you get the baseline model. Additional iterations re-run all 9 blocks with modified weights:

```
W_effective(iter) = W_base + scale[iter] * A[iter] @ B[iter]
```

where `A[iter]` is `(out_dim, rank)` and `B[iter]` is `(rank, in_dim)`. With 3 iterations you get **27 effective layers** while the base weights are shared — you only pay extra for the low-rank deltas.

## Architecture Diagram

```
Input tokens
     │
     ▼
┌──────────┐
│ tok_emb  │  (vocab=1024, dim=512)
│ + RMSNorm│
└────┬─────┘
     │  x0 (residual stream anchor)
     ▼
┌───────────────────────────────────────────────────────────────┐
│  ITERATION 0  (same weights as baseline)                      │
│  Block 0 → Block 1 → ... → Block 8    [W + Δ(0)]            │
├───────────────────────────────────────────────────────────────┤
│  ITERATION 1  (base weights + low-rank perturbation)          │
│  Block 0 → Block 1 → ... → Block 8    [W + Δ(1)]            │
├───────────────────────────────────────────────────────────────┤
│  ITERATION 2  (base weights + different perturbation)         │
│  Block 0 → Block 1 → ... → Block 8    [W + Δ(2)]            │
└────┬──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────┐
│ RMSNorm  │
│ + logits │  (tied embeddings + softcap)
└──────────┘
```

Across the 27 effective layers, the first 13 act as **encoder** (storing skip connections) and the last 14 as **decoder** (consuming them in reverse), preserving the baseline's U-Net-style skip structure.

## Scaling: Baseline → Deeper

| Config | Eff. Layers | Params | Artifact | Extra cost / iter |
|--------|-------------|--------|----------|-------------------|
| 9×1 (baseline) | 9 | 18.9M | 6.63 MB | — |
| 9×2, rank 32 | 18 | 20.9M | 8.27 MB | +1.6 MB |
| 9×3, rank 32 | 27 | 22.8M | 9.04 MB | +0.8 MB |
| 9×4, rank 32 | 36 | 24.7M | 10.39 MB | +1.4 MB |

All fit within the 16 MB budget. Each extra iteration adds 9 effective layers for just the cost of the rank-32 delta matrices.

## Parameter Breakdown (default: 9 blocks × 3 iters, rank 32)

| Component | Params | Description |
|-----------|--------|-------------|
| Embeddings | 524K | Shared tok_emb (tied to lm_head) |
| Base block weights | 16.5M | 9 blocks × 6 linear layers — identical to baseline |
| Low-rank deltas | 5.8M | 3 iters × 9 blocks × 6 layers × (A + B matrices) |
| Control tensors | 25K | attn_scale, mlp_scale, resid_mix, q_gain, skip_weights, iter_gate |
| **Total** | **22.8M** | 27 effective layers, artifact 9.04 MB |

## Key Design Decisions

**Why match the baseline at 1 iteration?**
With `NUM_ITERATIONS=1` the model is architecturally identical to the 9-layer baseline (same block count, same weights, same skip structure). This makes it easy to compare: any improvement comes purely from the extra iterations.

**Why low-rank deltas instead of just reusing the same weights?**
Reusing identical weights N times is equivalent to a single block with different residual scaling. The per-iteration deltas let each pass learn genuinely different transformations while sharing the bulk of the parameters.

**Why Muon for delta matrices?**
The A and B delta matrices are 2D and benefit from the same orthogonalized gradient updates that work well for the base weights. They get a separate Muon optimizer with `DELTA_LR`.

**Why initialize B to zero?**
Starting with zero deltas means the model begins as a standard weight-sharing recursive transformer, then gradually learns to differentiate iterations during training. This avoids instability at init.

## Environment Variables

All baseline hyperparameters are supported, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_BLOCKS` | 9 | Number of physical transformer blocks (9 = baseline) |
| `NUM_ITERATIONS` | 3 | Times to iterate through all blocks (1 = baseline equivalent) |
| `LORA_RANK` | 32 | Rank of per-iteration weight deltas |
| `DELTA_LR` | 0.04 | Learning rate for delta A/B matrices (Muon) |

## Running

```bash
# Train (8xH100)
torchrun --nproc_per_node=8 train_gpt_recursive.py

# Train baseline-equivalent (1 iteration, no deltas used)
NUM_ITERATIONS=1 torchrun --nproc_per_node=8 train_gpt_recursive.py

# Check artifact size
python check_artifact_size.py

# Sweep configurations
python check_artifact_size.py --sweep
```

## Future Directions

- **Test-time iteration scaling**: Use more iterations at inference than training
- **Progressive iteration schedule**: Start with 1 iteration, add more during training
- **Per-block iteration counts**: Let some blocks iterate more than others
- **Rank tuning**: With 43% headroom at default, room to increase rank or model dim
