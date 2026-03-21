# Future Work: Joint Optimization of Base Weights and Low-Rank Deltas

## Problem

When Muon (base weights) and Adam (delta A/B matrices) train simultaneously, training flatlines at ~6.0 loss (near random). Root cause: Muon strips gradient magnitude via Newton-Schulz orthogonalization while Adam preserves it — the identical gradient signal `dL/dW_eff` gets transformed incompatibly, causing the effective weight `W + scale*A@B` to oscillate.

**Evidence:**
- `DELTA_LR=0.0` (deltas frozen): loss 2.69 at step 200 — trains perfectly
- `DELTA_LR=0.04` (joint): loss 6.0 at step 200 — flatlined near random

## Research Summary

### Techniques Investigated

| Technique | Core Idea | Muon Fit | Complexity |
|-----------|-----------|----------|------------|
| **ReLoRA** | Periodically merge A@B into W, reset A/B, repeat | Good | Medium |
| **Muon for 2D deltas** | Reshape A/B to 2D so Muon can optimize them (same optimizer = no conflict) | Excellent | Low |
| **GaLore** | Project gradients to low-rank, no separate A/B params | Moderate | Medium-High |
| **DoRA** | Decompose weight into magnitude + direction, LoRA on direction only | Excellent | Medium |
| **PCGrad** | Detect conflicting gradients, project to remove conflict | Moderate | Medium |

### Key Papers
- [ReLoRA: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695)
- [Relaxed Recursive Transformers with Layer-wise LoRA](https://arxiv.org/html/2410.20672v1)
- [Improving Recursive Transformers with Mixture of LoRAs](https://arxiv.org/pdf/2512.12880)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354)
- [Gradient Surgery for Multi-Task Learning (PCGrad)](https://arxiv.org/abs/2001.06782)

## Recommended Approach: Muon for 2D Deltas + ReLoRA Merging

### Step 1: Restructure deltas as 2D params

Replace 3D `A (num_iters, out, rank)` with `nn.ParameterList` of 2D `(out, rank)` matrices. This lets Muon optimize them — eliminating the Muon/Adam mismatch entirely.

```python
# Before
self.A = nn.Parameter(torch.zeros(num_iterations, out_dim, rank))  # 3D, Adam only

# After
self.A_list = nn.ParameterList([
    nn.Parameter(torch.zeros(out_dim, rank)) for _ in range(num_iterations)
])  # 2D each, Muon compatible
```

Feed all 2D delta params into a separate Muon optimizer with `DELTA_LR` (potentially lower than `MATRIX_LR` since thin rectangular matrices get a `sqrt(rows/cols)` scaling boost from Muon).

### Step 2: ReLoRA periodic merge-and-reset

Every `RELORA_MERGE_EVERY` steps (e.g., 100):
1. Merge: `base.weight += delta_scale[i] * A[i] @ B[i]` for all iterations
2. Reset A to small random, B to zeros, delta_scale to 0.1
3. Clear Muon momentum buffers for delta params (stale after reset)
4. Brief LR warmup for deltas (5 steps)

This prevents deltas from growing large enough to interfere, while composing many rank-r updates into progressively higher effective rank. At ~30ms/step on 8xH100, 100 steps = 3 seconds, giving ~60 merges in 10 minutes.

### Step 3: Remove staged training

Delete `delta_phase2_frac` / `enter_phase2` logic. No more frozen params, no recompilation, no `find_unused_parameters=True` overhead.

### Step 4: Final merge before export

Before serialization, merge deltas one last time so the saved model has clean weights with no separate A/B params.

## Expected Impact

| Metric | Current (staged) | After fix |
|--------|-----------------|-----------|
| Step time (8xH100) | ~30ms | ~32ms |
| Full training budget used | 70% base + 30% delta | 100% joint |
| Optimizer conflict | Avoided by freezing | Eliminated by same optimizer |
| Effective rank of updates | rank r (single phase) | Cumulative from ~60 merges |

## Alternative: Train Base Only + Test-Time Recursion

If joint optimization proves too complex, a simpler approach:
1. Train a standard 9-layer baseline (no deltas, no recursion)
2. At inference only, run multiple iterations with small learned deltas
3. Train the deltas in a quick fine-tuning phase after base training

This sidesteps the optimization problem entirely but limits the model to learning recursion patterns only during fine-tuning.
