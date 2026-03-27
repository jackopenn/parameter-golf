# HyperLoRA: Weight Construction Math

## The Core Formula

Every weight matrix $W_{l,m}$ in the transformer (where $l$ = layer index, $m$ = matrix type like Q, K, V, O, fc, proj) is constructed as:

$$
W_{l,m} = \underbrace{\sum_{k=1}^{K} \alpha_k^{(l,m)} \cdot P_k^{(m)} \cdot \left(Q_k^{(m)}\right)^\top}_{\text{shared basis term}} + \underbrace{\lambda_{l,m} \cdot A_{l,m} \cdot B_{l,m}^\top}_{\text{LoRA residual}}
$$

where:

- $W_{l,m} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the full weight matrix used in the forward pass
- $K$ is the number of basis components (e.g. 24)
- $s$ is the rank of each basis component (e.g. 12)
- $r$ is the LoRA rank (e.g. 16)

---

## What Are P and Q? (The Basis Factors)

### Intuition

Think of a standard dense weight matrix $W \in \mathbb{R}^{512 \times 512}$ as a point in a 262,144-dimensional space. Each of the 9 layers has its own $W_Q$, $W_K$, etc. The key insight is that **these weight matrices across layers are not random independent points** — they tend to cluster near a low-dimensional subspace because all layers are doing roughly similar things (projecting the residual stream).

A **basis** is a small set of "template" matrices that span this subspace. Instead of storing 9 independent 512×512 matrices (one per layer), we store $K$ template matrices and then describe each layer's matrix as a weighted combination of those templates.

### What P and Q Actually Are

Each basis component $k$ is a **rank-$s$ matrix** stored in factored form:

$$
\underbrace{P_k^{(m)}}_{d_{\text{out}} \times s} \cdot \underbrace{\left(Q_k^{(m)}\right)^\top}_{s \times d_{\text{in}}} = \underbrace{M_k^{(m)}}_{d_{\text{out}} \times d_{\text{in}}}
$$

- $P_k^{(m)} \in \mathbb{R}^{d_{\text{out}} \times s}$ — the **left factor** (think: "output-side basis vectors")
- $Q_k^{(m)} \in \mathbb{R}^{d_{\text{in}} \times s}$ — the **right factor** (think: "input-side basis vectors")
- $s$ — the **rank** of each basis component (how expressive each template is)

The product $P_k \cdot Q_k^\top$ gives one full-sized $d_{\text{out}} \times d_{\text{in}}$ "template" matrix, but stored using only $(d_{\text{out}} + d_{\text{in}}) \times s$ parameters instead of $d_{\text{out}} \times d_{\text{in}}$.

**Why factor into P and Q instead of storing the full template?** Parameter efficiency. For a 512×512 matrix:
- Full template: 262,144 params
- Factored at rank $s=12$: $(512 + 512) \times 12 = 12,288$ params — a **21× compression** per template

### The Basis Set

For each matrix type $m$ (e.g., all Q projections across layers share one basis), the full basis is:

$$
\mathcal{B}^{(m)} = \left\{ P_1^{(m)} \cdot \left(Q_1^{(m)}\right)^\top, \; P_2^{(m)} \cdot \left(Q_2^{(m)}\right)^\top, \; \ldots, \; P_K^{(m)} \cdot \left(Q_K^{(m)}\right)^\top \right\}
$$

This is $K$ template matrices, each of rank $s$. Together they span a subspace of rank at most $K \cdot s$ in the space of $d_{\text{out}} \times d_{\text{in}}$ matrices.

### The Mixing Coefficients ($\alpha$)

The hypernetwork produces per-(layer, type) mixing coefficients:

$$
\boldsymbol{\alpha}^{(l,m)} = \text{HyperNet}^{(m)}\!\left(\mathbf{z}_{l,m}\right) \in \mathbb{R}^K
$$

where $\mathbf{z}_{l,m} \in \mathbb{R}^{d_{\text{embed}}}$ is a learnable conditioning vector for layer $l$, matrix type $m$.

The HyperNet is a tiny MLP:

$$
\text{HyperNet}(\mathbf{z}) = W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2
$$

The base weight is then:

$$
W_{\text{base}}^{(l,m)} = \sum_{k=1}^{K} \alpha_k^{(l,m)} \cdot P_k^{(m)} \cdot \left(Q_k^{(m)}\right)^\top
$$

**Efficiently computed** via einsum without ever forming the full $K$ template matrices:

$$
W_{\text{base}} = \underbrace{\left(\sum_k \alpha_k \cdot P_k\right)}_{d_{\text{out}} \times s} \cdot \underbrace{\left(\sum_k \alpha_k \cdot Q_k\right)^\top}_{s \times d_{\text{in}}}
$$

Wait — that's not quite right because the $\alpha$'s multiply the outer products, not the individual factors. The correct efficient computation is:

$$
W_{\text{base}}[i,j] = \sum_{k=1}^{K} \alpha_k \sum_{t=1}^{s} P_k[i,t] \cdot Q_k[j,t]
$$

In PyTorch: `torch.einsum('k, kos, kis -> oi', alpha, P, Q)`

---

## The LoRA Residual

The basis captures **what's shared across layers**. But each layer also needs to be unique. The LoRA term handles this:

$$
W_{\text{LoRA}}^{(l,m)} = \lambda_{l,m} \cdot \underbrace{A_{l,m}}_{d_{\text{out}} \times r} \cdot \underbrace{B_{l,m}^\top}_{r \times d_{\text{in}}}
$$

- $A_{l,m} \in \mathbb{R}^{d_{\text{out}} \times r}$ — per-layer, per-type left factor (init: Kaiming)
- $B_{l,m} \in \mathbb{R}^{r \times d_{\text{in}}}$ — per-layer, per-type right factor (init: **zeros**)
- $\lambda_{l,m} \in \mathbb{R}$ — learnable gating scalar (init: **0**)

Because $B$ and $\lambda$ are initialized to zero, the LoRA term contributes nothing at the start of training. The hypernetwork basis dominates early, and the LoRA term gradually "turns on" as $\lambda$ and $B$ move away from zero during training.

---

## Putting It All Together

The forward pass for a single linear layer at position $(l, m)$:

$$
\boxed{
\mathbf{y} = W_{l,m} \cdot \mathbf{x} = \left[\sum_{k=1}^{K} \alpha_k^{(l,m)} \cdot P_k^{(m)} \cdot \left(Q_k^{(m)}\right)^\top + \lambda_{l,m} \cdot A_{l,m} \cdot B_{l,m}^\top\right] \cdot \mathbf{x}
}
$$

---

## Parameter Counting

### What's Shared (stored once, used by all layers)

For one matrix type with shape $(d_{\text{out}}, d_{\text{in}})$:

$$
\text{Basis params} = K \times (d_{\text{out}} + d_{\text{in}}) \times s
$$

Example for $W_Q$ (512×512), $K=24$, $s=12$:

$$
24 \times (512 + 512) \times 12 = 294{,}912 \text{ params}
$$

### What's Per-Layer (stored once per layer)

For one matrix at one layer:

$$
\text{LoRA params} = (d_{\text{out}} + d_{\text{in}}) \times r + 1
$$

Plus $K$ mixing coefficients $\alpha_k$ (saved after training, tiny).

Example for $W_Q$ (512×512), $r=16$:

$$
(512 + 512) \times 16 + 1 + 24 = 16{,}409 \text{ params per layer}
$$

### Virtual Parameters (what the forward pass "sees")

$$
\text{Virtual params} = d_{\text{out}} \times d_{\text{in}} = 262{,}144
$$

### Amplification Ratio

$$
\text{Amplification} = \frac{\text{virtual params}}{\text{stored params}} = \frac{d_{\text{out}} \times d_{\text{in}}}{\frac{K(d_{\text{out}} + d_{\text{in}})s}{L} + (d_{\text{out}} + d_{\text{in}})r + K + 1}
$$

The basis cost is amortized across $L$ layers (since it's shared), while LoRA cost is per-layer. More layers = higher amplification.

### Full Model Budget (d=512, L=9, K=24, s=12, r=16)

| Component | Params | Notes |
|-----------|--------|-------|
| 6 basis sets | $6 \times 24 \times \bar{d} \times 12$ | ~1.77M (shared) |
| 54 LoRA pairs | $54 \times \bar{d} \times 16$ | ~0.99M (per-layer) |
| 54 $\alpha$ vectors | $54 \times 24$ | 1,296 (tiny) |
| 54 $\lambda$ scalars | 54 | 54 (tiny) |
| HyperNet MLPs | $6 \times (64 \times 128 + 128 \times 24)$ ≈ 67K | **discarded after training** |
| $z$ embeddings | $9 \times 6 \times 64$ = 3,456 | **discarded after training** |
| tok_emb | 524,288 | unchanged |
| skip_weights + scalars | ~30K | unchanged |
| **Total stored** | **~3.4M** | |
| **Total virtual** | **~16.5M** | |
| **Amplification** | **~4.9×** | |

---

## SVD Interpretation

If you're familiar with SVD, here's another way to think about it. A rank-$s$ matrix can be written as:

$$
M = U \Sigma V^\top = \sum_{i=1}^{s} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

Our basis component $P_k Q_k^\top$ is exactly this — $P_k$'s columns are the left singular vectors (times singular values), $Q_k$'s columns are the right singular vectors.

The mixing coefficient $\alpha_k$ then scales the entire rank-$s$ component. So the full basis term:

$$
\sum_k \alpha_k P_k Q_k^\top
$$

is a **weighted sum of rank-$s$ matrices**, giving a matrix of rank at most $K \times s$. With $K=24, s=12$, the basis can represent matrices up to rank 288 — more than enough for a 512×512 matrix (which has max rank 512).

The LoRA term adds $r$ more dimensions of rank, giving a total potential rank of $K \cdot s + r$.
