"""
Artifact Size Checker for Recursive Transformer

Checks whether the recursive transformer weights + code fit within the 16MB artifact limit.
We ship:
  1. The recursive transformer weights (quantized + compressed)
     - Base weights shared across iterations
     - Per-iteration low-rank delta matrices (A, B)
     - Control tensors (scales, gains, etc.)
  2. The code (train_gpt_recursive.py)

Usage:
    python check_artifact_size.py [--sweep]
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Import the model from the training script
sys.path.insert(0, str(Path(__file__).parent))
from train_gpt_recursive import (
    RecursiveGPT,
    quantize_state_dict_int8,
)


@dataclass
class RecursiveConfig:
    vocab_size: int = 1024
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    num_blocks: int = 9
    num_iterations: int = 3
    lora_rank: int = 32
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5


def estimate_artifact_size(
    config: RecursiveConfig,
    code_files: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Estimate the full artifact size."""
    model = RecursiveGPT(
        vocab_size=config.vocab_size,
        num_blocks=config.num_blocks,
        num_iterations=config.num_iterations,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        mlp_mult=config.mlp_mult,
        lora_rank=config.lora_rank,
        tie_embeddings=config.tie_embeddings,
        tied_embed_init_std=config.tied_embed_init_std,
        logit_softcap=config.logit_softcap,
        rope_base=config.rope_base,
        qk_gain_init=config.qk_gain_init,
    )

    # Quantize
    quant_obj, quant_stats = quantize_state_dict_int8(model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)

    weights_raw = len(raw_bytes)
    weights_compressed = len(compressed)

    # Code size
    code_total = 0
    if code_files is None:
        here = Path(__file__).parent
        code_files_paths = [here / "train_gpt_recursive.py"]
    else:
        code_files_paths = [Path(f) for f in code_files]

    code_sizes = {}
    for f in code_files_paths:
        if f.exists():
            size = f.stat().st_size
            code_sizes[f.name] = size
            code_total += size

    total = weights_compressed + code_total
    budget = 16_000_000

    # Parameter breakdown
    n_params = sum(p.numel() for p in model.parameters())
    base_params = 0
    delta_params = 0
    embed_params = 0
    other_params = 0
    for name, p in model.named_parameters():
        if "tok_emb" in name or "lm_head" in name:
            embed_params += p.numel()
        elif ".deltas." in name:
            delta_params += p.numel()
        elif ".base." in name:
            base_params += p.numel()
        else:
            other_params += p.numel()

    effective_layers = config.num_blocks * config.num_iterations

    if verbose:
        print("=" * 60)
        print("RECURSIVE TRANSFORMER ARTIFACT SIZE REPORT")
        print("=" * 60)

        print(f"\n--- Architecture ---")
        print(f"  Physical blocks:     {config.num_blocks}")
        print(f"  Iterations:          {config.num_iterations}")
        print(f"  Effective layers:    {effective_layers}")
        print(f"  LoRA rank:           {config.lora_rank}")
        print(f"  Model dim:           {config.model_dim}")
        print(f"  MLP mult:            {config.mlp_mult}")

        print(f"\n--- Parameter Breakdown ---")
        print(f"  Embedding params:    {embed_params:>12,}")
        print(f"  Base block params:   {base_params:>12,}")
        print(f"  Delta params:        {delta_params:>12,}")
        print(f"  Other (control):     {other_params:>12,}")
        print(f"  Total params:        {n_params:>12,}")

        # Compare to baseline with same effective depth
        baseline_params_est = effective_layers * (
            config.model_dim * config.model_dim * 4  # q,k,v,proj
            + config.model_dim * (config.mlp_mult * config.model_dim) * 2  # fc, proj
        ) + config.vocab_size * config.model_dim  # embeddings
        print(f"\n  Baseline equivalent ({effective_layers}L): ~{baseline_params_est:,}")
        print(f"  Compression ratio:   {baseline_params_est / n_params:.2f}x")

        print(f"\n--- Weights ---")
        print(f"  Raw state dict:  {weights_raw:>12,} bytes ({weights_raw / 1e6:.2f} MB)")
        print(f"  Int8 + zlib:     {weights_compressed:>12,} bytes ({weights_compressed / 1e6:.2f} MB)")
        print(f"  Compression:     {weights_raw / max(weights_compressed, 1):.2f}x")

        print(f"\n--- Code Files ---")
        for name, size in code_sizes.items():
            print(f"  {name}: {size:>8,} bytes")
        print(f"  Total code:      {code_total:>12,} bytes ({code_total / 1e6:.2f} MB)")

        print(f"\n--- Total Artifact ---")
        print(f"  Weights + Code:  {total:>12,} bytes ({total / 1e6:.2f} MB)")

        if total <= budget:
            headroom = budget - total
            print(f"  STATUS: FITS! ({headroom:,} bytes headroom, {headroom/budget*100:.1f}%)")
        else:
            overage = total - budget
            print(f"  STATUS: OVER BUDGET by {overage:,} bytes ({overage/1e6:.2f} MB)")

    return {
        "weights_raw": weights_raw,
        "weights_compressed": weights_compressed,
        "code_total": code_total,
        "total": total,
        "budget": budget,
        "fits": total <= budget,
        "n_params": n_params,
        "base_params": base_params,
        "delta_params": delta_params,
        "effective_layers": effective_layers,
    }


def sweep_configs():
    """Try different recursive transformer configurations and report sizes."""
    print("=" * 80)
    print("RECURSIVE TRANSFORMER CONFIG SWEEP")
    print("=" * 80)

    configs = [
        # (name, num_blocks, num_iterations, lora_rank, model_dim, mlp_mult)
        # --- Baseline-equivalent (1 iter = 9 blocks = baseline) ---
        ("9x1-baseline",  9, 1, 32, 512, 2),
        ("9x2-r16",       9, 2, 16, 512, 2),
        ("9x2-r32",       9, 2, 32, 512, 2),
        ("9x2-r48",       9, 2, 48, 512, 2),
        ("9x2-r64",       9, 2, 64, 512, 2),
        ("9x3-r16",       9, 3, 16, 512, 2),
        ("9x3-r32",       9, 3, 32, 512, 2),
        ("9x3-r48",       9, 3, 48, 512, 2),
        ("9x3-r64",       9, 3, 64, 512, 2),
        ("9x4-r32",       9, 4, 32, 512, 2),
        # --- Scaling up with headroom ---
        ("9x2-r32-3x",    9, 2, 32, 512, 3),
        ("9x3-r32-3x",    9, 3, 32, 512, 3),
        ("9x2-r32-d576",  9, 2, 32, 576, 2),
        ("9x3-r32-d576",  9, 3, 32, 576, 2),
        # --- Fewer blocks, more iters (smaller base, cheaper) ---
        ("5x2-r32",       5, 2, 32, 512, 2),
        ("5x3-r32",       5, 3, 32, 512, 2),
        ("3x3-r32",       3, 3, 32, 512, 2),
    ]

    print(f"\n{'Name':<22} {'Blocks':<7} {'Iters':<6} {'Rank':<5} {'Dim':<5} {'MLP':<4} "
          f"{'EffL':<5} {'Params':<12} {'Base':<10} {'Delta':<10} "
          f"{'Artifact MB':<12} {'Fits?':<6}")
    print("-" * 120)

    for name, nblocks, niters, rank, dim, mlp in configs:
        cfg = RecursiveConfig(
            num_blocks=nblocks,
            num_iterations=niters,
            lora_rank=rank,
            model_dim=dim,
            mlp_mult=mlp,
        )
        result = estimate_artifact_size(cfg, verbose=False)

        status = "YES" if result["fits"] else "NO"
        print(f"{name:<22} {nblocks:<7} {niters:<6} {rank:<5} {dim:<5} {mlp:<4} "
              f"{result['effective_layers']:<5} {result['n_params']:<12,} "
              f"{result['base_params']:<10,} {result['delta_params']:<10,} "
              f"{result['total']/1e6:<12.2f} {status:<6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check artifact size for recursive transformer")
    parser.add_argument("--sweep", action="store_true", help="Sweep different configurations")
    args = parser.parse_args()

    if args.sweep:
        sweep_configs()
    else:
        config = RecursiveConfig()
        result = estimate_artifact_size(config)
