"""
MLX implementation of Block Attention Residuals.
Provides naive, Phase-1 (batched inter-block), and Phase-2 (online-softmax merge)
implementations for benchmarking on Apple Silicon.

All functions operate on MLX arrays and follow the same mathematical
semantics as the PyTorch reference in baseline_pytorch/attn_res.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# RMSNorm (keys only — matches PyTorch RMSNorm)
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rms * weight


# ---------------------------------------------------------------------------
# Naive depth attention (mirrors PyTorch depth_attention exactly)
# ---------------------------------------------------------------------------

def depth_attention_naive(
    sources: mx.array,          # [N, B, T, D]
    pseudo_query: mx.array,     # [D]
    key_weight: mx.array,       # [D]
    eps: float = 1e-6,
) -> mx.array:
    """
    Full naive depth attention: stack → RMSNorm → logits → softmax → weighted sum.
    Returns: [B, T, D]
    """
    K = rms_norm(sources, key_weight, eps=eps)                       # [N, B, T, D]
    logits = mx.sum(K * pseudo_query[None, None, None, :], axis=-1)  # [N, B, T]
    weights = mx.softmax(logits, axis=0)                             # softmax over depth
    out = mx.sum(weights[..., None] * sources, axis=0)               # [B, T, D]
    return out


# ---------------------------------------------------------------------------
# Phase 1: Batched inter-block attention
# ---------------------------------------------------------------------------

def phase1_interblock(
    block_reps: mx.array,       # [N_prev, B, T, D] — completed blocks
    pseudo_queries: mx.array,   # [S, D] — all queries in current block
    key_weight: mx.array,       # [D]
    eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Phase 1 of Algorithm 1: batched inter-block attention.

    All S pseudo-queries attend over the N_prev cached block representations
    in a single batched operation.

    Returns:
        o1: [S, B, T, D]  — unnormalized weighted sums
        m1: [S, B, T]     — per-query row-max of logits
        l1: [S, B, T]     — per-query exp-sum (log-sum-exp denominator)
    """
    K = rms_norm(block_reps, key_weight, eps=eps)  # [N, B, T, D]

    # logits[s, n, b, t] = sum_d( K[n,b,t,d] * Q[s,d] )
    logits = mx.sum(
        K[None, :, :, :, :] * pseudo_queries[:, None, None, None, :],
        axis=-1,
    )  # [S, N, B, T]

    m1 = mx.max(logits, axis=1)                    # [S, B, T]
    exp_shifted = mx.exp(logits - m1[:, None, :, :])  # [S, N, B, T]
    l1 = mx.sum(exp_shifted, axis=1)               # [S, B, T]

    # o1[s, b, t, d] = sum_n( exp_shifted[s,n,b,t] * V[n,b,t,d] )
    o1 = mx.sum(
        exp_shifted[:, :, :, :, None] * block_reps[None, :, :, :, :],
        axis=1,
    )  # [S, B, T, D]

    return o1, m1, l1


# ---------------------------------------------------------------------------
# Phase 2: Online-softmax merge (reference Python implementation)
# ---------------------------------------------------------------------------

def phase2_online_merge_ref(
    o1: mx.array,               # [B, T, D]  — Phase-1 unnormalized output
    m1: mx.array,               # [B, T]     — Phase-1 row-max
    l1: mx.array,               # [B, T]     — Phase-1 exp-sum
    partial_block: mx.array,    # [B, T, D]  — current intra-block partial sum
    pseudo_query: mx.array,     # [D]
    key_weight: mx.array,       # [D]
    eps: float = 1e-6,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Phase 2 of Algorithm 1: single intra-block source merge via online softmax.

    Merges Phase-1 statistics (o1, m1, l1) with a single new source
    (the current partial_block) using the online softmax algorithm.

    Returns:
        h:    [B, T, D]  — merged output
        m:    [B, T]     — new combined row-max
        denom:[B, T]     — new combined exp-sum
    """
    # Compute logit for intra-block source
    k2 = rms_norm(partial_block[None, :, :, :], key_weight, eps=eps)[0]  # [B, T, D]
    m2 = mx.sum(k2 * pseudo_query[None, None, :], axis=-1)              # [B, T]

    # Single source: exp(m2 - m2) = 1, so l2 = 1, o2 = partial_block
    # Online softmax merge
    m = mx.maximum(m1, m2)
    a = mx.exp(m1 - m)          # rescale factor for Phase-1
    b = mx.exp(m2 - m)          # rescale factor for intra-block
    denom = a * l1 + b          # b * l2 where l2 = 1

    h = (a[:, :, None] * o1 + b[:, :, None] * partial_block) / denom[:, :, None]

    return h, m, denom


# ---------------------------------------------------------------------------
# Combined: naive_via_two_phase (for correctness testing)
# ---------------------------------------------------------------------------

def depth_attention_two_phase(
    blocks: List[mx.array],         # list of [B, T, D] — completed blocks
    partial_block: Optional[mx.array],  # [B, T, D] or None
    pseudo_query: mx.array,         # [D]
    key_weight: mx.array,           # [D]
    eps: float = 1e-6,
) -> mx.array:
    """
    Compute depth attention using two-phase decomposition.
    Should produce identical output to depth_attention_naive.
    Used for correctness verification.
    """
    if partial_block is None:
        # Only completed blocks — no Phase 2 needed
        block_tensor = mx.stack(blocks, axis=0)  # [N, B, T, D]
        return depth_attention_naive(block_tensor, pseudo_query, key_weight, eps)

    if len(blocks) == 0:
        # Only intra-block source — trivially return partial_block
        return partial_block

    # Phase 1: inter-block
    block_tensor = mx.stack(blocks, axis=0)  # [N, B, T, D]
    o1_batch, m1_batch, l1_batch = phase1_interblock(
        block_tensor,
        pseudo_query[None, :],  # [1, D]
        key_weight, eps,
    )
    o1 = o1_batch[0]    # [B, T, D]
    m1 = m1_batch[0]    # [B, T]
    l1 = l1_batch[0]    # [B, T]

    # Phase 2: merge with partial_block
    h, _, _ = phase2_online_merge_ref(o1, m1, l1, partial_block, pseudo_query, key_weight, eps)
    return h


# ---------------------------------------------------------------------------
# Block state container
# ---------------------------------------------------------------------------

@dataclass
class BlockState:
    """Mutable state for block-level inference."""
    blocks: List[mx.array] = field(default_factory=list)
    partial_block: Optional[mx.array] = None


# ---------------------------------------------------------------------------
# Full model wrapper (for decode benchmarks)
# ---------------------------------------------------------------------------

class BlockAttnResMLX:
    """
    MLX Block AttnRes model skeleton.
    Weights are loaded from converted PyTorch checkpoints.
    Supports naive, two-phase, and fused Phase-2 inference paths.
    """
    def __init__(self, dim: int, num_layers: int, num_heads: int,
                 ff_dim: int, block_size: int, vocab_size: int,
                 max_seq_len: int = 2048):
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.layers_per_block = block_size // 2

        # Pseudo-query vectors per sublayer (attn + mlp = 2 per layer)
        self.attn_queries = [mx.zeros((dim,)) for _ in range(num_layers)]
        self.mlp_queries = [mx.zeros((dim,)) for _ in range(num_layers)]
        self.attn_key_weights = [mx.ones((dim,)) for _ in range(num_layers)]
        self.mlp_key_weights = [mx.ones((dim,)) for _ in range(num_layers)]

    def naive_forward_block_attnres(
        self,
        blocks: List[mx.array],
        partial_block: mx.array,
        pseudo_query: mx.array,
        key_weight: mx.array,
        eps: float = 1e-6,
    ) -> mx.array:
        """Naive path: stack all sources and compute depth attention."""
        if partial_block is None:
            sources = blocks
        else:
            sources = blocks + [partial_block]
        stacked = mx.stack(sources, axis=0)
        return depth_attention_naive(stacked, pseudo_query, key_weight, eps)

    def two_phase_forward(
        self,
        blocks: List[mx.array],
        partial_block: Optional[mx.array],
        pseudo_query: mx.array,
        key_weight: mx.array,
        eps: float = 1e-6,
    ) -> mx.array:
        """Two-phase path: Phase 1 (inter-block) + Phase 2 (merge)."""
        return depth_attention_two_phase(
            blocks, partial_block, pseudo_query, key_weight, eps)
