"""
Phase-2 runner: unified interface for reference vs fused implementations.
"""

from __future__ import annotations
import mlx.core as mx
from mlx_impl.kernels import Phase2MergeFused, compute_m2
from mlx_impl.attnres_mlx import phase2_online_merge_ref


class Phase2Runner:
    """
    Dispatches Phase-2 online-softmax merge to either:
      - Reference Python implementation (phase2_online_merge_ref)
      - Fused Metal kernel (Phase2MergeFused)

    Both paths produce numerically equivalent results.
    """

    def __init__(self, use_fused: bool = True):
        self.use_fused = use_fused
        self._fused = Phase2MergeFused() if use_fused else None

    def run(
        self,
        o1: mx.array,              # [B, T, D]
        m1: mx.array,              # [B, T]
        l1: mx.array,              # [B, T]
        partial_block: mx.array,   # [B, T, D]
        pseudo_query: mx.array,    # [D]
        key_weight: mx.array,      # [D]
        eps: float = 1e-6,
    ):
        if self.use_fused:
            m2 = compute_m2(partial_block, pseudo_query, key_weight, eps)
            h, m_new, l_new = self._fused(o1, partial_block, m1, l1, m2)
            return h, m_new, l_new

        return phase2_online_merge_ref(
            o1, m1, l1, partial_block, pseudo_query, key_weight, eps)
