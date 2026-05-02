"""
Numerical equivalence tests for AttnRes implementations.

Verifies that:
  1. MLX naive == PyTorch naive (cross-framework)
  2. Phase1 + Phase2_ref == naive (decomposition correctness)
  3. Phase2_fused == Phase2_ref (kernel correctness)

Run: python -m pytest tests/test_numerical_equiv.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Try to import MLX — skip tests gracefully if unavailable
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _skip_if_no_mlx():
    if not HAS_MLX:
        import pytest
        pytest.skip("MLX not available")


def _skip_if_no_torch():
    if not HAS_TORCH:
        import pytest
        pytest.skip("PyTorch not available")


# ---------------------------------------------------------------------------
# Test 1: MLX naive depth attention internal consistency
# ---------------------------------------------------------------------------

def test_naive_uniform_attention_mlx():
    """Zero query → uniform weights → output = mean of sources."""
    _skip_if_no_mlx()
    from mlx_impl.attnres_mlx import depth_attention_naive

    N, B, T, D = 3, 2, 4, 64
    sources = mx.random.normal((N, B, T, D))
    q = mx.zeros((D,))
    w = mx.ones((D,))

    out = depth_attention_naive(sources, q, w)
    mx.eval(out)

    expected = mx.mean(sources, axis=0)
    mx.eval(expected)

    max_err = float(mx.max(mx.abs(out - expected)))
    print(f"  Uniform attention max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Uniform attention failed: {max_err}"


# ---------------------------------------------------------------------------
# Test 2: Two-phase decomposition == naive
# ---------------------------------------------------------------------------

def test_two_phase_equals_naive():
    """Phase1 + Phase2_ref produces same result as naive depth attention."""
    _skip_if_no_mlx()
    from mlx_impl.attnres_mlx import depth_attention_naive, depth_attention_two_phase

    B, T, D = 2, 8, 128
    N_blocks = 4

    np.random.seed(42)
    blocks_np = [np.random.randn(B, T, D).astype(np.float32) for _ in range(N_blocks)]
    partial_np = np.random.randn(B, T, D).astype(np.float32)
    q_np = np.random.randn(D).astype(np.float32) * 0.1
    w_np = np.ones(D, dtype=np.float32)

    blocks_mx = [mx.array(b) for b in blocks_np]
    partial_mx = mx.array(partial_np)
    q_mx = mx.array(q_np)
    w_mx = mx.array(w_np)

    # Naive: stack all sources
    all_sources = mx.stack(blocks_mx + [partial_mx], axis=0)
    out_naive = depth_attention_naive(all_sources, q_mx, w_mx)
    mx.eval(out_naive)

    # Two-phase
    out_twophase = depth_attention_two_phase(blocks_mx, partial_mx, q_mx, w_mx)
    mx.eval(out_twophase)

    max_abs = float(mx.max(mx.abs(out_naive - out_twophase)))
    out_naive_np = np.array(out_naive)
    out_twophase_np = np.array(out_twophase)
    denom = np.maximum(np.abs(out_naive_np), 1e-8)
    max_rel = float(np.max(np.abs(out_naive_np - out_twophase_np) / denom))

    print(f"  Two-phase vs naive: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
    assert max_abs < 1e-5, f"Two-phase decomposition failed: max_abs={max_abs}"


# ---------------------------------------------------------------------------
# Test 3: Fused Phase-2 == Reference Phase-2
# ---------------------------------------------------------------------------

def test_fused_equals_ref():
    """Fused Metal kernel produces same result as reference Python."""
    _skip_if_no_mlx()
    from mlx_impl.phase2_fused import Phase2Runner

    B, T, D = 2, 16, 256

    np.random.seed(123)
    o1 = mx.array(np.random.randn(B, T, D).astype(np.float32))
    partial = mx.array(np.random.randn(B, T, D).astype(np.float32))
    m1 = mx.array(np.random.randn(B, T).astype(np.float32))
    l1 = mx.abs(mx.array(np.random.randn(B, T).astype(np.float32))) + 1.0
    q = mx.array(np.random.randn(D).astype(np.float32) * 0.1)
    w = mx.ones((D,), dtype=mx.float32)
    mx.eval(o1, partial, m1, l1, q, w)

    # Reference
    runner_ref = Phase2Runner(use_fused=False)
    h_ref, m_ref, l_ref = runner_ref.run(o1, m1, l1, partial, q, w)
    mx.eval(h_ref, m_ref, l_ref)

    # Fused
    runner_fused = Phase2Runner(use_fused=True)
    h_fused, m_fused, l_fused = runner_fused.run(o1, m1, l1, partial, q, w)
    mx.eval(h_fused, m_fused, l_fused)

    max_abs_h = float(mx.max(mx.abs(h_ref - h_fused)))
    max_abs_m = float(mx.max(mx.abs(m_ref - m_fused)))
    max_abs_l = float(mx.max(mx.abs(l_ref - l_fused)))

    print(f"  Fused vs ref: h={max_abs_h:.2e}, m={max_abs_m:.2e}, l={max_abs_l:.2e}")
    assert max_abs_h < 1e-5, f"Fused h failed: {max_abs_h}"
    assert max_abs_m < 1e-5, f"Fused m failed: {max_abs_m}"
    assert max_abs_l < 1e-5, f"Fused l failed: {max_abs_l}"


# ---------------------------------------------------------------------------
# Test 4: Cross-framework equivalence (PyTorch CPU vs MLX)
# ---------------------------------------------------------------------------

def test_pytorch_vs_mlx_naive():
    """PyTorch depth_attention on CPU matches MLX depth_attention_naive."""
    _skip_if_no_mlx()
    _skip_if_no_torch()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline_pytorch"))
    from attn_res import depth_attention as pt_depth_attention, RMSNorm as PTRMSNorm
    from mlx_impl.attnres_mlx import depth_attention_naive as mlx_depth_attention

    N, B, T, D = 3, 2, 8, 64

    np.random.seed(999)
    sources_np = np.random.randn(N, B, T, D).astype(np.float32)
    q_np = np.random.randn(D).astype(np.float32) * 0.05

    # PyTorch
    sources_pt = torch.from_numpy(sources_np)
    q_pt = torch.from_numpy(q_np)
    norm_pt = PTRMSNorm(D)
    with torch.no_grad():
        norm_pt.weight.fill_(1.0)
        out_pt = pt_depth_attention(sources_pt, q_pt, norm_pt)
    out_pt_np = out_pt.numpy()

    # MLX
    sources_mx = mx.array(sources_np)
    q_mx = mx.array(q_np)
    w_mx = mx.ones((D,))
    out_mx = mlx_depth_attention(sources_mx, q_mx, w_mx)
    mx.eval(out_mx)
    out_mx_np = np.array(out_mx)

    max_abs = float(np.max(np.abs(out_pt_np - out_mx_np)))
    print(f"  PyTorch vs MLX: max_abs={max_abs:.2e}")
    assert max_abs < 1e-4, f"Cross-framework mismatch: {max_abs}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_naive_uniform_attention_mlx,
        test_two_phase_equals_naive,
        test_fused_equals_ref,
        test_pytorch_vs_mlx_naive,
    ]

    print("=" * 60)
    print("Numerical Equivalence Tests")
    print("=" * 60)

    passed, failed, skipped = 0, 0, 0
    for test in tests:
        name = test.__name__
        print(f"\n{name}:")
        try:
            test()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower():
                print(f"  ⊘ SKIPPED: {e}")
                skipped += 1
            else:
                print(f"  ✗ FAILED: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        exit(1)
