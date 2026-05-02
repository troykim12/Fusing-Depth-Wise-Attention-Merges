"""
Colab-ready Triton Phase-2 online-softmax merge checker.

What this script does:
  1. Checks CUDA + Triton availability.
  2. Runs numerical equivalence tests against a PyTorch reference.
  3. Benchmarks PyTorch CUDA reference vs fused Triton kernel.
  4. Optionally dumps Triton TTIR/TTGIR/PTX snippets for kernel inspection.
  5. Writes results to results/triton_phase2_colab.csv.

Colab usage:
  !python triton_phase2_colab.py --quick --dump-asm
  !python triton_phase2_colab.py --full

Notes:
  - Use Runtime > Change runtime type > GPU in Colab.
  - T4 is enough for quick verification. L4/A100 are better for stable benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics as stats
import sys
import textwrap
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as exc:  # import errors can vary across Colab images
    HAS_TRITON = False
    TRITON_IMPORT_ERROR = repr(exc)


# ---------------------------------------------------------------------------
# PyTorch reference implementation
# ---------------------------------------------------------------------------

def phase2_merge_ref_torch(
    o1: torch.Tensor,       # [B, T, D]
    partial: torch.Tensor,  # [B, T, D]
    m1: torch.Tensor,       # [B, T]
    l1: torch.Tensor,       # [B, T]
    m2: torch.Tensor,       # [B, T]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused PyTorch reference for Phase-2 online-softmax merge."""
    m = torch.maximum(m1, m2)
    a = torch.exp(m1 - m)
    b = torch.exp(m2 - m)
    denom = a * l1 + b  # l2 = 1 for the single intra-block source
    out = (a.unsqueeze(-1) * o1 + b.unsqueeze(-1) * partial) / denom.unsqueeze(-1)
    return out, m, denom


# ---------------------------------------------------------------------------
# Triton fused kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _phase2_merge_kernel(
        o1_ptr, partial_ptr, m1_ptr, l1_ptr, m2_ptr,
        out_ptr, mout_ptr, lout_ptr,
        BT: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One Triton program handles one flattened (B,T) position."""
        bt_idx = tl.program_id(0)

        # Triton 3.6 requires tl.exp inputs to be fp32/fp64.
        # Keep online-softmax merge scalars in fp32 even when o1/partial are fp16.
        m1_val = tl.load(m1_ptr + bt_idx).to(tl.float32)
        l1_val = tl.load(l1_ptr + bt_idx).to(tl.float32)
        m2_val = tl.load(m2_ptr + bt_idx).to(tl.float32)

        m_new = tl.maximum(m1_val, m2_val)
        a = tl.exp(m1_val - m_new)
        b = tl.exp(m2_val - m_new)
        denom = a * l1_val + b
        inv_denom = 1.0 / denom

        base_offset = bt_idx * D

        # Compile-time loop. This supports D > 1024 by iterating over D tiles.
        for d_start in tl.static_range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            mask = d_offs < D
            elem_idx = base_offset + d_offs

            o1_vals = tl.load(o1_ptr + elem_idx, mask=mask, other=0.0).to(tl.float32)
            p_vals = tl.load(partial_ptr + elem_idx, mask=mask, other=0.0).to(tl.float32)
            result = (a * o1_vals + b * p_vals) * inv_denom
            tl.store(out_ptr + elem_idx, result, mask=mask)

        tl.store(mout_ptr + bt_idx, m_new)
        tl.store(lout_ptr + bt_idx, denom)


def _choose_block_d(D: int) -> int:
    """Power-of-two block size accepted by tl.arange, capped for register pressure."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    return int(triton.next_power_of_2(min(D, 1024)))


def phase2_merge_fused_triton(
    o1: torch.Tensor,
    partial: torch.Tensor,
    m1: torch.Tensor,
    l1: torch.Tensor,
    m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Python wrapper for the Triton fused Phase-2 kernel."""
    if not HAS_TRITON:
        raise RuntimeError(f"Triton is not available: {TRITON_IMPORT_ERROR}")
    if not o1.is_cuda:
        raise RuntimeError("Inputs must be CUDA tensors")

    B, T_seq, D = o1.shape
    BT = B * T_seq

    o1_flat = o1.contiguous().view(-1)
    partial_flat = partial.contiguous().view(-1)
    m1_flat = m1.contiguous().view(-1)
    l1_flat = l1.contiguous().view(-1)
    m2_flat = m2.contiguous().view(-1)

    out_flat = torch.empty_like(o1_flat)
    mout_flat = torch.empty_like(m1_flat)
    lout_flat = torch.empty_like(l1_flat)

    block_d = _choose_block_d(D)
    grid = (BT,)
    _phase2_merge_kernel[grid](
        o1_flat, partial_flat, m1_flat, l1_flat, m2_flat,
        out_flat, mout_flat, lout_flat,
        BT=BT, D=D, BLOCK_D=block_d,
        num_warps=4,
    )

    return out_flat.view(B, T_seq, D), mout_flat.view(B, T_seq), lout_flat.view(B, T_seq)


# ---------------------------------------------------------------------------
# Test + benchmark utilities
# ---------------------------------------------------------------------------

def print_environment() -> None:
    print("=" * 80)
    print("Environment")
    print("=" * 80)
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Platform:      {platform.platform()}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA in torch: {torch.version.cuda}")
    print(f"Triton:        {triton.__version__ if HAS_TRITON else 'NOT AVAILABLE'}")

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:    {prop.total_memory / 1e9:.2f} GB")
        print(f"SM capability: {prop.major}.{prop.minor}")
    else:
        print("GPU:           NOT AVAILABLE")
    print()


def make_inputs(B: int, T: int, D: int, dtype: torch.dtype, seed: int = 42):
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    o1 = torch.randn(B, T, D, device=device, dtype=dtype, generator=gen)
    partial = torch.randn(B, T, D, device=device, dtype=dtype, generator=gen)
    m1 = torch.randn(B, T, device=device, dtype=dtype, generator=gen)
    l1 = torch.abs(torch.randn(B, T, device=device, dtype=dtype, generator=gen)) + 1.0
    m2 = torch.randn(B, T, device=device, dtype=dtype, generator=gen)
    return o1, partial, m1, l1, m2


def assert_close_report(name: str, ref: torch.Tensor, got: torch.Tensor, atol: float, rtol: float) -> dict:
    diff = (ref - got).abs()
    max_abs = diff.max().item()
    mean_abs = diff.float().mean().item()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name:>4s}: {status} | max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, atol={atol}, rtol={rtol}")
    if not ok:
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    return {"name": name, "max_abs": max_abs, "mean_abs": mean_abs}


def run_numerical_tests() -> None:
    print("=" * 80)
    print("Numerical equivalence tests")
    print("=" * 80)

    test_cases = [
        (2, 16, 256, torch.float32, 1e-5, 1e-5),
        (2, 16, 257, torch.float32, 1e-5, 1e-5),  # mask / non-power-of-two D test
        (2, 16, 512, torch.float16, 2e-2, 2e-2),
    ]

    for B, T, D, dtype, atol, rtol in test_cases:
        print(f"\nCase: B={B}, T={T}, D={D}, dtype={dtype}")
        inputs = make_inputs(B, T, D, dtype=dtype, seed=123)
        ref = phase2_merge_ref_torch(*inputs)
        got = phase2_merge_fused_triton(*inputs)
        torch.cuda.synchronize()
        assert_close_report("out", ref[0], got[0], atol=atol, rtol=rtol)
        assert_close_report("m", ref[1], got[1], atol=atol, rtol=rtol)
        assert_close_report("l", ref[2], got[2], atol=atol, rtol=rtol)

    print("\nAll numerical tests passed.\n")


def benchmark_cuda_events(fn: Callable[[], object], warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "median_ms": stats.median(times_ms),
        "p95_ms": sorted(times_ms)[max(0, int(0.95 * len(times_ms)) - 1)],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": stats.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def bench_one(B: int, T: int, D: int, dtype: torch.dtype, use_fused: bool, warmup: int, runs: int) -> dict:
    inputs = make_inputs(B, T, D, dtype=dtype, seed=999 + B + T + D)

    if use_fused:
        # Compile once outside measured region.
        phase2_merge_fused_triton(*inputs)
        torch.cuda.synchronize()
        fn = lambda: phase2_merge_fused_triton(*inputs)
        label = "phase2_fused_triton"
    else:
        fn = lambda: phase2_merge_ref_torch(*inputs)
        label = "phase2_ref_torch"

    result = benchmark_cuda_events(fn, warmup=warmup, runs=runs)
    result.update({
        "kernel": label,
        "B": B,
        "T": T,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup,
        "runs": runs,
    })
    return result


def run_benchmarks(mode: str, warmup: int, runs: int, out_csv: Path) -> list[dict]:
    print("=" * 80)
    print(f"Benchmarks: mode={mode}, warmup={warmup}, runs={runs}")
    print("=" * 80)

    if mode == "full":
        configs = [(B, T, D) for B in [1, 4] for T in [1, 16, 128, 512] for D in [512, 1024, 2048, 4096]]
    elif mode == "quick":
        configs = [(1, T, D) for T in [1, 16, 128] for D in [512, 1024, 2048]]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dtype = torch.float16
    rows: list[dict] = []

    for B, T, D in configs:
        print(f"\nB={B}, T={T}, D={D}, dtype=fp16")
        try:
            ref = bench_one(B, T, D, dtype=dtype, use_fused=False, warmup=warmup, runs=runs)
            fused = bench_one(B, T, D, dtype=dtype, use_fused=True, warmup=warmup, runs=runs)
            speedup = ref["median_ms"] / fused["median_ms"] if fused["median_ms"] > 0 else float("nan")
            ref["speedup_vs_ref"] = ""
            fused["speedup_vs_ref"] = speedup
            rows.extend([ref, fused])
            print(f"  ref:   {ref['median_ms']:.4f} ms")
            print(f"  fused: {fused['median_ms']:.4f} ms | speedup={speedup:.2f}x")
        except RuntimeError as exc:
            # Usually out-of-memory or Triton compile issue for a specific config.
            print(f"  SKIP: {exc}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = [
            "kernel", "B", "T", "D", "dtype", "device", "warmup", "runs",
            "mean_ms", "median_ms", "p95_ms", "std_ms", "min_ms", "max_ms", "speedup_vs_ref",
        ]
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {out_csv.resolve()}")

    return rows


# ---------------------------------------------------------------------------
# Triton IR / PTX dump helper
# ---------------------------------------------------------------------------

def dump_triton_asm(out_dir: Path, B: int = 1, T: int = 16, D: int = 512, dtype: torch.dtype = torch.float16) -> None:
    """Compile the kernel once and save TTIR/TTGIR/PTX/CUBIN metadata when available."""
    print("=" * 80)
    print("Triton compile artifact dump")
    print("=" * 80)

    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs(B, T, D, dtype=dtype, seed=2026)
    o1, partial, m1, l1, m2 = inputs

    BT = B * T
    block_d = _choose_block_d(D)
    out = torch.empty_like(o1.contiguous().view(-1))
    mout = torch.empty_like(m1.contiguous().view(-1))
    lout = torch.empty_like(l1.contiguous().view(-1))

    # Force compile. Newer Triton versions expose a CompiledKernel with .asm.
    compiled = _phase2_merge_kernel.warmup(
        o1.contiguous().view(-1), partial.contiguous().view(-1),
        m1.contiguous().view(-1), l1.contiguous().view(-1), m2.contiguous().view(-1),
        out, mout, lout,
        BT=BT, D=D, BLOCK_D=block_d,
        num_warps=4,
        grid=(BT,),
    )

    asm = getattr(compiled, "asm", None)
    if not asm:
        print("Compiled kernel did not expose .asm in this Triton version.")
        return

    print(f"Available artifact keys: {list(asm.keys())}")
    for key, value in asm.items():
        if value is None:
            continue
        # Avoid writing binary cubin as text.
        if isinstance(value, bytes):
            path = out_dir / f"phase2_kernel.{key}.bin"
            path.write_bytes(value)
            print(f"  wrote binary {key}: {path}")
            continue
        path = out_dir / f"phase2_kernel.{key}"
        text = str(value)
        path.write_text(text)
        print(f"  wrote {key}: {path}")

        if key.lower() in {"ptx", "ttir", "ttgir", "llir"}:
            print(f"\n--- {key} preview ---")
            print("\n".join(text.splitlines()[:80]))
            print("--- end preview ---\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-ready Triton Phase-2 checker (Jupyter-safe argparse)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run quick Colab-friendly benchmark")
    group.add_argument("--full", action="store_true", help="Run full benchmark sweep")
    parser.add_argument("--no-test", action="store_true", help="Skip numerical equivalence tests")
    parser.add_argument("--dump-asm", action="store_true", help="Dump Triton TTIR/TTGIR/PTX artifacts")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=None, help="Measured iterations")
    parser.add_argument("--out", type=str, default="results/triton_phase2_colab.csv", help="Output CSV path")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    print_environment()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. In Colab, select Runtime > Change runtime type > GPU, then rerun."
        )
    if not HAS_TRITON:
        raise SystemExit(
            "Triton import failed. In Colab, run: !pip install -q triton\n"
            f"Original import error: {TRITON_IMPORT_ERROR}"
        )

    if not args.no_test:
        run_numerical_tests()

    mode = "full" if args.full else "quick"
    warmup = args.warmup if args.warmup is not None else (20 if mode == "full" else 10)
    runs = args.runs if args.runs is not None else (100 if mode == "full" else 30)
    run_benchmarks(mode=mode, warmup=warmup, runs=runs, out_csv=Path(args.out))

    if args.dump_asm:
        dump_triton_asm(Path("results/triton_asm"))

    print("\nDone.")


if __name__ == "__main__":
    main()
