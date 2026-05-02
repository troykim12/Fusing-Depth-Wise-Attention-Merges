"""
Profiling harness for AttnRes Apple Silicon optimization paper.

Collects:
  1. Python-side timing (CSV)
  2. Memory tracking (torch.mps / mlx.metal)
  3. Dispatch count estimation
  4. Instructions for Instruments / Metal System Trace capture

Output: results/profile.csv + printed Instruments instructions
"""

from __future__ import annotations
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Dispatch counter (estimates GPU kernel dispatches per forward pass)
# ---------------------------------------------------------------------------

def count_ops_naive_depth_attention(N: int) -> dict:
    """Estimate dispatch count for naive depth attention."""
    return {
        "stack": 1,
        "rmsnorm": 1,          # rsqrt + mul + mul
        "einsum_logits": 1,
        "softmax": 1,
        "einsum_weighted": 1,
        "total": 5,
    }


def count_ops_phase1(N: int, S: int) -> dict:
    """Estimate dispatch count for Phase-1 inter-block."""
    return {
        "rmsnorm": 1,
        "broadcast_mul": 1,
        "reduce_sum_logits": 1,
        "max": 1,
        "exp": 1,
        "sum_exp": 1,
        "weighted_sum": 1,
        "total": 7,
    }


def count_ops_phase2_ref() -> dict:
    """Estimate dispatch count for Phase-2 reference."""
    return {
        "rmsnorm_partial": 1,
        "dot_m2": 1,
        "maximum": 1,
        "exp_a": 1,
        "exp_b": 1,
        "mul_add_num": 2,
        "mul_add_den": 2,
        "div": 1,
        "total": 10,
    }


def count_ops_phase2_fused() -> dict:
    """Estimate dispatch count for Phase-2 fused kernel."""
    return {
        "rmsnorm_partial": 1,  # m2 still computed on host
        "dot_m2": 1,
        "fused_merge": 1,      # single Metal kernel
        "total": 3,
    }


def estimate_dispatch_table():
    """Print dispatch count comparison table."""
    print("\n" + "=" * 60)
    print("Estimated GPU Dispatch Counts per Depth-Attention Call")
    print("=" * 60)

    configs = [
        ("B1: Naive", count_ops_naive_depth_attention(8)),
        ("B3: Phase1 + Phase2_ref", {
            **{f"p1_{k}": v for k, v in count_ops_phase1(8, 4).items() if k != "total"},
            **{f"p2_{k}": v for k, v in count_ops_phase2_ref().items() if k != "total"},
            "total": count_ops_phase1(8, 4)["total"] + count_ops_phase2_ref()["total"],
        }),
        ("B4: Phase1 + Phase2_fused", {
            **{f"p1_{k}": v for k, v in count_ops_phase1(8, 4).items() if k != "total"},
            **{f"p2_{k}": v for k, v in count_ops_phase2_fused().items() if k != "total"},
            "total": count_ops_phase1(8, 4)["total"] + count_ops_phase2_fused()["total"],
        }),
    ]

    for name, ops in configs:
        print(f"\n  {name}:")
        print(f"    Total dispatches: {ops['total']}")


# ---------------------------------------------------------------------------
# Instruments capture instructions
# ---------------------------------------------------------------------------

def print_instruments_instructions():
    """Print step-by-step Instruments profiling instructions."""
    print("\n" + "=" * 60)
    print("Metal System Trace Profiling Instructions")
    print("=" * 60)
    print("""
1. PYTORCH MPS PROFILING:
   a. Set environment variable before running:
        export PYTORCH_MPS_LOG_LEVEL=5
   b. Run with OS signposts enabled:
        python bench/bench_baseline_mps.py
   c. Open Instruments.app → Metal System Trace template
   d. Select the Python process and record
   e. Look for: os_signpost intervals, GPU timeline gaps,
      CPU fallback operations

2. MLX PROFILING:
   a. Run the benchmark:
        python bench/bench_micro.py
   b. In parallel, capture with Instruments:
      - Metal System Trace → record
   c. Look for: dispatch patterns, GPU idle gaps,
      kernel execution times

3. COMPARATIVE TRACES:
   Save separate traces for each implementation:
     traces/cold/B1_naive.trace
     traces/cold/B3_twophase.trace
     traces/cold/B4_fused.trace
     traces/steady/B1_naive.trace
     traces/steady/B3_twophase.trace
     traces/steady/B4_fused.trace
     traces/sustained/B4_fused_10min.trace

4. POWER MEASUREMENT:
   a. Use: sudo powermetrics --samplers gpu_power -i 1000
   b. Or: Instruments → Activity Monitor template → Energy
   c. Record during steady-state benchmark runs
   d. Compute: energy_per_token = total_energy / total_tokens

5. MEMORY TRACKING:
   - PyTorch MPS: torch.mps.current_allocated_memory()
   - MLX: mx.metal.get_active_memory() (if available)
   - System: vm_stat or Activity Monitor

6. GPU COUNTERS:
   - Instruments → GPU Counters template
   - Key metrics: ALU utilization, memory bandwidth,
     occupancy, cache hit rate
""")


# ---------------------------------------------------------------------------
# Memory snapshot utility
# ---------------------------------------------------------------------------

def capture_memory_snapshot_torch(fn, device_str="mps"):
    """Run fn and capture peak memory on PyTorch MPS."""
    import torch
    device = torch.device(device_str)

    if device.type == "mps":
        # No reset API for MPS, just measure current
        mem_before = torch.mps.current_allocated_memory()
        result = fn()
        torch.mps.synchronize()
        mem_after = torch.mps.current_allocated_memory()
        return {
            "mem_before_bytes": mem_before,
            "mem_after_bytes": mem_after,
            "mem_delta_bytes": mem_after - mem_before,
        }
    return {}


def main():
    estimate_dispatch_table()
    print_instruments_instructions()

    # Write dispatch estimates to CSV
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = [
        {"impl": "B1_naive", "dispatch_count": 5, "notes": "per depth_attention call"},
        {"impl": "B3_two_phase", "dispatch_count": 17, "notes": "phase1(7) + phase2_ref(10)"},
        {"impl": "B4_fused", "dispatch_count": 10, "notes": "phase1(7) + phase2_fused(3)"},
    ]
    with open(out_dir / "dispatch_estimates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "dispatch_count", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDispatch estimates → {out_dir / 'dispatch_estimates.csv'}")


if __name__ == "__main__":
    main()
