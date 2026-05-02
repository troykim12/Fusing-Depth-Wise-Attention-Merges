"""
Deep Microbenchmark: measure individual AttnRes kernels on Apple Silicon.
Focuses on Extreme Context Lengths and Peak Memory Usage to highlight Fused Kernel advantages.
"""

from __future__ import annotations
import csv
import os
import statistics as stats
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from mlx_impl.attnres_mlx import depth_attention_naive, phase1_interblock, phase2_online_merge_ref
from mlx_impl.phase2_fused import Phase2Runner


def sync_eval(*xs):
    """Force MLX lazy evaluation."""
    mx.eval(*xs)


def timed_run(fn, warmup=20, runs=50): # Runs 줄임 (큰 텐서 연산 시간이 길어짐)
    """Benchmark a function with warmup, returns timing and memory statistics."""
    # Warmup
    for _ in range(warmup):
        y = fn()
        if isinstance(y, tuple):
            sync_eval(*y)
        else:
            sync_eval(y)

    # MLX 메탈 메모리 피크 초기화 (이 기능이 지원되는 버전에 한함)
    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        pass

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, tuple):
            sync_eval(*y)
        else:
            sync_eval(y)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    # VRAM 사용량 추적
    try:
        peak_mem_mb = mx.metal.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        peak_mem_mb = 0.0

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": stats.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": stats.pstdev(latencies),
        "peak_mem_mb": peak_mem_mb,
    }


def bench_depth_attention_naive(B, T, D, N, dtype=mx.float16):
    sources = mx.random.normal((N, B, T, D)).astype(dtype)
    q = mx.random.normal((D,)).astype(dtype)
    w = mx.ones((D,), dtype=dtype)
    sync_eval(sources, q, w)
    return timed_run(lambda: depth_attention_naive(sources, q, w))


def bench_phase1(B, T, D, N, S=4, dtype=mx.float16):
    block_reps = mx.random.normal((N, B, T, D)).astype(dtype)
    queries = mx.random.normal((S, D)).astype(dtype)
    w = mx.ones((D,), dtype=dtype)
    sync_eval(block_reps, queries, w)
    return timed_run(lambda: phase1_interblock(block_reps, queries, w))


def bench_phase2(B, T, D, fused=False, dtype=mx.float16):
    o1 = mx.random.normal((B, T, D)).astype(dtype)
    partial = mx.random.normal((B, T, D)).astype(dtype)
    m1 = mx.random.normal((B, T)).astype(dtype)
    l1 = mx.abs(mx.random.normal((B, T)).astype(dtype)) + 1.0
    q = mx.random.normal((D,)).astype(dtype)
    w = mx.ones((D,), dtype=dtype)
    sync_eval(o1, partial, m1, l1, q, w)

    runner = Phase2Runner(use_fused=fused)
    return timed_run(lambda: runner.run(o1, m1, l1, partial, q, w))


def main():
    print("=" * 70)
    print("Deep AttnRes Microbenchmark — Extreme Scale on Apple Silicon")
    print("=" * 70)

    results = []

    # 파라미터 스케일업: 메모리 대역폭 한계까지 밀어붙입니다.
    B_values = [1, 8]             # High-throughput serving 모사
    T_values = [256, 1024, 4096]  # 긴 컨텍스트 윈도우 (8192는 OOM 위험이 있어 4096까지만)
    D_values = [1024, 4096]       # 대형 모델의 Hidden Dim
    N_values = [8, 16]

    total_configs = (
        len(B_values) * len(T_values) * len(D_values) * len(N_values) * 2
        + len(B_values) * len(T_values) * len(D_values) * 2
    )
    done = 0

    for B in B_values:
        for T_seq in T_values:
            for D in D_values:
                for N in N_values:
                    # 1. Naive depth attention
                    done += 1
                    print(f"[{done}/{total_configs}] naive B={B} T={T_seq} D={D} N={N}", end=" ")
                    try:
                        r = bench_depth_attention_naive(B, T_seq, D, N)
                        r.update({"impl": "B1_naive", "kernel": "depth_attention_naive",
                                  "B": B, "T": T_seq, "D": D, "N": N})
                        results.append(r)
                        print(f"→ {r['median_ms']:.2f}ms | Mem: {r['peak_mem_mb']:.1f}MB")
                    except Exception as e:
                        print(f"    OOM/SKIP: {e}")

                    # 2. Phase 1
                    done += 1
                    S = max(1, 8 // 2)
                    print(f"[{done}/{total_configs}] phase1 B={B} T={T_seq} D={D} N={N}", end=" ")
                    try:
                        r = bench_phase1(B, T_seq, D, N, S=S)
                        r.update({"impl": "B3_two_phase", "kernel": "phase1_interblock",
                                  "B": B, "T": T_seq, "D": D, "N": N})
                        results.append(r)
                        print(f"→ {r['median_ms']:.2f}ms | Mem: {r['peak_mem_mb']:.1f}MB")
                    except Exception as e:
                        print(f"    OOM/SKIP: {e}")

                # 3. Phase 2 ref vs fused
                for fused in [False, True]:
                    done += 1
                    label = "phase2_fused" if fused else "phase2_ref"
                    impl = "B4_fused" if fused else "B3_two_phase"
                    print(f"[{done}/{total_configs}] {label} B={B} T={T_seq} D={D}", end=" ")
                    try:
                        r = bench_phase2(B, T_seq, D, fused=fused)
                        r.update({"impl": impl, "kernel": label,
                                  "B": B, "T": T_seq, "D": D, "N": None})
                        results.append(r)
                        print(f"→ {r['median_ms']:.2f}ms | Mem: {r['peak_mem_mb']:.1f}MB")
                    except Exception as e:
                        print(f"    OOM/SKIP: {e}")

    # Write CSV
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "microbench_deep.csv"

    if results:
        fieldnames = ["impl", "kernel", "B", "T", "D", "N",
                      "mean_ms", "median_ms", "p95_ms", "std_ms", "min_ms", "max_ms", "peak_mem_mb"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {out_path}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
