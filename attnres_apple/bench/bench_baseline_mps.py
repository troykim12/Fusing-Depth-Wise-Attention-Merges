"""
PyTorch MPS Baseline Benchmark.

Measures naive depth_attention and full model inference on MPS backend.
Serves as the B1 (naive Block AttnRes) performance reference.

Output: results/baseline_mps.csv
"""

from __future__ import annotations
import csv
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline_pytorch"))

import torch
from attn_res import (
    depth_attention, RMSNorm,
    BlockAttnResTransformer, BaselineTransformer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline_pytorch"))
from device_utils import get_torch_device, sync_device, get_peak_memory


def timed_run_torch(fn, device, warmup=20, runs=100):
    """Benchmark with proper device sync."""
    for _ in range(warmup):
        _ = fn()
        sync_device(device)

    latencies = []
    for _ in range(runs):
        sync_device(device)
        t0 = time.perf_counter()
        _ = fn()
        sync_device(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": stats.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": stats.pstdev(latencies),
    }


def bench_depth_attention_torch(B, T, D, N, device):
    """Benchmark PyTorch depth_attention on given device."""
    sources = torch.randn(N, B, T, D, device=device, dtype=torch.float16)
    q = torch.zeros(D, device=device, dtype=torch.float16)
    norm = RMSNorm(D).to(device).half()

    def fn():
        return depth_attention(sources, q, norm)

    return timed_run_torch(fn, device)


@torch.no_grad()
def bench_model_forward_torch(model_cls, dim, num_layers, num_heads, block_size,
                               B, T, vocab_size, device, **extra_kwargs):
    """Benchmark full model forward pass."""
    ff_dim = int(dim * 0.45)
    kwargs = dict(vocab_size=vocab_size, dim=dim, num_layers=num_layers,
                  num_heads=num_heads, ff_dim=ff_dim, max_seq_len=T, dropout=0.0)
    kwargs.update(extra_kwargs)
    model = model_cls(**kwargs).to(device).half().eval()

    input_ids = torch.randint(0, vocab_size, (B, T), device=device)

    def fn():
        return model(input_ids)

    r = timed_run_torch(fn, device, warmup=10, runs=50)
    r["num_params"] = model.num_parameters()
    r["peak_mem"] = get_peak_memory(device)
    return r


def main():
    device = get_torch_device()
    print(f"Device: {device}")
    results = []

    # Kernel-level benchmarks
    print("\n--- Depth Attention Kernel Benchmarks ---")
    for B in [1, 4]:
        for T in [1, 16, 128, 512]:
            for D in [512, 1024]:
                for N in [4, 8]:
                    print(f"  B={B} T={T} D={D} N={N}", end=" ")
                    try:
                        r = bench_depth_attention_torch(B, T, D, N, device)
                        r.update({"impl": "B1_pytorch_mps", "kernel": "depth_attention",
                                  "B": B, "T": T, "D": D, "N": N})
                        results.append(r)
                        print(f"→ {r['median_ms']:.3f} ms")
                    except Exception as e:
                        print(f"SKIP: {e}")

    # Model-level benchmarks
    print("\n--- Model Forward Benchmarks ---")
    V = 32000
    for name, cls, extra in [
        ("baseline", BaselineTransformer, {}),
        ("block_attnres", BlockAttnResTransformer, {"block_size": 8}),
    ]:
        for dim, layers, heads in [(256, 4, 4), (512, 8, 8)]:
            T = 128
            B = 1
            print(f"  {name} dim={dim} L={layers}", end=" ")
            try:
                r = bench_model_forward_torch(
                    cls, dim, layers, heads, 8, B, T, V, device, **extra)
                r.update({"impl": f"B1_{name}", "kernel": "model_forward",
                          "B": B, "T": T, "D": dim, "N": layers})
                results.append(r)
                print(f"→ {r['median_ms']:.3f} ms, {r.get('num_params', 0):,} params")
            except Exception as e:
                print(f"SKIP: {e}")

    # Write CSV
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "baseline_mps.csv"

    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
