"""
Decode Benchmark: measure end-to-end autoregressive generation latency.

Compares B1 (naive), B3 (two-phase), and B4 (fused) implementations
across various prompt/generation length configurations.

Output: results/decode.csv
"""

from __future__ import annotations
import csv
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from mlx_impl.attnres_mlx import (
    depth_attention_naive,
    depth_attention_two_phase,
    rms_norm,
)


def sync_eval(*xs):
    mx.eval(*xs)


class SimpleBlockAttnResModel:
    """
    Minimal Block AttnRes model for decode benchmarking.
    No real attention/MLP — isolates the depth-attention overhead.

    This measures the overhead of the depth-attention mechanism itself,
    not the full transformer computation.
    """

    def __init__(self, dim, num_layers, block_size, mode="naive"):
        self.dim = dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.layers_per_block = block_size // 2
        self.mode = mode

        # Pseudo-queries and key weights (2 per layer: attn + mlp)
        self.attn_queries = [mx.zeros((dim,)) for _ in range(num_layers)]
        self.mlp_queries = [mx.zeros((dim,)) for _ in range(num_layers)]
        self.key_weights = [mx.ones((dim,)) for _ in range(num_layers)]

    def forward_naive(self, x):
        """B1: Naive depth attention at every sublayer."""
        B, T, D = x.shape
        blocks = [x]
        partial_block = x

        for i in range(self.num_layers):
            # Pre-attention depth attention
            if partial_block is None:
                sources = blocks
            else:
                sources = blocks + [partial_block]
            stacked = mx.stack(sources, axis=0)
            h = depth_attention_naive(stacked, self.attn_queries[i], self.key_weights[i])

            # Block boundary
            layer_number = i + 1
            if layer_number % self.layers_per_block == 0:
                blocks = blocks + [partial_block]
                partial_block = None

            # Simulate sublayer output (identity for benchmarking)
            attn_out = h * 0.01
            if partial_block is not None:
                partial_block = partial_block + attn_out
            else:
                partial_block = attn_out

            # Pre-MLP depth attention
            if partial_block is None:
                sources = blocks
            else:
                sources = blocks + [partial_block]
            stacked = mx.stack(sources, axis=0)
            h = depth_attention_naive(stacked, self.mlp_queries[i], self.key_weights[i])

            mlp_out = h * 0.01
            partial_block = partial_block + mlp_out

        # Final aggregation
        final_sources = blocks + [partial_block] if partial_block is not None else blocks
        stacked = mx.stack(final_sources, axis=0)
        return depth_attention_naive(stacked, mx.zeros((self.dim,)), mx.ones((self.dim,)))

    def forward_two_phase(self, x):
        """B3/B4: Two-phase depth attention."""
        B, T, D = x.shape
        blocks = [x]
        partial_block = x

        for i in range(self.num_layers):
            # Pre-attention: two-phase
            h = depth_attention_two_phase(
                blocks, partial_block, self.attn_queries[i], self.key_weights[i])

            layer_number = i + 1
            if layer_number % self.layers_per_block == 0:
                blocks = blocks + [partial_block]
                partial_block = None

            attn_out = h * 0.01
            if partial_block is not None:
                partial_block = partial_block + attn_out
            else:
                partial_block = attn_out

            # Pre-MLP: two-phase
            h = depth_attention_two_phase(
                blocks, partial_block, self.mlp_queries[i], self.key_weights[i])

            mlp_out = h * 0.01
            partial_block = partial_block + mlp_out

        final_sources = blocks + [partial_block] if partial_block is not None else blocks
        stacked = mx.stack(final_sources, axis=0)
        return depth_attention_naive(stacked, mx.zeros((self.dim,)), mx.ones((self.dim,)))

    def forward(self, x):
        if self.mode == "naive":
            return self.forward_naive(x)
        else:
            return self.forward_two_phase(x)


def run_decode_bench(model, B, T, gen_len, warmup=10, runs=30):
    """Simulate autoregressive decoding (depth-attention overhead only)."""
    x = mx.random.normal((B, T, model.dim))
    sync_eval(x)

    # Warmup
    for _ in range(warmup):
        out = model.forward(x)
        sync_eval(out)

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for step in range(gen_len):
            out = model.forward(x)
            sync_eval(out)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_median_ms": stats.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "ms_per_token": sum(latencies) / len(latencies) / gen_len,
        "tokens_per_sec": gen_len / (sum(latencies) / len(latencies) / 1000.0),
    }


def main():
    print("=" * 60)
    print("AttnRes Decode Benchmark — Apple Silicon")
    print("=" * 60)

    results = []

    configs = [
        {"dim": 512, "num_layers": 8, "block_size": 8},
        {"dim": 768, "num_layers": 12, "block_size": 8},
    ]

    for cfg in configs:
        for mode in ["naive", "two_phase"]:
            impl_name = "B1_naive" if mode == "naive" else "B3_two_phase"
            model = SimpleBlockAttnResModel(
                cfg["dim"], cfg["num_layers"], cfg["block_size"], mode=mode)

            for prompt_len in [16, 64]:
                for gen_len in [16, 64]:
                    label = (f"{impl_name} D={cfg['dim']} L={cfg['num_layers']} "
                             f"P={prompt_len} G={gen_len}")
                    print(f"  {label}", end=" ")
                    try:
                        r = run_decode_bench(model, B=1, T=prompt_len, gen_len=gen_len,
                                             warmup=5, runs=10)
                        r.update({
                            "impl": impl_name,
                            "D": cfg["dim"],
                            "num_layers": cfg["num_layers"],
                            "prompt_len": prompt_len,
                            "gen_len": gen_len,
                        })
                        results.append(r)
                        print(f"→ {r['latency_median_ms']:.1f} ms, "
                              f"{r['tokens_per_sec']:.1f} tok/s")
                    except Exception as e:
                        print(f"SKIP: {e}")

    # Write CSV
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "decode.csv"

    if results:
        fieldnames = ["impl", "D", "num_layers", "prompt_len", "gen_len",
                      "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
                      "ms_per_token", "tokens_per_sec"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
