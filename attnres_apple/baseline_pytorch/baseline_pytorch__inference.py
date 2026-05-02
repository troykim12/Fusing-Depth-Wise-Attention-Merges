"""
Inference Script for Attention Residuals (v3) — MPS-patched
=============================================================
Patched for Apple Silicon MPS support:
  - get_torch_device() selects CUDA > MPS > CPU
  - sync_device() handles MPS synchronization
  - benchmark_generation() properly syncs on MPS
  - Memory tracking via torch.mps.current_allocated_memory()
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from attn_res import (
    BlockAttnResTransformer,
    FullAttnResTransformer,
    BaselineTransformer,
)
from device_utils import get_torch_device, sync_device, get_peak_memory


# ---------------------------------------------------------------------------
# Sampling Utilities
# ---------------------------------------------------------------------------

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        indices_to_remove = sorted_mask.scatter(
            dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0, greedy=False):
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Generation Engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model, input_ids, max_new_tokens=128, temperature=1.0,
    top_k=0, top_p=1.0, greedy=False, eos_token_id=None,
    repetition_penalty=1.0,
):
    """Naive autoregressive generation (full recomputation per step)."""
    model.eval()
    device = input_ids.device
    B = input_ids.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        result = model(input_ids)
        next_logits = result["logits"][:, -1, :]

        if repetition_penalty != 1.0:
            for b in range(B):
                prev_tokens = input_ids[b].unique()
                penalty_logits = next_logits[b, prev_tokens]
                next_logits[b, prev_tokens] = torch.where(
                    penalty_logits > 0,
                    penalty_logits / repetition_penalty,
                    penalty_logits * repetition_penalty,
                )

        next_token = sample_next_token(
            next_logits, temperature=temperature,
            top_k=top_k, top_p=top_p, greedy=greedy,
        )

        if eos_token_id is not None:
            next_token = next_token.masked_fill(finished.unsqueeze(-1), eos_token_id)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

    return input_ids


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_model(args, device):
    ff_dim = int(args.dim * args.ff_ratio)
    if args.model == "attnres":
        model = BlockAttnResTransformer(
            args.vocab_size, args.dim, args.num_layers, args.num_heads,
            ff_dim, args.seq_len, args.block_size, dropout=0.0)
    elif args.model == "full_attnres":
        model = FullAttnResTransformer(
            args.vocab_size, args.dim, args.num_layers, args.num_heads,
            ff_dim, args.seq_len, dropout=0.0)
    elif args.model == "baseline":
        model = BaselineTransformer(
            args.vocab_size, args.dim, args.num_layers, args.num_heads,
            ff_dim, args.seq_len, dropout=0.0)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

    model = model.to(device)
    model.eval()
    num_params = model.num_parameters()
    print(f"Model: {args.model} | Params (excl. emb): {num_params:,}")
    return model


# ---------------------------------------------------------------------------
# Benchmark — MPS-aware
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_generation(
    model, device, vocab_size,
    prompt_len=64, gen_len=128, batch_size=1,
    num_warmup=3, num_runs=10,
):
    """Benchmark with proper MPS/CUDA sync."""
    model.eval()
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = generate(model, prompt, max_new_tokens=gen_len, greedy=True)
        sync_device(device)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        sync_device(device)
        t0 = time.perf_counter()
        output = generate(model, prompt, max_new_tokens=gen_len, greedy=True)
        sync_device(device)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    total_new_tokens = gen_len * batch_size
    latencies_ms = [l * 1000 for l in latencies]
    tokens_per_sec = [total_new_tokens / l for l in latencies]

    stats = {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "num_runs": num_runs,
        "latency_mean_ms": sum(latencies_ms) / len(latencies_ms),
        "latency_std_ms": (sum((l - sum(latencies_ms)/len(latencies_ms))**2
                               for l in latencies_ms) / len(latencies_ms)) ** 0.5,
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "tokens_per_sec_mean": sum(tokens_per_sec) / len(tokens_per_sec),
        "ms_per_token_mean": sum(latencies_ms) / len(latencies_ms) / gen_len,
        "peak_memory_bytes": get_peak_memory(device),
    }
    return stats


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

def inference_smoke_test():
    print("=" * 60)
    print("INFERENCE SMOKE TEST (MPS-patched)")
    print("=" * 60)

    device = get_torch_device()
    print(f"Device: {device}")
    V, D, L, H, FF, T = 500, 128, 4, 4, 64, 64

    for name, Model, kwargs in [
        ("Baseline", BaselineTransformer, {}),
        ("Block AttnRes", BlockAttnResTransformer, {"block_size": 4}),
        ("Full AttnRes", FullAttnResTransformer, {}),
    ]:
        print(f"\n--- {name} ---")
        model = Model(V, D, L, H, FF, max_seq_len=T, **kwargs).to(device)
        model.eval()

        prompt = torch.randint(0, V, (2, 8), device=device)
        output = generate(model, prompt, max_new_tokens=16, greedy=True)
        assert output.shape == (2, 24), f"Expected (2, 24), got {output.shape}"
        print(f"  Greedy: {prompt.shape} → {output.shape} ✓")

        output = generate(model, prompt, max_new_tokens=16,
                          temperature=0.8, top_k=50, top_p=0.9)
        assert output.shape == (2, 24)
        print(f"  Sampling: {output.shape} ✓")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Quick benchmark
    print(f"\n--- Benchmark (tiny model) ---")
    model = BaselineTransformer(V, D, L, H, FF, max_seq_len=T).to(device)
    model.eval()
    stats = benchmark_generation(
        model, device, V, prompt_len=8, gen_len=16,
        batch_size=1, num_warmup=2, num_runs=5)
    print(f"  Latency: {stats['latency_mean_ms']:.1f} ± {stats['latency_std_ms']:.1f} ms")
    print(f"  Tokens/sec: {stats['tokens_per_sec_mean']:.1f}")
    print(f"  ms/token: {stats['ms_per_token_mean']:.2f}")
    print(f"  Peak memory: {stats['peak_memory_bytes']:,} bytes")

    print("\n✓ All inference smoke tests passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Attention Residuals Inference (MPS-patched)")
    p.add_argument("--model", default="attnres",
                   choices=["baseline", "attnres", "full_attnres"])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--ff_ratio", type=float, default=0.45)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--bench_prompt_len", type=int, default=64)
    p.add_argument("--bench_gen_len", type=int, default=128)
    p.add_argument("--bench_batch_size", type=int, default=1)
    p.add_argument("--bench_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.smoke_test:
        inference_smoke_test()
        return

    device = get_torch_device()
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    model = build_model(args, device)

    if args.benchmark:
        print(f"\nBenchmarking {args.model}...")
        stats = benchmark_generation(
            model, device, args.vocab_size,
            prompt_len=args.bench_prompt_len,
            gen_len=args.bench_gen_len,
            batch_size=args.bench_batch_size,
            num_runs=args.bench_runs,
        )
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        return

    print("Use --smoke_test or --benchmark")


if __name__ == "__main__":
    main()
