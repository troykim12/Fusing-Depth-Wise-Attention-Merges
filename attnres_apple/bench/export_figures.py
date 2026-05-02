"""
Generate paper figures from benchmark CSV results.

Reads: results/microbench.csv, results/decode.csv
Writes: paper/figures/*.pdf

Requires: matplotlib (install with pip install matplotlib)
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figure generation")


def load_csv(path):
    if not path.exists():
        print(f"  {path} not found — skipping")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_micro_by_D(rows, out_dir):
    """Latency vs hidden dim D for each kernel type."""
    if not HAS_MPL:
        return

    kernels = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("B") == "1" and r.get("T") == "128":
            kernel = r["kernel"]
            D = int(r["D"])
            lat = float(r["median_ms"])
            kernels[kernel][D].append(lat)

    if not kernels:
        print("  No data for micro_by_D figure")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, d_data in sorted(kernels.items()):
        ds = sorted(d_data.keys())
        lats = [sum(d_data[d]) / len(d_data[d]) for d in ds]
        ax.plot(ds, lats, "o-", label=kernel)

    ax.set_xlabel("Hidden dimension D")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Microbenchmark: Latency vs D (B=1, T=128)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out_dir / "micro_by_D.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_decode_comparison(rows, out_dir):
    """Bar chart comparing decode performance across implementations."""
    if not HAS_MPL:
        return

    impls = defaultdict(list)
    for r in rows:
        if r.get("gen_len") == "64" and r.get("prompt_len") == "64":
            impl = r["impl"]
            tps = float(r["tokens_per_sec"])
            impls[impl].append(tps)

    if not impls:
        print("  No data for decode comparison figure")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    names = sorted(impls.keys())
    means = [sum(impls[n]) / len(impls[n]) for n in names]
    ax.bar(names, means, color=["#4C72B0", "#55A868", "#C44E52"][:len(names)])
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Decode Performance (P=64, G=64)")
    ax.grid(True, alpha=0.3, axis="y")

    path = out_dir / "decode_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    micro_rows = load_csv(base / "results" / "microbench.csv")
    decode_rows = load_csv(base / "results" / "decode.csv")

    if micro_rows:
        fig_micro_by_D(micro_rows, out_dir)
    if decode_rows:
        fig_decode_comparison(decode_rows, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
