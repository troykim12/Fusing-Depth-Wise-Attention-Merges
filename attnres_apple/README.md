# AttnRes Apple Silicon Optimization

Systems-level optimization of Block Attention Residuals inference on Apple M5 silicon.

## Paper

**Title:** Systems-Level Optimization of Block Attention Residuals on Apple Silicon: Two-Phase Inference with Fused Metal Kernels

**Thesis:** Restructuring naive Block AttnRes from per-layer depth-attention recomputation into a two-phase cached path with a fused Phase-2 Metal kernel reduces dispatch overhead, memory traffic, and intermediate tensor materialization on Apple silicon.

## Project Structure

```
attnres_apple/
├── baseline_pytorch/           # PyTorch reference (MPS-patched)
│   ├── attn_res.py             # Core AttnRes modules
│   ├── device_utils.py         # MPS device selection & sync
│   ├── train.py                # Training loop (if needed)
│   └── inference.py            # Naive generation loop
├── mlx_impl/                   # MLX optimized implementations
│   ├── attnres_mlx.py          # Naive + Phase1 + Phase2_ref
│   ├── kernels.py              # Custom Metal kernel (Phase2 fused)
│   └── phase2_fused.py         # Fused vs ref runner
├── bench/                      # Measurement scripts
│   ├── bench_baseline_mps.py   # PyTorch MPS baseline
│   ├── bench_micro.py          # Kernel-level microbenchmarks
│   ├── bench_decode.py         # Model-level decode benchmark
│   ├── bench_profile.py        # Profiling harness + instructions
│   └── export_figures.py       # Generate paper figures from CSV
├── tests/
│   └── test_numerical_equiv.py # Correctness verification
├── results/                    # CSV outputs (auto-generated)
├── traces/                     # Instruments traces (manual)
│   ├── cold/
│   ├── steady/
│   └── sustained/
└── paper/
    ├── main.tex                # Full paper (NeurIPS format)
    └── figures/                # Generated figures
```

## Execution Order

### Prerequisites
```bash
# On M5 MacBook Air:
pip install torch mlx numpy matplotlib
```

### Step 1: Verify correctness
```bash
python tests/test_numerical_equiv.py
```

### Step 2: PyTorch MPS baseline
```bash
python bench/bench_baseline_mps.py
# Output: results/baseline_mps.csv
```

### Step 3: MLX microbenchmarks
```bash
python bench/bench_micro.py
# Output: results/microbench.csv
```

### Step 4: Decode benchmarks
```bash
python bench/bench_decode.py
# Output: results/decode.csv
```

### Step 5: Profiling (instructions)
```bash
python bench/bench_profile.py
# Prints Metal System Trace instructions + dispatch estimates
```

### Step 6: Generate figures
```bash
python bench/export_figures.py
# Output: paper/figures/*.pdf
```

### Step 7: Compile paper
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Implementations Compared

| ID | Name | Description |
|----|------|-------------|
| B0 | Baseline | Standard Transformer (no AttnRes) |
| B1 | Naive | PyTorch Block AttnRes on MPS — reference |
| B2 | Cached | Cache completed block reps |
| B3 | Two-Phase | Phase 1 (batched) + Phase 2 (ref Python) |
| B4 | Fused | Phase 1 + Phase 2 (custom Metal kernel) |

## Key Design Decisions

1. **Training stays in PyTorch** — correctness baseline is clear
2. **Optimized inference in MLX** — native Apple silicon support, `compile()` for auto-fusion, `mx.fast.metal_kernel` for custom kernels
3. **Phase 2 is the fusion target** — elementwise online-softmax merge between repeated reads/writes; most dispatch reduction per engineering effort
4. **Timing always forces evaluation** — `mx.eval()` for MLX, `torch.mps.synchronize()` for PyTorch

## Hardware Note

The paper **must** report the exact MacBook Air M5 SKU:
- GPU core count (8 or 10)
- Unified memory capacity (16, 24, or 32 GB)
- Unified memory bandwidth (153 GB/s)
