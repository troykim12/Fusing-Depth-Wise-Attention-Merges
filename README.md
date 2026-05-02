# Fused Depth-Wise Attention Merges

Systems-level optimization of **Block Attention Residuals** inference using a two-phase cached schedule and fused Phase-2 online-softmax merge kernels for **Apple MLX/Metal** and **NVIDIA Triton**.

This repository contains the code, kernels, benchmark scripts, and result files for the project:

> **Fusing Depth-Wise Attention Merges: Systems-Level Optimization of Block Attention Residuals across Apple and NVIDIA Silicon**  
> Donghyun Kim, Stony Brook University

---

## Overview

Attention Residuals replace fixed residual accumulation with learned depth-wise softmax attention.

However, a naive Block AttnRes implementation can introduce unnecessary overhead during inference:

- repeated kernel launches,
- redundant global memory traffic,
- intermediate tensor materialization,
- repeated reconstruction of cached block representations.

This project implements a systems-level optimization for Block AttnRes inference:

1. **Phase 1:** Batched inter-block attention over cached block representations.
2. **Phase 2:** Sequential intra-block online-softmax merge.
3. **Fusion:** Phase-2 merge is fused into a single GPU kernel.

Two hardware/software paths are provided:

- **Apple Silicon:** MLX + custom Metal kernel
- **NVIDIA GPUs:** PyTorch + Triton kernel

---

## Key Results

Representative Phase-2 microbenchmark results at:

```text
B = 1
T = 128
D = 1024
dtype = fp16
```

| Platform | Kernel Framework | Reference Median | Fused Speedup |
|---|---:|---:|---:|
| Apple M5 MacBook Air | MLX + Metal | 0.246 ms | 1.24× |
| NVIDIA T4 | PyTorch + Triton | 0.151 ms | 1.76× |
| NVIDIA L4 | PyTorch + Triton | 0.145 ms | 1.73× |
| NVIDIA A100 | PyTorch + Triton | 0.142 ms | 1.70× |
| NVIDIA H100 | PyTorch + Triton | 0.080 ms | 1.85× |
| NVIDIA RTX PRO 6000 | PyTorch + Triton | 0.047 ms | 2.02× |

The fused kernel reduces intermediate memory traffic and improves arithmetic intensity while preserving numerical equivalence with the reference implementation.

---

## Repository Structure

```text
.
├── baseline_pytorch/
│   ├── attn_res.py
│   ├── baseline_pytorch__device_utils.py
│   └── baseline_pytorch__inference.py
│
├── mlx_impl/
│   ├── attnres_mlx.py
│   ├── kernels.py
│   └── phase2_fused.py
│
├── bench/
│   ├── bench_baseline_mps.py
│   ├── bench_micro.py
│   ├── bench_decode.py
│   ├── bench_profile.py
│   └── export_figures.py
│
├── tests/
│   └── test_numerical_equiv.py
│
├── triton_phase2_colab_v3.py
├── triton_phase2_colab_v3.ipynb
├── mlx_metal_results/
└── triton_results/
```

---

## Installation

### Apple Silicon / MLX

```bash
pip install mlx numpy matplotlib pytest
```

Optional PyTorch baseline:

```bash
pip install torch
```

### NVIDIA / Triton

For CUDA environments such as Google Colab:

```bash
pip install torch triton numpy matplotlib pytest
```

---

## Quick Start

### 1. Run numerical equivalence tests

```bash
python -m pytest tests/test_numerical_equiv.py -v
```

### 2. Run Apple MLX microbenchmarks

```bash
python bench/bench_micro.py
```

### 3. Run Apple decode benchmark

```bash
python bench/bench_decode.py
```

### 4. Run Triton benchmark on NVIDIA GPU

```bash
python triton_phase2_colab_v3.py --quick --dump-asm
```

For a longer benchmark sweep:

```bash
python triton_phase2_colab_v3.py --full
```

---

## Implementations Compared

| ID | Implementation | Description |
|---|---|---|
| B1 | Naive Block AttnRes | PyTorch reference implementation |
| B2 | Cached Block AttnRes | Reuses completed block representations |
| B3 | Two-Phase Block AttnRes | Batched Phase 1 + reference Phase 2 |
| B4 | Fused Phase-2 | Two-phase path with fused online-softmax merge |

---

## Method Summary

The Phase-2 online-softmax merge computes:

```text
m = max(m1, m2)
a = exp(m1 - m)
b = exp(m2 - m)
denom = a * l1 + b
out = (a * o1 + b * partial) / denom
```

The reference implementation materializes intermediate tensors across several operations.

The fused implementation keeps intermediate values in registers and performs the merge in a single custom GPU kernel.

---

## Roofline Interpretation

The optimization is motivated by a memory-traffic reduction rather than a reduction in arithmetic operations.

The reference implementation materializes intermediate tensors, while the fused implementation keeps those values inside registers.

As a result, fusion shifts the Phase-2 merge kernel toward higher arithmetic intensity and improves performance on memory-bound GPU workloads.

---

## Results

Precomputed benchmark outputs are included in:

```text
mlx_metal_results/
triton_results/
```

These directories contain CSV files for Apple MLX/Metal and NVIDIA Triton experiments.

---

## Numerical Equivalence

All optimized implementations are designed to preserve numerical equivalence with the PyTorch reference.

Reported tolerance:

```text
fp32 max absolute error < 1e-5
fp16 max absolute error < 1e-3
```

---

## Limitations

This repository is a research artifact, not a production inference library.

Known limitations:

- Apple MLX and NVIDIA Triton are not identical software stacks.
- Current implementation focuses mainly on the Phase-2 depth-wise merge kernel.
- Full end-to-end Triton model-level benchmarking is left for future work.
- mHC and DenseFormer extensions are analyzed conceptually but not fully implemented here.
- Apple M-series cross-generation benchmarking is left for future work.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{kim2026fuseddepthwiseattentionmerges,
  title  = {Fusing Depth-Wise Attention Merges: Systems-Level Optimization of Block Attention Residuals across Apple and NVIDIA Silicon},
  author = {Donghyun Kim},
  year   = {2026},
  note   = {Research artifact}
}
```

---

## License

MIT License
