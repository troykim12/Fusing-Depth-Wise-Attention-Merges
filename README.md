# Fused Depth-Wise Attention Merges

Systems-level optimization of Block Attention Residuals inference using a two-phase cached schedule and fused Phase-2 online-softmax merge kernels for Apple MLX/Metal and NVIDIA Triton.

This repository contains the code, kernels, benchmark scripts, and result CSVs used for the project:

**Fusing Depth-Wise Attention Merges: Systems-Level Optimization of Block Attention Residuals across Apple and NVIDIA Silicon**

## Overview

Attention Residuals replace fixed residual accumulation with learned depth-wise softmax attention.  
A naive Block AttnRes implementation can incur unnecessary kernel launches, global memory traffic, and intermediate tensor materialization during inference.

This project implements and benchmarks a systems-level optimization:

1. **Phase 1:** Batched inter-block attention over cached block representations.
2. **Phase 2:** Sequential intra-block online-softmax merge.
3. **Fusion:** Phase-2 merge is fused into a single GPU kernel.

Two implementations are provided:

- **Apple Silicon:** MLX + custom Metal kernel
- **NVIDIA GPUs:** PyTorch + Triton kernel

## Key Results

Representative Phase-2 microbenchmark results at `B=1, T=128, D=1024, fp16`:

| Platform | Kernel Framework | Reference Median | Fused Speedup |
|---|---:|---:|---:|
| Apple M5 MacBook Air | MLX + Metal | 0.246 ms | 1.24× |
| NVIDIA T4 | PyTorch + Triton | 0.151 ms | 1.76× |
| NVIDIA L4 | PyTorch + Triton | 0.145 ms | 1.73× |
| NVIDIA A100 | PyTorch + Triton | 0.142 ms | 1.70× |
| NVIDIA H100 | PyTorch + Triton | 0.080 ms | 1.85× |
| NVIDIA RTX PRO 6000 | PyTorch + Triton | 0.047 ms | 2.02× |

The fused kernel reduces intermediate memory traffic and improves arithmetic intensity while preserving numerical equivalence with the reference implementation.

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
├── mlx:metal_results/
└── triton_results/
