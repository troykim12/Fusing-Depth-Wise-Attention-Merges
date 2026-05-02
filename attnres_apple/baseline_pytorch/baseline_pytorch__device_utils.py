"""
Device utilities for Apple Silicon (MPS) support.
Provides unified device selection, synchronization, and memory tracking
across CUDA, MPS, and CPU backends.
"""

import time
from contextlib import contextmanager
from typing import Optional

import torch


def get_torch_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize GPU to ensure all queued operations are complete."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_peak_memory(device: torch.device) -> int:
    """Return current GPU memory allocated in bytes."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def reset_peak_memory(device: torch.device) -> None:
    """Reset peak memory tracking."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@contextmanager
def timed_region(device: torch.device, name: str = ""):
    """Context manager that times a GPU region with proper sync."""
    sync_device(device)
    t0 = time.perf_counter()
    yield
    sync_device(device)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    if name:
        print(f"  [{name}] {elapsed_ms:.3f} ms")


def device_info_str(device: torch.device) -> str:
    """Return a human-readable string describing the device."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"CUDA: {props.name}, {props.total_mem / 1e9:.1f} GB"
    elif device.type == "mps":
        return "Apple MPS (Metal Performance Shaders)"
    return "CPU"
