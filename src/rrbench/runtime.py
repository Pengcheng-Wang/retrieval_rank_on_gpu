from __future__ import annotations

import importlib.metadata as metadata
import platform
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def configure(seed: int, threads: int) -> None:
    if threads <= 0:
        raise ValueError("CPU thread count must be positive.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; CPU fallback would change the benchmark.")
    return torch.device("cuda", torch.cuda.current_device())


def environment_manifest() -> dict[str, object]:
    packages = {}
    for name in ("torch", "triton", "numpy", "faiss-gpu-cu12", "faiss-cpu", "torchrec", "fbgemm-gpu"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "tf32": False,
        "cpu_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {
            "name": device.name,
            "memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor],
            "multiprocessors": device.multi_processor_count,
        }
    return result


def measure(
    operation: Callable[[], T], warmup: int, iterations: int, cuda: bool
) -> tuple[T, list[float]]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("Nonnegative warmup and positive iterations are required.")
    for _ in range(warmup):
        operation()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    result: T
    for _ in range(iterations):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = operation()
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return result, samples
