from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

import fbgemm_gpu  # noqa: F401
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha


def _measure(function: Callable[[], None], warmup: int, samples: int) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(durations)
    free_mib, total_mib = (
        value // (1024 * 1024) for value in torch.cuda.mem_get_info()
    )
    return {
        "raw_ms": durations,
        "mean_ms": statistics.fmean(durations),
        "p50_ms": statistics.median(durations),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() // (1024 * 1024),
        "sampled_free_mib": free_mib,
        "total_mib": total_mib,
    }


def _inputs(
    batch_size: int,
    sequence_length: int,
    heads: int,
    attention_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = batch_size * sequence_length
    q = torch.empty(
        tokens,
        heads,
        attention_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.01, 0.01)
    k = torch.empty_like(q).uniform_(-0.01, 0.01)
    v = torch.empty(
        tokens,
        heads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).uniform_(-0.01, 0.01)
    offsets = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int64)
        * sequence_length
    )
    return q, k, v, offsets


def _forward(
    kernel: HammerKernel,
    sequence_length: int,
    attention_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return hstu_mha(
        max_seq_len=sequence_length,
        alpha=1.0 / attention_dim,
        q=q,
        k=k,
        v=v,
        seq_offsets=offsets,
        causal=True,
        dropout_pr=0.0,
        training=True,
        kernel=kernel,
    )


def benchmark_shape(
    batch_size: int,
    sequence_length: int,
    heads: int,
    attention_dim: int,
    value_dim: int,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    torch.manual_seed(1001)
    q, k, v, offsets = _inputs(
        batch_size,
        sequence_length,
        heads,
        attention_dim,
        value_dim,
    )
    with torch.inference_mode():
        reference = _forward(
            HammerKernel.PYTORCH,
            sequence_length,
            attention_dim,
            q,
            k,
            v,
            offsets,
        )
        candidate = _forward(
            HammerKernel.TRITON,
            sequence_length,
            attention_dim,
            q,
            k,
            v,
            offsets,
        )
    difference = candidate.float() - reference.float()
    fidelity = {
        "maximum_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "relative_l2_error": float(
            difference.norm() / reference.float().norm().clamp_min(1e-12)
        ),
    }

    forward = {}
    for name, kernel in (
        ("pytorch", HammerKernel.PYTORCH),
        ("triton", HammerKernel.TRITON),
    ):
        forward[name] = _measure(
            lambda selected=kernel: _forward(
                selected,
                sequence_length,
                attention_dim,
                q,
                k,
                v,
                offsets,
            ),
            warmup,
            samples,
        )

    backward = {}
    output_gradient = torch.randn_like(reference)
    for name, kernel in (
        ("pytorch", HammerKernel.PYTORCH),
        ("triton", HammerKernel.TRITON),
    ):
        q_grad = q.detach().requires_grad_()
        k_grad = k.detach().requires_grad_()
        v_grad = v.detach().requires_grad_()

        def forward_backward(selected=kernel):
            q_grad.grad = None
            k_grad.grad = None
            v_grad.grad = None
            output = _forward(
                selected,
                sequence_length,
                attention_dim,
                q_grad,
                k_grad,
                v_grad,
                offsets,
            )
            output.backward(output_gradient)

        backward[name] = _measure(forward_backward, warmup, samples)

    result = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "heads": heads,
        "attention_dim": attention_dim,
        "value_dim": value_dim,
        "tokens": batch_size * sequence_length,
        "dtype": "torch.bfloat16",
        "fidelity": fidelity,
        "forward": forward,
        "forward_backward": backward,
        "forward_speedup_triton_over_pytorch": forward["pytorch"]["mean_ms"]
        / forward["triton"]["mean_ms"],
        "forward_backward_speedup_triton_over_pytorch": backward["pytorch"][
            "mean_ms"
        ]
        / backward["triton"]["mean_ms"],
    }
    del q, k, v, offsets, reference, candidate
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[200, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = [
        benchmark_shape(
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            heads=args.heads,
            attention_dim=args.attention_dim,
            value_dim=args.value_dim,
            warmup=args.warmup,
            samples=args.samples,
        )
        for sequence_length in args.sequence_lengths
    ]
    output = {
        "source_commit": "acc36be02236750cd98d99246f7d7ef1412edd0e",
        "source_archive_sha256": (
            "67c91fe53fb390c793349dc4fd9bf013d1ed9a729745dc80e61492254d4c97db"
        ),
        "interpretation": (
            "Later portable PyTorch/Triton attention contrast; not the paper-era "
            "public model path or H100 CUDA result."
        ),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "warmup": args.warmup,
        "samples": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
