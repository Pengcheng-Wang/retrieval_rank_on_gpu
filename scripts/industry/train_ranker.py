from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from rrbench.industry_data import dense_features, training_maximum_gap
from rrbench.industry_models import make_industry_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[
            max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
        ],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "raw_ms": samples,
    }


def _batch_indices(
    size: int, batch_size: int, *, shuffle_seed: int | None
) -> Iterator[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def _to_device_batch(
    *,
    data: Any,
    split: str,
    indices: np.ndarray,
    candidates: np.ndarray,
    device: torch.device,
    maximum_gap: int,
    candidate_ranks: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    histories = np.asarray(data[f"{split}_histories"][indices])
    ratings = np.asarray(data[f"{split}_ratings"][indices])
    timestamps = np.asarray(data[f"{split}_timestamps"][indices])
    lengths = np.asarray(data[f"{split}_lengths"][indices])
    users = np.asarray(data[f"{split}_users"][indices])
    request_times = np.asarray(data[f"{split}_request_times"][indices])
    popularity = np.asarray(data["popularity"])
    dense = dense_features(
        histories=histories,
        ratings=ratings,
        lengths=lengths,
        candidates=candidates,
        request_times=request_times,
        timestamps=timestamps,
        popularity=popularity,
        max_history=histories.shape[1],
        maximum_popularity=int(popularity.max()),
        maximum_gap=maximum_gap,
    )
    if candidate_ranks is None:
        candidate_ranks = np.broadcast_to(
            np.arange(1, candidates.shape[1] + 1, dtype=np.int32),
            candidates.shape,
        ).copy()
    return {
        "histories": torch.as_tensor(histories, dtype=torch.long, device=device),
        "ratings": torch.as_tensor(ratings, dtype=torch.long, device=device),
        "timestamps": torch.as_tensor(timestamps, dtype=torch.long, device=device),
        "lengths": torch.as_tensor(lengths, dtype=torch.long, device=device),
        "users": torch.as_tensor(users, dtype=torch.long, device=device),
        "request_times": torch.as_tensor(
            request_times, dtype=torch.long, device=device
        ),
        "candidates": torch.as_tensor(
            candidates, dtype=torch.long, device=device
        ),
        "dense": torch.as_tensor(dense, dtype=torch.float32, device=device),
        "candidate_ranks": torch.as_tensor(
            candidate_ranks, dtype=torch.float32, device=device
        ),
    }


def _candidate_metrics(
    *,
    ranks: np.ndarray,
    present: np.ndarray,
    candidate_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queries": int(len(ranks)),
        "candidate_count": candidate_count,
        "target_present": int(present.sum()),
        "target_coverage": float(present.mean()),
    }
    for k in (10, 50, 200):
        hits = present & (ranks <= k)
        discounts = np.where(hits, 1.0 / np.log2(ranks + 1.0), 0.0)
        result[f"unconditional_hr@{k}"] = float(hits.mean())
        result[f"unconditional_ndcg@{k}"] = float(discounts.mean())
        if present.any():
            result[f"conditional_hr@{k}"] = float(hits[present].mean())
            result[f"conditional_ndcg@{k}"] = float(
                discounts[present].mean()
            )
        else:
            result[f"conditional_hr@{k}"] = None
            result[f"conditional_ndcg@{k}"] = None
    return result


def _evaluate(
    *,
    model: torch.nn.Module,
    data: Any,
    candidates_file: Any,
    data_split: str,
    candidate_split: str,
    device: torch.device,
    batch_size: int,
    maximum_gap: int,
    indices: np.ndarray | None = None,
    include_observations: bool = False,
) -> dict[str, Any]:
    candidates_all = np.asarray(
        candidates_file[f"{candidate_split}_candidates"]
    )[:, :200]
    targets_all = np.asarray(data[f"{data_split}_targets"])
    candidate_targets = np.asarray(
        candidates_file[f"{candidate_split}_targets"]
    )
    if not np.array_equal(targets_all[: len(candidate_targets)], candidate_targets):
        raise RuntimeError(
            f"{data_split} targets differ from the {candidate_split} candidate pool."
        )
    if indices is None:
        indices = np.arange(len(candidate_targets), dtype=np.int64)
    if len(indices) and int(indices.max()) >= len(candidate_targets):
        raise ValueError(
            f"{candidate_split} evaluation indices exceed the candidate file."
        )
    ranks = np.full(len(indices), np.inf, dtype=np.float64)
    present = np.zeros(len(indices), dtype=bool)
    model.eval()
    offset = 0
    with torch.inference_mode():
        for batch_indices in _batch_indices(
            len(indices), batch_size, shuffle_seed=None
        ):
            source_indices = indices[batch_indices]
            candidates = candidates_all[source_indices]
            batch = _to_device_batch(
                data=data,
                split=data_split,
                indices=source_indices,
                candidates=candidates,
                device=device,
                maximum_gap=maximum_gap,
            )
            scores = model(**batch)
            targets = torch.as_tensor(
                targets_all[source_indices], device=device
            )
            equal = batch["candidates"] == targets[:, None]
            batch_present = equal.any(dim=1)
            target_columns = equal.float().argmax(dim=1)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            batch_ranks = (
                (order == target_columns[:, None]).float().argmax(dim=1) + 1
            )
            size = len(source_indices)
            present[offset : offset + size] = batch_present.cpu().numpy()
            ranks[offset : offset + size] = batch_ranks.cpu().numpy()
            offset += size
    result = _candidate_metrics(
        ranks=ranks,
        present=present,
        candidate_count=200,
    )
    if include_observations:
        result["observations"] = {
            "query_indices": indices.astype(np.int32).tolist(),
            "target_present": present.astype(np.uint8).tolist(),
            "target_rank": np.where(present, ranks, 0).astype(np.uint16).tolist(),
        }
    return result


def _training_batch(
    *,
    data: Any,
    indices: np.ndarray,
    device: torch.device,
    maximum_gap: int,
) -> dict[str, torch.Tensor]:
    targets = np.asarray(data["train_targets"][indices])[:, None]
    negatives = np.asarray(data["train_negatives"][indices])
    candidates = np.concatenate((targets, negatives), axis=1)
    candidate_ranks = np.concatenate(
        (
            np.asarray(data["train_target_ranks"][indices])[:, None],
            np.asarray(data["train_negative_ranks"][indices]),
        ),
        axis=1,
    )
    return _to_device_batch(
        data=data,
        split="train",
        indices=indices,
        candidates=candidates,
        device=device,
        maximum_gap=maximum_gap,
        candidate_ranks=candidate_ranks,
    )


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _measure_forward(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    warmup: int,
    samples: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    def run() -> torch.Tensor:
        with torch.inference_mode(), _autocast(device, precision):
            return model(**batch)

    model.eval()
    for _ in range(warmup):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    output = run()
    for _ in range(samples):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
    result = {
        **_summary(durations),
        "precision": precision,
        "batch_size": int(batch["candidates"].shape[0]),
        "candidates_per_query": int(batch["candidates"].shape[1]),
        "history_length": int(batch["histories"].shape[1]),
        "candidate_scores_per_second": (
            batch["candidates"].numel() * 1000.0 / statistics.fmean(durations)
        ),
        "memory_scope": (
            "selected checkpoint model plus fixed inference batch after the "
            "training optimizer and gradients were released"
        ),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
        result["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved() / (1024 * 1024)
        )
    return result, output.float().detach()


def _top_overlap(reference: torch.Tensor, candidate: torch.Tensor, k: int) -> float:
    reference_ids = torch.topk(reference, k=k, dim=1).indices.cpu().numpy()
    candidate_ids = torch.topk(candidate, k=k, dim=1).indices.cpu().numpy()
    return float(
        np.mean(
            [
                len(set(left.tolist()).intersection(right.tolist())) / k
                for left, right in zip(reference_ids, candidate_ids)
            ]
        )
    )


def _benchmark(
    *,
    model: torch.nn.Module,
    data: Any,
    candidates_file: Any,
    device: torch.device,
    maximum_gap: int,
) -> dict[str, Any]:
    inference_indices = np.arange(16, dtype=np.int64)
    inference_candidates = np.asarray(
        candidates_file["test_candidates"][:16, :200]
    )
    inference_batch = _to_device_batch(
        data=data,
        split="test",
        indices=inference_indices,
        candidates=inference_candidates,
        device=device,
        maximum_gap=maximum_gap,
    )
    fp32, reference = _measure_forward(
        model=model,
        batch=inference_batch,
        device=device,
        precision="fp32",
        warmup=10,
        samples=50,
    )
    result: dict[str, Any] = {"inference": {"fp32": fp32}}
    if device.type == "cuda":
        try:
            bf16, bf16_output = _measure_forward(
                model=model,
                batch=inference_batch,
                device=device,
                precision="bf16",
                warmup=10,
                samples=50,
            )
            bf16["top10_set_overlap_with_fp32"] = _top_overlap(
                reference, bf16_output, 10
            )
            result["inference"]["bf16"] = bf16
        except Exception as error:
            result["inference"]["bf16"] = {
                "status": "unsupported",
                "error": f"{type(error).__name__}: {error}",
            }

    training_indices = np.arange(64, dtype=np.int64)
    training_batch = _training_batch(
        data=data,
        indices=training_indices,
        device=device,
        maximum_gap=maximum_gap,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(**training_batch)
        loss = F.cross_entropy(
            logits,
            torch.zeros(logits.shape[0], dtype=torch.long, device=device),
        )
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    model.train()
    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    final_loss = math.nan
    for _ in range(20):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        final_loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000.0)
    training = {
        **_summary(durations),
        "batch_size": 64,
        "candidates_per_context": 33,
        "final_loss": final_loss,
        "candidate_scores_per_second": (
            training_batch["candidates"].numel()
            * 1000.0
            / statistics.fmean(durations)
        ),
        "memory_scope": (
            "selected checkpoint model plus a fresh benchmark optimizer and "
            "fixed training batch"
        ),
    }
    if device.type == "cuda":
        training["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
        training["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved() / (1024 * 1024)
        )
    result["training_step"] = training
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["torchrec-dlrm", "torchrec-dcn", "din", "twin", "transact"],
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=[17, 42], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--training-negatives", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--maximum-train-examples", type=int)
    parser.add_argument("--maximum-eval-users", type=int)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.candidates, allow_pickle=False) as archive:
        candidates_file = {key: archive[key] for key in archive.files}
    with np.load(args.training_negatives, allow_pickle=False) as archive:
        hard_negatives = {key: archive[key] for key in archive.files}
    if not np.array_equal(hard_negatives["targets"], data["train_targets"]):
        raise RuntimeError("Hard-negative targets do not align with training data.")
    if not np.array_equal(hard_negatives["users"], data["train_users"]):
        raise RuntimeError("Hard-negative users do not align with training data.")
    if np.any(hard_negatives["negatives"] == data["train_targets"][:, None]):
        raise RuntimeError("Training targets leaked into hard negatives.")
    data["train_negatives"] = hard_negatives["negatives"]
    data["train_negative_ranks"] = hard_negatives["negative_ranks"]
    data["train_target_ranks"] = hard_negatives["target_ranks"]
    maximum_gap = training_maximum_gap(data)
    users = int(
        max(
            np.asarray(data["train_users"]).max(),
            np.asarray(data["validation_users"]).max(),
            np.asarray(data["test_users"]).max(),
        )
    ) + 1
    items = len(data["popularity"])
    model = make_industry_model(
        args.model,
        users=users,
        items=items,
        device=device,
        seed=args.seed,
        dimension=32,
        maximum_gap_seconds=maximum_gap,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )
    diagnostic_limits = None
    if args.maximum_eval_users is None:
        validation_indices = np.asarray(
            data["test_user_partition_validation_indices"]
        ).astype(np.int64)
        test_indices = np.setdiff1d(
            np.arange(len(data["test_targets"]), dtype=np.int64),
            validation_indices,
            assume_unique=False,
        )
    else:
        evaluation_size = min(
            args.maximum_eval_users,
            len(candidates_file["test_targets"]),
        )
        if evaluation_size < 2:
            raise ValueError("A diagnostic evaluation requires at least two users.")
        validation_size = max(1, evaluation_size // 2)
        validation_indices = np.arange(validation_size, dtype=np.int64)
        test_indices = np.arange(
            validation_size, evaluation_size, dtype=np.int64
        )
    if args.maximum_train_examples is not None or args.maximum_eval_users is not None:
        diagnostic_limits = {
            "maximum_train_examples": args.maximum_train_examples,
            "maximum_eval_users": args.maximum_eval_users,
            "validation_queries": len(validation_indices),
            "test_queries": len(test_indices),
        }
    selection = validation_indices
    initial_validation = _evaluate(
        model=model,
        data=data,
        candidates_file=candidates_file,
        data_split="test",
        candidate_split="test",
        device=device,
        batch_size=args.eval_batch_size,
        maximum_gap=maximum_gap,
        indices=selection,
    )
    epoch_results = [
        {
            "epoch": 0,
            "training_loss": None,
            "selection_validation": initial_validation,
        }
    ]
    best_metric = initial_validation["unconditional_ndcg@10"]
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    train_size = len(data["train_targets"])
    if args.maximum_train_examples is not None:
        train_size = min(train_size, args.maximum_train_examples)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in _batch_indices(
            train_size,
            args.batch_size,
            shuffle_seed=args.seed * 1000 + epoch,
        ):
            batch = _training_batch(
                data=data,
                indices=indices,
                device=device,
                maximum_gap=maximum_gap,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)
            loss = F.cross_entropy(
                logits,
                torch.zeros(
                    logits.shape[0], dtype=torch.long, device=device
                ),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate(
            model=model,
            data=data,
            candidates_file=candidates_file,
            data_split="test",
            candidate_split="test",
            device=device,
            batch_size=args.eval_batch_size,
            maximum_gap=maximum_gap,
            indices=selection,
        )
        metric = validation["unconditional_ndcg@10"]
        epoch_results.append(
            {
                "epoch": epoch,
                "training_loss": statistics.fmean(losses),
                "selection_validation": validation,
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            json.dumps(
                {
                    "model": args.model,
                    "seed": args.seed,
                    "epoch": epoch,
                    "loss": epoch_results[-1]["training_loss"],
                    "selection_ndcg@10": metric,
                    "best_epoch": best_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("No best checkpoint selected.")
    model.load_state_dict(best_state, strict=True)
    validation = _evaluate(
        model=model,
        data=data,
        candidates_file=candidates_file,
        data_split="test",
        candidate_split="test",
        device=device,
        batch_size=args.eval_batch_size,
        maximum_gap=maximum_gap,
        indices=validation_indices,
        include_observations=True,
    )
    test = _evaluate(
        model=model,
        data=data,
        candidates_file=candidates_file,
        data_split="test",
        candidate_split="test",
        device=device,
        batch_size=args.eval_batch_size,
        maximum_gap=maximum_gap,
        indices=test_indices,
        include_observations=True,
    )
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    base_model_class = f"{type(model.base).__module__}.{type(model.base).__name__}"
    upstream_class = getattr(model, "upstream_class", None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    checkpoint_path = args.output_dir / "model.pt"
    result_path = args.output_dir / "result.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "seed": args.seed,
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "data_sha256": _sha256(args.data),
            "candidates_sha256": _sha256(args.candidates),
            "training_negatives_sha256": _sha256(args.training_negatives),
            "protocol_sha256": _sha256(args.protocol),
        },
        checkpoint_path,
    )
    benchmark_result = None
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not args.skip_benchmark:
        benchmark_model = make_industry_model(
            args.model,
            users=users,
            items=items,
            device=device,
            seed=args.seed,
            dimension=32,
            maximum_gap_seconds=maximum_gap,
        )
        benchmark_model.load_state_dict(best_state, strict=True)
        benchmark_result = _benchmark(
            model=benchmark_model,
            data=data,
            candidates_file=candidates_file,
            device=device,
            maximum_gap=maximum_gap,
        )
        del benchmark_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    result = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torchrec": package_version("torchrec"),
            "fbgemm_gpu": package_version("fbgemm-gpu"),
            "triton": package_version("triton"),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
        },
        "model_class": model_class,
        "base_model_class": base_model_class,
        "upstream_class": upstream_class,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "best_epoch": best_epoch,
        "selected_residual_weight": float(best_state["residual_weight"]),
        "selection_metric": "unconditional_ndcg@10",
        "evaluation_split": {
            "validation": (
                "last event for a frozen deterministic 1000-user partition"
                if diagnostic_limits is None
                else f"diagnostic last-event partition ({len(validation_indices)} users)"
            ),
            "test": (
                "last event for the complementary users"
                if diagnostic_limits is None
                else f"diagnostic disjoint last-event partition ({len(test_indices)} users)"
            ),
            "upstream_retriever_training_overlap": False,
        },
        "diagnostic_limits": diagnostic_limits,
        "training_maximum_gap_seconds": maximum_gap,
        "best_selection_metric": best_metric,
        "epochs": epoch_results,
        "validation": validation,
        "test": test,
        "benchmark": benchmark_result,
        "inputs": {
            "data": str(args.data),
            "data_sha256": _sha256(args.data),
            "candidates": str(args.candidates),
            "candidates_sha256": _sha256(args.candidates),
            "training_negatives": str(args.training_negatives),
            "training_negatives_sha256": _sha256(args.training_negatives),
            "protocol": str(args.protocol),
            "protocol_sha256": _sha256(args.protocol),
        },
        "training_negative_source": "exact HSTU-large epoch100 candidate retrieval",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
