from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import fbgemm_gpu  # noqa: F401
from cpu_smoke import build_model
from data.reco_dataset import get_reco_dataset
from indexing.candidate_index import CandidateIndex
from indexing.mips_top_k import MIPSBruteForceTopK
from modeling.sequential.autoregressive_losses import LocalNegativesSampler, SampledSoftmaxLoss
from modeling.sequential.features import movielens_seq_features_from_row


def _slice_row(row: dict[str, Any], size: int) -> dict[str, Any]:
    return {
        key: value[:size] if isinstance(value, torch.Tensor) else value
        for key, value in row.items()
    }


def _load_model(model_id: str, checkpoint: Path, dataset, dtype: torch.dtype) -> torch.nn.Module:
    model = build_model(model_id, dataset)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("module."): value
        for key, value in payload["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device="cuda", dtype=dtype)


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def _measure(
    function: Callable[[], Any],
    warmup: int,
    samples: int,
) -> tuple[list[float], dict[str, int]]:
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
    free_mib, total_mib = (
        value // (1024 * 1024) for value in torch.cuda.mem_get_info()
    )
    memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() // (1024 * 1024),
        "sampled_free_mib": free_mib,
        "total_mib": total_mib,
    }
    return durations, memory


def _build_search(
    model: torch.nn.Module,
    dataset,
    dtype: torch.dtype,
) -> tuple[CandidateIndex, MIPSBruteForceTopK]:
    item_ids = torch.tensor(dataset.all_item_ids, device="cuda", dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        item_embeddings = model.get_item_embeddings(item_ids).to(dtype)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=-1, eps=1e-6)
    index = CandidateIndex(ids=item_ids, embeddings=item_embeddings)
    top_k = MIPSBruteForceTopK(item_embeddings=item_embeddings, item_ids=item_ids)
    return index, top_k


def _inference_mode(
    model_id: str,
    checkpoint: Path,
    dataset,
    eval_row: dict[str, Any],
    precision: str,
    batch_sizes: list[int],
    warmup: int,
    samples: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    if precision == "bf16":
        dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif precision == "fp32-tf32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif precision == "fp32-no-tf32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raise ValueError(f"Unknown precision {precision}")

    model = _load_model(model_id, checkpoint, dataset, dtype).eval()
    index, top_k = _build_search(model, dataset, dtype)
    result: dict[str, Any] = {
        "precision": precision,
        "dtype": str(dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "batches": {},
    }
    top_ids: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for batch_size in batch_sizes:
            row = _slice_row(eval_row, batch_size)
            features, _, _ = movielens_seq_features_from_row(
                row,
                torch.device("cuda"),
                max_output_length=11,
            )
            input_embeddings = model.get_item_embeddings(features.past_ids).to(dtype)

            def encode():
                return model.encode(
                    past_lengths=features.past_lengths,
                    past_ids=features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=features.past_payloads,
                )

            query = encode()

            def search():
                return index.get_top_k_outputs(
                    query_embeddings=query,
                    k=200,
                    top_k_module=top_k,
                    invalid_ids=features.past_ids,
                    return_embeddings=False,
                )

            def end_to_end():
                encoded = encode()
                return index.get_top_k_outputs(
                    query_embeddings=encoded,
                    k=200,
                    top_k_module=top_k,
                    invalid_ids=features.past_ids,
                    return_embeddings=False,
                )

            encode_samples, encode_memory = _measure(encode, warmup, samples)
            search_samples, search_memory = _measure(search, warmup, samples)
            end_samples, end_memory = _measure(end_to_end, warmup, samples)
            ids, _, _ = end_to_end()
            top_ids[batch_size] = ids.detach().cpu()
            result["batches"][str(batch_size)] = {
                "sequence_length": int(features.past_ids.shape[1]),
                "catalog_items": len(dataset.all_item_ids),
                "returned_items": 200,
                "encode": {
                    **_summary(encode_samples),
                    "raw_ms": encode_samples,
                    "memory": encode_memory,
                },
                "exact_search": {
                    **_summary(search_samples),
                    "raw_ms": search_samples,
                    "memory": search_memory,
                },
                "end_to_end": {
                    **_summary(end_samples),
                    "raw_ms": end_samples,
                    "memory": end_memory,
                    "queries_per_second": batch_size * 1000.0
                    / statistics.fmean(end_samples),
                },
            }
    del index, top_k, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, top_ids


def _training_benchmark(
    model_id: str,
    checkpoint: Path,
    dataset,
    train_row: dict[str, Any],
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = _load_model(model_id, checkpoint, dataset, torch.float32).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.98),
        weight_decay=0,
    )
    sampler = LocalNegativesSampler(
        num_items=dataset.max_item_id,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=dataset.all_item_ids,
        l2_norm=True,
        l2_norm_eps=1e-6,
    ).to("cuda")
    loss_fn = SampledSoftmaxLoss(
        num_to_sample=128,
        softmax_temperature=0.05,
        model=model,
    ).to("cuda")
    features, target_ids, _ = movielens_seq_features_from_row(
        train_row,
        torch.device("cuda"),
        max_output_length=11,
    )

    def step() -> float:
        model.zero_grad(set_to_none=True)
        features.past_ids.scatter_(1, features.past_lengths.view(-1, 1), target_ids)
        input_embeddings = model.get_item_embeddings(features.past_ids)
        outputs = model(
            past_lengths=features.past_lengths,
            past_ids=features.past_ids,
            past_embeddings=input_embeddings,
            past_payloads=features.past_payloads,
        )
        supervision_ids = features.past_ids
        mask = supervision_ids[:, 1:] != 0
        sampler._item_emb = model._embedding_module._item_emb
        loss = loss_fn(
            lengths=features.past_lengths,
            output_embeddings=outputs[:, :-1, :],
            supervision_ids=supervision_ids[:, 1:],
            supervision_embeddings=input_embeddings[:, 1:, :],
            supervision_weights=mask.float(),
            negatives_sampler=sampler,
        )
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    durations, memory = _measure(step, warmup, samples)
    final_loss = step()
    del model, optimizer, sampler, loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **_summary(durations),
        "raw_ms": durations,
        "batch_size": int(features.past_ids.shape[0]),
        "sequence_length": int(features.past_ids.shape[1]),
        "examples_per_second": features.past_ids.shape[0] * 1000.0
        / statistics.fmean(durations),
        "final_loss": final_loss,
        "memory": memory,
    }


def _overlap(reference: torch.Tensor, candidate: torch.Tensor, k: int) -> float:
    values = []
    for ref_row, candidate_row in zip(reference[:, :k], candidate[:, :k]):
        values.append(
            len(set(ref_row.tolist()).intersection(candidate_row.tolist())) / k
        )
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sasrec", "hstu", "hstu-large"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--training-warmup", type=int, default=3)
    parser.add_argument("--training-samples", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = get_reco_dataset("ml-1m", max_sequence_length=200, chronological=True)
    eval_row = next(
        iter(DataLoader(dataset.eval_dataset, batch_size=128, shuffle=False))
    )
    train_row = next(
        iter(DataLoader(dataset.train_dataset, batch_size=128, shuffle=False))
    )
    modes = {}
    mode_ids = {}
    for precision in ("fp32-no-tf32", "fp32-tf32", "bf16"):
        modes[precision], mode_ids[precision] = _inference_mode(
            model_id=args.model,
            checkpoint=args.checkpoint,
            dataset=dataset,
            eval_row=eval_row,
            precision=precision,
            batch_sizes=[1, 16, 128],
            warmup=args.warmup,
            samples=args.samples,
        )

    reference = mode_ids["fp32-no-tf32"]
    fidelity = {}
    for precision, results in mode_ids.items():
        fidelity[precision] = {
            str(batch_size): {
                "top10_set_overlap": _overlap(reference[batch_size], ids, 10),
                "top200_set_overlap": _overlap(reference[batch_size], ids, 200),
            }
            for batch_size, ids in results.items()
        }

    output = {
        "model": args.model,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "warmup": args.warmup,
        "samples": args.samples,
        "inference": modes,
        "precision_fidelity": fidelity,
        "training_step": _training_benchmark(
            model_id=args.model,
            checkpoint=args.checkpoint,
            dataset=dataset,
            train_row=train_row,
            warmup=args.training_warmup,
            samples=args.training_samples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
