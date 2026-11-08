from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import fbgemm_gpu  # noqa: F401
from benchmark_model import _build_search, _load_model
from data.reco_dataset import get_reco_dataset
from modeling.sequential.features import SequentialFeatures
from rrbench.industry_data import hard_negative_ids_and_ranks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chronological(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    for row, length in enumerate(lengths):
        size = int(length)
        output[row, :size] = values[row, :size][::-1]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--search-k", type=int, default=200)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--maximum-examples", type=int)
    args = parser.parse_args()
    if args.search_k <= args.negatives:
        raise ValueError("search-k must exceed the negative count.")

    with np.load(args.data, allow_pickle=False) as archive:
        histories = archive["train_histories"]
        ratings = archive["train_ratings"]
        timestamps = archive["train_timestamps"]
        lengths = archive["train_lengths"]
        targets = archive["train_targets"]
        users = archive["train_users"]
        request_times = archive["train_request_times"]
    size = len(targets)
    if args.maximum_examples is not None:
        size = min(size, args.maximum_examples)
        histories = histories[:size]
        ratings = ratings[:size]
        timestamps = timestamps[:size]
        lengths = lengths[:size]
        targets = targets[:size]
        users = users[:size]
        request_times = request_times[:size]

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = get_reco_dataset(
        "ml-1m", max_sequence_length=200, chronological=True
    )
    model = _load_model(
        "hstu-large", args.checkpoint, dataset, torch.float32
    ).eval()
    index, search = _build_search(model, dataset, torch.float32)
    all_retrieved = []
    with torch.inference_mode():
        for start in range(0, size, args.batch_size):
            stop = min(start + args.batch_size, size)
            batch_lengths_np = lengths[start:stop].astype(np.int64)
            chronological_ids = _chronological(
                histories[start:stop], batch_lengths_np
            )
            chronological_ratings = _chronological(
                ratings[start:stop], batch_lengths_np
            )
            chronological_times = _chronological(
                timestamps[start:stop], batch_lengths_np
            )
            batch = stop - start
            past_ids = torch.zeros(
                (batch, 211), dtype=torch.long, device="cuda"
            )
            past_ratings = torch.zeros_like(past_ids)
            past_times = torch.zeros_like(past_ids)
            past_ids[:, :200] = torch.as_tensor(
                chronological_ids, dtype=torch.long, device="cuda"
            )
            past_ratings[:, :200] = torch.as_tensor(
                chronological_ratings, dtype=torch.long, device="cuda"
            )
            past_times[:, :200] = torch.as_tensor(
                chronological_times, dtype=torch.long, device="cuda"
            )
            batch_lengths = torch.as_tensor(
                batch_lengths_np, dtype=torch.long, device="cuda"
            )
            past_times.scatter_(
                1,
                batch_lengths[:, None],
                torch.as_tensor(
                    request_times[start:stop],
                    dtype=torch.long,
                    device="cuda",
                )[:, None],
            )
            features = SequentialFeatures(
                past_lengths=batch_lengths,
                past_ids=past_ids,
                past_embeddings=None,
                past_payloads={
                    "timestamps": past_times,
                    "ratings": past_ratings,
                },
            )
            embeddings = model.get_item_embeddings(features.past_ids)
            query = model.encode(
                past_lengths=features.past_lengths,
                past_ids=features.past_ids,
                past_embeddings=embeddings,
                past_payloads=features.past_payloads,
            )
            ids, _, _ = index.get_top_k_outputs(
                query_embeddings=query,
                k=args.search_k,
                top_k_module=search,
                invalid_ids=features.past_ids,
                return_embeddings=False,
            )
            all_retrieved.append(ids.cpu().numpy().astype(np.int32))

    retrieved = np.concatenate(all_retrieved)
    negatives, negative_ranks, target_ranks = hard_negative_ids_and_ranks(
        retrieved, targets, args.negatives
    )
    if np.any(negatives == targets[:, None]):
        raise RuntimeError("Training targets leaked into hard negatives.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        negatives=negatives,
        negative_ranks=negative_ranks,
        target_ranks=target_ranks,
        targets=targets.astype(np.int32),
        users=users.astype(np.int32),
    )
    target_recall = float(np.any(retrieved == targets[:, None], axis=1).mean())
    receipt = {
        "output_sha256": _sha256(args.output),
        "data_sha256": _sha256(args.data),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "examples": size,
        "negatives_per_example": args.negatives,
        "search_k": args.search_k,
        "target_recall_at_search_k": target_recall,
        "missing_target_rank": args.search_k + 1,
        "search": "exact HSTU-large epoch100 full-catalog normalized inner product",
        "history_exclusion": True,
        "target_removed_from_negatives": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(args.receipt)


if __name__ == "__main__":
    main()
