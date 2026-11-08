from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import fbgemm_gpu  # noqa: F401
from benchmark_model import _load_model
from data.reco_dataset import get_reco_dataset
from indexing.candidate_index import CandidateIndex
from indexing.mips_top_k import MIPSBruteForceTopK
from modeling.sequential.features import movielens_seq_features_from_row
from rrbench.industry_data import rank_interleaved_union


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieve(
    *,
    model_id: str,
    checkpoint: Path,
    dataset: Any,
    split: torch.utils.data.Dataset,
    batch_size: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = _load_model(model_id, checkpoint, dataset, torch.float32).eval()
    item_ids = torch.tensor(
        dataset.all_item_ids, device="cuda", dtype=torch.int64
    ).unsqueeze(0)
    with torch.inference_mode():
        item_embeddings = F.normalize(
            model.get_item_embeddings(item_ids),
            p=2,
            dim=-1,
            eps=1e-6,
        )
    index = CandidateIndex(ids=item_ids, embeddings=item_embeddings)
    search = MIPSBruteForceTopK(
        item_embeddings=item_embeddings,
        item_ids=item_ids,
    )
    all_ids = []
    all_targets = []
    with torch.inference_mode():
        for row in DataLoader(split, batch_size=batch_size, shuffle=False):
            features, targets, _ = movielens_seq_features_from_row(
                row,
                torch.device("cuda"),
                max_output_length=11,
            )
            input_embeddings = model.get_item_embeddings(features.past_ids)
            query = model.encode(
                past_lengths=features.past_lengths,
                past_ids=features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=features.past_payloads,
            )
            ids, _, _ = index.get_top_k_outputs(
                query_embeddings=query,
                k=k,
                top_k_module=search,
                invalid_ids=features.past_ids,
                return_embeddings=False,
            )
            all_ids.append(ids.cpu().numpy().astype(np.int32))
            all_targets.append(targets.squeeze(1).cpu().numpy().astype(np.int32))
    del search, index, model
    torch.cuda.empty_cache()
    return np.concatenate(all_ids), np.concatenate(all_targets)


def _coverage(candidates: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        str(cap): float(
            np.any(candidates[:, :cap] == targets[:, None], axis=1).mean()
        )
        for cap in (50, 100, 200, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec-checkpoint", type=Path, required=True)
    parser.add_argument("--hstu-checkpoint", type=Path, required=True)
    parser.add_argument("--hstu-large-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retriever-k", type=int, default=500)
    parser.add_argument("--pool-cap", type=int, default=500)
    parser.add_argument("--maximum-users", type=int)
    args = parser.parse_args()
    if args.retriever_k < 500 or args.pool_cap < 500:
        raise ValueError("This protocol reports candidate coverage through rank 500.")
    if args.pool_cap > args.retriever_k:
        raise ValueError("The fixed pool cap cannot exceed retriever depth.")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = get_reco_dataset(
        "ml-1m", max_sequence_length=200, chronological=True
    )
    checkpoints = {
        "sasrec": args.sasrec_checkpoint,
        "hstu": args.hstu_checkpoint,
        "hstu-large": args.hstu_large_checkpoint,
    }
    results: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    validation_split = dataset.train_dataset
    test_split = dataset.eval_dataset
    if args.maximum_users is not None:
        user_count = min(args.maximum_users, len(validation_split), len(test_split))
        validation_split = Subset(validation_split, range(user_count))
        test_split = Subset(test_split, range(user_count))
    for model_id, checkpoint in checkpoints.items():
        validation_ids, validation_targets = _retrieve(
            model_id=model_id,
            checkpoint=checkpoint,
            dataset=dataset,
            split=validation_split,
            batch_size=args.batch_size,
            k=args.retriever_k,
        )
        test_ids, test_targets = _retrieve(
            model_id=model_id,
            checkpoint=checkpoint,
            dataset=dataset,
            split=test_split,
            batch_size=args.batch_size,
            k=args.retriever_k,
        )
        results[model_id] = {
            "validation": validation_ids,
            "test": test_ids,
        }
        targets.setdefault("validation", validation_targets)
        targets.setdefault("test", test_targets)
        if not np.array_equal(targets["validation"], validation_targets):
            raise RuntimeError("Validation targets changed across retrievers.")
        if not np.array_equal(targets["test"], test_targets):
            raise RuntimeError("Test targets changed across retrievers.")

    validation_union_diagnostic = rank_interleaved_union(
        [results[name]["validation"] for name in checkpoints],
        args.pool_cap,
    )
    test_union_diagnostic = rank_interleaved_union(
        [results[name]["test"] for name in checkpoints],
        args.pool_cap,
    )
    validation_pool = results["hstu-large"]["validation"][:, : args.pool_cap]
    test_pool = results["hstu-large"]["test"][:, : args.pool_cap]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        validation_targets=targets["validation"],
        test_targets=targets["test"],
        validation_candidates=validation_pool,
        test_candidates=test_pool,
        validation_rank_interleaved_union_diagnostic=validation_union_diagnostic,
        test_rank_interleaved_union_diagnostic=test_union_diagnostic,
        **{
            f"{split}_{model_id.replace('-', '_')}": arrays[split]
            for model_id, arrays in results.items()
            for split in ("validation", "test")
        },
    )
    receipt = {
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "source": {
            model_id: {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
            for model_id, checkpoint in checkpoints.items()
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "split_semantics": {
            "validation": "second-last event",
            "test": "last event",
        },
        "search": {
            "type": "exact full-catalog normalized inner product",
            "precision": "FP32 parameters with TF32 CUDA matrix multiplication enabled",
            "catalog_rows": len(dataset.all_item_ids),
            "retriever_k": args.retriever_k,
            "pool_cap": args.pool_cap,
            "fixed_pool": "hstu-large epoch100 exact retrieval order",
            "diagnostic_union": "rank-interleaved, ordered sasrec/hstu/hstu-large within each rank",
            "history_exclusion": True,
            "target_injection": False,
            "users": len(targets["test"]),
        },
        "coverage": {
            split: {
                **{
                    model_id: _coverage(arrays[split], targets[split])
                    for model_id, arrays in results.items()
                },
                "fixed_pool_hstu_large": _coverage(
                    validation_pool if split == "validation" else test_pool,
                    targets[split],
                ),
                "rank_interleaved_union_diagnostic": _coverage(
                    validation_union_diagnostic
                    if split == "validation"
                    else test_union_diagnostic,
                    targets[split],
                ),
            }
            for split in ("validation", "test")
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(args.receipt)


if __name__ == "__main__":
    main()
