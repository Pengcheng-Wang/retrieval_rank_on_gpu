from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rrbench.ann import Partition
from rrbench.io import bind_run_config, sha256, write_json
from rrbench.runtime import configure, require_cuda
from rrbench.semantic_ann import SemanticFaissANN, validate_partition
from rrbench.semantic_queries import validate_query_panel
from rrbench.semantic_rankers import query_representation


def common_candidate_pool(predictions: list[np.ndarray], limit: int) -> np.ndarray:
    if (
        not predictions or limit <= 0 or any(
            values.ndim != 2 or len(values) != len(predictions[0]) or values.dtype != np.int64
            for values in predictions
        )
    ):
        raise ValueError("Common candidate pools require aligned integer retriever outputs and a positive cap.")
    output = np.full((len(predictions[0]), limit), -1, dtype=np.int64)
    for query in range(len(output)):
        found = list(dict.fromkeys(
            int(values[query, rank])
            for rank in range(max(matrix.shape[1] for matrix in predictions))
            for values in predictions
            if rank < values.shape[1] and values[query, rank] >= 0
        ))[:limit]
        output[query, :len(found)] = found
    return output


def mine_candidates(
    item_file: Path, query_file: Path, panel_file: Path, partition_directory: Path,
    output: Path, candidates: int = 1000, probes: int = 256, seed: int = 17,
) -> dict[str, Any]:
    configure(seed, 4)
    device = require_cuda()
    if min(candidates, probes) <= 0 or max(candidates, probes) > 2048:
        raise ValueError("Candidate mining requires valid classicFAISSGPU k/probe limits.")
    items, queries = np.load(item_file, mmap_mode="r"), np.load(query_file, mmap_mode="r")
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    representation = query_representation(query_file)
    query_complete = json.loads((query_file.parent / "completed.json").read_text(encoding="utf-8"))
    if (
        query_complete["panel_sha256"] != sha256(panel_file) or query_complete["query_ids"] != panel["query_ids"]
        or queries.shape != (len(panel["query_ids"]), items.shape[1]) or candidates > len(items)
    ):
        raise ValueError("Candidate mining query vectors/panel IDs/corpus dimension are not aligned.")
    partition_receipt = json.loads((partition_directory / "validated-partition.json").read_text(encoding="utf-8"))
    partition_file = partition_directory / "partition.npz"
    if sha256(partition_file) != partition_receipt["partition_sha256"]:
        raise ValueError("Candidate mining shared IVF partition bytes changed.")
    data = np.load(partition_file)
    partition = Partition(*(data[name] for name in ("centroids", "assignments", "order", "offsets")))
    validate_partition(partition, len(items), items.shape[1], len(partition.centroids))
    if probes > len(partition.centroids):
        raise ValueError("Candidate mining probes exceed the frozen shared centroid count.")
    binding = {
        "items_sha256": sha256(item_file), "queries_sha256": sha256(query_file),
        "panel_sha256": sha256(panel_file), "partition_sha256": sha256(partition_file),
        "candidate_count": candidates, "probes": probes, "seed": seed,
        "backend": "faiss-gpu-fp16_scalar_codes", "representation": representation,
        "positive_injection": False, "retrieval_labels_loaded": False,
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(output / "ids.npy") != complete["ids_sha256"]:
            raise ValueError("Candidate mining output bytes changed.")
        values = np.load(output / "ids.npy", mmap_mode="r")
        if values.shape != (len(queries), candidates) or values.dtype != np.int64:
            raise ValueError("Candidate mining output shape/dtype changed.")
        return complete
    ann = SemanticFaissANN(items, partition, gpu=True, half=True)
    ids = np.lib.format.open_memmap(
        output / "ids.partial.npy", mode="w+", dtype=np.int64, shape=(len(queries), candidates),
    )
    started = time.perf_counter()
    for start in range(0, len(queries), 64):
        query = torch.as_tensor(np.array(queries[start:start + 64], dtype=np.float32), device=device)
        _, found = ann.search(query, candidates, probes)
        values = found.cpu().numpy()
        if ((values < -1) | (values >= len(items))).any():
            raise RuntimeError("Candidate mining produced invalid item IDs.")
        ids[start:start + len(query)] = values
        if start % 4096 == 0:
            ids.flush()
            write_json(output / "progress.json", {"query_rows": start + len(query), "expected_rows": len(queries)})
            print(f"LABEL_INDEPENDENT_CANDIDATES rows={start + len(query)}/{len(queries)}", flush=True)
    ids.flush()
    del ids
    (output / "ids.partial.npy").replace(output / "ids.npy")
    complete = {
        "complete": True, "queries": len(queries), "candidate_count": candidates,
        "catalog_items": len(items), "dimension": items.shape[1], "query_ids": panel["query_ids"],
        "elapsed_seconds": time.perf_counter() - started, "ids_sha256": sha256(output / "ids.npy"),
        "positive_injection": False, "binding": binding,
    }
    write_json(output / "completed.json", complete)
    return complete


def save_common_pool(sources: list[Path], output: Path, limit: int) -> dict[str, Any]:
    if not sources:
        raise ValueError("Common pool requires declared source retrievers.")
    receipts = [
        json.loads((path.parent / "completed.json").read_text(encoding="utf-8")) for path in sources
    ]
    panel_sha = receipts[0]["binding"]["panel_sha256"]
    query_ids = receipts[0]["query_ids"]
    for path, receipt in zip(sources, receipts):
        if (
            not receipt["complete"] or sha256(path) != receipt["ids_sha256"]
            or receipt["binding"]["panel_sha256"] != panel_sha or receipt["query_ids"] != query_ids
        ):
            raise ValueError("Common-pool sources do not use the same measured query panel/order.")
    binding = {
        "source_candidates": [sha256(path) for path in sources], "limit": limit,
        "query_panel_sha256": panel_sha, "query_ids": query_ids,
        "pool_policy": "Rank-interleaved retriever union without labels, positive injection or relevance-based truncation.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(output / "ids.npy") != completed["ids_sha256"]:
            raise ValueError("Common-pool candidate bytes changed.")
        return completed
    pool = common_candidate_pool([np.load(path, mmap_mode="r") for path in sources], limit)
    np.save(output / "ids.npy", pool)
    completed = {
        "complete": True, "queries": len(pool), "candidate_count": limit, "query_ids": query_ids,
        "ids_sha256": sha256(output / "ids.npy"), "binding": {"panel_sha256": panel_sha},
        "positive_injection": False,
    }
    write_json(output / "completed.json", completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-independent real semantic candidates and common pool.")
    commands = parser.add_subparsers(dest="operation", required=True)
    mine = commands.add_parser("mine")
    for name in ("items", "queries", "panel", "partition-directory", "output"):
        mine.add_argument("--" + name, required=True)
    mine.add_argument("--candidate-count", type=int, default=1000)
    mine.add_argument("--probes", type=int, default=256)
    common = commands.add_parser("common")
    common.add_argument("--source", action="append", required=True)
    common.add_argument("--output", required=True)
    common.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.operation == "mine":
        mine_candidates(
            Path(args.items), Path(args.queries), Path(args.panel), Path(args.partition_directory),
            Path(args.output), args.candidate_count, args.probes,
        )
    else:
        save_common_pool([Path(path) for path in args.source], Path(args.output), args.limit)


if __name__ == "__main__":
    main()
