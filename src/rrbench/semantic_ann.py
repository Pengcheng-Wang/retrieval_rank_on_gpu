from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rrbench.ann import Partition, SilverANN, train_partition
from rrbench.exact import exact_topk
from rrbench.io import append_jsonl, bind_run_config, sha256, write_json
from rrbench.metrics import latency_summary, recall_at_k
from rrbench.runtime import configure, environment_manifest, measure, require_cuda
from rrbench.semantic_ir import judged_metrics

BACKEND_STORAGE = {
    "faiss-cpu-fp32": (False, "fp32"), "faiss-gpu-fp32": (True, "fp32"),
    "faiss-gpu-fp16": (True, "fp16"),
    "silver-fp32": (True, "fp32"), "silver-fp16": (True, "fp16"),
    "silver-int8-global": (True, "int8-global"), "silver-int8-rowwise": (True, "int8-rowwise"),
}
SILVER_DIMENSIONS = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280)


def validate_backends(backends: list[str]) -> None:
    if not backends or len(backends) != len(set(backends)) or set(backends) - set(BACKEND_STORAGE):
        raise ValueError("ANN backend names must be unique entries in the explicit device/precision allowlist.")


def validate_partition(partition: Partition, rows: int, dimension: int, clusters: int) -> None:
    if (
        partition.centroids.shape != (clusters, dimension) or partition.centroids.dtype != np.float32
        or not np.isfinite(partition.centroids).all()
        or partition.assignments.dtype != np.int64 or partition.assignments.shape != (rows,)
        or (partition.assignments < 0).any() or (partition.assignments >= clusters).any()
        or partition.order.dtype != np.int64 or partition.order.shape != (rows,)
        or not np.array_equal(np.sort(partition.order), np.arange(rows))
        or partition.offsets.dtype != np.int64 or partition.offsets.shape != (clusters + 1,)
        or partition.offsets[0] != 0 or partition.offsets[-1] != rows or (np.diff(partition.offsets) < 0).any()
    ):
        raise ValueError("Shared partition shapes, types, permutation or list ranges are invalid.")
    counts = np.bincount(partition.assignments, minlength=clusters)
    if not np.array_equal(counts, np.diff(partition.offsets)):
        raise ValueError("Shared partition assignment counts differ from list boundaries.")
    if not np.array_equal(partition.assignments[partition.order], np.repeat(np.arange(clusters), counts)):
        raise ValueError("Shared partition ordered assignments differ between FAISS and Silver list membership.")


def validate_oracle(directory: Path, queries: int, items: int, dimension: int, k: int) -> dict[str, Any]:
    completed = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    if (
        not completed["complete"] or completed["queries"] != queries
        or completed["catalog_items"] != items or completed["dimension"] != dimension
    ):
        raise ValueError("Exact oracle receipt differs from the full catalog/query scope.")
    for filename, dtype in (("ids.npy", np.int64), ("scores.npy", np.float32)):
        path = directory / filename
        if sha256(path) != completed[filename + "_sha256"]:
            raise ValueError("Exact oracle payload checksum changed.")
        values = np.load(path, mmap_mode="r")
        if values.shape != (queries, k) or values.dtype != dtype:
            raise ValueError("Exact oracle payload shape or dtype changed.")
        if filename == "ids.npy" and ((values < 0) | (values >= items)).any():
            raise ValueError("Exact oracle contains out-of-catalog IDs.")
        if filename == "scores.npy" and not np.isfinite(values).all():
            raise ValueError("Exact oracle contains invalid scores.")
    return completed


def validate_journal(
    records: list[dict[str, Any]], output: Path, expected: set[str], queries: int, items: int,
) -> set[str]:
    verified = set()
    for row in records:
        condition = row["condition"]
        if condition not in expected or condition in verified:
            raise ValueError("ANN journal has an undeclared or duplicated condition.")
        if row["status"] == "ok":
            path = output / (condition + "-ids.npy")
            if sha256(path) != row["prediction_ids_sha256"]:
                raise ValueError("ANN candidate prediction checksum changed.")
            predicted = np.load(path, mmap_mode="r")
            k = row.get("k", predicted.shape[1])
            if predicted.dtype != np.int64 or predicted.shape != (queries, k):
                raise ValueError("ANN candidate prediction shape or dtype changed.")
            if ((predicted < -1) | (predicted >= items)).any():
                raise ValueError("ANN candidate predictions contain invalid external IDs.")
            for candidates in predicted:
                valid = candidates[candidates >= 0]
                if len(valid) != len(np.unique(valid)):
                    raise ValueError("ANN candidate predictions contain duplicate external IDs.")
        elif row["status"] not in ("unsupported", "unsupported_memory") or not row.get("reason"):
            raise ValueError("ANN condition is not a successful or explicitly unsupported outcome.")
        verified.add(condition)
    return verified


def validate_complete_ann(
    output: Path, items: int, dimension: int, queries: int, expected: set[str],
) -> dict[str, Any]:
    completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if not completed["complete"] or sha256(output / "ann.jsonl") != completed["ann_sha256"]:
        raise ValueError("Complete ANN result journal checksum changed; refusing to recertify its metrics/timings.")
    if (
        completed["catalog_items"] != items or completed["dimension"] != dimension
        or completed["panel_queries"] != queries or completed["conditions"] != sorted(expected)
    ):
        raise ValueError("Complete ANN receipt does not cover the exact declared scope/condition grid.")
    return completed


def make_shared_ivf(items: np.ndarray, partition: Partition, half: bool = False):
    import faiss

    quantizer = faiss.IndexFlatIP(items.shape[1])
    quantizer.add(partition.centroids)
    if half:
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, items.shape[1], len(partition.centroids),
            faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT, False,
        )
    else:
        index = faiss.IndexIVFFlat(
            quantizer, items.shape[1], len(partition.centroids), faiss.METRIC_INNER_PRODUCT,
        )
    index.is_trained = True
    for start in range(0, len(items), 65536):
        vectors = np.asarray(items[start:start + 65536], dtype=np.float32, order="C")
        identifiers = np.arange(start, start + len(vectors), dtype=np.int64)
        assignment = np.ascontiguousarray(partition.assignments[start:start + len(vectors)])
        index.add_core(
            len(vectors), faiss.swig_ptr(vectors), faiss.swig_ptr(identifiers), faiss.swig_ptr(assignment),
        )
    if index.ntotal != len(items) or index.code_size != items.shape[1] * (2 if half else 4):
        raise RuntimeError("FAISS did not store the whole corpus in its explicitly declared representation.")
    return index


class SemanticFaissANN:
    def __init__(self, items: np.ndarray, partition: Partition, gpu: bool, half: bool = False) -> None:
        import faiss
        import faiss.contrib.torch_utils

        self.cuda = gpu
        self.resources = None
        index = make_shared_ivf(items, partition, half)
        self.code_bytes = index.code_size
        if gpu:
            self.resources = faiss.StandardGpuResources()
            self.resources.setDefaultNullStreamAllDevices()
            self.resources.setTempMemory(512 << 20)
            options = faiss.GpuClonerOptions()
            options.useFloat16 = False
            options.useFloat16CoarseQuantizer = False
            if hasattr(options, "use_cuvs"):
                options.use_cuvs = False
            index = faiss.index_cpu_to_gpu(self.resources, torch.cuda.current_device(), index, options)
        self.index = index

    def search(self, queries: torch.Tensor, k: int, probes: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda and (k > 2048 or probes > 2048):
            raise ValueError("Classic FAISS GPU requires k/nprobe<=2048.")
        self.index.nprobe = probes
        return self.index.search(queries if self.cuda else queries.cpu(), k)


@torch.inference_mode()
def resident_exact_topk(
    items: np.ndarray, queries: torch.Tensor, k: int, chunk_size: int = 131072,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(queries.device)
        resident = items.size * 4 < free * 0.75
    else:
        resident = True
    tensor = torch.as_tensor(np.array(items, dtype=np.float32), device=queries.device) if resident else items
    return exact_topk(tensor, queries.float(), k, chunk_size=chunk_size)


def exact_panel(
    items: np.ndarray, queries: np.ndarray, output: Path, k: int, batch_size: int = 128,
) -> dict[str, Any]:
    device = require_cuda()
    binding = {
        "items_shape": list(items.shape), "queries_shape": list(queries.shape),
        "k": k, "batch_size": batch_size, "product": "FP32", "tf32": False,
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        return validate_oracle(output, len(queries), len(items), items.shape[1], k)
    free, _ = torch.cuda.mem_get_info(device)
    resident = items.size * 4 < free * 0.75
    database = torch.as_tensor(np.array(items, dtype=np.float32), device=device) if resident else items
    identifiers = np.lib.format.open_memmap(
        output / "ids.partial.npy", mode="w+", dtype=np.int64, shape=(len(queries), k),
    )
    scores = np.lib.format.open_memmap(
        output / "scores.partial.npy", mode="w+", dtype=np.float32, shape=(len(queries), k),
    )
    started = time.perf_counter()
    for position in range(0, len(queries), batch_size):
        query = torch.as_tensor(
            np.array(queries[position:position + batch_size], dtype=np.float32), device=device,
        )
        values, found = exact_topk(database, query, k, chunk_size=65536)
        identifiers[position:position + len(query)] = found.cpu().numpy()
        scores[position:position + len(query)] = values.cpu().numpy()
        write_json(output / "progress.json", {"rows": position + len(query), "expected": len(queries)})
        print(f"FULL_CATALOG_FP32_ORACLE rows={position + len(query)}/{len(queries)}", flush=True)
    identifiers.flush()
    scores.flush()
    del identifiers, scores, database
    torch.cuda.empty_cache()
    (output / "ids.partial.npy").replace(output / "ids.npy")
    (output / "scores.partial.npy").replace(output / "scores.npy")
    completed = {
        "complete": True, "queries": len(queries), "catalog_items": len(items),
        "dimension": items.shape[1], "fp32_database_resident": resident,
        "database_policy": "All canonical stored vectors cast toFP32; tiled exact full-catalog product.",
        "elapsed_seconds": time.perf_counter() - started,
        "ids.npy_sha256": sha256(output / "ids.npy"), "scores.npy_sha256": sha256(output / "scores.npy"),
    }
    write_json(output / "completed.json", completed)
    return completed


def benchmark(
    item_file: Path, query_file: Path, panel_file: Path, oracle: Path, output: Path,
    clusters: int = 16384, backends: list[str] | None = None,
    smoke_only: bool = False,
) -> None:
    configure(17, 4)
    device = require_cuda()
    items = np.load(item_file, mmap_mode="r")
    queries = np.load(query_file, mmap_mode="r")
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    if items.dtype != np.float16 or queries.shape != (len(panel["query_ids"]), items.shape[1]):
        raise ValueError("ANN must use aligned measured semantic item/query representations.")
    names = backends or [
        "faiss-cpu-fp32", "faiss-gpu-fp32", "faiss-gpu-fp16",
        "silver-fp32", "silver-fp16", "silver-int8-global", "silver-int8-rowwise",
    ]
    validate_backends(names)
    probes_grid, candidate_grid = ([2, 4], [10, 50]) if smoke_only else ([16, 64, 256], [100, 1000])
    warmup, samples_count, repeats = (2, 5, 1) if smoke_only else (20, 50, 3)
    if (
        min(probes_grid) <= 0 or max(probes_grid) > clusters or max(candidate_grid) > len(items)
        or clusters > len(items) or (not smoke_only and len(items) < 1000000)
    ):
        raise ValueError("Explicit smoke/production ANN grid does not fit its declared catalog/centroid universe.")
    binding = {
        "items_sha256": sha256(item_file), "queries_sha256": sha256(query_file),
        "panel_sha256": sha256(panel_file), "clusters": clusters, "seed": 17,
        "backends": names, "probes": probes_grid, "candidate_counts": candidate_grid,
        "latency_batches": [1, 16], "warmup": warmup, "samples": samples_count, "repeats": repeats,
        "smoke_only": smoke_only,
        "source_dtype": "FP16 canonical stored unit vectors", "partition": "shared fixed assignments",
    }
    bind_run_config(output, binding)
    bind_run_config(oracle, {
        "items_sha256": binding["items_sha256"], "queries_sha256": binding["queries_sha256"],
        "k": max(candidate_grid), "product": "FP32", "tf32": False,
    })
    if not (oracle / "completed.json").exists():
        exact_panel(items, queries, oracle / "panel", max(candidate_grid))
        write_json(oracle / "completed.json", {
            "complete": True, "panel_receipt_sha256": sha256(oracle / "panel" / "completed.json"),
        })
    if sha256(oracle / "panel" / "completed.json") != json.loads((oracle / "completed.json").read_text())["panel_receipt_sha256"]:
        raise ValueError("ANN oracle completion receipt changed.")
    validate_oracle(oracle / "panel", len(queries), len(items), items.shape[1], max(candidate_grid))
    truth = np.load(oracle / "panel" / "ids.npy", mmap_mode="r")
    bind_run_config(output / "partition", {
        "items_sha256": binding["items_sha256"], "clusters": clusters, "seed": 17, "iterations": 20,
    })
    partition_receipt = output / "partition" / "validated-partition.json"
    if (output / "partition" / "partition.npz").exists():
        saved = json.loads(partition_receipt.read_text(encoding="utf-8"))
        if sha256(output / "partition" / "partition.npz") != saved["partition_sha256"]:
            raise ValueError("Shared IVF partition artifact checksum changed.")
    partition = train_partition(items, clusters, 17, output / "partition")
    validate_partition(partition, len(items), items.shape[1], clusters)
    if not partition_receipt.exists():
        write_json(partition_receipt, {"partition_sha256": sha256(output / "partition" / "partition.npz")})
    qrels = {query: {int(item): int(grade) for item, grade in values.items()} for query, values in panel["qrels"].items()}
    query_ids = panel["query_ids"]
    journal = output / "ann.jsonl"
    _, gpu_total = torch.cuda.mem_get_info(device)
    unsupported = {
        backend for backend in names
        if (backend.startswith("silver") and items.shape[1] not in SILVER_DIMENSIONS)
        or (backend == "faiss-gpu-fp32" and items.size * 4 > gpu_total * 0.8)
    }
    expected_conditions = {
        condition for backend in names for condition in (
            [backend] if backend in unsupported else [
                f"{backend}-p{probes}-k{k}" for probes in probes_grid for k in candidate_grid
            ]
        )
    }
    records = []
    if journal.exists():
        records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    completed_conditions = validate_journal(records, output, expected_conditions, len(queries), len(items))
    if (output / "completed.json").exists():
        validate_complete_ann(output, len(items), items.shape[1], len(queries), expected_conditions)
        if completed_conditions != expected_conditions:
            raise ValueError("Completed ANN experiment has missing conditions.")
        print("COMPLETE_REAL_SEMANTIC_ANN_REUSED_WITH_VERIFIED_JOURNAL_AND_ARTIFACTS", flush=True)
        return
    for backend in names:
        started = time.perf_counter()
        if backend in unsupported and backend in completed_conditions:
            continue
        if items.shape[1] not in SILVER_DIMENSIONS and backend.startswith("silver"):
            append_jsonl(journal, {
                "condition": backend, "backend": backend, "status": "unsupported",
                "reason": "Pinned SilverTorch has no CUDA dispatch for this native dimension.",
                "dimension": items.shape[1], "catalog_items": len(items),
            })
            continue
        if backend == "faiss-gpu-fp32" and backend in unsupported:
            append_jsonl(journal, {
                "condition": backend, "backend": backend, "status": "unsupported_memory",
                "reason": "Full native FP32 corpus cannot fit one80GBGPU; no catalog/dimension reduction.",
                "dimension": items.shape[1], "catalog_items": len(items),
            })
            continue
        if backend.startswith("faiss"):
            cpu = backend.startswith("faiss-cpu")
            torch.set_num_threads(32 if cpu else 4)
            import faiss
            faiss.omp_set_num_threads(32 if cpu else 4)
            model = SemanticFaissANN(items, partition, BACKEND_STORAGE[backend][0], BACKEND_STORAGE[backend][1] == "fp16")
        elif backend.startswith("silver-"):
            torch.set_num_threads(4)
            model = SilverANN(items, partition, BACKEND_STORAGE[backend][1])
        else:
            raise ValueError("Unknown declared semantic ANN backend.")
        build_seconds = time.perf_counter() - started
        for probes in probes_grid:
            for k in candidate_grid:
                condition = f"{backend}-p{probes}-k{k}"
                if condition in completed_conditions:
                    continue
                predictions = np.empty((len(queries), k), dtype=np.int64)
                retrieval_seconds = 0.0
                quality_batch = 32 if model.cuda else 16
                for start in range(0, len(queries), quality_batch):
                    query = torch.as_tensor(
                        np.array(queries[start:start + quality_batch], dtype=np.float32),
                        device=device if model.cuda else torch.device("cpu"),
                    )
                    if model.cuda:
                        torch.cuda.synchronize()
                    before = time.perf_counter()
                    _, found = model.search(query, k, probes)
                    if model.cuda:
                        torch.cuda.synchronize()
                    retrieval_seconds += time.perf_counter() - before
                    predictions[start:start + len(query)] = found.cpu().numpy()
                np.save(output / (condition + "-ids.npy"), predictions)
                latency = {}
                for batch in (1, 16):
                    rng = np.random.default_rng(17 + batch)
                    selected_query_rows = [
                        rng.choice(len(queries), batch, replace=len(queries) < batch)
                        for _ in range(samples_count)
                    ]
                    query_batches = [
                        torch.as_tensor(
                            np.array(queries[chosen], dtype=np.float32),
                            device=device if model.cuda else torch.device("cpu"),
                        ) for chosen in selected_query_rows
                    ]
                    samples = []
                    for _ in range(repeats):
                        position = 0

                        def operation():
                            nonlocal position
                            query = query_batches[position % len(query_batches)]
                            position += 1
                            return model.search(query, k, probes)

                        _, measurements = measure(operation, warmup, samples_count, model.cuda)
                        samples.extend(measurements)
                    latency[str(batch)] = {
                        **latency_summary(samples, batch),
                        "query_row_batches": [chosen.tolist() for chosen in selected_query_rows],
                        "query_sampling": "Label-independent varying queries across the entire evaluated panel; same groups across backends.",
                    }
                row = {
                    "condition": condition, "backend": backend, "status": "ok", "probes": probes, "k": k,
                    "catalog_items": len(items), "dimension": items.shape[1], "index_build_seconds": build_seconds,
                    "retrieval_wall_seconds": retrieval_seconds, "quality_query_batch": quality_batch,
                    "geometric_recall_at_k": float(recall_at_k(predictions, truth, k).mean()),
                    "relevance": judged_metrics(query_ids, predictions, qrels, panel["binary_threshold"]),
                    "latency": latency, "environment": environment_manifest(),
                    "prediction_ids_sha256": sha256(output / (condition + "-ids.npy")),
                    "faiss_storage_contract": "IVF-SQfp16codes" if backend.endswith("fp16") and backend.startswith("faiss") else None,
                }
                append_jsonl(journal, row)
                print(f"REAL_SEMANTIC_ANN_COMPLETE {condition} recall={row['geometric_recall_at_k']:.6f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    completed_conditions = validate_journal(records, output, expected_conditions, len(queries), len(items))
    if completed_conditions != expected_conditions:
        raise RuntimeError("ANN experiment is missing declared conditions; no completion receipt may be published.")
    write_json(output / "completed.json", {
        "complete": True, "ann_sha256": sha256(journal), "catalog_items": len(items),
        "dimension": items.shape[1], "panel_queries": len(queries), "conditions": sorted(expected_conditions),
        "smoke_only": smoke_only,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Full real learned-vector ANN geometry, judged quality and timing.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=16384)
    parser.add_argument("--backend", action="append")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    benchmark(
        Path(args.items), Path(args.queries), Path(args.panel), Path(args.oracle), Path(args.output),
        args.clusters, args.backend, args.smoke_only,
    )


if __name__ == "__main__":
    main()
