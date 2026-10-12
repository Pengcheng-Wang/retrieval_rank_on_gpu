from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.metrics import latency_summary
from rrbench.runtime import configure, environment_manifest, require_cuda
from rrbench.semantic_ir import PassageStore, panel_metrics
from rrbench.semantic_models import (
    SemanticEncoder, TextReranker, configuration_digest, mrl_prefix, validate_model_files, validate_smoke_contract,
)
from rrbench.semantic_rankers import (
    make_cached_ranker, query_representation, validate_checkpoint_features, validate_fitting_query_alignment,
)
from rrbench.semantic_sampling import latency_query_rows, stable_rank


def device_telemetry(device: torch.device, require_exclusive: bool = False) -> dict[str, int]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    identifier = visible.split(",")[index] if visible else str(index)
    if require_exclusive:
        values = subprocess.check_output([
            "nvidia-smi", "--id=" + identifier, "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ], text=True).splitlines()
        other = [int(value.strip()) for value in values if value.strip() and int(value.strip()) != os.getpid()]
        if other:
            raise RuntimeError("GPU timing device has another compute process; no uncontended benchmark is published.")
    values = subprocess.check_output([
        "nvidia-smi", "--id=" + identifier,
        "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if len(values) != 3:
        raise RuntimeError("Device memory/utilization measurement has an unexpected NVML schema.")
    return {
        "memory_used_bytes": int(values[0].strip()) * (1 << 20),
        "device_memory_bytes": int(values[1].strip()) * (1 << 20),
        "gpu_utilization_percent": int(values[2].strip()),
    }


def pipeline_memory_bytes(
    items: int, dimension: int, index_bytes_per_value: int, encoder_parameters: int,
    reranker_parameters: int, cached_features: bool,
) -> int:
    if min(items, dimension, encoder_parameters) <= 0 or min(index_bytes_per_value, reranker_parameters) < 0:
        raise ValueError("Pipeline memory accounting requires valid catalog/model sizes.")
    return (
        items * dimension * (index_bytes_per_value + (2 if cached_features else 0))
        + (encoder_parameters + reranker_parameters) * 2 + (8 << 30)
    )


def transfer_candidates(candidates: torch.Tensor, boundary: str, device: torch.device) -> torch.Tensor:
    return transfer_candidate_payload(candidates, boundary, device)[0]


def transfer_candidate_payload(
    candidates: torch.Tensor, boundary: str, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if candidates.ndim != 2 or candidates.dtype != torch.int64:
        raise ValueError("Stage boundaries require a two-dimensional external candidate-ID tensor.")
    if boundary == "integrated":
        return candidates.to(device), None
    if boundary == "local-cascade":
        host = candidates.detach().cpu().contiguous()
        serialized = host.numpy().tobytes(order="C")
        received = torch.from_numpy(np.frombuffer(serialized, dtype=np.int64).copy().reshape(host.shape))
        return received.to(device), received
    raise ValueError("Unknown measured stage boundary; no assumed-RPC fallback is supported.")


def rank_candidates(candidates: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    if candidates.ndim != 2 or scores.shape != candidates.shape or not 0 < k <= candidates.shape[1]:
        raise ValueError("Final ranking requires aligned candidates/scores and a valid output cutoff.")
    if not torch.isfinite(scores[candidates >= 0]).all():
        raise ValueError("Valid candidate scores must be finite before final ranking.")
    values = scores.float().masked_fill(candidates < 0, -torch.inf)
    scores, positions = values.topk(k, dim=1)
    return candidates.gather(1, positions).masked_fill(~torch.isfinite(scores), -1)


def validate_candidates(candidates: np.ndarray, queries: int, items: int) -> None:
    if candidates.ndim != 2 or len(candidates) != queries or candidates.dtype != np.int64:
        raise ValueError("Reranking candidate IDs must align with their source query rows.")
    if ((candidates < -1) | (candidates >= items)).any():
        raise ValueError("Reranking candidate ID is outside the exact source catalog.")
    for row in candidates:
        found = row[row >= 0]
        if len(found) != len(np.unique(found)):
            raise ValueError("Reranking candidate pool contains duplicate external IDs.")


def score_text_panel(
    panel_file: Path, candidates_file: Path, source: Path, offsets: Path, specification: dict[str, Any],
    cache: Path, smoke_file: Path, output: Path, worker: int = 0, workers: int = 1,
) -> dict[str, Any]:
    if not 0 <= worker < workers or workers <= 0:
        raise ValueError("Text reranker worker must belong to its finite shard set.")
    configure(17, 4)
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    candidates = np.load(candidates_file, mmap_mode="r")
    store = PassageStore(source, offsets)
    validate_candidates(candidates, len(panel["queries"]), len(store.offsets) - 1)
    smoke = json.loads(smoke_file.read_text(encoding="utf-8"))
    if not smoke["complete"]:
        raise ValueError("Text model smoke must finish before reranking.")
    row = next(row for row in smoke["models"] if row["key"] == specification["key"])
    validate_smoke_contract(specification, row)
    validate_model_files(specification, cache, row["model_files"])
    rows = np.arange(worker, len(candidates), workers)
    maximum = min(500, candidates.shape[1])
    batch_size = 32 if specification["key"].startswith("qwen8") else 64
    binding = {
        "specification_sha256": configuration_digest(specification), "panel_sha256": sha256(panel_file),
        "candidates_sha256": sha256(candidates_file), "source_sha256": sha256(source),
        "offsets_sha256": sha256(offsets), "worker": worker, "workers": workers,
        "candidate_prefix": maximum, "pair_batch": batch_size,
        "instruction": panel.get("history_instruction", TextReranker.INSTRUCTION),
        "reranker_query_token_cap": panel.get("reranker_query_token_cap"),
        "candidate_policy": "No positive injection; exact fixed candidate IDs before labels are scored.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if (
            sha256(output / "scores.npy") != completed["scores_sha256"]
            or sha256(output / "query_rows.npy") != completed["query_rows_sha256"]
        ):
            raise ValueError("Complete reranker score shard bytes changed.")
        return completed
    model = TextReranker(specification, cache, require_cuda())
    scores = np.lib.format.open_memmap(
        output / "scores.partial.npy", mode="w+", dtype=np.float32, shape=(len(rows), maximum),
    )
    scores[:] = -np.inf
    torch.cuda.reset_peak_memory_stats()
    totals = {"real_tokens": 0, "padded_tokens": 0, "pairs": 0, "at_token_cap_pairs": 0, "bounded_query_pairs": 0}
    started = time.perf_counter()
    for index, query_row in enumerate(rows):
        identifiers = candidates[query_row, :maximum]
        valid = np.flatnonzero(identifiers >= 0)
        documents = store.fetch(np.asarray(identifiers[valid]))
        query = panel["queries"][query_row]
        for position in range(0, len(valid), batch_size):
            document_batch = documents[position:position + batch_size]
            values, measured = model.score(
                [query] * len(document_batch), document_batch,
                instruction=panel.get("history_instruction"),
                query_token_cap=panel.get("reranker_query_token_cap"),
            )
            scores[index, valid[position:position + len(document_batch)]] = values.cpu().numpy()
            totals["real_tokens"] += measured.real_tokens
            totals["padded_tokens"] += measured.padded_tokens
            totals["pairs"] += measured.rows
            totals["at_token_cap_pairs"] += measured.at_token_cap
            totals["bounded_query_pairs"] += measured.bounded_query_rows
        if index % 20 == 0:
            scores.flush()
            write_json(output / "progress.json", {
                "query_rows": index + 1, "expected_rows": len(rows), "totals": totals,
            })
            print(f"STRONG_TEXT_RERANKING {specification['key']} worker={worker} rows={index + 1}/{len(rows)}", flush=True)
    scores.flush()
    del scores
    (output / "scores.partial.npy").replace(output / "scores.npy")
    np.save(output / "query_rows.npy", rows)
    elapsed = time.perf_counter() - started
    completed = {
        "complete": True, "rows": len(rows), "worker": worker, "workers": workers,
        "parameters": model.parameters, "model": specification["model"], "revision": specification["revision"],
        "elapsed_seconds": elapsed, "totals": totals, "pairs_per_wall_second": totals["pairs"] / elapsed,
        "real_tokens_per_wall_second": totals["real_tokens"] / elapsed,
        "scores_sha256": sha256(output / "scores.npy"), "query_rows_sha256": sha256(output / "query_rows.npy"),
        "candidate_prefix": maximum, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "environment": environment_manifest(), "binding": binding,
    }
    write_json(output / "completed.json", completed)
    return completed


def collect_text_scores(
    panel_file: Path, candidates_file: Path, shards: list[Path], output: Path,
) -> dict[str, Any]:
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    candidates = np.load(candidates_file, mmap_mode="r")
    scores = np.full((len(candidates), min(500, candidates.shape[1])), -np.inf, dtype=np.float32)
    found = np.zeros(len(candidates), dtype=bool)
    receipts = []
    expected_spec = None
    expected_protocol = None
    for shard in shards:
        complete = json.loads((shard / "completed.json").read_text(encoding="utf-8"))
        binding = complete["binding"]
        if (
            not complete["complete"] or binding["panel_sha256"] != sha256(panel_file)
            or binding["candidates_sha256"] != sha256(candidates_file)
        ):
            raise ValueError("Text score shard belongs to a different panel or fixed candidate pool.")
        if expected_spec is None:
            expected_spec = binding["specification_sha256"]
            expected_protocol = {key: value for key, value in binding.items() if key != "worker"}
        if binding["specification_sha256"] != expected_spec:
            raise ValueError("Text score shards use different scorer specifications.")
        if {key: value for key, value in binding.items() if key != "worker"} != expected_protocol:
            raise ValueError("Text score shards use different corpus/offset/scoring protocol bindings.")
        if sha256(shard / "scores.npy") != complete["scores_sha256"] or sha256(shard / "query_rows.npy") != complete["query_rows_sha256"]:
            raise ValueError("Text score shard payload checksums changed.")
        rows = np.load(shard / "query_rows.npy")
        values = np.load(shard / "scores.npy", mmap_mode="r")
        if rows.ndim != 1 or rows.dtype != np.int64 or values.dtype != np.float32 or values.shape != (len(rows), scores.shape[1]):
            raise ValueError("Text score shard row mapping has a wrong shape or dtype.")
        if ((rows < 0) | (rows >= len(scores))).any() or len(np.unique(rows)) != len(rows) or found[rows].any():
            raise ValueError("Text score shards duplicate or mis-map evaluated queries.")
        if not np.isfinite(values[candidates[rows, :scores.shape[1]] >= 0]).all():
            raise ValueError("Text score shard has invalid scores for valid candidates.")
        scores[rows], found[rows] = values, True
        receipts.append(sha256(shard / "completed.json"))
    if not found.all():
        raise ValueError("Text reranking output omitted query rows; no quality result may be published.")
    binding = {
        "panel_sha256": sha256(panel_file), "candidates_sha256": sha256(candidates_file),
        "shard_receipts": receipts, "specification_sha256": expected_spec,
    }
    bind_run_config(output, binding)
    np.save(output / "scores.npy", scores)
    quality = {}
    for cap in (10, 50, 100, 500):
        if cap > scores.shape[1]:
            continue
        ranked = rank_candidates(
            torch.from_numpy(np.array(candidates[:, :cap], dtype=np.int64)),
            torch.from_numpy(scores[:, :cap]), cap,
        ).numpy()
        quality[str(cap)] = panel_metrics(panel, ranked)
        np.save(output / f"ranked-cap{cap}.npy", ranked)
    complete = {
        "complete": True, "rows": len(scores), "scores_sha256": sha256(output / "scores.npy"),
        "relevance_by_candidate_cap": quality, "candidate_source_sha256": binding["candidates_sha256"],
        "positive_injection": False, "scorer_specification_sha256": expected_spec,
        "quality_scope": "Entire scored candidate prefix sorted; serving takes its top10. Recall above the prefixcap is limited to returneditems.",
    }
    write_json(output / "completed.json", complete)
    return complete


@torch.inference_mode()
def score_cached_candidates(
    model: torch.nn.Module, items: np.ndarray, queries: np.ndarray, candidates: np.ndarray,
    assignments: np.ndarray, lengths: np.ndarray, centroids: np.ndarray,
    device: torch.device, pair_batch: int = 2048,
) -> np.ndarray:
    validate_candidates(candidates, len(queries), len(items))
    if (
        queries.shape[1] != items.shape[1] or assignments.shape != (len(items),)
        or lengths.shape != (len(items),) or pair_batch <= 0
    ):
        raise ValueError("Cached scorer full-corpus feature arrays are not aligned.")
    centroid_gpu = torch.as_tensor(centroids, device=device)
    scores = np.full(candidates.shape, -np.inf, dtype=np.float32)
    for query_row in range(len(queries)):
        ids = candidates[query_row]
        positions = np.flatnonzero(ids >= 0)
        query = torch.as_tensor(np.array(queries[query_row:query_row + 1], dtype=np.float32), device=device)
        cluster = int((query @ centroid_gpu.T).argmax(dim=1).item())
        for start in range(0, len(positions), pair_batch):
            selected = positions[start:start + pair_batch]
            identifiers = ids[selected]
            vectors = torch.as_tensor(np.array(items[identifiers], dtype=np.float32), device=device)
            sparse = torch.as_tensor(np.column_stack((
                np.full(len(selected), cluster, dtype=np.int64), assignments[identifiers],
                np.minimum(np.log2(np.maximum(lengths[identifiers], 1)).astype(np.int64), 31),
            )), device=device, dtype=torch.long)
            values = model(query.expand(len(selected), -1), vectors, sparse)
            if values.shape != (len(selected),) or not torch.isfinite(values).all():
                raise RuntimeError("Cached scorer returned invalid scores for valid candidates.")
            scores[query_row, selected] = values.cpu().numpy()
    return scores


def evaluate_cached_ranker(
    name: str, checkpoint: Path, item_file: Path, query_file: Path, panel_file: Path,
    candidates_file: Path, partition_file: Path, lengths_file: Path, output: Path,
) -> dict[str, Any]:
    configure(17, 4)
    items, queries = np.load(item_file, mmap_mode="r"), np.load(query_file, mmap_mode="r")
    candidates, lengths = np.load(candidates_file, mmap_mode="r"), np.load(lengths_file, mmap_mode="r")
    partition = np.load(partition_file)
    assignments, centroids = partition["assignments"], partition["centroids"]
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    validate_fitting_query_alignment(query_file, panel_file)
    representation = query_representation(query_file)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_features(state["binding"], item_file, partition_file, lengths_file, representation)
    if state["binding"]["model"] != name:
        raise ValueError("Cached evaluation checkpoint model differs from the declared architecture.")
    binding = {
        "checkpoint_sha256": sha256(checkpoint), "panel_sha256": sha256(panel_file),
        "items_sha256": sha256(item_file), "queries_sha256": sha256(query_file),
        "candidates_sha256": sha256(candidates_file), "partition_sha256": sha256(partition_file),
        "lengths_sha256": sha256(lengths_file), "same_candidate_policy": True,
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(output / "scores.npy") != complete["scores_sha256"]:
            raise ValueError("Cached evaluated score bytes changed.")
        return complete
    model = make_cached_ranker(name, items.shape[1], len(centroids), require_cuda(), 17).eval()
    model.load_state_dict(state["model"])
    started = time.perf_counter()
    scores = score_cached_candidates(
        model, items, queries, candidates, assignments, lengths, centroids, require_cuda(),
    )
    elapsed = time.perf_counter() - started
    np.save(output / "scores.npy", scores)
    quality = {}
    for cap in (10, 50, 100, 500, 1000):
        if cap > candidates.shape[1]:
            continue
        ranked = rank_candidates(
            torch.from_numpy(np.array(candidates[:, :cap])), torch.from_numpy(scores[:, :cap]), cap,
        ).numpy()
        quality[str(cap)] = panel_metrics(panel, ranked)
        np.save(output / f"ranked-cap{cap}.npy", ranked)
    complete = {
        "complete": True, "model_class": model.model_class, "query_rows": len(queries),
        "scores_sha256": sha256(output / "scores.npy"), "relevance_by_candidate_cap": quality,
        "scoring_wall_seconds": elapsed, "environment": environment_manifest(),
        "candidate_policy": "Fixed measured candidate IDs, no injected evaluation positives.",
    }
    write_json(output / "completed.json", complete)
    return complete


def benchmark_cached_boundary(
    name: str, checkpoint: Path, item_file: Path, query_file: Path, candidates_file: Path,
    partition_file: Path, lengths_file: Path, output: Path,
) -> dict[str, Any]:
    configure(17, 4)
    device = require_cuda()
    items, queries = np.load(item_file, mmap_mode="r"), np.load(query_file, mmap_mode="r")
    candidates = np.load(candidates_file, mmap_mode="r")
    partition, lengths = np.load(partition_file), np.load(lengths_file, mmap_mode="r")
    assignments, centroids = partition["assignments"], partition["centroids"]
    validate_candidates(candidates, len(queries), len(items))
    if items.shape[1] != queries.shape[1] or lengths.shape != (len(items),):
        raise ValueError("Cached boundary inputs have mismatched semantic feature shapes.")
    binding = {
        "model": name, "checkpoint_sha256": sha256(checkpoint),
        "inputs": {name: sha256(path) for name, path in {
            "items": item_file, "queries": query_file, "candidates": candidates_file,
            "partition": partition_file, "lengths": lengths_file,
        }.items()},
        "boundaries": ["integrated", "local-cascade"], "same_weights": True,
        "same_candidates": True, "precision": "FP32", "query_batches": [1, 16], "candidate_caps": [10, 50, 100, 500],
    }
    bind_run_config(output, binding)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_features(
        state["binding"], item_file, partition_file, lengths_file, query_representation(query_file),
    )
    needed = items.size * 2 + queries.size * 4 + (4 << 30)
    if needed > torch.cuda.mem_get_info(device)[0]:
        completed = {
            "complete": True, "status": "unsupported_memory",
            "reason": "Full canonicalFP16residentbank plus queries/scorer cannot fit the GPU; no catalog reduction.",
            "preflight_bytes": needed,
        }
        write_json(output / "completed.json", completed)
        return completed
    model = make_cached_ranker(name, items.shape[1], len(centroids), device, 17).eval()
    if state["binding"]["model"] != name or state["binding"]["dimension"] != items.shape[1]:
        raise ValueError("Cached boundary checkpoint model/semantic dimension differs from its declared arm.")
    model.load_state_dict(state["model"])
    item_gpu = torch.as_tensor(np.array(items), device=device)
    query_gpu = torch.as_tensor(np.array(queries, dtype=np.float32), device=device)
    assignment_gpu = torch.as_tensor(assignments, device=device)
    bucket_gpu = torch.as_tensor(
        np.minimum(np.log2(np.maximum(lengths, 1)).astype(np.int64), 31), device=device,
    )
    centroid_gpu = torch.as_tensor(centroids, device=device)
    query_clusters = (query_gpu @ centroid_gpu.T).argmax(dim=1)
    conditions, pairs = [], []
    for batch in (1, 16):
        if len(queries) < batch:
            continue
        rows = np.random.default_rng(17 + batch).choice(len(queries), batch, replace=False)
        for cap in (10, 50, 100, 500):
            if cap > candidates.shape[1]:
                continue
            source_ids = torch.as_tensor(np.array(candidates[rows, :cap]), device=device)
            if (source_ids < 0).any():
                raise ValueError("Boundary benchmark requires full valid fixed candidate prefixes.")
            repeated_queries = query_gpu[torch.as_tensor(rows, device=device)].repeat_interleave(cap, dim=0)
            outputs = {}
            for boundary in ("integrated", "local-cascade"):

                @torch.inference_mode()
                def operation():
                    ids, host_ids = transfer_candidate_payload(source_ids, boundary, device)
                    flattened = ids.reshape(-1)
                    if boundary == "integrated":
                        item = item_gpu[flattened].float()
                        sparse = torch.stack((
                            query_clusters[torch.as_tensor(rows, device=device)].repeat_interleave(cap),
                            assignment_gpu[flattened], bucket_gpu[flattened],
                        ), dim=1)
                    else:
                        if host_ids is None:
                            raise RuntimeError("Local cascade has no received host candidate payload.")
                        host = host_ids.numpy().reshape(-1)
                        item = torch.as_tensor(np.array(items[host], dtype=np.float32), device=device)
                        sparse = torch.as_tensor(np.column_stack((
                            query_clusters[torch.as_tensor(rows, device=device)].cpu().numpy().repeat(cap),
                            assignments[host],
                            np.minimum(np.log2(np.maximum(lengths[host], 1)).astype(np.int64), 31),
                        )), device=device, dtype=torch.long)
                    values = model(repeated_queries, item, sparse).reshape(batch, cap)
                    return rank_candidates(ids, values, min(10, cap)), values

                for _ in range(10):
                    operation()
                samples = []
                for _ in range(50):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    ranked, values = operation()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                outputs[boundary] = (ranked.cpu(), values.cpu())
                conditions.append({
                    "boundary": boundary, "query_batch": batch, "candidate_cap": cap,
                    "latency": latency_summary(samples, batch),
                })
            torch.testing.assert_close(outputs["integrated"][1], outputs["local-cascade"][1], rtol=1e-5, atol=1e-6)
            if not torch.equal(outputs["integrated"][0], outputs["local-cascade"][0]):
                raise RuntimeError("Same-score local stage boundaries changed final candidate ranking.")
            pairs.append({"query_batch": batch, "candidate_cap": cap, "same_scores_and_final_ids": True})
    complete = {
        "complete": True, "model_class": model.model_class, "conditions": conditions, "parity": pairs,
        "boundary_scope": "Measured in-process serialization/CPUfeaturefetch/H2D. No interprocess or fabricatednetworkRPC.",
        "excluded_shared_stages": "Same upstream queryencoding andANNretrieval excluded from incrementalboundary timer.",
        "environment": environment_manifest(),
    }
    write_json(output / "completed.json", complete)
    return complete


def benchmark_online_pipeline(
    registry_file: Path, encoder_key: str, ranker_key: str, backend: str,
    item_file: Path, partition_file: Path, panel_file: Path, source: Path, offsets: Path,
    cache: Path, smoke_file: Path, output: Path, checkpoint: Path | None = None,
    token_lengths: Path | None = None,
) -> dict[str, Any]:
    from rrbench.ann import Partition, SilverANN
    from rrbench.semantic_ann import BACKEND_STORAGE, SILVER_DIMENSIONS, SemanticFaissANN, validate_backends, validate_partition

    validate_backends([backend])
    configure(17, 4)
    device = require_cuda()
    idle_device = device_telemetry(device, require_exclusive=True)
    items = np.load(item_file, mmap_mode="r")
    data = np.load(partition_file)
    partition = Partition(*(data[name] for name in ("centroids", "assignments", "order", "offsets")))
    validate_partition(partition, len(items), items.shape[1], len(partition.centroids))
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    specification = registry["models"][encoder_key]
    smoke = json.loads(smoke_file.read_text(encoding="utf-8"))
    if not smoke["complete"]:
        raise ValueError("Online pipeline requires the complete actual-model smoke.")
    attested = {row["key"]: row for row in smoke["models"]}
    validate_smoke_contract(specification, attested[encoder_key])
    validate_model_files(specification, cache, attested[encoder_key]["model_files"])
    is_text = ranker_key in registry["models"] and "dimensions" not in registry["models"][ranker_key]
    if is_text:
        ranker_specification = registry["models"][ranker_key]
        validate_smoke_contract(ranker_specification, attested[ranker_key])
        validate_model_files(ranker_specification, cache, attested[ranker_key]["model_files"])
        ranker_parameters = attested[ranker_key]["parameters"]
    else:
        if checkpoint is None or token_lengths is None:
            raise ValueError("Cached online rankers require their actual fitted checkpoint and token features.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        current_representation = {
            "model": specification["model"], "revision": specification["revision"],
            "specification_sha256": configuration_digest(specification),
            "stored_dimension": items.shape[1], "stored_dtype": "float16",
            "normalization": "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage.",
            "query_instruction": panel.get("history_instruction", specification["query_instruction"]),
        }
        validate_checkpoint_features(
            state["binding"], item_file, partition_file, token_lengths, current_representation,
        )
        ranker_parameters = 0
    storage_bytes = (
        0 if not BACKEND_STORAGE[backend][0]
        else (1 if "int8" in backend else (2 if backend.endswith("fp16") else 4))
    )
    admission_bytes = pipeline_memory_bytes(
        len(items), items.shape[1], storage_bytes, attested[encoder_key]["parameters"],
        ranker_parameters, not is_text,
    )
    binding = {
        "encoder": encoder_key, "ranker": ranker_key, "backend": backend,
        "items_sha256": sha256(item_file), "partition_sha256": sha256(partition_file),
        "panel_sha256": sha256(panel_file), "source_sha256": sha256(source), "offsets_sha256": sha256(offsets),
        "model_registry_sha256": sha256(registry_file), "smoke_sha256": sha256(smoke_file),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
        "token_lengths_sha256": sha256(token_lengths) if token_lengths is not None else None,
        "boundaries": ["integrated", "local-cascade"], "candidate_caps": [10, 50, 100, 500],
        "query_batches": [1, 16], "probes": 64, "same_weights": True, "same_scorer": True,
        "same_precision": True, "same_candidates": True,
        "primary_warmup_batches": 1, "primary_measured_batches": 3,
        "timing_query_policy": "Fourdisjointhash-selectedbatches, shared acrossmodels/caps/boundaries.",
        "timing_estimand": "Descriptive synchronized wall-clock afteronewarmup, notsteady-stateSLO.",
        "history_bound_preprocessing": "Unique raw queries per scorerbatch; identical bounded tokens reused for repeated candidatepairs.",
    }
    bind_run_config(output, binding)
    if items.shape[1] not in specification["dimensions"]:
        raise ValueError("Online query representation does not match the native/MRL corpus dimension.")
    if backend.startswith("silver") and items.shape[1] not in SILVER_DIMENSIONS:
        result = {
            "complete": True, "status": "unsupported_dimension", "dimension": items.shape[1],
            "reason": "Pinned SilverTorch has no nativeCUDA dispatch for this dimension; no silentMRLsubstitution.",
        }
        write_json(output / "completed.json", result)
        return result
    _, total = torch.cuda.mem_get_info(device)
    if admission_bytes > total:
        result = {
            "complete": True, "status": "unsupported_memory", "estimated_minimum_and_workspace_bytes": admission_bytes,
            "device_memory_bytes": total, "reason": "Full declared index plus models cannot fit this singleGPUbudget; no silent reduction.",
            "memory_policy": "Preflight bound with8GiBworkspace, not measured memory or MFU.",
        }
        write_json(output / "completed.json", result)
        return result
    encoder = SemanticEncoder(specification, cache, device)
    if backend.startswith("faiss"):
        ann = SemanticFaissANN(items, partition, BACKEND_STORAGE[backend][0], BACKEND_STORAGE[backend][1] == "fp16")
    else:
        ann = SilverANN(items, partition, BACKEND_STORAGE[backend][1])
    store = PassageStore(source, offsets)
    if len(store.offsets) - 1 != len(items):
        raise ValueError("Online text lookup and ANN corpus use different item-ID universes.")
    if is_text:
        ranker = TextReranker(ranker_specification, cache, device)
        scorer_batch = 32 if ranker_key.startswith("qwen8") else 64
    else:
        ranker = make_cached_ranker(ranker_key, items.shape[1], len(partition.centroids), device, 17).eval()
        if state["binding"]["model"] != ranker_key or state["binding"]["dimension"] != items.shape[1]:
            raise ValueError("Online fitted scorer checkpoint does not match the declared model/dimension.")
        ranker.load_state_dict(state["model"])
        lengths = np.load(token_lengths, mmap_mode="r")
        if lengths.shape != (len(items),):
            raise ValueError("Online cached token-length feature does not cover the full corpus.")
        item_gpu = torch.as_tensor(np.array(items), device=device)
        assignment_gpu = torch.as_tensor(partition.assignments, device=device)
        buckets = np.minimum(np.log2(np.maximum(lengths, 1)).astype(np.int64), 31)
        bucket_gpu = torch.as_tensor(buckets, device=device)
        centroids_gpu = torch.as_tensor(partition.centroids, device=device)
    conditions, parity = [], []
    device_samples = [idle_device, device_telemetry(device, require_exclusive=True)]
    torch.cuda.reset_peak_memory_stats(device)
    for query_batch in (1, 16):
        if len(panel["queries"]) < query_batch:
            continue
        row_batches = latency_query_rows(panel, query_batch)
        for cap in (10, 50, 100, 500):
            if cap > len(items):
                continue
            outputs = {}
            order_key = stable_rank("online-boundary-order", 20261010, f"{query_batch}:{cap}")[0]
            boundaries = ["integrated", "local-cascade"]
            if order_key[0] % 2:
                boundaries.reverse()
            for boundary in boundaries:

                @torch.inference_mode()
                def operation(chosen):
                    texts = [panel["queries"][int(row)] for row in chosen]
                    query, _ = encoder.encode(texts, query=True, instruction=panel.get("history_instruction"))
                    if query.shape[1] != items.shape[1]:
                        query = mrl_prefix(query, items.shape[1])
                    query = query.to(torch.float16).float()
                    _, found = ann.search(query, cap, 64)
                    if not ann.cuda:
                        found = found.to(device)
                    ids, host_ids = transfer_candidate_payload(found.long(), boundary, device)
                    if (ids < 0).any():
                        raise ValueError("Online pipeline could not produce a full valid candidate prefix.")
                    if is_text:
                        text_ids = (
                            host_ids.numpy().reshape(-1) if host_ids is not None
                            else ids.cpu().numpy().reshape(-1)
                        )
                        documents = store.fetch(text_ids)
                        repeated = [text for text in texts for _ in range(cap)]
                        values = []
                        for start in range(0, len(documents), scorer_batch):
                            scores, _ = ranker.score(
                                repeated[start:start + scorer_batch], documents[start:start + scorer_batch],
                                instruction=panel.get("history_instruction"),
                                query_token_cap=panel.get("reranker_query_token_cap"),
                            )
                            values.append(scores)
                        scores = torch.cat(values).reshape(query_batch, cap)
                    else:
                        flat = ids.reshape(-1)
                        cluster = (query @ centroids_gpu.T).argmax(dim=1)
                        if boundary == "integrated":
                            vectors = item_gpu[flat].float()
                            sparse = torch.stack((
                                cluster.repeat_interleave(cap), assignment_gpu[flat], bucket_gpu[flat],
                            ), dim=1)
                        else:
                            if host_ids is None:
                                raise RuntimeError("Local cascade has no received host candidate payload.")
                            host = host_ids.numpy().reshape(-1)
                            vectors = torch.as_tensor(np.array(items[host], dtype=np.float32), device=device)
                            sparse = torch.as_tensor(np.column_stack((
                                cluster.cpu().numpy().repeat(cap), partition.assignments[host], buckets[host],
                            )), device=device, dtype=torch.long)
                        scores = ranker(query.repeat_interleave(cap, dim=0), vectors, sparse).reshape(query_batch, cap)
                    return ids, scores, rank_candidates(ids, scores, min(10, cap))

                operation(row_batches[0])
                samples = []
                results = []
                for chosen in row_batches[1:]:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    ids, values, ranked = operation(chosen)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
                    results.append((ids.cpu(), values.cpu(), ranked.cpu()))
                outputs[boundary] = results
                mean = float(np.mean(samples))
                conditions.append({
                    "boundary": boundary, "query_batch": query_batch, "candidate_cap": cap,
                    "warmup_query_rows": row_batches[0], "measured_query_row_batches": row_batches[1:],
                    "boundary_order": boundaries,
                    "full_online_latency": {
                        "samples_ms": samples, "sample_count": len(samples),
                        "latency_mean_ms": mean, "latency_p50_ms": float(np.median(samples)),
                        "observed_min_ms": float(min(samples)), "observed_max_ms": float(max(samples)),
                        "offline_queries_per_second": query_batch * 1000 / mean,
                        "scope": "Afteronewarmup,3distinctrealbatches; no tails,confidenceintervalsorSLOclaims.",
                    },
                })
                device_samples.append(device_telemetry(device, require_exclusive=True))
            for integrated, separate in zip(outputs["integrated"], outputs["local-cascade"]):
                if not torch.equal(integrated[0], separate[0]):
                    raise RuntimeError("Stage-boundary control changed upstream ANN candidate IDs.")
                torch.testing.assert_close(integrated[1], separate[1], rtol=1e-5, atol=1e-5)
                if not torch.equal(integrated[2], separate[2]):
                    raise RuntimeError("Same model/candidates changed final top-k at the stage boundary.")
            parity.append({"query_batch": query_batch, "candidate_cap": cap, "same_candidates_scores_final_ids": True})
    result = {
        "complete": True, "status": "ok", "conditions": conditions, "parity": parity,
        "timing_scope": "Actual querytokenization/LMencoding,fullANN,measuredcandidateboundary,feature/textfetch,scoringandfinaltopk.",
        "text_integrated_boundary": "Text requires host IDs and CPUtokenization in both paths; no transformer-kernelfusionclaim.",
        "network": "In-process byte serialization only; no interprocess transport or fabricatedRPCdelay.",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "environment": environment_manifest(),
        "device_memory_samples": device_samples,
        "maximum_observed_device_memory_used_bytes": max(sample["memory_used_bytes"] for sample in device_samples),
        "device_memory_sampling": "Measured startup/co-resident and post-conditionNVMLsamples outside latencytimers; not a continuouspeak.",
        "memory_caveat": "Torchallocatorpeak excludesFAISSexternalCUDAallocations; report deviceNVMLmemory separately.",
    }
    write_json(output / "completed.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong text reranking, fixed-pool quality and measured local cascade boundaries.")
    commands = parser.add_subparsers(dest="operation", required=True)
    text = commands.add_parser("text")
    text.add_argument("--registry", required=True)
    text.add_argument("--model", required=True)
    text.add_argument("--panel", required=True)
    text.add_argument("--candidates", required=True)
    text.add_argument("--source", required=True)
    text.add_argument("--offsets", required=True)
    text.add_argument("--cache", required=True)
    text.add_argument("--smoke", required=True)
    text.add_argument("--output", required=True)
    text.add_argument("--worker", type=int, default=0)
    text.add_argument("--workers", type=int, default=1)
    collect = commands.add_parser("collect")
    collect.add_argument("--panel", required=True)
    collect.add_argument("--candidates", required=True)
    collect.add_argument("--shard", required=True, action="append")
    collect.add_argument("--output", required=True)
    boundary = commands.add_parser("cached-boundary")
    for name in ("model", "checkpoint", "items", "queries", "candidates", "partition", "lengths", "output"):
        boundary.add_argument("--" + name, required=True)
    cached_quality = commands.add_parser("cached-quality")
    for name in ("model", "checkpoint", "items", "queries", "panel", "candidates", "partition", "lengths", "output"):
        cached_quality.add_argument("--" + name, required=True)
    pipeline = commands.add_parser("pipeline")
    for name in ("registry", "encoder", "ranker", "backend", "items", "partition", "panel", "source", "offsets", "cache", "smoke", "output"):
        pipeline.add_argument("--" + name, required=True)
    pipeline.add_argument("--checkpoint")
    pipeline.add_argument("--lengths")
    args = parser.parse_args()
    if args.operation == "text":
        registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        score_text_panel(
            Path(args.panel), Path(args.candidates), Path(args.source), Path(args.offsets),
            registry["models"][args.model], Path(args.cache), Path(args.smoke), Path(args.output),
            args.worker, args.workers,
        )
    elif args.operation == "collect":
        collect_text_scores(Path(args.panel), Path(args.candidates), [Path(path) for path in args.shard], Path(args.output))
    elif args.operation == "cached-boundary":
        benchmark_cached_boundary(
            args.model, Path(args.checkpoint), Path(args.items), Path(args.queries), Path(args.candidates),
            Path(args.partition), Path(args.lengths), Path(args.output),
        )
    elif args.operation == "pipeline":
        benchmark_online_pipeline(
            Path(args.registry), args.encoder, args.ranker, args.backend, Path(args.items), Path(args.partition),
            Path(args.panel), Path(args.source), Path(args.offsets), Path(args.cache), Path(args.smoke), Path(args.output),
            Path(args.checkpoint) if args.checkpoint else None, Path(args.lengths) if args.lengths else None,
        )
    else:
        evaluate_cached_ranker(
            args.model, Path(args.checkpoint), Path(args.items), Path(args.queries), Path(args.panel),
            Path(args.candidates), Path(args.partition), Path(args.lengths), Path(args.output),
        )


if __name__ == "__main__":
    main()
