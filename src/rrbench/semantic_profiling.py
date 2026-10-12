"""Primary B: one backend/session per process; profiling is never a latency sample.

All outputs are new, immutable runs. Pipeline v2 freezes whole raw-forward groups.
Historical batch32 query vectors are provenance/diagnostics only, never active inputs.
Active canonical query files are FP16; encoder/scorer/comparison query tensors stay FP32.
Frozen defaults: ANN/MLP10warmup+50samples; BGE2+8; run sessions0,1,2 separately.
FP32 pipeline references require --reference-only (2GiB gate, not deployment admission).
No serving tails, RPC, energy, MFU, or automatic qualification claims follow.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from rrbench.ann import Partition, SilverANN
from rrbench.io import append_jsonl, bind_run_config, sha256, write_json
from rrbench.semantic_ann import SemanticFaissANN, is_faiss_gpu_allocation_failure, validate_partition
from rrbench.semantic_cascade import device_telemetry, rank_candidates, transfer_candidate_payload, validate_candidates
from rrbench.semantic_profile_contracts import match_latency_reference as match_latency_reference
from rrbench.semantic_profile_contracts import match_quality_reference as match_quality_reference
from rrbench.semantic_profile_contracts import (
    BACKENDS, CLUSTERS, GIB, POLICY, QueryParityError, UnsupportedMemory, canonical_query_drift, canonical_query_payload,
    check_memory, condition_grid, digest, final_depth_metrics,
    match_quality_timing, planned_execution_groups, population_binding, positive_int, query_drift_statistics, query_schedule,
    require_execution_schedule, resolve_protocol_controls, timing_summary, validate_quality_cohort, validate_scorer_contract,
    select_execution_groups, validate_execution_groups, validate_partition_contract,
    validate_source_binding, validate_unit_queries, verify_completion,
)


class ProfilerUnavailable(RuntimeError):
    """No verified CUDA activity; CPU-only fallback cannot certify a GPU trace."""


def score_prefix(retrieved, cap, scorer):
    if retrieved.ndim != 2 or retrieved.dtype != torch.int64 or type(cap) is not int or not 0 < cap <= retrieved.shape[1]:
        raise ValueError("Scored prefix must fit the actual retrieved integer ID matrix.")
    prefix = retrieved[:, :cap]
    scores = scorer(prefix)
    return prefix, scores, rank_candidates(prefix, scores, min(10, cap))


def assert_parity(first, second, query_policy=None):
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Pipeline parity requires all four outputs: IDs, scores, final IDs and canonical query.")
    drift = canonical_query_drift(
        first[3].detach().cpu().numpy(), second[3].detach().cpu().numpy(), query_policy,
    )
    if not torch.equal(first[0], second[0]) or not torch.equal(first[2], second[2]):
        raise ValueError("Matched paths changed candidate/final IDs; quality cannot be joined to timing.")
    if not torch.allclose(first[1], second[1], rtol=1e-5, atol=1e-5):
        raise ValueError("Matched paths changed scorer scores beyond 1e-5.")
    return drift


def observation_arrays(results):
    if not results or any(len(result) != 4 for result in results):
        raise ValueError("Observation payloads must preserve all four pipeline outputs.")
    return {
        key: np.stack([result[column].detach().cpu().numpy() for result in results])
        for column, key in enumerate(("ids", "scores", "ranked", "raw_queries"))
    }


def memory_failure(error):
    return isinstance(error, (UnsupportedMemory, torch.cuda.OutOfMemoryError)) or (
        isinstance(error, RuntimeError) and is_faiss_gpu_allocation_failure(error)
    )


def require_cuda_trace(trace):
    if not any(
        event.get("cat") == "kernel" and event.get("ph") == "X" and event.get("dur", 0) > 0
        for event in trace.get("traceEvents", [])
    ):
        raise ProfilerUnavailable("No CUDA kernel activity was exported; verify CUPTI, not a CPU-only fallback.")


@torch.inference_mode()
def profiled_search(ann, queries, k, probes):
    """Unfiltered Silver adapter; runtime compares it to the untouched search before export."""
    scope = torch.profiler.record_function
    if not isinstance(ann, SilverANN):
        with scope("ann.full_native_search"):
            return ann.search(queries, k, probes)
    with scope("ann.centroid_selection"):
        selected = (queries.float() @ ann.centroids.T).topk(probes, dim=1).indices
        lengths = ann.lengths[selected]
    with scope("ann.dynamic_capacity_host_sync"):
        maximum = int(lengths.sum(dim=1).max().item())
        selected, lengths = selected.contiguous(), lengths.contiguous()
    if ann.dtype.startswith("int8"):
        from rrbench.quantization import quantize_symmetric

        with scope("ann.int8_query_quantization"):
            quantized = quantize_symmetric(queries, "rowwise")
        with scope("ann.selected_scoring"):
            scores, internal = torch.ops.st.fused_kmean_ann(
                ann.offsets, selected, lengths, ann.embeddings, quantized.values, maximum,
            )
        with scope("ann.int8_rescaling"):
            scales = ann.scales
            if scales is None:
                raise RuntimeError("INT8 item scales are missing.")
            if len(scales) > 1:
                scales = scales[internal.long().clamp_min(0), 0]
            scores = scores.float() * scales.reshape(scores.shape if scales.numel() > 1 else (1, 1))
            scores = (scores * quantized.scales).float().masked_fill(internal < 0, -torch.inf)
    else:
        with scope("ann.selected_scoring_including_query_cast"):
            scores, internal = ann.score(queries, selected, lengths, maximum)
    with scope("ann.topk_and_external_id_remap"):
        if k > scores.shape[1]:
            padding = k - scores.shape[1]
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
            internal = torch.nn.functional.pad(internal, (0, padding), value=-1)
        values, positions = scores.topk(k, dim=1)
        chosen = internal.long().gather(1, positions)
        ids = ann.order[chosen.clamp_min(0)].masked_fill(~torch.isfinite(values), -1)
        return values, ids


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sources(args):
    from rrbench.semantic_encoding import validate_completed
    from rrbench.semantic_rankers import query_representation

    query_file = args.queries if args.operation == "ann" else args.quality_queries
    panel_file = args.panel if args.operation == "ann" else (args.quality_panel or args.panel)
    paths = {
        "items": args.items, "item_receipt": args.items.parent / "completed.json",
        "item_ids": args.items.parent / "ids.npy", "merge": args.items.parent / "run-config.json",
        "partition": args.partition, "partition_binding": args.partition.parent / "run-config.json",
        "partition_receipt": args.partition.parent / "validated-partition.json",
        "queries": query_file, "query_receipt": query_file.parent / "completed.json",
        "panel": panel_file, "timing_panel": args.panel,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    item = validate_completed(args.items.parent, 8841823, 0)
    items, queries = np.load(args.items, mmap_mode="r"), np.load(query_file, mmap_mode="r")
    panel, representation = read_json(panel_file), query_representation(query_file)
    if args.items.name != "vectors.npy" or items.shape != (8841823, 1024):
        raise ValueError("Production and cost-shape smoke both require the complete 8,841,823-row source bank.")
    validate_source_binding(
        item, read_json(paths["merge"]), read_json(paths["partition_binding"]),
        read_json(paths["partition_receipt"]), read_json(paths["query_receipt"]), representation, panel, hashes,
    )
    validate_unit_queries(queries, len(panel["query_ids"]))
    with np.load(args.partition) as values:
        partition = Partition(*(values[name] for name in ("centroids", "assignments", "order", "offsets")))
    validate_partition_contract(len(partition.centroids), read_json(paths["partition_binding"]))
    validate_partition(partition, len(items), 1024, CLUSTERS)
    population = population_binding(read_json(args.panel), panel, hashes["timing_panel"], hashes["panel"])
    return items, partition, queries, panel, representation, population, paths


def artifact_paths(output, names):
    paths = {}
    for name in names:
        path = output / name
        if output.resolve() not in path.resolve().parents:
            raise ValueError("Artifact inventory contains an invalid relative path.")
        paths[name] = path
    return paths


def reuse_run(output, binding, expected):
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        bind_run_config(output, binding)
        return None
    if not (output / "completed.json").is_file():
        raise ValueError("Incomplete output is not reusable; choose a new immutable output directory.")
    bind_run_config(output, binding)
    receipt = read_json(output / "completed.json")
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    hashes = {name: sha256(path) for name, path in artifact_paths(output, receipt["artifacts"]).items()}
    verify_completion(receipt, binding, expected, records, hashes)
    return receipt


def finish_run(output, binding, expected, records):
    for record in records:
        append_jsonl(output / "journal.jsonl", record)
    names = {"run-config.json", "journal.jsonl"} | {name for row in records for name in row.get("files", [])}
    hashes = {name: sha256(path) for name, path in artifact_paths(output, names).items()}
    receipt = {
        "complete": True, "conditions": sorted(expected), "binding_sha256": digest(binding),
        "records_sha256": digest(records), "artifacts": hashes,
        "status": "ok" if all(row["status"] == "ok" for row in records) else "unsupported",
    }
    verify_completion(receipt, binding, expected, records, hashes)
    write_json(output / "completed.json", receipt)
    return receipt


def save_arrays(output, stem, **arrays):
    names = []
    for key, values in arrays.items():
        path = output / stem / (key + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npy")
        with partial.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
        names.append(path.relative_to(output).as_posix())
    return names


def observe(device, admission=0, headroom_bytes=8 * GIB):
    value = {
        **device_telemetry(device, require_exclusive=True),
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    check_memory(value, admission, headroom_bytes)
    free, _ = torch.cuda.mem_get_info(device)
    if free < headroom_bytes:
        raise UnsupportedMemory(f"CUDA free memory is below the declared {headroom_bytes}-byte headroom gate.")
    return {
        **value, "cuda_free_bytes": free, "headroom_gate_bytes": headroom_bytes,
        "scope": "Outside-timer NVML samples and Torch peak; external allocations/inter-sample peaks are not continuous measurements.",
    }


def timed_call(operation, device, headroom_bytes=8 * GIB):
    before = observe(device, headroom_bytes=headroom_bytes)
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return result, elapsed, [before, observe(device, headroom_bytes=headroom_bytes)]


def capture_trace(operation, path, device):
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise ProfilerUnavailable("CUDA profiler activities are unavailable; check the CUPTI installation.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True,
    ) as profiler:
        operation()
        torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(path))
    trace = read_json(path)
    require_cuda_trace(trace)
    return {
        "profiler": "torch.profiler CPU/CUDA with verified CUPTI kernel events; no nsys dependency.",
        "cuda_kernel_events": sum(event.get("cat") == "kernel" for event in trace["traceEvents"]),
        "trace_events": [{
            "name": event.key, "calls": event.count, "cpu_self_us": event.self_cpu_time_total,
            "cuda_device_total_us": event.device_time_total,
        } for event in profiler.key_averages()],
        "interpretation": "CPU self and CUDA device observations may overlap/nest; never sum into latency or a critical path.",
        "fused_scope": "Selected-item score primitive excludes ANN top-k/ID remapping and the neural ranker.",
        "hypothesis_only": "Host item()/dynamic capacity synchronization may bottleneck; inspect actual trace before concluding.",
    }


@torch.inference_mode()
def ann_predictions(ann, queries, k, probes, catalog_items, batch=16):
    positive_int(batch, "prediction batch")
    ids = np.empty((len(queries), k), dtype=np.int64)
    scores = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), batch):
        values, found = ann.search(queries[start:start + batch], k, probes)
        if found.dtype != torch.int64 or values.dtype != torch.float32:
            raise ValueError("ANN returned unexpected actual score/ID types.")
        ids[start:start + len(found)] = found.cpu().numpy()
        scores[start:start + len(found)] = values.cpu().numpy()
    validate_candidates(ids, len(queries), catalog_items)
    if not np.isfinite(scores[ids >= 0]).all():
        raise ValueError("Actual ANN predictions have nonfinite valid-item scores.")
    return scores, ids


def check_oracle_binding(binding, hashes):
    expected = {
        "items_sha256": hashes["items"], "queries_sha256": hashes["queries"],
        "k": 1000, "product": "FP32", "tf32": False,
    }
    if binding != expected:
        raise ValueError("Exact oracle is not bound to these canonical source vectors/FP32 products.")


class PipelineOperation:
    """Actual encode -> depth1000 -> prefix -> boundary -> scorer -> top10 operation."""

    def __init__(self, ann, encoder, ranker, items, partition, queries, panel, device, cap, probes, store=None, lengths=None):
        self.ann, self.encoder, self.ranker = ann, encoder, ranker
        self.items, self.partition, self.historical_queries, self.panel = items, partition, queries, panel
        self.device, self.cap, self.probes, self.store = device, cap, probes, store
        if cap not in (50, 500) or probes not in (64, 256):
            raise ValueError("Pipeline cap/probes must belong to the frozen deployment grid.")
        if store is None:
            if lengths is None or lengths.shape != (len(items),) or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 0).any():
                raise ValueError("Cached features require full-catalog positive integer token lengths.")
            self.buckets = np.minimum(np.log2(lengths).astype(np.int64), 31)
            self.item_gpu = torch.as_tensor(np.array(items), device=device)
            self.assignment_gpu = torch.as_tensor(partition.assignments, device=device)
            self.bucket_gpu = torch.as_tensor(self.buckets, device=device)
            self.centroids_gpu = torch.as_tensor(partition.centroids, device=device)

    @torch.inference_mode()
    def __call__(self, rows, boundary, raw=True, profile=False):
        from rrbench.semantic_models import mrl_prefix

        if not raw:
            raise ValueError("Pipeline v2 requires raw forwards; historical query vectors cannot be substituted.")
        scope = torch.profiler.record_function if profile else lambda name: nullcontext()
        texts = [self.panel["queries"][int(row)] for row in rows]
        with scope("pipeline.raw_query_tokenization_encoding_mrl"):
            if self.encoder is None:
                raise ValueError("Raw pipeline operation requires the actual query encoder.")
            query, _ = self.encoder.encode(texts, query=True)
            query = mrl_prefix(query, self.items.shape[1]).to(torch.float16).float()
        with scope("pipeline.ann_depth1000"):
            if profile:
                _, found = profiled_search(self.ann, query, 1000, self.probes)
            else:
                _, found = self.ann.search(query, 1000, self.probes)

        def scorer(prefix):
            with scope("pipeline.candidate_boundary"):
                ids, host = transfer_candidate_payload(prefix, boundary, self.device)
                if (ids < 0).any():
                    raise ValueError("Actual ANN did not produce a full valid scored prefix.")
            with scope("pipeline.feature_or_text_fetch_and_scorer"):
                output = []
                cluster = (query @ self.centroids_gpu.T).argmax(1) if self.store is None else None
                for row, text in enumerate(texts):
                    if self.store is not None:
                        cpu = host[row].numpy() if host is not None else ids[row].cpu().numpy()
                        documents = self.store.fetch(cpu)
                        values = [self.ranker.score([text] * len(documents[start:start + 64]),
                                  documents[start:start + 64])[0] for start in range(0, self.cap, 64)]
                        output.append(torch.cat(values))
                    else:
                        if host is None:
                            vectors = self.item_gpu[ids[row]].float()
                            sparse = torch.stack((
                                cluster[row].expand(self.cap), self.assignment_gpu[ids[row]], self.bucket_gpu[ids[row]],
                            ), dim=1)
                        else:
                            cpu = host[row].numpy()
                            vectors = torch.as_tensor(np.array(self.items[cpu], dtype=np.float32), device=self.device)
                            sparse = torch.as_tensor(np.column_stack((
                                np.full(self.cap, int(cluster[row].item())), self.partition.assignments[cpu], self.buckets[cpu],
                            )), device=self.device, dtype=torch.int64)
                        output.append(self.ranker(query[row].expand(self.cap, -1), vectors, sparse))
                return torch.stack(output)

        with scope("pipeline.prefix_scoring_and_final_top10"):
            _, scores, final = score_prefix(found, self.cap, scorer)
        return found, scores, final, query


@torch.inference_mode()
def execute_quality_groups(operation, manifest, after_group=None):
    validate_execution_groups(manifest)
    output = None
    for rows in manifest["groups"]:
        result = operation(rows, "integrated", raw=True)
        if len(result) != 4:
            raise ValueError("Raw grouped quality requires all four actual pipeline outputs.")
        arrays = tuple(value.detach().cpu().numpy() for value in result)
        if any(value.ndim != 2 or len(value) != len(rows) for value in arrays):
            raise ValueError("Raw quality outputs do not align with the complete execution group.")
        if output is None:
            output = [np.empty((manifest["query_count"], value.shape[1]), dtype=value.dtype) for value in arrays]
        for target, value in zip(output, arrays):
            if value.dtype != target.dtype or value.shape[1] != target.shape[1]:
                raise ValueError("Raw quality output shape/dtype changed between execution groups.")
            target[rows] = value
        if after_group is not None:
            after_group()
    return tuple(output)


def build_binding(paths, parameters):
    return {
        "sources": {name: sha256(path) for name, path in paths.items()},
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()},
        "scope": "Post-test exploratory; no test tuning, timing confidence intervals/tails, automatic qualification, RPC, MFU or energy.",
        "precision": "Canonical FP16 source -> FP32 queries; backend-native ANN arithmetic; TF32 off.",
    }


def profile_record(operation, args, condition, device):
    path = args.output / "profiles" / (condition + ".json")
    summary = capture_trace(operation, path, device)
    summary_path = path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return {
        "kind": "profile", "profile": summary, "trace_file": path.relative_to(args.output).as_posix(),
        "files": [path.relative_to(args.output).as_posix(), summary_path.relative_to(args.output).as_posix()],
    }


def check_adapter(ann, query, probes, output, stem):
    plain, annotated = ann.search(query, 1000, probes), profiled_search(ann, query, 1000, probes)
    if not torch.equal(plain[1], annotated[1]):
        raise ValueError("Profiler adapter changed external ANN IDs.")
    torch.testing.assert_close(plain[0], annotated[0], atol=0, rtol=0)
    return save_arrays(output, stem, plain_ids=plain[1].cpu().numpy(), traced_ids=annotated[1].cpu().numpy(),
                       plain_scores=plain[0].cpu().numpy(), traced_scores=annotated[0].cpu().numpy())


def run_ann(args, ann, queries, panel, partition, schedules, base, oracle):
    from rrbench.metrics import recall_at_k
    from rrbench.semantic_ir import panel_metrics

    records = []
    for probes in args.probes:
        scores, ids = ann_predictions(ann, queries, args.k, probes, len(partition.order))
        files = save_arrays(args.output, f"predictions/p{probes}-k{args.k}", ids=ids, scores=scores)
        quality = panel_metrics(panel, ids)
        for batch in args.batches:
            condition = f"{args.backend}-p{probes}-k{args.k}-b{batch}"
            schedule = schedules[batch]
            warmup = [queries[rows] for rows in schedule["warmup_rows"]]
            selected = [queries[rows] for rows in schedule["sample_rows"]]
            lists = (selected[0] @ torch.as_tensor(partition.centroids, device=queries.device).T).topk(probes, dim=1).indices.cpu().numpy()
            lengths = partition.lengths[lists]
            torch.cuda.reset_peak_memory_stats(queries.device)
            for query in warmup:
                ann.search(query, args.k, probes)
            telemetry = [observe(queries.device)]
            row = {
                **base, "condition": condition, "status": "ok", "batch": batch, "probes": probes,
                "k": args.k, "schedule": schedule, "files": list(files), "relevance": quality,
                "selected_list_lengths": lengths.tolist(), "selected_total_lengths": lengths.sum(1).tolist(),
                "list_scope": "Outside-timer shared-centroid FP32 reference selection; not FAISS internal phase timings.",
            }
            if oracle is not None:
                row["geometric_recall_at_k"] = float(recall_at_k(ids, oracle, args.k).mean())
            if args.profile:
                parity_files = check_adapter(ann, selected[0], probes, args.output, f"profiles/{condition}-adapter")

                def workload():
                    for query in selected:
                        profiled_search(ann, query, args.k, probes)

                profiled = profile_record(workload, args, condition, queries.device)
                row.update({**profiled, "files": row["files"] + parity_files + profiled["files"]})
            else:
                samples, measured_ids, measured_scores = [], [], []
                for rows, query in zip(schedule["sample_rows"], selected):
                    (values, found), elapsed, observed = timed_call(lambda: ann.search(query, args.k, probes), queries.device)
                    actual_ids, actual_scores = found.cpu().numpy(), values.cpu().numpy()
                    if not np.array_equal(actual_ids, ids[rows]) or not np.allclose(actual_scores, scores[rows], atol=1e-5, rtol=1e-5):
                        raise ValueError("Timed ANN batch changed canonical-panel predictions; no matched quality claim.")
                    samples.append(elapsed)
                    measured_ids.append(actual_ids)
                    measured_scores.append(actual_scores)
                    telemetry.extend(observed)
                row.update(kind="latency", latency=timing_summary(samples, batch))
                row["files"] += save_arrays(args.output, f"observations/{condition}", ids=np.stack(measured_ids), scores=np.stack(measured_scores))
            row["device_memory_samples"] = telemetry + [observe(queries.device)]
            records.append(row)
    return records


def pipeline_specifications(args, representation, partition, paths):
    from rrbench.semantic_models import configuration_digest, validate_model_files, validate_smoke_contract
    from rrbench.semantic_rankers import validate_checkpoint_features

    paths.update({name: getattr(args, name) for name in ("registry", "smoke", "source", "offsets")})
    registry, smoke = read_json(args.registry), read_json(args.smoke)
    if smoke.get("complete") is not True:
        raise ValueError("Pipeline requires completed pinned-model smoke attestation.")
    specs, attested = registry["models"], {row["key"]: row for row in smoke["models"]}
    if args.ranker == "bge-m3-rerank":
        validate_scorer_contract(args.ranker, specs[args.ranker])
    keys = [args.encoder] + ([args.ranker] if args.ranker == "bge-m3-rerank" else [])
    for key in keys:
        validate_smoke_contract(specs[key], attested[key])
        validate_model_files(specs[key], args.cache, attested[key]["model_files"])
        positive_int(attested[key]["parameters"], "attested parameter count")
    encoder = specs[args.encoder]
    if any(representation[key] != encoder[key] for key in ("model", "revision", "query_instruction")) or (
        representation["specification_sha256"] != configuration_digest(encoder)
    ):
        raise ValueError("Actual pipeline encoder revision/instruction differs from cached item/query representation.")
    state = None
    if args.ranker == "semantic-mlp":
        paths.update(checkpoint=args.checkpoint, lengths=args.lengths, checkpoint_receipt=args.checkpoint.parent / "completed.json")
        complete = read_json(paths["checkpoint_receipt"])
        if complete.get("complete") is not True or complete["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Frozen ranker checkpoint completion/hash changed.")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        validate_scorer_contract(args.ranker, state["binding"])
        validate_checkpoint_features(state["binding"], args.items, args.partition, args.lengths, representation)
        if (state["binding"]["model"], state["binding"]["dimension"], state["binding"]["clusters"]) != (
            args.ranker, 1024, len(partition.centroids),
        ):
            raise ValueError("Frozen scorer architecture/dimension/codebook differs from this deployment.")
    return specs, attested, state


def run_pipeline(args, operation, panel, population, schedules, base, deployment, execution_groups):
    from rrbench.semantic_ir import panel_metrics

    boundaries = ["integrated", "local-cascade"] if args.boundary == "both" else [args.boundary]
    count, batch, device, headroom = len(panel["query_ids"]), args.batch, operation.device, args.headroom_bytes
    schedule = schedules[batch]
    require_execution_schedule(execution_groups, schedule, args.warmup, args.samples)
    quality_arrays = execute_quality_groups(
        operation, execution_groups, after_group=lambda: observe(device, headroom_bytes=headroom),
    )
    query_storage, query_reference = canonical_query_payload(quality_arrays[3])
    quality_arrays = (*quality_arrays[:3], query_reference)
    validate_candidates(quality_arrays[0], count, len(operation.items))
    files = save_arrays(args.output, "quality", ids=quality_arrays[0], scores=quality_arrays[1],
                        ranked=quality_arrays[2], raw_queries=query_storage)
    historical_drift = {
        **query_drift_statistics(quality_arrays[3], np.array(operation.historical_queries, dtype=np.float32)),
        "comparison_role": "Historical batch32 diagnostic only; not a gate or invariance claim.",
    }
    write_json(args.output / "historical-query-drift.json", historical_drift)
    files += ["execution-groups.json", "historical-query-drift.json"]
    query_sha256 = sha256(args.output / "quality" / "raw_queries.npy")
    deployment = {
        **deployment, "candidate_ids_sha256": sha256(args.output / "quality" / "ids.npy"),
        "execution_queries_sha256": query_sha256, "queries_sha256": query_sha256,
        "execution_query_storage_dtype": str(query_storage.dtype),
        "execution_groups_sha256": execution_groups["execution_groups_sha256"],
    }
    point = {"deployment": deployment, "relevance": final_depth_metrics(panel_metrics(panel, quality_arrays[2])), "ranking_depth": 10}
    match_quality_timing(point, {"deployment": deployment})

    def check_result(result, rows, boundary, phase):
        copied = tuple(value.detach().cpu() for value in result)
        reference = tuple(torch.from_numpy(value[rows]) for value in quality_arrays)
        try:
            drift = assert_parity(copied, reference, args.canonical_query_parity)
        except QueryParityError as error:
            stem = f"rejected-query/{phase}-{boundary}"
            save_arrays(args.output, stem, **observation_arrays([copied]), cached_queries=quality_arrays[3][rows])
            write_json(args.output / stem / "drift.json", {"quality_rows": rows, **error.observation})
            raise
        return copied, {"quality_rows": rows, **drift}

    probe_groups = [execution_groups["groups"][index] for index in execution_groups["probe_group_indices"]]
    probe_rows = [row for group in probe_groups for row in group]
    parity = {boundary: [[], [], [], []] for boundary in ("integrated", "local-cascade")}
    for rows in probe_groups:
        for boundary in parity:
            result, _ = check_result(operation(rows, boundary), rows, boundary, "group-probe")
            for column, value in zip(parity[boundary], result):
                column.append(value.numpy())
        observe(device, headroom_bytes=headroom)
    probe_drift = {}
    for boundary, columns in parity.items():
        arrays = dict(zip(
            ("ids", "scores", "ranked", "raw_queries"), (np.concatenate(column) for column in columns),
        ))
        probe_drift[boundary] = canonical_query_drift(
            arrays["raw_queries"], quality_arrays[3][probe_rows], args.canonical_query_parity,
        )
        files += save_arrays(args.output, f"parity/{boundary}", **arrays)
    files += save_arrays(args.output, "parity/source", rows=np.asarray(probe_rows, dtype=np.int64),
                         quality_queries=quality_arrays[3][probe_rows])
    chosen = schedule["warmup_rows"] + schedule["sample_rows"]
    samples, results, telemetry, query_drift = ({boundary: [] for boundary in boundaries} for _ in range(4))
    torch.cuda.reset_peak_memory_stats(device)
    for index, rows in enumerate(chosen):
        order = boundaries if (int(digest([args.session, batch])[:2], 16) + index) % 2 else list(reversed(boundaries))
        for boundary in order:
            if index < args.warmup or args.profile:
                result = operation(rows, boundary)
                if args.profile and index >= args.warmup:
                    copied, drift = check_result(result, rows, boundary, "pretrace")
                    results[boundary].append(copied)
                    query_drift[boundary].append(drift)
                observe(device, headroom_bytes=headroom)
            else:
                result, elapsed, observed = timed_call(lambda: operation(rows, boundary), device, headroom)
                copied, drift = check_result(result, rows, boundary, "timed")
                samples[boundary].append(elapsed)
                results[boundary].append(copied)
                query_drift[boundary].append(drift)
                telemetry[boundary].extend(observed)
    records = []
    for boundary in boundaries:
        condition = f"{args.backend}-p{args.probes}-k1000-b{batch}-{args.ranker}-cap{args.candidate_cap}-{boundary}"
        row = {
            **base, **point, "condition": condition, "status": "ok", "batch": batch, "boundary": boundary,
            "population": population, "schedule": schedule, "files": list(files), "parity_queries": len(probe_rows),
            "historical_query_drift": historical_drift, "quality_scope": execution_groups["scope"],
            "unprofiled_pretrace_validation_pass": args.profile,
            "observation_phase": "pretrace" if args.profile else "timed",
            "query_parity": {"initial_group_probe": probe_drift[boundary], "observed_batches": query_drift[boundary]},
            "boundary_order_rule": "Alternate every batch from SHA256[session,batch] parity, including warmup; same across arms.",
            "scoring_protocol": "Per-query scorer calls; text pairs chunked64 identically across query batches.",
            "boundary_scope": "In-process serialization/feature fetch; text requires host IDs/tokenization in both paths; no RPC.",
        }
        row["files"] += save_arrays(args.output, f"observations/{boundary}", **observation_arrays(results[boundary]))
        if args.profile:
            query = torch.as_tensor(quality_arrays[3][chosen[-1]], device=device)
            row["files"] += check_adapter(operation.ann, query, args.probes, args.output, f"profiles/{condition}-adapter")

            def workload():
                for rows in chosen[args.warmup:]:
                    operation(rows, boundary, profile=True)

            profiled = profile_record(workload, args, condition, device)
            row.update({**profiled, "files": row["files"] + profiled["files"]})
        else:
            row.update(kind="latency", latency=timing_summary(samples[boundary], batch))
        row["device_memory_samples"] = telemetry[boundary] + [observe(device, headroom_bytes=headroom)]
        records.append(row)
    return records


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("ann", "pipeline"):
        child = commands.add_parser(operation)
        for name in ("items", "partition", "panel", "output"):
            child.add_argument("--" + name, type=Path, required=True)
        child.add_argument("--backend", choices=BACKENDS, required=True)
        child.add_argument("--session", type=int, required=True)
        protocol = "semantic-extension-v1.json" if operation == "ann" else "semantic-extension-pipeline-v2.json"
        child.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / protocol)
        child.add_argument("--warmup", type=int, help="Protocol default: ANN/MLP10, BGE2.")
        child.add_argument("--samples", type=int, help="Protocol default: ANN/MLP50, BGE8.")
        child.add_argument("--profile", action="store_true", help="Separate trace-only run; no latency samples.")
        if operation == "ann":
            child.add_argument("--queries", type=Path, required=True)
            child.add_argument("--batches", type=int, nargs="+", default=[1, 16])
            child.add_argument("--probes", type=int, nargs="+", default=[64, 256])
            child.add_argument("--k", type=int, default=1000)
            child.add_argument("--allow-batch64", action="store_true")
            child.add_argument("--smoke", action="store_true", help="Exactly100 real queries; full catalog unchanged.")
            child.add_argument("--oracle", type=Path)
        else:
            child.add_argument("--reference-only", action="store_true", help="FAISS FP32 only; 2GiB reference gate, not8GiB deployment admission.")
            child.add_argument("--cost-shape-smoke", action="store_true", help="Explicit100-row geometry with1warmup/5samples; excluded from deployment comparisons.")
            for name in ("registry", "source", "offsets", "cache", "smoke"):
                child.add_argument("--" + name, type=Path, required=True)
            child.add_argument("--historical-queries", "--quality-queries", dest="quality_queries", type=Path, required=True,
                               help="Attested historical query bank for lineage/drift diagnostics only, never active v2 quality or timing.")
            for name in ("quality-panel", "checkpoint", "lengths"):
                child.add_argument("--" + name, type=Path)
            child.add_argument("--encoder", choices=["qwen8"], default="qwen8")
            child.add_argument("--ranker", choices=["semantic-mlp", "bge-m3-rerank"], required=True)
            child.add_argument("--probes", type=int, choices=[64, 256], default=64)
            child.add_argument("--batch", type=int, choices=[1, 16], default=1)
            child.add_argument("--retrieval-depth", type=int, choices=[1000], default=1000)
            child.add_argument("--candidate-cap", type=int, choices=[50, 500], required=True)
            child.add_argument("--boundary", choices=["both", "integrated", "local-cascade"], default="both")
    return parser


def parse_arguments(argv=None):
    args = make_parser().parse_args(argv)
    vars(args).update(resolve_protocol_controls(vars(args), read_json(args.protocol)["profiling"]))
    return args


@torch.inference_mode()
def main(argv=None):
    from rrbench.runtime import configure, environment_manifest, require_cuda
    from rrbench.semantic_ann import validate_oracle
    from rrbench.semantic_cascade import pipeline_memory_bytes
    from rrbench.semantic_ir import PassageStore
    from rrbench.semantic_models import SemanticEncoder, TextReranker
    from rrbench.semantic_rankers import make_cached_ranker

    args = parse_arguments(argv)
    is_ann = args.operation == "ann"
    batches, probes = (args.batches, args.probes) if is_ann else ([args.batch], [args.probes])
    expected = condition_grid(args.backend, probes, args.k if is_ann else args.retrieval_depth, batches, is_ann and args.allow_batch64)
    if not is_ann:
        if args.ranker == "semantic-mlp" and (args.checkpoint is None or args.lengths is None):
            raise ValueError("Cached pipeline requires its frozen checkpoint and full token lengths.")
        boundaries = ["integrated", "local-cascade"] if args.boundary == "both" else [args.boundary]
        expected = [f"{expected[0]}-{args.ranker}-cap{args.candidate_cap}-{boundary}" for boundary in boundaries]
    timing_panel = read_json(args.panel)
    schedules = {batch: query_schedule(timing_panel, args.session, batch, args.warmup, args.samples) for batch in batches} if is_ann else {}
    if is_ann and args.smoke and len(timing_panel["query_ids"]) != 100:
        raise ValueError("Declared cost-shape smoke requires exactly100 real queries; use explicit fitting sample counts.")
    items, partition, queries, panel, representation, population, paths = load_sources(args)
    paths["protocol"] = args.protocol
    if not is_ann:
        validate_quality_cohort(panel, smoke=args.cost_shape_smoke)
        execution_groups = planned_execution_groups(panel, population, args.session, args.batch, args.cost_shape_smoke)
        schedules[args.batch] = select_execution_groups(execution_groups, args.warmup, args.samples)
        paths["historical_queries"] = paths.pop("queries")
        paths["historical_query_receipt"] = paths.pop("query_receipt")
    if max(probes) > len(partition.centroids) or len(panel["query_ids"]) < 100:
        raise ValueError("Full centroid coverage and at least100 real source queries are required.")
    for name in ("semantic_profiling", "semantic_profile_contracts", "ann", "semantic_ann", "semantic_cascade",
                 "semantic_models", "semantic_rankers", "semantic_ir", "semantic_encoding", "quantization", "runtime", "io"):
        paths["code_" + name] = Path(__file__).with_name(name + ".py")
    oracle = None
    if is_ann and args.oracle:
        paths.update({f"oracle_{name}": args.oracle / relative for name, relative in {
            "binding": "run-config.json", "receipt": "completed.json", "panel_receipt": "panel/completed.json",
            "panel_binding": "panel/run-config.json", "ids": "panel/ids.npy", "scores": "panel/scores.npy",
        }.items()})
    if not is_ann:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        specs, attested, state = pipeline_specifications(args, representation, partition, paths)
    binding = build_binding(paths, vars(args))
    binding.update(representation=representation, population=population, conditions=expected, catalog_items=8841823)
    if not is_ann:
        binding["execution_groups_sha256"] = execution_groups["execution_groups_sha256"]
    reused = reuse_run(args.output, binding, expected)
    if reused is not None:
        return reused
    if not is_ann:
        write_json(args.output / "execution-groups.json", execution_groups)
    if is_ann and args.oracle:
        check_oracle_binding(read_json(paths["oracle_binding"]), binding["sources"])
        receipt = read_json(paths["oracle_receipt"])
        if receipt.get("complete") is not True or receipt["panel_receipt_sha256"] != sha256(paths["oracle_panel_receipt"]):
            raise ValueError("Exact oracle panel receipt checksum changed.")
        validate_oracle(args.oracle / "panel", len(queries), len(items), 1024, 1000)
        oracle = np.load(paths["oracle_ids"], mmap_mode="r")
    configure(17, 4)
    device = require_cuda()
    base = {"sources_sha256": digest(binding["sources"]), "sources": binding["sources"], "representation": representation,
            "catalog_items": len(items), "environment": environment_manifest(), "smoke_only": args.smoke if is_ann else args.cost_shape_smoke,
            "session": args.session, "clusters": args.clusters, "point_role": args.point_role, "headroom_gate_bytes": args.headroom_bytes,
            "planned_sessions": args.planned_sessions, "recommended_backend_order": args.recommended_backend_order,
            "count_scope": {"warmup_batches": args.warmup, "measured_batches": args.samples, "policy": args.count_protocol},
            "backend": args.backend, "query_dtype": "torch.float32",
            "timing_scope": ("Resident preencoded-query ANN only; staging/build/quality excluded." if is_ann else
                             "Raw tokenization/encoding/MRL, ANN depth1000, prefix, local boundary/fetch, scorer, finaltop10.")}
    try:
        storage = 1 if "int8" in args.backend else (2 if args.backend.endswith("fp16") else 4)
        admission = items.size * storage + 8 * GIB
        if not is_ann:
            admission = pipeline_memory_bytes(len(items), 1024, storage, attested[args.encoder]["parameters"],
                                             attested[args.ranker]["parameters"] if state is None else 0, state is not None)
            admission += args.headroom_bytes - 8 * GIB
        admission += len(items) * 24 + partition.centroids.nbytes
        base["admission_estimate_bytes"] = admission
        base["initial_device"] = observe(device, admission, args.headroom_bytes)
        started = time.perf_counter()
        ann = (SemanticFaissANN(items, partition, True, args.backend.endswith("fp16")) if args.backend.startswith("faiss")
               else SilverANN(items, partition, args.backend.removeprefix("silver-")))
        torch.cuda.synchronize(device)
        base["index_build_seconds_excluded_from_online"] = time.perf_counter() - started
        actual = ann.code_bytes if isinstance(ann, SemanticFaissANN) else ann.embeddings.element_size() * 1024
        if actual != 1024 * storage:
            raise ValueError("Actual ANN storage/code bytes differ from the named backend.")
        base["actual_index_code_bytes_per_item"] = actual
        base["actual_index_value_dtype"] = str(ann.embeddings.dtype) if isinstance(ann, SilverANN) else ("float16" if storage == 2 else "float32")
        if is_ann:
            query_gpu = torch.as_tensor(np.array(queries, dtype=np.float32), device=device)
            records = run_ann(args, ann, query_gpu, panel, partition, schedules, base, oracle)
        else:
            store = PassageStore(args.source, args.offsets)
            if len(store.offsets) - 1 != len(items):
                raise ValueError("Text offsets and full ANN item-ID universe differ.")
            encoder = SemanticEncoder(specs[args.encoder], args.cache, device)
            ranker = (make_cached_ranker(args.ranker, 1024, len(partition.centroids), device, 17).eval() if state is not None
                      else TextReranker(specs[args.ranker], args.cache, device))
            if state is not None:
                ranker.load_state_dict(state["model"])
            operation = PipelineOperation(ann, encoder, ranker, items, partition, queries, panel, device,
                                          args.candidate_cap, args.probes, store=store if state is None else None,
                                          lengths=np.load(args.lengths, mmap_mode="r") if state is not None else None)
            torch.cuda.reset_peak_memory_stats(device)
            base["co_resident_device"] = observe(device, headroom_bytes=args.headroom_bytes)
            deployment = {
                **{key + "_sha256": binding["sources"][key] for key in ("items", "partition", "historical_queries", "source", "offsets")},
                "query_execution_policy": args.query_execution_policy, "query_batch": args.batch, "session": args.session,
                "quality_panel_sha256": binding["sources"]["panel"], "representation": representation,
                "backend": args.backend, "probes": args.probes, "retrieval_depth": 1000, "candidate_cap": args.candidate_cap,
                "scorer": {"name": args.ranker, "checkpoint_sha256": binding["sources"].get("checkpoint"),
                           "specification": specs[args.ranker] if state is None else state["binding"],
                           "model_files": attested[args.ranker]["model_files"] if state is None else None},
                "candidate_policy": POLICY, "positive_injection": False, "final_k": 10, "tf32": False,
                "clusters": args.clusters, "canonical_query_parity": args.canonical_query_parity,
                "scoring_protocol": "per-query; text pair-batch64; MLP FP32; text BF16; SDPA",
            }
            records = run_pipeline(args, operation, panel, population, schedules, base, deployment, execution_groups)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if not memory_failure(error) and not isinstance(error, ProfilerUnavailable):
            raise
        partial = [path.relative_to(args.output).as_posix() for path in args.output.rglob("*")
                   if path.is_file() and path.name != "run-config.json"]
        records = [{**base, "condition": name, "status": "unsupported_memory" if memory_failure(error) else "unsupported_profiler",
                    "reason": str(error), "files": partial, "failure_scope": "No successful whole-run timing/quality claim."}
                   for name in expected]
    if any(sha256(path) != binding["sources"][name] for name, path in paths.items()):
        raise ValueError("Input bytes changed during the run; completion refused.")
    result = finish_run(args.output, binding, expected, records)
    print(json.dumps({"status": result["status"], "output": str(args.output), "conditions": expected}))
    return result


if __name__ == "__main__":
    main()
