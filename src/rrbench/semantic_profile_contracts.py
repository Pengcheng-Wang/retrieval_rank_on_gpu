"""Pure Primary B scheduling, provenance, and completion contracts (no GPU or file IO)."""
from __future__ import annotations

import hashlib
import json

import numpy as np

BACKENDS = ("faiss-gpu-fp32", "silver-fp32", "faiss-gpu-fp16", "silver-fp16", "silver-int8-rowwise")
PIPELINE_CANDIDATES = ("faiss-gpu-fp16", "silver-int8-rowwise")
SESSIONS = (0, 1, 2)
CLUSTERS = 16384
QUERY_PARITY_LIMITS = {"maximum_absolute_difference": 1e-4, "minimum_cosine_similarity": 0.99999}
QUERY_EXECUTION_POLICY = "raw-execution-groups-v2"
EXECUTION_GEOMETRY = {
    "policy": QUERY_EXECUTION_POLICY, "production_rows": 1097, "production_group_warmup_batches": 10,
    "production_group_measured_batches": 50, "smoke_rows": 100,
    "smoke_group_warmup_batches": 1, "smoke_group_measured_batches": 5,
}
POLICY = "retrieve-depth-then-prefix-before-scoring"
GIB = 1 << 30
NORMALIZATION = "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage."


class UnsupportedMemory(RuntimeError):
    """An explicit admission/headroom failure, not a successful measurement."""


class QueryParityError(ValueError):
    def __init__(self, observation):
        self.observation = observation
        super().__init__("Canonical query drift exceeded frozen limits: " + json.dumps(observation, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a finite positive integer.")
    return value


def validate_session(session):
    if type(session) is not int or session not in SESSIONS:
        raise ValueError("Session must be one of the frozen restarted sessions0,1,2.")


def validate_partition_contract(actual_clusters, binding):
    if type(actual_clusters) is not int or actual_clusters != CLUSTERS or type(binding.get("clusters")) is not int or binding["clusters"] != CLUSTERS:
        raise ValueError("Frozen partition requires exactly16384 centroids AND binding clusters16384.")


def query_drift_statistics(actual, cached):
    if actual.ndim != 2 or actual.shape != cached.shape or not actual.size:
        raise ValueError("Canonical query comparisons require aligned nonempty matrices.")
    norms = []
    for values in (actual, cached):
        if (
            values.dtype != np.float32 or not np.isfinite(values).all()
            or (np.abs(values) > np.finfo(np.float16).max).any()
            or not np.array_equal(values, values.astype(np.float16).astype(np.float32))
        ):
            raise ValueError("Canonical query values must be finite FP16-rounded values represented in FP32.")
        lengths = np.linalg.norm(values, axis=1)
        if not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError("Canonical query comparisons require nonzero finite norms.")
        norms.append(lengths)
    cosine = np.clip(np.sum(actual * cached, axis=1) / (norms[0] * norms[1]), -1.0, 1.0)
    return {
        "maximum_absolute_difference": float(np.abs(actual - cached).max()),
        "minimum_cosine_similarity": float(cosine.min()),
        "changed_rounded_values": int(np.count_nonzero(actual != cached)),
        "query_count": len(actual), "value_count": int(actual.size),
        "scope": "FP32 comparisons of FP16-canonical values; cosine clipped to[-1,1] for rounding; statistics alone imply no acceptance.",
    }


def canonical_query_drift(actual, cached, limits=None):
    limits = dict(QUERY_PARITY_LIMITS) if limits is None else dict(limits)
    if limits != QUERY_PARITY_LIMITS:
        raise ValueError("Canonical query parity limits differ from the frozen pre-execution protocol.")
    observation = {**query_drift_statistics(actual, cached), "limits": limits}
    observation["scope"] += " Tolerance acceptance is not bit-exact."
    if (
        observation["maximum_absolute_difference"] > limits["maximum_absolute_difference"]
        or observation["minimum_cosine_similarity"] < limits["minimum_cosine_similarity"]
    ):
        raise QueryParityError(observation)
    return observation


def canonical_query_payload(queries):
    """Store already canonical values losslessly; FP32 promotion is a cast, not a byte view."""
    query_drift_statistics(queries, queries)
    stored = queries.astype(np.float16)
    return stored, stored.astype(np.float32)


def resolve_protocol_controls(parameters, profiling):
    if (
        profiling["ann_backends"] != list(BACKENDS) or profiling["restarted_sessions"] != list(SESSIONS)
        or profiling["pipeline_candidates"] != list(PIPELINE_CANDIDATES)
        or profiling["probes"] != [64, 256] or profiling["retrieval_depth"] != 1000
        or profiling["dimension"] != 1024 or profiling["proposed_observed_headroom_gib"] != 8
        or profiling.get("clusters") != CLUSTERS
        or {key: profiling.get("canonical_query_parity", {}).get(key) for key in QUERY_PARITY_LIMITS} != QUERY_PARITY_LIMITS
    ):
        raise ValueError("Frozen profiling protocol differs from this harness's source/grid contracts.")
    session, backend, operation = parameters["session"], parameters["backend"], parameters["operation"]
    validate_session(session)
    if backend not in BACKENDS or operation not in ("ann", "pipeline"):
        raise ValueError("Undeclared profiling operation/backend.")
    if operation == "pipeline" and any(
        profiling.get("pipeline_query_execution", {}).get(key) != value for key, value in EXECUTION_GEOMETRY.items()
    ):
        raise ValueError("Pipeline requires explicit raw-execution-groups-v2 protocol; v1 historical-cache execution is rejected.")
    reference = parameters.get("reference_only", False)
    if type(reference) is not bool or (reference and (operation != "pipeline" or backend != "faiss-gpu-fp32")):
        raise ValueError("Only the pipeline FAISS FP32 arm can be explicitly reference-only.")
    if operation == "pipeline" and not reference and backend not in PIPELINE_CANDIDATES:
        raise ValueError("Pipeline deployment candidates are FAISS half/Silver rowwiseINT8; FP32 needs --reference-only.")
    text = operation == "pipeline" and parameters["ranker"] == "bge-m3-rerank"
    defaults = {
        "warmup": profiling["text_warmup_batches" if text else "warmup_batches"],
        "samples": profiling["text_measured_batches" if text else "measured_batches"],
    }
    if operation == "pipeline" and parameters.get("cost_shape_smoke", False):
        defaults = {"warmup": 1, "samples": 5}
    controls = {key: defaults[key] if parameters.get(key) is None else parameters[key] for key in defaults}
    for key, value in controls.items():
        if positive_int(value, key) > 100:
            raise ValueError("Bounded warmup/samples must each be at most100.")
    order = list(BACKENDS) if session == 0 else (list(reversed(BACKENDS)) if session == 1 else list(BACKENDS[2:] + BACKENDS[:2]))
    return {
        **controls, "planned_sessions": list(SESSIONS), "recommended_backend_order": order,
        **({"query_execution_policy": QUERY_EXECUTION_POLICY} if operation == "pipeline" else {}),
        "clusters": CLUSTERS, "canonical_query_parity": dict(QUERY_PARITY_LIMITS),
        "headroom_bytes": (2 if reference else 8) * GIB,
        "point_role": "reference_only" if reference else ("deployment_candidate" if operation == "pipeline" else "ann_characterization"),
        "count_protocol": "frozen_defaults" if controls == defaults else "explicit_count_override",
    }


def validate_quality_cohort(panel, smoke=False):
    ids = panel["query_ids"]
    if smoke:
        if len(ids) != 100:
            raise ValueError("Cost-shape smoke requires exactly100 real rows and is not a deployment-quality point.")
        return
    groups = panel.get("evaluation_groups", [])
    if (
        len(ids) != 1097 or len(set(ids)) != 1097 or len(groups) != 1097
        or {group: groups.count(group) for group in set(groups)} != {"dev-small": 1000, "trec19": 43, "trec20": 54}
        or panel.get("threshold_by_group") != {"dev-small": 1, "trec19": 2, "trec20": 2}
    ):
        raise ValueError("Production pipeline quality requires the frozen1097 IR cohort with separate1000/43/54 populations.")


def validate_scorer_contract(ranker, specification):
    if ranker == "semantic-mlp" and specification.get("model") == ranker and type(specification.get("seed")) is int and specification["seed"] == 17:
        return
    if ranker == "bge-m3-rerank" and specification.get("model") == "BAAI/bge-reranker-v2-m3":
        return
    raise ValueError("Frozen pipeline scorer must be semantic-MLP seed17 or the actual BGE-reranker-v2-m3.")


def condition_grid(backend, probes=(64, 256), k=1000, batches=(1, 16), allow_batch64=False):
    if (
        backend not in BACKENDS or type(k) is not int or k != 1000
        or not probes or len(set(probes)) != len(probes) or set(probes) - {64, 256}
        or not batches or len(set(batches)) != len(batches)
        or set(batches) - ({1, 16, 64} if allow_batch64 else {1, 16})
        or any(type(value) is not int for value in (*probes, *batches))
    ):
        raise ValueError("Condition must belong to the declared Primary B backend/probes/k/batch grid.")
    return [f"{backend}-p{p}-k{k}-b{b}" for p in probes for b in batches]


def query_schedule(panel, session, batch, warmup=10, samples=50):
    validate_session(session)
    for name, value in (("batch", batch), ("warmup", warmup), ("samples", samples)):
        positive_int(value, name)
    ids = panel["query_ids"]
    if samples > 100 or warmup > 100:
        raise ValueError("A bounded session permits at most 100 warmup and 100 measured batches.")
    if (
        not ids or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids) or len(ids) < batch * (warmup + samples)
    ):
        raise ValueError("Insufficient unique source query IDs; lower explicit samples/batch, never repeat by default.")
    domain = f"primary-b-v2/session={session}/batch={batch}"
    order = sorted(range(len(ids)), key=lambda row: (digest([domain, 20261011, ids[row]]), ids[row]))
    value = {
        "session": session, "batch": batch, "query_ids_sha256": digest(ids),
        "warmup_rows": [order[start:start + batch] for start in range(len(order) - warmup * batch, len(order), batch)],
        "sample_rows": [order[start:start + batch] for start in range(0, samples * batch, batch)],
        "selection": "SHA256v2 IDs only; measured prefix and disjoint warmup suffix; unequal budgets retain paired measured prefixes.",
    }
    return {**value, "schedule_sha256": digest(value)}


def planned_execution_groups(panel, population, session, batch, smoke=False):
    validate_session(session)
    ids, mapping = panel["query_ids"], list(population["quality_rows"])
    if (
        type(batch) is not int or batch not in (1, 16) or type(smoke) is not bool
        or len(ids) != (100 if smoke else 1097) or len(set(ids)) != len(ids)
        or not mapping or len(set(mapping)) != len(mapping)
        or any(type(row) is not int or not 0 <= row < len(ids) for row in mapping)
    ):
        raise ValueError("Execution groups require the declared cohort, batch1/16 and an exact unique timing-to-quality mapping.")
    warmup, samples = (1, 5) if smoke else (10, 50)
    if len(mapping) < batch * (warmup + samples):
        raise ValueError("Timing population cannot fill immutable execution groups; reduced timing budgets do not shrink the geometry.")
    canonical = query_schedule({"query_ids": [ids[row] for row in mapping]}, session, batch, warmup, samples)
    groups = [[mapping[row] for row in group] for group in canonical["sample_rows"] + canonical["warmup_rows"]]
    used = {row for group in groups for row in group}
    remaining = sorted(
        (row for row in range(len(ids)) if row not in used),
        key=lambda row: (digest([f"primary-b-v2/session={session}/batch={batch}", 20261011, ids[row]]), ids[row]),
    )
    groups += [remaining[start:start + batch] for start in range(0, len(remaining), batch)]
    probe_groups = int(np.searchsorted(np.cumsum([len(group) for group in groups]), 100)) + 1
    manifest = {
        "query_execution_policy": QUERY_EXECUTION_POLICY, "session": session, "query_batch": batch,
        "query_count": len(ids), "query_ids_sha256": digest(ids), "smoke_geometry": smoke,
        "population": {**population, "quality_rows": mapping}, "groups": groups,
        "measured_group_indices": list(range(samples)),
        "warmup_group_indices": list(range(samples, samples + warmup)),
        "probe_group_indices": list(range(probe_groups)),
        "scope": "Quality is conditional on batch/session/ordered groups; sessions are not independent query samples.",
    }
    return {**manifest, "execution_groups_sha256": digest(manifest)}


def validate_execution_groups(manifest):
    groups = manifest["groups"]
    if (
        manifest.get("query_execution_policy") != QUERY_EXECUTION_POLICY
        or not groups or manifest["query_count"] != (100 if manifest["smoke_geometry"] else 1097)
        or digest({key: value for key, value in manifest.items() if key != "execution_groups_sha256"})
        != manifest.get("execution_groups_sha256")
        or any(not group or len(group) > manifest["query_batch"] for group in groups)
        or sorted(row for group in groups for row in group) != list(range(manifest["query_count"]))
    ):
        raise ValueError("Immutable execution group manifest changed or no longer covers each quality row exactly once.")


def select_execution_groups(manifest, warmup, samples):
    validate_execution_groups(manifest)
    if (
        positive_int(warmup, "warmup") > len(manifest["warmup_group_indices"])
        or positive_int(samples, "samples") > len(manifest["measured_group_indices"])
    ):
        raise ValueError("Requested count exceeds frozen execution groups; no regrouping or repeated queries is allowed.")
    value = {
        "session": manifest["session"], "batch": manifest["query_batch"], "row_space": "quality_panel",
        "query_ids_sha256": manifest["query_ids_sha256"], "execution_groups_sha256": manifest["execution_groups_sha256"],
        "warmup_rows": [list(manifest["groups"][index]) for index in manifest["warmup_group_indices"][-warmup:]],
        "sample_rows": [list(manifest["groups"][index]) for index in manifest["measured_group_indices"][:samples]],
    }
    return {**value, "schedule_sha256": digest(value)}


def require_execution_schedule(manifest, schedule, warmup, samples):
    if schedule != select_execution_groups(manifest, warmup, samples):
        raise ValueError("Execution schedule must replay selected ordered whole manifest groups; regrouping is forbidden.")


def timing_summary(samples, batch):
    positive_int(batch, "batch")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Every synchronized wall-time sample must be finite and positive.")
    return {
        "samples_ms": values.tolist(), "sample_count": len(values),
        "latency_mean_ms": float(values.mean()), "latency_p50_ms": float(np.median(values)),
        "observed_min_ms": float(values.min()), "observed_max_ms": float(values.max()),
        "offline_queries_per_second": batch * 1000 / float(values.mean()),
        "throughput_kind": "offline_batch_throughput",
        "scope": f"Descriptive {len(values)} synchronized batch observations; explicitly count-scoped.",
        "p99_warning": "At most100 varying batches/session: no reliable p99, timing confidence interval, serving QPS or SLO.",
    }


def population_binding(timing, quality, timing_sha, quality_sha):
    for source in (timing, quality):
        if any(name in source for name in ("users", "history_instruction", "history_item_rows")):
            raise ValueError("Primary B admits IR panels only, not user/history representations.")
        ids, texts = source["query_ids"], source["queries"]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(texts) or any(not text.strip() for text in texts):
            raise ValueError("Timing/quality population requires unique aligned IDs and nonempty text.")
    positions = {identifier: row for row, identifier in enumerate(quality["query_ids"])}
    rows = []
    for row, identifier in enumerate(timing["query_ids"]):
        target = positions.get(identifier)
        if (
            target is None or timing["queries"][row] != quality["queries"][target]
            or timing["qrels"].get(identifier) != quality["qrels"].get(identifier)
            or timing.get("binary_threshold") != quality.get("binary_threshold")
            or timing.get("threshold_by_group") != quality.get("threshold_by_group")
            or ("evaluation_groups" in timing) != ("evaluation_groups" in quality)
            or ("evaluation_groups" in timing and timing["evaluation_groups"][row] != quality["evaluation_groups"][target])
        ):
            raise ValueError("Timing population is not an exact text/judgment subset of the quality population.")
        rows.append(target)
    return {"timing_panel_sha256": timing_sha, "quality_panel_sha256": quality_sha, "quality_rows": rows}


def match_quality_timing(quality, timing):
    required = {
        "backend", "probes", "retrieval_depth", "candidate_cap", "items_sha256", "partition_sha256",
        "execution_queries_sha256", "execution_groups_sha256", "query_execution_policy", "query_batch", "session",
        "queries_sha256", "execution_query_storage_dtype",
        "quality_panel_sha256", "source_sha256", "offsets_sha256",
        "representation", "scorer", "candidate_policy", "positive_injection", "final_k", "tf32", "candidate_ids_sha256",
    }
    contract = quality.get("deployment", {})
    representation, scorer = contract.get("representation", {}), contract.get("scorer", {})
    if (
        not required.issubset(contract) or contract != timing.get("deployment")
        or contract["backend"] not in BACKENDS or contract["candidate_policy"] != POLICY
        or contract["positive_injection"] is not False or contract["tf32"] is not False
        or contract["retrieval_depth"] != 1000 or contract["candidate_cap"] not in (50, 500)
        or contract["probes"] not in (64, 256) or contract["final_k"] != 10
        or not contract["candidate_ids_sha256"]
        or contract["query_execution_policy"] != QUERY_EXECUTION_POLICY
        or not contract["execution_queries_sha256"] or not contract["execution_groups_sha256"]
        or contract["queries_sha256"] != contract["execution_queries_sha256"]
        or contract["execution_query_storage_dtype"] != "float16"
        or contract["query_batch"] not in (1, 16) or contract["session"] not in SESSIONS
        or any("batch" in point and point["batch"] != contract["query_batch"] for point in (quality, timing))
        or any("session" in point and point["session"] != contract["session"] for point in (quality, timing))
        or representation.get("model") != "Qwen/Qwen3-Embedding-8B"
        or representation.get("stored_dimension") != 1024 or representation.get("stored_dtype") != "float16"
        or representation.get("normalization") != NORMALIZATION
        or not all(representation.get(name) for name in ("revision", "specification_sha256", "query_instruction"))
        or scorer.get("name") not in ("semantic-mlp", "bge-m3-rerank")
        or not scorer.get("checkpoint_sha256" if scorer.get("name") == "semantic-mlp" else "model_files")
    ):
        raise ValueError("Quality/timing deployment contracts differ or are incomplete; no common-union substitution.")
    return digest(contract)


def match_quality_reference(candidate, reference):
    """Validate checked point records; return the shared reference identity, not an adoption decision."""
    if (
        candidate.get("status") != "ok" or reference.get("status") != "ok"
        or candidate.get("point_role") != "deployment_candidate" or reference.get("point_role") != "reference_only"
        or candidate.get("smoke_only") is not False or reference.get("smoke_only") is not False
    ):
        raise ValueError("Reference comparison requires successful nonsmoke deployment/reference-only points.")
    for point in (candidate, reference):
        match_quality_timing(point, point)
    actual, baseline = candidate["deployment"], reference["deployment"]
    excluded = {"backend", "candidate_ids_sha256"}
    if (
        actual["backend"] not in PIPELINE_CANDIDATES or baseline["backend"] != "faiss-gpu-fp32"
        or {key: value for key, value in actual.items() if key not in excluded}
        != {key: value for key, value in baseline.items() if key not in excluded}
    ):
        raise ValueError("Candidate and actual FAISS FP32 reference differ in panel/scorer/prefix/source contracts.")
    return digest(baseline)


def match_latency_reference(candidate, reference):
    identity = match_quality_reference(candidate, reference)
    fields = ("session", "batch", "boundary", "population", "schedule", "count_scope")
    protocol = reference.get("sources", {}).get("protocol")
    if (
        not protocol or candidate.get("sources", {}).get("protocol") != protocol
        or any(key not in candidate or key not in reference or candidate[key] != reference[key] for key in fields)
        or candidate.get("kind") != "latency" or reference.get("kind") != "latency"
    ):
        raise ValueError("Fresh latency reference must match protocol/session/batch/boundary/population and paired schedule.")
    for point in (candidate, reference):
        summary = point["latency"]
        counts, schedule = point["count_scope"], point["schedule"]
        if (
            summary != timing_summary(summary["samples_ms"], point["batch"])
            or summary["sample_count"] != counts["measured_batches"]
            or len(schedule["sample_rows"]) != counts["measured_batches"]
            or len(schedule["warmup_rows"]) != counts["warmup_batches"]
        ):
            raise ValueError("Latency reference raw samples do not match the declared count-scoped protocol.")
    return digest({
        "quality_reference": identity, "protocol": protocol, "session": reference["session"],
        "batch": reference["batch"], "boundary": reference["boundary"],
        "schedule_sha256": reference["schedule"]["schedule_sha256"],
    })


def final_depth_metrics(value):
    if isinstance(value, dict):
        return {key: final_depth_metrics(item) for key, item in value.items() if not key.endswith(("@100", "@1000"))}
    if isinstance(value, list):
        return [final_depth_metrics(item) for item in value]
    return value


def check_memory(observed, required_bytes=0, headroom_bytes=8 * GIB):
    free = observed["device_memory_bytes"] - observed["memory_used_bytes"]
    free -= max(0, observed.get("torch_peak_allocated_bytes", 0) - observed.get("torch_allocated_bytes", 0))
    if free < max(headroom_bytes, required_bytes):
        raise UnsupportedMemory(f"Conservative free-byte estimate {free} below admission/headroom gate {max(headroom_bytes, required_bytes)}.")


def verify_completion(receipt, binding, expected, records, actual_hashes):
    if receipt.get("complete") is not True or receipt.get("binding_sha256") != digest(binding):
        raise ValueError("Completion/source binding is missing or changed.")
    if receipt.get("conditions") != sorted(expected) or sorted(row["condition"] for row in records) != sorted(expected):
        raise ValueError("Completion does not cover every finite condition exactly once.")
    if receipt.get("records_sha256") != digest(records):
        raise ValueError("Completed journal metrics/timings changed; refusing to recertify.")
    required = {"run-config.json", "journal.jsonl"} | {name for row in records for name in row.get("files", [])}
    if set(receipt.get("artifacts", {})) != required or receipt["artifacts"] != actual_hashes:
        raise ValueError("Completed payload/journal/trace checksum inventory changed.")
    for row in records:
        if row.get("sources_sha256") != digest(binding["sources"]):
            raise ValueError("Condition source binding changed.")
        if row["status"] == "ok":
            if not row.get("files"):
                raise ValueError("Successful condition lacks persistent predictions or trace evidence.")
            if row["kind"] == "latency":
                value = timing_summary(row["latency"]["samples_ms"], row["batch"])
                if value != row["latency"] or value["sample_count"] != binding["parameters"]["samples"]:
                    raise ValueError("Completed timing summary/count differs from the raw sample journal.")
            elif row["kind"] != "profile" or "latency" in row:
                raise ValueError("Profile and latency observations cannot share a timing estimand.")
            elif row.get("trace_file") not in row["files"] or row.get("profile", {}).get("cuda_kernel_events", 0) <= 0:
                raise ValueError("Profile completion lacks verified CUDA trace evidence.")
        elif row["status"] not in ("unsupported_memory", "unsupported_profiler") or not row.get("reason"):
            raise ValueError("Unrecognized condition failure cannot be published as completion.")


def validate_source_binding(item, merge, partition_binding, partition_receipt, query, representation, panel, hashes):
    validate_partition_contract(CLUSTERS, partition_binding)
    expected = {key: value for key, value in representation.items() if key != "query_instruction"}
    if (
        item.get("complete") is not True or item["rows"] != 8841823
        or item["native_dimension"] != 4096 or item["stored_dimension"] != 1024
        or item.get("representation") != expected or item["vectors_sha256"] != hashes["items"]
        or expected["model"] != "Qwen/Qwen3-Embedding-8B"
        or expected["stored_dimension"] != 1024 or expected["stored_dtype"] != "float16"
        or expected["normalization"] != NORMALIZATION
        or not representation.get("query_instruction")
    ):
        raise ValueError("Full Qwen8 native4096/MRL1024 source bank representation is not attested.")
    position = 0
    for shard in item["source_shards"]:
        if shard["start"] != position or shard["rows"] <= 0 or not shard["receipt_sha256"]:
            raise ValueError("Full bank source shard coverage is not contiguous.")
        position += shard["rows"]
    if position != item["rows"] or merge != {"dimension": 1024, "rows": position, "sources": item["source_shards"]}:
        raise ValueError("Full bank source receipt and merge binding differ.")
    if partition_binding["items_sha256"] != hashes["items"] or partition_receipt["partition_sha256"] != hashes["partition"]:
        raise ValueError("Frozen partition belongs to another source bank or has changed.")
    if (
        query.get("complete") is not True or query["query_ids"] != panel["query_ids"]
        or query["rows"] != len(panel["query_ids"]) or query["panel_sha256"] != hashes["panel"]
        or query["native_dimension"] != 4096
    ):
        raise ValueError("Exact query source panel/IDs/order/native representation differs.")


def validate_unit_queries(values, rows):
    if values.shape != (rows, 1024) or values.dtype != np.float16:
        raise ValueError("Canonical MRL queries must be aligned float16 [rows,1024].")
    for start in range(0, rows, 8192):
        block = np.asarray(values[start:start + 8192], dtype=np.float32)
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=0.002):
            raise ValueError("MRL query rows must be finite and independently unit normalized.")
