"""CPU contract tests; CUDA equivalence tests exercise the real, optional backends."""

import copy
import importlib
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from rrbench.ann import Partition
from rrbench.io import sha256


def harness():
    assert importlib.util.find_spec("rrbench.semantic_profiling") is not None, (
        "The Primary B profiling harness has not been implemented."
    )
    return importlib.import_module("rrbench.semantic_profiling")


def panel(count=5000):
    return {
        "query_ids": [f"query-{row}" for row in range(count)],
        "queries": [f"real passage question {row}" for row in range(count)],
        "qrels": {f"query-{row}": {"999": 1} for row in range(count)},
        "binary_threshold": 1,
    }


def test_schedule_is_label_independent_paired_and_session_separated():
    module = harness()
    source = panel()
    original = module.query_schedule(source, session=1, batch=16, warmup=10, samples=100)
    altered = copy.deepcopy(source)
    altered["qrels"] = {}
    assert module.query_schedule(altered, 1, 16, 10, 100) == original
    assert module.query_schedule(source, 2, 16, 10, 100) != original
    rows = original["warmup_rows"] + original["sample_rows"]
    assert len(rows) == 110
    assert len({row for batch in rows for row in batch}) == 1760
    assert original["query_ids_sha256"] == module.digest(source["query_ids"])
    assert original["schedule_sha256"] == module.digest(
        {key: value for key, value in original.items() if key != "schedule_sha256"}
    )
    # Sampling source identities rather than file positions preserves selected IDs after a reorder.
    reordered = {key: list(reversed(source[key])) for key in ("query_ids", "queries")}
    second = module.query_schedule(reordered, 1, 16, 10, 100)
    assert [[source["query_ids"][row] for row in batch] for batch in rows] == [
        [reordered["query_ids"][row] for row in batch]
        for batch in second["warmup_rows"] + second["sample_rows"]
    ]


@pytest.mark.parametrize("field,value", [
    ("session", 3), ("session", -1), ("session", True), ("session", 1.5),
    ("warmup", 0), ("warmup", float("nan")), ("samples", 0), ("samples", 101),
    ("samples", float("inf")), ("batch", 0),
])
def test_schedule_rejects_nonpositive_nonintegral_or_unbounded_controls(field, value):
    values = dict(session=1, batch=1, warmup=10, samples=100)
    values[field] = value
    with pytest.raises(ValueError):
        harness().query_schedule(panel(), **values)


def test_schedule_does_not_silently_repeat_queries_to_fill_a_panel():
    module = harness()
    with pytest.raises(ValueError, match="unique"):
        module.query_schedule(panel(100), 1, 16, 10, 100)
    source = panel(100)
    source["query_ids"][1] = source["query_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        module.query_schedule(source, 1, 1, 1, 2)
    # Explicit bounded alternatives, not a requirement for a larger validation population.
    assert len(module.query_schedule(panel(4100), 1, 64, 10, 50)["sample_rows"]) == 50
    assert len(module.query_schedule(panel(1097), 1, 16, 10, 50)["sample_rows"]) == 50
    assert len(module.query_schedule(panel(100), 1, 1, 10, 90)["sample_rows"]) == 90


def test_frozen_three_sessions_and_batch64_default_fit_validation_without_repeats():
    module = harness()
    schedules = [module.query_schedule(panel(), session, 64) for session in (0, 1, 2)]
    assert len({schedule["schedule_sha256"] for schedule in schedules}) == 3
    for schedule in schedules:
        assert len(schedule["warmup_rows"]) == 10
        assert len(schedule["sample_rows"]) == 50
        assert len({row for group in schedule["warmup_rows"] + schedule["sample_rows"] for row in group}) == 3840


def test_expensive_text_samples_remain_a_paired_prefix_despite_different_warmup_budget():
    module = harness()
    source = panel(1097)
    mlp = module.query_schedule(source, 1, 16, 10, 50)
    text = module.query_schedule(source, 1, 16, 2, 8)
    assert text["sample_rows"] == mlp["sample_rows"][:8]
    assert set(row for group in text["warmup_rows"] for row in group).isdisjoint(
        row for group in mlp["sample_rows"] for row in group
    )
    assert len({row for group in mlp["warmup_rows"] + mlp["sample_rows"] for row in group}) == 960


def test_primary_grid_is_exact_and_batch64_is_explicitly_admitted():
    module = harness()
    assert set(module.BACKENDS) == {
        "faiss-gpu-fp32", "silver-fp32", "faiss-gpu-fp16", "silver-fp16",
        "silver-int8-rowwise",
    }
    assert module.condition_grid("silver-fp32") == [
        "silver-fp32-p64-k1000-b1", "silver-fp32-p64-k1000-b16",
        "silver-fp32-p256-k1000-b1", "silver-fp32-p256-k1000-b16",
    ]
    for arguments in (
        {"backend": "faiss-cpu-fp32"}, {"backend": "silver-int8-global"},
        {"backend": "silver-fp32", "probes": (16,)},
        {"backend": "silver-fp32", "k": 50},
        {"backend": "silver-fp32", "batches": (1, 1)},
        {"backend": "silver-fp32", "batches": (64,)},
    ):
        with pytest.raises(ValueError):
            module.condition_grid(**arguments)
    assert module.condition_grid("silver-fp32", batches=(64,), allow_batch64=True)
    with pytest.raises(module.UnsupportedMemory):
        module.check_memory({"memory_used_bytes": 73 << 30, "device_memory_bytes": 80 << 30})
    module.check_memory({"memory_used_bytes": 70 << 30, "device_memory_bytes": 80 << 30})


def test_summary_keeps_every_synchronized_observation_without_tail_or_ci_claims():
    module = harness()
    result = module.timing_summary([1.0, 9.0, 2.0], 16)
    assert result["samples_ms"] == [1.0, 9.0, 2.0]
    assert result["sample_count"] == 3
    assert result["latency_mean_ms"] == 4.0
    assert result["latency_p50_ms"] == 2.0
    assert result["observed_min_ms"] == 1.0
    assert result["observed_max_ms"] == 9.0
    assert result["offline_queries_per_second"] == 4000.0
    assert result["throughput_kind"] == "offline_batch_throughput"
    assert result["p99_warning"]
    assert not any(key in result for key in ("latency_p99_ms", "latency_mean_ci95_ms", "serving_qps"))
    for samples in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            module.timing_summary(samples, 1)


def test_prefix_is_chosen_before_scorer_and_labels_never_inject_candidates():
    module = harness()
    retrieved = torch.arange(1000, dtype=torch.int64).reshape(1, -1)
    seen = []

    def actual_toy_scorer(prefix):
        seen.append(prefix.clone())
        return prefix.float()

    prefix, scores, final = module.score_prefix(retrieved, 50, actual_toy_scorer)
    assert torch.equal(seen[0], retrieved[:, :50])
    assert prefix.shape == scores.shape == (1, 50)
    assert final.tolist() == [list(range(49, 39, -1))]
    from rrbench.semantic_ir import panel_metrics

    quality = panel_metrics(panel(1), final.numpy())
    assert quality["recall@10"] == 0.0  # The judged positive at 999 is not injected.
    with pytest.raises(ValueError, match="prefix"):
        module.score_prefix(retrieved, 1001, actual_toy_scorer)


def test_boundary_parity_requires_exact_ids_and_tolerant_scores():
    module = harness()
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    query = torch.zeros((1, 1024), dtype=torch.float32)
    query[:, 0] = 1
    first = (ids, torch.tensor([[0.9, 0.8]]), ids, query)
    module.assert_parity(first, (ids, first[1] + 1e-6, ids, query))
    with pytest.raises(ValueError, match="IDs"):
        module.assert_parity(first, (ids.flip(1), first[1], ids, query))
    with pytest.raises(ValueError, match="scores"):
        module.assert_parity(first, (ids, first[1] + 0.1, ids, query))


def parity_result():
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    query = torch.zeros((1, 1024), dtype=torch.float32)
    query[:, 0] = 1
    return ids, torch.tensor([[0.9, 0.8]]), ids, query


def test_identical_bge_like_outputs_cannot_hide_a_drifting_fourth_query():
    reference = parity_result()
    changed_query = reference[3].clone()
    changed_query[0, 1] = 0.001
    actual = (*reference[:3], changed_query.half().float())
    with pytest.raises(ValueError, match="query") as failure:
        harness().assert_parity(actual, reference)
    observed = failure.value.observation
    assert observed["maximum_absolute_difference"] >= 0.001
    assert observed["changed_rounded_values"] == 1
    assert observed["minimum_cosine_similarity"] >= 0.99999


def test_small_non_bit_exact_query_difference_is_recorded_under_frozen_limits():
    reference = parity_result()
    changed_query = reference[3].clone()
    changed_query[0, 1] = 5e-5
    actual = (*reference[:3], changed_query.half().float())
    observed = harness().assert_parity(actual, reference)
    assert 0 < observed["maximum_absolute_difference"] <= 1e-4
    assert observed["minimum_cosine_similarity"] >= 0.99999
    assert observed["changed_rounded_values"] == 1
    assert observed["query_count"] == 1
    assert observed["value_count"] == 1024
    assert observed["limits"] == {
        "maximum_absolute_difference": 1e-4, "minimum_cosine_similarity": 0.99999,
    }
    assert "not bit-exact" in observed["scope"]


def test_parity_rejects_accidentally_slicing_away_the_query():
    reference = parity_result()
    with pytest.raises(ValueError, match="four"):
        harness().assert_parity(reference[:3], reference[:3])


def test_cosine_gate_rejects_small_absolute_but_large_directional_drift():
    reference = parity_result()
    cached = torch.zeros_like(reference[3])
    cached[0, 0] = 1e-5
    actual = cached.flip(1).half().float()
    with pytest.raises(ValueError, match="query") as failure:
        harness().assert_parity((*reference[:3], actual), (*reference[:3], cached.half().float()))
    assert failure.value.observation["maximum_absolute_difference"] < 1e-4
    assert failure.value.observation["minimum_cosine_similarity"] == 0


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "not-rounded", "zero"])
def test_query_gate_rejects_invalid_or_noncanonical_comparisons(kind):
    reference = parity_result()
    query = reference[3].clone()
    if kind == "shape":
        query = query[:, :4]
    elif kind == "dtype":
        query = query.double()
    elif kind == "nonfinite":
        query[0, 1] = float("nan")
    elif kind == "not-rounded":
        query[0, 1] = 6e-5
    else:
        query.zero_()
    with pytest.raises(ValueError, match="query"):
        harness().assert_parity((*reference[:3], query), reference)


def test_observation_payloads_retain_all_four_pipeline_results():
    module = harness()
    result = parity_result()
    arrays = module.observation_arrays([result, result])
    assert set(arrays) == {"ids", "scores", "ranked", "raw_queries"}
    assert arrays["raw_queries"].shape == (2, 1, 1024)
    assert arrays["raw_queries"].dtype == np.float32
    np.testing.assert_array_equal(arrays["raw_queries"][0], result[3].numpy())
    with pytest.raises(ValueError, match="four"):
        module.observation_arrays([result[:3]])


def test_active_query_payload_is_lossless_fp16_with_value_preserving_fp32_promotion():
    module = harness()
    actual = np.vstack((parity_result()[3].numpy(), np.full((1, 1024), 0.03125, dtype=np.float32)))
    storage, promoted = module.canonical_query_payload(actual)
    assert storage.dtype == np.float16
    assert promoted.dtype == np.float32
    assert storage.shape == promoted.shape == actual.shape
    np.testing.assert_array_equal(promoted, actual)
    stream = BytesIO()
    np.save(stream, storage, allow_pickle=False)
    stream.seek(0)
    persisted = np.load(stream, allow_pickle=False)
    assert persisted.dtype == np.float16
    np.testing.assert_array_equal(persisted.astype(np.float32), actual)
    noncanonical = actual.copy()
    noncanonical[0, 1] = 6e-5
    with pytest.raises(ValueError, match="query"):
        module.canonical_query_payload(noncanonical)


def test_timing_population_must_map_to_exact_quality_texts_and_judgments():
    module = harness()
    full = panel(8)
    subset = {
        "query_ids": [full["query_ids"][row] for row in (7, 2)],
        "queries": [full["queries"][row] for row in (7, 2)],
        "qrels": {full["query_ids"][row]: full["qrels"][full["query_ids"][row]] for row in (7, 2)},
        "binary_threshold": 1,
    }
    mapping = module.population_binding(subset, full, "subset-sha", "quality-sha")
    assert mapping["quality_rows"] == [7, 2]
    assert mapping["timing_panel_sha256"] == "subset-sha"
    assert mapping["quality_panel_sha256"] == "quality-sha"
    subset["queries"][0] = "unmatched text"
    with pytest.raises(ValueError, match="population"):
        module.population_binding(subset, full, "subset-sha", "quality-sha")
    with pytest.raises(ValueError, match="IR"):
        module.population_binding({**full, "users": [1] * 8}, full, "a", "b")


def deployment():
    return {
        "backend": "silver-int8-rowwise", "probes": 64, "retrieval_depth": 1000, "candidate_cap": 50,
        "items_sha256": "items", "partition_sha256": "partition",
        "historical_queries_sha256": "historical-batch32",
        "execution_queries_sha256": "actual-new-canonical-query-bytes",
        "queries_sha256": "actual-new-canonical-query-bytes",
        "execution_query_storage_dtype": "float16",
        "execution_groups_sha256": "ordered-groups",
        "query_execution_policy": "raw-execution-groups-v2", "query_batch": 16, "session": 0,
        "quality_panel_sha256": "panel", "source_sha256": "source", "offsets_sha256": "offsets",
        "candidate_ids_sha256": "actual-single-index-candidates",
        "representation": source_receipts()["representation"],
        "scorer": {"name": "semantic-mlp", "checkpoint_sha256": "qwen8-checkpoint"},
        "candidate_policy": "retrieve-depth-then-prefix-before-scoring",
        "positive_injection": False, "final_k": 10, "tf32": False,
    }


def test_quality_timing_join_requires_contextual_query_geometry_and_actual_query_bytes():
    module = harness()
    quality = {"deployment": deployment()}
    timing = {"deployment": deployment(), "batch": 16}
    assert module.match_quality_timing(quality, timing) == module.digest(deployment())
    for field, value in (
        ("backend", "silver-fp16"), ("candidate_cap", 500), ("probes", 256),
        ("execution_queries_sha256", "different-vectors"), ("quality_panel_sha256", "other-population"),
        ("execution_groups_sha256", "different-groups"), ("query_batch", 1), ("session", 1),
        ("candidate_policy", "common-union"), ("positive_injection", True),
        ("candidate_ids_sha256", "common-union-ids"),
        ("scorer", {"name": "semantic-mlp", "checkpoint_sha256": "qwen06-checkpoint"}),
    ):
        changed = copy.deepcopy(timing)
        changed["deployment"][field] = value
        with pytest.raises(ValueError, match="deployment"):
            module.match_quality_timing(quality, changed)
    with pytest.raises(ValueError, match="deployment"):
        module.match_quality_timing({"deployment": {}}, {"deployment": {}})


def test_quality_join_requires_a_bound_actual_pool_and_encoder_precision_not_just_equal_metadata():
    module = harness()
    for field in ("candidate_ids_sha256", "representation"):
        broken = deployment()
        broken.pop(field)
        with pytest.raises(ValueError, match="deployment"):
            module.match_quality_timing({"deployment": broken}, {"deployment": copy.deepcopy(broken)})
    broken = deployment()
    broken["representation"] = {}
    with pytest.raises(ValueError, match="deployment"):
        module.match_quality_timing({"deployment": broken}, {"deployment": copy.deepcopy(broken)})


def test_queries_sha_alias_cannot_point_to_historical_bytes_or_a_float32_archive():
    module = harness()
    for key, value in (
        ("queries_sha256", "historical-batch32"), ("execution_query_storage_dtype", "float32"),
    ):
        broken = {**deployment(), key: value}
        with pytest.raises(ValueError, match="deployment"):
            module.match_quality_timing({"deployment": broken}, {"deployment": copy.deepcopy(broken)})


def test_final_quality_does_not_label_top10_outputs_as_a_top1000_ranking():
    from rrbench.semantic_ir import panel_metrics

    metrics = panel_metrics(panel(1), np.arange(10, dtype=np.int64).reshape(1, -1))
    restricted = harness().final_depth_metrics(metrics)
    assert "mrr@10" in restricted
    assert "recall@1000" not in restricted
    assert "ndcg@100" not in restricted["per_query"][0]
    assert restricted["per_query"][0]["query_id"] == "query-0"


def completion_fixture():
    module = harness()
    source = sha256(Path(__file__))
    binding = {"sources": {"test_source": source}, "parameters": {"samples": 2}}
    records = [{
        "condition": "c", "status": "ok", "kind": "latency", "batch": 1,
        "latency": module.timing_summary([1.0, 2.0], 1),
        "files": ["predictions/ids.npy"], "sources_sha256": module.digest(binding["sources"]),
    }]
    hashes = {name: source for name in ("run-config.json", "journal.jsonl", "predictions/ids.npy")}
    receipt = {
        "complete": True, "binding_sha256": module.digest(binding), "conditions": ["c"],
        "records_sha256": module.digest(records), "artifacts": hashes,
    }
    return receipt, binding, records, hashes


def test_completion_requires_all_conditions_payloads_and_unchanged_journal():
    module = harness()
    receipt, binding, records, hashes = completion_fixture()
    module.verify_completion(receipt, binding, ["c"], records, hashes)
    changed = copy.deepcopy(records)
    changed[0]["latency"]["samples_ms"][0] = 1000.0
    with pytest.raises(ValueError, match="journal"):
        module.verify_completion(receipt, binding, ["c"], changed, hashes)
    for bad in (
        {**receipt, "complete": False}, {**receipt, "conditions": []},
        {**receipt, "artifacts": {}}, {**receipt, "binding_sha256": "different-source"},
    ):
        with pytest.raises(ValueError):
            module.verify_completion(bad, binding, ["c"], records, hashes)
    with pytest.raises(ValueError, match="checksum"):
        module.verify_completion(receipt, binding, ["c"], records, {**hashes, "predictions/ids.npy": "drift"})
    with pytest.raises(ValueError):
        module.verify_completion(receipt, binding, ["c", "unexecuted"], records, hashes)


def test_unsupported_is_narrow_and_not_an_unexpected_backend_bug():
    module = harness()
    assert module.memory_failure(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert module.memory_failure(RuntimeError(
        "StandardGpuResources: alloc fail cudaMalloc out of memory"
    ))
    assert not module.memory_failure(RuntimeError("Unsupported scalar quantizer type"))
    assert not module.memory_failure(RuntimeError("CUDA illegal memory access"))
    assert not module.memory_failure(ValueError("out of memory"))


def test_profile_summary_refuses_a_cpu_only_trace_as_cuda_evidence():
    with pytest.raises(harness().ProfilerUnavailable, match="CUDA"):
        harness().require_cuda_trace({"traceEvents": [{"name": "aten::mm", "cat": "cpu_op", "dur": 2}]})


def source_receipts():
    representation = {
        "model": "Qwen/Qwen3-Embedding-8B", "revision": "r" * 40,
        "specification_sha256": "s" * 64, "stored_dimension": 1024, "stored_dtype": "float16",
        "normalization": "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage.",
    }
    shards = [{"start": 0, "rows": 8841823, "receipt_sha256": "shard-receipt"}]
    source = panel(100)
    item = {
        "complete": True, "rows": 8841823, "native_dimension": 4096, "stored_dimension": 1024,
        "representation": representation, "source_shards": shards, "vectors_sha256": "items",
    }
    query = {
        "complete": True, "rows": 100, "native_dimension": 4096, "panel_sha256": "panel",
        "query_ids": source["query_ids"],
    }
    return {
        "item": item,
        "merge": {"dimension": 1024, "rows": 8841823, "sources": shards},
        "partition_binding": {"items_sha256": "items", "clusters": 16384},
        "partition_receipt": {"partition_sha256": "partition"},
        "query": query, "representation": {**representation, "query_instruction": "Retrieve passages"},
        "panel": source, "hashes": {"items": "items", "partition": "partition", "panel": "panel"},
    }


def test_source_contract_rejects_unattested_same_dimension_banks_and_query_order():
    module = harness()
    values = source_receipts()
    module.validate_source_binding(**values)
    for field, child, value in (
        ("item", "rows", 8841822), ("item", "native_dimension", 1024),
        ("item", "complete", False), ("item", "source_shards", []),
        ("item", "vectors_sha256", "changed"),
        ("representation", "model", "Qwen/Qwen3-Embedding-0.6B"),
        ("representation", "stored_dtype", "float32"),
        ("representation", "stored_dimension", 4096),
        ("query", "query_ids", list(reversed(values["panel"]["query_ids"]))),
        ("query", "panel_sha256", "other-texts"),
        ("partition_binding", "items_sha256", "other-items"),
        ("partition_receipt", "partition_sha256", "other-partition"),
        ("partition_binding", "clusters", 8192),
    ):
        altered = copy.deepcopy(values)
        altered[field][child] = value
        with pytest.raises(ValueError, match="source|representation|bank|partition|query"):
            module.validate_source_binding(**altered)


def test_shared_partition_must_be_16384_even_when_8192_metadata_is_self_consistent():
    module = harness()
    module.validate_partition_contract(16384, {"clusters": 16384})
    for count, declared in ((8192, 8192), (16384, 8192), (8192, 16384)):
        with pytest.raises(ValueError, match="16384"):
            module.validate_partition_contract(count, {"clusters": declared})


def test_protocol_freezes_query_limits_and_cluster_count_before_any_execution():
    module = harness()
    profiling = frozen_profiling()
    parameters = {"operation": "ann", "session": 0, "backend": "faiss-gpu-fp16"}
    controls = module.resolve_protocol_controls(parameters, profiling)
    assert controls["clusters"] == 16384
    assert controls["canonical_query_parity"] == {
        "maximum_absolute_difference": 1e-4, "minimum_cosine_similarity": 0.99999,
    }
    for field, value in (
        ("maximum_absolute_difference", 1e-3), ("minimum_cosine_similarity", 0.999),
    ):
        changed = copy.deepcopy(profiling)
        changed["canonical_query_parity"][field] = value
        with pytest.raises(ValueError, match="protocol|parity"):
            module.resolve_protocol_controls(parameters, changed)
    with pytest.raises(ValueError, match="protocol"):
        module.resolve_protocol_controls(parameters, {**profiling, "clusters": 8192})


def test_cached_queries_are_finite_independently_normalized_canonical_fp16():
    module = harness()
    values = np.zeros((3, 1024), dtype=np.float16)
    values[:, 0] = 1
    module.validate_unit_queries(values, 3)
    for altered in (values.astype(np.float32), values[:, :512], values * 0.5):
        with pytest.raises(ValueError):
            module.validate_unit_queries(altered, 3)
    values[0, 1] = np.nan
    with pytest.raises(ValueError):
        module.validate_unit_queries(values, 3)


def test_cli_exposes_single_arm_sessions_and_separate_profiler_mode():
    module = harness()
    common = [
        "--items", "items.npy", "--partition", "partition.npz", "--panel", "panel.json",
        "--backend", "silver-fp32", "--session", "1", "--output", "new-output",
    ]
    ann = module.parse_arguments(["ann", *common, "--queries", "queries.npy"])
    assert (ann.warmup, ann.samples, ann.batches, ann.probes, ann.k) == (10, 50, [1, 16], [64, 256], 1000)
    assert not ann.profile
    assert module.make_parser().parse_args(["ann", *common, "--queries", "q.npy", "--profile"]).profile
    pipeline = module.parse_arguments([
        "pipeline", *[value.replace("silver-fp32", "silver-int8-rowwise") for value in common],
        "--registry", "registry.json", "--ranker", "semantic-mlp",
        "--source", "collection.tsv", "--offsets", "offsets.npy", "--cache", "models",
        "--smoke", "smoke.json", "--quality-queries", "q.npy", "--checkpoint", "model.pt",
        "--lengths", "lengths.npy", "--candidate-cap", "50",
    ])
    assert pipeline.encoder == "qwen8"
    assert pipeline.retrieval_depth == 1000
    assert pipeline.boundary == "both"
    assert (pipeline.warmup, pipeline.samples) == (10, 50)


def frozen_profiling(pipeline=False):
    filename = "semantic-extension-pipeline-v2.json" if pipeline else "semantic-extension-v1.json"
    path = Path(__file__).resolve().parents[1] / "configs" / filename
    return harness().read_json(path)["profiling"]


def test_frozen_protocol_defaults_are_scorer_scoped_and_reference_is_not_a_deployment_candidate():
    module = harness()
    protocol = frozen_profiling(pipeline=True)
    for ranker, expected in (("semantic-mlp", (10, 50)), ("bge-m3-rerank", (2, 8))):
        parameters = {"operation": "pipeline", "session": 0, "backend": "faiss-gpu-fp16", "ranker": ranker}
        controls = module.resolve_protocol_controls(parameters, protocol)
        assert (controls["warmup"], controls["samples"]) == expected
        assert controls["headroom_bytes"] == 8 << 30
        assert controls["point_role"] == "deployment_candidate"
        reference = module.resolve_protocol_controls(
            {**parameters, "backend": "faiss-gpu-fp32", "reference_only": True}, protocol,
        )
        assert (reference["warmup"], reference["samples"]) == expected
        assert reference["headroom_bytes"] == 2 << 30
        assert reference["point_role"] == "reference_only"
        assert reference["planned_sessions"] == [0, 1, 2]
    observed = {"memory_used_bytes": 77 << 30, "device_memory_bytes": 80 << 30}
    module.check_memory(observed, headroom_bytes=reference["headroom_bytes"])
    with pytest.raises(module.UnsupportedMemory):
        module.check_memory(observed)


def test_frozen_protocol_rejects_undeclared_pipeline_backends_reference_roles_and_sessions():
    module = harness()
    parameters = {"operation": "pipeline", "session": 0, "backend": "faiss-gpu-fp16", "ranker": "semantic-mlp"}
    for changes in (
        {"backend": "faiss-gpu-fp32"}, {"backend": "silver-fp32"}, {"backend": "silver-fp16"},
        {"reference_only": True}, {"session": 3}, {"session": True}, {"warmup": 0}, {"samples": 101},
    ):
        with pytest.raises(ValueError):
            module.resolve_protocol_controls({**parameters, **changes}, frozen_profiling(pipeline=True))


def test_pipeline_frozen_quality_cohort_is_1097_with_1000_43_54_groups_or_explicit_smoke():
    module = harness()
    source = panel(1097)
    source["evaluation_groups"] = ["dev-small"] * 1000 + ["trec19"] * 43 + ["trec20"] * 54
    source["threshold_by_group"] = {"dev-small": 1, "trec19": 2, "trec20": 2}
    module.validate_quality_cohort(source)
    for altered in (panel(7077), panel(100), {**source, "evaluation_groups": ["dev-small"] * 1097}):
        with pytest.raises(ValueError, match="1097|cohort"):
            module.validate_quality_cohort(altered)
    module.validate_quality_cohort(panel(100), smoke=True)
    with pytest.raises(ValueError, match="100"):
        module.validate_quality_cohort(source, smoke=True)


def test_frozen_scorers_require_seed17_and_the_actual_bge_v2_m3_model():
    module = harness()
    module.validate_scorer_contract("semantic-mlp", {"model": "semantic-mlp", "seed": 17})
    module.validate_scorer_contract("bge-m3-rerank", {"model": "BAAI/bge-reranker-v2-m3"})
    for key, binding in (
        ("semantic-mlp", {"model": "semantic-mlp", "seed": 23}),
        ("semantic-mlp", {"model": "semantic-mlp"}),
        ("bge-m3-rerank", {"model": "BAAI/bge-reranker-base"}),
    ):
        with pytest.raises(ValueError, match="scorer"):
            module.validate_scorer_contract(key, binding)


def test_candidate_quality_joins_only_to_its_matched_actual_faiss_fp32_reference():
    module = harness()
    reference = {
        "status": "ok", "point_role": "reference_only", "smoke_only": False,
        "deployment": {**deployment(), "backend": "faiss-gpu-fp32", "candidate_ids_sha256": "actual-fp32-ids"},
    }
    for backend in ("faiss-gpu-fp16", "silver-int8-rowwise"):
        candidate = {
            "status": "ok", "point_role": "deployment_candidate", "smoke_only": False,
            "deployment": {**deployment(), "backend": backend, "candidate_ids_sha256": backend + "-actual-ids"},
        }
        assert module.match_quality_reference(candidate, reference) == module.digest(reference["deployment"])
        for field, value in (
            ("candidate_cap", 500), ("probes", 256), ("quality_panel_sha256", "7077-instead-of-1097"),
            ("execution_queries_sha256", "another-actual-query-array"),
            ("execution_groups_sha256", "another-group-manifest"),
            ("scorer", {"name": "semantic-mlp", "checkpoint_sha256": "different-seed"}),
            ("candidate_policy", "common-union"), ("backend", "silver-fp32"),
        ):
            changed = copy.deepcopy(reference)
            changed["deployment"][field] = value
            with pytest.raises(ValueError, match="reference|deployment"):
                module.match_quality_reference(candidate, changed)
        with pytest.raises(ValueError):
            module.match_quality_reference(candidate, {**reference, "status": "unsupported_memory"})
        with pytest.raises(ValueError):
            module.match_quality_reference({**candidate, "smoke_only": True}, reference)


def test_protocol_file_is_read_only_and_its_sha_is_bound_without_node_specific_paths():
    module = harness()
    path = Path(__file__).resolve().parents[1] / "configs" / "semantic-extension-v1.json"
    before = sha256(path)
    args = module.parse_arguments([
        "ann", "--items", "bank.npy", "--partition", "ivf.npz", "--queries", "q.npy",
        "--panel", "validation.json", "--backend", "silver-fp32", "--session", "0",
        "--output", "new-output", "--protocol", str(path),
    ])
    binding = module.build_binding({"protocol": args.protocol}, vars(args))
    assert binding["sources"]["protocol"] == before == sha256(path)
    assert args.planned_sessions == [0, 1, 2]


def test_cli_bge_reference_defaults_are_2_8_and_explicit_overrides_remain_count_scoped():
    module = harness()
    arguments = [
        "pipeline", "--items", "items.npy", "--partition", "ivf.npz", "--panel", "panel1097.json",
        "--backend", "faiss-gpu-fp32", "--reference-only", "--session", "0", "--output", "new-reference",
        "--registry", "registry.json", "--ranker", "bge-m3-rerank", "--source", "collection.tsv",
        "--offsets", "offsets.npy", "--cache", "models", "--smoke", "smoke.json",
        "--quality-queries", "query-vectors.npy", "--candidate-cap", "500",
    ]
    reference = module.parse_arguments(arguments)
    assert (reference.warmup, reference.samples, reference.headroom_bytes) == (2, 8, 2 << 30)
    assert reference.point_role == "reference_only"
    assert reference.count_protocol == "frozen_defaults"
    profiled = module.parse_arguments([*arguments, "--profile", "--samples", "1"])
    assert profiled.count_protocol == "explicit_count_override"
    assert profiled.samples == 1
    assert "timing confidence interval" in module.timing_summary([1.0] * 8, 16)["p99_warning"]


def test_fresh_latency_reference_requires_same_session_boundary_rows_and_sample_count():
    module = harness()
    common = {
        "status": "ok", "kind": "latency", "smoke_only": False,
        "session": 0, "batch": 16, "boundary": "integrated",
        "population": {"timing_panel_sha256": "frozen1097", "quality_panel_sha256": "frozen1097"},
        "schedule": module.query_schedule(panel(1097), 0, 16),
        "sources": {"protocol": "frozen-protocol-sha"},
        "count_scope": {"warmup_batches": 10, "measured_batches": 50, "policy": "frozen_defaults"},
        "latency": module.timing_summary([2.0] * 50, 16),
    }
    candidate = {**common, "point_role": "deployment_candidate", "deployment": deployment()}
    reference = {
        **common, "point_role": "reference_only",
        "deployment": {**deployment(), "backend": "faiss-gpu-fp32", "candidate_ids_sha256": "fp32-actual"},
    }
    assert module.match_latency_reference(candidate, reference)
    for field, value in (
        ("session", 1), ("batch", 1), ("boundary", "local-cascade"), ("kind", "profile"),
        ("sources", {}), ("latency", module.timing_summary([2.0] * 3, 16)),
        ("schedule", module.query_schedule(panel(1097), 1, 16)),
        ("count_scope", {"warmup_batches": 1, "measured_batches": 3, "policy": "old-three-sample"}),
    ):
        with pytest.raises(ValueError, match="latency|reference|deployment"):
            module.match_latency_reference(candidate, {**reference, field: value})


class ExactCPU:
    """A real CPU dot-product reference, not a stand-in claiming CUDA measurements."""

    def __init__(self, items):
        self.items = torch.from_numpy(items.astype(np.float32))

    def search(self, queries, k, probes):
        return (queries @ self.items.T).topk(k, dim=1)


class SmallCPUEncoder:
    """A CPU linear encoder exercising raw dispatch, not evidence about Qwen/CUDA."""

    def __init__(self):
        self.model = torch.nn.Linear(3, 4, bias=False).eval()
        self.calls = []

    def encode(self, texts, query=True):
        assert query
        self.calls.append(tuple(texts))
        rows = torch.tensor([int(text.rsplit(" ", 1)[1]) for text in texts])
        return self.model(torch.nn.functional.one_hot(rows, 3).float()), None


def test_actual_cpu_candidate_generation_preserves_ids_scores_and_full_depth():
    module = harness()
    items = np.eye(4, dtype=np.float16)
    queries = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    scores, ids = module.ann_predictions(ExactCPU(items), queries, 4, 2, catalog_items=4, batch=1)
    assert ids.tolist() == [[3, 2, 1, 0], [0, 1, 2, 3]]
    np.testing.assert_array_equal(scores, [[4, 3, 2, 1], [4, 3, 2, 1]])


def test_pipeline_raw_operation_uses_fresh_forwards_not_the_historical_query_bank():
    module = harness()
    from rrbench.semantic_rankers import make_cached_ranker

    rng = np.random.default_rng(17)
    items = rng.normal(size=(1100, 4)).astype(np.float32)
    items = (items / np.linalg.norm(items, axis=1, keepdims=True)).astype(np.float16)
    queries = items[:3]
    partition = Partition(
        items[:2].astype(np.float32), np.zeros(len(items), dtype=np.int64),
        np.arange(len(items), dtype=np.int64), np.array([0, len(items), len(items)], dtype=np.int64),
    )
    ranker = make_cached_ranker("semantic-mlp", 4, 2, torch.device("cpu"), 17).eval()
    encoder = SmallCPUEncoder()
    operation = module.PipelineOperation(
        ExactCPU(items), encoder, ranker, items, partition, np.full_like(queries, np.nan), panel(3),
        torch.device("cpu"), cap=50, probes=64, lengths=np.full(len(items), 8, dtype=np.int64),
    )
    integrated = operation([0, 1], "integrated")
    local = operation([0, 1], "local-cascade")
    assert len(encoder.calls) == 2
    assert torch.isfinite(integrated[3]).all()
    module.assert_parity(integrated, local)
    assert integrated[0].shape == (2, 1000)
    assert integrated[1].shape == (2, 50)
    assert integrated[2].shape == (2, 10)
    for row in range(2):
        assert set(integrated[2][row].tolist()) <= set(integrated[0][row, :50].tolist())
        candidates = integrated[0][row, :50]
        expected = ranker(
            integrated[3][row].expand(50, -1),
            torch.from_numpy(items[candidates.numpy()].astype(np.float32)),
        )
        torch.testing.assert_close(integrated[1][row], expected)
    with pytest.raises(ValueError, match="historical|raw"):
        operation([0], "integrated", raw=False)
    operation.encoder = None
    with pytest.raises(ValueError, match="encoder"):
        operation([0], "integrated", raw=True)


def execution_population(source):
    return harness().population_binding(source, source, "timing-panel-sha", "quality-panel-sha")


def test_execution_groups_cover_every_quality_row_once_including_ordered_remainder():
    module = harness()
    source = panel(1097)
    manifest = module.planned_execution_groups(source, execution_population(source), 0, 16)
    groups = manifest["groups"]
    assert len(groups) == 69
    assert [len(group) for group in groups] == [16] * 68 + [9]
    assert sorted(row for group in groups for row in group) == list(range(1097))
    assert manifest["measured_group_indices"] == list(range(50))
    assert manifest["warmup_group_indices"] == list(range(50, 60))
    assert sum(len(groups[index]) for index in manifest["probe_group_indices"]) == 112
    assert manifest["query_execution_policy"] == "raw-execution-groups-v2"
    assert manifest["execution_groups_sha256"] == module.digest(
        {key: value for key, value in manifest.items() if key != "execution_groups_sha256"}
    )
    relabeled = {**source, "qrels": {}}
    assert module.planned_execution_groups(relabeled, execution_population(source), 0, 16) == manifest


def test_reduced_scorer_or_profile_budgets_only_select_frozen_whole_groups():
    module = harness()
    source = panel(1097)
    manifest = module.planned_execution_groups(source, execution_population(source), 1, 16)
    before = copy.deepcopy(manifest)
    full = module.select_execution_groups(manifest, 10, 50)
    text = module.select_execution_groups(manifest, 2, 8)
    profile = module.select_execution_groups(manifest, 1, 1)
    assert text["sample_rows"] == full["sample_rows"][:8]
    assert text["warmup_rows"] == full["warmup_rows"][-2:]
    assert profile["sample_rows"] == full["sample_rows"][:1]
    assert manifest == before
    assert all(rows in manifest["groups"] for rows in text["warmup_rows"] + text["sample_rows"])
    with pytest.raises(ValueError, match="group"):
        module.select_execution_groups(manifest, 10, 51)
    profile["sample_rows"][0][0] = -1
    assert manifest == before  # Selectors must not expose mutable aliases into the frozen manifest.


def test_execution_schedule_rejects_unregistered_or_reordered_timed_groups():
    module = harness()
    source = panel(1097)
    manifest = module.planned_execution_groups(source, execution_population(source), 0, 16)
    schedule = module.select_execution_groups(manifest, 2, 8)
    module.require_execution_schedule(manifest, schedule, 2, 8)
    for change in ("remixed", "reordered", "nonprefix"):
        modified = copy.deepcopy(schedule)
        if change == "remixed":
            modified["sample_rows"][0][0], modified["sample_rows"][1][0] = (
                modified["sample_rows"][1][0], modified["sample_rows"][0][0],
            )
        elif change == "reordered":
            modified["sample_rows"][0].reverse()
        else:
            modified["sample_rows"][0] = list(manifest["groups"][20])
        modified["schedule_sha256"] = module.digest(
            {key: value for key, value in modified.items() if key != "schedule_sha256"}
        )
        with pytest.raises(ValueError, match="group|schedule"):
            module.require_execution_schedule(manifest, modified, 2, 8)


def test_execution_manifest_rejects_corruption_and_empty_self_consistent_scope():
    module = harness()
    source = panel(100)
    manifest = module.planned_execution_groups(source, execution_population(source), 0, 16, smoke=True)
    changed = copy.deepcopy(manifest)
    changed["groups"][0][0] = changed["groups"][0][1]
    with pytest.raises(ValueError, match="group"):
        module.select_execution_groups(changed, 1, 1)
    empty = {
        **manifest, "groups": [], "query_count": 0, "probe_group_indices": [],
        "warmup_group_indices": [], "measured_group_indices": [],
    }
    empty["execution_groups_sha256"] = module.digest(
        {key: value for key, value in empty.items() if key != "execution_groups_sha256"}
    )
    with pytest.raises(ValueError, match="group"):
        module.execute_quality_groups(GroupGeometryCPUOperation(), empty)


def test_execution_groups_map_timing_subset_rows_to_canonical_full_quality_rows():
    module = harness()
    full = panel(1097)
    rows = [(row * 37) % 1097 for row in range(1000)]
    timing = {
        "query_ids": [full["query_ids"][row] for row in rows],
        "queries": [full["queries"][row] for row in rows],
        "qrels": {full["query_ids"][row]: full["qrels"][full["query_ids"][row]] for row in rows},
        "binary_threshold": 1,
    }
    mapping = module.population_binding(timing, full, "subset-sha", "full-sha")
    manifest = module.planned_execution_groups(full, mapping, 2, 16)
    expected = module.query_schedule(timing, 2, 16, 10, 50)
    selected = module.select_execution_groups(manifest, 10, 50)
    assert selected["sample_rows"] == [[rows[row] for row in group] for group in expected["sample_rows"]]
    assert selected["warmup_rows"] == [[rows[row] for row in group] for group in expected["warmup_rows"]]
    assert sorted(row for group in manifest["groups"] for row in group) == list(range(1097))
    assert set(row for group in selected["sample_rows"] for row in group) <= set(rows)


def test_100_row_smoke_has_six_full_groups_plus_four_and_probes_the_entire_manifest():
    module = harness()
    source = panel(100)
    manifest = module.planned_execution_groups(source, execution_population(source), 0, 16, smoke=True)
    assert [len(group) for group in manifest["groups"]] == [16] * 6 + [4]
    assert manifest["probe_group_indices"] == list(range(7))
    selected = module.select_execution_groups(manifest, 1, 5)
    assert len({row for group in selected["sample_rows"] + selected["warmup_rows"] for row in group}) == 96
    with pytest.raises(ValueError, match="group"):
        module.select_execution_groups(manifest, 2, 8)


class GroupGeometryCPUOperation:
    """CPU arithmetic deliberately dependent on complete group context, never a CUDA mock."""

    def __init__(self):
        self.calls = []

    def __call__(self, rows, boundary, raw=True):
        assert raw and boundary == "integrated"
        self.calls.append(list(rows))
        row = torch.tensor(rows, dtype=torch.int64)
        query = torch.nn.functional.normalize(torch.stack((
            row.float() + 1, torch.full_like(row, len(rows), dtype=torch.float32),
        ), 1), dim=1).half().float()
        return row[:, None], query[:, :1], row[:, None], query


def test_raw_quality_forwards_exact_groups_then_scatters_all_four_outputs():
    module = harness()
    source = panel(100)
    manifest = module.planned_execution_groups(source, execution_population(source), 0, 16, smoke=True)
    operation = GroupGeometryCPUOperation()
    arrays = module.execute_quality_groups(operation, manifest)
    assert operation.calls == manifest["groups"]
    assert len(arrays) == 4
    np.testing.assert_array_equal(arrays[0][:, 0], np.arange(100))
    np.testing.assert_array_equal(arrays[2][:, 0], np.arange(100))
    assert arrays[3].shape == (100, 2)
    for group in manifest["groups"]:
        replay = operation(group, "integrated")
        module.assert_parity(replay, tuple(torch.from_numpy(array[group]) for array in arrays))


def test_historical_query_drift_is_diagnostic_not_the_active_replay_gate():
    module = harness()
    reference = parity_result()
    old = reference[3].clone()
    old[0, 1] = 0.004
    old = old.half().float()
    diagnostic = module.query_drift_statistics(reference[3].numpy(), old.numpy())
    assert diagnostic["maximum_absolute_difference"] > 1e-4
    assert "limits" not in diagnostic
    assert module.assert_parity(reference, reference)["maximum_absolute_difference"] == 0
    with pytest.raises(ValueError, match="query"):
        module.assert_parity(reference, (*reference[:3], old))


def test_v2_matching_cannot_be_certified_by_policy_name_and_historical_sha_only():
    module = harness()
    for key in ("execution_queries_sha256", "execution_groups_sha256", "query_execution_policy"):
        incomplete = deployment()
        incomplete.pop(key)
        with pytest.raises(ValueError, match="deployment"):
            module.match_quality_timing({"deployment": incomplete}, {"deployment": copy.deepcopy(incomplete)})
    with pytest.raises(ValueError, match="deployment"):
        module.match_quality_timing({"deployment": deployment()}, {"deployment": deployment(), "batch": 1})


def test_pipeline_requires_explicit_v2_policy_but_ann_still_accepts_frozen_v1():
    module = harness()
    v1 = frozen_profiling()
    assert module.resolve_protocol_controls(
        {"operation": "ann", "backend": "silver-fp32", "session": 0}, v1,
    )["samples"] == 50
    parameters = {"operation": "pipeline", "backend": "faiss-gpu-fp16", "ranker": "semantic-mlp", "session": 0}
    with pytest.raises(ValueError, match="v2|raw-execution-groups"):
        module.resolve_protocol_controls(parameters, v1)
    v2 = module.resolve_protocol_controls(parameters, frozen_profiling(pipeline=True))
    assert v2["query_execution_policy"] == "raw-execution-groups-v2"


def test_pipeline_cli_rejects_v1_and_labels_query_input_historical():
    module = harness()
    arguments = [
        "pipeline", "--items", "items.npy", "--partition", "ivf.npz", "--panel", "panel1097.json",
        "--backend", "faiss-gpu-fp16", "--session", "0", "--output", "new-v2",
        "--registry", "registry.json", "--ranker", "bge-m3-rerank", "--source", "collection.tsv",
        "--offsets", "offsets.npy", "--cache", "models", "--smoke", "smoke.json",
        "--historical-queries", "historical-vectors.npy", "--candidate-cap", "50",
    ]
    parsed = module.parse_arguments(arguments)
    assert parsed.protocol.name == "semantic-extension-pipeline-v2.json"
    assert parsed.query_execution_policy == "raw-execution-groups-v2"
    v1 = Path(__file__).resolve().parents[1] / "configs" / "semantic-extension-v1.json"
    before = sha256(v1)
    with pytest.raises(ValueError, match="v2|raw-execution-groups"):
        module.parse_arguments([*arguments, "--protocol", str(v1)])
    assert sha256(v1) == before


def test_oracle_binding_cannot_mix_same_shaped_embedding_spaces():
    module = harness()
    binding = {"items_sha256": "items", "queries_sha256": "queries", "k": 1000, "product": "FP32", "tf32": False}
    module.check_oracle_binding(binding, {"items": "items", "queries": "queries"})
    for field, value in (("items_sha256", "other-items"), ("queries_sha256", "other-queries"), ("tf32", True)):
        with pytest.raises(ValueError, match="oracle"):
            module.check_oracle_binding({**binding, field: value}, {"items": "items", "queries": "queries"})


def test_file_binding_includes_real_source_bytes_and_parameter_changes():
    module = harness()
    path = Path(__file__)
    first = module.build_binding({"input": path}, {"samples": 10})
    assert first["sources"]["input"] == sha256(path)
    assert first["input_paths"]["input"] == str(path.resolve())
    assert first != module.build_binding({"input": path}, {"samples": 11})


def test_a_profile_receipt_without_cuda_trace_evidence_is_false_completion():
    module = harness()
    receipt, binding, records, hashes = completion_fixture()
    del records[0]["latency"]
    records[0]["kind"] = "profile"
    receipt["records_sha256"] = module.digest(records)
    with pytest.raises(ValueError, match="trace"):
        module.verify_completion(receipt, binding, ["c"], records, hashes)


def test_immutable_output_and_artifact_paths_are_not_implicitly_repaired():
    module = harness()
    with pytest.raises(ValueError, match="Incomplete"):
        module.reuse_run(Path(__file__).parent, {}, [])
    with pytest.raises(ValueError, match="Artifact"):
        module.artifact_paths(Path(__file__).parent, ["../src/rrbench/semantic_profiling.py"])


def test_memory_gate_accounts_for_observed_torch_peak_above_current_allocation():
    with pytest.raises(harness().UnsupportedMemory):
        harness().check_memory({
            "memory_used_bytes": 70 << 30, "device_memory_bytes": 80 << 30,
            "torch_allocated_bytes": 20 << 30, "torch_peak_allocated_bytes": 23 << 30,
        })


def test_main_rejects_invalid_controls_before_source_io_or_gpu_work():
    with pytest.raises(ValueError, match="warmup"):
        harness().main([
            "ann", "--items", "absent.npy", "--partition", "absent.npz", "--panel", "absent.json",
            "--queries", "absent-q.npy", "--backend", "silver-fp32", "--session", "1",
            "--warmup", "0", "--output", "must-not-be-created",
        ])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA backend equivalence requires CUDA.")
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "int8-rowwise"])
def test_instrumented_silver_matches_unmodified_search_on_real_cuda(dtype):
    pytest.importorskip("silvertorch.ops._load_ops")
    from rrbench.ann import SilverANN
    from rrbench.runtime import configure

    configure(17, 4)
    rng = np.random.default_rng(17)
    items = rng.normal(size=(512, 32)).astype(np.float32)
    items = (items / np.linalg.norm(items, axis=1, keepdims=True)).astype(np.float16)
    partition = Partition(
        items[:4].astype(np.float32), np.repeat(np.arange(4, dtype=np.int64), 128),
        np.arange(512, dtype=np.int64), np.arange(0, 513, 128, dtype=np.int64),
    )
    ann = SilverANN(items, partition, dtype)
    queries = torch.as_tensor(items[:3].astype(np.float32), device="cuda")
    expected = ann.search(queries, 10, 2)
    actual = harness().profiled_search(ann, queries, 10, 2)
    assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real FAISS GPU equivalence requires CUDA.")
@pytest.mark.parametrize("half", [False, True])
def test_instrumented_faiss_uses_the_real_native_search_and_declared_code_bytes(half):
    faiss = pytest.importorskip("faiss")
    if not hasattr(faiss, "StandardGpuResources"):
        pytest.skip("The installed FAISS build has no GPU backend.")
    from rrbench.semantic_ann import SemanticFaissANN

    items = np.eye(32, dtype=np.float16)
    partition = Partition(
        items[:2].astype(np.float32), np.zeros(len(items), dtype=np.int64),
        np.arange(len(items), dtype=np.int64), np.array([0, len(items), len(items)], dtype=np.int64),
    )
    ann = SemanticFaissANN(items, partition, gpu=True, half=half)
    queries = torch.as_tensor(items[:2].astype(np.float32), device="cuda")
    expected = ann.search(queries, 10, 2)
    actual = harness().profiled_search(ann, queries, 10, 2)
    assert ann.code_bytes == 32 * (2 if half else 4)
    assert torch.equal(expected[1], actual[1])
    torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)
