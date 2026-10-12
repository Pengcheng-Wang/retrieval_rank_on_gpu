import copy
import datetime as dt
import importlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from rrbench.io import sha256


@pytest.fixture
def run():
    if importlib.util.find_spec("rrbench.semantic_query_adaptation_run") is None:
        pytest.skip("The runner implementation has not been created yet.")
    return importlib.import_module("rrbench.semantic_query_adaptation_run")


def test_runner_module_exists():
    assert importlib.util.find_spec("rrbench.semantic_query_adaptation_run") is not None


@pytest.fixture
def work():
    # Keep test artifacts in the checkout, never the system temporary directory.
    import uuid
    import shutil

    root = Path("tests") / (".adaptation-test-" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


def panel(ids=("a", "b"), texts=("alpha", "beta")):
    return {
        "name": "fit", "query_ids": list(ids), "queries": list(texts),
        "qrels": {query: {"0": 1, "1": 1} for query in ids}, "binary_threshold": 1,
    }


def test_atomic_receipts_cannot_overwrite_or_escape(run, work):
    run.publish_json(work, "proof.json", {"complete": True})
    original = (work / "proof.json").read_bytes()
    with pytest.raises(FileExistsError):
        run.publish_json(work, "proof.json", {"complete": False})
    assert (work / "proof.json").read_bytes() == original
    for path in (".", "./", "../escape.json", "..\\escape.json", "/root.json", "C:\\root.json"):
        with pytest.raises(ValueError, match="relative|traversal|scope"):
            run.publish_json(work, path, {})
    assert not list(work.glob("*.pending-*"))


def test_json_duplicate_keys_and_nonfinite_values_fail(run, work):
    (work / "bad.json").write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        run.read_json(work / "bad.json")
    with pytest.raises(ValueError):
        run.publish_json(work, "nan.json", {"x": float("nan")})
    assert not (work / "nan.json").exists()


def test_complete_phase_reuse_checks_binding_inventory_and_bytes(run, work):
    binding = {"phase": "audit", "source": "fixed"}
    assert run.begin_phase(work, binding) is None
    run.publish_json(work, "audit.json", {"pass": True})
    result = run.finish_phase(work, binding, {"rows": 2})
    assert run.begin_phase(work, binding) == result
    with pytest.raises(ValueError, match="binding"):
        run.begin_phase(work, {**binding, "source": "changed"})
    (work / "audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|bytes"):
        run.begin_phase(work, binding)


def test_incomplete_or_uninventoried_phase_is_not_reusable(run, work):
    binding = {"phase": "mine"}
    run.begin_phase(work, binding)
    with pytest.raises(ValueError, match="Incomplete"):
        run.begin_phase(work, binding)
    run.finish_phase(work, binding, {})
    (work / "unlisted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        run.begin_phase(work, binding)


def test_panel_validation_preserves_all_positives_and_rejects_bad_rows(run):
    source = panel()
    source["qrels"]["a"]["2"] = 0
    assert run.validate_panel(source, rows=2, catalog_items=10) == [[0, 1], [0, 1]]
    for bad in (
        {**source, "query_ids": ["a", "a"]},
        {**source, "queries": ["alpha", "  "]},
        {**source, "queries": ["Ａlpha", "alpha"]},
        {**source, "qrels": {"a": {"10": 1}, "b": {"0": 1}}},
        {**source, "qrels": {"a": {"01": 1, "1": 1}, "b": {"0": 1}}},
        {**source, "queries": ["alpha"]},
    ):
        with pytest.raises(ValueError):
            run.validate_panel(bad, rows=2, catalog_items=10)


def test_cohort_disjointness_uses_canonical_text_not_only_ids(run):
    fit = panel(("a",), ("Café   Query",))
    validation = panel(("v",), ("beta",))
    final = panel(("z",), ("cafe\u0301 query",))
    with pytest.raises(ValueError, match="disjoint"):
        run.require_disjoint_panels(fit, validation, final)
    final["queries"] = ["gamma"]
    run.require_disjoint_panels(fit, validation, final)
    final["query_ids"] = ["a"]
    with pytest.raises(ValueError, match="disjoint"):
        run.require_disjoint_panels(fit, validation, final)


def test_query_receipt_requires_exact_panel_order_revision_and_bytes(run, work):
    source = panel()
    panel_path = work / "panel.json"
    run.publish_json(work, "panel.json", source)
    vectors = np.eye(2, dtype=np.float16)
    run.publish_array(work, "vectors.npy", vectors)
    spec = {
        "model": "model", "revision": "revision", "dimensions": [2],
        "query_instruction": run.QUERY_INSTRUCTION,
    }
    receipt = {
        "complete": True, "model": "model", "revision": "revision", "rows": 2,
        "native_dimension": 2, "mrl_representations": {},
        "query_ids": ["a", "b"], "panel_sha256": sha256(panel_path),
        "vectors_sha256": sha256(work / "vectors.npy"),
        "specification_sha256": run.configuration_digest(spec),
        "query_instruction": run.QUERY_INSTRUCTION,
    }
    actual = run.checked_queries(work / "vectors.npy", receipt, panel_path, source, spec)
    np.testing.assert_array_equal(actual, vectors)
    for field, value in (
        ("query_ids", ["b", "a"]), ("revision", "wrong"),
        ("panel_sha256", "wrong"), ("vectors_sha256", "wrong"), ("rows", 7077),
    ):
        with pytest.raises(ValueError):
            run.checked_queries(
                work / "vectors.npy", {**receipt, field: value}, panel_path, source, spec,
            )


class FullStringTokenizer:
    padding_side = "left"

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.seen = []

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        assert "max_length" not in kwargs
        self.seen.extend(texts)
        offset = len(self.seen) - len(texts)
        return {"length": self.lengths[offset:offset + len(texts)]}


def test_length_audit_uses_every_actual_instructed_string_and_actual_padding(run):
    tokenizer = FullStringTokenizer([3, 9, 512, 513])
    texts = ["a", "b", "c", "d"]
    result, lengths = run.token_audit(tokenizer, texts, [[0, 1], [2, 3]])
    assert tokenizer.seen == [f"Instruct: {run.QUERY_INSTRUCTION}\nQuery:{x}" for x in texts]
    assert result["count"] == 4
    assert result["overflow"] == 1
    assert result["pass"] is False
    assert result["real_tokens"] == 1037
    assert result["padded_tokens"] == 2 * 16 + 2 * 520
    assert result["max"] == 513
    assert result["generation_metadata"].startswith("N/A")
    np.testing.assert_array_equal(lengths, [3, 9, 512, 513])


def test_negative_sampling_is_query_stable_distinct_and_excludes_all_positives(run):
    query_ids = ["a", "b"]
    positives = [[0, 1, 2, 3, 4], [10, 11]]
    retrieved = np.array([np.arange(100), np.arange(20, 120)], dtype=np.int64)
    result = run.sample_negatives(query_ids, positives, retrieved, catalog_items=200)
    assert result["uniform_ids"].shape == (2, 64)
    assert result["hard_ids"].shape == (2, 12)
    assert result["positive_mask"][0].sum() == 5
    for row, expected in enumerate(positives):
        valid = result["candidate_ids"][row][result["valid_mask"][row]]
        marked = result["candidate_ids"][row][result["positive_mask"][row]]
        assert len(valid) == len(set(valid))
        assert set(marked) == set(expected)
        assert not set(result["hard_ids"][row]) & set(expected)
        assert not set(result["uniform_ids"][row]) & (
            set(expected) | set(result["hard_ids"][row])
        )
    reordered = run.sample_negatives(
        query_ids[::-1], positives[::-1], retrieved[::-1], catalog_items=200,
    )
    for name in result:
        np.testing.assert_array_equal(result[name], reordered[name][::-1])


def test_negative_sampling_preserves_large_positive_sets_and_rejects_bad_source_ids(run):
    positives = [list(range(110))]
    result = run.sample_negatives(["a"], positives, np.arange(100)[None, :], 200)
    assert result["positive_mask"].sum() == 110
    assert np.all(result["hard_ids"] == -1)
    assert result["valid_mask"].sum() == 174
    for retrieved in (np.array([[1, 1]]), np.array([[200]]), np.array([[-2]])):
        with pytest.raises(ValueError):
            run.sample_negatives(["a"], positives, retrieved, 200)
    with pytest.raises(ValueError, match="uniform|catalog"):
        run.sample_negatives(["a"], [list(range(190))], np.array([[195]]), 200)


def test_execution_groups_are_label_independent_total_and_fixed_across_contexts(run):
    ids = [f"q{i}" for i in range(1097)]
    groups = run.execution_groups(ids)
    assert sorted(row for group in groups for row in group) == list(range(1097))
    assert all(len(group) == 16 for group in groups[:-1])
    assert len(groups[-1]) == 9
    assert run.execution_groups(ids) == groups
    schedules = [run.runtime_schedule(groups, session) for session in range(3)]
    for schedule in schedules:
        assert len(schedule["warmup"]) == 10
        assert len(schedule["measured"]) == 50
        assert len(schedule["probes"]) == 100
        assert len(set(schedule["measured"])) == 50
        assert all(len(groups[index]) == 16 for index in schedule["probes"])
    assert schedules[0] != schedules[1]
    assert run.paired_order(0, 0) == ("base", "adapter")
    assert run.paired_order(1, 0) == ("adapter", "base")
    with pytest.raises(ValueError):
        run.runtime_schedule(groups, 3)


def test_optimizer_microbatches_weight_the_actual_short_block_equally(run):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    coefficients = torch.arange(1, 17, dtype=torch.float32) / 100
    seen = []

    def losses(rows):
        seen.extend(rows)
        return parameter * coefficients[rows]

    result = run.optimizer_step([parameter], optimizer, list(range(16)), losses)
    expected = 1 - 0.01 * coefficients.mean()
    torch.testing.assert_close(parameter.detach()[0], expected, rtol=0, atol=1e-7)
    assert seen == list(range(16))
    assert result["queries"] == 16 and result["microbatches"] == 2
    assert result["loss"] == pytest.approx(float(coefficients.mean()))


def test_optimizer_rejects_nonfinite_or_wrong_shape_losses_before_update(run):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1)
    for callback in (lambda rows: parameter * float("nan"), lambda rows: parameter.sum()):
        with pytest.raises(ValueError, match="per-query|finite"):
            run.optimizer_step([parameter], optimizer, [0, 1], callback)
    torch.testing.assert_close(parameter, torch.ones(1))


def test_checkpoint_roundtrip_restores_only_adapters_optimizer_and_rng(run, work):
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0)
    run.optimizer_step([parameter], optimizer, list(range(2)), lambda rows: parameter[rows] ** 2)
    binding = {"phase": "train", "source_sha256": "source"}
    state = run.make_checkpoint(
        {"adapter": parameter}, optimizer, binding, completed_steps=1, queries_seen=2,
        total_queries=4, effective_batch=2, loss_history=[1.0],
    )
    path = run.publish_checkpoint(work, state)
    assert path.name == "step-000001.pt"
    expected_random = (torch.rand(3), np.random.random(3), random.random())
    run.optimizer_step([parameter], optimizer, list(range(2)), lambda rows: parameter[rows] ** 2)
    expected_parameter = parameter.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restored, _ = run.read_checkpoint(work, binding, total_queries=4, effective_batch=2)
    assert set(restored["adapters"]) == {"adapter"}
    assert "model" not in restored and "base" not in restored
    parameter.data.copy_(restored["adapters"]["adapter"])
    optimizer.load_state_dict(restored["optimizer"])
    run.restore_rng(restored["rng"])
    torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.random(3), expected_random[1])
    assert random.random() == expected_random[2]
    run.optimizer_step([parameter], optimizer, list(range(2)), lambda rows: parameter[rows] ** 2)
    torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)
    assert run.tensor_tree_equal(optimizer.state_dict(), expected_optimizer)
    with pytest.raises(FileExistsError):
        run.publish_checkpoint(work, state)
    with pytest.raises(ValueError, match="final|epoch"):
        run.read_checkpoint(work, binding, total_queries=4, effective_batch=2, require_final=True)
    with pytest.raises(ValueError, match="binding"):
        run.read_checkpoint(work, {"source_sha256": "other"}, total_queries=4, effective_batch=2)


def test_checkpoint_rejects_incomplete_or_changed_payload_and_wrong_logical_position(run, work):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="position"):
        run.make_checkpoint(
            {"adapter": parameter}, optimizer, {}, completed_steps=781, queries_seen=50000,
            loss_history=[1.0] * 781,
        )
    (work / "step-000100.pt").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="incomplete|metadata"):
        run.read_checkpoint(work, {})


def test_final_checkpoint_requires_exact_782_steps_and_50000_queries(run):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0)
    run.optimizer_step([parameter], optimizer, [0], lambda rows: parameter.square())
    optimizer.state[parameter]["step"].fill_(782)
    state = run.make_checkpoint(
        {"adapter": parameter}, optimizer, {}, completed_steps=782, queries_seen=50000,
        loss_history=[1.0] * 782,
    )
    run.validate_checkpoint(state, {}, require_final=True)
    for field, value in (("completed_steps", 781), ("queries_seen", 49999), ("completed_epochs", 0)):
        with pytest.raises(ValueError):
            run.validate_checkpoint({**state, field: value}, {}, require_final=True)


def test_admission_enforces_bound_sources_and_deadline_without_gpu(run):
    protocol = json.loads((Path("configs") / "semantic-extension-query-adaptation-v1.json").read_text())
    admitted = dt.datetime(2026, 10, 12, 4, 7, 29, tzinfo=dt.timezone.utc)
    sources = {"items": "a" * 64}
    receipt = {
        "complete": True, "decision": "admitted", "primary_evidence_verified": True,
        "admitted_at_utc": admitted.isoformat(),
        "engineering_ready_deadline_utc": (admitted + dt.timedelta(hours=2)).isoformat(),
        "gpu_end_utc": (admitted + dt.timedelta(hours=5)).isoformat(),
        "hard_end_utc": (admitted + dt.timedelta(hours=6)).isoformat(),
        "protocol_sha256": "p" * 64, "sources": sources,
    }
    deadline = run.validate_admission(
        receipt, protocol, "p" * 64, sources, "audit",
        admitted + dt.timedelta(hours=1), now=admitted,
    )
    assert deadline == admitted + dt.timedelta(hours=1)
    for changed in (
        {**receipt, "sources": {"items": "b" * 64}},
        {**receipt, "primary_evidence_verified": False},
        {**receipt, "gpu_end_utc": (admitted + dt.timedelta(hours=7)).isoformat()},
    ):
        with pytest.raises(ValueError):
            run.validate_admission(
                changed, protocol, "p" * 64, sources, "audit",
                admitted + dt.timedelta(hours=1), now=admitted,
            )
    with pytest.raises(TimeoutError):
        run.check_deadline(admitted, now=admitted)


def test_memory_failure_classification_does_not_swallow_unrelated_runtime_errors(run):
    assert run.memory_failure(torch.cuda.OutOfMemoryError("test"))
    assert run.memory_failure(run.UnsupportedMemory("test"))
    assert run.memory_failure(RuntimeError("StandardGpuResources alloc fail: out of memory"))
    assert not run.memory_failure(RuntimeError("CUDA illegal memory access"))
    assert not run.memory_failure(ValueError("bad source"))
    with pytest.raises(run.UnsupportedMemory):
        run.validate_memory_sample({
            "memory_used_bytes": 49 << 30, "device_memory_bytes": 80 << 30,
            "torch_allocated_bytes": 1, "torch_peak_allocated_bytes": 1,
            "cuda_free_bytes": 31 << 30,
        })


def test_resident_exact_search_uses_existing_normal_tensor_and_full_catalog(run):
    bank = run.bank_tensor(np.eye(5, dtype=np.float16), torch.device("cpu"), torch.float32)
    assert not torch.is_inference(bank) and not bank.requires_grad
    pointer = bank.data_ptr()
    query = torch.tensor([[0.1, 0.8, 0.5, 0.2, 0.3]], requires_grad=True)
    scores, ids = run.exact_search(bank, query, k=3)
    assert ids.tolist() == [[1, 2, 4]]
    assert bank.data_ptr() == pointer
    scores.sum().backward()
    assert query.grad is not None and bank.grad is None


def test_paired_bootstrap_keeps_query_identity_and_population_separate(run):
    base = [{"query_id": "a", "mrr@10": 0.0}, {"query_id": "b", "mrr@10": 0.5}]
    adapter = [{"query_id": "a", "mrr@10": 0.5}, {"query_id": "b", "mrr@10": 1.0}]
    result = run.paired_bootstrap(base, adapter, metrics=("mrr@10",), draws=200)
    assert result["queries"] == 2
    assert result["metrics"]["mrr@10"]["difference"] == pytest.approx(0.5)
    assert result["metrics"]["mrr@10"]["ci95"] == [0.5, 0.5]
    with pytest.raises(ValueError, match="query"):
        run.paired_bootstrap(base, adapter[::-1], metrics=("mrr@10",), draws=200)


def test_fixed_protocol_rejects_recipe_or_measurement_changes(run):
    protocol = json.loads((Path("configs") / "semantic-extension-query-adaptation-v1.json").read_text())
    run.validate_protocol(protocol)
    for section, field, value in (
        ("training", "optimizer_updates", 781), ("training", "learning_rate", 3e-5),
        ("model", "merge", True), ("evaluation", "samples", 5),
        ("data", "fit_queries", 49999), ("memory", "maximum_observed_benchmark_gib", 80),
    ):
        changed = copy.deepcopy(protocol)
        changed[section][field] = value
        with pytest.raises(ValueError, match="fixed|protocol"):
            run.validate_protocol(changed)


def test_explicit_inputs_require_every_path_and_disallow_alias_or_traversal(run, work):
    paths = {}
    for key in run.INPUT_KEYS:
        target = work / key
        if key == "cache":
            target.mkdir()
        else:
            target.write_bytes(b"source")
        paths[key] = str(target.absolute())
    run.publish_json(work, "inputs.json", paths)
    actual = run.read_input_paths(work / "inputs.json")
    assert set(actual) == set(run.INPUT_KEYS)
    for suffix, changed in (
        ("missing", {key: value for key, value in paths.items() if key != "final_panel"}),
        ("alias", {**paths, "final_panel": paths["fit_panel"]}),
        ("traversal", {**paths, "final_panel": str(work / ".." / "final_panel")}),
    ):
        run.publish_json(work, suffix + ".json", changed)
        with pytest.raises(ValueError):
            run.read_input_paths(work / (suffix + ".json"))


def test_item_bank_checks_every_row_id_and_pinned_representation(run, work):
    run.publish_array(work, "items.npy", np.eye(2, dtype=np.float16))
    run.publish_array(work, "ids.npy", np.arange(2, dtype=np.int64))
    spec = {"model": "model", "revision": "revision", "dimensions": [2]}
    receipt = {
        "complete": True, "rows": 2, "stored_dimension": 2, "native_dimension": 2,
        "stored_dtype": "float16", "vectors_sha256": sha256(work / "items.npy"),
        "ids_sha256": sha256(work / "ids.npy"), "mrl_representations": {},
        "representation": {
            "model": "model", "revision": "revision", "stored_dimension": 2,
            "stored_dtype": "float16", "specification_sha256": run.configuration_digest(spec),
            "normalization": run.STORED_NORMALIZATION,
        },
        "source_shards": [{"start": 0, "rows": 2, "receipt_sha256": "a" * 64}],
    }
    values = run.checked_item_bank(work / "items.npy", work / "ids.npy", receipt, spec, rows=2)
    np.testing.assert_array_equal(values, np.eye(2, dtype=np.float16))
    changed = copy.deepcopy(receipt)
    changed["representation"]["revision"] = "other"
    with pytest.raises(ValueError):
        run.checked_item_bank(work / "items.npy", work / "ids.npy", changed, spec, rows=2)
    np.save(work / "ids.npy", np.array([1, 0]))
    changed = {**receipt, "ids_sha256": sha256(work / "ids.npy")}
    with pytest.raises(ValueError, match="row|mapping"):
        run.checked_item_bank(work / "items.npy", work / "ids.npy", changed, spec, rows=2)


def test_cli_exposes_only_bounded_explicit_phase_contract(run):
    parser = run.argument_parser()
    args = parser.parse_args([
        "--phase", "runtime", "--inputs", "private-inputs.json", "--protocol", "protocol.json",
        "--admission", "private-admission.json", "--output", "private-result",
        "--deadline", "2026-10-12T09:07:29Z", "--session", "2",
    ])
    assert args.phase == "runtime" and args.session == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "train"])


def test_cpu_audit_phase_persists_actual_strings_groups_and_failure_without_success(run, work):
    panels = {
        "fit": panel(("a", "b"), ("alpha", "beta")),
        "validation": panel(("v", "w"), ("gamma", "delta")),
        "final": panel(("x", "y"), ("epsilon", "zeta")),
    }
    paths = {
        name + "_panel": run.publish_json(work, name + "-source.json", value)
        for name, value in panels.items()
    }
    context = run.PilotContext(
        paths=paths, protocol={}, deadline=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1),
        output=work, binding={"sources": {name: sha256(path) for name, path in paths.items()}},
        specification={}, panels=panels, positives=[[0], [1]], items=np.eye(2, dtype=np.float16),
        fit_queries=np.eye(2, dtype=np.float16), validation_queries=np.eye(2, dtype=np.float16),
        partition=None, model_smoke={},
    )
    receipt = run.run_audit(context, tokenizer=FullStringTokenizer([3, 9, 4, 5, 6, 7]))
    assert receipt["complete"] and receipt["details"]["all_inputs_pass"]
    groups = run.read_json(work / "audit" / "groups.json")
    assert groups["final"]["query_ids"] == ["x", "y"]
    assert sorted(sum(groups["final"]["groups"], [])) == [0, 1]
    rendered = run.read_json(work / "audit" / "rendered-samples.json")
    assert rendered["training"] == f"Instruct: {run.QUERY_INSTRUCTION}\nQuery:alpha"
    assert run.run_audit(context, tokenizer=FullStringTokenizer([])) == receipt
    context.output = work / "overflow"
    with pytest.raises(ValueError, match="overflow"):
        run.run_audit(context, tokenizer=FullStringTokenizer([513, 3, 4, 5, 6, 7]))
    assert not (context.output / "audit" / "completed.json").exists()
    assert run.read_json(context.output / "audit" / "audit.json")["fit"]["overflow"] == 1


def test_smoke_schedule_cycles_exactly_100_queries_for_five_full_steps(run):
    blocks = [run.smoke_rows(step) for step in range(5)]
    assert all(len(block) == 64 for block in blocks)
    assert set(sum(blocks, [])) == set(range(100))
    assert blocks[1][:3] == [64, 65, 66]
    assert blocks[1][36] == 0
    with pytest.raises(ValueError):
        run.smoke_rows(5)


def test_quality_replay_checks_vectors_and_candidates_in_the_same_batch_scope(run):
    vectors = np.eye(2, dtype=np.float16)
    ids = np.array([[0, 1], [1, 0]], dtype=np.int64)
    scores = np.array([[1.0, 0], [1.0, 0]], dtype=np.float32)
    result = run.assert_quality_replay((vectors, scores, ids), (vectors, scores, ids))
    assert result["canonical_queries_exact"] and result["topk_ids_exact"]
    for actual in (
        (vectors[::-1], scores, ids), (vectors, scores, ids[::-1]),
        (vectors, scores + 0.1, ids),
    ):
        with pytest.raises(ValueError, match="replay"):
            run.assert_quality_replay(actual, (vectors, scores, ids))


def test_nested_completed_files_are_inventoried_not_ignored(run, work):
    binding = {"phase": "audit"}
    run.begin_phase(work, binding)
    run.publish_json(work, "child/completed.json", {"raw": True})
    receipt = run.finish_phase(work, binding, {})
    assert "child/completed.json" in receipt["artifacts"]


def test_timing_excludes_observation_and_synchronizes_actual_operation(run, monkeypatch):
    calls = []
    clock = iter((100, 10_000_100))
    monkeypatch.setattr(run, "observe_memory", lambda device: calls.append("observe") or {"ok": True})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(run.time, "perf_counter_ns", lambda: next(clock))
    actual, elapsed, memory = run.timed_operation(lambda: calls.append("operation") or 7, torch.device("cpu"))
    assert actual == 7 and elapsed == 10.0
    assert calls == ["observe", "sync", "operation", "sync", "observe"]
    assert len(memory) == 2


def test_runtime_summary_requires_50_matched_actual_samples_per_arm(run):
    records = [
        {"arm": arm, "sample": index, "group": index, "milliseconds": ms}
        for index in range(50) for arm, ms in (("base", 2.0), ("adapter", 3.0))
    ]
    result = run.runtime_summary(records)
    assert result["base"]["sample_count"] == 50
    assert result["adapter"]["mean_ms"] == 3.0
    assert result["paired_mean_difference_ms"] == 1.0
    assert len(result["base"]["samples_ms"]) == 50
    for bad in (records[:-1], [{**row, "milliseconds": float("nan")} for row in records]):
        with pytest.raises(ValueError):
            run.runtime_summary(bad)
    changed = copy.deepcopy(records)
    changed[-1]["group"] = 123
    with pytest.raises(ValueError, match="paired|group"):
        run.runtime_summary(changed)


def test_phase_dispatch_does_not_allocate_resources_or_allow_unbound_sessions(run, monkeypatch):
    called = []
    context = object()
    for phase in run.PHASES:
        if phase == "runtime":
            monkeypatch.setattr(run, "run_runtime", lambda ctx, session: called.append((ctx, session)) or {})
        else:
            monkeypatch.setattr(run, "run_" + phase, lambda ctx, name=phase: called.append((ctx, name)) or {})
    for phase in run.PHASES:
        run.execute_phase(context, phase, 2 if phase == "runtime" else None)
    assert called == [(context, name) for name in run.PHASES[:-1]] + [(context, 2)]
    with pytest.raises(ValueError, match="session"):
        run.execute_phase(context, "runtime", None)
    with pytest.raises(ValueError, match="session"):
        run.execute_phase(context, "train", 1)


def test_checkpoint_rejects_missing_optimizer_state_or_mismatched_step_counter(run):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0)
    with pytest.raises(ValueError, match="optimizer"):
        run.make_checkpoint({"adapter": parameter}, optimizer, {}, 1, 64, [1.0])
    run.optimizer_step([parameter], optimizer, [0], lambda rows: parameter.square())
    state = run.make_checkpoint({"adapter": parameter}, optimizer, {}, 1, 64, [1.0])
    changed = copy.deepcopy(state)
    changed["optimizer"]["state"][0]["step"].fill_(2)
    with pytest.raises(ValueError, match="optimizer"):
        run.validate_checkpoint(changed, {})
    changed = copy.deepcopy(state)
    changed["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="optimizer"):
        run.validate_checkpoint(changed, {})


def test_cuda_rng_capture_and_restore_touch_only_the_selected_device(run, monkeypatch):
    touched = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state",
        lambda device: touched.append(("get", device)) or torch.tensor([1, 2], dtype=torch.uint8),
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: touched.append(("set", device)),
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("must not touch other GPUs"))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: pytest.fail("must not touch other GPUs"))
    state = run.capture_rng(cuda=True)
    run.restore_rng(state)
    assert touched == [("get", 3), ("set", 3)]


def test_checkpoint_inventory_rejects_unknown_or_misnamed_payloads(run, work):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, weight_decay=0)
    run.optimizer_step([parameter], optimizer, [0], lambda rows: parameter.square())
    state = run.make_checkpoint({"adapter": parameter}, optimizer, {}, 1, 64, [1.0])
    run.publish_checkpoint(work, state)
    (work / "other-state.pt").write_bytes(b"unattested")
    with pytest.raises(ValueError, match="inventory|metadata"):
        run.read_checkpoint(work, {})
    (work / "other-state.pt").unlink()
    metadata = run.read_json(work / "step-000001.json")
    metadata["completed_steps"] = 2
    (work / "step-000001.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata|position"):
        run.read_checkpoint(work, {})


def test_main_persists_only_recognized_memory_failure_not_other_cuda_errors(run, work, monkeypatch):
    from types import SimpleNamespace

    context = SimpleNamespace(output=work, binding={})
    monkeypatch.setattr(run, "load_context", lambda args: context)
    arguments = [
        "--phase", "audit", "--inputs", "in.json", "--protocol", "p.json",
        "--admission", "a.json", "--output", str(work), "--deadline", "2026-10-12T06:07:29Z",
    ]

    def fail_memory(*_):
        raise torch.cuda.OutOfMemoryError("fixture allocation failed")

    monkeypatch.setattr(run, "execute_phase", fail_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run.main(arguments)
    receipt = run.read_json(work / "audit" / "failure.json")
    assert receipt["complete"] is False and receipt["status"] == "unsupported_memory"
    context.output = work / "other"

    def fail_other(*_):
        raise RuntimeError("CUDA illegal memory access")

    monkeypatch.setattr(run, "execute_phase", fail_other)
    with pytest.raises(RuntimeError, match="illegal"):
        run.main(arguments)
    assert not (context.output / "audit" / "failure.json").exists()


def test_source_recheck_rejects_changed_bytes_or_incomplete_inventory(run, work):
    path = run.publish_json(work, "source.json", {"frozen": True})
    sources = {"source": path}
    expected = {"source": sha256(path)}
    run.verify_file_hashes(sources, expected)
    with pytest.raises(ValueError, match="source|inventory"):
        run.verify_file_hashes({}, expected)
    path.write_text('{"frozen":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="source|checksum"):
        run.verify_file_hashes(sources, expected)


def test_memory_gate_records_the_actual_failed_observation_and_rejects_nan(run):
    sample = {
        "memory_used_bytes": 40 << 30, "device_memory_bytes": 80 << 30,
        "torch_allocated_bytes": 20 << 30, "torch_peak_allocated_bytes": 21 << 30,
        "cuda_free_bytes": 7 << 30,
    }
    with pytest.raises(run.UnsupportedMemory) as failed:
        run.validate_memory_sample(sample)
    assert failed.value.observation == sample
    with pytest.raises(ValueError, match="memory"):
        run.validate_memory_sample({**sample, "cuda_free_bytes": float("nan")})


def test_real_adapter_api_checkpoint_preserves_optimizer_parameter_mapping_on_cpu(run, work):
    from types import SimpleNamespace
    from rrbench import semantic_query_adaptation as core

    def model():
        torch.manual_seed(17)
        result = torch.nn.Module()
        result.config = SimpleNamespace(num_hidden_layers=1)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 4)
        layer.self_attn.v_proj = torch.nn.Linear(4, 2)
        result.layers = torch.nn.ModuleList([layer])
        core.attach_query_adapters(result)
        return result

    current = model()
    optimizer = run.new_optimizer(current)
    values = torch.eye(4)

    def losses(rows):
        attention = current.layers[0].self_attn
        return attention.q_proj(values[rows]).square().mean(1) + attention.v_proj(values[rows]).square().mean(1)

    run.optimizer_step(core.adapter_parameters(current), optimizer, [0, 1], losses)
    state = run.make_checkpoint(
        core.adapter_state(current), optimizer, {}, 1, 2, [1.0], total_queries=4, effective_batch=2,
    )
    run.publish_checkpoint(work, state)
    run.optimizer_step(core.adapter_parameters(current), optimizer, [2, 3], losses)
    expected = core.adapter_state(current)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    current = model()
    optimizer = run.new_optimizer(current)
    saved, _ = run.read_checkpoint(work, {}, total_queries=4, effective_batch=2)
    core.load_adapter_state(current, saved["adapters"])
    optimizer.load_state_dict(saved["optimizer"])
    run.restore_rng(saved["rng"])
    run.optimizer_step(core.adapter_parameters(current), optimizer, [2, 3], losses)
    assert run.tensor_tree_equal(core.adapter_state(current), expected)
    assert run.tensor_tree_equal(optimizer.state_dict(), expected_optimizer)


def test_all_retrieved_score_slots_are_checked_without_silently_accepting_nan(run):
    ids = np.array([[0, 1, -1]], dtype=np.int64)
    scores = np.array([[0.8, 0.5, -np.inf]], dtype=np.float32)
    run.checked_retrieval(ids, scores, rows=1, catalog_items=10, allow_missing=True)
    with pytest.raises(ValueError, match="missing"):
        run.checked_retrieval(ids, scores, rows=1, catalog_items=10)
    scores[0, 2] = np.nan
    with pytest.raises(ValueError, match="score"):
        run.checked_retrieval(ids, scores, rows=1, catalog_items=10, allow_missing=True)
