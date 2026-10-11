import numpy as np
import pytest
import torch

from rrbench import semantic_personalization_run as run
from rrbench.semantic_personalization import HistoryAttentionHead
from rrbench.io import sha256, write_json


def qualification_binding(seed=17):
    return {
        "items_sha256": "i", "protocol_sha256": "p", "panel_sha256": "q",
        "raw_queries_sha256": "r", "popularity_sha256": "pop", "history_metadata_sha256": "h",
        "coverage_receipt_sha256": "c", "implementation_sha256": "core",
        "runtime_sha256": "run", "metric_sha256": "metric", "statistics_sha256": "stats",
        "phase": "test", "checkpoint_sha256": f"checkpoint-{seed}",
        "admission_sha256": None if seed == 17 else "decision-hash",
    }


def admission_fixture(go=True):
    return {
        "complete": True, "phase": "validation", "model_seed": 17, "model_epochs": 3,
        "binding": {"items_sha256": "i", "protocol_sha256": "p", "checkpoint_sha256": "checkpoint-17"},
        "pilot_gate": {"go": go}, "go": go,
        "decision_status": "admitted" if go else "not_admitted",
        "decision_time_utc": "2026-10-11T10:00:00+00:00",
        "pilot_cutoff_utc": "2026-10-11T15:03:01.929Z",
    }


def test_raw_queries_require_exact_panel_identity_and_checked_bytes(tmp_path):
    query = tmp_path / "vectors.npy"
    np.save(query, np.eye(2, dtype=np.float16))
    panel = {"query_ids": ["q1", "q2"]}
    panel_file = tmp_path / "panel.json"
    write_json(panel_file, panel)
    receipt = tmp_path / "completed.json"
    write_json(receipt, {
        "complete": True, "vectors_sha256": sha256(query), "panel_sha256": sha256(panel_file),
        "query_ids": panel["query_ids"], "model": "model", "revision": "rev",
    })
    values = run.checked_raw_queries(query, receipt, panel_file, {
        "representation": {"model": "model", "revision": "rev"}, "stored_dimension": 2,
    })
    assert values.shape == (2, 2)
    write_json(panel_file, {"query_ids": ["q2", "q1"]})
    with pytest.raises(ValueError, match="query"):
        run.checked_raw_queries(query, receipt, panel_file, {
            "representation": {"model": "model", "revision": "rev"}, "stored_dimension": 2,
        })


def test_training_epoch_updates_the_head_and_counts_all_fitting_queries():
    torch.manual_seed(17)
    model = HistoryAttentionHead(4, maximum_history=2, hidden=3, bottleneck=3)
    items = torch.eye(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    histories = np.array([[0, 1], [2, 3], [1, -1]])
    candidates = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3]])
    positive = np.array([[False, True, False], [False, True, False], [False, True, False]])
    before = [value.detach().clone() for value in model.parameters()]
    loss, count = run.train_one_epoch(
        model, items, histories, candidates, positive, optimizer, batch_size=2, seed=17, epoch=0,
    )
    assert count == 3
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert items.grad is None


def test_sparse_pilot_stops_expansion_without_claiming_population_zero():
    users = np.array([10, 10, 20])
    quality = {
        "queries": 3, "recall@1000": 0.0,
        "per_query": [{"query_id": str(row), "recall@1000": 0.0} for row in range(3)],
    }
    result = run.pilot_gate(quality, quality, quality, users, draws=10)
    assert result["go"] is False
    assert result["evaluated_users"] == 2
    assert result["paired_rows"] == 3
    assert "not evidence" in result["interpretation"]


def test_smoke_certificate_cannot_authorize_a_different_frozen_bank():
    receipt = {
        "complete": True, "study_role": "correctness-only", "rows": 100,
        "item_vectors_sha256": "a", "fitting_panel_sha256": "b",
    }
    run.validate_smoke(receipt, "a", "b")
    with pytest.raises(ValueError, match="smoke"):
        run.validate_smoke(receipt, "c", "b")


def test_completed_model_reuse_rejects_inconsistent_fitting_scope(tmp_path):
    torch.save({"model": {"weight": torch.tensor([1.0])}, "binding": {"seed": 17}}, tmp_path / "model.pt")
    write_json(tmp_path / "completed.json", {
        "complete": True, "model_sha256": sha256(tmp_path / "model.pt"), "epochs": 3,
        "fitting_queries": 100, "binding": {"seed": 17},
    })
    run.checked_model(tmp_path, {"seed": 17}, queries=100)
    with pytest.raises(ValueError, match="scope"):
        run.checked_model(tmp_path, {"seed": 17}, queries=101)


def test_evaluation_reuse_is_bound_to_query_population_and_all_payload_bytes(tmp_path):
    np.save(tmp_path / "learned-ids.npy", np.array([[1, 2]], dtype=np.int64))
    binding = {"panel_sha256": "p", "items_sha256": "i"}
    write_json(tmp_path / "completed.json", {
        "complete": True, "binding": binding,
        "files": {"learned-ids.npy": sha256(tmp_path / "learned-ids.npy")},
    })
    run.checked_evaluation(tmp_path, binding)
    with pytest.raises(ValueError, match="binding"):
        run.checked_evaluation(tmp_path, {**binding, "panel_sha256": "different"})
    np.save(tmp_path / "learned-ids.npy", np.array([[3, 4]], dtype=np.int64))
    with pytest.raises(ValueError, match="payload"):
        run.checked_evaluation(tmp_path, binding)


def test_seed23_requires_passing_validation_of_the_fixed_seed17_checkpoint():
    valid = admission_fixture()
    run.validate_admission(valid, "i", "p")
    for field, value in (("phase", "test"), ("model_seed", 23), ("model_epochs", 2)):
        with pytest.raises(ValueError, match="admission"):
            run.validate_admission({**valid, field: value}, "i", "p")
    with pytest.raises(ValueError, match="admission"):
        run.validate_admission({**valid, "pilot_gate": {"go": False}}, "i", "p")


def test_test_phase_is_never_an_actionable_pilot_gate():
    assert run.phase_admission("test", {})["status"] == "not_applicable"
    assert run.phase_admission("test", {})["go"] is None
    with pytest.raises(ValueError, match="phase"):
        run.validate_evaluation_phase({"name": "kuairand-test"}, "validation")


def test_unadmitted_second_seed_is_not_a_missing_qualification_measurement():
    receipt = {
        "complete": True, "phase": "test", "model_seed": 17, "model_epochs": 3,
        "query_users": [1],
        "quality": {
            "learned": {"per_query": [{"query_id": "a", "recall@1000": 0.0}]},
            "training-popularity": {"per_query": [{"query_id": "a", "recall@1000": 0.0}]},
        },
        "incremental_user_encoding_timing": {"16": {"latency_mean_ms": 1.0}},
        "device_memory_samples": [{"memory_used_bytes": 1}],
        "binding": qualification_binding(),
    }
    result = run.qualification_summary(receipt, None, admission_fixture(False))
    assert result["two_seed_status"] == "not_assessed_by_design"
    assert result["research_qualification"] is False
    assert result["runtime_budget"]["batch16_within_5ms"]


def test_two_seed_qualification_uses_matching_user_means_and_fixed_seed_metric_average():
    def receipt(seed, metric):
        return {
            "complete": True, "phase": "test", "model_seed": seed, "model_epochs": 3,
            "query_users": [1, 1, 2],
            "quality": {
                "learned": {"per_query": [
                    {"query_id": "a", "recall@1000": metric},
                    {"query_id": "b", "recall@1000": 0.0},
                    {"query_id": "c", "recall@1000": 0.0},
                ]},
                "training-popularity": {"per_query": [
                    {"query_id": "a", "recall@1000": 0.0},
                    {"query_id": "b", "recall@1000": 0.0},
                    {"query_id": "c", "recall@1000": 0.0},
                ]},
            },
            "incremental_user_encoding_timing": {"16": {"latency_mean_ms": 1.0}},
            "device_memory_samples": [{"memory_used_bytes": 1}],
            "binding": qualification_binding(seed),
        }
    result = run.qualification_summary(
        receipt(17, 1.0), receipt(23, 0.5), admission_fixture(), admission_hash="decision-hash", draws=10,
    )
    assert result["effect"]["mean_user_effect"] == 0.1875
    assert result["effect"]["mean_event_or_query_effect"] == 0.25
    assert result["two_seed_status"] == "assessed"


def test_evaluation_reuse_rejects_changed_implementation_even_with_unchanged_inputs(tmp_path):
    core, runtime, metrics, statistics = [tmp_path / name for name in ("core.py", "run.py", "metrics.py", "stats.py")]
    for path in (core, runtime, metrics, statistics):
        path.write_text("version_one")
    np.save(tmp_path / "learned-ids.npy", np.array([[1]], dtype=np.int64))
    original = {
        "panel_sha256": "p", **run.evaluation_implementation(core, runtime, metrics, statistics),
    }
    write_json(tmp_path / "completed.json", {
        "complete": True, "binding": original,
        "files": {"learned-ids.npy": sha256(tmp_path / "learned-ids.npy")},
    })
    run.checked_evaluation(tmp_path, original)
    metrics.write_text("corrected_metric")
    changed = {"panel_sha256": "p", **run.evaluation_implementation(core, runtime, metrics, statistics)}
    with pytest.raises(ValueError, match="binding"):
        run.checked_evaluation(tmp_path, changed)


def test_late_pilot_decision_stops_expansion_as_time_budget_incomplete():
    validation = {
        "complete": True, "phase": "validation", "model_seed": 17, "model_epochs": 3,
        "binding": {"items_sha256": "i", "protocol_sha256": "p", "checkpoint_sha256": "c"},
        "pilot_gate": {"go": True},
    }
    result = run.make_admission_decision(
        validation, "2026-10-11T15:03:02Z", "2026-10-11T15:03:01.929Z",
    )
    assert result["go"] is False
    assert result["decision_status"] == "time_budget_incomplete"
    assert "not" in result["interpretation"]


def test_qualification_inclusion_follows_admission_not_stale_file_existence():
    assert run.select_second_seed({"complete": True, "go": False}, {"complete": True}) is None
    with pytest.raises(ValueError, match="admitted"):
        run.select_second_seed({"complete": True, "go": True}, None)


def test_qualification_cannot_combine_different_protocol_or_item_bindings():
    first = {"binding": {"protocol_sha256": "p", "items_sha256": "i", "panel_sha256": "q"}}
    second = {"binding": {"protocol_sha256": "other", "items_sha256": "i", "panel_sha256": "q"}}
    with pytest.raises(ValueError, match="binding"):
        run.validate_shared_evaluation_bindings(first, second)
