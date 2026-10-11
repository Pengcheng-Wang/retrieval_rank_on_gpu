import numpy as np
import pytest
import torch
import json
import sqlite3

from rrbench import semantic_personalization as study
from rrbench.io import sha256, write_json


def test_block_sampling_is_label_independent_balanced_and_deterministic():
    blocks = [(user, tick, tick - 1) for user in (3, 5, 8) for tick in range(10, 30)]
    selected = study.balanced_blocks(blocks, maximum=9, seed=17)
    assert selected == study.balanced_blocks(list(reversed(blocks)), maximum=9, seed=17)
    assert len(selected) == 9
    assert [sum(user == value for user, _ in selected) for value in (3, 5, 8)] == [3, 3, 3]


def test_padding_rejects_empty_and_out_of_catalog_histories():
    values = study.padded_histories([[2], [1, 3]], items=5, maximum=3)
    assert values.tolist() == [[2, -1, -1], [1, 3, -1]]
    with pytest.raises(ValueError, match="history"):
        study.padded_histories([[]], items=5, maximum=3)
    with pytest.raises(ValueError, match="history"):
        study.padded_histories([[5]], items=5, maximum=3)


def test_wrong_user_control_deranges_users_not_individual_query_rows():
    users = np.array([10, 10, 20, 30, 30])
    histories = [[0, 1], [0, 1], [2, 3], [4], [4]]
    changed = study.wrong_user_histories(histories, users, seed=17)
    assert changed[0] == changed[1]
    assert changed[3] == changed[4]
    assert all(a != b for a, b in zip(histories, changed))
    with pytest.raises(ValueError, match="frozen"):
        study.wrong_user_histories([[0], [1], [2]], np.array([10, 10, 20]), seed=17)


def test_order_control_preserves_the_history_multiset():
    histories = [[0, 1, 1, 2], [4], [0, 1, 1, 2]]
    changed = study.permute_history_order(histories, seed=17)
    assert changed[0] == changed[2]
    assert changed[0] != histories[0]
    assert all(sorted(a) == sorted(b) for a, b in zip(histories, changed))


def test_head_starts_as_same_history_mean_and_ignores_padding():
    torch.manual_seed(17)
    model = study.HistoryAttentionHead(dimension=4, maximum_history=3, hidden=3, bottleneck=3)
    features = torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0], [99, 99, 99, 99]]])
    mask = torch.tensor([[True, True, False]])
    expected = torch.tensor([[2 ** -0.5, 2 ** -0.5, 0, 0]])
    torch.testing.assert_close(model(features, mask), expected)
    features[:, 2] = -99
    torch.testing.assert_close(model(features, mask), expected)
    assert sum(value.numel() for value in study.HistoryAttentionHead(1024).parameters()) < 2_000_000


def test_multi_positive_loss_uses_all_positives_and_excludes_padding():
    logits = torch.tensor([[2., 1., 0., 1000.]], requires_grad=True)
    positive = torch.tensor([[True, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    expected = torch.logsumexp(logits[0, :3], 0) - torch.logsumexp(logits[0, :2], 0)
    loss = study.sampled_softmax_loss(logits, positive, valid, temperature=1.0)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 3] == 0
    with pytest.raises(ValueError, match="positive"):
        study.sampled_softmax_loss(logits.detach(), torch.zeros_like(positive), valid)


def test_training_candidate_sampling_keeps_positive_and_negative_sources_distinct():
    panel = {
        "query_ids": ["q"],
        "qrels": {"q": {"2": 1, "3": 1}},
        "logged_nonpositive_item_rows": {"q": [2, 5, 6]},
    }
    ids, positives, valid, audit = study.training_candidates(
        panel, items=10, seed=17, logged_limit=2, catalog_limit=3,
    )
    assert set(ids[positives]) == {2, 3}
    assert len(set(ids[valid])) == valid.sum()
    assert not set(ids[valid & ~positives]) & {2, 3}
    assert audit["logged_negative_pairs"] == 2
    assert audit["sampled_unjudged_negative_pairs"] == 3


def test_user_aggregation_preserves_the_full_query_denominator():
    quality = {
        "queries": 3, "recall@1000": 1 / 3,
        "per_query": [
            {"query_id": "a", "recall@1000": 1.0},
            {"query_id": "b", "recall@1000": 0.0},
            {"query_id": "c", "recall@1000": 0.0},
        ],
    }
    result = study.equal_user_metrics(quality, np.array([10, 10, 20]))
    assert result["queries"] == 3
    assert result["users"] == 2
    assert result["equal_user"]["recall@1000"] == 0.25
    assert result["query_weighted"]["recall@1000"] == 1 / 3
    assert result["hit_users@1000"] == 1


def test_recurrence_popularity_backfill_never_injects_targets_or_duplicates():
    predictions = study.recurrence_popularity([[2, 3, 2]], np.array([0, 1, 2, 3, 4]), k=4)
    assert predictions.tolist() == [[2, 3, 0, 1]]


def test_sampled_softmax_updates_only_the_user_head_not_the_item_bank():
    torch.manual_seed(17)
    items = torch.eye(4)
    model = study.HistoryAttentionHead(4, maximum_history=2, hidden=3, bottleneck=3)
    query = model(items[torch.tensor([[0, 1], [2, 3]])], torch.ones(2, 2, dtype=torch.bool))
    before = [value.detach().clone() for value in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    logits = query @ items.T
    loss = study.sampled_softmax_loss(
        logits,
        torch.tensor([[False, True, False, False], [False, False, True, False]]),
        torch.ones(2, 4, dtype=torch.bool),
    )
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert items.grad is None


def test_preparation_uses_only_training_blocks_and_strictly_prior_exposure_histories(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    original = np.array([100, 200, 300, 400], dtype=np.int64)
    np.save(catalog / "original_ids.npy", original)
    write_json(catalog / "corpus.json", {
        "rows": 4, "ids_sha256": sha256(catalog / "original_ids.npy"),
        "catalog_sha256": "fixture-catalog", "clock_policy_version": 2, "million_item_gate_pass": True,
    })
    with sqlite3.connect(catalog / "private-catalog.sqlite") as connection:
        connection.executescript(
            "CREATE TABLE events(user INTEGER,item INTEGER,timestamp INTEGER,label INTEGER);"
            "CREATE TABLE blocks(user INTEGER,timestamp INTEGER,lower INTEGER,upper INTEGER,phase INTEGER);"
            "CREATE TABLE semantic(item INTEGER);"
        )
        connection.executemany("INSERT INTO semantic VALUES(?)", [(int(item),) for item in original])
        connection.executemany("INSERT INTO blocks VALUES(?,?,?,?,?)", [
            (10, 20, 19, 21, 1), (10, 40, 39, 41, 1), (10, 60, 59, 61, 1), (10, 80, 79, 81, 2),
        ])
        connection.executemany("INSERT INTO events VALUES(?,?,?,?)", [
            (10, 100, 20, 0), (10, 200, 40, 1), (10, 300, 60, 1), (10, 400, 80, 1),
        ])
    output = tmp_path / "prepared"
    receipt = study.prepare_training_panel(catalog, output, maximum=20, seed=17, smoke=True)
    assert receipt["complete"]
    panel = json.loads((output / "training-panel.json").read_text())
    assert len(panel["query_ids"]) == 2
    assert "1:10:80" not in panel["query_ids"]
    first = panel["query_ids"].index("1:10:40")
    assert panel["history_item_rows"][first] == [0]
    assert panel["qrels"]["1:10:40"] == {"1": 1}
    assert all(
        all(upper <= lower for upper in bounds)
        for bounds, lower in zip(panel["history_upper_bounds"], panel["target_clock_lower"])
    )
    assert receipt["selection_audit"]["selected_no_history"] == 1


def test_prepared_training_payload_drift_is_not_recognized_as_valid_completion(tmp_path):
    write_json(tmp_path / "training-panel.json", {"name": "fixture"})
    write_json(tmp_path / "completed.json", {
        "complete": True, "panel_sha256": sha256(tmp_path / "training-panel.json"),
    })
    study.checked_training_panel(tmp_path)
    (tmp_path / "training-panel.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        study.checked_training_panel(tmp_path)


def test_training_history_clock_envelope_cannot_cross_the_target():
    panel = {
        "query_ids": ["q"], "users": [1], "history_item_rows": [[0, 1]],
        "history_upper_bounds": [[19, 25]], "target_clock_lower": [20],
        "qrels": {"q": {"2": 1}}, "catalog_items": 3,
    }
    with pytest.raises(ValueError, match="clock"):
        study.validate_history_panel(panel, items=3, require_clock_bounds=True)


def test_recurrent_and_novel_target_metrics_are_separate_from_primary_denominators():
    panel = {
        "query_ids": ["a", "b"], "users": [10, 20],
        "history_item_rows": [[0], [1]], "qrels": {"a": {"0": 1, "2": 1}, "b": {"3": 1}},
        "binary_threshold": 1,
    }
    rankings = np.array([[0, 4], [3, 4]], dtype=np.int64)
    result = study.target_strata(panel, rankings, np.array([4, 3, 1, 0, 1]))
    assert result["recurrent"]["queries"] == 1
    assert result["recurrent"]["equal_user"]["recall@1000"] == 1
    assert result["novel"]["queries"] == 2
    assert result["novel"]["equal_user"]["recall@1000"] == 0.5
    assert result["training_positive_support_zero"]["queries"] == 1
    assert result["training_positive_support_zero"]["equal_user"]["recall@1000"] == 1
    assert result["training_positive_support_one"]["queries"] == 1
    assert result["training_positive_support_one"]["equal_user"]["recall@1000"] == 0
    assert result["training_positive_support_two_to_nine"]["queries"] == 1
    assert result["training_positive_support_ten_plus"]["status"] == "no_targets_in_stratum"


def test_history_metadata_requires_existing_history_vectors_and_clock_provenance(tmp_path):
    import sqlite3

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    original = np.array([100, 200, 300], dtype=np.int64)
    np.save(catalog / "original_ids.npy", original)
    write_json(catalog / "corpus.json", {
        "rows": 3, "ids_sha256": sha256(catalog / "original_ids.npy"), "catalog_sha256": "fixture",
    })
    with sqlite3.connect(catalog / "private-catalog.sqlite") as connection:
        connection.executescript(
            "CREATE TABLE events(user INTEGER,item INTEGER,timestamp INTEGER,label INTEGER);"
            "CREATE TABLE blocks(user INTEGER,timestamp INTEGER,lower INTEGER,upper INTEGER,phase INTEGER);"
            "CREATE TABLE semantic(item INTEGER);"
        )
        connection.executemany("INSERT INTO semantic VALUES(?)", [(100,), (200,), (300,)])
        connection.executemany("INSERT INTO blocks VALUES(?,?,?,?,?)", [(1, 10, 9, 11, 1), (1, 20, 19, 21, 1)])
        connection.executemany("INSERT INTO events VALUES(?,?,?,?)", [(1, 100, 10, 0), (1, 200, 20, 1)])
    panel = {
        "query_ids": ["x"], "users": [1], "history_item_rows": [[0, 1]],
        "target_clock_lower": [30], "catalog_sha256": "fixture",
        "catalog_ids_sha256": sha256(catalog / "original_ids.npy"),
    }
    source = tmp_path / "validation.json"
    write_json(source, panel)
    result = study.history_metadata(catalog, source)
    assert result["history_upper_bounds"] == [[11, 21]]
    assert result["history_age_ms"] == [9]
    panel["history_item_rows"] = [[1]]
    write_json(source, panel)
    with pytest.raises(ValueError, match="history"):
        study.history_metadata(catalog, source)
