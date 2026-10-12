import json

import pytest

from rrbench.reporting import aggregate, publish, reject_private_fields


def test_repeat_aggregation_preserves_recall_and_units():
    rows = [
        {"backend": "a", "status": "ok", "latency_mean_ms": 1.0, "recall": 0.8},
        {"backend": "a", "status": "ok", "latency_mean_ms": 3.0, "recall": 0.8},
        {"backend": "b", "status": "unsupported"},
    ]
    result = aggregate(rows, ("backend",))
    assert len(result) == 1
    assert result[0]["measurements"] == 2
    assert result[0]["latency_mean_ms"] == 2
    assert result[0]["recall"] == 0.8


def test_publication_rejects_incomplete_journal(tmp_path):
    source = tmp_path / "input" / "experiment"
    source.mkdir(parents=True)
    (source / "ann.jsonl").write_text(json.dumps({"status": "ok"}) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        publish(tmp_path / "input", tmp_path / "output", tmp_path / "tables.md")


def test_publication_rejects_nested_operational_metadata():
    with pytest.raises(ValueError, match="access_token"):
        reject_private_fields({"environment": {"packages": {"access_token": "not-public"}}})
