import copy
import gzip
import json

import pytest

from rrbench.replication_reporting import BACKENDS, PROBES, protocol_matches, publish_replication, validate_rows


def complete_rows(seed=23):
    return [
        {
            "backend": backend, "probes": probe, "repeat": repeat,
            "seed": seed, "items": 1000000, "dimension": 128, "distribution": "clustered",
            "k": 2048, "batch_size": 16, "status": "ok",
            "samples_ms": [1.0] * 50, "per_query_recall": [0.8] * 128,
            "latency_mean_ms": 1.0, "recall": 0.8,
        }
        for backend in BACKENDS for probe in PROBES for repeat in range(3)
    ]


def test_complete_replication_rows_are_validated():
    validate_rows(complete_rows(), 23)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed", "bad_latency", "bad_recall"])
def test_invalid_replication_journals_are_rejected(change):
    rows = copy.deepcopy(complete_rows())
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "wrong_seed":
        rows[0]["seed"] = 42
    elif change == "bad_latency":
        rows[0]["samples_ms"][0] = -1.0
    else:
        rows[0]["recall"] = 0.9
    with pytest.raises(ValueError):
        validate_rows(rows, 23)


def test_replication_rejects_a_different_bound_protocol():
    with pytest.raises(ValueError, match="frozen|planned"):
        protocol_matches({"seed": 23}, 23)


def test_replication_publication_separates_seed_and_repeat_counts(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    baseline = [{**row, "experiment": "ann_1m_clustered_s17"} for row in complete_rows(17)]
    (reference / "ann-raw.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(row) + "\n" for row in baseline).encode(), mtime=0
    ))
    environment = {"tf32": False, "cpu_threads": 32}
    (reference / "summary.json").write_text(json.dumps({
        "environments": {"ann_1m_clustered_s17": environment}
    }))
    runs = {}
    for seed in (23, 42):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        (directory / "completed.json").write_text(json.dumps({
            "complete": True, "experiment": "synthetic ANN"
        }))
        (directory / "run-config.json").write_text(json.dumps({
            "experiment": "ann", "items": 1000000, "dimension": 128, "queries": 128,
            "seed": seed, "distribution": "clustered", "clusters": 1024, "ks": [2048],
            "probes": list(PROBES), "batches": [16], "warmup": 20, "iterations": 50,
            "repeats": 3, "threads": 32, "backends": list(BACKENDS),
        }))
        (directory / "manifest.json").write_text(json.dumps({"environment": environment}))
        (directory / "ann.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in complete_rows(seed))
        )
        runs[seed] = directory
    receipt = tmp_path / "resource.json"
    receipt.write_text(json.dumps({
        "restoration_verified": True, "restored_original_profile": True,
        "restored_gpu_count": 8, "other_nodes_modified": False,
    }))
    result = publish_replication(reference, runs, tmp_path / "public", tmp_path / "report.md", receipt)
    assert len(result["per_seed"]) == 72
    assert len(result["across_seed_anchor"]) == 6
    assert all(row["measurements"] == 3 for row in result["across_seed_anchor"])
    assert all(row["measurements"] == 3 for row in result["per_seed"])
    assert len(result["recall_constrained_frontier"]) == 36
    with pytest.raises(ValueError, match="new output"):
        publish_replication(reference, runs, tmp_path / "public", tmp_path / "report.md", receipt)
    assert "do not establish a consistent loss" in (tmp_path / "report.md").read_text(encoding="utf-8")
