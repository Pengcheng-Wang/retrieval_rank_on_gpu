from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_audit_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "hstu" / "audit.py"
    spec = importlib.util.spec_from_file_location("hstu_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_and_select_smoke(tmp_path: Path) -> None:
    audit = _load_audit_module()
    source = tmp_path / "ratings.csv"
    rows = [
        {
            "user_id": str(user_id),
            "sequence_item_ids": str([user_id, user_id + 1, user_id + 2]),
            "sequence_ratings": str([3, 4, 5]),
            "sequence_timestamps": str([1, 2, 3]),
        }
        for user_id in range(1, 6)
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    catalog = tmp_path / "movies.csv"
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["movie_id"])
        writer.writeheader()
        writer.writerows({"movie_id": item_id} for item_id in range(1, 9))

    summary = audit.summarize_csv(source, catalog)
    assert summary["users"] == 5
    assert summary["events"] == 15
    assert summary["history_length"]["mean"] == 2
    assert summary["nonmonotonic_timestamp_users"] == 0
    assert summary["evaluation_catalog"]["rows"] == 8
    assert summary["evaluation_catalog"]["unrated_catalog_rows"] == 1
    assert summary["split_coverage"]["warm_heldout_events"] == 4
    assert summary["split_coverage"]["cold_heldout_events"] == 1

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    receipt = audit.select_smoke_users(source, first, users=3, seed=42)
    audit.select_smoke_users(source, second, users=3, seed=42)
    assert first.read_bytes() == second.read_bytes()
    assert receipt["users"] == 3
    assert len(receipt["selected_user_ids_sha256"]) == 64


def test_parse_log_uses_final_epoch(tmp_path: Path) -> None:
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "rank 0:  batch-stat (train): step 0 (epoch 0 in 1.00s): 4.200000",
                "rank 0: eval @ epoch 0 in 2.00s: "
                "NDCG@10 0.1000, NDCG@50 0.2000, "
                "HR@10 0.3000, HR@50 0.4000, MRR 0.1500",
                "rank 0: eval @ epoch 1 in 2.00s: "
                "NDCG@10 0.0900, NDCG@50 0.2100, "
                "HR@10 0.3100, HR@50 0.4100, MRR 0.1400",
            ]
        )
        + "\n"
    )
    parsed = audit.parse_log(log)
    assert parsed["completed_epochs"] == 2
    assert parsed["final"]["epoch"] == 1
    assert parsed["final"]["ndcg@10"] == 0.09
    assert parsed["best_observed_ndcg10_diagnostic"]["epoch"] == 0
