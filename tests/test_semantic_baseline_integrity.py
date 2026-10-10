import sqlite3
import json

import numpy as np
import pytest
import torch

from rrbench.io import sha256, write_json
import rrbench.semantic_recommendation_baselines as mf_baseline
from rrbench.semantic_bm25 import validate_index_prefix, unicode61_terms
from rrbench.semantic_recommendation_baselines import validate_catalog_mapping, validate_final_mf, validate_mf_resume


def test_reversed_catalog_row_mapping_is_not_a_valid_mf_training_identity(tmp_path):
    np.save(tmp_path / "original_ids.npy", np.array([100, 200], dtype=np.int64))
    manifest = {"rows": 2, "ids_sha256": sha256(tmp_path / "original_ids.npy")}
    validate_catalog_mapping(tmp_path, manifest)
    np.save(tmp_path / "original_ids.npy", np.array([200, 100], dtype=np.int64))
    with pytest.raises(ValueError, match="mapping"):
        validate_catalog_mapping(tmp_path, manifest)


def test_mf_resume_cannot_claim_three_epochs_with_an_out_of_range_counter_and_one_loss():
    with pytest.raises(ValueError, match="epoch"):
        validate_mf_resume({"epochs": 4, "losses": [0.6], "elapsed_seconds": 10.0}, expected_epochs=3)


def test_bm25_stale_progress_cannot_reinsert_existing_rows_and_change_statistics():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE VIRTUAL TABLE passages USING fts5(body,content='')")
        connection.executemany("INSERT INTO passages(rowid,body) VALUES (?,?)", [(1, "one"), (2, "two"), (3, "three")])
        with pytest.raises(ValueError, match="prefix"):
            validate_index_prefix(connection, 2)


def test_bm25_zero_rowid_is_not_silently_returned_as_invalid_padding():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE VIRTUAL TABLE passages USING fts5(body,content='')")
        connection.executemany("INSERT INTO passages(rowid,body) VALUES (?,?)", [(0, "one"), (2, "two"), (3, "three")])
        with pytest.raises(ValueError, match="prefix"):
            validate_index_prefix(connection, 3)


def test_query_terms_use_actual_unicode61_normalization_not_python_casefold():
    assert unicode61_terms(["Straße", "e\u0301cole"]) == [["straße"], ["ecole"]]


def test_one_epoch_intermediate_mf_checkpoint_is_not_a_complete_control(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder fixture")
    write_json(tmp_path / "completed.json", {
        "complete": True, "model_sha256": sha256(checkpoint), "completed_epochs": 1,
        "requested_epochs": 3, "losses": [0.6],
    })
    with pytest.raises(ValueError, match="epochs"):
        validate_final_mf(checkpoint)


def test_mf_checkpoint_publication_interruption_keeps_last_committed_epoch_resumable(tmp_path, monkeypatch):
    first = {"epochs": 1, "model": {"weight": torch.ones(2)}, "losses": [0.6]}
    mf_baseline.publish_mf_epoch(tmp_path, first)
    original = json.loads((tmp_path / "training-state.json").read_text())
    publish = mf_baseline.write_json

    def interrupted_pointer(path, value):
        if path.name == "training-state.json":
            raise RuntimeError("simulated preempt before publishing new pointer")
        publish(path, value)

    monkeypatch.setattr(mf_baseline, "write_json", interrupted_pointer)
    with pytest.raises(RuntimeError, match="preempt"):
        mf_baseline.publish_mf_epoch(
            tmp_path, {"epochs": 2, "model": {"weight": torch.zeros(2)}, "losses": [0.6, 0.5]},
        )
    current = json.loads((tmp_path / "training-state.json").read_text())
    assert current == original
    checkpoint = tmp_path / current["checkpoint"]
    assert sha256(checkpoint) == current["checkpoint_sha256"]
    assert torch.load(checkpoint, weights_only=True)["epochs"] == 1
