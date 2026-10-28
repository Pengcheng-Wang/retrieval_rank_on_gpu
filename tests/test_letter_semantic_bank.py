import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def bank():
    assert (SCRIPTS / "bank_semantics.py").exists(), "Asynchronous single-writer semantic export is missing"
    return importlib.import_module("bank_semantics")


def test_only_sealed_row_aligned_chunks_are_ready(tmp_path):
    module = bank()
    root = tmp_path / "Instruments"
    root.mkdir()
    (root / "run-config.json").write_text(json.dumps({
        "dataset": "Instruments", "kind": "semantic-encoding-compatible-backbone",
        "track_id": "letter-openllama7b-v2-method-replication", "smoke_items": None, "chunk_size": 2,
    }))
    (root / "token-audit.json").write_text(json.dumps({"items": 4, "status": "pass"}))
    chunk = root / "chunks" / "items-00000-00002"
    chunk.mkdir(parents=True)
    np.save(chunk / "embeddings.npy", np.ones((2, 4096), dtype=np.float32), allow_pickle=False)
    (chunk / "completed.json").write_text(json.dumps({
        "status": "complete", "first_item": 0, "stop_item": 2, "shape": [2, 4096],
        "run_config_sha256": module.file_sha256(root / "run-config.json"),
        "embeddings_sha256": module.file_sha256(chunk / "embeddings.npy"),
    }))
    partial = root / "chunks" / "items-00002-00004"
    partial.mkdir()
    (partial / "embeddings.npy").write_bytes(b"partial")
    ready, complete = module.ready_artifacts(root)
    assert complete is False
    assert ready == [(Path("Instruments/chunks/items-00000-00002"), chunk)]
    (chunk / "embeddings.npy").write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.ready_artifacts(root)


def test_unstarted_semantic_output_is_pending(tmp_path):
    assert bank().ready_artifacts(tmp_path / "not-started") == ([], False)
