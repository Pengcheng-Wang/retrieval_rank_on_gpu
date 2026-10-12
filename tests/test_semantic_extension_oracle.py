import numpy as np
import pytest

from rrbench import semantic_extension_oracle as oracle
from rrbench.io import sha256, write_json


def test_oracle_binding_requires_exact_catalog_queries_and_fp32_products(tmp_path):
    items = tmp_path / "items.npy"
    queries = tmp_path / "queries.npy"
    panel = tmp_path / "panel.json"
    np.save(items, np.eye(2, dtype=np.float16))
    np.save(queries, np.eye(2, dtype=np.float16))
    write_json(panel, {"query_ids": ["a", "b"]})
    binding = oracle.source_binding(items, queries, panel)
    assert binding["items_sha256"] == sha256(items)
    assert binding["queries_sha256"] == sha256(queries)
    assert binding["k"] == 1000
    assert binding["product"] == "FP32"
    assert binding["tf32"] is False
    assert "panel_sha256" not in binding


def test_oracle_completion_reuse_checks_outer_and_inner_receipt_hashes(tmp_path):
    write_json(tmp_path / "completed.json", {
        "complete": True, "panel_receipt_sha256": "wrong",
    })
    panel = tmp_path / "panel"
    panel.mkdir()
    write_json(panel / "completed.json", {"complete": True})
    with pytest.raises(ValueError, match="receipt"):
        oracle.validate_outer_receipt(tmp_path)
