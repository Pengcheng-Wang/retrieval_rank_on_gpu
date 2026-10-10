import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_models import configuration_digest
from rrbench.semantic_queries import validate_query_panel


def test_query_completion_binds_exact_query_text_order_and_native_mrl_dimensions(tmp_path):
    spec = {"model": "example", "revision": "fixed", "dimensions": [4, 2]}
    panel = tmp_path / "panel.json"
    write_json(panel, {"query_ids": ["q1", "q2"], "queries": ["first", "second"]})
    np.save(tmp_path / "vectors.npy", np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float16))
    np.save(tmp_path / "vectors_mrl2.npy", np.eye(2, dtype=np.float16))
    write_json(tmp_path / "completed.json", {
        "complete": True, "rows": 2, "query_ids": ["q1", "q2"],
        "panel_sha256": sha256(panel), "specification_sha256": configuration_digest(spec),
        "native_dimension": 4, "vectors_sha256": sha256(tmp_path / "vectors.npy"),
        "mrl_representations": {"2": {"file": "vectors_mrl2.npy", "dimension": 2, "sha256": sha256(tmp_path / "vectors_mrl2.npy")}},
    })
    assert validate_query_panel(tmp_path, panel, spec)["rows"] == 2
    write_json(panel, {"query_ids": ["q2", "q1"], "queries": ["second", "first"]})
    with pytest.raises(ValueError, match="panel"):
        validate_query_panel(tmp_path, panel, spec)
