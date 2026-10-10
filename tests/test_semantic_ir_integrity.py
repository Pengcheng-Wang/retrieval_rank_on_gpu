import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_ir import validate_prepared_panels


def test_frozen_query_judgment_panel_reuse_checks_individual_panel_bytes(tmp_path):
    write_json(tmp_path / "dev-small.json", {"queries": ["first"], "qrels": {"q": {1: 1}}})
    write_json(tmp_path / "completed.json", {
        "complete": True, "panel_sha256": {"dev-small": sha256(tmp_path / "dev-small.json")},
    })
    validate_prepared_panels(tmp_path)
    write_json(tmp_path / "dev-small.json", {"queries": ["first"], "qrels": {"q": {2: 1}}})
    with pytest.raises(ValueError, match="panel"):
        validate_prepared_panels(tmp_path)
