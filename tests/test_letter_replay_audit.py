import copy
import importlib
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def audit():
    assert (SCRIPTS / "audit_replay.py").exists(), "Independent replay auditing is missing"
    return importlib.import_module("audit_replay")


def fixture():
    groups = {f"sid{i}": [i + 1] for i in range(1, 20)}
    groups["sid0"] = [0, 1]
    codes = {item: code for code, items in groups.items() for item in items}
    row = {
        "target_item": 1, "target_code": "sid0", "target_collision_size": 2,
        "codes": [f"sid{i}" for i in range(20)], "scores": list(range(20, 0, -1)),
        "item_predictions": list(range(10)),
        "metrics": {
            "code": {f"{m}@{k}": 1.0 for m in ("recall", "ndcg") for k in (1, 5, 10)},
            "item": {f"{m}@{k}": (0.0 if k == 1 else 1.0 if m == "recall" else 1 / math.log2(3))
                     for m in ("recall", "ndcg") for k in (1, 5, 10)},
        },
    }
    return row, codes, groups


def test_independent_audit_distinguishes_code_and_item_ranks():
    row, codes, groups = fixture()
    result = audit().audit_prediction(row, 1, codes, groups)
    assert result["code"]["ndcg@10"] == 1
    assert result["item"]["ndcg@10"] == pytest.approx(1 / math.log2(3))
    with pytest.raises(ValueError):
        audit().audit_prediction(row, 0, codes, groups)


def test_audit_rejects_duplicate_beams_nonfinite_scores_and_wrong_metrics():
    row, codes, groups = fixture()
    for field, value in (
        ("codes", ["sid0"] * 20),
        ("scores", [float("nan")] * 20),
        ("scores", list(range(20))),
        ("item_predictions", list(reversed(range(10)))),
    ):
        changed = copy.deepcopy(row)
        changed[field] = value
        with pytest.raises(ValueError):
            audit().audit_prediction(changed, 1, codes, groups)
    changed = copy.deepcopy(row)
    changed["metrics"]["item"]["ndcg@10"] = 1.0
    with pytest.raises(ValueError):
        audit().audit_prediction(changed, 1, codes, groups)
