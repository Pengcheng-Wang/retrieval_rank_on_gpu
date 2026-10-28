import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def compare():
    assert (SCRIPTS / "compare.py").exists(), "Paired result comparison is missing"
    import compare
    return compare


def test_paired_bootstrap_preserves_constant_within_user_difference():
    left = np.array([[0.0], [0.25], [0.5], [0.75]])
    result = compare().paired_bootstrap(left, left + 0.125, 20261027, 200)
    assert result["difference"] == [-0.125]
    assert result["ci_low"] == result["ci_high"] == [-0.125]


def test_identical_models_have_zero_observed_difference():
    values = np.array([[0, 0.0], [1, 0.5], [0, 0.0], [1, 1.0]])
    result = compare().paired_bootstrap(values, values, 42, 200)
    assert result["difference"] == [0.0, 0.0]
    assert result["ci_low"] == result["ci_high"] == [0.0, 0.0]


def test_paired_bootstrap_is_deterministic_and_rejects_bad_inputs():
    left = np.array([[0.0], [1.0], [0.0]])
    right = np.array([[1.0], [0.0], [0.0]])
    assert compare().paired_bootstrap(left, right, 7, 100) == compare().paired_bootstrap(left, right, 7, 100)
    with pytest.raises(ValueError):
        compare().paired_bootstrap(left, right[:2], 7, 100)
    with pytest.raises(ValueError):
        compare().paired_bootstrap(left * np.nan, right, 7, 100)
