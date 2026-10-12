import numpy as np
import pytest

from rrbench.semantic_token_audit import length_summary


def test_static_token_audit_counts_actual_overflow_not_cap_equality():
    result = length_summary(np.array([1, 8, 16, 17], dtype=np.uint32), 16)
    assert result["rows"] == 4
    assert result["raw_rows_exceeding_cap"] == 1
    assert result["strict_processed_rows"] == 4
    assert result["raw_length_quantiles"]["max"] == 17
    assert result["generation_finish_reason"] == "N/A: static text encoder inputs, no generated completions."


def test_invalid_token_lengths_are_not_certified_as_clean():
    with pytest.raises(ValueError, match="positive"):
        length_summary(np.array([0, 10]), 16)
