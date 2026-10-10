import numpy as np
import pytest
import json

from rrbench.inference import main, ranking_fidelity, score_fidelity, stable_rankings, training_arrays
from rrbench.io import sha256


def test_fp32_fidelity_is_a_fixed_score_gate_not_ranking_equivalence():
    reference = np.array([[1.0, 1.000001, 2.0]])
    actual = np.array([[1.000002, 1.0, 2.0]])
    valid = np.ones_like(reference, dtype=np.bool_)
    fidelity = score_fidelity(reference, actual, valid)
    assert fidelity["fp32_tolerance_pass"]
    assert ranking_fidelity(reference, actual, valid)["identical_full_order_fraction"] == 0


def test_score_gate_rejects_real_errors_and_undefined_normalization_is_explicit():
    valid = np.ones((1, 2), dtype=np.bool_)
    result = score_fidelity(np.zeros((1, 2)), np.ones((1, 2)), valid)
    assert not result["fp32_tolerance_pass"]
    assert result["normalized_rmse"] is None
    with pytest.raises(ValueError, match="finite"):
        score_fidelity(np.zeros((1, 2)), np.array([[np.inf, 0]]), valid)


def test_masked_ranking_and_trivial_full_set_cutoff():
    scores = np.array([[2.0, 100, 2.0, -1.0]])
    valid = np.array([[True, False, True, True]])
    assert stable_rankings(scores, valid).tolist() == [[0, 2, 3, -1]]
    result = ranking_fidelity(scores, scores, valid)
    assert result["top10_set_overlap"] == 1
    assert result["top100_is_full_candidate_set"]


def test_no_eligible_scores_or_shape_mismatch_fail_explicitly():
    with pytest.raises(ValueError, match="eligible"):
        score_fidelity(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="identical shapes"):
        score_fidelity(np.zeros((1, 2)), np.zeros((2, 2)), np.ones((1, 2), dtype=np.bool_))


def test_inference_training_reader_needs_no_label_or_holdout_fields(tmp_path):
    np.savez(
        tmp_path / "prepared.npz", user_counts=np.array([1, 2]),
        item_counts=np.array([1, 1, 1]), catalog=np.array([0, 1, 2]),
        train_seen=np.array([[True, False, False], [False, True, False]]),
    )
    digest = sha256(tmp_path / "prepared.npz")
    (tmp_path / "manifest.json").write_text(json.dumps({"prepared_sha256": digest}))
    users, items, catalog, seen, observed_digest = training_arrays(tmp_path)
    assert users.tolist() == [1, 2] and catalog.tolist() == [0, 1, 2]
    assert seen.shape == (2, 3) and observed_digest == digest
    (tmp_path / "manifest.json").write_text(json.dumps({"prepared_sha256": "wrong"}))
    with pytest.raises(ValueError, match="checksum"):
        training_arrays(tmp_path)


def test_inference_cli_rejects_missing_frozen_inputs_before_cuda(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["inference", "--output", "unused"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "requires --data and --checkpoints" in capsys.readouterr().err
