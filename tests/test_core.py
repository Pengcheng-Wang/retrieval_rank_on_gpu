import numpy as np
import pytest
import torch

from rrbench.exact import exact_topk
from rrbench.filters import InvertedFilter, conservative_positive, evaluate, exact_candidate_mask, parse_expression
from rrbench.metrics import latency_summary, recall_at_k, single_positive_metrics
from rrbench.quantization import quantize_symmetric
from rrbench.synthetic import generate_catalog
from rrbench.io import bind_run_config, write_json


def test_recall_ignores_padding_and_duplicates():
    assert recall_at_k(np.array([[1, 1, -1]]), np.array([[1, 2, 3]]), 3).item() == pytest.approx(1 / 3)


def test_single_positive_metrics_and_misses():
    result = single_positive_metrics(np.array([[2, 3, 4], [7, 8, 9]]), np.array([3, 4]), [1, 2])
    assert result["recall@1"] == 0
    assert result["recall@2"] == 0.5
    assert result["ndcg@2"] == pytest.approx(0.5 / np.log2(3))


def test_latency_is_not_serving_qps():
    result = latency_summary([1.0, 2.0, 3.0], 16)
    assert result["offline_queries_per_second"] == 8000
    assert "not serving" in result["throughput_kind"]


@pytest.mark.parametrize("mode", ["rowwise", "global"])
def test_symmetric_quantization_zero_and_outlier(mode):
    values = torch.tensor([[0.0, 0, 0], [1.0, -2, 100]])
    result = quantize_symmetric(values, mode)
    assert torch.isfinite(result.scales).all()
    torch.testing.assert_close(result.dequantize()[0], values[0])
    assert (result.dequantize() - values).abs().max() <= result.scales.max() / 2 + 1e-5


def test_exact_chunking_and_eligibility():
    items = torch.tensor([[1.0, 0], [0, 1], [-1, 0], [0.5, 0]])
    query = torch.tensor([[1.0, 0]])
    scores, ids = exact_topk(items, query, 3, torch.tensor([[False, True, True, True]]), chunk_size=2)
    assert ids.tolist() == [[3, 1, 2]]
    torch.testing.assert_close(scores, torch.tensor([[0.5, 0, -1]]))


@pytest.mark.parametrize("text", [
    "1:1", "1:1 AND NOT 2:2", "NOT (1:1 OR 2:2)",
    "NOT (1:1 AND NOT 2:2)", "(1:1 OR 2:2) AND NOT (1:2 AND 2:1)",
])
def test_conservative_not_never_discards_exact_matches(text):
    features = torch.tensor([[[1], [1]], [[1], [2]], [[2], [1]], [[2], [2]]])
    expression = parse_expression(text)
    truth = evaluate(expression, features)
    candidate = evaluate(conservative_positive(expression), features)
    assert not (truth & ~candidate).any()


@pytest.mark.parametrize("text", ["1:1 X 2:2", "(1:1", "AND 1:1", "1:1 2:2", "1:1);print(1)"])
def test_invalid_filters_fail_explicitly(text):
    with pytest.raises(ValueError):
        parse_expression(text)


def test_synthetic_catalog_reproducibility(tmp_path):
    first, queries = generate_catalog(tmp_path, 100, 16, 10, 17, "clustered")
    second, second_queries = generate_catalog(tmp_path, 100, 16, 10, 17, "clustered")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(queries, second_queries)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError):
        generate_catalog(tmp_path, 101, 16, 10, 17, "clustered")


@pytest.mark.parametrize("text", ["1:1", "1:1 AND NOT 2:2", "1:1 OR 2:2", "NOT (1:1 OR 2:2)"])
def test_inverted_reference_matches_forward(text):
    features = np.array([[[1, 1], [1, 2]], [[1, 2], [2, 2]], [[3, 3], [3, 3]]])
    expression = parse_expression(text)
    np.testing.assert_array_equal(InvertedFilter(features).search(expression), evaluate(expression, torch.tensor(features)).numpy())


@pytest.mark.parametrize("expressions", [
    ["1:1 AND NOT 2:2"] * 3,
    ["1:1 AND NOT 2:2", "NOT (1:1 OR 2:2)", "1:1 AND NOT 2:2"],
])
def test_batched_exact_verification_matches_individual_rows(expressions):
    features = torch.tensor([
        [[[1, 2], [2, 3]], [[2, 3], [1, 4]]],
        [[[1, 1], [1, 2]], [[1, 3], [1, 4]]],
        [[[2, 3], [2, 2]], [[1, 3], [1, 4]]],
    ])
    expected = torch.stack([evaluate(parse_expression(text), row) for text, row in zip(expressions, features)])
    torch.testing.assert_close(exact_candidate_mask(features, expressions), expected)


def test_completed_run_cannot_be_silently_reused_for_new_configuration(tmp_path):
    bind_run_config(tmp_path, {"seed": 17, "model": "mf"})
    bind_run_config(tmp_path, {"seed": 17, "model": "mf"})
    with pytest.raises(ValueError, match="different experiment"):
        bind_run_config(tmp_path, {"seed": 23, "model": "mf"})


def test_legacy_completed_artifact_cannot_be_rebound(tmp_path):
    (tmp_path / "completed.json").write_text('{"complete": true}')
    with pytest.raises(ValueError, match="retain it read-only"):
        bind_run_config(tmp_path, {"seed": 17})


def test_json_artifacts_have_platform_independent_line_endings(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"value": 17})
    assert b"\r" not in path.read_bytes()
