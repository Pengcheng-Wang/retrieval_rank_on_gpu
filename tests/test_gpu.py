import numpy as np
import pytest
import torch

from rrbench.ann import FaissANN, Partition, SilverANN
from rrbench.exact import exact_topk
from rrbench.filters import BloomFilter, evaluate, make_features, parse_expression
from rrbench.models import make_model
from rrbench.runtime import configure
from rrbench.quantization import quantize_symmetric

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def fixture_data():
    configure(17, 4)
    rng = np.random.default_rng(17)
    items = rng.standard_normal((100, 16), dtype=np.float32)
    queries = rng.standard_normal((100, 16), dtype=np.float32)
    centroids = rng.standard_normal((4, 16), dtype=np.float32)
    assignments = (items @ centroids.T).argmax(axis=1).astype(np.int64)
    order = np.argsort(assignments, kind="stable")
    offsets = np.concatenate(([0], np.bincount(assignments, minlength=4).cumsum()))
    return items, torch.tensor(queries, device="cuda"), Partition(centroids, assignments, order, offsets)


def test_100_row_all_probe_ann_matches_fp32_exact(fixture_data):
    pytest.importorskip("silvertorch.ops._load_ops")
    pytest.importorskip("faiss")
    items, queries, partition = fixture_data
    truth_scores, truth_ids = exact_topk(items, queries, 10)
    for backend in (SilverANN(items, partition, "fp32"), FaissANN(items, partition, False), FaissANN(items, partition, True)):
        scores, ids = backend.search(queries if backend.cuda else queries.cpu(), 10, 4)
        torch.testing.assert_close(ids.cuda().long(), truth_ids)
        torch.testing.assert_close(scores.cuda(), truth_scores, atol=1e-4, rtol=1e-4)


def test_partial_filter_and_external_ids(fixture_data):
    pytest.importorskip("silvertorch.ops._load_ops")
    items, queries, partition = fixture_data
    features = make_features(100, 17, 3, 1)
    # Exercise boundaries, nonuniform cluster lengths, and negated expressions.
    features[[0, 31, 32, 63, 64, 65, 99], 0, 0] = 1
    expressions = ["1:1", "1:1 AND NOT 2:1", "NOT (2:1 OR 3:1)", "(1:1 OR 2:2) AND NOT 3:1"]
    query = queries[: len(expressions)]
    truth_mask = torch.stack([evaluate(parse_expression(text), torch.tensor(features, device="cuda")) for text in expressions])
    truth_scores, truth_ids = exact_topk(items, query, 10, truth_mask)
    for mode in ("partial", "full"):
        backend = SilverANN(items, partition, "fp32", features, mode)
        scores, ids = backend.search(query, 10, 4, expressions)
        torch.testing.assert_close(ids.long(), truth_ids)
        torch.testing.assert_close(scores, truth_scores, atol=1e-4, rtol=1e-4)


def test_int8_raw_rescaling_does_not_overflow():
    pytest.importorskip("silvertorch.ops._load_ops")
    values = torch.full((2, 128), 127, dtype=torch.int8, device="cuda")
    query = values[:1]
    offsets = torch.tensor([0, 2], device="cuda")
    selected = torch.tensor([[0]], device="cuda")
    lengths = torch.tensor([[2]], device="cuda")
    scores, ids = torch.ops.st.fused_kmean_ann(offsets, selected, lengths, values, query, 2)
    valid = ids >= 0
    assert (scores[valid] == 2064512).all()
    assert torch.isfinite(scores.float() / 128).all()


@pytest.mark.parametrize("mode", ["global", "rowwise"])
def test_int8_all_probe_matches_dequantized_reference(fixture_data, mode):
    pytest.importorskip("silvertorch.ops._load_ops")
    items, queries, partition = fixture_data
    backend = SilverANN(items, partition, f"int8-{mode}")
    item_quantized = quantize_symmetric(torch.tensor(items), mode).dequantize().cuda()
    query_quantized = quantize_symmetric(queries, "rowwise").dequantize()
    scores, ids = backend.search(queries, 10, 4)
    expected = query_quantized @ item_quantized.T
    expected_scores = expected.topk(10, dim=1).values
    # Global symmetric INT8 can tie at adjacent integer score values.
    selected_scores = expected.gather(1, ids.long())
    torch.testing.assert_close(selected_scores, expected_scores, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-4)


def test_query_specific_eligibility_precedes_topk(fixture_data):
    pytest.importorskip("silvertorch.ops._load_ops")
    items, queries, partition = fixture_data
    eligible = torch.ones(2, 100, dtype=torch.bool, device="cuda")
    eligible[0, :50] = False
    eligible[1, 50:] = False
    backend = SilverANN(items, partition, "fp32")
    scores, ids = backend.search(queries[:2], 10, 4, eligible=eligible)
    expected_scores, expected_ids = exact_topk(items, queries[:2], 10, eligible)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", ["mf", "mlp", "torchrec-dlrm", "torchrec-dcn"])
def test_actual_ranker_forward_and_backward(name):
    if name.startswith("torchrec"):
        pytest.importorskip("torchrec")
    model = make_model(name, 100, 100, torch.device("cuda"), 17)
    users = torch.arange(100, device="cuda")
    items = users.flip(0)
    scores = model(users, items, torch.ones(100, 2, device="cuda"))
    assert scores.shape == (100,)
    assert torch.isfinite(scores).all()
    scores.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if name.startswith("torchrec"):
        assert model.model_class.startswith("torchrec.models.dlrm.")
