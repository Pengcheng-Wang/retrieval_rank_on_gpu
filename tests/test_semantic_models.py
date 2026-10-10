import numpy as np
import pytest
import sys
import torch
from types import SimpleNamespace

from rrbench.semantic_models import (
    configuration_digest, encode_shard, mrl_prefix, normalize, pool_hidden,
    model_file_manifest, validate_encoding_contract, validate_smoke_contract,
)


def test_last_token_pooling_handles_both_padding_sides():
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    expected = torch.stack((hidden[0, 4], hidden[1, 2]))
    torch.testing.assert_close(pool_hidden(hidden, mask, "last valid token"), expected)


def test_masked_mean_ignores_padding_and_normalizes():
    hidden = torch.tensor([[[1.0, 0], [0, 1], [100, 100]]])
    pooled = pool_hidden(hidden, torch.tensor([[1, 1, 0]]), "attention-mask mean")
    torch.testing.assert_close(pooled, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(normalize(pooled).norm(dim=1), torch.ones(1))
    with pytest.raises(ValueError, match="Zero embedding"):
        normalize(torch.zeros(1, 2))


def test_invalid_pooling_input_fails_instead_of_returning_padding():
    with pytest.raises(ValueError, match="nonpadding"):
        pool_hidden(torch.ones(1, 2, 3), torch.zeros(1, 2), "CLS")


def test_mrl_prefix_is_independently_renormalized_for_documents_and_queries():
    values = normalize(torch.tensor([[3.0, 4, 12, 0], [0, 5, 0, 12]]))
    reduced = mrl_prefix(values, 2)
    torch.testing.assert_close(reduced.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(reduced, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        mrl_prefix(values, 5)


def test_registry_configuration_binding_is_not_newline_dependent():
    assert configuration_digest({"dimension": 1024, "model": "example"}) == configuration_digest({
        "model": "example", "dimension": 1024,
    })


def test_invalid_batch_cannot_publish_a_zero_matrix_as_encoded(tmp_path):
    with pytest.raises(ValueError, match="positive before encoding"):
        encode_shard(tmp_path / "unused", tmp_path / "output", {}, tmp_path / "cache", -1)
    assert not (tmp_path / "output").exists()


def test_reuse_contract_cannot_accept_changed_pooling_or_missing_required_mrl():
    spec = {
        "model": "example", "revision": "fixed", "dimensions": [4, 2], "pooling": "CLS",
        "max_tokens": 512, "query_instruction": "task",
    }
    completed = {
        "specification_sha256": configuration_digest(spec),
        "native_dimension": 4, "mrl_representations": {"2": {"dimension": 2}},
    }
    validate_encoding_contract(spec, completed)
    with pytest.raises(ValueError, match="specification"):
        validate_encoding_contract({**spec, "pooling": "attention-mask mean"}, completed)
    with pytest.raises(ValueError, match="MRL"):
        validate_encoding_contract(spec, {**completed, "mrl_representations": {}})


def test_old_smoke_cannot_authorize_changed_token_budget_or_pooling():
    spec = {
        "revision": "fixed", "dimensions": [4, 2], "pooling": "CLS", "max_tokens": 512,
    }
    smoke = {
        "status": "ok", "rows": 100, "revision": "fixed",
        "specification_sha256": configuration_digest(spec), "dimension": 4,
        "mrl_representations": {"2": {}},
    }
    validate_smoke_contract(spec, smoke)
    with pytest.raises(ValueError, match="specification"):
        validate_smoke_contract({**spec, "max_tokens": 256}, smoke)


def test_safetensors_snapshot_does_not_require_unneeded_onnx_or_pickle_exports(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"fixture-weight-bytes")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    def download(*_, **kwargs):
        patterns = kwargs.get("allow_patterns", [])
        if "*.safetensors" not in patterns or "*.json" not in patterns:
            raise ValueError("Incomplete snapshot includes missing unrequested ONNX/pickle exports.")
        assert kwargs["local_files_only"]
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    receipt = model_file_manifest({"model": "example", "revision": "fixed"}, tmp_path)
    assert set(receipt["files"]) == {"model.safetensors", "config.json"}
