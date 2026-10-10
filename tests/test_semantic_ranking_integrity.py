import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rrbench.io import sha256, write_json
from rrbench.semantic_models import BatchStats, configuration_digest
from rrbench.semantic_queries import encode_query_panel
from rrbench.semantic_rankers import validate_checkpoint_features, validate_fitting_query_alignment
from rrbench.semantic_history import validate_history_panels


class Encoder:
    def __init__(self):
        self.specification = {
            "key": "fake", "model": "example", "revision": "fixed", "dimensions": [2],
            "query_instruction": "default-search", "max_tokens": 512,
        }
        self.native_dimension = 2
        self.instructions = []
        self.tokenizer = lambda texts, **kwargs: {"length": [10] * len(texts)}

    def format_texts(self, texts, query, instruction=None):
        self.instructions.append(instruction)
        return texts

    def encode(self, texts, query=False, instruction=None):
        self.instructions.append(instruction)
        return torch.tensor([[0.6, 0.8]] * len(texts)), BatchStats(len(texts), 10, 16, 0, 0.001)


def test_offline_history_queries_use_the_same_effective_instruction_as_online(tmp_path):
    panel = tmp_path / "panel.json"
    write_json(panel, {
        "queries": ["past exposure one", "past exposure two"], "query_ids": ["q1", "q2"],
        "history_instruction": "recommend videos",
    })
    encoder = Encoder()
    completed = encode_query_panel(encoder, panel, tmp_path / "out", 2)
    assert set(encoder.instructions) == {"recommend videos"}
    assert completed["query_instruction"] == "recommend videos"


def test_equal_dimension_checkpoint_cannot_use_another_encoder_or_cluster_codebook(tmp_path):
    for filename in ("items.npy", "partition.npz", "lengths.npy"):
        (tmp_path / filename).write_bytes(filename.encode())
    representation = {"model": "encoderA", "revision": "fixed", "stored_dimension": 1024}
    binding = {
        "sources": {
            "items": sha256(tmp_path / "items.npy"), "partition": sha256(tmp_path / "partition.npz"),
            "token_lengths": sha256(tmp_path / "lengths.npy"),
        },
        "representation": representation,
    }
    validate_checkpoint_features(
        binding, tmp_path / "items.npy", tmp_path / "partition.npz", tmp_path / "lengths.npy", representation,
    )
    with pytest.raises(ValueError, match="representation"):
        validate_checkpoint_features(
            binding, tmp_path / "items.npy", tmp_path / "partition.npz", tmp_path / "lengths.npy",
            {**representation, "model": "encoderB"},
        )
    (tmp_path / "partition.npz").write_bytes(b"different equal sized codebook")
    with pytest.raises(ValueError, match="partition"):
        validate_checkpoint_features(
            binding, tmp_path / "items.npy", tmp_path / "partition.npz", tmp_path / "lengths.npy", representation,
        )


def test_fitting_vectors_cannot_attach_valid_features_to_reordered_query_labels(tmp_path):
    panel = tmp_path / "panel.json"
    write_json(panel, {"query_ids": ["qA", "qB"], "queries": ["first", "second"]})
    vectors = tmp_path / "vectors" / "vectors.npy"
    vectors.parent.mkdir()
    np.save(vectors, np.eye(2, dtype=np.float16))
    write_json(vectors.parent / "completed.json", {
        "complete": True, "panel_sha256": sha256(panel), "rows": 2, "query_ids": ["qB", "qA"],
    })
    with pytest.raises(ValueError, match="alignment"):
        validate_fitting_query_alignment(vectors, panel)


def test_validation_and_test_only_history_receipt_cannot_claim_required_fit_panel_exists(tmp_path):
    write_json(tmp_path / "completed.json", {
        "complete": True, "panel_sha256": {"validation": "first", "test": "second"},
    })
    with pytest.raises(ValueError, match="fit"):
        validate_history_panels(tmp_path)
