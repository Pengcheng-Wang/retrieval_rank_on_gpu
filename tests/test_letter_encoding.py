import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def encoding():
    assert (SCRIPTS / "encode.py").exists(), "Audited semantic encoding is missing"
    return importlib.import_module("encode")


class Tokenizer:
    bos_token_id = 1
    unk_token_id = 0

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[1] + [4] * len(text) for text in texts]}


def test_author_text_rows_are_reordered_without_changing_fields():
    author = SimpleNamespace(generate_text=lambda items, fields: [
        [1, ["second title", "second description"]],
        [0, ["first title", "first description"]],
    ])
    rows = encoding().ordered_texts(author, {"1": {}, "0": {}})
    assert rows == [[0, ["first title", "first description"]],
                    [1, ["second title", "second description"]]]
    author.generate_text = lambda items, fields: [[0, ["title only"]]]
    with pytest.raises(ValueError):
        encoding().ordered_texts(author, {"0": {}})


def test_every_semantic_field_is_audited_without_truncation():
    rows = [[0, ["abc", ""]], [1, ["ok", "123456789"]]]
    audit = encoding().audit_texts(rows, Tokenizer(), 8)
    assert audit["status"] == "fail"
    assert audit["records"] == 4
    assert audit["strict_clean"] == 3
    assert audit["over_cap"] == 1
    assert audit["empty_cleaned_fields"] == 1
    assert audit["length_tokens_p50_p90_p99_max"][-1] == 10
    assert encoding().audit_texts(rows, Tokenizer(), 10)["status"] == "pass"


def test_declared_semantic_settings_cannot_be_ignored():
    spec = json.loads((ROOT / "configs" / "letter-openllama-v2-method-v1.json").read_text())
    encoding().validate_encoding_spec(spec)
    spec["semantic_encoding"]["max_tokens_per_field"] = 4096
    with pytest.raises(ValueError):
        encoding().validate_encoding_spec(spec)


def test_semantic_chunks_reject_wrong_rows_or_provenance(tmp_path):
    module = encoding()
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    array = np.zeros((2, 4096), dtype=np.float32)
    np.save(chunk / "embeddings.npy", array, allow_pickle=False)
    receipt = {
        "status": "complete", "first_item": 3, "stop_item": 5,
        "run_config_sha256": "identity", "shape": [2, 4096],
        "embeddings_sha256": module.file_sha256(chunk / "embeddings.npy"),
    }
    (chunk / "completed.json").write_text(json.dumps(receipt))
    np.testing.assert_array_equal(module.read_chunk(chunk, "identity", 3, 5), array)
    with pytest.raises(ValueError):
        module.read_chunk(chunk, "wrong-track", 3, 5)
    with pytest.raises(ValueError):
        module.read_chunk(chunk, "identity", 2, 4)
    array[0, 0] = 3
    np.save(chunk / "embeddings.npy", array, allow_pickle=False)
    with pytest.raises(ValueError):
        module.read_chunk(chunk, "identity", 3, 5)
