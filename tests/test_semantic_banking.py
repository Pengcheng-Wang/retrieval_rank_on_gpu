import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_banking import bank_shard, validate_bank_progress
from rrbench.semantic_models import configuration_digest


def shard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    spec = {"model": "example", "revision": "fixed", "dimensions": [2], "pooling": "CLS", "max_tokens": 512}
    (source / "corpus.tsv").write_text("0\tfirst\n1\tsecond\n", encoding="utf-8")
    source_digest = sha256(source / "corpus.tsv")
    np.save(source / "ids.npy", np.array([0, 1], dtype=np.int64))
    np.save(source / "vectors.npy", np.eye(2, dtype=np.float16))
    write_json(source / "run-config.json", {
        "specification_sha256": configuration_digest(spec), "source_sha256": source_digest,
        "model": "example", "revision": "fixed", "batch_size": 64, "max_rows": None,
    })
    write_json(source / "completed.json", {
        "complete": True, "rows": 2, "native_dimension": 2,
        "vectors_sha256": sha256(source / "vectors.npy"), "ids_sha256": sha256(source / "ids.npy"),
        "mrl_representations": {}, "specification_sha256": configuration_digest(spec),
        "source_sha256": source_digest,
    })
    return source, spec


def task(source):
    return {
        "key": "example", "corpus": "msmarco", "model": "tiny",
        "rows": 2, "start": 0, "batch_size": 64,
        "source_sha256": sha256(source / "corpus.tsv"), "output": str(source),
    }


def test_serial_bank_publishes_checked_completion_only_after_all_files(tmp_path):
    source, spec = shard(tmp_path)
    destination = tmp_path / "durable"
    result = bank_shard(source, destination, task(source), spec)
    assert result["complete"]
    assert sha256(destination / "vectors.npy") == sha256(source / "vectors.npy")
    assert bank_shard(source, destination, task(source), spec) == result


def test_corrupt_source_cannot_publish_success_shaped_durable_receipt(tmp_path):
    source, spec = shard(tmp_path)
    (source / "vectors.npy").write_bytes(b"broken")
    destination = tmp_path / "durable"
    with pytest.raises(ValueError, match="checksum"):
        bank_shard(source, destination, task(source), spec)
    assert not (destination / "completed.json").exists()


def test_unchanged_ids_and_model_do_not_authorize_old_source_text(tmp_path):
    source, spec = shard(tmp_path)
    expected = {**task(source), "source_sha256": "f" * 64}
    with pytest.raises(ValueError, match="source"):
        bank_shard(source, tmp_path / "durable", expected, spec)


def test_saved_progress_cannot_certify_a_missing_or_different_destination(tmp_path):
    source, spec = shard(tmp_path)
    entry = task(source)
    plan = {"tasks": [entry], "registry": {"models": {"tiny": spec}}}
    bank = tmp_path / "bank"
    target = bank / "embeddings" / "msmarco" / "tiny" / "example"
    completed = bank_shard(source, target, entry, spec)
    progress = {
        "plan_sha256": configuration_digest(plan), "bank_directory": str(bank.resolve()),
        "banked": {"example": {
            "rows": 2, "vectors_sha256": completed["vectors_sha256"],
            "receipt_sha256": sha256(target / "completed.json"),
        }},
    }
    assert validate_bank_progress(plan, bank, progress) == {"example"}
    with pytest.raises(ValueError, match="destination"):
        validate_bank_progress(plan, tmp_path / "different", progress)
    (target / "vectors.npy").unlink()
    with pytest.raises(FileNotFoundError):
        validate_bank_progress(plan, bank, progress)
