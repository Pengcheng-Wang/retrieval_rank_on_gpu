from pathlib import Path

import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_banking import process_artifact_requests


def fixture_request(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"expensive-checkpoint")
    write_json(source / "completed.json", {"complete": True})
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    request = {
        "source_root": str(source), "destination": "ranker/seed17",
        "files": {name: sha256(source / name) for name in ("model.pt", "completed.json")},
        "description": "Validated training checkpoint, single serialized writer.",
    }
    write_json(request_dir / "ranker.json", request)
    return source, request_dir, request


def test_same_writer_can_bank_completed_training_checkpoints_without_second_gen2_writer(tmp_path):
    source, requests, _ = fixture_request(tmp_path)
    bank = tmp_path / "bank"
    process_artifact_requests(requests, bank)
    assert sha256(bank / "artifacts" / "ranker" / "seed17" / "model.pt") == sha256(source / "model.pt")
    assert (bank / "artifacts" / "ranker" / "seed17" / "durable.json").exists()
    process_artifact_requests(requests, bank)
    assert (requests / "ranker.banked.json").exists()


def test_artifact_request_cannot_escape_publisher_source_or_destination_roots(tmp_path):
    _, requests, request = fixture_request(tmp_path)
    request["destination"] = "..\\outside"
    write_json(requests / "ranker.json", request)
    with pytest.raises(ValueError, match="relative"):
        process_artifact_requests(requests, tmp_path / "bank")


def test_expensive_checkpoint_corruption_cannot_publish_durable_success(tmp_path):
    source, requests, _ = fixture_request(tmp_path)
    (source / "model.pt").write_bytes(b"changed")
    bank = tmp_path / "bank"
    with pytest.raises(ValueError, match="checksum"):
        process_artifact_requests(requests, bank)
    assert not (bank / "artifacts" / "ranker" / "seed17" / "durable.json").exists()


def test_preexisting_partial_symlink_cannot_overwrite_outside_the_artifact_directory(tmp_path):
    _, requests, _ = fixture_request(tmp_path)
    bank = tmp_path / "bank"
    target = bank / "artifacts" / "ranker" / "seed17"
    target.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside-owner-data")
    try:
        (target / "model.pt.partial").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"This platform does not grant test symlink creation: {error}")
    with pytest.raises(ValueError, match="symlink"):
        process_artifact_requests(requests, bank)
    assert outside.read_bytes() == b"outside-owner-data"
    assert not (target / "durable.json").exists()


def test_manifest_replacement_cannot_acknowledge_the_new_digest_for_old_checkpoint_bytes(tmp_path, monkeypatch):
    _, requests, request = fixture_request(tmp_path)
    request_file = requests / "ranker.json"
    original_read = Path.read_bytes
    replaced = False

    def replaced_read(path):
        nonlocal replaced
        content = original_read(path)
        if path == request_file and not replaced:
            replaced = True
            changed = {**request, "files": {**request["files"], "model.pt": "0" * 64}}
            write_json(request_file, changed)
        return content

    monkeypatch.setattr(Path, "read_bytes", replaced_read)
    with pytest.raises(ValueError, match="changed"):
        process_artifact_requests(requests, tmp_path / "bank", {})
    assert not (requests / "ranker.banked.json").exists()


def test_existing_durable_receipt_cannot_be_an_ephemeral_outside_symlink(tmp_path):
    _, requests, _ = fixture_request(tmp_path)
    bank = tmp_path / "bank"
    process_artifact_requests(requests, bank)
    receipt = bank / "artifacts" / "ranker" / "seed17" / "durable.json"
    outside = tmp_path / "ephemeral-receipt.json"
    receipt.replace(outside)
    try:
        receipt.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Platform does not grant symlink creation: {error}")
    (requests / "ranker.banked.json").unlink()
    with pytest.raises(ValueError, match="symlink"):
        process_artifact_requests(requests, bank, {})
    assert not (requests / "ranker.banked.json").exists()
