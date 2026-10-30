import errno
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def broker():
    assert (SCRIPTS / "bank_remote.py").exists(), "Cross-node single-writer broker is missing"
    import bank_remote
    return bank_remote


def test_remote_snapshot_waits_without_claiming_completion(tmp_path):
    result = broker().snapshot_run(tmp_path / "run", tmp_path / "staging", tmp_path / "eval")
    assert result == {"model_complete": False, "evaluation_complete": False, "artifacts": []}


def test_remote_snapshot_binds_only_owned_sealed_checkpoints(tmp_path):
    from protocol import seal_checkpoint, verify_checkpoint
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 7}))
    seal_checkpoint(checkpoint)
    (run / "latest-state.json").write_text(json.dumps({
        "checkpoint": "checkpoint-7", "best_model_checkpoint": str(checkpoint),
    }))
    result = broker().snapshot_run(run, tmp_path / "stage", tmp_path / "eval")
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["kind"] == "checkpoint"
    assert verify_checkpoint(Path(artifact["path"])) == 7
    (checkpoint / "pytorch_model.bin").write_bytes(b"rotated or replaced")
    assert verify_checkpoint(Path(artifact["path"])) == 7


def test_remote_snapshot_rejects_checkpoint_path_escape(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest-state.json").write_text(json.dumps({"checkpoint": "../other/checkpoint-7"}))
    with pytest.raises(ValueError, match="checkpoint"):
        broker().snapshot_run(run, tmp_path / "stage", tmp_path / "eval")


def test_durable_acknowledgment_binds_the_exact_completed_model(tmp_path):
    from protocol import file_sha256, write_json
    run = tmp_path / "run"
    run.mkdir()
    (run / "pytorch_model.bin").write_bytes(b"selected weights")
    write_json(run / "run-config.json", {"seed": 42})
    write_json(run / "completed.json", {
        "status": "complete", "model_sha256": file_sha256(run / "pytorch_model.bin"),
        "run_config_sha256": file_sha256(run / "run-config.json"),
    })
    digest = file_sha256(run / "completed.json")
    broker().acknowledge_model(run, tmp_path / "acks", digest)
    assert json.loads((tmp_path / "acks" / "run.model.json").read_text())["completed_sha256"] == digest
    with pytest.raises(ValueError, match="checksum"):
        broker().acknowledge_model(run, tmp_path / "acks", "0" * 64)


def test_local_watcher_banks_both_latest_and_selected_best(tmp_path):
    from bank_watch import checkpoint_artifacts
    from protocol import seal_checkpoint
    run = tmp_path / "run"
    for step in (5, 7):
        checkpoint = run / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(f"{name}-{step}".encode())
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        seal_checkpoint(checkpoint)
    (run / "latest-state.json").write_text(json.dumps({
        "checkpoint": "checkpoint-7", "best_model_checkpoint": str(run / "checkpoint-5"),
    }))
    artifacts = checkpoint_artifacts(run, tmp_path / "stage")
    assert len(artifacts) == 2
    assert all(artifact["kind"] == "checkpoint" for artifact in artifacts)


def _sealed_remote_run(tmp_path):
    from protocol import seal_checkpoint

    run = tmp_path / "run"
    checkpoint = run / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    for name in ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 7}))
    seal_checkpoint(checkpoint)
    (run / "latest-state.json").write_text(json.dumps({
        "checkpoint": "checkpoint-7",
        "best_model_checkpoint": str(checkpoint),
    }))
    return run, checkpoint


def test_remote_snapshot_skips_copytree_error_when_source_checkpoint_rotates(
    tmp_path, monkeypatch, capsys,
):
    import bank

    module = broker()
    run, checkpoint = _sealed_remote_run(tmp_path)

    def rotate(source, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        (destination / "partial").write_bytes(b"incomplete")
        shutil.rmtree(source)
        raise shutil.Error([(
            str(source / "optimizer.pt"),
            str(destination / "optimizer.pt"),
            f"[Errno {errno.ENOENT}] No such file or directory",
        )])

    monkeypatch.setattr(bank.shutil, "copytree", rotate)
    result = module.snapshot_run(run, tmp_path / "stage", None, include_completed=False)

    assert result["artifacts"] == []
    assert "CHECKPOINT_ROTATED_DURING_SNAPSHOT" in capsys.readouterr().err
    assert not list((tmp_path / "stage").glob(".partial-*"))
    assert len(list((tmp_path / "stage").glob(".quarantine-*"))) == 1


def test_remote_snapshot_does_not_misclassify_staging_copy_error_as_rotation(
    tmp_path, monkeypatch, capsys,
):
    import bank

    module = broker()
    run, _ = _sealed_remote_run(tmp_path)

    def fail_staging(source, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        raise shutil.Error([(
            str(destination / "missing"),
            str(destination / "copy"),
            f"[Errno {errno.ENOENT}] No such file or directory",
        )])

    monkeypatch.setattr(bank.shutil, "copytree", fail_staging)
    with pytest.raises(shutil.Error):
        module.snapshot_run(run, tmp_path / "stage", None, include_completed=False)

    assert "ROTATED" not in capsys.readouterr().err
    assert not list((tmp_path / "stage").glob(".partial-*"))
    assert len(list((tmp_path / "stage").glob(".quarantine-*"))) == 1


def test_remote_snapshot_skips_marker_that_rotates_before_hashing(
    tmp_path, monkeypatch, capsys,
):
    module = broker()
    run, checkpoint = _sealed_remote_run(tmp_path)
    marker = checkpoint / "complete.json"
    original = module.file_sha256

    def rotate(path):
        if Path(path) == marker:
            marker.unlink()
            raise FileNotFoundError(errno.ENOENT, "rotated", str(marker))
        return original(path)

    monkeypatch.setattr(module, "file_sha256", rotate)
    result = module.snapshot_run(run, tmp_path / "stage", None, include_completed=False)

    assert result["artifacts"] == []
    assert "CHECKPOINT_ROTATED_DURING_SNAPSHOT" in capsys.readouterr().err
