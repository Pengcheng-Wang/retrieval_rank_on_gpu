import copy
import errno
import importlib
import json
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
from protocol import file_sha256, write_json


def helper():
    assert (SCRIPTS / "bank_baselines.py").is_file(), "Baseline snapshot helper is missing"
    return importlib.import_module("bank_baselines")


@pytest.fixture
def workspace():
    path = ROOT / ".local" / "letter" / "bank-tests" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def fixture_run(root, name="bpr-instruments-42", phase="checkpoint"):
    run = root / name
    run.mkdir()
    users = ["3", "7"]
    config = {
        "schema_version": 1, "kind": "declared-reconstruction-not-recovered-letter-settings",
        "run_class": "contract-gate", "model": "bpr", "dataset": "Instruments",
        "counts": {"users": 2, "items": 12, "optimizer_steps_per_epoch": 1},
        "training": {"max_epochs": 10, "validate_every": 2},
        "protocol": {"seen_item_exclusion": False},
        "provenance": {"data_sha256": {"inter": "a" * 64}},
    }
    write_json(run / "run-config.json", config)
    write_json(run / "user-map.json", users)
    predictions = [
        {"user": u, "model_user": index, "target": 0, "top10": list(range(10))}
        for index, u in enumerate(users, 1)
    ]
    metrics = {f"{m}@{k}": 1.0 for k in (5, 10) for m in ("recall", "ndcg")}
    prediction = run / "validation-top10-00002.json"
    write_json(prediction, predictions)
    selection = {"epoch": 2, "validation": metrics, "predictions": prediction.name,
                 "predictions_sha256": file_sha256(prediction)}
    history = [
        {"epoch": i, "mean_loss": 0.6, "optimizer_steps": 1, "seconds": 0.01}
        for i in range(1, 6)
    ]
    for epoch in (2, 4, 5):
        path = run / "checkpoints" / f"epoch-{epoch:05d}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"synthetic-checkpoint-{epoch}".encode())
        write_json(path.with_suffix(".json"), {
            "epoch": epoch, "sha256": file_sha256(path),
            "config_sha256": file_sha256(run / "run-config.json"),
            "selection": selection, "log": history[epoch - 1],
        })
    write_json(run / "epochs.json", history)
    if phase in {"model", "evaluation"}:
        freeze_run(run)
    if phase == "evaluation":
        finish_evaluation(run)
    return run


def freeze_run(run):
    config = json.loads((run / "run-config.json").read_text())
    latest = json.loads((run / "checkpoints" / "epoch-00005.json").read_text())
    selection = latest["selection"]
    selected = run / "checkpoints" / "epoch-00002.pt"
    write_json(run / "train-completed.json", {
        "status": "train-complete", "run_class": config["run_class"], "model": config["model"],
        "dataset": config["dataset"], "selected_epoch": 2, "last_epoch": 5,
        "last_checkpoint": "checkpoints/epoch-00005.pt",
        "selected_checkpoint": "checkpoints/epoch-00002.pt",
        "selected_checkpoint_sha256": file_sha256(selected),
        "validation": selection["validation"], "validation_predictions": selection["predictions"],
        "validation_predictions_sha256": selection["predictions_sha256"],
        "config_sha256": file_sha256(run / "run-config.json"),
        "user_map_sha256": file_sha256(run / "user-map.json"),
        "provenance": config["provenance"], "test_evaluated": False, "early_stopped": True,
    })


def finish_evaluation(run):
    config = json.loads((run / "run-config.json").read_text())
    frozen = json.loads((run / "train-completed.json").read_text())
    shutil.copyfile(run / frozen["validation_predictions"], run / "test-top10.json")
    write_json(run / "test-claimed.json", {
        "config_sha256": frozen["config_sha256"],
        "frozen_sha256": file_sha256(run / "train-completed.json"),
        "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"],
    })
    write_json(run / "completed.json", {
        "status": "complete", "run_class": config["run_class"], "kind": config["kind"],
        "model": config["model"], "dataset": config["dataset"], "test": frozen["validation"],
        "test_users": 2, "items": 12, "test_evaluated": True, "selected_epoch": 2,
        "validation": frozen["validation"], "config_sha256": frozen["config_sha256"],
        "frozen_train_receipt_sha256": file_sha256(run / "train-completed.json"),
        "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"],
        "test_predictions_sha256": file_sha256(run / "test-top10.json"),
        "user_map_sha256": frozen["user_map_sha256"], "provenance": config["provenance"],
    })


def arguments(root, runs):
    return SimpleNamespace(
        runs=runs, staging=root / "staging", destination=root / "fake-durable",
        ack_root=root / "acks", lock_file=root / "bank.lock",
        writer_host="node-0", interval=30,
    )


@pytest.fixture
def node0(monkeypatch):
    module = helper()
    monkeypatch.setattr(module.socket, "gethostname", lambda: "node-0")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("AZUREML_CR_NODE_RANK", "0")
    return module


def test_snapshot_contains_latest_and_earlier_best_without_false_completion(workspace):
    module = helper()
    run = fixture_run(workspace)
    artifact = module.create_snapshot(run, workspace / "stage")
    assert artifact["status"] == "ready" and artifact["phase"] == "checkpoint"
    snapshot = Path(artifact["path"])
    receipt = module.verify_snapshot(snapshot)
    assert receipt["latest_epoch"] == 5 and receipt["selection"]["epoch"] == 2
    assert receipt["latest_checkpoint_sha256"][:12] in artifact["identity"]
    payload = snapshot / "payload"
    assert (snapshot / "completed.json").is_file()
    assert not (payload / "completed.json").exists()
    assert not (payload / "train-completed.json").exists()
    assert sorted(p.name for p in (payload / "checkpoints").iterdir()) == [
        "epoch-00002.json", "epoch-00002.pt", "epoch-00005.json", "epoch-00005.pt"
    ]
    assert (payload / "validation-top10-00002.json").is_file()
    assert (payload / "run-config.json").read_bytes() == (run / "run-config.json").read_bytes()
    assert (payload / "user-map.json").read_bytes() == (run / "user-map.json").read_bytes()
    assert module.create_snapshot(run, workspace / "stage") == artifact


def test_missing_not_started_run_and_no_checkpoint_are_explicit_pending(workspace, capsys):
    module = helper()
    missing = module.create_snapshot(workspace / "missing", workspace / "stage")
    assert missing["status"] == "pending"
    empty = workspace / "empty"
    empty.mkdir()
    assert module.create_snapshot(empty, workspace / "stage")["status"] == "pending"
    assert "NOT_STARTED" in capsys.readouterr().err
    assert not (workspace / "stage").exists()


def test_snapshot_before_first_validation_needs_no_invented_selection(workspace):
    module = helper()
    run = fixture_run(workspace)
    shutil.rmtree(run / "checkpoints")
    folder = run / "checkpoints"
    folder.mkdir()
    (folder / "epoch-00001.pt").write_bytes(b"first epoch")
    write_json(folder / "epoch-00001.json", {
        "epoch": 1, "sha256": file_sha256(folder / "epoch-00001.pt"),
        "config_sha256": file_sha256(run / "run-config.json"), "selection": None,
        "log": {"epoch": 1, "mean_loss": 0.8, "optimizer_steps": 1},
    })
    artifact = module.create_snapshot(run, workspace / "stage")
    receipt = module.verify_snapshot(Path(artifact["path"]))
    assert receipt["selection"] is None
    assert not list((Path(artifact["path"]) / "payload").glob("validation-*.json"))


@pytest.mark.parametrize("target", [
    "checkpoints/epoch-00005.pt", "checkpoints/epoch-00002.pt", "validation-top10-00002.json",
    "run-config.json",
])
def test_hash_corruption_is_never_reported_as_rotation(workspace, target, capsys):
    module = helper()
    run = fixture_run(workspace)
    path = run.joinpath(*target.split("/"))
    if target.endswith(".pt"):
        path.write_bytes(b"tampered")
    else:
        value = json.loads(path.read_text())
        if target == "run-config.json":
            value["model"] = "hgn"
        else:
            value[0]["top10"].reverse()
        write_json(path, value)
    with pytest.raises(ValueError, match="checksum|config"):
        module.create_snapshot(run, workspace / "stage")
    assert "ROTATED" not in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["epoch", "selection_path", "future_best", "metrics"])
def test_inconsistent_checkpoint_metadata_is_rejected(workspace, mutation):
    module = helper()
    run = fixture_run(workspace)
    marker = run / "checkpoints" / "epoch-00005.json"
    record = json.loads(marker.read_text())
    if mutation == "epoch":
        record["epoch"] = 6
    elif mutation == "selection_path":
        record["selection"]["predictions"] = "../outside.json"
    elif mutation == "future_best":
        record["selection"]["epoch"] = 6
    else:
        record["selection"]["validation"]["ndcg@10"] = 0.1
    write_json(marker, record)
    with pytest.raises(ValueError):
        module.create_snapshot(run, workspace / "stage")


def test_only_expected_source_checkpoint_rotation_skips_cycle(workspace, monkeypatch, capsys):
    module = helper()
    run = fixture_run(workspace)
    original = module.shutil.copyfile

    def rotate(source, destination, *args, **kwargs):
        if Path(source) == run / "checkpoints" / "epoch-00005.pt":
            Path(source).unlink()
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "copyfile", rotate)
    result = module.create_snapshot(run, workspace / "stage")
    assert result["status"] == "rotated"
    assert "CHECKPOINT_ROTATED_DURING_SNAPSHOT" in capsys.readouterr().err
    assert not list((workspace / "stage").glob("baseline-*/completed.json"))
    assert list((workspace / "stage").glob(".bank-baselines-quarantine-*"))


def test_unexpected_missing_immutable_file_or_staging_failure_is_not_rotation(
    workspace, monkeypatch, capsys
):
    module = helper()
    run = fixture_run(workspace)
    (run / "user-map.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.create_snapshot(run, workspace / "stage")
    write_json(run / "user-map.json", ["3", "7"])

    def missing_destination(source, destination, *args, **kwargs):
        raise FileNotFoundError(errno.ENOENT, "missing staging destination", str(destination))

    monkeypatch.setattr(module.shutil, "copyfile", missing_destination)
    with pytest.raises(FileNotFoundError):
        module.create_snapshot(run, workspace / "stage")
    assert "ROTATED" not in capsys.readouterr().err


def test_partial_recovery_quarantines_only_owned_staging(workspace, monkeypatch, capsys):
    module = helper()
    run = fixture_run(workspace)
    stage = workspace / "stage"
    stage.mkdir()
    foreign = stage / f".bank-baselines-partial-{uuid.uuid4().hex}"
    foreign.mkdir()
    (foreign / "someone-elses-file").write_bytes(b"keep")
    original = module.shutil.copyfile

    def fail_copy(*args, **kwargs):
        raise PermissionError("deliberate copy failure")

    monkeypatch.setattr(module.shutil, "copyfile", fail_copy)
    with pytest.raises(PermissionError):
        module.create_snapshot(run, stage)
    owned = set(stage.glob(".bank-baselines-partial-*")) - {foreign}
    assert len(owned) == 1
    monkeypatch.setattr(module.shutil, "copyfile", original)
    artifact = module.create_snapshot(run, stage)
    module.verify_snapshot(Path(artifact["path"]))
    assert foreign.is_dir() and (foreign / "someone-elses-file").read_bytes() == b"keep"
    assert not next(iter(owned)).exists()
    assert len(list(stage.glob(".bank-baselines-quarantine-*"))) == 1
    assert "QUARANTINED_PARTIAL" in capsys.readouterr().err


def test_completed_local_snapshot_is_never_overwritten_or_repaired(workspace):
    module = helper()
    run = fixture_run(workspace)
    artifact = module.create_snapshot(run, workspace / "stage")
    corrupt = Path(artifact["path"]) / "payload" / "checkpoints" / "epoch-00005.pt"
    corrupt.write_bytes(b"damaged local snapshot")
    with pytest.raises(ValueError, match="checksum|manifest"):
        module.create_snapshot(run, workspace / "stage")
    assert corrupt.read_bytes() == b"damaged local snapshot"


def test_frozen_model_receipt_must_match_latest_selection(workspace):
    module = helper()
    run = fixture_run(workspace, phase="model")
    artifact = module.create_snapshot(run, workspace / "stage")
    assert artifact["phase"] == "model"
    payload = Path(artifact["path"]) / "payload"
    assert (payload / "train-completed.json").read_bytes() == (run / "train-completed.json").read_bytes()
    frozen = json.loads((run / "train-completed.json").read_text())
    frozen["selected_checkpoint_sha256"] = "0" * 64
    write_json(run / "train-completed.json", frozen)
    with pytest.raises(ValueError, match="selected|checksum"):
        module.create_snapshot(run, workspace / "stage")


def test_completed_evaluation_snapshot_copies_the_entire_immutable_run(workspace):
    module = helper()
    run = fixture_run(workspace, phase="evaluation")
    (run / "extra-audit.txt").write_text("private audit", encoding="utf-8")
    artifact = module.create_snapshot(run, workspace / "stage")
    assert artifact["phase"] == "evaluation"
    payload = Path(artifact["path"]) / "payload"
    assert module.bank.source_manifest(payload) == module.bank.source_manifest(run)
    assert (payload / "checkpoints" / "epoch-00004.pt").is_file()
    assert (payload / "test-top10.json").is_file()
    assert (payload / "extra-audit.txt").is_file()


def test_completed_evaluation_rejects_prediction_and_frozen_receipt_corruption(workspace):
    module = helper()
    run = fixture_run(workspace, phase="evaluation")
    completed = json.loads((run / "completed.json").read_text())
    completed["frozen_train_receipt_sha256"] = "0" * 64
    write_json(run / "completed.json", completed)
    with pytest.raises(ValueError, match="frozen|checksum"):
        module.create_snapshot(run, workspace / "stage")


def test_snapshot_cannot_be_staged_inside_its_source(workspace):
    run = fixture_run(workspace)
    with pytest.raises(ValueError, match="overlap|inside"):
        helper().create_snapshot(run, run / "staging")


def test_watcher_banks_under_common_lock_then_acknowledges_full_model_and_test(
    workspace, node0, monkeypatch
):
    run = fixture_run(workspace, phase="model")
    args = arguments(workspace, [run])
    original_copy, original_lock = node0.bank.copy_artifact, node0._locked
    lock_held = [False]
    bank_calls = []

    @contextmanager
    def track_lock(path):
        with original_lock(path):
            common = Path(path) == args.lock_file
            if common:
                lock_held[0] = True
            try:
                yield
            finally:
                if common:
                    lock_held[0] = False

    def copy_artifact(source, destination):
        assert lock_held[0], "Every durable write must hold the supplied common bank.lock"
        assert Path(source).parent == args.staging
        assert (Path(source) / "payload" / "checkpoints" / "epoch-00002.pt").is_file()
        bank_calls.append(destination)
        return original_copy(source, destination)

    monkeypatch.setattr(node0, "_locked", track_lock)
    monkeypatch.setattr(node0.bank, "copy_artifact", copy_artifact)
    banked = {}
    assert node0.watch_cycle(args, banked) is False
    model_ack_path = args.ack_root / f"{run.name}.model.json"
    model_ack = json.loads(model_ack_path.read_text())
    assert model_ack["train_completed_sha256"] == file_sha256(run / "train-completed.json")
    assert model_ack["status"] == "banked"
    assert not (args.ack_root / f"{run.name}.evaluation.json").exists()
    assert node0.watch_cycle(args, banked) is False
    assert len(bank_calls) == 1
    finish_evaluation(run)
    assert node0.watch_cycle(args, banked) is True
    final_ack = json.loads((args.ack_root / f"{run.name}.evaluation.json").read_text())
    assert final_ack["completed_sha256"] == file_sha256(run / "completed.json")
    assert json.loads(model_ack_path.read_text()) == model_ack
    assert len(bank_calls) == 2
    assert node0.watch_cycle(args, banked) is True
    assert len(bank_calls) == 2


def test_banking_failure_produces_no_ack_and_is_explicit(workspace, node0, monkeypatch, capsys):
    run = fixture_run(workspace, phase="evaluation")
    args = arguments(workspace, [run])

    def fail(*unused):
        raise OSError("deliberate durable transfer failure")

    monkeypatch.setattr(node0.bank, "copy_artifact", fail)
    assert node0.watch_cycle(args, {}) is False
    assert not list(args.ack_root.glob("*.json"))
    assert "deliberate durable transfer failure" in capsys.readouterr().err


def test_success_shaped_copy_without_verified_receipt_never_acknowledges(
    workspace, node0, monkeypatch
):
    run = fixture_run(workspace, phase="evaluation")
    args = arguments(workspace, [run])
    monkeypatch.setattr(node0.bank, "copy_artifact", lambda *unused: {})
    assert node0.watch_cycle(args, {}) is False
    assert not list(args.ack_root.glob("*.json"))


def test_watcher_waits_for_every_evaluation_and_missing_future_runs(workspace, node0):
    run = fixture_run(workspace, phase="evaluation")
    future = workspace / "future"
    args = arguments(workspace, [run, future])
    banked = {}
    assert node0.watch_cycle(args, banked) is False
    fixture_run(workspace, name="future", phase="model")
    assert node0.watch_cycle(args, banked) is False
    finish_evaluation(future)
    assert node0.watch_cycle(args, banked) is True


def test_wrong_existing_ack_is_an_error_not_an_overwrite(workspace, node0, capsys):
    run = fixture_run(workspace, phase="model")
    args = arguments(workspace, [run])
    assert node0.watch_cycle(args, {}) is False
    ack = args.ack_root / f"{run.name}.model.json"
    original = json.loads(ack.read_text())
    bad = copy.deepcopy(original)
    bad["train_completed_sha256"] = "0" * 64
    write_json(ack, bad)
    assert node0.watch_cycle(args, {}) is False
    assert json.loads(ack.read_text()) == bad
    assert "acknowledgment" in capsys.readouterr().err


def test_acknowledgment_cannot_point_to_another_valid_banked_run(workspace, node0, capsys):
    first = fixture_run(workspace, name="first", phase="evaluation")
    other = fixture_run(workspace, name="other", phase="evaluation")
    args = arguments(workspace, [first])
    assert node0.watch_cycle(args, {}) is True
    assert node0.watch_cycle(arguments(workspace, [other]), {}) is True
    ack = args.ack_root / "first.model.json"
    first_ack = json.loads(ack.read_text())
    other_ack = json.loads((args.ack_root / "other.model.json").read_text())
    for key in ("destination", "snapshot_identity", "bank_receipt_sha256"):
        first_ack[key] = other_ack[key]
    write_json(ack, first_ack)
    assert node0.watch_cycle(args, {}) is False
    assert "acknowledgment" in capsys.readouterr().err


def test_cached_banking_cannot_finish_with_missing_or_changed_acknowledgments(
    workspace, node0, capsys
):
    run = fixture_run(workspace, phase="evaluation")
    args = arguments(workspace, [run])
    banked = {}
    assert node0.watch_cycle(args, banked) is True
    final_ack = args.ack_root / f"{run.name}.evaluation.json"
    saved = final_ack.read_bytes()
    final_ack.unlink()
    assert node0.watch_cycle(args, banked) is False
    final_ack.write_bytes(saved + b" ")
    assert node0.watch_cycle(args, banked) is False
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("host,rank", [("node-1", "1"), ("node-2", "2"), ("node-0", "1")])
def test_only_node0_may_watch_or_write(workspace, monkeypatch, host, rank):
    module = helper()
    args = arguments(workspace, [workspace / "missing"])
    monkeypatch.setattr(module.socket, "gethostname", lambda: host)
    monkeypatch.setenv("NODE_RANK", rank)
    with pytest.raises((ValueError, RuntimeError), match="node-0"):
        module.watch_cycle(args, {})
    assert not args.destination.exists() and not args.staging.exists()


def test_cli_exposes_snapshot_and_node0_watch_without_training_commands():
    helper()
    for command, expected in [
        ([], ("snapshot", "watch")),
        (["snapshot"], ("--run", "--staging")),
        (["watch"], ("--runs", "--lock-file", "--writer-host", "--ack-root")),
    ]:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "bank_baselines.py"), *command, "--help"],
            check=True, capture_output=True, text=True,
        )
        assert all(text in result.stdout for text in expected)
