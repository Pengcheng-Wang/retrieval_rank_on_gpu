import copy
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))
import p5_bank as bank
import p5_checkpoint as checkpoint
from p5_source import digest_json, read_json, sha256, write_json


class Crash(RuntimeError):
    pass


@pytest.fixture
def workspace():
    path = ROOT / ".local" / "letter" / "p5-crash-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def training_parts():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    sampler = checkpoint.IndexSampler(10, 42)
    return model, optimizer, scheduler, sampler


def selection_state(epoch, score, selected=None):
    return {
        "epoch": epoch, "global_step": epoch * 10,
        "phase": "frozen" if epoch == 20 else "train",
        "best_metric": score, "selected_artifact": selected,
        "validation_history": [
            {"split": "valid", "users": 2, "primary": {"recall@10": score if i == epoch - 1 else score / 2}}
            for i in range(epoch)
        ],
    }


def test_final_epoch_crash_after_latest_recovers_selected_from_sealed_state(workspace, monkeypatch):
    parts = training_parts()
    contract = {"test": "same-scientific-contract"}
    first_state = selection_state(1, .1)
    first, _ = checkpoint.save_checkpoint(workspace, *parts, first_state, contract, selected=True)
    original_write = checkpoint.write_json

    def crash_before_selected(path, value):
        if Path(path).name == "selected.json":
            raise Crash("latest committed; selected publication did not happen")
        original_write(path, value)

    monkeypatch.setattr(checkpoint, "write_json", crash_before_selected)
    final_state = selection_state(20, .2, first.name)
    with pytest.raises(Crash):
        checkpoint.save_checkpoint(workspace, *parts, final_state, contract, selected=True)
    monkeypatch.setattr(checkpoint, "write_json", original_write)
    latest = checkpoint.pointer_path(workspace)
    restored = checkpoint.restore_checkpoint(latest, *parts, contract)
    assert restored["epoch"] == 20 and restored["best_metric"] == .2
    assert checkpoint.pointer_path(workspace, "selected").name == restored["selected_artifact"]
    assert latest != first
    assert (first / "p5-seal.json").exists()


@pytest.mark.parametrize("filename,when,new_committed", [
    ("latest.json", "before", False), ("latest.json", "after", True),
    ("selected.json", "before", True), ("selected.json", "after", True),
])
def test_every_checkpoint_pointer_publication_boundary(workspace, monkeypatch, filename, when, new_committed):
    parts = training_parts()
    contract = {"test": "pointer-boundaries"}
    first, _ = checkpoint.save_checkpoint(workspace, *parts, selection_state(1, .1), contract, selected=True)
    original_write = checkpoint.write_json

    def interrupted(path, value):
        if Path(path).name == filename and when == "before":
            raise Crash("before atomic pointer publication")
        original_write(path, value)
        if Path(path).name == filename and when == "after":
            raise Crash("after atomic pointer publication")

    monkeypatch.setattr(checkpoint, "write_json", interrupted)
    with pytest.raises(Crash):
        checkpoint.save_checkpoint(workspace, *parts, selection_state(20, .2, first.name),
                                   contract, selected=True)
    monkeypatch.setattr(checkpoint, "write_json", original_write)
    latest = checkpoint.pointer_path(workspace)
    state = checkpoint.restore_checkpoint(latest, *parts, contract)
    expected = latest if new_committed else first
    assert checkpoint.pointer_path(workspace, "selected") == expected
    assert state["selected_artifact"] == expected.name
    assert state["epoch"] == (20 if new_committed else 1)


@pytest.mark.parametrize("damage", ["missing", "stale", "malformed", "unsafe"])
def test_no_later_improvement_recovers_best_not_latest(workspace, damage):
    parts = training_parts()
    contract = {"test": "no-later-improvement"}
    state = selection_state(1, .2)
    best, _ = checkpoint.save_checkpoint(workspace, *parts, state, contract, selected=True)
    best_pointer = read_json(workspace / "selected.json")
    state["epoch"], state["global_step"], state["phase"] = 20, 200, "frozen"
    state["validation_history"] += [
        {"split": "valid", "users": 2, "primary": {"recall@10": .1}} for _ in range(19)
    ]
    with torch.no_grad():
        parts[0].weight.add_(1.)
    latest, _ = checkpoint.save_checkpoint(workspace, *parts, state, contract, selected=False)
    immutable_before = {str(p): sha256(p) for p in (workspace / "states").rglob("*") if p.is_file()}
    if damage == "missing":
        (workspace / "selected.json").unlink()
    elif damage == "stale":
        write_json(workspace / "selected.json", read_json(workspace / "latest.json"))
    elif damage == "malformed":
        (workspace / "selected.json").write_text("{interrupted", encoding="utf-8")
    else:
        write_json(workspace / "selected.json", {"artifact": "../outside"})
    write_json(workspace / "validation-history.json", [])
    assert checkpoint.pointer_path(workspace, "selected") == best
    assert checkpoint.pointer_path(workspace) == latest
    assert read_json(workspace / "selected.json") == best_pointer
    assert len(read_json(workspace / "validation-history.json")) == 20
    assert immutable_before == {str(p): sha256(p) for p in (workspace / "states").rglob("*") if p.is_file()}


def test_conflicting_frozen_selection_fails_closed(workspace):
    parts = training_parts()
    contract = {"test": "frozen-conflict"}
    best, _ = checkpoint.save_checkpoint(workspace, *parts, selection_state(20, .2), contract, selected=True)
    write_json(workspace / "frozen.json", {
        "selected": "p5-state-wrong", "selected_seal_sha256": "0" * 64, "epochs": 20,
    })
    with pytest.raises(ValueError, match="Frozen generation"):
        checkpoint.pointer_path(workspace, "selected")
    assert read_json(workspace / "selected.json")["artifact"] == best.name


@pytest.mark.parametrize("when", ["before", "after"])
def test_selection_recovery_itself_can_be_interrupted(workspace, monkeypatch, when):
    parts = training_parts()
    best, _ = checkpoint.save_checkpoint(workspace, *parts, selection_state(20, .2), {}, selected=True)
    (workspace / "selected.json").unlink()
    original_replace = Path.replace

    def interrupted(source, target):
        match = Path(target) == workspace / "selected.json"
        if match and when == "before":
            raise Crash("before recovery pointer commit")
        result = original_replace(source, target)
        if match and when == "after":
            raise Crash("after recovery pointer commit")
        return result

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(Crash):
        checkpoint.pointer_path(workspace, "selected")
    monkeypatch.setattr(Path, "replace", original_replace)
    assert checkpoint.pointer_path(workspace, "selected") == best


def test_selection_recovery_does_not_consume_rng(workspace):
    import random
    import numpy as np
    parts = training_parts()
    checkpoint.save_checkpoint(workspace, *parts, selection_state(20, .2), {}, selected=True)
    (workspace / "selected.json").unlink()
    state = checkpoint.rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    checkpoint.restore_rng(state)
    checkpoint.pointer_path(workspace, "selected")
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_restore_rejects_an_older_generation(workspace):
    parts = training_parts()
    old, _ = checkpoint.save_checkpoint(workspace, *parts, selection_state(1, .1), {}, selected=True)
    checkpoint.save_checkpoint(workspace, *parts, selection_state(20, .2), {}, selected=True)
    with pytest.raises(ValueError, match="different generation"):
        checkpoint.restore_checkpoint(old, *parts, {})


def completed_run(run):
    run.mkdir(parents=True, exist_ok=True)
    rows = [
        {"original_user": str(i), "original_target_item": i, "target": "item_" + str(i),
         "predictions": ["item_" + str(i)] * 20, "scores": [-1.] * 20}
        for i in range(2)
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (run / "test-predictions.jsonl").write_text(text, encoding="utf-8")
    (run / "updates.jsonl").write_text('{"step": 20}\n', encoding="utf-8")
    marker = "a" * 64
    write_json(run / "test-result.json", {
        "status": "complete", "split": "test", "users": 2, "expected_users": 2,
        "beams_per_user": 20, "seen_item_exclusion": False,
        "predictions_sha256": sha256(run / "test-predictions.jsonl"),
        "selected_seal_sha256": marker, "exact_letter_recipe_claim": False,
    })
    write_json(run / "completed.json", {"status": "complete", "test_evaluations": 1})
    write_json(run / "test-started.json", {"selected_seal_sha256": marker, "started_after_bank_ack": True})
    write_json(run / "frozen.json", {
        "status": "frozen-before-test", "selected": "p5-state-selected", "epochs": 20,
        "selected_seal_sha256": marker,
    })
    write_json(run / "p5-run-config.json", {"unit_test": True})
    write_json(run / "selected.json", {"artifact": "p5-state-selected", "seal_sha256": marker})
    write_json(run / "latest.json", {"artifact": "p5-state-latest", "seal_sha256": "b" * 64})
    history = []
    for epoch in range(1, 21):
        path = run / "validation" / f"epoch-{epoch:02d}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        history.append({"split": "valid", "users": 2, "predictions_sha256": sha256(path)})
    write_json(run / "validation-history.json", history)
    return rows


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
def test_interrupted_result_copy_cannot_publish_a_one_file_snapshot(workspace, monkeypatch):
    run = workspace / "run"
    completed_run(run)
    original_copy = bank.shutil.copyfile

    def crash_on_predictions(source, destination, *args, **kwargs):
        if Path(source).name == "test-predictions.jsonl":
            raise Crash("result header copied; predictions not copied")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(bank.shutil, "copyfile", crash_on_predictions)
    with pytest.raises(Crash):
        bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert not (run / "results-banked.json").exists()
    monkeypatch.setattr(bank.shutil, "copyfile", original_copy)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    acknowledgement = read_json(run / "results-banked.json")
    destination = Path(acknowledgement["destination"])
    assert (destination / "test-predictions.jsonl").is_file()
    assert (destination / "completed.json").is_file()
    assert (destination / "p5-result-seal.json").is_file()
    seal = read_json(destination / "p5-result-seal.json")
    assert "test-predictions.jsonl" in seal["files_sha256"]


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
@pytest.mark.parametrize("damage", ["missing_seal", "missing_inventory", "wrong_hash", "traversal"])
def test_existing_malformed_result_snapshot_is_preserved_and_rebuilt(workspace, damage):
    run = workspace / "run"
    completed_run(run)
    result = read_json(run / "test-result.json")
    name = "p5-results-" + digest_json(result)[:16]
    snapshot = run / "result-artifacts" / name
    snapshot.mkdir(parents=True)
    shutil.copyfile(run / "test-result.json", snapshot / "test-result.json")
    if damage != "missing_seal":
        entries = {"test-result.json": sha256(snapshot / "test-result.json")}
        if damage == "wrong_hash":
            entries["test-result.json"] = "0" * 64
        if damage == "traversal":
            entries["../test-predictions.jsonl"] = sha256(run / "test-predictions.jsonl")
        write_json(snapshot / "p5-result-seal.json", {
            "kind": "p5-reference-results-v1", "status": "sealed", "files_sha256": entries,
        })
    original_header = (snapshot / "test-result.json").read_bytes()
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    destination = Path(read_json(run / "results-banked.json")["destination"])
    assert (destination / "test-predictions.jsonl").exists()
    preserved = list((run / "result-artifacts" / "quarantine").rglob("test-result.json"))
    assert preserved and any(p.read_bytes() == original_header for p in preserved)


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
@pytest.mark.parametrize("filename", [
    "completed.json", "validation-history.json", "validation/epoch-01.jsonl", "validation/epoch-20.jsonl",
])
@pytest.mark.parametrize("when", ["before", "after"])
def test_result_copy_interruptions_never_become_final_snapshots(workspace, monkeypatch, filename, when):
    run = workspace / "run"
    completed_run(run)
    original = bank.shutil.copyfile

    def interrupted(source, target, *args, **kwargs):
        match = Path(source) == run / filename
        if match and when == "before":
            raise Crash("before file copy")
        result = original(source, target, *args, **kwargs)
        if match and when == "after":
            raise Crash("after file copy")
        return result

    monkeypatch.setattr(bank.shutil, "copyfile", interrupted)
    with pytest.raises(Crash):
        bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert not (run / "results-banked.json").exists()
    assert not [p for p in (run / "result-artifacts").glob("p5-results-*") if ".partial-" not in p.name]
    monkeypatch.setattr(bank.shutil, "copyfile", original)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    destination = Path(read_json(run / "results-banked.json")["destination"])
    assert len(list((destination / "validation").glob("*.jsonl"))) == 20
    assert (run / "result-artifacts" / "quarantine").exists()


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
@pytest.mark.parametrize("when", ["before_seal", "after_seal", "before_rename", "after_rename", "before_ack"])
def test_snapshot_seal_rename_acknowledgement_crash_boundaries(workspace, monkeypatch, when):
    run = workspace / "run"
    completed_run(run)
    original_write, original_rename = bank.write_json, Path.rename

    def interrupted_write(path, value):
        seal = Path(path).name == "p5-result-seal.json"
        ack = Path(path).name == "results-banked.json"
        if (seal and when == "before_seal") or (ack and when == "before_ack"):
            raise Crash(when)
        original_write(path, value)
        if seal and when == "after_seal":
            raise Crash(when)

    def interrupted_rename(source, target):
        publishing = ".partial-" in source.name and Path(target).parent == run / "result-artifacts"
        if publishing and when == "before_rename":
            raise Crash(when)
        value = original_rename(source, target)
        if publishing and when == "after_rename":
            raise Crash(when)
        return value

    monkeypatch.setattr(bank, "write_json", interrupted_write)
    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(Crash):
        bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert not (run / "results-banked.json").exists()
    monkeypatch.setattr(bank, "write_json", original_write)
    monkeypatch.setattr(Path, "rename", original_rename)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert read_json(run / "results-banked.json")["status"] == "verified"


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
@pytest.mark.parametrize("damage", ["missing_inventory", "wrong_hash", "bad_status", "extra_file", "missing_validation"])
def test_complete_directory_with_a_malformed_seal_is_not_trusted(workspace, damage):
    run = workspace / "run"
    completed_run(run)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    snapshot = next(p for p in (run / "result-artifacts").glob("p5-results-*") if ".partial-" not in p.name)
    seal = read_json(snapshot / "p5-result-seal.json")
    if damage == "missing_inventory":
        del seal["files_sha256"]["test-predictions.jsonl"]
    elif damage == "wrong_hash":
        seal["files_sha256"]["test-predictions.jsonl"] = "0" * 64
    elif damage == "bad_status":
        seal["status"] = "building"
    elif damage == "extra_file":
        (snapshot / "unexpected.json").write_text("{}")
    else:
        (snapshot / "validation/epoch-20.jsonl").unlink()
    write_json(snapshot / "p5-result-seal.json", seal)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert (run / "result-artifacts" / "quarantine").exists()
    assert read_json(snapshot / "p5-result-seal.json")["files_sha256"]["test-predictions.jsonl"] == sha256(run / "test-predictions.jsonl")


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
@pytest.mark.parametrize("damage", ["duplicate_user", "missing_user", "missing_beam", "nonfinite_score", "wrong_model", "short_budget"])
def test_self_consistent_hashes_do_not_make_incomplete_results_genuine(workspace, damage):
    run = workspace / "run"
    rows = completed_run(run)
    if damage == "duplicate_user":
        rows[1]["original_user"] = rows[0]["original_user"]
    elif damage == "missing_user":
        rows.pop()
    elif damage == "missing_beam":
        rows[0]["predictions"].pop()
    elif damage == "nonfinite_score":
        rows[0]["scores"][0] = float("nan")
    elif damage == "wrong_model":
        started = read_json(run / "test-started.json")
        started["selected_seal_sha256"] = "0" * 64
        write_json(run / "test-started.json", started)
    else:
        frozen = read_json(run / "frozen.json")
        frozen["epochs"] = 1
        write_json(run / "frozen.json", frozen)
    (run / "test-predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = read_json(run / "test-result.json")
    result["predictions_sha256"] = sha256(run / "test-predictions.jsonl")
    write_json(run / "test-result.json", result)
    with pytest.raises(ValueError):
        bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    assert not (run / "results-banked.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="Native node0 publication uses POSIX flock")
def test_false_legacy_ack_and_incomplete_durable_copy_are_preserved(workspace):
    run = workspace / "run"
    completed_run(run)
    result = read_json(run / "test-result.json")
    wrong = workspace / "durable" / ("p5-results-" + digest_json(result)[:16])
    wrong.mkdir(parents=True)
    shutil.copyfile(run / "test-result.json", wrong / "test-result.json")
    old_ack = {"status": "verified", "destination": str(wrong), "bank_receipt_sha256": "0" * 64}
    write_json(run / "results-banked.json", old_ack)
    bank.bank_results(run, workspace / "durable", workspace / "bank.lock")
    fixed = read_json(run / "results-banked.json")
    assert Path(fixed["destination"]) != wrong
    assert (wrong / "test-result.json").exists()
    assert not (wrong / "test-predictions.jsonl").exists()
    preserved = list((run / "result-artifacts" / "quarantine").glob("*results-banked.json"))
    assert preserved and any(read_json(p) == old_ack for p in preserved)
