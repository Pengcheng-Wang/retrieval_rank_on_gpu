import copy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_letter_p5_crash_consistency import completed_run, training_parts
from test_letter_p5_xnode import lease, workspace
from test_letter_p5_xnode_broker import broker


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))
import bank as durable
import p5_bank as results
import p5_checkpoint as checkpoints


def frozen_chain(workspace, monkeypatch, best_epoch=20):
    module = broker(workspace)
    xs = module.state
    root = xs.ROOT
    p5 = xs.P5
    run = p5 / "runs" / "example-cell"
    directory = xs.XROOT
    contract = {"dataset": "fixture-dataset", "same_scientific_contract": True}
    xs.write(run / "p5-run-config.json", contract)
    parts = training_parts()
    saved = {
        "epoch": 0, "global_step": 100, "phase": "train", "best_metric": -1.,
        "selected_artifact": None, "validation_history": [],
    }
    original, _ = checkpoints.save_checkpoint(run, *parts, saved, contract)
    current = lease()
    current["root"] = xs.checkpoint_header(original)
    current["contract_file_sha256"] = xs.sha(run / "p5-run-config.json")
    xs.write(run / "crossnode-attestation.json", {"lease_sha256": xs.digest(current)})
    ledger = directory / "ledger" / current["cell"] / current["lease_id"]
    headers, acks, envelopes = {}, {}, {}
    for epoch in range(1, 21):
        for phase in ("validate", "after-validation"):
            saved.update(
                global_step=100 + epoch * 10,
                epoch=epoch - 1 if phase == "validate" else epoch,
                phase="validate" if phase == "validate" else ("frozen" if epoch == 20 else "train"),
            )
            improved = phase == "after-validation" and epoch <= best_epoch
            if phase == "after-validation":
                score = epoch / 100 if improved else 0.
                if improved:
                    saved["best_metric"] = score
                saved["validation_history"].append({
                    "split": "valid", "users": 2, "primary": {"recall@10": score},
                })
                with torch.no_grad():
                    parts[0].weight.add_(.001)
            path, _ = checkpoints.save_checkpoint(run, *parts, saved, contract, selected=improved)
            selected = (
                xs.checkpoint_header(run / "states" / saved["selected_artifact"])
                if saved["selected_artifact"] else None
            )
            package = xs.publish_checkpoint(run, path, current, selected=selected)
            envelope = xs.verify_package(package)
            ack = xs.make_ack(package, current, {"files": xs.file_manifest(package)}, "fixture", "c" * 64)
            module.record_acceptance(ledger, current, envelope, ack)
            xs.write(run / ".xnode/acks" / (path.name + ".json"), ack)
            headers[envelope["generation"]] = xs.checkpoint_header(path)
            acks[envelope["generation"]], envelopes[envelope["generation"]] = ack, envelope
    head, accepted = module.accepted_chain(ledger, current)
    assert len(accepted) == 40 and head["phase"] == "frozen" and head["epoch"] == 20
    assert accepted[40]["envelope"]["selected"] == headers[best_epoch * 2]
    certificate = {
        "event_sha256": "c" * 64,
        "event": {"kind": "p5-xnode-ownership-v1", "status": "committed",
                  "owners": {current["cell"]: current}},
    }
    worker = module.Broker.__new__(module.Broker)
    worker.directory, worker.registry = directory, directory / "ownership"
    worker.context = SimpleNamespace(bank=results)
    worker.current = lambda: certificate
    copies = []

    def publish(source, destination):
        copies.append(str(destination))
        return durable.copy_artifact(source, destination)

    worker.copy_artifact = publish
    return SimpleNamespace(
        module=module, xs=xs, run=run, lease=current, contract=contract, worker=worker,
        head=head, accepted=accepted, headers=headers, acks=acks, envelopes=envelopes, copies=copies,
    )


def snapshot_for(chain, workspace, damage=None):
    xs, source = chain.xs, workspace / "result-source"
    completed_run(source)
    selected = copy.deepcopy(chain.envelopes[40]["selected"])
    latest = copy.deepcopy(chain.headers[40])
    contract = copy.deepcopy(chain.contract)
    if damage == "stale-selected":
        selected = copy.deepcopy(chain.headers[2])
    elif damage == "wrong-selected-artifact":
        selected["artifact"] = chain.headers[2]["artifact"]
    elif damage == "stale-latest":
        latest = copy.deepcopy(chain.headers[2])
    elif damage == "wrong-latest-seal":
        latest["seal_sha256"] = chain.headers[2]["seal_sha256"]
    elif damage == "wrong-contract":
        contract["same_scientific_contract"] = False
    xs.write(source / "p5-run-config.json", contract)
    xs.write(source / "latest.json", {
        "artifact": latest["artifact"], "seal_sha256": latest["seal_sha256"],
        "global_step": latest["step"], "phase": latest["phase"],
    })
    xs.write(source / "selected.json", {
        "artifact": selected["artifact"], "seal_sha256": selected["seal_sha256"],
    })
    frozen = xs.read(source / "frozen.json")
    frozen.update(selected=selected["artifact"], selected_seal_sha256=selected["seal_sha256"])
    xs.write(source / "frozen.json", frozen)
    for filename in ("test-result.json", "test-started.json"):
        value = xs.read(source / filename)
        value["selected_seal_sha256"] = selected["seal_sha256"]
        xs.write(source / filename, value)
    snapshot = results._build_result_snapshot_locked(source)
    local = chain.run / "result-artifacts" / snapshot.name
    shutil.copytree(snapshot, local)
    imported = chain.worker.directory / "results" / chain.lease["lease_id"] / snapshot.name
    shutil.copytree(snapshot, imported)
    ready = {
        "kind": "p5-xnode-results-ready-v1", "cell": chain.lease["cell"],
        "lease_sha256": xs.digest(chain.lease), "snapshot": str(local.relative_to(chain.run)),
        "result_seal_sha256": xs.sha(local / "p5-result-seal.json"), "head": chain.head,
    }
    xs.write(chain.run / ".xnode/results-ready.json", ready)
    results.verify_result_snapshot(local)
    return local, ready


DAMAGE = ("stale-selected", "wrong-selected-artifact", "stale-latest", "wrong-latest-seal", "wrong-contract")


@pytest.mark.parametrize("damage", DAMAGE)
def test_broker_rejects_internally_valid_result_not_bound_to_frozen_head(workspace, monkeypatch, damage):
    chain = frozen_chain(workspace, monkeypatch)
    snapshot, ready = snapshot_for(chain, workspace, damage)
    original_bytes = chain.xs.file_manifest(snapshot)
    with pytest.raises(ValueError, match="frozen|selected|contract"):
        chain.worker.remote_results(chain.lease, ready)
    assert chain.copies == []
    assert not (chain.run / "results-banked.json").exists()
    assert chain.xs.file_manifest(snapshot) == original_bytes
    assert not chain.xs.DURABLE.exists()


@pytest.mark.parametrize("damage", DAMAGE)
def test_worker_consumer_rejects_receipt_for_wrong_frozen_selection(workspace, monkeypatch, damage):
    chain = frozen_chain(workspace, monkeypatch)
    snapshot, ready = snapshot_for(chain, workspace, damage)
    xs = chain.xs
    receipt = {"files": xs.file_manifest(snapshot)}
    ack = {
        "kind": "p5-xnode-result-ack-v1", "status": "verified", "writer_host": xs.WRITER_HOST,
        "lease_id": chain.lease["lease_id"], "lease_sha256": xs.digest(chain.lease),
        "binding_version": 2, "frozen_head": chain.head, "frozen_checkpoint_ack": chain.acks[40],
        "result_seal_sha256": ready["result_seal_sha256"],
        "bank_receipt": receipt, "bank_receipt_sha256": xs.digest(receipt),
    }
    lease_path, message = workspace / "lease.json", workspace / "message.json"
    xs.write(lease_path, chain.lease)
    xs.write(message, ack)
    monkeypatch.setattr(chain.module.socket, "gethostname", lambda: xs.WORKER_HOST)
    with pytest.raises(ValueError, match="frozen|selected|contract"):
        chain.module.receive_message(lease_path, message)
    assert not (chain.run / "results-banked.json").exists()
    assert chain.copies == []


def test_producer_inventory_does_not_advertise_a_stale_selected_ancestor(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch)
    snapshot_for(chain, workspace, "stale-selected")
    path = workspace / "lease.json"
    chain.xs.write(path, chain.lease)
    monkeypatch.setattr(chain.module.socket, "gethostname", lambda: chain.xs.WORKER_HOST)
    with pytest.raises(ValueError, match="selected|frozen"):
        chain.module.remote_inventory(path)
    assert not (chain.run / "results-banked.json").exists()


def test_actual_frozen_selection_publishes_and_consumer_accepts(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch)
    snapshot, ready = snapshot_for(chain, workspace)
    ack = chain.worker.remote_results(chain.lease, ready)
    assert ack.get("binding_version") == 2
    assert ack.get("frozen_head") == chain.head
    assert ack.get("frozen_checkpoint_ack") == chain.acks[40]
    assert len(chain.copies) == 1
    assert chain.xs.validate_result_ack(ack, chain.lease, snapshot)
    lease_path, message = workspace / "lease.json", workspace / "message.json"
    chain.xs.write(lease_path, chain.lease)
    chain.xs.write(message, ack)
    monkeypatch.setattr(chain.module.socket, "gethostname", lambda: chain.xs.WORKER_HOST)
    chain.module.receive_message(lease_path, message)
    assert chain.xs.read(chain.run / "results-banked.json") == ack


def test_consumer_rejects_legacy_unbound_receipt_even_for_correct_predictions(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch)
    snapshot, ready = snapshot_for(chain, workspace)
    receipt = {"files": chain.xs.file_manifest(snapshot)}
    ack = {
        "kind": "p5-xnode-result-ack-v1", "status": "verified", "writer_host": chain.xs.WRITER_HOST,
        "lease_id": chain.lease["lease_id"], "lease_sha256": chain.xs.digest(chain.lease),
        "result_seal_sha256": ready["result_seal_sha256"],
        "bank_receipt": receipt, "bank_receipt_sha256": chain.xs.digest(receipt),
    }
    with pytest.raises(ValueError, match="frozen|binding"):
        chain.xs.validate_result_ack(ack, chain.lease, snapshot)


def test_earlier_winner_is_valid_when_the_actual_frozen_head_declares_it(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch, best_epoch=1)
    snapshot, ready = snapshot_for(chain, workspace)
    ack = chain.worker.remote_results(chain.lease, ready)
    assert ack["frozen_checkpoint_ack"]["envelope"]["selected"] == chain.headers[2]
    assert ack["frozen_head"]["generation"] == 40
    assert chain.xs.validate_result_ack(ack, chain.lease, snapshot)


def final_request_inventory(chain, requested_seals):
    xs = chain.xs
    chain.worker.config = {"input_manifest_sha256": "d" * 64}
    chain.worker.registration_anchor_sha256 = None
    registration = {
        "kind": "p5-xnode-registration-v1", "writer_host": xs.WRITER_HOST,
        "lease_sha256": xs.digest(chain.lease), "event_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
    }
    inventory = {
        "lease_sha256": xs.digest(chain.lease), "registration_sha256": xs.digest(registration),
        "packages": [], "acknowledged": {}, "guardian": None,
        "results_ready": None, "results_banked": False,
        "requests": [{"nonce": str(i), "lease_sha256": xs.digest(chain.lease),
                      "checkpoint_seal_sha256": seal} for i, seal in enumerate(requested_seals)],
    }
    sent = []
    chain.worker.rpc = lambda arguments: inventory
    chain.worker.send = lambda lease_, message: sent.append(message)
    return sent


def test_final_test_bank_request_cannot_authorize_an_unselected_ancestor(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch)
    sent = final_request_inventory(chain, [chain.headers[2]["seal_sha256"]])
    with pytest.raises(ValueError, match="frozen|selected"):
        chain.worker.remote(chain.lease)
    assert sent == []


def test_final_test_bank_requests_allow_exact_selected_and_latest_frozen_states(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch, best_epoch=1)
    sent = final_request_inventory(chain, [
        chain.headers[2]["seal_sha256"], chain.headers[40]["seal_sha256"],
    ])
    chain.worker.remote(chain.lease)
    assert [message["ack"]["checkpoint"] for message in sent] == [chain.headers[2], chain.headers[40]]


def test_result_marker_commit_holds_native_pointer_and_publication_locks(workspace, monkeypatch):
    chain = frozen_chain(workspace, monkeypatch)
    _, ready = snapshot_for(chain, workspace)
    ack = chain.worker.remote_results(chain.lease, ready)
    original_write = chain.xs.write
    committed = []

    def checked_write(path, value):
        if Path(path).name == "results-banked.json":
            for lock in (chain.run / ".p5-pointer.lock", chain.run / ".xnode/publish.lock"):
                with pytest.raises(OSError):
                    with chain.xs.locked(lock, nonblocking=True):
                        pass
            committed.append(value)
        original_write(path, value)

    lease_path, message = workspace / "lease.json", workspace / "message.json"
    original_write(lease_path, chain.lease)
    original_write(message, ack)
    monkeypatch.setattr(chain.module.socket, "gethostname", lambda: chain.xs.WORKER_HOST)
    monkeypatch.setattr(chain.xs, "write", checked_write)
    chain.module.receive_message(lease_path, message)
    assert committed == [ack]
