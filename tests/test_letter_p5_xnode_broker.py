import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from test_letter_p5_xnode import lease, package_fixture, protocol, workspace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def broker(workspace=None):
    path = SCRIPTS / "p5_xnode_broker.py"
    assert path.exists(), "The writer requires an owner/ancestry-verifying broker"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if workspace is not None:
        from test_letter_p5_xnode_configuration import configure
        configure(module.state, workspace)
    return module


def test_accepted_receipt_recovers_after_head_pointer_crash(workspace, monkeypatch):
    module, xs = broker(workspace), protocol(workspace)
    run, current, checkpoint = package_fixture(xs, workspace)
    package = xs.publish_checkpoint(run, checkpoint, current)
    envelope = xs.verify_package(package)
    ack = xs.make_ack(package, current, {"files": xs.file_manifest(package)}, "durable", "c" * 64)
    original = module.state.write

    def crash(path, value):
        if Path(path).name == "head.json":
            raise RuntimeError("head publication interrupted")
        original(path, value)

    monkeypatch.setattr(module.state, "write", crash)
    with pytest.raises(RuntimeError, match="head"):
        module.record_acceptance(workspace / "ledger", current, envelope, ack)
    monkeypatch.setattr(module.state, "write", original)
    head, accepted = module.accepted_chain(workspace / "ledger", current)
    assert head["generation"] == 1
    assert accepted[1]["ack"] == ack
    assert module.record_acceptance(workspace / "ledger", current, envelope, ack) == ack


def test_lost_historical_ack_can_be_replied_without_rolling_back_head(workspace):
    module, xs = broker(workspace), protocol(workspace)
    run, current, checkpoint = package_fixture(xs, workspace)
    package = xs.publish_checkpoint(run, checkpoint, current)
    first = xs.verify_package(package)
    ack = xs.make_ack(package, current, {"files": xs.file_manifest(package)}, "durable", "c" * 64)
    module.record_acceptance(workspace / "ledger", current, first, ack)
    from test_letter_p5_xnode import fake_checkpoint
    second_cp = fake_checkpoint(xs, run, 120, "333333333333")
    second_pkg = xs.publish_checkpoint(run, second_cp, current)
    second = xs.verify_package(second_pkg)
    ack2 = xs.make_ack(second_pkg, current, {"files": xs.file_manifest(second_pkg)}, "durable2", "c" * 64)
    module.record_acceptance(workspace / "ledger", current, second, ack2)
    assert module.find_accepted_ack(workspace / "ledger", current, first["checkpoint"]["seal_sha256"]) == ack
    assert module.accepted_chain(workspace / "ledger", current)[0]["generation"] == 2
    fork = copy.deepcopy(first)
    fork["checkpoint"]["seal_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="diverg"):
        module.record_acceptance(workspace / "ledger", current, fork, ack)


def test_pending_owner_handoff_never_falls_back_to_writer(workspace):
    module, xs = broker(workspace), protocol(workspace)
    xs.write(workspace / "transition-intent.json", {"status": "fencing", "destination": xs.WORKER_HOST})
    with pytest.raises(ValueError, match="committed|pending|fenced"):
        module.load_registry(workspace)


def test_broker_accepts_receipt_valid_sequence_two_registry_root(workspace):
    module, xs = broker(workspace), protocol(workspace)
    current = lease()
    event2 = {
        "kind": "p5-xnode-ownership-v1",
        "status": "committed",
        "sequence": 2,
        "previous_event_sha256": None,
        "owners": {
            cell: ({**current, "cell": cell} if cell != current["cell"] else current)
            for cell in xs.ALL_CELLS
        },
    }

    def cert(event):
        event_sha256 = module.event_sha(event)
        receipt = {
            "files": {
                "ownership.json": {"sha256": event_sha256, "bytes": 1},
            },
        }
        return {
            "event": event,
            "event_sha256": event_sha256,
            "writer_host": xs.WRITER_HOST,
            "bank_receipt": receipt,
            "bank_receipt_sha256": xs.digest(receipt),
        }

    certificate2 = cert(event2)
    event3 = {
        **event2,
        "sequence": 3,
        "previous_event_sha256": certificate2["event_sha256"],
    }
    certificate3 = cert(event3)
    xs.write(workspace / "committed" / "00000002.json", certificate2)
    xs.write(workspace / "committed" / "00000003.json", certificate3)
    assert module.load_registry(workspace)["event_sha256"] == certificate3["event_sha256"]

    event3["sequence"] = 4
    xs.write(workspace / "committed" / "00000003.json", cert(event3))
    with pytest.raises(ValueError, match="ancestry|receipt"):
        module.load_registry(workspace)


def compatibility_fixture(module, workspace):
    base = workspace / "runtime"
    base.mkdir()
    before = (
        "import json\nVALUE = 42\n"
        "def publish_checkpoint(value):\n    return value + VALUE\n"
        "def validate_result_ack(value):\n    return True\n"
    )
    after = before.replace("return True", "return bool(value)")
    after += (
        "def validate_result_binding(value):\n    return bool(value)\n"
        "def local_result_context(value):\n    return value\n"
    )
    (base / "p5_xnode_state.py").write_text(before)
    (base / "p5_xnode_control.py").write_text("UNCHANGED_CONTROL = True\n")
    updated = workspace / "new-state.py"
    updated.write_text(after)
    manifest = base / "runtime.json"
    module.state.write(manifest, {
        "kind": "p5-xnode-runtime-v1",
        "files_sha256": {name: module.state.sha(base / name) for name in
                         ("p5_xnode_state.py", "p5_xnode_control.py")},
    })
    return manifest, updated


def test_result_only_compatibility_leaves_every_checkpoint_statement_unchanged(workspace):
    module = broker(workspace)
    manifest, updated = compatibility_fixture(module, workspace)
    assert hasattr(module, "protocol_compatibility"), "Versioned result-only compatibility proof is required"
    proof = module.protocol_compatibility(manifest, updated)
    assert proof["runtime_manifest_sha256"] == module.state.sha(manifest)
    assert proof["updated_protocol_sha256"] == module.state.sha(updated)
    assert proof["checkpoint_protocol_unchanged"] is True
    assert proof["learner_control_unchanged"] is True
    assert proof["changed_existing_functions"] == ["validate_result_ack"]


@pytest.mark.parametrize("damage", ["checkpoint", "global", "new-helper", "old-runtime"])
def test_result_compatibility_rejects_changes_outside_named_result_functions(workspace, damage):
    module = broker(workspace)
    manifest, updated = compatibility_fixture(module, workspace)
    text = updated.read_text()
    if damage == "checkpoint":
        updated.write_text(text.replace("return value + VALUE", "return value - VALUE"))
    elif damage == "global":
        updated.write_text(text.replace("VALUE = 42", "VALUE = 43"))
    elif damage == "new-helper":
        updated.write_text(text + "\ndef altered_training(value):\n    return value\n")
    else:
        (manifest.parent / "p5_xnode_control.py").write_text("UNCHANGED_CONTROL = False\n")
    assert hasattr(module, "protocol_compatibility"), "Unverified runtime compatibility must not be admitted"
    with pytest.raises(ValueError, match="compatibility|unchanged|source"):
        module.protocol_compatibility(manifest, updated)


def registration_fixture(module, workspace):
    from test_letter_p5_xnode import lease
    current = lease()
    original = {
        "kind": "p5-xnode-ownership-v1", "status": "committed",
        "owners": {current["cell"]: current}, "broker_release_sha256": "a" * 64,
    }
    original_sha = module.event_sha(original)
    receipt = {"files": {"ownership.json": {"sha256": original_sha, "bytes": 1}}}
    certificate = {
        "event": original, "event_sha256": original_sha, "writer_host": module.state.WRITER_HOST,
        "bank_receipt": receipt, "bank_receipt_sha256": module.state.digest(receipt),
    }
    path = workspace / "original-certificate.json"
    module.state.write(path, certificate)
    new_event = copy.deepcopy(original)
    new_event.update(broker_release_sha256="b" * 64, previous_event_sha256=original_sha)
    new_certificate = {"event": new_event, "event_sha256": module.event_sha(new_event)}
    config = {"ownership_certificate": str(path), "input_manifest_sha256": "c" * 64}
    return current, config, new_certificate, certificate, module.state.sha(path)


def test_control_upgrade_preserves_immutable_worker_registration_anchor(workspace):
    module = broker(workspace)
    current, config, live, original, checksum = registration_fixture(module, workspace)
    assert hasattr(module, "registration_message"), "Unrestarted workers need their original registration anchor"
    message = module.registration_message(config, live, current, checksum)
    assert message["event_sha256"] == original["event_sha256"] != live["event_sha256"]
    assert message["lease_sha256"] == module.state.digest(current)


@pytest.mark.parametrize("damage", ["current-owner", "certificate-bytes"])
def test_legacy_registration_anchor_does_not_authorize_changed_ownership(workspace, damage):
    module = broker(workspace)
    current, config, live, original, checksum = registration_fixture(module, workspace)
    if damage == "current-owner":
        live["event"]["owners"][current["cell"]]["lease_id"] = "different-owner"
    else:
        with Path(config["ownership_certificate"]).open("a") as stream:
            stream.write("\n")
    assert hasattr(module, "registration_message"), "Legacy registration requires an exact owner/anchor guard"
    with pytest.raises(ValueError, match="owner|anchor|certificate"):
        module.registration_message(config, live, current, checksum)
