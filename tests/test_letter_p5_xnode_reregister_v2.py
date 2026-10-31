import copy
import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

from test_letter_p5_xnode_reregister import fixture as v1_fixture
from test_letter_p5_xnode_reregister import sha, write


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "p5_xnode_reregister_v2.py"
    assert path.exists(), "A deployable v2 UUID re-registration tool is required"
    spec = importlib.util.spec_from_file_location("p5_xnode_reregister_v2_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v2_fixture(tmp_path):
    input_manifest = tmp_path / "input-manifest.json"
    write(input_manifest, {"fixture": input_manifest.stem})
    v1, _, request, old_event, certificate = v1_fixture(
        tmp_path, input_manifest_sha256=sha(input_manifest),
    )
    old_bindings = json.loads(Path(request["old_bindings"]).read_text())
    for cell in request["cells"]:
        if old_bindings["cells"][cell]["role"] == "writer":
            root = old_event["owners"][cell]["root"]
            root.update(
                artifact=f"{cell.lower()}-root-90", seal_sha256=("1" * 64),
                step=90, model_sha256=("2" * 64), training_state_sha256=("3" * 64),
            )
    prior_sha = v1.event_sha(old_event)
    prior_receipt = {"files": {"ownership.json": {"sha256": prior_sha, "bytes": 1}}}
    certificate.update(
        event=old_event, event_sha256=prior_sha, bank_receipt=prior_receipt,
        bank_receipt_sha256=v1.digest(prior_receipt),
    )
    write(Path(request["registry"]) / "committed" / "0001.json", certificate)
    for cell in request["cells"]:
        if old_bindings["cells"][cell]["role"] == "worker":
            lease_id = old_event["owners"][cell]["lease_id"]
            record_path = (Path(request["registry"]).parent / "ledger" / cell / lease_id
                           / "accepted" / "g00000001.json")
            record = json.loads(record_path.read_text())
            record["ack"]["registry_event_sha256"] = prior_sha
            write(record_path, record)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        shutil.copyfile(SCRIPTS / name, runtime / name)
    maintenance_manifest = tmp_path / "maintenance-manifest.json"
    maintenance_proof = tmp_path / "maintenance-proof.json"
    base = tmp_path / "base"
    base.mkdir()
    for path in (maintenance_manifest, maintenance_proof):
        write(path, {"fixture": path.stem})
    template = {
        "kind": "p5-xnode-runtime-v1",
        "root": str(tmp_path), "p5_root": str(tmp_path / "p5"),
        "control_root": str(tmp_path / "control"),
        "durable_root": str(tmp_path.parent / (tmp_path.name + "-durable")),
        "python": str(Path(sys.executable).resolve()), "base": str(base),
        "input_manifest": str(input_manifest), "input_manifest_sha256": sha(input_manifest),
        "maintenance_manifest": str(maintenance_manifest),
        "maintenance_proof": str(maintenance_proof),
        "ownership_certificate": str(tmp_path / "obsolete-certificate.json"),
        "crossnode_proof": str(tmp_path / "obsolete-proof.json"),
        "files_sha256": {
            name: sha(runtime / name)
            for name in ("p5_xnode_control.py", "p5_xnode_state.py")
        },
    }
    template_path = runtime / "runtime-template.json"
    write(template_path, template)
    for cell in request["cells"]:
        if old_event["owners"][cell]["owner_node"] != request["old_bindings"]:
            pass
        assignment = json.loads(Path(request["old_bindings"]).read_text())["cells"][cell]
        if assignment["role"] == "writer":
            old = request["evidence"][cell]
            source = Path(old["package"]) / "checkpoint"
            durable = (tmp_path.parent / (tmp_path.name + "-durable") / "runs" / cell
                       / old_event["owners"][cell]["root"]["artifact"])
            durable.mkdir(parents=True)
            for member in source.iterdir():
                shutil.copyfile(member, durable / member.name)
            receipt = {"files": {
                member.name: {"sha256": sha(durable / member.name), "bytes": member.stat().st_size}
                for member in durable.iterdir() if member.is_file()
            }}
            write(durable / "bank-receipt.json", receipt)
            ack = {
                "status": "verified", "kind": "p5-full-state-v1", "artifact": durable.name,
                "seal_sha256": sha(source / "p5-seal.json"),
                "contract_sha256": old_event["owners"][cell]["root"]["contract_sha256"],
                "destination": str(durable), "bank_receipt_sha256": v1.digest(receipt),
                "writer_host": json.loads(Path(request["old_bindings"]).read_text())["writer_host"],
            }
            ack_path = tmp_path / "evidence" / cell / "bank-ack.json"
            write(ack_path, ack)
            request["evidence"][cell] = {
                "kind": "p5-uuid-reregistration-writer-root-v2", "cell": cell,
                "checkpoint": str(durable), "ack": str(ack_path),
                "contract": old["contract"],
            }
        else:
            request["evidence"][cell]["kind"] = "p5-uuid-reregistration-worker-root-v2"
    request.update(
        kind="p5-uuid-reregistration-request-v2",
        runtime_template=str(template_path),
    )
    request_path = tmp_path / "request-v2.json"
    write(request_path, request)
    return module(), v1, request_path, request, old_event, certificate


def prepare(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    return v2, request_path, request, old_event, certificate, v2.prepare_transition(request_path)


def test_prepare_emits_guardian_compatible_proof_runtime_and_registration(tmp_path):
    v2, _, request, _, _, plan = prepare(tmp_path)
    proof = plan["aggregate_proof"]
    assert set(proof) == {
        "kind", "status", "source_node", "destination_node",
        "runtime_manifest_sha256", "cells",
    }
    assert proof["kind"] == "p5-xnode-crossnode-proof-v1"
    assert proof["runtime_manifest_sha256"] == plan["runtime_manifest_sha256"]
    assert plan["event"]["crossnode_proof_sha256"] == v2.document_sha(proof)
    assert set(proof["cells"]) == {"worker-tid-small", "worker-tid-large"}
    assert set(plan["event"]["replacement_proofs_sha256"]) == set(request["cells"])
    transitioned = plan["event"]["transition_cells"]
    assert all(
        plan["event"]["owners"][cell]["control_manifest_sha256"]
        == plan["runtime_manifest_sha256"]
        for cell in transitioned
    )
    registration = v2.worker_registration(plan, "worker-tid-small")
    assert set(registration) == {
        "kind", "writer_host", "lease_sha256", "event_sha256",
        "input_manifest_sha256",
    }
    assert plan["event"]["sequence"] == 1


def test_prepare_uses_writer_bank_receipts_and_worker_xnode_ledgers(tmp_path):
    v2, _, _, _, _, plan = prepare(tmp_path)
    assert plan["cells"]["writer-cid"]["evidence_kind"] == "writer-bank"
    assert plan["cells"]["worker-tid-small"]["evidence_kind"] == "worker-xnode"
    assert "accepted_record_sha256" not in plan["cells"]["writer-cid"]
    assert "accepted_record_sha256" in plan["cells"]["worker-tid-small"]


def test_commit_publishes_canonical_state_and_activation_last(tmp_path, monkeypatch):
    v2, request_path, request, _, _, plan = prepare(tmp_path)
    bindings = plan["bindings"]
    monkeypatch.setattr(socket, "gethostname", lambda: bindings["writer_host"])
    result = v2.commit_transition(request_path, plan)
    xroot = Path(bindings["control_root"])
    assert Path(result["activation"]).samefile(xroot / "activation" / "last-published.json")
    activation = json.loads((xroot / "activation" / "last-published.json").read_text())
    assert activation["sequence"] == plan["event"]["sequence"]
    for relative, checksum in activation["files_sha256"].items():
        assert sha(xroot / relative) == checksum
    for cell in request["cells"]:
        assert (xroot / "leases" / f"{cell}.json").is_file()
        assert (xroot / "roots" / cell / "checkpoint" / "training-state.pt").is_file()
        assert (xroot / "roots" / cell / "p5-run-config.json").is_file()
        assert (xroot / "roots" / cell / "latest.json").is_file()
    for cell in ("worker-tid-small", "worker-tid-large"):
        transport = xroot / "worker-bootstrap" / cell / ".xnode"
        assert {path.name for path in transport.iterdir()} == {"registration.json"}


def test_interrupted_commit_is_not_activated(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    with pytest.raises(RuntimeError, match="activation"):
        v2.commit_transition(request_path, plan, fail_before_activation=True)
    with pytest.raises(ValueError, match="activated|activation"):
        v2.verify_activation(Path(plan["bindings"]["control_root"]),
                             plan["event"]["broker_release_sha256"])
    result = v2.activate_transition(request_path, plan)
    assert Path(result["activation"]).is_file()
    assert v2.verify_activation(
        Path(plan["bindings"]["control_root"]),
        plan["event"]["broker_release_sha256"],
    )["event_sha256"] == plan["event_sha256"]


def test_activation_rejects_stale_canonical_file(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    lease = xroot / "leases" / "worker-tid-small.json"
    lease.write_text(lease.read_text() + "\n")
    with pytest.raises(ValueError, match="activation|canonical|hash"):
        v2.verify_activation(xroot, plan["event"]["broker_release_sha256"])


def test_activation_recovery_cannot_bless_stale_staged_file(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    with pytest.raises(RuntimeError, match="activation"):
        v2.commit_transition(request_path, plan, fail_before_activation=True)
    xroot = Path(plan["bindings"]["control_root"])
    lease = xroot / "leases" / "worker-tid-small.json"
    lease.write_text(lease.read_text() + "\n")
    with pytest.raises(ValueError, match="canonical|staged|lease|hash"):
        v2.activate_transition(request_path, plan)


def test_broker_v3_authorizes_current_certificate_only_after_activation(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    broker.state.configure_bindings(
        Path(plan["bindings"]["control_root"]) / "current" / "bindings.json"
    )
    certificate = broker.activated_current(
        Path(plan["bindings"]["control_root"]),
        plan["event"]["broker_release_sha256"],
    )
    assert certificate["event_sha256"] == plan["event_sha256"]


def test_staged_runtime_preserves_guardian_registration_semantics(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    control_path = SCRIPTS / "p5_xnode_control.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_control_v2_test", control_path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    control.state._BINDINGS = None
    control.state.configure_bindings(xroot / "current" / "bindings.json")
    config = control.verify_runtime(xroot / "current" / "runtime-manifest.json")
    assert sha(xroot / "current" / "runtime-manifest.json") == plan["runtime_manifest_sha256"]
    assert config["files_sha256"] == plan["runtime_manifest"]["files_sha256"]
    cell = "worker-tid-small"
    run = Path(plan["bindings"]["p5_root"]) / "runs" / cell
    (run / ".xnode").mkdir(parents=True)
    shutil.copyfile(
        xroot / "worker-bootstrap" / cell / ".xnode" / "registration.json",
        run / ".xnode" / "registration.json",
    )
    lease = json.loads((xroot / "leases" / f"{cell}.json").read_text())
    registration = json.loads((run / ".xnode" / "registration.json").read_text())
    assert registration["lease_sha256"] == control.state.digest(lease)
    assert registration["event_sha256"] == json.loads(
        (xroot / "current" / "ownership-certificate.json").read_text()
    )["event_sha256"]
    assert registration["input_manifest_sha256"] == config["input_manifest_sha256"]
    assert registration["writer_host"] == control.state.WRITER_HOST
    assert control.registration(run, lease, config) == v2.worker_registration(plan, cell)


def test_broker_v3_rejects_incomplete_activation_inventory(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    activation_path = xroot / "activation" / "last-published.json"
    activation = json.loads(activation_path.read_text())
    activation["files_sha256"].pop("leases/worker-tid-small.json")
    write(activation_path, activation)
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_inventory_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    broker.state._BINDINGS = None
    broker.state.configure_bindings(xroot / "current" / "bindings.json")
    with pytest.raises(ValueError, match="activation|inventory|lease"):
        broker.activated_current(xroot, plan["event"]["broker_release_sha256"])


def test_broker_v3_release_rejects_certificate_hash_cycle(tmp_path):
    v2, _, _, _, _, plan = prepare(tmp_path)
    release = {
        "kind": "p5-xnode-broker-v3",
        "files_sha256": {
            "p5_xnode_broker.py": "a" * 64,
            "p5_xnode_state.py": "b" * 64,
            "p5_xnode_adopt.py": "c" * 64,
        },
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "ownership_certificate_sha256": "d" * 64,
    }
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_release_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    with pytest.raises(ValueError, match="cycle|certificate|anchor"):
        broker.validate_broker_v3_release(release)
