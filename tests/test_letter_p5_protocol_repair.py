import copy
import json
import shutil
from pathlib import Path

import pytest

from test_letter_p5_xnode import package_fixture, protocol, workspace
from test_letter_p5_xnode_broker import broker


def reseal(module, package):
    package = Path(package)
    module.write(package / "package-seal.json", {
        "kind": "p5-xnode-package-v1",
        "status": "sealed",
        "files": module.file_manifest(package, {"package-seal.json"}),
    })


def write_plain(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def test_broker_v4_release_has_acyclic_source_only_authority():
    module = broker()
    release = {
        "kind": "p5-xnode-broker-v4",
        "files_sha256": {
            "p5_xnode_broker.py": "a" * 64,
            "p5_xnode_state.py": "b" * 64,
            "p5_xnode_adopt.py": "c" * 64,
        },
        "runtime_manifest_sha256": "d" * 64,
    }

    assert module.validate_broker_v4_release(release) == release

    release["selected_lineage_readmission_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="cycle|release|source"):
        module.validate_broker_v4_release(release)


def test_broker_v4_serves_only_event_active_cells(workspace):
    module, xs = broker(workspace), protocol(workspace)
    owners = {
        cell: {"cell": cell, "owner_node": xs.WRITER_HOST}
        for cell in xs.ALL_CELLS
    }
    active = [sorted(xs.ALL_CELLS)[0]]
    certificate = {
        "event": {
            "owners": owners,
            "active_cells": active,
            "transition_cells": active,
            "selected_lineage_deployment": {
                "cells": active,
                "sha256": "a" * 64,
            },
        },
    }

    assert [cell for cell, _ in module.active_owners(certificate)] == active

    certificate["event"]["active_cells"] = sorted(xs.ALL_CELLS)
    with pytest.raises(ValueError, match="active|cell|recovery"):
        module.active_owners(certificate)


def selected_readmission_fixture(workspace):
    module, xs = broker(workspace), protocol(workspace)
    run, current, selected_path = package_fixture(xs, workspace)
    package = xs.publish_checkpoint(run, selected_path, current)
    envelope = xs.read(package / "envelope.json")
    selected = copy.deepcopy(envelope["checkpoint"])
    source_event = "1" * 64
    deployment_sha = "2" * 64
    attestation = {
        "kind": "p5-selected-lineage-readmission-attestation-v1",
        "historically_accepted": False,
        "source_event_sha256": source_event,
        "selected_lineage_deployment_sha256": deployment_sha,
        "lease_sha256": xs.digest(current),
        "root_checkpoint_seal_sha256": current["root"]["seal_sha256"],
        "selected_checkpoint_seal_sha256": selected["seal_sha256"],
        "scientific_contract_rewritten": False,
    }
    xs.write(package / "execution-attestation.json", attestation)
    readmission_envelope = {
        "kind": "p5-xnode-selected-readmission-v1",
        "purpose": "production",
        **{
            key: current[key]
            for key in ("cell", "owner_node", "gpu", "lease_id", "ownership_epoch")
        },
        "lease_sha256": xs.digest(current),
        "wire_generation": 0,
        "root_seal_sha256": current["root"]["seal_sha256"],
        "checkpoint": selected,
        "selected": selected,
        "execution_attestation_sha256": xs.sha(
            package / "execution-attestation.json"
        ),
    }
    xs.write(package / "envelope.json", readmission_envelope)
    reseal(xs, package)
    destination = xs.DURABLE / "selected-readmission" / current["cell"]
    shutil.copytree(package, destination)
    receipt = {"files": xs.file_manifest(destination)}
    write_plain(destination / "bank-receipt.json", receipt)
    authority = {
        "kind": "p5-selected-lineage-readmission-v1",
        "status": "verified",
        "parent_event_sha256": source_event,
        "selected_lineage_deployment_sha256": deployment_sha,
        "cells": {
            current["cell"]: {
                "historically_accepted": False,
                "lease_sha256": xs.digest(current),
                "root_seal_sha256": current["root"]["seal_sha256"],
                "selected": selected,
                "package": str(package.resolve()),
                "package_seal_sha256": xs.sha(package / "package-seal.json"),
                "destination": str(destination.resolve()),
                "bank_receipt_sha256": xs.sha(
                    destination / "bank-receipt.json"
                ),
            },
        },
    }
    xs.write(
        xs.XROOT / "current" / "selected-lineage-readmission.json",
        authority,
    )
    certificate = {
        "event_sha256": "3" * 64,
        "event": {
            "previous_event_sha256": source_event,
            "owners": {current["cell"]: current},
            "active_cells": [current["cell"]],
            "transition_cells": [current["cell"]],
            "selected_lineage_deployment": {
                "cells": [current["cell"]],
                "sha256": deployment_sha,
            },
        },
    }
    canonical_run = xs.P5 / "runs" / current["cell"]
    shutil.copytree(run, canonical_run)
    return module, xs, canonical_run, current, selected, package, certificate


def test_broker_issues_explicit_selected_readmission_ack(workspace):
    module, xs, _, current, selected, _, certificate = selected_readmission_fixture(
        workspace
    )

    ack = module.selected_readmission_ack(
        xs.XROOT, certificate, current, selected,
    )

    assert ack["kind"] == "p5-xnode-selected-readmission-ack-v1"
    assert ack["readmission"]["historically_accepted"] is False
    assert ack["readmission"]["wire_generation"] == 0
    assert xs.validate_ack(ack, current, selected)
    assert not (xs.XROOT / "ledger" / current["cell"]).exists()


def test_selected_readmission_rejects_parent_or_historical_claim(workspace):
    module, xs, _, current, selected, package, certificate = (
        selected_readmission_fixture(workspace)
    )
    envelope = xs.read(package / "envelope.json")
    envelope["parent_seal_sha256"] = current["root"]["seal_sha256"]
    xs.write(package / "envelope.json", envelope)
    reseal(xs, package)
    authority_path = xs.XROOT / "current" / "selected-lineage-readmission.json"
    authority = xs.read(authority_path)
    authority["cells"][current["cell"]]["package_seal_sha256"] = xs.sha(
        package / "package-seal.json"
    )
    xs.write(authority_path, authority)

    with pytest.raises(ValueError, match="readmission|parent|ancestry"):
        module.selected_readmission_ack(xs.XROOT, certificate, current, selected)


def test_worker_accepts_event_bound_selected_readmission_response(
    workspace, monkeypatch,
):
    module, xs, run, current, selected, _, certificate = (
        selected_readmission_fixture(workspace)
    )
    ack = module.selected_readmission_ack(
        xs.XROOT, certificate, current, selected,
    )
    nonce = "f" * 32
    xs.write(run / ".xnode" / "requests" / f"{nonce}.json", {
        "kind": "p5-xnode-checkpoint-request-v1",
        "nonce": nonce,
        "cell": current["cell"],
        "lease_id": current["lease_id"],
        "lease_sha256": xs.digest(current),
        "checkpoint_seal_sha256": selected["seal_sha256"],
    })
    lease_path = workspace / "lease.json"
    message_path = workspace / "message.json"
    xs.write(lease_path, current)
    xs.write(message_path, {
        "kind": "p5-xnode-checkpoint-response-v1",
        "nonce": nonce,
        "lease_sha256": xs.digest(current),
        "ack": ack,
        "registry_event_sha256": certificate["event_sha256"],
    })
    monkeypatch.setattr(module.socket, "gethostname", lambda: xs.WORKER_HOST)

    module.receive_message(lease_path, message_path)

    assert xs.read(run / ".xnode" / "responses" / f"{nonce}.json")["ack"] == ack


def test_broker_v4_activation_binds_readmission_and_active_inventory(
    workspace, monkeypatch,
):
    module, xs = broker(workspace), protocol(workspace)
    active = ["example-cell"]
    owners = {}
    runtime = {"kind": "fixture-runtime"}
    xs.write(xs.XROOT / "current" / "runtime-manifest.json", runtime)
    runtime_sha = xs.sha(xs.XROOT / "current" / "runtime-manifest.json")
    for cell in xs.ALL_CELLS:
        owners[cell] = {
            "cell": cell,
            "owner_node": xs.WORKER_HOST if cell == "example-cell" else xs.WRITER_HOST,
            "control_manifest_sha256": runtime_sha,
            "root": {
                "artifact": f"{cell}-root",
                "seal_sha256": "a" * 64,
            },
        }
    proof = {"runtime_manifest_sha256": runtime_sha}
    xs.write(xs.XROOT / "current" / "crossnode-proof.json", proof)
    xs.write(xs.XROOT / "current" / "bindings.json", {"fixture": True})
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        path = xs.XROOT / "current" / name
        path.write_bytes(name.encode())
    readmission = {
        "kind": "p5-selected-lineage-readmission-v1",
        "status": "verified",
        "cells": {},
    }
    xs.write(
        xs.XROOT / "current" / "selected-lineage-readmission.json",
        readmission,
    )
    event = {
        "kind": "p5-xnode-ownership-v1",
        "status": "committed",
        "sequence": 5,
        "previous_event_sha256": "1" * 64,
        "transition_kind": "p5-protocol-repair-v1",
        "transition_cells": active,
        "active_cells": active,
        "owners": owners,
        "broker_release_sha256": "2" * 64,
        "crossnode_proof_sha256": xs.sha(
            xs.XROOT / "current" / "crossnode-proof.json"
        ),
        "selected_lineage_readmission_sha256": xs.sha(
            xs.XROOT / "current" / "selected-lineage-readmission.json"
        ),
        "selected_lineage_deployment": {
            "sha256": "3" * 64,
            "cells": active,
        },
        "parent_activation_sha256": "4" * 64,
    }
    certificate = {
        "event": event,
        "event_sha256": module.event_sha(event),
    }
    xs.write(xs.XROOT / "current" / "ownership-certificate.json", certificate)
    expected = {
        "current/ownership-certificate.json",
        "current/runtime-manifest.json",
        "current/crossnode-proof.json",
        "current/bindings.json",
        "current/p5_xnode_control.py",
        "current/p5_xnode_state.py",
        "current/selected-lineage-readmission.json",
    }
    for cell in active:
        xs.write(xs.XROOT / "leases" / f"{cell}.json", owners[cell])
        root = xs.XROOT / "roots" / cell
        for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
            path = root / "checkpoint" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((cell + name).encode())
        xs.write(root / "p5-run-config.json", {"cell": cell})
        xs.write(root / "latest.json", {"artifact": owners[cell]["root"]["artifact"]})
        xs.write(
            xs.XROOT / "worker-bootstrap" / cell / ".xnode" / "registration.json",
            {"cell": cell},
        )
        expected.update({
            f"leases/{cell}.json",
            f"roots/{cell}/checkpoint/model.safetensors",
            f"roots/{cell}/checkpoint/training-state.pt",
            f"roots/{cell}/checkpoint/p5-seal.json",
            f"roots/{cell}/p5-run-config.json",
            f"roots/{cell}/latest.json",
            f"worker-bootstrap/{cell}/.xnode/registration.json",
        })
    xs.write(xs.XROOT / "activation" / "last-published.json", {
        "kind": "p5-xnode-activation-v2",
        "status": "activated",
        "sequence": event["sequence"],
        "event_sha256": certificate["event_sha256"],
        "broker_release_sha256": event["broker_release_sha256"],
        "parent_activation_sha256": event["parent_activation_sha256"],
        "files_sha256": {
            relative: xs.sha(xs.XROOT / relative)
            for relative in sorted(expected)
        },
    })
    monkeypatch.setattr(module, "load_registry", lambda _: certificate)

    assert module.activated_current(
        xs.XROOT, event["broker_release_sha256"]
    ) == certificate

    xs.write(
        xs.XROOT / "current" / "selected-lineage-readmission.json",
        {**readmission, "status": "tampered"},
    )
    with pytest.raises(ValueError, match="activation|canonical|readmission"):
        module.activated_current(xs.XROOT, event["broker_release_sha256"])


def test_broker_readmission_is_used_only_when_selection_is_not_accepted(
    monkeypatch,
):
    module = broker()
    selected = {"seal_sha256": "selected"}
    accepted = {"checkpoint": selected}
    recovered = {"checkpoint": selected, "kind": "readmission"}
    monkeypatch.setattr(module, "find_accepted_ack", lambda *_: accepted)
    monkeypatch.setattr(module, "selected_readmission_ack", lambda *_: recovered)
    assert module.selection_ack(
        Path("control"), {}, Path("ledger"), {}, selected,
    ) is accepted

    monkeypatch.setattr(
        module,
        "find_accepted_ack",
        lambda *_: (_ for _ in ()).throw(ValueError("missing")),
    )
    assert module.selection_ack(
        Path("control"), {}, Path("ledger"), {}, selected,
    ) is recovered


def test_final_test_readmission_requires_exact_frozen_selection(monkeypatch):
    module = broker()
    selected = {"artifact": "best", "seal_sha256": "selected"}
    recovered = {"checkpoint": selected, "kind": "readmission"}
    frozen = {
        "checkpoint": {"artifact": "latest", "seal_sha256": "latest"},
        "selected": selected,
    }
    monkeypatch.setattr(
        module,
        "find_accepted_ack",
        lambda *_: (_ for _ in ()).throw(ValueError("missing")),
    )
    monkeypatch.setattr(module, "selected_readmission_ack", lambda *_: recovered)
    request = {"checkpoint_seal_sha256": selected["seal_sha256"]}
    assert module.final_test_ack(
        Path("control"), {}, Path("ledger"), {}, request, frozen,
    ) is recovered

    request["checkpoint_seal_sha256"] = "foreign"
    assert module.final_test_ack(
        Path("control"), {}, Path("ledger"), {}, request, frozen,
    ) is None
