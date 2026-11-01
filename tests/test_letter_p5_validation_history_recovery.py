import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
from p5_source import digest_json
import p5_xnode_reregister as xreg


def module():
    path = SCRIPTS / "validation_history_recovery.py"
    assert path.exists(), "A versioned validation-history recovery tool is required"
    spec = importlib.util.spec_from_file_location(
        "validation_history_recovery_test", path,
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, expected_bytes=None):
    if expected_bytes is None:
        rows = [
            json.dumps({
                "original_user": user,
                "original_target_item": 1,
                "target": "1",
                "predictions": [str(i) for i in range(20)],
                "scores": [float(i) for i in range(20)],
            }, sort_keys=True)
            for user in ("u1", "u2")
        ]
        expected_bytes = ("\n".join(rows) + "\n").encode()
    run = tmp_path / "p5" / "runs" / "fixture-cell"
    contract_path = tmp_path / "contract.json"
    write(contract_path, {"fixture": True})
    expected = {
        "split": "valid",
        "users": 2,
        "expected_users": 2,
        "beams_per_user": 20,
        "seen_item_exclusion": False,
        "duplicate_canonical_beams": 0,
        "resumed_frozen_test_chunks": 0,
        "primary": {"recall@10": 0.5},
        "item_diagnostic": {"recall@10": 0.5},
        "predictions_sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }
    write(run / "validation-history.json", [expected for _ in range(20)])
    write(run / "latest.json", {"artifact": "latest"})
    write(run / "selected.json", {"artifact": "selected"})
    write(run / "frozen.json", {"status": "frozen-before-test"})
    write(run / "test-started.json", {"started": True})
    write(run / "test-result.json", {"status": "complete"})
    write(run / "completed.json", {"status": "complete"})
    write(run / "p5-run-config.json", {"fixture": True})
    write(run / "maintenance-attestation.json", {
        "kind": "fixture-maintenance",
    })
    (run / "updates.jsonl").write_text('{"fixture":true}\n')
    request = {
        "kind": "p5-validation-history-recovery-request-v1",
        "cell": "fixture-cell",
        "source_event_sha256": "b" * 64,
        "lease_sha256": "c" * 64,
        "root_seal_sha256": "d" * 64,
        "runtime_identity_sha256": "e" * 64,
        "input_identity_sha256": "f" * 64,
        "binding": {
            "host": socket.gethostname(),
            "gpu": 4,
            "gpu_uuid": "GPU-fixture",
        },
        "run": str(run),
        "contract": str(contract_path),
        "runtime_request": str(tmp_path / "runtime-request.json"),
        "output": str(tmp_path / "recovery"),
        "checkpoint_certificate": str(tmp_path / "checkpoint-certificate.json"),
        "ownership_certificate": str(
            tmp_path / "xroot" / "current"
            / "ownership-certificate.json"
        ),
        "lease": str(
            tmp_path / "xroot" / "leases" / "fixture-cell.json"
        ),
        "history_checkpoint": str(tmp_path / "history-checkpoint"),
        "history_bank_ack": str(tmp_path / "history-bank-ack.json"),
        "history_bank_receipt": str(
            tmp_path / "history-bank-receipt.json"
        ),
        "history_stage_receipt": str(
            tmp_path / "history-stage-receipt.json"
        ),
        "publication_fence": str(tmp_path / "publication-fence.json"),
        "bank_lock": str(tmp_path / "bank.lock"),
        "broker_lock": str(tmp_path / "xroot" / "broker.lock"),
        "writer_host": socket.gethostname(),
        "durable_evidence_destination": str(
            tmp_path / "durable-root" / "recovery-evidence"
        ),
        "result_bank_ack": str(run / "results-banked.json"),
        "result_binding_destination": str(
            tmp_path / "durable-root" / "result-binding"
        ),
        "result_destination_root": str(
            tmp_path / "durable-root" / "runs" / "fixture-cell"
        ),
        "bindings": str(tmp_path / "xroot" / "current" / "bindings.json"),
        "activation": str(
            tmp_path / "xroot" / "activation" / "last-published.json"
        ),
        "replay_worker": str(
            SCRIPTS / "validation_history_replay_worker.py"
        ),
        "replay_worker_sha256": sha(
            SCRIPTS / "validation_history_replay_worker.py"
        ),
        "proof_runner": str(SCRIPTS / "replacement_gpu_proof.py"),
        "proof_runner_sha256": sha(
            SCRIPTS / "replacement_gpu_proof.py"
        ),
        "frozen_runtime_files_sha256": {},
        "replay_pythonpath": [],
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "epochs": {},
    }
    frozen_runtime = tmp_path / "frozen-runtime"
    frozen_runtime.mkdir()
    for name in ("p5_run.py", "p5_checkpoint.py", "p5_source.py"):
        (frozen_runtime / name).write_text(f"# {name}\n")
        request["frozen_runtime_files_sha256"][name] = sha(
            frozen_runtime / name
        )
    dependency = tmp_path / "dependency" / "rrbench" / "io.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("# dependency\n")
    request["replay_pythonpath"].append({
        "path": str(tmp_path / "dependency"),
        "anchor": str(dependency),
        "anchor_sha256": sha(dependency),
    })
    plan_entries = [{
        "path": str(tmp_path / "durable" / "p5-state-root"),
        "bank_receipt_sha256": "0" * 64,
        "checkpoint_seal_sha256": "1" * 64,
        "step": 0,
        "epoch": 0,
        "phase": "train",
        "generation": 1,
    }]
    for epoch in range(1, 21):
        checkpoint = (
            tmp_path / "staged" / f"epoch-{epoch:02d}" / "checkpoint"
        )
        checkpoint.mkdir(parents=True)
        for name, data in (
            ("model.safetensors", f"model-{epoch}".encode()),
            ("training-state.pt", f"state-{epoch}".encode()),
        ):
            (checkpoint / name).write_bytes(data)
        write(checkpoint / "p5-seal.json", {
            "kind": "p5-full-state-v1",
            "status": "sealed",
            "contract_sha256": digest_json({"fixture": True}),
            "global_step": epoch * 10,
            "epoch": epoch - 1,
            "phase": "validate",
            "files_sha256": {
                "model.safetensors": sha(checkpoint / "model.safetensors"),
                "training-state.pt": sha(
                    checkpoint / "training-state.pt"
                ),
            },
        })
        source = tmp_path / "durable" / f"p5-state-epoch-{epoch:02d}"
        source_receipt_sha256 = hashlib.sha256(
            f"source-receipt-{epoch}".encode()
        ).hexdigest()
        write(checkpoint.parent / "stage-receipt.json", {
            "kind": "p5-validation-checkpoint-stage-v1",
            "status": "verified",
            "epoch": epoch,
            "source": str(source),
            "source_bank_receipt_sha256": source_receipt_sha256,
            "checkpoint_seal_sha256": sha(
                checkpoint / "p5-seal.json"
            ),
            "files": {
                name: {
                    "sha256": sha(checkpoint / name),
                    "bytes": (checkpoint / name).stat().st_size,
                }
                for name in (
                    "model.safetensors", "training-state.pt",
                    "p5-seal.json",
                )
            },
        })
        request["epochs"][str(epoch)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_seal_sha256": sha(checkpoint / "p5-seal.json"),
            "stage_receipt_sha256": sha(
                checkpoint.parent / "stage-receipt.json"
            ),
            "authority": "parent-event-inventory",
            "bank_ack": None,
            "bank_receipt": None,
        }
        plan_entries.append({
            "path": str(source),
            "bank_receipt_sha256": source_receipt_sha256,
            "checkpoint_seal_sha256": sha(checkpoint / "p5-seal.json"),
            "step": epoch * 10,
            "epoch": epoch - 1,
            "phase": "validate",
            "generation": len(plan_entries) + 1,
        })
        if epoch < 20:
            plan_entries.append({
                "path": str(
                    tmp_path / "durable"
                    / f"p5-state-train-{epoch:02d}"
                ),
                "bank_receipt_sha256": hashlib.sha256(
                    f"train-receipt-{epoch}".encode()
                ).hexdigest(),
                "checkpoint_seal_sha256": hashlib.sha256(
                    f"train-seal-{epoch}".encode()
                ).hexdigest(),
                "step": epoch * 10,
                "epoch": epoch,
                "phase": "train",
                "generation": len(plan_entries) + 1,
            })
    current_checkpoint = Path(request["epochs"]["20"]["checkpoint"])
    current_stage_path = current_checkpoint.parent / "stage-receipt.json"
    current_stage = json.loads(current_stage_path.read_text())
    current_ack_path = tmp_path / "epoch-20-bank-ack.json"
    current_bank_receipt_path = tmp_path / "epoch-20-bank-receipt.json"
    current_bank_receipt = {"files": current_stage["files"]}
    write(current_bank_receipt_path, current_bank_receipt)
    current_stage["source_bank_receipt_sha256"] = digest_json(
        current_bank_receipt
    )
    current_stage["source_bank_receipt_file_sha256"] = sha(
        current_bank_receipt_path
    )
    write(current_ack_path, {
        "kind": "p5-full-state-v1",
        "status": "verified",
        "artifact": Path(current_stage["source"]).name,
        "seal_sha256": sha(current_checkpoint / "p5-seal.json"),
        "contract_sha256": digest_json({"fixture": True}),
        "destination": current_stage["source"],
        "bank_receipt_sha256":
        current_stage["source_bank_receipt_sha256"],
    })
    current_stage["source_bank_ack_sha256"] = sha(current_ack_path)
    write(current_stage_path, current_stage)
    request["epochs"]["20"].update({
        "authority": "current-run-bank-ack",
        "bank_ack": str(current_ack_path),
        "bank_receipt": str(current_bank_receipt_path),
        "stage_receipt_sha256": sha(current_stage_path),
    })
    write(request["runtime_request"], {"fixture": True})
    parent_entries = plan_entries[:-1]
    checkpoint_root = {
        "artifact": Path(parent_entries[-1]["path"]).name,
        "seal_sha256": parent_entries[-1]["checkpoint_seal_sha256"],
        "contract_sha256": digest_json({"fixture": True}),
        "step": parent_entries[-1]["step"],
        "epoch": parent_entries[-1]["epoch"],
        "phase": parent_entries[-1]["phase"],
        "model_sha256": "3" * 64,
        "training_state_sha256": "4" * 64,
    }
    checkpoint_owner = {
        "kind": "p5-xnode-lease-v1",
        "cell": "fixture-cell",
        "root": checkpoint_root,
    }
    checkpoint_event = {
        "kind": "p5-xnode-ownership-v1",
        "sequence": 4,
        "status": "committed",
        "owners": {"fixture-cell": checkpoint_owner},
        "durable_recovery": {
            "fixture-cell": {
                "authority":
                "receipt-verified-durable-full-state-not-historical-acceptance",
                "historically_accepted": False,
                "chosen_checkpoint": parent_entries[-1]["path"],
                "chosen_checkpoint_seal_sha256":
                parent_entries[-1]["checkpoint_seal_sha256"],
                "root": checkpoint_root,
                "verified_checkpoints": parent_entries,
            },
        },
    }
    checkpoint_event_sha256 = xreg.event_sha(checkpoint_event)
    checkpoint_bank_receipt = {
        "files": {
            "ownership.json": {
                "sha256": checkpoint_event_sha256,
                "bytes": 1,
            },
        },
    }
    write(request["checkpoint_certificate"], {
        "event": checkpoint_event,
        "event_sha256": checkpoint_event_sha256,
        "bank_receipt": checkpoint_bank_receipt,
        "bank_receipt_sha256": xreg.digest(checkpoint_bank_receipt),
    })
    root = checkpoint_root
    lease = {
        "kind": "p5-xnode-lease-v1",
        "cell": "fixture-cell",
        "owner_node": socket.gethostname(),
        "gpu": 4,
        "gpu_uuid": "GPU-fixture",
        "lease_id": "fixture-lease",
        "ownership_epoch": 2,
        "root": root,
    }
    write(request["lease"], lease)
    current_event = {
        "kind": "p5-xnode-ownership-v1",
        "sequence": 5,
        "previous_event_sha256": checkpoint_event_sha256,
        "owners": {"fixture-cell": lease},
    }
    source_event_sha256 = xreg.event_sha(current_event)
    bank_receipt = {
        "files": {
            "ownership.json": {
                "sha256": source_event_sha256,
                "bytes": 1,
            },
        },
    }
    write(request["ownership_certificate"], {
        "event": current_event,
        "event_sha256": source_event_sha256,
        "bank_receipt": bank_receipt,
        "bank_receipt_sha256": xreg.digest(bank_receipt),
    })
    history_checkpoint = Path(request["history_checkpoint"])
    history_checkpoint.mkdir()
    (history_checkpoint / "model.safetensors").write_bytes(
        current_checkpoint.joinpath("model.safetensors").read_bytes()
    )
    (history_checkpoint / "training-state.pt").write_bytes(b"history-state")
    write(history_checkpoint / "p5-seal.json", {
        "kind": "p5-full-state-v1",
        "status": "sealed",
        "contract_sha256": digest_json({"fixture": True}),
        "global_step": 200,
        "epoch": 20,
        "phase": "frozen",
        "files_sha256": {
            "model.safetensors": sha(
                history_checkpoint / "model.safetensors"
            ),
            "training-state.pt": sha(
                history_checkpoint / "training-state.pt"
            ),
        },
    })
    history_files = {
        name: {
            "sha256": sha(history_checkpoint / name),
            "bytes": (history_checkpoint / name).stat().st_size,
        }
        for name in (
            "model.safetensors", "training-state.pt", "p5-seal.json",
        )
    }
    history_bank_receipt = {"files": history_files}
    write(request["history_bank_receipt"], history_bank_receipt)
    history_bank_ack = {
        "kind": "p5-full-state-v1",
        "status": "verified",
        "artifact": history_checkpoint.name,
        "seal_sha256": sha(history_checkpoint / "p5-seal.json"),
        "contract_sha256": digest_json({"fixture": True}),
        "destination": str(tmp_path / "durable-history"),
        "bank_receipt_sha256": digest_json(history_bank_receipt),
    }
    write(request["history_bank_ack"], history_bank_ack)
    write(request["history_stage_receipt"], {
        "kind": "p5-validation-checkpoint-stage-v1",
        "status": "verified",
        "purpose": "history-authority",
        "source": history_bank_ack["destination"],
        "source_bank_ack_sha256": sha(request["history_bank_ack"]),
        "source_bank_receipt_sha256": digest_json(history_bank_receipt),
        "source_bank_receipt_file_sha256": sha(
            request["history_bank_receipt"]
        ),
        "checkpoint_seal_sha256": sha(
            history_checkpoint / "p5-seal.json"
        ),
        "files": history_files,
    })
    selected_sha = "8" * 64
    write(run / "latest.json", {
        "artifact": history_checkpoint.name,
        "global_step": 200,
        "phase": "frozen",
        "seal_sha256": sha(history_checkpoint / "p5-seal.json"),
    })
    write(run / "selected.json", {
        "artifact": "selected",
        "seal_sha256": selected_sha,
    })
    write(run / "frozen.json", {
        "status": "frozen-before-test",
        "epochs": 20,
        "selected": "selected",
        "selected_seal_sha256": selected_sha,
    })
    (run / "test-predictions.jsonl").write_bytes(expected_bytes)
    write(run / "test-started.json", {
        "started_after_bank_ack": True,
        "selected_seal_sha256": selected_sha,
    })
    write(run / "test-result.json", {
        "status": "complete",
        "split": "test",
        "users": 2,
        "expected_users": 2,
        "beams_per_user": 20,
        "seen_item_exclusion": False,
        "selected_seal_sha256": selected_sha,
        "predictions_sha256": sha(run / "test-predictions.jsonl"),
    })
    write(run / "completed.json", {
        "status": "complete",
        "test_evaluations": 1,
    })
    request["source_event_sha256"] = source_event_sha256
    request["lease_sha256"] = xreg.digest(lease)
    request["root_seal_sha256"] = root["seal_sha256"]
    bindings = {
        "kind": "p5-xnode-bindings-v1",
        "root": str(tmp_path),
        "p5_root": str(tmp_path / "p5"),
        "control_root": str(tmp_path / "xroot"),
        "durable_root": str(tmp_path / "durable-root"),
        "writer_host": socket.gethostname(),
        "cells": {
            "fixture-cell": {
                "role": "writer",
                "gpu": 4,
                "gpu_uuid": "GPU-fixture",
            },
        },
    }
    write(request["bindings"], bindings)
    write(request["activation"], {
        "kind": "p5-xnode-activation-v2",
        "status": "activated",
        "event_sha256": source_event_sha256,
        "files_sha256": {
            "current/bindings.json": sha(request["bindings"]),
            "current/ownership-certificate.json": sha(
                request["ownership_certificate"]
            ),
            "leases/fixture-cell.json": sha(request["lease"]),
        },
    })
    write(request["publication_fence"], {
        "kind": "p5-validation-history-recovery-fence-v1",
        "status": "active",
        "cell": "fixture-cell",
        "source_event_sha256": source_event_sha256,
        "broker_stopped": True,
        "broker_pid": 2147483647,
        "broker_stopped_at_unix": 1,
    })
    request_path = tmp_path / "request.json"
    write(request_path, request)
    return request_path, run, expected, expected_bytes


class FakeBackend:
    def __init__(self, expected, output_bytes, state=None):
        self.expected = expected
        self.output_bytes = output_bytes
        self.state = state

    def replay(self, request, epoch, checkpoint, output):
        Path(output).write_bytes(self.output_bytes)
        state = self.state or {
            "epoch": epoch - 1,
            "phase": "validate",
            "global_step": epoch * 10,
            "validation_history": [self.expected for _ in range(epoch - 1)],
        }
        return {
            "metrics": self.expected,
            "state": state,
            "identity": {
                **request["binding"],
                "gpu_name": "fixture GPU",
            },
        }


def patch_authorities(recovery, monkeypatch, request_path, expected):
    import p5_bank

    request = json.loads(Path(request_path).read_text())
    history_header = {"seal_sha256": "1" * 64}
    monkeypatch.setattr(
        recovery,
        "_history_authority",
        lambda _: ([expected for _ in range(20)], history_header),
    )
    monkeypatch.setattr(
        recovery,
        "_stage_authority",
        lambda _, epoch: (
            Path(request["epochs"][str(epoch)]["checkpoint"]),
            json.loads(
                (
                    Path(request["epochs"][str(epoch)]["checkpoint"]).parent
                    / "stage-receipt.json"
                ).read_text()
            ),
        ),
    )
    monkeypatch.setattr(
        p5_bank, "_recover_selection_locked", lambda run: None,
    )


def recover_all(recovery, request_path, expected, output_bytes):
    return [
        recovery.recover_epoch(
            request_path, epoch,
            backend=FakeBackend(expected, output_bytes),
        )
        for epoch in range(1, 21)
    ]


def test_real_authorities_bind_lineage_without_rereading_durable_storage(
    tmp_path, monkeypatch,
):
    import torch
    from p5_checkpoint import IndexSampler

    recovery = module()
    request_path, _, expected, _ = fixture(tmp_path)
    request = json.loads(request_path.read_text())
    sampler = IndexSampler(2, 7)
    sampler.next(2)
    validation_sampler = sampler.state_dict()
    sampler.advance()
    frozen_sampler = sampler.state_dict()
    common = {
        "contract": {"fixture": True},
        "checkpoint_boundary":
        "after_complete_optimizer_update_no_partial_gradients",
        "optimizer": {"state": {}},
        "scheduler": {"step": 20},
        "rng": {"fixture": True},
        "schema_version": 1,
    }
    validation_payload = {
        **common,
        "sampler": validation_sampler,
        "state": {
            "phase": "validate",
            "epoch": 19,
            "global_step": 200,
            "validation_history": [expected for _ in range(19)],
            "best_metric": 0.5,
        },
    }
    frozen_payload = {
        **common,
        "sampler": frozen_sampler,
        "state": {
            "phase": "frozen",
            "epoch": 20,
            "global_step": 200,
            "validation_history": [expected for _ in range(20)],
            "best_metric": 0.5,
        },
    }
    monkeypatch.setattr(
        torch, "load",
        lambda path, **kwargs: (
            validation_payload
            if "epoch-20" in str(path) else frozen_payload
        ),
    )

    history, header = recovery._history_authority(request)
    checkpoint, receipt = recovery._stage_authority(request, 1)
    current_checkpoint, current_receipt = recovery._stage_authority(
        request, 20,
    )

    assert history == [expected for _ in range(20)]
    assert header["phase"] == "frozen"
    assert checkpoint == Path(request["epochs"]["1"]["checkpoint"])
    assert receipt["epoch"] == 1
    assert current_checkpoint == Path(request["epochs"]["20"]["checkpoint"])
    assert current_receipt["epoch"] == 20


def test_history_authority_rejects_unlinked_parent_event(
    tmp_path, monkeypatch,
):
    import torch

    recovery = module()
    request_path, _, expected, _ = fixture(tmp_path)
    request = json.loads(request_path.read_text())
    certificate = json.loads(
        Path(request["ownership_certificate"]).read_text()
    )
    certificate["event"]["previous_event_sha256"] = "0" * 64
    certificate["event_sha256"] = xreg.event_sha(certificate["event"])
    certificate["bank_receipt"]["files"]["ownership.json"]["sha256"] = (
        certificate["event_sha256"]
    )
    certificate["bank_receipt_sha256"] = xreg.digest(
        certificate["bank_receipt"]
    )
    write(request["ownership_certificate"], certificate)
    request["source_event_sha256"] = certificate["event_sha256"]
    write(request_path, request)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "contract": {"fixture": True},
        "checkpoint_boundary":
        "after_complete_optimizer_update_no_partial_gradients",
        "state": {
            "phase": "frozen",
            "epoch": 20,
            "validation_history": [expected for _ in range(20)],
        },
    })
    with pytest.raises(ValueError, match="authority differs"):
        recovery._history_authority(request)


def test_exact_state_comparison_is_dtype_and_bit_pattern_sensitive():
    import numpy as np
    import torch

    recovery = module()

    assert not recovery._exact_state_equal(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([-0.0], dtype=torch.float32),
    )
    assert not recovery._exact_state_equal(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int64),
    )
    assert recovery._exact_state_equal(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    assert not recovery._exact_state_equal(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(-0.0, dtype=torch.float32),
    )
    assert not recovery._exact_state_equal(
        np.array([0.0], dtype=np.float32),
        np.array([-0.0], dtype=np.float32),
    )


def test_validation_parent_accepts_legitimate_epoch20_new_best(
    tmp_path, monkeypatch,
):
    import torch
    from p5_checkpoint import IndexSampler

    recovery = module()
    validation = tmp_path / "validation"
    frozen = tmp_path / "p5-state-final"
    validation.mkdir()
    frozen.mkdir()
    sampler = IndexSampler(2, 9)
    sampler.next(2)
    before_sampler = sampler.state_dict()
    sampler.advance()
    after_sampler = sampler.state_dict()
    history = [
        {"primary": {"recall@10": 0.4}},
        {"primary": {"recall@10": 0.6}},
    ]
    common = {
        "contract": {"fixture": True},
        "checkpoint_boundary":
        "after_complete_optimizer_update_no_partial_gradients",
        "optimizer": {"state": {}},
        "scheduler": {"step": 2},
        "rng": {"fixture": True},
        "schema_version": 1,
    }
    before = {
        **common,
        "sampler": before_sampler,
        "state": {
            "phase": "validate",
            "epoch": 1,
            "global_step": 20,
            "validation_history": history[:-1],
            "best_metric": 0.4,
            "selected_artifact": "old-best",
        },
    }
    after = {
        **common,
        "sampler": after_sampler,
        "state": {
            "phase": "frozen",
            "epoch": 2,
            "global_step": 20,
            "validation_history": history,
            "best_metric": 0.6,
            "selected_artifact": frozen.name,
        },
    }
    monkeypatch.setattr(
        torch, "load",
        lambda path, **kwargs: (
            before if Path(path).parent == validation else after
        ),
    )

    recovery._verify_validation_frozen_parent(
        validation, frozen, history,
    )


def test_replays_and_atomically_publishes_exact_historical_validation(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    before = {
        name: sha(run / name)
        for name in ("latest.json", "selected.json", "frozen.json", "test-result.json")
    }

    result = recovery.recover_epoch(
        request_path, 1, backend=FakeBackend(expected, output_bytes),
    )

    canonical = run / "validation" / "epoch-01.jsonl"
    assert not canonical.exists()
    assert result["status"] == "replayed_exact"
    assert result["history_sha256_matches"] is True
    assert result["metrics"] == expected
    assert before == {
        name: sha(run / name)
        for name in ("latest.json", "selected.json", "frozen.json", "test-result.json")
    }
    assert recovery.verify_epoch(request_path, 1)["status"] == "verified"
    recover_all(recovery, request_path, expected, output_bytes)
    complete = recovery._publish_all(request_path)
    assert complete["status"] == "verified"
    manifest = json.loads(
        (run / "validation-history-recovery.json").read_text()
    )
    assert manifest["status"] == "published-and-result-bound"
    assert canonical.read_bytes() == output_bytes
    assert (run / "validation-history-recovery.json").is_file()
    request = json.loads(request_path.read_text())
    durable = Path(request["durable_evidence_destination"])
    assert (durable / "bank-receipt.json").is_file()
    assert (
        durable / "epochs" / "epoch-01-recovery.json"
    ).is_file()
    assert (durable / "epochs" / "epoch-01-stage.json").is_file()


@pytest.mark.parametrize(
    "damage", ["metrics", "sha", "state", "prefix", "existing"],
)
def test_recovery_fails_closed_on_any_history_or_publication_difference(
    tmp_path, monkeypatch, damage,
):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    replay_metrics = json.loads(json.dumps(expected))
    state = None
    if damage == "metrics":
        replay_metrics["primary"]["recall@10"] = 0.4
    elif damage == "sha":
        output_bytes = b'{"different":true}\n'
    elif damage == "state":
        state = {
            "epoch": 1,
            "phase": "train",
            "global_step": 10,
            "validation_history": [expected],
        }
    elif damage == "prefix":
        state = {
            "epoch": 0,
            "phase": "validate",
            "global_step": 10,
            "validation_history": [{"different": True}],
        }
    else:
        path = run / "validation" / "epoch-01.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"foreign")

    with pytest.raises(ValueError):
        recovery.recover_epoch(
            request_path,
            1,
            backend=FakeBackend(replay_metrics, output_bytes, state),
        )


def test_existing_exact_file_is_idempotently_verified(tmp_path, monkeypatch):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    path = run / "validation" / "epoch-01.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(output_bytes)

    result = recovery.recover_epoch(
        request_path, 1, backend=FakeBackend(expected, output_bytes),
    )

    assert result["status"] == "existing_exact_verified"
    assert path.read_bytes() == output_bytes
    recover_all(recovery, request_path, expected, output_bytes)
    assert recovery._publish_all(request_path)["status"] == "verified"


def test_replay_retry_preserves_immutable_receipt_identity(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    first = recovery.recover_epoch(
        request_path, 1, backend=FakeBackend(expected, output_bytes),
    )
    request = json.loads(request_path.read_text())
    receipt_path = (
        Path(request["output"]) / "epoch-01" / "recovery-receipt.json"
    )
    receipt_sha256 = sha(receipt_path)

    second = recovery.recover_epoch(
        request_path, 1,
        backend=FakeBackend({"different": True}, b"different"),
    )

    assert second == first
    assert sha(receipt_path) == receipt_sha256


def test_publish_never_overwrites_a_concurrent_conflicting_file(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    original_link = recovery.os.link

    def race(source, target):
        if (Path(target).name == "epoch-01.jsonl"
                and Path(target).parent == run / "validation"):
            Path(target).write_bytes(b"concurrent-foreign")
            raise FileExistsError(target)
        return original_link(source, target)

    monkeypatch.setattr(recovery.os, "link", race)
    with pytest.raises(ValueError, match="Concurrent|conflict"):
        recovery._publish_all(request_path)
    monkeypatch.setattr(recovery.os, "link", original_link)
    assert (run / "validation" / "epoch-01.jsonl").read_bytes() == (
        b"concurrent-foreign"
    )


def test_publication_banks_provenance_and_marks_run_before_first_link(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    original_link = recovery.os.link
    original_fsync = recovery.fsync_directory
    synchronized = {"run": False}

    def track_fsync(path):
        if Path(path) == run:
            synchronized["run"] = True
        return original_fsync(path)

    def crash_on_validation(source, target):
        if (Path(target).name == "epoch-01.jsonl"
                and Path(target).parent == run / "validation"):
            assert synchronized["run"] is True
            raise RuntimeError("simulated publication crash")
        return original_link(source, target)

    monkeypatch.setattr(recovery.os, "link", crash_on_validation)
    monkeypatch.setattr(recovery, "fsync_directory", track_fsync)
    with pytest.raises(RuntimeError, match="simulated"):
        recovery._publish_all(request_path)
    request = json.loads(request_path.read_text())
    marker = json.loads(
        (run / "validation-history-recovery.json").read_text()
    )
    assert marker["status"] == "published-and-result-bound"
    assert (
        run / "validation-history-recovery-complete.json"
    ).is_file()
    assert (
        Path(request["durable_evidence_destination"]) / "bank-receipt.json"
    ).is_file()
    monkeypatch.setattr(recovery.os, "link", original_link)
    assert recovery._publish_all(request_path)["status"] == "verified"
    assert recovery.verify_complete(request_path)["status"] == "verified"


def test_publication_retries_after_incomplete_durable_copy(
    tmp_path, monkeypatch,
):
    import bank

    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    original = bank.copy_artifact
    attempted = {"value": False}

    def interrupted(source, destination):
        if not attempted["value"]:
            attempted["value"] = True
            Path(destination).mkdir(parents=True)
            (Path(destination) / "partial").write_bytes(b"partial")
            raise OSError("simulated interrupted durable copy")
        return original(source, destination)

    monkeypatch.setattr(bank, "copy_artifact", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        recovery._publish_all(request_path)
    monkeypatch.setattr(bank, "copy_artifact", original)

    recovery._publish_all(request_path)
    request = json.loads(request_path.read_text())
    manifest = json.loads(
        (
            Path(request["run"]) / "validation-history-recovery.json"
        ).read_text()
    )

    assert "-retry-" in manifest["durable_evidence_destination"]
    assert (
        Path(manifest["durable_evidence_destination"])
        / "bank-receipt.json"
    ).is_file()


def test_result_banking_quarantines_incomplete_canonical_destination(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, run, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    request = json.loads(request_path.read_text())
    name = "p5-results-" + digest_json(
        json.loads((run / "test-result.json").read_text())
    )[:16]
    base = Path(request["result_destination_root"]) / name
    base.mkdir(parents=True)
    (base / "partial").write_bytes(b"preserved")
    old_retry = base.with_name(base.name + "-retry-old")
    old_retry.mkdir()
    (old_retry / "foreign").write_bytes(b"foreign")
    write(old_retry / "bank-receipt.json", {
        "files": {
            "foreign": {
                "sha256": sha(old_retry / "foreign"),
                "bytes": (old_retry / "foreign").stat().st_size,
            },
        },
    })

    complete = recovery.publish_all(request_path)

    assert complete["result_destination"] == str(base)
    assert (base / "bank-receipt.json").is_file()
    quarantined = [
        path
        for path in (base.parent / "quarantine").glob(
            name + "-incomplete-*"
        )
        if path.is_dir()
    ]
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial").read_bytes() == b"preserved"
    assert (old_retry / "foreign").read_bytes() == b"foreign"


def test_verify_rejects_changed_recorded_identity(tmp_path, monkeypatch):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recovery.recover_epoch(
        request_path, 1, backend=FakeBackend(expected, output_bytes),
    )
    request = json.loads(request_path.read_text())
    request["lease_sha256"] = "0" * 64
    write(request_path, request)
    with pytest.raises(ValueError, match="receipt|differs"):
        recovery.verify_epoch(request_path, 1)


def test_publish_requires_complete_20_epoch_request(tmp_path, monkeypatch):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    request = json.loads(request_path.read_text())
    del request["epochs"]["20"]
    write(request_path, request)
    for epoch in range(1, 20):
        recovery.recover_epoch(
            request_path, epoch,
            backend=FakeBackend(expected, output_bytes),
        )
    with pytest.raises(ValueError, match="all 20"):
        recovery._publish_all(request_path)


def test_request_rejects_noncanonical_activation_path(tmp_path):
    recovery = module()
    request_path, _, _, _ = fixture(tmp_path)
    request = json.loads(request_path.read_text())
    alternate = (
        tmp_path / "xroot" / "activation" / "stale-activation.json"
    )
    alternate.write_bytes(Path(request["activation"]).read_bytes())
    request["activation"] = str(alternate)
    write(request_path, request)

    with pytest.raises(ValueError, match="deployment authority"):
        recovery._request(request_path)


def test_publish_rehashes_staged_bytes_inside_fence(tmp_path, monkeypatch):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    receipt_path = (
        Path(json.loads(request_path.read_text())["output"])
        / "epoch-01" / "recovery-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    Path(receipt["staged_output"]).write_bytes(b"changed-after-verify")
    monkeypatch.setattr(
        recovery, "verify_epoch",
        lambda request, epoch: {"status": "verified"},
    )
    with pytest.raises(ValueError, match="Staged validation bytes changed"):
        recovery._publish_all(request_path)


def test_binds_durable_recovery_evidence_to_banked_result(
    tmp_path, monkeypatch,
):
    import p5_bank

    rows = []
    for user in ("u1", "u2"):
        rows.append(json.dumps({
            "original_user": user,
            "original_target_item": 1,
            "target": "1",
            "predictions": [str(i) for i in range(20)],
            "scores": [float(i) for i in range(20)],
        }, sort_keys=True))
    output_bytes = ("\n".join(rows) + "\n").encode()
    recovery = module()
    request_path, run, expected, output_bytes = fixture(
        tmp_path, output_bytes,
    )
    patch_authorities(recovery, monkeypatch, request_path, expected)
    selection_checked = {"value": False}

    def validate_selection(run_path):
        selection_checked["value"] = True

    monkeypatch.setattr(
        p5_bank, "_recover_selection_locked", validate_selection,
    )
    recover_all(recovery, request_path, expected, output_bytes)
    binding = recovery.publish_all(request_path)
    request = json.loads(request_path.read_text())

    destination = Path(request["result_binding_destination"])
    assert binding["status"] == "verified"
    assert (destination / "bank-receipt.json").is_file()
    assert (destination / "binding.json").is_file()
    assert selection_checked["value"] is True
    assert recovery.verify_complete(request_path) == binding
    manifest_path = run / "validation-history-recovery.json"
    manifest_sha256 = sha(manifest_path)
    assert recovery._publish_all(request_path) == binding
    assert sha(manifest_path) == manifest_sha256
    result_receipt_path = (
        Path(binding["result_destination"]) / "bank-receipt.json"
    )
    result_receipt = json.loads(result_receipt_path.read_text())
    first = next(iter(result_receipt["files"]))
    result_receipt["files"][first]["sha256"] = "0" * 64
    write(result_receipt_path, result_receipt)
    with pytest.raises(ValueError, match="result receipt changed"):
        recovery.verify_complete(request_path)


def test_complete_gate_rejects_replaced_unrelated_result_ack(
    tmp_path, monkeypatch,
):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recover_all(recovery, request_path, expected, output_bytes)
    recovery._publish_all(request_path)
    request = json.loads(request_path.read_text())
    result = tmp_path / "durable-root" / "invalid-result"
    result.mkdir()
    (result / "p5-result-seal.json").write_bytes(b"not-a-result-seal")
    receipt = {
        "files": {
            "p5-result-seal.json": {
                "sha256": sha(result / "p5-result-seal.json"),
                "bytes": (result / "p5-result-seal.json").stat().st_size,
            },
        },
    }
    write(result / "bank-receipt.json", receipt)
    write(request["result_bank_ack"], {
        "status": "verified",
        "destination": str(result),
        "bank_receipt_sha256": digest_json(receipt),
        "result_seal_sha256": sha(result / "p5-result-seal.json"),
    })
    with pytest.raises(ValueError, match="binding gate"):
        recovery.verify_complete(request_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "verified"),
        ("reconstructed_by_replay", False),
        ("stage_receipt_sha256", "0" * 64),
    ],
)
def test_verify_rejects_changed_replay_receipt_fields(
    tmp_path, monkeypatch, field, value,
):
    recovery = module()
    request_path, _, expected, output_bytes = fixture(tmp_path)
    patch_authorities(recovery, monkeypatch, request_path, expected)
    recovery.recover_epoch(
        request_path, 1, backend=FakeBackend(expected, output_bytes),
    )
    request = json.loads(request_path.read_text())
    receipt_path = (
        Path(request["output"]) / "epoch-01" / "recovery-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt differs"):
        recovery.verify_epoch(request_path, 1)
