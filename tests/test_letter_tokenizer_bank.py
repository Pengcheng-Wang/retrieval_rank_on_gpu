import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def module():
    assert (ROOT / "scripts" / "letter" / "bank_tokenizers.py").exists()
    return importlib.import_module("bank_tokenizers")


def make_run(path):
    import torch
    from protocol import file_sha256, write_json

    config = {"track_id": "letter-openllama7b-v2-method-replication", "synthetic_fixture": False,
              "dataset": "Instruments", "epochs": 20000}
    write_json(path / "run-config.json", config)
    for epoch in (2, 4):
        checkpoint = path / "checkpoints" / f"epoch-{epoch:05d}"
        checkpoint.mkdir(parents=True)
        state = {
            "epoch": epoch - 1, "spec": config, "model": {"weight": torch.tensor([epoch])},
            "optimizer": {}, "initted": [True] * 4, "selected_checkpoint": "checkpoints/epoch-00002",
            "python_rng": (), "numpy_rng": (), "torch_rng": torch.get_rng_state(), "cuda_rng": [],
            "best_loss": 1.0, "best_collision_rate": 0.01,
        }
        torch.save(state, checkpoint / "state.pt")
        write_json(checkpoint / "completed.json", {
            "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": epoch - 1,
            "state_sha256": file_sha256(checkpoint / "state.pt"),
        })
    write_json(path / "latest-state.json", {
        "epoch": 4, "checkpoint": str(path / "checkpoints" / "epoch-00004"),
    })
    return config


def test_quantizer_snapshot_preserves_latest_and_selected(tmp_path):
    run = tmp_path / "rq"
    make_run(run)
    packet = module().snapshot(run, tmp_path / "staging")
    assert packet["phase"] == "checkpoint"
    bundle = Path(packet["path"])
    assert (bundle / "payload/checkpoints/epoch-00004/state.pt").is_file()
    assert (bundle / "payload/checkpoints/epoch-00002/state.pt").is_file()
    assert not (bundle / "payload/completed.json").exists()
    module().verify_bundle(bundle)
    assert module().snapshot(run, tmp_path / "staging") == packet
    (bundle / "payload/checkpoints/epoch-00004/state.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError):
        module().verify_bundle(bundle)


def test_quantizer_snapshot_preserves_compatibility_checkpoint_ancestry(tmp_path):
    import torch
    from protocol import file_sha256, write_json

    run = tmp_path / "rq"
    config = make_run(run)
    parent = run / "checkpoints" / "epoch-00003"
    parent.mkdir()
    torch.save({
        "epoch": 2, "spec": config, "model": {"weight": torch.tensor([3])},
        "optimizer": {}, "initted": [True] * 4,
        "selected_checkpoint": "checkpoints/epoch-00002",
        "python_rng": (), "numpy_rng": (), "torch_rng": torch.get_rng_state(), "cuda_rng": [],
        "best_loss": 1.0, "best_collision_rate": 0.01,
    }, parent / "state.pt")
    write_json(parent / "completed.json", {
        "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": 2,
        "state_sha256": file_sha256(parent / "state.pt"),
    })
    parent_compatibility = parent / "compatibility"
    parent_compatibility.mkdir()
    write_json(parent_compatibility / "attestation.json", {
        "parent_checkpoint": None,
        "parent_attestation_sha256": None,
    })
    latest_compatibility = run / "checkpoints/epoch-00004/compatibility"
    latest_compatibility.mkdir()
    write_json(latest_compatibility / "attestation.json", {
        "parent_checkpoint": "checkpoints/epoch-00003",
        "parent_attestation_sha256":
            file_sha256(parent_compatibility / "attestation.json"),
    })
    packet = module().snapshot(run, tmp_path / "staging")
    payload = Path(packet["path"]) / "payload/checkpoints"
    assert (payload / "epoch-00003/state.pt").is_file()
    assert (payload / "epoch-00004/compatibility/attestation.json").is_file()


def test_quantizer_snapshot_never_admits_unsealed_or_redirected_state(tmp_path):
    assert module().snapshot(tmp_path / "missing", tmp_path / "staging") is None
    run = tmp_path / "rq"
    make_run(run)
    (run / "checkpoints/epoch-00004/state.pt").write_bytes(b"interrupted")
    with pytest.raises(ValueError):
        module().snapshot(run, tmp_path / "staging")
    (run / "latest-state.json").write_text(json.dumps({"epoch": 4, "checkpoint": "../other/epoch-00004"}))
    with pytest.raises(ValueError):
        module().snapshot(run, tmp_path / "staging")


def test_parameterization_cost_gate_is_never_banked_as_tokenizer_state(tmp_path):
    from protocol import write_json

    run = tmp_path / "rq"
    config = make_run(run)
    write_json(run / "run-config.json", {
        **config,
        "purpose": {
            "kind": "figure8_parameterization_cost_gate",
            "admissible_final_tokenizer": False,
        },
    })
    with pytest.raises(ValueError, match="parameterization|final tokenizer"):
        module().snapshot(run, tmp_path / "staging")
    assert not (tmp_path / "staging").exists()


def test_quantizer_acknowledgment_requires_the_exact_complete_run(tmp_path):
    from protocol import file_sha256, write_json

    run = tmp_path / "rq"
    make_run(run)
    write_json(run / "completed.json", {"status": "complete", "synthetic_fixture": False})
    digest = file_sha256(run / "completed.json")
    module().acknowledge(run, tmp_path / "acks", digest)
    assert json.loads((tmp_path / "acks/rq.tokenizer.json").read_text())["completed_sha256"] == digest
    with pytest.raises(ValueError):
        module().acknowledge(run, tmp_path / "acks", "0" * 64)


def test_node0_tokenizer_snapshot_does_not_need_ssh(tmp_path):
    run = tmp_path / "rq"
    make_run(run)
    args = SimpleNamespace(source_host="node-0", remote_staging=tmp_path / "staging")
    result = module().worker_snapshot(args, run)
    assert result["phase"] == "checkpoint"
    module().verify_bundle(Path(result["path"]))


def test_remote_tokenizer_snapshot_hides_cuda_from_cpu_banker(tmp_path, monkeypatch):
    calls = []

    def remote(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="null\n", stderr="")

    monkeypatch.setattr(module().subprocess, "run", remote)
    args = SimpleNamespace(
        source_host="node-1",
        source_project=tmp_path / "control",
        remote_staging=tmp_path / "staging",
    )
    assert module().worker_snapshot(args, tmp_path / "run") is None
    assert calls[0][:4] == ["ssh", "-n", "-o", "BatchMode=yes"]
    assert "CUDA_VISIBLE_DEVICES=-1" in calls[0][-1]
