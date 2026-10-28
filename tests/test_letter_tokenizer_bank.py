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
