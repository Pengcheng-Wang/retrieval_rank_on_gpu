import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))

import replacement_gpu_proof as proof


def test_runner_is_outside_frozen_p5_driver_inventory():
    assert Path(proof.__file__).name not in {
        path.name for path in SCRIPTS.glob("p5_*.py")
    }


def test_loads_learner_from_private_frozen_runtime_root(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "p5_run.py").write_text(
        "FROZEN_RUNTIME_SENTINEL = 'private-runtime'\n",
        encoding="utf-8",
    )

    loaded = proof._load_frozen_learner(runtime)

    assert loaded.FROZEN_RUNTIME_SENTINEL == "private-runtime"
    assert Path(loaded.__file__).resolve() == (runtime / "p5_run.py").resolve()


def test_restore_full_state_uses_public_frozen_api_and_captures_restored_rng():
    calls = []
    restored_rng = {"torch": [11, 12], "cuda": [[13, 14]]}

    class PublicCheckpoint:
        @staticmethod
        def restore_checkpoint(path, model, optimizer, scheduler, sampler, contract):
            calls.append(("restore", path, model, optimizer, scheduler, sampler, contract))
            return {"global_step": 37}

        @staticmethod
        def rng_state():
            calls.append(("rng",))
            return copy.deepcopy(restored_rng)

    arguments = (
        Path("checkpoint"), object(), object(), object(), object(), {"contract": "frozen"},
    )

    state, captured = proof._restore_full_state(PublicCheckpoint, *arguments)

    assert state == {"global_step": 37}
    assert captured == restored_rng
    assert [call[0] for call in calls] == ["restore", "rng"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SyntheticBackend:
    def __init__(self):
        self.loads = []
        self.snapshot = {
            "model": {"weight": [1.0, 2.0]},
            "optimizer": {"state": {"moment": [0.5]}},
            "scheduler": {"step": 7},
            "sampler": {"epoch": 2, "offset": 12},
            "rng": {"python": [3, 4], "numpy": [5, 6], "torch": [7, 8], "cuda": [[9, 10]]},
            "training_state": {"global_step": 11, "validation_history": [{"recall": 0.25}]},
        }
        self.update_result = {
            "sampled_rows": [4, 1, 9],
            "sampled_rows_sha256": "a" * 64,
            "loss": 1.25,
            "update": {"gradient_norm": 0.75, "lr_after_update": 1e-4},
            "resulting_state": {"global_step": 12},
        }

    def identity(self):
        return {
            "host": "node-2",
            "gpu": 6,
            "gpu_uuid": "GPU-current",
            "gpu_name": "Synthetic CUDA",
        }

    def cold_load(self, request, label):
        self.loads.append(label)
        return {
            "snapshot": copy.deepcopy(self.snapshot),
            "optimizer_memory": {
                "populated": True,
                "parameter_states": 3,
                "tensor_values": 18,
            },
        }

    def next_update(self, loaded):
        return copy.deepcopy(self.update_result)

    def geometry(self, loaded):
        return {
            "actual_generation": True,
            "beams": 20,
            "returns": 20,
            "probe_rows": 2,
            "complete_sequences": 40,
            "complete_catalog_sequences_only": True,
            "longest_catalog_target_completed": True,
        }


@pytest.fixture
def proof_request(tmp_path):
    control = tmp_path / "xnode"
    checkpoint = tmp_path / "durable" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "training-state.pt").write_bytes(b"state")
    contract = tmp_path / "private" / "contract.json"
    contract_value = {"contract": "synthetic"}
    write_json(contract, contract_value)
    seal = {
        "kind": "p5-full-state-v1",
        "status": "sealed",
        "contract_sha256": document_sha(contract_value),
        "files_sha256": {
            "model.safetensors": sha(checkpoint / "model.safetensors"),
            "training-state.pt": sha(checkpoint / "training-state.pt"),
        },
    }
    write_json(checkpoint / "p5-seal.json", seal)
    inputs = {}
    for name in proof.INPUT_NAMES:
        root = tmp_path / "private" / name
        root.mkdir(parents=True)
        anchor = root / ("p5_run.py" if name == "runtime" else "anchor.json")
        write_json(anchor, {"name": name})
        inputs[name] = {
            "path": str(root),
            "anchor": str(anchor),
            "anchor_sha256": sha(anchor),
        }
    value = {
        "kind": "p5-replacement-gpu-proof-request-v1",
        "cell": "Yelp-TID",
        "binding": {
            "host": "node-2",
            "gpu": 6,
            "gpu_uuid": "GPU-current",
            "control_root": str(control),
        },
        "checkpoint": str(checkpoint),
        "contract": str(contract),
        "contract_sha256": sha(contract),
        "inputs": inputs,
        "output": str(control / "gates" / "replacement-proof-yelp-tid"),
        "microbatch_size": 4,
    }
    path = tmp_path / "request.json"
    write_json(path, value)
    return path, value, seal


def test_emits_bound_proof_after_two_independent_cold_loads(proof_request):
    path, value, seal = proof_request
    backend = SyntheticBackend()

    result = proof.run_proof(path, backend=backend)

    assert backend.loads == ["cold-a", "cold-b"]
    assert result["kind"] == "p5-replacement-gpu-proof-v1"
    assert result["status"] == "pass"
    assert result["cell"] == "Yelp-TID"
    assert result["host"] == "node-2"
    assert result["gpu"] == 6
    assert result["gpu_uuid"] == "GPU-current"
    assert result["source_checkpoint_seal_sha256"] == sha(Path(value["checkpoint"]) / "p5-seal.json")
    assert result["contract_sha256"] == seal["contract_sha256"]
    assert result["model_sha256"] == seal["files_sha256"]["model.safetensors"]
    assert result["training_state_sha256"] == seal["files_sha256"]["training-state.pt"]
    assert result["cold_save_reload_bitwise_equal"] is True
    assert result["next_sampled_rows_and_losses_equal"] is True
    assert result["populated_optimizer_memory_pass"] is True
    assert result["complete_20beam_geometry_pass"] is True
    assert result["input_bindings_sha256"] == proof.document_sha(value["inputs"])
    assert json.loads((Path(value["output"]) / "p5-replacement-gpu-proof.json").read_text()) == result


def test_rejects_current_device_identity_mismatch(proof_request):
    path, _, _ = proof_request
    backend = SyntheticBackend()
    backend.identity = lambda: {
        "host": "node-2", "gpu": 6, "gpu_uuid": "GPU-stale", "gpu_name": "Synthetic CUDA",
    }

    with pytest.raises(ValueError, match="current host/index/UUID"):
        proof.run_proof(path, backend=backend)


def test_rejects_changed_exact_input_anchor(proof_request):
    path, value, _ = proof_request
    Path(value["inputs"]["corpus"]["anchor"]).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="input anchor"):
        proof.run_proof(path, backend=SyntheticBackend())


def test_rejects_output_outside_isolated_control_gates(proof_request):
    path, value, _ = proof_request
    value["output"] = str(Path(value["binding"]["control_root"]) / "runs" / "production")
    write_json(path, value)

    with pytest.raises(ValueError, match="control_root/gates"):
        proof.run_proof(path, backend=SyntheticBackend())


def test_rejects_cold_load_state_difference(proof_request):
    path, _, _ = proof_request
    backend = SyntheticBackend()
    original = backend.cold_load

    def cold_load(value, label):
        loaded = original(value, label)
        if label == "cold-b":
            loaded["snapshot"]["training_state"]["validation_history"][0]["recall"] = 0.5
        return loaded

    backend.cold_load = cold_load
    with pytest.raises(AssertionError, match="cold-loaded full state"):
        proof.run_proof(path, backend=backend)


def test_rejects_next_sample_or_loss_difference(proof_request):
    path, _, _ = proof_request
    backend = SyntheticBackend()
    calls = 0

    def next_update(_loaded):
        nonlocal calls
        calls += 1
        result = copy.deepcopy(backend.update_result)
        if calls == 2:
            result["loss"] += 0.01
        return result

    backend.next_update = next_update
    with pytest.raises(AssertionError, match="next sampled rows/loss/update"):
        proof.run_proof(path, backend=backend)


def test_rejects_empty_optimizer_memory(proof_request):
    path, _, _ = proof_request
    backend = SyntheticBackend()
    original = backend.cold_load

    def cold_load(value, label):
        loaded = original(value, label)
        loaded["optimizer_memory"] = {
            "populated": False, "parameter_states": 0, "tensor_values": 0,
        }
        return loaded

    backend.cold_load = cold_load
    with pytest.raises(ValueError, match="optimizer memory"):
        proof.run_proof(path, backend=backend)


@pytest.mark.parametrize(
    "change",
    [
        {"actual_generation": False},
        {"beams": 19, "complete_sequences": 38},
        {"returns": 19, "complete_sequences": 38},
        {"complete_sequences": 39},
        {"complete_catalog_sequences_only": False},
        {"longest_catalog_target_completed": False},
    ],
)
def test_rejects_incomplete_or_synthetic_beam_geometry(proof_request, change):
    path, _, _ = proof_request
    backend = SyntheticBackend()
    geometry = backend.geometry(None)
    geometry.update(change)
    backend.geometry = lambda _loaded: geometry

    with pytest.raises(ValueError, match="actual complete 20-beam geometry"):
        proof.run_proof(path, backend=backend)


def test_rejects_reusing_an_existing_attempt_directory(proof_request):
    path, value, _ = proof_request
    output = Path(value["output"])
    output.mkdir(parents=True)
    (output / "prior").write_text("do not relabel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new proof attempt"):
        proof.run_proof(path, backend=SyntheticBackend())
