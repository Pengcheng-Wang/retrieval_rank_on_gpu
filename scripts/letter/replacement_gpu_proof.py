"""Isolated replacement-GPU continuation proof for a durable P5 full-state root."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


INPUT_NAMES = {
    "native", "upstream", "manifest", "recipe", "model", "corpus", "runtime",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def document_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n")
    partial.replace(path)


def _load_frozen_learner(runtime_root):
    runtime_root = Path(runtime_root).resolve()
    learner = runtime_root / "p5_run.py"
    if not learner.is_file():
        raise ValueError("Frozen runtime root lacks p5_run.py")
    spec = importlib.util.spec_from_file_location("p5_replacement_frozen_run", learner)
    module = importlib.util.module_from_spec(spec)
    original = list(sys.path)
    try:
        sys.path.insert(0, str(runtime_root))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original
    return module


def _checkpoint_header(checkpoint, contract):
    checkpoint = Path(checkpoint)
    seal_path = checkpoint / "p5-seal.json"
    seal = read(seal_path)
    files = seal.get("files_sha256", {})
    if (seal.get("kind") != "p5-full-state-v1"
            or seal.get("status") != "sealed"
            or set(files) != {"model.safetensors", "training-state.pt"}):
        raise ValueError("Source is not a sealed P5 full-state checkpoint")
    for name, expected in files.items():
        if sha(checkpoint / name) != expected:
            raise ValueError("Source checkpoint bytes differ from its seal")
    if document_sha(read(contract)) != seal.get("contract_sha256"):
        raise ValueError("Source checkpoint and scientific contract differ")
    return seal


def _request(path):
    value = read(path)
    required = {
        "kind", "cell", "binding", "checkpoint", "contract", "contract_sha256",
        "inputs", "output", "microbatch_size",
    }
    if set(value) != required or value["kind"] != "p5-replacement-gpu-proof-request-v1":
        raise ValueError("Unknown replacement-GPU proof request")
    binding = value["binding"]
    if (set(binding) != {"host", "gpu", "gpu_uuid", "control_root"}
            or not isinstance(binding["host"], str) or not binding["host"]
            or type(binding["gpu"]) is not int or binding["gpu"] < 0
            or not isinstance(binding["gpu_uuid"], str) or not binding["gpu_uuid"]):
        raise ValueError("Private current host/index/UUID binding is incomplete")
    if type(value["microbatch_size"]) is not int or value["microbatch_size"] not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("Replacement proof must preserve a legal P5 microbatch")
    if set(value["inputs"]) != INPUT_NAMES:
        raise ValueError("Exact corpus/model/runtime input bindings are required")
    for name, item in value["inputs"].items():
        if set(item) != {"path", "anchor", "anchor_sha256"}:
            raise ValueError(f"Invalid {name} input binding")
        root, anchor = Path(item["path"]), Path(item["anchor"])
        if (not root.is_absolute() or not anchor.is_absolute()
                or anchor.resolve() != root.resolve()
                and not anchor.resolve().is_relative_to(root.resolve())
                or not anchor.is_file() or sha(anchor) != item["anchor_sha256"]):
            raise ValueError(f"Exact {name} input anchor changed")
    runtime = value["inputs"]["runtime"]
    if Path(runtime["anchor"]).resolve() != Path(runtime["path"]).resolve() / "p5_run.py":
        raise ValueError("Runtime anchor must be the private frozen p5_run.py")
    checkpoint, contract = Path(value["checkpoint"]), Path(value["contract"])
    if not checkpoint.is_absolute() or not contract.is_absolute():
        raise ValueError("Checkpoint and contract paths must be absolute private bindings")
    if sha(contract) != value["contract_sha256"]:
        raise ValueError("Scientific contract file changed")
    output = Path(value["output"]).resolve()
    gates = Path(binding["control_root"]).resolve() / "gates"
    if output == gates or not output.is_relative_to(gates):
        raise ValueError("Proof output must be a fresh child of control_root/gates")
    if output.exists():
        raise FileExistsError("Use a new proof attempt directory; never relabel an existing attempt")
    return value


def _equal(left, right):
    try:
        import numpy as np
        import torch
    except ImportError:
        np = torch = None
    if torch is not None and torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if np is not None and isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _require_equal(left, right, label):
    if not _equal(left, right):
        raise AssertionError(f"Replacement proof {label} differs")


def _valid_optimizer_memory(value):
    return (
        isinstance(value, dict)
        and value.get("populated") is True
        and type(value.get("parameter_states")) is int
        and value["parameter_states"] > 0
        and type(value.get("tensor_values")) is int
        and value["tensor_values"] > 0
    )


def _valid_geometry(value):
    return (
        isinstance(value, dict)
        and value.get("actual_generation") is True
        and value.get("beams") == 20
        and value.get("returns") == 20
        and type(value.get("probe_rows")) is int
        and value["probe_rows"] > 0
        and value.get("complete_sequences") == value["probe_rows"] * 20
        and value.get("complete_catalog_sequences_only") is True
        and value.get("longest_catalog_target_completed") is True
    )


def run_proof(request_path, backend=None):
    request = _request(request_path)
    seal = _checkpoint_header(request["checkpoint"], request["contract"])
    backend = NativeBackend() if backend is None else backend
    identity = backend.identity()
    expected = request["binding"]
    if any(identity.get(key) != expected[key] for key in ("host", "gpu", "gpu_uuid")):
        raise ValueError("Proof did not run on the private current host/index/UUID binding")
    output = Path(request["output"])
    output.mkdir(parents=True)
    first = backend.cold_load(request, "cold-a")
    second = backend.cold_load(request, "cold-b")
    _require_equal(first["snapshot"], second["snapshot"], "cold-loaded full state")
    if not (_valid_optimizer_memory(first["optimizer_memory"])
            and _valid_optimizer_memory(second["optimizer_memory"])):
        raise ValueError("Cold-loaded optimizer memory is absent or partial")
    geometry = backend.geometry(second)
    if not _valid_geometry(geometry):
        raise ValueError("Replacement proof lacks actual complete 20-beam geometry")
    first_update = backend.next_update(first)
    second_update = backend.next_update(second)
    _require_equal(first_update, second_update, "next sampled rows/loss/update")
    result = {
        "kind": "p5-replacement-gpu-proof-v1",
        "status": "pass",
        "cell": request["cell"],
        "host": identity["host"],
        "gpu": identity["gpu"],
        "gpu_uuid": identity["gpu_uuid"],
        "gpu_name": identity["gpu_name"],
        "source_checkpoint_seal_sha256": sha(Path(request["checkpoint"]) / "p5-seal.json"),
        "contract_sha256": seal["contract_sha256"],
        "model_sha256": seal["files_sha256"]["model.safetensors"],
        "training_state_sha256": seal["files_sha256"]["training-state.pt"],
        "input_bindings_sha256": document_sha(request["inputs"]),
        "microbatch_size": request["microbatch_size"],
        "cold_loads": ["cold-a", "cold-b"],
        "cold_save_reload_bitwise_equal": True,
        "next_sampled_rows_and_losses_equal": True,
        "populated_optimizer_memory_pass": True,
        "complete_20beam_geometry_pass": True,
        "optimizer_memory": second["optimizer_memory"],
        "next_update": second_update,
        "beam_geometry": geometry,
    }
    write(output / "p5-replacement-gpu-proof.json", result)
    return result


def _fingerprint(value):
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(canonical(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(str(array.dtype).encode())
            digest.update(canonical(list(array.shape)))
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(canonical(item))

    visit(value)
    return digest.hexdigest()


class NativeBackend:
    def __init__(self):
        self._setup = None
        self._learner = None
        self._checkpoint = None

    def identity(self):
        import torch

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit() or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one explicitly bound physical CUDA GPU")
        gpu = int(visible)
        fields = subprocess.check_output([
            "nvidia-smi", "--id=" + visible, "--query-gpu=uuid,name", "--format=csv,noheader",
        ], text=True).strip().split(",", 1)
        if len(fields) != 2:
            raise RuntimeError("Unable to identify the replacement GPU")
        return {
            "host": socket.gethostname(),
            "gpu": gpu,
            "gpu_uuid": fields[0].strip(),
            "gpu_name": fields[1].strip(),
        }

    @staticmethod
    def _adapted_setup(request, learner_module):
        runtime = request["inputs"]["runtime"]
        learner = Path(learner_module.__file__).resolve()
        if Path(runtime["anchor"]).resolve() != learner:
            raise ValueError("Loaded learner differs from the frozen runtime anchor")
        tree = ast.parse(learner.read_text(encoding="utf-8"))
        setup = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup"
        )
        legacy = ast.parse(
            'os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", "5"}',
            mode="eval",
        ).body
        changes = 0

        class ReplaceGuard(ast.NodeTransformer):
            def visit_If(self, node):
                nonlocal changes
                if ast.dump(node.test) == ast.dump(legacy):
                    changes += 1
                    return ast.copy_location(
                        ast.parse("_replacement_require_device(args)").body[0], node,
                    )
                return self.generic_visit(node)

        adapted = ReplaceGuard().visit(copy.deepcopy(setup))
        ast.fix_missing_locations(adapted)
        if changes != 1:
            raise ValueError("Frozen learner device-guard boundary changed")
        binding = request["binding"]

        def device_check(args):
            if (os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"])
                    or args.microbatch_size != request["microbatch_size"]
                    or args.gate_subset != 0):
                raise ValueError("Replacement proof changed device, batch, or full-data contract")

        namespace = dict(learner_module.__dict__)
        namespace["_replacement_require_device"] = device_check
        exec(compile(ast.Module(body=[adapted], type_ignores=[]), str(learner), "exec"), namespace)
        return namespace["setup"]

    def cold_load(self, request, label):
        if self._setup is None:
            self._learner = _load_frozen_learner(request["inputs"]["runtime"]["path"])
            self._checkpoint = sys.modules[self._learner.restore_checkpoint.__module__]
            self._setup = self._adapted_setup(request, self._learner)
        inputs = request["inputs"]
        args = SimpleNamespace(
            **{name: Path(inputs[name]["path"]) for name in (
                "native", "upstream", "manifest", "recipe", "model", "corpus"
            )},
            output=Path(request["output"]) / label,
            gate=None,
            resume=None,
            gate_subset=0,
            stop_steps=None,
            microbatch_size=request["microbatch_size"],
        )
        context = self._setup(args)
        contract = read(request["contract"])
        if context.contract != contract:
            raise ValueError("Cold-loaded runtime reconstructed a different scientific contract")
        state, captured_rng = self._checkpoint._restore_checkpoint_locked(
            Path(request["checkpoint"]),
            context.model,
            context.optimizer,
            context.scheduler,
            context.sampler,
            contract,
        )
        self._checkpoint.restore_rng(captured_rng)
        snapshot = {
            "model": _fingerprint(context.model.state_dict()),
            "optimizer": _fingerprint(context.optimizer.state_dict()),
            "scheduler": _fingerprint(context.scheduler.state_dict()),
            "sampler": _fingerprint(context.sampler.state_dict()),
            "rng": _fingerprint(self._checkpoint.rng_state()),
            "training_state": _fingerprint(state),
            "validation_state": _fingerprint({
                "best_metric": state.get("best_metric"),
                "selected_artifact": state.get("selected_artifact"),
                "validation_history": state.get("validation_history"),
                "test_started": state.get("test_started"),
            }),
        }
        parameter_states = len(context.optimizer.state)
        tensor_values = 0
        moments = 0
        import torch
        for values in context.optimizer.state.values():
            for name, value in values.items():
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():
                        raise ValueError("Optimizer memory contains nonfinite state")
                    tensor_values += value.numel()
                    if name in ("exp_avg", "exp_avg_sq") and value.numel():
                        moments += 1
        return {
            "context": context,
            "state": state,
            "captured_rng": captured_rng,
            "snapshot": snapshot,
            "optimizer_memory": {
                "populated": parameter_states > 0 and tensor_values > 0 and moments > 0,
                "parameter_states": parameter_states,
                "tensor_values": tensor_values,
                "moment_tensors": moments,
            },
        }

    def next_update(self, loaded):
        context, state = loaded["context"], loaded["state"]
        self._checkpoint.restore_rng(loaded["captured_rng"])
        indices = context.sampler.next(context.recipe["training"]["effective_batch_size"])
        if not indices:
            raise ValueError("Chosen root has no next training batch in its current epoch")
        result = self._learner.update(
            context.model,
            context.optimizer,
            context.scheduler,
            context.views["train"],
            indices,
            context.collator,
            loaded["context"].contract["operating_microbatch_size"],
            context.recipe["training"]["clip_norm"],
        )
        state["global_step"] += 1
        state["examples_seen"] += len(indices)
        resulting_state = {
            "model": _fingerprint(context.model.state_dict()),
            "optimizer": _fingerprint(context.optimizer.state_dict()),
            "scheduler": _fingerprint(context.scheduler.state_dict()),
            "sampler": _fingerprint(context.sampler.state_dict()),
            "rng": _fingerprint(self._checkpoint.rng_state()),
            "training_state": _fingerprint(state),
        }
        return {
            "sampled_rows": indices,
            "sampled_rows_sha256": result["sampled_rows_sha256"],
            "loss": result["loss"],
            "update": result,
            "resulting_state": resulting_state,
        }

    def geometry(self, loaded):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        context = loaded["context"]
        if (context.recipe["evaluation"]["beams"] != 20
                or context.recipe["evaluation"]["returns"] != 20):
            raise ValueError("Scientific contract no longer specifies complete 20-beam geometry")
        model = context.model
        model.eval()
        rows = [context.views["valid"][index] for index in range(2)]
        batch = tuple(t.to("cuda") for t in context.collator(rows))
        longest = max(context.catalog["token_ids"], key=len)

        class ForceLongestCatalogPath(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for index, prefix in enumerate(input_ids.tolist()):
                    prefix = prefix[1:]
                    if len(prefix) < len(longest) and prefix == longest[:len(prefix)]:
                        token = longest[len(prefix)]
                        if not torch.isfinite(scores[index, token]):
                            raise AssertionError("Longest catalog path is unavailable")
                        scores[index, token] += 100.
                return scores

        with torch.inference_mode():
            generated = self._learner.generation(
                model, batch, context.tokenizer, context.catalog, context.native,
                context.recipe, cache=True,
            )
            forced = self._learner.generation(
                model, tuple(t[:1] for t in batch), context.tokenizer, context.catalog,
                context.native, context.recipe, cache=True,
                logits_processor=LogitsProcessorList([ForceLongestCatalogPath()]),
            )
        completed = [row[1:row.index(1, 1) + 1] for row in forced.sequences.tolist()]
        return {
            "actual_generation": True,
            "beams": 20,
            "returns": 20,
            "probe_rows": 2,
            "complete_sequences": generated.sequences.shape[0],
            "complete_catalog_sequences_only": True,
            "longest_catalog_target_completed": longest in completed,
        }


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--request", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    completed = run_proof(arguments.request)
    print(
        "P5_REPLACEMENT_GPU_PROOF_PASS "
        f"{completed['cell']} {completed['host']} GPU{completed['gpu']} "
        f"{completed['gpu_uuid']}",
        flush=True,
    )
