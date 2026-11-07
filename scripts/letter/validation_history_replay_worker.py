"""Clean-process native P5 validation replay for recovery evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def load_proof_runner(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "validation_replay_proof_runner", path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_frozen_origins(backend, runtime_root):
    runtime_root = Path(runtime_root).resolve()
    origins = {
        "p5_run.py": Path(backend._learner.__file__).resolve(),
        "p5_checkpoint.py": Path(backend._checkpoint.__file__).resolve(),
    }
    source_module = sys.modules.get(backend._learner.read_json.__module__)
    if source_module is None or not getattr(source_module, "__file__", None):
        raise ValueError("Frozen learner source helper has no module origin")
    origins["p5_source.py"] = Path(source_module.__file__).resolve()
    for name, origin in origins.items():
        if origin != runtime_root / name:
            raise ValueError(f"Replay imported non-frozen runtime helper: {name}")
    return {name: str(origin) for name, origin in origins.items()}


def replay(request_path, proof_runner_path, output, response):
    proof = load_proof_runner(proof_runner_path)
    request = proof._request(request_path)
    backend = proof.NativeBackend()
    identity = backend.identity()
    expected = request["binding"]
    if any(
        identity.get(name) != expected[name]
        for name in ("host", "gpu", "gpu_uuid")
    ):
        raise ValueError("Validation replay ran on another GPU binding")
    loaded = backend.cold_load(request, "cold")
    origins = require_frozen_origins(
        backend, request["inputs"]["runtime"]["path"],
    )
    context = loaded["context"]
    metrics = backend._learner.evaluate(
        context.model,
        context.views["valid"],
        context.users,
        context.interactions,
        "valid",
        context.collator,
        context.tokenizer,
        context.catalog,
        context.native,
        context.recipe,
        Path(output),
    )
    state = loaded["state"]
    result = {
        "metrics": metrics,
        "state": {
            "phase": state.get("phase"),
            "epoch": state.get("epoch"),
            "validation_history": state.get("validation_history"),
        },
        "identity": identity,
        "module_origins": origins,
    }
    write(response, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--proof-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    replay(args.request, args.proof_runner, args.output, args.response)


if __name__ == "__main__":
    main()
