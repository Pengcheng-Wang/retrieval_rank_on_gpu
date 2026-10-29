"""Versioned I/O guard for the exact immutable P5 learner; no contract rewriting."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


BASE_HASHES = {
    "p5_assets.py": "d651ce24315c9ac2c8e1e3f8598eb11d8a269d6dd3c58524b28751d07cabe793",
    "p5_bank.py": "33b6a3e8948d983585829457f8db94c6055b51b52e3f22ace7606e76201f9c46",
    "p5_checkpoint.py": "f5aa8561ff20bb5df84675df3ee4ababbc2f5ee2dcef159de64e27442370e0c2",
    "p5_data.py": "7a6c8f67c3695205fb1a2dedfaad79ea8bfbff44f27c467374784d01ad78b977",
    "p5_gate.py": "fe9511e747e2e0ea908114f73cead8954dacb1aac287279dd576123c7755fbd6",
    "p5_index.py": "a434eaac6ea080792d4d8e440131fc0fbd0f81eb4f25a782b80586f9afb39fed",
    "p5_queue.py": "d48cfdfe372c72a5117ab71a2d7ec71203447aeee6f59f1fcc7953d326f4afd1",
    "p5_run.py": "6b9572e4cb3405b79c1247339bf78e21bbfb6904abf72ec30113533760c1c9b5",
    "p5_source.py": "e81f310eb56a2a347b325b854c1f4f11eb25d234a049e470617a2dd708b31f67",
}
BASE_RECIPE_SHA256 = "8e8d2900b299c7b3dd708126260c2ed0ee12b03ed31a25b33b7108b4be191f26"
PATCH_TARGETS = [
    "p5_checkpoint.save_checkpoint", "p5_checkpoint.restore_checkpoint",
    "p5_checkpoint.pointer_path", "p5_checkpoint.write_json",
    "p5_bank.result_snapshot_and_publication", "p5_queue.command",
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_base(base, seal_sha256):
    base = Path(base).resolve()
    scripts = base / "scripts" / "letter"
    for name, digest in BASE_HASHES.items():
        if _sha(scripts / name) != digest:
            raise ValueError(f"Unapproved immutable base source: {name}")
    marker = base / "p5-source-seal.json"
    if _sha(marker) != seal_sha256:
        raise ValueError("Immutable base seal differs from maintenance authorization")
    seal = json.loads(marker.read_text())
    for name, digest in seal["files_sha256"].items():
        if name.endswith(".py") or name.startswith(("configs/", "environments/")):
            if _sha(base / name) != digest:
                raise ValueError(f"Immutable base dependency changed: {name}")
    if _sha(base / "configs/letter-p5-reference-v1.json") != BASE_RECIPE_SHA256:
        raise ValueError("Maintenance cannot change the frozen scientific recipe")
    return base


def verify_release(manifest):
    manifest = Path(manifest).resolve()
    value = json.loads(manifest.read_text())
    if (value.get("kind") != "p5-crash-consistency-maintenance-v1"
            or value.get("patch_targets") != PATCH_TARGETS
            or set(value["files_sha256"]) != {"p5_maintenance.py", "p5_checkpoint.py", "p5_bank.py"}):
        raise ValueError("Unapproved maintenance scope")
    for name, digest in value["files_sha256"].items():
        if _sha(manifest.parent / name) != digest:
            raise ValueError("Maintenance source hash differs: " + name)
    if _sha(Path(__file__)) != value["files_sha256"]["p5_maintenance.py"]:
        raise ValueError("The invoked guard is not the attested guard")
    return value


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_rng_equal(a, b):
    import numpy as np
    import torch

    equal = (
        a["python"] == b["python"]
        and a["numpy"][0] == b["numpy"][0]
        and np.array_equal(a["numpy"][1], b["numpy"][1])
        and a["numpy"][2:] == b["numpy"][2:]
        and torch.equal(a["torch"], b["torch"])
        and len(a["cuda"]) == len(b["cuda"])
        and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    )
    if not equal:
        raise RuntimeError("The I/O guard changed a training RNG stream")


def wrap_child_command(original, base, manifest, proof):
    if len(original) < 2 or Path(original[1]).name != "p5_run.py":
        raise ValueError("Maintenance may launch only the unchanged learner through its guard")
    return [
        original[0], "-B", str(Path(__file__).resolve()),
        "--base-bundle", str(base), "--release-manifest", str(manifest),
        "--compatibility-proof", str(proof), "run", "--",
    ] + original[2:]


def install(base, manifest):
    release = verify_release(manifest)
    base = verify_base(base, release["base_source_seal_sha256"])
    folder = base / "scripts" / "letter"
    sys.dont_write_bytecode = True
    for name in ("p5_source", "p5_checkpoint", "p5_bank", "p5_queue", "p5_run"):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent != folder:
            raise RuntimeError("Maintenance requires a fresh process, not mixed source modules")
    sys.path.insert(0, str(folder))
    source = importlib.import_module("p5_source")
    original = importlib.import_module("p5_checkpoint")
    old_bank = importlib.import_module("p5_bank")
    queue = importlib.import_module("p5_queue")
    fixed = _module("p5_checkpoint_maintenance_v1", Path(manifest).parent / "p5_checkpoint.py")
    old_save, old_restore, old_write = original.save_checkpoint, original.restore_checkpoint, original.write_json

    def publish(path, value):
        path = Path(path)
        if path.name == "latest.json":
            artifact = path.parent / "states" / value["artifact"]
            original.verify_checkpoint(artifact)
            for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
                with (artifact / name).open("r+b") as stream:
                    os.fsync(stream.fileno())
            fixed.fsync_directory(artifact)
            fixed.fsync_directory(artifact.parent)
        old_write(path, value)
        if path.name in ("latest.json", "selected.json"):
            fixed.fsync_directory(path.parent)

    def save(run, model, optimizer, scheduler, sampler, state, contract, selected=False):
        before = original.rng_state()
        with fixed.pointer_lock(run):
            result = old_save(run, model, optimizer, scheduler, sampler, state, contract, selected)
            fixed._recover_selection_locked(run, contract)
        _assert_rng_equal(before, original.rng_state())
        return result

    def restore(path, model, optimizer, scheduler, sampler, contract):
        path = Path(path)
        run = path.parent.parent
        with fixed.pointer_lock(run):
            if fixed._pointer_path(run) != path:
                raise ValueError("Restore generation changed; do not reconcile a different latest")
            state = old_restore(path, model, optimizer, scheduler, sampler, contract)
            before = original.rng_state()
            fixed._recover_selection_locked(run, contract)
            _assert_rng_equal(before, original.rng_state())
            return state

    def pointer(run, pointer="latest"):
        before = original.rng_state()
        result = fixed.pointer_path(run, pointer)
        _assert_rng_equal(before, original.rng_state())
        return result

    original.write_json = publish
    original.save_checkpoint, original.restore_checkpoint, original.pointer_path = save, restore, pointer
    for name in ("pointer_lock", "fsync_directory", "recover_selection", "_recover_selection_locked"):
        setattr(original, name, getattr(fixed, name))
    fixed_bank = _module("p5_bank_maintenance_v1", Path(manifest).parent / "p5_bank.py")
    old_bank.bank_results, old_bank.watch = fixed_bank.bank_results, fixed_bank.watch
    old_bank.pointer_path = pointer
    learner = importlib.import_module("p5_run")
    if (learner.save_checkpoint is not save or learner.restore_checkpoint is not restore
            or learner.pointer_path is not pointer):
        raise RuntimeError("A learner imported before maintenance would bypass the guard")
    return SimpleNamespace(
        source=source, checkpoint=original, fixed_checkpoint=fixed, bank=fixed_bank,
        learner=learner, queue=queue, base=base, release=release,
        original_restore=old_restore, original_save=old_save,
    )


def verify_compatibility(manifest, proof):
    value = json.loads(Path(proof).read_text())
    expected_cells = {
        "Instruments-CID-s42", "Beauty-CID-s42", "Yelp-CID-s42",
        "Instruments-TID-s42", "Instruments-SemID-s42", "Beauty-TID-s42", "Yelp-TID-s42",
    }
    cells = value.get("cells", {})
    if (value.get("status") != "pass"
            or value.get("release_manifest_sha256") != _sha(manifest)
            or not value.get("model_optimizer_scheduler_sampler_rng_equal")
            or not value.get("subsequent_sampled_rows_and_losses_equal")
            or not value.get("scientific_contracts_unchanged")
            or set(cells) != expected_cells
            or any(cell.get("status") != "pass" or cell.get("updates_compared", 0) < 2
                   or not cell.get("model_optimizer_scheduler_sampler_rng_equal")
                   or not cell.get("sampled_rows_and_losses_equal")
                   or not cell.get("scientific_contract_unchanged") for cell in cells.values())):
        raise ValueError("A matching learned-state continuation proof is required")
    return value


def attest_run(context, run, manifest, proof):
    run = Path(run)
    binding = run / "p5-run-config.json"
    if not binding.exists():
        raise ValueError("Maintenance is restricted to existing bound reference runs")
    contract = context.source.read_json(binding)
    if (contract.get("driver_sha256") != BASE_HASHES
            or contract.get("recipe_sha256") != BASE_RECIPE_SHA256
            or contract.get("gate_subset") != 0):
        raise ValueError("Maintenance scientific contract differs from the frozen reference")
    attestation = {
        "kind": "p5-external-io-maintenance-v1",
        "scientific_contract_sha256": context.source.digest_json(contract),
        "scientific_contract_file_sha256": _sha(binding),
        "base_source_seal_sha256": context.release["base_source_seal_sha256"],
        "release_manifest_sha256": _sha(manifest), "compatibility_proof_sha256": _sha(proof),
        "base_driver_sha256": BASE_HASHES, "patch_targets": PATCH_TARGETS,
        "old_contract_not_rewritten": True,
    }
    marker = run / "maintenance-attestation.json"
    if marker.exists() and context.source.read_json(marker) != attestation:
        raise ValueError("A different maintenance attestation already owns this run")
    context.source.write_json(marker, attestation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--compatibility-proof", type=Path, required=True)
    parser.add_argument("mode", choices=("run", "bank", "queue"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trailing = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    verify_compatibility(args.release_manifest, args.compatibility_proof)
    context = install(args.base_bundle, args.release_manifest)
    if args.mode == "run":
        fit = context.learner.parser().parse_args(trailing)
        attest_run(context, fit.output, args.release_manifest, args.compatibility_proof)
        context.learner.train(fit)
    elif args.mode == "bank":
        bank_parser = argparse.ArgumentParser()
        for name in ("runs", "durable", "lock-file"):
            bank_parser.add_argument("--" + name, type=Path, required=True)
        bank_parser.add_argument("--writer-host", required=True)
        bank_parser.add_argument("--poll-seconds", type=int, default=30)
        bank_parser.add_argument("--once", action="store_true")
        context.bank.watch(bank_parser.parse_args(trailing))
    else:
        queue_parser = argparse.ArgumentParser()
        queue_parser.add_argument("--spec", type=Path, required=True)
        queue_args = queue_parser.parse_args(trailing)
        queue_args.gates_only, queue_args.prime_steps = False, 0
        original_command = context.queue.command

        def guarded_command(*values, **kwargs):
            return wrap_child_command(
                original_command(*values, **kwargs), args.base_bundle, args.release_manifest,
                args.compatibility_proof,
            )

        context.queue.command = guarded_command
        context.queue.queue(queue_args)


if __name__ == "__main__":
    main()
