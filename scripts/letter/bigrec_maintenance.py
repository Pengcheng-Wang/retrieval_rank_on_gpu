"""Versioned terminal-only overlay; original BIGRec files and bindings stay intact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace


BASE_TRAIN_SHA256 = "210a2b9ef7b37c2a89d3d866f6a65c76ebb7c0ed80ee4a071338a0725860b27f"
BASE_PIPELINE_SHA256 = "a72d250fda706a1422e1c18c39b3ab6a24038df30098ac2fa24965bc2459ae3d"
OVERLAY_FILES = ("bigrec_maintenance.py", "bigrec_terminal.py")
BASE_FILES = (
    "scripts/letter/bigrec_train.py", "scripts/letter/bigrec_checkpoint.py",
    "scripts/letter/bigrec_data.py", "scripts/letter/bigrec_source.py",
    "scripts/letter/bigrec_evaluate.py", "scripts/letter/bigrec_gate.py",
    "scripts/letter/bigrec_bank.py", "scripts/letter/bigrec_pipeline.py",
    "scripts/letter/protocol.py", "scripts/letter/prepare.py", "scripts/letter/author.py",
    "scripts/letter/runtime_patch.py", "scripts/letter/bank.py", "src/rrbench/io.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def patched_training_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != BASE_TRAIN_SHA256:
        raise ValueError("Terminal overlay supports only the independently verified frozen training bytes")
    replacements = (
        (b"def fit(args) -> None:\n    report = verify_admission(args.admission)\n",
         b"def fit(args) -> None:\n"
         b"    if _terminal.finish_if_terminal(args, _maintenance_binding):\n"
         b"        return\n"
         b"    report = verify_admission(args.admission)\n"),
        (b"            seal(checkpoint)\n",
         b"            _terminal.persist_decision(checkpoint, recipe, _maintenance_binding)\n"
         b"            seal(checkpoint)\n"),
        (b"    trainer.train(resume_from_checkpoint=str(resume) if resume else None)\n",
         b"    trainer.train(resume_from_checkpoint=str(resume) if resume else None)\n"
         b"    if _terminal.finish_if_terminal(args, _maintenance_binding, phase=\"after_training\"):\n"
         b"        return\n"),
    )
    for before, after in replacements:
        if b"\r\n" in source:
            before, after = before.replace(b"\n", b"\r\n"), after.replace(b"\n", b"\r\n")
        if source.count(before) != 1:
            raise ValueError("A frozen terminal-maintenance insertion anchor changed")
        source = source.replace(before, after)
    return source


def maintained_module(frozen_train: Path, binding: dict) -> ModuleType:
    helper = Path(__file__).with_name("bigrec_terminal.py")
    spec = importlib.util.spec_from_file_location("_bigrec_terminal_versioned_helper", helper)
    if spec is None or spec.loader is None:
        raise ImportError("Versioned terminal helper is unavailable")
    bigrec_terminal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bigrec_terminal)

    source = patched_training_source(frozen_train.read_bytes())
    module = ModuleType("_bigrec_terminal_maintained_frozen_v3")
    module.__file__ = str(frozen_train)
    module.__dict__.update(_terminal=bigrec_terminal, _maintenance_binding=binding)
    exec(compile(source, str(frozen_train) + "#terminal-maintenance-v1", "exec"), module.__dict__)
    return module


def _write(path: Path, value: dict) -> None:
    from protocol import write_json
    write_json(path, value)


def make_manifest(
    project: Path, output: Path, runs: list[Path], *, candidate: bool,
    evidence: list[Path] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError("Maintenance manifests are immutable")
    base = {name: sha(project / name) for name in BASE_FILES}
    if (base["scripts/letter/bigrec_train.py"] != BASE_TRAIN_SHA256
            or base["scripts/letter/bigrec_pipeline.py"] != BASE_PIPELINE_SHA256):
        raise ValueError("The frozen base implementation has changed")
    from bigrec_checkpoint import latest, verify
    allowed = {}
    for run in runs:
        if candidate and "gates" not in run.resolve().parts:
            raise ValueError("Candidate maintenance is limited to isolated gate runs")
        checkpoint = latest(run)
        if checkpoint is None:
            raise ValueError("Maintenance cannot restart a fit without its existing complete state")
        verify(checkpoint)
        allowed[str(run.resolve())] = {
            "run_config_sha256": sha(run / "run-config.json"),
            "activation_checkpoint": checkpoint.name,
            "activation_checkpoint_sha256": sha(checkpoint / "complete.json"),
        }
    proofs, proof_bodies = {}, []
    for path in evidence or []:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            raise ValueError("Maintenance proof did not pass")
        proofs[str(path.resolve())] = sha(path)
        proof_bodies.append(report)
    effective = hashlib.sha256(
        patched_training_source((project / "scripts/letter/bigrec_train.py").read_bytes())
    ).hexdigest()
    overlay = {name: sha(Path(__file__).parent / name) for name in OVERLAY_FILES}
    if not candidate:
        kinds = {report.get("kind") for report in proof_bodies}
        if kinds != {"bigrec_native_terminal_faults", "bigrec_native_maintenance_continuation"}:
            raise ValueError("Production maintenance requires native terminal and continuation proofs")
        for report in proof_bodies:
            if (report.get("base_train_sha256") != BASE_TRAIN_SHA256
                    or report.get("effective_training_sha256") != effective
                    or report.get("overlay_files") != overlay):
                raise ValueError("Native proof is bound to different maintenance bytes")
        continuation = next(p for p in proof_bodies if p["kind"] == "bigrec_native_maintenance_continuation")
        for run in runs:
            config = json.loads((run / "run-config.json").read_text(encoding="utf-8"))
            case = continuation["datasets"].get(config["dataset"])
            if (case is None or case.get("status") != "pass"
                    or case["original_checkpoint_sha256"] != allowed[str(run.resolve())]["activation_checkpoint_sha256"]):
                raise ValueError("Actual drained fit lacks a matching native before/after continuation proof")
    result = {
        "schema_version": 1, "id": "bigrec-terminal-v1", "status": "candidate" if candidate else "admitted",
        "project": str(project.resolve()), "base_files": base,
        "overlay_files": overlay, "effective_training_sha256": effective,
        "allowed_runs": allowed, "evidence": proofs, "created_unix": time.time(),
        "policy": "No math/budget change; terminal sealed decision bypasses train and outranks drain",
    }
    _write(output, result)
    return result


def load_manifest(path: Path, run: Path, allow_candidate: bool = False) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["id"] != "bigrec-terminal-v1" or (
        manifest["status"] != "admitted" and not (allow_candidate and manifest["status"] == "candidate")
    ):
        raise ValueError("Unadmitted terminal maintenance deployment")
    project = Path(manifest["project"])
    for name, digest in manifest["base_files"].items():
        if sha(project / name) != digest:
            raise ValueError(f"Frozen base bytes changed: {name}")
    for name, digest in manifest["overlay_files"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Versioned maintenance bytes changed: {name}")
    for name, digest in manifest["evidence"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Native maintenance evidence changed after admission")
    effective = hashlib.sha256(
        patched_training_source((project / "scripts/letter/bigrec_train.py").read_bytes())
    ).hexdigest()
    if effective != manifest["effective_training_sha256"]:
        raise ValueError("Effective terminal-maintenance implementation differs")
    entry = manifest["allowed_runs"].get(str(run.resolve()))
    if entry is None or sha(run / "run-config.json") != entry["run_config_sha256"]:
        raise ValueError("Maintenance cannot change or replace the frozen run configuration")
    binding = {
        "id": manifest["id"], "deployment_sha256": sha(path),
        "base_training_sha256": BASE_TRAIN_SHA256, "effective_training_sha256": effective,
        "overlay_files": manifest["overlay_files"],
    }
    return manifest, binding


def bind_activation(run: Path, manifest: dict, binding: dict) -> None:
    from bigrec_checkpoint import latest, verify

    path = run / "terminal-maintenance-activation.json"
    entry = manifest["allowed_runs"][str(run.resolve())]
    value = {"maintenance": binding, "original_run_config_sha256": entry["run_config_sha256"],
             "initial_checkpoint": entry["activation_checkpoint"],
             "initial_checkpoint_sha256": entry["activation_checkpoint_sha256"]}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Run is already bound to different immutable maintenance")
        return
    checkpoint = latest(run)
    if (checkpoint is None or checkpoint.name != entry["activation_checkpoint"]
            or sha(checkpoint / "complete.json") != entry["activation_checkpoint_sha256"]):
        raise ValueError("Fit state changed before maintenance activation")
    verify(checkpoint)
    _write(path, value)


def trace_training(path: Path):
    from transformers import Trainer
    from bigrec_train import batch_digest

    if path.exists():
        raise FileExistsError("Continuation proof traces are immutable")
    original = Trainer.training_step

    def observed(trainer, model, inputs, *args, **kwargs):
        digest = batch_digest(inputs)
        loss = original(trainer, model, inputs, *args, **kwargs)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step_before": trainer.state.global_step, "batch": digest,
                                     "loss": float(loss.detach().cpu())}) + "\n")
        return loss

    Trainer.training_step = observed
    return original


def run_train(args) -> None:
    manifest, binding = load_manifest(args.deployment, args.output, args.candidate)
    bind_activation(args.output, manifest, binding)
    if not args.resume:
        raise ValueError("State-preserving maintenance must resume, never start a new whole fit")
    project = Path(manifest["project"])
    module = maintained_module(project / "scripts/letter/bigrec_train.py", binding)
    original_step = trace_training(args.trace_output) if args.trace_output else None
    try:
        module.fit(args)
    finally:
        if original_step is not None:
            from transformers import Trainer
            Trainer.training_step = original_step


def run_pipeline(args) -> None:
    manifest, binding = load_manifest(args.deployment, args.run)
    bind_activation(args.run, manifest, binding)
    import bigrec_pipeline

    project = Path(manifest["project"])
    expected = str(project / "scripts/letter/bigrec_train.py")
    original = subprocess.Popen

    def maintained_popen(command, *values, **options):
        if isinstance(command, list) and len(command) > 1 and command[1] == expected:
            command = [
                command[0], str(Path(__file__).resolve()), "--project", str(project), "train",
                "--deployment", str(args.deployment), *command[2:],
            ]
        return original(command, *values, **options)

    subprocess.Popen = maintained_popen
    try:
        bigrec_pipeline.supervise(args)
    finally:
        subprocess.Popen = original


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--project", type=Path, required=True)
    boot, remaining = bootstrap.parse_known_args()
    sys.path.insert(0, str(boot.project.resolve() / "scripts/letter"))
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("manifest")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--runs", nargs="+", type=Path, required=True)
    make.add_argument("--candidate", action="store_true")
    make.add_argument("--evidence", nargs="+", type=Path)
    train = commands.add_parser("train")
    for name in ("deployment", "admission", "model", "output", "outbox", "gates"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--resume", required=True)
    train.add_argument("--stop-after-steps", type=int)
    train.add_argument("--gate-run", action="store_true")
    train.add_argument("--candidate", action="store_true")
    train.add_argument("--trace-output", type=Path)
    pipeline = commands.add_parser("pipeline")
    for name in ("deployment", "admission", "model", "gates", "run", "evaluation", "outbox", "catalog"):
        pipeline.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(remaining)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command == "manifest":
        make_manifest(boot.project.resolve(), args.output, args.runs,
                      candidate=args.candidate, evidence=args.evidence)
    elif args.command == "train":
        run_train(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
