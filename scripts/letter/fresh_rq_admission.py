"""CPU-only fresh-ID controller with explicit, native-first RQ compatibility admission."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys
import types

import fresh_ids as native
from bank import source_manifest
from fresh_pipeline import Runtime, driver
from protocol import file_sha256, require_hash


HELPERS = ("rq_recovery_compat", "rq_recovery_runtime", "rq_recovery_control", "rq_recovery_lineage")
ADMISSION_EVIDENCE_EXTENSIONS = {
    "export_was_corrected", "native_artifact_identity_claimed",
    "downstream_ids_repair_dependent", "corrected_index_sha256",
    "corrected_codes_sha256", "corrections_by_epoch_level_phase",
}


@contextmanager
def bank_lock(path: Path):
    import fcntl

    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def require_unredirected(path: Path, root: Path, label: str) -> Path:
    absolute, root_absolute = path.absolute(), root.absolute()
    try:
        relative = absolute.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"{label} escapes its authorized root") from error
    current = root_absolute
    for part in (Path(), *relative.parts):
        candidate = current if part == Path() else current / part
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{label} contains a redirected path component")
        current = candidate
    resolved, resolved_root = path.resolve(), root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} resolves outside its authorized root")
    return resolved


def compatible_existing_lineage(existing: dict, current: dict) -> bool:
    if existing == current:
        return True
    if not isinstance(existing, dict) or not isinstance(current, dict):
        return False
    old_admission = existing.get("queue_admission")
    if (not isinstance(old_admission, dict)
            or not re.fullmatch(
                r"[a-f0-9]{64}", old_admission.get("capsule_bank_receipt_sha256", ""),
            )):
        return False
    existing_core = {
        key: value for key, value in existing.items() if key != "queue_admission"
    }
    current_core = {
        key: value for key, value in current.items()
        if key != "queue_admission" and key not in ADMISSION_EVIDENCE_EXTENSIONS
    }
    return existing_core == current_core


class RepairAdmission:
    def __init__(
        self, integration: Path, root: Path, destination: Path,
        expected_integration_sha256: str,
    ):
        if (not re.fullmatch(r"[a-f0-9]{64}", expected_integration_sha256)
                or file_sha256(integration) != expected_integration_sha256):
            raise ValueError("Compatibility integration is not externally authorized")
        self.integration = integration
        self.integration_sha256 = expected_integration_sha256
        self.settings = native.read_json(integration)
        self.root, self.destination = root.resolve(), destination
        self.native_verifier = native.verify_rq
        cfg = self.settings
        required = {
            "scope", "helper_directory_on_node0", "helper", "helper_sha256", "helper_files",
            "trusted_admission_sha256", "capsule", "durable_capsule", "resume_chains",
        }
        if not required <= cfg.keys():
            raise ValueError("The private compatibility integration contract is incomplete")
        if (cfg["helper"] != "rq_recovery_lineage.verify_rq_with_compatibility"
                or set(cfg["helper_files"]) != {name + ".py" for name in HELPERS}
                or cfg["helper_files"]["rq_recovery_lineage.py"] != cfg["helper_sha256"]
                or not re.fullmatch(r"[a-f0-9]{64}", cfg["trusted_admission_sha256"])):
            raise ValueError("Invalid trusted compatibility helper/admission identity")
        self.helper_dir = Path(cfg["helper_directory_on_node0"])
        self.capsule, self.durable = Path(cfg["capsule"]), Path(cfg["durable_capsule"])
        require_unredirected(integration, self.root, "Compatibility integration")
        self.helper_dir = require_unredirected(
            self.helper_dir, self.root, "Compatibility helper directory",
        )
        self.capsule = require_unredirected(
            self.capsule, self.root, "Compatibility capsule",
        )
        self.durable = require_unredirected(
            self.durable, destination.resolve(), "Durable compatibility capsule",
        )
        if (self.durable.parent != destination.resolve() / "rq-recovery-v1"
                or self.durable.name != self.capsule.name):
            raise ValueError("Compatibility source/capsule paths escape their private allocation binding")
        self.helper = self._load_helpers()
        if set(cfg["scope"]) != set(self.helper.RUNS) or len(cfg["scope"]) != len(self.helper.RUNS):
            raise ValueError("Compatibility scope must exactly match the two admitted native runs")
        self.scope = frozenset(cfg["scope"])
        if set(cfg["resume_chains"]) != self.scope:
            raise ValueError("Every repaired native run requires an externally sealed resume chain")

    def _load_helpers(self):
        for name in HELPERS:
            path = self.helper_dir / (name + ".py")
            digest = self.settings["helper_files"][path.name]
            require_hash(path, digest)
            if path.is_symlink():
                raise ValueError("Compatibility helper source is redirected")
            loaded = sys.modules.get(name)
            if loaded is not None:
                origin = getattr(loaded, "__file__", None)
                if origin is None or Path(origin).resolve() != path.resolve():
                    raise ValueError("Compatibility helper namespace is already bound to another source")
            else:
                loaded = types.ModuleType(name)
                loaded.__file__, loaded.__package__ = str(path), ""
                sys.modules[name] = loaded
                try:
                    exec(compile(path.read_bytes(), str(path), "exec"), loaded.__dict__)
                except BaseException:
                    del sys.modules[name]
                    raise
        return sys.modules["rq_recovery_lineage"]

    def verify_capsule(self) -> dict:
        require_hash(self.integration, self.integration_sha256)
        cfg = self.settings
        require_hash(self.capsule / "admission.json", cfg["trusted_admission_sha256"])
        admission = native.read_json(self.capsule / "admission.json")
        if admission.get("status") != "admitted":
            raise ValueError("Repair capsule has not been admitted")
        for name in ("source.zip", "gates.zip"):
            require_hash(self.capsule / name, admission["artifacts"][name])
        require_hash(self.capsule / "preparation.json", admission["prepared_sha256"])
        expected = source_manifest(self.capsule)
        with bank_lock(self.root / "bank.lock"):
            receipt_path = require_unredirected(
                self.durable / "bank-receipt.json", self.durable,
                "Durable compatibility receipt",
            )
            receipt = native.read_json(receipt_path)
            if receipt != expected:
                raise ValueError("Repair capsule is not bound to its complete durable bank")
            for name, metadata in expected["files"].items():
                path = require_unredirected(
                    self.durable / name, self.durable, "Durable compatibility file",
                )
                require_hash(path, metadata["sha256"])
            bank_sha = file_sha256(receipt_path)
        return {
            "integration_sha256": self.integration_sha256,
            "helper_sha256": cfg["helper_sha256"], "helper_files": cfg["helper_files"],
            "capsule_admission_sha256": cfg["trusted_admission_sha256"],
            "capsule_bank_receipt_sha256": bank_sha,
            "verification": "native full-budget gate, artifact attestations, then complete banked capsule",
        }

    def verify_resume_chain(self, bundle: Path, native_result: dict) -> dict:
        run_id = native_result["rq_run_id"]
        settings = self.settings["resume_chains"][run_id]
        source = require_unredirected(
            Path(settings["source"]), self.root, "Resume-chain evidence",
        )
        durable = require_unredirected(
            Path(settings["durable"]), self.destination.resolve(), "Durable resume-chain evidence",
        )
        manifest_path = source / "manifest.json"
        require_hash(manifest_path, settings["manifest_sha256"])
        expected_files = source_manifest(source)
        with bank_lock(self.root / "bank.lock"):
            receipt = native.read_json(durable / "bank-receipt.json")
            if receipt != expected_files:
                raise ValueError("Resume-chain evidence is not bound to its complete durable bank")
            for name, metadata in expected_files["files"].items():
                path = require_unredirected(
                    durable / name, durable, "Durable resume-chain file",
                )
                require_hash(path, metadata["sha256"])
                if path.stat().st_size != metadata["bytes"]:
                    raise ValueError("Durable resume-chain file byte length differs")
            bank_sha = file_sha256(durable / "bank-receipt.json")
        manifest = native.read_json(manifest_path)
        variant = self.helper.RUNS[run_id]
        fixed = self.helper.VARIANTS[variant]
        required = {
            "run_id": run_id, "base_epoch": 16400,
            "base_state_sha256": fixed["base_state"], "final_epoch": 20000,
        }
        if (manifest.get("kind") != "rq_recovery_resume_chain"
                or any(manifest.get(key) != value for key, value in required.items())):
            raise ValueError("Resume-chain manifest identifies another native continuation")
        checkpoints = {}
        for checkpoint_root in (bundle / "payload" / "checkpoints", source / "checkpoints"):
            for path in sorted(checkpoint_root.glob("epoch-*")):
                match = re.fullmatch(r"epoch-(\d{5})", path.name)
                if match is None:
                    continue
                epoch = int(match.group(1))
                if epoch <= 16400:
                    continue
                if str(epoch) in checkpoints:
                    raise ValueError("Resume-chain checkpoint epoch is duplicated across evidence roots")
                attestation = self.helper.verify_checkpoint_attestation(
                    path, run_id, native_result["run_config_sha256"],
                    self.settings["trusted_admission_sha256"],
                )
                checkpoints[str(epoch)] = {
                    "attestation_sha256":
                        file_sha256(path / "compatibility" / "attestation.json"),
                    "state_sha256": attestation["state_sha256"],
                    "invocation": attestation["invocation"],
                    "resumed_from_epoch": attestation["resumed_from_epoch"],
                    "resumed_from_state_sha256": attestation["resumed_from_state_sha256"],
                }
        if checkpoints != manifest.get("checkpoints"):
            raise ValueError("Resume-chain manifest does not enumerate every repaired checkpoint")
        available = {16400: fixed["base_state"]}
        produced = set()
        invocations = manifest.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            raise ValueError("Resume-chain manifest contains no exact restoration receipts")

        def verify_restore(restoration: dict, expected_epoch: int, expected_state: str) -> None:
            receipt_path = require_unredirected(
                source / restoration["restore_receipt"], source,
                "Exact restoration receipt",
            )
            require_hash(receipt_path, restoration["restore_receipt_sha256"])
            restored = native.read_json(receipt_path)
            if restored != {
                "status": "exact", "epoch": expected_epoch, "state_sha256": expected_state,
                "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
            }:
                raise ValueError("Resume-chain restoration was partial or did not restore all RNG state")

        for invocation in sorted(invocations, key=lambda value: min(value["produced_epochs"])):
            identity = invocation.get("invocation")
            epochs = invocation.get("produced_epochs")
            resume_epoch = invocation.get("resumed_from_epoch")
            resume_state = invocation.get("resumed_from_state_sha256")
            if (not isinstance(identity, str) or not isinstance(epochs, list) or not epochs
                    or epochs != sorted(set(epochs))
                    or available.get(resume_epoch) != resume_state):
                raise ValueError("Resume-chain invocation does not continue an available full state")
            verify_restore(invocation, resume_epoch, resume_state)
            for epoch in epochs:
                checkpoint = checkpoints.get(str(epoch))
                if (checkpoint is None or checkpoint["invocation"] != identity
                        or checkpoint["resumed_from_epoch"] != resume_epoch
                        or checkpoint["resumed_from_state_sha256"] != resume_state
                        or epoch <= resume_epoch or epoch in produced):
                    raise ValueError("Resume-chain checkpoint is missing, duplicated or cross-invocation")
                available[epoch] = checkpoint["state_sha256"]
                produced.add(epoch)
        if produced != {int(epoch) for epoch in checkpoints} or available.get(20000) is None:
            raise ValueError("Resume-chain does not reach the exact final epoch through all checkpoints")
        additional = manifest.get("additional_restores", [])
        if not isinstance(additional, list):
            raise ValueError("Resume-chain additional restoration evidence is malformed")
        restored_epochs = set()
        for restoration in additional:
            epoch = restoration.get("epoch")
            state = restoration.get("state_sha256")
            checkpoint = checkpoints.get(str(epoch))
            if (available.get(epoch) != state or checkpoint is None
                    or restoration.get("invocation") != checkpoint["invocation"]):
                raise ValueError("Additional restoration does not name a checkpoint in the exact chain")
            verify_restore(restoration, epoch, state)
            restored_epochs.add(epoch)
        selected_epoch = native_result["selected_epoch"]
        if selected_epoch > 16400 and selected_epoch not in restored_epochs:
            raise ValueError("Post-correction selected state lacks an exact export restoration receipt")
        return {
            "status": "exact_full_state_chain", "run_id": run_id,
            "base_epoch": 16400, "final_epoch": 20000,
            "invocations": len(invocations), "checkpoints": len(checkpoints),
            "additional_exact_restores": len(additional),
            "manifest_sha256": settings["manifest_sha256"],
            "bank_receipt_sha256": bank_sha,
        }

    def __call__(self, bundle, spec, manifest, dataset, variant):
        require_hash(self.integration, self.integration_sha256)
        if native.verify_rq is not self.native_verifier:
            raise ValueError("The original native verifier was rebound; refusing alias recursion")
        native_result = self.native_verifier(bundle, spec, manifest, dataset, variant)
        if native_result["rq_run_id"] not in self.scope:
            return native_result
        immutable_native = copy.deepcopy(native_result)
        self._load_helpers()
        result = self.helper.verify_rq_with_compatibility(
            bundle, spec, manifest, dataset, variant,
            admission_sha256=self.settings["trusted_admission_sha256"],
            native_lineage=immutable_native,
        )
        if immutable_native != native_result:
            raise ValueError("Compatibility verification mutated the native verification result")
        resume_chain = self.verify_resume_chain(bundle, immutable_native)
        proof = self.verify_capsule()
        if "rq_compatibility" not in result:
            raise ValueError("Repaired RQ verification lost the mandatory compatibility lineage")
        return {
            **result, "rq_compatibility": {
                **result["rq_compatibility"],
                "queue_admission": {**proof, "resume_chain": resume_chain},
            },
        }


class CompatibilityRuntime(Runtime):
    def __init__(
        self, binding: Path, project: Path, control: Path, work: Path, integration: Path,
        expected_integration_sha256: str,
    ):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("The compatibility controller must be CPU-only; child GPU leases remain explicit")
        super().__init__(binding, project, control, work)
        self.rq_verifier = RepairAdmission(
            integration, self.root, self.destination, expected_integration_sha256,
        )
        self.operational_admission = {
            "kind": "native_first_rq_compatibility_guard",
            "integration_sha256": expected_integration_sha256,
            "controller_adapter_sha256": file_sha256(Path(__file__)),
            "helper_sha256": self.rq_verifier.settings["helper_sha256"],
            "trusted_admission_sha256": self.rq_verifier.settings["trusted_admission_sha256"],
            "scope": sorted(self.rq_verifier.scope),
            "controller_cuda_visible_devices": "",
            "training_cuda_visible_devices": [2, 3],
        }

    def admit(self, cell: dict, bundle: Path) -> Path:
        path = self.work / "stages" / cell["stage_id"]
        if path.exists() and cell["rq_run_id"] in self.rq_verifier.scope:
            staged = native.verify_stage(path)
            verified = self.rq_verifier(
                bundle, self.spec, native.read_json(self.manifest), cell["dataset"], cell["variant"],
            )
            expected = verified["rq_compatibility"]
            lineage = staged["experiment_lineage"]
            existing = lineage.get("rq_compatibility")
            if (lineage["rq"].get("rq_compatibility") != existing
                    or not compatible_existing_lineage(existing, expected)):
                raise ValueError("An existing repaired-ID stage lacks this exact admitted lineage; never rewrite it")
        return super().admit(cell, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("binding", "project", "control", "work", "integration", "adoption"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--integration-sha256", required=True)
    parser.add_argument("--checks", type=int, default=432)
    args = parser.parse_args()
    if not 1 <= args.checks <= 1008:
        parser.error("Use 1..1008 bounded checks")
    runtime = CompatibilityRuntime(
        args.binding, args.project, args.control, args.work, args.integration,
        args.integration_sha256,
    )
    driver(runtime, args.checks, adoption=args.adoption)


if __name__ == "__main__":
    main()
