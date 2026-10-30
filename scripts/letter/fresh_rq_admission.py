"""CPU-only fresh-ID controller with explicit, native-first RQ compatibility admission."""

from __future__ import annotations

import argparse
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


@contextmanager
def bank_lock(path: Path):
    import fcntl

    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class RepairAdmission:
    def __init__(self, integration: Path, root: Path, destination: Path):
        self.integration = integration
        self.integration_sha256 = file_sha256(integration)
        self.settings = native.read_json(integration)
        self.root, self.destination = root.resolve(), destination
        self.native_verifier = native.verify_rq
        cfg = self.settings
        required = {
            "scope", "helper_directory_on_node0", "helper", "helper_sha256", "helper_files",
            "trusted_admission_sha256", "capsule", "durable_capsule",
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
        if (not self.helper_dir.resolve().is_relative_to(self.root)
                or not self.capsule.resolve().is_relative_to(self.root)
                or self.durable.parent != destination / "rq-recovery-v1"
                or self.durable.name != self.capsule.name):
            raise ValueError("Compatibility source/capsule paths escape their private allocation binding")
        self.helper = self._load_helpers()
        if set(cfg["scope"]) != set(self.helper.RUNS) or len(cfg["scope"]) != len(self.helper.RUNS):
            raise ValueError("Compatibility scope must exactly match the two admitted native runs")
        self.scope = frozenset(cfg["scope"])

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
            receipt = native.read_json(self.durable / "bank-receipt.json")
            if receipt != expected:
                raise ValueError("Repair capsule is not bound to its complete durable bank")
            for name, metadata in expected["files"].items():
                require_hash(self.durable / name, metadata["sha256"])
            bank_sha = file_sha256(self.durable / "bank-receipt.json")
        return {
            "integration_sha256": self.integration_sha256,
            "helper_sha256": cfg["helper_sha256"], "helper_files": cfg["helper_files"],
            "capsule_admission_sha256": cfg["trusted_admission_sha256"],
            "capsule_bank_receipt_sha256": bank_sha,
            "verification": "native full-budget gate, artifact attestations, then complete banked capsule",
        }

    def __call__(self, bundle, spec, manifest, dataset, variant):
        require_hash(self.integration, self.integration_sha256)
        if native.verify_rq is not self.native_verifier:
            raise ValueError("The original native verifier was rebound; refusing alias recursion")
        self._load_helpers()
        # This pinned wrapper invokes fresh_ids.verify_rq first. That binding is never replaced.
        result = self.helper.verify_rq_with_compatibility(
            bundle, spec, manifest, dataset, variant,
            admission_sha256=self.settings["trusted_admission_sha256"],
        )
        if result["rq_run_id"] not in self.scope:
            return result
        proof = self.verify_capsule()
        if "rq_compatibility" not in result:
            raise ValueError("Repaired RQ verification lost the mandatory compatibility lineage")
        return {**result, "rq_compatibility": {**result["rq_compatibility"], "queue_admission": proof}}


class CompatibilityRuntime(Runtime):
    def __init__(self, binding: Path, project: Path, control: Path, work: Path, integration: Path):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("The compatibility controller must be CPU-only; child GPU leases remain explicit")
        super().__init__(binding, project, control, work)
        self.rq_verifier = RepairAdmission(integration, self.root, self.destination)
        self.operational_admission = {
            "kind": "native_first_rq_compatibility_guard",
            "integration_sha256": file_sha256(integration),
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
            if (lineage.get("rq_compatibility") != expected
                    or lineage["rq"].get("rq_compatibility") != expected):
                raise ValueError("An existing repaired-ID stage lacks this exact admitted lineage; never rewrite it")
        return super().admit(cell, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("binding", "project", "control", "work", "integration", "adoption"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checks", type=int, default=432)
    args = parser.parse_args()
    if not 1 <= args.checks <= 1008:
        parser.error("Use 1..1008 bounded checks")
    runtime = CompatibilityRuntime(args.binding, args.project, args.control, args.work, args.integration)
    driver(runtime, args.checks, adoption=args.adoption)


if __name__ == "__main__":
    main()
