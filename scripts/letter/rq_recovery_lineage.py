"""Additive provenance admission for the two compatibility-corrected RQ runs."""

from __future__ import annotations

import re
from pathlib import Path

from rq_recovery_compat import POLICY
from rq_recovery_control import HARNESS, ORIGINAL_SEAL, VARIANTS, read_json, require_sha
from rq_recovery_runtime import file_manifest, sha


RUNS = {f"openllama-v2-rq-Instruments-{variant}-s42": variant for variant in VARIANTS}


def _verify_common(directory, run_id, config_sha256, admission_sha256, artifact):
    if run_id not in RUNS or not isinstance(admission_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", admission_sha256,
    ):
        raise ValueError("A trusted admission digest is mandatory for these two repaired runs")
    fixed = VARIANTS[RUNS[run_id]]
    path = directory / "compatibility"
    marker = path / "attestation.json"
    result = read_json(marker)
    required = {
        "kind": "rq_compatibility_attestation", "purpose": "production", "artifact": artifact,
        "policy": POLICY, "run_id": run_id, "run_config_sha256": fixed["config"],
        "base_epoch": 16400, "base_state_sha256": fixed["base_state"],
        "original_seal_sha256": ORIGINAL_SEAL, "native_harness_sha256": HARNESS,
        "admission_sha256": admission_sha256,
    }
    if config_sha256 != fixed["config"] or any(result.get(key) != value for key, value in required.items()):
        raise ValueError("Compatibility evidence is not bound to this unchanged native contract")
    actual = file_manifest(path)
    actual.pop("attestation.json", None)
    if (actual != result["files"] or not {
        "source.zip", "gates.zip", "admission.json", "events.zip",
    }.issubset(actual)):
        raise ValueError("Compatibility evidence files are missing, changed or untracked")
    require_sha(path / "admission.json", admission_sha256)
    admission = read_json(path / "admission.json")
    if (admission.get("status") != "admitted" or admission.get("policy") != POLICY
            or admission.get("original_seal_sha256") != ORIGINAL_SEAL
            or admission.get("variants") != VARIANTS):
        raise ValueError("Compatibility admission identifies a different recovery")
    for name in ("source.zip", "gates.zip"):
        require_sha(path / name, admission["artifacts"][name])
    return result


def verify_checkpoint_attestation(checkpoint, run_id, config_sha256, admission_sha256):
    """Compose with native full-state verification; never reinterpret its state keys."""
    result = _verify_common(checkpoint, run_id, config_sha256, admission_sha256, "checkpoint")
    if not re.fullmatch(r"epoch-\d{5}", checkpoint.name):
        raise ValueError("Malformed repaired checkpoint directory")
    epoch = int(checkpoint.name.split("-")[1])
    receipt = read_json(checkpoint / "completed.json")
    if (not 16400 < epoch <= 20000 or result.get("epoch") != epoch
            or receipt.get("epoch") != epoch - 1 or receipt.get("status") != "complete"
            or receipt.get("kind") != "tokenizer_epoch_checkpoint"
            or result.get("state_sha256") != receipt.get("state_sha256")):
        raise ValueError("Compatibility receipt is not bound to its exact complete epoch")
    require_sha(checkpoint / "state.pt", receipt["state_sha256"])
    return result


def verify_rq_with_compatibility(bundle, spec, manifest, dataset, variant, *, admission_sha256=None):
    """Use instead of a private queue's verify_rq alias, leaving shared helpers unchanged."""
    from fresh_ids import verify_rq

    lineage = verify_rq(bundle, spec, manifest, dataset, variant)
    run_id = lineage["rq_run_id"]
    if run_id not in RUNS:
        return lineage
    payload = bundle / "payload"
    latest = payload / "checkpoints" / "epoch-20000"
    config_sha256 = lineage["run_config_sha256"]
    final = verify_checkpoint_attestation(latest, run_id, config_sha256, admission_sha256)
    if final["selected_checkpoint"] != lineage["selected_checkpoint"]:
        raise ValueError("Final compatibility attestation names a different selected model")
    selected_before_correction = lineage["selected_epoch"] <= 16400
    if not selected_before_correction:
        selected = verify_checkpoint_attestation(
            payload / lineage["selected_checkpoint"], run_id, config_sha256, admission_sha256,
        )
        if selected["state_sha256"] != lineage["selected_state_sha256"]:
            raise ValueError("Selected repaired-state provenance mismatch")
    export = payload / "export"
    exported = _verify_common(export, run_id, config_sha256, admission_sha256, "export")
    done = read_json(payload / "completed.json")
    reference = done.get("rq_compatibility", {})
    final_attestation_sha = sha(latest / "compatibility" / "attestation.json")
    export_attestation_sha = sha(export / "compatibility" / "attestation.json")
    if (exported.get("selected_epoch") != lineage["selected_epoch"]
            or exported.get("index_sha256") != lineage["index_sha256"]
            or exported.get("export_sha256") != lineage["export_sha256"]
            or exported.get("final_checkpoint_attestation_sha256") != final_attestation_sha
            or reference != {
                "policy": POLICY, "admission_sha256": admission_sha256,
                "export_attestation_sha256": export_attestation_sha,
                "base_state_sha256": VARIANTS[RUNS[run_id]]["base_state"],
            }):
        raise ValueError("Final export lost or changed its compatibility provenance")
    admission = read_json(export / "compatibility" / "admission.json")
    return {
        **lineage, "rq_compatibility": {
            "policy": POLICY, "classification": "compatibility correction at undefined native indexing",
            "not_claimed": "bitwise identity with a native execution past its crash",
            "admission_sha256": admission_sha256, "base_epoch": 16400,
            "base_state_sha256": reference["base_state_sha256"],
            "native_k_init_sha256": admission["native_k_init_sha256"],
            "executed_k_init_sha256": admission["executed_k_init_sha256"],
            "final_attestation_sha256": final_attestation_sha,
            "export_attestation_sha256": export_attestation_sha,
            "source_archive_sha256": admission["artifacts"]["source.zip"],
            "gates_archive_sha256": admission["artifacts"]["gates.zip"],
            "banked_evidence": admission["banked_evidence"],
            "selected_model_predates_correction": selected_before_correction,
            "compatibility_enabled_for_continuation_and_export": True,
            "export_corrections": exported["runtime_summary"]["phases"].get("export", 0),
            "continuation_runtime": final["runtime_summary"],
            "cumulative_runtime_at_export": exported["runtime_summary"],
            "original_scientific_recipe_and_selection_unchanged": True,
        },
    }
