"""Additive provenance admission for the two compatibility-corrected RQ runs."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from rq_recovery_compat import POLICY, endpoint_candidates
from rq_recovery_control import HARNESS, ORIGINAL_SEAL, VARIANTS, read_json, require_sha
from rq_recovery_runtime import array_sha, file_manifest, sha


RUNS = {f"openllama-v2-rq-Instruments-{variant}-s42": variant for variant in VARIANTS}


def _archive_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError("Compatibility event archive contains an unsafe member name")
    return path


def audit_event_archive(path):
    summary = {"tasks": 0, "failed_tasks": 0, "corrections": 0, "phases": {}, "worker_pids": []}
    details, pids, task_ids, used_evidence = {}, set(), set(), set()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Compatibility event archive contains duplicate members")
        content = {}
        for info in infos:
            name = _archive_name(info.filename).as_posix()
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK or info.is_dir():
                if mode == stat.S_IFLNK:
                    raise ValueError("Compatibility event archive contains a redirected member")
                continue
            content[name] = archive.read(info)
    task_files = sorted(name for name in content if PurePosixPath(name).name.startswith("tasks-")
                        and PurePosixPath(name).suffix == ".jsonl")
    if not task_files:
        raise ValueError("Compatibility event archive contains no worker task evidence")
    for name in task_files:
        for line in content[name].splitlines():
            event = json.loads(line)
            task_id = event.get("task_id")
            if not isinstance(task_id, str) or task_id in task_ids:
                raise ValueError("Compatibility event archive has a missing or duplicate task identity")
            task_ids.add(task_id)
            if (event.get("policy") != POLICY or event.get("status") != "success"
                    or event.get("actual_init_centroids_binding") is not True):
                raise ValueError("Compatibility event records a failed or unbound native task")
            context = event.get("context")
            call_id = context.get("call_id") if isinstance(context, dict) else None
            context_name = f"contexts/{call_id}.json"
            if call_id is None or context_name not in content:
                raise ValueError("Compatibility task lacks its parent clustering context")
            context_bytes = content[context_name]
            archived_context = json.loads(context_bytes)
            if (archived_context != context
                    or hashlib.sha256(context_bytes).hexdigest() != event.get("context_sha256")
                    or context.get("expected_input_sha256") != event.get("input_sha256")
                    or event.get("initial_seed") not in context.get("initialization_seeds", [])):
                raise ValueError("Compatibility worker task differs from its parent input/RNG context")
            before_keys = None
            after_keys = None
            for correction in event.get("corrections", []):
                evidence_name = _archive_name(correction.get("evidence", "")).as_posix()
                if evidence_name not in content or evidence_name in used_evidence:
                    raise ValueError("Compatibility correction evidence is missing or reused")
                evidence_bytes = content[evidence_name]
                if hashlib.sha256(evidence_bytes).hexdigest() != correction.get("evidence_sha256"):
                    raise ValueError("Compatibility correction evidence checksum mismatch")
                with np.load(io.BytesIO(evidence_bytes), allow_pickle=False) as evidence:
                    required = {
                        "X", "x_squared_norms", "distances", "potential", "draws", "candidates",
                        "rng_before_keys", "rng_after_draw_keys",
                    }
                    if not required <= set(evidence.files):
                        raise ValueError("Compatibility correction evidence is incomplete")
                    _, recomputed = endpoint_candidates(
                        evidence["distances"], evidence["potential"],
                        evidence["draws"], evidence["candidates"],
                    )
                    before_keys = evidence["rng_before_keys"]
                    after_keys = evidence["rng_after_draw_keys"]
                    evidence_input_sha256 = array_sha(evidence["X"])
                if recomputed is None or any(correction.get(key) != value
                                              for key, value in recomputed.items()):
                    raise ValueError("Compatibility correction does not satisfy the pinned endpoint policy")
                if evidence_input_sha256 != event.get("input_sha256"):
                    raise ValueError("Compatibility correction evidence names another worker input")
                if (not isinstance(correction.get("center"), int)
                        or not re.fullmatch(r"[0-9a-f]{64}", correction.get("rng_after_draw_sha256", ""))
                        or not isinstance(correction.get("rng_after_draw_position"), int)):
                    raise ValueError("Compatibility correction lacks RNG/center provenance")
                used_evidence.add(evidence_name)
                epoch, phase, level = str(context["epoch"]), context["phase"], str(context["level"])
                by_phase = details.setdefault(epoch, {}).setdefault(phase, {})
                by_phase[level] = by_phase.get(level, 0) + 1
            if event.get("corrections"):
                if (array_sha(before_keys) != event.get("rng_before_keys_sha256")
                        or array_sha(after_keys) != event.get("rng_after_keys_sha256")):
                    raise ValueError("Compatibility correction RNG evidence is inconsistent")
            count = len(event.get("corrections", []))
            phase = context["phase"]
            summary["tasks"] += 1
            summary["corrections"] += count
            summary["phases"][phase] = summary["phases"].get(phase, 0) + count
            pids.add(event["pid"])
    unclaimed = {
        name for name in content if PurePosixPath(name).suffix == ".npz"
    } - used_evidence
    if unclaimed:
        raise ValueError("Compatibility event archive contains unclaimed correction evidence")
    summary["worker_pids"] = sorted(pids)
    return {"summary": summary, "corrections_by_epoch_level_phase": details}


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
    recomputed = audit_event_archive(path / "events.zip")
    if (recomputed["summary"] != result.get("runtime_summary")
            or recomputed["summary"]["failed_tasks"] != 0
            or (artifact == "checkpoint"
                and recomputed["summary"]["phases"].get("export", 0) != 0)):
        raise ValueError("Compatibility runtime summary differs from independently audited events")
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


def verify_rq_with_compatibility(
    bundle, spec, manifest, dataset, variant, *,
    admission_sha256=None, native_lineage=None,
):
    """Use instead of a private queue's verify_rq alias, leaving shared helpers unchanged."""
    if native_lineage is None:
        from fresh_ids import verify_rq
        lineage = verify_rq(bundle, spec, manifest, dataset, variant)
    else:
        lineage = native_lineage
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
    audited_export = audit_event_archive(export / "compatibility" / "events.zip")
    export_corrections = audited_export["summary"]["phases"].get("export", 0)
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
            "export_corrections": export_corrections,
            "export_was_corrected": export_corrections > 0,
            "native_artifact_identity_claimed": False,
            "downstream_ids_repair_dependent":
                not selected_before_correction or export_corrections > 0,
            "corrected_index_sha256": lineage["index_sha256"],
            "corrected_codes_sha256": lineage["codes_sha256"],
            "corrections_by_epoch_level_phase":
                audited_export["corrections_by_epoch_level_phase"],
            "continuation_runtime": final["runtime_summary"],
            "cumulative_runtime_at_export": exported["runtime_summary"],
            "original_scientific_recipe_and_selection_unchanged": True,
        },
    }
