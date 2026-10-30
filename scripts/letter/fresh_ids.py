"""Fail-closed admission of completed RQ IDs into an isolated native T5 harness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import socket
from pathlib import Path, PurePosixPath, PureWindowsPath

from author import SOURCE_HASHES
from bank import source_manifest
from prepare import read_release
from protocol import collision_summary, file_sha256, require_hash, verify_checkpoint, write_json


KIND = "fresh-id-compatible-rebuild"
TRACK = "letter-openllama7b-v2-method-replication"
FROZEN_EXECUTION = "fceeeefb3c28b689d50ba25ee9810211478a1cdb3cb144ae2a11dcf0ba70990a"
FROZEN_HARNESS = {
    "scripts/letter/run.py": "8e42dcefacf678df886edf5428e6ccd8fbb98f0047a5dffb322efa1a9a7458a9",
    "scripts/letter/prepare.py": "997e5c7dd572c8bc8084809fd442b0f3f3010b7c28a430892e665dc3d1b40f1e",
    "scripts/letter/author.py": "722da1a9b023339ca211f67aade0ea85df56f32ff0b539c03c564294fad7432c",
    "scripts/letter/protocol.py": "b035963ab39926c0a992443021b46ebb5e3bc2236346ad6780394aa2ac410b22",
    "scripts/letter/runtime_patch.py": "75e392bf58e6fb91d8610bb85c4da84718d9bfc30770e58a739cce0b1c8db4b3",
    "scripts/letter/evaluate.py": "b8455f926ca0d834fb38084bde3a540b6064dc080428609f96e74e15bccdc0a3",
    "scripts/letter/trace_train.py": "d4bee249a00c391fa0d71956f0ff3bbe4289b116162e21e2807ff17a8f796c0f",
    "src/rrbench/io.py": "82d20d0ad50378e0502fa0bf1708f77a11ae937b7b1c75347205d2c1a1453630",
    "src/rrbench/__init__.py": "2ed5ebcdd0e661a5d7bcebc57bf97343b24114967d179cb8813a57def7671848",
}
FROZEN_BANK = {
    "bank.py": "444a4c5bd587078ed4e683be9e5ccf853e03b4b26ef36c63a31f66ad87008abc",
    "bank_watch.py": "65ba3c7083befcbd026df7bcd0384a56c12dc49f61a92780b537cd24a3d21c99",
    "bank_remote.py": "37b0619af97f0f38ee2e98d7dd3e1ec05238dd44f9919fa901cc8c1a40199e60",
}
T5_ASSETS = {
    "config.json": "7480260c5fcecd310fb7d7e65f0d5b1e31fe2d465ffe341028b4cd4b87c67ada",
    "generation_config.json": "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a",
    "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    "tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    "tokenizer_config.json": "d1e7146101aa96282057f374aa9c3b260fba2109e5990edb1775d6efc70ffa3c",
}
BEFORE = b'        "kind": "smoke" if args.smoke else "released-id-replay",\n'
AFTER = (
    b'        "kind": "smoke" if args.smoke else "fresh-id-compatible-rebuild",\n'
    b'        "experiment_lineage": manifest["experiment_lineage"],\n'
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def require_canonical(path: Path, expected: str) -> None:
    if path.is_symlink() or canonical_hash(path) != expected:
        raise ValueError(f"Frozen canonical-LF source checksum mismatch: {path}")


def patch_run(source: bytes) -> bytes:
    source = source.replace(b"\r\n", b"\n")
    if (hashlib.sha256(source).hexdigest() != FROZEN_HARNESS["scripts/letter/run.py"]
            or source.count(BEFORE) != 1):
        raise ValueError("Frozen run.py source checksum/context mismatch")
    return source.replace(BEFORE, AFTER)


def load_execution(path: Path, manifest: Path, method: Path, tokenizer_recipe: Path) -> dict:
    require_canonical(path, FROZEN_EXECUTION)
    spec = read_json(path)
    for source, key in ((manifest, "input_protocol_canonical_sha256"),
                        (method, "method_spec_canonical_sha256"),
                        (tokenizer_recipe, "tokenizer_recipe_canonical_sha256")):
        require_canonical(source, spec[key])
    recipe, method_spec = read_json(tokenizer_recipe), read_json(method)
    if (spec["track_id"] != TRACK or spec["run_kind"] != KIND
            or spec["exact_original_llama1_reproduction"] is not False
            or recipe["track_id"] != TRACK or method_spec["track_id"] != TRACK
            or spec["rq"]["options"]["epochs"] != 20000
            or spec["rq"]["variants"] != recipe["variants"]
            or spec["semantic_encoder"]["model_id"] != method_spec["model"]["id"]
            or spec["semantic_encoder"]["revision"] != method_spec["model"]["revision"]
            or spec["training"]["seeds"] != [42, 17, 23]
            or spec["queue"]["cuda_visible_devices"] != [2, 3]
            or spec["queue"]["writer_host"] != "node-0"):
        raise ValueError("Fresh execution recipe does not match the frozen method/queue")
    for dataset, variants in recipe["datasets"].items():
        if spec["datasets"][dataset]["variants"] != variants:
            raise ValueError("Dataset/variant matrix changed")
    return spec


def verify_original(upstream: Path, manifest: dict, dataset: str):
    return read_release(upstream, manifest, dataset)


def validate_codes(indices: dict, codes, items: int) -> dict:
    import numpy as np

    if (set(indices) != set(map(str, range(items))) or not isinstance(codes, np.ndarray)
            or codes.shape != (items, 4) or codes.dtype.kind not in "iu"
            or np.any(codes < 0) or np.any(codes >= 256)):
        raise ValueError("Fresh IDs need the complete contiguous catalog and integer N x 4 codes")
    for item, row in enumerate(codes.tolist()):
        expected = [f"<{letter}_{code}>" for letter, code in zip("abcd", row)]
        if indices[str(item)] != expected:
            raise ValueError("Index strings disagree with positional codes.npy entries")
    return collision_summary(indices)


def verify_inputs(semantic: Path, teacher: Path, spec: dict, manifest: dict, dataset: str) -> dict:
    import numpy as np
    import torch

    frozen, reference = spec["datasets"][dataset], manifest["datasets"][dataset]
    files = {
        "semantic_sha256": semantic / "embeddings.npy",
        "embedding_receipt_sha256": semantic / "embedding-receipt.json",
        "semantic_config_sha256": semantic / "run-config.json",
        "semantic_completed_sha256": semantic / "completed.json",
        "cf_sha256": teacher / "item-embeddings.pt",
        "cf_config_sha256": teacher / "run-config.json",
        "cf_completed_sha256": teacher / "completed.json",
        "teacher_sha256": teacher / "best.pt",
    }
    for key, path in files.items():
        if path.is_symlink():
            raise ValueError("Input provenance must not be redirected")
        require_hash(path, frozen[key])
    er, ec, ed = (read_json(semantic / name) for name in (
        "embedding-receipt.json", "run-config.json", "completed.json",
    ))
    tc, td = (read_json(teacher / name) for name in ("run-config.json", "completed.json"))
    n = reference["items"]
    semantic_array = np.load(files["semantic_sha256"], mmap_mode="r", allow_pickle=False)
    cf = torch.load(files["cf_sha256"], map_location="cpu", weights_only=True)
    if (semantic_array.shape != (n, 4096) or semantic_array.dtype != np.float32
            or not torch.is_tensor(cf) or list(cf.shape) != [n, 32] or cf.dtype != torch.float32
            or not torch.isfinite(cf).all()
            or any(not np.isfinite(semantic_array[i:i + 1024]).all() for i in range(0, n, 1024))):
        raise ValueError("Full real inputs must be finite float32 N x 4096 and N x 32")
    for record in (er, ec):
        if (record.get("track_id") != TRACK or record.get("dataset") != dataset
                or record.get("model_id") != spec["semantic_encoder"]["model_id"]
                or record.get("revision") != spec["semantic_encoder"]["revision"]
                or record.get("exact_original_llama1_reproduction") is not False
                or record.get("method_spec_sha256") != spec["method_spec_canonical_sha256"]
                or record.get("item_order") != "row_equals_zero_based_item_id"
                or record.get("item_json_sha256") != reference["sha256"]["item"]
                or record.get("dtype") != "float32"):
            raise ValueError("Semantic model, item order or method provenance mismatch")
    audit = er["token_audit"]
    if (er.get("status") != "pass" or er.get("smoke") is not False
            or er.get("encoder") != "OpenLLaMA-7B-v2" or er["shape"] != [n, 4096]
            or ed.get("status") != "complete" or ed.get("smoke") is not False
            or ed["kind"] != "semantic_embedding_bank" or ed["shape"] != [n, 4096]
            or er["embeddings_sha256"] != frozen["semantic_sha256"]
            or ed["embeddings_sha256"] != frozen["semantic_sha256"]
            or ed["embedding_receipt_sha256"] != frozen["embedding_receipt_sha256"]
            or er["run_config_sha256"] != frozen["semantic_config_sha256"]
            or not er["model_files_sha256"] or not er["tokenizer_files_sha256"]
            or er["model_files_sha256"] != ec["model_files_sha256"]
            or er["tokenizer_files_sha256"] != ec["tokenizer_files_sha256"]
            or audit.get("status") != "pass" or audit.get("records") != 2 * n
            or audit.get("strict_clean") != 2 * n or audit.get("over_cap") != 0):
        raise ValueError("Semantic bank is not a completed all-item/all-field bank")
    if (td.get("status") != "complete" or td.get("smoke") is not False
            or td.get("test_evaluated") is not False or tc.get("smoke") is not False
            or tc.get("kind") != "training-only-BCE-teacher-reconstruction"
            or td.get("kind") != tc["kind"] or tc.get("dataset") != dataset
            or tc["input_sha256"] != reference["sha256"]["inter"]
            or td["item_shape"] != [n, 32] or tc["model"]["hidden_units"] != 32
            or td["item_embeddings_sha256"] != frozen["cf_sha256"]
            or td["config_sha256"] != frozen["cf_config_sha256"]
            or td["teacher_sha256"] != frozen["teacher_sha256"]):
        raise ValueError("Dataset-specific training-only BCE teacher provenance mismatch")
    return {key: file_sha256(path) for key, path in files.items()}


def _rq_state(path: Path, config: dict) -> tuple[dict, dict]:
    import torch

    receipt = read_json(path / "completed.json")
    if (receipt.get("status") != "complete" or receipt.get("kind") != "tokenizer_epoch_checkpoint"
            or receipt.get("epoch") != int(path.name.split("-")[-1]) - 1):
        raise ValueError("Invalid tokenizer checkpoint completion receipt")
    require_hash(path / "state.pt", receipt["state_sha256"])
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    required = {"epoch", "spec", "model", "optimizer", "initted", "selected_checkpoint",
                "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "best_loss", "best_collision_rate"}
    if (set(state) != required or state["spec"] != config or state["epoch"] != receipt["epoch"]
            or len(state["initted"]) != 4 or not all(state["initted"])
            or not math.isfinite(state["best_collision_rate"])
            or not 0 <= state["best_collision_rate"] <= 1):
        raise ValueError("Tokenizer checkpoint state/specification mismatch")
    return state, receipt


def verify_rq(bundle: Path, spec: dict, manifest: dict, dataset: str, variant: str) -> dict:
    import numpy as np

    receipt = read_json(bundle / "completed.json")
    run_id = f"openllama-v2-rq-{dataset}-{variant}-s42"
    source_run = PurePosixPath(receipt.get("source_run", ""))
    if (receipt.get("status") != "complete" or receipt.get("kind") != "tokenizer_resume_bundle"
            or receipt.get("phase") != "completed" or receipt.get("track_id") != TRACK
            or source_run.name != run_id or not source_run.is_absolute() or ".." in source_run.parts):
        raise ValueError("Only a completed same-track final tokenizer bundle is admissible")
    payload = bundle / "payload"
    if source_manifest(payload) != receipt["payload_manifest"]:
        raise ValueError("Tokenizer bundle payload checksum mismatch")
    done, config = read_json(payload / "completed.json"), read_json(payload / "run-config.json")
    frozen, n = spec["datasets"][dataset], manifest["datasets"][dataset]["items"]
    if variant not in frozen["variants"]:
        raise ValueError("Undeclared dataset/variant")
    expected = {
        "dataset": dataset, "synthetic_fixture": False, "seed": spec["rq"]["seed"],
        "track_id": TRACK, "claim_type": spec["claim_type"],
        "method_spec_sha256": spec["method_spec_canonical_sha256"],
        "cf_provenance": "training_only_BCE_teacher_reconstruction",
        **{k: frozen[k] for k in ("semantic_sha256", "cf_sha256", "embedding_receipt_sha256")},
        "semantic_shape": [n, 4096], "cf_shape": [n, 32],
        **spec["rq"]["options"], **spec["rq"]["variants"][variant],
        "source": spec["rq"]["source"], "harness_sha256": spec["rq"]["harness_sha256"],
    }
    if config != expected or config.get("synthetic_fixture") is not False:
        raise ValueError("RQ run configuration differs from the frozen full-variant/input recipe")
    require_hash(payload / "run-config.json", done["run_config_sha256"])
    require_hash(payload / "completed.json", receipt["run_completed_sha256"])
    selected = done.get("selected_checkpoint", "")
    if (done.get("status") != "complete" or done.get("synthetic_fixture") is not False
            or done.get("track_id") != TRACK or done.get("claim_type") != spec["claim_type"]
            or not re.fullmatch(r"checkpoints/epoch-\d{5}", selected)
            or done.get("selected_epoch") != int(selected.split("-")[-1])
            or not 2000 <= done["selected_epoch"] <= 20000 or done["selected_epoch"] % 2000
            or receipt["selected_checkpoint"] != selected
            or receipt["latest_checkpoint"] != "checkpoints/epoch-20000"):
        raise ValueError("Invalid selected/final RQ checkpoint identity")
    latest = read_json(payload / "latest-state.json")
    if (latest.get("epoch") != 20000
            or latest.get("checkpoint") not in (
                str(source_run / "checkpoints" / "epoch-20000"), "checkpoints/epoch-20000")):
        raise ValueError("RQ has not completed all 20000 actual epochs in this run")
    last, last_receipt = _rq_state(payload / "checkpoints" / "epoch-20000", config)
    best, best_receipt = _rq_state(payload / selected, config)
    if (last["selected_checkpoint"] != selected or best["selected_checkpoint"] != selected
            or best["best_collision_rate"] != last["best_collision_rate"]):
        raise ValueError("Selected checkpoint is stale or not the native minimum-collision selection")
    export = read_json(payload / "export" / "export.json")
    require_hash(payload / "export" / "index.json", export["index_sha256"])
    indices = read_json(payload / "export" / "index.json")
    codes = np.load(payload / "export" / "codes.npy", allow_pickle=False)
    collisions = validate_codes(indices, codes, n)
    if (export != done["export"] or export["items"] != n or export["levels"] != 4
            or export["seed"] != 42
            or export["excess_collision_items_after"] != collisions["excess_collision_items"]
            or export["largest_group"] != collisions["largest_group"]
            or export["utilization_by_level"] != [len(np.unique(codes[:, i])) for i in range(4)]):
        raise ValueError("Frozen export, integer codes and collision receipt disagree")
    return {
        "rq_run_id": run_id, "run_config_sha256": file_sha256(payload / "run-config.json"),
        "run_completed_sha256": file_sha256(payload / "completed.json"),
        "bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "final_epoch": 20000, "final_state_sha256": last_receipt["state_sha256"],
        "selected_epoch": done["selected_epoch"], "selected_checkpoint": selected,
        "selected_state_sha256": best_receipt["state_sha256"],
        "selected_receipt_sha256": file_sha256(payload / selected / "completed.json"),
        "index_sha256": export["index_sha256"],
        "codes_sha256": file_sha256(payload / "export" / "codes.npy"),
        "export_sha256": file_sha256(payload / "export" / "export.json"),
        "collisions": collisions,
    }


def verify_durable(bundle: Path, durable: Path) -> dict:
    expected = source_manifest(bundle)
    receipt = read_json(durable / "bank-receipt.json")
    if receipt != expected:
        raise ValueError("Final tokenizer durable receipt checksum manifest mismatch")
    for name, metadata in expected["files"].items():
        path = durable / name
        if path.is_symlink():
            raise ValueError("Durable tokenizer proof must not be redirected")
        require_hash(path, metadata["sha256"])
    return {
        "bank_receipt_sha256": file_sha256(durable / "bank-receipt.json"),
        "bundle_completed_sha256": file_sha256(durable / "completed.json"),
        "run_completed_sha256": file_sha256(durable / "payload" / "completed.json"),
        "verification": "all durable bundle bytes checked under node0 shared bank lock",
    }


def derive_manifest(parent: dict, parent_sha: str, dataset: str, lineage: dict, indices, codes) -> dict:
    keys = ("items", "users", "interactions", "training_prefix_examples",
            "validation_users", "test_users", "sha256")
    reference = {k: copy.deepcopy(parent["datasets"][dataset][k]) for k in keys}
    reference["sha256"]["index"] = lineage["index_sha256"]
    reference["fresh_sid_inventory"] = validate_codes(indices, codes, reference["items"])
    return {
        "schema_version": 1, "kind": KIND, "status": "derived_inputs_not_training_results",
        "parent_manifest_sha256": parent_sha, "experiment_lineage": copy.deepcopy(lineage),
        "datasets": {dataset: reference},
        "paper_reference_context_only": parent.get("published_results_not_reproduced_measurements", {}),
        "claim": "Fresh compatible-method IDs with random native T5; not original-LLaMA1 reproduction",
    }


def reserve_stage(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)


def build_stage(output: Path, project: Path, upstream: Path, manifest_path: Path,
                execution_path: Path, dataset: str, index_path: Path, codes_path: Path,
                lineage: dict) -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError("Stage outputs are immutable, including incomplete stages")
    if output.resolve().is_relative_to(project.resolve()) or output.resolve().is_relative_to(upstream.resolve()):
        raise ValueError("Never stage into the active project or original upstream")
    manifest = read_json(manifest_path)
    verify_original(upstream, manifest, dataset)
    for name, digest in FROZEN_HARNESS.items():
        require_canonical(project / name, digest)
    native_files = {**SOURCE_HASHES["tiger"],
                    **{f"ckpt/TIGER/{name}": digest for name, digest in T5_ASSETS.items()}}
    for name, digest in native_files.items():
        require_hash(upstream / "LETTER-TIGER" / name, digest)
    require_hash(index_path, lineage["index_sha256"])
    require_hash(codes_path, lineage["codes_sha256"])
    indices, codes = read_json(index_path), np.load(codes_path, allow_pickle=False)
    derived = derive_manifest(manifest, file_sha256(manifest_path), dataset, lineage, indices, codes)
    reserve_stage(output)
    source_receipt = {}
    for name, digest in FROZEN_HARNESS.items():
        original = project / name
        target = output / "project" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = patch_run(original.read_bytes()) if name.endswith("/run.py") else canonical_bytes(original)
        target.write_bytes(content)
        source_receipt[name] = {
            "original_bytes_sha256": file_sha256(original), "original_canonical_sha256": digest,
            "executed_sha256": file_sha256(target),
            "change": "receipt kind + manifest experiment_lineage only" if name.endswith("/run.py")
                      else "canonical LF only",
        }
    for name in native_files:
        target = output / "upstream" / "LETTER-TIGER" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / "LETTER-TIGER" / name, target)
    for kind in ("item", "inter", "index"):
        target = output / "upstream" / "data" / dataset / f"{dataset}.{kind}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        source = index_path if kind == "index" else upstream / "data" / dataset / target.name
        shutil.copyfile(source, target)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copyfile(manifest_path, provenance / "parent-manifest.json")
    shutil.copyfile(execution_path, provenance / "execution.json")
    shutil.copyfile(codes_path, provenance / "codes.npy")
    write_json(output / "manifest.json", derived)
    verify_original(output / "upstream", derived, dataset)
    receipt = {
        "status": "ready", "kind": "isolated_fresh_id_inputs_and_source",
        "dataset": dataset, "experiment_lineage": lineage, "inputs": derived["datasets"][dataset]["sha256"],
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "source_patch": source_receipt, "adapter_sha256": file_sha256(Path(__file__)),
        "files": source_manifest(output)["files"],
    }
    write_json(output / "stage.json", receipt)
    return verify_stage(output)


def verify_stage(stage: Path) -> dict:
    import numpy as np

    receipt = read_json(stage / "stage.json")
    if receipt.get("status") != "ready" or receipt.get("kind") != "isolated_fresh_id_inputs_and_source":
        raise ValueError("Missing immutable source/input stage")
    for name, metadata in receipt["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Stage file escapes its immutable directory")
        path = stage / name
        if path.is_symlink():
            raise ValueError("Staged source/input is redirected")
        require_hash(path, metadata["sha256"])
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p != stage / "stage.json"}
    if actual != set(receipt["files"]):
        raise ValueError("Immutable stage file set changed")
    derived = read_json(stage / "manifest.json")
    parent = read_json(stage / "provenance" / "parent-manifest.json")
    dataset, lineage = receipt["dataset"], receipt["experiment_lineage"]
    require_hash(stage / "manifest.json", receipt["manifest_sha256"])
    require_hash(stage / "provenance" / "parent-manifest.json", derived["parent_manifest_sha256"])
    indices = verify_original(stage / "upstream", derived, dataset)[2]
    codes = np.load(stage / "provenance" / "codes.npy", allow_pickle=False)
    expected = derive_manifest(parent, derived["parent_manifest_sha256"], dataset, lineage, indices, codes)
    if derived != expected or receipt["inputs"] != derived["datasets"][dataset]["sha256"]:
        raise ValueError("Derived manifest, lineage, original item/inter hashes or index disagree")
    for name, digest in FROZEN_HARNESS.items():
        source = canonical_bytes(stage / "project" / name)
        if name.endswith("/run.py"):
            if source.count(AFTER) != 1:
                raise ValueError("Fresh run source lost its explicit lineage")
            source = source.replace(AFTER, BEFORE)
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Staged source changed beyond the guarded metadata-only patch")
    return receipt


def validate_audit(stage: Path, audit: Path) -> dict:
    receipt = verify_stage(stage)
    report = read_json(audit / "audit.json")
    manifest = read_json(stage / "manifest.json")
    reference = manifest["datasets"][receipt["dataset"]]
    if (report.get("status") != "pass" or report.get("dataset") != receipt["dataset"]
            or report.get("backend") != "tiger" or report["inputs"] != receipt["inputs"]
            or report["manifest_sha256"] != receipt["manifest_sha256"]
            or report["source"] != SOURCE_HASHES["tiger"] or report["model_max_length"] != 512):
        raise ValueError("Full fresh T5 audit does not bind this staged manifest/source")
    for split, count in (("train", reference["training_prefix_examples"]),
                         ("valid", reference["validation_users"]), ("test", reference["test_users"])):
        part = report["splits"][split]
        if (part["records"] != count or part["strict_clean"] != count or part["over_cap"] != 0
                or part["target_tokens_min_max"] != [5, 5]
                or part["input_tokens_p50_p90_p99_max"][-1] > 512):
            raise ValueError("Every fresh train/valid/test record needs four atomic codes + EOS")
    return report


def validate_run_identity(output: Path, stage: dict, seed: int, temperature: float, smoke: bool) -> dict:
    run = read_json(output / "run-config.json")
    if (run.get("kind") != ("smoke" if smoke else KIND)
            or run.get("experiment_lineage") != stage["experiment_lineage"]
            or run.get("dataset") != stage["dataset"] or run.get("inputs") != stage["inputs"]
            or run.get("seed") != seed or run.get("temperature") != temperature
            or (not smoke and stage["experiment_lineage"].get("synthetic_fixture") is not False)):
        raise ValueError("Run kind/lineage/input identity mismatch; never relabel or reuse this output")
    return run


def validate_training(output: Path, stage: Path, audit: Path, seed: int, temperature: float,
                      smoke: bool = False) -> dict:
    from runtime_patch import require_resume_patch
    from run import training_options

    staged = verify_stage(stage)
    audited = validate_audit(stage, audit)
    spec = read_json(stage / "provenance" / "execution.json")
    run = validate_run_identity(output, staged, seed, temperature, smoke)
    geometry = spec["gate"] if smoke else spec["datasets"][staged["dataset"]]
    options = training_options(
        seed=seed, batch_size=geometry["micro_batch_size"],
        effective_batch=spec["gate"]["effective_batch_size"] if smoke else 1024,
        world_size=2, learning_rate=0.0005, epochs=200,
    )
    if smoke:
        options.update(max_steps=5, logging_steps=1, save_strategy="steps", save_steps=1,
                       evaluation_strategy="no", load_best_model_at_end=False)
    expected_harness = {
        name: file_sha256(stage / "project" / "scripts" / "letter" / name)
        for name in ("run.py", "author.py", "protocol.py", "prepare.py", "runtime_patch.py")
    }
    if (run["options"] != options or run["world_size"] != 2
            or run["initialization"] != "author random T5 configuration"
            or run["author_source"] != SOURCE_HASHES["tiger"]
            or run["audit_sha256"] != file_sha256(audit / "audit.json")
            or run["tokenizer_vocab_sha256"] != audited["tokenizer_vocab_sha256"]
            or run["harness_sha256"] != expected_harness
            or run["trainer_source_sha256"] != require_resume_patch()):
        raise ValueError("Native T5 recipe/audit/harness differs from frozen execution")
    for key in ("d_model", "d_ff", "d_kv", "num_heads", "num_layers", "dropout_rate"):
        if run["model_config"][key] != spec["generator"][key]:
            raise ValueError("Native random T5 architecture changed")
    return run


def select_resume(output: Path, world_size: int = 2) -> Path:
    import torch

    if (output / "completed.json").exists():
        raise FileExistsError("Completed training outputs must never be resumed or overwritten")
    if not (output / "run-config.json").is_file():
        raise ValueError("Partial output has no immutable run identity")
    config = read_json(output / "run-config.json")
    checkpoints = sorted(output.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for checkpoint in checkpoints:
        if not (checkpoint / "complete.json").is_file():
            continue
        if checkpoint.is_symlink() or checkpoint.resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint escapes its own run")
        step = verify_checkpoint(checkpoint)
        state = read_json(checkpoint / "trainer_state.json")
        if (step != state["global_step"] or checkpoint.name != f"checkpoint-{step}"
                or not (checkpoint / "callback-state.json").is_file()
                or {p.name for p in checkpoint.glob("rng_state*.pth")}
                != {f"rng_state_{rank}.pth" for rank in range(world_size)}):
            raise ValueError("Exact continuation requires every rank RNG and callback state")
        if not (checkpoint / "training_args.bin").is_file():
            raise ValueError("Checkpoint has no native training-argument identity")
        arguments = torch.load(checkpoint / "training_args.bin", map_location="cpu", weights_only=False)
        best = state.get("best_model_checkpoint")
        if (Path(arguments.output_dir).resolve() != output.resolve()
                or arguments.seed != config["seed"]
                or arguments.data_seed != config["options"]["data_seed"]
                or (best and (Path(best).resolve().parent != output.resolve()
                              or not re.fullmatch(r"checkpoint-\d+", Path(best).name)))):
            raise ValueError("Checkpoint native arguments/selected model do not belong to its own run identity")
        return checkpoint
    raise ValueError("No complete sealed checkpoint; preserve this partial output")


def validate_scope(binding: Path) -> tuple[Path, Path]:
    from datetime import date
    from uuid import UUID

    if socket.gethostname() != "node-0":
        raise RuntimeError("Only the private binding's node0 writer may admit/bank downstream work")
    job = read_json(binding)
    required = {
        "name", "subscription", "resource_group", "workspace", "private_key", "nodes",
        "gpus_per_node", "shared_parent", "experiment_id", "durable_directory",
        "node_directory", "single_writer_node", "authorized_on", "authorization",
    }
    if (not isinstance(job, dict) or not required <= job.keys()
            or job.keys() - required - {"automation_id"}):
        raise ValueError("Private allocation binding schema is incomplete or has unknown fields")
    strings = required - {"nodes", "gpus_per_node", "single_writer_node"}
    if any(not isinstance(job[key], str) or not job[key]
           or job[key].strip() != job[key] or any(ord(c) < 32 for c in job[key]) for key in strings):
        raise ValueError("Private binding string fields must be nonempty and free of control characters")
    try:
        if str(UUID(job["subscription"])) != job["subscription"]:
            raise ValueError("Noncanonical subscription")
        if "automation_id" in job and str(UUID(job["automation_id"])) != job["automation_id"]:
            raise ValueError("Noncanonical automation identity")
        if date.fromisoformat(job["authorized_on"]).isoformat() != job["authorized_on"]:
            raise ValueError("Noncanonical authorization date")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Invalid private binding UUID or authorization date") from error
    if not (PurePosixPath(job["private_key"]).is_absolute()
            or PureWindowsPath(job["private_key"]).is_absolute()):
        raise ValueError("The private SSH key reference must be absolute")
    nodes = job["nodes"]
    if (not isinstance(nodes, list) or not nodes
            or any(type(node) is not int or node < 0 for node in nodes)
            or len(set(nodes)) != len(nodes) or 0 not in nodes
            or type(job["single_writer_node"]) is not int or job["single_writer_node"] != 0
            or type(job["gpus_per_node"]) is not int or job["gpus_per_node"] < 4):
        raise ValueError("Private binding must allocate node0 and GPU indices 2/3 with node0 as writer")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", job["experiment_id"]):
        raise ValueError("Private experiment identity must be a safe directory identifier")

    def absolute_path(key: str, trailing_slash: bool = False) -> PurePosixPath:
        value = job[key]
        if trailing_slash and value.endswith("/"):
            value = value[:-1]
        if (not value.startswith("/") or value.startswith("//")
                or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                       for part in value.split("/")[1:])):
            raise ValueError("Private scope paths must be canonical absolute POSIX paths")
        path = PurePosixPath(value)
        if path.as_posix() != value:
            raise ValueError("Private scope paths must not contain traversal or redundant separators")
        return path

    root = absolute_path("node_directory")
    shared = absolute_path("shared_parent", trailing_slash=True)
    destination = absolute_path("durable_directory")
    directory_id = job["experiment_id"].replace("-", "_")
    # Validate lexically: shared-mount IO belongs exclusively inside bank.lock.
    if (root.parent != PurePosixPath("/scratch") or root.name != directory_id
            or PurePosixPath("/data/gen2") not in shared.parents
            or destination.parent != shared or destination.name != directory_id
            or root == destination or root in destination.parents or destination in root.parents):
        raise ValueError("Private owned root and experiment-specific shared-parent boundary disagree")
    return Path(str(root)), Path(str(destination))


def attach_rq_compatibility(lineage: dict, rq: dict) -> None:
    if "rq_compatibility" in rq:
        lineage["rq_compatibility"] = copy.deepcopy(rq["rq_compatibility"])


def stage_real(args, *, rq_verifier=None) -> dict:
    import fcntl

    root, destination = validate_scope(args.binding)
    expected_durable = destination / "openllama-v2" / "tokenizers" / args.bundle.name
    if (args.durable_bundle != expected_durable
            or not args.bundle.resolve().is_relative_to(root)
            or not args.output.resolve().is_relative_to(root)):
        raise ValueError("Bundle/stage/durable paths escape the assigned task roots")
    spec = load_execution(args.execution, args.manifest, args.method_spec, args.tokenizer_recipe)
    manifest = read_json(args.manifest)
    verify_original(args.upstream, manifest, args.dataset)
    inputs = verify_inputs(args.semantic, args.teacher, spec, manifest, args.dataset)
    verifier = verify_rq if rq_verifier is None else rq_verifier
    rq = verifier(args.bundle, spec, manifest, args.dataset, args.variant)
    with (root / "bank.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        durable = verify_durable(args.bundle, args.durable_bundle)
    lineage = {
        "kind": KIND, "track_id": TRACK, "claim_type": spec["claim_type"],
        "synthetic_fixture": False, "exact_original_llama1_reproduction": False,
        "dataset": args.dataset, "variant": args.variant,
        "semantic_encoder": spec["semantic_encoder"], "downstream_generator": spec["generator"],
        "method_model_identity_applies_to": "semantic_encoder_only_in_this_T5_track",
        "method_spec_sha256": spec["method_spec_canonical_sha256"],
        "execution_sha256": file_sha256(args.execution),
        "tokenizer_recipe_sha256": spec["tokenizer_recipe_canonical_sha256"],
        "input_provenance": inputs, "rq": rq, "durable_proof": durable,
        "index_sha256": rq["index_sha256"], "codes_sha256": rq["codes_sha256"],
    }
    attach_rq_compatibility(lineage, rq)
    return build_stage(
        args.output, args.project, args.upstream, args.manifest, args.execution, args.dataset,
        args.bundle / "payload" / "export" / "index.json",
        args.bundle / "payload" / "export" / "codes.npy", lineage,
    )


def fixture_codes(items: int):
    import numpy as np

    if not 0 < items <= 65536:
        raise ValueError("The catalog-format fixture supports 1..65536 items")
    # The first two positions reversibly encode item ID; all positions exercise the vocabulary.
    return np.array([[i % 256, (i // 256 + 31 * i) % 256,
                      (73 * i) % 256, (17 * i + i // 256) % 256]
                     for i in range(items)], dtype=np.int64)


def stage_fixture(args) -> dict:
    import numpy as np

    spec = load_execution(args.execution, args.manifest, args.method_spec, args.tokenizer_recipe)
    manifest = read_json(args.manifest)
    verify_original(args.upstream, manifest, args.dataset)
    if args.output.exists():
        raise FileExistsError("Fixture stage already exists; no overwrite")
    fixture = args.output.with_name(args.output.name + "-fixture-ids")
    reserve_stage(fixture)
    n = manifest["datasets"][args.dataset]["items"]
    codes = fixture_codes(n)
    indices = {str(i): [f"<{c}_{v}>" for c, v in zip("abcd", row)]
               for i, row in enumerate(codes.tolist())}
    write_json(fixture / "index.json", indices)
    np.save(fixture / "codes.npy", codes, allow_pickle=False)
    lineage = {
        "kind": "catalog_format_fixture_only", "track_id": "synthetic_fresh_id_format_gate",
        "claim_type": "synthetic_correctness_fixture", "synthetic_fixture": True,
        "exact_original_llama1_reproduction": False, "rq_completed": False,
        "dataset": args.dataset, "variant": "format_fixture",
        "semantic_encoder": None, "downstream_generator": spec["generator"],
        "execution_sha256": file_sha256(args.execution),
        "index_sha256": file_sha256(fixture / "index.json"),
        "codes_sha256": file_sha256(fixture / "codes.npy"),
        "scope": "Full real interaction audit and tiny native T5 format/resume only; no real training admission",
    }
    write_json(fixture / "fixture-receipt.json", lineage)
    return build_stage(args.output, args.project, args.upstream, args.manifest, args.execution,
                       args.dataset, fixture / "index.json", fixture / "codes.npy", lineage)


def equal_state(left, right) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_state(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right))
    return left == right


def probe_gate(stage: Path, audit: Path, checkpoint: Path) -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from author import default_args, load_backend
    from evaluate import generate_fixed, verify_model_binding
    from prepare import tokenizer_digest
    from protocol import PrefixTrie, code_to_items, sequence_records

    staged = verify_stage(stage)
    run = validate_training(checkpoint, stage, audit, 42, 1.0, smoke=True)
    done = read_json(checkpoint / "completed.json")
    verify_model_binding(checkpoint, done, run, staged["dataset"], staged["inputs"])
    manifest = read_json(stage / "manifest.json")
    interactions, _, indices = read_release(stage / "upstream", manifest, staged["dataset"])
    author = load_backend(stage / "upstream", "tiger")
    args = default_args(author, stage / "upstream", staged["dataset"])
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, local_files_only=True, model_max_length=512)
    if tokenizer_digest(tokenizer) != run["tokenizer_vocab_sha256"]:
        raise ValueError("Fresh gate tokenizer differs from its training vocabulary")
    rows = author.data.SeqRecDataset(args, mode="test")
    inputs, targets = author.collator.TestCollator(args, tokenizer)([rows[0]])
    groups = code_to_items(indices)
    trie = PrefixTrie([[0] + tokenizer.encode(code) for code in sorted(groups)])
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    with torch.inference_mode():
        generated = generate_fixed(model, inputs.to("cuda"),
                                   lambda _, prefix: trie.allowed(prefix.tolist()), 20)
    codes = [text.strip().replace(" ", "") for text in
             tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)]
    if (len(codes) != 20 or len(set(codes)) != 20 or any(code not in groups for code in codes)
            or any(tokenizer.eos_token_id not in row[1:].tolist() for row in generated.sequences)
            or targets[0] != "".join(indices[str(sequence_records(interactions, "test")[0]["target"])])):
        raise ValueError("Native full-trie four-code + EOS generation/target parity failed")
    examples = read_json(audit / "examples.json")
    return {
        "status": "pass", "beams": 20, "max_new_tokens": 10, "valid_unique_codes": codes,
        "finish_reason": "EOS observed in every beam",
        "train_example": next(row for row in examples if row["split"] == "train"),
        "test_example": next(row for row in examples if row["split"] == "test"),
        "template": "unchanged unprompted native TIGER; no chat/thinking wrapper",
        "not_a_quality_measurement": True,
    }


def check_gate(stage: Path, audit: Path, gate: Path) -> dict:
    import torch

    staged = verify_stage(stage)
    continuous, resumed = gate / "continuous", gate / "resumed"
    configs = [validate_training(p, stage, audit, 42, 1.0, smoke=True) for p in (continuous, resumed)]
    if configs[0] != configs[1]:
        raise ValueError("Continuation gate configurations differ")
    for run in (continuous, resumed):
        done = read_json(run / "completed.json")
        require_hash(run / "run-config.json", done["run_config_sha256"])
        require_hash(run / "pytorch_model.bin", done["model_sha256"])
        if done["status"] != "complete" or done["global_step"] != 5:
            raise ValueError("Native five-update gate is not complete")
        verify_checkpoint(run / "checkpoint-5")
    compared = ["pytorch_model.bin", "optimizer.pt", "scheduler.pt",
                "rng_state_0.pth", "rng_state_1.pth"]
    for name in compared:
        values = [torch.load(p / "checkpoint-5" / name, map_location="cpu", weights_only=False)
                  for p in (continuous, resumed)]
        if not equal_state(*values):
            raise ValueError(f"Fresh-ID continuation is not bitwise identical: {name}")
    callbacks = [read_json(p / "checkpoint-5" / "callback-state.json") for p in (continuous, resumed)]
    if callbacks[0] != callbacks[1]:
        raise ValueError("Callback state changed during exact continuation")
    trace_hashes = {}
    for rank in range(2):
        paths = [gate / "traces" / part / f"rank-{rank}.jsonl"
                 for part in ("continuous", "stopped", "resumed")]
        traces = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        if not traces[0] or traces[0] != traces[1] + traces[2] or len(traces[0]) != 10:
            raise ValueError("Ordered actual training input/label tensors differ after resume")
        trace_hashes[str(rank)] = {part: file_sha256(path) for part, path in
                                  zip(("continuous", "stopped", "resumed"), paths)}
    states = [read_json(p / "trainer_state.json") for p in (continuous, resumed)]
    histories = [[row for row in state["log_history"] if "loss" in row or "eval_loss" in row]
                 for state in states]
    if histories[0] != histories[1]:
        raise ValueError("Ordered loss/learning-rate/validation history changed across resume")
    result = {
        "status": "pass", "kind": "native_fresh_id_compatibility_and_resume_gate",
        "not_a_quality_measurement": True, "experiment_lineage": staged["experiment_lineage"],
        "stage_sha256": file_sha256(stage / "stage.json"),
        "audit_sha256": file_sha256(audit / "audit.json"), "records": 100, "updates": 5,
        "workers": 2, "micro_batch_size": 16, "effective_batch_size": 64, "interrupt_step": 3,
        "model_optimizer_scheduler_and_every_rank_rng": "bitwise_equal",
        "callback_state": callbacks[0], "ordered_input_and_label_tensors": "identical",
        "ordered_loss_and_lr_history": "identical", "trace_sha256": trace_hashes,
        "checkpoint_receipts": {part: file_sha256(gate / part / "checkpoint-5" / "complete.json")
                                for part in ("continuous", "resumed")},
        "generation_probe": probe_gate(stage, audit, continuous),
        "scope": "Five-update format/continuation gate; prior full-data production-geometry proofs remain separate",
    }
    if (gate / "gate.json").exists():
        raise FileExistsError("Native gate evidence is immutable")
    write_json(gate / "gate.json", result)
    return result


def verify_evaluation(stage: Path, audit: Path, checkpoint: Path, evaluation: Path) -> dict:
    import torch
    from protocol import code_to_items, rank_metrics

    staged = verify_stage(stage)
    run = read_json(checkpoint / "run-config.json")
    validate_training(checkpoint, stage, audit, run["seed"], run["temperature"])
    done, trained = read_json(evaluation / "completed.json"), read_json(checkpoint / "completed.json")
    config = read_json(evaluation / "evaluation-config.json")
    manifest = read_json(stage / "manifest.json")
    interactions, _, indices = read_release(stage / "upstream", manifest, staged["dataset"])
    if (done["status"] != "complete" or trained["status"] != "complete"
            or config["source_run"] != run or config["dataset"] != staged["dataset"]
            or config["checkpoint_sha256"] != trained["model_sha256"]
            or config["split"] != "test" or config["beams"] != 20 or config["max_new_tokens"] != 10
            or config["seen_item_exclusion"] is not False or config["expected_users"] != len(interactions)):
        raise ValueError("Final fresh evaluation/run/fixed-budget binding mismatch")
    require_hash(checkpoint / "run-config.json", trained["run_config_sha256"])
    require_hash(checkpoint / "pytorch_model.bin", trained["model_sha256"])
    require_hash(evaluation / "predictions.jsonl", done["predictions_sha256"])
    groups, seen = code_to_items(indices), set()
    totals = {"code": {}, "item": {}}
    with (evaluation / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            user, codes, scores = row["user"], row["codes"], row["scores"]
            if user not in interactions or user in seen:
                raise ValueError("Missing/duplicated/unknown evaluation users")
            target, target_code = interactions[user][-1], "".join(indices[str(interactions[user][-1])])
            if (row["target_item"] != target or row["target_code"] != target_code
                    or len(codes) != 20 or len(set(codes)) != 20 or any(c not in groups for c in codes)
                    or len(scores) != 20 or any(not math.isfinite(v) for v in scores)
                    or scores != sorted(scores, reverse=True)
                    or row["target_collision_size"] != len(groups[target_code])):
                raise ValueError("Fresh prediction code/target/beam/collision mismatch")
            expanded = list(dict.fromkeys(item for code in codes for item in groups[code]))[:10]
            metrics = {"code": rank_metrics(codes, target_code), "item": rank_metrics(expanded, target)}
            if row["item_predictions"] != expanded or row["metrics"] != metrics:
                raise ValueError("Saved fresh code/item metrics do not independently recompute")
            seen.add(user)
            for kind, values in metrics.items():
                for key, value in values.items():
                    totals[kind][key] = totals[kind].get(key, 0.0) + value
    metrics = {kind: {k: v / len(interactions) for k, v in values.items()}
               for kind, values in totals.items()}
    if seen != set(interactions) or done["users"] != len(seen) or done["metrics"] != metrics:
        raise ValueError("Fresh evaluation is not the complete all-user recomputed result")
    selected_name = Path(trained["best_checkpoint"]).name
    if not re.fullmatch(r"checkpoint-\d+", selected_name):
        raise ValueError("Invalid selected native T5 checkpoint")
    selected = checkpoint / selected_name
    verify_checkpoint(selected)
    state, selected_state = read_json(checkpoint / "trainer_state.json"), read_json(selected / "trainer_state.json")
    losses = [row["eval_loss"] for row in state["log_history"] if "eval_loss" in row]
    if (not losses or min(losses) != trained["best_metric"]
            or selected_state["best_metric"] != trained["best_metric"]
            or Path(selected_state["best_model_checkpoint"]).name != selected_name
            or not equal_state(torch.load(selected / "pytorch_model.bin", map_location="cpu"),
                               torch.load(checkpoint / "pytorch_model.bin", map_location="cpu"))):
        raise ValueError("Final T5 weights are not the minimum-validation-loss checkpoint")
    return {
        "status": "verified_single_fresh_id_cell", "study_status": "partial",
        "experiment_lineage": staged["experiment_lineage"], "seed": run["seed"],
        "temperature": run["temperature"], "all_users": len(seen), "beams": 20,
        "code_metrics": metrics["code"], "original_item_diagnostic": metrics["item"],
        "model_sha256": trained["model_sha256"], "predictions_sha256": done["predictions_sha256"],
        "selected_checkpoint": selected_name, "selected_epoch": selected_state["epoch"],
        "scope": "Compatible semantic encoder + reconstructed teacher + random native T5; not original-backbone reproduction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("stage", "fixture"):
        sub = commands.add_parser(command)
        for name in ("project", "upstream", "manifest", "execution", "method-spec",
                     "tokenizer-recipe", "output"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        sub.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
        if command == "stage":
            for name in ("binding", "bundle", "durable-bundle", "semantic", "teacher"):
                sub.add_argument(f"--{name}", type=Path, required=True)
            sub.add_argument("--variant", choices=("semantic", "cf", "diversity", "full"), required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--stage", type=Path, required=True)
    gate = commands.add_parser("check-gate")
    for name in ("stage", "audit", "gate"):
        gate.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-gate":
        result = check_gate(args.stage, args.audit, args.gate)
        print(json.dumps({key: result[key] for key in (
            "status", "kind", "model_optimizer_scheduler_and_every_rank_rng",
            "ordered_input_and_label_tensors", "stage_sha256",
        )}), flush=True)
        return
    result = (stage_real(args) if args.command == "stage" else
              stage_fixture(args) if args.command == "fixture" else verify_stage(args.stage))
    print(json.dumps({"status": result["status"], "dataset": result["dataset"],
                      "lineage": result["experiment_lineage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
