import ast
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
from protocol import file_sha256, write_json


def module():
    assert (SCRIPTS / "fresh_ids.py").is_file(), "Strict fresh-ID admission is missing"
    return importlib.import_module("fresh_ids")


def test_frozen_bank_sources_match_their_canonical_hashes():
    m = module()
    for name, digest in m.FROZEN_BANK.items():
        assert m.canonical_hash(SCRIPTS / name) == digest


def pipeline():
    assert (SCRIPTS / "fresh_pipeline.py").is_file(), "Bounded fresh-ID queue is missing"
    return importlib.import_module("fresh_pipeline")


def execution():
    path = ROOT / "configs" / "letter-fresh-id-execution-v1.json"
    assert path.is_file(), "Fresh-ID execution recipe must be frozen before launch"
    return json.loads(path.read_text(encoding="utf-8"))


def catalog(path):
    items = {str(i): {"title": f"item {i}"} for i in range(4)}
    inter = {"u0": [0, 1, 2, 3], "u1": [3, 2, 1, 0]}
    indices = {str(i): [f"<{c}_{i}>" for c in "abcd"] for i in range(4)}
    folder = path / "data" / "Instruments"
    for kind, data in (("item", items), ("inter", inter), ("index", indices)):
        write_json(folder / f"Instruments.{kind}.json", data)
    return {"datasets": {"Instruments": {
        "items": 4, "users": 2, "interactions": 8,
        "training_prefix_examples": 2, "validation_users": 2, "test_users": 2,
        "sha256": {k: file_sha256(folder / f"Instruments.{k}.json")
                   for k in ("item", "inter", "index")},
    }}}


def index_fixture():
    codes = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 255, 0, 255], [1, 2, 3, 4]])
    indices = {str(i): [f"<{c}_{n}>" for c, n in zip("abcd", row)]
               for i, row in enumerate(codes.tolist())}
    return indices, codes


def bundle_fixture(path):
    """Contract fixture, NOT an actual RQ fit or a pipeline-admissible production dataset."""
    spec = execution()
    config = {
        "dataset": "Instruments", "synthetic_fixture": False, "seed": 42,
        "track_id": spec["track_id"], "claim_type": spec["claim_type"],
        "method_spec_sha256": spec["method_spec_canonical_sha256"],
        "cf_provenance": "training_only_BCE_teacher_reconstruction",
        **{k: spec["datasets"]["Instruments"][k] for k in (
            "semantic_sha256", "cf_sha256", "embedding_receipt_sha256",
        )},
        "semantic_shape": [4, 4096], "cf_shape": [4, 32],
        **spec["rq"]["options"], "alpha": 0.02, "beta": 0.0001,
        "source": spec["rq"]["source"], "harness_sha256": spec["rq"]["harness_sha256"],
    }
    payload = path / "payload"
    write_json(payload / "run-config.json", config)
    for epoch in (2000, 20000):
        checkpoint = payload / "checkpoints" / f"epoch-{epoch:05d}"
        checkpoint.mkdir(parents=True)
        state = {
            "epoch": epoch - 1, "spec": config, "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"state": {}}, "initted": [True] * 4,
            "selected_checkpoint": "checkpoints/epoch-02000",
            "python_rng": (), "numpy_rng": (), "torch_rng": torch.get_rng_state(),
            "cuda_rng": [], "best_loss": 1.0, "best_collision_rate": 0.25,
        }
        torch.save(state, checkpoint / "state.pt")
        write_json(checkpoint / "completed.json", {
            "status": "complete", "kind": "tokenizer_epoch_checkpoint",
            "epoch": epoch - 1, "state_sha256": file_sha256(checkpoint / "state.pt"),
        })
    indices, codes = index_fixture()
    write_json(payload / "export" / "index.json", indices)
    np.save(payload / "export" / "codes.npy", codes, allow_pickle=False)
    export = {
        "items": 4, "levels": 4, "seed": 42, "collision_retries": 1,
        "excess_collision_items_before": 1, "excess_collision_items_after": 1,
        "largest_group": 2, "utilization_by_level": [3, 3, 3, 3],
        "index_sha256": file_sha256(payload / "export" / "index.json"),
    }
    write_json(payload / "export" / "export.json", export)
    write_json(payload / "latest-state.json", {
        "epoch": 20000,
        "checkpoint": "/scratch/unit-only/openllama-v2-rq-Instruments-full-s42/checkpoints/epoch-20000",
    })
    write_json(payload / "completed.json", {
        "status": "complete", "synthetic_fixture": False, "track_id": spec["track_id"],
        "claim_type": spec["claim_type"], "selected_epoch": 2000,
        "selected_checkpoint": "checkpoints/epoch-02000", "export": export,
        "run_config_sha256": file_sha256(payload / "run-config.json"),
    })
    seal_bundle(path, spec)
    return spec, {"datasets": {"Instruments": {"items": 4}}}


def seal_bundle(path, spec):
    from bank import source_manifest

    write_json(path / "completed.json", {
        "status": "complete", "kind": "tokenizer_resume_bundle", "phase": "completed",
        "source_run": "/scratch/unit-only/openllama-v2-rq-Instruments-full-s42",
        "track_id": spec["track_id"], "latest_checkpoint": "checkpoints/epoch-20000",
        "selected_checkpoint": "checkpoints/epoch-02000",
        "run_completed_sha256": file_sha256(path / "payload" / "completed.json"),
        "payload_manifest": source_manifest(path / "payload"),
    })


def change_json(path, **updates):
    value = json.loads(path.read_text())
    value.update(updates)
    write_json(path, value)


def test_source_patch_changes_only_two_receipt_metadata_entries():
    m = module()
    original = (SCRIPTS / "run.py").read_bytes()
    patched = m.patch_run(original)
    old_tree, new_tree = ast.parse(original), ast.parse(patched)
    old_receipt = next(n.value for n in ast.walk(old_tree)
                       if isinstance(n, ast.Assign) and any(
                           isinstance(t, ast.Name) and t.id == "receipt" for t in n.targets))
    new_receipt = next(n.value for n in ast.walk(new_tree)
                       if isinstance(n, ast.Assign) and any(
                           isinstance(t, ast.Name) and t.id == "receipt" for t in n.targets))
    values = {k.value: v for k, v in zip(new_receipt.keys, new_receipt.values)}
    assert ast.unparse(values["kind"]) == "'smoke' if args.smoke else 'fresh-id-compatible-rebuild'"
    assert ast.unparse(values["experiment_lineage"]) == "manifest['experiment_lineage']"
    new_receipt.keys = [k for k in new_receipt.keys if k.value != "experiment_lineage"]
    new_receipt.values = [values[k.value] for k in new_receipt.keys]
    for i, key in enumerate(new_receipt.keys):
        if key.value == "kind":
            new_receipt.values[i] = old_receipt.values[i]
    assert ast.dump(old_tree) == ast.dump(new_tree)
    assert m.patch_run(original.replace(b"\n", b"\r\n")) == patched
    with pytest.raises(ValueError, match="source|checksum"):
        m.patch_run(original.replace(b"warmup_ratio", b"warmup_factor"))
    with pytest.raises(ValueError, match="source|checksum"):
        m.patch_run(patched)
    assert (SCRIPTS / "run.py").read_bytes() == original


@pytest.mark.parametrize("section,key,value", [
    ("training", "learning_rate", 0.001), ("training", "epochs", 20),
    ("training", "effective_batch_size", 512), ("evaluation", "beams", 10),
    ("generator", "num_layers", 2), ("gate", "updates", 2),
])
def test_execution_recipe_changes_are_rejected_before_launch(tmp_path, section, key, value):
    spec = execution()
    spec[section][key] = value
    path = tmp_path / "altered-execution.json"
    write_json(path, spec)
    with pytest.raises(ValueError, match="checksum|frozen"):
        module().load_execution(
            path, ROOT / "configs" / "letter-reproduction-v1.json",
            ROOT / "configs" / "letter-openllama-v2-method-v1.json",
            ROOT / "configs" / "letter-tokenizer-reconstruction-v1.json",
        )


@pytest.mark.parametrize("kind", ["item", "inter", "index"])
def test_original_data_change_is_rejected(tmp_path, kind):
    manifest = catalog(tmp_path)
    module().verify_original(tmp_path, manifest, "Instruments")
    path = tmp_path / "data" / "Instruments" / f"Instruments.{kind}.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        module().verify_original(tmp_path, manifest, "Instruments")


@pytest.mark.parametrize("problem", [
    "missing", "extra", "noncontiguous", "position", "range", "fifth", "float", "bool", "mismatch",
])
def test_fresh_index_requires_exact_integer_four_code_catalog(problem):
    indices, codes = index_fixture()
    if problem == "missing":
        del indices["3"]
    elif problem == "extra":
        indices["4"] = indices["0"]
    elif problem == "noncontiguous":
        indices["04"] = indices.pop("3")
    elif problem == "position":
        indices["0"][0] = "<b_1>"
    elif problem == "range":
        indices["0"][0] = "<a_256>"
        codes[0, 0] = 256
    elif problem == "fifth":
        indices["0"].append("<e_0>")
    elif problem == "float":
        codes = codes.astype(np.float32)
    elif problem == "bool":
        codes = codes.astype(bool)
    else:
        codes[0, 0] = 2
    with pytest.raises(ValueError):
        module().validate_codes(indices, codes, 4)


def test_collisions_are_retained_without_a_fifth_uniqueness_token():
    indices, codes = index_fixture()
    report = module().validate_codes(indices, codes, 4)
    assert report["unique_sids"] == 3
    assert report["excess_collision_items"] == 1


def test_catalog_format_fixture_exercises_all_1024_atomic_sid_tokens():
    m = module()
    assert hasattr(m, "fixture_codes"), "The explicit synthetic format gate needs a code generator"
    codes = m.fixture_codes(9922)
    assert codes.shape == (9922, 4)
    assert len(set(map(tuple, codes.tolist()))) == 9922
    assert [len(np.unique(codes[:, i])) for i in range(4)] == [256] * 4


def test_final_bundle_binds_selected_and_latest_and_durable_bytes(tmp_path):
    path = tmp_path / "bundle"
    spec, manifest = bundle_fixture(path)
    report = module().verify_rq(path, spec, manifest, "Instruments", "full")
    assert report["selected_epoch"] == 2000
    assert report["final_epoch"] == 20000
    assert report["index_sha256"] == file_sha256(path / "payload" / "export" / "index.json")
    from bank import copy_artifact

    durable = tmp_path / "durable"
    copy_artifact(path, durable)
    module().verify_durable(path, durable)
    (durable / "payload" / "export" / "index.json").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        module().verify_durable(path, durable)


@pytest.mark.parametrize("problem", [
    "partial_bundle", "unfinished_epoch", "synthetic", "track", "variant", "method",
    "teacher", "semantic_shape", "selected_epoch", "selected_pointer", "checkpoint_spec",
    "checkpoint_corrupt", "cross_run_latest", "bundle_track", "index_content",
])
def test_rq_partial_cross_track_or_wrong_input_never_admitted(tmp_path, problem):
    path = tmp_path / "bundle"
    spec, manifest = bundle_fixture(path)
    payload = path / "payload"
    if problem == "partial_bundle":
        change_json(path / "completed.json", phase="checkpoint")
    elif problem == "unfinished_epoch":
        change_json(payload / "latest-state.json", epoch=19999)
    elif problem in ("synthetic", "track", "variant", "method", "teacher", "semantic_shape"):
        key, value = {
            "synthetic": ("synthetic_fixture", True), "track": ("track_id", "released-id-replay"),
            "variant": ("beta", 0.0), "method": ("method_spec_sha256", "0" * 64),
            "teacher": ("cf_sha256", "0" * 64), "semantic_shape": ("semantic_shape", [4, 32]),
        }[problem]
        change_json(payload / "run-config.json", **{key: value})
        change_json(payload / "completed.json",
                    run_config_sha256=file_sha256(payload / "run-config.json"))
    elif problem == "selected_epoch":
        change_json(payload / "completed.json", selected_epoch=4000)
    elif problem == "selected_pointer":
        change_json(payload / "completed.json", selected_checkpoint="../other/epoch-02000")
    elif problem == "checkpoint_spec":
        checkpoint = payload / "checkpoints" / "epoch-02000"
        state = torch.load(checkpoint / "state.pt", weights_only=False)
        state["spec"] = {**state["spec"], "track_id": "other"}
        torch.save(state, checkpoint / "state.pt")
        change_json(checkpoint / "completed.json", state_sha256=file_sha256(checkpoint / "state.pt"))
    elif problem == "checkpoint_corrupt":
        (payload / "checkpoints" / "epoch-02000" / "state.pt").write_bytes(b"corrupt")
    elif problem == "cross_run_latest":
        change_json(payload / "latest-state.json",
                    checkpoint="/scratch/other/checkpoints/epoch-20000")
    elif problem == "bundle_track":
        change_json(path / "completed.json", track_id="other")
    else:
        np.save(payload / "export" / "codes.npy", np.zeros((4, 4), dtype=np.int64))
    if problem not in ("partial_bundle", "bundle_track"):
        seal_bundle(path, spec)
    with pytest.raises(ValueError):
        module().verify_rq(path, spec, manifest, "Instruments", "full")


def test_derived_manifest_retains_interactions_but_not_released_claims(tmp_path):
    manifest = catalog(tmp_path)
    manifest["datasets"]["Instruments"]["unique_released_sids"] = 2
    manifest["published_results_not_reproduced_measurements"] = {"paper": "context"}
    parent = copy.deepcopy(manifest)
    indices, codes = index_fixture()
    lineage = {"kind": "fresh-id-compatible-rebuild", "index_sha256": "a" * 64}
    derived = module().derive_manifest(manifest, "b" * 64, "Instruments", lineage, indices, codes)
    assert manifest == parent
    assert derived["datasets"]["Instruments"]["sha256"]["inter"] == parent["datasets"]["Instruments"]["sha256"]["inter"]
    assert derived["datasets"]["Instruments"]["sha256"]["item"] == parent["datasets"]["Instruments"]["sha256"]["item"]
    assert derived["datasets"]["Instruments"]["sha256"]["index"] == "a" * 64
    assert derived["parent_manifest_sha256"] == "b" * 64
    assert derived["experiment_lineage"] == lineage
    assert "published_results_not_reproduced_measurements" not in derived
    assert "unique_released_sids" not in derived["datasets"]["Instruments"]
    assert derived["paper_reference_context_only"] == {"paper": "context"}


def test_stage_output_is_exclusive_even_if_empty(tmp_path):
    output = tmp_path / "reserved"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module().reserve_stage(output)
    (output / "owner.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        module().reserve_stage(output)
    assert (output / "owner.txt").read_text() == "keep"


def test_wrong_labeled_run_or_cross_index_resume_is_rejected(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    lineage = {"kind": "fresh-id-compatible-rebuild", "synthetic_fixture": False}
    stage = {
        "experiment_lineage": lineage, "dataset": "Instruments",
        "inputs": {"index": "a" * 64},
    }
    run = {
        "kind": "released-id-replay", "experiment_lineage": lineage, "dataset": "Instruments",
        "seed": 42, "temperature": 1.0, "inputs": stage["inputs"],
    }
    write_json(output / "run-config.json", run)
    with pytest.raises(ValueError, match="lineage|kind|identity"):
        module().validate_run_identity(output, stage, 42, 1.0, smoke=False)
    run["kind"] = "fresh-id-compatible-rebuild"
    run["inputs"] = {"index": "b" * 64}
    write_json(output / "run-config.json", run)
    with pytest.raises(ValueError, match="lineage|kind|identity"):
        module().validate_run_identity(output, stage, 42, 1.0, smoke=False)


def test_queue_is_frozen_seed_first_and_temperature_pairs_share_stage():
    cells = pipeline().queue_cells(execution())
    assert len(cells) == 39
    assert [c["seed"] for c in cells] == [42] * 13 + [17] * 13 + [23] * 13
    assert len({c["run_id"] for c in cells}) == 39
    for dataset in ("Instruments", "Beauty", "Yelp"):
        pair = [c for c in cells if c["dataset"] == dataset and c["seed"] == 42
                and c["variant"] == "full"]
        assert {c["temperature"] for c in pair} == {0.7, 1.0}
        assert len({c["stage_id"] for c in pair}) == 1


def test_readiness_waits_without_creating_stage_or_reading_gpu(tmp_path):
    cell = pipeline().queue_cells(execution())[0]
    status = pipeline().find_bundle([tmp_path / "incoming"], cell["rq_run_id"])
    assert status is None
    assert not list(tmp_path.iterdir())


def test_partial_evaluation_is_preserved_and_next_attempt_is_exclusive(tmp_path):
    first = tmp_path / "evaluation-attempt-001"
    first.mkdir()
    (first / "predictions.jsonl").write_text("partial\n")
    assert pipeline().next_evaluation(tmp_path) == tmp_path / "evaluation-attempt-002"
    assert (first / "predictions.jsonl").read_text() == "partial\n"


def test_rendezvous_never_uses_shared_standalone_port():
    command = pipeline().torchrun_prefix("cell")
    assert "--standalone" not in command
    assert "--rdzv_endpoint=localhost:0" in command
    assert "--nproc_per_node=2" in command
    assert command != pipeline().torchrun_prefix("cell")


def test_gate_comparison_detects_optimizer_rng_and_order_changes():
    m = module()
    assert hasattr(m, "equal_state"), "Bitwise native continuation comparison is missing"
    state = {"optimizer": {"moment": torch.tensor([1.0, 2.0])},
             "rng": (np.array([2, 3]), torch.tensor([1, 2], dtype=torch.uint8))}
    assert m.equal_state(state, copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed["optimizer"]["moment"][1] = 3
    assert not m.equal_state(state, changed)
    changed = copy.deepcopy(state)
    changed["rng"][0][0] = 4
    assert not m.equal_state(state, changed)


def test_real_launch_rejects_synthetic_stage_before_a_gpu_command():
    p = pipeline()
    assert hasattr(p, "require_real_stage"), "Synthetic fixtures need a hard launch boundary"
    with pytest.raises(ValueError, match="synthetic|fixture"):
        p.require_real_stage({"experiment_lineage": {"synthetic_fixture": True}})


def test_incomplete_or_missing_rank_state_cannot_resume(tmp_path):
    from types import SimpleNamespace

    m = module()
    write_json(tmp_path / "run-config.json", {
        "kind": "fresh-id-compatible-rebuild", "seed": 42, "options": {"data_seed": 42},
    })
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "pytorch_model.bin", "rng_state_0.pth"):
        torch.save({"value": torch.tensor([1])}, checkpoint / name)
    torch.save(SimpleNamespace(output_dir=str(tmp_path), seed=42, data_seed=42),
               checkpoint / "training_args.bin")
    write_json(checkpoint / "trainer_state.json", {"global_step": 3})
    write_json(checkpoint / "callback-state.json", {"early_stopping_patience_counter": 0})
    from protocol import seal_checkpoint

    with pytest.raises(ValueError, match="sealed"):
        m.select_resume(tmp_path)
    seal_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="rank RNG"):
        m.select_resume(tmp_path)
    torch.save({"value": torch.tensor([2])}, checkpoint / "rng_state_1.pth")
    seal_checkpoint(checkpoint)
    assert m.select_resume(tmp_path) == checkpoint
    torch.save(SimpleNamespace(output_dir=str(tmp_path / "different-run"), seed=42, data_seed=42),
               checkpoint / "training_args.bin")
    seal_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="own run|identity"):
        m.select_resume(tmp_path)


def test_local_summary_without_final_durable_ack_is_still_pending(tmp_path):
    p = pipeline()
    assert hasattr(p, "cell_is_banked"), "Local results alone must not advance the durable queue"
    cell = p.queue_cells(execution())[0]
    result = tmp_path / "results" / cell["run_id"] / "result.json"
    write_json(result, {"status": "verified_single_fresh_id_cell"})
    assert not p.cell_is_banked(tmp_path, cell)
    write_json(tmp_path / "acks" / f"{cell['run_id']}.json", {
        "status": "banked", "result_sha256": file_sha256(result), "run_id": cell["run_id"],
    })
    assert p.cell_is_banked(tmp_path, cell)
    write_json(result, {"status": "changed"})
    with pytest.raises(ValueError):
        p.cell_is_banked(tmp_path, cell)


def input_fixture(root):
    m, spec = module(), execution()
    manifest = catalog(root / "upstream")
    semantic, teacher = root / "semantic", root / "teacher"
    semantic.mkdir()
    teacher.mkdir()
    np.save(semantic / "embeddings.npy", np.zeros((4, 4096), dtype=np.float32))
    torch.save(torch.zeros((4, 32), dtype=torch.float32), teacher / "item-embeddings.pt")
    torch.save({"weight": torch.ones(1)}, teacher / "best.pt")
    ec = {
        "track_id": spec["track_id"], "dataset": "Instruments",
        "model_id": spec["semantic_encoder"]["model_id"],
        "revision": spec["semantic_encoder"]["revision"],
        "exact_original_llama1_reproduction": False,
        "method_spec_sha256": spec["method_spec_canonical_sha256"],
        "item_order": "row_equals_zero_based_item_id", "dtype": "float32",
        "item_json_sha256": manifest["datasets"]["Instruments"]["sha256"]["item"],
        "model_files_sha256": {"unit-only": "a" * 64},
        "tokenizer_files_sha256": {"unit-only": "b" * 64},
    }
    write_json(semantic / "run-config.json", ec)
    er = {
        **ec, "status": "pass", "smoke": False, "encoder": "OpenLLaMA-7B-v2",
        "shape": [4, 4096], "run_config_sha256": file_sha256(semantic / "run-config.json"),
        "embeddings_sha256": file_sha256(semantic / "embeddings.npy"),
        "token_audit": {"status": "pass", "records": 8, "strict_clean": 8, "over_cap": 0},
    }
    write_json(semantic / "embedding-receipt.json", er)
    write_json(semantic / "completed.json", {
        "status": "complete", "smoke": False, "kind": "semantic_embedding_bank",
        "shape": [4, 4096], "embeddings_sha256": er["embeddings_sha256"],
        "embedding_receipt_sha256": file_sha256(semantic / "embedding-receipt.json"),
    })
    write_json(teacher / "run-config.json", {
        "kind": "training-only-BCE-teacher-reconstruction", "smoke": False,
        "dataset": "Instruments", "model": {"hidden_units": 32},
        "input_sha256": manifest["datasets"]["Instruments"]["sha256"]["inter"],
    })
    write_json(teacher / "completed.json", {
        "kind": "training-only-BCE-teacher-reconstruction", "status": "complete",
        "smoke": False, "test_evaluated": False, "item_shape": [4, 32],
        "item_embeddings_sha256": file_sha256(teacher / "item-embeddings.pt"),
        "teacher_sha256": file_sha256(teacher / "best.pt"),
        "config_sha256": file_sha256(teacher / "run-config.json"),
    })
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
    spec["datasets"]["Instruments"].update({k: file_sha256(p) for k, p in files.items()})
    return m, semantic, teacher, spec, manifest, files


@pytest.mark.parametrize("key", [
    "semantic_sha256", "embedding_receipt_sha256", "semantic_config_sha256",
    "semantic_completed_sha256", "cf_sha256", "cf_config_sha256", "cf_completed_sha256",
    "teacher_sha256",
])
def test_each_full_real_input_and_provenance_file_is_hash_bound(tmp_path, key):
    m, semantic, teacher, spec, manifest, files = input_fixture(tmp_path)
    assert m.verify_inputs(semantic, teacher, spec, manifest, "Instruments")
    path = files[key]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_inputs(semantic, teacher, spec, manifest, "Instruments")


def private_binding_fixture():
    return {
        "name": "fixture-job", "subscription": "00000000-0000-4000-8000-000000000001",
        "resource_group": "fixture-group", "workspace": "fixture-workspace",
        "private_key": r"C:\fixture\id_rsa", "nodes": [0, 1, 2], "gpus_per_node": 8,
        "shared_parent": "/data/gen2/fixture-shared/reports/",
        "experiment_id": "fixture-study-1", "node_directory": "/scratch/fixture_study_1",
        "durable_directory": "/data/gen2/fixture-shared/reports/fixture_study_1",
        "single_writer_node": 0, "authorized_on": "2000-01-01",
        "authorization": "Unit-test fixture only",
    }


def test_private_binding_drives_scope_without_an_allocation_literal(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m.socket, "gethostname", lambda: "node-0")
    path = tmp_path / "binding.json"
    write_json(path, private_binding_fixture())
    root, destination = m.validate_scope(path)
    assert root.as_posix() == "/scratch/fixture_study_1"
    assert destination.as_posix() == "/data/gen2/fixture-shared/reports/fixture_study_1"


@pytest.mark.parametrize("key,value", [
    ("node_directory", "."), ("node_directory", "/"), ("node_directory", "/scratch"),
    ("node_directory", "/scratch/other"), ("node_directory", "/scratch/nested/fixture_study_1"),
    ("node_directory", "/scratch/other/../fixture_study_1"),
    ("node_directory", "/scratch//fixture_study_1"),
    ("node_directory", r"C:\scratch\fixture_study_1"),
    ("durable_directory", "/data/gen2/fixture-shared/reports"),
    ("durable_directory", "/data/gen2/other/fixture_study_1"),
    ("durable_directory", "/data/gen2/fixture-shared/reports/other"),
    ("durable_directory", "/data/gen2/fixture-shared/reports/nested/fixture_study_1"),
    ("shared_parent", "/"), ("shared_parent", "/data/gen2"),
    ("shared_parent", "/scratch/fixture_study_1/shared"),
    ("shared_parent", "//data/gen2/fixture-shared/reports"),
    ("experiment_id", "../escape"), ("experiment_id", "different-study"),
    ("single_writer_node", 1), ("single_writer_node", False),
    ("nodes", [1, 2]), ("nodes", [0, 0]), ("nodes", [0, True]),
    ("nodes", "0,1,2"), ("gpus_per_node", 2), ("gpus_per_node", 8.0),
    ("subscription", "not-a-uuid"), ("authorized_on", "not-a-date"),
    ("private_key", "relative-key"), ("authorization", ""),
    ("unrecognized_scope_override", "anything"),
])
def test_private_binding_rejects_unsafe_or_inconsistent_scope(tmp_path, monkeypatch, key, value):
    m = module()
    monkeypatch.setattr(m.socket, "gethostname", lambda: "node-0")
    binding = private_binding_fixture()
    binding[key] = value
    path = tmp_path / "binding.json"
    write_json(path, binding)
    with pytest.raises((ValueError, RuntimeError)):
        m.validate_scope(path)


def test_private_binding_requires_its_schema_and_node_zero(tmp_path, monkeypatch):
    m = module()
    path = tmp_path / "binding.json"
    write_json(path, private_binding_fixture())
    monkeypatch.setattr(m.socket, "gethostname", lambda: "node-1")
    with pytest.raises(RuntimeError, match="node0"):
        m.validate_scope(path)
    monkeypatch.setattr(m.socket, "gethostname", lambda: "node-0")
    binding = private_binding_fixture()
    del binding["shared_parent"]
    write_json(path, binding)
    with pytest.raises(ValueError, match="schema"):
        m.validate_scope(path)


def test_runtime_versions_work_without_touching_an_older_queue(tmp_path, monkeypatch):
    import inspect

    p = pipeline()
    assert "work" in inspect.signature(p.Runtime).parameters, "Work version must be explicitly selected"
    root, destination = tmp_path / "owned", tmp_path / "durable"
    root.mkdir()
    project, control = root / "project", root / "control-fixture"
    monkeypatch.setattr(p, "validate_scope", lambda _: (root, destination))
    monkeypatch.setattr(p, "require_canonical", lambda *args: None)
    monkeypatch.setattr(p, "load_execution", lambda *args: execution())
    old = root / "fresh-t5-previous" / "queue-binding.json"
    write_json(old, {"immutable": "previous queue"})
    before = old.read_bytes()
    work = root / "fresh-t5-next"
    runtime = p.Runtime(tmp_path / "private-binding.json", project, control, work)
    assert runtime.work == work
    assert runtime.bank_root == destination / "openllama-v2" / work.name
    assert old.read_bytes() == before
    for invalid in (root, project, control, root / "upstream",
                    root / "nested" / "fresh-t5-next", tmp_path / "fresh-t5-outside"):
        with pytest.raises(ValueError, match="work"):
            p.Runtime(tmp_path / "private-binding.json", project, control, invalid)
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == work)
    with pytest.raises(ValueError, match="work"):
        p.Runtime(tmp_path / "private-binding.json", project, control, work)


@pytest.mark.parametrize("evaluation_state", ["new", "partial", "completed"])
def test_complete_run_is_banked_once_before_any_evaluation(tmp_path, monkeypatch, evaluation_state):
    p = pipeline()
    runtime = object.__new__(p.Runtime)
    runtime.work, runtime.spec = tmp_path / "work", execution()
    cell = p.queue_cells(runtime.spec)[0]
    stage = runtime.work / "stages" / cell["stage_id"]
    events = []
    monkeypatch.setattr(runtime, "admit", lambda *args: stage)
    monkeypatch.setattr(runtime, "gate", lambda *args: None)
    monkeypatch.setattr(runtime, "train", lambda *args: events.append("train"))

    def bank(source, relative):
        if relative == f"runs/{cell['run_id']}":
            events.append("bank-final-run")

    def invoke(*args, **kwargs):
        events.append("launch-evaluation")

    def verify(*args):
        events.append("verify-evaluation")
        return {"status": "unit_fixture_only"}

    monkeypatch.setattr(runtime, "bank", bank)
    monkeypatch.setattr(runtime, "invoke", invoke)
    monkeypatch.setattr(p, "verify_evaluation", verify)
    previous = runtime.work / "evaluations" / cell["run_id"] / "evaluation-attempt-001"
    if evaluation_state == "completed":
        write_json(previous / "completed.json", {"status": "unit_fixture_only"})
    elif evaluation_state == "partial":
        previous.mkdir(parents=True)
        (previous / "predictions.jsonl").write_text("preserve this partial attempt\n")
    runtime.execute_cell(cell, tmp_path / "bundle")
    assert events.count("bank-final-run") == 1
    assert events.index("train") < events.index("bank-final-run") < events.index("verify-evaluation")
    if evaluation_state != "completed":
        assert events.index("bank-final-run") < events.index("launch-evaluation")
    if evaluation_state == "partial":
        assert (previous / "predictions.jsonl").read_text() == "preserve this partial attempt\n"


def test_failed_final_model_banking_prevents_test_evaluation(tmp_path, monkeypatch):
    p = pipeline()
    runtime = object.__new__(p.Runtime)
    runtime.work, runtime.spec = tmp_path / "work", execution()
    cell = p.queue_cells(runtime.spec)[0]
    monkeypatch.setattr(runtime, "admit", lambda *args: tmp_path / "stage")
    monkeypatch.setattr(runtime, "gate", lambda *args: None)
    monkeypatch.setattr(runtime, "train", lambda *args: None)
    evaluation_calls = []

    def bank(source, relative):
        if relative == f"runs/{cell['run_id']}":
            raise RuntimeError("deliberately failed final-model bank")

    monkeypatch.setattr(runtime, "bank", bank)
    monkeypatch.setattr(runtime, "invoke", lambda *args, **kwargs: evaluation_calls.append("launch"))
    monkeypatch.setattr(p, "verify_evaluation",
                        lambda *args: evaluation_calls.append("verify") or {"status": "unit_fixture_only"})
    with pytest.raises(RuntimeError, match="deliberately failed"):
        runtime.execute_cell(cell, tmp_path / "bundle")
    assert evaluation_calls == []


@pytest.mark.parametrize("exit_code", [0, 17])
def test_failed_exporter_is_detected_before_training_startup(tmp_path, exit_code):
    p = pipeline()
    assert hasattr(p, "wait_for_exporter"), "Training needs an exporter startup gate"
    log = tmp_path / "exporter.log"
    log.write_text("")
    exporter = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({exit_code})"])
    exporter.wait(timeout=10)
    with pytest.raises(p.ExporterFailure, match=str(exit_code)):
        p.wait_for_exporter(exporter, log, lambda: False)


def test_invoke_never_starts_a_fit_with_a_dead_exporter(tmp_path, monkeypatch):
    p = pipeline()
    runtime = object.__new__(p.Runtime)
    runtime.work, runtime.root = tmp_path / "work", tmp_path
    marker = tmp_path / "training-started"
    argv = [sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).write_text('started')"]
    monkeypatch.setattr(runtime, "command", lambda *args, **kwargs: (argv, os.environ.copy()))
    exporter = subprocess.Popen([sys.executable, "-c", "raise SystemExit(29)"])
    exporter.wait(timeout=10)
    with pytest.raises(p.ExporterFailure, match="29"):
        runtime.invoke(argv, "fixture", exporter=exporter, completed=lambda: False)
    assert not marker.exists()
    with pytest.raises(ValueError, match="both"):
        runtime.invoke(argv, "fixture", exporter=exporter)


@pytest.mark.skipif(os.name != "posix", reason="Owned Linux process-session drain runs in the native gate")
@pytest.mark.parametrize("launcher", ["python_tree", "torchrun"])
def test_failed_exporter_gracefully_drains_only_owned_training_tree(tmp_path, launcher):
    p = pipeline()
    assert hasattr(p, "wait_for_training"), "A blocked fit must be supervised for exporter death"
    ready, sealed = tmp_path / "worker-ready", tmp_path / "sealed-fixture-state.json"
    worker_code = (
        "import signal,sys,time\nfrom pathlib import Path\n"
        f"def drain(*args):\n Path({str(sealed)!r}).write_text('fixture drained at safe boundary');sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,drain)\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "while True: time.sleep(.05)\n"
    )
    launcher_code = (
        "import signal,subprocess,sys,time\n"
        f"worker=subprocess.Popen([sys.executable,'-c',{worker_code!r}])\n"
        "def drain(*args):\n worker.wait(timeout=10);sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,drain)\n"
        "while True: time.sleep(.05)\n"
    )
    if launcher == "torchrun":
        fixture = tmp_path / "training_fixture.py"
        fixture.write_text(
            "import os,signal,sys,time\nfrom pathlib import Path\n"
            "rank=os.environ['RANK']\n"
            f"def drain(*args):\n Path({str(sealed)!r}+rank).write_text('fixture drained at safe boundary');sys.exit(0)\n"
            "signal.signal(signal.SIGTERM,drain)\n"
            f"Path({str(ready)!r}+rank).write_text('ready')\n"
            "while True: time.sleep(.05)\n"
        )
        argv = p.torchrun_prefix("exporter-supervision-fixture") + [str(fixture)]
        ready_files = [Path(str(ready) + str(rank)) for rank in (0, 1)]
        sealed_files = [Path(str(sealed) + str(rank)) for rank in (0, 1)]
    else:
        argv = [sys.executable, "-c", launcher_code]
        ready_files, sealed_files = [ready], [sealed]
    training = subprocess.Popen(argv, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                 start_new_session=True)
    exporter = subprocess.Popen([sys.executable, "-c", "raise SystemExit(19)"])
    try:
        exporter.wait(timeout=10)
        deadline = time.monotonic() + 30
        while not all(path.exists() for path in ready_files) and time.monotonic() < deadline:
            time.sleep(.02)
        assert all(path.exists() for path in ready_files)
        with pytest.raises(p.ExporterFailure, match="19"):
            p.wait_for_training(training, exporter, lambda: False, tmp_path)
        assert training.poll() is not None
        assert all(path.read_text() == "fixture drained at safe boundary" for path in sealed_files)
        assert unrelated.poll() is None
        failure = json.loads((tmp_path / "supervision.json").read_text())
        assert failure["exporter_pid"] == exporter.pid
        assert failure["training_pid"] == training.pid
        assert failure["drain"] == "completed"
    finally:
        if training.poll() is None:
            os.killpg(training.pid, 15)
            training.wait(timeout=15)
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_normal_exporter_exit_after_validated_completion_is_allowed(tmp_path):
    p = pipeline()
    assert hasattr(p, "wait_for_training"), "Normal exporter completion needs a distinct state"
    training = subprocess.Popen([sys.executable, "-c", "pass"])
    exporter = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        exporter.wait(timeout=10)
        assert p.wait_for_training(training, exporter, lambda: True, tmp_path) == 0
        assert not (tmp_path / "supervision.json").exists()
    finally:
        if training.poll() is None:
            training.terminate()
            training.wait(timeout=10)
