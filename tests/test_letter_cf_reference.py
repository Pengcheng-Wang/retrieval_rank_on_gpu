import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def implementation():
    assert (SCRIPTS / "cf_reference.py").is_file(), "Frozen reference consumer is missing"
    return importlib.import_module("cf_reference")


@pytest.fixture
def reference(tmp_path, monkeypatch):
    from bank import source_manifest
    from cf import MODEL_SHA256
    from protocol import file_sha256, write_json

    module = implementation()
    runtime = {
        "python": "3.10.21", "torch": "1.13.1+cu117", "numpy": "1.24.4",
        "device": "cpu", "precision": "float32", "threads": 2, "interop_threads": 1,
    }
    monkeypatch.setattr(module, "runtime_signature", lambda: dict(runtime))
    source = tmp_path / "current-source"
    source.mkdir()
    for name in ("mechanisms.py", "cf.py", "prepare.py", "protocol.py"):
        shutil.copyfile(SCRIPTS / name, source / name)
    monkeypatch.setattr(module, "SCRIPT_DIR", source)
    upstream = tmp_path / "upstream"
    interactions = {str(i): [0, 1, 2, 3, 4] for i in range(100)}
    items = {str(i): {"title": str(i), "description": ""} for i in range(16)}
    indices = {str(i): [f"<a_{i}>", "<b_0>", "<c_0>", "<d_0>"] for i in range(16)}
    hashes = {}
    for name, value in (("inter", interactions), ("item", items), ("index", indices)):
        path = upstream / "data/Instruments" / f"Instruments.{name}.json"
        write_json(path, value)
        hashes[name] = file_sha256(path)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"datasets": {"Instruments": {
        "users": 100, "items": 16, "interactions": 500, "sha256": hashes,
    }}})
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "best.pt").write_bytes(b"owned selected teacher fixture")
    (teacher / "item-embeddings.pt").write_bytes(b"owned item bank fixture")
    write_json(teacher / "run-config.json", {
        "kind": "training-only-BCE-teacher-reconstruction", "dataset": "Instruments",
        "epochs": 201, "seed": 42, "model": {"hidden_units": 32, "maxlen": 50},
        "input_sha256": hashes["inter"], "source_model_sha256": MODEL_SHA256,
    })
    teacher_done = {
        "status": "complete", "smoke": False, "test_evaluated": False, "epoch": 80,
        "config_sha256": file_sha256(teacher / "run-config.json"),
        "teacher_sha256": file_sha256(teacher / "best.pt"),
        "item_embeddings_sha256": file_sha256(teacher / "item-embeddings.pt"),
    }
    write_json(teacher / "completed.json", teacher_done)
    bundle = tmp_path / "reference"
    payload = bundle / "payload"
    archived = bundle / "source"
    archived.mkdir(parents=True)
    for path in source.iterdir():
        shutil.copyfile(path, archived / path.name)
    (archived / "runner.py").write_text("# fixture runner\n")
    predictions = [{"user": str(i), "target_item": 4, "top10": list(range(10))}
                   for i in range(100)]
    prediction_path = payload / "Instruments/predictions.json"
    write_json(prediction_path, predictions)
    score = sum(1 / math.log2(6) for _ in range(100)) / 100
    result = {
        "users": 100, "items": 16, "selected_epoch": 80, "device": "cpu",
        "history_limit": 50, "seen_item_exclusion": False,
        "tie_rule": "original_item_id_ascending",
        "teacher_sha256": teacher_done["teacher_sha256"],
        "predictions_sha256": file_sha256(prediction_path), "independently_recomputed": True,
        "metrics": {"recall@5": 1.0, "recall@10": 1.0, "ndcg@5": score, "ndcg@10": score},
        "wall_seconds": 1.0,
    }
    config = {
        "kind": "frozen_32d_teacher_reference_only", "quantized_replacement": False,
        "tokenizer_analyzed": False, "runtime": runtime, "history_limit": 50,
        "seen_item_exclusion": False, "tie_rule": "original_item_id_ascending",
        "ranking_helper": "mechanisms.rank_cf", "manifest_sha256": file_sha256(manifest),
        "runner_sha256": file_sha256(archived / "runner.py"),
        "evaluator_sha256": {p.name: file_sha256(p) for p in source.iterdir()},
        "teachers": {"Instruments": {
            "completion_sha256": file_sha256(teacher / "completed.json"),
            "config_sha256": teacher_done["config_sha256"],
            "selected_teacher_sha256": teacher_done["teacher_sha256"],
            "item_embeddings_sha256": teacher_done["item_embeddings_sha256"],
            "selected_epoch": 80, "fit_test_evaluated": False,
        }},
    }
    write_json(payload / "run-config.json", config)
    write_json(payload / "admission.json", {
        "status": "pass", "test_scored": False, "validation_probe_users_per_dataset": 100,
        "run_config_sha256": file_sha256(payload / "run-config.json"),
    })
    write_json(payload / "Instruments/test-access.json", {
        "status": "started", "admission_sha256": file_sha256(payload / "admission.json"),
        "teacher_sha256": teacher_done["teacher_sha256"], "fit_receipt_unchanged": True,
    })
    write_json(payload / "Instruments/completed.json", {"status": "complete", **result})
    done = {
        "status": "complete", "kind": config["kind"], "test_evaluated": True,
        "quantized_replacement": False, "tokenizer_analyzed": False,
        "run_config_sha256": file_sha256(payload / "run-config.json"),
        "results": {"Instruments": result},
        "files_sha256": {p.relative_to(payload).as_posix(): file_sha256(p)
                         for p in payload.rglob("*") if p.is_file()},
    }
    write_json(payload / "completed.json", done)
    write_json(bundle / "completed.json", {
        "status": "complete", "kind": "frozen_teacher_reference_bundle",
        "reference_completed_sha256": file_sha256(payload / "completed.json"),
        "payload_manifest": source_manifest(payload), "source_manifest": source_manifest(archived),
    })
    published = tmp_path / "published.json"
    write_json(published, {
        "kind": "frozen_32d_cf_teacher_reference", "status": "complete",
        "evaluation_completed_sha256": file_sha256(payload / "completed.json"),
        "evaluation_config_sha256": file_sha256(payload / "run-config.json"),
        "manifest_sha256": file_sha256(manifest), "runtime": runtime,
        "evaluator_sha256": config["evaluator_sha256"], "runner_sha256": config["runner_sha256"],
        "results": {"Instruments": result},
    })
    args = SimpleNamespace(
        cf_reference=bundle, cf_reference_manifest=published, teacher=teacher,
        manifest=manifest, upstream=upstream, dataset="Instruments", device="cpu",
        output=tmp_path / "analysis",
    )
    return SimpleNamespace(args=args, result=result, payload=payload, source=source,
                           runtime=runtime, module=module)


def test_verified_reference_is_copied_without_model_scoring(reference):
    result = reference.module.reuse_reference(reference.args)
    copied = reference.args.output / "cf-original-predictions.json"
    assert copied.read_bytes() == (reference.payload / "Instruments/predictions.json").read_bytes()
    assert result["metrics"] == reference.result["metrics"]
    assert result["reused_frozen_reference"] is True
    with pytest.raises(FileExistsError):
        reference.module.reuse_reference(reference.args)


@pytest.mark.parametrize("field,value", [
    ("device", "cuda:0"), ("threads", 4), ("interop_threads", 2),
    ("precision", "float64"), ("torch", "different"), ("python", "different"),
])
def test_runtime_drift_is_rejected_before_output(reference, monkeypatch, field, value):
    changed = {**reference.runtime, field: value}
    monkeypatch.setattr(reference.module, "runtime_signature", lambda: changed)
    with pytest.raises(ValueError, match="runtime|CPU"):
        reference.module.reuse_reference(reference.args)
    assert not reference.args.output.exists()


@pytest.mark.parametrize("member", [
    "Instruments/predictions.json", "Instruments/test-access.json", "run-config.json",
])
def test_payload_tampering_is_rejected(reference, member):
    with (reference.payload / member).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum|inventory|manifest"):
        reference.module.reuse_reference(reference.args)
    assert not reference.args.output.exists()


def test_different_teacher_and_input_are_not_compatible(reference):
    (reference.args.teacher / "best.pt").write_bytes(b"different model")
    with pytest.raises(ValueError, match="checksum"):
        reference.module.reuse_reference(reference.args)


def test_output_must_not_overlap_the_immutable_reference(reference):
    reference.args.output = reference.args.cf_reference / "new-output"
    with pytest.raises(ValueError, match="overlap"):
        reference.module.reuse_reference(reference.args)


def test_orchestration_only_source_change_is_allowed(reference):
    path = reference.source / "mechanisms.py"
    text = path.read_text().replace("def analyze(args) -> None:", "def analyze(args) -> None:\n    pass")
    path.write_text(text)
    assert reference.module.reuse_reference(reference.args)["reused_frozen_reference"]


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace("128):", "64):"),
    lambda s: s + "\nrank_cf = lambda *args: None\n",
    lambda s: s.replace("import math", "import math as other_math"),
])
def test_scorer_or_global_changes_are_not_hidden_by_module_version(reference, mutation):
    path = reference.source / "mechanisms.py"
    path.write_text(mutation(path.read_text()))
    with pytest.raises(ValueError, match="scoring|source"):
        reference.module.reuse_reference(reference.args)


def test_non_scoring_dependency_changes_are_rejected(reference):
    with (reference.source / "cf.py").open("a") as stream:
        stream.write("\n# different helper\n")
    with pytest.raises(ValueError, match="checksum"):
        reference.module.reuse_reference(reference.args)


def test_replacement_requires_cache_instead_of_recomputing_the_original(tmp_path):
    import mechanisms
    from test_letter_mechanisms import analysis_fixture

    args = analysis_fixture(tmp_path)
    args.rank_replacement = True
    with pytest.raises(ValueError, match="frozen reference"):
        mechanisms.analyze(args)
    assert not args.output.exists()


def test_reference_cannot_bypass_the_complete_tokenizer_gate(tmp_path, monkeypatch):
    import mechanisms
    from test_letter_mechanisms import analysis_fixture

    args = analysis_fixture(tmp_path)
    args.rank_replacement = True
    args.cf_reference = tmp_path / "cached"
    args.cf_reference_manifest = tmp_path / "published.json"
    (args.tokenizer_run / "checkpoints/epoch-20000/state.pt").unlink()
    monkeypatch.setattr(implementation(), "verified_reference",
                        lambda _: pytest.fail("Cache was read before final tokenizer validation"))
    with pytest.raises((ValueError, FileNotFoundError)):
        mechanisms.analyze(args)
    assert not args.output.exists()


@pytest.mark.parametrize("input_name", ["tokenizer_run", "teacher", "upstream"])
def test_analysis_cannot_write_inside_immutable_inputs(tmp_path, input_name):
    import mechanisms
    from test_letter_mechanisms import analysis_fixture

    args = analysis_fixture(tmp_path)
    args.output = getattr(args, input_name) / "derived-analysis"
    with pytest.raises(ValueError, match="overlap"):
        mechanisms.analyze(args)
    assert not args.output.exists()


def test_replacement_invokes_neural_scoring_only_for_quantized_vectors(tmp_path, monkeypatch):
    import cf
    import mechanisms
    import torch
    from protocol import file_sha256, write_json
    from test_letter_mechanisms import analysis_fixture

    args = analysis_fixture(tmp_path)
    args.rank_replacement = True
    args.cf_reference = tmp_path / "cached"
    args.cf_reference_manifest = tmp_path / "published.json"
    args.sasrec = tmp_path / "sasrec"
    args.device = "cpu"
    config = json.loads((args.teacher / "run-config.json").read_text())
    config["model"] = {"maxlen": 50}
    write_json(args.teacher / "run-config.json", config)
    done = json.loads((args.teacher / "completed.json").read_text())
    done.update(config_sha256=file_sha256(args.teacher / "run-config.json"),
                teacher_sha256="a" * 64)
    write_json(args.teacher / "completed.json", done)
    rows = torch.load(args.teacher / "item-embeddings.pt", weights_only=True)
    weights = torch.cat([torch.zeros(1, 32), rows])

    class Model(torch.nn.Module):
        def __init__(self, users, items, options):
            super().__init__()
            self.item_emb = torch.nn.Embedding(items + 1, 32)

    monkeypatch.setattr(cf, "load_model_class", lambda _: Model)
    monkeypatch.setattr(mechanisms, "load_teacher_model_state",
                        lambda *args: {"item_emb.weight": weights.clone()})
    cache = implementation()
    monkeypatch.setattr(cache, "verified_reference", lambda _: None)
    monkeypatch.setattr(cache, "reuse_reference",
                        lambda _: {"reused_frozen_reference": True, "users": 2})
    calls = []

    def score(model, interactions, length, device, output):
        calls.append(output.name)
        assert output.name == "cf-quantized-predictions.json"
        write_json(output, [])
        return {"users": 2, "predictions_sha256": file_sha256(output)}

    monkeypatch.setattr(mechanisms, "rank_cf", score)
    mechanisms.analyze(args)
    assert calls == ["cf-quantized-predictions.json"]
    result = json.loads((args.output / "completed.json").read_text())
    assert result["cf_original_reference"]["reused_frozen_reference"]
