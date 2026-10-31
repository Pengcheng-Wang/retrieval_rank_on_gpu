import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    return importlib.import_module("figure8_tokenizer")


def compare_module():
    return importlib.import_module("figure8_tokenizer_compare")


def execution():
    return json.loads(
        (ROOT / "configs" / "letter-fresh-id-execution-v1.json").read_text()
    )


def normalized_bundle_config(spec, items):
    frozen = execution()
    levels = len(spec["codes"])
    return {
        **spec,
        "cf_provenance": "training_only_BCE_teacher_reconstruction",
        "cf_sha256": frozen["datasets"]["Instruments"]["cf_sha256"],
        "cf_shape": [items, 32],
        "claim_type": frozen["claim_type"],
        "embedding_receipt_sha256":
            frozen["datasets"]["Instruments"]["embedding_receipt_sha256"],
        "harness_sha256":
            "21e3909b8ea84793779820db8442e83b12a8597e3caf9d294d9f18bcb95a4532",
        "learning_rate": 0.001,
        "method_spec_sha256": frozen["method_spec_canonical_sha256"],
        "num_workers": 0,
        "purpose": {
            "admissible_final_tokenizer": True,
            "kind": "tokenizer_reconstruction",
        },
        "runtime_parameterization": {
            "clusters": spec["clusters"],
            "codebook_shapes": [
                [size, spec["dimension"]] for size in spec["codes"]
            ],
            "emitted_sid_length": levels,
            "epsilons": spec["epsilons"],
            "last_token_prefix": "abcdefgh"[levels - 1],
            "levels": levels,
        },
        "selection": "first minimum collision rate at scheduled evaluations",
        "semantic_sha256":
            frozen["datasets"]["Instruments"]["semantic_sha256"],
        "semantic_shape": [items, 4096],
        "source": frozen["rq"]["source"],
        "synthetic_fixture": False,
        "track_id": frozen["track_id"],
        "weight_decay": 0.0001,
    }


def test_tokenizer_queue_is_exact_and_isolated():
    cells = module().tokenizer_cells()

    assert [cell["arm"] for cell in cells] == [
        "alpha-0p001", "alpha-0p01", "alpha-0p1",
        "beta-0p00001", "beta-0p001", "beta-0p01",
        "clusters-5", "clusters-15", "clusters-20",
        "levels-2", "levels-6", "levels-8",
        "codes-64", "codes-128", "codes-512",
    ]
    assert len({cell["run_id"] for cell in cells}) == 15
    assert len({cell["stage_id"] for cell in cells}) == 15
    assert all(cell["dataset"] == "Instruments" for cell in cells)
    assert all(cell["seed"] == 42 and cell["temperature"] == 0.7 for cell in cells)
    assert all(cell["learning_rate"] == 0.0005 and cell["epochs"] == 200 for cell in cells)
    assert all(cell["world_size"] == 2 and cell["effective_batch_size"] == 1024 for cell in cells)
    assert all("figure8-tokenizer" in cell["run_id"] for cell in cells)
    assert not any("fresh-t5-" in cell["run_id"] for cell in cells)


def test_frozen_tokenizer_plan_has_no_selection_output():
    plan = module().load_plan(ROOT / "configs" / "letter-figure8-tokenizer-v1.json")

    assert plan["work_basename"] == "fresh-t5-figure8-tokenizer-v1"
    assert plan["arms"] == [cell["arm"] for cell in module().tokenizer_cells()]
    assert plan["baseline"]["temperature"] == 0.7
    assert plan["test_selection"] == "descriptive only; no winner or recipe selection"
    assert not module().contains_winner_field(plan)


def test_operational_v2_plan_preserves_the_frozen_scientific_grid():
    original = module().load_plan(
        ROOT / "configs" / "letter-figure8-tokenizer-v1.json"
    )
    replacement = module().load_plan(
        ROOT / "configs" / "letter-figure8-tokenizer-v2.json"
    )

    assert replacement["kind"] == "letter_figure8_tokenizer_downstream_v2"
    assert replacement["work_basename"] == "fresh-t5-figure8-tokenizer-v2"
    for key in (
        "dataset", "seed", "temperature", "learning_rate", "epochs",
        "world_size", "effective_batch_size", "beams", "seen_item_exclusion",
        "max_new_tokens", "arms", "baseline", "panels", "metric_identities",
        "comparison_metrics", "bootstrap_seed", "bootstrap_replicates",
        "selection", "test_selection",
    ):
        assert replacement[key] == original[key]


def test_operational_plan_version_and_work_namespace_cannot_be_mixed(tmp_path):
    plan = json.loads(
        (ROOT / "configs" / "letter-figure8-tokenizer-v1.json").read_text()
    )
    plan["kind"] = "letter_figure8_tokenizer_downstream_v2"
    path = tmp_path / "mixed-plan.json"
    path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="version|namespace"):
        module().load_plan(path)


@pytest.mark.parametrize(
    "field",
    ["temperature", "max_new_tokens", "seen_item_exclusion", "selection", "arms"],
)
def test_frozen_tokenizer_plan_rejects_scientific_drift(tmp_path, field):
    plan = json.loads(
        (ROOT / "configs" / "letter-figure8-tokenizer-v1.json").read_text()
    )
    plan[field] = [] if field == "arms" else "changed"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="frozen scientific grid"):
        module().load_plan(path)


def test_arm_specs_change_exactly_one_tokenizer_factor():
    plan = module().load_plan(ROOT / "configs" / "letter-figure8-tokenizer-v1.json")
    baseline = module().baseline_tokenizer_spec()

    for cell in module().tokenizer_cells():
        spec = module().arm_tokenizer_spec(plan, cell["arm"])
        changed = {
            key for key in ("codes", "clusters", "alpha", "beta")
            if spec[key] != baseline[key]
        }
        assert len(changed) == 1
        assert spec["seed"] == 42 and spec["epochs"] == 20000
        assert len(spec["codes"]) == len(spec["epsilons"])


def test_real_levels2_operational_spec_projects_to_frozen_science():
    plan = module().load_plan(
        ROOT / "configs" / "letter-figure8-tokenizer-v1.json"
    )
    expected = module().arm_tokenizer_spec(plan, "levels-2")
    spec = {
        "alpha": 0.02,
        "batch_size": 1024,
        "beta": 0.0001,
        "cf": "/scratch/letter_20261028_a77dfb13/runs/Instruments-cf-bce-s42/item-embeddings.pt",
        "checkpoint_every": 500,
        "clusters": 10,
        "codes": [256, 256],
        "control": "/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/control",
        "dataset": "Instruments",
        "dimension": 32,
        "embedding_receipt": "/scratch/letter_20261028_a77dfb13/embeddings-openllama-v2-v1/Instruments/embedding-receipt.json",
        "epochs": 20000,
        "epsilons": [0.0, 0.003],
        "eval_step": 2000,
        "gpu": 1,
        "host": "node-1",
        "kind": "figure8-one-factor-tokenizer-v1",
        "layers": [2048, 1024, 512, 256, 128, 64],
        "manifest": "/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/control/configs/letter-reproduction-v1.json",
        "method_spec": "/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/control/configs/letter-openllama-v2-method-v1.json",
        "output": "/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/runs/levels-2",
        "root": "/scratch/letter_20261028_a77dfb13",
        "seed": 42,
        "semantic": "/scratch/letter_20261028_a77dfb13/embeddings-openllama-v2-v1/Instruments/embeddings.npy",
    }

    assert hashlib.sha256(
        (json.dumps(spec, indent=2) + "\n").encode()
    ).hexdigest() == "395ac8a97ec3d089e6f7a93bfc5f7f8f0a1123fbab74a7beccefd9a3ad973b2e"
    assert module().validate_arm_launch_spec(
        spec, expected, "levels-2",
    ) == expected
    with pytest.raises(ValueError, match="scientific"):
        module().validate_arm_launch_spec(
            {**spec, "epsilons": [0.0, 0.0]}, expected, "levels-2",
        )
    with pytest.raises(ValueError, match="operational"):
        module().validate_arm_launch_spec(
            {**spec, "output": spec["output"].replace("levels-2", "levels-6")},
            expected,
            "levels-2",
        )


def test_real_alpha_operational_spec_accepts_pinned_incoming_teacher_path():
    plan = module().load_plan(
        ROOT / "configs" / "letter-figure8-tokenizer-v2.json"
    )
    expected = module().arm_tokenizer_spec(plan, "alpha-0p001")
    spec = {
        **expected,
        "cf": "/scratch/letter_fixture/incoming/Instruments-cf-bce-s42/item-embeddings.pt",
        "checkpoint_every": 500,
        "control": "/scratch/letter_fixture/figure8-sweeps-v1/control",
        "embedding_receipt":
            "/scratch/letter_fixture/embeddings-openllama-v2-v1/Instruments/embedding-receipt.json",
        "gpu": 0,
        "host": "node-0",
        "kind": "figure8-one-factor-tokenizer-v1",
        "manifest":
            "/scratch/letter_fixture/figure8-sweeps-v1/control/configs/letter-reproduction-v1.json",
        "method_spec":
            "/scratch/letter_fixture/figure8-sweeps-v1/control/configs/letter-openllama-v2-method-v1.json",
        "output": "/scratch/letter_fixture/figure8-sweeps-v1/runs/alpha-0p001",
        "root": "/scratch/letter_fixture",
        "semantic":
            "/scratch/letter_fixture/embeddings-openllama-v2-v1/Instruments/embeddings.npy",
    }

    assert module().validate_arm_launch_spec(
        spec, expected, "alpha-0p001",
    ) == expected


def test_real_levels2_bundle_config_requires_normalized_provenance():
    plan = module().load_plan(
        ROOT / "configs" / "letter-figure8-tokenizer-v1.json"
    )
    expected = module().arm_tokenizer_spec(plan, "levels-2")
    execution = json.loads(
        (ROOT / "configs" / "letter-fresh-id-execution-v1.json").read_text()
    )
    config = {
        **expected,
        "cf_provenance": "training_only_BCE_teacher_reconstruction",
        "cf_sha256": execution["datasets"]["Instruments"]["cf_sha256"],
        "cf_shape": [9922, 32],
        "claim_type": execution["claim_type"],
        "embedding_receipt_sha256":
            execution["datasets"]["Instruments"]["embedding_receipt_sha256"],
        "harness_sha256":
            "21e3909b8ea84793779820db8442e83b12a8597e3caf9d294d9f18bcb95a4532",
        "learning_rate": 0.001,
        "method_spec_sha256": execution["method_spec_canonical_sha256"],
        "num_workers": 0,
        "purpose": {
            "admissible_final_tokenizer": True,
            "kind": "tokenizer_reconstruction",
        },
        "runtime_parameterization": {
            "clusters": 10,
            "codebook_shapes": [[256, 32], [256, 32]],
            "emitted_sid_length": 2,
            "epsilons": [0.0, 0.003],
            "last_token_prefix": "b",
            "levels": 2,
        },
        "seed": 42,
        "selection": "first minimum collision rate at scheduled evaluations",
        "semantic_sha256":
            execution["datasets"]["Instruments"]["semantic_sha256"],
        "semantic_shape": [9922, 4096],
        "source": execution["rq"]["source"],
        "synthetic_fixture": False,
        "track_id": execution["track_id"],
        "weight_decay": 0.0001,
    }

    assert module().validate_arm_run_config(
        config, expected, execution, items=9922,
    ) == config
    with pytest.raises(ValueError, match="runtime parameterization"):
        module().validate_arm_run_config(
            {
                **config,
                "runtime_parameterization": {
                    **config["runtime_parameterization"],
                    "levels": 4,
                },
            },
            expected,
            execution,
            items=9922,
        )
    with pytest.raises(ValueError, match="provenance"):
        module().validate_arm_run_config(
            {**config, "cf_sha256": "0" * 64},
            expected,
            execution,
            items=9922,
        )


@pytest.mark.parametrize(
    ("levels", "size"),
    [(2, 64), (4, 256), (6, 128), (8, 512)],
)
def test_variable_identifier_bank_accepts_requested_shapes(levels, size):
    indices, codes = identifier_bank(items=6, levels=levels, size=size)

    receipt = module().validate_identifier_bank(
        indices, codes, items=6, codebook_sizes=[size] * levels,
    )

    assert receipt["levels"] == levels
    assert receipt["codebook_sizes"] == [size] * levels
    assert receipt["target_tokens"] == levels + 1
    assert receipt["prefixes"] == list("abcdefgh"[:levels])


def test_variable_identifier_bank_rejects_incomplete_catalog():
    indices, codes = identifier_bank(items=6, levels=4, size=256)
    del indices["5"]

    with pytest.raises(ValueError, match="complete contiguous catalog"):
        module().validate_identifier_bank(indices, codes, 6, [256] * 4)


def test_variable_identifier_bank_rejects_wrong_shape_range_and_prefix():
    indices, codes = identifier_bank(items=6, levels=2, size=64)
    with pytest.raises(ValueError, match="shape"):
        module().validate_identifier_bank(indices, codes[:, :1], 6, [64, 64])

    bad_range = codes.copy()
    bad_range[0, 1] = 64
    with pytest.raises(ValueError, match="level 1 range"):
        module().validate_identifier_bank(indices, bad_range, 6, [64, 64])

    bad_prefix = dict(indices)
    bad_prefix["0"] = ["<a_0>", "<c_0>"]
    with pytest.raises(ValueError, match="prefix"):
        module().validate_identifier_bank(bad_prefix, codes, 6, [64, 64])


def test_variable_audit_requires_atomic_l_plus_one_targets_and_input_cap():
    bank = {"levels": 8, "target_tokens": 9}
    report = audit_report(target_tokens=9, input_max=161, atomic=True)

    assert module().validate_variable_audit(report, bank, expected_records=3) == report

    report["splits"]["test"]["target_tokens_min_max"] = [8, 9]
    with pytest.raises(ValueError, match="L plus EOS"):
        module().validate_variable_audit(report, bank, expected_records=3)

    report = audit_report(target_tokens=9, input_max=513, atomic=True)
    with pytest.raises(ValueError, match="512"):
        module().validate_variable_audit(report, bank, expected_records=3)

    report = audit_report(target_tokens=9, input_max=161, atomic=False)
    with pytest.raises(ValueError, match="atomic"):
        module().validate_variable_audit(report, bank, expected_records=3)


def test_generation_contract_requires_budget_unique_valid_beams_and_coverage():
    indices, codes = identifier_bank(items=24, levels=8, size=64)
    bank = module().validate_identifier_bank(indices, codes, 24, [64] * 8)
    catalog = {"".join(value): [int(item)] for item, value in indices.items()}
    expected_users = ["0", "1"]
    rows = [
        {"user": user, "codes": list(catalog)[:20]}
        for user in expected_users
    ]

    receipt = module().validate_evaluation_rows(
        rows, expected_users, catalog, beams=20, max_new_tokens=10, bank=bank,
    )

    assert receipt["users"] == 2
    assert receipt["beams"] == 20
    assert receipt["item_predictions"]["0"] == list(range(10))

    with pytest.raises(ValueError, match="generation budget"):
        module().validate_evaluation_rows(
            rows, expected_users, catalog, beams=20, max_new_tokens=8, bank=bank,
        )
    duplicate = [dict(rows[0]), dict(rows[1])]
    duplicate[0]["codes"] = [rows[0]["codes"][0]] * 20
    with pytest.raises(ValueError, match="unique"):
        module().validate_evaluation_rows(
            duplicate, expected_users, catalog, 20, 10, bank,
        )
    invalid = [dict(rows[0]), dict(rows[1])]
    invalid[0]["codes"] = [*rows[0]["codes"][:19], "<invalid>"]
    with pytest.raises(ValueError, match="catalog"):
        module().validate_evaluation_rows(invalid, expected_users, catalog, 20, 10, bank)
    with pytest.raises(ValueError, match="coverage"):
        module().validate_evaluation_rows(rows[:1], expected_users, catalog, 20, 10, bank)


def test_queue_binding_is_immutable_and_separate(tmp_path):
    binding = {
        "kind": "figure8_tokenizer_queue_v1",
        "work": str(tmp_path / "fresh-t5-figure8-tokenizer-v1"),
        "durable_root": str(tmp_path / "durable" / "figure8-tokenizer-v1"),
        "queue": module().tokenizer_cells(),
    }

    marker = module().bind_tokenizer_queue(tmp_path, binding)

    assert json.loads(marker.read_text()) == binding
    assert module().bind_tokenizer_queue(tmp_path, binding) == marker
    with pytest.raises(ValueError, match="frozen"):
        module().bind_tokenizer_queue(tmp_path, {**binding, "queue": []})


def test_queue_binding_does_not_freeze_incremental_arm_admissions():
    cells = module().tokenizer_cells()
    first = module().queue_binding(
        plan_sha256="a" * 64,
        work=Path("/scratch/fresh-t5-figure8-tokenizer-v1"),
        durable_root=Path("/durable/fresh-t5-figure8-tokenizer-v1"),
        gpu_indexes=(0, 1),
        source_sha256={"queue": "b" * 64},
    )
    later = module().queue_binding(
        plan_sha256="a" * 64,
        work=Path("/scratch/fresh-t5-figure8-tokenizer-v1"),
        durable_root=Path("/durable/fresh-t5-figure8-tokenizer-v1"),
        gpu_indexes=(0, 1),
        source_sha256={"queue": "b" * 64},
    )

    assert first == later
    assert first["queue"] == cells
    assert "admissions" not in first


def test_arm_admission_requires_exact_spec_bundle_and_durable_receipt(tmp_path):
    arm = "levels-2"
    spec = {
        "dataset": "Instruments", "codes": [256, 256],
        "epsilons": [0.0, 0.003], "dimension": 32,
        "layers": [2048, 1024, 512, 256, 128, 64], "clusters": 10,
        "alpha": 0.02, "beta": 0.0001, "seed": 42, "epochs": 20000,
        "batch_size": 1024, "eval_step": 2000,
    }
    spec_path = tmp_path / f"{arm}.json"
    launch_spec = {
        **spec,
        "cf": "/scratch/letter_fixture/runs/Instruments-cf-bce-s42/item-embeddings.pt",
        "checkpoint_every": 500,
        "control": "/scratch/letter_fixture/figure8-sweeps-v1/control",
        "embedding_receipt":
            "/scratch/letter_fixture/embeddings-openllama-v2-v1/Instruments/embedding-receipt.json",
        "gpu": 1,
        "host": "node-1",
        "kind": "figure8-one-factor-tokenizer-v1",
        "manifest":
            "/scratch/letter_fixture/figure8-sweeps-v1/control/configs/letter-reproduction-v1.json",
        "method_spec":
            "/scratch/letter_fixture/figure8-sweeps-v1/control/configs/letter-openllama-v2-method-v1.json",
        "output": "/scratch/letter_fixture/figure8-sweeps-v1/runs/levels-2",
        "root": "/scratch/letter_fixture",
        "semantic":
            "/scratch/letter_fixture/embeddings-openllama-v2-v1/Instruments/embeddings.npy",
    }
    spec_path.write_text(json.dumps(launch_spec))
    bundle = tmp_path / "durable" / "levels-2-completed-deadbeef"
    payload = bundle / "payload"
    (payload / "export").mkdir(parents=True)
    (bundle / "completed.json").write_text('{"status":"complete"}')
    (payload / "completed.json").write_text('{"status":"complete"}')
    indices, codes = identifier_bank(6, 2, 256)
    (payload / "run-config.json").write_text(
        json.dumps(normalized_bundle_config(spec, len(indices)))
    )
    (payload / "export" / "index.json").write_text(json.dumps(indices))
    np.save(payload / "export" / "codes.npy", codes, allow_pickle=False)
    write_bank_receipt(bundle)
    audit = {
        "status": "pass", "arm": arm, "bundle": bundle.name,
        "selected_epoch": 12000, "final_epoch": 20000,
        "bundle_completed_sha256": sha(bundle / "completed.json"),
        "run_completed_sha256": sha(payload / "completed.json"),
        "index_sha256": sha(payload / "export" / "index.json"),
        "codes_sha256": sha(payload / "export" / "codes.npy"),
        "durable_bank_receipt_sha256": sha(bundle / "bank-receipt.json"),
    }

    admitted = module().verify_arm_admission(
        arm, audit, spec_path, tmp_path / "durable",
        expected_spec=spec, execution=execution(),
    )

    assert admitted["bundle"] == bundle.name
    assert admitted["bank"]["levels"] == 2
    assert admitted["spec_sha256"] == sha(spec_path)
    audit["bundle"] = "levels-6-completed-cross-arm"
    with pytest.raises(ValueError, match="arm bundle"):
        module().verify_arm_admission(
            arm, audit, spec_path, tmp_path / "durable",
            expected_spec=spec, execution=execution(),
        )
    audit["bundle"] = bundle.name
    (bundle / "bank-receipt.json").unlink()
    with pytest.raises(ValueError, match="durable"):
        module().verify_arm_admission(
            arm, audit, spec_path, tmp_path / "durable",
            expected_spec=spec, execution=execution(),
        )


@pytest.mark.parametrize(
    "work_basename",
    ("fresh-t5-figure8-tokenizer-v1", "fresh-t5-figure8-tokenizer-v2"),
)
def test_arm_admission_path_must_be_exact_durable_namespace(tmp_path, work_basename):
    destination = tmp_path / "destination"
    durable = destination / "openllama-v2" / work_basename
    exact = durable / "operational" / "admissions" / "levels-2" / "admission.json"
    exact.parent.mkdir(parents=True)
    exact.write_text("{}")

    assert module().require_durable_admission_path(
        exact, destination, "levels-2", work_basename,
    ) == exact.resolve()
    local = tmp_path / work_basename / "admissions" / "levels-2"
    local.mkdir(parents=True)
    (local / "admission.json").write_text("{}")
    with pytest.raises(ValueError, match="durable admission namespace"):
        module().require_durable_admission_path(
            local / "admission.json", destination, "levels-2", work_basename,
        )


def test_derived_manifest_binds_arm_bank_and_rejects_cross_arm_lineage():
    indices, codes = identifier_bank(6, 6, 128)
    parent = {"datasets": {"Instruments": {
        "items": 6, "users": 3, "interactions": 18,
        "training_prefix_examples": 3, "validation_users": 3, "test_users": 3,
        "sha256": {"item": "a", "inter": "b", "index": "old"},
    }}}
    lineage = {
        "arm": "levels-6", "index_sha256": "c" * 64, "codes_sha256": "d" * 64,
        "bundle_completed_sha256": "e" * 64,
    }

    derived = module().derive_arm_manifest(
        parent, "f" * 64, lineage, indices, codes, [128] * 6,
    )

    assert derived["experiment_lineage"] == lineage
    assert derived["datasets"]["Instruments"]["sha256"]["index"] == "c" * 64
    assert derived["datasets"]["Instruments"]["figure8_identifier_bank"]["levels"] == 6
    with pytest.raises(ValueError, match="cross-arm"):
        module().validate_stage_lineage(
            derived, {**lineage, "arm": "codes-128"},
        )


def test_resume_identity_rejects_cross_arm_or_warm_start():
    cell = module().tokenizer_cells()[0]
    run = {
        "arm": cell["arm"], "run_id": cell["run_id"], "stage_id": cell["stage_id"],
        "bundle_completed_sha256": "a" * 64, "seed": 42, "temperature": 0.7,
        "initialization": "author random T5 configuration",
    }

    assert module().validate_resume_identity(run, cell, "a" * 64) == run
    with pytest.raises(ValueError, match="cross-arm"):
        module().validate_resume_identity({**run, "arm": "beta-0p01"}, cell, "a" * 64)
    with pytest.raises(ValueError, match="warm"):
        module().validate_resume_identity(
            {**run, "initialization": "checkpoint from another arm"}, cell, "a" * 64,
        )


def test_completed_evaluation_selection_rejects_duplicates(tmp_path):
    root = tmp_path / "evaluations"
    for name in ("evaluation-attempt-001", "evaluation-attempt-002"):
        path = root / name
        path.mkdir(parents=True)
        (path / "completed.json").write_text("{}")

    with pytest.raises(ValueError, match="Ambiguous completed evaluations"):
        module().completed_evaluation(root)


def test_execute_cell_banks_run_before_test_and_banks_result():
    cell = module().tokenizer_cells()[0]

    class Runtime:
        def __init__(self):
            self.events = []

        def admit(self, value, admission):
            self.events.append(("admit", value["arm"], admission["bundle"]))
            return "stage"

        def gate(self, value, stage):
            self.events.append(("gate", value["arm"], stage))

        def train(self, value, stage):
            self.events.append(("train", value["arm"], stage))
            return "run"

        def bank_run(self, value, run):
            self.events.append(("bank_run", value["arm"], run))

        def evaluate(self, value, stage, run):
            assert self.events[-1][0] == "bank_run"
            self.events.append(("evaluate", value["arm"], run))
            return "evaluation"

        def bank_evaluation(self, value, evaluation):
            self.events.append(("bank_evaluation", value["arm"], evaluation))

        def publish(self, value, stage, run, evaluation):
            self.events.append(("publish", value["arm"], evaluation))
            return {"status": "banked"}

    runtime = Runtime()
    result = module().execute_cell(runtime, cell, {"bundle": "alpha-0p001-completed-a"})

    assert result == {"status": "banked"}
    assert [event[0] for event in runtime.events] == [
        "admit", "gate", "train", "bank_run", "evaluate",
        "bank_evaluation", "publish",
    ]


def test_queue_supervises_training_with_durable_checkpoint_banker():
    source = (
        ROOT / "scripts" / "letter" / "figure8_tokenizer_queue.py"
    ).read_text(encoding="utf-8")

    assert "checkpoint-watcher" in source
    assert "exporter=banker" in source
    assert "checkpoint_artifacts" in source


def test_next_admitted_cell_uses_frozen_order_without_blocking_on_unready_arm():
    cells = module().tokenizer_cells()
    admissions = {
        cells[2]["arm"]: {"bundle": "alpha-0p1-completed-a"},
        cells[5]["arm"]: {"bundle": "beta-0p01-completed-b"},
    }

    assert module().next_admitted_cell(cells, admissions, set()) == cells[2]
    assert module().next_admitted_cell(
        cells, admissions, {cells[2]["run_id"]},
    ) == cells[5]
    assert module().next_admitted_cell(
        cells, admissions, {cells[2]["run_id"], cells[5]["run_id"]},
    ) is None


def test_panel_aggregation_reuses_baseline_and_has_no_winner():
    points = {
        arm: {"metrics": {"code": {"recall@10": index / 100}}}
        for index, arm in enumerate([cell["arm"] for cell in module().tokenizer_cells()], 1)
    }
    baseline = {"run_id": "baseline-t07", "metrics": {"code": {"recall@10": 0.1}}}

    result = module().aggregate_panels(points, baseline)

    assert set(result["panels"]) == {"alpha", "beta", "clusters", "levels", "codes"}
    assert all(panel["baseline"] == baseline for panel in result["panels"].values())
    assert not module().contains_winner_field(result)
    assert result["selection"] == "descriptive only; no test-based winner"


def test_baseline_comparison_requires_exact_frozen_result(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"status":"verified_single_fresh_id_cell"}')
    digest = sha(result)

    assert compare_module().verify_baseline_result(result, digest)["status"].startswith(
        "verified"
    )
    with pytest.raises(ValueError, match="baseline result"):
        compare_module().verify_baseline_result(result, "0" * 64)


def identifier_bank(items: int, levels: int, size: int):
    codes = np.array([
        [(item * (level + 1)) % size for level in range(levels)]
        for item in range(items)
    ], dtype=np.int64)
    prefixes = "abcdefgh"
    indices = {
        str(item): [f"<{prefixes[level]}_{value}>" for level, value in enumerate(row)]
        for item, row in enumerate(codes.tolist())
    }
    return indices, codes


def audit_report(target_tokens: int, input_max: int, atomic: bool):
    split = {
        "records": 3, "strict_clean": 3, "over_cap": 0,
        "input_tokens_p50_p90_p99_max": [1, 2, 3, input_max],
        "target_tokens_min_max": [target_tokens, target_tokens],
        "atomic_identifier_tokens": atomic,
    }
    return {"status": "pass", "splits": {
        "train": dict(split), "valid": dict(split), "test": dict(split),
    }}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bank_receipt(bundle: Path):
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "bank-receipt.json":
            files[path.relative_to(bundle).as_posix()] = {
                "sha256": sha(path), "bytes": path.stat().st_size,
            }
    (bundle / "bank-receipt.json").write_text(json.dumps({"files": files}))
