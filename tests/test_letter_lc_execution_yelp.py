import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def yelp():
    assert importlib.util.find_spec("lc_yelp"), "Versioned Yelp control is missing"
    return importlib.import_module("lc_yelp")


def test_yelp_geometry_records_native_and_amended_padding():
    lib = yelp()
    native = lib.geometry(225061, 32, 4, 2)
    amended = lib.geometry(225061, 64, 4, 1)
    assert native["microbatches_per_epoch"] == 1759
    assert native["distributed_padding_examples"] == 91
    assert native["integral_accumulation_epoch"] is False
    assert native["updates_per_epoch"] is None
    assert amended["microbatches_per_epoch"] == 880
    assert amended["distributed_padding_examples"] == 219
    assert amended["updates_per_epoch"] == 880
    assert amended["effective_batch"] == native["effective_batch"] == 256


def test_yelp_recipe_only_amends_train_geometry():
    lib = yelp()
    manifest = json.loads((ROOT / "configs" / "letter-reproduction-v1.json").read_text())
    recipe = lib.yelp_recipe(manifest, "Yelp")
    assert (recipe["micro_batch"], recipe["accumulation"], recipe["world_size"]) == (64, 1, 4)
    assert recipe["effective_batch"] == 256
    assert (recipe["epochs"], recipe["learning_rate"], recipe["temperature"]) == (6, 3e-4, 1.0)
    assert (recipe["lora_r"], recipe["lora_alpha"], recipe["lora_dropout"]) == (16, 32, 0.05)
    assert recipe["tasks"] == ["seqrec"] and recipe["context"] == 2048
    with pytest.raises(ValueError, match="Beauty"):
        lib.yelp_recipe(manifest, "Beauty")
    with pytest.raises(ValueError):
        lib.yelp_recipe(manifest, "Instruments")


def test_yelp_private_fit_adaptation_is_narrow_and_source_guarded():
    lib = yelp()
    source = (ROOT / "scripts" / "letter" / "lc_run.py").read_text()
    import ast
    fit = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    original = "\n".join(source.splitlines()[fit.lineno - 1:fit.end_lineno]) + "\n"
    patched = lib.adapt_fit_source(original)
    assert "Subset(train, range(512)), Subset(valid, range(512))" in patched
    assert "range(64)" in patched
    assert "write_reload_reference(" in patched
    assert original.count("range(100)") == 2
    with pytest.raises(ValueError):
        lib.adapt_fit_source(original.replace("range(100)", "range(99)"))


def test_preparation_never_counts_as_real_gpu_admission():
    lib = yelp()
    with pytest.raises(ValueError, match="GPU"):
        lib.require_gpu_admission({"status": "pass", "kind": "cpu-preparation"}, {}, {})
