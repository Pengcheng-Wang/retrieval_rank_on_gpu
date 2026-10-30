import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def rq():
    assert (SCRIPTS / "tokenizer_run.py").exists(), "Tokenizer lifecycle adapter is missing"
    return importlib.import_module("tokenizer_run")


def test_tokenizer_spec_rejects_silent_zip_truncation():
    rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 9922)
    with pytest.raises(ValueError, match="levels"):
        rq().validate_spec([256] * 8, [0, 0, 0, 0.003], 10, 32, 9922)
    with pytest.raises(ValueError, match="initialization"):
        rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 100)
    with pytest.raises(ValueError, match="initialization"):
        rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 60000)


def test_default_and_nondefault_cluster_bounds_are_feasible():
    assert rq().cluster_bounds(256, 10) == (10, 60)
    low, high = rq().cluster_bounds(512, 5)
    assert 5 * low <= 512 <= 5 * high
    with pytest.raises(ValueError):
        rq().cluster_bounds(8, 10)


def test_eight_level_export_and_collision_accounting():
    codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
                      [1, 1, 2, 3, 4, 5, 6, 7]])
    result = rq().format_codes(codes)
    assert result["0"][-1] == "<h_7>"
    assert rq().collision_groups(codes) == [[0, 1]]
    with pytest.raises(ValueError):
        rq().format_codes(np.array([[-1, 2]]))


def test_figure8_gate_is_bounded_and_never_final_tokenizer_identity():
    assert rq().experiment_purpose(False, 20000) == {
        "kind": "tokenizer_reconstruction",
        "admissible_final_tokenizer": True,
    }
    assert rq().experiment_purpose(True, 2) == {
        "kind": "figure8_parameterization_cost_gate",
        "admissible_final_tokenizer": False,
    }
    with pytest.raises(ValueError, match="bounded"):
        rq().experiment_purpose(True, 11)


def test_runtime_receipt_checks_actual_nondefault_codebooks_and_sid_length():
    def layer(codes, epsilon):
        return SimpleNamespace(
            n_e=codes,
            e_dim=32,
            sk_epsilon=epsilon,
            embedding=SimpleNamespace(weight=SimpleNamespace(shape=(codes, 32))),
        )

    codes = [512] * 8
    epsilons = [0] * 7 + [0.003]
    model = SimpleNamespace(
        n_clusters=20,
        e_dim=32,
        num_emb_list=codes,
        sk_epsilons=epsilons,
        rq=SimpleNamespace(vq_layers=[layer(count, epsilon)
                                     for count, epsilon in zip(codes, epsilons)]),
    )
    receipt = rq().runtime_model_receipt(model, codes, epsilons, 20, 32)
    assert receipt == {
        "levels": 8,
        "codebook_shapes": [[512, 32]] * 8,
        "clusters": 20,
        "epsilons": epsilons,
        "emitted_sid_length": 8,
        "last_token_prefix": "h",
    }
    model.rq.vq_layers[-1].embedding.weight.shape = (256, 32)
    with pytest.raises(ValueError, match="codebook"):
        rq().runtime_model_receipt(model, codes, epsilons, 20, 32)


def test_input_embeddings_are_aligned_finite_matrices():
    semantic = np.ones((512, 16), dtype=np.float32)
    collaborative = np.ones((512, 32), dtype=np.float32)
    rq().validate_embeddings(semantic, collaborative, 32)
    with pytest.raises(ValueError, match="aligned"):
        rq().validate_embeddings(semantic, collaborative[1:], 32)
    semantic[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rq().validate_embeddings(semantic, collaborative, 32)


def test_interrupted_output_is_quarantined_without_deleting_it(tmp_path):
    directory = tmp_path / "epoch-00100"
    directory.mkdir()
    (directory / "state.pt.tmp").write_bytes(b"interrupted")
    displaced = rq().fresh_output_directory(directory, "completed.json")
    assert directory.is_dir() and not list(directory.iterdir())
    assert displaced.parent == tmp_path
    assert (displaced / "state.pt.tmp").read_bytes() == b"interrupted"
    (directory / "completed.json").write_text('{"status":"complete"}')
    with pytest.raises(FileExistsError):
        rq().fresh_output_directory(directory, "completed.json")


def test_compatible_encoder_requires_its_explicit_track():
    spec = ROOT / "configs" / "letter-openllama-v2-method-v1.json"
    config = json.loads(spec.read_text())
    receipt = {
        "encoder": "OpenLLaMA-7B-v2", "track_id": config["track_id"],
        "model_id": config["model"]["id"], "revision": config["model"]["revision"],
        "exact_original_llama1_reproduction": False, "smoke": False,
        "method_spec_sha256": rq().file_sha256(spec),
    }
    with pytest.raises(ValueError):
        rq().encoder_track(receipt, None)
    assert rq().encoder_track(receipt, spec) == config["track_id"]
    with pytest.raises(ValueError):
        rq().encoder_track({**receipt, "smoke": True}, spec)
    with pytest.raises(ValueError):
        rq().encoder_track({**receipt, "revision": "different"}, spec)
    with pytest.raises(ValueError):
        rq().encoder_track({**receipt, "method_spec_sha256": "different"}, spec)


def test_original_encoder_gate_stays_distinct_from_variant():
    spec = ROOT / "configs" / "letter-openllama-v2-method-v1.json"
    receipt = {"encoder": "LLaMA1-7B"}
    assert rq().encoder_track(receipt, None) == "letter-original-llama1"
    with pytest.raises(ValueError):
        rq().encoder_track(receipt, spec)
    with pytest.raises(ValueError):
        rq().encoder_track({**receipt, "exact_original_llama1_reproduction": False}, None)
