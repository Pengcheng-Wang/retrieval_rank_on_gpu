"""Explicit private-source/CUDA integration gates; never a CPU fit fallback."""

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def test_real_frozen_base_lora_ce_reload_and_title_grounding():
    admission = os.environ.get("BIGREC_ADMISSION")
    if admission is None:
        pytest.skip("Requires a hash-bound full-corpus BIGRec admission and explicitly allocated GPU")
    from bigrec_gate import model_gate

    result = model_gate(
        Path(admission), Path(os.environ["BIGREC_MODEL"]),
        Path(os.environ["BIGREC_GATE_OUTPUT"]),
    )
    assert result["status"] == "pass"
    assert result["vanilla_ce"] and result["qv_only_gradients"]
    assert result["frozen_base"] and result["adapter_reload"]
    assert result["last_token_pooling"] and result["four_beam_generation"]


def test_real_cold_resume_complete_state_and_ordered_inputs():
    root = os.environ.get("BIGREC_RESUME_GATE")
    if root is None:
        pytest.skip("Run the production-shaped continuous/interrupted processes first")
    from bigrec_gate import compare_continuations

    result = compare_continuations(Path(root))
    assert result["status"] == "pass"
    assert result["actual_optimizer_updates"] >= 5
    assert result["adapter_bitwise"] and result["optimizer_bitwise"]
    assert result["scheduler_exact"] and result["scaler_exact"]
    assert result["rng_exact"] and result["sampler_exact"]
    assert result["ordered_batches_exact"] and result["validation_state_exact"]
