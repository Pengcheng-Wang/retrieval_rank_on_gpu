"""Explicit opt-in real checkpoint generation probe; never touches held-out test users."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


@pytest.mark.skipif(not os.environ.get("LETTER_LC_GATE_RUN"), reason="Completed real LC gate required")
def test_real_lc_validation_generation_and_raw_scores():
    import torch
    from lc_checkpoint import snapshot_identity
    from lc_evaluate import INFERENCE_PRECISION, POLICIES, load_evaluation_model, make_constraint
    from lc_run import load_inputs, setup_runtime
    from protocol import file_sha256

    run = Path(os.environ["LETTER_LC_GATE_RUN"]).resolve()
    run.resolve().relative_to((ROOT / ".local" / "letter").resolve())
    checkpoint = run / "checkpoint-8"
    task = ROOT.parent
    args = SimpleNamespace(
        upstream=task / "upstream", manifest=ROOT / "configs" / "letter-reproduction-v1.json",
        method_spec=ROOT / "configs" / "letter-openllama-v2-method-v1.json",
        execution_spec=ROOT / "configs" / "letter-lc-compatible-execution-v1.json",
        model=task / "models" / "openllama-v2-e5961def" / "converted",
        model_receipt=task / "models" / "openllama-v2-e5961def" / "completed.json",
        audit=ROOT / ".local" / "letter" / "lc-audit" / "full-001-Instruments",
        dataset="Instruments",
    )
    rank, world, runtime = setup_runtime(require_world=False)
    assert world == 1 and rank == 0
    inputs = load_inputs(args)
    model = load_evaluation_model(inputs, rank, checkpoint)
    model.config.use_cache = True
    tokenizer = inputs["test_tokenizer"]
    collator = inputs["author"].collator.TestCollator(inputs["native"], tokenizer)
    batch, _ = collator([inputs["valid"][0]])
    batch = {key: value.to(rank) for key, value in batch.items()}
    width = batch["input_ids"].shape[1]
    codes = [[tokenizer.convert_tokens_to_ids(token) for token in code]
             for code in inputs["indices"].values()]
    catalog = {"".join(code) for code in inputs["indices"].values()}
    results = []
    with torch.inference_mode():
        for policy in POLICIES:
            output = model.generate(
                **batch, num_beams=20, num_return_sequences=20, max_new_tokens=5,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                prefix_allowed_tokens_fn=make_constraint(codes, 2, width, policy),
                eos_token_id=2, pad_token_id=0, length_penalty=1.0, early_stopping=True,
            )
            generated = output.sequences[:, width:].cpu().tolist()
            assert len(generated) == 20
            assert all(len(seq) == 5 and seq[-1] == 2 and 2 not in seq[:-1] for seq in generated)
            assert torch.isfinite(output.sequences_scores).all()
            predictions = [
                text.strip().replace(" ", "") for text in tokenizer.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
            ]
            invalid = sum(code not in catalog for code in predictions)
            if policy == "full-prefix-trie-correction":
                assert invalid == 0
            results.append({"policy": policy, "beams": 20, "output_tokens": 5,
                            "all_stopped_on_eos": True, "all_scores_finite": True,
                            "invalid_codes": invalid, "unique_codes": len(set(predictions))})
    report = {
        "status": "pass", "scope": "one fixed validation prompt; no test quality measurement",
        "checkpoint": snapshot_identity(checkpoint),
        "inputs": inputs["identity"], "policies": results, "runtime": runtime,
        "probe_source_sha256": file_sha256(Path(__file__)),
        "inference_precision": INFERENCE_PRECISION,
    }
    with (run / "generation-probe.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
