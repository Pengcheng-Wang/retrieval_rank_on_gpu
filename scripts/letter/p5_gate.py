"""Native ~128-row GPU gate, complete beams, worst-length inputs, and cold resume."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LogitsProcessor, LogitsProcessorList

from p5_checkpoint import pointer_path, verify_checkpoint
from p5_run import evaluate, generation, native_loss, parser, setup, update
from p5_source import digest_json, read_json, sha256, write_json


def assert_equal(a, b, path="root"):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise AssertionError(f"Cold continuation tensor changed: {path}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Cold continuation keys changed: {path}")
        for key in a:
            assert_equal(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Cold continuation length changed: {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, path + "." + str(index))
    elif a != b:
        raise AssertionError(f"Cold continuation value changed: {path}: {a} != {b}")


def child(args, output, steps, resume=None):
    command = [sys.executable, str(Path(__file__).with_name("p5_run.py"))]
    for name in ("native", "upstream", "manifest", "recipe", "model", "corpus"):
        command.extend(["--" + name, str(getattr(args, name))])
    command.extend([
        "--output", str(output), "--gate-subset", "128", "--stop-steps", str(steps),
        "--microbatch-size", str(args.microbatch_size),
    ])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    output.mkdir(parents=True, exist_ok=True)
    log = output / "child.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ))
    if result.returncode:
        text = log.read_text(encoding="utf-8")
        if "CUDA out of memory" in text:
            raise torch.cuda.OutOfMemoryError(f"Native training microbatch failed: {log}")
        raise RuntimeError(f"Native gate child failed ({result.returncode}): {log}\n{text[-5000:]}")
    return pointer_path(output)


def run_gate(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("continuous", "interrupted", "resumed"):
        if (args.output / name / "latest.json").exists():
            raise FileExistsError("Use a new gate attempt directory; do not relabel an earlier attempt")
    continuous = child(args, args.output / "continuous", 5)
    interrupted = child(args, args.output / "interrupted", 2)
    resumed = child(args, args.output / "resumed", 5, interrupted)
    a_seal, b_seal = verify_checkpoint(continuous), verify_checkpoint(resumed)
    if a_seal["files_sha256"]["model.safetensors"] != b_seal["files_sha256"]["model.safetensors"]:
        raise AssertionError("Native model bytes differ after cold continuation")
    a = torch.load(continuous / "training-state.pt", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "training-state.pt", map_location="cpu", weights_only=False)
    assert_equal(a, b)
    continuous_trace = [
        json.loads(line) for line in (args.output / "continuous" / "updates.jsonl").read_text().splitlines()
    ]
    resumed_trace = [
        json.loads(line)
        for line in (args.output / "resumed" / "updates.jsonl").read_text().splitlines()
    ]
    assert_equal(continuous_trace[2:], resumed_trace)
    del a, b
    args.gate_subset = 128
    context = setup(args)
    model = context.model
    model.load_state_dict(load_file(str(continuous / "model.safetensors")), strict=True)
    model.eval()
    tokenizer, collator, native, catalog = (
        context.tokenizer, context.collator, context.native, context.catalog,
    )
    probe_rows = [context.views["valid"][i] for i in range(2)]
    probe = tuple(t.to("cuda") for t in collator(probe_rows))
    with torch.inference_mode():
        encoded = model.encoder(
            input_ids=probe[0], attention_mask=probe[1], whole_word_ids=probe[2],
            whole_word_embedding_type="shijie", return_dict=True,
        )
        direct = model(
            input_ids=probe[0], attention_mask=probe[1], whole_word_ids=probe[2],
            whole_word_embedding_type="shijie", labels=probe[3], return_dict=True,
        ).logits
        precomputed = model(
            encoder_outputs=encoded, attention_mask=probe[1], labels=probe[3], return_dict=True,
        ).logits
        torch.testing.assert_close(direct, precomputed, rtol=0, atol=0)
        without_words = model.encoder(
            input_ids=probe[0], attention_mask=probe[1], whole_word_ids=torch.zeros_like(probe[2]),
            whole_word_embedding_type="shijie", return_dict=True,
        ).last_hidden_state
        word_effect = (encoded.last_hidden_state - without_words).abs().max().item()
        if word_effect <= 1e-5:
            raise AssertionError("Whole-word IDs are not active in the actual generation encoder")
        cached = generation(model, probe, tokenizer, catalog, native, context.recipe, cache=True)
        uncached = generation(model, probe, tokenizer, catalog, native, context.recipe, cache=False)
        if not torch.equal(cached.sequences, uncached.sequences):
            raise AssertionError("Native multi-example 20-beam cache/reordering parity failed")
        torch.testing.assert_close(cached.sequences_scores, uncached.sequences_scores, rtol=1e-5, atol=1e-5)
        single = tuple(t.to("cuda") for t in collator(probe_rows[:1]))
        short_encoded = model.encoder(
            input_ids=single[0], attention_mask=single[1], whole_word_ids=single[2],
            whole_word_embedding_type="shijie", return_dict=True,
        ).last_hidden_state
        torch.testing.assert_close(
            short_encoded[0], encoded.last_hidden_state[0, :single[0].shape[1]], rtol=2e-5, atol=2e-5,
        )
        longest = max(catalog["token_ids"], key=len)

        class ForceLongestCatalogPath(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for i, prefix in enumerate(input_ids.tolist()):
                    prefix = prefix[1:]
                    if len(prefix) < len(longest) and prefix == longest[:len(prefix)]:
                        token = longest[len(prefix)]
                        if not torch.isfinite(scores[i, token]):
                            raise AssertionError("Longest full title is not reachable in the full-catalog trie")
                        scores[i, token] += 100.
                return scores

        forced = generation(
            model, single, tokenizer, catalog, native, context.recipe,
            logits_processor=LogitsProcessorList([ForceLongestCatalogPath()]),
        )
        completed = [row[1:row.index(1, 1) + 1] for row in forced.sequences.tolist()]
        if longest not in completed:
            raise AssertionError("Native 20-beam generation cannot complete the longest catalog target")
    del direct, precomputed, encoded, without_words, short_encoded, cached, uncached, forced, probe, single
    # This is deliberately an actual longest-record training batch, not a short sampled smoke.
    longest_rows = read_json(args.corpus / "longest-records.json")
    rows = [(r["input"], r["target_text"]) for r in longest_rows]
    rows = (rows * math_ceil_div(128, len(rows)))[:128]
    torch.cuda.reset_peak_memory_stats()
    worst = update(
        model, context.optimizer, context.scheduler, rows, list(range(128)), collator,
        args.microbatch_size, context.recipe["training"]["clip_norm"],
    )
    peak = torch.cuda.max_memory_allocated()
    metrics = evaluate(
        model, context.views["valid"], context.users, context.interactions, "valid", collator,
        tokenizer, catalog, native, context.recipe, args.output / "gate-validation-predictions.jsonl",
        limit=100,
    )
    report = {
        "status": "pass", "kind": "native-P5-GPU-admission-not-a-result",
        "dataset": context.audit["dataset"], "representation": context.audit["representation"],
        "recipe_sha256": sha256(args.recipe), "admission_sha256": sha256(args.corpus / "admission.json"),
        "driver_sha256": context.contract["driver_sha256"],
        "operating_microbatch_size": args.microbatch_size,
        "operating_accumulation": 128 // args.microbatch_size,
        "source_nominal_microbatch": context.recipe["training"]["microbatch_size"],
        "microbatch_adaptation": "memory-only gate; effective128, loss weighting and all tokens retained",
        "training_support_rows": 128, "continuous_updates": 5, "cold_restart_after_updates": 2,
        "model_bytes_equal_after_cold_restart": True,
        "optimizer_scheduler_rng_sampler_epoch_state_equal": True,
        "subsequent_sampled_rows_losses_and_updates_equal": True,
        "native_encoder_precompute_logits_exact": True, "whole_word_effect_max_abs": word_effect,
        "pretrained_load_diagnostics": model.pretrained_load_diagnostics,
        "batch_padding_invariance": True, "cached_uncached_multiuser20beam_parity": True,
        "longest_catalog_target_tokens_including_eos": len(longest),
        "longest_catalog_target_completed_with_full20beam_trie": True,
        "longest_path_probe_uses_gate_only_score_boost_not_an_evaluation_policy": True,
        "worst_length_microbatch_peak_bytes": peak, "worst_length_update": worst,
        "validation_users_generated": metrics["users"], "beams_per_user": metrics["beams_per_user"],
        "complete_catalog_sequences_only": True,
        "native_empty_prefix_masking_calls": catalog.get("_empty_beam_prefix_count", 0),
        "first_empty_prefix_diagnostic": catalog.get("_first_empty_beam_prefix"),
        "validation_prediction_sha256": metrics["predictions_sha256"],
        "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - started,
        "test_labels_scored": False,
    }
    write_json(args.output / "gate.json", report)
    print(f"P5_GPU_GATE_PASS {report['dataset']} {report['representation']} "
          f"micro={args.microbatch_size} peak={peak}", flush=True)
    return report


def math_ceil_div(a, b):
    return (a + b - 1) // b


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        run_gate(arguments)
    except torch.cuda.OutOfMemoryError as error:
        write_json(arguments.output / "gate.json", {
            "status": "memory-gate-failed", "operating_microbatch_size": arguments.microbatch_size,
            "error": str(error), "not_a_training_recipe_or_score_search": True,
        })
        raise SystemExit(13)
