"""Frozen-checkpoint LC decoding: native position sets and separately named trie correction."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from protocol import (
    PrefixTrie, code_to_items, expand_item_predictions, file_sha256, rank_metrics,
    sequence_records, shard_positions, verify_coverage, write_json,
)


POLICIES = ("author-position-sets", "full-prefix-trie-correction")
INFERENCE_PRECISION = (
    "native BF16 base and saved embedding/head, FP32 LoRA under PEFT 0.4; unmerged, not int8"
)


def make_constraint(codes, eos, input_width, policy):
    if policy not in POLICIES or input_width < 1 or not codes or any(len(c) != 4 for c in codes):
        raise ValueError("Invalid LC decoder policy or four-token catalog")
    trie = PrefixTrie([list(code) + [eos] for code in codes])
    positions = [sorted({code[i] for code in codes}) for i in range(4)] + [[eos]]

    def allowed(batch_id, sentence):
        sequence = sentence.tolist() if hasattr(sentence, "tolist") else list(sentence)
        if len(sequence) < input_width:
            raise ValueError("Generation sequence is shorter than its input width")
        generated = sequence[input_width:]
        if eos in generated:
            return [eos]
        if len(generated) > 4:
            raise ValueError("Generation exceeded four SID tokens before EOS")
        choices = positions[len(generated)] if policy == POLICIES[0] else trie.allowed(generated)
        if not choices:
            raise ValueError("Generated prefix is outside the declared catalog trie")
        return choices

    return allowed


def score_beams(codes, scores, target, indices):
    if len(codes) != len(scores) or not codes or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Every raw beam requires a finite score")
    ordered = sorted(zip(codes, scores), key=lambda pair: pair[1], reverse=True)
    codes, scores = [p[0] for p in ordered], [float(p[1]) for p in ordered]
    catalog = code_to_items(indices)
    target_code = "".join(indices[str(target)])
    # The released LC evaluator does NOT deduplicate beams and sums NDCG contributions.
    native = {}
    for k in (1, 5, 10):
        hits = [index + 1 for index, code in enumerate(codes[:k]) if code == target_code]
        native[f"recall@{k}"] = float(bool(hits))
        native[f"ndcg@{k}"] = sum(1 / math.log2(rank + 1) for rank in hits)
    items = expand_item_predictions(codes, indices, 20)
    return {
        "raw_codes": codes, "raw_scores": scores, "code": native,
        "item_predictions": items, "item": rank_metrics(items, target),
        "invalid_codes": sum(code not in catalog for code in codes),
        "duplicate_codes": len(codes) - len(set(codes)),
        "target_code": target_code, "target_collision_size": len(catalog[target_code]),
    }


def load_evaluation_model(inputs, rank, checkpoint):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import LlamaConfig
    from lc_checkpoint import assert_identical, verify

    receipt = verify(checkpoint)
    config = LlamaConfig.from_pretrained(inputs["model"], local_files_only=True)
    config.pretraining_tp = 1
    config._attn_implementation = "eager"
    base = inputs["author"].model.LETTER.from_pretrained(
        inputs["model"], config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": rank}, local_files_only=True,
    )
    base.set_hyper(inputs["recipe"]["temperature"])
    base.resize_token_embeddings(len(inputs["tokenizer"]))
    model = PeftModel.from_pretrained(
        base, checkpoint, is_trainable=False, torch_dtype=torch.bfloat16, device_map={"": rank},
    ).eval()
    path = checkpoint / receipt["adapter_file"]
    saved = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(
        path, map_location="cpu",
    )
    current = get_peft_model_state_dict(model)
    expected = {
        key: value.to(torch.float32 if "lora_" in key else torch.bfloat16)
        for key, value in saved.items()
    }
    if current.keys() != saved.keys() or any(
        current[key].dtype != expected[key].dtype for key in current
    ):
        raise ValueError("Native BF16 evaluation adapter inventory or precision differs")
    assert_identical(current, expected, "native_bf16_adapter_reload")
    if getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Native LC evaluation reloads the BF16 base, not the int8 training copy")
    model.config.use_cache = True
    return model


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from lc_checkpoint import verify, verify_ack
    from lc_run import load_inputs, setup_runtime

    rank, world, runtime = setup_runtime(require_world=False)
    completed = json.loads((args.run / "completed.json").read_text())
    if completed.get("status") != "complete" or completed.get("kind") != "full-fit":
        raise ValueError("Held-out evaluation requires a frozen completed full-budget fit")
    chosen = args.run / "snapshots" / completed["selected_checkpoint"]
    receipt = verify(chosen)
    acknowledgment = json.loads((args.run / "acks" / (chosen.name + ".json")).read_text())
    verify_ack(chosen, acknowledgment)
    if completed["selected_complete_sha256"] != file_sha256(chosen / "complete.json"):
        raise ValueError("Selected checkpoint changed after freezing")
    run_config = json.loads((args.run / "run-config.json").read_text())
    if file_sha256(args.run / "run-config.json") != receipt["files"]["run-config.json"]:
        raise ValueError("Checkpoint/run identity mismatch")
    inputs = load_inputs(args)
    if inputs["identity"] != run_config["inputs"]:
        raise ValueError("Evaluation inputs differ from the frozen training inputs")
    author, native, tokenizer = inputs["author"], inputs["native"], inputs["test_tokenizer"]
    model = load_evaluation_model(inputs, rank, chosen)
    model.config.use_cache = True
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 2
    test = author.data.SeqRecDataset(native, mode="test", prompt_id=0, sample_num=-1)
    records = sequence_records(inputs["interactions"], "test")
    positions = shard_positions(len(test), rank, world)
    loader = DataLoader(
        Subset(test, positions), batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=author.collator.TestCollator(native, tokenizer),
    )
    codes = [[tokenizer.convert_tokens_to_ids(token) for token in code]
             for code in inputs["indices"].values()]
    config = {
        "schema_version": 1, "claim": run_config["claim"], "decoder_policy": args.decoder_policy,
        "native_primary": args.decoder_policy == POLICIES[0],
        "checkpoint": completed, "run_config_sha256": file_sha256(args.run / "run-config.json"),
        "num_beams": 20, "num_return_sequences": 20, "max_new_tokens": 5,
        "sampling": False, "seen_item_exclusion": False, "filter_invalid_codes": False,
        "test_prompt_id": 0, "all_users": len(records), "drop_last": False,
        "batch_size": args.batch_size, "world_size": world,
        "evaluator_sha256": file_sha256(Path(__file__)), "runtime": runtime,
        "inference_precision": INFERENCE_PRECISION,
        "code_duplicates": "Native LC behavior: retain; sum repeated-target NDCG contributions.",
        "item_tie_policy": "Stable distinct-code order, ascending original item ID within a code.",
    }
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "run-config.json", config)
        marker = args.run / f"evaluation-{chosen.name}-{args.decoder_policy}.json"
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"output": str(args.output), "identity": config["run_config_sha256"]}, stream)
    if world > 1:
        torch.distributed.barrier()
    offset = 0
    started = time.monotonic()
    prediction_file = args.output / f"predictions-rank-{rank}.jsonl"
    with prediction_file.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch, targets in loader:
            batch = {key: value.to(rank) for key, value in batch.items()}
            width = batch["input_ids"].shape[1]
            output = model.generate(
                **batch, num_beams=20, num_return_sequences=20, max_new_tokens=5,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                prefix_allowed_tokens_fn=make_constraint(codes, 2, width, args.decoder_policy),
                eos_token_id=2, pad_token_id=0, length_penalty=1.0, early_stopping=True,
            )
            generated = output.sequences[:, width:].cpu().tolist()
            if any(len(seq) != 5 or seq[-1] != 2 or 2 in seq[:-1] for seq in generated):
                raise ValueError("LC generated output did not finish four code tokens with EOS")
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            texts = [text.strip().replace(" ", "") for text in texts]
            scores = output.sequences_scores.float().cpu().tolist()
            for index, target_code in enumerate(targets):
                position = positions[offset]
                record = records[position]
                if target_code != "".join(inputs["indices"][str(record["target"])]):
                    raise ValueError("Native test target/canonical target mismatch")
                result = score_beams(texts[index * 20:(index + 1) * 20],
                                     scores[index * 20:(index + 1) * 20],
                                     record["target"], inputs["indices"])
                result.update(user=record["user"], target=record["target"], position=position)
                stream.write(json.dumps(result) + "\n")
                offset += 1
            stream.flush()
    if offset != len(positions):
        raise ValueError("Evaluation shard coverage incomplete")
    write_json(args.output / f"completed-rank-{rank}.json", {
        "status": "complete", "users": offset, "sha256": file_sha256(prediction_file),
    })
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        rows = []
        for worker in range(world):
            path = args.output / f"predictions-rank-{worker}.jsonl"
            marker = json.loads((args.output / f"completed-rank-{worker}.json").read_text())
            if marker["sha256"] != file_sha256(path):
                raise ValueError("Prediction shard checksum mismatch")
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        verify_coverage([record["user"] for record in records], [row["user"] for row in rows])
        result = {
            "status": "complete", "users": len(rows), "decoder_policy": args.decoder_policy,
            "code": {key: sum(row["code"][key] for row in rows) / len(rows)
                     for key in rows[0]["code"]},
            "item": {key: sum(row["item"][key] for row in rows) / len(rows)
                     for key in rows[0]["item"]},
            "invalid_codes": sum(row["invalid_codes"] for row in rows),
            "duplicate_codes": sum(row["duplicate_codes"] for row in rows),
            "run_config_sha256": file_sha256(args.output / "run-config.json"),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(args.output / "completed.json", result)
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


def main():
    from lc_run import add_input_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-policy", choices=POLICIES, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
