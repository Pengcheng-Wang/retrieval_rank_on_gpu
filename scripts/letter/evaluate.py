"""Fixed-budget, all-user T5 evaluation with code and original-item metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from author import default_args, load_backend
from prepare import read_release, tokenizer_digest
from protocol import (
    PrefixTrie, file_sha256, rank_metrics, code_to_items, sequence_records,
    verify_coverage, write_json,
)
from rrbench.io import append_jsonl


def generate_fixed(model, inputs: dict, prefix_allowed_tokens, beams: int):
    if beams < 10:
        raise ValueError("The declared top-10 protocol requires at least ten beams")
    return model.generate(
        **inputs, max_new_tokens=10, prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=beams, num_return_sequences=beams,
        output_scores=True, return_dict_in_generate=True, early_stopping=True,
    )


def verify_model_binding(checkpoint: Path, completed: dict, config: dict, dataset: str, inputs: dict):
    if (completed.get("status") != "complete"
            or completed["run_config_sha256"] != file_sha256(checkpoint / "run-config.json")
            or completed["model_sha256"] != file_sha256(checkpoint / "pytorch_model.bin")
            or config["dataset"] != dataset or config["inputs"] != inputs):
        raise ValueError("Checkpoint, input or run-configuration identity mismatch")


def evaluate(args) -> None:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA; no silent CPU fallback")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evaluation outputs are immutable; use a new directory")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    interactions, _, indices = read_release(args.upstream, manifest, args.dataset)
    author = load_backend(args.upstream, "tiger")
    author.utils.set_seed(args.seed)
    dataset_args = default_args(author, args.upstream, args.dataset)
    authored = author.data.SeqRecDataset(dataset_args, mode=args.split)
    records = sequence_records(interactions, args.split)
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint, local_files_only=True, model_max_length=512)
    config = json.loads((args.checkpoint / "run-config.json").read_text(encoding="utf-8"))
    completed = json.loads((args.checkpoint / "completed.json").read_text(encoding="utf-8"))
    verify_model_binding(args.checkpoint, completed, config, args.dataset,
                         manifest["datasets"][args.dataset]["sha256"])
    if config["tokenizer_vocab_sha256"] != tokenizer_digest(tokenizer):
        raise ValueError("Checkpoint, dataset or tokenizer identity mismatch")
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True).cuda().eval()
    groups = code_to_items(indices)
    trie = PrefixTrie([[0] + tokenizer.encode(code) for code in sorted(groups)])
    prefix = lambda batch_id, sentence: trie.allowed(sentence.tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "evaluation-config.json", {
        "dataset": args.dataset, "split": args.split, "seed": args.seed,
        "checkpoint_sha256": completed["model_sha256"], "beams": args.beams,
        "batch_size": args.batch_size, "max_new_tokens": 10,
        "seen_item_exclusion": False, "constraint": "full-prefix catalog trie",
        "expected_users": len(records), "source_run": config,
    })
    totals = {"code": {}, "item": {}}
    seen = []
    started = time.monotonic()
    collator = author.collator.TestCollator(dataset_args, tokenizer)
    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            batch = [authored[i] for i in range(offset, min(offset + args.batch_size, len(records)))]
            inputs, targets = collator(batch)
            output = generate_fixed(model, inputs.to("cuda"), prefix, args.beams)
            decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            scores = output.sequences_scores.cpu().tolist()
            for b, target_code in enumerate(targets):
                record = records[offset + b]
                pairs = sorted(zip(
                    decoded[b * args.beams:(b + 1) * args.beams],
                    scores[b * args.beams:(b + 1) * args.beams],
                ), key=lambda pair: pair[1], reverse=True)
                codes = [text.strip().replace(" ", "") for text, _ in pairs]
                if any(code not in groups for code in codes):
                    raise ValueError("Full-prefix generation emitted an invalid semantic ID")
                if len(set(codes)) != len(codes):
                    raise ValueError("Duplicate beams would distort the native NDCG metric")
                item_predictions = []
                for code in codes:
                    item_predictions.extend(groups[code])
                item_predictions = list(dict.fromkeys(item_predictions))[:10]
                metrics = {
                    "code": rank_metrics(codes, target_code),
                    "item": rank_metrics(item_predictions, record["target"]),
                }
                for kind, values in metrics.items():
                    for name, value in values.items():
                        totals[kind][name] = totals[kind].get(name, 0.0) + value
                seen.append(record["user"])
                append_jsonl(args.output / "predictions.jsonl", {
                    "user": record["user"], "target_item": record["target"],
                    "target_code": target_code, "codes": codes,
                    "scores": [value for _, value in pairs],
                    "item_predictions": item_predictions,
                    "target_collision_size": len(groups[target_code]), "metrics": metrics,
                })
            if offset % (args.batch_size * 50) == 0:
                print(f"EVALUATED {len(seen)}/{len(records)}", flush=True)
    verify_coverage([r["user"] for r in records], seen)
    write_json(args.output / "completed.json", {
        "status": "complete", "users": len(seen), "elapsed_seconds": time.monotonic() - started,
        "metrics": {kind: {k: v / len(seen) for k, v in values.items()}
                    for kind, values in totals.items()},
        "predictions_sha256": file_sha256(args.output / "predictions.jsonl"),
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("upstream", "manifest", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beams", type=int, choices=(20,), default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
