"""Audit every released sequence and actual T5-tokenized record before fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from author import SOURCE_HASHES, default_args, load_backend
from protocol import collision_summary, file_sha256, require_hash, sequence_records, write_json


def compare_author_rows(authored, records, indices, backend: str) -> None:
    if backend != "tiger":
        raise ValueError("This parity check is for the unprompted author TIGER dataset")
    if len(authored) != len(records):
        raise ValueError("Author/canonical split lengths differ")
    for i, record in enumerate(records):
        expected = {
            "input_ids": "".join("".join(indices[str(item)]) for item in record["history"]),
            "labels": "".join(indices[str(record["target"])]),
        }
        if authored[i] != expected:
            raise ValueError(f"Author/canonical split mismatch at record {i}")


def validate_token_rows(inputs, labels, limit: int, eos: int) -> dict[str, int]:
    if len(inputs) != len(labels):
        raise ValueError("Input/target length mismatch")
    for input_ids, label_ids in zip(inputs, labels):
        if not input_ids or not label_ids:
            raise ValueError("Tokenized record has empty input/labels")
        if max(len(input_ids), len(label_ids)) > limit:
            raise ValueError("Tokenized record exceeds the configured length")
        if label_ids[-1] != eos:
            raise ValueError("Tokenized target lost EOS")
    return {"records": len(inputs)}


def read_release(upstream: Path, manifest: dict, dataset: str) -> tuple[dict, dict, dict]:
    folder = upstream / "data" / dataset
    expected = manifest["datasets"][dataset]
    loaded = {}
    for kind, digest in expected["sha256"].items():
        path = folder / f"{dataset}.{kind}.json"
        require_hash(path, digest)
        loaded[kind] = json.loads(path.read_text(encoding="utf-8"))
    inter, items, indices = (loaded[k] for k in ("inter", "item", "index"))
    domain = set(map(str, range(expected["items"])))
    if set(items) != domain or set(indices) != domain or len(inter) != expected["users"]:
        raise ValueError("Released ID domain or user count differs from the frozen manifest")
    if any(str(i) not in domain for seq in inter.values() for i in seq):
        raise ValueError("An interaction item is absent from the catalog")
    if sum(map(len, inter.values())) != expected["interactions"]:
        raise ValueError("Released interaction count differs from the frozen manifest")
    return inter, items, indices


def tokenizer_digest(tokenizer) -> str:
    return hashlib.sha256(json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def make_tiger_tokenizer(author, indices):
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        str(author.directory / "ckpt" / "TIGER"),
        model_max_length=512, local_files_only=True,
    )
    tokens = sorted({token for code in indices.values() for token in code})
    tokenizer.add_tokens(tokens)
    for token in tokens:
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            raise ValueError(f"Semantic-ID token is not atomic: {token}")
    return tokenizer


def audit_tiger(upstream: Path, manifest_path: Path, dataset: str, output: Path) -> dict:
    import numpy as np

    started = time.monotonic()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interactions, _, indices = read_release(upstream, manifest, dataset)
    author = load_backend(upstream, "tiger")
    args = default_args(author, upstream, dataset)
    author.utils.set_seed(42)
    tokenizer = make_tiger_tokenizer(author, indices)
    report: dict[str, Any] = {
        "backend": "tiger", "dataset": dataset, "status": "checking",
        "manifest_sha256": file_sha256(manifest_path),
        "inputs": manifest["datasets"][dataset]["sha256"],
        "source": SOURCE_HASHES["tiger"],
        "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
        "vocab_size": len(tokenizer), "model_max_length": 512,
        "generation_finish_reason": "N/A (static recommendation targets)",
        "collision_summary": collision_summary(indices), "splits": {},
    }
    examples = []
    for split in ("train", "valid", "test"):
        authored = author.data.SeqRecDataset(args, mode=split)
        records = sequence_records(interactions, split)
        compare_author_rows(authored, records, indices, "tiger")
        input_lengths, target_lengths = [], []
        for offset in range(0, len(authored), 512):
            batch = [authored[i] for i in range(offset, min(offset + 512, len(authored)))]
            input_ids = tokenizer([r["input_ids"] for r in batch], truncation=False)["input_ids"]
            label_ids = tokenizer([r["labels"] for r in batch], truncation=False)["input_ids"]
            validate_token_rows(input_ids, label_ids, 512, tokenizer.eos_token_id)
            if any(len(ids) != 5 for ids in label_ids):
                raise ValueError("Released four-token SID target does not tokenize to four codes + EOS")
            input_lengths.extend(map(len, input_ids))
            target_lengths.extend(map(len, label_ids))
            if offset == 0:
                examples.extend({"split": split, **records[i], **batch[i]} for i in range(min(100, len(batch))))
        report["splits"][split] = {
            "records": len(records), "strict_clean": len(records), "over_cap": 0,
            "input_tokens_p50_p90_p99_max": np.quantile(input_lengths, [.5, .9, .99, 1]).tolist(),
            "target_tokens_min_max": [min(target_lengths), max(target_lengths)],
        }
        print(f"AUDITED {dataset} {split}: {len(records)} rows", flush=True)
    if report["splits"]["train"]["records"] != manifest["datasets"][dataset]["training_prefix_examples"]:
        raise ValueError("Training prefix count differs from the reference")
    report.update(status="pass", elapsed_seconds=time.monotonic() - started)
    write_json(output / "audit.json", report)
    write_json(output / "examples.json", examples)
    tokenizer.save_pretrained(output / "tokenizer")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "audit.json").exists():
        raise FileExistsError("Use a new audit directory; existing receipts are immutable")
    audit_tiger(args.upstream.resolve(), args.manifest.resolve(), args.dataset, args.output)


if __name__ == "__main__":
    main()
