"""Full-data audit for variable-length Figure 8 tokenizer identifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from author import SOURCE_HASHES, default_args, load_backend
from figure8_tokenizer import validate_identifier_bank
from prepare import (
    compare_author_rows,
    make_tiger_tokenizer,
    read_release,
    tokenizer_digest,
    validate_token_rows,
)
from protocol import collision_summary, file_sha256, sequence_records, write_json


def audit_variable_tiger(
    upstream: Path, manifest_path: Path, dataset: str, codes_path: Path,
    codebook_sizes: list[int], output: Path,
) -> dict:
    started = time.monotonic()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interactions, _, indices = read_release(upstream, manifest, dataset)
    codes = np.load(codes_path, allow_pickle=False)
    bank = validate_identifier_bank(indices, codes, len(indices), codebook_sizes)
    author = load_backend(upstream, "tiger")
    args = default_args(author, upstream, dataset)
    author.utils.set_seed(42)
    tokenizer = make_tiger_tokenizer(author, indices)
    tokens = sorted({token for values in indices.values() for token in values})
    if any(len(tokenizer.encode(token, add_special_tokens=False)) != 1 for token in tokens):
        raise ValueError("Every Figure8 identifier token must be atomic")
    report = {
        "backend": "tiger",
        "dataset": dataset,
        "status": "checking",
        "manifest_sha256": file_sha256(manifest_path),
        "inputs": manifest["datasets"][dataset]["sha256"],
        "source": SOURCE_HASHES["tiger"],
        "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
        "vocab_size": len(tokenizer),
        "model_max_length": 512,
        "identifier_bank": bank,
        "collision_summary": collision_summary(indices),
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        authored = author.data.SeqRecDataset(args, mode=split)
        records = sequence_records(interactions, split)
        compare_author_rows(authored, records, indices, "tiger")
        input_lengths, target_lengths = [], []
        for offset in range(0, len(authored), 512):
            batch = [authored[i] for i in range(offset, min(offset + 512, len(authored)))]
            input_ids = tokenizer([row["input_ids"] for row in batch], truncation=False)["input_ids"]
            label_ids = tokenizer([row["labels"] for row in batch], truncation=False)["input_ids"]
            validate_token_rows(input_ids, label_ids, 512, tokenizer.eos_token_id)
            if any(len(row) != bank["target_tokens"] for row in label_ids):
                raise ValueError("Figure8 target does not tokenize to L atomic codes plus EOS")
            input_lengths.extend(map(len, input_ids))
            target_lengths.extend(map(len, label_ids))
        report["splits"][split] = {
            "records": len(records),
            "strict_clean": len(records),
            "over_cap": 0,
            "atomic_identifier_tokens": True,
            "input_tokens_p50_p90_p99_max":
                np.quantile(input_lengths, [.5, .9, .99, 1]).tolist(),
            "target_tokens_min_max": [min(target_lengths), max(target_lengths)],
        }
    report.update(status="pass", elapsed_seconds=time.monotonic() - started)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", report)
    tokenizer.save_pretrained(output / "tokenizer")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "manifest", "codes", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments",), required=True)
    parser.add_argument("--codebook-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Figure8 audits are immutable; use a new output directory")
    audit_variable_tiger(
        args.upstream.resolve(), args.manifest.resolve(), args.dataset,
        args.codes.resolve(), args.codebook_sizes, args.output.resolve(),
    )


if __name__ == "__main__":
    main()
