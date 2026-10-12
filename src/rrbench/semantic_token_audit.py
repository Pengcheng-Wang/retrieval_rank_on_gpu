from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_models import configuration_digest, pinned_snapshot


def length_summary(lengths: np.ndarray, cap: int) -> dict[str, Any]:
    if (
        lengths.ndim != 1 or not len(lengths) or cap <= 0
        or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 0).any()
    ):
        raise ValueError("Token audit requires positive integer lengths and a positive cap.")
    values = np.quantile(lengths, [0.5, 0.9, 0.99])
    return {
        "rows": len(lengths), "token_cap": cap,
        "raw_length_quantiles": {
            "p50": float(values[0]), "p90": float(values[1]), "p99": float(values[2]),
            "max": int(lengths.max()),
        },
        "raw_rows_exceeding_cap": int((lengths > cap).sum()),
        "raw_rows_equal_to_cap": int((lengths == cap).sum()),
        "strict_processed_rows": int((np.minimum(lengths, cap) > 0).sum()),
        "processed_length_max": int(np.minimum(lengths, cap).max()),
        "generation_finish_reason": "N/A: static text encoder inputs, no generated completions.",
        "input_policy": "Model-card tokenizer once; explicit document-prefix truncation at the recorded cap.",
        "audit_scope": "Every static input record, not a sampled training-trace certification.",
        "pass": True,
    }


def audit_corpus(
    corpus: Path, specification: dict[str, Any], cache: Path, output: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    snapshot = pinned_snapshot(specification, cache)
    token_files = {
        filename: sha256(snapshot / filename)
        for filename in (
            "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.txt", "vocab.json", "merges.txt", "added_tokens.json",
        ) if (snapshot / filename).exists()
    }
    if not token_files:
        raise ValueError("No pinned tokenizer files are available for a full-input audit.")
    binding = {
        "specification_sha256": configuration_digest(specification), "tokenizer_files": token_files,
        "source_shards": manifest["shards"], "expected_rows": manifest["rows"],
        "packages": {
            package: importlib.metadata.version(package) for package in ("transformers", "tokenizers")
        },
        "mode": "document text; no query instruction or chat-template wrapper",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if sha256(output / "raw_lengths.npy") != complete["raw_lengths_sha256"]:
            raise ValueError("Full tokenizer-audit length array changed.")
        measured = length_summary(
            np.load(output / "raw_lengths.npy", mmap_mode="r"), specification["max_tokens"],
        )
        if measured != complete["lengths"] or measured["rows"] != manifest["rows"]:
            raise ValueError("Full tokenizer-audit receipt differs from its measured input lengths.")
        return complete
    tokenizer = AutoTokenizer.from_pretrained(
        specification["model"], revision=specification["revision"], cache_dir=cache,
        trust_remote_code=False,
    )
    temporary = output / "raw_lengths.partial.npy"
    lengths = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint32, shape=(manifest["rows"],),
    )
    started, position, special_marker_rows = time.perf_counter(), 0, 0
    for shard in manifest["shards"]:
        source = corpus / shard["file"]
        if sha256(source) != shard["sha256"] or shard["start"] != position:
            raise ValueError("Token audit source bytes or global row order changed.")
        texts = []
        source_rows = 0

        def flush_batch() -> None:
            nonlocal position
            encoded = tokenizer(
                texts, padding=False, truncation=False, return_length=True,
                return_attention_mask=False, return_token_type_ids=False,
            )
            actual = np.asarray(encoded["length"], dtype=np.uint32)
            if actual.shape != (len(texts),) or (actual == 0).any():
                raise ValueError("Static text has invalid post-tokenization lengths.")
            lengths[position:position + len(texts)] = actual
            position += len(texts)
            texts.clear()

        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                identifier, separator, text = line.partition("\t")
                if not separator or not text.strip() or int(identifier) != position + len(texts):
                    raise ValueError("Token audit requires every nonempty globally mapped corpus row.")
                special_marker_rows += int("<|im_start|>" in text or "<|im_end|>" in text)
                texts.append(text.rstrip("\n"))
                source_rows += 1
                if len(texts) == 1024:
                    flush_batch()
        if texts:
            flush_batch()
        if source_rows != shard["rows"]:
            raise ValueError("Token audit omitted or duplicated declared shard rows.")
        lengths.flush()
        write_json(output / "progress.json", {"rows": position, "expected_rows": manifest["rows"]})
        print(f"FULL_INPUT_TOKEN_AUDIT {specification['key']} rows={position}", flush=True)
    if position != manifest["rows"]:
        raise ValueError("Full static-token audit did not process every declared corpus record.")
    measured = length_summary(lengths, specification["max_tokens"])
    lengths.flush()
    del lengths
    temporary.replace(output / "raw_lengths.npy")
    complete = {
        "complete": True, "lengths": measured, "model": specification["model"],
        "revision": specification["revision"], "specification_sha256": binding["specification_sha256"],
        "tokenizer_files": token_files, "raw_lengths_sha256": sha256(output / "raw_lengths.npy"),
        "raw_chat_control_marker_rows": special_marker_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "completed.json", complete)
    print(f"ALL_STATIC_INPUTS_AUDITED {specification['key']} rows={position}", flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Full static corpus token-length audit before LM encoding.")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    audit_corpus(
        Path(args.corpus), registry["models"][args.model], Path(args.cache), Path(args.output),
    )


if __name__ == "__main__":
    main()
