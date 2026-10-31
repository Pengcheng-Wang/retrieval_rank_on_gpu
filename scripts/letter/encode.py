"""Audited, resumable calls to the author's unchanged semantic encoder routine."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from download_openllama import MODEL_ID, REVISION, safe_filename, validate_reader_version, validate_track
from prepare import read_release, tokenizer_digest
from protocol import bind_run_config, file_sha256, require_hash, write_json
from tokenizer_run import fresh_output_directory


SOURCE_HASHES = {
    "amazon_text_emb.py": "05c2ce863e39e24b703bd754f6c1b4929b06c51a6718d5798359d9f8cbf73f7b",
    "utils.py": "7f6b3481e6a161fb8a903a037bf751575e899e77da51bb326e340673a1730c50",
}


def load_author_encoder(upstream: Path):
    source = upstream / "data_process"
    for name, digest in SOURCE_HASHES.items():
        require_hash(source / name, digest)
    if "utils" in sys.modules or "amazon_text_emb" in sys.modules:
        raise RuntimeError("Author encoder requires a fresh process without generic module conflicts")
    sys.path.insert(0, str(source.resolve()))
    return importlib.import_module("amazon_text_emb")


def ordered_texts(author, items: dict) -> list:
    rows = sorted(author.generate_text(items, ["title", "description"]), key=lambda row: row[0])
    if (len(rows) != len(items) or [row[0] for row in rows] != list(range(len(items)))
            or any(len(row[1]) != 2 or not all(isinstance(s, str) for s in row[1]) for row in rows)):
        raise ValueError("Author semantic fields must cover every item with title and description")
    return rows


def audit_texts(rows: list, tokenizer, limit: int) -> dict:
    if not rows or limit < 1:
        raise ValueError("Semantic audit needs a nonempty corpus and positive token limit")
    lengths, empty, unknown = [], 0, 0
    fields = [text for _, texts in rows for text in texts]
    for offset in range(0, len(fields), 256):
        texts = fields[offset:offset + 256]
        encoded = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        if len(encoded) != len(texts):
            raise ValueError("Tokenizer changed semantic field coverage")
        for text, tokens in zip(texts, encoded):
            if not tokens or tokens[0] != tokenizer.bos_token_id or tokens.count(tokenizer.bos_token_id) != 1:
                raise ValueError("Semantic field lost its single native BOS")
            lengths.append(len(tokens))
            empty += int(text == "")
            unknown += tokens.count(tokenizer.unk_token_id)
    over = sum(length > limit for length in lengths)
    return {
        "status": "pass" if not over else "fail", "items": len(rows), "records": len(fields),
        "strict_clean": len(fields) - over, "over_cap": over, "limit": limit,
        "length_tokens_p50_p90_p99_max": np.quantile(lengths, [.5, .9, .99, 1]).tolist(),
        "empty_cleaned_fields": empty, "unknown_tokens": unknown,
        "generation_finish_reason": "N/A (static item metadata, not generated traces)",
        "template": "none; native single BOS, no automatic EOS, no chat wrapper",
        "pooling": "native masked mean per field, then equal mean of title/description",
    }


def read_chunk(path: Path, identity: str, first: int, stop: int) -> np.ndarray:
    receipt = json.loads((path / "completed.json").read_text(encoding="utf-8"))
    if (receipt.get("status") != "complete" or receipt.get("run_config_sha256") != identity
            or receipt.get("first_item") != first or receipt.get("stop_item") != stop
            or receipt.get("shape") != [stop - first, 4096]):
        raise ValueError("Semantic chunk belongs to a different model, corpus or item interval")
    require_hash(path / "embeddings.npy", receipt["embeddings_sha256"])
    array = np.load(path / "embeddings.npy", allow_pickle=False)
    if (array.shape != (stop - first, 4096) or array.dtype != np.float32
            or not np.isfinite(array).all()):
        raise ValueError("Invalid semantic chunk shape, dtype or values")
    return array


def validate_encoding_spec(spec: dict) -> None:
    validate_track(spec)
    expected = {
        "semantic_encoding": {
            "dtype": "float32", "fields": ["title", "description"],
            "max_tokens_per_field": 2048, "normalization": "none",
        },
        "tokenizer": {
            "use_fast": False, "legacy": True, "bos": 1, "eos": 2, "padding": 0,
            "automatic_eos": False, "apply_chat_template": False,
        },
    }
    if any(spec[section].get(key) != value
           for section, settings in expected.items() for key, value in settings.items()):
        raise ValueError("Declared semantic/tokenizer settings differ from the native execution contract")


def verify_backbone(args, spec: dict) -> dict:
    validate_encoding_spec(spec)
    receipt = json.loads(args.model_receipt.read_text(encoding="utf-8"))
    if (receipt.get("status") != "downloaded_converted_verified_not_runtime_validated"
            or receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION
            or receipt.get("track_id") != spec["track_id"]
            or receipt.get("exact_original_llama1_reproduction") is not False
            or receipt.get("spec_sha256") != file_sha256(args.method_spec)):
        raise ValueError("Semantic encoder cannot reuse a differently identified backbone")
    validate_reader_version(receipt["conversion_reader"]["torch"])
    for name, digest in receipt["converted_files_sha256"].items():
        require_hash(args.model / safe_filename(name), digest)
    return receipt


def bank_artifact(args, source: Path, relative: Path) -> None:
    if args.bank_destination is None:
        return
    import fcntl
    from bank import copy_artifact

    if socket.gethostname() != args.writer_host:
        raise RuntimeError("Only the designated node may persist semantic artifacts")
    args.bank_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.bank_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        copy_artifact(source, args.bank_destination / relative)
    print(f"BANKED_SEMANTIC {relative}", flush=True)


def run(args) -> None:
    import torch
    from transformers import AutoModel, LlamaTokenizer

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    spec = json.loads(args.method_spec.read_text(encoding="utf-8"))
    backbone = verify_backbone(args, spec)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    author = load_author_encoder(args.upstream)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model, legacy=True, local_files_only=True, model_max_length=2048,
    )
    tokenizer.pad_token_id = 0
    if (len(tokenizer) != 32000 or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2
            or tokenizer.is_fast or not tokenizer.add_bos_token or tokenizer.add_eos_token):
        raise ValueError("Unexpected slow-tokenizer vocabulary or special-token policy")
    token_files = {k: v for k, v in backbone["converted_files_sha256"].items()
                   if k in {"tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}}
    model_files = {k: v for k, v in backbone["converted_files_sha256"].items()
                   if k.startswith("model") or k == "config.json"}
    prepared = []
    for dataset in args.datasets:
        _, items, _ = read_release(args.upstream, manifest, dataset)
        rows = ordered_texts(author, items)
        output = args.output / dataset
        config = {
            "kind": "semantic-encoding-compatible-backbone", "track_id": spec["track_id"],
            "exact_original_llama1_reproduction": False, "dataset": dataset,
            "model_id": MODEL_ID, "revision": REVISION, "model_files_sha256": model_files,
            "tokenizer_files_sha256": token_files, "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
            "method_spec_sha256": file_sha256(args.method_spec), "author_source_sha256": SOURCE_HASHES,
            "harness_sha256": file_sha256(Path(__file__)), "chunk_size": args.chunk_size,
            "smoke_items": args.smoke_items, "batch_size": 1, "dtype": "float32",
            "item_order": "row_equals_zero_based_item_id",
            "item_json_sha256": manifest["datasets"][dataset]["sha256"]["item"],
            "token_limit": 2048, "word_drop_ratio": -1, "normalization": "none",
            "deterministic_algorithms": True, "allow_tf32": False,
        }
        bind_run_config(output, config)
        audit = audit_texts(rows, tokenizer, 2048)
        write_json(output / "token-audit.json", audit)
        write_json(output / "examples.json", rows[:5])
        if audit["status"] != "pass":
            raise ValueError(f"{dataset} contains over-limit semantic fields; no encoding admitted")
        print(f"AUDITED_SEMANTIC {dataset} {json.dumps(audit)}", flush=True)
        prepared.append((dataset, output, rows, audit, config))
    pending = []
    for dataset, output, rows, audit, config in prepared:
        if (output / "completed.json").is_file():
            completed = json.loads((output / "completed.json").read_text())
            require_hash(output / "embeddings.npy", completed["embeddings_sha256"])
            require_hash(output / "embedding-receipt.json", completed["embedding_receipt_sha256"])
            receipt = json.loads((output / "embedding-receipt.json").read_text())
            require_hash(output / "run-config.json", receipt["run_config_sha256"])
            bank_artifact(args, output, Path(dataset) / "complete")
            print(f"ALREADY_ENCODED {dataset}", flush=True)
        else:
            pending.append((dataset, output, rows, audit, config))
    if args.audit_only or not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic encoding requires an explicitly allocated CUDA device")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model, loading = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        use_safetensors=True, local_files_only=True, output_loading_info=True,
    )
    if (loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]
            or set(loading["unexpected_keys"]) - {"lm_head.weight"}):
        raise ValueError(f"Semantic base-model loading changed parameter coverage: {loading}")
    model = model.cuda().eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The semantic encoder must use native float32 parameters")
    draining = [False]
    signal.signal(signal.SIGTERM, lambda sig, frame: draining.__setitem__(0, True))
    chunks_this_call = 0
    for dataset, output, rows, audit, config in pending:
        started = time.monotonic()
        identity = file_sha256(output / "run-config.json")
        total = min(args.smoke_items, len(rows)) if args.smoke_items is not None else len(rows)
        chunks = []
        recovery = args.output / "interrupted" / dataset
        for first in range(0, total, args.chunk_size):
            stop = min(first + args.chunk_size, total)
            chunk = output / "chunks" / f"items-{first:05d}-{stop:05d}"
            if (chunk / "completed.json").is_file():
                read_chunk(chunk, identity, first, stop)
                print(f"REUSED_SEMANTIC_CHUNK {dataset} {first}:{stop}", flush=True)
            else:
                fresh_output_directory(chunk, "completed.json", recovery)
                original_args = SimpleNamespace(
                    dataset=dataset, root=str(chunk), plm_name="openllama-v2",
                    max_sent_len=2048, device=torch.device("cuda:0"),
                )
                local_rows = [[item - first, texts] for item, texts in rows[first:stop]]
                author.generate_item_embedding(original_args, local_rows, tokenizer, model, word_drop_ratio=-1)
                generated = chunk / f"{dataset}.emb-openllama-v2-td.npy"
                array = np.load(generated, allow_pickle=False)
                if array.shape != (stop - first, 4096) or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError("Author encoder produced an invalid semantic tensor")
                generated.rename(chunk / "embeddings.npy")
                write_json(chunk / "completed.json", {
                    "status": "complete", "kind": "semantic_embedding_chunk",
                    "first_item": first, "stop_item": stop, "run_config_sha256": identity,
                    "shape": list(array.shape), "embeddings_sha256": file_sha256(chunk / "embeddings.npy"),
                })
                chunks_this_call += 1
                print(f"ENCODED_SEMANTIC {dataset} {stop}/{total}", flush=True)
            bank_artifact(args, chunk, Path(dataset) / "chunks" / chunk.name)
            chunks.append((chunk, first, stop))
            if draining[0] or (args.stop_after_chunks is not None and chunks_this_call >= args.stop_after_chunks):
                print("SEMANTIC_ENCODER_DRAINED_AT_COMPLETE_CHUNK", flush=True)
                return
        array = np.concatenate([read_chunk(path, identity, first, stop) for path, first, stop in chunks])
        if array.shape != (total, 4096):
            raise ValueError("Assembled semantic bank lost item coverage")
        temporary = output / "embeddings.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(output / "embeddings.npy")
        digest = file_sha256(output / "embeddings.npy")
        write_json(output / "embedding-receipt.json", {
            "status": "pass", "smoke": args.smoke_items is not None,
            "encoder": "OpenLLaMA-7B-v2", "track_id": spec["track_id"],
            "exact_original_llama1_reproduction": False,
            "model_id": MODEL_ID, "revision": REVISION, "dataset": dataset,
            "item_order": config["item_order"], "item_json_sha256": config["item_json_sha256"],
            "model_files_sha256": model_files, "tokenizer_files_sha256": token_files,
            "author_source_sha256": SOURCE_HASHES, "run_config_sha256": identity,
            "token_audit": audit, "shape": list(array.shape), "dtype": "float32",
            "embeddings_sha256": digest, "method_spec_sha256": file_sha256(args.method_spec),
        })
        write_json(output / "completed.json", {
            "status": "complete", "kind": "semantic_embedding_bank", "smoke": args.smoke_items is not None,
            "embeddings_sha256": digest, "shape": list(array.shape),
            "embedding_receipt_sha256": file_sha256(output / "embedding-receipt.json"),
            "seconds_this_invocation": time.monotonic() - started,
        })
        bank_artifact(args, output, Path(dataset) / "complete")
        print(f"SEMANTIC_BANK_COMPLETE {dataset} {digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "model", "model-receipt", "method-spec", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--smoke-items", type=int)
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--bank-destination", type=Path)
    parser.add_argument("--bank-lock", type=Path)
    parser.add_argument("--writer-host")
    args = parser.parse_args()
    if (args.chunk_size < 1 or any(v is not None and v < 1 for v in (args.smoke_items, args.stop_after_chunks))
            or len(set(args.datasets)) != len(args.datasets)):
        parser.error("Intervals must be positive and datasets unique")
    if any((args.bank_destination, args.bank_lock, args.writer_host)) and not all(
        (args.bank_destination, args.bank_lock, args.writer_host)
    ):
        parser.error("Bank destination, lock and designated writer must be supplied together")
    run(args)


if __name__ == "__main__":
    main()
