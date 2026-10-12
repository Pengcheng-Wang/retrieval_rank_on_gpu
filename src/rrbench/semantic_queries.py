from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.metrics import latency_summary
from rrbench.runtime import configure, environment_manifest, require_cuda
from rrbench.semantic_models import (
    SemanticEncoder, configuration_digest, mrl_prefix, validate_encoding_contract,
    validate_model_files, validate_smoke_contract,
)
from rrbench.semantic_token_audit import length_summary


def validate_query_panel(output: Path, panel: Path, specification: dict[str, Any]) -> dict[str, Any]:
    completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    source = json.loads(panel.read_text(encoding="utf-8"))
    if not completed["complete"] or completed["panel_sha256"] != sha256(panel):
        raise ValueError("Query completion is bound to a different panel text/order.")
    if completed["query_ids"] != source["query_ids"] or completed["rows"] != len(source["query_ids"]):
        raise ValueError("Query embedding rows do not align with the exact source query IDs.")
    effective_instruction = source.get("history_instruction", specification.get("query_instruction", ""))
    if completed.get("query_instruction") != effective_instruction:
        raise ValueError("Query completion uses a different effective query instruction.")
    validate_encoding_contract(specification, completed)
    representations = [{
        "file": "vectors.npy", "dimension": completed["native_dimension"],
        "sha256": completed["vectors_sha256"],
    }, *completed["mrl_representations"].values()]
    for representation in representations:
        path = output / representation["file"]
        if sha256(path) != representation["sha256"]:
            raise ValueError("Query embedding bytes changed from their complete receipt.")
        values = np.load(path, mmap_mode="r")
        if values.dtype != np.float16 or values.shape != (completed["rows"], representation["dimension"]):
            raise ValueError("Query embedding representation shape or dtype changed.")
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values.astype(np.float32), axis=1), 1, atol=0.002,
        ):
            raise ValueError("Query embeddings are not finite unit-normalized stored representations.")
    return completed


def encode_query_panel(
    encoder: SemanticEncoder, panel: Path, output: Path, batch_size: int,
) -> dict[str, Any]:
    specification = encoder.specification
    source = json.loads(panel.read_text(encoding="utf-8"))
    queries, ids = source["queries"], source["query_ids"]
    effective_instruction = source.get("history_instruction", specification["query_instruction"])
    if not queries or len(queries) != len(ids) or len(set(ids)) != len(ids) or batch_size <= 0:
        raise ValueError("Query panel must contain aligned unique IDs and a positive batch size.")
    bind_run_config(output, {
        "panel_sha256": sha256(panel), "specification_sha256": configuration_digest(specification),
        "batch_size": batch_size, "stored_dtype": "float16", "mode": "model-cardqueryinstruction",
        "effective_query_instruction": effective_instruction,
    })
    if (output / "completed.json").exists():
        return validate_query_panel(output, panel, specification)
    vectors = np.lib.format.open_memmap(
        output / "vectors.partial.npy", mode="w+", dtype=np.float16,
        shape=(len(queries), encoder.native_dimension),
    )
    mrl = {
        dimension: np.lib.format.open_memmap(
            output / f"vectors_mrl{dimension}.partial.npy", mode="w+", dtype=np.float16,
            shape=(len(queries), dimension),
        ) for dimension in specification["dimensions"] if dimension != encoder.native_dimension
    }
    lengths = np.empty(len(queries), dtype=np.uint32)
    measured, started = [], time.perf_counter()
    for position in range(0, len(queries), batch_size):
        batch = queries[position:position + batch_size]
        formatted = encoder.format_texts(batch, query=True, instruction=effective_instruction)
        raw = encoder.tokenizer(
            formatted, padding=False, truncation=False, return_length=True,
            return_attention_mask=False, return_token_type_ids=False,
        )
        lengths[position:position + len(batch)] = raw["length"]
        encoded, stats = encoder.encode(batch, query=True, instruction=effective_instruction)
        vectors[position:position + len(batch)] = encoded.cpu().numpy().astype(np.float16)
        for dimension, representation in mrl.items():
            representation[position:position + len(batch)] = mrl_prefix(encoded, dimension).cpu().numpy().astype(np.float16)
        measured.append(vars(stats))
    vectors.flush()
    del vectors
    (output / "vectors.partial.npy").replace(output / "vectors.npy")
    reduced = {}
    for dimension, representation in mrl.items():
        representation.flush()
        del representation
        filename = f"vectors_mrl{dimension}.npy"
        (output / f"vectors_mrl{dimension}.partial.npy").replace(output / filename)
        reduced[str(dimension)] = {
            "file": filename, "dimension": dimension, "sha256": sha256(output / filename),
            "transform": "Same FP32 unit-normalized prefix used for document representations.",
        }
    latency = {}
    for count in (1, 16):
        batch = (queries * count)[:count]
        for _ in range(5):
            encoder.encode(batch, query=True, instruction=effective_instruction)
        samples = [
            encoder.encode(batch, query=True, instruction=effective_instruction)[1].seconds * 1000
            for _ in range(30)
        ]
        latency[str(count)] = latency_summary(samples, count)
    completed = {
        "complete": True, "rows": len(queries), "query_ids": ids,
        "panel_sha256": sha256(panel), "specification_sha256": configuration_digest(specification),
        "model": specification["model"], "revision": specification["revision"],
        "native_dimension": encoder.native_dimension, "vectors_sha256": sha256(output / "vectors.npy"),
        "mrl_representations": reduced, "stored_dtype": "float16", "compute_dtype": "bfloat16",
        "query_instruction": effective_instruction, "length_audit": length_summary(
            lengths, specification["max_tokens"],
        ),
        "elapsed_seconds": time.perf_counter() - started, "batch_stats": measured,
        "online_query_encoding_latency": latency, "environment": environment_manifest(),
    }
    write_json(output / "completed.json", completed)
    validate_query_panel(output, panel, specification)
    print(f"SEMANTIC_QUERY_PANEL_COMPLETE {specification['key']} rows={len(queries)}", flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned semantic query vectors and measured online encoder costs.")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel", required=True, action="append")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-gate")
    parser.add_argument("--lease-owner")
    parser.add_argument("--parent", type=int)
    parser.add_argument("--parent-start", type=int)
    args = parser.parse_args()
    if args.start_gate:
        from rrbench.process_lease import guard_worker_parent, wait_for_start_gate

        if args.lease_owner is None or args.parent is None or args.parent_start is None:
            parser.error("start-gate requires lease-owner, parent and parent-start")
        guard_worker_parent(args.parent, args.parent_start)
        wait_for_start_gate(Path(args.start_gate), args.lease_owner, -2, args.parent, args.parent_start)
    configure(17, 4)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    specification = registry["models"][args.model]
    smoke = json.loads(Path(args.smoke).read_text(encoding="utf-8"))
    if not smoke["complete"]:
        raise ValueError("Model smoke must be complete before query encoding.")
    row = next(row for row in smoke["models"] if row["key"] == args.model)
    validate_smoke_contract(specification, row)
    validate_model_files(specification, Path(args.cache), row["model_files"])
    encoder = SemanticEncoder(specification, Path(args.cache), require_cuda())
    for panel in args.panel:
        path = Path(panel)
        encode_query_panel(
            encoder, path, Path(args.output) / path.stem, 32 if args.model == "qwen8" else 64,
        )


if __name__ == "__main__":
    main()
