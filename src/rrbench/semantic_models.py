from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import numpy as np
import torch

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.runtime import configure, environment_manifest, require_cuda

MODEL_ASSET_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "README.md"]


def configuration_digest(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def validate_encoding_contract(specification: dict[str, Any], completed: dict[str, Any]) -> None:
    if completed.get("specification_sha256") != configuration_digest(specification):
        raise ValueError("Semantic completion is bound to a different or unattested encoding specification.")
    native = max(specification["dimensions"])
    if completed["native_dimension"] != native:
        raise ValueError("Semantic completion has a different native dimension.")
    required = {str(dimension) for dimension in specification["dimensions"] if dimension != native}
    if set(completed["mrl_representations"]) != required:
        raise ValueError("Semantic completion does not contain the complete required MRL representation set.")


def validate_smoke_contract(specification: dict[str, Any], smoke: dict[str, Any]) -> None:
    if smoke.get("specification_sha256") != configuration_digest(specification):
        raise ValueError("Semantic smoke is bound to a different or unattested model specification.")
    if smoke["status"] != "ok" or smoke["rows"] < 100 or smoke["revision"] != specification["revision"]:
        raise ValueError("Semantic smoke is not a successful100-real-row test of the pinned revision.")
    if "dimensions" in specification:
        native = max(specification["dimensions"])
        required = {str(dimension) for dimension in specification["dimensions"] if dimension != native}
        if smoke["dimension"] != native or set(smoke["mrl_representations"]) != required:
            raise ValueError("Semantic smoke does not cover every required native/MRL dimension.")


def pinned_snapshot(specification: dict[str, Any], cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        specification["model"], revision=specification["revision"], cache_dir=cache,
        local_files_only=True, allow_patterns=MODEL_ASSET_PATTERNS,
    ))


def model_file_manifest(specification: dict[str, Any], cache: Path) -> dict[str, Any]:
    snapshot = pinned_snapshot(specification, cache)
    files = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in sorted(snapshot.rglob("*")) if path.is_file()
        and any(fnmatch.fnmatch(str(path.relative_to(snapshot)), pattern) for pattern in MODEL_ASSET_PATTERNS)
    }
    if not any(filename.endswith(".safetensors") for filename in files):
        raise ValueError("Pinned model snapshot has no verified safetensors weights.")
    return {
        "revision": specification["revision"], "files": files, "files_sha256": configuration_digest(files),
        "asset_policy": "Required safetensors/config/tokenizer/model-card assets; alternateONNX/pickleexports not loaded.",
    }


def validate_model_files(specification: dict[str, Any], cache: Path, expected: dict[str, Any]) -> None:
    if model_file_manifest(specification, cache) != expected:
        raise ValueError("Local pinned model/tokenizer files differ from their successful smoke attestation.")


def model_registry(config: Path, destination: Path) -> dict[str, Any]:
    configuration = json.loads(config.read_text(encoding="utf-8"))
    config_digest = configuration_digest(configuration)
    if destination.exists():
        registry = json.loads(destination.read_text(encoding="utf-8"))
        if registry["configuration_sha256"] != config_digest:
            raise ValueError("Model registry is bound to a different experiment configuration.")
        return registry
    entries = {}
    for item in configuration["embeddings"] + configuration["rerankers"]:
        url = "https://huggingface.co/api/models/" + item["model"]
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        if metadata.get("private") or metadata.get("gated") or metadata.get("disabled"):
            raise ValueError("The declared semantic model is not publicly accessible.")
        actual_license = metadata.get("cardData", {}).get("license")
        if actual_license not in ("apache-2.0", "mit"):
            raise ValueError("Unexpected semantic-model license; do not substitute a model silently.")
        entries[item["key"]] = {
            **item, "revision": metadata["sha"], "model_card_license": actual_license,
            "metadata_url": url, "model_card_url": f"https://huggingface.co/{item['model']}/blob/{metadata['sha']}/README.md",
            "declared_weight_parameters": metadata.get("safetensors", {}).get("total"),
        }
    registry = {
        "configuration_sha256": config_digest, "models": entries,
        "snapshot_policy": "Pinned public revisions, safetensors only, trust_remote_code=False.",
    }
    write_json(destination, registry)
    return registry


def normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("Model output must be a finite embedding matrix.")
    norms = values.norm(dim=-1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("Zero embedding cannot represent a valid cosine retrieval vector.")
    return values / norms


def mrl_prefix(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if values.ndim != 2 or not 0 < dimension <= values.shape[1]:
        raise ValueError("MRL prefix dimension must fit the native embedding.")
    return normalize(values[:, :dimension])


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.any(dim=1).all():
        raise ValueError("Pooling requires aligned hidden states and at least one nonpadding token.")
    if method == "CLS":
        return hidden[:, 0]
    if method == "attention-mask mean":
        weights = mask[..., None].float()
        return (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1)
    if method.startswith("last valid token"):
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = (positions[None, :] * mask).max(dim=1).values.long()
        return hidden[torch.arange(len(hidden), device=hidden.device), last]
    raise ValueError("Unknown declared embedding pooling policy.")


@dataclass
class BatchStats:
    rows: int
    real_tokens: int
    padded_tokens: int
    at_token_cap: int
    seconds: float


class SemanticEncoder:
    def __init__(self, specification: dict[str, Any], cache: Path, device: torch.device) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.specification = specification
        self.device = device
        self.is_qwen = specification["model"].startswith("Qwen/")
        self.tokenizer = AutoTokenizer.from_pretrained(
            specification["model"], revision=specification["revision"],
            cache_dir=cache, trust_remote_code=False,
            padding_side="left" if self.is_qwen else "right",
        )
        self.model = AutoModel.from_pretrained(
            specification["model"], revision=specification["revision"],
            cache_dir=cache, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.native_dimension = self.model.config.hidden_size
        if self.native_dimension != max(specification["dimensions"]):
            raise ValueError("Native model dimension differs from the pinned experiment specification.")
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def format_texts(self, texts: list[str], query: bool, instruction: str | None = None) -> list[str]:
        if not query:
            return texts
        task = self.specification["query_instruction"] if instruction is None else instruction
        if self.is_qwen:
            return [f"Instruct: {task}\nQuery:{text}" for text in texts]
        return [task + text for text in texts]

    @torch.inference_mode()
    def encode(
        self, texts: list[str], query: bool = False, instruction: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[torch.Tensor, BatchStats]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Encoding requires a nonempty batch of nonempty text.")
        formatted = self.format_texts(texts, query, instruction)
        limit = max_tokens or self.specification["max_tokens"]
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        batch = self.tokenizer(
            formatted, padding=True, truncation=True, max_length=limit,
            return_tensors="pt", pad_to_multiple_of=8,
        ).to(self.device)
        output = self.model(**batch, **({"use_cache": False} if self.is_qwen else {}))
        vectors = normalize(pool_hidden(output.last_hidden_state, batch["attention_mask"], self.specification["pooling"]))
        torch.cuda.synchronize(self.device)
        lengths = batch["attention_mask"].sum(dim=1)
        stats = BatchStats(
            len(texts), int(lengths.sum().item()), batch["input_ids"].numel(),
            int((lengths >= limit).sum().item()), time.perf_counter() - started,
        )
        return vectors, stats


class TextReranker:
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, specification: dict[str, Any], cache: Path, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        self.specification, self.device = specification, device
        self.is_qwen = specification["model"].startswith("Qwen/")
        self.tokenizer = AutoTokenizer.from_pretrained(
            specification["model"], revision=specification["revision"],
            cache_dir=cache, trust_remote_code=False,
            padding_side="left" if self.is_qwen else "right",
        )
        model_class = AutoModelForCausalLM if self.is_qwen else AutoModelForSequenceClassification
        self.model = model_class.from_pretrained(
            specification["model"], revision=specification["revision"],
            cache_dir=cache, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        if self.is_qwen:
            self.prefix = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
            self.suffix = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
            self.yes = self.tokenizer.encode("yes", add_special_tokens=False)
            self.no = self.tokenizer.encode("no", add_special_tokens=False)
            if len(self.yes) != 1 or len(self.no) != 1 or self.yes == self.no:
                raise ValueError("Official Qwen yes/no scorer requires distinct single tokens.")

    def tokenize(self, queries: list[str], documents: list[str], instruction: str | None = None):
        if len(queries) != len(documents) or not queries:
            raise ValueError("Reranker requires aligned nonempty query/document batches.")
        limit = self.specification["max_tokens"]
        if self.is_qwen:
            texts = [
                f"<Instruct>: {instruction or self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
                for query, doc in zip(queries, documents)
            ]
            budget = limit - len(self.prefix) - len(self.suffix)
            if budget <= 0:
                raise ValueError("Qwen score grammar exceeds the token budget.")
            encoded = self.tokenizer(
                texts, padding=False, truncation=True, max_length=budget,
                add_special_tokens=False, return_attention_mask=False,
            )
            body = [self.prefix + tokens + self.suffix for tokens in encoded["input_ids"]]
            return self.tokenizer.pad({"input_ids": body}, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        return self.tokenizer(
            queries, documents, padding=True, truncation="only_second", max_length=limit,
            pad_to_multiple_of=8, return_tensors="pt",
        )

    @torch.inference_mode()
    def score(self, queries: list[str], documents: list[str], instruction: str | None = None) -> tuple[torch.Tensor, BatchStats]:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        inputs = self.tokenize(queries, documents, instruction).to(self.device)
        if self.is_qwen:
            output = self.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1].float()
            scores = logits[:, self.yes[0]] - logits[:, self.no[0]]
        else:
            scores = self.model(**inputs).logits.reshape(-1).float()
        if not torch.isfinite(scores).all() or len(scores) != len(queries):
            raise RuntimeError("Text reranker produced invalid scores.")
        torch.cuda.synchronize(self.device)
        lengths = inputs["attention_mask"].sum(dim=1)
        return scores, BatchStats(
            len(queries), int(lengths.sum().item()), inputs["input_ids"].numel(),
            int((lengths >= self.specification["max_tokens"]).sum().item()),
            time.perf_counter() - started,
        )


def encode_shard(
    source: Path, destination: Path, spec: dict[str, Any], cache: Path, batch_size: int,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0 or (max_rows is not None and max_rows <= 0):
        raise ValueError("Batch size and optional row limit must be positive before encoding.")
    configure(17, 4)
    device = require_cuda()
    started = time.perf_counter()
    binding = {
        "model": spec["model"], "revision": spec["revision"], "source_sha256": sha256(source),
        "max_rows": max_rows, "batch_size": batch_size, "max_tokens": spec["max_tokens"],
        "native_dim": max(spec["dimensions"]), "stored_dtype": "float16",
        "specification_sha256": configuration_digest(spec),
    }
    if (destination / "completed.json").exists():
        bind_run_config(destination, binding)
        completed = json.loads((destination / "completed.json").read_text(encoding="utf-8"))
        validate_encoding_contract(spec, completed)
        if sha256(destination / "vectors.npy") != completed["vectors_sha256"]:
            raise ValueError("Complete semantic embedding shard checksum changed.")
        ids_path = destination / "ids.npy"
        if not ids_path.exists() or sha256(ids_path) != completed["ids_sha256"]:
            raise ValueError("Complete semantic shard ID mapping is missing or corrupted.")
        if np.load(ids_path, mmap_mode="r").shape != (completed["rows"],):
            raise ValueError("Semantic shard ID mapping has the wrong length.")
        for dimension, representation in completed["mrl_representations"].items():
            if sha256(destination / representation["file"]) != representation["sha256"]:
                raise ValueError("Complete MRL representation checksum changed.")
        return completed
    bind_run_config(destination, binding)
    encoder = SemanticEncoder(spec, cache, device)
    texts, identifiers = [], []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            identifier, separator, text = line.partition("\t")
            if not separator or not text.strip():
                raise ValueError("Corpus shard contains an invalid or empty item text.")
            identifiers.append(int(identifier))
            texts.append(text.rstrip("\n"))
            if max_rows and len(texts) >= max_rows:
                break
    count = len(texts)
    if not count:
        raise ValueError("No corpus rows are available for encoding.")
    temporary = destination / "vectors.partial.npy"
    vectors = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(count, encoder.native_dimension)
    )
    mrl = {
        dimension: np.lib.format.open_memmap(
            destination / f"vectors_mrl{dimension}.partial.npy", mode="w+",
            dtype=np.float16, shape=(count, dimension),
        )
        for dimension in spec["dimensions"] if dimension != encoder.native_dimension
    }
    stats = []
    written = 0
    for start in range(0, count, batch_size):
        encoded, measured = encoder.encode(texts[start:start + batch_size])
        vectors[start:start + len(encoded)] = encoded.cpu().numpy().astype(np.float16)
        for dimension, representation in mrl.items():
            representation[start:start + len(encoded)] = mrl_prefix(encoded, dimension).cpu().numpy().astype(np.float16)
        written += len(encoded)
        stats.append(vars(measured))
        if len(stats) % 100 == 0:
            vectors.flush()
            write_json(destination / "progress.json", {
                "completed_rows": start + len(encoded), "expected_rows": count,
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"ENCODING {spec['key']} rows={start + len(encoded)}/{count}", flush=True)
    if written != count:
        raise RuntimeError("Encoding did not write every declared row; no complete artifact may be published.")
    vectors.flush()
    del vectors
    temporary.replace(destination / "vectors.npy")
    representations = {}
    for dimension, representation in mrl.items():
        representation.flush()
        filename = f"vectors_mrl{dimension}.npy"
        (destination / f"vectors_mrl{dimension}.partial.npy").replace(destination / filename)
        representations[str(dimension)] = {
            "file": filename, "dimension": dimension, "sha256": sha256(destination / filename),
            "transform": "Native embedding prefix normalized in FP32 before FP16 storage.",
        }
    np.save(destination / "ids.npy", np.asarray(identifiers, dtype=np.int64))
    result = {
        "complete": True, "rows": count, "native_dimension": encoder.native_dimension,
        "parameters": encoder.parameters, "model": spec["model"], "revision": spec["revision"],
        "vectors_sha256": sha256(destination / "vectors.npy"), "ids_sha256": sha256(destination / "ids.npy"),
        "stored_dtype": "float16", "model_compute_dtype": "bfloat16", "attention": "sdpa",
        "normalization": "FP32 unit normalization before FP16 storage; exact oracle uses stored vectors",
        "elapsed_seconds": time.perf_counter() - started,
        "real_tokens": sum(row["real_tokens"] for row in stats),
        "padded_tokens": sum(row["padded_tokens"] for row in stats),
        "at_token_cap_rows": sum(row["at_token_cap"] for row in stats),
        "batch_samples": stats, "environment": environment_manifest(),
        "source_sha256": binding["source_sha256"],
        "mrl_representations": representations, "written_rows": written,
        "specification_sha256": binding["specification_sha256"],
    }
    write_json(destination / "completed.json", result)
    print(f"ENCODING_COMPLETE {spec['key']} rows={count} dim={encoder.native_dimension}", flush=True)
    return result


def smoke_models(
    registry: dict[str, Any], source: Path, cache: Path, destination: Path, production_batches: bool = False,
) -> None:
    configure(17, 4)
    device = require_cuda()
    texts = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            texts.append(line.partition("\t")[2].strip())
            if len(texts) == 100:
                break
    if len(texts) != 100:
        raise ValueError("A100-row real-text smoke panel is required.")
    results = []
    for key, spec in registry["models"].items():
        started = time.perf_counter()
        batch_size = (32 if key.startswith("qwen8") else 64) if production_batches else 8
        torch.cuda.reset_peak_memory_stats(device)
        if "dimensions" in spec:
            encoder = SemanticEncoder(spec, cache, device)
            measured = []
            batches = []
            for start in range(0, 100, batch_size):
                values, stats = encoder.encode(texts[start:start + batch_size])
                if values.shape[1] != max(spec["dimensions"]):
                    raise RuntimeError("Semantic encoder smoke has the wrong dimension.")
                batches.append(values.cpu().numpy())
                measured.append(vars(stats))
            single, _ = encoder.encode([texts[0]])
            grouped, _ = encoder.encode(texts[:4])
            error = float((single[0] - grouped[0]).abs().max().item())
            if error > 0.03:
                raise RuntimeError("Embedding pooling/padding smoke differs unexpectedly across batching.")
            queries, _ = encoder.encode(["What is the capital of China?", "Explain gravity"], query=True)
            mrl_smoke = {
                str(dimension): {
                    "query_shape": list(mrl_prefix(queries, dimension).shape),
                    "document_shape": list(mrl_prefix(torch.tensor(np.concatenate(batches)), dimension).shape),
                }
                for dimension in spec["dimensions"] if dimension != encoder.native_dimension
            }
            row = {
                "key": key, "rows": 100, "dimension": encoder.native_dimension,
                "parameters": encoder.parameters, "revision": spec["revision"],
                "padding_batch_max_absolute_error": error,
                "query_shape": list(queries.shape), "batch_stats": measured,
                "status": "ok", "kind": "embedding",
                "mrl_representations": mrl_smoke,
            }
            del encoder, single, grouped, values, queries
        else:
            reranker = TextReranker(spec, cache, device)
            values, measured = [], []
            for start in range(0, 100, batch_size):
                scores, stats = reranker.score(
                    ["Find a passage that describes the topic in the document."] * min(batch_size, 100 - start),
                    texts[start:start + batch_size],
                )
                values.extend(scores.tolist())
                measured.append(vars(stats))
            reference, _ = reranker.score(
                ["What is the capital of China?"] * 2,
                ["Beijing is the capital of China.", "Gravity attracts masses toward each other."],
            )
            if not (reference[0] > reference[1]).item():
                raise RuntimeError("Model-card scoring smoke failed on a declared relevant/irrelevant pair.")
            row = {
                "key": key, "rows": 100, "parameters": reranker.parameters,
                "revision": spec["revision"], "batch_stats": measured,
                "known_pair_order_correct": True, "status": "ok", "kind": "reranker",
            }
            del reranker, scores, reference
        row["seconds"] = time.perf_counter() - started
        row["specification_sha256"] = configuration_digest(spec)
        row["model_files"] = model_file_manifest(spec, cache)
        row["production_batch_size"] = batch_size
        row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        results.append(row)
        write_json(destination / "smoke-progress.json", results)
        torch.cuda.empty_cache()
    write_json(destination / "smoke.json", {"complete": True, "rows_per_model": 100, "models": results})


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned real-text semantic models and shard encoding.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    registry = subparsers.add_parser("registry")
    registry.add_argument("--config", required=True)
    registry.add_argument("--output", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--production-batches", action="store_true")
    smoke.add_argument("--start-gate")
    smoke.add_argument("--lease-owner")
    smoke.add_argument("--parent", type=int)
    smoke.add_argument("--parent-start", type=int)
    encode = subparsers.add_parser("encode")
    for command in (smoke, encode):
        command.add_argument("--registry", required=True)
        command.add_argument("--source", required=True)
        command.add_argument("--cache", required=True)
        command.add_argument("--output", required=True)
    encode.add_argument("--model", required=True)
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.operation == "registry":
        result = model_registry(Path(args.config), Path(args.output))
        print({key: spec["revision"] for key, spec in result["models"].items()})
    else:
        pinned = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        if args.operation == "smoke":
            if args.start_gate:
                from rrbench.process_lease import guard_worker_parent, wait_for_start_gate

                if args.lease_owner is None or args.parent is None or args.parent_start is None:
                    parser.error("start-gate requires lease-owner, parent and parent-start")
                guard_worker_parent(args.parent, args.parent_start)
                wait_for_start_gate(
                    Path(args.start_gate), args.lease_owner, -1, args.parent, args.parent_start,
                )
            smoke_models(
                pinned, Path(args.source), Path(args.cache), Path(args.output), args.production_batches,
            )
        else:
            encode_shard(
                Path(args.source), Path(args.output), pinned["models"][args.model],
                Path(args.cache), args.batch_size, args.max_rows,
            )


if __name__ == "__main__":
    main()
