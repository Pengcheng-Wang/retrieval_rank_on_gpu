from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.runtime import configure, environment_manifest, require_cuda


def query_representation(query_file: Path) -> dict[str, Any]:
    completed = json.loads((query_file.parent / "completed.json").read_text(encoding="utf-8"))
    values = np.load(query_file, mmap_mode="r")
    if query_file.name == "vectors.npy":
        expected = completed["vectors_sha256"]
        dimension = completed["native_dimension"]
    else:
        match = next(
            (representation for representation in completed["mrl_representations"].values()
             if representation["file"] == query_file.name), None,
        )
        if match is None:
            raise ValueError("Query vector file is not a measured native/MRL representation.")
        expected, dimension = match["sha256"], match["dimension"]
    if not completed["complete"] or values.dtype != np.float16 or values.shape != (completed["rows"], dimension):
        raise ValueError("Query representation payload is incomplete or has an invalid shape/dtype.")
    if sha256(query_file) != expected:
        raise ValueError("Query representation payload checksum changed.")
    return {
        "model": completed["model"], "revision": completed["revision"],
        "specification_sha256": completed["specification_sha256"],
        "stored_dimension": dimension, "stored_dtype": "float16",
        "normalization": "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage.",
        "query_instruction": completed["query_instruction"],
    }


def validate_checkpoint_features(
    binding: dict[str, Any], items: Path, partition: Path, lengths: Path,
    representation: dict[str, Any],
) -> None:
    if binding.get("representation") != representation:
        raise ValueError("Checkpoint uses a different semantic encoder/representation/query instruction.")
    for name, path in (("items", items), ("partition", partition), ("token_lengths", lengths)):
        if binding["sources"].get(name) != sha256(path):
            raise ValueError(f"Checkpoint {name} feature source differs from the trained catalog/codebook.")


def validate_fitting_query_alignment(query_file: Path, panel_file: Path) -> None:
    completed = json.loads((query_file.parent / "completed.json").read_text(encoding="utf-8"))
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    values = np.load(query_file, mmap_mode="r")
    if (
        not completed["complete"] or completed["panel_sha256"] != sha256(panel_file)
        or completed["query_ids"] != panel["query_ids"] or completed["rows"] != len(panel["query_ids"])
        or len(values) != len(panel["query_ids"])
    ):
        raise ValueError("Fitting query-vector/label panel alignment differs in source bytes, IDs or row order.")


def semantic_features(query: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    if query.ndim != 2 or query.shape != item.shape:
        raise ValueError("Semantic ranker requires aligned query/item feature matrices.")
    return torch.cat((query, item, query * item, (query - item).abs()), dim=1).float()


class CachedSemanticMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("Semantic ranker dimension must be positive.")
        self.network = nn.Sequential(
            nn.Linear(dimension * 4, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        self.model_class = f"{type(self).__module__}.{type(self).__name__}"

    def forward(self, query: torch.Tensor, item: torch.Tensor, sparse: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(semantic_features(query, item)).squeeze(-1)


class CachedTorchRec(nn.Module):
    def __init__(self, dimension: int, clusters: int, dcn: bool, device: torch.device) -> None:
        super().__init__()
        from torchrec.models.dlrm import DLRM, DLRM_DCN
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
        from torchrec.modules.embedding_modules import EmbeddingBagCollection

        self.keys = ["query_cluster", "item_cluster", "length_bucket"]
        collection = EmbeddingBagCollection(tables=[
            EmbeddingBagConfig(
                name=name + "_table", embedding_dim=128, num_embeddings=cardinality,
                feature_names=[name],
            ) for name, cardinality in zip(self.keys, (clusters, clusters, 32))
        ], device=device)
        arguments = {
            "embedding_bag_collection": collection, "dense_in_features": dimension * 4,
            "dense_arch_layer_sizes": [512, 128], "over_arch_layer_sizes": [512, 256, 1],
            "dense_device": device,
        }
        self.model = DLRM_DCN(**arguments, dcn_num_layers=3, dcn_low_rank_dim=64) if dcn else DLRM(**arguments)
        self.model_class = f"{type(self.model).__module__}.{type(self.model).__name__}"
        self.to(device)

    def forward(self, query: torch.Tensor, item: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        if sparse.shape != (len(query), 3) or sparse.dtype != torch.int64:
            raise ValueError("TorchRec needs three aligned transferable integer sparse features.")
        count = len(query)
        values = sparse.T.contiguous().reshape(-1)
        collection = KeyedJaggedTensor(
            keys=self.keys, values=values, lengths=torch.ones(3 * count, dtype=torch.long, device=query.device),
            stride=count, length_per_key=[count] * 3, offset_per_key=[0, count, 2 * count, 3 * count],
            index_per_key={name: index for index, name in enumerate(self.keys)},
        )
        return self.model(semantic_features(query, item), collection).squeeze(-1)


def make_cached_ranker(
    name: str, dimension: int, clusters: int, device: torch.device, seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    if name == "semantic-mlp":
        return CachedSemanticMLP(dimension).to(device)
    if name in ("torchrec-dlrm", "torchrec-dcn"):
        return CachedTorchRec(dimension, clusters, name.endswith("dcn"), device)
    raise ValueError("Unknown declared cached semantic ranker.")


def prepare_pairs(panel: dict[str, Any], candidates: np.ndarray, seed: int, negatives: int = 12) -> np.ndarray:
    if candidates.ndim != 2 or len(candidates) != len(panel["query_ids"]) or negatives <= 0:
        raise ValueError("Fitting candidates must align with the frozen query groups.")
    rng, rows = np.random.default_rng(seed), []
    for query, identifier in enumerate(panel["query_ids"]):
        labels = {int(item): int(grade) for item, grade in panel["qrels"][identifier].items()}
        positives = sorted(item for item, grade in labels.items() if grade >= 1)
        if not positives:
            raise ValueError("Fitting query has no official positive.")
        alternatives = [
            int(item) for item in candidates[query] if item >= 0 and int(item) not in positives
        ]
        alternatives = list(dict.fromkeys(alternatives))
        if not alternatives:
            raise ValueError("Fitting query has no label-independent hard-negative candidates.")
        selected = rng.choice(alternatives, min(negatives, len(alternatives)), replace=False)
        rows.extend((query, item, 1) for item in positives)
        rows.extend((query, int(item), 0) for item in selected)
    return np.asarray(rows, dtype=np.int64)


def train_ranker(
    name: str, item_file: Path, query_file: Path, panel_file: Path, candidates_file: Path,
    partition_file: Path, token_lengths_file: Path, output: Path,
    seed: int = 17, epochs: int = 3, batch_size: int = 2048,
) -> dict[str, Any]:
    if min(epochs, batch_size) <= 0:
        raise ValueError("Training epochs and batch size must be positive.")
    configure(seed, 4)
    device = require_cuda()
    items = np.load(item_file, mmap_mode="r")
    queries = np.load(query_file, mmap_mode="r")
    candidates = np.load(candidates_file, mmap_mode="r")
    partition = np.load(partition_file)
    assignments, centroids = partition["assignments"], partition["centroids"]
    lengths = np.load(token_lengths_file, mmap_mode="r")
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    if panel["name"] not in ("fit", "kuairand-fit") or items.shape[1] != queries.shape[1] or lengths.shape != (len(items),):
        raise ValueError("Cached ranker may fit only matching training query/corpus features.")
    representation = query_representation(query_file)
    validate_fitting_query_alignment(query_file, panel_file)
    item_receipt = json.loads((item_file.parent / "completed.json").read_text(encoding="utf-8"))
    expected_items = {key: value for key, value in representation.items() if key != "query_instruction"}
    if (
        not item_receipt["complete"] or item_receipt.get("representation") != expected_items
        or sha256(item_file) != item_receipt["vectors_sha256"]
    ):
        raise ValueError("Fitting query and item vectors use different or unattested encoder representations.")
    binding = {
        "model": name, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "dimension": items.shape[1], "clusters": len(centroids),
        "representation": representation,
        "sources": {
            key: sha256(path) for key, path in {
                "items": item_file, "queries": query_file, "panel": panel_file, "candidates": candidates_file,
                "partition": partition_file, "token_lengths": token_lengths_file,
            }.items()
        },
        "learning_rate": 0.001, "negative_samples": 12,
        "query_features": "Transferable semantic vector/cluster; no queryIDembedding.",
        "negative_truth": "Unjudged retrieved alternatives used as conventional weak negatives; not exhaustive relevance truth.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text())
        if sha256(output / "model.pt") != completed["checkpoint_sha256"]:
            raise ValueError("Complete fitted ranker checkpoint changed.")
        return completed
    rows = prepare_pairs(panel, candidates, seed)
    centroid_gpu = torch.as_tensor(centroids, device=device)
    query_cluster = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), 512):
        query = torch.as_tensor(np.array(queries[start:start + 512], dtype=np.float32), device=device)
        query_cluster[start:start + len(query)] = (query @ centroid_gpu.T).argmax(dim=1).cpu().numpy()
    del centroid_gpu, query
    model = make_cached_ranker(name, items.shape[1], len(centroids), device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    started, epoch_start, losses, previous_training_seconds = time.perf_counter(), 0, [], 0.0
    checkpoint = output / "training-state.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Training resume checkpoint is bound to a different fitting protocol.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, losses = state["completed_epochs"], state["losses"]
        if "training_elapsed_seconds" not in state:
            raise ValueError("Training resume checkpoint lacks measured accumulated runtime; use a new run.")
        previous_training_seconds = state["training_elapsed_seconds"]
    for epoch in range(epoch_start, epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(rows))
        total_loss, count = 0.0, 0
        for start in range(0, len(rows), batch_size):
            selected = rows[order[start:start + batch_size]]
            qids, item_ids = selected[:, 0], selected[:, 1]
            query = torch.as_tensor(np.array(queries[qids], dtype=np.float32), device=device)
            item = torch.as_tensor(np.array(items[item_ids], dtype=np.float32), device=device)
            sparse = torch.as_tensor(np.column_stack((
                query_cluster[qids], assignments[item_ids],
                np.minimum(np.floor(np.log2(np.maximum(lengths[item_ids], 1))).astype(np.int64), 31),
            )), device=device, dtype=torch.long)
            target = torch.as_tensor(selected[:, 2], device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(query, item, sparse)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Semantic fitting loss is nonfinite; no checkpoint is published.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
            count += len(selected)
        losses.append(total_loss / count)
        temporary = output / "training-state.partial.pt"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "binding": binding,
            "completed_epochs": epoch + 1, "losses": losses,
            "training_elapsed_seconds": previous_training_seconds + time.perf_counter() - started,
        }, temporary)
        temporary.replace(checkpoint)
        write_json(output / "progress.json", {"completed_epochs": epoch + 1, "losses": losses})
        print(f"SEMANTIC_RANKER_EPOCH {name} seed={seed} epoch={epoch + 1} loss={losses[-1]:.6f}", flush=True)
    model.eval()
    temporary = output / "model.partial.pt"
    torch.save({"model": model.state_dict(), "binding": binding}, temporary)
    temporary.replace(output / "model.pt")
    completed = {
        "complete": True, "name": name, "model_class": model.model_class, "seed": seed,
        "parameters": sum(value.numel() for value in model.parameters()), "training_pairs": len(rows),
        "epochs": epochs, "losses": losses,
        "training_seconds": previous_training_seconds + time.perf_counter() - started,
        "training_runtime_scope": "Measured fitting-worker segments accumulated across epoch-complete resume; downtime excluded.",
        "checkpoint_sha256": sha256(output / "model.pt"), "environment": environment_manifest(),
        "torchrec_implementation": "Actual unsharded localTorchRecDLRM/DCN; no assumed automaticFBGEMM servingoptimization.",
        "feature_budget": {
            "dense_semantic_dimension": items.shape[1] * 4, "sparse_embedding_dimension": 128,
            "dense_arch": [512, 128], "over_arch": [512, 256, 1], "cross_layers": 3, "cross_rank": 64,
        },
    }
    write_json(output / "completed.json", completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised semantic MLP and actual TorchRec DLRM/DCN fitting.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--token-lengths", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train_ranker(
        args.model, Path(args.items), Path(args.queries), Path(args.panel), Path(args.candidates),
        Path(args.partition), Path(args.token_lengths), Path(args.output), args.seed,
    )


if __name__ == "__main__":
    main()
