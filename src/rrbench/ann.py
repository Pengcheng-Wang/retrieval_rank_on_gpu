from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rrbench.filters import BloomFilter, exact_candidate_mask
from rrbench.io import write_json
from rrbench.quantization import quantize_symmetric


@dataclass
class Partition:
    centroids: np.ndarray
    assignments: np.ndarray
    order: np.ndarray
    offsets: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def train_partition(
    items: np.ndarray, clusters: int, seed: int, directory: Path, training_iterations: int = 20
) -> Partition:
    import faiss

    if not 1 < clusters <= len(items):
        raise ValueError("Require 1 < cluster count <= item count.")
    saved = directory / "partition.npz"
    if saved.exists():
        data = np.load(saved)
        if data["centroids"].shape != (clusters, items.shape[1]) or len(data["order"]) != len(items):
            raise ValueError("Saved IVF partition shape does not match the dataset.")
        return Partition(*(data[key] for key in ("centroids", "assignments", "order", "offsets")))
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    sample_count = min(len(items), max(clusters * 40, 100000))
    selected = rng.choice(len(items), sample_count, replace=False)
    training = np.asarray(items[selected], dtype=np.float32, order="C")
    gpu_available = torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources")
    kmeans = faiss.Kmeans(
        items.shape[1], clusters, niter=training_iterations, seed=seed,
        gpu=gpu_available, spherical=True, verbose=False
    )
    kmeans.train(training)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32, order="C")
    quantizer = faiss.IndexFlatIP(items.shape[1])
    quantizer.add(centroids)
    resources = None
    if gpu_available:
        resources = faiss.StandardGpuResources()
        resources.setDefaultNullStreamAllDevices()
        search_index = faiss.index_cpu_to_gpu(resources, 0, quantizer)
    else:
        search_index = quantizer
    assignments = np.empty(len(items), dtype=np.int64)
    for position in range(0, len(items), 262144):
        _, found = search_index.search(np.asarray(items[position : position + 262144]), 1)
        assignments[position : position + len(found)] = found[:, 0]
    order = np.argsort(assignments, kind="stable")
    lengths = np.bincount(assignments, minlength=clusters)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    np.savez(saved, centroids=centroids, assignments=assignments, order=order, offsets=offsets)
    write_json(directory / "partition.json", {
        "seed": seed, "clusters": clusters, "training_iterations": training_iterations,
        "training_rows": sample_count, "build_seconds": time.perf_counter() - start,
        "spherical": True, "metric": "inner_product", "shared_assignments": True,
        "description": "One trained quantizer and fixed assignments shared by every IVF backend",
    })
    return Partition(centroids, assignments, order, offsets)


def make_faiss_ivf(items: np.ndarray, partition: Partition):
    import faiss

    quantizer = faiss.IndexFlatIP(items.shape[1])
    quantizer.add(partition.centroids)
    index = faiss.IndexIVFFlat(quantizer, items.shape[1], len(partition.centroids), faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    for start in range(0, len(items), 262144):
        vectors = np.ascontiguousarray(items[start : start + 262144])
        ids = np.arange(start, start + len(vectors), dtype=np.int64)
        assignment = np.ascontiguousarray(partition.assignments[start : start + len(vectors)])
        index.add_core(len(vectors), faiss.swig_ptr(vectors), faiss.swig_ptr(ids), faiss.swig_ptr(assignment))
    if index.ntotal != len(items):
        raise RuntimeError("FAISS index did not ingest the complete shared partition.")
    return index


class FaissANN:
    def __init__(self, items: np.ndarray, partition: Partition, gpu: bool, half: bool = False):
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401

        self.cuda = gpu
        self.resources = None
        index = make_faiss_ivf(items, partition)
        if gpu:
            self.resources = faiss.StandardGpuResources()
            self.resources.setDefaultNullStreamAllDevices()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = half
            if hasattr(options, "use_cuvs"):
                options.use_cuvs = False
            index = faiss.index_cpu_to_gpu(self.resources, torch.cuda.current_device(), index, options)
        self.index = index

    def search(self, queries: torch.Tensor, k: int, probes: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda and (k > 2048 or probes > 2048):
            raise ValueError("Classic FAISS GPU supports k and nprobe <= 2048.")
        self.index.nprobe = probes
        return self.index.search(queries if self.cuda else queries.cpu(), k)


class FaissHNSW:
    def __init__(self, items: np.ndarray, neighbors: int = 32):
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401

        self.cuda = False
        self.index = faiss.IndexHNSWFlat(items.shape[1], neighbors, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 100
        self.index.add(np.ascontiguousarray(items))

    def search(self, queries: torch.Tensor, k: int, probes: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.index.hnsw.efSearch = max(k, probes)
        return self.index.search(queries.cpu(), k)


class SilverANN:
    def __init__(
        self, items: np.ndarray, partition: Partition, dtype: str,
        features: np.ndarray | None = None, filter_mode: str = "partial",
    ) -> None:
        import silvertorch.ops._load_ops  # noqa: F401

        if dtype not in ("fp32", "fp16", "int8-global", "int8-rowwise"):
            raise ValueError("Unsupported SilverTorch representation.")
        if filter_mode not in ("partial", "full"):
            raise ValueError("filter_mode must be partial or full.")
        self.cuda = True
        self.dtype = dtype
        self.filter_mode = filter_mode
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.centroids = torch.as_tensor(partition.centroids, device=self.device)
        self.order = torch.as_tensor(partition.order, device=self.device)
        self.offsets = torch.as_tensor(partition.offsets, device=self.device)
        self.lengths = torch.as_tensor(partition.lengths, device=self.device)
        sorted_values = torch.from_numpy(np.asarray(items[partition.order]))
        self.scales = None
        if dtype.startswith("int8"):
            quantized = quantize_symmetric(sorted_values, dtype.split("-")[1])
            self.embeddings = quantized.values.to(self.device)
            self.scales = quantized.scales.to(self.device)
        else:
            self.embeddings = sorted_values.to(
                device=self.device, dtype=torch.float16 if dtype == "fp16" else torch.float32
            )
        self.bloom = None
        self.features = None
        if features is not None:
            self.features = torch.as_tensor(features[partition.order], device=self.device)
            self.bloom = BloomFilter(self.features)

    def probes(self, queries: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        selected = (queries.float() @ self.centroids.T).topk(count, dim=1).indices
        lengths = self.lengths[selected]
        maximum = int(lengths.sum(dim=1).max().item())
        return selected.contiguous(), lengths.contiguous(), maximum

    def score(
        self, queries: torch.Tensor, selected: torch.Tensor, lengths: torch.Tensor,
        maximum: int, plans: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_scales = None
        if self.dtype.startswith("int8"):
            quantized = quantize_symmetric(queries, "rowwise")
            query_values, query_scales = quantized.values, quantized.scales
        else:
            query_values = queries.to(self.embeddings.dtype).contiguous()
        if plans is not None:
            if self.bloom is None:
                raise ValueError("Filter plans require a Bloom index.")
            if self.filter_mode == "partial":
                partial = self.bloom.partial(plans, self.offsets[selected], lengths)
                scores, internal_ids = torch.ops.st.fused_kmean_ann_with_partial_masks(
                    self.offsets, selected, lengths, self.embeddings, query_values, maximum, *partial
                )
            else:
                full = self.bloom.search(plans, packed=True)
                scores, internal_ids = torch.ops.st.fused_kmean_ann(
                    self.offsets, selected, lengths, self.embeddings, query_values, maximum,
                    filtering_bit_mask=full,
                )
        else:
            scores, internal_ids = torch.ops.st.fused_kmean_ann(
                self.offsets, selected, lengths, self.embeddings, query_values, maximum
            )
        if query_scales is not None:
            item_scales = self.scales
            if item_scales is None:
                raise RuntimeError("INT8 item scales are missing.")
            if len(item_scales) > 1:
                item_scales = item_scales[internal_ids.long().clamp_min(0), 0]
            scores = scores.float() * item_scales.reshape(scores.shape if item_scales.numel() > 1 else (1, 1))
            scores = scores * query_scales
        return scores.float().masked_fill(internal_ids < 0, -torch.inf), internal_ids

    def search(
        self, queries: torch.Tensor, k: int, probes: int,
        filter_texts: list[str] | None = None,
        plans: tuple[torch.Tensor, torch.Tensor] | None = None,
        eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected, lengths, maximum = self.probes(queries, probes)
        if filter_texts is not None:
            if self.bloom is None or self.features is None or len(filter_texts) != len(queries):
                raise ValueError("Filtering requires aligned metadata and one expression per query.")
            if plans is None:
                plans = self.bloom.plans(filter_texts)
        scores, internal_ids = self.score(queries, selected, lengths, maximum, plans)
        if eligible is not None:
            if eligible.shape != (len(queries), len(self.order)) or eligible.dtype != torch.bool:
                raise ValueError("Eligibility must be Boolean [queries, original item rows].")
            external = self.order[internal_ids.long().clamp_min(0)]
            rows = torch.arange(len(queries), device=self.device)[:, None]
            scores.masked_fill_(~eligible[rows, external], -torch.inf)
        if filter_texts is not None:
            # Verification precedes top-k so Bloom false positives cannot consume
            # the finite candidate budget; negative literals were conservatively removed.
            features = self.features[internal_ids.long().clamp_min(0)]
            scores.masked_fill_(~exact_candidate_mask(features, filter_texts), -torch.inf)
        if k > scores.shape[1]:
            padding = k - scores.shape[1]
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
            internal_ids = torch.nn.functional.pad(internal_ids, (0, padding), value=-1)
        top_scores, positions = scores.topk(k, dim=1)
        internal = internal_ids.long().gather(1, positions)
        external = self.order[internal.clamp_min(0)]
        external = external.masked_fill(~torch.isfinite(top_scores), -1)
        return top_scores, external
