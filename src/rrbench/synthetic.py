from __future__ import annotations

from pathlib import Path

import numpy as np

from rrbench.io import write_json


def generate_catalog(
    directory: Path, count: int, dimension: int, queries: int, seed: int, distribution: str
) -> tuple[np.ndarray, np.ndarray]:
    if min(count, dimension, queries) <= 0:
        raise ValueError("Catalog dimensions and query count must be positive.")
    if distribution not in ("isotropic", "clustered"):
        raise ValueError("Unknown synthetic distribution.")
    directory.mkdir(parents=True, exist_ok=True)
    expected = {
        "count": count,
        "dimension": dimension,
        "queries": queries,
        "seed": seed,
        "distribution": distribution,
        "normalized": True,
        "generator": "numpy PCG64; float32 normals; fixed 65536-row generation chunks",
        "purpose": "synthetic systems measurements, not recommendation quality",
    }
    manifest = directory / "synthetic.json"
    if manifest.exists():
        import json

        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Existing catalog has a different configuration.")
        return np.load(directory / "items.npy", mmap_mode="r"), np.load(directory / "queries.npy")
    rng = np.random.default_rng(seed)
    cluster_count = 64
    centers = rng.standard_normal((cluster_count, dimension), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    items = np.lib.format.open_memmap(
        directory / "items.npy", mode="w+", dtype=np.float32, shape=(count, dimension)
    )
    for start in range(0, count, 65536):
        size = min(65536, count - start)
        block = rng.standard_normal((size, dimension), dtype=np.float32)
        if distribution == "clustered":
            block = block * (0.15 / np.sqrt(dimension))
            block += centers[rng.integers(0, cluster_count, size)]
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        items[start : start + size] = block
    items.flush()
    query_rng = np.random.default_rng(seed + 100003)
    query_vectors = query_rng.standard_normal((queries, dimension), dtype=np.float32)
    if distribution == "clustered":
        query_vectors *= 0.15 / np.sqrt(dimension)
        query_vectors += centers[query_rng.integers(0, cluster_count, queries)]
    query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12)
    np.save(directory / "queries.npy", query_vectors)
    write_json(manifest, expected)
    return items, query_vectors
