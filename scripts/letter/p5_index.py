"""Full-catalog adaptations of the pinned P5 identifier construction."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering

from p5_source import cid_functions, digest_json, native_args, native_cid_format, write_json


class MetadataUnavailable(ValueError):
    pass


def metadata_view(items, categories=False):
    domain = set(map(str, range(len(items))))
    if set(items) != domain:
        raise ValueError("Metadata must cover contiguous original item IDs 0..N-1")
    rows = []
    for original in range(len(items)):
        row = dict(items[str(original)])
        if "title" not in row or not isinstance(row["title"], str):
            raise MetadataUnavailable(f"Original item {original} has no string title")
        if categories:
            if "categories" not in row:
                raise MetadataUnavailable("Released category metadata is absent; SemID is unavailable")
            value = row["categories"]
            if not isinstance(value, str):
                raise MetadataUnavailable("Expected the released comma-serialized category path")
            path = value.split(",") if value else []
            if any(not label for label in path):
                raise MetadataUnavailable("Cannot unambiguously admit this released category path")
            # The release stores one path as a comma-joined string; P5 expects [path].
            row["categories"] = [path]
        rows.append(row)
    id2item = {str(i + 1): str(i) for i in range(len(rows))}
    meta_dict = {str(i): i for i in range(len(rows))}
    return rows, meta_dict, id2item


def validate_sequences(interactions, items):
    if not interactions or len({int(u) for u in interactions}) != len(interactions):
        raise ValueError("Users must be unique numeric original IDs")
    users = sorted(interactions, key=int)
    for user in users:
        seq = interactions[user]
        if len(seq) < 5 or any(type(i) is not int or not 0 <= i < items for i in seq):
            raise ValueError("Invalid released sequence or padding confused with item zero")
    return users


def training_graph(interactions, items):
    users = validate_sequences(interactions, items)
    pairs, occurrences = defaultdict(int), Counter()
    for user in users:
        train = interactions[user][:-2]
        occurrences.update(train)
        for pair in combinations(train, 2):
            pairs[pair] += 1
    ordered = sorted(pairs.items(), key=lambda pair: pair[1], reverse=True)
    edges = {}
    for (a, b), frequency in ordered:
        edges[a, b] = frequency
        edges[b, a] = frequency
    coordinates = list(edges)
    graph = sparse.csr_matrix((
        np.array([edges[p] for p in coordinates], dtype=np.float64),
        ([p[0] for p in coordinates], [p[1] for p in coordinates]),
    ), shape=(items, items), dtype=np.float64)
    graph.sort_indices()
    cold = sorted(set(range(items)) - set(occurrences))
    bidirectional = [
        (a, b, value, pairs[b, a])
        for (a, b), value in pairs.items() if a < b and (b, a) in pairs
    ]
    report = {
        "graph_policy": "native-descending-oriented-overwrite-full-catalog",
        "shape": list(graph.shape), "nnz": graph.nnz, "dtype": str(graph.dtype),
        "training_items": len(occurrences), "cold_original_ids": cold,
        "cold_zero_rows": bool(graph[cold].nnz == 0), "self_pairs": sum(a == b for a, b in pairs),
        "bidirectional_pairs": len(bidirectional),
        "unequal_reverse_counts": sum(x != y for _, _, x, y in bidirectional),
        "undirected_sum_difference_examples": [
            {"items": [a, b], "direction_counts": [x, y], "native_weight": edges[a, b],
             "not_used_undirected_sum": x + y} for a, b, x, y in bidirectional[:20]
        ],
        "native_matrix_size_would_be": len(occurrences),
        "maximum_fitting_original_id": max(occurrences),
        "fitting_sequences_sha256": digest_json([interactions[u][:-2] for u in users]),
    }
    return graph, ordered, report


def cid_mapping(native, interactions, items, output, width=20, leaf_size=500, max_depth=8):
    if min(items, leaf_size) < width:
        raise ValueError("Native spectral clustering requires enough catalog/cluster rows")
    graph, pairs, report = training_graph(interactions, items)
    started = time.monotonic()
    clustering = SpectralClustering(
        n_clusters=width, assign_labels="cluster_qr", random_state=0, affinity="precomputed",
    ).fit(graph)
    labels = clustering.labels_.tolist()
    paths = {i: [str(label)] for i, label in enumerate(labels)}
    helpers = cid_functions(native.source)
    calls = []

    def split(group_labels, group_pairs, reverse_functions, depth):
        for group in range(width):
            size = group_labels.count(group)
            if size <= leaf_size:
                continue
            if depth >= max_depth:
                raise ValueError("Native eight-level CID tree still has an oversized leaf")
            result = helpers.one_further_indexing(
                group, group_labels, group_pairs, width, paths, reverse_functions, depth + 1,
            )
            child_labels, child_pairs, _, child_reverse = result
            calls.append({"depth": depth + 1, "items": size, "clusters": len(set(child_labels))})
            print(f"CID_SPLIT depth={depth + 1} items={size}", flush=True)
            split(child_labels, child_pairs, child_reverse, depth + 1)

    split(labels, pairs, [lambda i: i], 1)
    mapping, vocabulary = native_cid_format(native.source, paths)
    if len(mapping) != items or len(set(mapping.values())) != items:
        raise ValueError("Native CID formatted leaves must uniquely identify the entire catalog")
    args = native_args("Instruments", "CID", items)
    args.data_dir, args.task = str(Path(output).resolve()) + "/", "native"
    folder = Path(output) / "native" / "CF_indices"
    write_json(folder / "c20_500_CF_index.json", paths)
    write_json(folder / "computed_20_500_CF_index.json", [mapping, vocabulary])
    loaded, full_vocab = native.CF_index.construct_indices_from_cluster(args)
    if loaded != mapping:
        raise ValueError("Native item-representation loader changed the CID mapping")
    codes = [native.item_rep_method.CF_representation(str(i + 1), loaded) for i in range(items)]
    report.update(
        elapsed_seconds=time.monotonic() - started, recursion=calls,
        max_depth=max(map(len, paths.values())), root_storage="value-equivalent float64 CSR",
        recursive_storage="unchanged native dense helper", root_labels=labels,
        max_leaf_size=max(Counter(tuple(v) for v in paths.values()).values()),
        code_format="CID_generation.py standard method 2 -> native CF_representation",
        cold_effect="Adding zero rows can also alter the warm-item spectral partition",
    )
    return codes, sorted(set(full_vocab)), report


def identifiers(native, dataset, representation, interactions, items, output):
    validate_sequences(interactions, len(items))
    args = native_args(dataset, representation, len(items))
    if representation == "CID":
        return cid_mapping(native, interactions, len(items), output)
    if representation == "SemID" and dataset != "Instruments":
        raise MetadataUnavailable(f"{dataset} SemID unavailable: release has no category metadata")
    rows, meta_dict, id2item = metadata_view(items, categories=representation == "SemID")
    if representation == "TID":
        codes = [
            native.item_rep_method.title_representation(str(i + 1), rows, meta_dict, id2item)
            for i in range(len(items))
        ]
        titles = [row["title"] for row in rows]
        counts = Counter(titles)
        return codes, [], {
            "raw_empty_titles": sum(not title for title in titles),
            "raw_duplicate_groups": sum(count > 1 for count in counts.values()),
            "raw_excess_collision_items": sum(count - 1 for count in counts.values()),
            "name_override_rows": sum("name" in row for row in rows),
            "title_transformation": "unchanged native title_representation; no cleaning or truncation",
        }
    before = random.getstate()
    try:
        random.seed(42)
        categories, levels = native.item_rep_method.build_category_map(
            args, rows, meta_dict, id2item,
        )
    finally:
        random.setstate(before)
    codes = [
        native.item_rep_method.content_based_representation(args, str(i + 1), categories, levels)
        for i in range(len(items))
    ]
    if any(not code for code in codes) or len(set(codes)) != len(items):
        raise ValueError("Native SemID normalization does not uniquely cover the released catalog")

    class VocabularyCollector:
        vocab = {}

        def add_tokens(self, values):
            self.tokens = sorted(set(values))

    collector = VocabularyCollector()
    native.utils.content_based_representation_non_hierarchical(args, categories, levels, collector)
    write_json(Path(output) / "category-map.json", {
        "p5_id_to_native_category_path_and_leaf": categories, "level_categories": levels,
        "original_i_to_p5_i_plus_one": True,
    })
    return codes, collector.tokens, {
        "category_source": "exact released comma-serialized Instruments category path",
        "adaptation": "split literal comma, wrap as [path]; native first-path/root removal",
        "roots": dict(Counter(row["categories"][0][0] for row in rows if row["categories"][0])),
        "seed": 42, "rows": len(items),
        "empty_released_paths": sum(not row["categories"][0] for row in rows),
        "root_only_paths": sum(len(row["categories"][0]) == 1 for row in rows),
        "native_item_id_fallback_rows": sum(len(row["categories"][0]) <= 1 for row in rows),
        "native_item_id_fallback_original_ids": [
            i for i, row in enumerate(rows) if len(row["categories"][0]) <= 1
        ],
        "fallback_semantics": "native numeric item-ID category token; not recovered taxonomy",
        "mapping_sha256": digest_json(categories),
        "metadata_not_original_amazon_category_file": True,
    }
