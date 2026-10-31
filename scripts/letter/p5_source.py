"""Verified, private native P5 code; no author source or weights are vendored."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


COMMIT = "bc9a4b1bcae36d58d304f7ddaaa5fbe5e1be1ca0"
ARCHIVE_SHA256 = "3e64c60b8333f6e97f67452ba55ef5eca2485e01290aa4f6419445bde6426e34"
SOURCE_HASHES = {
    "main.py": "3d2feb421bd23615fd9fd1697d4970e63b0cd94a0c4fd2f9b1238fefe3ec7bd5",
    "data.py": "17d9a40e8595d5adfa2ea06b4ed9edc7ca9a391d9f696cb2a57e004a72de17f9",
    "item_rep_method.py": "2eeefc5d7db7888fe8344e7300c8dc61e82a67ce471fdbed869a57bde87ae674",
    "modeling_p5.py": "742f58b95ba9f187aeb012d77b0a3775af4bf27083325e959bc28d6c4139ef6d",
    "prompt.py": "367a221ad8846f1a35ced8c89a6bbf7117324c909ba9acbb07c3da2dc5eb02af",
    "utils.py": "d795b034549d211149c6ecaefe664be6166d351e666422e90ce49279265f9539",
    "generation_trie.py": "04afa617e905bb69c0051c1ccef548514298ce5a68e1df32c2c774dc6b06351f",
    "CF_index.py": "db993c147f51d8775109da3584f79a22d35ccb86a46682e480539e73a2089193",
    "CID_generation.py": "b41b4eb7cfc11f648531fa2ecc6cf3b2b17da49415be936de3441fa097ddd119",
    "requirements.txt": "3d62c94488fa23228f8530d6cf7512793bf66e2073072182d786e7ad7b6eb4ba",
    "README.md": "c10451a07f38bcc3de082cc9b18233132c3760a8346b59779871e1ffc91df1e1",
}
MODEL_ID = "google-t5/t5-small"
MODEL_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
MODEL_WEIGHTS_SHA256 = "bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc"
REPRESENTATIONS = {"TID": "title", "CID": "CF", "SemID": "content_based"}
RUNTIME = {
    "torch": "1.13.1+cu117", "transformers": "4.26.0", "numpy": "1.24.4",
    "scipy": "1.10.1", "scikit-learn": "1.2.2", "tokenizers": "0.13.3",
    "safetensors": "0.3.3",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"P5 input checksum mismatch: {path}")


def verify_native(source):
    source = Path(source).resolve()
    for name, digest in SOURCE_HASHES.items():
        require_hash(source / name, digest)
    return source


def runtime_versions():
    import importlib.metadata

    versions = {name: importlib.metadata.version(name) for name in RUNTIME}
    if versions != RUNTIME:
        raise RuntimeError(f"Use the independently locked P5 environment: {versions}")
    return versions


def load_native(source):
    source = verify_native(source)
    names = ("CF_index", "utils", "item_rep_method", "prompt", "data",
             "generation_trie", "modeling_p5")
    for name in names:
        if name in sys.modules:
            if Path(getattr(sys.modules[name], "__file__", "")).parent != source:
                raise RuntimeError(f"Native module collision: {name}; use a fresh process")
    sys.path.insert(0, str(source))
    loaded = {}
    for name in names:
        if name in sys.modules:
            loaded[name] = sys.modules[name]
            continue
        path = source / (name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if name == "data":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            removed = 0
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "item_rep_method":
                    kept = [alias for alias in node.names if alias.name != "change_base"]
                    removed += len(node.names) - len(kept)
                    node.names = kept
            if removed != 1:
                raise RuntimeError("Unexpected unused native change_base import")
            exec(compile(tree, str(path), "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(source=source, **loaded)


def cid_functions(source):
    """Load only the verified fixed-width helper, not the script's data/fit side effects."""
    import numpy as np
    from sklearn.cluster import SpectralClustering

    source = verify_native(source)
    path = source / "CID_generation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"composite_function", "one_further_indexing"}
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise RuntimeError("Native fixed-width clustering helper is missing")
    namespace = {"np": np, "SpectralClustering": SpectralClustering}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def native_cid_format(source, labels):
    """Execute standard CID_generation method 2's actual serialization block."""
    path = verify_native(source) / "CID_generation.py"
    text = path.read_text(encoding="utf-8")
    text = text[text.index("###### apply indexing method 2,"):]
    tree = ast.parse(text, filename=str(path))
    selected, started = [], False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "reformed_item_CF_index_map"
            for t in node.targets
        ):
            started = True
        if started and isinstance(node, ast.With):
            break
        if started:
            selected.append(node)
    if not selected:
        raise RuntimeError("Native CID formatting block is missing")
    namespace = {"data": {str(i): value for i, value in labels.items()}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["full_index"], namespace["vocab"]


def native_args(dataset, representation, items):
    return SimpleNamespace(
        task=dataset.lower(), item_representation=REPRESENTATIONS[representation],
        number_of_items=items + 1, max_history=20, whole_word_embedding="shijie",
        evaluation_template_id=0, remove_last_item=False, remove_first_item=False,
        cluster_number=20, cluster_size=500, last_token_no_repetition=False,
        optimal_width_in_CF=False, data_order="remapped_sequential",
    )


def load_recipe(path):
    recipe = read_json(path)
    if (recipe.get("schema_version") != 1 or recipe.get("status") !=
            "declared-reference-reconstruction-not-recovered-letter-baseline"):
        raise ValueError("Not the declared P5 reference recipe")
    if recipe["source"]["commit"] != COMMIT or recipe["source"]["files_sha256"] != SOURCE_HASHES:
        raise ValueError("P5 source pin differs")
    if recipe["model"] != {
        "id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0",
        "weights_sha256": MODEL_WEIGHTS_SHA256, "initialization": "pretrained-native-P5",
    } or recipe["runtime"] != RUNTIME:
        raise ValueError("P5 model/runtime pin differs")
    train = recipe["training"]
    if (train["epochs"], train["effective_batch_size"], train["learning_rate"],
            train["warmup_ratio"], train["weight_decay"], train["adam_epsilon"],
            train["clip_norm"], train["seed"]) != (20, 128, .001, .05, .01, 1e-6, 1., 42):
        raise ValueError("Frozen source-justified training recipe differs")
    if recipe["indexing"]["graph_policy"] != "native-descending-oriented-overwrite-full-catalog":
        raise ValueError("The graph policy must be declared before fitting")
    if (recipe["evaluation"]["beams"] != 20 or recipe["evaluation"]["returns"] != 20
            or recipe["evaluation"]["seen_item_exclusion"] is not False):
        raise ValueError("Do not reduce the full-catalog beam/evaluation contract")
    return recipe
