import json

import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_encoding import build_plan, merge_corpus, validate_completed, worker_tasks


def fixture_plan(tmp_path):
    corpus = tmp_path / "data" / "msmarco"
    corpus.mkdir(parents=True)
    shards = []
    for number, start in enumerate((0, 2, 4)):
        source = corpus / f"corpus_{number:03d}.tsv"
        source.write_text(f"{start}\tfirst\n{start + 1}\tsecond\n", encoding="utf-8")
        shards.append({"file": source.name, "start": start, "rows": 2, "sha256": sha256(source)})
    write_json(corpus / "corpus.json", {"rows": 6, "shards": shards})
    registry = {
        "models": {
            "qwen8": {"key": "qwen8", "corpora": ["msmarco"], "dimensions": [4096, 1024]},
            "qwen06": {"key": "qwen06", "corpora": ["msmarco"], "dimensions": [1024]},
            "rerank": {"key": "rerank"},
        }
    }
    return build_plan(registry, {"msmarco": corpus}, tmp_path / "embeddings", 4)


def test_finite_plan_covers_each_model_and_real_shard_once(tmp_path):
    plan = fixture_plan(tmp_path)
    tasks = [task for worker in range(4) for task in worker_tasks(plan, worker)]
    assert len(tasks) == 6
    assert len({task["key"] for task in tasks}) == 6
    assert sum(task["rows"] for task in tasks) == 12
    assert {task["model"] for task in tasks} == {"qwen8", "qwen06"}
    with pytest.raises(ValueError, match="worker"):
        worker_tasks(plan, 4)


def test_plan_rejects_changed_text_shard_before_expensive_encoding(tmp_path):
    fixture_plan(tmp_path)
    source = tmp_path / "data" / "msmarco" / "corpus_000.tsv"
    source.write_text("0\tchanged\n", encoding="utf-8")
    registry = {"models": {"qwen8": {"corpora": ["msmarco"], "dimensions": [4096, 1024]}}}
    with pytest.raises(ValueError, match="checksum"):
        build_plan(registry, {"msmarco": source.parent}, tmp_path / "out", 4)


def test_first_large_model_wave_is_balanced_across_nodes_not_concentrated_on_node_zero(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shards = []
    for number in range(34):
        source = corpus / f"corpus_{number:03d}.tsv"
        source.write_text(f"{number}\treal\n", encoding="utf-8")
        shards.append({"file": source.name, "start": number, "rows": 1, "sha256": sha256(source)})
    write_json(corpus / "corpus.json", {"rows": 34, "shards": shards})
    registry = {"models": {"qwen8": {"corpora": ["msmarco"], "dimensions": [4096, 1024]}}}
    plan = build_plan(registry, {"msmarco": corpus}, tmp_path / "out", 24)
    counts = [sum(task["worker"] // 8 == node for task in plan["tasks"]) for node in range(3)]
    assert max(counts) - min(counts) <= 1


def write_encoded(directory, ids, vectors):
    directory.mkdir(parents=True)
    np.save(directory / "ids.npy", np.asarray(ids, dtype=np.int64))
    np.save(directory / "vectors.npy", np.asarray(vectors, dtype=np.float16))
    write_json(directory / "completed.json", {
        "complete": True, "rows": len(ids), "native_dimension": 2,
        "ids_sha256": sha256(directory / "ids.npy"),
        "vectors_sha256": sha256(directory / "vectors.npy"), "mrl_representations": {},
    })


def test_completion_checks_ids_not_only_vector_bytes(tmp_path):
    write_encoded(tmp_path / "encoded", [7, 8], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="IDs"):
        validate_completed(tmp_path / "encoded", 2, 0)


def test_merge_refuses_a_missing_shard_and_preserves_global_row_mapping(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_encoded(first, [0, 1], [[1, 0], [0, 1]])
    tasks = [
        {"output": str(first), "start": 0, "rows": 2},
        {"output": str(second), "start": 2, "rows": 2},
    ]
    with pytest.raises(FileNotFoundError):
        merge_corpus(tasks, tmp_path / "merged", 2)
    assert not (tmp_path / "merged" / "completed.json").exists()
    write_encoded(second, [2, 3], [[0.6, 0.8], [0.8, 0.6]])
    result = merge_corpus(tasks, tmp_path / "merged", 2)
    assert result["rows"] == 4
    np.testing.assert_array_equal(np.load(tmp_path / "merged" / "ids.npy"), np.arange(4))
    np.testing.assert_array_equal(
        np.load(tmp_path / "merged" / "vectors.npy"),
        np.asarray([[1, 0], [0, 1], [0.6, 0.8], [0.8, 0.6]], dtype=np.float16),
    )
    assert json.loads((tmp_path / "merged" / "completed.json").read_text())["complete"]


def test_merged_mrl_matrix_width_is_not_relabelled_as_the_encoder_native_dimension(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "ids.npy", np.array([0], dtype=np.int64))
    np.save(source / "vectors.npy", np.array([[1, 0, 0, 0]], dtype=np.float16))
    np.save(source / "vectors_mrl2.npy", np.array([[1, 0]], dtype=np.float16))
    write_json(source / "completed.json", {
        "complete": True, "rows": 1, "native_dimension": 4,
        "ids_sha256": sha256(source / "ids.npy"), "vectors_sha256": sha256(source / "vectors.npy"),
        "mrl_representations": {"2": {"file": "vectors_mrl2.npy", "dimension": 2, "sha256": sha256(source / "vectors_mrl2.npy")}},
    })
    result = merge_corpus([{"output": str(source), "start": 0, "rows": 1}], tmp_path / "merged", 2)
    assert result["native_dimension"] == 4
    assert result["stored_dimension"] == 2
    assert validate_completed(tmp_path / "merged", 1, 0)["stored_dimension"] == 2
