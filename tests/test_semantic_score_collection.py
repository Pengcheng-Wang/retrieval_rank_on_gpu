import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_cascade import collect_text_scores


def fixture(tmp_path, source_mismatch=False, score_dtype=np.float32):
    panel, candidates = tmp_path / "panel.json", tmp_path / "candidates.npy"
    write_json(panel, {
        "query_ids": ["q1", "q2"], "queries": ["one", "two"],
        "qrels": {"q1": {"0": 1}, "q2": {"1": 1}}, "binary_threshold": 1,
    })
    np.save(candidates, np.tile(np.arange(10, dtype=np.int64), (2, 1)))
    shards = []
    for worker in range(2):
        shard = tmp_path / str(worker)
        shard.mkdir()
        np.save(shard / "query_rows.npy", np.array([worker], dtype=np.int64))
        np.save(shard / "scores.npy", np.arange(10, dtype=score_dtype)[None])
        write_json(shard / "completed.json", {
            "complete": True, "rows": 1,
            "binding": {
                "panel_sha256": sha256(panel), "candidates_sha256": sha256(candidates),
                "specification_sha256": "model", "worker": worker, "workers": 2,
                "source_sha256": "source-different" if source_mismatch and worker == 1 else "source",
                "offsets_sha256": "offsets", "candidate_prefix": 10,
            },
            "scores_sha256": sha256(shard / "scores.npy"), "query_rows_sha256": sha256(shard / "query_rows.npy"),
        })
        shards.append(shard)
    return panel, candidates, shards


def test_different_corpus_or_offset_bindings_cannot_be_combined_into_one_quality_result(tmp_path):
    panel, candidates, shards = fixture(tmp_path, source_mismatch=True)
    with pytest.raises(ValueError, match="protocol"):
        collect_text_scores(panel, candidates, shards, tmp_path / "out")
    assert not (tmp_path / "out" / "completed.json").exists()


def test_integer_scores_cannot_be_silently_converted_into_verified_fp32_model_outputs(tmp_path):
    panel, candidates, shards = fixture(tmp_path, score_dtype=np.int64)
    with pytest.raises(ValueError, match="dtype"):
        collect_text_scores(panel, candidates, shards, tmp_path / "out")


def test_complete_fp32_shards_preserve_exact_query_rows_on_collection(tmp_path):
    panel, candidates, shards = fixture(tmp_path)
    result = collect_text_scores(panel, candidates, shards, tmp_path / "out")
    assert result["complete"] and result["rows"] == 2
    np.testing.assert_array_equal(np.load(tmp_path / "out" / "scores.npy"), np.tile(np.arange(10), (2, 1)))
