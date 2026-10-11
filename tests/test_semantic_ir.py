import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_ir import PassageStore, judged_metrics, split_fitting_queries
from rrbench.semantic_ir import panel_metrics


def test_graded_ndcg_and_binary_recall_use_distinct_trec_relevance_contracts():
    qrels = {"q": {7: 3, 8: 1, 9: 2}}
    result = judged_metrics(["q"], np.array([[8, 7, 9, -1]]), qrels, binary_threshold=2)
    assert result["recall@1"] == 0
    assert result["recall@10"] == 1
    assert result["mrr@10"] == 0.5
    assert 0 < result["ndcg@10"] < 1
    assert result["ndcg_gain"] == "Raw graded relevance, matching default trec_eval gain."


def test_duplicate_candidates_do_not_inflate_relevance_gain():
    with pytest.raises(ValueError, match="duplicate"):
        judged_metrics(["q"], np.array([[7, 7]]), {"q": {7: 3}}, 2)


def test_fitting_query_groups_are_frozen_and_exclude_final_query_ids_and_text():
    queries = {"0": "same", "1": "other", "2": "same", "3": "heldout words", "4": "third", "5": "fourth"}
    first = split_fitting_queries(queries, {"3"}, {"heldout words"}, 2, 1, 17)
    second = split_fitting_queries(dict(reversed(list(queries.items()))), {"3"}, {"heldout words"}, 2, 1, 17)
    assert first == second
    train, validation = first
    assert len(train) == 2 and len(validation) == 1
    assert not set(train) & set(validation)
    assert "3" not in train + validation
    assert len({queries[identifier] for identifier in train + validation}) == 3


def test_random_access_preserves_passage_id_mapping_and_rejects_corrupt_offset(tmp_path):
    source = tmp_path / "catalog.tsv"
    lines = [b"0\tfirst\n", b"1\tsecond\n", b"2\tthird\n"]
    source.write_bytes(b"".join(lines))
    offsets = tmp_path / "offsets.npy"
    np.save(offsets, np.cumsum([0] + [len(line) for line in lines], dtype=np.uint64))
    store = PassageStore(source, offsets)
    assert store.fetch(np.array([2, 0])) == ["third", "first"]
    with pytest.raises(ValueError, match="ID"):
        store.fetch(np.array([-1]))
    del store
    np.save(offsets, np.array([0, 0, len(lines[0]) + len(lines[1]), sum(map(len, lines))], dtype=np.uint64))
    with pytest.raises(ValueError, match="offset"):
        PassageStore(source, offsets)


def test_combined_final_panel_keeps_msmarco_and_trec_binary_thresholds_separate():
    panel = {
        "query_ids": ["dev-small:q", "trec19:q"], "qrels": {
            "dev-small:q": {"7": 1}, "trec19:q": {"7": 1, "9": 2},
        },
        "evaluation_groups": ["dev-small", "trec19"],
        "threshold_by_group": {"dev-small": 1, "trec19": 2}, "binary_threshold": 1,
    }
    result = panel_metrics(panel, np.array([[7, 9], [7, 9]], dtype=np.int64))
    assert result["by_group"]["dev-small"]["mrr@10"] == 1
    assert result["by_group"]["trec19"]["mrr@10"] == 0.5
    assert result["by_group"]["trec19"]["binary_relevance_threshold"] == 2
