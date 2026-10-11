import pytest

from rrbench.semantic_sampling import latency_query_rows, select_text_rows


def test_text_query_selection_does_not_use_judgments_and_preserves_all_trec_rows():
    panel = {
        "query_ids": ["dev:a", "dev:b", "dev:c", "trec:x"],
        "evaluation_groups": ["dev-small", "dev-small", "dev-small", "trec19"],
        "qrels": {"dev:a": {"1": 1}, "dev:b": {"2": 1}, "dev:c": {"3": 1}, "trec:x": {"4": 2}},
    }
    first = select_text_rows(panel, seed=17, maximum_dev=2)
    panel["qrels"] = {query: {} for query in panel["query_ids"]}
    assert select_text_rows(panel, seed=17, maximum_dev=2) == first
    assert 3 in first and len(first) == 3


def test_expensive_recommendation_text_panel_is_one_frozen_query_per_user():
    panel = {"query_ids": ["a", "b", "c", "d"], "users": [10, 10, 11, 12]}
    selected = select_text_rows(panel, seed=17)
    assert len(selected) == 3
    assert {panel["users"][index] for index in selected} == {10, 11, 12}
    assert selected == sorted(selected)


def test_sampling_requires_unique_query_identity_and_explicit_population():
    with pytest.raises(ValueError, match="unique"):
        select_text_rows({"query_ids": ["a", "a"], "users": [1, 2]})


def test_latency_query_batches_are_distinct_label_blind_and_shared_across_models():
    panel = {"query_ids": [str(index) for index in range(100)], "qrels": {}}
    batches = latency_query_rows(panel, 16, seed=17)
    assert len(batches) == 4
    assert all(len(batch) == 16 for batch in batches)
    assert len({row for batch in batches for row in batch}) == 64
    panel["qrels"] = {query: {"1": 3} for query in panel["query_ids"]}
    assert latency_query_rows(panel, 16, seed=17) == batches
