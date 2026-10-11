import pytest

from rrbench.semantic_tables import unique_quality_point


def test_result_table_lookup_requires_the_exact_stage_model_panel_and_seed():
    rows = [
        {"corpus": "msmarco", "encoder": "qwen8", "stage": "exact_full_catalog_retrieval", "dimension": 1024, "query_panel": "trec19", "ndcg@10": 0.8},
        {"corpus": "msmarco", "encoder": "qwen8", "stage": "exact_full_catalog_retrieval", "dimension": 4096, "query_panel": "trec19", "ndcg@10": 0.9},
    ]
    assert unique_quality_point(rows, corpus="msmarco", encoder="qwen8", dimension=1024,
                                stage="exact_full_catalog_retrieval", query_panel="trec19")["ndcg@10"] == 0.8
    with pytest.raises(ValueError, match="exactly one"):
        unique_quality_point(rows, corpus="msmarco", encoder="qwen8")


def test_missing_experiment_point_cannot_be_replaced_by_a_zero_default():
    with pytest.raises(ValueError, match="exactly one"):
        unique_quality_point([], corpus="kuairand")
