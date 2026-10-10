import torch

from rrbench.semantic_models import TextReranker


class Tokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, *documents, **kwargs):
        self.seen.append((texts, documents, kwargs))
        if kwargs.get("return_attention_mask") is False:
            return {"input_ids": [list(range(len(text.split()))) for text in texts]}
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def decode(self, values, **_):
        return " ".join("token" for _ in values)


def test_long_history_query_cannot_consume_entire_cross_encoder_document_budget():
    reranker = object.__new__(TextReranker)
    reranker.specification = {"max_tokens": 512}
    reranker.is_qwen = False
    reranker.tokenizer = Tokenizer()
    queries = ["word " * 1000]
    bounded = reranker.bound_queries(queries, query_token_cap=160)
    assert len(bounded[0].split()) == 160
    assert queries[0] != bounded[0]


def test_short_official_ir_query_is_not_changed_by_optional_history_guard():
    reranker = object.__new__(TextReranker)
    reranker.tokenizer = Tokenizer()
    queries = ["capital of China"]
    assert reranker.bound_queries(queries, query_token_cap=160) == queries
