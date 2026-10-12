from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class Expression:
    operator: str
    feature: int = 0
    value: int = 0
    children: tuple[Expression, ...] = ()

    def render(self) -> str:
        if self.operator == "TERM":
            return f"{self.feature}:{self.value}"
        if self.operator == "NOT":
            return f"NOT ({self.children[0].render()})"
        if self.operator == "TRUE":
            return ""
        return f"({self.children[0].render()} {self.operator} {self.children[1].render()})"


@lru_cache(maxsize=1024)
def parse_expression(text: str) -> Expression:
    token_pattern = re.compile(r"\s*(\d+:\d+|AND\b|OR\b|NOT\b|\(|\))")
    tokens = []
    position = 0
    while position < len(text):
        match = token_pattern.match(text, position)
        if not match:
            if not text[position:].strip():
                break
            raise ValueError(f"Invalid filter syntax at position {position}.")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        return Expression("TRUE")
    cursor = 0

    def unary() -> Expression:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Unexpected end of filter.")
        token = tokens[cursor]
        cursor += 1
        if token == "NOT":
            return Expression("NOT", children=(unary(),))
        if token == "(":
            result = disjunction()
            if cursor >= len(tokens) or tokens[cursor] != ")":
                raise ValueError("Unmatched filter parenthesis.")
            cursor += 1
            return result
        if ":" not in token:
            raise ValueError(f"Expected a filter term, got {token!r}.")
        feature, value = map(int, token.split(":"))
        return Expression("TERM", feature, value)

    def conjunction() -> Expression:
        nonlocal cursor
        result = unary()
        while cursor < len(tokens) and tokens[cursor] == "AND":
            cursor += 1
            result = Expression("AND", children=(result, unary()))
        return result

    def disjunction() -> Expression:
        nonlocal cursor
        result = conjunction()
        while cursor < len(tokens) and tokens[cursor] == "OR":
            cursor += 1
            result = Expression("OR", children=(result, conjunction()))
        return result

    result = disjunction()
    if cursor != len(tokens):
        raise ValueError("Unexpected trailing filter tokens.")
    return result


def conservative_positive(expression: Expression, negate: bool = False) -> Expression:
    if expression.operator == "TRUE":
        return Expression("TRUE")
    if expression.operator == "NOT":
        return conservative_positive(expression.children[0], not negate)
    if expression.operator == "TERM":
        return Expression("TRUE") if negate else expression
    operator = expression.operator
    if negate:
        operator = "OR" if operator == "AND" else "AND"
    left, right = (conservative_positive(child, negate) for child in expression.children)
    if operator == "OR" and "TRUE" in (left.operator, right.operator):
        return Expression("TRUE")
    if operator == "AND":
        if left.operator == "TRUE":
            return right
        if right.operator == "TRUE":
            return left
    return Expression(operator, children=(left, right))


def evaluate(expression: Expression, features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("Features must end in [feature, values_per_feature] dimensions.")
    if expression.operator == "TRUE":
        return torch.ones(features.shape[:-2], dtype=torch.bool, device=features.device)
    if expression.operator == "TERM":
        column = expression.feature - 1
        if column < 0 or column >= features.shape[-2]:
            return torch.zeros(features.shape[:-2], dtype=torch.bool, device=features.device)
        return (features[..., column, :] == expression.value).any(dim=-1)
    if expression.operator == "NOT":
        return ~evaluate(expression.children[0], features)
    left, right = (evaluate(child, features) for child in expression.children)
    return left & right if expression.operator == "AND" else left | right


def exact_candidate_mask(features: torch.Tensor, expressions: list[str]) -> torch.Tensor:
    if features.ndim != 4 or len(features) != len(expressions):
        raise ValueError("Candidate features must be [queries, candidates, features, values].")
    groups = {}
    for row, text in enumerate(expressions):
        groups.setdefault(text, []).append(row)
    if len(groups) == 1:
        return evaluate(parse_expression(expressions[0]), features)
    result = torch.empty(features.shape[:2], device=features.device, dtype=torch.bool)
    for text, rows in groups.items():
        indices = torch.tensor(rows, device=features.device)
        result[indices] = evaluate(parse_expression(text), features[indices])
    return result


def make_features(count: int, seed: int, feature_count: int = 6, values_per_feature: int = 2) -> np.ndarray:
    if min(count, feature_count, values_per_feature) <= 0:
        raise ValueError("Feature dimensions must be positive.")
    rng = np.random.default_rng(seed)
    result = rng.integers(1, 65, (count, feature_count, values_per_feature), dtype=np.int64)
    return np.sort(result, axis=-1)


class BloomFilter:
    def __init__(
        self, features: torch.Tensor, k: int = 5, hash_k: int = 7, multiplier: float = 5.0
    ) -> None:
        import silvertorch.ops._load_ops  # noqa: F401

        if features.ndim != 3 or features.dtype != torch.long:
            raise ValueError("Bloom features must be int64 [items, features, values].")
        if not 1 <= k <= hash_k or multiplier <= 1.0:
            raise ValueError("Require 1 <= k <= hash_k and multiplier > 1.0.")
        self.features = features
        self.k, self.hash_k = k, hash_k
        count, feature_count, value_count = features.shape
        feature_ids = torch.arange(1, feature_count + 1, dtype=torch.int32, device=features.device)
        offsets = torch.arange(
            0, (count * feature_count + 1) * value_count, value_count,
            dtype=torch.long, device=features.device
        )
        self.index, self.offsets = torch.ops.st.bloom_index_build(
            feature_ids, offsets, features.reshape(-1).contiguous(), multiplier, k
        )

    def plans(self, texts: list[str], conservative: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        expressions = [parse_expression(text) for text in texts]
        if conservative:
            expressions = [conservative_positive(expression) for expression in expressions]
        query_texts = [expression.render() for expression in expressions]
        _, plans = torch.ops.st.parse_expression_query_batch(
            query_texts, torch.full((len(texts),), self.k, dtype=torch.long),
            self.hash_k, True, 64
        )
        return tuple(t.to(self.features.device) for t in plans)

    def search(self, plans: tuple[torch.Tensor, torch.Tensor], packed: bool = False) -> torch.Tensor:
        result = torch.ops.st.bloom_index_search_batch(
            self.index, self.offsets, *plans, self.k, self.hash_k, not packed
        )
        return result if packed else result[:, : len(self.features)]

    def partial(
        self, plans: tuple[torch.Tensor, torch.Tensor], offsets: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.st.bloom_index_search_batch_return_partial_response(
            self.index, self.offsets, *plans, offsets.contiguous(), lengths.contiguous(),
            self.k, self.hash_k
        )


class InvertedFilter:
    """Exact sorted-posting CPU reference; not Meta's internal inverted index."""

    def __init__(self, features: np.ndarray) -> None:
        if features.ndim != 3:
            raise ValueError("Inverted-index features must be [items, features, values].")
        self.count = len(features)
        self.postings = {}
        for feature in range(features.shape[1]):
            flat = features[:, feature, :].reshape(-1)
            order = np.argsort(flat, kind="stable")
            sorted_values = flat[order]
            values, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
            for value, start, count in zip(values, starts, counts):
                self.postings[(feature + 1, int(value))] = np.unique(
                    order[start : start + count] // features.shape[2]
                )
        self.universe = np.arange(self.count, dtype=np.int64)

    def ids(self, expression: Expression) -> np.ndarray:
        if expression.operator == "TRUE":
            return self.universe
        if expression.operator == "TERM":
            return self.postings.get((expression.feature, expression.value), np.empty(0, dtype=np.int64))
        if expression.operator == "NOT":
            return np.setdiff1d(self.universe, self.ids(expression.children[0]), assume_unique=True)
        left, right = (self.ids(child) for child in expression.children)
        return np.intersect1d(left, right, assume_unique=True) if expression.operator == "AND" else np.union1d(left, right)

    def search(self, expression: Expression) -> np.ndarray:
        mask = np.zeros(self.count, dtype=np.bool_)
        mask[self.ids(expression)] = True
        return mask
