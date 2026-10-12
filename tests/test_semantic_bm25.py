import numpy as np

from rrbench.io import sha256, write_json
from rrbench.semantic_bm25 import build_bm25, search_bm25


def test_real_bm25_full_catalog_lookup_is_label_independent_and_returns_external_ids(tmp_path):
    source = tmp_path / "collection.tsv"
    source.write_text("0\tBeijing is the capital of China\n1\tgravity attracts mass\n2\tChina has Beijing as its capital\n", encoding="utf-8")
    index = tmp_path / "index"
    manifest = build_bm25(source, index, expected_rows=3)
    assert manifest["catalog_items"] == 3
    result = search_bm25(index, ["capital China", 'not-present OR "operator'], 3)
    assert set(result[0, :2]) == {0, 2}
    assert result[1].tolist() == [-1, -1, -1]
