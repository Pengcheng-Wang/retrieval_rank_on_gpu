import numpy as np
import pytest

from rrbench.io import sha256, write_json
from rrbench.semantic_profiles import (
    prepare_profiles, training_positive_profiles, validate_profile_inventory, validate_profile_user_map,
    validate_profile_item_bank,
)


def test_static_semantic_profiles_use_only_positive_training_rows_with_repeated_event_weights():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float16)
    pairs = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 2, 0], [1, 2, 0]], dtype=np.int64)
    profiles, supported, counts = training_positive_profiles(pairs, vectors, users=2)
    assert supported.tolist() == [0]
    assert counts.tolist() == [3, 0]
    np.testing.assert_allclose(profiles[0].astype(np.float32), np.array([2, 1]) / np.sqrt(5), atol=0.001)
    assert profiles.dtype == np.float16


def test_bad_training_user_or_zero_norm_profile_is_not_silently_replaced():
    with pytest.raises(ValueError, match="user"):
        training_positive_profiles(np.array([[3, 0, 1]], dtype=np.int64), np.eye(2, dtype=np.float16), 1)
    with pytest.raises(ValueError, match="norm"):
        training_positive_profiles(
            np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64),
            np.array([[1, 0], [-1, 0]], dtype=np.float16), 1,
        )


def test_profile_user_map_cannot_turn_broken_mapping_into_positive_support_attrition():
    with pytest.raises(ValueError, match="bijection"):
        validate_profile_user_map({"10": 0, "11": 99}, 2)
    with pytest.raises(ValueError, match="bijection"):
        validate_profile_user_map({"10": 0, "11": 0}, 2)


def test_profile_item_bank_must_have_the_actual_admitted_shape_not_only_declared_dimensions(tmp_path):
    items = tmp_path / "items"
    items.mkdir()
    np.save(items / "vectors.npy", np.eye(2, dtype=np.float16))
    write_json(items / "completed.json", {
        "complete": True, "rows": 2, "stored_dimension": 1024,
        "vectors_sha256": sha256(items / "vectors.npy"),
    })
    ann = tmp_path / "ann"
    ann.mkdir()
    write_json(ann / "run-config.json", {"items_sha256": sha256(items / "vectors.npy")})
    with pytest.raises(ValueError, match="shape"):
        validate_profile_item_bank(items / "vectors.npy", ann, 2)


def test_incomplete_profile_receipt_with_empty_phase_set_cannot_be_reused(tmp_path):
    write_json(tmp_path / "completed.json", {"complete": False, "panels": {}})
    with pytest.raises(ValueError, match="complete"):
        validate_profile_inventory(tmp_path, {}, users=2, dimension=2)


def test_valid_profile_preparation_preserves_retained_query_ids_and_original_qrels(tmp_path):
    data, items, ann, panels = (tmp_path / name for name in ("data", "items", "ann", "panels"))
    for directory in (data, items, ann, panels):
        directory.mkdir()
    np.save(data / "pairs.npy", np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.int64))
    write_json(data / "users.json", {"10": 0, "11": 1})
    np.save(data / "training-positive-popularity.npy", np.array([2, 0, 0], dtype=np.int64))
    write_json(data / "completed.json", {
        "complete": True, "pairs": 3, "users": 2, "items": 3, "positives": 2,
        "pairs_sha256": sha256(data / "pairs.npy"), "users_sha256": sha256(data / "users.json"),
        "positive_popularity_sha256": sha256(data / "training-positive-popularity.npy"),
        "input_contract": {"catalog_sha256": "catalog", "catalog_ids_sha256": "ids"},
    })
    matrix = np.zeros((3, 1024), dtype=np.float16)
    matrix[:, 0] = 1
    np.save(items / "vectors.npy", matrix)
    np.save(items / "ids.npy", np.arange(3, dtype=np.int64))
    write_json(items / "completed.json", {
        "complete": True, "rows": 3, "native_dimension": 4096, "stored_dimension": 1024,
        "vectors_sha256": sha256(items / "vectors.npy"), "ids_sha256": sha256(items / "ids.npy"),
        "mrl_representations": {}, "representation": {"model": "Qwen/Qwen3-Embedding-8B"},
    })
    write_json(ann / "run-config.json", {"items_sha256": sha256(items / "vectors.npy")})
    for phase in ("validation", "test"):
        write_json(panels / (phase + ".json"), {
            "queries": ["first", "second"], "query_ids": ["a", "b"], "users": [10, 11],
            "qrels": {"a": {"2": 1}, "b": {"1": 1}}, "catalog_sha256": "catalog", "catalog_ids_sha256": "ids",
        })
    output = tmp_path / "out"
    prepare_profiles(data, items / "vectors.npy", panels, output, ann)
    import json

    measured = json.loads((output / "test.json").read_text())
    assert measured["query_ids"] == ["a"]
    assert measured["qrels"] == {"a": {"2": 1}}
    assert measured["original_query_rows"] == [0]
    assert "queries" not in measured
    prepare_profiles(data, items / "vectors.npy", panels, output, ann)
