import numpy as np
import pytest

from scripts.letter.generation_bias import (
    analyze_code_distribution,
    bootstrap_mean_ci,
    fractional_rank_bins,
    generated_user_shares,
    midrank_bins,
    target_code_share,
    target_token_counts,
)


def test_fractional_rank_bins_splits_a_tie_across_boundaries():
    memberships = fractional_rank_bins(
        {"a": 10, "b": 5, "c": 5, "d": 1, "e": 0},
        bins=5,
    )

    assert memberships["a"] == pytest.approx([1, 0, 0, 0, 0])
    assert memberships["b"] == pytest.approx([0, 0.5, 0.5, 0, 0])
    assert memberships["c"] == pytest.approx([0, 0.5, 0.5, 0, 0])
    assert memberships["d"] == pytest.approx([0, 0, 0, 1, 0])
    assert memberships["e"] == pytest.approx([0, 0, 0, 0, 1])


def test_fractional_rank_bins_does_not_break_an_all_equal_tie_arbitrarily():
    memberships = fractional_rank_bins({"a": 0, "b": 0, "c": 0, "d": 0, "e": 0})

    for weights in memberships.values():
        assert weights == pytest.approx([0.2] * 5)


def test_target_code_share_normalizes_target_occurrences_and_code_positions():
    item_codes = {
        0: ("a0", "b0"),
        1: ("a1", "b0"),
    }
    memberships = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([0.0, 1.0]),
        "b0": np.array([0.5, 0.5]),
    }

    share = target_code_share([0, 0, 1], item_codes, memberships)

    assert share == pytest.approx([7 / 12, 5 / 12])
    assert share.sum() == pytest.approx(1.0)


def test_target_token_counts_can_limit_the_analysis_to_one_code_level():
    item_codes = {
        0: ("a0", "b0"),
        1: ("a1", "b0"),
    }

    assert target_token_counts([0, 0, 1], item_codes) == {
        "a0": 2,
        "a1": 1,
        "b0": 3,
    }
    assert target_token_counts([0, 0, 1], item_codes, levels={0}) == {
        "a0": 2,
        "a1": 1,
    }


def test_generated_user_shares_gives_each_user_beam_and_position_equal_mass():
    code_tokens = {
        "x": ("a0", "b0"),
        "y": ("a1", "b0"),
    }
    memberships = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([0.0, 1.0]),
        "b0": np.array([0.5, 0.5]),
    }
    beams = [["x", "y"], ["y", "y"]]

    shares = generated_user_shares(
        beams,
        code_tokens,
        memberships,
        top_k=2,
        rank_weight="uniform",
    )

    assert shares[0] == pytest.approx([0.5, 0.5])
    assert shares[1] == pytest.approx([0.25, 0.75])
    assert shares.mean(axis=0) == pytest.approx([0.375, 0.625])
    assert shares.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_generated_user_shares_supports_reciprocal_rank_sensitivity():
    code_tokens = {
        "popular": ("a0",),
        "tail": ("a1",),
    }
    memberships = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([0.0, 1.0]),
    }

    shares = generated_user_shares(
        [["popular", "tail"]],
        code_tokens,
        memberships,
        top_k=2,
        rank_weight="reciprocal",
    )

    assert shares[0] == pytest.approx([2 / 3, 1 / 3])


def test_generated_user_shares_supports_dcg_rank_sensitivity():
    code_tokens = {
        "popular": ("a0",),
        "tail": ("a1",),
    }
    memberships = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([0.0, 1.0]),
    }

    shares = generated_user_shares(
        [["popular", "tail"]],
        code_tokens,
        memberships,
        top_k=2,
        rank_weight="dcg",
    )

    expected_first = 1 / (1 + 1 / np.log2(3))
    assert shares[0] == pytest.approx([expected_first, 1 - expected_first])


def test_generated_user_shares_can_limit_each_beam_to_one_code_level():
    code_tokens = {"x": ("a0", "b0")}
    memberships = {
        "a0": np.array([1.0, 0.0]),
        "b0": np.array([0.0, 1.0]),
    }

    shares = generated_user_shares(
        [["x"]],
        code_tokens,
        memberships,
        top_k=1,
        levels={0},
    )

    assert shares[0] == pytest.approx([1.0, 0.0])


def test_midrank_bins_keeps_a_tie_together():
    memberships = midrank_bins({"a": 10, "b": 5, "c": 5, "d": 1, "e": 0}, bins=5)

    assert memberships["b"] == pytest.approx(memberships["c"])
    assert memberships["b"] == pytest.approx([0, 0, 1, 0, 0])


def test_bootstrap_mean_ci_is_exact_for_constant_user_shares():
    estimate = bootstrap_mean_ci(
        np.array([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]),
        replicates=20,
        seed=7,
    )

    assert estimate["mean"] == pytest.approx([0.75, 0.25])
    assert estimate["low"] == pytest.approx([0.75, 0.25])
    assert estimate["high"] == pytest.approx([0.75, 0.25])


def test_analyze_code_distribution_reports_training_generation_shift_and_ratio():
    item_codes = {
        0: ("a0", "b0"),
        1: ("a1", "b1"),
    }
    code_tokens = {
        "x": item_codes[0],
        "y": item_codes[1],
    }

    result = analyze_code_distribution(
        target_items=[0, 0, 1],
        item_codes=item_codes,
        beams_by_user=[["x", "y"], ["y", "y"]],
        code_tokens=code_tokens,
        bins=2,
        top_k=2,
        rank_weight="uniform",
        bootstrap_replicates=20,
        bootstrap_seed=3,
    )

    assert result["training_share"] == pytest.approx([2 / 3, 1 / 3])
    assert result["generated_share"] == pytest.approx([0.25, 0.75])
    assert result["shift"] == pytest.approx([-5 / 12, 5 / 12])
    assert result["amplification"] == pytest.approx([0.375, 2.25])
    assert result["generated_ci"]["mean"] == pytest.approx(result["generated_share"])


def test_analyze_code_distribution_can_define_bins_from_catalog_assignments():
    item_codes = {
        0: ("a0",),
        1: ("a1",),
    }

    result = analyze_code_distribution(
        target_items=[0, 0, 0, 1],
        popularity_items=[0, 1],
        item_codes=item_codes,
        beams_by_user=[["x", "y"]],
        code_tokens={"x": item_codes[0], "y": item_codes[1]},
        bins=2,
        top_k=2,
        bootstrap_replicates=10,
        bootstrap_seed=3,
    )

    assert result["training_share"] == pytest.approx([0.5, 0.5])
    assert result["generated_share"] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("top_k", [0, 3])
def test_generated_user_shares_rejects_invalid_rank_cutoffs(top_k):
    with pytest.raises(ValueError, match="top_k"):
        generated_user_shares(
            [["x", "y"]],
            {"x": ("a0",), "y": ("a1",)},
            {"a0": np.array([1.0]), "a1": np.array([1.0])},
            top_k=top_k,
        )
