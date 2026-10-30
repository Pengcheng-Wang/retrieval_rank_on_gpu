from collections import defaultdict
import math


import numpy as np


def analyze_code_distribution(
    *,
    target_items,
    popularity_items=None,
    item_codes,
    beams_by_user,
    code_tokens,
    bins=5,
    top_k=20,
    rank_weight="uniform",
    levels=None,
    membership_method="fractional",
    bootstrap_replicates=2000,
    bootstrap_seed=20261027,
):
    counts = target_token_counts(
        target_items if popularity_items is None else popularity_items,
        item_codes,
        levels,
    )
    if membership_method == "fractional":
        memberships = fractional_rank_bins(counts, bins)
    elif membership_method == "midrank":
        memberships = midrank_bins(counts, bins)
    else:
        raise ValueError("unsupported membership_method")
    training = target_code_share(
        target_items,
        item_codes,
        memberships,
        levels=levels,
    )
    user_shares = generated_user_shares(
        beams_by_user,
        code_tokens,
        memberships,
        top_k=top_k,
        rank_weight=rank_weight,
        levels=levels,
    )
    generated_ci = bootstrap_mean_ci(
        user_shares,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    generated = np.asarray(generated_ci["mean"])
    shift = generated - training
    return {
        "code_types": len(counts),
        "training_share": training.tolist(),
        "generated_share": generated.tolist(),
        "shift": shift.tolist(),
        "amplification": [
            None if baseline == 0 else observed / baseline
            for observed, baseline in zip(generated, training)
        ],
        "generated_ci": generated_ci,
        "shift_ci": {
            "low": (np.asarray(generated_ci["low"]) - training).tolist(),
            "high": (np.asarray(generated_ci["high"]) - training).tolist(),
        },
    }


def fractional_rank_bins(counts, bins=5):
    if not counts or type(bins) is not int or bins <= 0:
        raise ValueError("counts must be nonempty and bins must be positive")
    grouped = defaultdict(list)
    for code, count in counts.items():
        if not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0:
            raise ValueError("counts must be finite and nonnegative")
        grouped[count].append(code)
    memberships = {}
    total = len(counts)
    start = 0
    for count in sorted(grouped, reverse=True):
        codes = grouped[count]
        end = start + len(codes)
        weights = np.zeros(bins, dtype=np.float64)
        for bin_index in range(bins):
            lower = total * bin_index / bins
            upper = total * (bin_index + 1) / bins
            weights[bin_index] = max(0.0, min(end, upper) - max(start, lower)) / len(codes)
        for code in codes:
            memberships[code] = weights.copy()
        start = end
    return memberships


def midrank_bins(counts, bins=5):
    if not counts or type(bins) is not int or bins <= 0:
        raise ValueError("counts must be nonempty and bins must be positive")
    grouped = defaultdict(list)
    for code, count in counts.items():
        if not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0:
            raise ValueError("counts must be finite and nonnegative")
        grouped[count].append(code)
    memberships = {}
    total = len(counts)
    start = 0
    for count in sorted(grouped, reverse=True):
        codes = grouped[count]
        end = start + len(codes)
        bin_index = min(int(((start + end) / 2) * bins / total), bins - 1)
        weights = np.zeros(bins, dtype=np.float64)
        weights[bin_index] = 1
        for code in codes:
            memberships[code] = weights.copy()
        start = end
    return memberships


def target_token_counts(target_items, item_codes, levels=None):
    counts = {}
    for tokens in item_codes.values():
        for token in _select_levels(tokens, levels):
            counts[token] = 0
    for item in target_items:
        for token in _select_levels(item_codes[item], levels):
            counts[token] += 1
    if not counts:
        raise ValueError("no code tokens selected")
    return counts


def bootstrap_mean_ci(samples, *, replicates, seed):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("samples must be a nonempty finite matrix")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    rng = np.random.default_rng(seed)
    means = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0).tolist(),
        "low": np.percentile(means, 2.5, axis=0).tolist(),
        "high": np.percentile(means, 97.5, axis=0).tolist(),
    }


def _select_levels(tokens, levels):
    if levels is None:
        return tuple(tokens)
    selected = tuple(token for level, token in enumerate(tokens) if level in levels)
    if not selected:
        raise ValueError("levels select no code tokens")
    return selected


def _bin_count(memberships):
    if not memberships:
        raise ValueError("memberships must be nonempty")
    bins = {len(np.asarray(weights)) for weights in memberships.values()}
    if len(bins) != 1 or next(iter(bins)) <= 0:
        raise ValueError("membership vectors must have one positive width")
    return next(iter(bins))


def target_code_share(target_items, item_codes, memberships, levels=None):
    bins = _bin_count(memberships)
    total = np.zeros(bins, dtype=np.float64)
    observations = 0
    for item in target_items:
        tokens = _select_levels(item_codes[item], levels)
        if not tokens:
            raise ValueError("item codes must be nonempty")
        for token in tokens:
            total += memberships[token]
            observations += 1
    if observations == 0:
        raise ValueError("target_items must be nonempty")
    return total / observations


def generated_user_shares(
    beams_by_user,
    code_tokens,
    memberships,
    *,
    top_k,
    rank_weight="uniform",
    levels=None,
):
    bins = _bin_count(memberships)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not beams_by_user or any(len(beams) < top_k for beams in beams_by_user):
        raise ValueError("top_k exceeds available beams")
    if rank_weight == "uniform":
        weights = np.ones(top_k, dtype=np.float64)
    elif rank_weight == "reciprocal":
        weights = 1 / np.arange(1, top_k + 1, dtype=np.float64)
    elif rank_weight == "dcg":
        weights = 1 / np.log2(np.arange(2, top_k + 2, dtype=np.float64))
    else:
        raise ValueError("unsupported rank_weight")
    weights /= weights.sum()
    shares = np.zeros((len(beams_by_user), bins), dtype=np.float64)
    for user_index, beams in enumerate(beams_by_user):
        for rank, code in enumerate(beams[:top_k]):
            tokens = _select_levels(code_tokens[code], levels)
            if not tokens:
                raise ValueError("generated codes must contain learned tokens")
            beam_share = sum((memberships[token] for token in tokens), np.zeros(bins))
            shares[user_index] += weights[rank] * beam_share / len(tokens)
    return shares
