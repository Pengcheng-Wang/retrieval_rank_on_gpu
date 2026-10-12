from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_history import frozen_exposure_history
from rrbench.semantic_ir import panel_metrics
from rrbench.semantic_models import normalize
from rrbench.semantic_recommendation_baselines import validate_catalog_mapping
from rrbench.semantic_sampling import stable_rank


def balanced_blocks(
    blocks: Iterable[tuple[int, int, int]], maximum: int, seed: int,
) -> dict[tuple[int, int], int]:
    if maximum <= 0:
        raise ValueError("Fitting block budget must be positive.")
    grouped = defaultdict(list)
    identities = set()
    for user, timestamp, lower in blocks:
        identity = int(user), int(timestamp)
        if identity in identities:
            raise ValueError("Fitting block identity is duplicated.")
        identities.add(identity)
        grouped[identity[0]].append((stable_rank("extension-fit-block", seed, f"{user}:{timestamp}"), identity, int(lower)))
    if not grouped:
        raise ValueError("No admitted training blocks are available.")
    for values in grouped.values():
        values.sort()
    users = sorted(grouped, key=lambda user: stable_rank("extension-fit-user", seed, str(user)))
    result = {}
    for position in range(max(map(len, grouped.values()))):
        for user in users:
            if position < len(grouped[user]):
                _, identity, lower = grouped[user][position]
                result[identity] = lower
                if len(result) == maximum:
                    return result
    return result


def padded_histories(histories: list[list[int]], items: int, maximum: int = 12) -> np.ndarray:
    if not histories or min(items, maximum) <= 0:
        raise ValueError("A nonempty history panel and positive catalog/width are required.")
    result = np.full((len(histories), maximum), -1, dtype=np.int64)
    for row, history in enumerate(histories):
        if (
            not 0 < len(history) <= maximum
            or any(not isinstance(item, (int, np.integer)) or not 0 <= item < items for item in history)
        ):
            raise ValueError("An empty, oversized or out-of-catalog history is invalid.")
        result[row, :len(history)] = history
    return result


def wrong_user_histories(histories: list[list[int]], users: np.ndarray, seed: int) -> list[list[int]]:
    if users.ndim != 1 or len(users) != len(histories):
        raise ValueError("User identities and frozen histories must align.")
    by_user = {}
    for user, history in zip(users.tolist(), histories):
        if user in by_user and by_user[user] != history:
            raise ValueError("Inconsistent frozen held-out histories within a user.")
        by_user[user] = history
    ordered = sorted(by_user, key=lambda user: stable_rank("extension-wrong-user", seed, str(user)))
    if len(ordered) < 2:
        raise ValueError("Wrong-user control needs at least two users.")
    mapping = {user: ordered[(index + 1) % len(ordered)] for index, user in enumerate(ordered)}
    return [list(by_user[mapping[user]]) for user in users.tolist()]


def permute_history_order(histories: list[list[int]], seed: int) -> list[list[int]]:
    result, known = [], {}
    for history in histories:
        key = tuple(history)
        if not key:
            raise ValueError("Order control cannot invent a missing history.")
        if key not in known:
            digest = stable_rank("extension-order", seed, repr(key))[0]
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            order = rng.permutation(len(key))
            changed = [key[index] for index in order]
            if len(set(key)) > 1 and changed == history:
                changed = list(key[1:] + key[:1])
            known[key] = changed
        result.append(list(known[key]))
    return result


def mean_history(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (
        features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool
        or not mask.any(dim=1).all() or not torch.isfinite(features[mask]).all()
    ):
        raise ValueError("History vectors need aligned nonempty valid masks.")
    selected = features.float().masked_fill(~mask[..., None], 0)
    return normalize(selected.sum(dim=1) / mask.sum(dim=1, keepdim=True))

class HistoryAttentionHead(nn.Module):
    def __init__(
        self, dimension: int, maximum_history: int = 12, hidden: int = 128, bottleneck: int = 256,
    ) -> None:
        super().__init__()
        if min(dimension, maximum_history, hidden, bottleneck) <= 0:
            raise ValueError("User-head dimensions must be positive.")
        self.dimension, self.maximum_history = dimension, maximum_history
        self.key = nn.Linear(dimension, hidden)
        self.recency = nn.Embedding(maximum_history, hidden)
        self.attention = nn.Linear(hidden, 1)
        self.residual = nn.Sequential(
            nn.Linear(dimension, bottleneck), nn.GELU(), nn.Linear(bottleneck, dimension),
        )
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3 or features.shape[2] != self.dimension
            or features.shape[1] > self.maximum_history or mask.shape != features.shape[:2]
            or mask.dtype != torch.bool or not mask.any(dim=1).all()
            or not torch.isfinite(features[mask]).all()
        ):
            raise ValueError("User head requires aligned finite nonempty histories.")
        features = features.float().masked_fill(~mask[..., None], 0)
        positions = torch.arange(features.shape[1], device=features.device)[None, :]
        ages = (mask.sum(dim=1, keepdim=True) - 1 - positions).clamp_min(0)
        hidden = torch.tanh(self.key(features) + self.recency(ages))
        logits = self.attention(hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = logits.softmax(dim=1)
        pooled = (features * weights[..., None]).sum(dim=1)
        return normalize(pooled + 0.1 * self.residual(pooled))


def sampled_softmax_loss(
    logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor, temperature: float = 0.05,
) -> torch.Tensor:
    if (
        logits.ndim != 2 or positive.shape != logits.shape or valid.shape != logits.shape
        or positive.dtype != torch.bool or valid.dtype != torch.bool or temperature <= 0
        or (positive & ~valid).any() or not positive.any(dim=1).all()
        or not (valid & ~positive).any(dim=1).all() or not torch.isfinite(logits[valid]).all()
    ):
        raise ValueError("Sampled retrieval loss requires finite positives and distinct valid negatives.")
    scaled = logits.float() / temperature
    denominator = scaled.masked_fill(~valid, -torch.inf).logsumexp(dim=1)
    numerator = scaled.masked_fill(~positive, -torch.inf).logsumexp(dim=1)
    return (denominator - numerator).mean()


def training_candidates(
    panel: dict[str, Any], items: int, seed: int, logged_limit: int = 12, catalog_limit: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if items <= 1 or min(logged_limit, catalog_limit) < 0 or logged_limit + catalog_limit <= 0:
        raise ValueError("Negative sampling needs a valid catalog and positive total budget.")
    rows, positive_counts = [], []
    logged_count, unjudged_count = 0, 0
    for query in panel["query_ids"]:
        positive = sorted(int(item) for item, grade in panel["qrels"][query].items() if grade >= 1)
        if not positive or any(not 0 <= item < items for item in positive):
            raise ValueError("Fitting query lacks valid catalog positives.")
        rng = np.random.default_rng(int.from_bytes(stable_rank("extension-negatives", seed, query)[0][:8], "little"))
        logged = sorted(set(panel["logged_nonpositive_item_rows"][query]) - set(positive))
        if any(not isinstance(item, int) or not 0 <= item < items for item in logged):
            raise ValueError("Logged negative source contains invalid item rows.")
        if len(logged) > logged_limit:
            logged = rng.choice(logged, logged_limit, replace=False).tolist()
        blocked = set(positive + logged)
        sampled = []
        budget = min(catalog_limit, items - len(blocked))
        while len(sampled) < budget:
            value = int(rng.integers(items))
            if value not in blocked:
                blocked.add(value)
                sampled.append(value)
        if not logged and not sampled:
            raise ValueError("Fitting query has no admissible weak-negative alternatives.")
        rows.append(positive + logged + sampled)
        positive_counts.append(len(positive))
        logged_count += len(logged)
        unjudged_count += len(sampled)
    if not rows:
        raise ValueError("Fitting query population is empty.")
    ids = np.full((len(rows), max(map(len, rows))), -1, dtype=np.int64)
    positives = np.zeros_like(ids, dtype=bool)
    for row, (values, count) in enumerate(zip(rows, positive_counts)):
        ids[row, :len(values)] = values
        positives[row, :count] = True
    return ids, positives, ids >= 0, {
        "positive_pairs": sum(positive_counts), "logged_negative_pairs": logged_count,
        "sampled_unjudged_negative_pairs": unjudged_count,
    }


def equal_user_metrics(quality: dict[str, Any], users: np.ndarray) -> dict[str, Any]:
    rows = quality["per_query"]
    if users.ndim != 1 or len(users) != len(rows) or quality["queries"] != len(rows) or not rows:
        raise ValueError("Equal-user aggregation requires every unchanged panel row.")
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    result = {"queries": len(rows), "users": len(unique), "equal_user": {}, "query_weighted": {}}
    for metric in rows[0]:
        if metric == "query_id":
            continue
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or not np.isclose(values.mean(), quality[metric]):
            raise ValueError("Query means changed before equal-user aggregation.")
        result["query_weighted"][metric] = float(values.mean())
        result["equal_user"][metric] = float((np.bincount(inverse, weights=values) / counts).mean())
        if metric.startswith("recall@"):
            result["hit_users@" + metric.removeprefix("recall@")] = int(len(np.unique(users[values > 0])))
    return result


def recurrence_popularity(histories: list[list[int]], popularity_order: np.ndarray, k: int) -> np.ndarray:
    if (
        popularity_order.ndim != 1 or not 0 < k <= len(popularity_order)
        or not np.array_equal(np.sort(popularity_order), np.arange(len(popularity_order)))
    ):
        raise ValueError("Popularity order must cover the full catalog exactly once.")
    result = []
    for history in histories:
        if any(not 0 <= item < len(popularity_order) for item in history):
            raise ValueError("Recurrence history is outside the frozen catalog.")
        selected = list(dict.fromkeys(reversed(history)))
        known = set(selected)
        for item in popularity_order:
            if len(selected) >= k:
                break
            if int(item) not in known:
                selected.append(int(item))
                known.add(int(item))
        result.append(selected[:k])
    return np.asarray(result, dtype=np.int64)


def validate_history_panel(panel: dict[str, Any], items: int, require_clock_bounds: bool = False) -> None:
    identifiers = panel["query_ids"]
    if (
        not identifiers or len(identifiers) != len(set(identifiers))
        or len(panel["users"]) != len(identifiers)
        or len(panel["history_item_rows"]) != len(identifiers)
        or panel["catalog_items"] != items or set(panel["qrels"]) != set(identifiers)
    ):
        raise ValueError("History panel identities, population or catalog mapping differ.")
    padded_histories(panel["history_item_rows"], items)
    for query in identifiers:
        labels = panel["qrels"][query]
        if not labels or any(not 0 <= int(item) < items or int(grade) != 1 for item, grade in labels.items()):
            raise ValueError("Conditional history query lacks valid known catalog positives.")
    if require_clock_bounds:
        bounds, lower = panel["history_upper_bounds"], panel["target_clock_lower"]
        if len(bounds) != len(identifiers) or len(lower) != len(identifiers):
            raise ValueError("History clock envelope population differs.")
        for history, upper, target in zip(panel["history_item_rows"], bounds, lower):
            if len(upper) != len(history) or any(value > target for value in upper):
                raise ValueError("A history clock envelope crosses the target lower bound.")


def checked_training_panel(directory: Path) -> dict[str, Any]:
    receipt = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    if not receipt["complete"] or sha256(directory / "training-panel.json") != receipt["panel_sha256"]:
        raise ValueError("Prepared fitting panel completion or checksum changed.")
    return receipt


def prepare_training_panel(
    catalog: Path, output: Path, maximum: int = 50000, seed: int = 20261011, smoke: bool = False,
) -> dict[str, Any]:
    manifest = json.loads((catalog / "corpus.json").read_text(encoding="utf-8"))
    if (
        manifest["clock_policy_version"] != 2 or not manifest["million_item_gate_pass"]
        or (not smoke and manifest["rows"] != 1798875)
    ):
        raise ValueError("Preparation requires the original audited recommendation catalog.")
    original = validate_catalog_mapping(catalog, manifest)
    mapped = {int(item): row for row, item in enumerate(original)}
    database = catalog / "private-catalog.sqlite"
    database_hash = sha256(database)
    binding = {
        "catalog_sha256": manifest["catalog_sha256"], "catalog_ids_sha256": manifest["ids_sha256"],
        "database_sha256": database_hash, "maximum_blocks": maximum, "seed": seed,
        "history": "Same last12 admitted-training exposure items; upper<=target lower, no label filtering.",
        "selection": "User-balanced hash-selected training blocks before label/history eligibility.",
        "study_role": "smoke" if smoke else "full-catalog-exploratory-fitting",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        return checked_training_panel(output)
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        training_blocks = list(connection.execute(
            "SELECT user,timestamp,lower FROM blocks WHERE phase=1 ORDER BY user,timestamp"
        ))
        selected = balanced_blocks(training_blocks, maximum, seed)
        histories = defaultdict(list)
        for user, item, upper in connection.execute(
            "SELECT e.user,e.item,b.upper FROM events e JOIN blocks b USING(user,timestamp) "
            "JOIN semantic s USING(item) WHERE b.phase=1 ORDER BY e.user,b.upper,e.timestamp,e.item"
        ):
            histories[user].append((mapped[item], upper))
        history_arrays = {
            user: (
                np.asarray([item for item, _ in values], dtype=np.int64),
                np.asarray([upper for _, upper in values], dtype=np.int64),
            ) for user, values in histories.items()
        }
        del histories
        positives, negatives = defaultdict(set), defaultdict(set)
        for user, timestamp, item, label in connection.execute(
            "SELECT e.user,e.timestamp,e.item,e.label FROM events e JOIN blocks b USING(user,timestamp) "
            "JOIN semantic s USING(item) WHERE b.phase=1 ORDER BY e.user,e.timestamp,e.item"
        ):
            if (user, timestamp) in selected:
                (positives if label else negatives)[(user, timestamp)].add(mapped[item])
    panel = {
        "name": "kuairand-fit", "query_ids": [], "users": [], "history_item_rows": [],
        "history_upper_bounds": [], "target_clock_lower": [], "qrels": {},
        "logged_nonpositive_item_rows": {}, "catalog_items": len(original),
        "catalog_sha256": manifest["catalog_sha256"], "catalog_ids_sha256": manifest["ids_sha256"],
        "binary_threshold": 1, "history_contract": binding["history"],
        "evaluation_status": "Post-test exploratory fitting; no held-out labels/history are used.",
    }
    skipped_history, skipped_positive = 0, 0
    for (user, timestamp), lower in sorted(selected.items()):
        arrays = history_arrays.get(user)
        if arrays is None:
            skipped_history += 1
            continue
        history = frozen_exposure_history(*arrays, target_lower=lower, maximum=12)
        if not len(history):
            skipped_history += 1
            continue
        known = positives[(user, timestamp)]
        if not known:
            skipped_positive += 1
            continue
        boundary = np.searchsorted(arrays[1], lower, side="right")
        upper = arrays[1][max(0, boundary - 12):boundary]
        query = f"1:{user}:{timestamp}"
        panel["query_ids"].append(query)
        panel["users"].append(user)
        panel["history_item_rows"].append(history.tolist())
        panel["history_upper_bounds"].append(upper.tolist())
        panel["target_clock_lower"].append(lower)
        panel["qrels"][query] = {str(item): 1 for item in sorted(known)}
        panel["logged_nonpositive_item_rows"][query] = sorted(negatives[(user, timestamp)] - known)
    validate_history_panel(panel, len(original), require_clock_bounds=True)
    if sha256(database) != database_hash:
        raise ValueError("Audited training source changed during fitting preparation.")
    write_json(output / "training-panel.json", panel)
    receipt = {
        "complete": True, "panel_sha256": sha256(output / "training-panel.json"),
        "catalog_items": len(original), "queries": len(panel["query_ids"]),
        "users": len(set(panel["users"])), "input_contract": binding,
        "selection_audit": {
            "source_training_blocks": len(training_blocks), "selected_blocks": len(selected),
            "selected_no_history": skipped_history, "selected_no_covered_positive": skipped_positive,
            "retained_queries": len(panel["query_ids"]),
        },
    }
    write_json(output / "completed.json", receipt)
    print("CONTROLLED_PERSONALIZATION_FITTING_PANEL_VERIFIED", receipt["selection_audit"], flush=True)
    return receipt


def target_strata(
    panel: dict[str, Any], rankings: np.ndarray, positive_support: np.ndarray,
) -> dict[str, Any]:
    if (
        positive_support.ndim != 1 or (positive_support < 0).any()
        or len(rankings) != len(panel["query_ids"]) or len(panel["users"]) != len(rankings)
    ):
        raise ValueError("Target diagnostic rows and training-positive support must align.")
    groups = {
        "recurrent": lambda item, history: item in history,
        "novel": lambda item, history: item not in history,
        "training_positive_support_zero": lambda item, history: positive_support[item] == 0,
        "training_positive_support_one": lambda item, history: positive_support[item] == 1,
        "training_positive_support_two_to_nine": lambda item, history: 2 <= positive_support[item] < 10,
        "training_positive_support_ten_plus": lambda item, history: positive_support[item] >= 10,
    }
    result = {}
    for name, predicate in groups.items():
        indices, qrels = [], {}
        for row, query in enumerate(panel["query_ids"]):
            history = set(panel["history_item_rows"][row])
            labels = {}
            for item, grade in panel["qrels"][query].items():
                item = int(item)
                if not 0 <= item < len(positive_support):
                    raise ValueError("Diagnostic positive target is outside the frozen catalog.")
                if predicate(item, history):
                    labels[item] = grade
            if labels:
                indices.append(row)
                qrels[query] = labels
        if not indices:
            result[name] = {"status": "no_targets_in_stratum", "queries": 0, "users": 0}
            continue
        selected = {
            "query_ids": [panel["query_ids"][row] for row in indices],
            "qrels": qrels, "binary_threshold": 1,
        }
        quality = panel_metrics(selected, rankings[indices])
        result[name] = {
            "status": "ok", **equal_user_metrics(quality, np.asarray(panel["users"])[indices]),
            "known_positive_targets": sum(map(len, qrels.values())),
            "population": "Only queries containing targets in this declared stratum; original primary means remain separate.",
        }
    return result


def history_metadata(catalog: Path, panel_file: Path) -> dict[str, Any]:
    manifest = json.loads((catalog / "corpus.json").read_text())
    original = validate_catalog_mapping(catalog, manifest)
    panel = json.loads(panel_file.read_text())
    if panel["catalog_sha256"] != manifest["catalog_sha256"] or panel["catalog_ids_sha256"] != manifest["ids_sha256"]:
        raise ValueError("History diagnostic and catalog identity differ.")
    mapped = {int(item): row for row, item in enumerate(original)}
    known_users = set(panel["users"])
    histories = defaultdict(list)
    database = catalog / "private-catalog.sqlite"
    source_hash = sha256(database)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for user, item, upper in connection.execute(
            "SELECT e.user,e.item,b.upper FROM events e JOIN blocks b USING(user,timestamp) "
            "JOIN semantic s USING(item) WHERE b.phase=1 ORDER BY e.user,b.upper,e.timestamp,e.item"
        ):
            if user in known_users:
                histories[user].append((mapped[item], upper))
    arrays = {
        user: (np.asarray([item for item, _ in values]), np.asarray([upper for _, upper in values]))
        for user, values in histories.items()
    }
    upper_bounds, ages = [], []
    for user, given, lower in zip(panel["users"], panel["history_item_rows"], panel["target_clock_lower"]):
        items, upper = arrays[user]
        expected = frozen_exposure_history(items, upper, lower, 12)
        if expected.tolist() != given:
            raise ValueError("Existing frozen history differs from the audited training source.")
        boundary = np.searchsorted(upper, lower, side="right")
        bounds = upper[max(0, boundary - 12):boundary]
        upper_bounds.append(bounds.tolist())
        ages.append(int(lower - bounds[-1]))
    if sha256(database) != source_hash:
        raise ValueError("History diagnostic database changed while reading.")
    return {
        "complete": True, "panel_sha256": sha256(panel_file), "database_sha256": source_hash,
        "query_ids": panel["query_ids"], "history_upper_bounds": upper_bounds, "history_age_ms": ages,
        "clock_status": "Declared recorded-clock envelope distance, not verified online feedback availability.",
    }
