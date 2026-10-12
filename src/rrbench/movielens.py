from __future__ import annotations

import urllib.request
import zipfile
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rrbench.io import sha256, write_json

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
LICENSE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"


@dataclass
class RecommendationData:
    train_users: np.ndarray
    train_items: np.ndarray
    train_labels: np.ndarray
    eval_users: np.ndarray
    validation_targets: np.ndarray
    test_targets: np.ndarray
    train_seen: np.ndarray
    validation_seen: np.ndarray
    test_seen: np.ndarray
    catalog: np.ndarray
    user_counts: np.ndarray
    item_counts: np.ndarray

    @property
    def users(self) -> int:
        return len(self.user_counts)

    @property
    def items(self) -> int:
        return len(self.item_counts)


def chronological_split(ratings: np.ndarray) -> RecommendationData:
    if ratings.ndim != 2 or ratings.shape[1] != 4 or len(ratings) == 0:
        raise ValueError("Ratings must have [user, item, rating, timestamp] columns.")
    _, users = np.unique(ratings[:, 0], return_inverse=True)
    _, items = np.unique(ratings[:, 1], return_inverse=True)
    user_count, item_count = users.max() + 1, items.max() + 1
    train_seen = np.zeros((user_count, item_count), dtype=np.bool_)
    validation_seen = np.zeros_like(train_seen)
    test_seen = np.zeros_like(train_seen)
    training_indices = []
    eval_users, validation, test = [], [], []
    order = np.lexsort((items, ratings[:, 3], users))
    boundaries = np.searchsorted(users[order], np.arange(user_count + 1))
    for user in range(user_count):
        indices = order[boundaries[user] : boundaries[user + 1]]
        positives = indices[ratings[indices, 2] >= 4]
        times = np.unique(ratings[positives, 3])
        if len(times) < 3:
            continue
        validation_time, test_time = times[-2:]
        train = indices[ratings[indices, 3] < validation_time]
        if not (ratings[train, 2] >= 4).any():
            continue
        training_indices.extend(train.tolist())
        train_seen[user, items[train]] = True
        validation_seen[user, items[indices[ratings[indices, 3] < validation_time]]] = True
        test_seen[user, items[indices[ratings[indices, 3] < test_time]]] = True
        # Same-second ratings are simultaneous; pick a deterministic target
        # within each held-out timestamp rather than inventing an event order.
        validation.append(int(items[positives[ratings[positives, 3] == validation_time]].min()))
        test.append(int(items[positives[ratings[positives, 3] == test_time]].min()))
        eval_users.append(user)
    if not eval_users:
        raise ValueError("No users have three distinct positive timestamps.")
    train = np.asarray(training_indices, dtype=np.int64)
    labels = (ratings[train, 2] >= 4).astype(np.float32)
    catalog = np.unique(items[train[labels.astype(bool)]])
    return RecommendationData(
        users[train], items[train], labels, np.asarray(eval_users),
        np.asarray(validation), np.asarray(test), train_seen, validation_seen, test_seen,
        catalog, np.bincount(users[train], minlength=user_count),
        np.bincount(items[train], minlength=item_count),
    )


def load_movielens(directory: Path) -> RecommendationData:
    directory.mkdir(parents=True, exist_ok=True)
    prepared = directory / "prepared.npz"
    if prepared.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if sha256(prepared) != manifest["prepared_sha256"]:
            raise ValueError("Prepared MovieLens checksum differs from its recorded manifest.")
        arrays = np.load(prepared)
        return RecommendationData(**{name: arrays[name] for name in RecommendationData.__dataclass_fields__})
    archive = directory / "ml-1m.zip"
    if not archive.exists():
        temporary = directory / "ml-1m.zip.download"
        urllib.request.urlretrieve(MOVIELENS_URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        license_text = source.read("ml-1m/README").decode("utf-8", errors="strict")
        (directory / "MOVIELENS_LICENSE.txt").write_text(license_text, encoding="utf-8")
        lines = source.read("ml-1m/ratings.dat").decode("ascii").splitlines()
    ratings = np.asarray([line.split("::") for line in lines], dtype=np.int64)
    data = chronological_split(ratings)
    np.savez_compressed(prepared, **vars(data))
    write_json(directory / "manifest.json", {
        "dataset": "MovieLens 1M", "source": MOVIELENS_URL, "license_source": LICENSE_URL,
        "archive_sha256": sha256(archive), "raw_ratings": len(ratings),
        "prepared_sha256": sha256(prepared),
        "training_events": len(data.train_users), "training_positives": int(data.train_labels.sum()),
        "evaluation_users": len(data.eval_users), "warm_catalog_items": len(data.catalog),
        "split": "per-user last two distinct positive timestamps; all fitting events strictly before validation",
        "temporal_limitation": "transductive pooled training, not a global-time deployment simulation",
        "timestamp_ties": "simultaneous; one deterministic target per held-out timestamp",
        "positive_label": "rating >= 4",
        "catalog": "items with at least one positive training interaction",
        "negative_sampling": "uniform warm catalog excluding all observed training items; no future-label blacklist",
        "redistribution": "prohibited without GroupLens permission; do not commit or redistribute source/prepared data",
    })
    return data
