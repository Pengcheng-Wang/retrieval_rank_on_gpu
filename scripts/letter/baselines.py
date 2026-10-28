"""Pinned native conventional references, not recovered LETTER baseline recipes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import cf
import prepare
import protocol
from cf import TeacherSampler, teacher_sequences
from prepare import read_release
from protocol import bind_run_config, file_sha256, rank_metrics, require_hash
from protocol import verify_coverage, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PINS = {
    "recbole": {
        "repository": "https://github.com/RUCAIBox/RecBole",
        "commit": "362d31f00af801d7d99bc635c902d1df1405e79d",
        "license": "MIT",
        "sha256": {
            "model/general_recommender/bpr.py": "b7f97b2c501e6c56c57d17bc25e829fc345380b71f2d4ff43e8aee6e1639b24e",
            "model/general_recommender/lightgcn.py": "2e8d48381ca656cda29c759948cb58fdedf4f50573580845f33c5b1d14cfec31",
            "model/sequential_recommender/hgn.py": "33b12b7d1fb6147406231caceb92102e86925b2ddfc285a6a5539f87ea176d70",
            "model/sequential_recommender/bert4rec.py": "d7136bee1b5e20c10766f4f3c5d0281859ccaf06c3ff77be5ec0ecfbae2a5c82",
            "model/abstract_recommender.py": "f7d0b66b4b2952a321aafd67f713da4ea81bbaa4c148eccbc824784ad30999d2",
            "model/layers.py": "4d1e03f7a78f4e9df7ff35cc09a64797bb75645048e9e6380cb21944e7f02f39",
            "model/loss.py": "83a942315a0927dd80bcf47a43ba7fe9877ddeb5352232814484d2d2c7e20ec9",
            "model/init.py": "eafcf5a6cef245222d6e3cdbfdeae47b33635e5496c7d5945053c55b39ea53b7",
            "data/interaction.py": "4a72fbdb14b7be157336ee0c93f1b9596feb2f4065736cde2b15cf10e35ef801",
            "data/transform.py": "63c083a32e8ec8d98bd15627292b03faf40e61dcabe5958155ce6a98e5fcb693",
        },
    },
    "sasrec": {
        "repository": "https://github.com/pmixer/SASRec.pytorch",
        "commit": "29b9922adfc0dd1710a5f8fad8dddace2639fc76",
        "license": "Apache-2.0",
        "sha256": {
            "model.py": cf.MODEL_SHA256,
            "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        },
    },
    "caser": {
        "repository": "https://github.com/graytowne/caser_pytorch",
        "commit": "c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1",
        "license": "LGPL-3.0",
        "sha256": {
            "caser.py": "63dd99f91021d99093d33cc1c71e182acdca4c8d3c366e00369981a297a22d31",
            "interactions.py": "ec5ca786403e307c79b729edf16da89bc65d4c8985c26a16751417c44137dc38",
            "utils.py": "8f63d96e9e3449a71543e382157c594104999df0ee36f484e2c4416ca2ab3ab1",
            "LICENSE": "da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768",
        },
    },
}
ARCHITECTURES = {
    "bpr": {"embedding_size": 64},
    "lightgcn": {"embedding_size": 64, "n_layers": 2, "reg_weight": 1e-5,
                 "require_pow": False},
    "hgn": {"embedding_size": 64, "history_length": 20, "pooling_type": "average",
            "reg_weight": [0.0, 0.0], "loss_type": "CE"},
    "bert4rec": {
        "history_length": 20, "n_layers": 2, "n_heads": 2, "hidden_size": 64,
        "inner_size": 256, "hidden_dropout_prob": 0.2, "attn_dropout_prob": 0.2,
        "hidden_act": "gelu", "layer_norm_eps": 1e-12, "initializer_range": 0.02,
        "mask_ratio": 0.2, "ft_ratio": 0.5, "loss_type": "CE",
    },
    "sasrec": {"hidden_units": 50, "maxlen": 50, "num_blocks": 2,
               "num_heads": 1, "dropout_rate": 0.5},
    "caser": {"L": 5, "T": 3, "d": 50, "nh": 16, "nv": 4, "drop": 0.5,
              "ac_conv": "relu", "ac_fc": "relu", "total_negatives": 3},
}
BATCH_SIZES = {"bpr": 1024, "lightgcn": 1024, "hgn": 256,
               "bert4rec": 256, "sasrec": 128, "caser": 512}
TRAINING = {
    "optimizer": "Adam", "learning_rate": 0.001, "adam_eps": 1e-8,
    "max_epochs": 200, "sasrec_max_epochs": 201, "validate_every": 5,
    "early_stop_checks": 10, "selection_metric": "ndcg@10",
    "evaluation_batch_size": 128, "candidate_chunk_size": 1024,
}
PROTOCOL = {
    "split": "train=s[:-2];valid=s[-2];test=s[-1]",
    "item_mapping": "released_i_to_model_i+1_padding_0",
    "user_mapping": "numeric_original_user_key_order_1_based",
    "candidate_catalog": "all_real_items", "seen_item_exclusion": False,
    "ties": "score_descending_then_original_item_id_ascending",
    "test_policy": "once_after_validation_selected_checkpoint_frozen",
}
RUNTIME = {"python": "3.10", "torch": "1.13.1+cu117", "numpy": "1.24.4",
           "scipy": "1.10.1", "recbole": "1.2.0"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_pairs, parse_constant=reject_constant)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_recipe(path: Path) -> dict:
    recipe = _read_json(path)
    if set(recipe) != {"schema_version", "status", "sources", "models", "training",
                       "protocol", "runtime", "declarations"}:
        raise ValueError("Unexpected or missing baseline recipe fields")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1 or recipe["status"] != (
        "declared-reconstruction-not-recovered-letter-settings"
    ):
        raise ValueError("Not a version-1 declared reconstruction recipe")
    for field, expected in (("sources", SOURCE_PINS), ("training", TRAINING),
                            ("protocol", PROTOCOL), ("runtime", RUNTIME)):
        if _digest(recipe[field]) != _digest(expected):
            raise ValueError(f"Frozen version-1 {field} contract differs")
    if set(recipe["models"]) != set(ARCHITECTURES):
        raise ValueError("Missing or unknown model")
    for name, architecture in ARCHITECTURES.items():
        expected = {"architecture": architecture, "batch_size": BATCH_SIZES[name],
                    "optimizer_betas": [0.9, 0.98 if name == "sasrec" else 0.999],
                    "weight_decay": 1e-6 if name == "caser" else 0.0}
        if _digest(recipe["models"][name]) != _digest(expected):
            raise ValueError(f"Frozen {name} architecture/optimizer contract differs")
    if not isinstance(recipe["declarations"], list) or not recipe["declarations"] or not all(
        isinstance(item, str) and item for item in recipe["declarations"]
    ):
        raise ValueError("Recipe must explicitly declare reconstruction decisions")
    return recipe


@dataclass
class Release:
    users: list[str]
    train: dict[int, list[int]]
    valid: dict[int, int]
    test: dict[int, int]
    items: int

    def histories(self, split):
        if split == "valid":
            return {u: list(s) for u, s in self.train.items()}
        if split == "test":
            return {u: s + [self.valid[u]] for u, s in self.train.items()}
        raise ValueError("Evaluation split must be valid or test")


def release_from_interactions(interactions, item_count):
    if type(item_count) is not int or item_count <= 0 or not isinstance(interactions, dict):
        raise ValueError("Invalid released catalog")
    if not interactions or any(
        not isinstance(u, str) or not u.isdecimal() for u in interactions
    ) or len({int(u) for u in interactions}) != len(interactions):
        raise ValueError("Original user keys must be distinct numeric IDs")
    if any(not isinstance(s, list) or len(s) < 5 or any(
        type(i) is not int or not 0 <= i < item_count for i in s
    ) for s in interactions.values()):
        raise ValueError("Invalid released sequence: need five events and real integer item IDs")
    users, train, valid, test = teacher_sequences(interactions)
    return Release(users, train, valid, test, item_count)


def load_release(upstream, manifest_path, dataset):
    manifest = _read_json(manifest_path)
    interactions, items, _ = read_release(Path(upstream), manifest, dataset)
    return release_from_interactions(interactions, len(items))


def _validate_train(train, items):
    if type(items) is not int or items <= 0 or not train or (
        set(train) != set(range(1, len(train) + 1))
    ):
        raise ValueError("Training IDs must be dense one-based model IDs")
    if any(not s or any(type(i) is not int or not 1 <= i <= items for i in s)
           for s in train.values()):
        raise ValueError("Invalid training-only item sequence")


class TrainView:
    """Only the counts and binary training graph required by native RecBole classes."""

    def __init__(self, train, items):
        _validate_train(train, items)
        self.train = train
        self.user_num, self.item_num = len(train) + 1, items + 1

    def num(self, field):
        return {"user_id": self.user_num, "item_id": self.item_num}[field]

    def inter_matrix(self, form="coo"):
        import scipy.sparse as sp

        edges = sorted((u, i) for u, s in self.train.items() for i in set(s))
        users, items = zip(*edges)
        return sp.coo_matrix(
            (np.ones(len(edges), dtype=np.float32), (users, items)),
            shape=(self.user_num, self.item_num),
        ).asformat(form)


class TrainingNegatives:
    def __init__(self, train, items):
        _validate_train(train, items)
        self.items = items
        self.seen = {u: np.asarray(sorted(set(s)), dtype=np.int64) for u, s in train.items()}
        if any(len(s) >= items for s in self.seen.values()):
            raise ValueError("Every training user needs an eligible real negative")

    def draw(self, rng, user, count):
        if type(count) is not int or count < 1:
            raise ValueError("Negative count must be positive")
        values = rng.randint(1, self.items + 1, size=count)
        rejected = np.isin(values, self.seen[user])
        while rejected.any():
            values[rejected] = rng.randint(1, self.items + 1, size=int(rejected.sum()))
            rejected = np.isin(values, self.seen[user])
        return values


def verify_source_files(directory, hashes):
    directory = Path(directory)
    for name, digest in hashes.items():
        require_hash(directory.joinpath(*name.split("/")), digest)
    return dict(hashes)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _caser_modules(source):
    verify_source_files(source, SOURCE_PINS["caser"]["sha256"])
    utils = _load_module("letter_baseline_caser_utils", source / "utils.py")
    previous = sys.modules.get("utils")
    try:
        sys.modules["utils"] = utils
        model = _load_module("letter_baseline_caser", source / "caser.py")
        interactions = _load_module("letter_baseline_caser_interactions",
                                    source / "interactions.py")
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return model.Caser, interactions._sliding_window


def caser_records(train, source):
    if any(len(s) < 3 for s in train.values()):
        raise ValueError("Original Caser requires at least three real training targets")
    _, window = _caser_modules(Path(source))
    records = []
    for user, sequence in train.items():
        for row in window(np.asarray(sequence, dtype=np.int64), 8):
            history, targets = row[:5].tolist(), row[5:].tolist()
            if len(row) != 8 or any(i <= 0 for i in targets):
                raise ValueError("Native Caser produced a padded/malformed training target")
            records.append({"user_id": user, "history": history, "targets": targets})
    return records


def training_records(data, model, history_length):
    if model in {"bpr", "lightgcn"}:
        return [{"user_id": u, "target": item}
                for u, sequence in data.train.items() for item in sequence]
    if model not in {"hgn", "bert4rec"} or history_length < 1:
        raise ValueError("Unsupported training-record contract")
    rows = []
    for user, sequence in data.train.items():
        for position in range(1, len(sequence)):
            end = position + (model == "bert4rec")
            capacity = history_length + (model == "bert4rec")
            rows.append({"user_id": user, "history": sequence[max(0, end-capacity):end],
                         "target": sequence[position]})
    return rows


def padded_histories(histories, length, *, left=False):
    if length < 1 or not histories or any(not h for h in histories):
        raise ValueError("Histories and capacity must be nonempty")
    values = np.zeros((len(histories), length), dtype=np.int64)
    for row, history in enumerate(histories):
        tail = history[-length:]
        if any(type(i) is not int or i <= 0 for i in tail):
            raise ValueError("Observed histories must contain real one-based item IDs")
        if left:
            values[row, -len(tail):] = tail
        else:
            values[row, :len(tail)] = tail
    return values


def bert_inference_inputs(histories, history_length=20):
    body = padded_histories(histories, history_length)
    return {"item_id_list": np.column_stack((np.zeros(len(histories), dtype=np.int64), body)),
            "item_length": np.asarray([min(len(h), history_length) + 1 for h in histories])}


def bert_mask_batch(prefixes, users, negative_pool, rng, *, history_length=20,
                    mask_ratio=0.2, ft_ratio=0.5):
    capacity = history_length + 1
    count = int(mask_ratio * capacity)
    if not 0 < mask_ratio <= 1 or not 0 <= ft_ratio <= 1 or count < 1:
        raise ValueError("BERT masking needs a positive effective mask capacity")
    if len(prefixes) != len(users) or any(len(p) < 2 for p in prefixes):
        raise ValueError("Mask complete training prefixes, including the final training target")
    masked = padded_histories(prefixes, capacity)
    positives = np.zeros((len(prefixes), count), dtype=np.int64)
    negatives = np.zeros_like(positives)
    indices = np.zeros_like(positives)
    terminal_batch = rng.random_sample() < ft_ratio
    for row, (user, prefix) in enumerate(zip(users, prefixes)):
        terminal = min(len(prefix), capacity) - 1
        positions = np.asarray([terminal]) if terminal_batch else (
            np.flatnonzero(rng.random_sample(terminal) < mask_ratio) + 1
        )[-count:]
        # Native loss treats position zero as padding and divides by effective mask count.
        if not len(positions):
            positions = np.asarray([terminal])
        n = len(positions)
        positives[row, :n] = masked[row, positions]
        negatives[row, :n] = negative_pool.draw(rng, user, n)
        indices[row, :n] = positions
        masked[row, positions] = negative_pool.items + 1
    return {"Mask_item_id_list": masked, "Pos_item_id": positives,
            "Neg_item_id": negatives, "MASK_INDEX": indices}


class EpochSampler:
    def __init__(self, data, name, settings, seed, caser_source=None):
        self.data, self.name, self.settings = data, name, settings
        self.rng, self.epochs = np.random.RandomState(seed), 0
        self.negatives = TrainingNegatives(data.train, data.items)
        self.batch_size = settings["batch_size"]
        architecture = settings["architecture"]
        self.teacher = None
        if name == "sasrec":
            self.teacher = TeacherSampler(data.train, data.items, self.batch_size,
                                          architecture["maxlen"], seed)
            self.records = []
            self.steps = max(1, len(data.users) // self.batch_size)
            self.record_count = self.steps * self.batch_size
        else:
            self.records = caser_records(data.train, caser_source) if name == "caser" else (
                training_records(data, name, architecture.get("history_length", 0))
            )
            self.record_count = len(self.records)
            self.steps = math.ceil(self.record_count / self.batch_size)
        if self.steps < 1:
            raise ValueError("An epoch must contain at least one optimizer update")

    def state(self):
        return {"name": self.name, "epochs": self.epochs, "rng": self.rng.get_state(),
                "teacher": self.teacher.state() if self.teacher else None}

    def restore(self, state):
        if set(state) != {"name", "epochs", "rng", "teacher"} or state["name"] != self.name:
            raise ValueError("Sampler checkpoint belongs to a different contract")
        if type(state["epochs"]) is not int or state["epochs"] < 0:
            raise ValueError("Invalid sampler epoch")
        self.rng.set_state(state["rng"])
        self.epochs = state["epochs"]
        if self.teacher:
            self.teacher.restore(state["teacher"])
        elif state["teacher"] is not None:
            raise ValueError("Unexpected SASRec sampler state")

    def batches(self):
        if self.teacher:
            for _ in range(self.steps):
                yield dict(zip(("user_id", "sequence", "positive", "negative"),
                               self.teacher.next_batch()))
        else:
            permutation = self.rng.permutation(self.record_count)
            for start in range(0, self.record_count, self.batch_size):
                rows = [self.records[i] for i in permutation[start:start+self.batch_size]]
                yield self._batch(rows)
        self.epochs += 1

    def _batch(self, rows):
        users = [r["user_id"] for r in rows]
        batch = {"user_id": np.asarray(users, dtype=np.int64)}
        architecture = self.settings["architecture"]
        if self.name == "bert4rec":
            return bert_mask_batch(
                [r["history"] for r in rows], users, self.negatives, self.rng,
                history_length=architecture["history_length"],
                mask_ratio=architecture["mask_ratio"], ft_ratio=architecture["ft_ratio"],
            )
        if self.name == "caser":
            batch.update(
                sequence=np.asarray([r["history"] for r in rows], dtype=np.int64),
                positive=np.asarray([r["targets"] for r in rows], dtype=np.int64),
                negative=np.asarray([self.negatives.draw(self.rng, u, 3) for u in users]),
            )
        else:
            batch["item_id"] = np.asarray([r["target"] for r in rows], dtype=np.int64)
            if self.name in {"bpr", "lightgcn"}:
                batch["neg_item_id"] = np.asarray(
                    [self.negatives.draw(self.rng, u, 1)[0] for u in users]
                )
            else:
                batch["item_id_list"] = padded_histories(
                    [r["history"] for r in rows], architecture["history_length"]
                )
        return batch


def sampled_bce(positive, negative):
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Sampled BCE needs nonempty positive and negative logits")
    return (torch.nn.functional.softplus(-positive).mean()
            + torch.nn.functional.softplus(negative).mean())


def mask_padding(scores, item_count):
    if scores.ndim != 2 or scores.shape[1] != item_count + 1 or scores.shape[0] < 1:
        raise ValueError("Native scores must cover padding plus every real item")
    if not torch.isfinite(scores).all():
        raise ValueError("Native full-catalog scores must be finite before masking padding")
    result = scores.clone()
    result[:, 0] = -torch.inf
    return result


def top_items(scores, k=10):
    if scores.ndim != 2 or not torch.isneginf(scores[:, 0]).all() or (
        not torch.isfinite(scores[:, 1:]).all()
    ) or not 1 <= k <= scores.shape[1] - 1:
        raise ValueError("Invalid padding-masked full catalog or ranking cutoff")
    # Stable sort preserves ascending released IDs among exactly equal scores.
    return torch.argsort(scores[:, 1:], dim=1, descending=True, stable=True)[:, :k]


def aggregate_predictions(users, targets, predictions, item_count):
    verify_coverage(users, [p["user"] for p in predictions])
    verify_coverage(users, list(targets))
    totals = {f"{m}@{k}": 0.0 for k in (5, 10) for m in ("recall", "ndcg")}
    if not users:
        raise ValueError("Evaluation requires every released user")
    for prediction in predictions:
        items = prediction["top10"]
        target = targets[prediction["user"]]
        if (len(items) != min(10, item_count) or len(set(items)) != len(items)
                or any(type(i) is not int or not 0 <= i < item_count for i in items)
                or type(target) is not int or not 0 <= target < item_count):
            raise ValueError("Predictions/targets must be unique original real item IDs")
        for metric, value in rank_metrics(items, target, (5, 10)).items():
            totals[metric] += value
    return {metric: total / len(users) for metric, total in totals.items()}


def _recbole_source():
    distribution = importlib.metadata.distribution("recbole")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    pin = SOURCE_PINS["recbole"]
    if distribution.version != "1.2.0" or direct.get("url") != pin["repository"] or (
        direct.get("vcs_info", {}).get("commit_id") != pin["commit"]
    ):
        raise ValueError("RecBole must be installed from the exact pinned Git revision")
    spec = importlib.util.find_spec("recbole")
    if spec is None or spec.origin is None:
        raise ImportError("Pinned RecBole package is absent")
    verify_source_files(Path(spec.origin).parent, pin["sha256"])
    return {**copy.deepcopy(pin), "installed_direct_url": direct}


class NativeAdapter:
    def __init__(self, name, data, settings, device, *, sasrec_source=None, caser_source=None,
                 candidate_chunk_size=1024):
        self.name, self.data, self.settings = name, data, settings
        self.device = torch.device(device)
        self.candidate_chunk_size = candidate_chunk_size
        architecture = copy.deepcopy(settings["architecture"])
        if name == "sasrec":
            if sasrec_source is None:
                raise ValueError("The pinned external --sasrec source is required")
            verify_source_files(sasrec_source, SOURCE_PINS["sasrec"]["sha256"])
            self.source = copy.deepcopy(SOURCE_PINS["sasrec"])
            self.model = cf.load_model_class(Path(sasrec_source))(
                len(data.users), data.items, SimpleNamespace(**architecture, device=str(device))
            ).to(self.device)
            for parameter in self.model.parameters():
                if parameter.ndim >= 2:
                    torch.nn.init.xavier_normal_(parameter)
        elif name == "caser":
            if caser_source is None:
                raise ValueError("The pinned external --caser source is required")
            cls, _ = _caser_modules(Path(caser_source))
            self.source = copy.deepcopy(SOURCE_PINS["caser"])
            self.model = cls(len(data.users) + 1, data.items + 1,
                             SimpleNamespace(**architecture)).to(self.device)
        else:
            self.source = _recbole_source()
            from recbole.data.interaction import Interaction
            from recbole.model.general_recommender.bpr import BPR
            from recbole.model.general_recommender.lightgcn import LightGCN
            from recbole.model.sequential_recommender.bert4rec import BERT4Rec
            from recbole.model.sequential_recommender.hgn import HGN

            self.Interaction = Interaction
            config = {
                **architecture, "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
                "NEG_PREFIX": "neg_", "LIST_SUFFIX": "_list",
                "ITEM_LIST_LENGTH_FIELD": "item_length",
                "MAX_ITEM_LIST_LENGTH": architecture.get("history_length", 1)
                                       + (name == "bert4rec"),
                "device": self.device,
                "MASK_ITEM_SEQ": "Mask_item_id_list", "POS_ITEMS": "Pos_item_id",
                "NEG_ITEMS": "Neg_item_id", "MASK_INDEX": "MASK_INDEX",
            }
            cls = {"bpr": BPR, "lightgcn": LightGCN, "hgn": HGN, "bert4rec": BERT4Rec}[name]
            self.model = cls(config, TrainView(data.train, data.items)).to(self.device)
        self.clear_caches()
        check_parameters(self.model)

    def clear_caches(self):
        if self.name == "lightgcn":
            self.model.restore_user_e = self.model.restore_item_e = None
        if self.name == "caser":
            self.model.cache_x = None

    def _tensors(self, batch):
        return {key: torch.as_tensor(value, dtype=torch.long, device=self.device)
                for key, value in batch.items()}

    def loss(self, batch):
        self.clear_caches()
        if self.name == "sasrec":
            positive, negative = self.model(
                batch["user_id"], batch["sequence"], batch["positive"], batch["negative"]
            )
            effective = torch.as_tensor(batch["positive"] != 0, device=self.device)
            return sampled_bce(positive[effective], negative[effective])
        if self.name == "caser":
            tensors = self._tensors(batch)
            candidates = torch.cat((tensors["positive"], tensors["negative"]), dim=1)
            logits = self.model(tensors["sequence"], tensors["user_id"][:, None],
                                candidates, for_pred=False).reshape(len(candidates), -1)
            return sampled_bce(logits[:, :3], logits[:, 3:])
        if self.name == "bert4rec" and not np.all((batch["MASK_INDEX"] > 0).any(axis=1)):
            raise ValueError("BERT batch has a zero-effective-mask row")
        return self.model.calculate_loss(self.Interaction(self._tensors(batch)))

    def scores(self, users, histories):
        if len(users) != len(histories) or not users:
            raise ValueError("Scoring needs aligned users and observed histories")
        if self.model.training:
            raise ValueError("Full-catalog scoring requires model.eval()")
        architecture = self.settings["architecture"]
        if self.name == "sasrec":
            sequences = padded_histories(histories, architecture["maxlen"], left=True)
            scores = self.model.predict(np.asarray(users), sequences,
                                        np.arange(self.data.items + 1))
        elif self.name == "caser":
            history = self._tensors({"history": padded_histories(histories, 5, left=True)})[
                "history"
            ]
            user_tensor = torch.as_tensor(users, dtype=torch.long, device=self.device)[:, None]
            parts = []
            for start in range(0, self.data.items + 1, self.candidate_chunk_size):
                candidates = torch.arange(start, min(start + self.candidate_chunk_size,
                                                     self.data.items + 1), device=self.device)
                candidates = candidates.unsqueeze(0).expand(len(users), -1)
                parts.append(self.model(history, user_tensor, candidates, for_pred=False)
                             .reshape(len(users), candidates.shape[1]))
            scores = torch.cat(parts, dim=1)
        else:
            batch = {"user_id": users}
            if self.name == "hgn":
                batch["item_id_list"] = padded_histories(histories, architecture["history_length"])
            elif self.name == "bert4rec":
                batch.update(bert_inference_inputs(histories, architecture["history_length"]))
            scores = self.model.full_sort_predict(self.Interaction(self._tensors(batch)))
            if self.name in {"bpr", "lightgcn"}:
                scores = scores.reshape(len(users), self.data.items + 1)
        return mask_padding(scores, self.data.items)


def seed_everything(seed, device):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG checkpoint")
    if len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Visible CUDA device count changed on continuation")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def check_parameters(model):
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite native model parameter")


def check_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("Missing or nonfinite native model gradients")


def scalar_loss(loss):
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss).all():
        raise ValueError("Native training loss is not a finite scalar")
    return loss.reshape(())


def score_users(adapter, data, split, batch_size):
    histories = data.histories(split)
    targets = data.valid if split == "valid" else data.test
    predictions = []
    adapter.clear_caches()
    adapter.model.eval()
    with torch.inference_mode():
        for start in range(1, len(data.users) + 1, batch_size):
            users = list(range(start, min(start + batch_size, len(data.users) + 1)))
            scores = adapter.scores(users, [histories[u] for u in users])
            top = top_items(scores, min(10, data.items)).cpu().tolist()
            predictions.extend(
                {"user": data.users[u-1], "model_user": u, "target": targets[u]-1, "top10": items}
                for u, items in zip(users, top)
            )
    metrics = aggregate_predictions(
        data.users, {data.users[u-1]: target-1 for u, target in targets.items()},
        predictions, data.items,
    )
    return metrics, predictions


def _runtime_receipt():
    actual = {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
              **{name: importlib.metadata.version(name)
                 for name in ("torch", "numpy", "scipy", "recbole")}}
    if actual != RUNTIME:
        raise ValueError(f"Use the separate pinned baseline runtime, not LETTER's venv: {actual}")
    return {**actual, "python_full": sys.version, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": True, "tf32": False,
            "cublas_workspace_config": ":4096:8",
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _build_run(args):
    recipe = load_recipe(args.recipe)
    runtime = _runtime_receipt()
    data = load_release(args.upstream, args.manifest, args.dataset)
    if data.items < 10:
        raise ValueError("Reference evaluation requires at least ten real items")
    if args.gate and sum(map(len, data.train.values())) > 100:
        raise ValueError("A contract gate is restricted to at most 100 synthetic training events")
    manifest = _read_json(args.manifest)
    if args.gate and manifest.get("purpose") != "synthetic-native-contract-gate":
        raise ValueError("Use an explicitly labeled synthetic manifest for the small GPU gate")
    if not args.gate and manifest.get("purpose") == "synthetic-native-contract-gate":
        raise ValueError("Synthetic gate data cannot be labeled a reference fit")
    settings = copy.deepcopy(recipe["models"][args.model])
    training = copy.deepcopy(recipe["training"])
    training["max_epochs"] = training.pop("sasrec_max_epochs") if args.model == "sasrec" else (
        training["max_epochs"]
    )
    training.pop("sasrec_max_epochs", None)
    if args.gate:
        settings["batch_size"] = 8
        training.update(max_epochs=2, validate_every=1, evaluation_batch_size=3)
    seed_everything(args.seed, args.device)
    adapter = NativeAdapter(
        args.model, data, settings, args.device, sasrec_source=args.sasrec,
        caser_source=args.caser, candidate_chunk_size=training["candidate_chunk_size"],
    )
    source_files = {
        "scripts/letter/baselines.py": Path(__file__),
        "scripts/letter/cf.py": Path(cf.__file__),
        "scripts/letter/prepare.py": Path(prepare.__file__),
        "scripts/letter/protocol.py": Path(protocol.__file__),
        "scripts/letter/author.py": Path(prepare.__file__).with_name("author.py"),
        "src/rrbench/io.py": ROOT / "src" / "rrbench" / "io.py",
    }
    environment = ROOT / "environments" / "letter-baselines"
    config = {
        "schema_version": 1, "kind": recipe["status"],
        "run_class": "contract-gate" if args.gate else "conventional-reference",
        "model": args.model, "dataset": args.dataset, "seed": args.seed, "device": args.device,
        "model_settings": settings, "training": training, "protocol": recipe["protocol"],
        "declarations": recipe["declarations"],
        "provenance": {
            "source": adapter.source, "runtime": runtime,
            "recipe_sha256": file_sha256(args.recipe),
            "manifest_sha256": file_sha256(args.manifest),
            "data_sha256": manifest["datasets"][args.dataset]["sha256"],
            "training_sequences_sha256": _digest(data.train),
            "code_sha256": {name: file_sha256(path) for name, path in source_files.items()},
            "environment_sha256": {name: file_sha256(environment / name)
                                    for name in ("pyproject.toml", "uv.lock")},
        },
        "counts": {"users": len(data.users), "items": data.items,
                   "training_events": sum(map(len, data.train.values())),
                   "training_prefixes": sum(len(s)-1 for s in data.train.values()),
                   "binary_training_edges": sum(len(set(s)) for s in data.train.values())},
    }
    return data, adapter, config


@contextmanager
def _train_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "train.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        lock.unlink()


def _save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _checkpoint_name(epoch):
    return f"checkpoints/epoch-{epoch:05d}.pt"


def bind_user_map(directory, users, *, resume):
    path = directory / "user-map.json"
    if resume:
        if not path.is_file() or _read_json(path) != users:
            raise ValueError("Original user mapping is missing or differs from the frozen input")
    else:
        write_json(path, users)


def _read_checkpoint(directory, config_sha, epoch=None):
    markers = sorted((directory / "checkpoints").glob("epoch-*.json"))
    if not markers:
        raise ValueError("No complete epoch-boundary checkpoint exists")
    marker = markers[-1] if epoch is None else (
        directory / "checkpoints" / f"epoch-{epoch:05d}.json"
    )
    receipt = _read_json(marker)
    path = marker.with_suffix(".pt")
    if type(receipt["epoch"]) is not int or receipt["epoch"] < 1 or (
        path.name != f"epoch-{receipt['epoch']:05d}.pt"
    ):
        raise ValueError("Checkpoint filename/receipt identity mismatch")
    require_hash(path, receipt["sha256"])
    if receipt["config_sha256"] != config_sha:
        raise ValueError("Checkpoint is bound to another run configuration")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or state["config_sha256"] != config_sha or (
        state["epoch"] != receipt["epoch"] or state["sampler"]["epochs"] != state["epoch"]
    ) or state["selection"] != receipt["selection"]:
        raise ValueError("Checkpoint state/receipt identity mismatch")
    return state, receipt


def _save_checkpoint(directory, state):
    epoch = state["epoch"]
    path = directory / _checkpoint_name(epoch)
    _save_torch(path, state)
    receipt = {"epoch": epoch, "sha256": file_sha256(path),
               "config_sha256": state["config_sha256"], "selection": state["selection"],
               "log": state["history"][-1]}
    write_json(path.with_suffix(".json"), receipt)
    write_json(directory / "epochs.json", state["history"])
    markers = sorted((directory / "checkpoints").glob("epoch-*.json"))
    retained = set()
    for marker in markers[-2:]:
        record = _read_json(marker)
        retained.add(record["epoch"])
        if record["selection"] is not None:
            retained.add(record["selection"]["epoch"])
    for marker in markers:
        if _read_json(marker)["epoch"] not in retained:
            marker.with_suffix(".pt").unlink()
            marker.unlink()
    retained_predictions = {
        _read_json(m)["selection"]["predictions"]
        for m in (directory / "checkpoints").glob("epoch-*.json")
        if _read_json(m)["selection"] is not None
    }
    for path in directory.glob("validation-top10-*.json"):
        if path.name not in retained_predictions:
            path.unlink()
    return receipt


def train(args):
    output = args.output
    if (output / "train-completed.json").exists() or (output / "completed.json").exists():
        raise FileExistsError("Training selection is already frozen; do not refit this run")
    data, adapter, config = _build_run(args)
    training = config["training"]
    settings = config["model_settings"]
    sampler = EpochSampler(data, args.model, settings, args.seed, args.caser)
    config["counts"].update(training_records_per_epoch=sampler.record_count,
                            optimizer_steps_per_epoch=sampler.steps)
    optimizer = torch.optim.Adam(
        adapter.model.parameters(), lr=training["learning_rate"],
        betas=tuple(settings["optimizer_betas"]), eps=training["adam_eps"],
        weight_decay=settings["weight_decay"],
    )
    with _train_lock(output):
        existing = [p for p in output.iterdir() if p.name != "train.lock"]
        if existing and not args.resume:
            raise FileExistsError("Use a new empty run directory or explicit --resume")
        if args.resume and not (output / "run-config.json").is_file():
            raise ValueError("Cannot resume without the immutable run configuration")
        bind_run_config(output, config)
        config_sha = file_sha256(output / "run-config.json")
        bind_user_map(output, data.users, resume=args.resume)
        first_epoch, selection, bad_checks, history = 1, None, 0, []
        if args.resume:
            state, _ = _read_checkpoint(output, config_sha)
            adapter.model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            sampler.restore(state["sampler"])
            adapter.clear_caches()
            check_parameters(adapter.model)
            restore_rng(state["rng"])
            first_epoch = state["epoch"] + 1
            selection, bad_checks, history = state["selection"], state["bad_checks"], state["history"]
        draining = [False]
        previous_handler = signal.signal(signal.SIGTERM,
                                         lambda *_: draining.__setitem__(0, True))
        try:
            last_epoch = first_epoch - 1
            for epoch in range(first_epoch, training["max_epochs"] + 1):
                if bad_checks >= training["early_stop_checks"]:
                    break
                started = time.monotonic()
                adapter.clear_caches()
                adapter.model.train()
                total_loss, steps = 0.0, 0
                for batch in sampler.batches():
                    optimizer.zero_grad(set_to_none=True)
                    loss = scalar_loss(adapter.loss(batch))
                    loss.backward()
                    check_gradients(adapter.model)
                    optimizer.step()
                    check_parameters(adapter.model)
                    adapter.clear_caches()
                    total_loss += loss.item()
                    steps += 1
                if steps != sampler.steps or sampler.epochs != epoch:
                    raise ValueError("Epoch coverage/update count mismatch")
                log = {"epoch": epoch, "mean_loss": total_loss / steps, "optimizer_steps": steps}
                if epoch % training["validate_every"] == 0 or epoch == training["max_epochs"]:
                    metrics, predictions = score_users(
                        adapter, data, "valid", training["evaluation_batch_size"]
                    )
                    log["validation"] = metrics
                    if selection is None or metrics["ndcg@10"] > selection["validation"]["ndcg@10"]:
                        prediction_file = output / f"validation-top10-{epoch:05d}.json"
                        write_json(prediction_file, predictions)
                        selection = {"epoch": epoch, "validation": metrics,
                                     "predictions": prediction_file.name,
                                     "predictions_sha256": file_sha256(prediction_file)}
                        bad_checks = 0
                    else:
                        bad_checks += 1
                log["seconds"] = time.monotonic() - started
                history.append(log)
                state = {
                    "schema_version": 1, "config_sha256": config_sha, "epoch": epoch,
                    "model": adapter.model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state(), "rng": capture_rng(), "selection": selection,
                    "bad_checks": bad_checks, "history": history,
                }
                _save_checkpoint(output, state)
                last_epoch = epoch
                print(json.dumps(log, allow_nan=False), flush=True)
                if draining[0] or (args.stop_after_epoch is not None and epoch >= args.stop_after_epoch):
                    return {"status": "drained", "epoch": epoch, "test_evaluated": False}
            if selection is None:
                raise ValueError("No validation-selected checkpoint; training is not complete")
            _, selected_receipt = _read_checkpoint(output, config_sha, selection["epoch"])
            require_hash(output / selection["predictions"], selection["predictions_sha256"])
            receipt = {
                "status": "train-complete", "run_class": config["run_class"],
                "model": args.model, "dataset": args.dataset, "selected_epoch": selection["epoch"],
                "last_epoch": last_epoch, "last_checkpoint": _checkpoint_name(last_epoch),
                "selected_checkpoint": _checkpoint_name(selection["epoch"]),
                "selected_checkpoint_sha256": selected_receipt["sha256"],
                "validation": selection["validation"], "validation_predictions": selection["predictions"],
                "validation_predictions_sha256": selection["predictions_sha256"],
                "config_sha256": config_sha, "user_map_sha256": file_sha256(output / "user-map.json"),
                "provenance": config["provenance"], "test_evaluated": False,
                "early_stopped": bad_checks >= training["early_stop_checks"],
            }
            write_json(output / "train-completed.json", receipt)
            return receipt
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def evaluate(args):
    output = args.output
    if (output / "completed.json").exists() or (output / "test-claimed.json").exists():
        raise FileExistsError("Test evaluation is single-use; this run has already claimed it")
    frozen_path = output / "train-completed.json"
    frozen = _read_json(frozen_path)
    data, adapter, config = _build_run(args)
    sampler = EpochSampler(data, args.model, config["model_settings"], args.seed, args.caser)
    config["counts"].update(training_records_per_epoch=sampler.record_count,
                            optimizer_steps_per_epoch=sampler.steps)
    if not (output / "run-config.json").is_file():
        raise ValueError("Frozen evaluation requires an existing run configuration")
    bind_run_config(output, config)
    config_sha = file_sha256(output / "run-config.json")
    if frozen["status"] != "train-complete" or frozen["config_sha256"] != config_sha:
        raise ValueError("Training selection is not frozen against this configuration")
    require_hash(output / frozen["selected_checkpoint"], frozen["selected_checkpoint_sha256"])
    require_hash(output / frozen["validation_predictions"], frozen["validation_predictions_sha256"])
    require_hash(output / "user-map.json", frozen["user_map_sha256"])
    state, _ = _read_checkpoint(output, config_sha, frozen["selected_epoch"])
    if state["selection"]["epoch"] != frozen["selected_epoch"] or (
        state["selection"]["validation"] != frozen["validation"]
    ):
        raise ValueError("Frozen validation selection differs from the selected checkpoint")
    adapter.model.load_state_dict(state["model"], strict=True)
    adapter.clear_caches()
    check_parameters(adapter.model)
    with (output / "test-claimed.json").open("x", encoding="utf-8") as stream:
        json.dump({"config_sha256": config_sha, "frozen_sha256": file_sha256(frozen_path),
                   "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    metrics, predictions = score_users(
        adapter, data, "test", config["training"]["evaluation_batch_size"]
    )
    prediction_file = output / "test-top10.json"
    write_json(prediction_file, predictions)
    receipt = {
        "status": "complete", "run_class": config["run_class"], "kind": config["kind"],
        "model": args.model, "dataset": args.dataset, "test": metrics,
        "test_users": len(predictions), "items": data.items, "test_evaluated": True,
        "selected_epoch": frozen["selected_epoch"], "validation": frozen["validation"],
        "config_sha256": config_sha, "frozen_train_receipt_sha256": file_sha256(frozen_path),
        "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"],
        "test_predictions_sha256": file_sha256(prediction_file),
        "user_map_sha256": frozen["user_map_sha256"], "provenance": config["provenance"],
    }
    write_json(output / "completed.json", receipt)
    return receipt


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", choices=tuple(ARCHITECTURES), required=True)
        sub.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
        for name in ("upstream", "manifest", "output"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        sub.add_argument("--recipe", type=Path,
                         default=ROOT / "configs" / "letter-conventional-baselines-v1.json")
        sub.add_argument("--sasrec", type=Path)
        sub.add_argument("--caser", type=Path)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
        sub.add_argument("--gate", action="store_true",
                         help="Two-epoch, <=100-event synthetic contract gate; never a paper fit")
        if command == "train":
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after-epoch", type=int)
    return parser


def main():
    args = make_parser().parse_args()
    if args.command == "train" and args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    result = train(args) if args.command == "train" else evaluate(args)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
