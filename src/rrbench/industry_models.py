from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn


class TorchRecSequenceRanker(nn.Module):
    def __init__(
        self,
        *,
        users: int,
        items: int,
        dimension: int,
        dcn: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        from torchrec.models.dlrm import DLRM, DLRM_DCN
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
        from torchrec.modules.embedding_modules import EmbeddingBagCollection

        collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="user_table",
                    embedding_dim=dimension,
                    num_embeddings=users,
                    feature_names=["user"],
                ),
                EmbeddingBagConfig(
                    name="candidate_table",
                    embedding_dim=dimension,
                    num_embeddings=items,
                    feature_names=["candidate"],
                ),
                EmbeddingBagConfig(
                    name="last_item_table",
                    embedding_dim=dimension,
                    num_embeddings=items,
                    feature_names=["last_item"],
                ),
            ],
            device=device,
        )
        arguments = {
            "embedding_bag_collection": collection,
            "dense_in_features": 4,
            "dense_arch_layer_sizes": [64, dimension],
            "over_arch_layer_sizes": [128, 64, 1],
            "dense_device": device,
        }
        if dcn:
            self.model = DLRM_DCN(
                **arguments,
                dcn_num_layers=3,
                dcn_low_rank_dim=32,
            )
        else:
            self.model = DLRM(**arguments)
        self.to(device)
        self.model_class = f"{type(self.model).__module__}.{type(self.model).__name__}"

    def forward(
        self,
        *,
        users: torch.Tensor,
        candidates: torch.Tensor,
        dense: torch.Tensor,
        histories: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        batch, candidate_count = candidates.shape
        flat_count = batch * candidate_count
        flat_users = users[:, None].expand(-1, candidate_count).reshape(-1)
        flat_candidates = candidates.reshape(-1)
        flat_last_items = histories[:, 0, None].expand(-1, candidate_count).reshape(-1)
        sparse = KeyedJaggedTensor(
            keys=["user", "candidate", "last_item"],
            values=torch.cat((flat_users, flat_candidates, flat_last_items)),
            lengths=torch.ones(
                3 * flat_count, dtype=torch.long, device=users.device
            ),
            stride=flat_count,
            length_per_key=[flat_count, flat_count, flat_count],
            offset_per_key=[0, flat_count, 2 * flat_count, 3 * flat_count],
            index_per_key={"user": 0, "candidate": 1, "last_item": 2},
        )
        return self.model(dense.reshape(flat_count, -1), sparse).reshape(
            batch, candidate_count
        )


class DINRanker(nn.Module):
    def __init__(self, *, users: int, items: int, dimension: int) -> None:
        super().__init__()
        self.item = nn.Embedding(items, dimension, padding_idx=0)
        self.user = nn.Embedding(users, 16)
        self.attention = nn.Sequential(
            nn.Linear(dimension * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(dimension * 3 + 16 + 4, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        users: torch.Tensor,
        candidates: torch.Tensor,
        dense: torch.Tensor,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidate_count = candidates.shape
        history_length = histories.shape[1]
        history = self.item(histories)
        candidate = self.item(candidates)
        history_expanded = history[:, None].expand(
            -1, candidate_count, -1, -1
        )
        candidate_expanded = candidate[:, :, None].expand(
            -1, -1, history_length, -1
        )
        attention_input = torch.cat(
            (
                history_expanded,
                candidate_expanded,
                history_expanded - candidate_expanded,
                history_expanded * candidate_expanded,
            ),
            dim=-1,
        )
        logits = self.attention(attention_input).squeeze(-1)
        mask = (
            torch.arange(history_length, device=histories.device)[None, :]
            < lengths[:, None]
        )
        logits = logits.masked_fill(~mask[:, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(weights[..., None] * history_expanded, dim=2)
        user = self.user(users)[:, None].expand(-1, candidate_count, -1)
        features = torch.cat(
            (candidate, pooled, candidate * pooled, user, dense),
            dim=-1,
        )
        return self.head(features).squeeze(-1)


class TWINRanker(nn.Module):
    def __init__(
        self,
        *,
        users: int,
        items: int,
        dimension: int,
        esu_items: int,
        maximum_gap_seconds: int,
    ) -> None:
        super().__init__()
        self.esu_items = esu_items
        self.maximum_gap_seconds = max(maximum_gap_seconds, 1)
        self.item = nn.Embedding(items, dimension, padding_idx=0)
        self.user = nn.Embedding(users, 16)
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.cross_bias = nn.Linear(2, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(dimension * 3 + 16 + 4, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        users: torch.Tensor,
        candidates: torch.Tensor,
        dense: torch.Tensor,
        histories: torch.Tensor,
        ratings: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
        request_times: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidate_count = candidates.shape
        history_length = histories.shape[1]
        history = self.item(histories)
        candidate = self.item(candidates)
        query = self.query(candidate)
        key = self.key(history)
        scores = torch.einsum("bcd,bld->bcl", query, key) / math.sqrt(
            query.shape[-1]
        )
        gaps = torch.clamp(request_times[:, None] - timestamps, min=0)
        gap_feature = torch.log1p(gaps.float()) / math.log1p(
            self.maximum_gap_seconds
        )
        rating_feature = torch.clamp(ratings.float(), min=0) / 5.0
        cross_features = torch.stack((rating_feature, gap_feature), dim=-1)
        scores = scores + self.cross_bias(cross_features).squeeze(-1)[:, None]
        mask = (
            torch.arange(history_length, device=histories.device)[None, :]
            < lengths[:, None]
        )
        scores = scores.masked_fill(
            ~mask[:, None], torch.finfo(scores.dtype).min
        )
        k = min(self.esu_items, history_length)
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        value = self.value(history)
        value_expanded = value[:, None].expand(-1, candidate_count, -1, -1)
        selected = torch.gather(
            value_expanded,
            2,
            top_indices[..., None].expand(-1, -1, -1, value.shape[-1]),
        )
        weights = torch.softmax(top_scores, dim=-1)
        pooled = torch.sum(weights[..., None] * selected, dim=2)
        user = self.user(users)[:, None].expand(-1, candidate_count, -1)
        features = torch.cat(
            (candidate, pooled, candidate * pooled, user, dense),
            dim=-1,
        )
        return self.head(features).squeeze(-1)


class TransActRanker(nn.Module):
    def __init__(
        self,
        *,
        users: int,
        items: int,
        item_dimension: int,
        action_dimension: int,
        sequence_length: int,
        latest_embeddings: int,
    ) -> None:
        super().__init__()
        vendor = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "transact"
            / "transact_code"
        )
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        transact_module = importlib.import_module("transact")
        config_module = importlib.import_module("transact_config")
        if Path(transact_module.__file__).resolve().parent != vendor:
            raise RuntimeError("Imported TransAct module is not the vendored snapshot.")
        if Path(config_module.__file__).resolve().parent != vendor:
            raise RuntimeError("Imported TransAct config is not the vendored snapshot.")
        config = config_module.TransActConfig(
            seq_len=sequence_length,
            # Inputs are Unix seconds, so the numerical window is also seconds.
            time_window_ms=30 * 24 * 60 * 60,
            latest_n_emb=latest_embeddings,
            concat_candidate_emb=True,
            concat_max_pool=True,
            action_vocab=range(0, 6),
            action_emb_dim=action_dimension,
            item_emb_dim=item_dimension,
            num_layer=2,
            nhead=2,
            dim_feedforward=128,
        )
        self.sequence_length = sequence_length
        self.item = nn.Embedding(items, item_dimension, padding_idx=0)
        self.user = nn.Embedding(users, 16)
        self.transact = transact_module.TransAct(config)
        transformer_dimension = action_dimension + item_dimension * 2
        transact_output = transformer_dimension * (latest_embeddings + 1)
        self.head = nn.Sequential(
            nn.Linear(transact_output + item_dimension + 16 + 4, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )
        self.upstream_class = (
            f"{type(self.transact).__module__}.{type(self.transact).__name__}"
        )

    def forward(
        self,
        *,
        users: torch.Tensor,
        candidates: torch.Tensor,
        dense: torch.Tensor,
        histories: torch.Tensor,
        ratings: torch.Tensor,
        timestamps: torch.Tensor,
        request_times: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidate_count = candidates.shape
        history_ids = histories[:, : self.sequence_length]
        history_ratings = ratings[:, : self.sequence_length]
        history_times = timestamps[:, : self.sequence_length]
        history_ids = history_ids[:, None].expand(-1, candidate_count, -1).reshape(
            batch * candidate_count, self.sequence_length
        )
        history_ratings = history_ratings[:, None].expand(
            -1, candidate_count, -1
        ).reshape(batch * candidate_count, self.sequence_length)
        history_times = history_times[:, None].expand(
            -1, candidate_count, -1
        ).reshape(batch * candidate_count, self.sequence_length)
        flat_candidates = candidates.reshape(-1)
        candidate_embedding = self.item(flat_candidates)
        transformed = self.transact(
            action_type_seq=history_ratings,
            item_embedding_seq=self.item(history_ids),
            action_time_seq=history_times,
            request_time=request_times[:, None]
            .expand(-1, candidate_count)
            .reshape(-1),
            item_embedding=candidate_embedding,
        )
        user = self.user(users)[:, None].expand(-1, candidate_count, -1).reshape(
            batch * candidate_count, -1
        )
        features = torch.cat(
            (
                transformed,
                candidate_embedding,
                user,
                dense.reshape(batch * candidate_count, -1),
            ),
            dim=-1,
        )
        return self.head(features).reshape(batch, candidate_count)


class RetrievalResidualRanker(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.residual_weight = nn.Parameter(torch.zeros(()))
        self.upstream_class = getattr(
            base, "model_class", getattr(base, "upstream_class", None)
        )

    def forward(
        self,
        *,
        candidate_ranks: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        prior = -torch.log(candidate_ranks.float())
        return prior + self.residual_weight * self.base(**inputs)


def make_industry_model(
    name: str,
    *,
    users: int,
    items: int,
    device: torch.device,
    seed: int,
    dimension: int = 32,
    maximum_gap_seconds: int = 1,
) -> nn.Module:
    if min(users, items, dimension) <= 0:
        raise ValueError("Model dimensions must be positive.")
    torch.manual_seed(seed)
    if name == "torchrec-dlrm":
        model: nn.Module = TorchRecSequenceRanker(
            users=users,
            items=items,
            dimension=dimension,
            dcn=False,
            device=device,
        )
    elif name == "torchrec-dcn":
        model = TorchRecSequenceRanker(
            users=users,
            items=items,
            dimension=dimension,
            dcn=True,
            device=device,
        )
    elif name == "din":
        model = DINRanker(users=users, items=items, dimension=dimension)
    elif name == "twin":
        model = TWINRanker(
            users=users,
            items=items,
            dimension=dimension,
            esu_items=20,
            maximum_gap_seconds=maximum_gap_seconds,
        )
    elif name == "transact":
        model = TransActRanker(
            users=users,
            items=items,
            item_dimension=dimension,
            action_dimension=16,
            sequence_length=50,
            latest_embeddings=4,
        )
    else:
        raise ValueError(f"Unknown industry model: {name}")
    return RetrievalResidualRanker(model).to(device)
