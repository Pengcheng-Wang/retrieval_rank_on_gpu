from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dimension: int = 128) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def forward(self, users: torch.Tensor, items: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(-1)


class MLPRanker(nn.Module):
    def __init__(self, users: int, items: int, dimension: int = 32) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.network = nn.Sequential(
            nn.Linear(dimension * 3 + 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        user, item = self.user(users), self.item(items)
        return self.network(torch.cat((user, item, user * item, dense), dim=1)).squeeze(-1)


class TorchRecRanker(nn.Module):
    def __init__(self, users: int, items: int, dimension: int, dcn: bool, device: torch.device) -> None:
        super().__init__()
        from torchrec.models.dlrm import DLRM, DLRM_DCN
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
        from torchrec.modules.embedding_modules import EmbeddingBagCollection

        collection = EmbeddingBagCollection(tables=[
            EmbeddingBagConfig(name="user_table", embedding_dim=dimension, num_embeddings=users, feature_names=["user"]),
            EmbeddingBagConfig(name="item_table", embedding_dim=dimension, num_embeddings=items, feature_names=["item"]),
        ], device=device)
        arguments = {
            "embedding_bag_collection": collection,
            "dense_in_features": 2,
            "dense_arch_layer_sizes": [64, dimension],
            "over_arch_layer_sizes": [128, 64, 1],
            "dense_device": device,
        }
        self.model = DLRM_DCN(**arguments, dcn_num_layers=2, dcn_low_rank_dim=32) if dcn else DLRM(**arguments)
        self.to(device)
        self.model_class = f"{type(self.model).__module__}.{type(self.model).__name__}"

    def forward(self, users: torch.Tensor, items: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        count = len(users)
        sparse = KeyedJaggedTensor(
            keys=["user", "item"], values=torch.cat((users, items)),
            lengths=torch.ones(2 * count, dtype=torch.long, device=users.device),
            stride=count, length_per_key=[count, count], offset_per_key=[0, count, 2 * count],
            index_per_key={"user": 0, "item": 1},
        )
        return self.model(dense, sparse).squeeze(-1)


def make_model(
    name: str, users: int, items: int, device: torch.device, seed: int,
    mf_dimension: int = 128, rank_dimension: int = 32,
) -> nn.Module:
    if min(users, items, mf_dimension, rank_dimension) <= 0:
        raise ValueError("Model dimensions must be positive.")
    torch.manual_seed(seed)
    if name == "mf":
        return MatrixFactorization(users, items, mf_dimension).to(device)
    if name == "mlp":
        return MLPRanker(users, items, rank_dimension).to(device)
    if name in ("torchrec-dlrm", "torchrec-dcn"):
        return TorchRecRanker(users, items, rank_dimension, name.endswith("dcn"), device)
    raise ValueError(f"Unknown ranker: {name}")
