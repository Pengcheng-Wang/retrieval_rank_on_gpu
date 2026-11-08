from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gin
import torch
from torch.utils.data import DataLoader

import fbgemm_gpu  # noqa: F401
from data.reco_dataset import get_reco_dataset
from modeling.sequential.autoregressive_losses import LocalNegativesSampler, SampledSoftmaxLoss
from modeling.sequential.embedding_modules import LocalEmbeddingModule
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.features import movielens_seq_features_from_row
from modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
from modeling.similarity_utils import get_similarity_function


MODEL_CONFIGS = {
    "sasrec": ("SASRec", "configs/ml-1m/sasrec-sampled-softmax-n128-final.gin"),
    "hstu": ("HSTU", "configs/ml-1m/hstu-sampled-softmax-n128-final.gin"),
    "hstu-large": ("HSTU", "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin"),
}


def build_model(model_id: str, dataset) -> torch.nn.Module:
    module_type, config_path = MODEL_CONFIGS[model_id]
    gin.clear_config()
    gin.parse_config_file(config_path, skip_unknown=True)
    embedding = LocalEmbeddingModule(num_items=dataset.max_item_id, item_embedding_dim=50)
    interaction, _ = get_similarity_function(
        "DotProduct",
        query_embedding_dim=50,
        item_embedding_dim=50,
    )
    preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=211,
        embedding_dim=50,
        dropout_rate=0.2,
    )
    postprocessor = L2NormEmbeddingPostprocessor(embedding_dim=50, eps=1e-6)
    return get_sequential_encoder(
        module_type=module_type,
        max_sequence_length=200,
        max_output_length=11,
        embedding_module=embedding,
        interaction_module=interaction,
        input_preproc_module=preprocessor,
        output_postproc_module=postprocessor,
        verbose=False,
    )


def run(model_id: str, batch_size: int) -> dict[str, object]:
    dataset = get_reco_dataset("ml-1m", max_sequence_length=200, chronological=True)
    row = next(iter(DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=False)))
    model = build_model(model_id, dataset)
    features, target_ids, _ = movielens_seq_features_from_row(
        row,
        torch.device("cpu"),
        max_output_length=11,
    )
    features.past_ids.scatter_(1, features.past_lengths.view(-1, 1), target_ids)
    input_embeddings = model.get_item_embeddings(features.past_ids)
    outputs = model(
        past_lengths=features.past_lengths,
        past_ids=features.past_ids,
        past_embeddings=input_embeddings,
        past_payloads=features.past_payloads,
    )
    supervision_ids = features.past_ids
    sampler = LocalNegativesSampler(
        num_items=dataset.max_item_id,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=dataset.all_item_ids,
        l2_norm=True,
        l2_norm_eps=1e-6,
    )
    loss_fn = SampledSoftmaxLoss(
        num_to_sample=8,
        softmax_temperature=0.05,
        model=model,
    )
    mask = supervision_ids[:, 1:] != 0
    loss = loss_fn(
        lengths=features.past_lengths,
        output_embeddings=outputs[:, :-1, :],
        supervision_ids=supervision_ids[:, 1:],
        supervision_embeddings=input_embeddings[:, 1:, :],
        supervision_weights=mask.float(),
        negatives_sampler=sampler,
    )
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    result = {
        "model": model_id,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "output_shape": list(outputs.shape),
        "loss": float(loss.detach()),
        "loss_finite": math.isfinite(float(loss.detach())),
        "gradients_finite": bool(gradients_finite),
    }
    if not result["loss_finite"] or not result["gradients_finite"]:
        raise RuntimeError(f"Nonfinite CPU smoke result: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=sorted(MODEL_CONFIGS),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = [run(model_id, args.batch_size) for model_id in args.models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
