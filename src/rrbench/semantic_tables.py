from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.reporting import table


def unique_quality_point(rows: list[dict[str, Any]], **required: Any) -> dict[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in required.items())]
    if len(selected) != 1:
        raise ValueError("Report requires exactly one verified measured point for its complete selection.")
    return selected[0]


def render_report(summary: dict[str, Any], statistics: dict[str, Any]) -> str:
    if not summary["complete"] or not statistics["complete"]:
        raise ValueError("Final scientific report requires complete verified measurements and statistics.")
    quality, online, ann = summary["quality_rows"], summary["online_rows"], summary["ann"]
    content = [
        "# Real-corpus GPU retrieval and ranking results", "",
        "_Full real-catalog semantic study; source-pinned models, actual FAISS/SilverTorch/TorchRec, and GPT-6-Astra review._", "",
        "---", "",
        "## 📌 Main findings", "",
        "The study evaluates **8,841,823 real MS MARCO passages** and **1,798,875 "
        "training-observed, caption-covered KuaiRand videos**. All catalog rows were encoded "
        "and checksum-validated; the study does not substitute interaction counts or a smaller "
        "catalog for million-item retrieval.", "",
        "Faster GPU retrieval and a stronger semantic model are separate interventions. "
        "The FP32 shared-partition ANN comparison isolates implementations; native-versus-MRL "
        "exact retrieval isolates representation dimensions. Same-scorer online comparisons "
        "keep query encoder, index, weights, precision and candidate IDs fixed within each "
        "boundary control. This is an offline/local study, not a reproduction of Meta's "
        "private QPS, TCO or engagement gains.[^1]", "",
        "## 📏 Scale, model strength and protocols", "",
        table(["Track", "Real item catalog", "Primary final queries", "Strong text subset"], [
            ["Semantic IR", "8,841,823", "6,980 dev-small + 43/54 TREC", "1,000 dev-small + all 43/54 TREC"],
            ["Recommendation", "1,798,875", "2,707 conditional warm/history queries", "819 fixed users, one retained query each"],
        ]), "",
        "IR and personalized recommendation must not be conflated. The recommendation split "
        "uses whole recorded-clock-block boundary quarantine, frozen training exposure histories, "
        "and retrospective 2026 captions. Exact historical exposure/feedback availability is "
        "unestablished. Only **14.78% of all test-positive events** belong to the frozen semantic "
        "catalog; the reported conditional query means cannot be multiplied by that event "
        "coverage to manufacture unconditional recall.[^2]", "",
        "The encoder arms are MiniLM-L6 (384D), BGE-large (1024D), Qwen3-Embedding-0.6B "
        "(1024D), and Qwen3-Embedding-8B (native 4096D plus independently normalized 1024D MRL). "
        "They are trained retrievers, not arbitrary chatbot hidden-state embeddings. "
        "The pinned Qwen cards report stronger English/Chinese retrieval aggregates for the "
        "larger embedding model, but those publisher scores are not this experiment's "
        "recommendation results.[^3]", "",
        "Offline item encoding cost is amortizable and must not be added to every online "
        "request. The following cost is the **sum of worker wall time** across complete shards, "
        "not parallel elapsed time or hardware MFU; it includes loading, CPU tokenization "
        "and I/O. Brief declared correctness smokes shared owned encoder GPUs.", "",
        table(["Catalog", "Encoder", "Items", "Native D", "Summed worker hours", "Rows / worker s"], [
            [row["corpus"], row["encoder"], f"{row['items']:,}", str(row["native_dimension"]),
             f"{row['sum_worker_hours']:.2f}", f"{row['item_rows_per_worker_second']:.1f}"]
            for row in summary["encoding_rows"]
        ]), "",
        "## 🔍 Exact semantic retrieval quality", "",
        "Exhaustive FP32 inner products operate over the identical canonical FP16-stored "
        "vectors, with TF32 disabled. FP32 arithmetic does not recover precision already "
        "lost at storage. These rows have no IVF approximation confound.", "",
    ]
    rows = []
    for model, dimension in (("minilm", 384), ("bge-large", 1024), ("qwen06", 1024), ("qwen8", 1024), ("qwen8", 4096)):
        points = [
            unique_quality_point(
                quality, corpus="msmarco", encoder=model, dimension=dimension,
                stage="exact_full_catalog_retrieval", query_panel=panel,
            ) for panel in ("dev-small", "trec19", "trec20")
        ]
        rows.append([model, str(dimension), f"{points[0]['mrr@10']:.4f}",
                     f"{points[1]['ndcg@10']:.4f}", f"{points[2]['ndcg@10']:.4f}",
                     f"{points[0]['recall@1000']:.4f}"])
    content += [
        table(["Encoder", "Stored D", "Dev MRR@10", "TREC19 NDCG@10", "TREC20 NDCG@10", "Dev Recall@1000"], rows), "",
        "TREC binary recall uses grade ≥2; NDCG retains raw relevance grades. Unjudged items "
        "contribute zero gain without becoming exhaustive negative truth. Pretrained models "
        "may have seen MS MARCO; this is not untouched zero-shot generalization.[^4]", "",
        "Native 4096D is not supported by the pinned SilverTorch dispatch. It is compared "
        "with FAISS and exact retrieval over the full corpus, not silently relabeled as "
        "SilverTorch. Native FP32 storage exceeds one 80GB GPU; explicit unsupported outcomes "
        "remain in the raw matrix.", "",
        "The full native **IVF-SQhalf could not be constructed** on a clean A100 80GB, "
        "including a second probe with scratch reduced to 128MiB. Search in those GPU "
        "IVF cells is N/A, not zero quality or an implicit CPU fallback. The precise "
        "allocator/packing cause is not established. A separate classic, device-only "
        "contiguous FAISS FlatFP16 control held all 8,841,823 native vectors and executed "
        "one fixed sanity query, leaving "
        f"{summary['native_memory_controls']['flat_half_feasibility']['GPUfree_after_bytes'] / (1 << 30):.2f} GiB free. "
        "That proves layout-specific storage/search feasibility, not IVF feasibility, "
        "FP32-oracle equivalence, quality/latency-grid performance, or native LM/reranker "
        "co-fit. Raw vector bytes alone are not a sufficient deployment budget.", "",
        "## ⚙️ Actual shared-partition ANN performance", "",
        "Representative fixed operating point: 16,384 shared spherical IVF clusters, "
        "**256 probes, top-1,000**, one query. Complete search includes centroid probing, "
        "scoring, top-k and external-ID remapping. Geometry recall is overlap against "
        "exhaustive top-1,000, not relevance recall.", "",
    ]
    rows = []
    for corpus in ("msmarco", "kuairand"):
        for backend in ("faiss-cpu-fp32", "faiss-gpu-fp32", "faiss-gpu-fp16", "silver-fp32", "silver-fp16", "silver-int8-global", "silver-int8-rowwise"):
            point = unique_quality_point(ann, corpus=corpus, encoder="qwen8", dimension=1024,
                                         backend=backend, probes=256, k=1000, status="ok")
            latency = point["latency"]["1"]["latency_mean_ms"]
            rows.append([corpus, backend, f"{latency:.3f}", f"{point['geometric_recall_at_k']:.4f}"])
    content += [
        table(["Catalog", "Backend", "Mean query ms", "Geometric Recall@1000"], rows), "",
        "This finite-grid comparison does not prove a globally optimal operating point or "
        "offered-load serving QPS. INT8-global and rowwise methods are named adaptations with "
        "rescaling before top-k; compare their geometry and relevance before claiming lossless "
        "speed. The new FAISS FP16 arm uses actual half scalar codes. Earlier IVFFlat "
        "`useFloat16`-flag rows stored FP32 values and must not be cited as verified FP16 "
        "measurements; original archives are preserved.[^5]", "",
        "## 🧠 Strong rerankers and actual TorchRec", "",
        "The common pool is a label-independent, rank-interleaved union of the 0.6B and "
        "8B-MRL retrievers. Caps are applied to its ordered candidate prefix **before** "
        "reranking. Each scorer sees the same IDs; positives are never injected into final "
        "retrieval. Strong text scorers use the official BGE logit or Qwen final yes-minus-no "
        "logit grammar.[^3]", "",
    ]
    rows = []
    for architecture in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank"):
        values = [
            unique_quality_point(quality, corpus="msmarco", architecture=architecture,
                                 candidate_cap=500, stage="common_pool_text_ranker", query_panel=panel)
            for panel in ("dev-small", "trec19", "trec20")
        ]
        rows.append([architecture, f"{values[0]['mrr@10']:.4f}", f"{values[1]['ndcg@10']:.4f}", f"{values[2]['ndcg@10']:.4f}"])
    for architecture in ("semantic-mlp", "torchrec-dlrm", "torchrec-dcn"):
        values = [
            [
                unique_quality_point(quality, corpus="msmarco", architecture=architecture, seed=seed,
                                     candidate_cap=500, stage="common_pool_cached_ranker", query_panel=panel)
                for seed in (17, 23, 42)
            ] for panel in ("dev-small", "trec19", "trec20")
        ]
        rows.append([architecture + " (3-seed mean)",
                     f"{np.mean([point['mrr@10'] for point in values[0]]):.4f}",
                     f"{np.mean([point['ndcg@10'] for point in values[1]]):.4f}",
                     f"{np.mean([point['ndcg@10'] for point in values[2]]):.4f}"])
    content += [
        table(["Same common 500 candidates", "Selected dev MRR@10", "TREC19 NDCG@10", "TREC20 NDCG@10"], rows), "",
        "TorchRec arms instantiate actual DLRM and DLRM_DCN with EmbeddingBagCollection/"
        "KeyedJaggedTensor. Dense 512→128 and over-architecture 512→256→1 layers, 128D "
        "sparse embeddings, and three low-rank cross layers make them materially larger "
        "than the initial MovieLens harness. They learn from fitting-only query groups "
        "with transferable features, not query-ID shortcuts. They are nevertheless a "
        "cheaper information class than text cross-encoders; architectural efficiency does "
        "not guarantee equal model quality.[^6]", "",
        "All 819 selected recommendation users have **zero known positives in the common "
        "first-500 pool**. Reranking cannot recover an absent candidate. The corresponding "
        "zero metrics are a qrel-derived ceiling of this pipeline, not evidence that three "
        "scorers are equivalent or semantically incapable. Full 2,707-query actual-retriever/"
        "cached quality is reported separately.", "",
        "## 🏗️ One-GPU online and local cascade costs", "",
        "```mermaid", "flowchart LR",
        "    accTitle: Measured Real Catalog Pipeline",
        "    accDescr: Real raw queries are encoded online, searched against the full catalog, and scored by the same co-resident model; only the measured local boundary differs.",
        '    raw_query["Raw query / frozen history"] --> query_encoder["Pinned LM query encoder"]',
        '    query_encoder --> full_index["Full real FAISS or SilverTorch index"]',
        '    full_index --> resident_path["Resident candidate / feature path"]',
        '    full_index --> local_cascade["Measured CPU serialization / fetch / H2D"]',
        '    resident_path --> same_scorer["Same scorer weights and candidates"]',
        '    local_cascade --> same_scorer',
        '    same_scorer --> final_topk["Verified final top-k parity"]',
        "```", "",
        "Each cell holds the actual Qwen8 query encoder, full index and chosen scorer "
        "simultaneously on one A100 80GB. It measures raw query tokenization/encoding, "
        "complete ANN, actual ID/feature/text movement, scoring and final selection. "
        "The local-cascade control is in-process serialization/fetch/H2D—not invented RPC. "
        "Text needs CPU IDs/tokenization in both paths, and no transformer-kernel fusion "
        "is claimed.", "",
        "The table shows integrated versus local-cascade wall time at **batch16, cap500**. "
        "Every point has one warmup and only three measured distinct batches; all raw "
        "samples and min-max ranges are retained. These are descriptive means, not tail "
        "SLOs, confidence intervals, or demonstrated steady-state performance.", "",
    ]
    rows = []
    for corpus in ("msmarco", "kuairand"):
        for ranker in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank", "torchrec-dcn"):
            for backend in ("faiss-gpu-fp32", "silver-fp32"):
                integrated = unique_quality_point(online, corpus=corpus, ranker=ranker, backend=backend,
                                                   boundary="integrated", query_batch=16, candidate_cap=500, status="ok")
                separate = unique_quality_point(online, corpus=corpus, ranker=ranker, backend=backend,
                                                boundary="local-cascade", query_batch=16, candidate_cap=500, status="ok")
                rows.append([corpus, ranker, backend, f"{integrated['latency_mean_ms']:.1f}",
                             f"{separate['latency_mean_ms']:.1f}"])
    content += [
        table(["Catalog", "Scorer", "ANN", "Integrated batch ms", "Local cascade batch ms"], rows), "",
        "ANN-only speedups can be hidden by query encoding and heavy reranking. FAISS GPU "
        "also supports resident GPU scoring, so residency is not unique to SilverTorch. "
        "Use full-pipeline costs and the right candidate/quality policy when assessing "
        "deployment—not a kernel-only speedup. NVML startup/co-resident/post-condition memory "
        "samples include external FAISS allocations; they are not continuous peaks. "
        "No MFU is fabricated from utilization.", "",
        "## 🎯 Recommendation interpretation and diagnostic", "",
    ]
    rows = []
    for architecture in ("logged-mf", "training-positive-popularity"):
        for phase in ("validation", "test"):
            if architecture == "logged-mf":
                values = [unique_quality_point(quality, corpus="kuairand", architecture=architecture, seed=seed,
                                               phase=phase, stage="full_catalog_id_baseline") for seed in (17, 23, 42)]
                recall = np.mean([point["recall@1000"] for point in values])
            else:
                recall = unique_quality_point(quality, corpus="kuairand", architecture=architecture,
                                              phase=phase, stage="full_catalog_nonpersonalized_baseline")["recall@1000"]
            rows.append([architecture, phase, f"{recall:.6f}"])
    content += [
        table(["Full catalog ID/control baseline", "Conditional panel", "Recall@1000"], rows), "",
        "The logged MF baseline fits all 4,498,635 admitted catalog-covered training "
        "exposures, including 1,203,235 positive rows, for 983 users and the full "
        "1,798,875-item table. It has 230,381,824 parameters. Its rapidly falling training "
        "loss and weak full-catalog recall are compatible with sparse item support and "
        "pointwise observed-exposure versus full-catalog ranking mismatch. They do not "
        "establish general MF inferiority.", "",
        "One fixed, test-informed exploratory diagnostic averages all positive training "
        "exposure vectors per user, including event multiplicity. Exact test Recall@1000 "
        "is 0.0003694 versus 0.0014777 for exposure-history exact retrieval. This jointly "
        "changes label selection, history length, aggregation and query encoding; its null "
        "result cannot identify positive selection alone, nor rule out better personalized "
        "semantic models. Caption duration identity matched exactly in 864/864 valid "
        "sample pairs, versus one after permutation, strengthening ID-namespace sanity "
        "without validating every caption.", "",
        "## 🧪 Uncertainty, limitations and adoption lessons", "",
        table(["Paired exact-retrieval contrast", "Panel / metric", "Mean difference", "95% empirical query interval"], [
            [row["contrast"], row["query_panel"] + " / " + row["metric"],
             f"{row['effect']['mean_event_or_query_effect']:+.5f}",
             f"[{row['effect']['ci95_mean_event_or_query_effect'][0]:+.5f}, "
             f"{row['effect']['ci95_mean_event_or_query_effect'][1]:+.5f}]"]
            for row in statistics["comparisons"]
            if row.get("stage") == "Exact-fullcatalogFP32retrievalfromcanonicalstoredvectors"
            and row["metric"] in ("mrr@10", "ndcg@10")
            and ((row["query_panel"] == "dev-small" and row["metric"] == "mrr@10")
                 or (row["query_panel"] in ("trec19", "trec20") and row["metric"] == "ndcg@10"))
        ]), "",
        "Paired query/user resampling is computed from unchanged per-query IDs, with "
        "2,000 draws for empirical IR contrasts and 20,000 user-cluster draws for the MF/"
        "popularity comparison. Report both query-weighted and equal-user sensitivity "
        "where appropriate. Zero-width intervals from all-zero observed metrics are not "
        "proof of exact population performance. Three training seeds do not triple the "
        "independent sample count.", "",
        "The expensive text subset and reduced online sampling are transparent post-test "
        "compute-budget amendments, frozen before bulk text inference. Full catalog sizes "
        "and primary retrieval/cached query panels remain unchanged. The common-pool "
        "quality cannot be paired with single-retriever timing as one deployed point. "
        "Sparse recommendation judgments and warm-target coverage severely bound "
        "recommendation conclusions.", "",
        "For adoption, first validate semantic/personalization signal and candidate recall; "
        "then choose dimensionality/quantization against exact judged quality; only then "
        "optimize index and stage boundaries. Strong pretrained retrieval or cross-encoding "
        "cannot compensate for incorrect eligibility, poor user representations or absent "
        "candidates. TorchRec is an implementation baseline, not a universal winning ranker. "
        "The public SilverTorch code does not supply Meta's private production models.", "",
        "[Full measured CSV/JSON artifacts](../results/2026-10-10-real-semantic/summary.json) "
        "and paired statistical comparisons accompany this report. Source revisions, model "
        "cards, dataset/license contracts, input length audits and complete checksums are "
        "in the [design](semantic-study-design.md) and pinned configs.", "",
        "[^1]: Meta SilverTorch public source, paper and engineering overview. https://github.com/meta-recsys/silvertorch/tree/22c2007ed5d28ec8304de081cc2393778708ad12 ; https://doi.org/10.1145/3805712.3809755 ; https://engineering.fb.com/2026/05/26/ml-applications/silvertorch-index-as-model-new-retrieval-paradigm-recommendation-systems/",
        "[^2]: KuaiRand dataset, supplementary captions and field descriptions. https://doi.org/10.1145/3511808.3557624 ; https://zenodo.org/records/10439422 ; https://zenodo.org/records/18159199 ; https://github.com/chongminggao/KuaiRand",
        "[^3]: Qwen3 embedding/reranking family and pinned model cards. https://arxiv.org/abs/2506.05176 ; https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/README.md ; https://huggingface.co/Qwen/Qwen3-Reranker-8B/blob/77d193c791ed757ca307ee72715aa132723da912/README.md",
        "[^4]: MS MARCO usage terms and TREC guidelines. https://github.com/microsoft/msmarco/blob/master/README.md ; https://github.com/microsoft/msmarco/blob/master/TREC-Deep-Learning-2020.md",
        "[^5]: Pinned FAISS GPU cloner. https://github.com/facebookresearch/faiss/blob/v1.14.1/faiss/gpu/GpuCloner.cpp",
        "[^6]: Actual TorchRec DLRM implementation. https://github.com/meta-pytorch/torchrec/blob/v1.8.0-rc1/torchrec/models/dlrm.py",
    ]
    return "\n".join(content) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact measured semantic study tables and interpretation.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    statistics = json.loads(Path(args.statistics).read_text(encoding="utf-8"))
    text = render_report(summary, statistics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
