# Measured benchmark tables

_Generated from checksummed numerical journals. See results.md for interpretation._

---

## 📊 Representative ANN operating points

Batch 16, k=2048; IVF probes=64 at 1M/20M and probes=24 at 80M. HNSW uses efSearch=2048. Values are means across repeats; SD is repeat variation, not a serving-tail interval.

| Experiment | Backend | Batch ms | Repeat SD ms | ANN recall | Repeats |
| --- | --- | --- | --- | --- | --- |
| ann_1m_clustered_s17 | faiss-ivf-cpu | 9.5958 | 1.1224 | 0.9999 | 3 |
| ann_1m_clustered_s17 | faiss-ivf-gpu | 3.1485 | 0.0136 | 1.0000 | 3 |
| ann_1m_clustered_s17 | silver-fp16 | 0.9605 | 0.1007 | 0.8035 | 3 |
| ann_1m_clustered_s17 | silver-fp32 | 1.0057 | 0.0058 | 0.9999 | 3 |
| ann_1m_clustered_s17 | silver-int8-global | 1.1074 | 0.0032 | 0.7442 | 3 |
| ann_1m_clustered_s17 | silver-int8-rowwise | 1.1441 | 0.0061 | 0.8364 | 3 |
| ann_1m_cpuwheel_s17 | faiss-hnsw-cpu | 14.4437 | 0.2841 | 0.9954 | 3 |
| ann_1m_cpuwheel_s17 | faiss-ivf-cpu | 7.9995 | 0.8979 | 0.9999 | 3 |
| ann_1m_isotropic_s17 | faiss-ivf-cpu | 8.1449 | 0.0936 | 0.2489 | 3 |
| ann_1m_isotropic_s17 | faiss-ivf-gpu | 3.0607 | 0.0011 | 0.2489 | 3 |
| ann_1m_isotropic_s17 | silver-fp16 | 0.8934 | 0.0274 | 0.2489 | 3 |
| ann_1m_isotropic_s17 | silver-fp32 | 1.0038 | 0.0249 | 0.2489 | 3 |
| ann_1m_isotropic_s17 | silver-int8-global | 1.1719 | 0.1242 | 0.2489 | 3 |
| ann_1m_isotropic_s17 | silver-int8-rowwise | 1.1547 | 0.0013 | 0.2489 | 3 |
| ann_20m_clustered_s17 | faiss-ivf-cpu | 36.3222 | 1.6066 | 0.9999 | 3 |
| ann_20m_clustered_s17 | faiss-ivf-gpu | 6.7686 | 0.2067 | 1.0000 | 3 |
| ann_20m_clustered_s17 | silver-fp16 | 1.7827 | 0.0400 | 0.6376 | 3 |
| ann_20m_clustered_s17 | silver-fp32 | 2.5582 | 0.0565 | 0.9998 | 3 |
| ann_20m_clustered_s17 | silver-int8-global | 1.6012 | 0.0083 | 0.5331 | 3 |
| ann_20m_clustered_s17 | silver-int8-rowwise | 1.8055 | 0.0073 | 0.6968 | 3 |
| ann_20m_isolated_anchor_s17 | faiss-ivf-cpu | 32.9881 | 2.3897 | 0.9999 | 3 |
| ann_20m_isolated_anchor_s17 | faiss-ivf-gpu | 6.7787 | 0.2100 | 1.0000 | 3 |
| ann_20m_isolated_anchor_s17 | silver-fp32 | 2.5336 | 0.0157 | 0.9998 | 3 |
| ann_80m_clustered_s17 | faiss-ivf-cpu | 41.0344 | 1.0902 | 0.5419 | 3 |
| ann_80m_clustered_s17 | faiss-ivf-gpu | 6.4879 | 0.1609 | 0.5419 | 3 |
| ann_80m_clustered_s17 | silver-int8-global | 2.0530 | 0.2012 | 0.3449 | 3 |

## ⚖️ Recall-matched ANN frontier

20M clustered items, batch 16, k=2048. For each arm, choose the fastest measured probe setting meeting the target. A speedup at mismatched recall is not treated as comparable.

| Recall target | Backend | Chosen probes | Measured recall | Batch ms |
| --- | --- | --- | --- | --- |
| 0.50 | faiss-ivf-cpu | 24 | 0.8156 | 24.3128 |
| 0.50 | faiss-ivf-gpu | 24 | 0.8156 | 4.7559 |
| 0.50 | silver-fp16 | 24 | 0.5804 | 1.3686 |
| 0.50 | silver-fp32 | 24 | 0.8156 | 1.8564 |
| 0.50 | silver-int8-global | 64 | 0.5331 | 1.6012 |
| 0.50 | silver-int8-rowwise | 24 | 0.6317 | 1.4697 |
| 0.80 | faiss-ivf-cpu | 24 | 0.8156 | 24.3128 |
| 0.80 | faiss-ivf-gpu | 24 | 0.8156 | 4.7559 |
| 0.80 | silver-fp16 | - | not attained | - |
| 0.80 | silver-fp32 | 24 | 0.8156 | 1.8564 |
| 0.80 | silver-int8-global | - | not attained | - |
| 0.80 | silver-int8-rowwise | - | not attained | - |
| 0.95 | faiss-ivf-cpu | 64 | 0.9999 | 32.9881 |
| 0.95 | faiss-ivf-gpu | 64 | 1.0000 | 6.7686 |
| 0.95 | silver-fp16 | - | not attained | - |
| 0.95 | silver-fp32 | 64 | 0.9998 | 2.5336 |
| 0.95 | silver-int8-global | - | not attained | - |
| 0.95 | silver-int8-rowwise | - | not attained | - |

## 🔬 Isolated scoring and scratch

20M items, batch 16, probes=64. Query quantization, centroid probing, and top-k are excluded.

| Representation | Scoring implementation | Batch ms | Incremental PyTorch peak MiB |
| --- | --- | --- | --- |
| fp16 | fused-native | 1.1710 | 70.6 |
| fp32 | fused-native | 2.0004 | 93.2 |
| int8-global | fused-native | 0.7459 | 93.3 |
| fp16 | torch-gather-bmm | 5.4725 | 1504.6 |
| fp32 | torch-gather-bmm | 6.4203 | 2969.5 |

## 🛡️ Exact filtering versus Bloom approximation

10M items, positive query `1:1`, Bloom multiplier 5. Raw Bloom is approximate; the exact row pays verification cost.

| Backend | Mean batch ms | Output false positives | Output false negatives |
| --- | --- | --- | --- |
| exact-inverted-cpu | 1.8293 | 0 | 0 |
| exact-forward-cpu | 8.8507 | 0 | 0 |
| exact-forward-gpu | 0.6171 | 0 | 0 |
| silver-bloom-packed | 0.1235 | 479 | 0 |
| silver-bloom-exact-bool | 0.7750 | 0 | 0 |

## 🧠 Trained full-catalog recommendation

Same 2,000 users, three training seeds, validation-selected checkpoints. SD below is across seeds.

| Model | Seeds | NDCG@10 | Seed SD | Recall@10 | Recall@100 | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| mf | 3 | 0.03079 | 0.00103 | 0.06483 | 0.31783 | 1247488 |
| mlp | 3 | 0.02911 | 0.00089 | 0.05717 | 0.30900 | 332865 |
| torchrec-dcn | 3 | 0.03098 | 0.00073 | 0.06317 | 0.32017 | 347361 |
| torchrec-dlrm | 3 | 0.03060 | 0.00197 | 0.06133 | 0.32450 | 327073 |

## 🔗 Complete local pipeline runtime

1M synthetic items, 512 candidates, probes=24, batch=16. Reranker weights are random; these are runtime and candidate-geometry results only.

| ANN/filter backend | Actual reranker | Batch ms | Candidate recall | Fill |
| --- | --- | --- | --- | --- |
| silver-fp32 | mlp | 2.6169 | 0.9999 | 1.0000 |
| silver-fp32 | torchrec-dcn | 3.0822 | 0.9999 | 1.0000 |
| silver-fp32 | torchrec-dlrm | 2.9967 | 0.9999 | 1.0000 |
| silver-int8-global | mlp | 2.8775 | 0.6935 | 1.0000 |
| silver-int8-global | torchrec-dcn | 3.3543 | 0.6935 | 1.0000 |
| silver-int8-global | torchrec-dlrm | 3.3211 | 0.6935 | 1.0000 |
| silver-int8-global-full | mlp | 2.7964 | 0.6935 | 1.0000 |
| silver-int8-global-full | torchrec-dcn | 3.2818 | 0.6935 | 1.0000 |
| silver-int8-global-full | torchrec-dlrm | 3.1823 | 0.6935 | 1.0000 |
| faiss-ivf-cpu | mlp | 9.6673 | 1.0000 | 1.0000 |
| faiss-ivf-cpu | torchrec-dcn | 9.7808 | 1.0000 | 1.0000 |
| faiss-ivf-cpu | torchrec-dlrm | 9.8738 | 1.0000 | 1.0000 |
| faiss-ivf-gpu | mlp | 2.0343 | 1.0000 | 1.0000 |
| faiss-ivf-gpu | torchrec-dcn | 2.2184 | 1.0000 | 1.0000 |
| faiss-ivf-gpu | torchrec-dlrm | 2.1465 | 1.0000 | 1.0000 |

## 🧮 Paper-sized filtering adaptation

40M synthetic items with six two-valued attributes per item, one query per batch. Multipliers control adaptive Bloom width; this is not the private production attribute distribution.

| Multiplier | Backend | Batch ms | Bloom index GiB | Output false positives |
| --- | --- | --- | --- | --- |
| 8.6 | exact-inverted-cpu | 7.0978 | 2.403 | 0 |
| 8.6 | exact-forward-cpu | 69.0867 | 2.403 | 0 |
| 8.6 | exact-forward-gpu | 2.5410 | 2.403 | 0 |
| 8.6 | silver-bloom-packed | 0.1505 | 2.403 | 619 |
| 8.6 | silver-bloom-exact-bool | 2.5850 | 2.403 | 0 |
| 17.1 | exact-inverted-cpu | 7.2707 | 4.778 | 0 |
| 17.1 | exact-forward-cpu | 60.3538 | 4.778 | 0 |
| 17.1 | exact-forward-gpu | 2.5286 | 4.778 | 0 |
| 17.1 | silver-bloom-packed | 0.1574 | 4.778 | 0 |
| 17.1 | silver-bloom-exact-bool | 2.6590 | 4.778 | 0 |

## 🔧 Eligibility root-cause profile

1M items, batch 16, probes=24. Both exact-verification implementations were checked to return identical masks; these stage timings exclude unrelated work.

| Stage | Mean batch ms |
| --- | --- |
| fused_partial_scoring | 1.0472 |
| sequential_exact_verification | 11.1613 |
| batched_exact_verification | 1.7737 |
| complete_retrieval_filter | 3.6662 |

## 🎯 Frozen learned ANN-to-rerank quality

MovieLens, 500 candidates, all 64 clusters probed. This isolates representation/filtering changes from cluster approximation; metrics average the three frozen training seeds.

| ANN backend | Frozen reranker | Geometric candidate recall | Target in candidates | NDCG@10 |
| --- | --- | --- | --- | --- |
| faiss-ivf-cpu | mf | 1.0000 | 0.7028 | 0.03079 |
| faiss-ivf-cpu | mlp | 1.0000 | 0.7028 | 0.02916 |
| faiss-ivf-cpu | torchrec-dcn | 1.0000 | 0.7028 | 0.03118 |
| faiss-ivf-cpu | torchrec-dlrm | 1.0000 | 0.7028 | 0.03086 |
| faiss-ivf-gpu | mf | 1.0000 | 0.7028 | 0.03079 |
| faiss-ivf-gpu | mlp | 1.0000 | 0.7028 | 0.02916 |
| faiss-ivf-gpu | torchrec-dcn | 1.0000 | 0.7028 | 0.03118 |
| faiss-ivf-gpu | torchrec-dlrm | 1.0000 | 0.7028 | 0.03086 |
| silver-fp16 | mf | 0.9992 | 0.7028 | 0.03079 |
| silver-fp16 | mlp | 0.9992 | 0.7028 | 0.02916 |
| silver-fp16 | torchrec-dcn | 0.9992 | 0.7028 | 0.03118 |
| silver-fp16 | torchrec-dlrm | 0.9992 | 0.7028 | 0.03086 |
| silver-fp32 | mf | 1.0000 | 0.7028 | 0.03079 |
| silver-fp32 | mlp | 1.0000 | 0.7028 | 0.02916 |
| silver-fp32 | torchrec-dcn | 1.0000 | 0.7028 | 0.03118 |
| silver-fp32 | torchrec-dlrm | 1.0000 | 0.7028 | 0.03086 |
| silver-int8-global | mf | 0.9938 | 0.7027 | 0.03079 |
| silver-int8-global | mlp | 0.9938 | 0.7027 | 0.02916 |
| silver-int8-global | torchrec-dcn | 0.9938 | 0.7027 | 0.03118 |
| silver-int8-global | torchrec-dlrm | 0.9938 | 0.7027 | 0.03086 |
| silver-int8-rowwise | mf | 0.9957 | 0.7037 | 0.03079 |
| silver-int8-rowwise | mlp | 0.9957 | 0.7037 | 0.02916 |
| silver-int8-rowwise | torchrec-dcn | 0.9957 | 0.7037 | 0.03122 |
| silver-int8-rowwise | torchrec-dlrm | 0.9957 | 0.7037 | 0.03082 |

## 📦 Numerical artifacts

The `results/2026-10-10` directory contains full sweep CSVs, sanitized compressed raw journals, environment metadata, and source checksums. Unsupported FAISS configurations remain explicit in raw ANN rows.
