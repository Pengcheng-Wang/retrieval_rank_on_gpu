# Independent data-seed replication

_Bounded follow-up milestone, 2026-10-10. Initial artifacts remain immutable._

---

## 📋 Protocol and scope

The original seed17 is a read-only reference. Seeds23 and42 regenerate both the clustered catalog/query vectors and their KMeans partition. Each IVF arm shares that seed's fixed partition. This measures joint data/index sensitivity, not held-out recommendation quality or Meta production serving QPS.

1M normalized 128D items, 1,024 clusters, 128 queries, top-2048, probes 8/24/64/256, batch16, 20 warmups, 50 timed samples, three repeats, 32 CPU threads; TF32 disabled.

## 🎯 Findings

At the fixed 64-probe anchor, the FAISS-GPU/SilverTorch-FP32 latency ratio ranges from 3.052x to 3.131x. SilverTorch FP32 geometric recall ranges from 0.99992 to 0.99995.

The INT8-global 64-probe recalls range from 0.7442 to 0.7679; rowwise INT8 ranges from 0.8336 to 0.8414. Both INT8 arms lose geometric recall relative to FP32 in all three data/index seeds.

## 📊 Fixed 64-probe anchor

| Seed | Backend | Mean batch ms | Repeat SD ms | Geometric recall |
| --- | --- | --- | --- | --- |
| 17 | faiss-ivf-cpu | 9.5958 | 1.1224 | 0.9999 |
| 17 | faiss-ivf-gpu | 3.1485 | 0.0136 | 1.0000 |
| 17 | silver-fp16 | 0.9605 | 0.1007 | 0.8035 |
| 17 | silver-fp32 | 1.0057 | 0.0058 | 0.9999 |
| 17 | silver-int8-global | 1.1074 | 0.0032 | 0.7442 |
| 17 | silver-int8-rowwise | 1.1441 | 0.0061 | 0.8364 |
| 23 | faiss-ivf-cpu | 12.5600 | 0.9068 | 1.0000 |
| 23 | faiss-ivf-gpu | 3.2160 | 0.1322 | 1.0000 |
| 23 | silver-fp16 | 0.8678 | 0.0044 | 0.8071 |
| 23 | silver-fp32 | 1.0486 | 0.0088 | 1.0000 |
| 23 | silver-int8-global | 1.1042 | 0.0030 | 0.7679 |
| 23 | silver-int8-rowwise | 1.1647 | 0.0010 | 0.8414 |
| 42 | faiss-ivf-cpu | 10.4502 | 1.2886 | 0.9999 |
| 42 | faiss-ivf-gpu | 3.1602 | 0.0292 | 1.0000 |
| 42 | silver-fp16 | 0.8636 | 0.0082 | 0.8019 |
| 42 | silver-fp32 | 1.0354 | 0.0392 | 0.9999 |
| 42 | silver-int8-global | 1.1772 | 0.1270 | 0.7495 |
| 42 | silver-int8-rowwise | 1.1548 | 0.0070 | 0.8336 |

## ⚖️ Paired FP32 comparisons

| Seed | FAISS-GPU / SilverTorch latency | FAISS-CPU / SilverTorch latency | FAISS recall | SilverTorch recall |
| --- | --- | --- | --- | --- |
| 17 | 3.131x | 9.541x | 1.00000 | 0.99994 |
| 23 | 3.067x | 11.977x | 0.99999 | 0.99995 |
| 42 | 3.052x | 10.093x | 0.99999 | 0.99992 |

## 🔬 Across-seed variation

These standard deviations are across three seed-level means, not the within-run sample confidence intervals or serving-tail uncertainty.

| Backend | Mean batch ms | Data-seed SD ms | Mean recall | Data-seed recall SD |
| --- | --- | --- | --- | --- |
| faiss-ivf-cpu | 10.8686 | 1.5258 | 0.9999 | 0.00002 |
| faiss-ivf-gpu | 3.1749 | 0.0360 | 1.0000 | 0.00000 |
| silver-fp16 | 0.8973 | 0.0547 | 0.8041 | 0.00268 |
| silver-fp32 | 1.0299 | 0.0220 | 0.9999 | 0.00002 |
| silver-int8-global | 1.1296 | 0.0413 | 0.7538 | 0.01244 |
| silver-int8-rowwise | 1.1546 | 0.0103 | 0.8371 | 0.00395 |

## 🎯 Recall-constrained probe grid

Each row chooses the fastest tested probe count meeting a geometric-recall target. The coarse grid can overdeliver recall; these are constraints, not an exact continuous iso-recall fit.

| Seed | Backend | Target | Chosen probes | Measured recall | Batch ms |
| --- | --- | --- | --- | --- | --- |
| 17 | faiss-ivf-cpu | 0.8 | 24 | 0.9999 | 3.3974 |
| 17 | faiss-ivf-gpu | 0.8 | 24 | 0.9999 | 1.4864 |
| 17 | silver-fp32 | 0.8 | 24 | 0.9999 | 0.8136 |
| 17 | silver-fp16 | 0.8 | 24 | 0.8035 | 0.8532 |
| 17 | silver-int8-global | 0.8 | - | not attained | - |
| 17 | silver-int8-rowwise | 0.8 | 64 | 0.8364 | 1.1441 |
| 17 | faiss-ivf-cpu | 0.95 | 24 | 0.9999 | 3.3974 |
| 17 | faiss-ivf-gpu | 0.95 | 24 | 0.9999 | 1.4864 |
| 17 | silver-fp32 | 0.95 | 24 | 0.9999 | 0.8136 |
| 17 | silver-fp16 | 0.95 | - | not attained | - |
| 17 | silver-int8-global | 0.95 | - | not attained | - |
| 17 | silver-int8-rowwise | 0.95 | - | not attained | - |
| 23 | faiss-ivf-cpu | 0.8 | 24 | 1.0000 | 3.5384 |
| 23 | faiss-ivf-gpu | 0.8 | 24 | 1.0000 | 1.4763 |
| 23 | silver-fp32 | 0.8 | 24 | 1.0000 | 0.8161 |
| 23 | silver-fp16 | 0.8 | 24 | 0.8071 | 0.8466 |
| 23 | silver-int8-global | 0.8 | - | not attained | - |
| 23 | silver-int8-rowwise | 0.8 | 64 | 0.8414 | 1.1647 |
| 23 | faiss-ivf-cpu | 0.95 | 24 | 1.0000 | 3.5384 |
| 23 | faiss-ivf-gpu | 0.95 | 24 | 1.0000 | 1.4763 |
| 23 | silver-fp32 | 0.95 | 24 | 1.0000 | 0.8161 |
| 23 | silver-fp16 | 0.95 | - | not attained | - |
| 23 | silver-int8-global | 0.95 | - | not attained | - |
| 23 | silver-int8-rowwise | 0.95 | - | not attained | - |
| 42 | faiss-ivf-cpu | 0.8 | 24 | 0.9999 | 3.4698 |
| 42 | faiss-ivf-gpu | 0.8 | 24 | 1.0000 | 1.4599 |
| 42 | silver-fp32 | 0.8 | 24 | 0.9999 | 0.8140 |
| 42 | silver-fp16 | 0.8 | 24 | 0.8019 | 0.8436 |
| 42 | silver-int8-global | 0.8 | - | not attained | - |
| 42 | silver-int8-rowwise | 0.8 | 64 | 0.8336 | 1.1548 |
| 42 | faiss-ivf-cpu | 0.95 | 24 | 0.9999 | 3.4698 |
| 42 | faiss-ivf-gpu | 0.95 | 24 | 1.0000 | 1.4599 |
| 42 | silver-fp32 | 0.95 | 24 | 0.9999 | 0.8140 |
| 42 | silver-fp16 | 0.95 | - | not attained | - |
| 42 | silver-int8-global | 0.95 | - | not attained | - |
| 42 | silver-int8-rowwise | 0.95 | - | not attained | - |

## 🛡️ Interpretation boundaries

The new runs require an empty, idle measurement GPU; the original keepalive supervisor and profile are restored afterward. Different elapsed-time windows and thermal/allocator conditions remain possible across seeds. The raw journal retains every timing sample and per-query geometric recall.

Three synthetic data seeds test repeatability of this recipe, not general recommendation-model quality or every embedding distribution. The original MovieLens checkpoints and test evaluations are unchanged; the TorchRec precision/compilation milestone is not part of this wake.

## ⚖️ Efficient high-recall operating points

Using each arm's fastest tested setting meeting recall>=0.95, the paired FP32 GPU latency ratios are:

| Seed | FAISS chosen probes | SilverTorch chosen probes | FAISS-GPU / SilverTorch latency |
| --- | --- | --- | --- |
| 17 | 24 | 24 | 1.827x |
| 23 | 24 | 24 | 1.809x |
| 42 | 24 | 24 | 1.793x |

This is the more efficient grid-constrained comparison; the larger fixed-64-probe ratio is not the fastest configuration either system can use.
