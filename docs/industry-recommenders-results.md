# Measured industrial-ranker extension

## Result boundary

Every number in this document comes from this repository's own runs on the
authorized A100 allocation. Paper and production values appear only in the
[research report](industry-recommenders-study.md) as cited context.

The completed HSTU/SASRec retrieval results remain unchanged. This extension
tests candidate rankers on a separate fixed pool; it does not merge retrieval
and ranking architectures into one misleading leaderboard.

## Executed experiment

| Item | Measured/frozen value |
|---|---:|
| Users | 6,040 |
| Events | 1,000,209 |
| Catalog rows | 3,883 |
| Training positions | 50,000 |
| Validation queries | 1,000 last events from a frozen user partition |
| Test queries | 5,040 last events from complementary users |
| Maximum history | 200 |
| Training candidates | 1 positive + 32 exact HSTU-large hard negatives |
| Ranking candidates | first 200 from epoch-100 HSTU-large |
| Seeds | 17, 42 |
| Learned epochs | 1-10, with epoch 0 as the frozen retrieval prior |
| GPU | NVIDIA A100-SXM4-80GB |
| Quality precision | FP32 parameters with TF32 CUDA matrix multiplication |

The training hard-negative artifact contains 50,000 contexts. The observed
target appeared in HSTU-large's exact top 200 for 87.978% of them. The target
was removed from every negative list and supplied only as the supervised
positive.

The model identity receipts confirm actual
`torchrec.models.dlrm.DLRM`, actual `torchrec.models.dlrm.DLRM_DCN`, and the
unchanged upstream `transact.TransAct` module. DIN and TWIN are independent
paper-derived implementations because no reusable licensed official release
was available.

## Retrieval ceiling

The held-out target was never inserted into validation or test candidates.

| Retriever/list | Test coverage@50 | @100 | @200 | @500 |
|---|---:|---:|---:|---:|
| HSTU-large fixed pool | 59.21% | 70.18% | 78.75% | 88.23% |
| HSTU | 57.02% | 67.68% | 77.26% | 87.58% |
| SASRec | 54.68% | 65.65% | 75.10% | 85.62% |
| Three-way interleave diagnostic | 58.27% | 69.74% | 78.45% | 88.10% |

The initial 100-user gate proposed the fixed-width three-way interleave. The
gate rejected it because truncation can discard items from HSTU-large's own
depth. The failed construction is preserved as a diagnostic; all rankers use
the stronger, simpler HSTU-large list.

HSTU-large was already selected in the completed retrieval study on this same
MovieLens last-event benchmark. The ranker validation/test users are disjoint,
but this extension is not an untouched test of retriever-family selection.

At the ranking cap of 200, **21.25% of test targets are unreachable by every
ranker**. Conditional metrics must not hide that ceiling.

## Residual validation gate

Every ranker predicts a residual over
\(-\log(\text{HSTU-large retrieval rank})\). The residual scale is initialized
to zero, so epoch 0 exactly reproduces the frozen candidate order. A learned
checkpoint is admitted only if it improves validation NDCG@10 over epoch 0.

| Model | Epoch-0 validation NDCG@10 | Best learned NDCG@10 | Difference | Passed seeds |
|---|---:|---:|---:|---:|
| TorchRec DLRM | 0.18151 | 0.12420 | -0.05730 | 0/2 |
| TorchRec DLRM_DCN | 0.18151 | 0.13568 | -0.04583 | 0/2 |
| DIN | 0.18151 | 0.15241 | -0.02910 | 0/2 |
| TWIN-style | 0.18151 | 0.15444 | -0.02707 | 0/2 |
| Official TransAct module + head | 0.18151 | 0.17886 | -0.00265 | 0/2 |

**No learned ranker passed the frozen validation gate.** TransAct came closest,
but both seeds still underperformed the retrieval prior. Therefore every
selected checkpoint is epoch 0, and no learned ranker receives a confirmatory
test claim.

## Selected test quality

Because all five arms selected epoch 0, their accepted test output is exactly
the frozen HSTU-large order:

| Metric | Selected value |
|---|---:|
| HR@10 | 0.32619 |
| NDCG@10 | 0.18660 |
| Conditional HR@10 | 0.41421 |
| Conditional NDCG@10 | 0.23695 |
| HR@50 | 0.59206 |
| NDCG@50 | 0.24574 |
| Candidate coverage / HR@200 | 0.78750 |

This is a null adoption result, not a tie among five successful rankers. The
correct decision under this protocol is **do not add a learned ranking stage**.
The experiments show that industrial pedigree and additional model structure
do not guarantee a cascade-level gain when public data lacks the feature,
label, and traffic signals used by the production systems.

## A100 structure cost

The following measurements execute each full model even though its learned
residual failed admission. They quantify counterfactual architecture cost, not
the cost of the selected epoch-0 system, which can skip the ranker entirely.

Inference is synchronized batch `16 x 200` candidate scoring. The common input
is a padded 200-event history; TransAct consumes its configured most-recent 50
events. Values are two-seed means from 50 independent samples after 10 warmups.
These are offline batch measurements, not serving QPS or p99.

| Model | FP32 mean | FP32 p95 | FP32 scores/s | Peak allocated | Training step |
|---|---:|---:|---:|---:|---:|
| TorchRec DLRM | **0.928 ms** | 0.941 ms | 3.45M | 30.4 MiB | **3.458 ms** |
| TWIN-style | 1.031 ms | 1.055 ms | 3.10M | 41.8 MiB | 3.922 ms |
| TorchRec DLRM_DCN | 1.097 ms | 1.132 ms | 2.92M | 34.2 MiB | 4.197 ms |
| DIN | 2.996 ms | 3.015 ms | 1.07M | 646.2 MiB | 5.392 ms |
| TransAct | 10.771 ms | 10.911 ms | 0.30M | 384.0 MiB | 13.710 ms |

Training-step timing uses batch `64 x 33` candidates, 3 warmups, and 20
independent synchronized samples. DIN's explicit candidate-by-history
activation expansion consumed the most FP32 inference memory. TransAct had the
largest measured training allocation at 997.2 MiB.

### Complete-run wall time

These bounded receipts include data load, epoch-0 validation, 10 learned
epochs, per-epoch selection validation, final validation/test, efficiency
measurements, checkpointing, and serialized durable publication:

| Model | Mean complete run |
|---|---:|
| TorchRec DLRM | 48.5 s |
| TWIN-style | 57.5 s |
| TorchRec DLRM_DCN | 69.0 s |
| DIN | 80.0 s |
| TransAct | 150.5 s |

Runs were concurrent across separate A100s, so these rows are per-run wall time
and must not be summed into allocation wall time.

## Precision result

BF16 timing was not uniformly faster:

| Model | BF16 mean | Change versus FP32 |
|---|---:|---:|
| DLRM | 1.131 ms | 21.8% slower |
| DLRM_DCN | 1.398 ms | 27.4% slower |
| TWIN-style | 1.273 ms | 23.5% slower |
| DIN | 2.043 ms | 31.8% faster |
| TransAct | 5.144 ms | 52.2% faster |

Top-10 overlap is 1.0 because the selected residual scale is zero and the
retrieval prior determines the output. It is therefore **not** evidence that a
learned BF16 ranker is quality-equivalent to FP32.

## Preserved failures and corrections

1. A fixed-width three-retriever interleave reduced candidate coverage and was
   rejected before final ranker training.
2. The first full matrix decompressed compressed NPZ arrays on every batch.
   GPUs stayed mostly idle and no epoch completed. The attempt logs were
   preserved and the loader was fixed to materialize each array once.
3. A complete matrix was rejected because second-last-event validation labels
   had been observed by the official HSTU autoregressive training split. The
   full experiment was rerun on disjoint last-event user partitions.
4. A complete random-negative matrix was rejected because its easy catalog
   negatives did not match hard HSTU candidates at evaluation.
5. A complete hard-negative matrix was preserved as a diagnostic after review
   found that it discarded the strong retrieval score and forced an
   uninitialized ranker to replace the baseline order.
6. An initial residual matrix searched only 64 training candidates and assigned
   rank 65 to every missing target. It was preserved and replaced with exact
   top-200 ranks matching evaluation depth.
7. The final residual protocol admitted epoch 0 and reran all 10 model/seed
   cells. Every learned arm failed the validation expansion gate.
8. The first aggregate transfer was rejected after local hash/JSON validation
   detected SSH debug lines interleaved into the base64 stream. The final
   archive was filtered, archive-hash checked, file-hash checked, and JSON
   parsed before replacing the local copy.

## What this says about industrial practice

The deep-research answer and the measured answer are complementary:

- Industry does not use a single universal “SOTA recommender.” Retrieval,
  candidate-aware ranking, realtime sequence modeling, long-term user
  representation, online training, and policy layers solve different
  problems.
- Official code should be preferred where released. That is why the executed
  arms use real TorchRec DLRM/DCN and unchanged TransAct.
- More modern structure did not automatically win when its production inputs
  and scale were removed.
- On this matched public task, no learned residual earned the right to add
  latency after the retrieval score was preserved as the null baseline.

## Reproducible evidence

The checksummed package is
[`results/2026-11-08-industry-ranking`](../results/2026-11-08-industry-ranking/README.md).
It includes every per-seed result, raw independent timing sample, learning
gate, training curve, run-wall receipt, candidate ceiling, hard-negative
receipt, protocol, implementation identity, and SHA-256 manifest.

Exact GPT-6-Astra/MAX was unavailable. No substitute is labeled Astra and no
numeric one-million-token context claim is made.
