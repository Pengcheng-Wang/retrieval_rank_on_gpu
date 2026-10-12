# Query-only adaptation evidence

_Complete aggregate-only export of one fixed Qwen0.6 LoRA pilot._

---

The [report](../../docs/query-adaptation-results.md) explains the mixed
head-rank, Recall@1000 and runtime outcomes. One seed17 epoch fits50,000
queries without changing the8,841,823-passage item bank. Its final782-update
checkpoint is the only evaluated adapter; smoke state and intermediate
checkpoints do not select a winner.

| File | Complete contents |
| --- | --- |
| `quality.csv` | All20 paired metric contrasts across validation/dev/TREC19/TREC20, including negative and inconclusive intervals |
| `timing.csv` | All18 context/path/arm records;50 measured batches and10 warmups per record |
| `timing-samples.csv` | All900 positive raw batch durations, preserving context, path, arm and sample index |
| `summary.json` | Fixed recipe, training/token counts, full precision quality effects, paired runtime summaries, sampled memory and source bindings |
| `checksums.json` | Package-byte hashes and canonical-LF scientific source/report seals |

Intervals are pointwise95% paired original-query bootstraps, not adjusted
family evidence. Every Recall@1000 point estimate is negative; validation
and TREC20 intervals exclude zero. All Recall@10/100 intervals include zero,
and TREC20's Recall@100 point estimate is positive. Previously inspected
panels and one training seed remain post-test exploratory.

Runtime compares fresh base models with attached, unmerged FP32 adapters.
Three fresh paired process contexts are not three quality replications.
Both models and the whole FP32 item bank are resident, with one upload
outside request timing. Batch1 encoding is a cost diagnostic, not a
batch1 retrieval-quality point. No serving, MFU or energy claim is made.

Memory samples describe the resident harness; administrative admission
memory is separate and is not a continuous peak. Resource drains and
original native keeper restoration were verified.

Opaque source/checkpoint/group digests preserve provenance without exporting
raw query or passage IDs, texts, model weights, credentials, allocation
access, device UUIDs, process IDs or private control paths. The private
evidence packet is not redistributed.
