# HSTU reproduction aggregate

_Checksummed public-data quality, scale, and A100 efficiency results_

---

## ðŸ“‹ Contents

| File | Content |
| --- | --- |
| `protocol.json` | Frozen source, data, model, metric, precision, and gate contract |
| `quality.csv` | Published targets, final results, best-epoch diagnostics, wall time, memory, and tolerance decisions |
| `efficiency.csv` | Raw aggregate FP32/TF32/BF16 encoding, exact-search, end-to-end, and training-step measurements |
| `scale-probes.json` | Complete MovieLens-20M and Amazon Books audits and bounded probe outcomes |
| `summary.json` | Source-bound combined result, smoke evidence, precision fidelity, scale probes, and reviewer limitation |
| `checksums.json` | SHA-256 and byte count for every other file in this directory |

## ðŸŽ¯ Interpretation

The final MovieLens-1M SASRec, HSTU, and HSTU-large checkpoints all pass the
predeclared HR@10 and NDCG@10 reproduction tolerances. The scale probes are
separate one-epoch or 100-user execution evidence; they are not substituted
for the 101-epoch primary quality result.

The human-readable analysis is in
[`docs/hstu-results.md`](../../docs/hstu-results.md). Raw datasets,
checkpoints, access details, and operational logs are intentionally excluded
from Git.
