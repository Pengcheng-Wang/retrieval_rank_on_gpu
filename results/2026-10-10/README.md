# Initial GPU retrieval study: numerical artifacts

_Sanitized results from the completed 2026-10-10 study; no allocation access or per-user MovieLens records._

---

## 📋 Read these first

| Artifact | Content |
| --- | --- |
| [Interpretation](../../docs/results.md) | Measured findings, adoption implications, and limitations |
| [Operating-point tables](../../docs/measured-tables.md) | Generated representative comparisons |
| [Protocol](../../docs/benchmark-protocol.md) | Dataset, timing, exactness, model, and fairness contracts |
| `summary.json` | Aggregate ANN/stage/pipeline groups, environments, source hashes, learned-model summaries |
| `quality.json` | Aggregate held-out metrics for 12 selected model/seed runs |
| `upstream-audit.json` | Reproduced Bloom, INT8, and mask-convention hazards |
| `pipeline-profile.json` | Verified-equivalent sequential versus batched exact filtering |
| `files.json` | SHA-256 for every numerical/report artifact in this directory |

## 📊 Full sweeps and raw evidence

`ann.csv`, `stages.csv`, `filter.csv`, `pipeline.csv`, `rank-timing.csv`, and
`trained_ann.csv` expose the complete measured sweeps. The corresponding
`*-raw.jsonl.gz` files preserve individual latency samples and original
configuration/repeat records.

`unsupported` means the requested classic FAISS GPU configuration is outside
its declared k/nprobe limit. `superseded` preserves an earlier, slower or
less appropriate adapter/control measurement but excludes it from current
comparison tables. Neither status is converted into a successful result.

Every latency is a batch duration in **milliseconds**. Derived throughput
is sequential offline queries per second, not serving QPS under a tail-latency
SLO. `probes` is IVF `nprobe`; for HNSW, inspect `effort_parameter` and
`effective_effort` instead.

The learned-model summary strips user/item targets and per-user IDs. MovieLens
source/prepared data and model checkpoints are not distributed in this public
results folder. Synthetic per-query geometric recalls are not user engagement
records.

## 🔁 Regenerate the publication

After collecting complete private numerical journals into an ignored directory:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\publish_results.py `
  --input .local\collected `
  --output results\2026-10-10 `
  --tables docs\measured-tables.md
```

The publisher rejects incomplete experiment folders and unsafe identifier
fields. It produces compressed sanitized journals, CSVs, metadata, and
checksummed Markdown tables; it does not read credentials or contact AML.
