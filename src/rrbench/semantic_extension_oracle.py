from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.runtime import configure, require_cuda
from rrbench.semantic_ann import exact_panel, validate_oracle
from rrbench.semantic_cascade import device_telemetry
from rrbench.semantic_profiling import load_sources


def source_binding(items: Path, queries: Path, panel: Path) -> dict[str, object]:
    if not panel.is_file():
        raise ValueError("Oracle query panel must exist.")
    return {
        "items_sha256": sha256(items), "queries_sha256": sha256(queries),
        "k": 1000, "product": "FP32", "tf32": False,
    }


def validate_outer_receipt(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "completed.json").read_text())
    if not receipt["complete"] or sha256(directory / "panel/completed.json") != receipt["panel_receipt_sha256"]:
        raise ValueError("Full-catalog oracle completion receipt changed.")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-bound full-catalog geometry reference for the finite extension.")
    for name in ("items", "queries", "panel", "partition", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    configure(17, 4)
    device = require_cuda()
    device_telemetry(device, require_exclusive=True)
    items, _, queries, _, _, _, _ = load_sources(SimpleNamespace(
        operation="ann", items=args.items, queries=args.queries, panel=args.panel, partition=args.partition,
    ))
    binding = source_binding(args.items, args.queries, args.panel)
    bind_run_config(args.output, binding)
    provenance = {
        "panel_sha256": sha256(args.panel), "partition_sha256": sha256(args.partition),
        "implementation_sha256": sha256(Path(__file__)),
        "exact_implementation_sha256": sha256(Path(exact_panel.__code__.co_filename)),
        "queries": len(queries), "items": len(items),
        "scope": "Whole frozen FP16 catalog, exhaustive FP32 arithmetic and TF32 disabled; not index/pipeline latency.",
    }
    provenance_path = args.output / "source-provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Extension oracle provenance changed; use a new immutable version.")
    if not provenance_path.exists():
        write_json(provenance_path, provenance)
    if (args.output / "completed.json").exists():
        validate_outer_receipt(args.output)
        validate_oracle(args.output / "panel", len(queries), len(items), 1024, 1000)
        return
    exact_panel(items, queries, args.output / "panel", 1000)
    write_json(args.output / "completed.json", {
        "complete": True, "panel_receipt_sha256": sha256(args.output / "panel/completed.json"),
    })
    validate_outer_receipt(args.output)
    validate_oracle(args.output / "panel", len(queries), len(items), 1024, 1000)
    print("EXTENSION_FULL_CATALOG_GEOMETRY_REFERENCE_VERIFIED", len(items), len(queries), flush=True)


if __name__ == "__main__":
    main()
