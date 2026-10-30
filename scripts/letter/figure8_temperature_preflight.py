"""Prepare and bank the seven-point Figure 8 temperature preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from figure8_temperature import (
    SWEEP_TEMPERATURES,
    assemble_preflight,
    load_plan,
    parse_gpu_pair,
    temperature_cells,
    validate_preflight,
)
from fresh_ids import file_sha256, read_json, write_json
from fresh_pipeline import Runtime, torchrun_prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("binding", "project", "control", "work", "bundle", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.batches <= 10:
        parser.error("Use 1..10 ordered preflight batches")
    plan = load_plan(args.plan.resolve())
    if args.batches != plan["preflight_batches_per_rank"]:
        raise ValueError("Figure8 preflight batch count differs from the frozen plan")
    runtime = Runtime(
        args.binding, args.project, args.control, args.work,
        gpu_indexes=parse_gpu_pair(args.gpu_pair),
    )
    if runtime.work.name != plan["work_basename"]:
        raise ValueError("Figure8 temperature preflight work namespace differs")
    stage = runtime.admit(temperature_cells()[0], args.bundle.resolve())
    audit = runtime.work / "audits" / "Instruments-full-ids-v1"
    runtime.audit(stage, audit)
    root = runtime.work / "temperature-preflight-v1"
    receipt_path = root / "receipt.json"
    readme = root / "README.md"
    readme_text = (
        "# Figure 8 temperature preflight v1\n\n"
        "Seven setup-only probes verify identical seed42 model initialization and "
        "rank-local ordered training batches for temperatures "
        "0.6, 0.7, 0.8, 0.9, 1.0, 1.1 and 1.2. No optimizer step or test "
        "evaluation is performed.\n"
    )
    if readme.is_file() and readme.read_text(encoding="utf-8") != readme_text:
        raise ValueError("Figure8 temperature preflight README is immutable")
    if not readme.exists():
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(readme_text, encoding="utf-8")
    if receipt_path.is_file():
        validate_preflight(read_json(receipt_path))
        return
    probes = root / "probes"
    for label, temperature in SWEEP_TEMPERATURES:
        if not receipt_path.is_file():
            output = probes / label
            if output.exists():
                if all((output / f"rank-{rank}.json").is_file() for rank in range(2)):
                    continue
                raise FileExistsError(
                    f"Interrupted Figure8 preflight probe is preserved: {output}"
                )
            runtime.invoke(
                torchrun_prefix(f"figure8-temperature-preflight-{label}") + [
                    runtime.scripts / "figure8_temperature_probe.py",
                    "--stage", stage,
                    "--audit", audit,
                    "--output", output,
                    "--temperature", temperature,
                    "--seed", 42,
                    "--batches", args.batches,
                    "--learning-rate", plan["learning_rate"],
                    "--epochs", plan["epochs"],
                    "--batch-size",
                    runtime.spec["datasets"]["Instruments"]["micro_batch_size"],
                    "--effective-batch", plan["effective_batch_size"],
                ],
                f"figure8-temperature-preflight-{label}",
                gpu=True,
            )
    if not receipt_path.is_file():
        receipt = assemble_preflight(
            probes, stage / "stage.json", audit / "audit.json", args.batches,
        )
        receipt.update({
            "plan_sha256": file_sha256(args.plan),
            "probe_source_sha256": file_sha256(
                runtime.scripts / "figure8_temperature_probe.py"
            ),
            "preflight_source_sha256": file_sha256(Path(__file__)),
            "gpu_indexes": list(runtime.gpu_indexes),
        })
        write_json(receipt_path, receipt)
    runtime.bank(root, "operational/temperature-preflight-v1")


if __name__ == "__main__":
    main()
