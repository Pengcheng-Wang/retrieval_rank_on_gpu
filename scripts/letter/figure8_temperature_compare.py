"""Write the immutable seven-point Figure 8 temperature comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from figure8_temperature import SWEEP_TEMPERATURES, aggregate_sweep, load_plan
from fresh_ids import file_sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan.resolve())
    arms = {}
    for value in args.arm:
        label, separator, path = value.partition("=")
        if (not separator or label in arms
                or label not in {name for name, _ in SWEEP_TEMPERATURES}):
            raise ValueError("Use each exact --arm LABEL=EVALUATION_DIRECTORY once")
        arms[label] = Path(path).resolve()
    if args.output.exists():
        raise FileExistsError("Figure8 temperature comparisons are immutable")
    result = aggregate_sweep(
        arms,
        seed=plan["bootstrap_seed"],
        replicates=plan["bootstrap_replicates"],
    )
    result.update({
        "plan_sha256": file_sha256(args.plan),
        "analysis_sha256": file_sha256(Path(__file__)),
        "numpy_version": np.__version__,
    })
    write_json(args.output, result)


if __name__ == "__main__":
    main()
