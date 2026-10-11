from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from rrbench.process_lease import guard_worker_parent, wait_for_start_gate


def run_registered_module(
    parent: int, parent_start: int, gate: Path, owner: str, worker: int,
    module: str, arguments: list[str],
) -> None:
    guard_worker_parent(parent, parent_start)
    wait_for_start_gate(gate, owner, worker, parent, parent_start)
    original = sys.argv
    try:
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = original


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unchanged module only after owned-child startup approval.")
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--parent-start", type=int, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run_registered_module(
        args.parent, args.parent_start, args.gate, args.owner, args.worker, args.module, args.arguments,
    )


if __name__ == "__main__":
    main()
