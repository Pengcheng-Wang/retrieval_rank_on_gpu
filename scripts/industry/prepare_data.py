from __future__ import annotations

import argparse
import json
from pathlib import Path

from rrbench.industry_data import prepare_ranking_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=200)
    parser.add_argument("--maximum-examples", type=int, default=50000)
    parser.add_argument("--per-user-cap", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    receipt = prepare_ranking_data(
        sequence_csv=args.sequences,
        catalog_csv=args.catalog,
        output=args.output,
        max_history=args.max_history,
        maximum_examples=args.maximum_examples,
        per_user_cap=args.per_user_cap,
        negatives=args.negatives,
        seed=args.seed,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(args.receipt)


if __name__ == "__main__":
    main()
