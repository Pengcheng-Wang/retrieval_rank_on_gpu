from pathlib import Path
import argparse

from rrbench.reporting import publish


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish sanitized, complete benchmark journals.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tables", required=True)
    args = parser.parse_args()
    result = publish(Path(args.input), Path(args.output), Path(args.tables))
    print(f"Published {len(result['ann'])} ANN groups and {len(result['quality_runs'])} trained runs.")


if __name__ == "__main__":
    main()
