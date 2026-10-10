from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "third_party" / "silvertorch-source-sha256.json").read_text())
    vendor = root / "third_party" / "silvertorch"
    failures = []
    for record in manifest["files"]:
        path = vendor.joinpath(*record["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(str(path.relative_to(root)))
    if failures:
        raise SystemExit("Upstream snapshot differs from its pinned source manifest: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} unchanged upstream files at {manifest['commit']}.")


if __name__ == "__main__":
    main()
