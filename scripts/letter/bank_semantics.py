"""Export sealed semantic chunks independently of the GPU encoding processes."""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
from pathlib import Path

from bank import copy_artifact
from encode import read_chunk
from protocol import file_sha256, require_hash, write_json


def ready_artifacts(output: Path) -> tuple[list[tuple[Path, Path]], bool]:
    if not (output / "run-config.json").is_file() or not (output / "token-audit.json").is_file():
        return [], False
    config = json.loads((output / "run-config.json").read_text())
    audit = json.loads((output / "token-audit.json").read_text())
    if (config["dataset"] not in ("Instruments", "Beauty", "Yelp")
            or config["kind"] != "semantic-encoding-compatible-backbone"
            or config["track_id"] != "letter-openllama7b-v2-method-replication"
            or config["smoke_items"] is not None or audit["status"] != "pass"):
        raise ValueError("Semantic exporter requires a full, audited compatible-backbone run")
    identity = file_sha256(output / "run-config.json")
    ready = []
    for path in sorted((output / "chunks").glob("items-*")):
        if not (path / "completed.json").is_file():
            continue
        match = re.fullmatch(r"items-(\d+)-(\d+)", path.name)
        if match is None:
            raise ValueError("Malformed sealed semantic chunk name")
        first, stop = map(int, match.groups())
        if (first < 0 or first % config["chunk_size"] != 0
                or stop != min(first + config["chunk_size"], audit["items"]) or first >= stop):
            raise ValueError("Semantic chunk falls outside its declared catalog partition")
        read_chunk(path, identity, first, stop)
        ready.append((Path(config["dataset"]) / "chunks" / path.name, path))
    complete = output / "completed.json"
    if not complete.is_file():
        return ready, False
    receipt = json.loads(complete.read_text())
    if (receipt["status"] != "complete" or receipt["smoke"] is not False
            or receipt["shape"] != [audit["items"], 4096]):
        raise ValueError("The completed semantic bank does not cover its full catalog")
    require_hash(output / "embeddings.npy", receipt["embeddings_sha256"])
    require_hash(output / "embedding-receipt.json", receipt["embedding_receipt_sha256"])
    embedding = json.loads((output / "embedding-receipt.json").read_text())
    if embedding["run_config_sha256"] != identity or embedding["track_id"] != config["track_id"]:
        raise ValueError("Complete semantic bank has mismatched model/recipe lineage")
    expected_chunks = (audit["items"] + config["chunk_size"] - 1) // config["chunk_size"]
    if len(ready) != expected_chunks:
        raise ValueError("A completed semantic bank is missing sealed chunks")
    ready.append((Path(config["dataset"]) / "complete", output))
    return ready, True


def watch(args) -> None:
    import fcntl

    if args.writer_host != "node-0" or socket.gethostname() != args.writer_host:
        raise RuntimeError("Only node0 may run the semantic durable exporter")
    banked = set()
    print("ASYNCHRONOUS_SEMANTIC_EXPORTER_READY", flush=True)
    while True:
        all_done, count = True, 0
        for output in args.outputs:
            ready, complete = ready_artifacts(output)
            all_done = all_done and complete
            for relative, source in ready:
                marker_sha = file_sha256(source / "completed.json")
                identity = (str(relative), marker_sha)
                if identity in banked:
                    continue
                args.lock_file.parent.mkdir(parents=True, exist_ok=True)
                with args.lock_file.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    copy_artifact(source, args.destination / relative)
                    if relative.name == "complete":
                        write_json(args.ack_root / f"{relative.parts[0]}.semantic.json", {
                            "status": "banked", "completed_sha256": marker_sha,
                            "run_config_sha256": file_sha256(source / "run-config.json"),
                        })
                banked.add(identity)
                count += 1
                print(f"BANKED_SEMANTIC_ASYNC {relative}", flush=True)
        if all_done:
            print("ALL_SEMANTIC_BANKS_DURABLY_VERIFIED", flush=True)
            return
        print(f"SEMANTIC_EXPORT_PENDING new_artifacts={count}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", type=Path, required=True)
    for name in ("destination", "lock-file", "ack-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--writer-host", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 30 or len(set(args.outputs)) != len(args.outputs):
        parser.error("Use a minimum30-second interval and unique output roots")
    watch(args)


if __name__ == "__main__":
    main()
