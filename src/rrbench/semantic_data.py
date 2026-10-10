from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PureWindowsPath
import tarfile
import time
import urllib.request

import numpy as np

from rrbench.io import sha256, write_json

PASSAGE_COUNT = 8841823
MSMARCO_BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
SOURCES = {
    "collection.tar.gz": {"url": MSMARCO_BASE + "collection.tar.gz"},
    "queries.tar.gz": {"url": MSMARCO_BASE + "queries.tar.gz"},
    "qrels.train.tsv": {"url": MSMARCO_BASE + "qrels.train.tsv"},
    "qrels.dev.tsv": {"url": MSMARCO_BASE + "qrels.dev.tsv"},
    "collectionandqueries.tar.gz": {"url": MSMARCO_BASE + "collectionandqueries.tar.gz"},
    "KuaiRand-1K.tar.gz": {
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        "md5": "6b0b9c8222d67fcd4c676218edca3f1f", "bytes": 1135436720,
    },
    "kuairand_video_captions.csv": {
        "url": "https://zenodo.org/records/18159199/files/kuairand_video_captions.csv",
        "md5": "9fd4e7587ffd0c8dc289d013858fb229", "bytes": 3222932768,
    },
}


def download_file(directory: Path, name: str, source: dict[str, str | int]) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    target, receipt = directory / name, directory / (name + ".receipt.json")
    if receipt.exists():
        known = json.loads(receipt.read_text(encoding="utf-8"))
        if not target.exists() or sha256(target) != known["sha256"] or known["url"] != source["url"]:
            raise ValueError("Existing download differs from its recorded source receipt.")
        return known
    temporary = target.with_suffix(target.suffix + ".download")
    start = time.perf_counter()
    request = urllib.request.Request(str(source["url"]), headers={"User-Agent": "rrbench-research/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        expected = response.headers.get("Content-Length")
        digest, md5, count = hashlib.sha256(), hashlib.md5(), 0
        while block := response.read(8 << 20):
            stream.write(block)
            digest.update(block)
            md5.update(block)
            count += len(block)
        if expected and count != int(expected):
            raise ValueError("Downloaded byte count differs from HTTP Content-Length.")
    if source.get("bytes") is not None and count != source["bytes"]:
        raise ValueError("Downloaded dataset size differs from its official record.")
    if source.get("md5") is not None and md5.hexdigest() != source["md5"]:
        raise ValueError("Dataset checksum differs from the official Zenodo record.")
    temporary.replace(target)
    result = {
        "url": source["url"], "file": name, "bytes": count, "sha256": digest.hexdigest(),
        "md5": md5.hexdigest(), "download_seconds": time.perf_counter() - start,
        "external_md5_verified": "md5" in source,
    }
    write_json(receipt, result)
    print(f"DOWNLOAD_VERIFIED {name} bytes={count}", flush=True)
    return result


def safe_extract(archive: Path, destination: Path, include: set[str] | None = None) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            parts = Path(member.name).parts
            if (
                member.name.startswith(("/", "\\")) or PureWindowsPath(member.name).drive
                or ".." in parts or member.issym() or member.islnk()
            ):
                raise ValueError("Dataset archive contains an unsafe path or link.")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Dataset archive contains a non-file payload.")
            if include is not None and member.name not in include:
                continue
            filename = destination.joinpath(*parts)
            if not filename.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Dataset archive escapes its explicit destination.")
            filename.parent.mkdir(parents=True, exist_ok=True)
            payload = stream.extractfile(member)
            if payload is None:
                raise ValueError("Dataset archive member has no payload.")
            with filename.open("wb") as output:
                while block := payload.read(8 << 20):
                    output.write(block)
            names.append(member.name)
    return names


def shard_passages(collection: Path, destination: Path, rows_per_shard: int = 262144) -> dict[str, object]:
    manifest_path = destination / "corpus.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["source_sha256"] != sha256(collection):
            raise ValueError("Prepared corpus is bound to a different collection.")
        for shard in existing["shards"]:
            if sha256(destination / shard["file"]) != shard["sha256"]:
                raise ValueError("Prepared shard differs from its recorded receipt.")
        offsets_path = destination / "passage_offsets.npy"
        if not offsets_path.exists() or sha256(offsets_path) != existing["offsets_sha256"]:
            raise ValueError("Prepared passage offsets are missing or corrupted.")
        if np.load(offsets_path, mmap_mode="r").shape != (existing["rows"] + 1,):
            raise ValueError("Prepared passage-offset count differs from the corpus.")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    offsets = np.lib.format.open_memmap(
        destination / "passage_offsets.npy", mode="w+", dtype=np.uint64, shape=(PASSAGE_COUNT + 1,)
    )
    count, empty, position, shard_rows = 0, 0, 0, 0
    shard_number, output, digest, shards = -1, None, None, []
    started = time.perf_counter()
    with collection.open("rb") as stream:
        for line in stream:
            if count >= PASSAGE_COUNT:
                raise ValueError("Collection has more rows than its declared official corpus.")
            identifier, separator, text = line.partition(b"\t")
            if not separator or int(identifier) != count:
                raise ValueError("Collection IDs are not unique ordered contiguous passage rows.")
            text.decode("utf-8", errors="strict")
            if not text.strip():
                empty += 1
            offsets[count] = position
            position += len(line)
            if count % rows_per_shard == 0:
                if output is not None:
                    output.close()
                    shards[-1].update(rows=shard_rows, sha256=digest.hexdigest())
                shard_number += 1
                filename = f"corpus_{shard_number:03d}.tsv"
                output = (destination / filename).open("wb")
                digest, shard_rows = hashlib.sha256(), 0
                shards.append({"file": filename, "start": count})
            output.write(line)
            digest.update(line)
            count, shard_rows = count + 1, shard_rows + 1
    if output is not None:
        output.close()
        shards[-1].update(rows=shard_rows, sha256=digest.hexdigest())
    if count != PASSAGE_COUNT:
        raise ValueError("Collection row count differs from the official full corpus.")
    offsets[count] = position
    offsets.flush()
    result = {
        "dataset": "MS MARCO passage v1", "rows": count, "unique_ids": count,
        "id_contract": "Contiguous unique integer IDs verified for every row, starting at0.",
        "empty_text_rows": empty, "source_sha256": sha256(collection),
        "offsets_sha256": sha256(destination / "passage_offsets.npy"),
        "rows_per_shard": rows_per_shard, "shards": shards,
        "preparation_seconds": time.perf_counter() - started,
        "license": "Official noncommercial research terms; corpus not redistributed in public Git.",
    }
    write_json(manifest_path, result)
    print(f"FULL_REAL_CORPUS_VERIFIED rows={count} shards={len(shards)} empty={empty}", flush=True)
    return result


def acquire(directory: Path, msmarco_only: bool = False) -> None:
    download = directory / "downloads"
    sources = {name: source for name, source in SOURCES.items() if not msmarco_only or name.startswith(("collection", "queries", "qrels"))}
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(download_file, download, name, source) for name, source in sources.items()]
        receipts = [future.result() for future in futures]
    raw = directory / "raw"
    msmarco, kuairand = raw / "msmarco", raw / "kuairand"
    for name, destination in (
        ("collection.tar.gz", msmarco), ("queries.tar.gz", msmarco),
        ("collectionandqueries.tar.gz", msmarco), ("KuaiRand-1K.tar.gz", kuairand),
    ):
        if msmarco_only and name.startswith("KuaiRand"):
            continue
        marker = destination / (name + ".extracted.json")
        if not marker.exists():
            include = {
                "queries.dev.small.tsv", "qrels.dev.small.tsv", "queries.eval.small.tsv",
            } if name == "collectionandqueries.tar.gz" else None
            members = safe_extract(download / name, destination, include)
            write_json(marker, {"archive_sha256": sha256(download / name), "members": members})
    collection = msmarco / "collection.tsv"
    if not collection.exists():
        raise ValueError("Official collection archive did not contain collection.tsv.")
    corpus = shard_passages(collection, directory / "msmarco")
    dev_queries = msmarco / "queries.dev.small.tsv"
    if not dev_queries.exists() or sum(1 for _ in dev_queries.open("rb")) != 6980:
        raise ValueError("Official dev-small query panel does not have the expected6980 rows.")
    write_json(directory / ("msmarco-acquisition.json" if msmarco_only else "acquisition.json"), {
        "download_receipts": receipts, "msmarco_corpus": corpus,
        "kuairand": "Archive checksum verified; temporal and text join counts must be audited before experiment.",
    })
    print("REAL_DATA_ACQUISITION_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and audit the real semantic research corpora.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--msmarco-only", action="store_true")
    args = parser.parse_args()
    acquire(Path(args.output), args.msmarco_only)


if __name__ == "__main__":
    main()
