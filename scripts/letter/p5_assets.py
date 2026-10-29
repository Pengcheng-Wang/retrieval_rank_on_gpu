"""Acquire and verify only the pinned Apache T5 safetensors snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from p5_source import MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
from p5_source import read_json, require_hash, sha256, write_json


ASSETS = {
    "README.md", "config.json", "generation_config.json", "model.safetensors",
    "spiece.model", "tokenizer.json", "tokenizer_config.json",
}


def verify_assets(output):
    output = Path(output)
    receipt = read_json(output / "asset-receipt.json")
    if (receipt["model_id"], receipt["revision"], receipt["license"]) != (
        MODEL_ID, MODEL_REVISION, "apache-2.0",
    ) or set(receipt["files_sha256"]) != ASSETS:
        raise ValueError("Different or incomplete pretrained T5 assets")
    for name, digest in receipt["files_sha256"].items():
        require_hash(output / name, digest)
    require_hash(output / "model.safetensors", MODEL_WEIGHTS_SHA256)
    return receipt


def acquire(output):
    output = Path(output)
    if (output / "asset-receipt.json").exists():
        return verify_assets(output)
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=90) as response:
        metadata = json.load(response)
    if (metadata["sha"] != MODEL_REVISION or metadata.get("gated")
            or metadata["cardData"]["license"] != "apache-2.0"):
        raise ValueError("Pretrained model identity, license or access differs")
    inventory = {a["rfilename"]: a for a in metadata["siblings"] if a["rfilename"] in ASSETS}
    if set(inventory) != ASSETS:
        raise ValueError("Pinned T5 snapshot lacks a required safe asset")
    hashes = {}
    for name, asset in inventory.items():
        path = output / name
        if not path.exists():
            download = output / (name + ".partial")
            with urllib.request.urlopen(
                f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}",
                timeout=180,
            ) as response, download.open("wb") as stream:
                while chunk := response.read(8 << 20):
                    stream.write(chunk)
            download.replace(path)
        if path.stat().st_size != asset["size"]:
            raise ValueError(f"Wrong model asset size: {name}")
        if asset.get("lfs"):
            require_hash(path, asset["lfs"]["sha256"])
        else:
            content = path.read_bytes()
            if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != asset["blobId"]:
                raise ValueError(f"Wrong model Git blob: {name}")
        hashes[name] = sha256(path)
        print(f"VERIFIED_ASSET {name} {hashes[name]}", flush=True)
    require_hash(output / "model.safetensors", MODEL_WEIGHTS_SHA256)
    write_json(output / "asset-receipt.json", {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0",
        "files_sha256": hashes, "inventory": inventory,
        "downloaded_pickle_loaded": False,
    })
    return verify_assets(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    acquire(parser.parse_args().output)
