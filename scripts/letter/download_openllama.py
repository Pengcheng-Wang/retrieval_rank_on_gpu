"""Acquire only the separately declared Apache-licensed compatible backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from protocol import file_sha256, require_hash, write_json


MODEL_ID = "openlm-research/open_llama_7b_v2"
REVISION = "e5961def23172a2384543940e773ab676033c963"
TOKENIZER_SHA = "91b289e85fa20fd375d8b33dc12f77616f18abc6359804471d1fafcb425fecb8"
LICENSE_URL = (
    "https://raw.githubusercontent.com/openlm-research/open_llama/"
    "6e7f73eab7e799e2464f38ed977e537bae02873e/LICENSE"
)


def safe_filename(name: str) -> str:
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("A model asset must be a plain snapshot filename")
    return name


def validate_architecture(config: dict) -> None:
    expected = {
        "model_type": "llama", "hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "max_position_embeddings": 2048, "vocab_size": 32000,
        "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Backbone architecture differs from the declared compatible variant")
    if (config.get("auto_map") or config.get("num_key_value_heads", 32) != 32
            or config.get("rope_scaling") is not None or config.get("rope_theta", 10000) != 10000
            or config.get("pretraining_tp", 1) != 1):
        raise ValueError("Unsupported remote-code, attention or position configuration")


def validate_reader_version(version: str) -> None:
    match = re.match(r"^(\d+)\.(\d+)\.", version)
    if match is None or tuple(map(int, match.groups())) < (2, 6):
        raise ValueError("Downloaded pickle tensors require patched Torch >=2.6, never the legacy trainer")


def validate_track(spec: dict) -> None:
    if (spec["model"]["id"] != MODEL_ID or spec["model"]["revision"] != REVISION
            or spec["exact_original_llama1_reproduction"] is not False):
        raise ValueError("This downloader admits only the explicit licensed alternative")


def acquire(spec_path: Path, output: Path, converter_python: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_track(spec)
    marker = output / "completed.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (existing.get("model_id") != MODEL_ID or existing.get("revision") != REVISION
                or existing.get("track_id") != spec["track_id"]
                or existing.get("exact_original_llama1_reproduction") is not False):
            raise ValueError("Existing backbone receipt belongs to a different track/model")
        validate_reader_version(existing["conversion_reader"]["torch"])
        for name, digest in existing["converted_files_sha256"].items():
            require_hash(output / "converted" / safe_filename(name), digest)
        print("LICENSED_BACKBONE_ALREADY_VERIFIED", flush=True)
        return
    info = HfApi(token=False).model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated or (info.cardData or {}).get("license") != "apache-2.0":
        raise ValueError("Model identity/access/license metadata does not match the declared variant")
    allowed = {
        "README.md", "config.json", "generation_config.json", "pytorch_model.bin.index.json",
        "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
        "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    }
    assets = {f.rfilename: f for f in info.siblings if f.rfilename in allowed}
    if set(assets) != allowed:
        raise ValueError("The pinned model snapshot has an unexpected asset inventory")
    snapshot = output / "snapshot"
    snapshot_download(
        MODEL_ID, revision=REVISION, token=False, local_dir=snapshot,
        local_dir_use_symlinks=False, allow_patterns=sorted(allowed),
        resume_download=True, max_workers=2,
    )
    hashes = {}
    for name, asset in assets.items():
        path = snapshot / safe_filename(name)
        if path.stat().st_size != asset.size:
            raise ValueError(f"Model asset size mismatch: {name}")
        if asset.lfs:
            require_hash(path, asset.lfs["sha256"])
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != asset.blob_id:
                raise ValueError(f"Git-blob checksum mismatch: {name}")
        hashes[name] = file_sha256(path)
    require_hash(snapshot / "tokenizer.model", TOKENIZER_SHA)
    config = json.loads((snapshot / "config.json").read_text())
    validate_architecture(config)
    write_json(output / "verified-source.json", {
        "status": "pass", "model_id": MODEL_ID, "revision": REVISION,
        "spec_sha256": file_sha256(spec_path), "files_sha256": hashes,
    })
    environment = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    subprocess.run([
        converter_python, str(Path(__file__).resolve()), "--convert-verified-only",
        "--spec", str(spec_path.resolve()), "--output", str(output.resolve()),
    ], check=True, env=environment)
    print("LICENSED_BACKBONE_SNAPSHOT_VERIFIED", flush=True)


def convert_verified(spec_path: Path, output: Path) -> None:
    import torch
    import safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    validate_reader_version(torch.__version__)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_track(spec)
    receipt = json.loads((output / "verified-source.json").read_text(encoding="utf-8"))
    if (receipt["status"] != "pass" or receipt["model_id"] != MODEL_ID
            or receipt["revision"] != REVISION or receipt["spec_sha256"] != file_sha256(spec_path)):
        raise ValueError("Unverified or differently specified model source")
    snapshot = output / "snapshot"
    hashes = receipt["files_sha256"]
    for name, digest in hashes.items():
        require_hash(snapshot / safe_filename(name), digest)
    require_hash(snapshot / "tokenizer.model", TOKENIZER_SHA)
    validate_architecture(json.loads((snapshot / "config.json").read_text()))
    if (output / "completed.json").exists():
        raise FileExistsError("Refusing to overwrite an already completed conversion")
    converted = output / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    index = json.loads((snapshot / "pytorch_model.bin.index.json").read_text())
    new_map = {}
    total_bytes = 0
    tensor_count = 0
    for filename in sorted(set(index["weight_map"].values())):
        safe_filename(filename)
        if filename not in hashes:
            raise ValueError("Checkpoint index points outside the verified snapshot")
        # Never fall back to unrestricted pickle loading for downloaded weights.
        state = torch.load(snapshot / filename, map_location="cpu", weights_only=True)
        expected = {key for key, value in index["weight_map"].items() if value == filename}
        if set(state) != expected or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError("Unexpected checkpoint tensor inventory")
        state = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("The checkpoint contains nonfinite tensors")
        name = filename.replace("pytorch_model", "model").replace(".bin", ".safetensors")
        temporary = converted / f"{name}.partial"
        save_file(state, temporary, metadata={"format": "pt"})
        with safe_open(temporary, framework="pt", device="cpu") as restored:
            if set(restored.keys()) != expected:
                raise ValueError("Converted checkpoint keys changed")
            for key, tensor in state.items():
                restored_tensor = restored.get_tensor(key)
                if (tensor.dtype != restored_tensor.dtype or tensor.shape != restored_tensor.shape
                        or not torch.equal(tensor.reshape(-1).view(torch.uint8),
                                           restored_tensor.reshape(-1).view(torch.uint8))):
                    raise ValueError(f"Checkpoint conversion changed tensor bytes: {key}")
                total_bytes += tensor.numel() * tensor.element_size()
                tensor_count += 1
                new_map[key] = name
        temporary.replace(converted / name)
        del state
        print(f"CONVERTED_AND_VERIFIED {name}", flush=True)
    write_json(converted / "model.safetensors.index.json", {
        "metadata": {"total_size": total_bytes}, "weight_map": new_map,
    })
    for name in sorted(hashes):
        if not name.endswith(".bin") and name != "pytorch_model.bin.index.json":
            shutil.copyfile(snapshot / name, converted / name)
    with urllib.request.urlopen(LICENSE_URL, timeout=30) as response:
        license_bytes = response.read()
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise ValueError("Publisher license retrieval did not return Apache-2.0")
    (converted / "LICENSE").write_bytes(license_bytes)
    converted_hashes = {p.name: file_sha256(p) for p in sorted(converted.iterdir()) if p.is_file()}
    write_json(output / "completed.json", {
        "status": "downloaded_converted_verified_not_runtime_validated",
        "track_id": spec["track_id"], "exact_original_llama1_reproduction": False,
        "model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0",
        "source_files_sha256": hashes, "converted_files_sha256": converted_hashes,
        "tensor_count": tensor_count, "tensor_bytes": total_bytes,
        "conversion": "weights_only load, identical dtype/value safetensors round-trip",
        "conversion_reader": {"torch": torch.__version__, "safetensors": safetensors.__version__},
        "spec_sha256": file_sha256(spec_path),
        "tokenizer_use_fast": False, "tokenizer_legacy": True,
    })
    print("LICENSED_BACKBONE_CONVERSION_VERIFIED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter-python")
    parser.add_argument("--convert-verified-only", action="store_true")
    args = parser.parse_args()
    if args.convert_verified_only:
        convert_verified(args.spec, args.output)
    elif args.converter_python:
        acquire(args.spec, args.output, args.converter_python)
    else:
        parser.error("--converter-python is required for a separate, patched tensor reader")


if __name__ == "__main__":
    main()
