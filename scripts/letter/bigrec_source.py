"""Private pinned-source and safe-backbone identities for compatible BIGRec."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from protocol import file_sha256, require_hash


REVISION = "16fc86bf2af905c29d089e5f60b35b857e9e0fe3"
ARCHIVE_SHA256 = "4acc47e62d2beb8d35f71b5985480584dc3266c8626279bbc29249936b86b58a"
SOURCE_HASHES = {
    "train.py": "7458f76d8c61b3d46a8f710cc0cefa3f83dc100fa7ccdb49f753adfb7b98023c",
    "inference.py": "b2c86dd6c499e8770b6754ac55d29d7e8d779781c64043148f741cfbf8cde3f5",
    "data/movie/evaluate.py": "66955fa6d9278ccb17eb4ca0096559b032520de4c1bc5c420379fb252473649d",
    "data/game/process.ipynb": "21df3c71ba04e9a0a9d9c6caee12d9f043e03926b0c164dd9d72710e4bc6bfe8",
}
MODEL_ID = "openlm-research/open_llama_7b_v2"
MODEL_REVISION = "e5961def23172a2384543940e773ab676033c963"
MODEL_HASHES = {
    "config.json": "a184b02448fd8164d686a73262846759a86862b8ee0d1ef141d6289aacb892e2",
    "generation_config.json": "d36794768a4d6604d5107e6512264cad829f073b69718e08273751897ef0c3b9",
    "model-00001-of-00002.safetensors": "a4c1a4daa945d56842583673572003387ad4f08b536ecdef01f03bb42f0ef2f5",
    "model-00002-of-00002.safetensors": "db3672c83cb21be600f38996c2d01e0f77d36a2bbe2420ccdd9c529b6b18f100",
    "model.safetensors.index.json": "1fea933c4e1e0beb48d84c40cc25bdcda4282e737cca2cfc012cdbe1e5d0c8ba",
    "tokenizer.model": "91b289e85fa20fd375d8b33dc12f77616f18abc6359804471d1fafcb425fecb8",
    "tokenizer_config.json": "5a979515b1e98186bdceb56439d17f24bfa68baa669e50e150cc3933c07da3a2",
    "special_tokens_map.json": "3833453daeb8222e7a04fc85758cfe429dab30f5fc901d18d389105c501784ad",
}


def load_pure_renderer(path: Path, expected_sha256: str):
    require_hash(path, expected_sha256)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_prompt"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one native generate_prompt")
    function = functions[0]
    if function.decorator_list or any(isinstance(n, (ast.Call, ast.Import, ast.ImportFrom))
                                      for n in ast.walk(function)):
        raise ValueError("The prompt renderer must be pure and import-free")
    namespace = {"__builtins__": {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["generate_prompt"]


def verify_native(path: Path) -> dict:
    require_hash(path / "upstream.zip", ARCHIVE_SHA256)
    for name, digest in SOURCE_HASHES.items():
        require_hash(path / name, digest)
    return {"revision": REVISION, "archive_sha256": ARCHIVE_SHA256, "files": SOURCE_HASHES}


def verify_model(path: Path, full: bool = True) -> dict:
    receipt_path = path.parent / "completed.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt["model_id"] != MODEL_ID or receipt["revision"] != MODEL_REVISION
            or receipt["license"] != "Apache-2.0"):
        raise ValueError("Backbone does not match the approved compatible model")
    for name, expected in MODEL_HASHES.items():
        if receipt["converted_files_sha256"].get(name) != expected:
            raise ValueError(f"Converted model receipt mismatch: {name}")
        if full or not name.endswith(".safetensors"):
            require_hash(path / name, expected)
        elif not (path / name).is_file():
            raise ValueError(f"Missing safe backbone shard: {name}")
    if list(path.glob("*.bin")) or list(path.glob("*.pt")):
        raise ValueError("The safe backbone directory must not contain pickle weights")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if set(index["weight_map"].values()) != {n for n in MODEL_HASHES if n.endswith(".safetensors")}:
        raise ValueError("Unexpected safe backbone index/shard mapping")
    return {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "files": MODEL_HASHES,
        "parent_receipt_sha256": file_sha256(receipt_path),
        "weight_loading": "local_files_only_use_safetensors",
    }


def load_tokenizer(model: Path):
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        str(model), local_files_only=True, legacy=True, add_bos_token=True,
        add_eos_token=False, model_max_length=2**31,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (1, 2, 0):
        raise ValueError("Unexpected native tokenizer special tokens")
    if getattr(tokenizer, "chat_template", None) or tokenizer.is_fast or tokenizer.add_eos_token:
        raise ValueError("No fast tokenizer, automatic EOS, or chat template is allowed")
    return tokenizer
