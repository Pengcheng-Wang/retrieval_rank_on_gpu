"""Load unchanged author modules from a separately acquired, verified snapshot."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from protocol import require_hash


SOURCE_HASHES = {
    "tiger": {
        "data.py": "90fd819745b730eef4d7beb38bd77483324de051f80686d9834d17880e079a64",
        "collator.py": "28db5cab3329d4c02b3563b9af279e27c5d969ed301c6bff95d1fa52e36746da",
        "utils.py": "f405865a3101d1158f0b1ce56438c77640a1bc8a48139f6b5d731c6696d7f5f8",
        "modeling_letter.py": "454d544a8ae62323509ca5e2e5d5bc397c28527d286d426264eb2bc74a2c3b03",
    },
    "lc-rec": {
        "data.py": "cfd7733325670792c0da32393752c49116cd2b0ce097ffd629cda1c224bddecc",
        "collator.py": "e1ec6dec9aadf3c123ea45656fe2c58dc102ac075ca78bf9dff7df47cb5c29b8",
        "utils.py": "8c1139a496cd74479e862472a57dd5c5eccb3870e17fb64b2f4d02063fd252e5",
        "modeling_letter.py": "c4798e82bfd0c5a16acebb551e08d0e421765d028981288ee6680cdc7f49c9fc",
        "prompt.py": "2415f1a2bece1fd27f961efd5beb3a52e0e997ba39c1f3825fec1292bf84f7d7",
    },
}


def verify_backend_source(upstream: Path, backend: str) -> Path:
    directory = upstream / {"tiger": "LETTER-TIGER", "lc-rec": "LETTER-LC-Rec"}[backend]
    for name, digest in SOURCE_HASHES[backend].items():
        require_hash(directory / name, digest)
    return directory


def load_backend(upstream: Path, backend: str) -> SimpleNamespace:
    directory = verify_backend_source(upstream, backend).resolve()
    for name in ("data", "utils", "collator", "modeling_letter", "prompt"):
        if name in sys.modules:
            path = getattr(sys.modules[name], "__file__", None)
            if path is None or Path(path).resolve().parent != directory:
                raise RuntimeError(f"Author module name conflict: {name}; use a fresh process")
    sys.path.insert(0, str(directory))
    return SimpleNamespace(
        directory=directory,
        data=importlib.import_module("data"),
        utils=importlib.import_module("utils"),
        collator=importlib.import_module("collator"),
        model=importlib.import_module("modeling_letter"),
    )


def default_args(author: SimpleNamespace, upstream: Path, dataset: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for add in (author.utils.parse_global_args, author.utils.parse_train_args,
                author.utils.parse_dataset_args):
        parser = add(parser)
    args = parser.parse_args([])
    args.dataset = dataset
    args.data_path = str(upstream / "data")
    args.index_file = ".index.json"
    args.max_his_len = 20
    args.add_prefix = False
    args.tasks = "seqrec"
    args.train_prompt_sample_num = "1"
    args.train_data_sample_num = "0"
    return args
