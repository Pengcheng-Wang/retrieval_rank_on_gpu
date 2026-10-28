import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def downloader():
    assert (SCRIPTS / "download_openllama.py").exists(), "Licensed-backbone acquisition is missing"
    import download_openllama
    return download_openllama


def test_backbone_architecture_cannot_silently_change():
    config = {
        "model_type": "llama", "hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "max_position_embeddings": 2048, "vocab_size": 32000,
        "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    downloader().validate_architecture(config)
    with pytest.raises(ValueError):
        downloader().validate_architecture({**config, "hidden_size": 512})
    with pytest.raises(ValueError):
        downloader().validate_architecture({**config, "auto_map": {"AutoModel": "remote.Code"}})


def test_model_file_names_cannot_escape_the_snapshot():
    assert downloader().safe_filename("pytorch_model-00001-of-00002.bin")
    for name in ("../secret", "/absolute.bin", "folder/model.bin", "x\\y.bin"):
        with pytest.raises(ValueError):
            downloader().safe_filename(name)


def test_downloaded_pickle_requires_a_patched_reader():
    for version in ("1.13.1+cu117", "2.5.1", "unknown"):
        with pytest.raises(ValueError):
            downloader().validate_reader_version(version)
    downloader().validate_reader_version("2.6.0")
    downloader().validate_reader_version("2.13.0+cu129")
