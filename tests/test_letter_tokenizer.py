import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def rq():
    assert (SCRIPTS / "tokenizer_run.py").exists(), "Tokenizer lifecycle adapter is missing"
    return importlib.import_module("tokenizer_run")


def test_tokenizer_spec_rejects_silent_zip_truncation():
    rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 9922)
    with pytest.raises(ValueError, match="levels"):
        rq().validate_spec([256] * 8, [0, 0, 0, 0.003], 10, 32, 9922)
    with pytest.raises(ValueError, match="initialization"):
        rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 100)
    with pytest.raises(ValueError, match="initialization"):
        rq().validate_spec([256] * 4, [0, 0, 0, 0.003], 10, 32, 60000)


def test_default_and_nondefault_cluster_bounds_are_feasible():
    assert rq().cluster_bounds(256, 10) == (10, 60)
    low, high = rq().cluster_bounds(512, 5)
    assert 5 * low <= 512 <= 5 * high
    with pytest.raises(ValueError):
        rq().cluster_bounds(8, 10)


def test_eight_level_export_and_collision_accounting():
    codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7],
                      [1, 1, 2, 3, 4, 5, 6, 7]])
    result = rq().format_codes(codes)
    assert result["0"][-1] == "<h_7>"
    assert rq().collision_groups(codes) == [[0, 1]]
    with pytest.raises(ValueError):
        rq().format_codes(np.array([[-1, 2]]))


def test_input_embeddings_are_aligned_finite_matrices():
    semantic = np.ones((512, 16), dtype=np.float32)
    collaborative = np.ones((512, 32), dtype=np.float32)
    rq().validate_embeddings(semantic, collaborative, 32)
    with pytest.raises(ValueError, match="aligned"):
        rq().validate_embeddings(semantic, collaborative[1:], 32)
    semantic[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rq().validate_embeddings(semantic, collaborative, 32)


def test_interrupted_output_is_quarantined_without_deleting_it(tmp_path):
    directory = tmp_path / "epoch-00100"
    directory.mkdir()
    (directory / "state.pt.tmp").write_bytes(b"interrupted")
    displaced = rq().fresh_output_directory(directory, "completed.json")
    assert directory.is_dir() and not list(directory.iterdir())
    assert displaced.parent == tmp_path
    assert (displaced / "state.pt.tmp").read_bytes() == b"interrupted"
    (directory / "completed.json").write_text('{"status":"complete"}')
    with pytest.raises(FileExistsError):
        rq().fresh_output_directory(directory, "completed.json")
