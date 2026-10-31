import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def cf():
    assert (SCRIPTS / "cf.py").exists(), "BCE teacher adapter is missing"
    import cf
    return cf


def test_cf_split_mapping_preserves_padding_and_future_isolation():
    prepare = cf().teacher_sequences
    users, train, valid, test = prepare({"7": [0, 1, 2, 3, 4], "3": [2, 3, 4, 5, 6]})
    assert users == ["3", "7"]
    assert train == {1: [3, 4, 5], 2: [1, 2, 3]}
    assert valid == {1: 6, 2: 4} and test == {1: 7, 2: 5}
    assert prepare({"7": [0, 1, 2, 3, 6], "3": [2, 3, 4, 5, 0]})[1] == train


def test_synchronous_sampler_resume_repeats_next_batch():
    sampler = cf().TeacherSampler({1: [1, 2, 3], 2: [2, 4, 5]}, 8, 4, 5, 42)
    sampler.next_batch()
    state = sampler.state()
    expected = sampler.next_batch()
    sampler.restore(state)
    actual = sampler.next_batch()
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_synchronous_sampler_uses_only_training_positives():
    train = {1: [1, 2, 3], 2: [2, 4, 5]}
    sampler = cf().TeacherSampler(train, 8, 16, 5, 42)
    users, seq, pos, neg = sampler.next_batch()
    for u, s, p, n in zip(users, seq, pos, neg):
        assert set(s) - {0} <= set(train[u])
        assert set(p) - {0} <= set(train[u])
        assert not (set(n) - {0}) & set(train[u])
        assert s[-1] == train[u][-2] and p[-1] == train[u][-1]


def test_teacher_export_drops_one_padding_row():
    weights = np.arange(4 * 3).reshape(4, 3)
    np.testing.assert_array_equal(cf().export_item_rows(weights, 3), weights[1:])
    with pytest.raises(ValueError):
        cf().export_item_rows(weights, 4)


def test_sampler_matches_author_single_worker_distribution():
    import importlib.util
    import os

    source = Path(os.environ.get("LETTER_SASREC", ROOT / ".local" / "letter" / "sasrec"))
    if not (source / "utils.py").exists():
        pytest.skip("The external pinned SASRec source is required for parity testing")
    spec = importlib.util.spec_from_file_location("sasrec_reference_utils", source / "utils.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    train = {1: [1, 2, 3], 2: [2, 4, 5]}

    class BatchReady(Exception):
        pass

    class OneBatch:
        batch = None

        def put(self, rows):
            self.batch = tuple(np.asarray(column) for column in rows)
            raise BatchReady

    queue = OneBatch()
    rng = np.random.get_state()
    try:
        with pytest.raises(BatchReady):
            reference.sample_function(train, 2, 8, 16, 5, queue, 42)
    finally:
        np.random.set_state(rng)
    actual = cf().TeacherSampler(train, 8, 16, 5, 42).next_batch()
    for left, right in zip(queue.batch, actual):
        np.testing.assert_array_equal(left, right)


def test_teacher_exposes_an_epoch_boundary_drain():
    import subprocess
    result = subprocess.run([sys.executable, str(SCRIPTS / "cf.py"), "--help"],
                            check=True, capture_output=True, text=True)
    assert "--stop-after-epoch" in result.stdout
