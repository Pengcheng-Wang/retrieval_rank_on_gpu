"""Opt-in native GPU gates; every artifact stays under LETTER_BASELINE_GATE_ROOT."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))
pytestmark = pytest.mark.skipif(
    os.environ.get("LETTER_BASELINE_GPU_GATE") != "1",
    reason="Opt in only on the authorized idle GPU1 and pinned baseline runtime",
)


@pytest.fixture(scope="module")
def gate_root():
    import baselines

    assert torch.cuda.is_available()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    path = Path(os.environ["LETTER_BASELINE_GATE_ROOT"]) / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    folder = path / "upstream" / "data" / "Instruments"
    folder.mkdir(parents=True)
    interactions = {}
    lengths = [3, 4, 5, 8, 9, 10, 20, 25]
    for user, length in enumerate(lengths):
        train = [(user * 3 + position) % 29 for position in range(length)]
        if user == 2:
            train[2] = train[0]
        interactions[str(user)] = train + [train[-1], (user * 3 + length + 1) % 32]
    payloads = {"inter": interactions,
                "item": {str(i): {"title": f"synthetic-{i}"} for i in range(32)},
                "index": {str(i): [f"<synthetic_{i}>"] for i in range(32)}}
    hashes = {}
    for name, value in payloads.items():
        file = folder / f"Instruments.{name}.json"
        baselines.write_json(file, value)
        hashes[name] = baselines.file_sha256(file)
    manifest = {"purpose": "synthetic-native-contract-gate", "datasets": {
        "Instruments": {"users": 8, "items": 32, "interactions": sum(lengths) + 16,
                        "training_prefix_examples": sum(lengths) - 8, "sha256": hashes}
    }}
    baselines.write_json(path / "manifest.json", manifest)
    print(f"BASELINE_GATE_ROOT={path}", flush=True)
    return path


def assert_exact(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu()), path
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{path}[{i}]")
    else:
        assert left == right, path


def command_arguments(root, model, output, *extra):
    return [
        "train", "--model", model, "--upstream", str(root / "upstream"),
        "--manifest", str(root / "manifest.json"), "--dataset", "Instruments",
        "--recipe", str(ROOT / "configs" / "letter-conventional-baselines-v1.json"),
        "--output", str(output), "--seed", "42", "--device", "cuda:0", "--gate",
        "--sasrec", os.environ["LETTER_SASREC"], "--caser", os.environ["LETTER_CASER"],
        *extra,
    ]


def arguments(root, model, output, *extra):
    import baselines

    return baselines.make_parser().parse_args(command_arguments(root, model, output, *extra))


def run_cli(root, model, output, *extra, command="train", sigterm=False):
    argv = command_arguments(root, model, output, *extra)
    argv[0] = command
    launcher = [sys.executable, str(ROOT / "scripts" / "letter" / "baselines.py")]
    if sigterm:
        # Raise a real SIGTERM inside the first native batch, after train installed its handler.
        code = """
import signal, sys
sys.path.insert(0, 'scripts/letter')
import baselines
original = baselines.NativeAdapter.loss
sent = False
def interrupt(self, batch):
    global sent
    value = original(self, batch)
    if not sent:
        sent = True
        signal.raise_signal(signal.SIGTERM)
    return value
baselines.NativeAdapter.loss = interrupt
baselines.main()
"""
        launcher = [sys.executable, "-c", code]
    result = subprocess.run(launcher + argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", ["bpr", "lightgcn", "hgn", "bert4rec", "sasrec", "caser"])
def test_native_gpu_exact_continuation_and_one_shot_test(gate_root, name):
    import baselines

    full = gate_root / name / "uninterrupted"
    resumed = gate_root / name / "resumed"
    reference = run_cli(gate_root, name, full)
    drain = (run_cli(gate_root, name, resumed, sigterm=True) if name == "hgn" else
             run_cli(gate_root, name, resumed, "--stop-after-epoch", "1"))
    assert drain["status"] == "drained"
    assert drain["epoch"] == 1
    assert not (resumed / "train-completed.json").exists()
    restored = run_cli(gate_root, name, resumed, "--resume")
    assert reference["status"] == restored["status"] == "train-complete"
    assert not (full / "completed.json").exists()
    left = torch.load(full / reference["last_checkpoint"], map_location="cpu")
    right = torch.load(resumed / restored["last_checkpoint"], map_location="cpu")
    for key in ("model", "optimizer", "rng", "sampler", "epoch", "selection", "bad_checks"):
        assert_exact(left[key], right[key], key)
    assert reference["selected_epoch"] == restored["selected_epoch"]
    assert reference["validation"] == restored["validation"]

    data = baselines.load_release(gate_root / "upstream", gate_root / "manifest.json",
                                  "Instruments")
    recipe = baselines.load_recipe(ROOT / "configs" / "letter-conventional-baselines-v1.json")
    adapter = baselines.NativeAdapter(
        name, data, recipe["models"][name], "cuda:0",
        sasrec_source=Path(os.environ["LETTER_SASREC"]),
        caser_source=Path(os.environ["LETTER_CASER"]),
    )
    adapter.model.load_state_dict(left["model"], strict=True)
    adapter.clear_caches()
    adapter.model.eval()
    users = list(data.train)
    histories = [data.train[u] for u in users]
    with torch.inference_mode():
        scores = adapter.scores(users, histories)
        assert scores.shape == (8, 33)
        assert torch.isneginf(scores[:, 0]).all()
        assert torch.isfinite(scores[:, 1:]).all()
        assert baselines.top_items(scores).shape == (8, 10)
        if name == "caser":
            singles = torch.cat([adapter.scores([u], [h]) for u, h in zip(users, histories)])
            torch.testing.assert_close(scores[:, 1:], singles[:, 1:], rtol=1e-5, atol=1e-6)
            assert adapter.model.item_embeddings.weight[0].abs().sum() > 0
        if name == "bert4rec":
            inputs = baselines.bert_inference_inputs(histories)
            rebuilt = adapter.model.reconstruct_test_data(
                torch.as_tensor(inputs["item_id_list"], device="cuda:0"),
                torch.as_tensor(inputs["item_length"], device="cuda:0"),
            ).cpu().tolist()
            assert rebuilt[-1] == histories[-1][-20:] + [33]
        if name == "lightgcn":
            graph = baselines.TrainView(data.train, data.items).inter_matrix().toarray()
            adjacency = np.block([[np.zeros((9, 9)), graph],
                                  [graph.T, np.zeros((33, 33))]])
            inv = (np.sum(adjacency, axis=1) + 1e-7) ** -0.5
            expected = inv[:, None] * adjacency * inv[None, :]
            np.testing.assert_allclose(
                adapter.model.norm_adj_matrix.to_dense().cpu().numpy(), expected,
                rtol=1e-6, atol=1e-7,
            )
            adapter.clear_caches()
            assert adapter.model.restore_user_e is adapter.model.restore_item_e is None
        if name == "sasrec":
            assert adapter.model.item_emb.weight.shape == (33, 50)

    args = arguments(gate_root, name, resumed)
    args.command = "evaluate"
    evaluation = run_cli(gate_root, name, resumed, command="evaluate")
    assert evaluation["status"] == "complete"
    assert evaluation["test_users"] == 8
    assert len(evaluation["test"]) == 4
    assert evaluation["run_class"] == "contract-gate"
    predictions = json.loads((resumed / "test-top10.json").read_text())
    assert len(predictions) == 8
    assert all(len(p["top10"]) == len(set(p["top10"])) == 10 for p in predictions)
    assert all(0 <= i < 32 for p in predictions for i in p["top10"])
    with pytest.raises(FileExistsError):
        baselines.evaluate(args)
    baselines.write_json(gate_root / name / "gate-passed.json", {
        "status": "pass", "model": name, "exact_continuation": True,
        "native_forward_loss_scores": True, "test_claim_guard": True,
        "cold_process_continuation": True,
        "drain": "real-SIGTERM-during-first-batch" if name == "hgn" else "stop-after-epoch-1",
        "uninterrupted_receipt": str(full / "train-completed.json"),
        "resumed_receipt": str(resumed / "completed.json"),
    })
