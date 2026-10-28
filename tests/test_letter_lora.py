"""Synthetic-weight LC-Rec API checks, not authorization or model-quality evidence."""

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def test_original_lc_rec_int8_lora_gradients_loss_and_reload(tmp_path):
    source = os.environ.get("LETTER_UPSTREAM")
    if source is None:
        pytest.skip("Explicit isolated Linux/CUDA LETTER environment is required")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_int8_training
    from transformers import LlamaConfig, LlamaForCausalLM
    from author import load_backend

    assert torch.cuda.is_available()
    author = load_backend(Path(source), "lc-rec")
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=128, pretraining_tp=1,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    base_path = tmp_path / "synthetic-base"
    LlamaForCausalLM(config).save_pretrained(base_path)
    model = author.model.LETTER.from_pretrained(base_path, load_in_8bit=True, device_map={"": 0})
    model.set_hyper(0.8)
    model.resize_token_embeddings(68)
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["embed_tokens", "lm_head"], bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    for name, parameter in model.named_parameters():
        if "original_module" in name and any(x in name for x in ("embed_tokens", "lm_head")):
            parameter.requires_grad = False
    active = {n: p for n, p in model.named_parameters() if "modules_to_save.default" in n}
    originals = {n: p.detach().clone() for n, p in model.named_parameters() if "original_module" in n}
    assert len(active) == 2 and all(p.requires_grad for p in active.values())
    inputs = torch.tensor([[1, 64, 65, 66, 67, 2], [1, 66, 64, 65, 67, 2]], device="cuda")
    labels = inputs.clone()
    labels[:, :3] = -100
    model.config.use_cache = False
    model.train()
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
    oracle = torch.nn.functional.cross_entropy(
        (output.logits[:, :-1] / 0.8).reshape(-1, 68), labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, oracle)
    output.loss.backward()
    for name, parameter in active.items():
        assert parameter.grad is not None and parameter.grad[64:].abs().sum().item() > 0, name
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in originals:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)
    model.eval()
    with torch.inference_mode():
        expected = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(adapter_path)
    base = LlamaForCausalLM.from_pretrained(base_path, load_in_8bit=True, device_map={"": 0})
    base.resize_token_embeddings(68)
    base = prepare_model_for_int8_training(base)
    reloaded = PeftModel.from_pretrained(base, adapter_path).eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=1e-5)
