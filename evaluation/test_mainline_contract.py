from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "train" / "train_dwarf.py"
SPEC = importlib.util.spec_from_file_location("dwarf_public_trainer_contract", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def small_config(**overrides):
    values = {
        "vocab_size": 128,
        "embedding_dim": 64,
        "num_heads": 4,
        "ffn_dim": 96,
        "seq_len": 256,
        "dropout": 0.0,
    }
    values.update(overrides)
    return trainer.DwarfConfig(**values)


def test_dsqg_npci_removed_but_global_hisa_npci_retained() -> None:
    model = trainer.DwarfForCausalLM(small_config())
    dsqg = [block.attn for block in model.blocks if isinstance(block, trainer.DSQGBlock)]
    assert len(dsqg) == 9
    assert all(not hasattr(module, "npci_theta_k") for module in dsqg)
    assert all(not hasattr(module, "npci_theta_v") for module in dsqg)

    global_hisa = model.blocks[3].attn
    assert hasattr(global_hisa, "npci_theta_k")
    assert hasattr(global_hisa, "npci_theta_v")


def test_zero_auxiliary_has_no_activation_dependency() -> None:
    model = trainer.DwarfForCausalLM(small_config())
    model.blocks = nn.ModuleList()
    model.norm = nn.Identity()
    _, auxiliary = model.forward_hidden(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        return_auxiliary=True,
    )
    assert auxiliary.shape == ()
    assert auxiliary.item() == 0.0
    assert not auxiliary.requires_grad


def test_eval_hisa_zero_auxiliary_has_no_output_dependency() -> None:
    block = trainer.GlobalMixerBlock(small_config(hisa_backend="eager"))
    block.eval()
    values = torch.randn(1, 64, 64, requires_grad=True)
    _, auxiliary = block(values)
    assert auxiliary.item() == 0.0
    assert not auxiliary.requires_grad


def test_short_training_hisa_zero_route_aux_has_no_logits_dependency() -> None:
    block = trainer.GlobalMixerBlock(small_config(hisa_backend="eager"))
    block.train()
    values = torch.randn(1, 64, 64, requires_grad=True)
    _, auxiliary = block(values)
    assert auxiliary.item() == 0.0
    assert not auxiliary.requires_grad


def test_public_model_hisa_policy_is_explicit_not_environment_resolved(monkeypatch) -> None:
    monkeypatch.setenv("DWARF_HISA_V16_BACKEND", "eager")
    monkeypatch.setenv("DWARF_HISA_V16_TOKEN_SELECTION", "canonical")
    monkeypatch.setenv("DWARF_HISA_V16_BLOCK_Q", "32")
    monkeypatch.setenv("DWARF_HISA_V16_BWD", "atomic")
    config = small_config(
        hisa_backend="triton",
        hisa_token_selection_mode="auto",
        hisa_local_backend="flex",
        hisa_triton_block_q=16,
        hisa_backward_impl="atomic_masked",
    )
    model = trainer.DwarfForCausalLM(config)
    hisa = model.blocks[3].attn
    assert hisa.backend == config.hisa_backend
    assert hisa.token_selection_mode == config.hisa_token_selection_mode
    assert hisa.local_backend == config.hisa_local_backend
    assert hisa.triton_block_q == config.hisa_triton_block_q
    assert hisa.backward_impl == config.hisa_backward_impl
    metadata = trainer.model_metadata(model)
    assert metadata["config"]["hisa_backend"] == "triton"
    assert metadata["hisa"]["execution"]["backward_impl"] == "atomic_masked"

    direct = trainer.HierarchicalSparseAttentionV19HISACausal(D=64, H=4, hd=16)
    assert direct.backend == "auto"
    assert direct.token_selection_mode == "auto"
    assert direct.triton_block_q == 16
    assert direct.backward_impl == "atomic_masked"
    hisa_module = sys.modules[direct.__class__.__module__]
    with pytest.raises(RuntimeError, match="FlexAttention"):
        hisa_module._resolve_attention_execution(
            backend="triton",
            is_cuda=True,
            triton_available=True,
            flex_available=False,
        )


def test_pre_v19_checkpoint_kind_is_rejected(tmp_path: Path) -> None:
    model = trainer.DwarfForCausalLM(small_config())
    optimizer = trainer.build_optimizer(model)
    path = tmp_path / "v18.pt"
    torch.save({"kind": "dwarf-55m-resume-v1"}, path)
    with pytest.raises(ValueError, match="not a 55M DWARF resumable checkpoint"):
        trainer.restore_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            architecture=trainer.model_metadata(model),
            dataset={},
        )


def test_optimizer_load_rejects_tampered_group_policy() -> None:
    model = trainer.DwarfForCausalLM(small_config())
    optimizer = trainer.build_optimizer(model)
    valid = optimizer.state_dict()

    tampered_base_lr = copy.deepcopy(valid)
    adamw = next(
        item for item in tampered_base_lr["optimizers"] if item["name"] == "adamw"
    )
    adamw["state"]["param_groups"][0]["base_lr"] *= 2.0
    with pytest.raises(ValueError, match="immutable metadata"):
        optimizer.load_state_dict(tampered_base_lr, expected_lr_factor=1.0)

    tampered_lr = copy.deepcopy(valid)
    for item in tampered_lr["optimizers"]:
        for group in item["state"]["param_groups"]:
            group["lr"] = group["base_lr"] * 0.5
    with pytest.raises(ValueError, match="scheduled learning rate"):
        optimizer.load_state_dict(tampered_lr, expected_lr_factor=1.0)
