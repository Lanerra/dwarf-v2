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

    direct = trainer.HierarchicalSparseAttentionV16HISACausal(D=64, H=4, hd=16)
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
            local_backend="flex",
            flex_available=False,
        )


def test_legacy_dsqg_npci_migration_is_exact_and_preserves_hisa() -> None:
    assert hasattr(trainer, "migrate_legacy_dsqg_npci_model_state")
    assert hasattr(trainer, "migrate_legacy_dsqg_npci_optimizer_state")
    model = trainer.DwarfForCausalLM(small_config())
    current = copy.deepcopy(model.state_dict())
    legacy = copy.deepcopy(current)
    expected_removed = set()
    for index, block in enumerate(model.blocks):
        if not isinstance(block, trainer.DSQGBlock):
            continue
        for suffix in ("npci_theta_k", "npci_theta_v"):
            name = f"blocks.{index}.attn.{suffix}"
            legacy[name] = torch.zeros(model.config.num_heads)
            expected_removed.add(name)

    migrated, removed = trainer.migrate_legacy_dsqg_npci_model_state(legacy, model)
    assert removed == expected_removed
    assert migrated.keys() == current.keys()
    assert "blocks.3.attn.npci_theta_k" in migrated
    assert "blocks.3.attn.npci_theta_v" in migrated

    malformed = copy.deepcopy(legacy)
    malformed["blocks.3.attn.retired_npci"] = torch.zeros(model.config.num_heads)
    with pytest.raises(ValueError, match="model keys"):
        trainer.migrate_legacy_dsqg_npci_model_state(malformed, model)

    old_ids = list(range(20))
    saved_optimizer = {
        "kind": "dwarf-muon-adamw-v1",
        "optimizers": [
            {
                "name": "muon",
                "state": {"state": {}, "param_groups": []},
            },
            {
                "name": "adamw",
                "state": {
                    "state": {value: {"step": torch.tensor(1.0)} for value in old_ids},
                    "param_groups": [{"name": "adam_npci", "params": old_ids}],
                },
            },
        ],
    }
    migrated_optimizer = trainer.migrate_legacy_dsqg_npci_optimizer_state(
        saved_optimizer,
        dsqg_blocks_before_global=3,
        total_dsqg_blocks=9,
    )
    npci_group = migrated_optimizer["optimizers"][1]["state"]["param_groups"][0]
    assert migrated_optimizer["kind"] == "dwarf-muon-adamw-v2"
    assert npci_group["params"] == [6, 7]
    assert set(migrated_optimizer["optimizers"][1]["state"]["state"]) == {6, 7}
    reversed_optimizer = copy.deepcopy(saved_optimizer)
    reversed_optimizer["optimizers"][1]["state"]["param_groups"][0][
        "params"
    ].reverse()
    with pytest.raises(ValueError, match="canonical contiguous order"):
        trainer.migrate_legacy_dsqg_npci_optimizer_state(
            reversed_optimizer,
            dsqg_blocks_before_global=3,
            total_dsqg_blocks=9,
        )


def test_legacy_checkpoint_payload_migrates_into_live_optimizer_and_fails_closed(
    monkeypatch,
) -> None:
    model = trainer.DwarfForCausalLM(small_config())
    reference = copy.deepcopy(model)
    optimizer = trainer.build_optimizer(model)
    current_architecture = trainer.model_metadata(model)
    legacy_model = dict(model.state_dict())
    for index, block in enumerate(model.blocks):
        if isinstance(block, trainer.DSQGBlock):
            legacy_model[f"blocks.{index}.attn.npci_theta_k"] = torch.zeros(
                model.config.num_heads
            )
            legacy_model[f"blocks.{index}.attn.npci_theta_v"] = torch.zeros(
                model.config.num_heads
            )

    legacy_optimizer = copy.deepcopy(optimizer.state_dict())
    legacy_optimizer["kind"] = "dwarf-muon-adamw-v1"
    adamw = next(
        item for item in legacy_optimizer["optimizers"] if item["name"] == "adamw"
    )["state"]
    npci_group = next(
        group for group in adamw["param_groups"] if group["name"] == "adam_npci"
    )
    hisa_ids = list(npci_group["params"])
    first_npci_id = hisa_ids[0]
    assert hisa_ids == [first_npci_id, first_npci_id + 1]
    old_npci_ids = list(range(first_npci_id, first_npci_id + 20))
    npci_group_index = adamw["param_groups"].index(npci_group)
    for group in adamw["param_groups"][npci_group_index + 1 :]:
        group["params"] = [parameter_id + 18 for parameter_id in group["params"]]
    npci_group["params"] = old_npci_ids
    retained_hisa_ids = set(old_npci_ids[6:8])
    for parameter_id in old_npci_ids:
        if parameter_id in retained_hisa_ids:
            continue
        adamw["state"][parameter_id] = {"step": torch.tensor(1.0)}

    checkpoint = {
        "kind": "dwarf-55m-resume-v1",
        "model": legacy_model,
        "optimizer": legacy_optimizer,
        "architecture": trainer._legacy_v1_architecture(current_architecture),
    }
    migrated = trainer.migrate_legacy_checkpoint_payload(
        checkpoint,
        model=model,
        architecture=current_architecture,
    )
    assert migrated["kind"] == "dwarf-55m-resume-v2"
    assert migrated["architecture"] == current_architecture
    assert len(migrated["migration"]["removed_model_keys"]) == 18
    assert hasattr(trainer, "validate_checkpoint_migration_receipt")
    trainer.validate_checkpoint_migration_receipt(
        migrated["migration"],
        model=model,
        architecture=current_architecture,
    )
    forged_receipt = copy.deepcopy(migrated["migration"])
    forged_receipt["legacy_sources"]["train_dwarf.py"] = "0" * 64
    with pytest.raises(ValueError, match="migration receipt"):
        trainer.validate_checkpoint_migration_receipt(
            forged_receipt,
            model=model,
            architecture=current_architecture,
        )
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    native_payload = trainer.checkpoint_payload(
        step=1,
        model=model,
        optimizer=optimizer,
        architecture=current_architecture,
        dataset={"fixture": "native-v2"},
    )
    assert native_payload["migration"] is None
    model._checkpoint_migration_receipt = forged_receipt
    with pytest.raises(ValueError, match="migration receipt"):
        trainer.checkpoint_payload(
            step=1,
            model=model,
            optimizer=optimizer,
            architecture=current_architecture,
            dataset={"fixture": "forged-v2"},
        )
    delattr(model, "_checkpoint_migration_receipt")
    model.load_state_dict(migrated["model"], strict=True)
    optimizer.load_state_dict(migrated["optimizer"])

    model.blocks[3].attn.backend = "eager"
    reference.blocks[3].attn.backend = "eager"
    model.eval()
    reference.eval()
    inputs = torch.arange(64, dtype=torch.long).reshape(1, 64) % model.config.vocab_size
    torch.manual_seed(91)
    expected_logits, expected_auxiliary = reference(inputs, return_auxiliary=True)
    expected_loss = expected_logits.float().square().mean() + expected_auxiliary.float()
    expected_loss.backward()
    torch.manual_seed(91)
    actual_logits, actual_auxiliary = model(inputs, return_auxiliary=True)
    actual_loss = actual_logits.float().square().mean() + actual_auxiliary.float()
    actual_loss.backward()
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_auxiliary, expected_auxiliary)
    assert torch.equal(actual_loss, expected_loss)
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    actual_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert reference_gradients.keys() == actual_gradients.keys()
    for name in reference_gradients:
        torch.testing.assert_close(
            actual_gradients[name],
            reference_gradients[name],
            rtol=1e-6,
            atol=2e-8,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )

    wrong_source = copy.deepcopy(checkpoint)
    wrong_source["architecture"]["sources"]["train_dwarf.py"] = "0" * 64
    with pytest.raises(ValueError, match="source manifest"):
        trainer.migrate_legacy_checkpoint_payload(
            wrong_source,
            model=model,
            architecture=current_architecture,
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
