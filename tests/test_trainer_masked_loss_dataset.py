from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer(monkeypatch):
    monkeypatch.setenv("DWARF_DISABLE_BNB", "1")
    monkeypatch.setenv("DWARF_LIGER", "0")
    monkeypatch.setenv("DWARF_TORCH_COMPILE", "0")
    monkeypatch.setenv("DWARF_Q6_G128", "0")
    monkeypatch.delenv("DWARF_DSQG_W", raising=False)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    try:
        spec = importlib.util.spec_from_file_location(
            f"trainer_masked_loss_{os.getpid()}_{id(monkeypatch)}",
            TRAINER,
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        for path in [str(ROOT / "kernels"), str(ROOT)]:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def test_streamed_linear_ce_loss_matches_selected_target_rows(monkeypatch):
    mod = load_trainer(monkeypatch)
    hidden = torch.tensor(
        [
            [[0.2, -0.1, 0.4], [0.5, 0.0, -0.3], [0.1, 0.7, -0.2]],
            [[-0.4, 0.3, 0.2], [0.9, -0.6, 0.1], [0.0, 0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [
            [0.1, 0.2, -0.3],
            [-0.5, 0.4, 0.2],
            [0.3, -0.1, 0.6],
            [0.7, 0.0, -0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2, 1], [3, 1, 2]], dtype=torch.long)
    target_mask = torch.tensor([[False, True, False], [True, False, True]])

    loss_sum, n_valid = mod._streamed_linear_ce_loss(
        hidden,
        targets,
        weight,
        chunk_rows=2,
        grad_denom=None,
        loss_mask=target_mask,
    )

    logits = F.linear(hidden.reshape(-1, hidden.size(-1)), weight)
    flat_mask = target_mask.reshape(-1)
    expected = F.cross_entropy(logits[flat_mask], targets.reshape(-1)[flat_mask], reduction="sum")
    assert n_valid == int(target_mask.sum().item())
    assert torch.allclose(loss_sum.cpu(), expected)


def test_prepare_dataset_loss_masks_absent_uses_all_token_ce(monkeypatch):
    mod = load_trainer(monkeypatch)
    train = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    val = torch.arange(4, dtype=torch.int32).reshape(1, 4)

    train_mask, val_mask, stats = mod._prepare_dataset_loss_masks(
        {"train": train, "val": val}, train, val, use_liger_ce=False
    )

    assert train_mask.dtype == torch.bool
    assert val_mask.dtype == torch.bool
    assert train_mask.shape == train.shape
    assert val_mask.shape == val.shape
    assert bool(train_mask.all())
    assert bool(val_mask.all())
    assert stats["source"] == "all_token"
    assert stats["uses_sparse_loss_mask"] is False
    assert stats["train_real_tokens"] == train[:, 1:].numel()
    assert stats["val_real_tokens"] == val[:, 1:].numel()


def test_prepare_dataset_loss_masks_validate_shape_and_next_token_alignment(monkeypatch):
    mod = load_trainer(monkeypatch)
    train = torch.zeros((2, 4), dtype=torch.int32)
    val = torch.zeros((1, 4), dtype=torch.int32)
    train_mask = torch.zeros_like(train, dtype=torch.bool)
    val_mask = torch.zeros_like(val, dtype=torch.int64)
    train_mask[0, 2] = True  # token column 2 trains prediction at input position 1
    val_mask[0, 3] = 1       # token column 3 trains prediction at input position 2

    out_train_mask, out_val_mask, stats = mod._prepare_dataset_loss_masks(
        {
            "train": train,
            "val": val,
            "train_loss_mask": train_mask,
            "val_loss_mask": val_mask,
        },
        train,
        val,
        use_liger_ce=False,
    )

    assert out_train_mask.dtype == torch.bool
    assert out_val_mask.dtype == torch.bool
    assert out_train_mask[:, 1:].sum().item() == 1
    assert out_val_mask[:, 1:].sum().item() == 1
    assert stats["source"] == "dataset"
    assert stats["uses_sparse_loss_mask"] is True
    assert stats["train_real_tokens"] == 1
    assert stats["val_real_tokens"] == 1

    bad = {"train_loss_mask": torch.ones((2, 3), dtype=torch.bool), "val_loss_mask": torch.ones_like(val, dtype=torch.bool)}
    with pytest.raises(ValueError, match="train_loss_mask.*shape"):
        mod._prepare_dataset_loss_masks(bad, train, val, use_liger_ce=False)


def test_prepare_dataset_loss_masks_fail_fast_for_zero_train_targets(monkeypatch):
    mod = load_trainer(monkeypatch)
    train = torch.zeros((2, 4), dtype=torch.int32)
    val = torch.zeros((1, 4), dtype=torch.int32)
    train_mask = torch.zeros_like(train, dtype=torch.bool)
    train_mask[:, 0] = True  # prompt/BOS-only marks vanish after next-token shift
    val_mask = torch.ones_like(val, dtype=torch.bool)

    with pytest.raises(ValueError, match="zero train target rows"):
        mod._prepare_dataset_loss_masks(
            {"train_loss_mask": train_mask, "val_loss_mask": val_mask},
            train,
            val,
            use_liger_ce=False,
        )


def test_prepare_dataset_loss_masks_reject_sparse_masks_with_liger(monkeypatch):
    mod = load_trainer(monkeypatch)
    train = torch.zeros((1, 4), dtype=torch.int32)
    val = torch.zeros((1, 4), dtype=torch.int32)
    train_mask = torch.ones_like(train, dtype=torch.bool)
    val_mask = torch.ones_like(val, dtype=torch.bool)
    train_mask[0, 2] = False

    with pytest.raises(RuntimeError, match="Liger fused CE does not support sparse loss masks"):
        mod._prepare_dataset_loss_masks(
            {"train_loss_mask": train_mask, "val_loss_mask": val_mask},
            train,
            val,
            use_liger_ce=True,
        )


def test_checkpoint_config_records_loss_mask_stats(monkeypatch):
    mod = load_trainer(monkeypatch)
    cfg = {"dataset": {"path": "synthetic.pt"}}
    stats = {
        "source": "dataset",
        "uses_sparse_loss_mask": True,
        "train_real_tokens": 3,
        "train_target_slots": 9,
        "val_real_tokens": 1,
        "val_target_slots": 3,
    }

    returned = mod._attach_loss_mask_stats_to_checkpoint_config(cfg, stats)

    assert returned is cfg
    assert cfg["dataset"]["loss_mask"] == stats
