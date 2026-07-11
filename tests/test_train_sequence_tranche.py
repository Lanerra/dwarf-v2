from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("v2_train_tranche_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_train_tranche_selects_a_contiguous_non_overlapping_window() -> None:
    trainer = load_trainer()
    rows = torch.arange(40).reshape(10, 4)
    mask = torch.ones_like(rows, dtype=torch.bool)

    tranche, tranche_mask, metadata = trainer.select_train_tranche(
        train_data=rows,
        train_loss_mask=mask,
        max_train_seqs=3,
        offset_text="4",
    )

    assert tranche.tolist() == rows[4:7].tolist()
    assert tranche_mask.tolist() == mask[4:7].tolist()
    assert metadata == {"mode": "contiguous", "offset": 4, "count": 3, "end": 7}


def test_explicit_train_tranche_rejects_horizon_overrun() -> None:
    trainer = load_trainer()
    rows = torch.arange(40).reshape(10, 4)
    mask = torch.ones_like(rows, dtype=torch.bool)

    with pytest.raises(ValueError, match="exceeds dataset rows"):
        trainer.select_train_tranche(
            train_data=rows,
            train_loss_mask=mask,
            max_train_seqs=3,
            offset_text="8",
        )
