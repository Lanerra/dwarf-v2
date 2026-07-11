from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_dsqg_dataset_quarantine.py"
SPEC = importlib.util.spec_from_file_location("repair_dsqg_dataset_quarantine", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_replace_rows_replaces_only_quarantined_rows_without_mutating_input() -> None:
    original = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32)
    replacement = torch.tensor([[101, 102, 103], [201, 202, 203]], dtype=torch.int32)

    repaired = MODULE.replace_rows(original, [2, 0], replacement)

    assert original.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert repaired.tolist() == [[201, 202, 203], [4, 5, 6], [101, 102, 103]]


def test_replace_rows_rejects_nonmatching_replacement_shape() -> None:
    original = torch.zeros((3, 4), dtype=torch.int32)

    with pytest.raises(ValueError, match="shape"):
        MODULE.replace_rows(original, [0], torch.ones((1, 3), dtype=torch.int32))
