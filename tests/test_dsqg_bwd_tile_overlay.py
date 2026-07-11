from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


OVERLAY = (
    Path(__file__).resolve().parents[1]
    / "kernel_overlays"
    / "bwd_tile_tuning"
    / "dsqg_attention_v20_bf16_se.py"
)


def _load_overlay():
    spec = importlib.util.spec_from_file_location("dsqg_bwd_tile_overlay_test", OVERLAY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_backward_launch_defaults_to_forward_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_overlay()
    monkeypatch.delenv("DWARF_DSQG_BWD_BLOCK_N", raising=False)
    monkeypatch.delenv("DWARF_DSQG_BWD_NUM_WARPS", raising=False)
    monkeypatch.delenv("DWARF_DSQG_BWD_NUM_STAGES", raising=False)

    assert module.resolve_backward_launch_config(64, 4, 2) == (64, 4, 2)


def test_backward_launch_accepts_explicit_backward_only_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_overlay()
    monkeypatch.setenv("DWARF_DSQG_BWD_BLOCK_N", "32")

    assert module.resolve_backward_launch_config(64, 4, 2) == (32, 4, 2)


def test_backward_launch_rejects_unsupported_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_overlay()
    monkeypatch.setenv("DWARF_DSQG_BWD_BLOCK_N", "48")

    with pytest.raises(ValueError, match="power of two"):
        module.resolve_backward_launch_config(64, 4, 2)
