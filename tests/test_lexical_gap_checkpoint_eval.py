from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_lexical_gap_checkpoints.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_lexical_gap_checkpoints", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_answer_rank_metrics_reports_topk_mrr_and_margin() -> None:
    mod = load_eval_module()
    logits = torch.tensor(
        [
            [[0.0, 0.2, 3.0, 1.0], [0.0, 4.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[2, 2]], dtype=torch.long)
    answer_mask = torch.tensor([[True, True]])

    metrics = mod.answer_rank_metrics(logits, labels, answer_mask, prefix="lex")

    assert metrics["lex_answer_tokens"] == 2.0
    assert metrics["lex_top1_acc"] == pytest.approx(0.5)
    assert metrics["lex_top5_acc"] == pytest.approx(1.0)
    assert metrics["lex_mean_rank"] == pytest.approx(1.5)
    assert metrics["lex_mrr"] == pytest.approx((1.0 + 0.5) / 2.0)
    assert metrics["lex_mean_gold_margin"] == pytest.approx((2.0 + -2.0) / 2.0)


def test_build_comparison_reports_deltas_and_selects_winner() -> None:
    mod = load_eval_module()
    report = mod.build_comparison(
        [
            {"name": "DSQG-D", "lex_answer_ce": 2.0, "lex_mrr": 0.20, "lex_top1_acc": 0.0, "lex_mean_rank": 50.0},
            {"name": "DSQG-W", "lex_answer_ce": 1.5, "lex_mrr": 0.25, "lex_top1_acc": 0.1, "lex_mean_rank": 30.0},
        ]
    )

    assert report["best_by_answer_ce"] == "DSQG-W"
    assert report["best_by_mrr"] == "DSQG-W"
    assert report["w_minus_d_answer_ce"] == pytest.approx(-0.5)
    assert report["w_minus_d_mrr"] == pytest.approx(0.05)
    assert report["w_minus_d_mean_rank"] == pytest.approx(-20.0)


def test_set_eval_env_can_enable_dsqg_w_width_cell(monkeypatch) -> None:
    mod = load_eval_module()

    mod._set_eval_env(dsqg_w=True, sites="2,6,final", width_cell=True, width_bottleneck=32, width_gate_init=-5.0)

    import os

    assert os.environ["DWARF_DSQG_W"] == "1"
    assert os.environ["DWARF_DSQG_W_WIDTH_CELL"] == "1"
    assert os.environ["DWARF_DSQG_W_WIDTH_BOTTLENECK"] == "32"
    assert os.environ["DWARF_DSQG_W_WIDTH_GATE_INIT"] == "-5.0"
