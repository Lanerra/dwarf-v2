from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateBatch, CandidateType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/candidate_recall_audit_dsqg_w.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("candidate_recall_audit_dsqg_w", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def make_candidate_batch() -> CandidateBatch:
    cand_token_indices = torch.tensor(
        [[
            [-1, -1, -1],
            [0, -1, -1],
            [1, 0, -1],
            [0, 1, -1],
        ]],
        dtype=torch.long,
    )
    cand_types = torch.tensor(
        [[
            [int(CandidateType.NULL), int(CandidateType.NULL), int(CandidateType.NULL)],
            [int(CandidateType.QUESTION), int(CandidateType.NULL), int(CandidateType.NULL)],
            [int(CandidateType.HISA_EVIDENCE), int(CandidateType.LOCAL), int(CandidateType.NULL)],
            [int(CandidateType.LOCAL), int(CandidateType.LONG_OFFSET), int(CandidateType.NULL)],
        ]],
        dtype=torch.long,
    )
    cand_mask = cand_token_indices >= 0
    cand_sources = torch.ones_like(cand_token_indices)
    cand_states = torch.zeros(1, 4, 3, 8)
    valid_count = cand_mask.sum(dim=-1)
    return CandidateBatch(
        cand_states=cand_states,
        cand_types=cand_types,
        cand_sources=cand_sources,
        cand_mask=cand_mask,
        cand_token_indices=cand_token_indices,
        valid_candidate_count=valid_count,
        telemetry={
            "dsqg_w_candidate_duplicate_rate": torch.tensor(0.25),
            "dsqg_w_candidate_invalid_rate": torch.tensor(0.10),
            "dsqg_w_valid_candidate_count": valid_count.float().mean(),
        },
    )


def test_compute_gold_evidence_candidate_recall_overall_and_by_type() -> None:
    mod = load_audit_module()
    batch = make_candidate_batch()
    gold = torch.tensor(
        [[
            [-1, -1],
            [0, -1],
            [1, 0],
            [2, 0],
        ]],
        dtype=torch.long,
    )

    metrics = mod.compute_gold_evidence_candidate_recall(batch, gold)

    assert metrics["dsqg_w_gold_evidence_candidate_recall"] == pytest.approx(0.80)
    assert metrics["dsqg_w_gold_evidence_candidate_count"] == 5
    assert metrics["dsqg_w_gold_evidence_candidate_hit_count"] == 4
    assert metrics["dsqg_w_gold_evidence_candidate_recall_by_type"]["QUESTION"] == pytest.approx(0.20)
    assert metrics["dsqg_w_gold_evidence_candidate_recall_by_type"]["HISA_EVIDENCE"] == pytest.approx(0.20)
    assert metrics["dsqg_w_gold_evidence_candidate_recall_by_type"]["LOCAL"] == pytest.approx(0.40)
    assert metrics["dsqg_w_gold_evidence_candidate_recall_by_type"]["LONG_OFFSET"] == pytest.approx(0.0)


def test_compute_recall_ignores_future_gold_indices() -> None:
    mod = load_audit_module()
    batch = make_candidate_batch()
    gold = torch.tensor([[[3], [3], [3], [3]]], dtype=torch.long)

    metrics = mod.compute_gold_evidence_candidate_recall(batch, gold)

    assert metrics["dsqg_w_gold_evidence_candidate_count"] == 1
    assert metrics["dsqg_w_gold_evidence_candidate_recall"] == pytest.approx(0.0)


def test_synthetic_audit_reports_candidate_telemetry_and_perfect_recall() -> None:
    mod = load_audit_module()
    report = mod.run_synthetic_audit(batch=2, seq_len=16, d=16, n_heads=4, max_candidates=24)

    assert report["pass"] is True
    assert report["metrics"]["dsqg_w_gold_evidence_candidate_recall"] == pytest.approx(1.0)
    assert report["metrics"]["dsqg_w_gold_evidence_candidate_recall_by_type"]["HISA_EVIDENCE"] > 0.0
    assert report["candidate_telemetry"]["dsqg_w_candidate_fraction_question"] > 0.0
    assert report["candidate_telemetry"]["dsqg_w_candidate_fraction_hisa_evidence"] > 0.0
    assert report["candidate_telemetry"]["dsqg_w_candidate_fraction_l3_skip"] > 0.0
    assert report["candidate_telemetry"]["dsqg_w_valid_candidate_count"] <= 24
