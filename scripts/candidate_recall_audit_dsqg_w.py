#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kernels.dsqg_w.dsqg_w_mvp import (
    CandidateBatch,
    CandidateProvider,
    CandidateType,
    DSQGWConfig,
)


def _as_python(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().cpu().item())
        return value.detach().cpu().tolist()
    return value


def _gold_values_for_position(gold_evidence_indices: torch.Tensor, b: int, t: int) -> list[int]:
    if gold_evidence_indices.ndim == 1:
        vals = gold_evidence_indices.detach().cpu().tolist()
    elif gold_evidence_indices.ndim == 2:
        vals = gold_evidence_indices[b].detach().cpu().tolist()
    elif gold_evidence_indices.ndim == 3:
        vals = gold_evidence_indices[b, t].detach().cpu().tolist()
    else:
        raise ValueError("gold_evidence_indices must have rank 1, 2, or 3")

    out: list[int] = []
    seen: set[int] = set()
    for val in vals:
        idx = int(val)
        if idx < 0 or idx > t or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def compute_gold_evidence_candidate_recall(
    candidates: CandidateBatch,
    gold_evidence_indices: torch.Tensor,
) -> dict[str, Any]:
    """Measure whether causal gold evidence token indices entered DSQG-W candidates.

    Recall denominator includes only gold indices that are valid for a query row:
    ``0 <= gold_token_index <= query_position``.  Duplicate gold indices within a
    row are counted once so duplicated labels do not inflate recall.
    """

    if candidates.cand_token_indices.ndim != 3:
        raise ValueError("candidates.cand_token_indices must have shape [B, T, J]")
    bsz, seq_len, _ = candidates.cand_token_indices.shape
    if gold_evidence_indices.shape[0] != bsz:
        raise ValueError("gold_evidence_indices batch dimension must match candidates")

    type_hit_counts = {ctype.name: 0 for ctype in CandidateType}
    total = 0
    hits = 0

    for b in range(bsz):
        for t in range(seq_len):
            valid_slots = candidates.cand_mask[b, t]
            row_tokens = candidates.cand_token_indices[b, t][valid_slots]
            row_types = candidates.cand_types[b, t][valid_slots]
            for gold_idx in _gold_values_for_position(gold_evidence_indices, b, t):
                total += 1
                token_matches = row_tokens == int(gold_idx)
                if bool(token_matches.any().item()):
                    hits += 1
                    matched_types = row_types[token_matches].detach().cpu().tolist()
                    for type_id in set(int(v) for v in matched_types):
                        type_hit_counts[CandidateType(type_id).name] += 1

    recall = float(hits) / float(total) if total else 0.0
    by_type = {
        name: (float(count) / float(total) if total else 0.0)
        for name, count in type_hit_counts.items()
    }
    return {
        "dsqg_w_gold_evidence_candidate_recall": recall,
        "dsqg_w_gold_evidence_candidate_count": total,
        "dsqg_w_gold_evidence_candidate_hit_count": hits,
        "dsqg_w_gold_evidence_candidate_recall_by_type": by_type,
        "dsqg_w_gold_evidence_candidate_hit_count_by_type": type_hit_counts,
    }


def _make_per_position_indices(batch: int, seq_len: int, width: int, *, base_lag: int) -> torch.Tensor:
    out = torch.full((batch, seq_len, width), -1, dtype=torch.long)
    for b in range(batch):
        for t in range(seq_len):
            for j in range(width):
                lag = base_lag + j
                out[b, t, j] = max(0, t - lag)
    return out


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            _validate_record(record, source=f"{path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No audit records found in {path}")
    return records


def _validate_record(record: dict[str, Any], *, source: str) -> None:
    required = [
        "id",
        "tokens",
        "answer_positions",
        "gold_evidence_indices",
        "question_indices",
        "hisa_evidence_indices",
        "l3_skip_indices",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{source} missing required fields: {missing}")
    if not isinstance(record["tokens"], list) or not record["tokens"]:
        raise ValueError(f"{source} tokens must be a non-empty list")
    seq_len = len(record["tokens"])
    for key in ["answer_positions", "gold_evidence_indices", "question_indices", "hisa_evidence_indices", "l3_skip_indices"]:
        if not isinstance(record[key], list):
            raise ValueError(f"{source} {key} must be a list")
        for idx in record[key]:
            if not isinstance(idx, int):
                raise ValueError(f"{source} {key} contains non-integer index {idx!r}")
            if idx < 0 or idx >= seq_len:
                raise ValueError(f"{source} {key} index {idx} outside token range 0..{seq_len - 1}")
    for answer_pos in record["answer_positions"]:
        for gold_idx in record["gold_evidence_indices"]:
            if gold_idx > answer_pos:
                raise ValueError(f"{source} gold evidence index {gold_idx} is future to answer position {answer_pos}")


def _record_gold_tensor(record: dict[str, Any]) -> torch.Tensor:
    seq_len = len(record["tokens"])
    gold = torch.full((1, seq_len, len(record["gold_evidence_indices"])), -1, dtype=torch.long)
    gold_values = torch.tensor(record["gold_evidence_indices"], dtype=torch.long)
    for answer_pos in record["answer_positions"]:
        gold[0, int(answer_pos), :] = gold_values
    return gold


def _indices_tensor(record: dict[str, Any], key: str) -> torch.Tensor:
    return torch.tensor([record[key]], dtype=torch.long)


def _merge_recall_metrics(per_record: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(m["dsqg_w_gold_evidence_candidate_count"]) for m in per_record)
    hits = sum(int(m["dsqg_w_gold_evidence_candidate_hit_count"]) for m in per_record)
    type_hit_counts = {ctype.name: 0 for ctype in CandidateType}
    for metrics in per_record:
        for name, count in metrics["dsqg_w_gold_evidence_candidate_hit_count_by_type"].items():
            type_hit_counts[name] += int(count)
    return {
        "dsqg_w_gold_evidence_candidate_recall": float(hits) / float(total) if total else 0.0,
        "dsqg_w_gold_evidence_candidate_count": total,
        "dsqg_w_gold_evidence_candidate_hit_count": hits,
        "dsqg_w_gold_evidence_candidate_recall_by_type": {
            name: (float(count) / float(total) if total else 0.0)
            for name, count in type_hit_counts.items()
        },
        "dsqg_w_gold_evidence_candidate_hit_count_by_type": type_hit_counts,
    }


def _mean_telemetry(values: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for telemetry in values for key in telemetry})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(telemetry[key]) for telemetry in values if key in telemetry]
        out[key] = sum(vals) / float(len(vals)) if vals else 0.0
    return out


def run_jsonl_audit(
    records: list[dict[str, Any]],
    *,
    d: int = 512,
    n_heads: int = 8,
    max_candidates: int = 32,
    seed: int = 20260628,
    min_recall: float = 0.8,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")
    cfg = DSQGWConfig(
        d=d,
        n_heads=n_heads,
        max_candidates=max_candidates,
        bottleneck=64,
        k_question=4,
        k_hisa_evidence=8,
        k_chunk=0,
        k_l3_skip=4,
        local_offsets=(1, 2, 4, 8),
        long_offsets=(16, 32, 64, 128, 256, 512, 1024, 2048),
    )
    provider = CandidateProvider(cfg)
    per_record_metrics: list[dict[str, Any]] = []
    per_record_telemetry: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        torch.manual_seed(seed + record_index)
        seq_len = len(record["tokens"])
        final_states = torch.randn(1, seq_len, d)
        l3_states = final_states + 0.125 * torch.randn_like(final_states)
        candidates = provider.build(
            final_states,
            l3_states=l3_states,
            question_indices=_indices_tensor(record, "question_indices"),
            hisa_evidence_indices=_indices_tensor(record, "hisa_evidence_indices"),
            l3_skip_indices=_indices_tensor(record, "l3_skip_indices"),
        )
        metrics = compute_gold_evidence_candidate_recall(candidates, _record_gold_tensor(record))
        telemetry = {key: _as_python(value) for key, value in candidates.telemetry.items()}
        per_record_metrics.append(metrics)
        per_record_telemetry.append(telemetry)
        examples.append(
            {
                "id": record["id"],
                "answer_positions": list(record["answer_positions"]),
                "gold_evidence_indices": list(record["gold_evidence_indices"]),
                "recall": metrics["dsqg_w_gold_evidence_candidate_recall"],
                "hit_count": metrics["dsqg_w_gold_evidence_candidate_hit_count"],
                "gold_count": metrics["dsqg_w_gold_evidence_candidate_count"],
                "recall_by_type": metrics["dsqg_w_gold_evidence_candidate_recall_by_type"],
            }
        )

    merged_metrics = _merge_recall_metrics(per_record_metrics)
    telemetry = _mean_telemetry(per_record_telemetry)
    return {
        "config": {
            "dataset_examples": len(records),
            "d": d,
            "n_heads": n_heads,
            "max_candidates": max_candidates,
            "candidate_path": "LOCAL_LONG_QUESTION_HISA_EVIDENCE_L3_SKIP_NULL",
            "seed": seed,
            "min_recall": min_recall,
        },
        "metrics": merged_metrics,
        "candidate_telemetry": telemetry,
        "examples": examples,
        "pass": bool(
            merged_metrics["dsqg_w_gold_evidence_candidate_recall"] >= min_recall
            and telemetry["dsqg_w_valid_candidate_count"] <= max_candidates
            and telemetry["dsqg_w_candidate_fraction_question"] > 0.0
            and telemetry["dsqg_w_candidate_fraction_hisa_evidence"] > 0.0
            and telemetry["dsqg_w_candidate_fraction_l3_skip"] > 0.0
        ),
    }


def run_synthetic_audit(
    *,
    batch: int = 2,
    seq_len: int = 64,
    d: int = 512,
    n_heads: int = 8,
    max_candidates: int = 32,
    seed: int = 20260628,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    final_states = torch.randn(batch, seq_len, d)
    # Different tensor to prove L3/HISA source plumbing exists while keeping the
    # audit independent of a real trained checkpoint.
    l3_states = final_states + 0.125 * torch.randn_like(final_states)

    cfg = DSQGWConfig(
        d=d,
        n_heads=n_heads,
        max_candidates=max_candidates,
        bottleneck=64,
        k_question=4,
        k_hisa_evidence=8,
        k_chunk=0,
        k_l3_skip=4,
        local_offsets=(1, 2, 4, 8),
        long_offsets=(16, 32, 64, 128, 256, 512, 1024, 2048),
    )
    provider = CandidateProvider(cfg)

    question_indices = torch.arange(cfg.k_question, dtype=torch.long).repeat(batch, 1)
    hisa_evidence_indices = _make_per_position_indices(batch, seq_len, cfg.k_hisa_evidence, base_lag=1)
    l3_skip_indices = _make_per_position_indices(batch, seq_len, cfg.k_l3_skip, base_lag=cfg.k_hisa_evidence + 1)
    # Gold evidence is the strongest/nearest synthetic HISA evidence candidate.
    gold_evidence_indices = hisa_evidence_indices[:, :, :1].clone()

    candidates = provider.build(
        final_states,
        l3_states=l3_states,
        question_indices=question_indices,
        hisa_evidence_indices=hisa_evidence_indices,
        l3_skip_indices=l3_skip_indices,
    )
    metrics = compute_gold_evidence_candidate_recall(candidates, gold_evidence_indices)
    telemetry = {key: _as_python(value) for key, value in candidates.telemetry.items()}

    report = {
        "config": {
            "batch": batch,
            "seq_len": seq_len,
            "d": d,
            "n_heads": n_heads,
            "max_candidates": max_candidates,
            "candidate_path": "LOCAL_LONG_QUESTION_HISA_EVIDENCE_L3_SKIP_NULL",
            "seed": seed,
        },
        "metrics": metrics,
        "candidate_telemetry": telemetry,
        "pass": bool(
            metrics["dsqg_w_gold_evidence_candidate_recall"] >= 0.999
            and telemetry["dsqg_w_valid_candidate_count"] <= max_candidates
            and telemetry["dsqg_w_candidate_fraction_hisa_evidence"] > 0.0
            and telemetry["dsqg_w_candidate_fraction_question"] > 0.0
            and telemetry["dsqg_w_candidate_fraction_l3_skip"] > 0.0
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="DSQG-W candidate-recall audit")
    parser.add_argument("--jsonl", type=Path, default=None, help="Lexical-gap JSONL audit set. Omit for synthetic audit.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()

    if args.jsonl is not None:
        report = run_jsonl_audit(
            load_jsonl_records(args.jsonl),
            d=args.d,
            n_heads=args.n_heads,
            max_candidates=args.max_candidates,
            seed=args.seed,
            min_recall=args.min_recall,
        )
    else:
        report = run_synthetic_audit(
            batch=args.batch,
            seq_len=args.seq_len,
            d=args.d,
            n_heads=args.n_heads,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
