#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_OBJECTIVE = ROOT / "scripts/frozen_trunk_objective_dsqg_w.py"
MICROTRAIN = ROOT / "scripts/microtrain_dsqg_w_lexical_gap.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_objective_module():
    return _load_module(FROZEN_OBJECTIVE, "frozen_trunk_objective_dsqg_w_for_parity")


def load_microtrain_module():
    return _load_module(MICROTRAIN, "microtrain_dsqg_w_lexical_gap_for_parity")


def _float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().float().cpu().item())


def _answer_logits(logits: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    return logits[answer_mask.to(device=logits.device)]


def _rank_metrics(micro, result, batch, *, prefix: str) -> dict[str, float]:
    raw = micro.answer_rank_metrics(result.logits, batch.labels, batch.answer_mask, prefix=prefix)
    return {key: float(value) for key, value in raw.items()}


def _scalar_telemetry_diff(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, dict[str, float]]:
    diffs: dict[str, float] = {}
    for key in sorted(set(a) & set(b)):
        av = a[key]
        bv = b[key]
        if torch.is_tensor(av) and av.numel() != 1:
            continue
        if torch.is_tensor(bv) and bv.numel() != 1:
            continue
        if isinstance(av, (int, float)) or torch.is_tensor(av):
            if isinstance(bv, (int, float)) or torch.is_tensor(bv):
                diffs[key] = abs(_float(av) - _float(bv))
    return (max(diffs.values()) if diffs else 0.0), diffs


def _make_reference_model(obj, trainer, *, seed: int, vocab_size: int, seq_len: int):
    torch.manual_seed(int(seed))
    model = obj.make_tiny_model(trainer, vocab_size=vocab_size, ffn_dim=64, seq_len=seq_len)
    model.eval()
    return model


def run_parity_harness(
    *,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    output_dir: Path | str,
    train_size: int = 64,
    val_size: int = 16,
    seed: int = 20260628,
    candidate_backend: str = "reference",
    atol: float = 0.0,
) -> dict[str, Any]:
    if candidate_backend != "reference":
        raise ValueError(f"unsupported candidate_backend {candidate_backend!r}; available: reference")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    obj = load_objective_module()
    micro = load_microtrain_module()

    train_records, val_records = micro.generate_lexical_gap_dataset(train_size=train_size, val_size=val_size, seed=seed)
    tokenizer = obj.load_tokenizer(tokenizer_path)
    val_batch, val_meta = obj.build_tokenized_lexical_gap_batch(val_records, tokenizer)
    vocab_size = int(val_meta["tokenizer_vocab_size"])
    seq_len = int(val_batch.input_ids.shape[1])

    trainer = obj.load_trainer(enable_objective=True, suffix="parity")
    reference_model = _make_reference_model(obj, trainer, seed=seed, vocab_size=vocab_size, seq_len=seq_len)
    candidate_model = _make_reference_model(obj, trainer, seed=seed, vocab_size=vocab_size, seq_len=seq_len)
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    candidate_model.eval()

    with torch.no_grad():
        ref = obj.compute_frozen_dsqg_w_objective(reference_model, val_batch)
        cand = obj.compute_frozen_dsqg_w_objective(candidate_model, val_batch)

    answer_diff = _max_abs_diff(_answer_logits(ref.logits, val_batch.answer_mask), _answer_logits(cand.logits, val_batch.answer_mask))
    full_diff = _max_abs_diff(ref.logits, cand.logits)
    loss_abs_diff = abs(float(ref.loss.detach().cpu().item()) - float(cand.loss.detach().cpu().item()))

    ref_rank = _rank_metrics(micro, ref, val_batch, prefix="val")
    cand_rank = _rank_metrics(micro, cand, val_batch, prefix="val")
    rank_diffs = {key: abs(ref_rank[key] - cand_rank[key]) for key in sorted(ref_rank)}
    rank_metric_max_abs_diff = max(rank_diffs.values()) if rank_diffs else 0.0
    telemetry_max_abs_diff, telemetry_diffs = _scalar_telemetry_diff(ref.telemetry, cand.telemetry)

    reference_summary = {
        "loss": float(ref.loss.detach().cpu().item()),
        **ref_rank,
    }
    candidate_summary = {
        "loss": float(cand.loss.detach().cpu().item()),
        **cand_rank,
    }
    report = {
        "pass": bool(
            loss_abs_diff <= float(atol)
            and full_diff <= float(atol)
            and answer_diff <= float(atol)
            and rank_metric_max_abs_diff <= float(atol)
            and telemetry_max_abs_diff <= float(atol)
        ),
        "objective": "dsqg_w_reference_candidate_parity",
        "reference_backend": "reference",
        "candidate_backend": candidate_backend,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": vocab_size,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "val_answer_tokens": float(val_batch.answer_mask.sum().item()),
        "atol": float(atol),
        "loss_abs_diff": loss_abs_diff,
        "full_logits_max_abs_diff": full_diff,
        "answer_logits_max_abs_diff": answer_diff,
        "rank_metric_max_abs_diff": rank_metric_max_abs_diff,
        "rank_metric_abs_diffs": rank_diffs,
        "scalar_telemetry_max_abs_diff": telemetry_max_abs_diff,
        "scalar_telemetry_abs_diffs": telemetry_diffs,
        "reference": reference_summary,
        "candidate": candidate_summary,
    }
    report_path = output / "parity_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in DSQG-W reference-vs-candidate parity harness")
    parser.add_argument("--enable", action="store_true", help="Run parity. Omit to report disabled/skipped.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_parity"))
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--candidate-backend", default="reference")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    if not args.enable:
        report = {
            "enabled": False,
            "skipped": True,
            "pass": True,
            "reason": "pass --enable to run DSQG-W parity",
        }
    else:
        report = run_parity_harness(
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed,
            candidate_backend=args.candidate_backend,
            atol=args.atol,
        )
        report["enabled"] = True
        report["skipped"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
