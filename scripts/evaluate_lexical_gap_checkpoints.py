#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
FROZEN_OBJECTIVE = ROOT / "scripts/frozen_trunk_objective_dsqg_w.py"
MICROTRAIN = ROOT / "scripts/microtrain_dsqg_w_lexical_gap.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_frozen_objective_module():
    return _load_module(FROZEN_OBJECTIVE, "frozen_trunk_objective_for_checkpoint_eval")


def load_microtrain_module():
    return _load_module(MICROTRAIN, "microtrain_dsqg_w_lexical_gap_for_checkpoint_eval")


def _set_eval_env(
    *,
    dsqg_w: bool,
    sites: str = "2,6,final",
    width_cell: bool = False,
    width_bottleneck: int = 64,
    width_gate_init: float = -2.5,
) -> None:
    os.environ["DWARF_DISABLE_BNB"] = "1"
    os.environ["DWARF_LIGER"] = "0"
    os.environ["DWARF_TORCH_COMPILE"] = "0"
    os.environ["DWARF_Q6_G128"] = "0"
    os.environ["DWARF_BS"] = "1"
    os.environ["DWARF_GA"] = "1"
    os.environ["DWARF_MAX_ACC_STEPS"] = "1"
    if dsqg_w:
        os.environ["DWARF_DSQG_W"] = "1"
        os.environ["DWARF_DSQG_W_SITES"] = str(sites)
        os.environ["DWARF_DSQG_W_MAX_CANDIDATES"] = "16"
        os.environ["DWARF_DSQG_W_BOTTLENECK"] = "64"
        os.environ["DWARF_DSQG_W_WIDTH_CELL"] = "1" if width_cell else "0"
        os.environ["DWARF_DSQG_W_WIDTH_BOTTLENECK"] = str(int(width_bottleneck))
        os.environ["DWARF_DSQG_W_WIDTH_GATE_INIT"] = str(float(width_gate_init))
        os.environ["DWARF_DSQG_W_QUESTION"] = "1"
        os.environ["DWARF_DSQG_W_HISA_L3"] = "1"
        os.environ["DWARF_DSQG_W_K_QUESTION"] = "4"
        os.environ["DWARF_DSQG_W_K_HISA_EVIDENCE"] = "4"
        os.environ["DWARF_DSQG_W_K_L3_SKIP"] = "2"
    else:
        for key in [
            "DWARF_DSQG_W",
            "DWARF_DSQG_W_SITES",
            "DWARF_DSQG_W_WIDTH_CELL",
            "DWARF_DSQG_W_WIDTH_BOTTLENECK",
            "DWARF_DSQG_W_WIDTH_GATE_INIT",
            "DWARF_DSQG_W_QUESTION",
            "DWARF_DSQG_W_HISA_L3",
        ]:
            os.environ.pop(key, None)


def load_trainer_module(
    *,
    dsqg_w: bool,
    suffix: str,
    sites: str = "2,6,final",
    width_cell: bool = False,
    width_bottleneck: int = 64,
    width_gate_init: float = -2.5,
):
    _set_eval_env(
        dsqg_w=dsqg_w,
        sites=sites,
        width_cell=width_cell,
        width_bottleneck=width_bottleneck,
        width_gate_init=width_gate_init,
    )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    try:
        return _load_module(TRAINER, f"dwarf_v2_trainer_lexical_gap_eval_{suffix}")
    finally:
        for path in [str(ROOT / "kernels"), str(ROOT)]:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def make_full_model(trainer_mod, *, device: torch.device):
    model = trainer_mod.TriadicJ96Dsr(
        vocab_size=trainer_mod.VOCAB_SIZE,
        embedding_dim=trainer_mod.EMBEDDING_DIM,
        num_heads=trainer_mod.NUM_HEADS,
        ffn_dim=trainer_mod.FFN_DIM,
        seq_len=trainer_mod.MAX_SEQ_LEN,
        dsr_layer=trainer_mod.DSR_LAYER,
        dropout=0.0,
        num_chunks=trainer_mod.NUM_CHUNKS,
        top_k_chunks=trainer_mod.TOP_K_CHUNKS,
    )
    return model.to(device)


def load_full_checkpoint(model, checkpoint_path: Path | str, *, device: torch.device) -> dict[str, Any]:
    # This evaluator is for local trainer-produced full checkpoints. They include
    # metadata pickled with protocol 5, which PyTorch's restricted weights-only
    # unpickler cannot read yet.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state, strict=True)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
    }


def make_causal_answer_targets(batch) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros_like(batch.input_ids)
    predict_mask = torch.zeros_like(batch.answer_mask, dtype=torch.bool)
    answer_rows, answer_cols = torch.where(batch.answer_mask.bool())
    valid = answer_cols > 0
    rows = answer_rows[valid]
    cols = answer_cols[valid]
    labels[rows, cols - 1] = batch.input_ids[rows, cols]
    predict_mask[rows, cols - 1] = True
    if not predict_mask.any():
        raise ValueError("lexical-gap batch has no causal answer targets")
    return labels, predict_mask


def answer_rank_metrics(logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor, *, prefix: str) -> dict[str, float]:
    labels = labels.to(device=logits.device)
    answer_mask = answer_mask.to(device=logits.device)
    masked_logits = logits[answer_mask]
    masked_labels = labels[answer_mask]
    if masked_labels.numel() == 0:
        raise ValueError("answer_mask must select at least one token")
    target_scores = masked_logits.gather(1, masked_labels.view(-1, 1)).squeeze(1)
    ranks = (masked_logits > target_scores.view(-1, 1)).sum(dim=1).to(torch.float32) + 1.0
    sorted_ranks = torch.sort(ranks).values
    n = int(sorted_ranks.numel())
    median_rank = sorted_ranks[n // 2] if n % 2 == 1 else (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2]) / 2.0
    competitor_scores = masked_logits.masked_fill(
        F.one_hot(masked_labels, num_classes=masked_logits.shape[-1]).bool(),
        torch.finfo(masked_logits.dtype).min,
    ).amax(dim=1)
    gold_margin = target_scores - competitor_scores
    reciprocal = 1.0 / ranks
    return {
        f"{prefix}_answer_tokens": float(masked_labels.numel()),
        f"{prefix}_answer_ce": float(F.cross_entropy(masked_logits, masked_labels).detach().float().cpu().item()),
        f"{prefix}_top1_acc": float((ranks <= 1).float().mean().detach().cpu().item()),
        f"{prefix}_top5_acc": float((ranks <= 5).float().mean().detach().cpu().item()),
        f"{prefix}_top10_acc": float((ranks <= 10).float().mean().detach().cpu().item()),
        f"{prefix}_top100_acc": float((ranks <= 100).float().mean().detach().cpu().item()),
        f"{prefix}_mean_rank": float(ranks.mean().detach().cpu().item()),
        f"{prefix}_median_rank": float(median_rank.detach().cpu().item()),
        f"{prefix}_min_rank": float(ranks.min().detach().cpu().item()),
        f"{prefix}_max_rank": float(ranks.max().detach().cpu().item()),
        f"{prefix}_mrr": float(reciprocal.mean().detach().cpu().item()),
        f"{prefix}_mean_gold_margin": float(gold_margin.mean().detach().cpu().item()),
    }


def _record_metrics_by_kind(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kind in sorted({str(record.get("kind", "unknown")) for record in records}):
        row_indices = [idx for idx, record in enumerate(records) if str(record.get("kind", "unknown")) == kind]
        if not row_indices:
            continue
        row_tensor = torch.tensor(row_indices, device=logits.device, dtype=torch.long)
        kind_logits = logits.index_select(0, row_tensor)
        kind_labels = labels.index_select(0, row_tensor)
        kind_mask = mask.index_select(0, row_tensor)
        if kind_mask.any():
            out[kind] = answer_rank_metrics(kind_logits, kind_labels, kind_mask, prefix="lex")
    return out


def _scalar_telemetry(telemetry: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in telemetry.items():
        if torch.is_tensor(value) and value.numel() == 1:
            out[key] = float(value.detach().float().cpu().item())
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def evaluate_checkpoint(
    *,
    name: str,
    checkpoint_path: Path | str,
    dsqg_w: bool,
    records: list[dict[str, Any]],
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    sites: str = "2,6,final",
    width_cell: bool = False,
    width_bottleneck: int = 64,
    width_gate_init: float = -2.5,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    obj = load_frozen_objective_module()
    tokenizer = obj.load_tokenizer(tokenizer_path)
    batch, batch_meta = obj.build_tokenized_lexical_gap_batch(records, tokenizer)
    labels, answer_predict_mask = make_causal_answer_targets(batch)
    trainer = load_trainer_module(
        dsqg_w=dsqg_w,
        suffix=name.lower().replace(" ", "_"),
        sites=sites,
        width_cell=width_cell,
        width_bottleneck=width_bottleneck,
        width_gate_init=width_gate_init,
    )
    model = make_full_model(trainer, device=device)
    load_report = load_full_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    input_ids = batch.input_ids.to(device)
    labels = labels.to(device)
    answer_predict_mask = answer_predict_mask.to(device)
    with torch.no_grad():
        if dsqg_w:
            logits = model(
                input_ids,
                dsqg_w_question_indices=batch.question_indices.to(device),
                dsqg_w_hisa_evidence_indices=batch.hisa_evidence_indices.to(device),
                dsqg_w_l3_skip_indices=batch.l3_skip_indices.to(device),
            )
        else:
            logits = model(input_ids)
    metrics = answer_rank_metrics(logits, labels, answer_predict_mask, prefix="lex")
    by_kind = _record_metrics_by_kind(logits, labels, answer_predict_mask, records)
    telemetry = _scalar_telemetry(getattr(model, "dsqg_w_last_telemetry", {})) if dsqg_w else {}
    return {
        "name": name,
        "checkpoint_path": str(checkpoint_path),
        "dsqg_w_enabled": bool(dsqg_w),
        "dsqg_w_width_cell": bool(width_cell) if dsqg_w else False,
        "dsqg_w_width_bottleneck": int(width_bottleneck) if dsqg_w and width_cell else None,
        "dsqg_w_width_gate_init": float(width_gate_init) if dsqg_w and width_cell else None,
        "dsqg_w_sites": list(getattr(model, "dsqg_w_site_keys", ())) if dsqg_w else [],
        "load_report": load_report,
        "examples": len(records),
        "tokenized": True,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": int(batch_meta["tokenizer_vocab_size"]),
        "max_seq_len": int(batch.input_ids.shape[1]),
        **metrics,
        "by_kind": by_kind,
        "telemetry": telemetry,
    }


def build_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("need at least two runs to compare")
    by_name = {run["name"]: run for run in runs}
    d_run = by_name.get("DSQG-D") or runs[0]
    w_run = by_name.get("DSQG-W") or runs[1]

    def delta(key: str) -> float | None:
        if key not in d_run or key not in w_run:
            return None
        return float(w_run[key] - d_run[key])

    return {
        "objective": "lexical_gap_causal_answer_checkpoint_eval",
        "runs": runs,
        "best_by_answer_ce": min(runs, key=lambda row: row["lex_answer_ce"])["name"],
        "best_by_mrr": max(runs, key=lambda row: row["lex_mrr"])["name"],
        "best_by_top1": max(runs, key=lambda row: row["lex_top1_acc"])["name"],
        "best_by_mean_rank": min(runs, key=lambda row: row["lex_mean_rank"])["name"],
        "w_minus_d_answer_ce": delta("lex_answer_ce"),
        "w_minus_d_mrr": delta("lex_mrr"),
        "w_minus_d_top1_acc": delta("lex_top1_acc"),
        "w_minus_d_top5_acc": delta("lex_top5_acc"),
        "w_minus_d_mean_rank": delta("lex_mean_rank"),
        "w_minus_d_mean_gold_margin": delta("lex_mean_gold_margin"),
    }


def run_comparison(
    *,
    dsqg_d_checkpoint: Path,
    dsqg_w_checkpoint: Path,
    output_path: Path,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    val_size: int = 144,
    seed: int = 20260628,
    sites: str = "2,6,final",
    dsqg_w_width_cell: bool = False,
    dsqg_w_width_bottleneck: int = 64,
    dsqg_w_width_gate_init: float = -2.5,
    device: str | None = None,
) -> dict[str, Any]:
    micro = load_microtrain_module()
    _, val_records = micro.generate_lexical_gap_dataset(train_size=max(1, val_size), val_size=val_size, seed=seed)
    runs = [
        evaluate_checkpoint(
            name="DSQG-D",
            checkpoint_path=dsqg_d_checkpoint,
            dsqg_w=False,
            records=val_records,
            tokenizer_path=tokenizer_path,
            sites=sites,
            device=device,
        ),
        evaluate_checkpoint(
            name="DSQG-W",
            checkpoint_path=dsqg_w_checkpoint,
            dsqg_w=True,
            records=val_records,
            tokenizer_path=tokenizer_path,
            sites=sites,
            width_cell=dsqg_w_width_cell,
            width_bottleneck=dsqg_w_width_bottleneck,
            width_gate_init=dsqg_w_width_gate_init,
            device=device,
        ),
    ]
    report = build_comparison(runs)
    report.update({
        "seed": int(seed),
        "val_examples": int(val_size),
        "sites": sites,
        "dsqg_w_width_cell": bool(dsqg_w_width_cell),
        "dsqg_w_width_bottleneck": int(dsqg_w_width_bottleneck) if dsqg_w_width_cell else None,
        "dsqg_w_width_gate_init": float(dsqg_w_width_gate_init) if dsqg_w_width_cell else None,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate matched DSQG-D/DSQG-W checkpoints on causal lexical-gap answer prediction")
    parser.add_argument("--dsqg-d-checkpoint", type=Path, required=True)
    parser.add_argument("--dsqg-w-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/lexical_gap_checkpoint_eval.json"))
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--val-size", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--sites", default="2,6,final")
    parser.add_argument("--dsqg-w-width-cell", action="store_true", help="Instantiate the DSQG-W checkpoint with the width cell enabled.")
    parser.add_argument("--dsqg-w-width-bottleneck", type=int, default=64)
    parser.add_argument("--dsqg-w-width-gate-init", type=float, default=-2.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    report = run_comparison(
        dsqg_d_checkpoint=args.dsqg_d_checkpoint,
        dsqg_w_checkpoint=args.dsqg_w_checkpoint,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        val_size=args.val_size,
        seed=args.seed,
        sites=args.sites,
        dsqg_w_width_cell=args.dsqg_w_width_cell,
        dsqg_w_width_bottleneck=args.dsqg_w_width_bottleneck,
        dsqg_w_width_gate_init=args.dsqg_w_width_gate_init,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
