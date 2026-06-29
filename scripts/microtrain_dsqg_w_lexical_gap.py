#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_OBJECTIVE = ROOT / "scripts/frozen_trunk_objective_dsqg_w.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_objective_module():
    spec = importlib.util.spec_from_file_location("frozen_trunk_objective_dsqg_w_for_microtrain", FROZEN_OBJECTIVE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


LEXICAL_FACTS = [
    {"kind": "metal_symbol", "subject": "copper", "answer": "Cu", "evidence": "copper"},
    {"kind": "metal_symbol", "subject": "sodium", "answer": "Na", "evidence": "sodium"},
    {"kind": "metal_symbol", "subject": "iron", "answer": "Fe", "evidence": "iron"},
    {"kind": "metal_symbol", "subject": "gold", "answer": "Au", "evidence": "gold"},
    {"kind": "juvenile", "subject": "dog", "answer": "puppy", "evidence": "dog"},
    {"kind": "juvenile", "subject": "cat", "answer": "kitten", "evidence": "cat"},
    {"kind": "juvenile", "subject": "horse", "answer": "foal", "evidence": "horse"},
    {"kind": "juvenile", "subject": "cow", "answer": "calf", "evidence": "cow"},
    {"kind": "color", "subject": "banana", "answer": "yellow", "evidence": "banana"},
    {"kind": "color", "subject": "snow", "answer": "white", "evidence": "snow"},
    {"kind": "color", "subject": "grass", "answer": "green", "evidence": "grass"},
    {"kind": "color", "subject": "mars", "answer": "red", "evidence": "mars"},
]

TEMPLATES = [
    "Fact: The target item is {subject}. Bridge: use the learned {kind} mapping. Question: Which answer matches the target? Answer: {answer}",
    "Context: A clue mentions {subject}. Rule: infer the associated {kind}. Query: What should be produced? Answer: {answer}",
    "Background: Remember {subject} for the {kind} relation. Prompt: Give the linked response. Answer: {answer}",
    "Evidence: The relevant word is {subject}. Instruction: transfer through the {kind} table. Question: What is the result? Answer: {answer}",
    "Source: {subject} appears in the passage. Task: choose its {kind} companion. Answer: {answer}",
    "Note: The semantic key is {subject}. Request: output the paired {kind} value. Answer: {answer}",
]


def _find_positions(tokens: list[str], predicate) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if predicate(token)]


def _record_from_fact(*, split: str, index: int, fact: dict[str, str], template: str) -> dict[str, Any]:
    prompt = template.format(**fact)
    tokens = prompt.split()
    answer_positions = [idx for idx, token in enumerate(tokens) if token == fact["answer"]]
    if not answer_positions:
        raise ValueError(f"template did not expose answer token {fact['answer']!r}: {prompt}")
    evidence_positions = _find_positions(tokens, lambda token: fact["evidence"] in token.strip(".,:;!?"))
    if not evidence_positions:
        raise ValueError(f"template did not expose evidence token {fact['evidence']!r}: {prompt}")
    answer_pos = answer_positions[-1]
    question_start = min(_find_positions(tokens, lambda token: token in {"Question:", "Query:", "Prompt:", "Task:", "Request:"}) or [max(0, answer_pos - 5)])
    question_indices = [idx for idx in range(question_start, answer_pos) if idx < answer_pos][-4:]
    hisa_evidence_indices = evidence_positions[:2]
    if len(hisa_evidence_indices) < 4:
        pre_answer = [idx for idx in range(0, answer_pos) if idx not in hisa_evidence_indices]
        hisa_evidence_indices = (hisa_evidence_indices + pre_answer[: 4 - len(hisa_evidence_indices)])[:4]
    l3_skip_indices = [idx for idx in [0, 1, max(0, question_start - 1)] if idx < answer_pos]
    l3_skip_indices = list(dict.fromkeys(l3_skip_indices))[:3]
    return {
        "id": f"{split}_{index:04d}_{fact['kind']}_{fact['subject']}",
        "split": split,
        "kind": fact["kind"],
        "subject": fact["subject"],
        "answer": fact["answer"],
        "prompt": prompt,
        "tokens": tokens,
        "answer_positions": [answer_pos],
        "gold_evidence_indices": evidence_positions[:1],
        "question_indices": question_indices,
        "hisa_evidence_indices": hisa_evidence_indices,
        "l3_skip_indices": l3_skip_indices,
    }


def generate_lexical_gap_dataset(*, train_size: int = 128, val_size: int = 32, seed: int = 20260628) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)

    def make_split(split: str, size: int, *, offset: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i in range(size):
            fact = LEXICAL_FACTS[(i + offset) % len(LEXICAL_FACTS)]
            template = TEMPLATES[(i + rng.randrange(len(TEMPLATES))) % len(TEMPLATES)]
            records.append(_record_from_fact(split=split, index=i, fact=fact, template=template))
        rng.shuffle(records)
        return records

    return make_split("train", train_size, offset=0), make_split("val", val_size, offset=3)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _loss(obj, model, batch) -> float:
    with torch.no_grad():
        result = obj.compute_frozen_dsqg_w_objective(model, batch)
    return float(result.loss.detach().float().cpu().item())


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
    if n % 2 == 1:
        median_rank = float(sorted_ranks[n // 2].detach().cpu().item())
    else:
        median_rank = float(((sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2]) / 2.0).detach().cpu().item())
    return {
        f"{prefix}_answer_tokens": float(masked_labels.numel()),
        f"{prefix}_top1_acc": float((ranks <= 1).to(torch.float32).mean().detach().cpu().item()),
        f"{prefix}_top5_acc": float((ranks <= 5).to(torch.float32).mean().detach().cpu().item()),
        f"{prefix}_top10_acc": float((ranks <= 10).to(torch.float32).mean().detach().cpu().item()),
        f"{prefix}_mean_rank": float(ranks.mean().detach().cpu().item()),
        f"{prefix}_median_rank": median_rank,
        f"{prefix}_min_rank": float(ranks.min().detach().cpu().item()),
        f"{prefix}_max_rank": float(ranks.max().detach().cpu().item()),
    }


def _loss_and_rank_metrics(obj, model, batch, *, prefix: str) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        result = obj.compute_frozen_dsqg_w_objective(model, batch)
    loss = float(result.loss.detach().float().cpu().item())
    return loss, answer_rank_metrics(result.logits, batch.labels, batch.answer_mask, prefix=prefix)


def _changed_names(before: dict[str, torch.Tensor], model, *, prefix: str | None) -> list[str]:
    changed: list[str] = []
    for name, param in model.named_parameters():
        if prefix is not None and not name.startswith(prefix):
            continue
        if prefix is None and name.startswith("dsqg_w."):
            continue
        old = before.get(name)
        if old is not None and not torch.equal(old, param.detach()):
            changed.append(name)
    return changed


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def run_microtrain(
    *,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    output_dir: Path | str,
    train_size: int = 128,
    val_size: int = 32,
    steps: int = 16,
    lr: float = 1e-3,
    seed: int = 20260628,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_records, val_records = generate_lexical_gap_dataset(train_size=train_size, val_size=val_size, seed=seed)
    _write_jsonl(output / "train.jsonl", train_records)
    _write_jsonl(output / "val.jsonl", val_records)

    obj = load_objective_module()
    tokenizer = obj.load_tokenizer(tokenizer_path)
    train_batch, train_meta = obj.build_tokenized_lexical_gap_batch(train_records, tokenizer)
    val_batch, val_meta = obj.build_tokenized_lexical_gap_batch(val_records, tokenizer)
    vocab_size = int(train_meta["tokenizer_vocab_size"])
    seq_len = max(int(train_batch.input_ids.shape[1]), int(val_batch.input_ids.shape[1]))

    trainer = obj.load_trainer(enable_objective=True, suffix="microtrain")
    torch.manual_seed(seed)
    model = obj.make_tiny_model(trainer, vocab_size=vocab_size, ffn_dim=64, seq_len=seq_len)
    obj.prepare_model_for_frozen_dsqg_w_objective(model)
    optimizer = obj.make_dsqg_w_optimizer(model, lr=lr)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    train_initial, train_rank_initial = _loss_and_rank_metrics(obj, model, train_batch, prefix="train_initial")
    val_initial, val_rank_initial = _loss_and_rank_metrics(obj, model, val_batch, prefix="val_initial")
    train_losses_before_step: list[float] = []
    step_reports: list[dict[str, Any]] = []
    model.train()
    for _ in range(int(steps)):
        step_report = obj.run_one_frozen_dsqg_w_step(model, train_batch, optimizer)
        step_reports.append(step_report)
        train_losses_before_step.append(float(step_report["loss_before_step"]))

    model.eval()
    train_final, train_rank_final = _loss_and_rank_metrics(obj, model, train_batch, prefix="train_final")
    val_final, val_rank_final = _loss_and_rank_metrics(obj, model, val_batch, prefix="val_final")
    changed_dsqg_w = _changed_names(before, model, prefix="dsqg_w.")
    changed_frozen = _changed_names(before, model, prefix=None)

    checkpoint = obj.save_dsqg_w_checkpoint(
        model,
        output / "checkpoint",
        metadata={
            "seed": seed,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_vocab_size": vocab_size,
            "train_examples": len(train_records),
            "val_examples": len(val_records),
            "train_jsonl": str(output / "train.jsonl"),
            "val_jsonl": str(output / "val.jsonl"),
            "steps": int(steps),
            "lr": float(lr),
            "train_loss_initial": train_initial,
            "train_loss_final": train_final,
            "val_loss_initial": val_initial,
            "val_loss_final": val_final,
            "candidate_settings": obj._candidate_settings(),
        },
    )

    torch.manual_seed(seed)
    fresh_model = obj.make_tiny_model(trainer, vocab_size=vocab_size, ffn_dim=64, seq_len=seq_len)
    obj.prepare_model_for_frozen_dsqg_w_objective(fresh_model)
    load_report = obj.load_dsqg_w_checkpoint(fresh_model, checkpoint["state_path"])
    roundtrip_loss = _loss(obj, fresh_model, train_batch)
    roundtrip_delta = roundtrip_loss - train_final

    telemetry = dict(step_reports[-1]["telemetry"]) if step_reports else {}
    rank_metrics: dict[str, float] = {}
    rank_metrics.update(train_rank_initial)
    rank_metrics.update(train_rank_final)
    rank_metrics.update(val_rank_initial)
    rank_metrics.update(val_rank_final)
    for split in ["train", "val"]:
        for metric in ["top1_acc", "top5_acc", "top10_acc", "mean_rank", "median_rank", "min_rank", "max_rank"]:
            rank_metrics[f"{split}_{metric}_initial"] = rank_metrics.pop(f"{split}_initial_{metric}")
            rank_metrics[f"{split}_{metric}_final"] = rank_metrics.pop(f"{split}_final_{metric}")
            rank_metrics[f"{split}_{metric}_delta"] = rank_metrics[f"{split}_{metric}_final"] - rank_metrics[f"{split}_{metric}_initial"]
        rank_metrics[f"{split}_answer_tokens_initial"] = rank_metrics.pop(f"{split}_initial_answer_tokens")
        rank_metrics[f"{split}_answer_tokens_final"] = rank_metrics.pop(f"{split}_final_answer_tokens")
    report = {
        "pass": bool(
            _finite(train_initial)
            and _finite(train_final)
            and _finite(val_initial)
            and _finite(val_final)
            and train_final < train_initial
            and len(changed_dsqg_w) > 0
            and len(changed_frozen) == 0
            and abs(roundtrip_delta) < 1e-6
            and not load_report["missing_keys"]
            and not load_report["unexpected_keys"]
        ),
        "objective": "dsqg_w_lexical_gap_microtrain",
        "tokenized": True,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": vocab_size,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "steps": int(steps),
        "lr": float(lr),
        "train_loss_initial": train_initial,
        "train_loss_final": train_final,
        "train_loss_delta": train_final - train_initial,
        "val_loss_initial": val_initial,
        "val_loss_final": val_final,
        "val_loss_delta": val_final - val_initial,
        **rank_metrics,
        "train_losses_before_step": train_losses_before_step,
        "train_answer_tokens": float(train_batch.answer_mask.sum().item()),
        "val_answer_tokens": float(val_batch.answer_mask.sum().item()),
        "changed_dsqg_w_param_count": len(changed_dsqg_w),
        "changed_frozen_param_count": len(changed_frozen),
        "changed_frozen_param_names": changed_frozen,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "checkpoint_roundtrip_loss": roundtrip_loss,
        "checkpoint_roundtrip_loss_delta": roundtrip_delta,
        "telemetry": telemetry,
        "train_jsonl": str(output / "train.jsonl"),
        "val_jsonl": str(output / "val.jsonl"),
    }
    report_path = output / "microtrain_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in frozen-trunk DSQG-W lexical-gap microtrainer")
    parser.add_argument("--enable", action="store_true", help="Run the microtrainer. Omit to report the disabled default.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_microtrain"))
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    if not args.enable:
        report = {
            "enabled": False,
            "skipped": True,
            "pass": True,
            "reason": "pass --enable to run the DSQG-W lexical-gap microtrainer",
        }
    else:
        report = run_microtrain(
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
        )
        report["enabled"] = True
        report["skipped"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
