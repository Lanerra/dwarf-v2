#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from kernels.dsqg_w.dsqg_w_mvp import answer_masked_loss

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


class FrozenDSQGWBatch:
    def __init__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        answer_mask: torch.Tensor,
        question_indices: torch.Tensor | None = None,
        hisa_evidence_indices: torch.Tensor | None = None,
        l3_skip_indices: torch.Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.answer_mask = answer_mask
        self.question_indices = question_indices
        self.hisa_evidence_indices = hisa_evidence_indices
        self.l3_skip_indices = l3_skip_indices


class FrozenObjectiveResult:
    def __init__(self, *, loss: torch.Tensor, logits: torch.Tensor, telemetry: dict[str, float]) -> None:
        self.loss = loss
        self.logits = logits
        self.telemetry = telemetry


def _set_common_env() -> None:
    os.environ["DWARF_DISABLE_BNB"] = "1"
    os.environ["DWARF_LIGER"] = "0"
    os.environ["DWARF_TORCH_COMPILE"] = "0"
    os.environ["DWARF_Q6_G128"] = "0"


def load_trainer(*, enable_objective: bool, suffix: str):
    _set_common_env()
    if enable_objective:
        os.environ["DWARF_DSQG_W"] = "1"
        os.environ["DWARF_DSQG_W_QUESTION"] = "1"
        os.environ["DWARF_DSQG_W_HISA_L3"] = "1"
        os.environ["DWARF_DSQG_W_MAX_CANDIDATES"] = os.environ.get("DWARF_DSQG_W_MAX_CANDIDATES", "16")
        os.environ["DWARF_DSQG_W_BOTTLENECK"] = os.environ.get("DWARF_DSQG_W_BOTTLENECK", "64")
        os.environ["DWARF_DSQG_W_K_QUESTION"] = os.environ.get("DWARF_DSQG_W_K_QUESTION", "4")
        os.environ["DWARF_DSQG_W_K_HISA_EVIDENCE"] = os.environ.get("DWARF_DSQG_W_K_HISA_EVIDENCE", "4")
        os.environ["DWARF_DSQG_W_K_L3_SKIP"] = os.environ.get("DWARF_DSQG_W_K_L3_SKIP", "2")
    else:
        for key in ["DWARF_DSQG_W", "DWARF_DSQG_W_QUESTION", "DWARF_DSQG_W_HISA_L3"]:
            os.environ.pop(key, None)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    try:
        spec = importlib.util.spec_from_file_location(f"dwarf_v2_trainer_frozen_dsqgw_{suffix}", TRAINER)
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


def make_tiny_model(
    trainer_mod,
    *,
    vocab_size: int = 128,
    ffn_dim: int = 64,
    seq_len: int = 32,
    device: torch.device | str | None = None,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = trainer_mod.TriadicJ96Dsr(
        vocab_size=vocab_size,
        embedding_dim=trainer_mod.EMBEDDING_DIM,
        num_heads=trainer_mod.NUM_HEADS,
        ffn_dim=ffn_dim,
        seq_len=seq_len,
        dsr_layer=trainer_mod.DSR_LAYER,
        dropout=0.0,
        num_chunks=trainer_mod.NUM_CHUNKS,
        top_k_chunks=trainer_mod.TOP_K_CHUNKS,
    )
    return model.to(device)


def prepare_model_for_frozen_dsqg_w_objective(model) -> dict[str, int]:
    if not getattr(model, "dsqg_w_enabled", False) or getattr(model, "dsqg_w", None) is None:
        raise ValueError("DSQG-W must be enabled before preparing the frozen objective")

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.dsqg_w.parameters():
        param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {
        "trainable_param_count": int(trainable),
        "frozen_param_count": int(frozen),
    }


def _scalar_telemetry(telemetry: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in telemetry.items():
        if torch.is_tensor(value) and value.numel() == 1:
            out[key] = float(value.detach().float().cpu().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _to_device(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


def compute_frozen_dsqg_w_objective(model, batch: FrozenDSQGWBatch) -> FrozenObjectiveResult:
    if not getattr(model, "dsqg_w_enabled", False):
        raise ValueError("DSQG-W must be enabled for the frozen objective")
    device = _model_device(model)
    input_ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)
    answer_mask = batch.answer_mask.to(device)
    logits = model(
        input_ids,
        dsqg_w_question_indices=_to_device(batch.question_indices, device),
        dsqg_w_hisa_evidence_indices=_to_device(batch.hisa_evidence_indices, device),
        dsqg_w_l3_skip_indices=_to_device(batch.l3_skip_indices, device),
    )
    loss = answer_masked_loss(logits, labels, answer_mask)
    telemetry = _scalar_telemetry(getattr(model, "dsqg_w_last_telemetry", {}))
    telemetry.update(
        {
            "dsqg_w_objective_enabled": 1.0,
            "dsqg_w_objective_answer_ce": float(loss.detach().float().cpu().item()),
            "dsqg_w_objective_answer_tokens": float(answer_mask.bool().sum().item()),
            "dsqg_w_objective_batch_tokens": float(input_ids.numel()),
        }
    )
    return FrozenObjectiveResult(loss=loss, logits=logits, telemetry=telemetry)


def make_synthetic_batch(*, batch: int = 1, seq_len: int = 16, vocab_size: int = 128, seed: int = 20260628) -> FrozenDSQGWBatch:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    answer_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    answer_mask[:, -1] = True
    question_indices = torch.arange(0, 4, dtype=torch.long).repeat(batch, 1)
    hisa_evidence_indices = torch.arange(1, 5, dtype=torch.long).repeat(batch, 1)
    l3_skip_indices = torch.tensor([[5, 6]], dtype=torch.long).repeat(batch, 1)
    return FrozenDSQGWBatch(
        input_ids=input_ids,
        labels=labels,
        answer_mask=answer_mask,
        question_indices=question_indices,
        hisa_evidence_indices=hisa_evidence_indices,
        l3_skip_indices=l3_skip_indices,
    )


def run_smoke_objective(*, enable: bool | None = None, seed: int = 20260628) -> dict[str, Any]:
    if enable is None:
        enable = os.getenv("DWARF_DSQG_W_FROZEN_OBJECTIVE", "0") == "1"
    if not enable:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "DWARF_DSQG_W_FROZEN_OBJECTIVE is disabled",
            "pass": True,
        }

    trainer = load_trainer(enable_objective=True, suffix="smoke")
    torch.manual_seed(seed)
    model = make_tiny_model(trainer, vocab_size=128, ffn_dim=64, seq_len=16)
    counts = prepare_model_for_frozen_dsqg_w_objective(model)
    model.train()
    batch = make_synthetic_batch(batch=1, seq_len=16, vocab_size=128, seed=seed + 1)
    result = compute_frozen_dsqg_w_objective(model, batch)
    result.loss.backward()

    grad_names = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and param.grad.detach().abs().sum().item() > 0.0
    ]
    grad_scope_ok = bool(grad_names) and all(name.startswith("dsqg_w.") for name in grad_names)
    telemetry = dict(result.telemetry)
    telemetry.update({key: float(value) for key, value in counts.items()})
    return {
        "enabled": True,
        "skipped": False,
        "objective": "frozen_trunk_answer_only_ce",
        "loss": float(result.loss.detach().float().cpu().item()),
        "grad_param_names": grad_names,
        "grad_scope_ok": grad_scope_ok,
        "telemetry": telemetry,
        "pass": bool(torch.isfinite(result.loss).item() and grad_scope_ok and telemetry["dsqg_w_objective_answer_tokens"] > 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Disabled-by-default DSQG-W frozen-trunk answer-only CE smoke")
    parser.add_argument("--enable", action="store_true", help="Run the objective smoke. Otherwise report the disabled default.")
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    report = run_smoke_objective(enable=args.enable, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
