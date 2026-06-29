#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
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


class LoadedTokenizer:
    def __init__(self, path: Path | str) -> None:
        from tokenizers import Tokenizer

        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def get_vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())


def load_tokenizer(path: Path | str) -> LoadedTokenizer:
    return LoadedTokenizer(path)


def _set_common_env() -> None:
    os.environ["DWARF_DISABLE_BNB"] = "1"
    os.environ["DWARF_LIGER"] = "0"
    os.environ["DWARF_TORCH_COMPILE"] = "0"
    os.environ["DWARF_Q6_G128"] = "0"


def load_trainer(*, enable_objective: bool, suffix: str, dsqg_w_sites: str | None = None):
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
        if dsqg_w_sites is not None:
            os.environ["DWARF_DSQG_W_SITES"] = str(dsqg_w_sites)
        else:
            os.environ.pop("DWARF_DSQG_W_SITES", None)
    else:
        for key in ["DWARF_DSQG_W", "DWARF_DSQG_W_QUESTION", "DWARF_DSQG_W_HISA_L3", "DWARF_DSQG_W_SITES"]:
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
    modules = getattr(model, "dsqg_w_blocks", None)
    if modules is not None and len(modules) > 0:
        for param in modules.parameters():
            param.requires_grad_(True)
    else:
        for param in model.dsqg_w.parameters():
            param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {
        "trainable_param_count": int(trainable),
        "frozen_param_count": int(frozen),
    }


def is_dsqg_w_param_name(name: str) -> bool:
    return name.startswith("dsqg_w.") or name.startswith("dsqg_w_blocks.")


def make_dsqg_w_optimizer(model, *, lr: float = 1e-4, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    if not getattr(model, "dsqg_w_enabled", False) or getattr(model, "dsqg_w", None) is None:
        raise ValueError("DSQG-W must be enabled before constructing its optimizer")
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    non_dsqg = [name for name, _ in named_params if not is_dsqg_w_param_name(name)]
    if non_dsqg:
        raise ValueError(f"frozen objective optimizer saw non-DSQG-W trainable params: {non_dsqg}")
    params = [param for _, param in named_params]
    if not params:
        raise ValueError("no trainable DSQG-W parameters found")
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))


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


def _param_snapshot(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _changed_names(before: dict[str, torch.Tensor], model, *, prefix: str | None) -> list[str]:
    changed: list[str] = []
    for name, param in model.named_parameters():
        if prefix == "dsqg_w.":
            if not is_dsqg_w_param_name(name):
                continue
        elif prefix is not None and not name.startswith(prefix):
            continue
        if prefix is None and is_dsqg_w_param_name(name):
            continue
        old = before.get(name)
        if old is not None and not torch.equal(old, param.detach()):
            changed.append(name)
    return changed


def run_one_frozen_dsqg_w_step(
    model,
    batch: FrozenDSQGWBatch,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    before = _param_snapshot(model)
    optimizer.zero_grad(set_to_none=True)
    result = compute_frozen_dsqg_w_objective(model, batch)
    result.loss.backward()
    grad_names = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and param.grad.detach().abs().sum().item() > 0.0
    ]
    grad_scope_ok = bool(grad_names) and all(is_dsqg_w_param_name(name) for name in grad_names)
    optimizer.step()
    changed_dsqg_w = _changed_names(before, model, prefix="dsqg_w.")
    changed_frozen = _changed_names(before, model, prefix=None)

    telemetry = dict(result.telemetry)
    telemetry.update(
        {
            "dsqg_w_step_lr": float(optimizer.param_groups[0]["lr"]),
            "dsqg_w_step_grad_param_count": float(len(grad_names)),
            "dsqg_w_step_changed_param_count": float(len(changed_dsqg_w)),
        }
    )
    with torch.no_grad():
        post = compute_frozen_dsqg_w_objective(model, batch)
    telemetry["dsqg_w_objective_answer_ce_after_step"] = float(post.loss.detach().float().cpu().item())

    return {
        "step": 1,
        "loss_before_step": float(result.loss.detach().float().cpu().item()),
        "loss_after_step": float(post.loss.detach().float().cpu().item()),
        "grad_param_names": grad_names,
        "grad_scope_ok": grad_scope_ok,
        "changed_dsqg_w_param_names": changed_dsqg_w,
        "changed_dsqg_w_param_count": len(changed_dsqg_w),
        "changed_frozen_param_names": changed_frozen,
        "changed_frozen_param_count": len(changed_frozen),
        "telemetry": telemetry,
        "pass": bool(grad_scope_ok and len(changed_dsqg_w) > 0 and len(changed_frozen) == 0),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _candidate_settings() -> dict[str, int]:
    return {
        "max_candidates": int(os.environ.get("DWARF_DSQG_W_MAX_CANDIDATES", "16")),
        "k_question": int(os.environ.get("DWARF_DSQG_W_K_QUESTION", "4")),
        "k_hisa_evidence": int(os.environ.get("DWARF_DSQG_W_K_HISA_EVIDENCE", "4")),
        "k_l3_skip": int(os.environ.get("DWARF_DSQG_W_K_L3_SKIP", "2")),
    }


def _dsqg_w_checkpoint_module(model):
    blocks = getattr(model, "dsqg_w_blocks", None)
    if blocks is not None and len(blocks) > 1:
        return blocks, "model.dsqg_w_blocks.state_dict"
    return model.dsqg_w, "model.dsqg_w.state_dict"


def save_dsqg_w_checkpoint(model, output_dir: Path | str, *, metadata: dict[str, Any] | None = None) -> dict[str, str]:
    if not getattr(model, "dsqg_w_enabled", False) or getattr(model, "dsqg_w", None) is None:
        raise ValueError("DSQG-W must be enabled before saving its checkpoint")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "dsqg_w_state.pt"
    metadata_path = out_dir / "dsqg_w_metadata.json"
    module, contains = _dsqg_w_checkpoint_module(model)
    state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    torch.save({"dsqg_w_state_dict": state_dict}, state_path)
    sidecar = {
        "contains": contains,
        "state_path": str(state_path),
        "metadata": _jsonable(metadata or {}),
        "git_commit": _git_commit(),
    }
    metadata_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "state_path": str(state_path),
        "metadata_path": str(metadata_path),
        "git_commit": sidecar["git_commit"],
    }


def load_dsqg_w_checkpoint(model, state_path: Path | str) -> dict[str, list[str]]:
    if not getattr(model, "dsqg_w_enabled", False) or getattr(model, "dsqg_w", None) is None:
        raise ValueError("DSQG-W must be enabled before loading its checkpoint")
    payload = torch.load(state_path, map_location=_model_device(model), weights_only=True)
    state_dict = payload["dsqg_w_state_dict"]
    blocks = getattr(model, "dsqg_w_blocks", None)
    if blocks is not None and len(blocks) > 1:
        incompatible = blocks.load_state_dict(state_dict, strict=False)
    else:
        incompatible = model.dsqg_w.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


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


def load_lexical_gap_records(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            for key in ["tokens", "answer_positions", "question_indices", "hisa_evidence_indices", "l3_skip_indices"]:
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing required key {key!r}")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contained no records")
    return records


def _build_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    vocab = {"<pad>": 0}
    for record in records:
        for token in record["tokens"]:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _pad_index_rows(records: list[dict[str, Any]], key: str, width: int | None = None) -> torch.Tensor:
    if width is None:
        width = max(len(record[key]) for record in records)
    out = torch.full((len(records), width), -1, dtype=torch.long)
    for row, record in enumerate(records):
        values = [int(value) for value in record[key]][:width]
        if values:
            out[row, : len(values)] = torch.tensor(values, dtype=torch.long)
    return out


def build_lexical_gap_batch(records: list[dict[str, Any]]) -> tuple[FrozenDSQGWBatch, dict[str, int]]:
    vocab = _build_vocab(records)
    max_len = max(len(record["tokens"]) for record in records)
    input_ids = torch.zeros((len(records), max_len), dtype=torch.long)
    labels = torch.zeros_like(input_ids)
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, record in enumerate(records):
        ids = torch.tensor([vocab[token] for token in record["tokens"]], dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        labels[row, : ids.numel()] = ids
        for pos in record["answer_positions"]:
            answer_mask[row, int(pos)] = True
    return (
        FrozenDSQGWBatch(
            input_ids=input_ids,
            labels=labels,
            answer_mask=answer_mask,
            question_indices=_pad_index_rows(records, "question_indices"),
            hisa_evidence_indices=_pad_index_rows(records, "hisa_evidence_indices"),
            l3_skip_indices=_pad_index_rows(records, "l3_skip_indices"),
        ),
        vocab,
    )


def _record_token_char_spans(record: dict[str, Any]) -> list[tuple[int, int]]:
    prompt = record["prompt"]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in record["tokens"]:
        start = prompt.find(token, cursor)
        if start < 0:
            raise ValueError(f"could not align token {token!r} in record {record.get('id', '<unknown>')!r}")
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def _first_encoded_token_overlapping(offsets: list[tuple[int, int]], span: tuple[int, int]) -> int:
    span_start, span_end = span
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if tok_start < span_end and tok_end > span_start:
            return idx
    return -1


def _encoded_tokens_overlapping(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    span_start, span_end = span
    return [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > tok_start and tok_start < span_end and tok_end > span_start
    ]


def _map_word_positions_to_encoded_indices(record: dict[str, Any], offsets: list[tuple[int, int]], key: str) -> list[int]:
    spans = _record_token_char_spans(record)
    mapped: list[int] = []
    for pos in record[key]:
        pos = int(pos)
        mapped.append(_first_encoded_token_overlapping(offsets, spans[pos]) if 0 <= pos < len(spans) else -1)
    return mapped


def build_tokenized_lexical_gap_batch(
    records: list[dict[str, Any]],
    tokenizer: LoadedTokenizer,
) -> tuple[FrozenDSQGWBatch, dict[str, Any]]:
    encodings = [tokenizer.encode(record["prompt"]) for record in records]
    max_len = max(len(encoding.ids) for encoding in encodings)
    input_ids = torch.zeros((len(records), max_len), dtype=torch.long)
    labels = torch.zeros_like(input_ids)
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    q_rows: list[list[int]] = []
    h_rows: list[list[int]] = []
    l3_rows: list[list[int]] = []
    for row, (record, encoding) in enumerate(zip(records, encodings)):
        ids = torch.tensor(encoding.ids, dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        labels[row, : ids.numel()] = ids
        offsets = [(int(start), int(end)) for start, end in encoding.offsets]
        spans = _record_token_char_spans(record)
        for pos in record["answer_positions"]:
            pos = int(pos)
            if 0 <= pos < len(spans):
                for tok_idx in _encoded_tokens_overlapping(offsets, spans[pos]):
                    answer_mask[row, tok_idx] = True
        q_rows.append(_map_word_positions_to_encoded_indices(record, offsets, "question_indices"))
        h_rows.append(_map_word_positions_to_encoded_indices(record, offsets, "hisa_evidence_indices"))
        l3_rows.append(_map_word_positions_to_encoded_indices(record, offsets, "l3_skip_indices"))

    def pad_rows(rows: list[list[int]]) -> torch.Tensor:
        width = max(len(row) for row in rows)
        out = torch.full((len(rows), width), -1, dtype=torch.long)
        for row_idx, row_values in enumerate(rows):
            if row_values:
                out[row_idx, : len(row_values)] = torch.tensor(row_values, dtype=torch.long)
        return out

    if not answer_mask.any():
        raise ValueError("tokenized lexical-gap batch produced no answer tokens")
    return (
        FrozenDSQGWBatch(
            input_ids=input_ids,
            labels=labels,
            answer_mask=answer_mask,
            question_indices=pad_rows(q_rows),
            hisa_evidence_indices=pad_rows(h_rows),
            l3_skip_indices=pad_rows(l3_rows),
        ),
        {
            "tokenized": True,
            "tokenizer_path": tokenizer.path,
            "tokenizer_vocab_size": tokenizer.get_vocab_size(),
            "max_seq_len": int(max_len),
            "answer_tokens": float(answer_mask.sum().item()),
        },
    )


def run_lexical_gap_overfit_smoke(
    *,
    jsonl_path: Path | str,
    tokenizer_path: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    steps: int = 8,
    lr: float = 1e-3,
    seed: int = 20260628,
) -> dict[str, Any]:
    records = load_lexical_gap_records(jsonl_path)
    if tokenizer_path is None:
        batch, vocab = build_lexical_gap_batch(records)
        batch_meta: dict[str, Any] = {"tokenized": False, "vocab_size": len(vocab)}
        model_vocab_size = max(128, len(vocab))
    else:
        tokenizer = load_tokenizer(tokenizer_path)
        batch, batch_meta = build_tokenized_lexical_gap_batch(records, tokenizer)
        model_vocab_size = int(batch_meta["tokenizer_vocab_size"])
    trainer = load_trainer(enable_objective=True, suffix="lexical_gap_overfit")
    torch.manual_seed(seed)
    model = make_tiny_model(
        trainer,
        vocab_size=model_vocab_size,
        ffn_dim=64,
        seq_len=batch.input_ids.shape[1],
    )
    counts = prepare_model_for_frozen_dsqg_w_objective(model)
    model.train()
    optimizer = make_dsqg_w_optimizer(model, lr=lr)
    initial = _param_snapshot(model)

    losses: list[float] = []
    step_reports: list[dict[str, Any]] = []
    for _ in range(int(steps)):
        report = run_one_frozen_dsqg_w_step(model, batch, optimizer)
        step_reports.append(report)
        losses.append(float(report["loss_before_step"]))
    with torch.no_grad():
        final_result = compute_frozen_dsqg_w_objective(model, batch)
    loss_final = float(final_result.loss.detach().float().cpu().item())
    loss_initial = float(losses[0])
    changed_frozen_final = _changed_names(initial, model, prefix=None)
    changed_dsqg_w_final = _changed_names(initial, model, prefix="dsqg_w.")
    min_changed_dsqg_w = min(int(report["changed_dsqg_w_param_count"]) for report in step_reports)
    max_changed_frozen = max(int(report["changed_frozen_param_count"]) for report in step_reports + [{"changed_frozen_param_count": len(changed_frozen_final)}])
    telemetry = dict(final_result.telemetry)
    telemetry.update({key: float(value) for key, value in counts.items()})
    for key, value in batch_meta.items():
        if isinstance(value, (int, float, bool)):
            telemetry[f"dsqg_w_overfit_{key}"] = float(value)
    telemetry.update(
        {
            "dsqg_w_overfit_lr": float(lr),
            "dsqg_w_overfit_steps": float(steps),
            "dsqg_w_overfit_loss_initial": loss_initial,
            "dsqg_w_overfit_loss_final": loss_final,
            "dsqg_w_overfit_loss_delta": loss_final - loss_initial,
            "dsqg_w_overfit_changed_dsqg_w_param_count": float(len(changed_dsqg_w_final)),
            "dsqg_w_overfit_changed_frozen_param_count": float(len(changed_frozen_final)),
        }
    )
    report = {
        "enabled": True,
        "skipped": False,
        "objective": "frozen_trunk_answer_only_ce_overfit_smoke",
        "dataset": str(jsonl_path),
        "dataset_examples": len(records),
        "tokenized": bool(batch_meta.get("tokenized", False)),
        "tokenizer_path": batch_meta.get("tokenizer_path"),
        "tokenizer_vocab_size": batch_meta.get("tokenizer_vocab_size"),
        "vocab_size": int(batch_meta.get("tokenizer_vocab_size", batch_meta.get("vocab_size", model_vocab_size))),
        "steps": int(steps),
        "losses_before_step": losses,
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "loss_delta": loss_final - loss_initial,
        "answer_tokens": float(batch.answer_mask.sum().item()),
        "min_changed_dsqg_w_param_count": min_changed_dsqg_w,
        "changed_dsqg_w_param_count_final": len(changed_dsqg_w_final),
        "max_changed_frozen_param_count": max_changed_frozen,
        "changed_frozen_param_names_final": changed_frozen_final,
        "telemetry": telemetry,
        "pass": bool(loss_final < loss_initial and min_changed_dsqg_w > 0 and max_changed_frozen == 0),
    }
    if checkpoint_dir is not None:
        report["checkpoint"] = save_dsqg_w_checkpoint(
            model,
            checkpoint_dir,
            metadata={
                "seed": seed,
                "jsonl_path": str(jsonl_path),
                "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
                "tokenizer_vocab_size": batch_meta.get("tokenizer_vocab_size"),
                "vocab_size": report["vocab_size"],
                "seq_len": int(batch.input_ids.shape[1]),
                "steps": int(steps),
                "lr": float(lr),
                "losses_before_step": losses,
                "loss_final": loss_final,
                "candidate_settings": _candidate_settings(),
            },
        )
    return report


def run_smoke_objective(*, enable: bool | None = None, seed: int = 20260628, step: bool = False) -> dict[str, Any]:
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
    if step:
        optimizer = make_dsqg_w_optimizer(model, lr=float(os.environ.get("DWARF_DSQG_W_FROZEN_LR", "0.001")))
        report = run_one_frozen_dsqg_w_step(model, batch, optimizer)
        report.update(
            {
                "enabled": True,
                "skipped": False,
                "objective": "frozen_trunk_answer_only_ce_step",
            }
        )
        report["telemetry"].update({key: float(value) for key, value in counts.items()})
        return report

    result = compute_frozen_dsqg_w_objective(model, batch)
    result.loss.backward()

    grad_names = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and param.grad.detach().abs().sum().item() > 0.0
    ]
    grad_scope_ok = bool(grad_names) and all(is_dsqg_w_param_name(name) for name in grad_names)
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
    parser.add_argument("--step", action="store_true", help="Run one DSQG-W-only optimizer step after the objective smoke.")
    parser.add_argument("--overfit-jsonl", type=Path, default=None, help="Run a tiny multi-step JSONL overfit smoke when --enable is also set.")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer JSON for real-token-ID lexical-gap overfit smoke.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional output directory for DSQG-W-only checkpoint artifacts.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    if args.overfit_jsonl is not None:
        if not args.enable:
            report = {
                "enabled": False,
                "skipped": True,
                "reason": "pass --enable to run the DSQG-W lexical-gap overfit smoke",
                "pass": True,
            }
        else:
            report = run_lexical_gap_overfit_smoke(
                jsonl_path=args.overfit_jsonl,
                tokenizer_path=args.tokenizer,
                checkpoint_dir=args.save_dir,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
            )
    else:
        report = run_smoke_objective(enable=args.enable, seed=args.seed, step=args.step)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
