#!/usr/bin/env python3
"""Evaluate a DSQG-W checkpoint on a requested validation split without training."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
DEFAULT_BASE_DATASET = ROOT / "datasets" / "dwarf_base_v1_olmo1tok_2048_2b.pt"


def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def env_from_run_config(run_config: Path, *, dataset: Path, batch_size: int, max_val_seqs: int) -> dict[str, str]:
    cfg = load_json(run_config)
    env = {str(k): str(v) for k, v in cfg.get("env", {}).items()}
    env.update(
        {
            "DWARF_DATASET": str(dataset),
            "DWARF_BS": str(int(batch_size)),
            "DWARF_MAX_VAL_SEQS": str(int(max_val_seqs)),
            "DWARF_MAX_TRAIN_SEQS": "1",
            "DWARF_TORCH_COMPILE": "0",
            "DWARF_LIGER": "0",
            "DWARF_Q6_G128": "0",
            "DWARF_PIN_DATASET": "0",
        }
    )
    return env


def import_trainer(env: dict[str, str]):
    for key, value in env.items():
        if key == "CUDA_VISIBLE_DEVICES":
            continue
        os.environ[key] = value
    spec = importlib.util.spec_from_file_location("dwarf_base_val_eval_trainer", str(TRAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer from {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model(module, checkpoint: Path, device: torch.device):
    model = module.TriadicJ96Dsr(
        vocab_size=module.VOCAB_SIZE,
        embedding_dim=module.EMBEDDING_DIM,
        num_heads=module.NUM_HEADS,
        ffn_dim=module.FFN_DIM,
        seq_len=module.MAX_SEQ_LEN,
        dsr_layer=module.DSR_LAYER,
        scale_embed_init_val=module.SCALE_EMBED_INIT_VAL,
        dropout=module.DROPOUT,
        num_chunks=module.NUM_CHUNKS,
        top_k_chunks=module.TOP_K_CHUNKS,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    missing_learnable = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing_learnable:
        raise RuntimeError(f"checkpoint missing learnable keys: {missing_learnable[:20]}")
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}, {"missing": missing, "unexpected": unexpected}


def dataset_stats(mask: torch.Tensor) -> dict[str, Any]:
    target = mask[:, 1:]
    real = int(target.sum().item())
    slots = int(target.numel())
    return {"real_tokens": real, "target_slots": slots, "real_fraction": real / max(slots, 1)}


def scalar_telemetry(telemetry: dict[str, Any]) -> dict[str, float]:
    keep = (
        "dsqg_w_gate_mean",
        "dsqg_w_delta_to_x_ratio",
        "dsqg_w_hisa_source_mass",
        "dsqg_w_candidate_score_bias_norm",
        "dsqg_w_candidate_score_mean",
        "dsqg_w_typed_mixer_gate_mean",
        "dsqg_w_valid_candidate_count",
        "read_mix_weight_norm",
    )
    out: dict[str, float] = {}
    for key in keep:
        value = telemetry.get(key)
        if torch.is_tensor(value) and value.numel() == 1:
            out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def evaluate_split(module, model, val_data: torch.Tensor, val_loss_mask: torch.Tensor, device: torch.device) -> tuple[float, dict[str, Any]]:
    start = time.time()
    loss = float(module.evaluate(model, val_data, str(device), loss_mask=val_loss_mask))
    elapsed = time.time() - start
    ppl = math.exp(min(loss, 20.0))
    telemetry = scalar_telemetry(getattr(model, "dsqg_w_last_telemetry", {}) or {})
    routing_entropy = getattr(model.blocks[module.DSR_LAYER].attn, "_routing_entropy", None)
    if torch.is_tensor(routing_entropy):
        routing_entropy = float(routing_entropy.detach().cpu().item())
    elif not isinstance(routing_entropy, (int, float)):
        routing_entropy = None
    return loss, {"ppl": ppl, "elapsed_s": elapsed, "telemetry_last_batch": telemetry, "routing_entropy_last_batch": routing_entropy}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_BASE_DATASET)
    ap.add_argument("--max-val-seqs", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    env = env_from_run_config(args.run_config, dataset=args.dataset, batch_size=args.batch_size, max_val_seqs=args.max_val_seqs)
    module = import_trainer(env)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device required for DSQG/HISA kernels")

    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    val_data = dataset["val"].to(dtype=torch.int32).contiguous()
    if val_data.size(1) != module.MAX_SEQ_LEN:
        raise ValueError(f"dataset val seq_len={val_data.size(1)} != trainer seq_len={module.MAX_SEQ_LEN}")
    val_data = val_data[: args.max_val_seqs]
    dummy_train = dataset["train"][:1].to(dtype=torch.int32).contiguous()
    _, val_mask_all, mask_stats = module._prepare_dataset_loss_masks(dataset, dummy_train, dataset["val"].to(dtype=torch.int32).contiguous(), use_liger_ce=False)
    val_loss_mask = val_mask_all[: args.max_val_seqs].contiguous()

    model, ckpt, load_info = build_model(module, args.checkpoint, device)
    torch.cuda.reset_peak_memory_stats()
    val_loss, eval_info = evaluate_split(module, model, val_data, val_loss_mask, device)
    peak = int(torch.cuda.max_memory_allocated())
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_config": str(args.run_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_label": "original_base_v1",
        "device": torch.cuda.get_device_name(0),
        "max_val_seqs": int(args.max_val_seqs),
        "batch_size_env": int(args.batch_size),
        "eval_batch_size_effective": max(1, int(args.batch_size) // 2),
        "val_shape": list(val_data.shape),
        "val_mask_stats": dataset_stats(val_loss_mask),
        "trainer_mask_source": mask_stats.get("source"),
        "checkpoint_config": ckpt.get("config", {}) if isinstance(ckpt, dict) else {},
        "load_info": load_info,
        "val_loss": val_loss,
        "val_ppl": eval_info["ppl"],
        "elapsed_s": eval_info["elapsed_s"],
        "peak_memory_allocated_bytes": peak,
        "telemetry_last_batch": eval_info["telemetry_last_batch"],
        "routing_entropy_last_batch": eval_info["routing_entropy_last_batch"],
        "verification": {
            "pass": math.isfinite(val_loss) and eval_info["ppl"] > 0.0 and val_data.size(0) == int(args.max_val_seqs),
            "errors": [],
        },
    }
    if payload["load_info"]["missing"]:
        payload["verification"]["errors"].append("missing checkpoint keys")
    if not payload["verification"]["pass"]:
        payload["verification"]["errors"].append("non-finite or incomplete eval")
    write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "device": payload["device"],
        "max_val_seqs": payload["max_val_seqs"],
        "val_loss": payload["val_loss"],
        "val_ppl": payload["val_ppl"],
        "elapsed_s": payload["elapsed_s"],
        "verification": payload["verification"],
    }, indent=2), flush=True)
    return 0 if payload["verification"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
