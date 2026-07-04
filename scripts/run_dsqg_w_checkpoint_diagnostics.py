#!/usr/bin/env python3
"""Posthoc DSQG-W checkpoint diagnostics.

Loads a DSQG-W run directory/checkpoint under the current code and computes fixed
train/validation-batch forward telemetry plus CE-only, aux-only, weighted-aux,
and combined gradient norms.  This is intended to answer whether the width-cell
content path (v_proj/lateral_up/gate) receives CE signal separately from the
short direct routing-aux path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


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


def checkpoint_from_run_config(run_config: Path) -> Path:
    cfg = load_json(run_config)
    env = cfg.get("env", {})
    ckpt_dir = Path(env["DWARF_CHECKPOINT_DIR"])
    base = env["DWARF_CKPT_BASE_NAME"]
    for candidate in (ckpt_dir / f"{base}_best.pt", ckpt_dir / f"{base}_ep1.pt"):
        if candidate.exists():
            return candidate
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")


def trainer_env_from_run_config(run_config: Path) -> dict[str, str]:
    cfg = load_json(run_config)
    env = {str(k): str(v) for k, v in cfg.get("env", {}).items()}
    # Diagnostics should not compile or use Liger; they must match model topology,
    # not train throughput. Preserve architecture/config envs from the run.
    env["DWARF_TORCH_COMPILE"] = "0"
    env["DWARF_LIGER"] = "0"
    env["DWARF_Q6_G128"] = "0"
    env["DWARF_PIN_DATASET"] = "0"
    return env


def load_trainer_module(run_config: Path):
    env = trainer_env_from_run_config(run_config)
    for key, value in env.items():
        if key == "CUDA_VISIBLE_DEVICES":
            # Device selection belongs to the caller process environment.  Do not
            # remap devices after torch import.
            continue
        os.environ[key] = value
    spec = importlib.util.spec_from_file_location("dwarf_v2_diag_trainer", str(TRAIN_SCRIPT))
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    return model, ckpt if isinstance(ckpt, dict) else {"raw_checkpoint_type": type(ckpt).__name__}


def load_dataset(env: dict[str, str]) -> dict[str, torch.Tensor]:
    dataset_path = Path(env["DWARF_DATASET"])
    cache = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "train" not in cache or "val" not in cache:
        raise ValueError(f"unsupported dataset cache shape at {dataset_path}")
    return cache


def select_batch(cache: dict[str, Any], split: str, *, batch_size: int, seq_len: int, seed: int, offset: int) -> torch.Tensor:
    data = cache[split]
    if not torch.is_tensor(data):
        raise ValueError(f"cache[{split!r}] is not a tensor")
    if data.size(1) < seq_len:
        raise ValueError(f"{split} seq_len {data.size(1)} < requested {seq_len}")
    rng = random.Random(seed + (0 if split == "train" else 10_000) + offset)
    max_start = max(int(data.size(0)) - batch_size, 0)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    batch = data[start:start + batch_size, :seq_len].long().contiguous()
    return batch


def scalarize_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    keep_prefixes = (
        "dsqg_w_gate", "dsqg_w_delta", "dsqg_w_x_norm", "dsqg_w_read_norm",
        "dsqg_w_width", "dsqg_w_typed_mixer", "dsqg_w_hisa", "dsqg_w_candidate",
        "read_mix_weight_norm",
    )
    out: dict[str, Any] = {}
    for key, value in telemetry.items():
        if not key.startswith(keep_prefixes):
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
            elif key == "dsqg_w_typed_read_norms":
                out[key] = value.detach().cpu().float().tolist()
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def grad_norm_for_named_params(named_params: list[tuple[str, torch.nn.Parameter]], predicate) -> float | None:
    total_sq = 0.0
    seen = 0
    for name, param in named_params:
        if not predicate(name) or param.grad is None:
            continue
        grad = param.grad.detach().float()
        if grad.numel() == 0:
            continue
        norm = float(grad.norm().item())
        if math.isfinite(norm):
            total_sq += norm * norm
            seen += 1
    if seen == 0:
        return None
    return math.sqrt(total_sq)


def grad_groups(model: torch.nn.Module) -> dict[str, float | None]:
    named = list(model.named_parameters())
    score_terms = (
        ".width_cell.q_proj", ".width_cell.k_proj", ".width_cell.rel_diff_proj",
        ".width_cell.rel_prod_proj", ".width_cell.rel_diff_score", ".width_cell.rel_prod_score",
        ".width_cell.type_pair_bias", ".width_cell.source_pair_bias", ".width_cell.self_bias",
    )
    groups = {
        "w_width_score_gn": lambda n: any(term in n for term in score_terms),
        "w_width_v_gn": lambda n: ".width_cell.v_proj" in n,
        "w_width_up_gn": lambda n: ".width_cell.lateral_up" in n,
        "w_width_gate_gn": lambda n: ".width_cell.gate" in n,
        "w_mix_gate_gn": lambda n: ".typed_mixer.gate" in n,
        "w_main_gate_gn": lambda n: n.endswith(".gate") and "dsqg_w_blocks." in n and ".width_cell." not in n and ".typed_mixer." not in n,
        "w_all_gate_gn": lambda n: n.endswith(".gate") and ("dsqg_w" in n or ".width_cell." in n or ".typed_mixer." in n),
    }
    return {key: grad_norm_for_named_params(named, pred) for key, pred in groups.items()}


def gate_logits(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    vals: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if name.endswith(".gate") and "dsqg_w" in name:
            vals.append(param.detach().float().reshape(-1))
            if ".width_cell.gate" in name:
                out["w_width_gate_logit_mean"] = float(param.detach().float().mean().item())
            elif ".typed_mixer.gate" in name:
                out["w_mix_gate_logit_mean"] = float(param.detach().float().mean().item())
            elif ".dsqg_w_blocks." in name:
                out["w_gate_logit_mean"] = float(param.detach().float().mean().item())
    if vals:
        cat = torch.cat(vals)
        out["w_all_gate_logit_mean"] = float(cat.mean().item())
        out["w_all_gate_logit_min"] = float(cat.min().item())
        out["w_all_gate_logit_max"] = float(cat.max().item())
    return out


def make_inputs(module, batch: torch.Tensor, device: torch.device):
    x = batch[:, :-1].to(device, non_blocking=True).long()
    y = batch[:, 1:].to(device, non_blocking=True).long()
    target_mask = torch.ones_like(y, dtype=torch.bool, device=device)
    q_idx, hisa_idx, l3_skip_idx = module._dsqg_w_training_candidate_indices(x)
    return x, y, target_mask, q_idx, hisa_idx, l3_skip_idx


def forward_hidden(module, model, batch: torch.Tensor, device: torch.device):
    x, y, target_mask, q_idx, hisa_idx, l3_skip_idx = make_inputs(module, batch, device)
    hidden = model.forward_hidden(
        x,
        dsqg_w_question_indices=q_idx,
        dsqg_w_hisa_evidence_indices=hisa_idx,
        dsqg_w_l3_skip_indices=l3_skip_idx,
    )
    telemetry = scalarize_telemetry(getattr(model, "dsqg_w_last_telemetry", {}) or {})
    return hidden, y, target_mask, telemetry


def zero_grad(model) -> None:
    model.zero_grad(set_to_none=True)


def run_loss_mode(module, model, batch: torch.Tensor, device: torch.device, mode: str, aux_weight: float) -> dict[str, Any]:
    zero_grad(model)
    hidden, y, target_mask, telemetry = forward_hidden(module, model, batch, device)
    n_rows = int(target_mask.sum().item())
    losses: dict[str, float] = {}
    if mode == "combined":
        # Aux branches off inside the recomposer before final hidden projection.
        # Backprop it first and retain the graph so the streamed CE backward can
        # still traverse the shared forward graph.  Doing streamed CE first frees
        # the graph and makes the subsequent aux backward invalid.
        aux = module._dsqg_w_width_aux_loss(model, model)
        aux_value = module._dsqg_w_width_aux_value(model, model)
        if aux is None:
            losses["aux_loss"] = None
        else:
            losses["aux_loss"] = float(aux.detach().item() if aux_value is None else aux_value)
            (aux * float(aux_weight)).backward(retain_graph=True)
        total_ce, _ = module._streamed_linear_ce_loss(
            hidden,
            y,
            model.out.weight,
            chunk_rows=int(getattr(module, "CE_CHUNK", 4096)),
            grad_denom=float(max(n_rows, 1)),
            loss_mask=target_mask,
        )
        losses["ce_loss_sum"] = float(total_ce.detach().item())
        losses["ce_loss_mean"] = float(total_ce.detach().item() / float(max(n_rows, 1)))
    else:
        if mode == "ce":
            total_ce, _ = module._streamed_linear_ce_loss(
                hidden,
                y,
                model.out.weight,
                chunk_rows=int(getattr(module, "CE_CHUNK", 4096)),
                grad_denom=float(max(n_rows, 1)),
                loss_mask=target_mask,
            )
            losses["ce_loss_sum"] = float(total_ce.detach().item())
            losses["ce_loss_mean"] = float(total_ce.detach().item() / float(max(n_rows, 1)))
        if mode in {"aux", "weighted_aux"}:
            aux = module._dsqg_w_width_aux_loss(model, model)
            aux_value = module._dsqg_w_width_aux_value(model, model)
            if aux is None:
                losses["aux_loss"] = None
            else:
                losses["aux_loss"] = float(aux.detach().item() if aux_value is None else aux_value)
                scale = aux_weight if mode == "weighted_aux" else 1.0
                (aux * float(scale)).backward()
    groups = grad_groups(model)
    zero_grad(model)
    return {"mode": mode, "losses": losses, "grad_norms": groups, "forward_telemetry": telemetry}


def run_split(module, model, cache: dict[str, Any], split: str, *, batch_size: int, seq_len: int, seed: int, offset: int, aux_weight: float, device: torch.device, modes: list[str]) -> dict[str, Any]:
    batch = select_batch(cache, split, batch_size=batch_size, seq_len=seq_len, seed=seed, offset=offset)
    result = {
        "split": split,
        "batch_shape": list(batch.shape),
        "batch_seed": seed,
        "batch_offset": offset,
        "gate_logits": gate_logits(model),
        "modes": [],
    }
    for mode in modes:
        result["modes"].append(run_loss_mode(module, model, batch, device, mode, aux_weight))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Variant run directory containing run_config.json")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--modes", nargs="+", default=["ce", "aux", "weighted_aux", "combined"], choices=["ce", "aux", "weighted_aux", "combined"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    run_config = run_dir / "run_config.json"
    if not run_config.exists():
        raise FileNotFoundError(run_config)
    env = trainer_env_from_run_config(run_config)
    aux_weight = float(env.get("DWARF_DSQG_W_WIDTH_AUX_WEIGHT", "0"))
    checkpoint = args.checkpoint.resolve() if args.checkpoint else checkpoint_from_run_config(run_config).resolve()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    module = load_trainer_module(run_config)
    cache = load_dataset(env)
    model, ckpt = build_model(module, checkpoint, device)
    model.eval()

    payload: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_config": str(run_config),
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "aux_weight": aux_weight,
        "checkpoint_config": ckpt.get("config", {}) if isinstance(ckpt, dict) else {},
        "diagnostics": [],
    }
    for split in args.splits:
        payload["diagnostics"].append(
            run_split(
                module,
                model,
                cache,
                split,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=args.seed,
                offset=args.offset,
                aux_weight=aux_weight,
                device=device,
                modes=args.modes,
            )
        )

    out_path = args.output
    if out_path is None:
        out_path = run_dir / "diagnostics" / f"checkpoint_diagnostics_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(out_path, payload)
    print(f"checkpoint diagnostics -> {out_path}", flush=True)
    for split_result in payload["diagnostics"]:
        print(f"\n[{split_result['split']}] gate_logits={split_result['gate_logits']}", flush=True)
        for mode_result in split_result["modes"]:
            print(f"  {mode_result['mode']}: losses={mode_result['losses']} grad_norms={mode_result['grad_norms']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
