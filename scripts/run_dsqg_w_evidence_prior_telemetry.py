#!/usr/bin/env python3
"""Checkpoint-local DSQG-W evidence/prior/quotas telemetry runner.

This is intentionally post-hoc: it loads existing run_config/checkpoint pairs,
runs a tiny validation slice, records the new candidate evidence telemetry, then
checks the zero-init evidence prior composer for behavior preservation across the
actual sourcewise/Triton path plus eager sourcewise and materialized dense paths.
Conditional HISA quotas are evaluated only if occupancy telemetry indicates a
candidate-retention problem.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
DEFAULT_RUN_ROOT = ROOT / "runs" / "dsqg_w_reset_quality_20260704_122732_100k_aux0"
DEFAULT_VARIANT = "w_typed_aux0"

# Keep imports after constants so CUDA_VISIBLE_DEVICES can be set by the caller
# before the process starts. Do not mutate visible devices inside this script.
import torch  # noqa: E402


def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DSQG-W evidence/prior telemetry",
        "",
        f"- run_root: `{payload['run_root']}`",
        f"- variant: `{payload['variant_id']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- device: `{payload['device']}`",
        f"- batches: {payload['max_batches']} × batch_size {payload['batch_size']}",
        "",
        "## Evidence baseline",
        "",
    ]
    ev = payload["evidence_baseline"]
    for key in payload["interesting_keys"]:
        if key in ev["telemetry_mean"]:
            lines.append(f"- `{key}`: {ev['telemetry_mean'][key]:.6g}")
    lines.extend(["", "## Prior composer behavior", ""])
    lines.append("| path | max_abs_hidden_delta | CE_delta | prior_norm | prior_abs_mean | enabled |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in payload["prior_checks"]:
        pm = item["prior"]["telemetry_mean"]
        lines.append(
            "| {path} | {delta:.6g} | {ce_delta:.6g} | {prior_norm:.6g} | {prior_abs:.6g} | {enabled:.0f} |".format(
                path=item["path"],
                delta=item["max_abs_hidden_delta"],
                ce_delta=item["ce_delta"],
                prior_norm=pm.get("dsqg_w_prior_norm", float("nan")),
                prior_abs=pm.get("dsqg_w_prior_abs_mean", float("nan")),
                enabled=pm.get("dsqg_w_evidence_prior_enabled", 0.0),
            )
        )
    lines.extend(["", "## Conditional quota decision", ""])
    q = payload["quota_decision"]
    lines.append(f"- decision: **{q['decision']}**")
    lines.append(f"- reason: {q['reason']}")
    if payload.get("quota_check"):
        qc = payload["quota_check"]
        qm = qc["telemetry_mean"]
        lines.append(f"- quota_hisa_max: {qc['quota_hisa_max']}")
        lines.append(f"- clipped_fraction: {qm.get('dsqg_w_candidate_quota_hisa_clipped_fraction', float('nan')):.6g}")
        lines.append(f"- valid_candidate_count: {qm.get('dsqg_w_valid_candidate_count', float('nan')):.6g}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_run_config(run_root: Path, variant_id: str) -> dict[str, Any]:
    path = run_root / "pretrain" / variant_id / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_from_config(cfg: dict[str, Any]) -> Path:
    env = cfg["env"]
    ckpt_dir = Path(env["DWARF_CHECKPOINT_DIR"])
    base = env["DWARF_CKPT_BASE_NAME"]
    for name in (f"{base}_best.pt", f"{base}_ep1.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no checkpoint under {ckpt_dir}")


def env_for_pass(cfg: dict[str, Any], *, path_mode: str, prior: bool, quotas: bool, quota_hisa_max: int = 2) -> dict[str, str]:
    env = {str(k): str(v) for k, v in cfg.get("env", {}).items()}
    env.update(
        {
            "PYTHONPATH": ".",
            "DWARF_TORCH_COMPILE": "0",
            "DWARF_LIGER": "0",
            "DWARF_Q6_G128": "0",
            "DWARF_PIN_DATASET": "0",
            "DWARF_DSQG_W_FAST_TELEMETRY": "1",
            "DWARF_DSQG_W_EVIDENCE_PRIOR": "1" if prior else "0",
            "DWARF_DSQG_W_EVIDENCE_PRIOR_CLIP": "2.0",
            "DWARF_DSQG_W_EVIDENCE_PRIOR_INIT_SCALE": "0.0",
            "DWARF_DSQG_W_CANDIDATE_QUOTAS": "1" if quotas else "0",
            "DWARF_DSQG_W_QUOTA_HISA_MAX": str(int(quota_hisa_max)),
        }
    )
    if path_mode == "triton":
        env["DWARF_DSQG_W_SOURCEWISE"] = "1"
        env["DWARF_DSQG_W_TRITON_SOURCEWISE"] = "1"
    elif path_mode == "sourcewise":
        env["DWARF_DSQG_W_SOURCEWISE"] = "1"
        env["DWARF_DSQG_W_TRITON_SOURCEWISE"] = "0"
    elif path_mode == "dense":
        env["DWARF_DSQG_W_SOURCEWISE"] = "0"
        env["DWARF_DSQG_W_TRITON_SOURCEWISE"] = "0"
    else:
        raise ValueError(f"unknown path_mode={path_mode}")
    return env


@contextlib.contextmanager
def patched_environ(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def load_train_module(env: dict[str, str], label: str):
    for p in (ROOT, ROOT / "kernels", ROOT / "train"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    name = f"dsqg_w_telemetry_train_{label}_{time.time_ns()}"
    with patched_environ(env):
        spec = importlib.util.spec_from_file_location(name, str(TRAIN_SCRIPT))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {TRAIN_SCRIPT}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def load_model(mod, checkpoint: Path, device: str):
    model = mod.TriadicJ96Dsr(
        vocab_size=getattr(mod, "VOCAB_SIZE"),
        embedding_dim=getattr(mod, "EMBEDDING_DIM"),
        num_heads=getattr(mod, "NUM_HEADS"),
        ffn_dim=getattr(mod, "FFN_DIM"),
        seq_len=getattr(mod, "MAX_SEQ_LEN"),
        dsr_layer=getattr(mod, "DSR_LAYER"),
        scale_embed_init_val=getattr(mod, "SCALE_EMBED_INIT_VAL", 0.15),
        dropout=0.0,
        num_chunks=getattr(mod, "NUM_CHUNKS", 32),
        top_k_chunks=getattr(mod, "TOP_K_CHUNKS", 4),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any("_orig_mod." in k for k in state):
        state = {k.replace("._orig_mod", "").replace("_orig_mod.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.endswith("causal_mask")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {
        "missing_learnable_keys": [k for k in missing if not k.endswith("causal_mask")],
        "unexpected_keys": list(unexpected),
        "param_count": sum(p.numel() for p in model.parameters()),
    }


def load_val_batches(mod, env: dict[str, str], *, batch_size: int, max_batches: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    dataset = Path(env.get("DWARF_DATASET", str(ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt")))
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    cache = torch.load(dataset, weights_only=True, map_location="cpu")
    val = cache["val"].to(dtype=torch.int32).contiguous()
    _, val_mask, _ = mod._prepare_dataset_loss_masks(cache, cache["train"].to(dtype=torch.int32), val, use_liger_ce=False)
    n = min(int(batch_size) * int(max_batches), int(val.shape[0]))
    val = val[:n]
    val_mask = val_mask[:n]
    batches = [val[i : i + batch_size] for i in range(0, n, batch_size)]
    masks = [val_mask[i : i + batch_size] for i in range(0, n, batch_size)]
    return batches, masks


def tensor_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        val = float(value.detach().float().cpu().item())
    elif isinstance(value, (float, int)):
        val = float(value)
    else:
        return None
    if math.isfinite(val):
        return val
    return None


def summarize_telemetry(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for k in keys:
        vals = [row[k] for row in rows if k in row and math.isfinite(row[k])]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


@torch.inference_mode()
def run_single_pass(
    cfg: dict[str, Any],
    checkpoint: Path,
    *,
    path_mode: str,
    prior: bool,
    quotas: bool,
    quota_hisa_max: int,
    batch_size: int,
    max_batches: int,
    device: str,
) -> dict[str, Any]:
    env = env_for_pass(cfg, path_mode=path_mode, prior=prior, quotas=quotas, quota_hisa_max=quota_hisa_max)
    mod = load_train_module(env, f"{path_mode}_{'prior' if prior else 'base'}_{'quota' if quotas else 'noquota'}")
    batches, masks = load_val_batches(mod, env, batch_size=batch_size, max_batches=max_batches)
    with patched_environ(env):
        model, load_info = load_model(mod, checkpoint, device)
        rows: list[dict[str, float]] = []
        hidden_cpu: list[torch.Tensor] = []
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for batch, mask in zip(batches, masks):
            x = batch[:, :-1].to(device, non_blocking=True).long()
            y = batch[:, 1:].to(device, non_blocking=True).long()
            target_mask = mask[:, 1:].to(device, non_blocking=True)
            q_idx, hisa_idx, l3_skip_idx = mod._dsqg_w_training_candidate_indices(x)
            with mod._amp_context(device):
                hidden = model.forward_hidden(
                    x,
                    dsqg_w_question_indices=q_idx,
                    dsqg_w_hisa_evidence_indices=hisa_idx,
                    dsqg_w_l3_skip_indices=l3_skip_idx,
                )
                loss_sum, n_rows = mod._streamed_linear_ce_loss(
                    hidden,
                    y,
                    model.out.weight,
                    chunk_rows=getattr(mod, "CE_CHUNK", 2048),
                    grad_denom=None,
                    loss_mask=target_mask,
                )
            telemetry = getattr(model, "dsqg_w_last_telemetry", {}) or {}
            row = {k: v for k, v in ((k, tensor_scalar(v)) for k, v in telemetry.items()) if v is not None}
            rows.append(row)
            hidden_cpu.append(hidden.detach().float().cpu())
            total_loss += float(loss_sum.item())
            total_tokens += int(n_rows)
            del x, y, target_mask, hidden, loss_sum
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.time() - t0
        result = {
            "path_mode": path_mode,
            "prior": bool(prior),
            "quotas": bool(quotas),
            "quota_hisa_max": int(quota_hisa_max),
            "loss": total_loss / max(total_tokens, 1),
            "tokens": int(total_tokens),
            "elapsed_s": elapsed_s,
            "telemetry_mean": summarize_telemetry(rows),
            "telemetry_rows": rows,
            "load_info": load_info,
            "hidden_cpu": hidden_cpu,
        }
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def strip_hidden(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out.pop("hidden_cpu", None)
    return out


def compare_hidden(a: dict[str, Any], b: dict[str, Any]) -> float:
    max_delta = 0.0
    for ha, hb in zip(a["hidden_cpu"], b["hidden_cpu"]):
        max_delta = max(max_delta, float((ha - hb).abs().max().item()))
    return max_delta


def quota_decision(telemetry: dict[str, float], *, monopoly_threshold: float, missing_question_threshold: float) -> dict[str, Any]:
    monopoly = telemetry.get("dsqg_w_candidate_hisa_monopoly_row_fraction", 0.0)
    missing_q = telemetry.get("dsqg_w_candidate_missing_question_row_fraction", 0.0)
    valid_count = telemetry.get("dsqg_w_valid_candidate_count", 0.0)
    if monopoly >= monopoly_threshold:
        return {
            "decision": "run_quotas",
            "reason": f"HISA monopoly rows {monopoly:.3f} >= threshold {monopoly_threshold:.3f}",
            "monopoly": monopoly,
            "missing_question": missing_q,
            "valid_candidate_count": valid_count,
        }
    if missing_q >= missing_question_threshold:
        return {
            "decision": "run_quotas",
            "reason": f"missing-question rows {missing_q:.3f} >= threshold {missing_question_threshold:.3f}",
            "monopoly": monopoly,
            "missing_question": missing_q,
            "valid_candidate_count": valid_count,
        }
    return {
        "decision": "skip_quotas",
        "reason": (
            f"HISA monopoly rows {monopoly:.3f} < {monopoly_threshold:.3f} and "
            f"missing-question rows {missing_q:.3f} < {missing_question_threshold:.3f}"
        ),
        "monopoly": monopoly,
        "missing_question": missing_q,
        "valid_candidate_count": valid_count,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--prior-check-batches", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paths", nargs="+", default=["triton", "sourcewise", "dense"], choices=["triton", "sourcewise", "dense"])
    ap.add_argument("--quota-hisa-max", type=int, default=2)
    ap.add_argument("--monopoly-threshold", type=float, default=0.50)
    ap.add_argument("--missing-question-threshold", type=float, default=0.05)
    args = ap.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = args.device if args.device != "cuda" else "cuda"
    run_root = args.run_root.resolve()
    cfg = read_run_config(run_root, args.variant)
    checkpoint = checkpoint_from_config(cfg)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (args.output_root or (run_root / "evidence_prior_telemetry" / f"{args.variant}_{stamp}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"run_root={run_root}", flush=True)
    print(f"variant={args.variant}", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == 'cuda' else device}", flush=True)
    print(f"out_root={out_root}", flush=True)

    evidence = run_single_pass(
        cfg,
        checkpoint,
        path_mode="triton" if "triton" in args.paths else args.paths[0],
        prior=False,
        quotas=False,
        quota_hisa_max=args.quota_hisa_max,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
    )
    print("evidence baseline complete", flush=True)

    prior_checks = []
    for path_mode in args.paths:
        base = run_single_pass(
            cfg,
            checkpoint,
            path_mode=path_mode,
            prior=False,
            quotas=False,
            quota_hisa_max=args.quota_hisa_max,
            batch_size=args.batch_size,
            max_batches=args.prior_check_batches,
            device=device,
        )
        prior = run_single_pass(
            cfg,
            checkpoint,
            path_mode=path_mode,
            prior=True,
            quotas=False,
            quota_hisa_max=args.quota_hisa_max,
            batch_size=args.batch_size,
            max_batches=args.prior_check_batches,
            device=device,
        )
        item = {
            "path": path_mode,
            "base": strip_hidden(base),
            "prior": strip_hidden(prior),
            "max_abs_hidden_delta": compare_hidden(base, prior),
            "ce_delta": float(prior["loss"] - base["loss"]),
        }
        prior_checks.append(item)
        print(
            f"prior check {path_mode}: max_abs_hidden_delta={item['max_abs_hidden_delta']:.6g} ce_delta={item['ce_delta']:.6g}",
            flush=True,
        )
        del base, prior
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qd = quota_decision(
        evidence["telemetry_mean"],
        monopoly_threshold=args.monopoly_threshold,
        missing_question_threshold=args.missing_question_threshold,
    )
    quota_check = None
    if qd["decision"] == "run_quotas":
        quota = run_single_pass(
            cfg,
            checkpoint,
            path_mode="triton" if "triton" in args.paths else args.paths[0],
            prior=True,
            quotas=True,
            quota_hisa_max=args.quota_hisa_max,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=device,
        )
        quota_check = strip_hidden(quota)
        print("quota check complete", flush=True)
    else:
        print(f"quota check skipped: {qd['reason']}", flush=True)

    interesting_keys = [
        "dsqg_w_valid_candidate_count",
        "dsqg_w_candidate_multi_evidence_fraction",
        "dsqg_w_candidate_evidence_count_mean",
        "dsqg_w_candidate_hisa_monopoly_row_fraction",
        "dsqg_w_candidate_missing_question_row_fraction",
        "dsqg_w_candidate_quota_hisa_clipped_fraction",
        "dsqg_w_candidate_pre_fraction_question",
        "dsqg_w_candidate_fraction_question",
        "dsqg_w_candidate_pre_fraction_hisa_evidence",
        "dsqg_w_candidate_fraction_hisa_evidence",
        "dsqg_w_candidate_pre_source_fraction_hisa",
        "dsqg_w_candidate_post_source_fraction_hisa",
        "dsqg_w_candidate_pre_source_fraction_final",
        "dsqg_w_candidate_post_source_fraction_final",
        "dsqg_w_candidate_pre_source_fraction_l3",
        "dsqg_w_candidate_post_source_fraction_l3",
    ]
    payload = {
        "timestamp": stamp,
        "run_root": str(run_root),
        "variant_id": args.variant,
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == "cuda" else device,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "prior_check_batches": args.prior_check_batches,
        "paths": args.paths,
        "interesting_keys": interesting_keys,
        "evidence_baseline": strip_hidden(evidence),
        "prior_checks": prior_checks,
        "quota_decision": qd,
        "quota_check": quota_check,
    }
    json_path = out_root / "evidence_prior_telemetry.json"
    md_path = out_root / "summary.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
