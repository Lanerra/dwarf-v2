#!/usr/bin/env python3
"""Run ARC-Easy/PIQA/LAMBADA external trio for arbitrary DSQG-W run dirs.

This bridges DWARF-v2 run_config.json files into the legacy DWARF external eval
harness by injecting ARCH_CONFIGS at runtime. Results are written under
<run-root>/external_trio/<variant>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/home/dlewis3/Desktop/AI/DWARF")
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
TOKENIZER = ROOT / "tokenizers" / "olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
PYTHON_EVAL_ROOTS = [LEGACY_ROOT, LEGACY_ROOT / "analysis_shortlist", LEGACY_ROOT / "evals", LEGACY_ROOT / "tools"]
TASKS = ["arc_easy", "piqa", "lambada"]


def _setup_legacy_imports():
    for p in reversed(PYTHON_EVAL_ROOTS):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    import analysis_shortlist.eval_external as eval_external  # type: ignore
    return eval_external


def _json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n", encoding="utf-8")


def _arch_config_from_run_config(run_config: Path) -> dict[str, Any]:
    cfg = json.loads(run_config.read_text(encoding="utf-8"))
    env = cfg.get("env", {})
    # Keep only architecture/model-construction toggles. Runtime geometry
    # (CUDA, batch, dataset, checkpoint dirs) is controlled by this eval runner.
    eval_env = {
        k: str(v)
        for k, v in env.items()
        if k.startswith("DWARF_DSQG_W")
        or k in {
            "DWARF_HISA_STAGE2_REP_R",
            "DWARF_PURE_DSQG",
            "DWARF_Q6_G128",
            "DWARF_FFN_DIM",
            "DWARF_TORCH_COMPILE",
            "DWARF_LIGER",
        }
    }
    eval_env["DWARF_TORCH_COMPILE"] = "0"
    eval_env["DWARF_LIGER"] = "0"
    eval_env["DWARF_Q6_G128"] = "0"
    return {
        "arch": "triadic_j96_dsr",
        "D": 512,
        "H": 8,
        "FFN": 1536,
        "L": 10,
        "full_layer": 3,
        "train_script": str(TRAIN_SCRIPT),
        "model_class": "TriadicJ96Dsr",
        "tokenizer": str(TOKENIZER),
        "eval_env": eval_env,
    }


def _checkpoint_from_run_config(run_config: Path) -> Path:
    cfg = json.loads(run_config.read_text(encoding="utf-8"))
    env = cfg.get("env", {})
    ckpt_dir = Path(env["DWARF_CHECKPOINT_DIR"])
    base = env["DWARF_CKPT_BASE_NAME"]
    for name in (f"{base}_best.pt", f"{base}_ep1.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")


def run_variant(eval_external, run_root: Path, variant_id: str, *, max_examples: int | None = None) -> dict[str, Any]:
    run_dir = run_root / "pretrain" / variant_id
    run_config = run_dir / "run_config.json"
    if not run_config.exists():
        raise FileNotFoundError(run_config)
    checkpoint = _checkpoint_from_run_config(run_config)
    label = f"{run_root.name}_{variant_id}_external_trio"
    arch_name = f"dwarf_v2_matrix_{run_root.name}_{variant_id}".replace("-", "_")
    eval_external.ARCH_CONFIGS[arch_name] = _arch_config_from_run_config(run_config)
    eval_external.MAX_SEQ_LEN = 2048
    eval_external.TOKENIZER = str(TOKENIZER)

    out_root = run_root / "external_trio" / variant_id
    out_root.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== external trio {variant_id} ===", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"arch={arch_name}", flush=True)
    print(f"torch cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    tokenizer = eval_external.load_tokenizer()
    load_t0 = time.time()
    model = eval_external.load_model_from_arch(arch_name, str(checkpoint), "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext_t0 = time.time()
    external_results = eval_external.run_evaluation(model, tokenizer, device, tasks=TASKS, max_examples=max_examples)
    eval_external.print_summary(label, external_results)
    payload = {
        "label": label,
        "variant_id": variant_id,
        "checkpoint": str(checkpoint),
        "arch": arch_name,
        "n_params": n_params,
        "timestamp": started,
        "tasks": TASKS,
        "max_examples": max_examples,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "run_config": str(run_config),
        "load_elapsed_s": time.time() - load_t0,
        "elapsed_s": time.time() - ext_t0,
        "results": external_results,
    }
    out = out_root / f"external_trio_{variant_id}_{started}.json"
    _write_json(out, payload)
    print(f"external trio -> {out}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"variant_id": variant_id, "external_trio_json": str(out), "payload": payload}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["no_w", "w_typed_aux0"])
    ap.add_argument("--external-max", type=int, default=None)
    args = ap.parse_args(argv)
    run_root = args.run_root.resolve()
    eval_external = _setup_legacy_imports()
    all_results = [run_variant(eval_external, run_root, v, max_examples=args.external_max) for v in args.variants]
    combined = {
        "run_root": str(run_root),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "tasks": TASKS,
        "results": all_results,
    }
    out = run_root / "external_trio" / f"combined_external_trio_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(out, combined)
    print(f"combined external trio -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
