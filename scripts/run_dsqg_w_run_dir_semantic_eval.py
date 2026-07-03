#!/usr/bin/env python3
"""Run semantic-transfer evals for DSQG-W matrix run directories.

This generic helper mirrors scripts/run_reset200k_checkpoint_evals_3090.py but
accepts arbitrary run roots created by the corrected 200-step matrix runner. Run
it in a separate process with CUDA_VISIBLE_DEVICES set to the desired eval GPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import os
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _setup_legacy_imports():
    for p in reversed(PYTHON_EVAL_ROOTS):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    import analysis_shortlist.eval_external as eval_external  # type: ignore

    eval_semantic = _load_module("legacy_eval_semantic_transfer", LEGACY_ROOT / "evals" / "eval_semantic_transfer.py")
    core = _load_module("legacy_semantic_transfer_core", LEGACY_ROOT / "tools" / "semantic_transfer_eval.py")
    return eval_external, eval_semantic, core


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
    best = ckpt_dir / f"{base}_best.pt"
    if best.exists():
        return best
    ep1 = ckpt_dir / f"{base}_ep1.pt"
    if ep1.exists():
        return ep1
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")


def _run_run_dir(eval_external, eval_semantic, core, run_dir: Path, *, run_root: Path, suite: str) -> dict[str, Any]:
    run_config = run_dir / "run_config.json"
    if not run_config.exists():
        raise FileNotFoundError(run_config)
    checkpoint = _checkpoint_from_run_config(run_config)
    variant_id = run_dir.name
    label = f"{run_root.name}_{variant_id}"
    arch_name = f"dwarf_v2_matrix_{run_root.name}_{variant_id}".replace("-", "_")

    eval_external.ARCH_CONFIGS[arch_name] = _arch_config_from_run_config(run_config)
    eval_external.MAX_SEQ_LEN = 2048
    eval_external.TOKENIZER = str(TOKENIZER)

    out_root = run_root / "semantic_transfer" / variant_id
    out_root.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== semantic {variant_id} ===", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"arch={arch_name}", flush=True)
    print(f"torch cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    tokenizer = eval_external.load_tokenizer()
    load_t0 = time.time()
    model = eval_external.load_model_from_arch(arch_name, str(checkpoint), "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    semantic_t0 = time.time()
    examples = eval_semantic.build_suite_examples(core, suite=suite)
    overall, by_family, example_rows = eval_semantic.run_core_evaluation(
        core,
        examples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=2048,
    )
    payload = eval_semantic.build_payload(
        label=f"{label}_{suite}",
        checkpoint=str(checkpoint),
        arch=arch_name,
        suite=suite,
        overall=overall,
        by_family=by_family,
        examples=example_rows,
        elapsed_s=time.time() - semantic_t0,
        metadata={
            "model_loading": "DWARF-v2 injected ARCH_CONFIGS via run_config.json",
            "tokenizer": str(TOKENIZER),
            "max_seq_len": 2048,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "run_config": str(run_config),
            "run_root": str(run_root),
        },
    )
    sem_json = out_root / f"semantic_transfer_{variant_id}_{suite}_{started}.json"
    sem_md = out_root / f"semantic_transfer_{variant_id}_{suite}_{started}.md"
    eval_semantic.write_json(sem_json, payload)
    eval_semantic.write_markdown(sem_md, payload)
    result = {
        "variant_id": variant_id,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "arch": arch_name,
        "n_params": n_params,
        "load_elapsed_s": time.time() - load_t0,
        "semantic_transfer": {"overall": overall, "by_family": by_family, "elapsed_s": payload["elapsed_s"]},
        "outputs": {"semantic_transfer_json": str(sem_json), "semantic_transfer_md": str(sem_md)},
    }
    print(f"semantic-transfer -> {sem_json}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _select_run_dirs(run_root: Path, variant_ids: list[str] | None) -> list[Path]:
    pretrain = run_root / "pretrain"
    if variant_ids:
        return [pretrain / v for v in variant_ids]
    return sorted(p for p in pretrain.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--variant-ids", nargs="*", default=None)
    ap.add_argument("--semantic-suite", default="builtin_v3_deconfounded")
    args = ap.parse_args(argv)

    run_root = args.run_root.resolve()
    run_dirs = _select_run_dirs(run_root, args.variant_ids)
    eval_external, eval_semantic, core = _setup_legacy_imports()
    results = []
    for run_dir in run_dirs:
        results.append(_run_run_dir(eval_external, eval_semantic, core, run_dir, run_root=run_root, suite=args.semantic_suite))
    combined = {
        "run_root": str(run_root),
        "semantic_suite": args.semantic_suite,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "results": results,
    }
    combined_path = run_root / "semantic_transfer" / f"combined_semantic_transfer_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(combined_path, combined)
    print(f"combined semantic summary -> {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
