#!/usr/bin/env python3
"""Run DSQG-W reset-200k checkpoint evals on a selected CUDA device.

This bridges the legacy DWARF eval harness to DWARF-v2 checkpoints by injecting
arch configs from each run_config.json. Results are written under the DWARF-v2
run directory, not the legacy DWARF tree.
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
RUN_ROOT = ROOT / "runs" / "dsqg_w_reset_real_20260702_124636"
TRAIN_SCRIPT = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
TOKENIZER = ROOT / "tokenizers" / "olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
PYTHON_EVAL_ROOTS = [LEGACY_ROOT, LEGACY_ROOT / "analysis_shortlist", LEGACY_ROOT / "evals", LEGACY_ROOT / "tools"]
TASKS = ["arc_easy", "piqa", "lambada"]

VARIANTS: dict[str, dict[str, Any]] = {
    "B_dsr_rep4": {
        "label": "reset200k_B_dsr_rep4_no_w",
        "run_dir": RUN_ROOT / "pretrain" / "B_dsr_rep4",
        "checkpoint": RUN_ROOT / "pretrain" / "B_dsr_rep4" / "checkpoints" / "d512_l10_dsqg_w_pretrain_B_dsr_rep4_best.pt",
    },
    "E_fast_l3_3site": {
        "label": "reset200k_E_fast_l3_3site",
        "run_dir": RUN_ROOT / "pretrain" / "E_fast_l3_3site",
        "checkpoint": RUN_ROOT / "pretrain" / "E_fast_l3_3site" / "checkpoints" / "d512_l10_dsqg_w_pretrain_E_fast_l3_3site_best.pt",
    },
    "F_fast_l3_final": {
        "label": "reset200k_F_fast_l3_final",
        "run_dir": RUN_ROOT / "pretrain" / "F_fast_l3_final",
        "checkpoint": RUN_ROOT / "pretrain" / "F_fast_l3_final" / "checkpoints" / "d512_l10_dsqg_w_pretrain_F_fast_l3_final_best.pt",
    },
}


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
    # Load the legacy semantic core by absolute path. DWARF-v2's trainer later
    # installs its own `tools` package in sys.modules, so relying on
    # `import tools.semantic_transfer_eval` is order-sensitive when cwd/PYTHONPATH
    # point at DWARF-v2.
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


def _arch_config_from_run_config(run_config: Path) -> dict[str, Any]:
    cfg = json.loads(run_config.read_text(encoding="utf-8"))
    env = cfg.get("env", {})
    # Keep only architecture/model-construction toggles. Runtime geometry (CUDA,
    # batch, dataset, checkpoint dirs) is controlled by this eval runner.
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n", encoding="utf-8")


def _run_variant(eval_external, eval_semantic, core, variant_id: str, *, semantic_suite: str, external_max: int | None, skip_semantic: bool, skip_external: bool) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    label = variant["label"]
    checkpoint = Path(variant["checkpoint"])
    run_dir = Path(variant["run_dir"])
    run_config = run_dir / "run_config.json"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not run_config.exists():
        raise FileNotFoundError(run_config)

    arch_name = f"dwarf_v2_reset200k_{variant_id}"
    arch_config = _arch_config_from_run_config(run_config)
    eval_external.ARCH_CONFIGS[arch_name] = arch_config
    eval_external.MAX_SEQ_LEN = 2048
    eval_external.TOKENIZER = str(TOKENIZER)

    out_root = RUN_ROOT / "evals" / label
    out_root.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== {variant_id} :: {label} ===", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"arch={arch_name}", flush=True)
    print(f"torch cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    tokenizer = eval_external.load_tokenizer()
    load_t0 = time.time()
    model = eval_external.load_model_from_arch(arch_name, str(checkpoint), "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result: dict[str, Any] = {
        "variant_id": variant_id,
        "label": label,
        "checkpoint": str(checkpoint),
        "arch": arch_name,
        "n_params": n_params,
        "started": started,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "load_elapsed_s": time.time() - load_t0,
        "outputs": {},
    }

    if not skip_semantic:
        semantic_t0 = time.time()
        examples = eval_semantic.build_suite_examples(core, suite=semantic_suite)
        overall, by_family, example_rows = eval_semantic.run_core_evaluation(
            core,
            examples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=2048,
        )
        payload = eval_semantic.build_payload(
            label=f"{label}_{semantic_suite}",
            checkpoint=str(checkpoint),
            arch=arch_name,
            suite=semantic_suite,
            overall=overall,
            by_family=by_family,
            examples=example_rows,
            elapsed_s=time.time() - semantic_t0,
            metadata={
                "model_loading": "DWARF-v2 injected ARCH_CONFIGS via run_config.json",
                "tokenizer": str(TOKENIZER),
                "max_seq_len": 2048,
                "device": result["device"],
                "run_config": str(run_config),
            },
        )
        sem_json = out_root / f"semantic_transfer_{label}_{semantic_suite}_{started}.json"
        sem_md = out_root / f"semantic_transfer_{label}_{semantic_suite}_{started}.md"
        eval_semantic.write_json(sem_json, payload)
        eval_semantic.write_markdown(sem_md, payload)
        result["outputs"]["semantic_transfer_json"] = str(sem_json)
        result["outputs"]["semantic_transfer_md"] = str(sem_md)
        result["semantic_transfer"] = {"overall": overall, "by_family": by_family, "elapsed_s": payload["elapsed_s"]}
        print(f"semantic-transfer -> {sem_json}", flush=True)

    if not skip_external:
        ext_t0 = time.time()
        external_results = eval_external.run_evaluation(model, tokenizer, device, tasks=TASKS, max_examples=external_max)
        eval_external.print_summary(label, external_results)
        ext_payload = {
            "label": label,
            "variant_id": variant_id,
            "checkpoint": str(checkpoint),
            "arch": arch_name,
            "n_params": n_params,
            "timestamp": started,
            "tasks": TASKS,
            "max_examples": external_max,
            "device": result["device"],
            "run_config": str(run_config),
            "elapsed_s": time.time() - ext_t0,
            "results": external_results,
        }
        ext_json = out_root / f"external_trio_{label}_{started}.json"
        _write_json(ext_json, ext_payload)
        result["outputs"]["external_trio_json"] = str(ext_json)
        result["external_trio"] = {"results": external_results, "elapsed_s": ext_payload["elapsed_s"]}
        print(f"external trio -> {ext_json}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary_path = out_root / f"eval_summary_{label}_{started}.json"
    _write_json(summary_path, result)
    result["outputs"]["summary_json"] = str(summary_path)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--semantic-suite", default="builtin_v3_deconfounded")
    ap.add_argument("--external-max", type=int, default=None)
    ap.add_argument("--skip-semantic", action="store_true")
    ap.add_argument("--skip-external", action="store_true")
    args = ap.parse_args(argv)

    eval_external, eval_semantic, core = _setup_legacy_imports()
    all_results = []
    for variant_id in args.variants:
        all_results.append(
            _run_variant(
                eval_external,
                eval_semantic,
                core,
                variant_id,
                semantic_suite=args.semantic_suite,
                external_max=args.external_max,
                skip_semantic=args.skip_semantic,
                skip_external=args.skip_external,
            )
        )
    combined_path = RUN_ROOT / "evals" / f"combined_reset200k_evals_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(combined_path, {"results": all_results})
    print(f"combined summary -> {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
