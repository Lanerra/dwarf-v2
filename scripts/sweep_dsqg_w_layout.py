#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MICROTRAIN = ROOT / "scripts/microtrain_dsqg_w_lexical_gap.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_microtrain_module():
    spec = importlib.util.spec_from_file_location("microtrain_dsqg_w_lexical_gap_for_layout", MICROTRAIN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def layout_id(site_spec: str) -> str:
    cleaned = "_".join(part.strip().replace("layer_", "") for part in str(site_spec).split(",") if part.strip())
    return "sites_" + cleaned.replace("-", "m").replace(".", "p")


def _compact_layout(site_spec: str, report: dict[str, Any]) -> dict[str, Any]:
    params = int(report["dsqg_w_trainable_param_count"])
    params_m = max(params / 1_000_000.0, 1e-12)
    return {
        "layout_id": layout_id(site_spec),
        "site_spec": site_spec,
        "pass": bool(report["pass"]),
        "dsqg_w_sites": list(report["dsqg_w_sites"]),
        "dsqg_w_site_count": int(report["dsqg_w_site_count"]),
        "dsqg_w_trainable_param_count": int(report["dsqg_w_trainable_param_count"]),
        "train_examples": int(report["train_examples"]),
        "val_examples": int(report["val_examples"]),
        "steps": int(report["steps"]),
        "lr": float(report["lr"]),
        "train_loss_delta": float(report["train_loss_delta"]),
        "val_loss_delta": float(report["val_loss_delta"]),
        "val_loss_final": float(report["val_loss_final"]),
        "val_mean_rank_initial": float(report["val_mean_rank_initial"]),
        "val_mean_rank_final": float(report["val_mean_rank_final"]),
        "val_mean_rank_delta": float(report["val_mean_rank_delta"]),
        "val_top1_acc_initial": float(report["val_top1_acc_initial"]),
        "val_top1_acc_final": float(report["val_top1_acc_final"]),
        "val_top1_acc_delta": float(report["val_top1_acc_delta"]),
        "val_top5_acc_initial": float(report["val_top5_acc_initial"]),
        "val_top5_acc_final": float(report["val_top5_acc_final"]),
        "val_top5_acc_delta": float(report["val_top5_acc_delta"]),
        "changed_dsqg_w_param_count": int(report["changed_dsqg_w_param_count"]),
        "changed_frozen_param_count": int(report["changed_frozen_param_count"]),
        "checkpoint_roundtrip_loss_delta": float(report["checkpoint_roundtrip_loss_delta"]),
        "top1_per_million_params": float(report["val_top1_acc_final"]) / params_m,
        "top5_per_million_params": float(report["val_top5_acc_final"]) / params_m,
        "report_path": str(report["report_path"]),
        "checkpoint_state": str(report["checkpoint"]["state_path"]),
    }


def run_id(site_spec: str, *, steps: int, lr: float) -> str:
    lr_text = f"{float(lr):g}".replace("-", "m").replace(".", "p")
    return f"{layout_id(site_spec)}_steps{int(steps)}_lr{lr_text}"


def run_layout_experiment(
    *,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    output_dir: Path | str,
    site_specs: list[str],
    train_size: int = 64,
    val_size: int = 16,
    steps: int = 16,
    lr: float = 1e-3,
    seed: int = 20260628,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    micro = load_microtrain_module()
    layouts: list[dict[str, Any]] = []
    for site_spec in site_specs:
        run_dir = output / layout_id(site_spec)
        report_path = run_dir / "microtrain_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = micro.run_microtrain(
                tokenizer_path=tokenizer_path,
                output_dir=run_dir,
                train_size=train_size,
                val_size=val_size,
                steps=steps,
                lr=lr,
                seed=seed,
                dsqg_w_sites=site_spec,
            )
        layouts.append(_compact_layout(site_spec, report))

    passing = [row for row in layouts if row["pass"]]
    rank_source = passing or layouts
    best_by_val_top5 = max(rank_source, key=lambda row: (row["val_top5_acc_final"], row["val_top1_acc_final"], -row["val_mean_rank_final"]))
    best_by_val_top1 = max(rank_source, key=lambda row: (row["val_top1_acc_final"], row["val_top5_acc_final"], -row["val_mean_rank_final"]))
    best_by_val_mean_rank = min(rank_source, key=lambda row: (row["val_mean_rank_final"], row["val_loss_final"]))
    unstable = [
        row
        for row in layouts
        if (not row["pass"])
        or row["changed_frozen_param_count"] != 0
        or abs(row["checkpoint_roundtrip_loss_delta"]) > 1e-6
    ]
    summary = {
        "pass": bool(layouts and all(row["pass"] for row in layouts) and not unstable),
        "objective": "dsqg_w_layout_experiment",
        "tokenizer_path": str(tokenizer_path),
        "layout_count": len(layouts),
        "site_specs": list(site_specs),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "layouts": layouts,
        "best_by_val_top5": best_by_val_top5,
        "best_by_val_top1": best_by_val_top1,
        "best_by_val_mean_rank": best_by_val_mean_rank,
        "unstable_layouts": unstable,
    }
    summary_path = output / "layout_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_layout_grid_experiment(
    *,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    output_dir: Path | str,
    site_specs: list[str],
    steps_grid: list[int],
    lr_grid: list[float],
    train_size: int = 64,
    val_size: int = 16,
    seed: int = 20260628,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    micro = load_microtrain_module()
    runs: list[dict[str, Any]] = []
    for site_spec in site_specs:
        for steps in steps_grid:
            for lr in lr_grid:
                rid = run_id(site_spec, steps=steps, lr=lr)
                run_dir = output / rid
                report_path = run_dir / "microtrain_report.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                else:
                    report = micro.run_microtrain(
                        tokenizer_path=tokenizer_path,
                        output_dir=run_dir,
                        train_size=train_size,
                        val_size=val_size,
                        steps=int(steps),
                        lr=float(lr),
                        seed=seed,
                        dsqg_w_sites=site_spec,
                    )
                row = _compact_layout(site_spec, report)
                row["run_id"] = rid
                runs.append(row)

    passing = [row for row in runs if row["pass"]]
    rank_source = passing or runs
    best_by_val_top1 = max(rank_source, key=lambda row: (row["val_top1_acc_final"], row["val_top5_acc_final"], -row["val_mean_rank_final"]))
    best_by_val_top5 = max(rank_source, key=lambda row: (row["val_top5_acc_final"], row["val_top1_acc_final"], -row["val_mean_rank_final"]))
    best_by_val_mean_rank = min(rank_source, key=lambda row: (row["val_mean_rank_final"], row["val_loss_final"]))
    best_by_param_efficiency_top1 = max(rank_source, key=lambda row: (row["top1_per_million_params"], row["val_top1_acc_final"]))
    best_by_param_efficiency_top5 = max(rank_source, key=lambda row: (row["top5_per_million_params"], row["val_top5_acc_final"]))
    unstable = [
        row
        for row in runs
        if (not row["pass"])
        or row["changed_frozen_param_count"] != 0
        or abs(row["checkpoint_roundtrip_loss_delta"]) > 1e-6
    ]
    summary = {
        "pass": bool(runs and all(row["pass"] for row in runs) and not unstable),
        "objective": "dsqg_w_layout_grid_experiment",
        "tokenizer_path": str(tokenizer_path),
        "run_count": len(runs),
        "site_specs": list(site_specs),
        "steps_grid": [int(v) for v in steps_grid],
        "lr_grid": [float(v) for v in lr_grid],
        "train_size": int(train_size),
        "val_size": int(val_size),
        "seed": int(seed),
        "runs": runs,
        "best_by_val_top1": best_by_val_top1,
        "best_by_val_top5": best_by_val_top5,
        "best_by_val_mean_rank": best_by_val_mean_rank,
        "best_by_param_efficiency_top1": best_by_param_efficiency_top1,
        "best_by_param_efficiency_top5": best_by_param_efficiency_top5,
        "unstable_runs": unstable,
    }
    summary_path = output / "layout_grid_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_site_specs(text: str) -> list[str]:
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_int_grid(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("integer grid must contain at least one value")
    return values


def _parse_float_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("float grid must contain at least one value")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in DSQG-W site-layout microtrain experiment")
    parser.add_argument("--enable", action="store_true", help="Run the layout experiment. Omit to report disabled/skipped.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_layout_experiment"))
    parser.add_argument("--site-specs", default="final;6,final;2,6,final", help="Semicolon-separated site specs; commas separate sites inside a layout.")
    parser.add_argument("--grid", action="store_true", help="Run every site spec across --steps-grid and --lr-grid.")
    parser.add_argument("--steps-grid", default="8,16,32", help="Comma-separated steps for --grid mode.")
    parser.add_argument("--lr-grid", default="0.001,0.003", help="Comma-separated learning rates for --grid mode.")
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    if not args.enable:
        report = {
            "enabled": False,
            "skipped": True,
            "pass": True,
            "reason": "pass --enable to run DSQG-W layout experiment",
        }
    else:
        site_specs = _parse_site_specs(args.site_specs)
        if args.grid:
            report = run_layout_grid_experiment(
                tokenizer_path=args.tokenizer,
                output_dir=args.output_dir,
                site_specs=site_specs,
                steps_grid=_parse_int_grid(args.steps_grid),
                lr_grid=_parse_float_grid(args.lr_grid),
                train_size=args.train_size,
                val_size=args.val_size,
                seed=args.seed,
            )
        else:
            report = run_layout_experiment(
                tokenizer_path=args.tokenizer,
                output_dir=args.output_dir,
                site_specs=site_specs,
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
