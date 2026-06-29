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
    spec = importlib.util.spec_from_file_location("microtrain_dsqg_w_lexical_gap_for_sweep", MICROTRAIN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _lr_label(lr: float) -> str:
    return f"{lr:g}".replace("-", "m").replace(".", "p")


def expand_sweep_grid(
    *,
    size_grid: list[tuple[int, int]],
    step_grid: list[int],
    lr_grid: list[float],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for train_size, val_size in size_grid:
        for steps in step_grid:
            for lr in lr_grid:
                run_id = f"train{train_size}_val{val_size}_steps{steps}_lr{_lr_label(float(lr))}"
                configs.append(
                    {
                        "run_id": run_id,
                        "train_size": int(train_size),
                        "val_size": int(val_size),
                        "steps": int(steps),
                        "lr": float(lr),
                    }
                )
    return configs


def _compact_run(run_id: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "pass": bool(report["pass"]),
        "train_examples": int(report["train_examples"]),
        "val_examples": int(report["val_examples"]),
        "steps": int(report["steps"]),
        "lr": float(report["lr"]),
        "train_loss_initial": float(report["train_loss_initial"]),
        "train_loss_final": float(report["train_loss_final"]),
        "train_loss_delta": float(report["train_loss_delta"]),
        "val_loss_initial": float(report["val_loss_initial"]),
        "val_loss_final": float(report["val_loss_final"]),
        "val_loss_delta": float(report["val_loss_delta"]),
        "changed_dsqg_w_param_count": int(report["changed_dsqg_w_param_count"]),
        "changed_frozen_param_count": int(report["changed_frozen_param_count"]),
        "checkpoint_roundtrip_loss_delta": float(report["checkpoint_roundtrip_loss_delta"]),
        "report_path": str(report["report_path"]),
        "checkpoint_state": str(report["checkpoint"]["state_path"]),
    }


def run_sweep(
    *,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    output_dir: Path | str,
    size_grid: list[tuple[int, int]],
    step_grid: list[int],
    lr_grid: list[float],
    seed: int = 20260628,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    micro = load_microtrain_module()
    configs = expand_sweep_grid(size_grid=size_grid, step_grid=step_grid, lr_grid=lr_grid)
    runs: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        run_dir = output / config["run_id"]
        report_path = run_dir / "microtrain_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = micro.run_microtrain(
                tokenizer_path=tokenizer_path,
                output_dir=run_dir,
                train_size=config["train_size"],
                val_size=config["val_size"],
                steps=config["steps"],
                lr=config["lr"],
                seed=seed + index,
            )
        runs.append(_compact_run(config["run_id"], report))

    passing_runs = [run for run in runs if run["pass"]]
    best_by_val = min(passing_runs or runs, key=lambda run: (run["val_loss_delta"], run["val_loss_final"]))
    best_by_train = min(passing_runs or runs, key=lambda run: (run["train_loss_delta"], run["train_loss_final"]))
    unstable_runs = [
        run
        for run in runs
        if (not run["pass"])
        or run["changed_frozen_param_count"] != 0
        or abs(run["checkpoint_roundtrip_loss_delta"]) > 1e-6
    ]
    summary = {
        "pass": bool(runs and all(run["pass"] for run in runs) and not unstable_runs),
        "objective": "dsqg_w_microtrain_sweep",
        "tokenizer_path": str(tokenizer_path),
        "run_count": len(runs),
        "size_grid": [[train, val] for train, val in size_grid],
        "step_grid": [int(step) for step in step_grid],
        "lr_grid": [float(lr) for lr in lr_grid],
        "seed": int(seed),
        "runs": runs,
        "best_by_val_loss_delta": best_by_val,
        "best_by_train_loss_delta": best_by_train,
        "unstable_runs": unstable_runs,
    }
    summary_path = output / "sweep_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_size_grid(text: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("x", ":").split(":", 1)
        sizes.append((int(left), int(right)))
    return sizes


def _parse_int_grid(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an opt-in DSQG-W lexical-gap microtrain sweep")
    parser.add_argument("--enable", action="store_true", help="Run the sweep. Omit to report disabled/skipped.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_microtrain_sweep"))
    parser.add_argument("--size-grid", default="64:16,128:32")
    parser.add_argument("--steps-grid", default="4,8,16,32")
    parser.add_argument("--lr-grid", default="0.0003,0.001,0.003")
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    if not args.enable:
        report = {
            "enabled": False,
            "skipped": True,
            "pass": True,
            "reason": "pass --enable to run the DSQG-W microtrain sweep",
        }
    else:
        report = run_sweep(
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            size_grid=_parse_size_grid(args.size_grid),
            step_grid=_parse_int_grid(args.steps_grid),
            lr_grid=_parse_float_grid(args.lr_grid),
            seed=args.seed,
        )
        report["enabled"] = True
        report["skipped"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
