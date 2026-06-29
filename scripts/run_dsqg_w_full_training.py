#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
DEFAULT_DATASET = ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt"


def _str(value: Any) -> str:
    return str(value)


def build_run_config(
    *,
    output_dir: Path | str,
    run_name: str = "dsqg_w_2_6_final_pilot",
    gpu: str = "0",
    max_acc_steps: int = 25,
    train_seqs: int = 256,
    val_seqs: int = 128,
    batch_size: int = 1,
    grad_accum: int = 1,
    epochs: int = 1,
    log_interval: int = 1,
    passkey_trials: int = 2,
    sites: str = "2,6,final",
    max_candidates: int = 16,
    bottleneck: int = 64,
    width_cell: bool = False,
    width_bottleneck: int = 64,
    width_gate_init: float = -2.5,
    width_aux_weight: float = 0.0,
    lr: float | None = None,
    dataset: Path | str = DEFAULT_DATASET,
    tokenizer: Path | str = DEFAULT_TOKENIZER,
    python: Path | str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    checkpoint_dir = out / "checkpoints"
    stdout_path = out / "trainer.stdout.log"
    stderr_path = out / "trainer.stderr.log"
    py = str(python or sys.executable)
    env: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": ".",
        "DWARF_TOKENIZER": str(tokenizer),
        "DWARF_DATASET": str(dataset),
        "DWARF_CHECKPOINT_DIR": str(checkpoint_dir),
        "DWARF_CKPT_BASE_NAME": f"d512_l10_dsqg_w_{run_name}",
        "DWARF_EPOCHS": str(int(epochs)),
        "DWARF_MAX_ACC_STEPS": str(int(max_acc_steps)),
        "DWARF_MAX_TRAIN_SEQS": str(int(train_seqs)),
        "DWARF_MAX_VAL_SEQS": str(int(val_seqs)),
        "DWARF_BS": str(int(batch_size)),
        "DWARF_GA": str(int(grad_accum)),
        "DWARF_LOG_INTERVAL": str(int(log_interval)),
        "DWARF_PASSKEY_TRIALS": str(int(passkey_trials)),
        "DWARF_DSQG_W": "1",
        "DWARF_DSQG_W_SITES": str(sites),
        "DWARF_DSQG_W_MAX_CANDIDATES": str(int(max_candidates)),
        "DWARF_DSQG_W_BOTTLENECK": str(int(bottleneck)),
        "DWARF_DSQG_W_WIDTH_CELL": "1" if width_cell else "0",
        "DWARF_DSQG_W_WIDTH_BOTTLENECK": str(int(width_bottleneck)),
        "DWARF_DSQG_W_WIDTH_GATE_INIT": str(float(width_gate_init)),
        "DWARF_DSQG_W_WIDTH_AUX_WEIGHT": str(float(width_aux_weight)),
        "DWARF_DSQG_W_QUESTION": "1",
        "DWARF_DSQG_W_HISA_L3": "1",
        "DWARF_DSQG_W_K_QUESTION": "4",
        "DWARF_DSQG_W_K_HISA_EVIDENCE": "4",
        "DWARF_DSQG_W_K_L3_SKIP": "2",
        "DWARF_TORCH_COMPILE": "0",
        "DWARF_LIGER": "0",
        "DWARF_Q6_G128": "0",
        "DWARF_PIN_DATASET": "0",
    }
    if lr is not None:
        env["DWARF_LR"] = str(float(lr))
    return {
        "objective": "dsqg_w_full_training_launcher",
        "run_name": run_name,
        "root": str(ROOT),
        "trainer": str(TRAINER),
        "output_dir": str(out),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "config_path": str(out / "run_config.json"),
        "command": [py, str(TRAINER.relative_to(ROOT))],
        "env": env,
    }


def _jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, sort_keys=True))


def write_config(config: dict[str, Any]) -> Path:
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path(config["env"]["DWARF_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
    path = Path(config["config_path"])
    path.write_text(json.dumps(_jsonable_config(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def execute_config(config: dict[str, Any]) -> dict[str, Any]:
    write_config(config)
    env = os.environ.copy()
    env.update(config["env"])
    stdout_path = Path(config["stdout_path"])
    stderr_path = Path(config["stderr_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            config["command"],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    elapsed_s = time.time() - t0
    return {
        "returncode": int(completed.returncode),
        "elapsed_s": elapsed_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a bounded real-trainer DSQG-W full-run pilot")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_full_training_pilot"))
    parser.add_argument("--run-name", default="dsqg_w_2_6_final_pilot")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-acc-steps", type=int, default=25)
    parser.add_argument("--train-seqs", type=int, default=256)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--passkey-trials", type=int, default=2)
    parser.add_argument("--sites", default="2,6,final")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--width-cell", action="store_true", help="Enable the opt-in DSQG-W candidate lateral width cell.")
    parser.add_argument("--width-bottleneck", type=int, default=64)
    parser.add_argument("--width-gate-init", type=float, default=-2.5)
    parser.add_argument("--width-aux-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true", help="Write run_config.json and exit without executing trainer.")
    parser.add_argument("--execute", action="store_true", help="Execute trainer after writing run_config.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_run_config(
        output_dir=args.output_dir,
        run_name=args.run_name,
        gpu=args.gpu,
        max_acc_steps=args.max_acc_steps,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        log_interval=args.log_interval,
        passkey_trials=args.passkey_trials,
        sites=args.sites,
        max_candidates=args.max_candidates,
        bottleneck=args.bottleneck,
        width_cell=args.width_cell,
        width_bottleneck=args.width_bottleneck,
        width_gate_init=args.width_gate_init,
        width_aux_weight=args.width_aux_weight,
        lr=args.lr,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        python=args.python,
    )
    config_path = write_config(config)
    execute = bool(args.execute and not args.dry_run)
    report: dict[str, Any] = {
        "pass": True,
        "executed": False,
        "config_path": str(config_path),
        **config,
    }
    if execute:
        exec_report = execute_config(config)
        report.update(exec_report)
        report["executed"] = True
        report["pass"] = exec_report["returncode"] == 0
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)
