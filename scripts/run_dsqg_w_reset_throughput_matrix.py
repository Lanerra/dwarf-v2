#!/usr/bin/env python3
"""Run the DSQG-W systems/objective-reset throughput gate matrix.

Acceptance ladder for the current reset:
  1. no_w control
  2. final-site full-candidate W, sourcewise Triton, no typed_mixer, no width_cell, aux=0
  3. +typed_mixer
  4. +width_cell alone (collapse detector)
  5. +typed_mixer +width_cell

This is bench-only trainer evidence, not a quality claim. Serious 100K+ sweeps stay
blocked until the full-candidate W path reaches the configured tok/s floor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_dsqg_w_full_training.py"
DEFAULT_PYTHON = Path("/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python")

BENCH_RE = re.compile(
    r"\[BENCH\] first_step_ms=([0-9.]+) trailing_avg_ms=([0-9.]+) "
    r"steady_tok_s=([0-9]+) approx_compile_overhead_ms=([0-9.]+)"
)
MEM_RE = re.compile(r"\[BENCH\] peak_vram=([0-9]+)MB")
STEP_RE = re.compile(r"\[ep\d+ step (\d+)/(\d+)\].*?ce=([0-9.]+).*? ([0-9]+) tok/s(.*)")

VARIANTS: dict[str, dict[str, Any]] = {
    "no_w": {
        "args": ["--disable-dsqg-w"],
        "description": "DSR backbone control with DSQG-W disabled",
    },
    "w_base_final": {
        "args": ["--sites", "final", "--sourcewise", "--triton-sourcewise", "--width-aux-weight", "0.0"],
        "description": "final-site full-candidate W, no typed_mixer, no width_cell, aux=0",
    },
    "w_typed_final": {
        "args": ["--sites", "final", "--sourcewise", "--triton-sourcewise", "--typed-mixer", "--width-aux-weight", "0.0"],
        "description": "w_base_final + typed_mixer, aux=0",
    },
    "w_width_final": {
        "args": ["--sites", "final", "--sourcewise", "--triton-sourcewise", "--width-cell", "--width-aux-weight", "0.0"],
        "description": "w_base_final + width_cell alone, aux=0 collapse detector",
    },
    "w_typed_width_final": {
        "args": [
            "--sites",
            "final",
            "--sourcewise",
            "--triton-sourcewise",
            "--typed-mixer",
            "--width-cell",
            "--width-aux-weight",
            "0.0",
        ],
        "description": "w_base_final + typed_mixer + width_cell, aux=0",
    },
}


def _base_cmd(args: argparse.Namespace, variant: str, out_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(LAUNCHER.relative_to(ROOT)),
        "--run-name",
        variant,
        "--output-dir",
        str(out_dir),
        "--gpu",
        str(args.gpu),
        "--max-acc-steps",
        str(args.max_acc_steps),
        "--train-seqs",
        str(args.train_seqs),
        "--val-seqs",
        str(args.val_seqs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--log-interval",
        str(args.log_interval),
        "--passkey-trials",
        "0",
        "--max-candidates",
        str(args.max_candidates),
        "--k-question",
        str(args.k_question),
        "--k-hisa-evidence",
        str(args.k_hisa_evidence),
        "--k-l3-skip",
        str(args.k_l3_skip),
        "--local-offsets",
        "none",
        "--long-offsets",
        "none",
    ]
    cmd.extend(VARIANTS[variant]["args"])
    cmd.append("--execute")
    return cmd


def _parse_result(name: str, variant_dir: Path, proc: subprocess.CompletedProcess[str], cmd: list[str]) -> dict[str, Any]:
    trainer_stdout = variant_dir / "trainer.stdout.log"
    trainer_stderr = variant_dir / "trainer.stderr.log"
    stdout_text = trainer_stdout.read_text(errors="ignore") if trainer_stdout.exists() else ""
    stderr_text = trainer_stderr.read_text(errors="ignore") if trainer_stderr.exists() else ""
    bench_matches = BENCH_RE.findall(stdout_text)
    mem_matches = MEM_RE.findall(stdout_text)
    step_matches = STEP_RE.findall(stdout_text)
    result: dict[str, Any] = {
        "name": name,
        "description": VARIANTS[name]["description"],
        "returncode": int(proc.returncode),
        "command": cmd,
        "trainer_stdout": str(trainer_stdout),
        "trainer_stderr": str(trainer_stderr),
    }
    if bench_matches:
        first_ms, trailing_ms, tok_s, compile_ms = bench_matches[-1]
        result.update(
            {
                "first_step_ms": float(first_ms),
                "trailing_avg_ms": float(trailing_ms),
                "steady_tok_s": int(tok_s),
                "approx_compile_overhead_ms": float(compile_ms),
            }
        )
    if mem_matches:
        result["peak_vram_mb"] = int(mem_matches[-1])
    if step_matches:
        step, total, ce, tok_s, tail = step_matches[-1]
        result["last_step"] = {"step": int(step), "total": int(total), "ce": float(ce), "tok_s": int(tok_s), "telemetry_tail": tail.strip()}
    if proc.returncode != 0:
        result["trainer_stderr_tail"] = stderr_text[-4000:]
        result["launcher_stdout_tail"] = proc.stdout[-4000:] if proc.stdout else ""
        result["launcher_stderr_tail"] = proc.stderr[-4000:] if proc.stderr else ""
    return result


def _write_summary(output_dir: Path, results: list[dict[str, Any]], floor_tok_s: int) -> None:
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    by_name = {r["name"]: r for r in results}
    lines = ["# DSQG-W systems/objective reset throughput matrix", ""]
    lines.append(f"Throughput floor for promotion: **{floor_tok_s:,} tok/s**. Aux is forced to 0.0 in all W variants.")
    lines.append("")
    lines.append("| variant | rc | steady tok/s | trailing ms | peak MB | last CE | gate status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in results:
        tok = int(r.get("steady_tok_s", 0) or 0)
        status = "PASS" if r["name"] == "no_w" or tok >= floor_tok_s else "BLOCKED"
        ce = r.get("last_step", {}).get("ce", 0.0)
        lines.append(
            f"| {r['name']} | {r['returncode']} | {tok:,} | {r.get('trailing_avg_ms', 0):.1f} | "
            f"{r.get('peak_vram_mb', 0)} | {ce:.4f} | {status} |"
        )
    if "w_width_final" in by_name and "w_base_final" in by_name:
        base = max(int(by_name["w_base_final"].get("steady_tok_s", 0) or 0), 1)
        width = int(by_name["w_width_final"].get("steady_tok_s", 0) or 0)
        lines.extend(["", f"Width-cell alone ratio vs W base: **{width / base:.3f}x** ({width:,} / {base:,} tok/s)."])
    lines.append("")
    lines.append("## Variant descriptions")
    for name, spec in VARIANTS.items():
        if name in by_name:
            lines.append(f"- `{name}`: {spec['description']}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DSQG-W reset throughput acceptance matrix")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / f"dsqg_w_reset_throughput_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--variants", default="no_w,w_base_final,w_typed_final,w_width_final,w_typed_width_final")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-acc-steps", type=int, default=12)
    parser.add_argument("--train-seqs", type=int, default=48)
    parser.add_argument("--val-seqs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--k-question", type=int, default=4)
    parser.add_argument("--k-hisa-evidence", type=int, default=4)
    parser.add_argument("--k-l3-skip", type=int, default=2)
    parser.add_argument("--floor-tok-s", type=int, default=50_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed variant")
    args = parser.parse_args(argv)

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known={sorted(VARIANTS)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for variant in requested:
        variant_dir = args.output_dir / variant
        cmd = _base_cmd(args, variant, variant_dir)
        if args.dry_run:
            results.append({"name": variant, "description": VARIANTS[variant]["description"], "returncode": 0, "command": cmd})
            continue
        print(f"[matrix] running {variant}: {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        env.update({"PYTHONPATH": ".", "DWARF_BENCH_ONLY": "1", "PYTHONUNBUFFERED": "1"})
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        result = _parse_result(variant, variant_dir, proc, cmd)
        results.append(result)
        _write_summary(args.output_dir, results, args.floor_tok_s)
        print(
            f"[matrix] {variant} rc={result['returncode']} tok/s={result.get('steady_tok_s', 0)} "
            f"summary={args.output_dir / 'summary.md'}",
            flush=True,
        )
        if proc.returncode != 0 and not args.keep_going:
            return int(proc.returncode or 1)
    _write_summary(args.output_dir, results, args.floor_tok_s)
    print(f"[matrix] summary: {args.output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
