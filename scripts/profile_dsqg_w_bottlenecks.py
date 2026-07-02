#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_dsqg_w_full_training.py"
PYTHON = Path("/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python")

BENCH_RE = re.compile(
    r"\[BENCH\] first_step_ms=([0-9.]+) trailing_avg_ms=([0-9.]+) "
    r"steady_tok_s=([0-9]+) approx_compile_overhead_ms=([0-9.]+)"
)
MEM_RE = re.compile(r"\[BENCH\] peak_vram=([0-9]+)MB")
PROFILE_PATH_RE = re.compile(r"\[DSQG-W profile\] key averages: (.+)")


def _variant_args(name: str) -> list[str]:
    base = [
        str(PYTHON),
        str(LAUNCHER.relative_to(ROOT)),
        "--run-name",
        name,
        "--max-acc-steps",
        "8",
        "--train-seqs",
        "32",
        "--val-seqs",
        "4",
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--log-interval",
        "1",
        "--passkey-trials",
        "0",
        "--execute",
    ]
    if name == "no_w":
        return base + ["--disable-dsqg-w"]
    if name == "triton_final":
        return base + ["--sites", "final", "--sourcewise", "--triton-sourcewise"]
    if name == "triton_6_final":
        return base + ["--sites", "6,final", "--sourcewise", "--triton-sourcewise"]
    if name == "triton_2_6_final":
        return base + ["--sites", "2,6,final", "--sourcewise", "--triton-sourcewise"]
    if name == "triton_no_dsr":
        return base + ["--sites", "2,6,final", "--sourcewise", "--triton-sourcewise", "--no-dsr-candidates"]
    if name == "triton_split":
        return base + ["--sites", "2,6,final", "--sourcewise", "--triton-sourcewise"]
    raise ValueError(f"unknown variant {name}")


def _extract_interesting_profile_rows(table_path: Path, limit: int = 30) -> list[str]:
    if not table_path.exists():
        return []
    rows = []
    needles = (
        "dsqg_w/",
        "DSQGWSourcewiseTritonCompactRead",
        "_dsqg_w_sourcewise",
        "aten::native_layer_norm",
        "aten::layer_norm",
        "aten::linear",
        "aten::addmm",
        "aten::mm",
        "aten::cat",
        "aten::gelu",
        "aten::argsort",
        "aten::gather",
        "aten::where",
    )
    for line in table_path.read_text(errors="ignore").splitlines():
        if any(needle in line for needle in needles):
            rows.append(line.rstrip())
    return rows[:limit]


def _run_variant(name: str, root_out: Path, args: argparse.Namespace) -> dict:
    variant_out = root_out / name
    trainer_out = variant_out / "trainer"
    trace_dir = variant_out / "trace"
    table_path = trace_dir / "key_averages.txt"
    cmd = _variant_args(name)
    cmd[cmd.index("--run-name") + 1] = name
    cmd.extend(["--output-dir", str(trainer_out)])
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": ".",
            "DWARF_BENCH_ONLY": "1",
            "DWARF_PROFILE_DSQG_W": "1",
            "DWARF_PROFILE_DSQG_W_TRACE_DIR": str(trace_dir),
            "DWARF_PROFILE_DSQG_W_TABLE": str(table_path),
            "DWARF_PROFILE_DSQG_W_WAIT": str(args.wait),
            "DWARF_PROFILE_DSQG_W_WARMUP": str(args.warmup),
            "DWARF_PROFILE_DSQG_W_ACTIVE": str(args.active),
            "DWARF_DSQG_W_GEOMETRY_AUDIT": "1",
        }
    )
    if name == "triton_split":
        env["DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION"] = "v20_split"
    variant_out.mkdir(parents=True, exist_ok=True)
    wrapper_stdout = variant_out / "launcher.stdout.log"
    wrapper_stderr = variant_out / "launcher.stderr.log"
    with wrapper_stdout.open("w", encoding="utf-8") as out, wrapper_stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err, text=True, check=False)
    trainer_stdout = trainer_out / "trainer.stdout.log"
    trainer_stderr = trainer_out / "trainer.stderr.log"
    text = trainer_stdout.read_text(errors="ignore") if trainer_stdout.exists() else ""
    bench = BENCH_RE.findall(text)
    mem = MEM_RE.findall(text)
    profile_paths = PROFILE_PATH_RE.findall(text)
    result = {
        "name": name,
        "returncode": int(proc.returncode),
        "command": cmd,
        "trainer_stdout": str(trainer_stdout),
        "trainer_stderr": str(trainer_stderr),
        "profile_table": str(table_path),
        "profile_trace_dir": str(trace_dir),
        "reported_profile_paths": profile_paths,
        "interesting_profile_rows": _extract_interesting_profile_rows(table_path),
    }
    if bench:
        first_ms, trailing_ms, tok_s, compile_ms = bench[-1]
        result.update(
            {
                "first_step_ms": float(first_ms),
                "trailing_avg_ms": float(trailing_ms),
                "steady_tok_s": int(tok_s),
                "approx_compile_overhead_ms": float(compile_ms),
            }
        )
    if mem:
        result["peak_vram_mb"] = int(mem[-1])
    if proc.returncode != 0:
        result["launcher_stderr_tail"] = wrapper_stderr.read_text(errors="ignore")[-4000:]
        result["trainer_stderr_tail"] = trainer_stderr.read_text(errors="ignore")[-4000:] if trainer_stderr.exists() else ""
    return result


def _write_summary(root_out: Path, results: list[dict]) -> None:
    (root_out / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# DSQG-W bottleneck profiler summary", ""]
    lines.append("| variant | rc | trailing ms | tok/s | peak MB | profile table |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['returncode']} | {r.get('trailing_avg_ms', 0):.1f} | "
            f"{r.get('steady_tok_s', 0)} | {r.get('peak_vram_mb', 0)} | `{r['profile_table']}` |"
        )
    for r in results:
        lines.extend(["", f"## {r['name']}", "", "```text"])
        rows = r.get("interesting_profile_rows") or ["<no matching profiler rows extracted>"]
        lines.extend(rows)
        lines.append("```")
    (root_out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal DSQG-W profiler matrix")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / f"dsqg_w_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--variants", default="no_w,triton_final,triton_6_final,triton_2_6_final,triton_no_dsr,triton_split")
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3)
    args = parser.parse_args(argv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in variants:
        print(f"[profile] running {variant}", flush=True)
        result = _run_variant(variant, args.output_dir, args)
        results.append(result)
        _write_summary(args.output_dir, results)
        if result["returncode"] != 0:
            print(f"[profile] {variant} failed; stopping", file=sys.stderr, flush=True)
            return int(result["returncode"] or 1)
    print(f"[profile] summary: {args.output_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
