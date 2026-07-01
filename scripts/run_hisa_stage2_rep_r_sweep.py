#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/run_dsqg_w_full_training.py"
PARSER_PATH = ROOT / "scripts/parse_dsqg_w_ladder.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load_module(LAUNCHER_PATH, "run_dsqg_w_full_training_for_hisa_rep_sweep")
parser_mod = _load_module(PARSER_PATH, "parse_dsqg_w_ladder_for_hisa_rep_sweep")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None
    return out or None


def parse_rep_rs(text: str) -> list[int]:
    reps: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        rep = int(part)
        if rep < 0:
            raise ValueError(f"rep_r must be non-negative, got {rep}")
        reps.append(rep)
    if not reps:
        raise ValueError("at least one rep_r value is required")
    return reps


def build_rep_config(*, args: argparse.Namespace, out_root: Path, rep_r: int) -> dict[str, Any]:
    run_name = f"hisa_stage2_rep{rep_r}_{'w' if args.dsqg_w else 'donly'}_{args.max_acc_steps}step"
    run_dir = out_root / f"rep{rep_r}"
    config = launcher.build_run_config(
        output_dir=run_dir,
        run_name=run_name,
        gpu=args.gpu,
        max_acc_steps=args.max_acc_steps,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        log_interval=args.log_interval,
        passkey_trials=args.passkey_trials,
        dsqg_w=args.dsqg_w,
        sites=args.sites,
        max_candidates=args.max_candidates,
        bottleneck=args.bottleneck,
        gate_init=args.gate_init,
        fuse_init_std=args.fuse_init_std,
        sourcewise=args.sourcewise,
        triton_sourcewise=args.triton_sourcewise,
        typed_mixer=args.typed_mixer,
        query_type_bias=args.query_type_bias,
        typed_hisa_reps=args.typed_hisa_reps,
        dsr_candidates=not args.no_dsr_candidates,
        local_offsets=args.local_offsets,
        long_offsets=args.long_offsets,
        hisa_stage2_rep_r=rep_r,
        pure_dsqg=False,
        lr=args.lr,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        python=args.python,
    )
    config["env"].update(
        {
            "HISA_TELEMETRY": "1" if args.hisa_telemetry else "0",
            "DWARF_DISABLE_BNB": "1" if args.disable_bnb else "0",
            "DWARF_LIGER": "1" if args.liger else "0",
            "DWARF_CKPT": "none",
            "DWARF_PIN_DATASET": "1" if args.pin_dataset else "0",
            "DWARF_BENCH_ONLY": "1" if args.bench_only else "0",
        }
    )
    config["hisa_stage2_rep_r_sweep"] = {
        "rep_r": int(rep_r),
        "expected_steps": int(args.max_acc_steps),
        "expected_gpu": args.expected_gpu,
        "dsqg_w": bool(args.dsqg_w),
        "bench_only": bool(args.bench_only),
    }
    return config


def compact_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    steps = metrics.get("steps") or []
    last = steps[-1] if steps else {}
    return {
        "rep_r": result.get("rep_r"),
        "health_pass": result.get("health", {}).get("pass"),
        "health_errors": result.get("health", {}).get("errors", []),
        "returncode": result.get("returncode"),
        "final_step": metrics.get("final_step"),
        "planned_steps": metrics.get("planned_steps"),
        "final_ce": metrics.get("final_ce"),
        "val_ppl": metrics.get("val_ppl"),
        "passkey_mean": metrics.get("passkey_mean"),
        "avg_logged_tok_s": metrics.get("avg_logged_tok_s"),
        "last_logged_tok_s": last.get("tok_s"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "elapsed_s": metrics.get("elapsed_s"),
        "hisa_stage2_rep_r": metrics.get("hisa_stage2_rep_r"),
        "stage2_frac": metrics.get("stage2_frac", last.get("stage2_frac")),
        "routing_ent": metrics.get("routing_ent", last.get("routing_ent")),
        "w_hisa": metrics.get("w_hisa", last.get("w_hisa")),
        "w_smean": metrics.get("w_smean", last.get("w_smean")),
        "w_score": metrics.get("w_score", last.get("w_score")),
        "run_dir": result.get("run_dir"),
        "stdout_path": metrics.get("stdout_path"),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((row for row in rows if row.get("rep_r") == 0), rows[0] if rows else {})
    base_tok = baseline.get("avg_logged_tok_s")
    base_ppl = baseline.get("val_ppl")
    base_ce = baseline.get("final_ce")
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        tok = row.get("avg_logged_tok_s")
        if isinstance(tok, (int, float)) and isinstance(base_tok, (int, float)) and float(base_tok) != 0.0:
            out["tok_s_vs_rep0"] = float(tok) / float(base_tok)
        ppl = row.get("val_ppl")
        if isinstance(ppl, (int, float)) and isinstance(base_ppl, (int, float)):
            out["delta_val_ppl_vs_rep0"] = float(ppl) - float(base_ppl)
        ce = row.get("final_ce")
        if isinstance(ce, (int, float)) and isinstance(base_ce, (int, float)):
            out["delta_final_ce_vs_rep0"] = float(ce) - float(base_ce)
        enriched.append(out)
    return {"pass": all(bool(row.get("health_pass")) for row in enriched), "rows": enriched}


def format_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(path: Path, *, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = summary.get("rows", [])
    lines: list[str] = []
    lines.append(f"# HISA Stage-2 rep_r sweep — {manifest['started_at'][:10]}")
    lines.append("")
    lines.append(f"Commit: `{manifest.get('git_commit')}`")
    lines.append(f"Mode: {'DSQG-W enabled' if manifest['args']['dsqg_w'] else 'D-only / DSQG-W disabled'}")
    lines.append(f"Device target: `CUDA_VISIBLE_DEVICES={manifest['args']['gpu']}`, expected `{manifest['args']['expected_gpu']}`")
    lines.append(f"Dataset: `{manifest['args']['dataset']}`")
    lines.append(
        f"Common run: `{manifest['args']['max_acc_steps']}` optimizer steps, "
        f"train_seqs={manifest['args']['train_seqs']}, val_seqs={manifest['args']['val_seqs']}, "
        f"bs={manifest['args']['batch_size']}, ga={manifest['args']['grad_accum']}, "
        f"log_interval={manifest['args']['log_interval']}, passkey_trials={manifest['args']['passkey_trials']}."
    )
    lines.append("")
    lines.append("| rep_r | health | avg tok/s | vs rep0 | final CE | ΔCE | val PPL | ΔPPL | passkey | stage2 frac | routing ent | peak VRAM | elapsed |")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        health = "PASS" if row.get("health_pass") else "FAIL"
        lines.append(
            f"| {row.get('rep_r')} | {health} | {format_value(row.get('avg_logged_tok_s'), digits=0)} | "
            f"{format_value(row.get('tok_s_vs_rep0'), digits=3)}x | "
            f"{format_value(row.get('final_ce'), digits=4)} | {format_value(row.get('delta_final_ce_vs_rep0'), digits=4)} | "
            f"{format_value(row.get('val_ppl'), digits=2)} | {format_value(row.get('delta_val_ppl_vs_rep0'), digits=2)} | "
            f"{format_value(row.get('passkey_mean'), digits=1)}% | {format_value(row.get('stage2_frac'), digits=3)} | "
            f"{format_value(row.get('routing_ent'), digits=3)} | {format_value(row.get('peak_vram_mb'), digits=0)} MB | "
            f"{format_value(row.get('elapsed_s'), digits=0)}s |"
        )
    lines.append("")
    lines.append("Raw per-variant logs and configs are under each `rep*/` run directory. `sweep_results.json` has parsed step-level telemetry.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rep(*, args: argparse.Namespace, out_root: Path, rep_r: int, dry_run: bool) -> dict[str, Any]:
    config = build_rep_config(args=args, out_root=out_root, rep_r=rep_r)
    config_path = launcher.write_config(config)
    run_dir = Path(config["output_dir"])
    result: dict[str, Any] = {
        "rep_r": int(rep_r),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "dry_run": bool(dry_run),
        "returncode": None,
        "metrics": {},
        "health": {"pass": bool(dry_run), "errors": [] if dry_run else ["not executed"]},
    }
    if dry_run:
        write_json(run_dir / "run_result.json", result)
        return result

    print(f"[rep{rep_r}] starting run_dir={run_dir}", flush=True)
    exec_report = launcher.execute_config(config)
    result.update(exec_report)
    parsed = parser_mod.parse_run_dir(
        run_dir,
        expected_steps=None if args.bench_only else int(args.max_acc_steps),
        expected_gpu=args.expected_gpu,
        require_dsqg_w=bool(args.dsqg_w),
        expected_stage2_rep_r=int(rep_r),
        returncode=int(exec_report["returncode"]),
    )
    result["metrics"] = parsed["metrics"]
    result["health"] = parsed["health"]
    row = compact_row(result)
    print(
        f"[rep{rep_r}] {'PASS' if row['health_pass'] else 'FAIL'} rc={row['returncode']} "
        f"step={row['final_step']}/{row['planned_steps']} ce={row['final_ce']} "
        f"ppl={row['val_ppl']} tok_s={row['avg_logged_tok_s']} "
        f"stage2_frac={row['stage2_frac']} vram={row['peak_vram_mb']}",
        flush=True,
    )
    write_json(run_dir / "run_result.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run a HISA Stage-2 query-representative rep_r sweep with real trainer telemetry")
    parser.add_argument("--out-root", type=Path, default=ROOT / f"results/hisa_stage2_rep_r_sweep_{stamp}")
    parser.add_argument("--rep-rs", default="0,1,2,4,8", help="Comma-separated rep_r grid; 0 is rowmax reference.")
    parser.add_argument("--gpu", default="0", help="PyTorch CUDA_VISIBLE_DEVICES value; on this workstation 0 maps to RTX 4090.")
    parser.add_argument("--expected-gpu", default="RTX 4090")
    parser.add_argument("--dataset", type=Path, default=launcher.DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=launcher.DEFAULT_TOKENIZER)
    parser.add_argument("--python", type=Path, default=Path("/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python"))
    parser.add_argument("--max-acc-steps", type=int, default=200)
    parser.add_argument("--train-seqs", type=int, default=256)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--passkey-trials", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dsqg-w", action="store_true", help="Enable DSQG-W; default is D-only to isolate HISA Stage-2.")
    parser.add_argument("--sites", default="2,6,final")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--gate-init", type=float, default=-2.0)
    parser.add_argument("--fuse-init-std", type=float, default=0.02)
    parser.add_argument("--sourcewise", action="store_true")
    parser.add_argument("--triton-sourcewise", action="store_true")
    parser.add_argument("--typed-mixer", action="store_true")
    parser.add_argument("--query-type-bias", action="store_true")
    parser.add_argument("--typed-hisa-reps", action="store_true")
    parser.add_argument("--no-dsr-candidates", action="store_true")
    parser.add_argument("--local-offsets", default="none")
    parser.add_argument("--long-offsets", default="none")
    parser.add_argument("--no-hisa-telemetry", dest="hisa_telemetry", action="store_false")
    parser.add_argument("--no-disable-bnb", dest="disable_bnb", action="store_false")
    parser.add_argument("--no-liger", dest="liger", action="store_false")
    parser.add_argument("--pin-dataset", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(hisa_telemetry=True, disable_bnb=True, liger=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    reps = parse_rep_rs(args.rep_rs)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        "objective": "hisa_stage2_rep_r_sweep",
        "git_commit": _git_commit(),
        "root": str(ROOT),
        "out_root": str(out_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "rep_rs": reps,
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
    }
    write_json(out_root / "sweep_manifest.json", manifest)

    results: list[dict[str, Any]] = []
    for rep_r in reps:
        result = run_rep(args=args, out_root=out_root, rep_r=rep_r, dry_run=bool(args.dry_run))
        results.append(result)
        if args.stop_on_fail and not result.get("health", {}).get("pass"):
            print(f"[sweep] stopping after failed rep{rep_r}", flush=True)
            break

    rows = [compact_row(result) for result in results]
    summary = summarize_rows(rows)
    final = {
        "manifest": manifest,
        "results": results,
        "summary": summary,
        "elapsed_s": time.time() - started,
        "pass": bool(summary.get("pass")),
    }
    write_json(out_root / "sweep_results.json", final)
    write_summary(out_root / "summary.md", manifest=manifest, summary=summary)
    print(json.dumps(_jsonable({"out_root": out_root, "pass": final["pass"], "summary_path": out_root / "summary.md"}), indent=2, sort_keys=True))
    return final


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["pass"] else 2)
