#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

STEP_RE = re.compile(
    r"\[ep(?P<epoch>\d+) step (?P<step>\d+)/(?P<total>\d+)\]\s+ce=(?P<ce>[-+0-9.eE]+).*?(?P<toks>[-+0-9.eE]+) tok/s(?P<tail>.*)$"
)
VAL_PPL_RE = re.compile(r"Ep\s+\d+/\d+\s+\|\s+Val PPL\s+(?P<ppl>[-+0-9.eE]+)")
GPU_RE = re.compile(r"GPU:\s*(?P<gpu>.+)$")
STAGE2_RE = re.compile(r"HISA Stage-2 selector:\s*rep_r=(?P<rep>[-+0-9]+)")
SUMMARY_RE = re.compile(r"peak_vram=(?P<vram>\d+)MB\s+elapsed=(?P<elapsed>\d+)s")
PASSKEY_RE = re.compile(r"Passkey mean=(?P<passkey>[-+0-9.eE]+)%")
GIT_RE = re.compile(r"git=(?P<git>[0-9a-f]+)")
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[-+0-9.eE]+)")
HEALTH_ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "out of memory",
    "non-finite",
    "nonfinite",
    "NaN",
    "nan loss",
    "inf loss",
)


def _maybe_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_step_tail(tail: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for match in KV_RE.finditer(tail):
        value = _maybe_float(match.group("value"))
        if value is not None:
            parsed[match.group("key")] = value
    return parsed


def parse_trainer_text(text: str, *, stdout_path: str | Path | None = None) -> dict[str, Any]:
    lines = text.splitlines()
    steps: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "stdout_path": str(stdout_path) if stdout_path is not None else None,
        "gpu": None,
        "git": None,
        "loss_mask_line": None,
        "dsqg_w_banner": None,
        "hisa_stage2_rep_r": None,
        "val_ppl": None,
        "passkey_mean": None,
        "peak_vram_mb": None,
        "elapsed_s": None,
        "steps": steps,
        "health_errors": [],
    }
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if "GPU:" in stripped:
            m = GPU_RE.search(stripped)
            if m:
                metrics["gpu"] = m.group("gpu").strip()
        if stripped.startswith("git="):
            m = GIT_RE.search(stripped)
            if m:
                metrics["git"] = m.group("git")
        if "train:" in stripped and "val:" in stripped and "train_real=" in stripped:
            metrics["loss_mask_line"] = stripped
        if "DSQG-W recomposer" in stripped:
            metrics["dsqg_w_banner"] = stripped
        if "HISA Stage-2 selector" in stripped:
            m = STAGE2_RE.search(stripped)
            if m:
                metrics["hisa_stage2_rep_r"] = int(m.group("rep"))
        m = STEP_RE.search(stripped)
        if m:
            step = {
                "line": line_no,
                "epoch": int(m.group("epoch")),
                "step": int(m.group("step")),
                "total_steps": int(m.group("total")),
                "ce": float(m.group("ce")),
                "tok_s": float(m.group("toks")),
            }
            step.update(_parse_step_tail(m.group("tail")))
            steps.append(step)
        m = VAL_PPL_RE.search(stripped)
        if m:
            metrics["val_ppl"] = _maybe_float(m.group("ppl"))
        m = PASSKEY_RE.search(stripped)
        if m:
            metrics["passkey_mean"] = _maybe_float(m.group("passkey"))
        m = SUMMARY_RE.search(stripped)
        if m:
            metrics["peak_vram_mb"] = int(m.group("vram"))
            metrics["elapsed_s"] = int(m.group("elapsed"))
        for pattern in HEALTH_ERROR_PATTERNS:
            if pattern in stripped:
                if pattern in {"non-finite", "nonfinite"} and "skip_nonfinite" in stripped:
                    continue
                metrics["health_errors"].append({"line": line_no, "pattern": pattern, "text": stripped[:240]})
                break
    if steps:
        last = steps[-1]
        metrics["final_step"] = last["step"]
        metrics["planned_steps"] = last["total_steps"]
        metrics["final_ce"] = last["ce"]
        metrics["avg_logged_tok_s"] = sum(float(s["tok_s"]) for s in steps) / len(steps)
        for key in (
            "w_gate",
            "w_gate_logit",
            "w_dx",
            "w_hisa",
            "w_score",
            "w_smean",
            "w_mix_gate",
            "w_mix_gate_logit",
            "w_width_gate",
            "w_width_gate_logit",
            "w_width_delta",
            "w_width_ent",
            "w_width_self",
            "w_width_qh",
            "w_width_hq",
            "w_width_xfer",
            "w_width_ep",
            "w_rel_diff",
            "w_rel_prod",
            "w_width_score_gn",
            "w_width_v_gn",
            "w_width_up_gn",
            "w_width_gate_gn",
            "w_mix_gate_gn",
            "w_all_gate_gn",
            "w_cache",
            "w_fast",
            "w_fast_bypass",
            "w_trainable",
            "w_mat",
            "w_sem_bypass",
            "w_det",
            "w_j",
            "routing_ent",
        ):
            if key in last:
                metrics[key] = last[key]
    else:
        metrics["final_step"] = None
        metrics["planned_steps"] = None
        metrics["final_ce"] = None
        metrics["avg_logged_tok_s"] = None
    return metrics


def parse_trainer_stdout(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return parse_trainer_text(p.read_text(encoding="utf-8", errors="replace"), stdout_path=p)


def health_check(
    metrics: dict[str, Any],
    *,
    returncode: int | None = None,
    expected_steps: int | None = None,
    expected_gpu: str | None = None,
    require_dsqg_w: bool | None = None,
    expected_stage2_rep_r: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if returncode is not None and int(returncode) != 0:
        errors.append(f"returncode={returncode}")
    if metrics.get("health_errors"):
        errors.append("trainer log contains health error patterns")
    final_step = metrics.get("final_step")
    planned_steps = metrics.get("planned_steps")
    if expected_steps is not None:
        if final_step != expected_steps or planned_steps != expected_steps:
            errors.append(f"final step {final_step}/{planned_steps} != expected {expected_steps}/{expected_steps}")
    elif final_step is None:
        errors.append("no optimizer step lines parsed")
    for key in ("final_ce", "val_ppl"):
        value = metrics.get(key)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{key} is not finite: {value}")
    if expected_gpu and expected_gpu not in str(metrics.get("gpu") or ""):
        errors.append(f"GPU mismatch: expected contains {expected_gpu!r}, saw {metrics.get('gpu')!r}")
    if expected_stage2_rep_r is not None and metrics.get("hisa_stage2_rep_r") != expected_stage2_rep_r:
        errors.append(
            f"HISA rep_r mismatch: expected {expected_stage2_rep_r}, saw {metrics.get('hisa_stage2_rep_r')}"
        )
    banner = str(metrics.get("dsqg_w_banner") or "")
    if require_dsqg_w is True and "enabled" not in banner:
        errors.append(f"DSQG-W expected enabled, banner={banner!r}")
    if require_dsqg_w is False and "disabled" not in banner:
        errors.append(f"DSQG-W expected disabled, banner={banner!r}")
    return {"pass": not errors, "errors": errors}


def parse_run_dir(
    run_dir: str | Path,
    *,
    expected_steps: int | None = None,
    expected_gpu: str | None = None,
    require_dsqg_w: bool | None = None,
    expected_stage2_rep_r: int | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    stdout = run / "trainer.stdout.log"
    metrics = parse_trainer_stdout(stdout)
    health = health_check(
        metrics,
        returncode=returncode,
        expected_steps=expected_steps,
        expected_gpu=expected_gpu,
        require_dsqg_w=require_dsqg_w,
        expected_stage2_rep_r=expected_stage2_rep_r,
    )
    return {"run_dir": str(run), "metrics": metrics, "health": health}


def summarize_lane(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    baseline_metrics = results[0].get("metrics", {}) if results else {}
    base_ppl = baseline_metrics.get("val_ppl")
    base_ce = baseline_metrics.get("final_ce")
    for result in results:
        metrics = result.get("metrics", {})
        row = {
            "variant_id": result.get("variant_id"),
            "label": result.get("label"),
            "run_dir": result.get("run_dir"),
            "health_pass": result.get("health", {}).get("pass"),
            "health_errors": result.get("health", {}).get("errors", []),
            "returncode": result.get("returncode"),
            "final_step": metrics.get("final_step"),
            "final_ce": metrics.get("final_ce"),
            "val_ppl": metrics.get("val_ppl"),
            "avg_logged_tok_s": metrics.get("avg_logged_tok_s"),
            "peak_vram_mb": metrics.get("peak_vram_mb"),
            "w_dx": metrics.get("w_dx"),
            "w_hisa": metrics.get("w_hisa"),
            "w_gate": metrics.get("w_gate"),
            "w_gate_logit": metrics.get("w_gate_logit"),
            "w_mix_gate": metrics.get("w_mix_gate"),
            "w_mix_gate_logit": metrics.get("w_mix_gate_logit"),
            "w_width_gate": metrics.get("w_width_gate"),
            "w_width_gate_logit": metrics.get("w_width_gate_logit"),
            "w_width_delta": metrics.get("w_width_delta"),
            "w_width_ent": metrics.get("w_width_ent"),
            "w_width_self": metrics.get("w_width_self"),
            "w_width_qh": metrics.get("w_width_qh"),
            "w_width_hq": metrics.get("w_width_hq"),
            "w_width_xfer": metrics.get("w_width_xfer"),
            "w_width_ep": metrics.get("w_width_ep"),
            "w_rel_diff": metrics.get("w_rel_diff"),
            "w_rel_prod": metrics.get("w_rel_prod"),
            "w_width_score_gn": metrics.get("w_width_score_gn"),
            "w_width_v_gn": metrics.get("w_width_v_gn"),
            "w_width_up_gn": metrics.get("w_width_up_gn"),
            "w_width_gate_gn": metrics.get("w_width_gate_gn"),
            "w_mix_gate_gn": metrics.get("w_mix_gate_gn"),
            "w_all_gate_gn": metrics.get("w_all_gate_gn"),
            "w_fast": metrics.get("w_fast"),
            "w_fast_bypass": metrics.get("w_fast_bypass"),
            "w_trainable": metrics.get("w_trainable"),
            "w_mat": metrics.get("w_mat"),
            "w_sem_bypass": metrics.get("w_sem_bypass"),
            "w_det": metrics.get("w_det"),
            "w_j": metrics.get("w_j"),
            "hisa_stage2_rep_r": metrics.get("hisa_stage2_rep_r"),
        }
        if isinstance(base_ppl, (int, float)) and isinstance(row["val_ppl"], (int, float)):
            row["delta_val_ppl_vs_a"] = row["val_ppl"] - base_ppl
        if isinstance(base_ce, (int, float)) and isinstance(row["final_ce"], (int, float)):
            row["delta_final_ce_vs_a"] = row["final_ce"] - base_ce
        rows.append(row)
    return {"pass": all(bool(r.get("health_pass")) for r in rows), "rows": rows}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Parse DSQG-W/HISA ladder trainer logs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--expected-gpu", default=None)
    parser.add_argument("--expected-stage2-rep-r", type=int, default=None)
    parser.add_argument("--returncode", type=int, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require-dsqg-w", action="store_true")
    group.add_argument("--forbid-dsqg-w", action="store_true")
    args = parser.parse_args(argv)
    require_dsqg_w = True if args.require_dsqg_w else False if args.forbid_dsqg_w else None
    result = parse_run_dir(
        args.run_dir,
        expected_steps=args.expected_steps,
        expected_gpu=args.expected_gpu,
        require_dsqg_w=require_dsqg_w,
        expected_stage2_rep_r=args.expected_stage2_rep_r,
        returncode=args.returncode,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parsed = main()
    raise SystemExit(0 if parsed["health"]["pass"] else 2)
