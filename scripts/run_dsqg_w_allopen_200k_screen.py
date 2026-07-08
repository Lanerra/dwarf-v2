#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/dlewis3/Desktop/AI/DWARF-v2")
PY = Path(os.environ.get("PY", "/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python"))
TRAIN_GPU = os.environ.get("TRAIN_GPU", "0")  # PyTorch cuda:0 is RTX 4090 on this workstation.
EVAL_GPU = os.environ.get("EVAL_GPU", "1")    # PyTorch cuda:0 inside eval proc is RTX 3090.
STAMP = os.environ.get("STAMP") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(os.environ.get("RUN_ROOT", ROOT / "runs" / f"dsqg_w_allopen_200k_{STAMP}"))
VARIANT = os.environ.get("VARIANT", "allopen_200k")
DATASET = Path(os.environ.get("DATASET", ROOT / "datasets" / "dwarf_base_v1_olmo1tok_2048_2b.pt"))
TRAIN_SEQS = int(os.environ.get("TRAIN_SEQS", "200000"))
VAL_SEQS = int(os.environ.get("VAL_SEQS", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "100"))
PASSKEY_TRIALS = int(os.environ.get("PASSKEY_TRIALS", "10"))
WIDTH_AUX_WEIGHT = float(os.environ.get("WIDTH_AUX_WEIGHT", "0.001"))
GATE_INIT = float(os.environ.get("GATE_INIT", "-1.5"))
GATE_LR_MULT = float(os.environ.get("GATE_LR_MULT", "3.0"))
SEMANTIC_SUITE = os.environ.get("SEMANTIC_SUITE", "builtin_v3_deconfounded")

STEP_RE = re.compile(r"\[ep(?P<epoch>\d+) step (?P<step>\d+)/(?:\d+)\] ce=(?P<ce>[0-9.]+).*?(?P<tok_s>\d+) tok/s(?P<tail>.*)$")
PPL_RE = re.compile(r"Ep (?P<epoch>\d+)/(?:\d+) \| Val PPL (?P<ppl>[0-9.]+)")
PASSKEY_RE = re.compile(r"Passkey mean=(?P<mean>[0-9.]+)%")
PEAK_RE = re.compile(r"peak_vram=(?P<mb>\d+)MB")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, label: str) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log(f"START {label}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, check=False).returncode
    log(f"DONE {label}: rc={rc} elapsed_s={time.time() - t0:.1f}")
    return int(rc)


def steps_for(train_seqs: int) -> int:
    return int(math.ceil(float(train_seqs) / float(BATCH_SIZE * GRAD_ACCUM)))


def parse_train_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "stdout_path": str(path),
        "stdout_exists": path.exists(),
        "last_step": None,
        "val_ppl": None,
        "passkey_mean": None,
        "peak_vram_mb": None,
        "traceback_absent": True,
        "runtime_error_absent": True,
        "oom_absent": True,
        "gpu_line": None,
        "dataset_line": None,
        "tok_s_logs": [],
    }
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Traceback" in line:
            metrics["traceback_absent"] = False
        if "RuntimeError" in line:
            metrics["runtime_error_absent"] = False
        if "CUDA out of memory" in line:
            metrics["oom_absent"] = False
        if line.startswith("  GPU:"):
            metrics["gpu_line"] = line.strip()
        if line.strip().startswith("train:"):
            metrics["dataset_line"] = line.strip()
        m = STEP_RE.search(line)
        if m:
            step = {
                "epoch": int(m.group("epoch")),
                "step": int(m.group("step")),
                "ce": float(m.group("ce")),
                "tok_s": int(m.group("tok_s")),
                "tail": m.group("tail").strip(),
                "line": line.strip(),
            }
            metrics["last_step"] = step
            metrics["tok_s_logs"].append(step["tok_s"])
        p = PPL_RE.search(line)
        if p:
            metrics["val_ppl"] = float(p.group("ppl"))
        pk = PASSKEY_RE.search(line)
        if pk:
            metrics["passkey_mean"] = float(pk.group("mean"))
        pv = PEAK_RE.search(line)
        if pv:
            metrics["peak_vram_mb"] = int(pv.group("mb"))
    vals = metrics.get("tok_s_logs") or []
    if vals:
        metrics["mean_last4_tok_s"] = sum(vals[-4:]) / min(4, len(vals))
    return metrics


def train() -> dict[str, Any]:
    out_dir = RUN_ROOT / "pretrain" / VARIANT
    cmd = [
        str(PY),
        "scripts/run_dsqg_w_full_training.py",
        "--run-name", VARIANT,
        "--output-dir", str(out_dir),
        "--gpu", TRAIN_GPU,
        "--dataset", str(DATASET),
        "--seq-len", "2048",
        "--max-acc-steps", str(steps_for(TRAIN_SEQS)),
        "--train-seqs", str(TRAIN_SEQS),
        "--val-seqs", str(VAL_SEQS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM),
        "--epochs", "1",
        "--log-interval", str(LOG_INTERVAL),
        "--passkey-trials", str(PASSKEY_TRIALS),
        "--sites", "final",
        "--max-candidates", "16",
        "--bottleneck", "64",
        "--gate-init", str(GATE_INIT),
        "--gate-lr-mult", str(GATE_LR_MULT),
        "--fuse-init-std", "0.02",
        "--sourcewise",
        "--triton-sourcewise",
        "--width-cell",
        "--width-bottleneck", "64",
        "--width-gate-init", str(GATE_INIT),
        "--width-aux-weight", str(WIDTH_AUX_WEIGHT),
        "--width-entropy-floor", "1.5",
        "--width-entropy-weight", "0.25",
        "--typed-mixer",
        "--typed-mixer-bottleneck", "64",
        "--typed-mixer-gate-init", str(GATE_INIT),
        "--evidence-prior",
        "--evidence-prior-clip", "2.0",
        "--evidence-prior-init-scale", "0.0",
        "--candidate-quotas",
        "--quota-hisa-max", "4",
        "--query-type-bias",
        "--typed-hisa-reps",
        "--hisa-stage2-rep-r", "4",
        "--k-question", "4",
        "--k-hisa-evidence", "4",
        "--k-l3-skip", "2",
        "--execute",
    ]
    rc = run_cmd(
        cmd,
        env={"PYTHONUNBUFFERED": "1", "DWARF_SE_MAX_ABORT": "4.0"},
        label=f"train {VARIANT}",
    )
    metrics = parse_train_log(out_dir / "trainer.stdout.log")
    metrics.update({
        "returncode": rc,
        "variant": VARIANT,
        "train_seqs": TRAIN_SEQS,
        "out_dir": str(out_dir),
        "dataset": str(DATASET),
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_acc_steps": steps_for(TRAIN_SEQS),
        "width_aux_weight": WIDTH_AUX_WEIGHT,
        "gate_init": GATE_INIT,
        "gate_lr_mult": GATE_LR_MULT,
    })
    write_json(RUN_ROOT / "metrics" / f"train_{VARIANT}.json", metrics)
    return metrics


def latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_evidence() -> dict[str, Any]:
    rc = run_cmd(
        [
            str(PY),
            "scripts/run_dsqg_w_evidence_prior_telemetry.py",
            "--run-root", str(RUN_ROOT),
            "--variant", VARIANT,
            "--batch-size", "2",
            "--max-batches", "4",
            "--prior-check-batches", "1",
            "--paths", "triton",
        ],
        env={"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"},
        label=f"evidence {VARIANT}",
    )
    latest = latest_file(RUN_ROOT / "evidence_prior_telemetry", "*/evidence_prior_telemetry.json")
    summary: dict[str, Any] = {"returncode": rc, "telemetry_path": str(latest) if latest else None}
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        summary.update({
            "device": payload.get("device"),
            "quota_decision": payload.get("quota_decision"),
            "evidence_mean": payload.get("evidence_baseline", {}).get("telemetry_mean"),
        })
    write_json(RUN_ROOT / "metrics" / f"evidence_{VARIANT}.json", summary)
    return summary


def run_semantic() -> dict[str, Any]:
    rc = run_cmd(
        [
            str(PY),
            "scripts/run_dsqg_w_run_dir_semantic_eval.py",
            "--run-root", str(RUN_ROOT),
            "--variant-ids", VARIANT,
            "--semantic-suite", SEMANTIC_SUITE,
        ],
        env={"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"},
        label=f"semantic {VARIANT}",
    )
    latest = latest_file(RUN_ROOT / "semantic_transfer", "combined_semantic_transfer_*.json")
    summary: dict[str, Any] = {"returncode": rc, "combined_path": str(latest) if latest else None}
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        rows = [r for r in payload.get("results", []) if r.get("variant_id") == VARIANT]
        if rows:
            summary.update({
                "device": payload.get("device"),
                "checkpoint": rows[0].get("checkpoint"),
                "overall": rows[0].get("semantic_transfer", {}).get("overall"),
            })
    write_json(RUN_ROOT / "metrics" / f"semantic_{VARIANT}.json", summary)
    return summary


def run_external_trio() -> dict[str, Any]:
    rc = run_cmd(
        [
            str(PY),
            "scripts/run_dsqg_w_run_dir_external_trio.py",
            "--run-root", str(RUN_ROOT),
            "--variants", VARIANT,
        ],
        env={"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"},
        label=f"external_trio {VARIANT}",
    )
    latest = latest_file(RUN_ROOT / "external_trio", "combined_external_trio_*.json")
    summary: dict[str, Any] = {"returncode": rc, "combined_path": str(latest) if latest else None}
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        rows = [r for r in payload.get("results", []) if r.get("variant_id") == VARIANT]
        if rows:
            summary.update({
                "device": payload.get("device"),
                "checkpoint": rows[0].get("payload", {}).get("checkpoint"),
                "results": rows[0].get("payload", {}).get("results"),
                "external_trio_json": rows[0].get("external_trio_json"),
            })
    write_json(RUN_ROOT / "metrics" / f"external_trio_{VARIANT}.json", summary)
    return summary


def verify(train_metrics: dict[str, Any], evidence: dict[str, Any], semantic: dict[str, Any], external: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if train_metrics.get("returncode") != 0:
        errors.append(f"train rc={train_metrics.get('returncode')}")
    for flag in ("traceback_absent", "runtime_error_absent", "oom_absent"):
        if not train_metrics.get(flag):
            errors.append(f"trainer {flag}=false")
    if "RTX 4090" not in str(train_metrics.get("gpu_line")):
        errors.append(f"trainer GPU mismatch: {train_metrics.get('gpu_line')}")
    if train_metrics.get("last_step", {}).get("step") != steps_for(TRAIN_SEQS):
        errors.append(f"final step mismatch: {train_metrics.get('last_step')}")
    for name, payload in (("evidence", evidence), ("semantic", semantic), ("external_trio", external)):
        if payload.get("returncode") != 0:
            errors.append(f"{name} rc={payload.get('returncode')}")
        if payload.get("device") and payload.get("device") != "NVIDIA GeForce RTX 3090":
            errors.append(f"{name} device={payload.get('device')}")
    return {"pass": not errors, "errors": errors}


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_ROOT / "allopen_200k_config.json", {
        "run_root": RUN_ROOT,
        "variant": VARIANT,
        "mode": "scratch_200k_allopen_width_typed_prior_querybias_quotas_open_gates",
        "train_gpu": TRAIN_GPU,
        "eval_gpu": EVAL_GPU,
        "dataset": DATASET,
        "train_seqs": TRAIN_SEQS,
        "val_seqs": VAL_SEQS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_acc_steps": steps_for(TRAIN_SEQS),
        "width_aux_weight": WIDTH_AUX_WEIGHT,
        "gate_init": GATE_INIT,
        "gate_lr_mult": GATE_LR_MULT,
        "python": PY,
    })
    log(f"RUN_ROOT={RUN_ROOT}")
    t = train()
    if t.get("returncode") != 0:
        write_json(RUN_ROOT / "metrics" / "allopen_200k_verification.json", {"pass": False, "errors": ["train failed"], "train": t})
        return 2
    e = run_evidence()
    s = run_semantic()
    x = run_external_trio()
    v = verify(t, e, s, x)
    write_json(RUN_ROOT / "metrics" / "stage_allopen_final.json", {"train": t, "evidence": e, "semantic": s, "external_trio": x, "verification": v})
    write_json(RUN_ROOT / "metrics" / "allopen_200k_verification.json", v)
    log(f"all-open verification: {json.dumps(v, sort_keys=True)}")
    log("all-open 200K screen complete")
    return 0 if v["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
