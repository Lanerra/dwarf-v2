#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/dlewis3/Desktop/AI/DWARF-v2")
PY = Path(os.environ.get("PY", "/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python"))
TRAIN_GPU = os.environ.get("TRAIN_GPU", "0")  # On this box, CUDA_VISIBLE_DEVICES=0 exposes the RTX 4090 to PyTorch.
EVAL_GPU = os.environ.get("EVAL_GPU", "1")    # Secondary/eval GPU, normally RTX 3090.
STAMP = os.environ.get("STAMP") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(os.environ.get("RUN_ROOT", ROOT / "runs" / f"dsqg_w_prior_ladder_{STAMP}"))
VAL_SEQS = int(os.environ.get("VAL_SEQS", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "50"))
PASSKEY_TRIALS = int(os.environ.get("PASSKEY_TRIALS", "0"))
SEMANTIC_SUITE = os.environ.get("SEMANTIC_SUITE", "builtin_v3_deconfounded")

# Promotion gate: intentionally permissive enough for a 50K early-training screen,
# but strict enough to avoid launching 100K after a clearly broken prior run.
PROMOTE_LIMITS = {
    "max_final_ce": float(os.environ.get("PROMOTE_MAX_FINAL_CE", "4.60")),
    "max_val_ppl": float(os.environ.get("PROMOTE_MAX_VAL_PPL", "95.0")),
    "min_semantic_choice_accuracy": float(os.environ.get("PROMOTE_MIN_SEM_CHOICE", "0.70")),
    "max_semantic_token_weighted_nll": float(os.environ.get("PROMOTE_MAX_SEM_TOKEN_NLL", "5.75")),
    "max_missing_question_fraction": float(os.environ.get("PROMOTE_MAX_MISSING_Q", "0.05")),
    "max_hisa_monopoly_fraction": float(os.environ.get("PROMOTE_MAX_HISA_MONOPOLY", "0.50")),
}

STEP_RE = re.compile(r"\[ep(?P<epoch>\d+) step (?P<step>\d+)/(?:\d+)\] ce=(?P<ce>[0-9.]+).*?(?P<tok_s>\d+) tok/s(?P<tail>.*)$")
PPL_RE = re.compile(r"Ep (?P<epoch>\d+)/(?:\d+) \| Val PPL (?P<ppl>[0-9.]+)")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None, label: str) -> int:
    log(f"START {label}: {' '.join(cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    t0 = time.time()
    completed = subprocess.run(cmd, cwd=cwd, env=merged_env, text=True, check=False)
    elapsed = time.time() - t0
    log(f"DONE {label}: rc={completed.returncode} elapsed_s={elapsed:.1f}")
    return int(completed.returncode)


def stage_steps(train_seqs: int) -> int:
    return int(math.ceil(float(train_seqs) / float(BATCH_SIZE * GRAD_ACCUM)))


def train_stage(train_seqs: int, variant: str) -> dict[str, Any]:
    out_dir = RUN_ROOT / "pretrain" / variant
    args = [
        str(PY),
        "scripts/run_dsqg_w_full_training.py",
        "--run-name", variant,
        "--output-dir", str(out_dir),
        "--gpu", TRAIN_GPU,
        "--max-acc-steps", str(stage_steps(train_seqs)),
        "--train-seqs", str(train_seqs),
        "--val-seqs", str(VAL_SEQS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM),
        "--epochs", "1",
        "--log-interval", str(LOG_INTERVAL),
        "--passkey-trials", str(PASSKEY_TRIALS),
        "--sites", "final",
        "--sourcewise",
        "--triton-sourcewise",
        "--typed-mixer",
        "--evidence-prior",
        "--evidence-prior-clip", "2.0",
        "--evidence-prior-init-scale", "0.0",
        "--execute",
    ]
    rc = run_cmd(args, env={"PYTHONUNBUFFERED": "1"}, label=f"train {variant}")
    metrics = parse_train_log(out_dir / "trainer.stdout.log")
    metrics.update({"variant": variant, "train_seqs": train_seqs, "returncode": rc, "out_dir": str(out_dir)})
    write_json(RUN_ROOT / "metrics" / f"train_{variant}.json", metrics)
    return metrics


def parse_train_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "stdout_path": str(path),
        "stdout_exists": path.exists(),
        "traceback_absent": True,
        "runtime_error_absent": True,
        "prior_enabled_banner": False,
        "quota_disabled_env": False,
        "gpu_line": None,
        "last_step": None,
        "val_ppl": None,
        "complete_summary": False,
    }
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Traceback" in line:
            metrics["traceback_absent"] = False
        if "RuntimeError" in line:
            metrics["runtime_error_absent"] = False
        if line.startswith("  GPU:"):
            metrics["gpu_line"] = line.strip()
        if "evidence_prior" in line.lower() or "prior" in line.lower():
            # Keep weak marker; the authoritative env marker is run_config.json.
            metrics["prior_log_mentions"] = True
        if "DSR + R_PLANES=4" in line and "Summary" in line:
            metrics["complete_summary"] = True
        m = STEP_RE.search(line)
        if m:
            metrics["last_step"] = {
                "epoch": int(m.group("epoch")),
                "step": int(m.group("step")),
                "ce": float(m.group("ce")),
                "tok_s": int(m.group("tok_s")),
                "tail": m.group("tail").strip(),
                "line": line.strip(),
            }
        p = PPL_RE.search(line)
        if p:
            metrics["val_ppl"] = float(p.group("ppl"))
    cfg_path = path.parent / "run_config.json"
    metrics["run_config_exists"] = cfg_path.exists()
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        env = cfg.get("env", {})
        metrics["prior_enabled_banner"] = env.get("DWARF_DSQG_W_EVIDENCE_PRIOR") == "1"
        metrics["quota_disabled_env"] = env.get("DWARF_DSQG_W_CANDIDATE_QUOTAS") == "0"
        metrics["config_env_subset"] = {
            k: env.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "DWARF_MAX_TRAIN_SEQS",
                "DWARF_MAX_ACC_STEPS",
                "DWARF_BS",
                "DWARF_GA",
                "DWARF_DSQG_W_EVIDENCE_PRIOR",
                "DWARF_DSQG_W_CANDIDATE_QUOTAS",
                "DWARF_DSQG_W_QUOTA_HISA_MAX",
                "DWARF_DSQG_W_K_QUESTION",
                "DWARF_DSQG_W_K_HISA_EVIDENCE",
                "DWARF_DSQG_W_K_L3_SKIP",
            ]
        }
    return metrics


def run_semantic(variant: str) -> dict[str, Any]:
    rc = run_cmd(
        [
            str(PY),
            "scripts/run_dsqg_w_run_dir_semantic_eval.py",
            "--run-root", str(RUN_ROOT),
            "--variant-ids", variant,
            "--semantic-suite", SEMANTIC_SUITE,
        ],
        env={"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."},
        label=f"semantic {variant}",
    )
    latest = latest_file(RUN_ROOT / "semantic_transfer", "combined_semantic_transfer_*.json")
    summary: dict[str, Any] = {"returncode": rc, "combined_path": str(latest) if latest else None, "overall": None}
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            if result.get("variant_id") == variant:
                summary["overall"] = result.get("semantic_transfer", {}).get("overall")
                summary["device"] = payload.get("device")
                summary["checkpoint"] = result.get("checkpoint")
                break
    write_json(RUN_ROOT / "metrics" / f"semantic_{variant}.json", summary)
    return summary


def run_evidence(variant: str) -> dict[str, Any]:
    rc = run_cmd(
        [
            str(PY),
            "scripts/run_dsqg_w_evidence_prior_telemetry.py",
            "--run-root", str(RUN_ROOT),
            "--variant", variant,
            "--batch-size", "2",
            "--max-batches", "4",
            "--prior-check-batches", "1",
            "--paths", "triton",
        ],
        env={"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."},
        label=f"evidence {variant}",
    )
    latest = latest_file(RUN_ROOT / "evidence_prior_telemetry", "*/evidence_prior_telemetry.json")
    summary: dict[str, Any] = {"returncode": rc, "telemetry_path": str(latest) if latest else None, "quota_decision": None, "evidence_mean": None}
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        summary["quota_decision"] = payload.get("quota_decision")
        summary["evidence_mean"] = payload.get("evidence_baseline", {}).get("telemetry_mean")
        summary["device"] = payload.get("device")
    write_json(RUN_ROOT / "metrics" / f"evidence_{variant}.json", summary)
    return summary


def latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def promotion_decision(train: dict[str, Any], semantic: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    ok = True

    def fail(reason: str) -> None:
        nonlocal ok
        ok = False
        reasons.append(reason)

    if train.get("returncode") != 0:
        fail(f"train returncode={train.get('returncode')}")
    if not train.get("traceback_absent"):
        fail("traceback found in trainer stdout")
    if not train.get("runtime_error_absent"):
        fail("RuntimeError found in trainer stdout")
    if not train.get("prior_enabled_banner"):
        fail("run_config does not record DWARF_DSQG_W_EVIDENCE_PRIOR=1")
    if not train.get("quota_disabled_env"):
        fail("run_config does not record candidate quotas disabled")
    if "RTX 4090" not in str(train.get("gpu_line")):
        fail(f"trainer GPU line was not RTX 4090: {train.get('gpu_line')}")

    last_step = train.get("last_step") or {}
    ce = last_step.get("ce")
    if ce is None or not math.isfinite(float(ce)):
        fail("missing/nonfinite final CE")
    elif float(ce) > PROMOTE_LIMITS["max_final_ce"]:
        fail(f"final CE {ce:.4f} > {PROMOTE_LIMITS['max_final_ce']:.4f}")

    ppl = train.get("val_ppl")
    if ppl is None or not math.isfinite(float(ppl)):
        fail("missing/nonfinite Val PPL")
    elif float(ppl) > PROMOTE_LIMITS["max_val_ppl"]:
        fail(f"Val PPL {ppl:.2f} > {PROMOTE_LIMITS['max_val_ppl']:.2f}")

    overall = semantic.get("overall") or {}
    choice = overall.get("choice_accuracy")
    token_nll = overall.get("token_weighted_target_nll")
    if semantic.get("returncode") != 0:
        fail(f"semantic returncode={semantic.get('returncode')}")
    if choice is None or not math.isfinite(float(choice)):
        fail("missing/nonfinite semantic choice_accuracy")
    elif float(choice) < PROMOTE_LIMITS["min_semantic_choice_accuracy"]:
        fail(f"semantic choice_accuracy {choice:.4f} < {PROMOTE_LIMITS['min_semantic_choice_accuracy']:.4f}")
    if token_nll is None or not math.isfinite(float(token_nll)):
        fail("missing/nonfinite semantic token-weighted NLL")
    elif float(token_nll) > PROMOTE_LIMITS["max_semantic_token_weighted_nll"]:
        fail(f"semantic token-weighted NLL {token_nll:.4f} > {PROMOTE_LIMITS['max_semantic_token_weighted_nll']:.4f}")

    if evidence.get("returncode") != 0:
        fail(f"evidence telemetry returncode={evidence.get('returncode')}")
    qd = evidence.get("quota_decision") or {}
    missing_q = qd.get("missing_question")
    monopoly = qd.get("monopoly")
    if missing_q is not None and float(missing_q) > PROMOTE_LIMITS["max_missing_question_fraction"]:
        fail(f"missing-question fraction {float(missing_q):.4f} > {PROMOTE_LIMITS['max_missing_question_fraction']:.4f}")
    if monopoly is not None and float(monopoly) > PROMOTE_LIMITS["max_hisa_monopoly_fraction"]:
        fail(f"HISA monopoly fraction {float(monopoly):.4f} > {PROMOTE_LIMITS['max_hisa_monopoly_fraction']:.4f}")

    if ok:
        reasons.append("50K cleared CE/PPL, semantic-transfer, GPU/config, and evidence-occupancy gates")
    return {"promote": ok, "reasons": reasons, "limits": PROMOTE_LIMITS}


def run_stage_with_evals(train_seqs: int, variant: str) -> dict[str, Any]:
    train = train_stage(train_seqs, variant)
    evidence = run_evidence(variant) if train.get("returncode") == 0 else {"returncode": -1, "skipped": "train failed"}
    semantic = run_semantic(variant) if train.get("returncode") == 0 else {"returncode": -1, "skipped": "train failed"}
    stage = {"train": train, "evidence": evidence, "semantic": semantic}
    write_json(RUN_ROOT / "metrics" / f"stage_{variant}.json", stage)
    return stage


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        RUN_ROOT / "ladder_config.json",
        {
            "run_root": str(RUN_ROOT),
            "train_gpu": TRAIN_GPU,
            "eval_gpu": EVAL_GPU,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "val_seqs": VAL_SEQS,
            "log_interval": LOG_INTERVAL,
            "passkey_trials": PASSKEY_TRIALS,
            "semantic_suite": SEMANTIC_SUITE,
            "promotion_limits": PROMOTE_LIMITS,
            "python": str(PY),
        },
    )
    log(f"RUN_ROOT={RUN_ROOT}")
    log("starting 50K prior/no-quota stage")
    stage_50k = run_stage_with_evals(50_000, "w_prior_noquota_50k")
    decision = promotion_decision(stage_50k["train"], stage_50k["semantic"], stage_50k["evidence"])
    write_json(RUN_ROOT / "metrics" / "promotion_50k_to_100k.json", decision)
    log(f"promotion decision: {json.dumps(decision, sort_keys=True)}")
    if not decision["promote"]:
        log("not launching 100K; 50K did not clear promotion gate")
        return 0

    log("50K looks promising; starting 100K prior/no-quota stage")
    stage_100k = run_stage_with_evals(100_000, "w_prior_noquota_100k")
    write_json(RUN_ROOT / "metrics" / "stage_100k_final.json", stage_100k)
    log("ladder complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
