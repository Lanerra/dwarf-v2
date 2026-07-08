#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/dlewis3/Desktop/AI/DWARF-v2")
PY = Path(os.environ.get("PY", "/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python"))
EVAL_GPU = os.environ.get("EVAL_GPU", "1")
STAMP = os.environ.get("STAMP") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(os.environ.get("RUN_ROOT", ROOT / "runs" / f"dsqg_w_100k_evaltime_attribution_{STAMP}"))
SOURCE_RUN_CONFIG = Path(os.environ.get(
    "SOURCE_RUN_CONFIG",
    ROOT / "runs/dsqg_w_prior_ladder_20260704_183100/pretrain/w_prior_noquota_100k/run_config.json",
))
SEMANTIC_SUITE = os.environ.get("SEMANTIC_SUITE", "builtin_v3_deconfounded")

VARIANTS = {
    "eval_typed_prior": {
        "description": "Original trained checkpoint architecture at eval: typed mixer on, scalar prior on.",
        "DWARF_DSQG_W_TYPED_MIXER": "1",
        "DWARF_DSQG_W_EVIDENCE_PRIOR": "1",
    },
    "eval_typed_no_prior": {
        "description": "Same trained checkpoint, scalar prior disabled at eval; tests direct prior contribution.",
        "DWARF_DSQG_W_TYPED_MIXER": "1",
        "DWARF_DSQG_W_EVIDENCE_PRIOR": "0",
    },
    "eval_prior_no_typed": {
        "description": "Same trained checkpoint, typed mixer disabled at eval while scalar prior remains; approximate prior-only forward ablation, not retrained.",
        "DWARF_DSQG_W_TYPED_MIXER": "0",
        "DWARF_DSQG_W_EVIDENCE_PRIOR": "1",
    },
    "eval_no_typed_no_prior": {
        "description": "Same trained checkpoint, typed mixer and scalar prior disabled at eval; approximate W-minus-helpers forward ablation, not retrained.",
        "DWARF_DSQG_W_TYPED_MIXER": "0",
        "DWARF_DSQG_W_EVIDENCE_PRIOR": "0",
    },
}


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], *, label: str) -> int:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"})
    log(f"START {label}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, check=False).returncode
    log(f"DONE {label}: rc={rc} elapsed_s={time.time() - t0:.1f}")
    return int(rc)


def prepare_configs() -> list[str]:
    cfg = json.loads(SOURCE_RUN_CONFIG.read_text(encoding="utf-8"))
    source_env = cfg.get("env", {})
    source_checkpoint_dir = Path(source_env["DWARF_CHECKPOINT_DIR"])
    source_base = source_env["DWARF_CKPT_BASE_NAME"]
    assert (source_checkpoint_dir / f"{source_base}_best.pt").exists(), source_checkpoint_dir
    variant_ids: list[str] = []
    for variant, overrides in VARIANTS.items():
        out_dir = RUN_ROOT / "pretrain" / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        new_cfg = json.loads(json.dumps(cfg))
        new_cfg["run_name"] = variant
        new_cfg["output_dir"] = str(out_dir)
        new_cfg["config_path"] = str(out_dir / "run_config.json")
        new_cfg["stdout_path"] = str(out_dir / "trainer.stdout.log")
        new_cfg["stderr_path"] = str(out_dir / "trainer.stderr.log")
        new_cfg["attribution_note"] = overrides["description"]
        env = new_cfg["env"]
        env.update({k: v for k, v in overrides.items() if k.startswith("DWARF_")})
        # Keep checkpoint pointers on the trained 100K checkpoint. These are eval-time
        # architecture toggles against one checkpoint, not retrained variants.
        env["DWARF_CHECKPOINT_DIR"] = str(source_checkpoint_dir)
        env["DWARF_CKPT_BASE_NAME"] = source_base
        env["DWARF_DSQG_W_CANDIDATE_QUOTAS"] = "0"
        env["DWARF_DSQG_W_QUOTA_HISA_MAX"] = "0"
        (out_dir / "run_config.json").write_text(json.dumps(new_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        variant_ids.append(variant)
    write_json(RUN_ROOT / "attribution_config.json", {
        "run_root": str(RUN_ROOT),
        "source_run_config": str(SOURCE_RUN_CONFIG),
        "source_checkpoint_dir": str(source_checkpoint_dir),
        "source_base": source_base,
        "eval_gpu": EVAL_GPU,
        "variant_order": variant_ids,
        "variants": VARIANTS,
        "important_caveat": "These are eval-time toggles on one trained typed+prior checkpoint. Missing/unexpected checkpoint keys are expected when typed mixer is disabled; this is not a substitute for retrained attribution.",
    })
    return variant_ids


def latest(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def summarize(variant_ids: list[str], semantic_rc: int, external_rc: int) -> dict[str, Any]:
    out: dict[str, Any] = {"semantic_returncode": semantic_rc, "external_returncode": external_rc, "variants": {}}
    sem_path = latest(RUN_ROOT / "semantic_transfer", "combined_semantic_transfer_*.json")
    ext_path = latest(RUN_ROOT / "external_trio", "combined_external_trio_*.json")
    out["semantic_combined"] = str(sem_path) if sem_path else None
    out["external_combined"] = str(ext_path) if ext_path else None
    sem_payload = json.loads(sem_path.read_text()) if sem_path else {"results": []}
    ext_payload = json.loads(ext_path.read_text()) if ext_path else {"results": []}
    for vid in variant_ids:
        row: dict[str, Any] = {"description": VARIANTS[vid]["description"]}
        sem_rows = [r for r in sem_payload.get("results", []) if r.get("variant_id") == vid]
        if sem_rows:
            row["semantic"] = sem_rows[0].get("semantic_transfer", {}).get("overall")
            row["semantic_json"] = sem_rows[0].get("outputs", {}).get("semantic_transfer_json")
        ext_rows = [r for r in ext_payload.get("results", []) if r.get("variant_id") == vid]
        if ext_rows:
            payload = ext_rows[0].get("payload", {})
            row["external"] = payload.get("results")
            row["external_json"] = ext_rows[0].get("external_trio_json")
        out["variants"][vid] = row
    write_json(RUN_ROOT / "attribution_summary.json", out)
    return out


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    variant_ids = prepare_configs()
    sem_rc = run_cmd([
        str(PY),
        "scripts/run_dsqg_w_run_dir_semantic_eval.py",
        "--run-root", str(RUN_ROOT),
        "--variant-ids", *variant_ids,
        "--semantic-suite", SEMANTIC_SUITE,
    ], label="semantic eval-time attribution")
    ext_rc = run_cmd([
        str(PY),
        "scripts/run_dsqg_w_run_dir_external_trio.py",
        "--run-root", str(RUN_ROOT),
        "--variants", *variant_ids,
    ], label="external trio eval-time attribution")
    summary = summarize(variant_ids, sem_rc, ext_rc)
    log(f"wrote {RUN_ROOT / 'attribution_summary.json'}")
    return 0 if sem_rc == 0 and ext_rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
