#!/usr/bin/env python3
"""Run a gated 4090 parity check plus BWD16/W4 overnight FWE smoke ladder."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fwe_dedup_artifact import validate_contract
from prepare_hisa_dsqg_fwe_2b import build_dry_run_config, write_dry_run_config

PYTHON = Path("/home/dlewis3/Desktop/AI/DWARF/.venv/bin/python")
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
PARITY_SCRIPT = ROOT / "scripts" / "compare_dsqg_kernel_overlay.py"
ARTIFACT = Path(
    "/home/dlewis3/Desktop/AI/DWARF/datasets/"
    "dsqg_fineweb_edu_dedup_3ba9d605_olmo1tok50280_s2048_split20260710_ce2b_decontam_repaired.pt"
)
MANIFEST = ARTIFACT.with_suffix(".manifest.json")
DECONTAM = ARTIFACT.with_suffix(".decontam.json")
TOKENIZER = ROOT / "tokenizers" / "olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
EXPECTED_GPU = "NVIDIA GeForce RTX 4090"
BATCH_SIZE = 8
GRAD_ACCUM = 2
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM
WSD_TOTAL_STEPS = 25_000
WSD_PHASES = {"warmup_steps": 1_250, "stable_steps": 20_000, "decay_steps": 3_750}
SEED = 20260710

STEP_RE = re.compile(r"\[ep(?P<epoch>\d+) step (?P<step>\d+)/(?P<planned>\d+)\] ce=(?P<ce>[0-9.eE+-]+).*? (?P<tok_s>[0-9.]+) tok/s")
PPL_RE = re.compile(r"Ep \d+/\d+ \| Val PPL (?P<ppl>[0-9.eE+-]+)")
VRAM_RE = re.compile(r"peak_vram=(?P<vram>[0-9.]+)MB")
KERNEL_RE = re.compile(r"Kernel: .*?module=(?P<path>[^;]+);")


@dataclass(frozen=True)
class Variant:
    variant_id: str
    kernel_dir: Path
    backward_env: dict[str, str]


@dataclass(frozen=True)
class Stage:
    stage_id: str
    variant_id: str
    train_seqs: int
    train_seq_offset: int
    schedule_step_offset: int
    resume_from: str | None = None
    requires_overlay_gate: bool = False


VARIANTS = {
    "canonical": Variant("canonical", (ROOT / "kernels").resolve(), {}),
    "bwd16_w4": Variant(
        "bwd16_w4",
        (ROOT / "kernel_overlays" / "bwd_tile_tuning").resolve(),
        {
            "DWARF_DSQG_BWD_BLOCK_N": "16",
            "DWARF_DSQG_BWD_NUM_WARPS": "4",
            "DWARF_DSQG_BWD_NUM_STAGES": "2",
        },
    ),
}


def build_ladder_stages() -> tuple[Stage, ...]:
    return (
        Stage("canonical_20k", "canonical", 20_000, 0, 0),
        Stage("bwd16_w4_20k", "bwd16_w4", 20_000, 0, 0),
        Stage("canonical_a_50k", "canonical", 50_000, 0, 0),
        Stage("bwd16_w4_50k", "bwd16_w4", 50_000, 0, 0),
        Stage("canonical_b_50k", "canonical", 50_000, 0, 0),
        Stage("bwd16_w4_100k", "bwd16_w4", 50_000, 50_000, 3_125, "bwd16_w4_50k", True),
        Stage("bwd16_w4_200k", "bwd16_w4", 100_000, 100_000, 6_250, "bwd16_w4_100k", True),
        Stage("bwd16_w4_400k", "bwd16_w4", 200_000, 200_000, 12_500, "bwd16_w4_200k", True),
    )


def next_stage_allowed(*, stage_id: str, gates: dict[str, bool]) -> bool:
    if stage_id in {"bwd16_w4_100k", "bwd16_w4_200k", "bwd16_w4_400k"}:
        return bool(gates.get("overlay_50k", False))
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_variant_env(variant: Variant) -> dict[str, str]:
    env = {"DWARF_DSQG_KERNEL_DIR": str(variant.kernel_dir)}
    env.update(variant.backward_env)
    return env


def expected_steps(stage: Stage) -> int:
    return int(math.ceil(stage.train_seqs / EFFECTIVE_BATCH))


def verify_visible_gpu(*, gpu: str, expected_gpu: str) -> dict[str, Any]:
    code = """
import json
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f'expected exactly one visible CUDA device, got {torch.cuda.device_count()}')
p = torch.cuda.get_device_properties(0)
print(json.dumps({'name': p.name, 'total_memory': p.total_memory, 'visible_count': torch.cuda.device_count()}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(
        [str(PYTHON), "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if payload["name"] != expected_gpu:
        raise RuntimeError(f"GPU mapping mismatch: expected {expected_gpu!r}, got {payload['name']!r}")
    return payload


def validate_artifact_once(*, verify_sha256: bool) -> dict[str, Any]:
    contract = validate_contract(
        artifact_path=ARTIFACT,
        manifest_path=MANIFEST,
        decontam_path=DECONTAM,
        tokenizer_path=TOKENIZER,
        verify_artifact_sha256=verify_sha256,
    )
    return {
        "artifact": str(contract.artifact_path),
        "manifest": str(contract.manifest_path),
        "decontam": str(contract.decontam_path),
        "tokenizer": str(contract.tokenizer_path),
        "train_rows": contract.train_rows,
        "validation_rows": contract.validation_rows,
        "artifact_sha256_verified": contract.artifact_sha256 is not None,
    }


def build_stage_config(*, stage: Stage, out_root: Path, gpu: str) -> dict[str, Any]:
    run_dir = out_root / "stages" / stage.stage_id
    config = build_dry_run_config(
        output_dir=run_dir,
        artifact_path=ARTIFACT,
        manifest_path=MANIFEST,
        decontam_path=DECONTAM,
        tokenizer_path=TOKENIZER,
        gpu=gpu,
        train_seqs=stage.train_seqs,
        val_seqs=512,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        checkpoint_strategy="none",
        schedule_total_steps=WSD_TOTAL_STEPS,
        schedule_step_offset=stage.schedule_step_offset,
        verify_artifact_sha256=False,
    )
    variant = VARIANTS[stage.variant_id]
    env = config["env"]
    env.update(build_variant_env(variant))
    env.update(
        {
            "DWARF_TRAIN_SEQ_OFFSET": str(stage.train_seq_offset),
            "DWARF_SEED": str(SEED),
            "DWARF_CKPT_BASE_NAME": stage.stage_id,
            "PYTHONUNBUFFERED": "1",
        }
    )
    if stage.resume_from is not None:
        previous_checkpoint = out_root / "stages" / stage.resume_from / "checkpoints" / f"{stage.resume_from}_ep1.pt"
        env.update({"DWARF_RESUME": str(previous_checkpoint), "DWARF_SKIP_SCHED": "1"})
    config["variant"] = _jsonable(variant)
    config["stage"] = asdict(stage)
    config["expected"] = {
        "gpu": EXPECTED_GPU,
        "steps": expected_steps(stage),
        "kernel_module": str((variant.kernel_dir / "dsqg_attention_v20_bf16_se.py").resolve()),
        "wsd": {"total_steps": WSD_TOTAL_STEPS, **WSD_PHASES, "step_offset": stage.schedule_step_offset},
        "train_tranche": {"offset": stage.train_seq_offset, "count": stage.train_seqs, "end": stage.train_seq_offset + stage.train_seqs},
    }
    write_dry_run_config(config)
    return config


def _parse_metrics(log_path: Path, *, expected_kernel: str, expected_local_steps: int) -> tuple[dict[str, Any], list[str]]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    step_matches = list(STEP_RE.finditer(text))
    ppl_matches = list(PPL_RE.finditer(text))
    vram_matches = list(VRAM_RE.finditer(text))
    kernel_matches = list(KERNEL_RE.finditer(text))
    metrics: dict[str, Any] = {
        "final_step": int(step_matches[-1].group("step")) if step_matches else None,
        "planned_steps": int(step_matches[-1].group("planned")) if step_matches else None,
        "final_ce": float(step_matches[-1].group("ce")) if step_matches else None,
        "avg_logged_tok_s": (statistics.median(float(match.group("tok_s")) for match in step_matches[-20:]) if step_matches else None),
        "val_ppl": float(ppl_matches[-1].group("ppl")) if ppl_matches else None,
        "peak_vram_mb": float(vram_matches[-1].group("vram")) if vram_matches else None,
        "kernel_module": kernel_matches[-1].group("path").strip() if kernel_matches else None,
    }
    errors: list[str] = []
    if "Traceback" in text or "[FATAL]" in text or "OutOfMemoryError" in text:
        errors.append("trainer fatal/traceback/OOM detected")
    if metrics["final_step"] != expected_local_steps or metrics["planned_steps"] != expected_local_steps:
        errors.append(f"planned step mismatch: got {metrics['final_step']}/{metrics['planned_steps']}, expected {expected_local_steps}")
    for key in ("final_ce", "avg_logged_tok_s", "val_ppl", "peak_vram_mb"):
        value = metrics[key]
        if value is None or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite {key}")
    if metrics["kernel_module"] != expected_kernel:
        errors.append(f"kernel mismatch: got {metrics['kernel_module']!r}, expected {expected_kernel!r}")
    return metrics, errors


def run_stage(*, stage: Stage, config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    run_dir = Path(config["output_dir"])
    expected = config["expected"]
    stdout_path = run_dir / "trainer.stdout.log"
    stderr_path = run_dir / "trainer.stderr.log"
    command_path = run_dir / "command.json"
    write_json(command_path, {"command": config["command"], "env": config["env"]})
    result: dict[str, Any] = {
        "stage": asdict(stage),
        "run_dir": str(run_dir),
        "config_path": config["contract_path"],
        "command_path": str(command_path),
        "returncode": None,
        "elapsed_s": None,
        "metrics": {},
        "health": {"pass": bool(dry_run), "errors": [] if dry_run else ["not executed"]},
    }
    if dry_run:
        write_json(run_dir / "run_result.json", result)
        return result
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(config["env"])
    t0 = time.monotonic()
    print(f"[stage] START {stage.stage_id} steps={expected['steps']} run_dir={run_dir}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(config["command"], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    result["returncode"] = int(completed.returncode)
    result["elapsed_s"] = time.monotonic() - t0
    metrics, errors = _parse_metrics(
        stdout_path,
        expected_kernel=expected["kernel_module"],
        expected_local_steps=expected["steps"],
    )
    result["metrics"] = metrics
    if completed.returncode != 0:
        errors.insert(0, f"trainer returncode={completed.returncode}")
    if stage.resume_from is not None:
        resume_path = Path(config["env"]["DWARF_RESUME"])
        if not resume_path.is_file():
            errors.insert(0, f"missing resume checkpoint {resume_path}")
    checkpoint_path = run_dir / "checkpoints" / f"{stage.stage_id}_ep1.pt"
    if not checkpoint_path.is_file():
        errors.append(f"missing stage checkpoint {checkpoint_path}")
    result["checkpoint_path"] = str(checkpoint_path)
    result["health"] = {"pass": not errors, "errors": errors}
    write_json(run_dir / "run_result.json", result)
    print(
        f"[stage] {'PASS' if not errors else 'FAIL'} {stage.stage_id} rc={result['returncode']} "
        f"step={metrics.get('final_step')}/{metrics.get('planned_steps')} "
        f"ppl={metrics.get('val_ppl')} tok_s={metrics.get('avg_logged_tok_s')} "
        f"vram={metrics.get('peak_vram_mb')} elapsed_s={result['elapsed_s']:.1f}",
        flush=True,
    )
    return result


def run_parity(*, out_root: Path, gpu: str, expected_gpu: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baseline = VARIANTS["canonical"].kernel_dir / "dsqg_attention_v20_bf16_se.py"
    candidate = VARIANTS["bwd16_w4"].kernel_dir / "dsqg_attention_v20_bf16_se.py"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
    for seed in (20260710, 20260711, 20260712):
        verify_visible_gpu(gpu=gpu, expected_gpu=expected_gpu)
        output = out_root / "parity" / f"seed_{seed}.json"
        completed = subprocess.run(
            [
                str(PYTHON), str(PARITY_SCRIPT),
                "--baseline-kernel", str(baseline),
                "--candidate-kernel", str(candidate),
                "--json-out", str(output),
                "--batch-size", "1", "--seq-len", "2048", "--seed", str(seed),
                "--atol", "0.003", "--rtol", "0.02",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False, "error": "missing parity output"}
        row = {"seed": seed, "returncode": completed.returncode, "result": payload, "stdout": completed.stdout, "stderr": completed.stderr}
        write_json(out_root / "parity" / f"seed_{seed}.run.json", row)
        if completed.returncode != 0 or not payload.get("passed") or payload.get("device") != expected_gpu:
            raise RuntimeError(f"parity failed for seed={seed}: {row}")
        results.append(row)
        print(f"[parity] PASS seed={seed}", flush=True)
    return results


def evaluate_overlay_50k_gate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    controls = [results["canonical_a_50k"]["metrics"], results["canonical_b_50k"]["metrics"]]
    candidate = results["bwd16_w4_50k"]["metrics"]
    control_ppl = statistics.mean(float(row["val_ppl"]) for row in controls)
    control_tok_s = statistics.mean(float(row["avg_logged_tok_s"]) for row in controls)
    candidate_ppl = float(candidate["val_ppl"])
    candidate_tok_s = float(candidate["avg_logged_tok_s"])
    quality_pass = candidate_ppl <= control_ppl * 1.005
    no_material_throughput_regression = candidate_tok_s >= control_tok_s * 0.985
    return {
        "control_mean_val_ppl": control_ppl,
        "control_mean_tok_s": control_tok_s,
        "candidate_val_ppl": candidate_ppl,
        "candidate_tok_s": candidate_tok_s,
        "ppl_ratio": candidate_ppl / control_ppl,
        "throughput_ratio": candidate_tok_s / control_tok_s,
        "quality_pass": quality_pass,
        "no_material_throughput_regression": no_material_throughput_regression,
        "pass": quality_pass and no_material_throughput_regression,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / f"runs/bwd_tile_promotion_overnight_{stamp}")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--expected-gpu", default=EXPECTED_GPU)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-artifact-sha256", action="store_true")
    return parser.parse_args(argv)


def run_ladder(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "root": str(ROOT),
        "created_at": datetime.now().isoformat(),
        "gpu": verify_visible_gpu(gpu=args.gpu, expected_gpu=args.expected_gpu),
        "variants": VARIANTS,
        "wsd": {"total_steps": WSD_TOTAL_STEPS, **WSD_PHASES},
        "stages": build_ladder_stages(),
        "dry_run": bool(args.dry_run),
    }
    plan["artifact"] = validate_artifact_once(verify_sha256=not args.skip_artifact_sha256)
    write_json(out_root / "overnight_plan.json", plan)
    results: dict[str, dict[str, Any]] = {}
    gates: dict[str, bool] = {}
    parity: list[dict[str, Any]] = []
    status = "passed"
    failure: str | None = None
    try:
        if not args.dry_run:
            parity = run_parity(out_root=out_root, gpu=args.gpu, expected_gpu=args.expected_gpu)
        for stage in build_ladder_stages():
            if not next_stage_allowed(stage_id=stage.stage_id, gates=gates):
                status = "stopped_by_gate"
                failure = f"stage {stage.stage_id} blocked by failed overlay_50k gate"
                break
            verify_visible_gpu(gpu=args.gpu, expected_gpu=args.expected_gpu)
            config = build_stage_config(stage=stage, out_root=out_root, gpu=args.gpu)
            result = run_stage(stage=stage, config=config, dry_run=args.dry_run)
            results[stage.stage_id] = result
            if not result["health"]["pass"]:
                status = "failed"
                failure = f"stage {stage.stage_id} failed health: {result['health']['errors']}"
                break
            if stage.stage_id == "canonical_b_50k":
                gate = (
                    {"pass": True, "dry_run": True}
                    if args.dry_run
                    else evaluate_overlay_50k_gate(results)
                )
                gates["overlay_50k"] = bool(gate["pass"])
                write_json(out_root / "overlay_50k_gate.json", gate)
                if args.dry_run:
                    print("[gate] overlay_50k dry-run contract prevalidated", flush=True)
                else:
                    print(f"[gate] overlay_50k pass={gate['pass']} throughput_ratio={gate['throughput_ratio']:.4f} ppl_ratio={gate['ppl_ratio']:.4f}", flush=True)
                if not gate["pass"]:
                    status = "stopped_by_gate"
                    failure = "overlay failed 50K quality/throughput gate"
                    break
    except Exception as exc:
        status = "failed"
        failure = f"{type(exc).__name__}: {exc}"
    payload = {
        "status": status,
        "failure": failure,
        "parity": parity,
        "stages": results,
        "gates": gates,
        "out_root": str(out_root),
    }
    write_json(out_root / "ladder_results.json", payload)
    verification = {
        "pass": status == "passed",
        "status": status,
        "failure": failure,
        "required_parity_seeds": 3,
        "completed_stage_ids": list(results),
        "expected_stage_ids": [stage.stage_id for stage in build_ladder_stages()],
        "all_completed_stages_healthy": all(result["health"]["pass"] for result in results.values()),
        "overlay_50k_gate": gates.get("overlay_50k"),
        "result_path": str(out_root / "ladder_results.json"),
    }
    write_json(out_root / "completion_verification.json", verification)
    print(f"[ladder] status={status} result={out_root / 'ladder_results.json'}", flush=True)
    if status != "passed":
        raise SystemExit(1)
    return payload


def main(argv: list[str] | None = None) -> int:
    run_ladder(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
