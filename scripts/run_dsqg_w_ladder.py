#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/run_dsqg_w_full_training.py"
PARSER_PATH = ROOT / "scripts/parse_dsqg_w_ladder.py"
DEFAULT_SAME_FAMILY_DATASET = (
    ROOT
    / "datasets/dsqg_v2_semantic_curriculum_2048_20260629/dsqg_v2_semantic_curriculum_2048_train4096_val512_same_family.pt"
)
DEFAULT_PRETRAIN_DATASET = ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load_module(LAUNCHER_PATH, "run_dsqg_w_full_training_for_ladder")
parser_mod = _load_module(PARSER_PATH, "parse_dsqg_w_ladder_for_runner")


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    dsqg_w: bool
    hisa_stage2_rep_r: int
    sites: str | None = None
    typed_mixer: bool = False
    query_type_bias: bool = False
    typed_hisa_reps: bool = False
    dsr_candidates: bool = True
    local_offsets: str = "none"
    long_offsets: str = "none"
    sourcewise: bool = False
    triton_sourcewise: bool = False
    detach_recomposer: bool = False
    fast_evidence_mean: bool = False
    k_question: int = 4
    k_hisa_evidence: int = 4
    k_l3_skip: int = 2
    pure_dsqg: bool = False
    gate_init: float = -2.0
    fuse_init_std: float = 0.02
    bottleneck: int = 128
    typed_mixer_bottleneck: int = 64
    typed_mixer_gate_init: float = -2.0
    notes: str = ""

    def launcher_kwargs(self) -> dict[str, Any]:
        return {
            "dsqg_w": self.dsqg_w,
            "hisa_stage2_rep_r": self.hisa_stage2_rep_r,
            "typed_mixer": self.typed_mixer,
            "query_type_bias": self.query_type_bias,
            "typed_hisa_reps": self.typed_hisa_reps,
            "dsr_candidates": self.dsr_candidates,
            "local_offsets": self.local_offsets,
            "long_offsets": self.long_offsets,
            "sourcewise": self.sourcewise,
            "triton_sourcewise": self.triton_sourcewise,
            "detach_recomposer": self.detach_recomposer,
            "fast_evidence_mean": self.fast_evidence_mean,
            "k_question": self.k_question,
            "k_hisa_evidence": self.k_hisa_evidence,
            "k_l3_skip": self.k_l3_skip,
            "pure_dsqg": self.pure_dsqg,
            "gate_init": self.gate_init,
            "fuse_init_std": self.fuse_init_std,
            "bottleneck": self.bottleneck,
            "typed_mixer_bottleneck": self.typed_mixer_bottleneck,
            "typed_mixer_gate_init": self.typed_mixer_gate_init,
        }


VARIANTS: tuple[Variant, ...] = (
    Variant(
        variant_id="P_pure_dsqg_v1",
        label="Pure DSQG-D v1 no-HISA control",
        dsqg_w=False,
        hisa_stage2_rep_r=0,
        pure_dsqg=True,
        notes="Original-style pure DSQG-D control; HISA/DSR disabled and no DSQG-W.",
    ),
    Variant(
        variant_id="A_dsr_rowmax",
        label="D-only rowmax Stage-2 baseline",
        dsqg_w=False,
        hisa_stage2_rep_r=0,
        notes="Backbone only; no DSQG-W, row-max HISA Stage-2.",
    ),
    Variant(
        variant_id="B_dsr_rep4",
        label="D-only query-representative Stage-2",
        dsqg_w=False,
        hisa_stage2_rep_r=4,
        notes="Backbone only; isolates HISA query-representative Stage-2.",
    ),
    Variant(
        variant_id="E_fast_l3_3site",
        label="Fast aligned-L3 detached W at L2/L6/final",
        dsqg_w=True,
        hisa_stage2_rep_r=4,
        sites="2,6,final",
        sourcewise=True,
        triton_sourcewise=True,
        detach_recomposer=True,
        fast_evidence_mean=True,
        k_question=0,
        k_hisa_evidence=0,
        k_l3_skip=0,
        notes=(
            "Production-candidate fast semantic-W screen: no per-token candidate gathers; "
            "uses aligned L3/DSR state as J=1 read vector with forward-only detached perturbation."
        ),
    ),
    Variant(
        variant_id="F_fast_l3_final",
        label="Fast aligned-L3 detached W at final only",
        dsqg_w=True,
        hisa_stage2_rep_r=4,
        sites="final",
        sourcewise=True,
        triton_sourcewise=True,
        detach_recomposer=True,
        fast_evidence_mean=True,
        k_question=0,
        k_hisa_evidence=0,
        k_l3_skip=0,
        notes=(
            "Placement control for the fast aligned-L3 W path; expected as lower-bound/final-only evidence, "
            "not the preferred promoted architecture if 3-site is healthy."
        ),
    ),
    Variant(
        variant_id="C_dfed_w_min",
        label="D-fed W minimal composer",
        dsqg_w=True,
        hisa_stage2_rep_r=4,
        typed_mixer=False,
        query_type_bias=False,
        typed_hisa_reps=False,
        notes="DSQG-W composes over D/HISA selected candidates with opened gate/fuse; typed features off.",
    ),
    Variant(
        variant_id="D_dfed_w_full",
        label="D-fed typed W full composer",
        dsqg_w=True,
        hisa_stage2_rep_r=4,
        typed_mixer=True,
        query_type_bias=True,
        typed_hisa_reps=True,
        notes="Full D-fed typed mixer + query type bias + typed representative labels.",
    ),
)

DEFAULT_VARIANT_IDS = "A_dsr_rowmax,B_dsr_rep4,C_dfed_w_min,D_dfed_w_full"


def select_variants(variant_ids: str) -> tuple[Variant, ...]:
    requested = [part.strip() for part in variant_ids.split(",") if part.strip()]
    by_id = {variant.variant_id: variant for variant in VARIANTS}
    if not requested:
        return VARIANTS
    unknown = [variant_id for variant_id in requested if variant_id not in by_id]
    if unknown:
        valid = ",".join(variant.variant_id for variant in VARIANTS)
        raise ValueError(f"unknown variant id(s): {','.join(unknown)}; valid={valid}")
    return tuple(by_id[variant_id] for variant_id in requested)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Variant):
        return {
            "variant_id": value.variant_id,
            "label": value.label,
            "dsqg_w": value.dsqg_w,
            "hisa_stage2_rep_r": value.hisa_stage2_rep_r,
            "sites": value.sites,
            "typed_mixer": value.typed_mixer,
            "query_type_bias": value.query_type_bias,
            "typed_hisa_reps": value.typed_hisa_reps,
            "dsr_candidates": value.dsr_candidates,
            "local_offsets": value.local_offsets,
            "long_offsets": value.long_offsets,
            "sourcewise": value.sourcewise,
            "triton_sourcewise": value.triton_sourcewise,
            "detach_recomposer": value.detach_recomposer,
            "fast_evidence_mean": value.fast_evidence_mean,
            "k_question": value.k_question,
            "k_hisa_evidence": value.k_hisa_evidence,
            "k_l3_skip": value.k_l3_skip,
            "pure_dsqg": value.pure_dsqg,
            "gate_init": value.gate_init,
            "fuse_init_std": value.fuse_init_std,
            "notes": value.notes,
        }
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


def _lane_dataset(args: argparse.Namespace, lane: str) -> Path:
    if lane == "same_family":
        return Path(args.same_family_dataset)
    if lane == "pretrain":
        return Path(args.pretrain_dataset)
    raise ValueError(f"unknown lane {lane!r}")


def build_variant_config(
    *,
    args: argparse.Namespace,
    lane: str,
    variant: Variant,
    out_root: Path,
) -> dict[str, Any]:
    run_name = f"{lane}_{variant.variant_id}"
    run_dir = out_root / lane / variant.variant_id
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
        sites=variant.sites or args.sites,
        max_candidates=args.max_candidates,
        width_cell=args.width_cell,
        width_bottleneck=args.width_bottleneck,
        width_gate_init=args.width_gate_init,
        width_aux_weight=args.width_aux_weight,
        width_entropy_floor=args.width_entropy_floor,
        width_entropy_weight=args.width_entropy_weight,
        lr=args.lr,
        dataset=_lane_dataset(args, lane),
        tokenizer=args.tokenizer,
        python=args.python,
        **variant.launcher_kwargs(),
    )
    config["ladder"] = {
        "lane": lane,
        "variant": _jsonable(variant),
        "expected_steps": int(args.max_acc_steps),
        "expected_gpu": args.expected_gpu,
    }
    return config


def run_variant(
    *,
    args: argparse.Namespace,
    lane: str,
    variant: Variant,
    out_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    print(f"[{lane}] {variant.variant_id}: {variant.label}", flush=True)
    config = build_variant_config(args=args, lane=lane, variant=variant, out_root=out_root)
    config_path = launcher.write_config(config)
    run_dir = Path(config["output_dir"])
    result: dict[str, Any] = {
        "lane": lane,
        "variant_id": variant.variant_id,
        "label": variant.label,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "dry_run": bool(dry_run),
        "variant": _jsonable(variant),
        "returncode": None,
        "metrics": {},
        "health": {"pass": bool(dry_run), "errors": [] if dry_run else ["not executed"]},
    }
    if not dry_run:
        exec_report = launcher.execute_config(config)
        result.update(exec_report)
        parsed = parser_mod.parse_run_dir(
            run_dir,
            expected_steps=int(args.max_acc_steps),
            expected_gpu=args.expected_gpu,
            require_dsqg_w=variant.dsqg_w,
            expected_stage2_rep_r=None if variant.pure_dsqg else variant.hisa_stage2_rep_r,
            returncode=int(exec_report["returncode"]),
        )
        result["metrics"] = parsed["metrics"]
        result["health"] = parsed["health"]
        status = "PASS" if result["health"]["pass"] else "FAIL"
        metrics = result["metrics"]
        print(
            f"[{lane}] {variant.variant_id}: {status} rc={result['returncode']} "
            f"step={metrics.get('final_step')}/{metrics.get('planned_steps')} "
            f"ce={metrics.get('final_ce')} ppl={metrics.get('val_ppl')} "
            f"w_dx={metrics.get('w_dx')} vram={metrics.get('peak_vram_mb')}",
            flush=True,
        )
    write_json(run_dir / "run_result.json", result)
    return result


def run_lane(
    args: argparse.Namespace,
    lane: str,
    out_root: Path,
    *,
    variants: tuple[Variant, ...],
    dry_run: bool,
) -> dict[str, Any]:
    dataset = _lane_dataset(args, lane)
    if not dataset.exists():
        raise FileNotFoundError(f"{lane} dataset does not exist: {dataset}")
    lane_results: list[dict[str, Any]] = []
    for variant in variants:
        result = run_variant(args=args, lane=lane, variant=variant, out_root=out_root, dry_run=dry_run)
        lane_results.append(result)
        if not dry_run and not result["health"]["pass"]:
            print(f"[{lane}] stopping after failed variant {variant.variant_id}", flush=True)
            break
    summary = parser_mod.summarize_lane(lane_results)
    summary.update({"lane": lane, "dataset": str(dataset), "variants_completed": len(lane_results)})
    write_json(out_root / f"{lane}_summary.json", summary)
    return {"lane": lane, "dataset": str(dataset), "results": lane_results, "summary": summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run a gated DSQG-D/HISA → DSQG-W architecture ladder")
    parser.add_argument("--out-root", type=Path, default=ROOT / f"runs/dsqg_w_dfed_ladder_{stamp}")
    parser.add_argument("--gpu", default="0", help="PyTorch CUDA_VISIBLE_DEVICES value; on this workstation 0 maps to RTX 4090.")
    parser.add_argument("--expected-gpu", default="RTX 4090")
    parser.add_argument("--same-family-dataset", type=Path, default=DEFAULT_SAME_FAMILY_DATASET)
    parser.add_argument("--pretrain-dataset", type=Path, default=DEFAULT_PRETRAIN_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=launcher.DEFAULT_TOKENIZER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lanes", choices=("same_family", "pretrain", "both"), default="both")
    parser.add_argument(
        "--variant-ids",
        default=DEFAULT_VARIANT_IDS,
        help="Comma-separated variant ids to run in order. Use P_pure_dsqg_v1,D_dfed_w_full for the pure-vs-W baseline gate.",
    )
    parser.add_argument("--max-acc-steps", type=int, default=64)
    parser.add_argument("--train-seqs", type=int, default=512)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--passkey-trials", type=int, default=0)
    parser.add_argument("--sites", default="final")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--width-cell", action="store_true")
    parser.add_argument("--width-bottleneck", type=int, default=64)
    parser.add_argument("--width-gate-init", type=float, default=-2.5)
    parser.add_argument("--width-aux-weight", type=float, default=0.0)
    parser.add_argument("--width-entropy-floor", type=float, default=1.5)
    parser.add_argument("--width-entropy-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    variants = select_variants(args.variant_ids)
    started = time.time()
    manifest = {
        "objective": "dsqg_w_dfed_ladder",
        "git_commit": _git_commit(),
        "root": str(ROOT),
        "out_root": str(out_root),
        "dry_run": bool(args.dry_run),
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "variants": [_jsonable(v) for v in variants],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out_root / "ladder_manifest.json", manifest)
    lanes = ["same_family", "pretrain"] if args.lanes == "both" else [args.lanes]
    results: dict[str, Any] = {"manifest": manifest, "lanes": {}, "pretrain_skipped": False, "pretrain_skip_reason": None}
    for lane in lanes:
        if lane == "pretrain" and args.lanes == "both":
            same = results["lanes"].get("same_family", {})
            if not same.get("summary", {}).get("pass"):
                results["pretrain_skipped"] = True
                results["pretrain_skip_reason"] = "same_family lane did not pass mechanical health gate"
                print(f"[pretrain] skipped: {results['pretrain_skip_reason']}", flush=True)
                break
        lane_result = run_lane(args, lane, out_root, variants=variants, dry_run=bool(args.dry_run))
        results["lanes"][lane] = lane_result
    results["elapsed_s"] = time.time() - started
    results["pass"] = all(lane_result["summary"].get("pass") for lane_result in results["lanes"].values())
    if results["pretrain_skipped"]:
        results["pass"] = False
    write_json(out_root / "ladder_results.json", results)
    print(json.dumps(_jsonable({"out_root": out_root, "pass": results["pass"], "pretrain_skipped": results["pretrain_skipped"]}), indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    final = main()
    raise SystemExit(0 if final["pass"] else 2)
