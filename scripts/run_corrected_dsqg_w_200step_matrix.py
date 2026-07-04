#!/usr/bin/env python3
"""Run the corrected DSQG-W 200-step smoke matrix.

This is a narrow, evidence-oriented matrix for the post-fix DSQG-W implementation:
- no-W matched DSR control
- legacy aligned-L3 bypass control
- corrected trainable aligned-L3 path
- candidate-scoring final variants with typed reps / typed mixer / relation width cell
- post-DSR+final full variant

Training runs execute sequentially through scripts/run_dsqg_w_full_training.py and
are parsed with scripts/parse_dsqg_w_ladder.py. Semantic-transfer eval is run by
a separate helper so it can use a different CUDA_VISIBLE_DEVICES mapping.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
DEFAULT_DATASET = ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load_module(LAUNCHER_PATH, "run_dsqg_w_full_training_for_corrected_matrix")
parser_mod = _load_module(PARSER_PATH, "parse_dsqg_w_ladder_for_corrected_matrix")


@dataclass(frozen=True)
class MatrixVariant:
    variant_id: str
    label: str
    dsqg_w: bool = True
    sites: str = "final"
    sourcewise: bool = True
    triton_sourcewise: bool = True
    detach_recomposer: bool = False
    fast_evidence_mean: bool = False
    allow_fast_evidence_mean_bypass: bool = False
    typed_hisa_reps: bool = False
    query_type_bias: bool = False
    typed_mixer: bool = False
    width_cell: bool = False
    k_question: int = 4
    k_hisa_evidence: int = 4
    k_l3_skip: int = 2
    hisa_stage2_rep_r: int = 4
    gate_init: float = -2.0
    gate_lr_mult: float = 1.25
    width_gate_init: float = -2.5
    typed_mixer_gate_init: float = -2.5
    notes: str = ""

    def launcher_kwargs(self) -> dict[str, Any]:
        return {
            "dsqg_w": self.dsqg_w,
            "sites": self.sites,
            "sourcewise": self.sourcewise,
            "triton_sourcewise": self.triton_sourcewise,
            "detach_recomposer": self.detach_recomposer,
            "fast_evidence_mean": self.fast_evidence_mean,
            "typed_hisa_reps": self.typed_hisa_reps,
            "query_type_bias": self.query_type_bias,
            "typed_mixer": self.typed_mixer,
            "width_cell": self.width_cell,
            "k_question": self.k_question,
            "k_hisa_evidence": self.k_hisa_evidence,
            "k_l3_skip": self.k_l3_skip,
            "hisa_stage2_rep_r": self.hisa_stage2_rep_r,
            "gate_init": self.gate_init,
            "gate_lr_mult": self.gate_lr_mult,
            "width_gate_init": self.width_gate_init,
            "typed_mixer_gate_init": self.typed_mixer_gate_init,
            "local_offsets": "none",
            "long_offsets": "none",
            "dsr_candidates": True,
            "pure_dsqg": False,
        }


VARIANTS: tuple[MatrixVariant, ...] = (
    MatrixVariant(
        variant_id="A_no_w",
        label="matched no-W DSR rep4 control",
        dsqg_w=False,
        sourcewise=False,
        triton_sourcewise=False,
        notes="Matched DSR/HISA query-representative Stage-2 backbone with DSQG-W disabled.",
    ),
    MatrixVariant(
        variant_id="B_legacy_fast_l3_final",
        label="legacy aligned-L3 final bypass",
        sites="final",
        detach_recomposer=True,
        fast_evidence_mean=True,
        allow_fast_evidence_mean_bypass=True,
        k_question=0,
        k_hisa_evidence=0,
        k_l3_skip=0,
        notes="Historical fast evidence-mean aligned-L3 path; explicit bypass opt-in for compatibility only.",
    ),
    MatrixVariant(
        variant_id="C_scored_l3_final",
        label="corrected trainable aligned-L3 final",
        sites="final",
        detach_recomposer=True,
        fast_evidence_mean=True,
        k_question=0,
        k_hisa_evidence=0,
        k_l3_skip=0,
        notes="Same requested fast-L3 shape, but corrected implementation forces trainable scored L3 candidate path.",
    ),
    MatrixVariant(
        variant_id="D_candidate_final",
        label="final candidate scorer + typed HISA reps",
        sites="final",
        typed_hisa_reps=True,
        query_type_bias=True,
        notes="True D/DSR-fed candidate-scoring W at final only; typed reps and query type bias on, no mixer/width cell.",
    ),
    MatrixVariant(
        variant_id="E_candidate_final_typed_mix",
        label="final candidate scorer + typed mixer",
        sites="final",
        typed_hisa_reps=True,
        query_type_bias=True,
        typed_mixer=True,
        notes="Adds typed candidate-set mixer before scoring.",
    ),
    MatrixVariant(
        variant_id="F_candidate_final_width",
        label="final candidate scorer + relation width cell",
        sites="final",
        typed_hisa_reps=True,
        query_type_bias=True,
        width_cell=True,
        notes="Adds DSQGWWidthCell with relation diff/product features, no typed mixer.",
    ),
    MatrixVariant(
        variant_id="G_candidate_final_full",
        label="final full corrected DSQG-W",
        sites="final",
        typed_hisa_reps=True,
        query_type_bias=True,
        typed_mixer=True,
        width_cell=True,
        notes="Full corrected final-only candidate composer: typed reps + query bias + typed mixer + relation width cell.",
    ),
    MatrixVariant(
        variant_id="H_candidate_6_final_full",
        label="post-DSR L6+final full corrected DSQG-W",
        sites="6,final",
        typed_hisa_reps=True,
        query_type_bias=True,
        typed_mixer=True,
        width_cell=True,
        notes="Post-retrieval composition interleaved before later sparse depth plus final recomposition; avoids pre-DSR L2.",
    ),
)

DEFAULT_VARIANT_IDS = ",".join(v.variant_id for v in VARIANTS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MatrixVariant):
        return {k: _jsonable(v) for k, v in value.__dict__.items()}
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
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def select_variants(variant_ids: str) -> tuple[MatrixVariant, ...]:
    requested = [v.strip() for v in variant_ids.split(",") if v.strip()]
    by_id = {v.variant_id: v for v in VARIANTS}
    unknown = [v for v in requested if v not in by_id]
    if unknown:
        raise ValueError(f"unknown variant id(s): {','.join(unknown)}; valid={','.join(by_id)}")
    return tuple(by_id[v] for v in requested)


def build_config(args: argparse.Namespace, variant: MatrixVariant, out_root: Path) -> dict[str, Any]:
    run_dir = out_root / "pretrain" / variant.variant_id
    config = launcher.build_run_config(
        output_dir=run_dir,
        run_name=f"pretrain_{variant.variant_id}",
        gpu=args.gpu,
        max_acc_steps=args.max_acc_steps,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        log_interval=args.log_interval,
        passkey_trials=args.passkey_trials,
        max_candidates=args.max_candidates,
        bottleneck=args.bottleneck,
        width_bottleneck=args.width_bottleneck,
        width_aux_weight=args.width_aux_weight,
        width_entropy_floor=args.width_entropy_floor,
        width_entropy_weight=args.width_entropy_weight,
        typed_mixer_bottleneck=args.typed_mixer_bottleneck,
        lr=args.lr,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        python=args.python,
        **variant.launcher_kwargs(),
    )
    env = config["env"]
    env["DWARF_DSQG_W_ALLOW_FAST_EVIDENCE_MEAN_BYPASS"] = "1" if variant.allow_fast_evidence_mean_bypass else "0"
    # Width transfer aux is meaningful only when DSQG-W's width cell is active.
    # Keep no-W and non-width ablations from failing the trainer's aux-telemetry guard.
    if not (variant.dsqg_w and variant.width_cell):
        env["DWARF_DSQG_W_WIDTH_AUX_WEIGHT"] = "0.0"
    config["matrix"] = {
        "variant": _jsonable(variant),
        "expected_steps": int(args.max_acc_steps),
        "expected_gpu": args.expected_gpu,
        "passkey_interpretation": "plumbing-only at 200 steps; passkey usually emerges around steps 1400-2000 in ~2400-step runs",
    }
    return config


def run_variant(args: argparse.Namespace, variant: MatrixVariant, out_root: Path, dry_run: bool) -> dict[str, Any]:
    print(f"[matrix] {variant.variant_id}: {variant.label}", flush=True)
    config = build_config(args, variant, out_root)
    config_path = launcher.write_config(config)
    run_dir = Path(config["output_dir"])
    result: dict[str, Any] = {
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
            expected_stage2_rep_r=variant.hisa_stage2_rep_r,
            returncode=int(exec_report["returncode"]),
        )
        result["metrics"] = parsed["metrics"]
        result["health"] = parsed["health"]
        status = "PASS" if result["health"]["pass"] else "FAIL"
        m = result["metrics"]
        print(
            f"[matrix] {variant.variant_id}: {status} rc={result['returncode']} "
            f"step={m.get('final_step')}/{m.get('planned_steps')} ce={m.get('final_ce')} "
            f"ppl={m.get('val_ppl')} tok/s={m.get('avg_logged_tok_s')} "
            f"w_gate={m.get('w_gate')} w_dx={m.get('w_dx')} w_j={m.get('w_j')} "
            f"vram={m.get('peak_vram_mb')}",
            flush=True,
        )
    write_json(run_dir / "run_result.json", result)
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    base = results[0].get("metrics", {}) if results else {}
    base_ppl = base.get("val_ppl")
    base_ce = base.get("final_ce")
    for result in results:
        m = result.get("metrics", {})
        row = {
            "variant_id": result.get("variant_id"),
            "label": result.get("label"),
            "health_pass": result.get("health", {}).get("pass"),
            "health_errors": result.get("health", {}).get("errors", []),
            "returncode": result.get("returncode"),
            "run_dir": result.get("run_dir"),
            "final_step": m.get("final_step"),
            "final_ce": m.get("final_ce"),
            "val_ppl": m.get("val_ppl"),
            "passkey_mean": m.get("passkey_mean"),
            "avg_logged_tok_s": m.get("avg_logged_tok_s"),
            "peak_vram_mb": m.get("peak_vram_mb"),
            "w_gate": m.get("w_gate"),
            "w_gate_logit": m.get("w_gate_logit"),
            "w_dx": m.get("w_dx"),
            "w_hisa": m.get("w_hisa"),
            "w_score": m.get("w_score"),
            "w_smean": m.get("w_smean"),
            "w_mix_gate": m.get("w_mix_gate"),
            "w_mix_gate_logit": m.get("w_mix_gate_logit"),
            "w_width_gate": m.get("w_width_gate"),
            "w_width_gate_logit": m.get("w_width_gate_logit"),
            "w_width_delta": m.get("w_width_delta"),
            "w_width_ent": m.get("w_width_ent"),
            "w_width_self": m.get("w_width_self"),
            "w_width_qh": m.get("w_width_qh"),
            "w_width_hq": m.get("w_width_hq"),
            "w_width_xfer": m.get("w_width_xfer"),
            "w_width_ep": m.get("w_width_ep"),
            "w_rel_diff": m.get("w_rel_diff"),
            "w_rel_prod": m.get("w_rel_prod"),
            "w_width_score_gn": m.get("w_width_score_gn"),
            "w_width_v_gn": m.get("w_width_v_gn"),
            "w_width_up_gn": m.get("w_width_up_gn"),
            "w_width_gate_gn": m.get("w_width_gate_gn"),
            "w_mix_gate_gn": m.get("w_mix_gate_gn"),
            "w_all_gate_gn": m.get("w_all_gate_gn"),
            "w_fast": m.get("w_fast"),
            "w_fast_bypass": m.get("w_fast_bypass"),
            "w_trainable": m.get("w_trainable"),
            "w_mat": m.get("w_mat"),
            "w_sem_bypass": m.get("w_sem_bypass"),
            "w_det": m.get("w_det"),
            "w_j": m.get("w_j"),
        }
        if isinstance(base_ppl, (int, float)) and isinstance(row["val_ppl"], (int, float)):
            row["delta_val_ppl_vs_a"] = row["val_ppl"] - base_ppl
        if isinstance(base_ce, (int, float)) and isinstance(row["final_ce"], (int, float)):
            row["delta_final_ce_vs_a"] = row["final_ce"] - base_ce
        rows.append(row)
    return {"pass": all(bool(r.get("health_pass")) for r in rows), "rows": rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / f"runs/dsqg_w_corrected_200step_matrix_{stamp}")
    parser.add_argument("--variant-ids", default=DEFAULT_VARIANT_IDS)
    parser.add_argument("--gpu", default="0", help="PyTorch CUDA_VISIBLE_DEVICES for training; verified 0=>RTX 4090 on this host.")
    parser.add_argument("--expected-gpu", default="RTX 4090")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-acc-steps", type=int, default=200)
    parser.add_argument("--train-seqs", type=int, default=800)
    parser.add_argument("--val-seqs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--passkey-trials", type=int, default=1, help="Plumbing-only at 200 steps; not a promotion metric.")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--width-bottleneck", type=int, default=64)
    parser.add_argument("--width-aux-weight", type=float, default=0.01)
    parser.add_argument("--width-entropy-floor", type=float, default=1.5)
    parser.add_argument("--width-entropy-weight", type=float, default=0.25)
    parser.add_argument("--typed-mixer-bottleneck", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    variants = select_variants(args.variant_ids)
    manifest = {
        "objective": "corrected_dsqg_w_200step_matrix",
        "git_commit": _git_commit(),
        "root": str(ROOT),
        "out_root": str(out_root),
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "variants": [_jsonable(v) for v in variants],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "passkey_note": "Dennis expects passkey to stay near 0% this early; ignore it as architecture signal.",
    }
    write_json(out_root / "matrix_manifest.json", manifest)
    started = time.time()
    results: list[dict[str, Any]] = []
    for variant in variants:
        result = run_variant(args, variant, out_root, dry_run=bool(args.dry_run))
        results.append(result)
        if not args.dry_run and not result["health"]["pass"]:
            print(f"[matrix] stopping after failed variant {variant.variant_id}", flush=True)
            break
    summary = summarize(results)
    payload = {
        "manifest": manifest,
        "results": results,
        "summary": summary,
        "elapsed_s": time.time() - started,
        "pass": bool(summary["pass"]),
    }
    write_json(out_root / "matrix_results.json", payload)
    print(json.dumps(_jsonable({"out_root": out_root, "pass": payload["pass"], "elapsed_s": payload["elapsed_s"]}), indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    final = main()
    raise SystemExit(0 if final["pass"] else 2)
