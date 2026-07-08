#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
DEFAULT_DATASET = ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt"


def _str(value: Any) -> str:
    return str(value)


def _env_bool(env: dict[str, str], key: str) -> bool:
    return env.get(key, "0") == "1"


def _env_float_or_none(env: dict[str, str], key: str) -> float | None:
    raw = env.get(key, "")
    if str(raw).strip() == "":
        return None
    return float(raw)


def _env_offsets(env: dict[str, str], key: str) -> list[int]:
    raw = env.get(key, "")
    if str(raw).strip().lower() in {"", "none", "off", "disable", "disabled", "[]"}:
        return []
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _dsqg_w_legacy_mode_labels(env: dict[str, str]) -> list[str]:
    ebh = _env_bool(env, "DWARF_DSQG_W_EVIDENCE_BINDING_HUB")
    packet = _env_bool(env, "DWARF_DSQG_W_EBH_SOURCEWISE_PACKET")
    labels: list[str] = []
    if ebh and not packet:
        labels.append("legacy_materialized_ebh")
    if ebh and packet and not _env_bool(env, "DWARF_DSQG_W_EBH_SCORE_FEATURES"):
        labels.append("legacy_packet_no_score")
    if _env_bool(env, "DWARF_DSQG_W_EBH_PAIR_MIXER"):
        labels.append("legacy_ebh_pair_mixer")
    if ebh and packet and (_env_bool(env, "DWARF_DSQG_W_WIDTH_CELL") or _env_bool(env, "DWARF_DSQG_W_TYPED_MIXER")):
        labels.append("legacy_packet_semantic_approx")
    if _env_bool(env, "DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL"):
        labels.append("legacy_projected_width_control")
    if _env_bool(env, "DWARF_DSQG_W_FAST_EVIDENCE_MEAN") and _env_bool(env, "DWARF_DSQG_W_ALLOW_FAST_EVIDENCE_MEAN_BYPASS"):
        labels.append("legacy_fast_evidence_mean_bypass")
    if not _env_bool(env, "DWARF_PRE_HISA_EMA"):
        labels.append("legacy_no_pre_hisa_ema")
    return labels


def _dsqg_w_lane_label(env: dict[str, str]) -> str:
    if not _env_bool(env, "DWARF_DSQG_W"):
        return "disabled"
    if _dsqg_w_legacy_mode_labels(env):
        return "legacy_guarded"
    if (
        _env_bool(env, "DWARF_DSQG_W_SOURCEWISE")
        and _env_bool(env, "DWARF_DSQG_W_TRITON_SOURCEWISE")
        and _env_bool(env, "DWARF_DSQG_W_WIDTH_CELL")
        and _env_bool(env, "DWARF_DSQG_W_TYPED_MIXER")
        and not _env_bool(env, "DWARF_DSQG_W_EVIDENCE_BINDING_HUB")
        and _env_bool(env, "DWARF_PRE_HISA_EMA")
    ):
        return "lane_a_no_ebh_lateral_open"
    if (
        _env_bool(env, "DWARF_DSQG_W_SOURCEWISE")
        and _env_bool(env, "DWARF_DSQG_W_TRITON_SOURCEWISE")
        and _env_bool(env, "DWARF_DSQG_W_EVIDENCE_BINDING_HUB")
        and _env_bool(env, "DWARF_DSQG_W_EBH_SOURCEWISE_PACKET")
        and _env_bool(env, "DWARF_DSQG_W_EBH_SCORE_FEATURES")
        and _env_bool(env, "DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM")
        and not _env_bool(env, "DWARF_DSQG_W_WIDTH_CELL")
        and not _env_bool(env, "DWARF_DSQG_W_TYPED_MIXER")
        and _env_bool(env, "DWARF_PRE_HISA_EMA")
    ):
        return "lane_b_ebh_packet_triton_score"
    return "other"


def _dsqg_w_manifest(env: dict[str, str]) -> dict[str, Any]:
    sites = [part.strip() for part in env["DWARF_DSQG_W_SITES"].split(",") if part.strip()]
    return {
        "enabled": _env_bool(env, "DWARF_DSQG_W"),
        "sourcewise": _env_bool(env, "DWARF_DSQG_W_SOURCEWISE"),
        "triton_sourcewise": _env_bool(env, "DWARF_DSQG_W_TRITON_SOURCEWISE"),
        "max_candidates": int(env["DWARF_DSQG_W_MAX_CANDIDATES"]),
        "bottleneck": int(env["DWARF_DSQG_W_BOTTLENECK"]),
        "gate_init": float(env["DWARF_DSQG_W_GATE_INIT"]),
        "fuse_init_std": float(env["DWARF_DSQG_W_FUSE_INIT_STD"]),
        "typed_mixer": _env_bool(env, "DWARF_DSQG_W_TYPED_MIXER"),
        "typed_mixer_bottleneck": int(env["DWARF_DSQG_W_TYPED_MIXER_BOTTLENECK"]),
        "typed_mixer_gate_init": float(env["DWARF_DSQG_W_TYPED_MIXER_GATE_INIT"]),
        "force_typed_mixer_gate": _env_float_or_none(env, "DWARF_DSQG_W_FORCE_TYPED_MIXER_GATE"),
        "width_cell": _env_bool(env, "DWARF_DSQG_W_WIDTH_CELL"),
        "width_bottleneck": int(env["DWARF_DSQG_W_WIDTH_BOTTLENECK"]),
        "width_gate_init": float(env["DWARF_DSQG_W_WIDTH_GATE_INIT"]),
        "force_width_gate": _env_float_or_none(env, "DWARF_DSQG_W_FORCE_WIDTH_GATE"),
        "evidence_binding_hub": _env_bool(env, "DWARF_DSQG_W_EVIDENCE_BINDING_HUB"),
        "ebh_bottleneck": int(env["DWARF_DSQG_W_EBH_BOTTLENECK"]),
        "ebh_gate_init": float(env["DWARF_DSQG_W_EBH_GATE_INIT"]),
        "force_ebh_gate": _env_float_or_none(env, "DWARF_DSQG_W_FORCE_EBH_GATE"),
        "ebh_phase_bands": int(env["DWARF_DSQG_W_EBH_PHASE_BANDS"]),
        "ebh_score_features": _env_bool(env, "DWARF_DSQG_W_EBH_SCORE_FEATURES"),
        "ebh_sourcewise_packet": _env_bool(env, "DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"),
        "ebh_triton_lane_accum": _env_bool(env, "DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM"),
        "ebh_pair_mixer": _env_bool(env, "DWARF_DSQG_W_EBH_PAIR_MIXER"),
        "ebh_pair_rank": int(env["DWARF_DSQG_W_EBH_PAIR_RANK"]),
        "ebh_pair_gate_init": float(env["DWARF_DSQG_W_EBH_PAIR_GATE_INIT"]),
        "force_ebh_pair_gate": _env_float_or_none(env, "DWARF_DSQG_W_FORCE_EBH_PAIR_GATE"),
        "query_type_bias": _env_bool(env, "DWARF_DSQG_W_QUERY_TYPE_BIAS"),
        "evidence_prior": _env_bool(env, "DWARF_DSQG_W_EVIDENCE_PRIOR"),
        "evidence_prior_clip": float(env["DWARF_DSQG_W_EVIDENCE_PRIOR_CLIP"]),
        "evidence_prior_init_scale": float(env["DWARF_DSQG_W_EVIDENCE_PRIOR_INIT_SCALE"]),
        "candidate_quotas": _env_bool(env, "DWARF_DSQG_W_CANDIDATE_QUOTAS"),
        "quota_hisa_max": int(env["DWARF_DSQG_W_QUOTA_HISA_MAX"]),
        "local_offsets": _env_offsets(env, "DWARF_DSQG_W_LOCAL_OFFSETS"),
        "long_offsets": _env_offsets(env, "DWARF_DSQG_W_LONG_OFFSETS"),
        "sourcewise_width_cell_fusion": _env_bool(env, "DWARF_DSQG_W_SOURCEWISE_WIDTH_CELL_FUSION"),
        "projected_width_control": _env_bool(env, "DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL"),
        "triton_transformed_compact_read": _env_bool(env, "DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ"),
        "active_site_mode": "multi_site" if len(sites) > 1 else "single_site",
        "site_scheduling_policy": "fixed_env_order",
        "sites": sites,
        "pre_hisa_ema_policy": "enabled_required_for_promoted_lanes" if _env_bool(env, "DWARF_PRE_HISA_EMA") else "legacy_ablation_disabled",
        "layer_layout_marker": "trainer_runtime_hash",
        "lane_label": _dsqg_w_lane_label(env),
        "legacy_guarded_modes": _dsqg_w_legacy_mode_labels(env),
    }


def build_run_config(
    *,
    output_dir: Path | str,
    run_name: str = "dsqg_w_2_6_final_pilot",
    gpu: str = "0",
    max_acc_steps: int = 25,
    train_seqs: int = 256,
    val_seqs: int = 128,
    batch_size: int = 1,
    grad_accum: int = 1,
    epochs: int = 1,
    log_interval: int = 1,
    passkey_trials: int = 2,
    dsqg_w: bool = True,
    sites: str = "2,6,final",
    max_candidates: int = 16,
    bottleneck: int = 64,
    gate_init: float = -2.5,
    gate_lr_mult: float = 1.25,
    fuse_init_std: float = 0.02,
    sourcewise: bool = False,
    triton_sourcewise: bool = False,
    detach_recomposer: bool = False,
    fast_evidence_mean: bool = False,
    width_cell: bool = False,
    width_bottleneck: int = 64,
    width_gate_init: float = -2.5,
    width_aux_weight: float = 0.0,
    width_entropy_floor: float = 1.5,
    width_entropy_weight: float = 0.25,
    force_width_gate: float | None = None,
    typed_mixer: bool = False,
    typed_mixer_bottleneck: int = 64,
    typed_mixer_gate_init: float = -2.5,
    force_typed_mixer_gate: float | None = None,
    evidence_binding_hub: bool = False,
    ebh_bottleneck: int = 256,
    ebh_gate_init: float = -5.0,
    force_ebh_gate: float | None = None,
    ebh_phase_bands: int = 4,
    ebh_score_features: bool = True,
    ebh_pair_mixer: bool = False,
    ebh_pair_rank: int = 64,
    ebh_pair_gate_init: float = -2.5,
    force_ebh_pair_gate: float | None = None,
    ebh_sourcewise_packet: bool = False,
    ebh_triton_lane_accum: bool = False,
    evidence_prior: bool = False,
    evidence_prior_clip: float = 2.0,
    evidence_prior_init_scale: float = 0.0,
    candidate_quotas: bool = False,
    quota_hisa_max: int = 0,
    query_type_bias: bool = False,
    typed_hisa_reps: bool = False,
    dsr_candidates: bool = True,
    local_offsets: str = "none",
    long_offsets: str = "none",
    hisa_stage2_rep_r: int = 4,
    hisa_top_m: int | None = None,
    k_question: int = 4,
    k_hisa_evidence: int = 4,
    k_l3_skip: int = 2,
    pure_dsqg: bool = False,
    pre_hisa_ema: bool = True,
    lr: float | None = None,
    min_lr_ratio: float | None = None,
    lr_warmup_steps: int | None = None,
    seq_len: int | None = None,
    resume: Path | str | None = None,
    skip_opt: bool = False,
    skip_sched: bool = False,
    dataset: Path | str = DEFAULT_DATASET,
    tokenizer: Path | str = DEFAULT_TOKENIZER,
    python: Path | str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    checkpoint_dir = out / "checkpoints"
    stdout_path = out / "trainer.stdout.log"
    stderr_path = out / "trainer.stderr.log"
    py = str(python or sys.executable)
    sourcewise = bool(sourcewise or triton_sourcewise)
    env: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": ".",
        "DWARF_TOKENIZER": str(tokenizer),
        "DWARF_DATASET": str(dataset),
        "DWARF_CHECKPOINT_DIR": str(checkpoint_dir),
        "DWARF_CKPT_BASE_NAME": f"d512_l10_dsqg_w_{run_name}",
        "DWARF_EPOCHS": str(int(epochs)),
        "DWARF_MAX_ACC_STEPS": str(int(max_acc_steps)),
        "DWARF_MAX_TRAIN_SEQS": str(int(train_seqs)),
        "DWARF_MAX_VAL_SEQS": str(int(val_seqs)),
        "DWARF_BS": str(int(batch_size)),
        "DWARF_GA": str(int(grad_accum)),
        "DWARF_LOG_INTERVAL": str(int(log_interval)),
        "DWARF_PASSKEY_TRIALS": str(int(passkey_trials)),
        "DWARF_DSQG_W": "1" if dsqg_w else "0",
        "DWARF_DSQG_W_SITES": str(sites),
        "DWARF_DSQG_W_MAX_CANDIDATES": str(int(max_candidates)),
        "DWARF_DSQG_W_BOTTLENECK": str(int(bottleneck)),
        "DWARF_DSQG_W_GATE_INIT": str(float(gate_init)),
        "DWARF_DSQG_W_GATE_LR_MULT": str(float(gate_lr_mult)),
        "DWARF_DSQG_W_FUSE_INIT_STD": str(float(fuse_init_std)),
        "DWARF_DSQG_W_SOURCEWISE": "1" if sourcewise else "0",
        "DWARF_DSQG_W_TRITON_SOURCEWISE": "1" if triton_sourcewise else "0",
        "DWARF_DSQG_W_DETACH_RECOMPOSER": "1" if detach_recomposer else "0",
        "DWARF_DSQG_W_FAST_EVIDENCE_MEAN": "1" if fast_evidence_mean else "0",
        "DWARF_DSQG_W_WIDTH_CELL": "1" if width_cell else "0",
        "DWARF_DSQG_W_WIDTH_BOTTLENECK": str(int(width_bottleneck)),
        "DWARF_DSQG_W_WIDTH_GATE_INIT": str(float(width_gate_init)),
        "DWARF_DSQG_W_WIDTH_AUX_WEIGHT": str(float(width_aux_weight)),
        "DWARF_DSQG_W_WIDTH_ENTROPY_FLOOR": str(float(width_entropy_floor)),
        "DWARF_DSQG_W_WIDTH_ENTROPY_WEIGHT": str(float(width_entropy_weight)),
        "DWARF_DSQG_W_FORCE_WIDTH_GATE": "" if force_width_gate is None else str(float(force_width_gate)),
        "DWARF_DSQG_W_TYPED_MIXER": "1" if typed_mixer else "0",
        "DWARF_DSQG_W_TYPED_MIXER_BOTTLENECK": str(int(typed_mixer_bottleneck)),
        "DWARF_DSQG_W_TYPED_MIXER_GATE_INIT": str(float(typed_mixer_gate_init)),
        "DWARF_DSQG_W_FORCE_TYPED_MIXER_GATE": "" if force_typed_mixer_gate is None else str(float(force_typed_mixer_gate)),
        "DWARF_DSQG_W_EVIDENCE_BINDING_HUB": "1" if evidence_binding_hub else "0",
        "DWARF_DSQG_W_EBH_BOTTLENECK": str(int(ebh_bottleneck)),
        "DWARF_DSQG_W_EBH_GATE_INIT": str(float(ebh_gate_init)),
        "DWARF_DSQG_W_FORCE_EBH_GATE": "" if force_ebh_gate is None else str(float(force_ebh_gate)),
        "DWARF_DSQG_W_EBH_PHASE_BANDS": str(int(ebh_phase_bands)),
        "DWARF_DSQG_W_EBH_SCORE_FEATURES": "1" if ebh_score_features else "0",
        "DWARF_DSQG_W_EBH_PAIR_MIXER": "1" if ebh_pair_mixer else "0",
        "DWARF_DSQG_W_EBH_PAIR_RANK": str(int(ebh_pair_rank)),
        "DWARF_DSQG_W_EBH_PAIR_GATE_INIT": str(float(ebh_pair_gate_init)),
        "DWARF_DSQG_W_FORCE_EBH_PAIR_GATE": "" if force_ebh_pair_gate is None else str(float(force_ebh_pair_gate)),
        "DWARF_DSQG_W_EBH_SOURCEWISE_PACKET": "1" if ebh_sourcewise_packet else "0",
        "DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM": "1" if ebh_triton_lane_accum else "0",
        "DWARF_DSQG_W_EVIDENCE_PRIOR": "1" if evidence_prior else "0",
        "DWARF_DSQG_W_EVIDENCE_PRIOR_CLIP": str(float(evidence_prior_clip)),
        "DWARF_DSQG_W_EVIDENCE_PRIOR_INIT_SCALE": str(float(evidence_prior_init_scale)),
        "DWARF_DSQG_W_CANDIDATE_QUOTAS": "1" if candidate_quotas else "0",
        "DWARF_DSQG_W_QUOTA_HISA_MAX": str(int(quota_hisa_max)),
        "DWARF_DSQG_W_QUERY_TYPE_BIAS": "1" if query_type_bias else "0",
        "DWARF_DSQG_W_TYPED_HISA_REPS": "1" if typed_hisa_reps else "0",
        "DWARF_DSQG_W_DSR_CANDIDATES": "1" if dsr_candidates else "0",
        "DWARF_DSQG_W_LOCAL_OFFSETS": str(local_offsets),
        "DWARF_DSQG_W_LONG_OFFSETS": str(long_offsets),
        "DWARF_HISA_STAGE2_REP_R": str(int(hisa_stage2_rep_r)),
        "DWARF_DSQG_W_QUESTION": "1",
        "DWARF_DSQG_W_HISA_L3": "1",
        "DWARF_DSQG_W_K_QUESTION": str(int(k_question)),
        "DWARF_DSQG_W_K_HISA_EVIDENCE": str(int(k_hisa_evidence)),
        "DWARF_DSQG_W_K_L3_SKIP": str(int(k_l3_skip)),
        "DWARF_TORCH_COMPILE": "0",
        "DWARF_LIGER": "0",
        "DWARF_Q6_G128": "0",
        "DWARF_PURE_DSQG": "1" if pure_dsqg else "0",
        "DWARF_PRE_HISA_EMA": "1" if pre_hisa_ema else "0",
        "DWARF_PIN_DATASET": "0",
    }
    env.setdefault("DWARF_DSQG_W_SOURCEWISE_WIDTH_CELL_FUSION", "0")
    env.setdefault("DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL", "0")
    env.setdefault("DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ", "0")
    if lr is not None:
        env["DWARF_LR"] = str(float(lr))
    if min_lr_ratio is not None:
        env["DWARF_MIN_LR_RATIO"] = str(float(min_lr_ratio))
    if lr_warmup_steps is not None:
        env["DWARF_LR_WARMUP_STEPS"] = str(int(lr_warmup_steps))
    if hisa_top_m is not None:
        env["DWARF_HISA_TOP_M"] = str(int(hisa_top_m))
    if seq_len is not None:
        env["DWARF_SEQ_LEN"] = str(int(seq_len))
    if resume is not None:
        env["DWARF_RESUME"] = str(resume)
    if skip_opt:
        env["DWARF_SKIP_OPT"] = "1"
    if skip_sched:
        env["DWARF_SKIP_SCHED"] = "1"
    return {
        "objective": "dsqg_w_full_training_launcher",
        "run_name": run_name,
        "root": str(ROOT),
        "trainer": str(TRAINER),
        "output_dir": str(out),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "config_path": str(out / "run_config.json"),
        "command": [py, str(TRAINER.relative_to(ROOT))],
        "env": env,
        "manifest": {"dsqg_w": _dsqg_w_manifest(env)},
    }


def _jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, sort_keys=True))


def write_config(config: dict[str, Any]) -> Path:
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path(config["env"]["DWARF_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
    path = Path(config["config_path"])
    path.write_text(json.dumps(_jsonable_config(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def execute_config(config: dict[str, Any]) -> dict[str, Any]:
    write_config(config)
    env = os.environ.copy()
    env.update(config["env"])
    stdout_path = Path(config["stdout_path"])
    stderr_path = Path(config["stderr_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            config["command"],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    elapsed_s = time.time() - t0
    return {
        "returncode": int(completed.returncode),
        "elapsed_s": elapsed_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a bounded real-trainer DSQG-W full-run pilot")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dsqg_w_full_training_pilot"))
    parser.add_argument("--run-name", default="dsqg_w_2_6_final_pilot")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-acc-steps", type=int, default=25)
    parser.add_argument("--train-seqs", type=int, default=256)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--passkey-trials", type=int, default=2)
    parser.add_argument("--disable-dsqg-w", action="store_true", help="Run the DSR backbone without DSQG-W recomposition.")
    parser.add_argument("--sites", default="2,6,final")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--gate-init", type=float, default=-2.5)
    parser.add_argument("--gate-lr-mult", type=float, default=1.25)
    parser.add_argument("--fuse-init-std", type=float, default=0.02)
    parser.add_argument("--sourcewise", action="store_true", help="Enable opt-in source-wise DSQG-W score/read accumulation.")
    parser.add_argument(
        "--triton-sourcewise",
        action="store_true",
        help="Enable the opt-in Triton DSQG-W sourcewise path with recompute backward; requires --sourcewise.",
    )
    parser.add_argument("--detach-recomposer", action="store_true", help="Run DSQG-W recomposition as a detached forward perturbation to avoid W backward cost.")
    parser.add_argument("--fast-evidence-mean", action="store_true", help="Use the experimental fast evidence-mean DSQG-W path instead of score/read kernels.")
    parser.add_argument("--width-cell", action="store_true", help="Enable the opt-in DSQG-W candidate lateral width cell.")
    parser.add_argument("--width-bottleneck", type=int, default=64)
    parser.add_argument("--width-gate-init", type=float, default=-2.5)
    parser.add_argument("--width-aux-weight", type=float, default=0.0)
    parser.add_argument("--width-entropy-floor", type=float, default=1.5)
    parser.add_argument("--width-entropy-weight", type=float, default=0.25)
    parser.add_argument("--force-width-gate", type=float, default=None, help="Hard-clamp width-cell gate value in [0,1] for lateral-open probes.")
    parser.add_argument("--typed-mixer", action="store_true", help="Enable the typed candidate-set mixer before DSQG-W scoring.")
    parser.add_argument("--typed-mixer-bottleneck", type=int, default=64)
    parser.add_argument("--typed-mixer-gate-init", type=float, default=-2.5)
    parser.add_argument("--force-typed-mixer-gate", type=float, default=None, help="Hard-clamp typed-mixer gate value in [0,1] for lateral-open probes.")
    parser.add_argument("--evidence-binding-hub", action="store_true", help="Enable the TPJ-like DSQG-W evidence binding hub before W scoring/readout.")
    parser.add_argument("--ebh-bottleneck", type=int, default=256)
    parser.add_argument("--ebh-gate-init", type=float, default=-5.0)
    parser.add_argument("--force-ebh-gate", type=float, default=None, help="Hard-clamp EBH output/bind gate value in [0,1] for admission probes.")
    parser.add_argument("--ebh-phase-bands", type=int, default=4)
    parser.add_argument("--no-ebh-score-features", action="store_true", help="Disable scalar score features in the EBH alignment path.")
    parser.add_argument("--ebh-pair-mixer", action="store_true", help="Enable bounded low-rank O(K^2) EBH candidate-candidate mixing before lane reduction.")
    parser.add_argument("--ebh-pair-rank", type=int, default=64)
    parser.add_argument("--ebh-pair-gate-init", type=float, default=-2.5)
    parser.add_argument("--force-ebh-pair-gate", type=float, default=None, help="Hard-clamp EBH pair-mixer gate value in [0,1].")
    parser.add_argument("--ebh-sourcewise-packet", action="store_true", help="Use sourcewise EBH packet accumulation instead of materializing [B,T,J,D] candidate states.")
    parser.add_argument("--ebh-triton-lane-accum", action="store_true", help="Use the Triton-backed EBH sourcewise packet lane-accumulation fast path.")
    parser.add_argument("--evidence-prior", action="store_true", help="Enable the DSQG-W scalar evidence/source/type prior composer.")
    parser.add_argument("--evidence-prior-clip", type=float, default=2.0)
    parser.add_argument("--evidence-prior-init-scale", type=float, default=0.0)
    parser.add_argument("--candidate-quotas", action="store_true", help="Enable DSQG-W candidate quotas for HISA evidence slots.")
    parser.add_argument("--quota-hisa-max", type=int, default=0)
    parser.add_argument("--query-type-bias", action="store_true", help="Enable query-conditioned candidate-type score bias.")
    parser.add_argument("--typed-hisa-reps", action="store_true", help="Label first four HISA evidence candidates as representative evidence slots.")
    parser.add_argument("--no-dsr-candidates", action="store_true", help="Disable direct HISA/DSR selected-token candidates and use fallback offset candidates only.")
    parser.add_argument("--local-offsets", default="none", help="Comma-separated DSQG-W local offset candidates; default none for D-fed W runs.")
    parser.add_argument("--long-offsets", default="none", help="Comma-separated DSQG-W long offset candidates; default none for D-fed W runs.")
    parser.add_argument("--hisa-stage2-rep-r", type=int, default=4, help="Query-representative HISA Stage-2 selector representatives; use 0 for legacy row-max diagnostic fallback.")
    parser.add_argument("--hisa-top-m", type=int, default=None, help="HISA token-refinement top-m; 8192 CPT uses 16 to stay under Triton shared-memory limits when resuming 2048 checkpoints.")
    parser.add_argument("--k-question", type=int, default=4, help="Number of DSQG-W question candidates; use 0 to disable.")
    parser.add_argument("--k-hisa-evidence", type=int, default=4, help="Number of DSQG-W HISA evidence candidates; use 0 to disable.")
    parser.add_argument("--k-l3-skip", type=int, default=2, help="Number of DSQG-W L3 skip candidates; use 0 to disable.")
    parser.add_argument("--pure-dsqg", action="store_true", help="Disable HISA/DSR and run the pure DSQG-D v1 control layout.")
    parser.add_argument("--disable-pre-hisa-ema", action="store_true", help="Disable the L2 pre-HISA causal EMA/preIF K/V injection path.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-opt", action="store_true", help="Skip optimizer state when resuming from a checkpoint.")
    parser.add_argument("--skip-sched", action="store_true", help="Skip scheduler state when resuming from a checkpoint.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true", help="Write run_config.json and exit without executing trainer.")
    parser.add_argument("--execute", action="store_true", help="Execute trainer after writing run_config.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_run_config(
        output_dir=args.output_dir,
        run_name=args.run_name,
        gpu=args.gpu,
        max_acc_steps=args.max_acc_steps,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        log_interval=args.log_interval,
        passkey_trials=args.passkey_trials,
        dsqg_w=not args.disable_dsqg_w,
        sites=args.sites,
        max_candidates=args.max_candidates,
        bottleneck=args.bottleneck,
        gate_init=args.gate_init,
        gate_lr_mult=args.gate_lr_mult,
        fuse_init_std=args.fuse_init_std,
        sourcewise=args.sourcewise,
        triton_sourcewise=args.triton_sourcewise,
        detach_recomposer=args.detach_recomposer,
        fast_evidence_mean=args.fast_evidence_mean,
        width_cell=args.width_cell,
        width_bottleneck=args.width_bottleneck,
        width_gate_init=args.width_gate_init,
        width_aux_weight=args.width_aux_weight,
        width_entropy_floor=args.width_entropy_floor,
        width_entropy_weight=args.width_entropy_weight,
        force_width_gate=args.force_width_gate,
        typed_mixer=args.typed_mixer,
        typed_mixer_bottleneck=args.typed_mixer_bottleneck,
        typed_mixer_gate_init=args.typed_mixer_gate_init,
        force_typed_mixer_gate=args.force_typed_mixer_gate,
        evidence_binding_hub=args.evidence_binding_hub,
        ebh_bottleneck=args.ebh_bottleneck,
        ebh_gate_init=args.ebh_gate_init,
        force_ebh_gate=args.force_ebh_gate,
        ebh_phase_bands=args.ebh_phase_bands,
        ebh_score_features=not args.no_ebh_score_features,
        ebh_pair_mixer=args.ebh_pair_mixer,
        ebh_pair_rank=args.ebh_pair_rank,
        ebh_pair_gate_init=args.ebh_pair_gate_init,
        force_ebh_pair_gate=args.force_ebh_pair_gate,
        ebh_sourcewise_packet=args.ebh_sourcewise_packet,
        ebh_triton_lane_accum=args.ebh_triton_lane_accum,
        evidence_prior=args.evidence_prior,
        evidence_prior_clip=args.evidence_prior_clip,
        evidence_prior_init_scale=args.evidence_prior_init_scale,
        candidate_quotas=args.candidate_quotas,
        quota_hisa_max=args.quota_hisa_max,
        query_type_bias=args.query_type_bias,
        typed_hisa_reps=args.typed_hisa_reps,
        dsr_candidates=not args.no_dsr_candidates,
        local_offsets=args.local_offsets,
        long_offsets=args.long_offsets,
        hisa_stage2_rep_r=args.hisa_stage2_rep_r,
        hisa_top_m=args.hisa_top_m,
        k_question=args.k_question,
        k_hisa_evidence=args.k_hisa_evidence,
        k_l3_skip=args.k_l3_skip,
        pure_dsqg=args.pure_dsqg,
        pre_hisa_ema=not args.disable_pre_hisa_ema,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        lr_warmup_steps=args.lr_warmup_steps,
        seq_len=args.seq_len,
        resume=args.resume,
        skip_opt=args.skip_opt,
        skip_sched=args.skip_sched,
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        python=args.python,
    )
    config_path = write_config(config)
    execute = bool(args.execute and not args.dry_run)
    report: dict[str, Any] = {
        "pass": True,
        "executed": False,
        "config_path": str(config_path),
        **config,
    }
    if execute:
        exec_report = execute_config(config)
        report.update(exec_report)
        report["executed"] = True
        report["pass"] = exec_report["returncode"] == 0
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)
