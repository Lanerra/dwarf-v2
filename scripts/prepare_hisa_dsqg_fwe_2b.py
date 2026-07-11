#!/usr/bin/env python3
"""Prepare, but never execute, a DWARF-v2 HISA/DSQG run on audited FWE-Dedup."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from fwe_dedup_artifact import FWEArtifactContract, validate_contract


def build_dry_run_config(
    *,
    output_dir: Path | str,
    artifact_path: Path | str,
    manifest_path: Path | str,
    decontam_path: Path | str,
    tokenizer_path: Path | str,
    gpu: str = "0",
    train_seqs: int = 50_000,
    val_seqs: int = 512,
    batch_size: int = 8,
    grad_accum: int = 2,
    checkpoint_strategy: str = "none",
    verify_artifact_sha256: bool = False,
) -> dict[str, Any]:
    if train_seqs <= 0 or val_seqs <= 0 or batch_size <= 0 or grad_accum <= 0:
        raise ValueError("train/validation sequence counts, batch size, and gradient accumulation must be positive")
    contract: FWEArtifactContract = validate_contract(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        decontam_path=decontam_path,
        tokenizer_path=tokenizer_path,
        verify_artifact_sha256=verify_artifact_sha256,
    )
    out = Path(output_dir).resolve()
    max_acc_steps = int(math.ceil(train_seqs / (batch_size * grad_accum)))
    checkpoint_dir = out / "checkpoints"
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(ROOT),
        "DWARF_TOKENIZER": str(contract.tokenizer_path),
        "DWARF_DATASET": str(contract.artifact_path),
        "DWARF_VOCAB_SIZE": str(contract.vocab_size),
        "DWARF_SEQ_LEN": str(contract.seq_len),
        "DWARF_CHECKPOINT_DIR": str(checkpoint_dir),
        "DWARF_CKPT_BASE_NAME": "d512_l10_hisa_dsqg_fwe_dedup",
        "DWARF_EPOCHS": "1",
        "DWARF_MAX_ACC_STEPS": str(max_acc_steps),
        "DWARF_MAX_TRAIN_SEQS": str(train_seqs),
        "DWARF_MAX_VAL_SEQS": str(val_seqs),
        "DWARF_BS": str(batch_size),
        "DWARF_GA": str(grad_accum),
        "DWARF_LOG_INTERVAL": "10",
        "DWARF_PASSKEY_TRIALS": "10",
        "DWARF_CKPT": str(checkpoint_strategy),
        "DWARF_PURE_DSQG": "0",
        "DWARF_DSQG_W": "0",
        "DWARF_Q6_G128": "0",
        "DWARF_PRE_HISA_EMA": "1",
        "DWARF_HISA_STAGE2_REP_R": "4",
        "DWARF_HISA_TOP_M": "64",
    }
    return {
        "mode": "dry_run_only",
        "executed": False,
        "root": str(ROOT),
        "trainer": str(TRAINER),
        "output_dir": str(out),
        "contract_path": str(out / "run_contract.json"),
        "command": [sys.executable, str(TRAINER.relative_to(ROOT))],
        "env": env,
        "artifact_contract": {
            "artifact": str(contract.artifact_path),
            "manifest": str(contract.manifest_path),
            "decontam": str(contract.decontam_path),
            "tokenizer": str(contract.tokenizer_path),
            "seq_len": contract.seq_len,
            "vocab_size": contract.vocab_size,
            "train_rows": contract.train_rows,
            "validation_rows": contract.validation_rows,
            "artifact_sha256_verified": contract.artifact_sha256 is not None,
        },
        "architecture_contract": {
            "hisa_enabled": True,
            "pure_dsqg": False,
            "dsqg_w": False,
            "q6_g128": False,
            "pre_hisa_ema": True,
        },
    }


def write_dry_run_config(config: dict[str, Any]) -> Path:
    path = Path(config["contract_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a non-executing DWARF-v2 HISA/DSQG FWE-Dedup run contract")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decontam", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-seqs", type=int, default=50_000)
    parser.add_argument("--val-seqs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--checkpoint-strategy", choices=("none", "every_other", "all", "full_attn"), default="none")
    parser.add_argument("--verify-artifact-sha256", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_dry_run_config(
        output_dir=args.output_dir,
        artifact_path=args.artifact,
        manifest_path=args.manifest,
        decontam_path=args.decontam,
        tokenizer_path=args.tokenizer,
        gpu=args.gpu,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        checkpoint_strategy=args.checkpoint_strategy,
        verify_artifact_sha256=args.verify_artifact_sha256,
    )
    contract_path = write_dry_run_config(config)
    return {**config, "contract_path": str(contract_path)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
