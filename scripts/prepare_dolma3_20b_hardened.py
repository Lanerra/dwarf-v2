#!/usr/bin/env python3
"""Prepare, but never execute, the hardened 20B Dolma 3 DWARF run."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_dolma3_20b_hardened.py"
EXPECTED_SOURCE_DATASET = "allenai/dolma3_mix-150B-1025"
EXPECTED_SOURCE_REVISION = "afa92bfb22366821c5e6cd427cdd036b34b713ef"
EXPECTED_TRAIN_INPUT_TOKENS = 20_000_000_000
EXPECTED_SEQUENCE_LENGTH = 2_048
MILESTONE_INPUT_TOKENS = (
    500_000_000,
    1_000_000_000,
    2_000_000_000,
    5_000_000_000,
    10_000_000_000,
    15_000_000_000,
    20_000_000_000,
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(*, artifact_path: Path | str, release_path: Path | str, tokenizer_path: Path | str,
                     verify_artifact_sha256: bool = False,
                     verify_artifact_payload: bool = False) -> dict[str, Any]:
    artifact = Path(artifact_path).resolve()
    release_file = Path(release_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    if not artifact.is_file() or not release_file.is_file() or not tokenizer.is_file():
        raise FileNotFoundError("artifact, release manifest, and tokenizer must all exist")
    release = json.loads(release_file.read_text(encoding="utf-8"))
    if release.get("format") != "dwarf-dolma3-base-release-v1":
        raise ValueError("unexpected Dolma 3 release format")
    source = release.get("source", {})
    if source.get("dataset") != EXPECTED_SOURCE_DATASET or source.get("revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("release is not pinned to the approved Dolma 3 Mix 150B source revision")
    policy = release.get("policy", {})
    if policy.get("sequence_length") != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError("release sequence length must be 2048")
    if policy.get("document_format") != "raw_text" or policy.get("loss_mask_policy") != "implicit_shifted_causal_all_targets":
        raise ValueError("release is not raw all-token causal base-pretraining data")
    counts = release.get("counts", {})
    train_rows = int(counts.get("train_sequences", 0))
    validation_rows = int(counts.get("validation_sequences", 0))
    train_input_tokens = int(counts.get("train_input_tokens", 0))
    if train_rows <= 0 or validation_rows <= 0 or train_input_tokens != EXPECTED_TRAIN_INPUT_TOKENS:
        raise ValueError("release does not contain the exact approved 20B training-input-token budget")
    tokenizer_info = release.get("tokenizer", {})
    tokenizer_sha256 = sha256_file(tokenizer)
    if tokenizer_info.get("sha256") != tokenizer_sha256:
        raise ValueError("tokenizer SHA-256 does not match the sealed release")
    vocab_size = int(tokenizer_info.get("vocab_size", 0))
    if vocab_size != 50_282:
        raise ValueError("hardened trainer requires the 50,282-vocabulary OLMo1 tokenizer")
    artifact_info = release.get("dataset", {})
    if artifact_info.get("path") != artifact.name:
        raise ValueError("release dataset path does not identify the supplied artifact")
    artifact_sha256 = None
    if verify_artifact_sha256:
        artifact_sha256 = sha256_file(artifact)
        if artifact_info.get("sha256") != artifact_sha256:
            raise ValueError("dataset SHA-256 does not match the sealed release")
    if verify_artifact_payload:
        payload = torch.load(artifact, map_location="cpu", mmap=True, weights_only=False)
        train = payload.get("train")
        validation = payload.get("val")
        if (
            not torch.is_tensor(train)
            or not torch.is_tensor(validation)
            or tuple(train.shape) != (train_rows, EXPECTED_SEQUENCE_LENGTH)
            or tuple(validation.shape) != (validation_rows, EXPECTED_SEQUENCE_LENGTH)
            or train.dtype != torch.int32
            or validation.dtype != torch.int32
            or payload.get("vocab_size") != vocab_size
        ):
            raise ValueError("packed tensor payload does not match the sealed Dolma 3 release contract")
        metadata = payload.get("metadata", {})
        if metadata.get("sequence_length") != EXPECTED_SEQUENCE_LENGTH:
            raise ValueError("packed tensor metadata sequence length does not match the sealed release")
    return {
        "artifact": str(artifact),
        "release": str(release_file),
        "tokenizer": str(tokenizer),
        "artifact_sha256_verified": artifact_sha256 is not None,
        "artifact_payload_verified": verify_artifact_payload,
        "artifact_sha256": artifact_sha256,
        "release_sha256": sha256_file(release_file),
        "tokenizer_sha256": tokenizer_sha256,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_input_tokens": train_input_tokens,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "vocab_size": vocab_size,
    }


def build_dry_run_config(*, output_dir: Path | str, artifact_path: Path | str, release_path: Path | str,
                         tokenizer_path: Path | str, gpu: str, batch_size: int, grad_accum: int,
                         verify_artifact_sha256: bool = False,
                         verify_artifact_payload: bool = False) -> dict[str, Any]:
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    artifact_contract = validate_release(
        artifact_path=artifact_path,
        release_path=release_path,
        tokenizer_path=tokenizer_path,
        verify_artifact_sha256=verify_artifact_sha256,
        verify_artifact_payload=verify_artifact_payload,
    )
    out = Path(output_dir).resolve()
    max_acc_steps = math.ceil(artifact_contract["train_rows"] / (batch_size * grad_accum))
    wsd_warmup_steps = math.ceil(max_acc_steps * 0.05)
    wsd_decay_steps = math.ceil(max_acc_steps * 0.15)
    wsd_stable_steps = max_acc_steps - wsd_warmup_steps - wsd_decay_steps
    checkpoint_dir = out / "checkpoints"
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(ROOT),
        "DWARF_DISABLE_BNB": "1",
        "DWARF_TORCH_COMPILE": "0",
        "DWARF_TOKENIZER": artifact_contract["tokenizer"],
        "DWARF_DATASET": artifact_contract["artifact"],
        "DWARF_VOCAB_SIZE": str(artifact_contract["vocab_size"]),
        "DWARF_SEQ_LEN": str(artifact_contract["sequence_length"]),
        "DWARF_CHECKPOINT_DIR": str(checkpoint_dir),
        "DWARF_CKPT_BASE_NAME": "d512_l10_dolma3_20b_hardened",
        "DWARF_EPOCHS": "1",
        "DWARF_MAX_ACC_STEPS": str(max_acc_steps),
        "DWARF_MAX_TRAIN_SEQS": str(artifact_contract["train_rows"]),
        "DWARF_MAX_VAL_SEQS": str(artifact_contract["validation_rows"]),
        "DWARF_BS": str(batch_size),
        "DWARF_GA": str(grad_accum),
        "DWARF_LOG_INTERVAL": "25",
        "DWARF_CKPT": "none",
        "DWARF_LR_SCHEDULE": "wsd",
        "DWARF_SCHEDULE_TOTAL_STEPS": str(max_acc_steps),
        "DWARF_WSD_WARMUP_STEPS": str(wsd_warmup_steps),
        "DWARF_WSD_STABLE_STEPS": str(wsd_stable_steps),
        "DWARF_WSD_DECAY_STEPS": str(wsd_decay_steps),
        "DWARF_SAVE_EVERY_STEPS": "500",
        "DWARF_EVAL_EVERY_STEPS": "1000",
        "DWARF_MILESTONE_INPUT_TOKENS": ",".join(str(value) for value in MILESTONE_INPUT_TOKENS),
        "DWARF_DSQG_W": "0",
        "DWARF_PURE_DSQG": "0",
        "DWARF_Q6_G128": "0",
    }
    return {
        "mode": "dry_run_only",
        "executed": False,
        "status": "prepared_pending_artifact_audit_and_deployed_stack_sweep",
        "root": str(ROOT),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "output_dir": str(out),
        "contract_path": str(out / "run_contract.json"),
        "command": [sys.executable, str(TRAINER.relative_to(ROOT))],
        "env": env,
        "artifact_contract": artifact_contract,
        "resume_contract": {
            "strict_mid_epoch": True,
            "rolling_full_state_checkpoints": 2,
            "save_every_optimizer_steps": 500,
            "checkpoint_payload": [
                "model", "optimizer", "scheduler", "global_step", "epoch", "next_acc_step",
                "epoch_permutation", "python_torch_cuda_rng", "artifact_and_trainer_contract",
            ],
        },
        "schedule_contract": {
            "kind": "wsd",
            "total_steps": max_acc_steps,
            "warmup_steps": wsd_warmup_steps,
            "stable_steps": wsd_stable_steps,
            "decay_steps": wsd_decay_steps,
        },
        "evaluation_contract": {
            "validation_ppl_every_optimizer_steps": 1000,
            "external_promotion_milestones_input_tokens": [
                value for value in MILESTONE_INPUT_TOKENS if value >= 2_000_000_000
            ],
            "external_gpu": "separate RTX 3090 lane after checkpoint reconstruction verification",
        },
        "launch_gates": [
            "sealed artifact SHA-256 verification",
            "exact-token train+validation benchmark audit report",
            "deployed-stack 4090 BS/GA sweep with >=4GiB VRAM headroom",
            "fresh-process production-geometry checkpoint/resume smoke",
            "live CUDA_VISIBLE_DEVICES-to-GPU UUID verification",
        ],
    }


def write_dry_run_config(config: dict[str, Any]) -> Path:
    path = Path(config["contract_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--verify-artifact-sha256", action="store_true")
    parser.add_argument("--verify-artifact-payload", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = build_dry_run_config(
        output_dir=args.output_dir,
        artifact_path=args.artifact,
        release_path=args.release,
        tokenizer_path=args.tokenizer,
        gpu=args.gpu,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        verify_artifact_sha256=args.verify_artifact_sha256,
        verify_artifact_payload=args.verify_artifact_payload,
    )
    contract_path = write_dry_run_config(config)
    return {**config, "contract_path": str(contract_path)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
