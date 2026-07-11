#!/usr/bin/env python3
"""Validate an immutable audited FineWeb-Edu-Dedup artifact for DWARF-v2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


class ArtifactContractError(ValueError):
    """Raised when an external FWE-Dedup artifact is not safe to use."""


@dataclass(frozen=True)
class FWEArtifactContract:
    artifact_path: Path
    manifest_path: Path
    decontam_path: Path
    tokenizer_path: Path
    seq_len: int
    vocab_size: int
    train_rows: int
    validation_rows: int
    artifact_sha256: str | None


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactContractError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_int(payload: dict[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactContractError(f"{label}.{key} must be a positive integer")
    return int(value)


def validate_contract(
    *,
    artifact_path: Path | str,
    manifest_path: Path | str,
    decontam_path: Path | str,
    tokenizer_path: Path | str,
    verify_artifact_sha256: bool = False,
) -> FWEArtifactContract:
    """Validate audit metadata and the mmap-loadable tensor payload.

    Hashing the multi-gigabyte payload is opt-in; size, payload structure, tokenizer
    identity, and zero-match audit are always checked.
    """

    artifact = Path(artifact_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    decontam_file = Path(decontam_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    if not artifact.is_file():
        raise ArtifactContractError(f"artifact missing: {artifact}")
    if not tokenizer.is_file():
        raise ArtifactContractError(f"tokenizer missing: {tokenizer}")

    manifest = _load_json(manifest_file, "manifest")
    if manifest.get("schema_version") != "dsqg.fineweb_edu_dedup.repaired.v1":
        raise ArtifactContractError("manifest schema_version is not the audited repaired FWE-Dedup schema")
    declared_size = _expect_int(manifest, "output_size_bytes", "manifest")
    if artifact.stat().st_size != declared_size:
        raise ArtifactContractError(
            f"artifact size {artifact.stat().st_size} does not match manifest output_size_bytes {declared_size}"
        )

    packing = manifest.get("packing")
    token_meta = manifest.get("tokenizer")
    if not isinstance(packing, dict) or not isinstance(token_meta, dict):
        raise ArtifactContractError("manifest requires packing and tokenizer objects")
    seq_len = _expect_int(packing, "seq_len", "manifest.packing")
    train_rows = _expect_int(packing, "train_rows", "manifest.packing")
    validation_rows = _expect_int(packing, "validation_rows", "manifest.packing")
    vocab_size = _expect_int(token_meta, "vocab_size", "manifest.tokenizer")
    declared_tokenizer_hash = token_meta.get("sha256")
    if not isinstance(declared_tokenizer_hash, str) or _sha256(tokenizer) != declared_tokenizer_hash:
        raise ArtifactContractError("tokenizer SHA-256 does not match the audited artifact manifest")

    decontam = _load_json(decontam_file, "decontamination report")
    if decontam.get("schema_version") != "dwarf.dataset_decontam.v2":
        raise ArtifactContractError("decontamination report has an unsupported schema_version")
    summary = decontam.get("summary")
    if not isinstance(summary, dict) or summary.get("match_count") != 0 or decontam.get("matches") != []:
        raise ArtifactContractError("decontamination report must have match_count=0 and an empty matches list")

    try:
        payload = torch.load(artifact, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:  # pragma: no cover - message is platform-specific
        raise ArtifactContractError(f"artifact cannot be mmap-loaded: {artifact}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError("artifact payload must be a tensor dictionary")
    for split, expected_rows in (("train", train_rows), ("val", validation_rows)):
        tensor = payload.get(split)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ArtifactContractError(f"artifact {split} tensor is missing or not rank-2")
        if tuple(tensor.shape) != (expected_rows, seq_len):
            raise ArtifactContractError(
                f"artifact {split} shape {tuple(tensor.shape)} does not match ({expected_rows}, {seq_len})"
            )
        if tensor.dtype not in {torch.int32, torch.int64}:
            raise ArtifactContractError(f"artifact {split} dtype must be int32 or int64, got {tensor.dtype}")

    declared_artifact_hash = manifest.get("output_sha256")
    if declared_artifact_hash is not None and not isinstance(declared_artifact_hash, str):
        raise ArtifactContractError("manifest output_sha256 must be a string when present")
    if verify_artifact_sha256:
        if not declared_artifact_hash:
            raise ArtifactContractError("manifest has no output_sha256 for requested artifact verification")
        if _sha256(artifact) != declared_artifact_hash:
            raise ArtifactContractError("artifact SHA-256 does not match the audited manifest")

    return FWEArtifactContract(
        artifact_path=artifact,
        manifest_path=manifest_file,
        decontam_path=decontam_file,
        tokenizer_path=tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        train_rows=train_rows,
        validation_rows=validation_rows,
        artifact_sha256=declared_artifact_hash if verify_artifact_sha256 else None,
    )
