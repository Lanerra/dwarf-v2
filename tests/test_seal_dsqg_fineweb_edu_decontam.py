from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEALER_PATH = ROOT / "scripts" / "seal_dsqg_fineweb_edu_decontam.py"
CONTRACT_PATH = ROOT / "scripts" / "fwe_dedup_artifact.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seal_zero_match_audit_creates_trainable_immutable_manifest(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("fixture-tokenizer", encoding="utf-8")
    artifact = tmp_path / "fwe.pt"
    torch.save({"train": torch.zeros((3, 8), dtype=torch.int32), "val": torch.ones((2, 8), dtype=torch.int32)}, artifact)
    source_manifest = tmp_path / "fwe.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "dsqg.fineweb_edu_dedup.v1",
                "output": str(artifact),
                "output_size_bytes": artifact.stat().st_size,
                "output_sha256": "source-hash-is-intentionally-recomputed",
                "packing": {"seq_len": 8, "train_rows": 3, "validation_rows": 2},
                "tokenizer": {"sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(), "vocab_size": 32},
                "document_provenance": {"document_split_before_packing": True},
            }
        ),
        encoding="utf-8",
    )
    decontam = tmp_path / "fwe.decontam.json"
    decontam.write_text(
        json.dumps({"schema_version": "dwarf.dataset_decontam.v2", "summary": {"match_count": 0}, "matches": []}),
        encoding="utf-8",
    )
    sealed = tmp_path / "fwe.audited.manifest.json"

    sealer = load_module("seal_dsqg_fineweb_edu_decontam", SEALER_PATH)
    result = sealer.seal_zero_match_audit(
        artifact_path=artifact,
        source_manifest_path=source_manifest,
        decontam_path=decontam,
        output_manifest_path=sealed,
    )

    assert result["manifest"] == str(sealed)
    payload = json.loads(sealed.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dsqg.fineweb_edu_dedup.repaired.v1"
    assert payload["output"] == str(artifact.resolve())
    assert payload["output_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert payload["decontamination"]["status"] == "passed_zero_match_exact_token_audit"

    contract = load_module("fwe_dedup_artifact", CONTRACT_PATH).validate_contract(
        artifact_path=artifact,
        manifest_path=sealed,
        decontam_path=decontam,
        tokenizer_path=tokenizer,
        verify_artifact_sha256=True,
    )
    assert contract.train_rows == 3


def test_seal_rejects_audit_with_matches(tmp_path: Path) -> None:
    sealer = load_module("seal_dsqg_fineweb_edu_decontam", SEALER_PATH)
    artifact = tmp_path / "fwe.pt"
    artifact.write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "dsqg.fineweb_edu_dedup.v1"}), encoding="utf-8")
    decontam = tmp_path / "decontam.json"
    decontam.write_text(json.dumps({"schema_version": "dwarf.dataset_decontam.v2", "summary": {"match_count": 1}, "matches": [{"row": 0}]}), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="zero-match"):
        sealer.seal_zero_match_audit(
            artifact_path=artifact,
            source_manifest_path=manifest,
            decontam_path=decontam,
            output_manifest_path=tmp_path / "sealed.json",
        )
