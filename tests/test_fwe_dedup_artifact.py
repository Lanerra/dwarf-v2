from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "fwe_dedup_artifact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fwe_dedup_artifact", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_contract(tmp_path: Path, *, match_count: int = 0) -> tuple[Path, Path, Path, Path]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("fixture-tokenizer", encoding="utf-8")
    artifact = tmp_path / "fwe.pt"
    torch.save(
        {
            "train": torch.zeros((3, 8), dtype=torch.int32),
            "val": torch.ones((2, 8), dtype=torch.int32),
            "source_id_train": torch.zeros(3, dtype=torch.int16),
            "source_id_val": torch.zeros(2, dtype=torch.int16),
        },
        artifact,
    )
    manifest = tmp_path / "fwe.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "dsqg.fineweb_edu_dedup.repaired.v1",
                "output": str(artifact),
                "output_size_bytes": artifact.stat().st_size,
                "packing": {"seq_len": 8, "train_rows": 3, "validation_rows": 2},
                "tokenizer": {
                    "path": str(tokenizer),
                    "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                    "vocab_size": 32,
                },
            }
        ),
        encoding="utf-8",
    )
    decontam = tmp_path / "fwe.decontam.json"
    decontam.write_text(
        json.dumps(
            {
                "schema_version": "dwarf.dataset_decontam.v2",
                "artifact": str(artifact),
                "summary": {"match_count": match_count},
                "matches": [] if match_count == 0 else [{"benchmark": "fixture"}],
            }
        ),
        encoding="utf-8",
    )
    return artifact, manifest, decontam, tokenizer


def test_validate_contract_accepts_audited_payload_and_reports_runtime_shape(tmp_path: Path) -> None:
    artifact, manifest, decontam, tokenizer = write_contract(tmp_path)
    module = load_module()

    contract = module.validate_contract(
        artifact_path=artifact,
        manifest_path=manifest,
        decontam_path=decontam,
        tokenizer_path=tokenizer,
    )

    assert contract.seq_len == 8
    assert contract.vocab_size == 32
    assert contract.train_rows == 3
    assert contract.validation_rows == 2
    assert contract.artifact_sha256 is None


def test_validate_contract_rejects_any_decontamination_match(tmp_path: Path) -> None:
    artifact, manifest, decontam, tokenizer = write_contract(tmp_path, match_count=1)
    module = load_module()

    with pytest.raises(module.ArtifactContractError, match="match_count"):
        module.validate_contract(
            artifact_path=artifact,
            manifest_path=manifest,
            decontam_path=decontam,
            tokenizer_path=tokenizer,
        )
