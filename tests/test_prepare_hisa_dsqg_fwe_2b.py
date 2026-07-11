from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_hisa_dsqg_fwe_2b.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_hisa_dsqg_fwe_2b", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_audited_artifact(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("fixture-tokenizer", encoding="utf-8")
    artifact = tmp_path / "fwe.pt"
    torch.save(
        {"train": torch.zeros((3, 8), dtype=torch.int32), "val": torch.ones((2, 8), dtype=torch.int32)},
        artifact,
    )
    import hashlib

    manifest = tmp_path / "fwe.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "dsqg.fineweb_edu_dedup.repaired.v1",
                "output_size_bytes": artifact.stat().st_size,
                "packing": {"seq_len": 8, "train_rows": 3, "validation_rows": 2},
                "tokenizer": {"sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(), "vocab_size": 32},
            }
        ),
        encoding="utf-8",
    )
    decontam = tmp_path / "fwe.decontam.json"
    decontam.write_text(
        json.dumps({"schema_version": "dwarf.dataset_decontam.v2", "summary": {"match_count": 0}, "matches": []}),
        encoding="utf-8",
    )
    return artifact, manifest, decontam, tokenizer


def test_build_dry_run_contract_targets_v2_hisa_hybrid_and_forbids_dsqg_w(tmp_path: Path) -> None:
    artifact, manifest, decontam, tokenizer = write_audited_artifact(tmp_path)
    module = load_module()

    config = module.build_dry_run_config(
        output_dir=tmp_path / "run",
        artifact_path=artifact,
        manifest_path=manifest,
        decontam_path=decontam,
        tokenizer_path=tokenizer,
        gpu="0",
        train_seqs=50_000,
        batch_size=8,
        grad_accum=2,
    )

    env = config["env"]
    assert Path(config["root"]).resolve() == ROOT.resolve()
    assert Path(config["trainer"]).resolve().parent.parent == ROOT.resolve()
    assert env["DWARF_DATASET"] == str(artifact)
    assert env["DWARF_TOKENIZER"] == str(tokenizer)
    assert env["DWARF_VOCAB_SIZE"] == "32"
    assert env["DWARF_SEQ_LEN"] == "8"
    assert env["DWARF_DSQG_W"] == "0"
    assert env["DWARF_PURE_DSQG"] == "0"
    assert env["DWARF_Q6_G128"] == "0"
    assert env["DWARF_PRE_HISA_EMA"] == "1"
    assert env["DWARF_CKPT"] == "none"
    assert env["DWARF_MAX_ACC_STEPS"] == "3125"
    assert config["mode"] == "dry_run_only"
    assert config["command"][-1] == "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def test_write_dry_run_contract_never_executes_trainer(tmp_path: Path) -> None:
    artifact, manifest, decontam, tokenizer = write_audited_artifact(tmp_path)
    module = load_module()

    result = module.main(
        [
            "--output-dir", str(tmp_path / "run"),
            "--artifact", str(artifact),
            "--manifest", str(manifest),
            "--decontam", str(decontam),
            "--tokenizer", str(tokenizer),
        ]
    )

    contract_path = Path(result["contract_path"])
    assert result["executed"] is False
    assert contract_path.exists()
    assert json.loads(contract_path.read_text(encoding="utf-8"))["mode"] == "dry_run_only"
