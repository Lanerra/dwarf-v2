from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_dsqg_v2_semantic_curriculum.py"
TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_dsqg_v2_semantic_curriculum", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_semantic_curriculum_artifact_has_shifted_answer_masks_and_manifest(tmp_path: Path) -> None:
    mod = load_builder()
    report = mod.build_curriculum(
        output_dir=tmp_path,
        tokenizer_path=TOKENIZER,
        train_size=24,
        val_size=12,
        seq_len=256,
        seed=20260629,
    )

    dataset_path = Path(report["dataset_path"])
    manifest_path = Path(report["manifest_path"])
    audit_path = Path(report["audit_path"])
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert audit_path.exists()

    payload = torch.load(dataset_path, weights_only=True)
    assert payload["train"].shape == (24, 256)
    assert payload["val"].shape == (12, 256)
    assert payload["train_loss_mask"].shape == payload["train"].shape
    assert payload["val_loss_mask"].shape == payload["val"].shape
    assert payload["train_loss_mask"][:, 1:].any()
    assert not payload["train_loss_mask"][:, 0].any()
    assert payload["train_source_id"].shape == (24,)
    assert payload["val_source_id"].shape == (12,)

    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert manifest["dataset_shape"]["seq_len"] == 256
    assert manifest["mask_alignment"] == "token-column aligned; trainer uses loss_mask[:, 1:] for next-token targets"
    assert manifest["architecture_note"] == "DSQG-W overlays semantic width on the DSQG-D retrieval/depth backbone; it is not a DSQG-D replacement."
    assert audit["pass"] is True
    assert audit["leakage"]["prompt_hash_overlap_count"] == 0
    assert audit["leakage"]["family_id_overlap_count"] == 0
    assert audit["splits"]["train"]["real_loss_tokens"] == int(payload["train_loss_mask"][:, 1:].sum().item())
    assert set(audit["splits"]["train"]["bucket_counts"]) >= {"lexical_gap", "copy_conflict", "relation_bridge", "retrieval_guardrail"}


def test_builder_fails_if_val_family_leaks_into_train() -> None:
    mod = load_builder()
    train_records = [{"id": "train", "prompt_hash": "a", "family_id": "same", "template_id": "t1", "bucket": "lexical_gap", "answer_token_count": 1}]
    val_records = [{"id": "val", "prompt_hash": "b", "family_id": "same", "template_id": "t2", "bucket": "lexical_gap", "answer_token_count": 1}]

    audit = mod.audit_records(train_records, val_records, train_loss_tokens=1, val_loss_tokens=1)

    assert audit["pass"] is False
    assert audit["leakage"]["family_id_overlap_count"] == 1
