from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_dsqg_fineweb_edu_dedup_2b.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("dsqg_fineweb_edu_dedup_2b_builder", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ledger_rejects_duplicate_id_and_normalized_text_across_document_splits(tmp_path: Path) -> None:
    builder = load_builder()
    ledger = builder.DocumentLedger(tmp_path / "docs.sqlite")
    try:
        first = ledger.admit(
            document_id="doc-a",
            text="Alpha\r\nBeta  ",
            split="train",
            metadata={"url": "https://a.example"},
        )
        duplicate_id = ledger.admit(
            document_id="doc-a",
            text="Different document",
            split="validation",
            metadata={},
        )
        duplicate_text = ledger.admit(
            document_id="doc-b",
            text="Alpha\nBeta",
            split="validation",
            metadata={},
        )

        assert first.accepted is True
        assert first.normalized_text == "Alpha\nBeta"
        assert duplicate_id.accepted is False
        assert duplicate_id.reason == "duplicate_id"
        assert duplicate_text.accepted is False
        assert duplicate_text.reason == "duplicate_normalized_text"
        assert ledger.summary() == {
            "accepted": 1,
            "by_split": {"train": 1},
            "duplicate_id": 1,
            "duplicate_normalized_text": 1,
        }
    finally:
        ledger.close()


def test_document_hash_split_is_stable_and_document_disjoint() -> None:
    builder = load_builder()
    first = builder.assign_document_split(
        document_id="stable-doc-123",
        source_revision="pinned-revision",
        validation_buckets=10,
        split_buckets=1_000,
    )
    second = builder.assign_document_split(
        document_id="stable-doc-123",
        source_revision="pinned-revision",
        validation_buckets=10,
        split_buckets=1_000,
    )

    assert first == second
    assert first in {"train", "validation"}
    assert builder.assign_document_split(
        document_id="stable-doc-123",
        source_revision="other-pinned-revision",
        validation_buckets=10,
        split_buckets=1_000,
    ) in {"train", "validation"}


def test_build_from_records_packs_only_document_partitioned_rows_and_writes_manifest(tmp_path: Path) -> None:
    builder = load_builder()
    output = tmp_path / "dsqg_fwe_dedup.pt"
    manifest = tmp_path / "dsqg_fwe_dedup.manifest.json"
    ledger_path = tmp_path / "dsqg_fwe_dedup.documents.sqlite"
    config = builder.BuildConfig(
        source_revision="test-revision",
        dataset_name="unit-test-dsqg-fwe-dedup",
        seq_len=4,
        train_rows=2,
        validation_rows=1,
        eos_id=99,
        vocab_size=128,
        tokenizer_path="unit-tokenizer.json",
        output_path=output,
        manifest_path=manifest,
        ledger_path=ledger_path,
        validation_buckets=1_000,
        split_buckets=1_000,
    )
    records = [
        {"id": "train-a", "text": "a b c", "metadata": {"url": "https://a.example"}},
        {"id": "train-b", "text": "d e f", "metadata": {"url": "https://b.example"}},
        {"id": "train-c", "text": "g h i", "metadata": {"url": "https://c.example"}},
    ]

    def force_partition(document_id: str, source_revision: str, validation_buckets: int, split_buckets: int) -> str:
        return "validation" if document_id == "train-c" else "train"

    report = builder.build_from_records(
        records=records,
        config=config,
        encode=lambda text: [ord(token[0]) for token in text.split()],
        split_assigner=force_partition,
    )

    assert report["complete"] is True
    assert report["train_rows"] == 2
    assert report["validation_rows"] == 1
    assert output.exists()
    assert manifest.exists()
    assert ledger_path.exists()
    assert report["actual_train_next_token_targets"] == 6

    payload = builder.torch.load(output, map_location="cpu", weights_only=False)
    assert payload["train"].shape == (2, 4)
    assert payload["val"].shape == (1, 4)
    assert payload["train"].tolist() == [[97, 98, 99, 99], [100, 101, 102, 99]]
    assert payload["val"].tolist() == [[103, 104, 105, 99]]
    assert payload["source_id_train"].tolist() == [0, 0]

    manifest_data = builder.json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["source"]["revision"] == "test-revision"
    assert manifest_data["document_provenance"]["document_split_before_packing"] is True
    assert manifest_data["deduplication"]["duplicate_normalized_text"] == 0


def test_build_from_records_fails_closed_when_a_target_split_cannot_fill(tmp_path: Path) -> None:
    builder = load_builder()
    config = builder.BuildConfig(
        source_revision="test-revision",
        dataset_name="unit-test-dsqg-fwe-dedup",
        seq_len=4,
        train_rows=1,
        validation_rows=1,
        eos_id=99,
        vocab_size=128,
        tokenizer_path="unit-tokenizer.json",
        output_path=tmp_path / "incomplete.pt",
        manifest_path=tmp_path / "incomplete.manifest.json",
        ledger_path=tmp_path / "incomplete.documents.sqlite",
        validation_buckets=1_000,
        split_buckets=1_000,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        builder.build_from_records(
            records=[{"id": "only-train", "text": "a b c", "metadata": {}}],
            config=config,
            encode=lambda text: [ord(token[0]) for token in text.split()],
            split_assigner=lambda *args: "train",
        )
