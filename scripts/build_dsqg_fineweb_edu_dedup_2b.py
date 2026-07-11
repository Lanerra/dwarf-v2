#!/usr/bin/env python3
"""Build a pinned, document-deduplicated FineWeb-Edu DSQG pretraining artifact.

The builder streams the public SmolLM `fineweb-edu-dedup` source at an immutable
revision, assigns a document-disjoint validation split before tokenization, keeps
an SQLite identity/provenance ledger, tight-packs OLMo-tokenizer IDs with EOS, and
emits an immutable `.pt` artifact plus manifest.  It resumes by replaying the
same deterministic stream order and skipping ledgered document IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "HuggingFaceTB/smollm-corpus"
DEFAULT_CONFIG = "fineweb-edu-dedup"
DEFAULT_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"
DEFAULT_DATASET_NAME = "dsqg_fineweb_edu_dedup_olmo1tok_2048_2b"
DEFAULT_TOKENIZER = Path("/home/dlewis3/Desktop/AI/DWARF/tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json")
DEFAULT_TRAIN_ROWS = 977_040  # 2,000,000,880 next-token targets at seq_len=2048.
DEFAULT_VALIDATION_ROWS = 8_192


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str | None
    normalized_text: str
    normalized_sha256: str


@dataclass(frozen=True)
class BuildConfig:
    source_revision: str
    dataset_name: str
    seq_len: int
    train_rows: int
    validation_rows: int
    eos_id: int
    vocab_size: int
    tokenizer_path: str
    output_path: Path
    manifest_path: Path
    ledger_path: Path
    validation_buckets: int
    split_buckets: int
    state_path: Path | None = None
    train_memmap_path: Path | None = None
    validation_memmap_path: Path | None = None
    benchmark_cache: Path | None = None
    source_dataset: str = DEFAULT_DATASET
    source_config: str = DEFAULT_CONFIG
    source_split: str = "train"
    selection_seed: int = 20260710
    shuffle_buffer: int = 100_000


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Use a documented, conservative identity normalization without rewriting content."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def assign_document_split(
    document_id: str,
    source_revision: str,
    validation_buckets: int,
    split_buckets: int,
) -> str:
    if not (0 < validation_buckets < split_buckets):
        raise ValueError("validation_buckets must be positive and less than split_buckets")
    payload = f"{source_revision}\0{document_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % split_buckets
    return "validation" if bucket < validation_buckets else "train"


class DocumentLedger:
    """Persistent exact-document identity ledger for selected source records."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                normalized_sha256 TEXT NOT NULL UNIQUE,
                split TEXT NOT NULL CHECK(split IN ('train', 'validation')),
                metadata_json TEXT NOT NULL,
                normalized_characters INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                kind TEXT PRIMARY KEY,
                count INTEGER NOT NULL
            )
            """
        )
        self.connection.commit()
        self._pending_writes = 0

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def checkpoint(self) -> None:
        self.connection.commit()
        self._pending_writes = 0

    def _flush_periodically(self) -> None:
        self._pending_writes += 1
        if self._pending_writes >= 1_000:
            self.checkpoint()

    def has_document(self, document_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (str(document_id),)
        ).fetchone() is not None

    def _increment(self, kind: str) -> None:
        self.connection.execute(
            "INSERT INTO events(kind, count) VALUES (?, 1) ON CONFLICT(kind) DO UPDATE SET count = count + 1",
            (kind,),
        )

    def admit(self, document_id: str, text: str, split: str, metadata: Mapping[str, Any]) -> Admission:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {split}")
        normalized = normalize_text(text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if self.has_document(document_id):
            self._increment("duplicate_id")
            self._flush_periodically()
            return Admission(False, "duplicate_id", normalized, digest)
        duplicate_text = self.connection.execute(
            "SELECT 1 FROM documents WHERE normalized_sha256 = ?", (digest,)
        ).fetchone()
        if duplicate_text is not None:
            self._increment("duplicate_normalized_text")
            self._flush_periodically()
            return Admission(False, "duplicate_normalized_text", normalized, digest)
        self.connection.execute(
            """
            INSERT INTO documents(document_id, normalized_sha256, split, metadata_json, normalized_characters, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(document_id),
                digest,
                split,
                json.dumps(dict(metadata), sort_keys=True, default=str),
                len(normalized),
                utc_now_iso(),
            ),
        )
        self._increment("accepted")
        self._flush_periodically()
        return Admission(True, None, normalized, digest)

    def summary(self) -> dict[str, Any]:
        self.checkpoint()
        by_split = {
            str(split): int(count)
            for split, count in self.connection.execute(
                "SELECT split, COUNT(*) FROM documents GROUP BY split ORDER BY split"
            )
        }
        events = {str(kind): int(count) for kind, count in self.connection.execute("SELECT kind, count FROM events")}
        return {
            "accepted": int(events.get("accepted", 0)),
            "by_split": by_split,
            "duplicate_id": int(events.get("duplicate_id", 0)),
            "duplicate_normalized_text": int(events.get("duplicate_normalized_text", 0)),
        }


class PackedSplitWriter:
    """A resumable fixed-row tight packer. Documents never cross split boundaries."""

    def __init__(self, path: Path, rows: int, seq_len: int, *, existing_rows: int = 0, buffer: list[int] | None = None):
        if rows <= 0 or seq_len <= 1:
            raise ValueError("rows must be positive and seq_len must exceed one")
        self.path = path
        self.rows = int(rows)
        self.seq_len = int(seq_len)
        self.rows_written = int(existing_rows)
        self.buffer = list(buffer or [])
        if not (0 <= self.rows_written <= self.rows):
            raise ValueError("invalid existing row count")
        if len(self.buffer) >= self.seq_len:
            raise ValueError("resume buffer must be shorter than one packed row")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self.path.exists() else "w+"
        self.memmap = np.memmap(self.path, dtype=np.int32, mode=mode, shape=(self.rows, self.seq_len))

    @property
    def complete(self) -> bool:
        return self.rows_written >= self.rows

    @property
    def remaining_slots(self) -> int:
        return max(0, (self.rows - self.rows_written) * self.seq_len - len(self.buffer))

    def add_document(self, token_ids: Iterable[int], eos_id: int) -> int:
        if self.complete:
            return 0
        payload = [int(token) for token in token_ids] + [int(eos_id)]
        if not payload:
            return 0
        accepted = payload[: self.remaining_slots]
        self.buffer.extend(accepted)
        while len(self.buffer) >= self.seq_len and not self.complete:
            self.memmap[self.rows_written, :] = np.asarray(self.buffer[: self.seq_len], dtype=np.int32)
            del self.buffer[: self.seq_len]
            self.rows_written += 1
        if self.complete:
            self.buffer.clear()
        return len(accepted)

    def checkpoint(self) -> None:
        self.memmap.flush()

    def close(self) -> None:
        self.memmap.flush()
        del self.memmap


def _state_payload(
    *,
    config: BuildConfig,
    docs_seen: int,
    docs_rejected_short: int,
    train: PackedSplitWriter,
    validation: PackedSplitWriter,
) -> dict[str, Any]:
    return {
        "schema_version": "dsqg.fineweb_edu_dedup.state.v1",
        "source_revision": config.source_revision,
        "dataset_name": config.dataset_name,
        "seq_len": config.seq_len,
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
        "docs_seen": docs_seen,
        "docs_rejected_short": docs_rejected_short,
        "train": {"rows_written": train.rows_written, "buffer": train.buffer},
        "validation": {"rows_written": validation.rows_written, "buffer": validation.buffer},
        "updated_at": utc_now_iso(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_state(path: Path | None, config: BuildConfig) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_revision": config.source_revision,
        "dataset_name": config.dataset_name,
        "seq_len": config.seq_len,
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
    }
    mismatched = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
    if mismatched:
        raise ValueError(f"resume state does not match requested build: {mismatched}")
    return state


def _default_memmap_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_suffix(output_path.suffix + ".train.i32"),
        output_path.with_suffix(output_path.suffix + ".validation.i32"),
    )


def _tokenizer_encoder(tokenizer_path: Path) -> tuple[Callable[[str], list[int]], int, int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError(f"{tokenizer_path} has no <|endoftext|> token")
    return lambda text: [int(token) for token in tokenizer.encode(text).ids], tokenizer.get_vocab_size(), int(eos_id)


def _materialize_payload(config: BuildConfig, train_path: Path, validation_path: Path) -> dict[str, Any]:
    train_mm = np.memmap(train_path, dtype=np.int32, mode="r", shape=(config.train_rows, config.seq_len))
    validation_mm = np.memmap(validation_path, dtype=np.int32, mode="r", shape=(config.validation_rows, config.seq_len))
    try:
        train = torch.from_numpy(np.array(train_mm, copy=True)).to(dtype=torch.int32)
        validation = torch.from_numpy(np.array(validation_mm, copy=True)).to(dtype=torch.int32)
    finally:
        del train_mm
        del validation_mm
    return {
        "train": train,
        "val": validation,
        "source_id_train": torch.zeros((config.train_rows,), dtype=torch.int16),
        "source_id_val": torch.zeros((config.validation_rows,), dtype=torch.int16),
        "source_id_map": {"fineweb_edu_dedup": 0},
        "vocab_size": config.vocab_size,
        "seq_len": config.seq_len,
        "eos_id": config.eos_id,
        "tokenizer": config.tokenizer_path,
        "tokenizer_path": config.tokenizer_path,
        "dataset": config.dataset_name,
        "source_mix": {"fineweb_edu_dedup": 1.0},
        "n_seqs": {
            "train": {"fineweb_edu_dedup": config.train_rows},
            "val": {"fineweb_edu_dedup": config.validation_rows},
            "total": {"fineweb_edu_dedup": config.train_rows + config.validation_rows},
        },
        "packing_mode": "tight_eod_document_split",
        "fim_rate": 0.0,
    }


def _benchmark_cache_manifest(cache_dir: Path | None) -> dict[str, Any]:
    if cache_dir is None:
        return {"status": "not_configured"}
    files = sorted(cache_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"benchmark cache has no JSON files: {cache_dir}")
    return {
        "status": "configured_for_postbuild_exact_token_audit",
        "path": str(cache_dir),
        "files": [{"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
    }


def build_from_records(
    *,
    records: Iterable[Mapping[str, Any]],
    config: BuildConfig,
    encode: Callable[[str], list[int]],
    split_assigner: Callable[[str, str, int, int], str] = assign_document_split,
    min_doc_chars: int = 1,
    checkpoint_every_docs: int = 10_000,
) -> dict[str, Any]:
    if config.output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {config.output_path}")
    state = _load_state(config.state_path, config)
    default_train_path, default_validation_path = _default_memmap_paths(config.output_path)
    train_path = config.train_memmap_path or default_train_path
    validation_path = config.validation_memmap_path or default_validation_path
    train_state = (state or {}).get("train", {})
    validation_state = (state or {}).get("validation", {})
    train = PackedSplitWriter(train_path, config.train_rows, config.seq_len, existing_rows=int(train_state.get("rows_written", 0)), buffer=train_state.get("buffer", []))
    validation = PackedSplitWriter(
        validation_path,
        config.validation_rows,
        config.seq_len,
        existing_rows=int(validation_state.get("rows_written", 0)),
        buffer=validation_state.get("buffer", []),
    )
    ledger = DocumentLedger(config.ledger_path)
    docs_seen = int((state or {}).get("docs_seen", 0))
    docs_rejected_short = int((state or {}).get("docs_rejected_short", 0))
    started = time.monotonic()
    try:
        for record in records:
            if train.complete and validation.complete:
                break
            docs_seen += 1
            document_id = str(record.get("id", ""))
            text = record.get("text", "")
            metadata = record.get("metadata", {})
            if not document_id or not isinstance(text, str) or not isinstance(metadata, Mapping):
                docs_rejected_short += 1
                continue
            split = split_assigner(document_id, config.source_revision, config.validation_buckets, config.split_buckets)
            writer = train if split == "train" else validation
            if writer.complete or ledger.has_document(document_id):
                continue
            normalized = normalize_text(text)
            if len(normalized) < min_doc_chars:
                docs_rejected_short += 1
                continue
            token_ids = encode(normalized)
            if not token_ids:
                docs_rejected_short += 1
                continue
            if min(token_ids) < 0 or max(token_ids) >= config.vocab_size:
                raise ValueError(f"token IDs outside vocab range for document {document_id}")
            admission = ledger.admit(document_id, normalized, split, metadata)
            if admission.accepted:
                writer.add_document(token_ids, config.eos_id)
            if config.state_path is not None and docs_seen % checkpoint_every_docs == 0:
                train.checkpoint()
                validation.checkpoint()
                ledger.checkpoint()
                _write_json(
                    config.state_path,
                    _state_payload(
                        config=config,
                        docs_seen=docs_seen,
                        docs_rejected_short=docs_rejected_short,
                        train=train,
                        validation=validation,
                    ),
                )
        train.checkpoint()
        validation.checkpoint()
        ledger.checkpoint()
        if not (train.complete and validation.complete):
            raise RuntimeError(
                "incomplete dataset build: "
                f"train_rows={train.rows_written}/{config.train_rows}, "
                f"validation_rows={validation.rows_written}/{config.validation_rows}"
            )
        payload = _materialize_payload(config, train_path, validation_path)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, config.output_path)
        ledger_summary = ledger.summary()
        manifest = {
            "schema_version": "dsqg.fineweb_edu_dedup.v1",
            "architecture": "DSQG",
            "name": config.dataset_name,
            "created_at": utc_now_iso(),
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command_line": sys.argv,
            },
            "output": str(config.output_path),
            "output_sha256": sha256_file(config.output_path),
            "output_size_bytes": config.output_path.stat().st_size,
            "source": {
                "dataset": config.source_dataset,
                "config": config.source_config,
                "revision": config.source_revision,
                "split": config.source_split,
                "streaming_selection": "fixed-seed shuffled stream; stop after document-unique train/validation quotas",
                "selection_seed": config.selection_seed,
                "shuffle_buffer": config.shuffle_buffer,
            },
            "tokenizer": {
                "path": config.tokenizer_path,
                "sha256": sha256_file(Path(config.tokenizer_path)) if Path(config.tokenizer_path).is_file() else None,
                "vocab_size": config.vocab_size,
                "eos_id": config.eos_id,
            },
            "packing": {
                "mode": "tight_eod_document_split",
                "seq_len": config.seq_len,
                "document_split_before_packing": True,
                "train_rows": config.train_rows,
                "validation_rows": config.validation_rows,
                "train_input_positions": config.train_rows * config.seq_len,
                "validation_input_positions": config.validation_rows * config.seq_len,
                "actual_train_next_token_targets": config.train_rows * (config.seq_len - 1),
                "actual_validation_next_token_targets": config.validation_rows * (config.seq_len - 1),
            },
            "document_provenance": {
                "ledger_path": str(config.ledger_path),
                "ledger_sha256": sha256_file(config.ledger_path),
                "document_split_before_packing": True,
                "split_assignment": "blake2b-64(source_revision + NUL + document_id) modulo split_buckets",
                "validation_buckets": config.validation_buckets,
                "split_buckets": config.split_buckets,
                "normalization": "Unicode NFC; CRLF/CR to LF; trim trailing whitespace per line and outer whitespace",
            },
            "deduplication": ledger_summary,
            "decontamination": _benchmark_cache_manifest(config.benchmark_cache),
            "runtime": {
                "docs_seen": docs_seen,
                "docs_rejected_short_or_invalid": docs_rejected_short,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "resume_state_path": str(config.state_path) if config.state_path else None,
            },
        }
        _write_json(config.manifest_path, manifest)
        if config.state_path is not None:
            _write_json(
                config.state_path,
                {
                    **_state_payload(
                        config=config,
                        docs_seen=docs_seen,
                        docs_rejected_short=docs_rejected_short,
                        train=train,
                        validation=validation,
                    ),
                    "complete": True,
                    "artifact": str(config.output_path),
                },
            )
        return {
            "complete": True,
            "train_rows": config.train_rows,
            "validation_rows": config.validation_rows,
            "actual_train_next_token_targets": config.train_rows * (config.seq_len - 1),
            "output": str(config.output_path),
            "manifest": str(config.manifest_path),
            "ledger": str(config.ledger_path),
        }
    finally:
        train.close()
        validation.close()
        ledger.close()


def stream_huggingface_records(
    *,
    dataset: str,
    config_name: str,
    split: str,
    revision: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[Mapping[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(dataset, config_name, split=split, revision=revision, streaming=True)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from stream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--validation-rows", type=int, default=DEFAULT_VALIDATION_ROWS)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--validation-buckets", type=int, default=10)
    parser.add_argument("--split-buckets", type=int, default=1_000)
    parser.add_argument("--min-doc-chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--checkpoint-every-docs", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "datasets" / f"{DEFAULT_DATASET_NAME}.pt")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--train-memmap", type=Path)
    parser.add_argument("--validation-memmap", type=Path)
    parser.add_argument("--benchmark-cache", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")
    output = args.output.resolve()
    manifest = args.manifest or output.with_suffix(".manifest.json")
    ledger = args.ledger or output.with_suffix(".documents.sqlite")
    state = args.state or output.with_suffix(".state.json")
    encoder, vocab_size, eos_id = _tokenizer_encoder(args.tokenizer)
    config = BuildConfig(
        source_revision=args.revision,
        dataset_name=args.dataset_name,
        seq_len=args.seq_len,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        eos_id=eos_id,
        vocab_size=vocab_size,
        tokenizer_path=str(args.tokenizer.resolve()),
        output_path=output,
        manifest_path=manifest.resolve(),
        ledger_path=ledger.resolve(),
        state_path=state.resolve(),
        train_memmap_path=args.train_memmap.resolve() if args.train_memmap else None,
        validation_memmap_path=args.validation_memmap.resolve() if args.validation_memmap else None,
        validation_buckets=args.validation_buckets,
        split_buckets=args.split_buckets,
        benchmark_cache=args.benchmark_cache.resolve() if args.benchmark_cache else None,
        source_dataset=args.dataset,
        source_config=args.config,
        source_split=args.split,
        selection_seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    print(
        json.dumps(
            {
                "event": "dsqg_fineweb_edu_dedup_build_start",
                "dataset": args.dataset,
                "config": args.config,
                "revision": args.revision,
                "train_rows": args.train_rows,
                "validation_rows": args.validation_rows,
                "seq_len": args.seq_len,
                "target_train_next_token_targets": args.train_rows * (args.seq_len - 1),
                "output": str(output),
                "manifest": str(manifest),
                "ledger": str(ledger),
                "state": str(state),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report = build_from_records(
        records=stream_huggingface_records(
            dataset=args.dataset,
            config_name=args.config,
            split=args.split,
            revision=args.revision,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
        config=config,
        encode=encoder,
        min_doc_chars=args.min_doc_chars,
        checkpoint_every_docs=args.checkpoint_every_docs,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
