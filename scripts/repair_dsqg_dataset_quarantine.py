#!/usr/bin/env python3
"""Create an immutable replacement artifact for exact benchmark-contaminated rows.

The source artifact is never modified. Quarantined rows are replaced with fresh,
source-pinned, document-unique, train-split FineWeb-Edu-Dedup material selected
from a separate deterministic stream. Candidate documents containing any exact
benchmark probe are rejected before packing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module("dsqg_repair_builder", SCRIPT_DIR / "build_dsqg_fineweb_edu_dedup_2b.py")
DECONTAM = _load_module("dsqg_repair_decontam", SCRIPT_DIR / "dwarf_dataset_decontam_fast.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def replace_rows(original: torch.Tensor, row_indices: list[int], replacements: torch.Tensor) -> torch.Tensor:
    if original.ndim != 2 or replacements.ndim != 2:
        raise ValueError("original and replacements must be 2D tensors")
    if len(row_indices) != replacements.shape[0]:
        raise ValueError("replacement row count does not match quarantine row count")
    if original.shape[1] != replacements.shape[1]:
        raise ValueError("replacement shape does not match sequence shape")
    if len(set(row_indices)) != len(row_indices) or any(index < 0 or index >= original.shape[0] for index in row_indices):
        raise ValueError("quarantine row indices must be unique and in range")
    repaired = original.clone()
    for replacement_index, row_index in enumerate(row_indices):
        repaired[row_index] = replacements[replacement_index].to(dtype=original.dtype)
    return repaired


def _clone_sqlite(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite ledger: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _row_has_probe(tokens: list[int], anchors: dict[int, dict[tuple[int, ...], list[Any]]], anchor_tokens: int) -> bool:
    if len(tokens) < anchor_tokens:
        return False
    rolling = DECONTAM.token_hash(tokens[:anchor_tokens])
    high = pow(DECONTAM.BASE, anchor_tokens - 1, 1 << 64)
    for offset in range(len(tokens) - anchor_tokens + 1):
        candidates = anchors.get(rolling)
        if candidates:
            prefix = tuple(tokens[offset : offset + anchor_tokens])
            for anchor, probes in candidates.items():
                if prefix != anchor:
                    continue
                for probe in probes:
                    end = offset + len(probe.tokens)
                    if end <= len(tokens) and tuple(tokens[offset:end]) == probe.tokens:
                        return True
        if offset + anchor_tokens == len(tokens):
            break
        old = (tokens[offset] + DECONTAM.BIAS) & DECONTAM.MASK64
        new = (tokens[offset + anchor_tokens] + DECONTAM.BIAS) & DECONTAM.MASK64
        rolling = (rolling - ((old * high) & DECONTAM.MASK64)) & DECONTAM.MASK64
        rolling = ((rolling * DECONTAM.BASE) + new) & DECONTAM.MASK64
    return False


def select_clean_train_rows(
    *,
    source_dataset: str,
    source_config: str,
    source_revision: str,
    source_split: str,
    selection_seed: int,
    shuffle_buffer: int,
    tokenizer_path: Path,
    vocab_size: int,
    eos_id: int,
    seq_len: int,
    rows: int,
    min_doc_chars: int,
    ledger_path: Path,
    benchmark_cache: Path,
    scratch_memmap: Path,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    encode, tokenizer_vocab, tokenizer_eos = BUILDER._tokenizer_encoder(tokenizer_path)
    if tokenizer_vocab != vocab_size or tokenizer_eos != eos_id:
        raise ValueError("tokenizer metadata does not match source artifact")
    tokenizer = DECONTAM.load_tokenizer(str(tokenizer_path))
    probes = DECONTAM.load_benchmark_probes(
        benchmark_cache,
        tokenizer,
        min_tokens=16,
        max_tokens=None,
        max_examples_per_benchmark=None,
    )
    anchors = DECONTAM.prefix_index(probes, 16)
    ledger = BUILDER.DocumentLedger(ledger_path)
    writer = BUILDER.PackedSplitWriter(scratch_memmap, rows=rows, seq_len=seq_len)
    selected: list[dict[str, Any]] = []
    stats = {"records_seen": 0, "existing_or_nontrain": 0, "short_or_invalid": 0, "probe_rejected": 0}
    try:
        for record in BUILDER.stream_huggingface_records(
            dataset=source_dataset,
            config_name=source_config,
            split=source_split,
            revision=source_revision,
            seed=selection_seed,
            shuffle_buffer=shuffle_buffer,
        ):
            if writer.complete:
                break
            stats["records_seen"] += 1
            document_id = str(record.get("id", ""))
            text = record.get("text", "")
            metadata = record.get("metadata", {})
            if not document_id or not isinstance(text, str) or not isinstance(metadata, Mapping):
                stats["short_or_invalid"] += 1
                continue
            split = BUILDER.assign_document_split(document_id, source_revision, 10, 1_000)
            if split != "train" or ledger.has_document(document_id):
                stats["existing_or_nontrain"] += 1
                continue
            normalized = BUILDER.normalize_text(text)
            if len(normalized) < min_doc_chars:
                stats["short_or_invalid"] += 1
                continue
            token_ids = encode(normalized)
            if not token_ids or min(token_ids) < 0 or max(token_ids) >= vocab_size:
                stats["short_or_invalid"] += 1
                continue
            if _row_has_probe(token_ids, anchors, 16):
                stats["probe_rejected"] += 1
                continue
            admission = ledger.admit(document_id, normalized, split, metadata)
            if not admission.accepted:
                continue
            writer.add_document(token_ids, eos_id)
            selected.append(
                {
                    "document_id": document_id,
                    "normalized_sha256": admission.normalized_sha256,
                    "url": str(metadata.get("url", "")),
                    "file_path": str(metadata.get("file_path", "")),
                }
            )
        if not writer.complete:
            raise RuntimeError(f"could not pack {rows} clean replacement rows: {stats}")
        writer.checkpoint()
        replacement_mm = np.memmap(scratch_memmap, dtype=np.int32, mode="r", shape=(rows, seq_len))
        try:
            replacement = torch.from_numpy(np.array(replacement_mm, copy=True)).to(dtype=torch.int32)
        finally:
            del replacement_mm
        return replacement, selected, stats
    finally:
        writer.close()
        ledger.close()


def repair_artifact(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output_manifest.exists() or args.output_ledger.exists():
        raise FileExistsError("repair outputs already exist; refusing to overwrite immutable artifacts")
    report = json.loads(args.decontam.read_text(encoding="utf-8"))
    quarantined = report.get("summary", {}).get("quarantine_rows", {}).get("train", [])
    if not quarantined:
        raise ValueError("decontamination report has no quarantined train rows")
    rows = [int(row) for row in quarantined]
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    packing = source_manifest["packing"]
    source = source_manifest["source"]
    tokenizer = source_manifest["tokenizer"]
    if int(packing["seq_len"]) != 2048:
        raise ValueError("repair script is intentionally fixed to the 2048-token dataset contract")

    _clone_sqlite(args.ledger, args.output_ledger)
    scratch_memmap = args.output.with_suffix(".repair.train.i32")
    replacement, replacement_docs, selection_stats = select_clean_train_rows(
        source_dataset=str(source["dataset"]),
        source_config=str(source["config"]),
        source_revision=str(source["revision"]),
        source_split=str(source["split"]),
        selection_seed=args.repair_seed,
        shuffle_buffer=int(source["shuffle_buffer"]),
        tokenizer_path=Path(tokenizer["path"]),
        vocab_size=int(tokenizer["vocab_size"]),
        eos_id=int(tokenizer["eos_id"]),
        seq_len=int(packing["seq_len"]),
        rows=len(rows),
        min_doc_chars=args.min_doc_chars,
        ledger_path=args.output_ledger,
        benchmark_cache=args.benchmark_cache,
        scratch_memmap=scratch_memmap,
    )
    try:
        payload = torch.load(args.artifact, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("train"), torch.Tensor):
            raise TypeError("source artifact has no train tensor")
        payload["train"] = replace_rows(payload["train"], rows, replacement)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
    finally:
        scratch_memmap.unlink(missing_ok=True)

    repaired_manifest = dict(source_manifest)
    repaired_manifest.update(
        {
            "schema_version": "dsqg.fineweb_edu_dedup.repaired.v1",
            "created_at": utc_now_iso(),
            "name": f"{source_manifest['name']}_decontam_repaired",
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "output_size_bytes": args.output.stat().st_size,
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command_line": sys.argv,
            },
        }
    )
    repaired_manifest["document_provenance"] = dict(source_manifest["document_provenance"])
    repaired_manifest["document_provenance"].update(
        {"ledger_path": str(args.output_ledger), "ledger_sha256": sha256_file(args.output_ledger)}
    )
    repaired_ledger = BUILDER.DocumentLedger(args.output_ledger)
    try:
        repaired_manifest["deduplication"] = repaired_ledger.summary()
    finally:
        repaired_ledger.close()
    repaired_manifest["decontamination"] = {
        "status": "repair_pending_postbuild_exact_token_audit",
        "parent_artifact": str(args.artifact),
        "parent_artifact_sha256": sha256_file(args.artifact),
        "parent_decontamination_report": str(args.decontam),
        "parent_decontamination_report_sha256": sha256_file(args.decontam),
        "quarantined_train_rows": rows,
        "repair_selection_seed": args.repair_seed,
        "replacement_documents": replacement_docs,
        "selection_stats": selection_stats,
        "benchmark_cache": str(args.benchmark_cache),
    }
    args.output_manifest.write_text(json.dumps(repaired_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(args.output), "manifest": str(args.output_manifest), "ledger": str(args.output_ledger), "rows_replaced": rows, "selection_stats": selection_stats}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--decontam", type=Path, required=True)
    parser.add_argument("--benchmark-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--repair-seed", type=int, default=20260710_1)
    parser.add_argument("--min-doc-chars", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = repair_artifact(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
