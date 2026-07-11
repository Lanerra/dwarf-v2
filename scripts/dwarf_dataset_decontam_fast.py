#!/usr/bin/env python3
"""Linear-time exact token decontamination audit for DWARF dataset artifacts.

Each benchmark probe is indexed by an exact fixed-size token prefix.  Dataset rows
are scanned once with a rolling hash over that prefix; a candidate is reported only
after its entire token sequence is compared exactly.  This has the same matching
semantics as the legacy per-probe-length scan without rescanning every row once per
observed benchmark length.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

SCHEMA_VERSION = "dwarf.dataset_decontam.v2"
MASK64 = (1 << 64) - 1
BASE = 1_000_003
BIAS = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class Probe:
    benchmark: str
    example_index: int
    field: str
    severity: str
    text: str
    tokens: tuple[int, ...]


def utc_now_iso() -> str:
    return datetime.now().astimezone(timezone.utc).isoformat(timespec="seconds")


def load_artifact(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict dataset artifact, got {type(obj).__name__}")
    return obj


def load_manifest(artifact_path: Path, manifest_path: Path | None) -> tuple[dict[str, Any] | None, Path]:
    path = manifest_path or artifact_path.with_suffix(".manifest.json")
    if not path.exists():
        return None, path
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping manifest at {path}")
    return data, path


def meta(obj: dict[str, Any], manifest: dict[str, Any] | None, key: str) -> Any:
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    return obj.get(key, metadata.get(key, manifest.get(key) if manifest else None))


def source_id_map(obj: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, int] | None:
    raw = obj.get("source_id_map") or (manifest or {}).get("source_id_map")
    if not isinstance(raw, dict):
        return None
    result: dict[str, int] = {}
    for name, value in raw.items():
        try:
            result[str(name)] = int(value)
        except (TypeError, ValueError):
            continue
    return result or None


def load_tokenizer(tokenizer_ref: str, *, allow_remote: bool = False):
    path = Path(tokenizer_ref).expanduser()
    if path.is_file():
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True, local_files_only=not allow_remote)


def encode(tokenizer, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    return [int(token) for token in getattr(encoded, "ids", encoded)]


def text_fields(example: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    context = example.get("context")
    if isinstance(context, str) and context.strip():
        yield "context", "medium", context.strip()
    full = example.get("full_text")
    if isinstance(full, str) and full.strip():
        yield "full_text", "high", full.strip()
    target = example.get("target")
    if isinstance(context, str) and isinstance(target, str) and context.strip() and target.strip():
        yield "context_plus_target", "high", f"{context.strip()} {target.strip()}"
    choices = example.get("choices")
    label = example.get("label")
    if isinstance(context, str) and isinstance(choices, list) and isinstance(label, int) and 0 <= label < len(choices):
        answer = choices[label]
        if isinstance(answer, str) and answer.strip():
            yield "context_plus_label", "high", f"{context.strip()} {answer.strip()}"
    if isinstance(choices, list):
        for index, choice in enumerate(choices):
            if isinstance(choice, str) and choice.strip():
                yield f"choice_{index}", "low", choice.strip()


def load_benchmark_probes(
    cache_dir: Path,
    tokenizer,
    *,
    min_tokens: int,
    max_tokens: int | None,
    max_examples_per_benchmark: int | None,
) -> list[Probe]:
    probes: list[Probe] = []
    seen: set[tuple[int, ...]] = set()
    for path in sorted(cache_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        if max_examples_per_benchmark is not None:
            records = records[:max_examples_per_benchmark]
        for example_index, example in enumerate(records):
            if not isinstance(example, dict):
                continue
            for field, severity, text in text_fields(example):
                tokens = tuple(encode(tokenizer, text))
                if len(tokens) < min_tokens:
                    continue
                if max_tokens is not None and len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                if not tokens or tokens in seen:
                    continue
                seen.add(tokens)
                probes.append(Probe(path.stem, example_index, field, severity, text[:240], tokens))
    return probes


def token_hash(tokens: Iterable[int]) -> int:
    value = 0
    for token in tokens:
        value = ((value * BASE) + ((int(token) + BIAS) & MASK64)) & MASK64
    return value


def prefix_index(probes: Iterable[Probe], anchor_tokens: int) -> dict[int, dict[tuple[int, ...], list[Probe]]]:
    if anchor_tokens <= 0:
        raise ValueError("anchor_tokens must be positive")
    result: dict[int, dict[tuple[int, ...], list[Probe]]] = defaultdict(lambda: defaultdict(list))
    for probe in probes:
        if len(probe.tokens) < anchor_tokens:
            continue
        anchor = probe.tokens[:anchor_tokens]
        result[token_hash(anchor)][anchor].append(probe)
    return result


def search_token_rows(
    data: torch.Tensor,
    probes: list[Probe],
    *,
    split: str,
    anchor_tokens: int = 16,
    source_ids: torch.Tensor | None = None,
    source_id_map: dict[str, int] | None = None,
    max_rows: int | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    """Return exact full-probe matches after one rolling-hash pass per row."""
    if data.ndim != 2:
        raise ValueError(f"{split} tensor must be 2D, got {tuple(data.shape)}")
    anchors = prefix_index(probes, anchor_tokens)
    inverse_source = {value: name for name, value in (source_id_map or {}).items()}
    rows = int(data.shape[0]) if max_rows is None or max_rows <= 0 else min(int(data.shape[0]), max_rows)
    high = pow(BASE, anchor_tokens - 1, 1 << 64)
    matches: list[dict[str, Any]] = []

    for row_index in range(rows):
        row = [int(value) for value in data[row_index].tolist()]
        if len(row) < anchor_tokens:
            continue
        source_id = None
        source_name = None
        if source_ids is not None and source_ids.ndim == 1 and row_index < source_ids.shape[0]:
            source_id = int(source_ids[row_index].item())
            source_name = inverse_source.get(source_id, str(source_id))

        rolling = token_hash(row[:anchor_tokens])
        for offset in range(len(row) - anchor_tokens + 1):
            candidates = anchors.get(rolling)
            if candidates:
                window = tuple(row[offset : offset + anchor_tokens])
                for anchor, anchored_probes in candidates.items():
                    if window != anchor:
                        continue
                    for probe in anchored_probes:
                        end = offset + len(probe.tokens)
                        if end <= len(row) and tuple(row[offset:end]) == probe.tokens:
                            match = {
                                "split": split,
                                "row": row_index,
                                "offset": offset,
                                "length": len(probe.tokens),
                                "benchmark": probe.benchmark,
                                "example_index": probe.example_index,
                                "field": probe.field,
                                "severity": probe.severity,
                                "text_preview": probe.text,
                            }
                            if source_ids is not None:
                                match["source_id"] = source_id
                                match["source"] = source_name
                            matches.append(match)
                            if max_matches is not None and len(matches) >= max_matches:
                                return matches
            if offset + anchor_tokens == len(row):
                break
            old = (row[offset] + BIAS) & MASK64
            new = (row[offset + anchor_tokens] + BIAS) & MASK64
            rolling = (rolling - ((old * high) & MASK64)) & MASK64
            rolling = ((rolling * BASE) + new) & MASK64
    return matches


def summarize_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    quarantined: dict[str, list[int]] = defaultdict(list)
    unique_rows: set[tuple[str, int]] = set()
    for match in matches:
        benchmark, severity = str(match["benchmark"]), str(match["severity"])
        by_benchmark.setdefault(benchmark, {}).setdefault(severity, 0)
        by_benchmark[benchmark][severity] += 1
        source = str(match.get("source") or "unknown")
        by_source.setdefault(source, {}).setdefault(severity, 0)
        by_source[source][severity] += 1
        row_key = (str(match["split"]), int(match["row"]))
        if row_key not in unique_rows:
            unique_rows.add(row_key)
            quarantined[row_key[0]].append(row_key[1])
    return {
        "match_count": len(matches),
        "unique_quarantine_rows": {split: len(rows) for split, rows in quarantined.items()},
        "by_benchmark": by_benchmark,
        "by_source": by_source,
        "quarantine_rows": {split: sorted(rows) for split, rows in quarantined.items()},
    }


def decontaminate_artifact(
    artifact_path: Path,
    cache_dir: Path,
    *,
    tokenizer_ref: str | None = None,
    allow_remote_tokenizer: bool = False,
    splits: list[str] | None = None,
    min_tokens: int = 16,
    max_tokens: int | None = None,
    max_examples_per_benchmark: int | None = None,
    max_rows: int | None = None,
    max_matches: int | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if min_tokens < 1:
        raise ValueError("min_tokens must be positive")
    obj = load_artifact(artifact_path)
    manifest, manifest_file = load_manifest(artifact_path, manifest_path)
    tokenizer_name = tokenizer_ref or meta(obj, manifest, "tokenizer_path") or meta(obj, manifest, "tokenizer")
    if not tokenizer_name:
        raise KeyError("No tokenizer supplied and artifact has no tokenizer/tokenizer_path metadata")
    tokenizer = load_tokenizer(str(tokenizer_name), allow_remote=allow_remote_tokenizer)
    probes = load_benchmark_probes(
        cache_dir,
        tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        max_examples_per_benchmark=max_examples_per_benchmark,
    )
    source_map = source_id_map(obj, manifest)
    matches: list[dict[str, Any]] = []
    scanned: dict[str, Any] = {}
    for split in splits or ["train", "val"]:
        data = obj.get(split)
        if not isinstance(data, torch.Tensor):
            continue
        n_rows = int(data.shape[0]) if max_rows is None or max_rows <= 0 else min(int(data.shape[0]), max_rows)
        scanned[split] = {"rows": n_rows, "shape": list(data.shape)}
        source_ids = obj.get(f"source_id_{split}") if isinstance(obj.get(f"source_id_{split}"), torch.Tensor) else None
        split_matches = search_token_rows(
            data,
            probes,
            split=split,
            anchor_tokens=min_tokens,
            source_ids=source_ids,
            source_id_map=source_map,
            max_rows=max_rows,
            max_matches=None if max_matches is None else max_matches - len(matches),
        )
        matches.extend(split_matches)
        if max_matches is not None and len(matches) >= max_matches:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "artifact": str(artifact_path),
        "benchmark_cache_dir": str(cache_dir),
        "tokenizer": str(tokenizer_name),
        "manifest": str(manifest_file) if manifest is not None else None,
        "probe_count": len(probes),
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "max_examples_per_benchmark": max_examples_per_benchmark,
        "scanned": scanned,
        "source_id_map": source_map,
        "summary": summarize_matches(matches),
        "matches": matches,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--benchmark-cache", type=Path, default=Path("analysis_shortlist/logs/benchmark_cache"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--allow-remote-tokenizer", action="store_true")
    parser.add_argument("--split", action="append", choices=["train", "val"])
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-examples-per-benchmark", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-matches", type=int)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = decontaminate_artifact(
        args.artifact,
        args.benchmark_cache,
        tokenizer_ref=args.tokenizer,
        allow_remote_tokenizer=args.allow_remote_tokenizer,
        splits=args.split,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_examples_per_benchmark=args.max_examples_per_benchmark,
        max_rows=args.max_rows,
        max_matches=args.max_matches,
        manifest_path=args.manifest,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": report["artifact"], "probe_count": report["probe_count"], "scanned": report["scanned"], "summary": report["summary"], "out": str(args.out) if args.out else None}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
