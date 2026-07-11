from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dwarf_dataset_decontam_fast.py"
SPEC = importlib.util.spec_from_file_location("dwarf_dataset_decontam_fast", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def probe(tokens: list[int], *, benchmark: str = "fixture"):
    return MODULE.Probe(
        benchmark=benchmark,
        example_index=0,
        field="full_text",
        severity="high",
        text="fixture",
        tokens=tuple(tokens),
    )


def test_search_token_rows_finds_full_probe_after_shared_16_token_prefix() -> None:
    prefix = list(range(100, 116))
    matched = probe(prefix + [700, 701])
    prefix_only_collision = probe(prefix + [800, 801], benchmark="must_not_match")
    rows = torch.tensor(
        [
            [1, *prefix, 700, 701, 2],
            [1, *prefix, 999, 998, 2],
        ],
        dtype=torch.int32,
    )

    matches = MODULE.search_token_rows(
        rows,
        [matched, prefix_only_collision],
        split="train",
        anchor_tokens=16,
    )

    assert matches == [
        {
            "split": "train",
            "row": 0,
            "offset": 1,
            "length": 18,
            "benchmark": "fixture",
            "example_index": 0,
            "field": "full_text",
            "severity": "high",
            "text_preview": "fixture",
        }
    ]


def test_search_token_rows_accepts_probe_equal_to_anchor_length() -> None:
    exact = probe(list(range(16)))
    rows = torch.tensor([list(range(16))], dtype=torch.int32)

    matches = MODULE.search_token_rows(rows, [exact], split="validation", anchor_tokens=16)

    assert len(matches) == 1
    assert matches[0]["offset"] == 0
    assert matches[0]["length"] == 16


def test_decontamination_defaults_scan_train_and_validation(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "fixture.pt"
    payload = {
        "train": torch.tensor([[1, 2, 3]], dtype=torch.int32),
        "val": torch.tensor([[4, 5, 6]], dtype=torch.int32),
    }
    seen_splits: list[str] = []

    monkeypatch.setattr(MODULE, "load_artifact", lambda _: payload)
    monkeypatch.setattr(MODULE, "load_manifest", lambda *_: ({}, None))
    monkeypatch.setattr(MODULE, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(MODULE, "load_benchmark_probes", lambda *_args, **_kwargs: [probe(list(range(16)))])
    monkeypatch.setattr(MODULE, "source_id_map", lambda *_: {})

    def fake_search(data, _probes, *, split, **_kwargs):
        seen_splits.append(split)
        return []

    monkeypatch.setattr(MODULE, "search_token_rows", fake_search)
    report = MODULE.decontaminate_artifact(artifact, tmp_path, tokenizer_ref="fixture-tokenizer")

    assert seen_splits == ["train", "val"]
    assert set(report["scanned"]) == {"train", "val"}
