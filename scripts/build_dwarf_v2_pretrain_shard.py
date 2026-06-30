#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets/dwarf_base_v1_olmo1tok_2048_2b.pt"
DEFAULT_OUTPUT = ROOT / "datasets/dsqg_v2_pretrain_shard_2048_20260629"
ARCHITECTURE_NOTE = "DSQG-W overlays semantic width on the DSQG-D retrieval/depth backbone; it is not a DSQG-D replacement."
PACKING_MODE = "source-stratified-subset-full-token-loss"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def invert_source_map(source_id_map: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in source_id_map.items()}


def allocate_counts(total: int, source_mix: dict[str, float], available: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    names = sorted(source_mix)
    raw = {name: float(source_mix[name]) * total for name in names}
    counts = {name: min(int(raw[name]), int(available.get(name, 0))) for name in names}
    remaining = total - sum(counts.values())
    # Largest fractional remainder first, then larger target weight, then name for determinism.
    order = sorted(names, key=lambda n: (raw[n] - int(raw[n]), float(source_mix[n]), n), reverse=True)
    while remaining > 0:
        progressed = False
        for name in order:
            if remaining <= 0:
                break
            if counts[name] < int(available.get(name, 0)):
                counts[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError(f"not enough available rows to allocate {total} examples: {available}")
    return dict(sorted(counts.items()))


def select_split(
    data: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    id_to_name: dict[int, str],
    source_mix: dict[str, float],
    size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    available: dict[str, int] = {}
    ids_by_name: dict[str, torch.Tensor] = {}
    for source_id, name in id_to_name.items():
        idx = torch.nonzero(source_ids == int(source_id), as_tuple=False).flatten()
        ids_by_name[name] = idx
        available[name] = int(idx.numel())
    counts = allocate_counts(int(size), source_mix, available)

    chosen: list[torch.Tensor] = []
    chosen_source_ids: list[torch.Tensor] = []
    for name, count in counts.items():
        if count <= 0:
            continue
        idx = ids_by_name[name]
        perm = torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(rng.randrange(2**31)))[:count]
        selected = idx[perm]
        chosen.append(data[selected].clone())
        chosen_source_ids.append(source_ids[selected].clone())
    rows = torch.cat(chosen, dim=0)
    out_source_ids = torch.cat(chosen_source_ids, dim=0).to(torch.int16)
    order = torch.randperm(rows.shape[0], generator=torch.Generator().manual_seed(rng.randrange(2**31)))
    return rows[order].contiguous(), out_source_ids[order].contiguous(), counts


def source_counts(source_ids: torch.Tensor, id_to_name: dict[int, str]) -> dict[str, int]:
    c = Counter(int(x) for x in source_ids.tolist())
    return dict(sorted((id_to_name[k], int(v)) for k, v in c.items()))


def split_summary(data: torch.Tensor, loss_mask: torch.Tensor, source_ids: torch.Tensor, id_to_name: dict[int, str]) -> dict[str, Any]:
    target_slots = int(loss_mask[:, 1:].numel())
    real_loss_tokens = int(loss_mask[:, 1:].sum().item())
    return {
        "rows": int(data.shape[0]),
        "seq_len": int(data.shape[1]),
        "source_counts": source_counts(source_ids, id_to_name),
        "real_loss_tokens": real_loss_tokens,
        "target_slots": target_slots,
        "real_loss_fraction": float(real_loss_tokens) / float(target_slots),
        "min_token_id": int(data.min().item()),
        "max_token_id": int(data.max().item()),
    }


def build_pretrain_shard(
    *,
    source_dataset: Path | str = DEFAULT_SOURCE,
    output_dir: Path | str = DEFAULT_OUTPUT,
    train_size: int = 4096,
    val_size: int = 512,
    seed: int = 20260629,
) -> dict[str, Any]:
    source_dataset = Path(source_dataset)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(source_dataset, map_location="cpu", weights_only=True, mmap=True)
    required = {"train", "val", "source_id_train", "source_id_val", "source_id_map", "source_mix", "vocab_size"}
    missing = sorted(required - set(cache))
    if missing:
        raise KeyError(f"source dataset missing required keys: {missing}")

    source_id_map = {str(k): int(v) for k, v in dict(cache["source_id_map"]).items()}
    id_to_name = invert_source_map(source_id_map)
    source_mix = {str(k): float(v) for k, v in dict(cache["source_mix"]).items()}
    if set(source_mix) != set(source_id_map):
        raise ValueError("source_mix keys must match source_id_map keys")
    vocab_size = int(cache["vocab_size"])
    seq_len = int(cache.get("seq_len", int(cache["train"].shape[1])))
    rng = random.Random(int(seed))

    train, train_source_id, planned_train_counts = select_split(
        cache["train"], cache["source_id_train"], id_to_name=id_to_name, source_mix=source_mix, size=int(train_size), rng=rng
    )
    val, val_source_id, planned_val_counts = select_split(
        cache["val"], cache["source_id_val"], id_to_name=id_to_name, source_mix=source_mix, size=int(val_size), rng=rng
    )
    train_loss_mask = torch.ones_like(train, dtype=torch.bool)
    val_loss_mask = torch.ones_like(val, dtype=torch.bool)

    audit = {
        "pass": bool(
            tuple(train.shape) == (int(train_size), seq_len)
            and tuple(val.shape) == (int(val_size), seq_len)
            and int(train.max().item()) < vocab_size
            and int(val.max().item()) < vocab_size
            and bool(train_loss_mask[:, 1:].all())
            and bool(val_loss_mask[:, 1:].all())
        ),
        "source_dataset": str(source_dataset),
        "planned_source_counts": {"train": planned_train_counts, "val": planned_val_counts},
        "splits": {
            "train": split_summary(train, train_loss_mask, train_source_id, id_to_name),
            "val": split_summary(val, val_loss_mask, val_source_id, id_to_name),
        },
    }
    if not audit["pass"]:
        raise ValueError(f"pretrain shard audit failed: {json.dumps(audit, sort_keys=True)}")

    dataset_path = out / f"dsqg_v2_pretrain_shard_{seq_len}_train{train_size}_val{val_size}.pt"
    manifest_path = out / "manifest.json"
    audit_path = out / "audit.json"
    payload = {
        "train": train,
        "val": val,
        "train_loss_mask": train_loss_mask,
        "val_loss_mask": val_loss_mask,
        "source_id_train": train_source_id,
        "source_id_val": val_source_id,
        "train_source_id": train_source_id,
        "val_source_id": val_source_id,
        "source_id_map": source_id_map,
        "source_mix": source_mix,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "eos_id": int(cache.get("eos_id", 50279)),
        "tokenizer_path": str(cache.get("tokenizer_path", "")),
        "dataset": "dsqg_v2_pretrain_shard",
        "metadata": {
            "base_dataset": str(cache.get("dataset", source_dataset.name)),
            "base_dataset_path": str(source_dataset),
            "architecture_note": ARCHITECTURE_NOTE,
            "packing_mode": PACKING_MODE,
            "seed": int(seed),
        },
    }
    torch.save(payload, dataset_path)
    manifest = {
        "name": "dsqg_v2_pretrain_shard",
        "version": "20260629-full-token-v1",
        "created_by": str(Path(__file__).relative_to(ROOT)),
        "git_commit": git_commit(),
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": sha256_file(source_dataset),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "audit_path": str(audit_path),
        "dataset_shape": {"seq_len": seq_len, "train_rows": int(train_size), "val_rows": int(val_size)},
        "source_id_map": source_id_map,
        "source_mix": source_mix,
        "packing_mode": PACKING_MODE,
        "loss_mask": "all token columns marked true; trainer applies loss_mask[:, 1:] for next-token targets",
        "architecture_note": ARCHITECTURE_NOTE,
        "intended_architecture": "DSQG-D backbone + DSQG-W semantic-width overlay; full-token pretraining objective.",
        "compatible_trainer": "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py",
        "audit_summary": audit,
    }
    audit.update({"dataset_path": str(dataset_path), "manifest_path": str(manifest_path), "dataset_sha256": manifest["dataset_sha256"]})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "pass": True,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "train_real_loss_tokens": audit["splits"]["train"]["real_loss_tokens"],
        "val_real_loss_tokens": audit["splits"]["val"]["real_loss_tokens"],
        "source_counts_train": audit["splits"]["train"]["source_counts"],
        "source_counts_val": audit["splits"]["val"]["source_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a small DWARF-v2-shaped shard from the actual base_v1 pretraining artifact")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args(argv)
    report = build_pretrain_shard(
        source_dataset=args.source_dataset,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
