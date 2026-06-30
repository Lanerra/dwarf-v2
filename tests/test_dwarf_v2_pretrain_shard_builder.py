from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_dwarf_v2_pretrain_shard.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_dwarf_v2_pretrain_shard", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_source_artifact(path: Path) -> None:
    seq_len = 8
    train = torch.arange(30 * seq_len, dtype=torch.int32).reshape(30, seq_len)
    val = torch.arange(1000, 1000 + 12 * seq_len, dtype=torch.int32).reshape(12, seq_len)
    source_id_train = torch.tensor([0] * 15 + [1] * 9 + [2] * 6, dtype=torch.int16)
    source_id_val = torch.tensor([0] * 6 + [1] * 4 + [2] * 2, dtype=torch.int16)
    torch.save(
        {
            "dataset": "fake_base_v1",
            "train": train,
            "val": val,
            "source_id_train": source_id_train,
            "source_id_val": source_id_val,
            "source_id_map": {"web": 0, "code": 1, "long": 2},
            "source_mix": {"web": 0.5, "code": 0.3, "long": 0.2},
            "vocab_size": 2000,
            "seq_len": seq_len,
            "eos_id": 1999,
            "tokenizer_path": "fake-tokenizer.json",
        },
        path,
    )


def test_pretrain_shard_preserves_source_mix_and_full_loss_masks(tmp_path: Path) -> None:
    mod = load_builder()
    source = tmp_path / "source.pt"
    write_source_artifact(source)

    report = mod.build_pretrain_shard(
        source_dataset=source,
        output_dir=tmp_path / "out",
        train_size=10,
        val_size=5,
        seed=7,
    )

    dataset_path = Path(report["dataset_path"])
    manifest_path = Path(report["manifest_path"])
    audit_path = Path(report["audit_path"])
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert audit_path.exists()

    shard = torch.load(dataset_path, weights_only=True)
    assert shard["train"].shape == (10, 8)
    assert shard["val"].shape == (5, 8)
    assert shard["train_loss_mask"].shape == shard["train"].shape
    assert shard["val_loss_mask"].shape == shard["val"].shape
    assert bool(shard["train_loss_mask"].all()) is True
    assert bool(shard["val_loss_mask"].all()) is True
    assert shard["metadata"]["base_dataset"] == "fake_base_v1"
    assert shard["metadata"]["packing_mode"] == "source-stratified-subset-full-token-loss"

    audit = json.loads(audit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert audit["pass"] is True
    assert audit["splits"]["train"]["real_loss_tokens"] == 10 * 7
    assert audit["splits"]["val"]["real_loss_tokens"] == 5 * 7
    assert audit["splits"]["train"]["source_counts"] == {"code": 3, "long": 2, "web": 5}
    assert audit["splits"]["val"]["source_counts"] == {"code": 1, "long": 1, "web": 3}
    assert manifest["intended_architecture"] == "DSQG-D backbone + DSQG-W semantic-width overlay; full-token pretraining objective."
