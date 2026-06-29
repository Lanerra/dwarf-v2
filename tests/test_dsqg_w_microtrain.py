from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/microtrain_dsqg_w_lexical_gap.py"
TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_microtrain_module():
    spec = importlib.util.spec_from_file_location("microtrain_dsqg_w_lexical_gap", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_generate_lexical_gap_dataset_has_train_val_records() -> None:
    mod = load_microtrain_module()

    train, val = mod.generate_lexical_gap_dataset(train_size=16, val_size=8, seed=20260628)

    assert len(train) == 16
    assert len(val) == 8
    all_records = train + val
    assert len({record["id"] for record in all_records}) == len(all_records)
    assert all(record["split"] == "train" for record in train)
    assert all(record["split"] == "val" for record in val)
    for record in all_records:
        assert "Answer:" in record["prompt"]
        assert record["tokens"][record["answer_positions"][0]] == record["answer"]
        assert record["gold_evidence_indices"]
        assert record["hisa_evidence_indices"]
        assert record["question_indices"]
        assert record["l3_skip_indices"]


def test_answer_rank_metrics_measure_topk_and_rank() -> None:
    mod = load_microtrain_module()
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 4.0, 3.0, 2.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
            ]
        ]
    )
    labels = torch.tensor([[2, 2]], dtype=torch.long)
    answer_mask = torch.tensor([[True, True]])

    metrics = mod.answer_rank_metrics(logits, labels, answer_mask, prefix="toy")

    assert metrics["toy_answer_tokens"] == 2.0
    assert metrics["toy_top1_acc"] == pytest.approx(0.5)
    assert metrics["toy_top5_acc"] == pytest.approx(1.0)
    assert metrics["toy_mean_rank"] == pytest.approx(2.0)
    assert metrics["toy_median_rank"] == pytest.approx(2.0)


def test_microtrainer_runs_tokenized_train_val_checkpoint_roundtrip(tmp_path: Path) -> None:
    mod = load_microtrain_module()

    report = mod.run_microtrain(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "microtrain",
        train_size=12,
        val_size=6,
        steps=2,
        lr=1e-3,
        seed=20260628,
    )

    assert report["pass"] is True
    assert report["objective"] == "dsqg_w_lexical_gap_microtrain"
    assert report["tokenized"] is True
    assert report["train_examples"] == 12
    assert report["val_examples"] == 6
    assert report["steps"] == 2
    assert report["train_loss_final"] < report["train_loss_initial"]
    assert report["train_mean_rank_initial"] > 0.0
    assert report["train_mean_rank_final"] > 0.0
    assert "train_mean_rank_delta" in report
    assert 0.0 <= report["train_top1_acc_initial"] <= 1.0
    assert 0.0 <= report["train_top1_acc_final"] <= 1.0
    assert report["val_mean_rank_initial"] > 0.0
    assert report["val_mean_rank_final"] > 0.0
    assert "val_mean_rank_delta" in report
    assert 0.0 <= report["val_top5_acc_initial"] <= 1.0
    assert 0.0 <= report["val_top5_acc_final"] <= 1.0
    assert report["val_loss_initial"] > 0.0
    assert report["val_loss_final"] > 0.0
    assert report["checkpoint_roundtrip_loss_delta"] == pytest.approx(0.0)
    assert report["changed_frozen_param_count"] == 0
    assert report["changed_dsqg_w_param_count"] > 0
    assert Path(report["checkpoint"]["state_path"]).exists()
    assert Path(report["checkpoint"]["metadata_path"]).exists()
    report_path = Path(report["report_path"])
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["checkpoint"]["metadata_path"] == report["checkpoint"]["metadata_path"]
