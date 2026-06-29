from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/parity_dsqg_w_reference.py"
TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_parity_module():
    spec = importlib.util.spec_from_file_location("parity_dsqg_w_reference", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parity_harness_reference_backend_matches_itself(tmp_path: Path) -> None:
    mod = load_parity_module()

    report = mod.run_parity_harness(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "parity",
        train_size=8,
        val_size=4,
        seed=20260628,
        candidate_backend="reference",
    )

    assert report["pass"] is True
    assert report["objective"] == "dsqg_w_reference_candidate_parity"
    assert report["candidate_backend"] == "reference"
    assert report["reference_backend"] == "reference"
    assert report["loss_abs_diff"] == pytest.approx(0.0)
    assert report["full_logits_max_abs_diff"] == pytest.approx(0.0)
    assert report["answer_logits_max_abs_diff"] == pytest.approx(0.0)
    assert report["rank_metric_max_abs_diff"] == pytest.approx(0.0)
    assert report["scalar_telemetry_max_abs_diff"] == pytest.approx(0.0)
    assert report["reference"]["val_top5_acc"] >= 0.0
    assert report["candidate"]["val_top5_acc"] == pytest.approx(report["reference"]["val_top5_acc"])
    assert Path(report["report_path"]).exists()
    saved = json.loads(Path(report["report_path"]).read_text())
    assert saved["answer_logits_max_abs_diff"] == pytest.approx(0.0)


def test_parity_harness_rejects_unknown_candidate_backend(tmp_path: Path) -> None:
    mod = load_parity_module()

    with pytest.raises(ValueError, match="unsupported candidate_backend"):
        mod.run_parity_harness(
            tokenizer_path=TOKENIZER,
            output_dir=tmp_path / "parity",
            train_size=4,
            val_size=2,
            seed=20260628,
            candidate_backend="triton_missing",
        )
