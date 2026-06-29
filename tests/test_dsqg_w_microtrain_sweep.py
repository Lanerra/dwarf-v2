from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_dsqg_w_microtrain.py"
TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_dsqg_w_microtrain", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_expand_sweep_grid_orders_configs_deterministically() -> None:
    mod = load_sweep_module()

    configs = mod.expand_sweep_grid(
        size_grid=[(8, 4), (12, 6)],
        step_grid=[1, 2],
        lr_grid=[1e-3, 3e-4],
    )

    assert [config["run_id"] for config in configs] == [
        "train8_val4_steps1_lr0p001",
        "train8_val4_steps1_lr0p0003",
        "train8_val4_steps2_lr0p001",
        "train8_val4_steps2_lr0p0003",
        "train12_val6_steps1_lr0p001",
        "train12_val6_steps1_lr0p0003",
        "train12_val6_steps2_lr0p001",
        "train12_val6_steps2_lr0p0003",
    ]


def test_sweep_runs_configs_and_ranks_by_validation_delta(tmp_path: Path) -> None:
    mod = load_sweep_module()

    summary = mod.run_sweep(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "sweep",
        size_grid=[(8, 4)],
        step_grid=[1, 2],
        lr_grid=[1e-3],
        seed=20260628,
    )

    assert summary["pass"] is True
    assert summary["objective"] == "dsqg_w_microtrain_sweep"
    assert summary["run_count"] == 2
    assert len(summary["runs"]) == 2
    assert summary["best_by_val_loss_delta"]["val_loss_delta"] == min(run["val_loss_delta"] for run in summary["runs"])
    assert all(run["pass"] for run in summary["runs"])
    assert all(Path(run["report_path"]).exists() for run in summary["runs"])
    assert Path(summary["summary_path"]).exists()
    saved = json.loads(Path(summary["summary_path"]).read_text())
    assert saved["best_by_val_loss_delta"]["run_id"] == summary["best_by_val_loss_delta"]["run_id"]


def test_sweep_reuses_existing_run_reports_without_rerunning(tmp_path: Path, monkeypatch) -> None:
    mod = load_sweep_module()
    run_dir = tmp_path / "sweep" / "train8_val4_steps1_lr0p001"
    run_dir.mkdir(parents=True)
    existing_report = {
        "pass": True,
        "train_examples": 8,
        "val_examples": 4,
        "steps": 1,
        "lr": 1e-3,
        "train_loss_initial": 10.0,
        "train_loss_final": 9.9,
        "train_loss_delta": -0.1,
        "val_loss_initial": 10.1,
        "val_loss_final": 10.0,
        "val_loss_delta": -0.1,
        "changed_dsqg_w_param_count": 19,
        "changed_frozen_param_count": 0,
        "checkpoint_roundtrip_loss_delta": 0.0,
        "report_path": str(run_dir / "microtrain_report.json"),
        "checkpoint": {"state_path": str(run_dir / "checkpoint" / "dsqg_w_state.pt")},
    }
    (run_dir / "microtrain_report.json").write_text(json.dumps(existing_report), encoding="utf-8")

    class ShouldNotRun:
        def run_microtrain(self, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("existing run should have been reused")

    monkeypatch.setattr(mod, "load_microtrain_module", lambda: ShouldNotRun())

    summary = mod.run_sweep(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "sweep",
        size_grid=[(8, 4)],
        step_grid=[1],
        lr_grid=[1e-3],
        seed=20260628,
    )

    assert summary["pass"] is True
    assert summary["run_count"] == 1
    assert summary["runs"][0]["run_id"] == "train8_val4_steps1_lr0p001"
    assert summary["runs"][0]["val_loss_delta"] == -0.1
