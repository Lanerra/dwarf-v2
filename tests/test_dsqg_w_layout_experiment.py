from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_dsqg_w_layout.py"
TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


def load_layout_module():
    spec = importlib.util.spec_from_file_location("sweep_dsqg_w_layout", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_layout_id_sanitizes_site_specs() -> None:
    mod = load_layout_module()

    assert mod.layout_id("final") == "sites_final"
    assert mod.layout_id("6,final") == "sites_6_final"
    assert mod.layout_id("2,6,final") == "sites_2_6_final"


def test_layout_experiment_runs_multiple_site_specs_and_ranks(tmp_path: Path) -> None:
    mod = load_layout_module()

    summary = mod.run_layout_experiment(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "layout",
        site_specs=["final", "2,final"],
        train_size=8,
        val_size=4,
        steps=1,
        lr=1e-3,
        seed=20260628,
    )

    assert summary["pass"] is True
    assert summary["objective"] == "dsqg_w_layout_experiment"
    assert summary["layout_count"] == 2
    assert [row["dsqg_w_sites"] for row in summary["layouts"]] == [["final"], ["layer_2", "final"]]
    assert summary["best_by_val_top5"]["val_top5_acc_final"] == max(row["val_top5_acc_final"] for row in summary["layouts"])
    assert summary["best_by_val_mean_rank"]["val_mean_rank_final"] == min(row["val_mean_rank_final"] for row in summary["layouts"])
    assert all(Path(row["report_path"]).exists() for row in summary["layouts"])
    saved = json.loads(Path(summary["summary_path"]).read_text())
    assert saved["layout_count"] == 2


def test_layout_experiment_uses_same_seed_for_each_layout(tmp_path: Path, monkeypatch) -> None:
    mod = load_layout_module()
    seen_seeds = []

    class FakeMicro:
        def run_microtrain(self, **kwargs):
            seen_seeds.append(kwargs["seed"])
            site_spec = kwargs["dsqg_w_sites"]
            sites = ["final"] if site_spec == "final" else ["layer_2", "final"]
            return {
                "pass": True,
                "dsqg_w_sites": sites,
                "dsqg_w_site_count": len(sites),
                "dsqg_w_trainable_param_count": 10 * len(sites),
                "train_examples": kwargs["train_size"],
                "val_examples": kwargs["val_size"],
                "steps": kwargs["steps"],
                "lr": kwargs["lr"],
                "train_loss_delta": -0.1,
                "val_loss_delta": -0.1,
                "val_loss_final": 9.0,
                "val_mean_rank_initial": 100.0,
                "val_mean_rank_final": 10.0,
                "val_mean_rank_delta": -90.0,
                "val_top1_acc_initial": 0.0,
                "val_top1_acc_final": 0.1,
                "val_top1_acc_delta": 0.1,
                "val_top5_acc_initial": 0.0,
                "val_top5_acc_final": 0.2,
                "val_top5_acc_delta": 0.2,
                "changed_dsqg_w_param_count": 1,
                "changed_frozen_param_count": 0,
                "checkpoint_roundtrip_loss_delta": 0.0,
                "report_path": str(kwargs["output_dir"] / "microtrain_report.json"),
                "checkpoint": {"state_path": str(kwargs["output_dir"] / "checkpoint" / "dsqg_w_state.pt")},
            }

    monkeypatch.setattr(mod, "load_microtrain_module", lambda: FakeMicro())

    mod.run_layout_experiment(
        tokenizer_path=TOKENIZER,
        output_dir=tmp_path / "layout",
        site_specs=["final", "2,final"],
        train_size=8,
        val_size=4,
        steps=1,
        lr=1e-3,
        seed=1234,
    )

    assert seen_seeds == [1234, 1234]
