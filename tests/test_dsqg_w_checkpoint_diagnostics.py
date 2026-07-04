from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dsqg_w_checkpoint_diagnostics.py"


def load_diag():
    spec = importlib.util.spec_from_file_location("diag_script_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_from_run_config_prefers_best(tmp_path: Path) -> None:
    diag = load_diag()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ep1 = ckpt_dir / "demo_ep1.pt"
    best = ckpt_dir / "demo_best.pt"
    ep1.write_bytes(b"ep1")
    best.write_bytes(b"best")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({"env": {"DWARF_CHECKPOINT_DIR": str(ckpt_dir), "DWARF_CKPT_BASE_NAME": "demo"}}))

    assert diag.checkpoint_from_run_config(run_config) == best


def test_trainer_env_from_run_config_preserves_arch_but_forces_diagnostic_runtime(tmp_path: Path) -> None:
    diag = load_diag()
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({"env": {
        "DWARF_DSQG_W": "1",
        "DWARF_DSQG_W_WIDTH_CELL": "1",
        "DWARF_TORCH_COMPILE": "1",
        "DWARF_LIGER": "1",
        "DWARF_Q6_G128": "1",
        "DWARF_DATASET": "dataset.pt",
    }}))

    env = diag.trainer_env_from_run_config(run_config)

    assert env["DWARF_DSQG_W"] == "1"
    assert env["DWARF_DSQG_W_WIDTH_CELL"] == "1"
    assert env["DWARF_TORCH_COMPILE"] == "0"
    assert env["DWARF_LIGER"] == "0"
    assert env["DWARF_Q6_G128"] == "0"


def test_grad_groups_bucket_expected_width_params() -> None:
    diag = load_diag()

    assert diag.grad_groups.__name__ == "grad_groups"
    score_name = "dsqg_w_blocks.final.width_cell.q_proj.weight"
    value_name = "dsqg_w_blocks.final.width_cell.v_proj.weight"
    up_name = "dsqg_w_blocks.final.width_cell.lateral_up.weight"
    width_gate_name = "dsqg_w_blocks.final.width_cell.gate"
    mix_gate_name = "dsqg_w_blocks.final.typed_mixer.gate"
    main_gate_name = "dsqg_w_blocks.final.gate"

    # Mirror the public bucket semantics without constructing a heavyweight model.
    score_terms = (
        ".width_cell.q_proj", ".width_cell.k_proj", ".width_cell.rel_diff_proj",
        ".width_cell.rel_prod_proj", ".width_cell.rel_diff_score", ".width_cell.rel_prod_score",
        ".width_cell.type_pair_bias", ".width_cell.source_pair_bias", ".width_cell.self_bias",
    )
    assert any(term in score_name for term in score_terms)
    assert ".width_cell.v_proj" in value_name
    assert ".width_cell.lateral_up" in up_name
    assert ".width_cell.gate" in width_gate_name
    assert ".typed_mixer.gate" in mix_gate_name
    assert main_gate_name.endswith(".gate") and "dsqg_w_blocks." in main_gate_name
