from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_corrected_dsqg_w_200step_matrix.py"


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("run_corrected_dsqg_w_200step_matrix_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_corrected_matrix_width_aux_only_enabled_for_width_variants(tmp_path: Path) -> None:
    mod = load_matrix_module()
    out_root = tmp_path / "matrix"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--variant-ids",
            "A_no_w,D_candidate_final,G_candidate_final_full",
            "--width-aux-weight",
            "0.01",
            "--max-acc-steps",
            "2",
            "--train-seqs",
            "4",
            "--val-seqs",
            "2",
            "--dry-run",
        ]
    )

    assert result["pass"] is True
    cfg_a = json.loads((out_root / "pretrain/A_no_w/run_config.json").read_text(encoding="utf-8"))
    cfg_d = json.loads((out_root / "pretrain/D_candidate_final/run_config.json").read_text(encoding="utf-8"))
    cfg_g = json.loads((out_root / "pretrain/G_candidate_final_full/run_config.json").read_text(encoding="utf-8"))

    assert cfg_a["env"]["DWARF_DSQG_W"] == "0"
    assert cfg_a["env"]["DWARF_DSQG_W_WIDTH_CELL"] == "0"
    assert cfg_a["env"]["DWARF_DSQG_W_WIDTH_AUX_WEIGHT"] == "0.0"

    assert cfg_d["env"]["DWARF_DSQG_W"] == "1"
    assert cfg_d["env"]["DWARF_DSQG_W_WIDTH_CELL"] == "0"
    assert cfg_d["env"]["DWARF_DSQG_W_WIDTH_AUX_WEIGHT"] == "0.0"

    assert cfg_g["env"]["DWARF_DSQG_W"] == "1"
    assert cfg_g["env"]["DWARF_DSQG_W_WIDTH_CELL"] == "1"
    assert cfg_g["env"]["DWARF_DSQG_W_WIDTH_AUX_WEIGHT"] == "0.01"
    assert cfg_g["env"]["DWARF_DSQG_W_GATE_LR_MULT"] == "1.25"
