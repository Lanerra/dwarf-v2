from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = ROOT / "scripts/run_hisa_stage2_rep_r_sweep.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hisa_stage2_rep_r_sweep_dry_run_writes_rep_configs(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_hisa_stage2_rep_r_sweep_test")
    dataset = tmp_path / "dataset.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "rep_sweep"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--dataset",
            str(dataset),
            "--rep-rs",
            "0,1,4",
            "--max-acc-steps",
            "3",
            "--train-seqs",
            "4",
            "--val-seqs",
            "2",
            "--dry-run",
        ]
    )

    assert result["pass"] is True
    manifest = json.loads((out_root / "sweep_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rep_rs"] == [0, 1, 4]
    assert manifest["args"]["dsqg_w"] is False
    assert (out_root / "summary.md").exists()
    results = json.loads((out_root / "sweep_results.json").read_text(encoding="utf-8"))
    assert [row["rep_r"] for row in results["summary"]["rows"]] == [0, 1, 4]

    cfg0 = json.loads((out_root / "rep0/run_config.json").read_text(encoding="utf-8"))
    cfg1 = json.loads((out_root / "rep1/run_config.json").read_text(encoding="utf-8"))
    cfg4 = json.loads((out_root / "rep4/run_config.json").read_text(encoding="utf-8"))
    assert cfg0["env"]["DWARF_DSQG_W"] == "0"
    assert cfg0["env"]["DWARF_HISA_STAGE2_REP_R"] == "0"
    assert cfg1["env"]["DWARF_HISA_STAGE2_REP_R"] == "1"
    assert cfg4["env"]["DWARF_HISA_STAGE2_REP_R"] == "4"
    assert cfg4["env"]["HISA_TELEMETRY"] == "1"
    assert cfg4["env"]["DWARF_LIGER"] == "1"
    assert cfg4["env"]["DWARF_CKPT"] == "none"


def test_hisa_stage2_rep_r_sweep_dsqg_w_dry_run_sets_w_env(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_hisa_stage2_rep_r_sweep_w_test")
    dataset = tmp_path / "dataset.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "rep_sweep_w"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--dataset",
            str(dataset),
            "--rep-rs",
            "2",
            "--dsqg-w",
            "--typed-mixer",
            "--query-type-bias",
            "--typed-hisa-reps",
            "--dry-run",
        ]
    )

    assert result["pass"] is True
    cfg = json.loads((out_root / "rep2/run_config.json").read_text(encoding="utf-8"))
    assert cfg["env"]["DWARF_DSQG_W"] == "1"
    assert cfg["env"]["DWARF_HISA_STAGE2_REP_R"] == "2"
    assert cfg["env"]["DWARF_DSQG_W_TYPED_MIXER"] == "1"
    assert cfg["env"]["DWARF_DSQG_W_QUERY_TYPE_BIAS"] == "1"
    assert cfg["env"]["DWARF_DSQG_W_TYPED_HISA_REPS"] == "1"
