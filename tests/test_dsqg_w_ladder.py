from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSER_SCRIPT = ROOT / "scripts/parse_dsqg_w_ladder.py"
RUNNER_SCRIPT = ROOT / "scripts/run_dsqg_w_ladder.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_trainer_stdout_extracts_health_and_w_metrics(tmp_path: Path) -> None:
    mod = load_module(PARSER_SCRIPT, "parse_dsqg_w_ladder_test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stdout = run_dir / "trainer.stdout.log"
    stdout.write_text(
        "\n".join(
            [
                "  GPU: NVIDIA GeForce RTX 4090",
                "  DSQG-W recomposer sites=final: enabled J<=16 bottleneck=128 gate_init=-2.0 fuse_init_std=0.02 candidates=DSR_SELECTED_QUESTION_L3_SKIP_NULL",
                "  HISA Stage-2 selector: rep_r=4 (0=rowmax baseline)",
                "  train: 128 seqs  val: 64 seqs  host_dtype=torch.int32 train_real=161/262,016 (0.06%) val_real=82/131,008 (0.06%)",
                "  [ep1 step 8/16] ce=9.5 se_max=0.1 grad_norm=1.0 lr=1.0e-4 1234 tok/s routing_ent=2.5 w_gate=0.119 w_dx=0.101 w_hisa=0.62 w_score=0.002 w_smean=0.21 w_mix_gate=0.119",
                "  [ep1 step 16/16] ce=8.5 se_max=0.1 grad_norm=1.0 lr=1.0e-4 2345 tok/s routing_ent=2.4 w_gate=0.120 w_dx=0.111 w_hisa=0.61 w_score=0.003 w_smean=0.22 w_mix_gate=0.121 w_fast=1.000 w_det=1.000 w_j=1.000",
                "Ep 1/1 | Val PPL 12345.67 *",
                "  Passkey mean=12.5%",
                "  peak_vram=3456MB  elapsed=42s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = mod.parse_run_dir(
        run_dir,
        expected_steps=16,
        expected_gpu="RTX 4090",
        require_dsqg_w=True,
        expected_stage2_rep_r=4,
        returncode=0,
    )

    assert result["health"]["pass"] is True
    metrics = result["metrics"]
    assert metrics["final_step"] == 16
    assert metrics["final_ce"] == 8.5
    assert metrics["val_ppl"] == 12345.67
    assert metrics["peak_vram_mb"] == 3456
    assert metrics["w_dx"] == 0.111
    assert metrics["w_fast"] == 1.0
    assert metrics["w_det"] == 1.0
    assert metrics["w_j"] == 1.0
    assert metrics["avg_logged_tok_s"] == (1234 + 2345) / 2


def test_ladder_runner_dry_run_writes_variant_configs(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_dsqg_w_ladder_test")
    dataset = tmp_path / "same_family.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "ladder"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--lanes",
            "same_family",
            "--same-family-dataset",
            str(dataset),
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
    manifest = json.loads((out_root / "ladder_manifest.json").read_text(encoding="utf-8"))
    assert [v["variant_id"] for v in manifest["variants"]] == [
        "A_dsr_rowmax",
        "B_dsr_rep4",
        "C_dfed_w_min",
        "D_dfed_w_full",
    ]
    assert (out_root / "same_family_summary.json").exists()
    cfg_a = json.loads((out_root / "same_family/A_dsr_rowmax/run_config.json").read_text(encoding="utf-8"))
    cfg_d = json.loads((out_root / "same_family/D_dfed_w_full/run_config.json").read_text(encoding="utf-8"))
    assert cfg_a["env"]["DWARF_DSQG_W"] == "0"
    assert cfg_a["env"]["DWARF_HISA_STAGE2_REP_R"] == "0"
    assert cfg_d["env"]["DWARF_DSQG_W"] == "1"
    assert cfg_d["env"]["DWARF_DSQG_W_DSR_CANDIDATES"] == "1"
    assert cfg_d["env"]["DWARF_DSQG_W_LOCAL_OFFSETS"] == "none"
    assert cfg_d["env"]["DWARF_DSQG_W_TYPED_MIXER"] == "1"


def test_ladder_runner_variant_ids_filter_tighter_attribution_set(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_dsqg_w_ladder_filter_test")
    dataset = tmp_path / "same_family.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "tight_ladder"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--lanes",
            "same_family",
            "--same-family-dataset",
            str(dataset),
            "--variant-ids",
            "B_dsr_rep4,C_dfed_w_min,D_dfed_w_full",
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
    manifest = json.loads((out_root / "ladder_manifest.json").read_text(encoding="utf-8"))
    assert [v["variant_id"] for v in manifest["variants"]] == [
        "B_dsr_rep4",
        "C_dfed_w_min",
        "D_dfed_w_full",
    ]
    assert not (out_root / "same_family/A_dsr_rowmax/run_config.json").exists()
    assert (out_root / "same_family/B_dsr_rep4/run_config.json").exists()
    assert (out_root / "same_family/C_dfed_w_min/run_config.json").exists()
    assert (out_root / "same_family/D_dfed_w_full/run_config.json").exists()


def test_ladder_runner_can_select_fast_aligned_l3_reset_variants(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_dsqg_w_ladder_fast_l3_test")
    dataset = tmp_path / "pretrain.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "fast_l3_ladder"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--lanes",
            "pretrain",
            "--pretrain-dataset",
            str(dataset),
            "--variant-ids",
            "B_dsr_rep4,E_fast_l3_3site,F_fast_l3_final",
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
    manifest = json.loads((out_root / "ladder_manifest.json").read_text(encoding="utf-8"))
    assert [v["variant_id"] for v in manifest["variants"]] == [
        "B_dsr_rep4",
        "E_fast_l3_3site",
        "F_fast_l3_final",
    ]
    cfg_b = json.loads((out_root / "pretrain/B_dsr_rep4/run_config.json").read_text(encoding="utf-8"))
    cfg_e = json.loads((out_root / "pretrain/E_fast_l3_3site/run_config.json").read_text(encoding="utf-8"))
    cfg_f = json.loads((out_root / "pretrain/F_fast_l3_final/run_config.json").read_text(encoding="utf-8"))
    assert cfg_b["env"]["DWARF_DSQG_W"] == "0"
    assert cfg_b["env"]["DWARF_HISA_STAGE2_REP_R"] == "4"
    for cfg in (cfg_e, cfg_f):
        env = cfg["env"]
        assert env["DWARF_DSQG_W"] == "1"
        assert env["DWARF_HISA_STAGE2_REP_R"] == "4"
        assert env["DWARF_DSQG_W_SOURCEWISE"] == "1"
        assert env["DWARF_DSQG_W_TRITON_SOURCEWISE"] == "1"
        assert env["DWARF_DSQG_W_DETACH_RECOMPOSER"] == "1"
        assert env["DWARF_DSQG_W_FAST_EVIDENCE_MEAN"] == "1"
        assert env["DWARF_DSQG_W_K_QUESTION"] == "0"
        assert env["DWARF_DSQG_W_K_HISA_EVIDENCE"] == "0"
        assert env["DWARF_DSQG_W_K_L3_SKIP"] == "0"
        assert env["DWARF_DSQG_W_LOCAL_OFFSETS"] == "none"
        assert env["DWARF_DSQG_W_LONG_OFFSETS"] == "none"
    assert cfg_e["env"]["DWARF_DSQG_W_SITES"] == "2,6,final"
    assert cfg_f["env"]["DWARF_DSQG_W_SITES"] == "final"


def test_ladder_runner_can_select_pure_dsqg_control(tmp_path: Path) -> None:
    mod = load_module(RUNNER_SCRIPT, "run_dsqg_w_ladder_pure_test")
    dataset = tmp_path / "pretrain.pt"
    dataset.write_bytes(b"not loaded during dry-run")
    out_root = tmp_path / "pure_ladder"

    result = mod.main(
        [
            "--out-root",
            str(out_root),
            "--lanes",
            "pretrain",
            "--pretrain-dataset",
            str(dataset),
            "--variant-ids",
            "P_pure_dsqg_v1,D_dfed_w_full",
            "--max-acc-steps",
            "3",
            "--train-seqs",
            "4",
            "--val-seqs",
            "2",
            "--sites",
            "6,final",
            "--dry-run",
        ]
    )

    assert result["pass"] is True
    manifest = json.loads((out_root / "ladder_manifest.json").read_text(encoding="utf-8"))
    assert [v["variant_id"] for v in manifest["variants"]] == ["P_pure_dsqg_v1", "D_dfed_w_full"]
    cfg_p = json.loads((out_root / "pretrain/P_pure_dsqg_v1/run_config.json").read_text(encoding="utf-8"))
    cfg_d = json.loads((out_root / "pretrain/D_dfed_w_full/run_config.json").read_text(encoding="utf-8"))
    assert cfg_p["env"]["DWARF_PURE_DSQG"] == "1"
    assert cfg_p["env"]["DWARF_DSQG_W"] == "0"
    assert cfg_d["env"]["DWARF_PURE_DSQG"] == "0"
    assert cfg_d["env"]["DWARF_DSQG_W_SITES"] == "6,final"
