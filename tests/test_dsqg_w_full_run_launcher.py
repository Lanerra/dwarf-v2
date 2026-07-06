from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_dsqg_w_full_training.py"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("run_dsqg_w_full_training", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_full_run_launcher_builds_winning_layout_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "full_run",
        run_name="unit",
        max_acc_steps=3,
        train_seqs=12,
        val_seqs=8,
        batch_size=1,
        grad_accum=1,
    )

    env = cfg["env"]
    assert cfg["run_name"] == "unit"
    assert cfg["command"][-1] == "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["DWARF_DSQG_W"] == "1"
    assert env["DWARF_DSQG_W_SITES"] == "2,6,final"
    assert env["DWARF_DSQG_W_MAX_CANDIDATES"] == "16"
    assert env["DWARF_DSQG_W_BOTTLENECK"] == "64"
    assert env["DWARF_DSQG_W_GATE_INIT"] == "-2.5"
    assert env["DWARF_DSQG_W_GATE_LR_MULT"] == "1.25"
    assert env["DWARF_DSQG_W_FUSE_INIT_STD"] == "0.02"
    assert env["DWARF_DSQG_W_DSR_CANDIDATES"] == "1"
    assert env["DWARF_DSQG_W_LOCAL_OFFSETS"] == "none"
    assert env["DWARF_DSQG_W_LONG_OFFSETS"] == "none"
    assert env["DWARF_HISA_STAGE2_REP_R"] == "4"
    assert env["DWARF_DSQG_W_WIDTH_CELL"] == "0"
    assert env["DWARF_DSQG_W_QUESTION"] == "1"
    assert env["DWARF_DSQG_W_HISA_L3"] == "1"
    assert env["DWARF_MAX_ACC_STEPS"] == "3"
    assert env["DWARF_MAX_TRAIN_SEQS"] == "12"
    assert env["DWARF_MAX_VAL_SEQS"] == "8"
    assert env["DWARF_BS"] == "1"
    assert env["DWARF_GA"] == "1"
    assert env["DWARF_CKPT_BASE_NAME"].endswith("unit")
    assert Path(env["DWARF_CHECKPOINT_DIR"]).name == "checkpoints"


def test_full_run_launcher_can_enable_width_cell_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "width_run",
        run_name="width_unit",
        width_cell=True,
        width_bottleneck=12,
        width_gate_init=-3.5,
        width_aux_weight=0.25,
        width_entropy_floor=1.25,
        width_entropy_weight=0.4,
    )

    env = cfg["env"]
    assert env["DWARF_DSQG_W_WIDTH_CELL"] == "1"
    assert env["DWARF_DSQG_W_WIDTH_BOTTLENECK"] == "12"
    assert env["DWARF_DSQG_W_WIDTH_GATE_INIT"] == "-3.5"
    assert env["DWARF_DSQG_W_WIDTH_AUX_WEIGHT"] == "0.25"
    assert env["DWARF_DSQG_W_WIDTH_ENTROPY_FLOOR"] == "1.25"
    assert env["DWARF_DSQG_W_WIDTH_ENTROPY_WEIGHT"] == "0.4"


def test_full_run_launcher_can_enable_sourcewise_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "sourcewise_run",
        run_name="sourcewise_unit",
        sourcewise=True,
    )

    assert cfg["env"]["DWARF_DSQG_W_SOURCEWISE"] == "1"
    assert cfg["env"]["DWARF_DSQG_W_TRITON_SOURCEWISE"] == "0"


def test_full_run_launcher_can_enable_triton_sourcewise_prototype_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "triton_sourcewise_run",
        run_name="triton_sourcewise_unit",
        sourcewise=True,
        triton_sourcewise=True,
    )

    assert cfg["env"]["DWARF_DSQG_W_SOURCEWISE"] == "1"
    assert cfg["env"]["DWARF_DSQG_W_TRITON_SOURCEWISE"] == "1"


def test_full_run_launcher_can_enable_fast_detached_evidence_mean_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "fast_mean_run",
        run_name="fast_mean_unit",
        sourcewise=True,
        triton_sourcewise=True,
        detach_recomposer=True,
        fast_evidence_mean=True,
        k_question=0,
        k_hisa_evidence=0,
        k_l3_skip=0,
    )

    env = cfg["env"]
    assert env["DWARF_DSQG_W_SOURCEWISE"] == "1"
    assert env["DWARF_DSQG_W_TRITON_SOURCEWISE"] == "1"
    assert env["DWARF_DSQG_W_DETACH_RECOMPOSER"] == "1"
    assert env["DWARF_DSQG_W_FAST_EVIDENCE_MEAN"] == "1"
    assert env["DWARF_DSQG_W_K_QUESTION"] == "0"
    assert env["DWARF_DSQG_W_K_HISA_EVIDENCE"] == "0"
    assert env["DWARF_DSQG_W_K_L3_SKIP"] == "0"


def test_full_run_launcher_can_disable_dsqg_w_for_backbone_controls(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "d_only",
        run_name="d_only_unit",
        dsqg_w=False,
        hisa_stage2_rep_r=4,
    )

    env = cfg["env"]
    assert env["DWARF_DSQG_W"] == "0"
    assert env["DWARF_HISA_STAGE2_REP_R"] == "4"


def test_full_run_launcher_can_opt_into_rowmax_stage2_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "rowmax_diagnostic",
        run_name="rowmax_unit",
        hisa_stage2_rep_r=0,
    )

    assert cfg["env"]["DWARF_HISA_STAGE2_REP_R"] == "0"


def test_full_run_launcher_can_enable_typed_mixer_and_query_rep_hisa_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "combined_run",
        run_name="combined_unit",
        typed_mixer=True,
        typed_mixer_bottleneck=12,
        gate_init=-2.0,
        fuse_init_std=0.03,
        typed_mixer_gate_init=-2.0,
        query_type_bias=True,
        typed_hisa_reps=True,
        local_offsets="1,2",
        long_offsets="none",
        hisa_stage2_rep_r=4,
    )

    env = cfg["env"]
    assert env["DWARF_DSQG_W_TYPED_MIXER"] == "1"
    assert env["DWARF_DSQG_W_GATE_INIT"] == "-2.0"
    assert env["DWARF_DSQG_W_FUSE_INIT_STD"] == "0.03"
    assert env["DWARF_DSQG_W_TYPED_MIXER_BOTTLENECK"] == "12"
    assert env["DWARF_DSQG_W_TYPED_MIXER_GATE_INIT"] == "-2.0"
    assert env["DWARF_DSQG_W_QUERY_TYPE_BIAS"] == "1"
    assert env["DWARF_DSQG_W_TYPED_HISA_REPS"] == "1"
    assert env["DWARF_DSQG_W_DSR_CANDIDATES"] == "1"
    assert env["DWARF_DSQG_W_LOCAL_OFFSETS"] == "1,2"
    assert env["DWARF_DSQG_W_LONG_OFFSETS"] == "none"
    assert env["DWARF_HISA_STAGE2_REP_R"] == "4"


def test_full_run_launcher_can_enable_evidence_binding_hub_env(tmp_path: Path) -> None:
    mod = load_launcher_module()

    cfg = mod.build_run_config(
        output_dir=tmp_path / "ebh_run",
        run_name="ebh_unit",
        evidence_binding_hub=True,
        ebh_bottleneck=48,
        ebh_gate_init=-2.0,
        ebh_phase_bands=3,
        ebh_score_features=False,
        ebh_sourcewise_packet=True,
    )

    env = cfg["env"]
    assert env["DWARF_DSQG_W_EVIDENCE_BINDING_HUB"] == "1"
    assert env["DWARF_DSQG_W_EBH_BOTTLENECK"] == "48"
    assert env["DWARF_DSQG_W_EBH_GATE_INIT"] == "-2.0"
    assert env["DWARF_DSQG_W_EBH_PHASE_BANDS"] == "3"
    assert env["DWARF_DSQG_W_EBH_SCORE_FEATURES"] == "0"
    assert env["DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"] == "1"


def test_full_run_launcher_can_configure_cpt_resume_env(tmp_path: Path) -> None:
    mod = load_launcher_module()
    resume = tmp_path / "seed.pt"
    dataset = tmp_path / "cpt_8192.pt"

    cfg = mod.build_run_config(
        output_dir=tmp_path / "cpt_run",
        run_name="cpt_unit",
        dataset=dataset,
        seq_len=8192,
        resume=resume,
        skip_opt=True,
        skip_sched=True,
        lr=2e-5,
        min_lr_ratio=0.5,
        lr_warmup_steps=0,
        hisa_top_m=16,
        batch_size=1,
        grad_accum=16,
    )

    env = cfg["env"]
    assert env["DWARF_DATASET"] == str(dataset)
    assert env["DWARF_SEQ_LEN"] == "8192"
    assert env["DWARF_RESUME"] == str(resume)
    assert env["DWARF_SKIP_OPT"] == "1"
    assert env["DWARF_SKIP_SCHED"] == "1"
    assert env["DWARF_LR"] == "2e-05"
    assert env["DWARF_MIN_LR_RATIO"] == "0.5"
    assert env["DWARF_LR_WARMUP_STEPS"] == "0"
    assert env["DWARF_HISA_TOP_M"] == "16"



def test_full_run_launcher_dry_run_writes_config_without_executing(tmp_path: Path) -> None:
    mod = load_launcher_module()

    report = mod.main([
        "--output-dir", str(tmp_path / "dry"),
        "--run-name", "dry_unit",
        "--max-acc-steps", "2",
        "--train-seqs", "8",
        "--val-seqs", "4",
        "--dry-run",
    ])

    assert report["pass"] is True
    assert report["executed"] is False
    assert Path(report["config_path"]).exists()
    saved = json.loads(Path(report["config_path"]).read_text())
    assert saved["env"]["DWARF_DSQG_W_SITES"] == "2,6,final"
    assert saved["env"]["DWARF_MAX_ACC_STEPS"] == "2"
    assert Path(saved["stdout_path"]).name == "trainer.stdout.log"
