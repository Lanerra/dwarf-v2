from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hisa_dsqg_bwd_tile_overnight_ladder.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("bwd_tile_overnight_ladder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ladder_plan_uses_a_single_400k_wsd_horizon_with_disjoint_candidate_tranches() -> None:
    runner = load_runner()

    stages = runner.build_ladder_stages()
    by_id = {stage.stage_id: stage for stage in stages}

    assert [stage.stage_id for stage in stages] == [
        "canonical_20k",
        "bwd16_w4_20k",
        "canonical_a_50k",
        "bwd16_w4_50k",
        "canonical_b_50k",
        "bwd16_w4_100k",
        "bwd16_w4_200k",
        "bwd16_w4_400k",
    ]
    assert runner.WSD_TOTAL_STEPS == 25_000
    assert runner.WSD_PHASES == {"warmup_steps": 1_250, "stable_steps": 20_000, "decay_steps": 3_750}
    assert [
        (by_id[name].train_seqs, by_id[name].train_seq_offset, by_id[name].schedule_step_offset)
        for name in ("bwd16_w4_50k", "bwd16_w4_100k", "bwd16_w4_200k", "bwd16_w4_400k")
    ] == [
        (50_000, 0, 0),
        (50_000, 50_000, 3_125),
        (100_000, 100_000, 6_250),
        (200_000, 200_000, 12_500),
    ]


def test_variant_environment_is_hermetic_and_overlay_only_changes_kernel_surface() -> None:
    runner = load_runner()

    canonical = runner.build_variant_env(runner.VARIANTS["canonical"])
    candidate = runner.build_variant_env(runner.VARIANTS["bwd16_w4"])

    assert canonical["DWARF_DSQG_KERNEL_DIR"] == str((ROOT / "kernels").resolve())
    assert "DWARF_DSQG_BWD_BLOCK_N" not in canonical
    assert candidate["DWARF_DSQG_KERNEL_DIR"] == str((ROOT / "kernel_overlays" / "bwd_tile_tuning").resolve())
    assert candidate["DWARF_DSQG_BWD_BLOCK_N"] == "16"
    assert candidate["DWARF_DSQG_BWD_NUM_WARPS"] == "4"
    assert candidate["DWARF_DSQG_BWD_NUM_STAGES"] == "2"


def test_candidate_scale_stages_require_the_50k_quality_gate() -> None:
    runner = load_runner()

    assert runner.next_stage_allowed(stage_id="bwd16_w4_100k", gates={"overlay_50k": True})
    assert not runner.next_stage_allowed(stage_id="bwd16_w4_100k", gates={"overlay_50k": False})
    assert runner.next_stage_allowed(stage_id="canonical_a_50k", gates={})


def test_stage_contract_writes_json_with_the_overlay_kernel_path(tmp_path: Path) -> None:
    runner = load_runner()
    stage = next(stage for stage in runner.build_ladder_stages() if stage.stage_id == "bwd16_w4_20k")

    config = runner.build_stage_config(stage=stage, out_root=tmp_path, gpu="0")

    assert Path(config["contract_path"]).is_file()


def test_dry_run_completes_all_conditional_stage_contracts(tmp_path: Path) -> None:
    runner = load_runner()

    payload = runner.run_ladder(
        runner.parse_args(["--out-root", str(tmp_path), "--dry-run", "--skip-artifact-sha256"])
    )

    assert payload["status"] == "passed"
    assert set(payload["stages"]) == {stage.stage_id for stage in runner.build_ladder_stages()}
