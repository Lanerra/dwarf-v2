from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("v2_wsd_schedule_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wsd_uses_explicit_global_horizon_and_continuation_offset() -> None:
    trainer = load_trainer()

    config = trainer.build_lr_schedule_config(
        run_steps=3_125,
        environ={
            "DWARF_LR_SCHEDULE": "wsd",
            "DWARF_SCHEDULE_TOTAL_STEPS": "12500",
            "DWARF_WSD_WARMUP_STEPS": "625",
            "DWARF_WSD_STABLE_STEPS": "10000",
            "DWARF_WSD_DECAY_STEPS": "1875",
            "DWARF_SCHEDULE_STEP_OFFSET": "3125",
        },
    )

    assert config.kind == "wsd"
    assert config.total_steps == 12_500
    assert config.warmup_steps == 625
    assert config.stable_steps == 10_000
    assert config.decay_steps == 1_875
    assert config.step_offset == 3_125


def test_wsd_is_linear_warmup_then_stable_then_cosine_decay() -> None:
    trainer = load_trainer()
    config = trainer.build_lr_schedule_config(
        run_steps=10,
        environ={
            "DWARF_LR_SCHEDULE": "wsd",
            "DWARF_SCHEDULE_TOTAL_STEPS": "10",
            "DWARF_WSD_WARMUP_STEPS": "2",
            "DWARF_WSD_STABLE_STEPS": "5",
            "DWARF_WSD_DECAY_STEPS": "3",
        },
    )

    assert trainer.lr_schedule_multiplier(step=0, config=config, min_lr_ratio=0.1) == pytest.approx(0.5)
    assert trainer.lr_schedule_multiplier(step=1, config=config, min_lr_ratio=0.1) == pytest.approx(1.0)
    assert trainer.lr_schedule_multiplier(step=2, config=config, min_lr_ratio=0.1) == pytest.approx(1.0)
    assert trainer.lr_schedule_multiplier(step=6, config=config, min_lr_ratio=0.1) == pytest.approx(1.0)
    assert trainer.lr_schedule_multiplier(step=9, config=config, min_lr_ratio=0.1) == pytest.approx(0.1)


def test_wsd_rejects_a_run_that_would_cross_its_declared_horizon() -> None:
    trainer = load_trainer()

    with pytest.raises(ValueError, match="exceeds schedule horizon"):
        trainer.build_lr_schedule_config(
            run_steps=3_125,
            environ={
                "DWARF_LR_SCHEDULE": "wsd",
                "DWARF_SCHEDULE_TOTAL_STEPS": "5000",
                "DWARF_WSD_WARMUP_STEPS": "250",
                "DWARF_WSD_STABLE_STEPS": "4000",
                "DWARF_WSD_DECAY_STEPS": "750",
                "DWARF_SCHEDULE_STEP_OFFSET": "3125",
            },
        )


def test_trainer_scheduler_uses_the_global_schedule_contract() -> None:
    source = TRAINER.read_text(encoding="utf-8")

    assert "lr_schedule = build_lr_schedule_config(run_steps=run_total_steps)" in source
    assert "return lr_schedule_multiplier(step=step, config=lr_schedule, min_lr_ratio=MIN_LR_RATIO)" in source


def test_wsd_continuation_rejects_double_loading_scheduler_progress() -> None:
    trainer = load_trainer()
    config = trainer.build_lr_schedule_config(
        run_steps=3_125,
        environ={
            "DWARF_LR_SCHEDULE": "wsd",
            "DWARF_SCHEDULE_TOTAL_STEPS": "12500",
            "DWARF_WSD_WARMUP_STEPS": "625",
            "DWARF_WSD_STABLE_STEPS": "10000",
            "DWARF_WSD_DECAY_STEPS": "1875",
            "DWARF_SCHEDULE_STEP_OFFSET": "3125",
        },
    )

    with pytest.raises(ValueError, match="DWARF_SKIP_SCHED=1"):
        trainer.validate_schedule_resume(config=config, resume_path="checkpoint.pt", skip_scheduler_state=False)

    trainer.validate_schedule_resume(config=config, resume_path="checkpoint.pt", skip_scheduler_state=True)
