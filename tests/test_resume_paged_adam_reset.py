from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train" / "train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("trainer_resume_opt", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = []
        self.loaded: list[dict[str, object]] = []

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loaded.append(state)

    def state_dict(self) -> dict[str, object]:
        return {"state": {}}

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none

    def step(self) -> None:
        pass


def test_multi_optimizer_can_preserve_muon_while_explicitly_resetting_paged_adam_state() -> None:
    trainer = load_trainer()
    muon = FakeOptimizer()
    adamw = FakeOptimizer()
    optimizer = trainer._MultiOptimizer((("muon", muon), ("adamw", adamw)))
    state = {
        "kind": "multi",
        "optimizers": [
            {"name": "muon", "state_dict": {"state": {1: {"momentum_buffer": "muon"}}}},
            {"name": "adamw", "state_dict": {"state": {2: {"__bnb_optimizer_quant_state__": "paged"}}}},
        ],
    }

    outcome = optimizer.load_state_dict(state, skip_state_names={"adamw"})

    assert outcome == {"loaded": ["muon"], "skipped": ["adamw"]}
    assert muon.loaded == [{"state": {1: {"momentum_buffer": "muon"}}}]
    assert adamw.loaded == []


def test_multi_optimizer_keeps_existing_full_state_restore_by_default() -> None:
    trainer = load_trainer()
    muon = FakeOptimizer()
    adamw = FakeOptimizer()
    optimizer = trainer._MultiOptimizer((("muon", muon), ("adamw", adamw)))
    state = {
        "kind": "multi",
        "optimizers": [
            {"name": "muon", "state_dict": {"state": {}}},
            {"name": "adamw", "state_dict": {"state": {}}},
        ],
    }

    outcome = optimizer.load_state_dict(state)

    assert outcome == {"loaded": ["muon", "adamw"], "skipped": []}
    assert len(muon.loaded) == len(adamw.loaded) == 1
