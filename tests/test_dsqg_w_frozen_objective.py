from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frozen_trunk_objective_dsqg_w.py"


def load_objective_module():
    spec = importlib.util.spec_from_file_location("frozen_trunk_objective_dsqg_w", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def tiny_batch(mod, *, batch: int = 1, seq: int = 8):
    input_ids = torch.randint(0, 128, (batch, seq), dtype=torch.long)
    labels = input_ids.clone()
    answer_mask = torch.zeros(batch, seq, dtype=torch.bool)
    answer_mask[:, -1] = True
    return mod.FrozenDSQGWBatch(
        input_ids=input_ids,
        labels=labels,
        answer_mask=answer_mask,
        question_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.long).expand(batch, -1),
        hisa_evidence_indices=torch.tensor([[1, 2, 3, 4]], dtype=torch.long).expand(batch, -1),
        l3_skip_indices=torch.tensor([[5, 6]], dtype=torch.long).expand(batch, -1),
    )


def test_frozen_objective_smoke_is_disabled_by_default() -> None:
    mod = load_objective_module()

    report = mod.run_smoke_objective(enable=False)

    assert report["enabled"] is False
    assert report["pass"] is True
    assert report["skipped"] is True
    assert report["reason"] == "DWARF_DSQG_W_FROZEN_OBJECTIVE is disabled"


def test_prepare_model_freezes_trunk_and_leaves_only_dsqg_w_trainable(monkeypatch) -> None:
    mod = load_objective_module()
    trainer = mod.load_trainer(enable_objective=True, suffix="freeze_test")
    model = mod.make_tiny_model(trainer, vocab_size=128, ffn_dim=64, seq_len=16)

    counts = mod.prepare_model_for_frozen_dsqg_w_objective(model)

    assert counts["trainable_param_count"] > 0
    assert counts["frozen_param_count"] > counts["trainable_param_count"]
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(name.startswith("dsqg_w.") for name in trainable_names)
    assert not model.embedding.weight.requires_grad
    assert not model.out.weight.requires_grad


def test_answer_only_ce_objective_backprops_only_dsqg_w_parameters(monkeypatch) -> None:
    mod = load_objective_module()
    trainer = mod.load_trainer(enable_objective=True, suffix="objective_test")
    model = mod.make_tiny_model(trainer, vocab_size=128, ffn_dim=64, seq_len=16)
    mod.prepare_model_for_frozen_dsqg_w_objective(model)
    batch = tiny_batch(mod)

    result = mod.compute_frozen_dsqg_w_objective(model, batch)
    result.loss.backward()

    assert result.loss.item() > 0.0
    assert result.telemetry["dsqg_w_objective_answer_tokens"] == pytest.approx(1.0)
    assert result.telemetry["dsqg_w_objective_answer_ce"] == pytest.approx(result.loss.item())
    assert result.telemetry["dsqg_w_candidate_fraction_question"] > 0.0
    assert result.telemetry["dsqg_w_candidate_fraction_hisa_evidence"] > 0.0
    assert result.telemetry["dsqg_w_candidate_fraction_l3_skip"] > 0.0

    grad_names = {name for name, param in model.named_parameters() if param.grad is not None and param.grad.abs().sum() > 0}
    assert grad_names
    assert all(name.startswith("dsqg_w.") for name in grad_names)
    assert model.embedding.weight.grad is None
    assert model.out.weight.grad is None


def test_compute_objective_rejects_disabled_dsqg_w_model() -> None:
    mod = load_objective_module()
    trainer = mod.load_trainer(enable_objective=False, suffix="disabled_test")
    model = mod.make_tiny_model(trainer, vocab_size=128, ffn_dim=64, seq_len=16)
    batch = tiny_batch(mod)

    with pytest.raises(ValueError, match="DSQG-W must be enabled"):
        mod.compute_frozen_dsqg_w_objective(model, batch)


def test_single_optimizer_step_updates_only_dsqg_w_parameters() -> None:
    mod = load_objective_module()
    trainer = mod.load_trainer(enable_objective=True, suffix="step_test")
    model = mod.make_tiny_model(trainer, vocab_size=128, ffn_dim=64, seq_len=16)
    mod.prepare_model_for_frozen_dsqg_w_objective(model)
    optimizer = mod.make_dsqg_w_optimizer(model, lr=1e-3)
    batch = tiny_batch(mod)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith("dsqg_w.") or name in {"embedding.weight", "norm.weight"}
    }
    before_embedding = model.embedding.weight.detach().clone()
    before_out = model.out.weight.detach().clone()

    report = mod.run_one_frozen_dsqg_w_step(model, batch, optimizer)

    assert report["pass"] is True
    assert report["step"] == 1
    assert report["grad_scope_ok"] is True
    assert report["changed_dsqg_w_param_count"] > 0
    assert report["changed_frozen_param_count"] == 0
    assert report["telemetry"]["dsqg_w_step_lr"] == pytest.approx(1e-3)
    assert report["telemetry"]["dsqg_w_objective_answer_tokens"] == pytest.approx(1.0)

    changed_dsqg_w = [
        name
        for name, old in before.items()
        if name.startswith("dsqg_w.") and not torch.equal(old, dict(model.named_parameters())[name].detach())
    ]
    assert changed_dsqg_w
    torch.testing.assert_close(before_embedding, model.embedding.weight.detach())
    torch.testing.assert_close(before_out, model.out.weight.detach())


def test_step_smoke_is_disabled_by_default() -> None:
    mod = load_objective_module()

    report = mod.run_smoke_objective(enable=False, step=True)

    assert report["enabled"] is False
    assert report["skipped"] is True
    assert report["pass"] is True
