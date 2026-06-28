from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frozen_trunk_objective_dsqg_w.py"
LEXICAL_GAP_JSONL = ROOT / "audits/dsqg_w_lexical_gap_mini.jsonl"
OLMO_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"


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


def test_build_lexical_gap_batch_from_jsonl_records() -> None:
    mod = load_objective_module()

    records = mod.load_lexical_gap_records(LEXICAL_GAP_JSONL)
    batch, vocab = mod.build_lexical_gap_batch(records)

    assert len(records) == 3
    assert batch.input_ids.shape == batch.labels.shape == batch.answer_mask.shape
    assert batch.input_ids.shape[0] == 3
    assert int(batch.answer_mask.sum().item()) == 3
    assert batch.question_indices.shape == (3, 4)
    assert batch.hisa_evidence_indices.shape == (3, 4)
    assert batch.l3_skip_indices.shape == (3, 3)
    assert vocab["Cu"] == int(batch.labels[0, 19].item())
    assert vocab["puppy"] == int(batch.labels[1, 19].item())
    assert vocab["yellow"] == int(batch.labels[2, 19].item())


def test_multistep_lexical_gap_smoke_reduces_loss_and_keeps_trunk_frozen() -> None:
    mod = load_objective_module()

    report = mod.run_lexical_gap_overfit_smoke(
        jsonl_path=LEXICAL_GAP_JSONL,
        steps=4,
        lr=1e-3,
        seed=20260628,
    )

    assert report["pass"] is True
    assert report["enabled"] is True
    assert report["objective"] == "frozen_trunk_answer_only_ce_overfit_smoke"
    assert report["steps"] == 4
    assert report["dataset_examples"] == 3
    assert report["answer_tokens"] == pytest.approx(3.0)
    assert report["loss_final"] < report["loss_initial"]
    assert report["loss_delta"] < 0.0
    assert report["max_changed_frozen_param_count"] == 0
    assert report["min_changed_dsqg_w_param_count"] > 0


def test_build_real_tokenizer_lexical_gap_batch_maps_answer_and_candidates() -> None:
    mod = load_objective_module()

    records = mod.load_lexical_gap_records(LEXICAL_GAP_JSONL)
    tokenizer = mod.load_tokenizer(OLMO_TOKENIZER)
    batch, meta = mod.build_tokenized_lexical_gap_batch(records, tokenizer)

    assert meta["tokenizer_path"] == str(OLMO_TOKENIZER)
    assert meta["tokenizer_vocab_size"] > 1000
    assert batch.input_ids.shape == batch.labels.shape == batch.answer_mask.shape
    assert batch.input_ids.shape[0] == 3
    assert int(batch.answer_mask.sum().item()) >= 3
    assert batch.input_ids.max().item() < meta["tokenizer_vocab_size"]
    assert batch.question_indices.shape[0] == 3
    assert batch.hisa_evidence_indices.shape[0] == 3
    assert batch.l3_skip_indices.shape[0] == 3
    assert (batch.question_indices >= 0).any().item()
    assert (batch.hisa_evidence_indices >= 0).any().item()
    assert (batch.l3_skip_indices >= 0).any().item()


def test_real_tokenizer_lexical_gap_overfit_smoke_reduces_loss() -> None:
    mod = load_objective_module()

    report = mod.run_lexical_gap_overfit_smoke(
        jsonl_path=LEXICAL_GAP_JSONL,
        tokenizer_path=OLMO_TOKENIZER,
        steps=3,
        lr=1e-3,
        seed=20260628,
    )

    assert report["pass"] is True
    assert report["tokenized"] is True
    assert report["tokenizer_vocab_size"] > 1000
    assert report["loss_final"] < report["loss_initial"]
    assert report["max_changed_frozen_param_count"] == 0
    assert report["min_changed_dsqg_w_param_count"] > 0


def test_dsqg_w_only_checkpoint_roundtrip_reproduces_tokenized_loss(tmp_path: Path) -> None:
    mod = load_objective_module()
    seed = 20260628
    records = mod.load_lexical_gap_records(LEXICAL_GAP_JSONL)
    tokenizer = mod.load_tokenizer(OLMO_TOKENIZER)
    batch, meta = mod.build_tokenized_lexical_gap_batch(records, tokenizer)
    trainer = mod.load_trainer(enable_objective=True, suffix="checkpoint_roundtrip")

    torch.manual_seed(seed)
    model = mod.make_tiny_model(
        trainer,
        vocab_size=meta["tokenizer_vocab_size"],
        ffn_dim=64,
        seq_len=batch.input_ids.shape[1],
    )
    mod.prepare_model_for_frozen_dsqg_w_objective(model)
    optimizer = mod.make_dsqg_w_optimizer(model, lr=1e-3)
    for _ in range(3):
        mod.run_one_frozen_dsqg_w_step(model, batch, optimizer)
    with torch.no_grad():
        saved_loss = float(mod.compute_frozen_dsqg_w_objective(model, batch).loss.detach().cpu().item())

    save_report = mod.save_dsqg_w_checkpoint(
        model,
        tmp_path / "roundtrip",
        metadata={
            "seed": seed,
            "jsonl_path": str(LEXICAL_GAP_JSONL),
            "tokenizer_path": str(OLMO_TOKENIZER),
            "tokenizer_vocab_size": meta["tokenizer_vocab_size"],
            "seq_len": batch.input_ids.shape[1],
            "loss_final": saved_loss,
            "candidate_settings": {"max_candidates": 16, "k_question": 4, "k_hisa_evidence": 4, "k_l3_skip": 2},
        },
    )

    state_payload = torch.load(save_report["state_path"], map_location="cpu", weights_only=True)
    assert set(state_payload) == {"dsqg_w_state_dict"}
    assert state_payload["dsqg_w_state_dict"]
    sidecar = json.loads(Path(save_report["metadata_path"]).read_text())
    assert sidecar["contains"] == "model.dsqg_w.state_dict"
    assert sidecar["metadata"]["tokenizer_path"] == str(OLMO_TOKENIZER)
    assert sidecar["metadata"]["candidate_settings"]["max_candidates"] == 16
    assert sidecar["git_commit"]

    torch.manual_seed(seed)
    fresh_model = mod.make_tiny_model(
        trainer,
        vocab_size=meta["tokenizer_vocab_size"],
        ffn_dim=64,
        seq_len=batch.input_ids.shape[1],
    )
    mod.prepare_model_for_frozen_dsqg_w_objective(fresh_model)
    load_report = mod.load_dsqg_w_checkpoint(fresh_model, save_report["state_path"])
    assert load_report["missing_keys"] == []
    assert load_report["unexpected_keys"] == []
    with torch.no_grad():
        loaded_loss = float(mod.compute_frozen_dsqg_w_objective(fresh_model, batch).loss.detach().cpu().item())
    assert loaded_loss == pytest.approx(saved_loss)
