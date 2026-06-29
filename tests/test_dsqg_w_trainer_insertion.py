from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer(monkeypatch, *, dsqg_w: bool, question: bool = False, hisa_l3: bool = False, sites: str | None = None):
    monkeypatch.setenv("DWARF_DISABLE_BNB", "1")
    monkeypatch.setenv("DWARF_LIGER", "0")
    monkeypatch.setenv("DWARF_TORCH_COMPILE", "0")
    if dsqg_w:
        monkeypatch.setenv("DWARF_DSQG_W", "1")
        monkeypatch.setenv("DWARF_DSQG_W_MAX_CANDIDATES", "16")
        monkeypatch.setenv("DWARF_DSQG_W_BOTTLENECK", "64")
        if sites is not None:
            monkeypatch.setenv("DWARF_DSQG_W_SITES", sites)
        else:
            monkeypatch.delenv("DWARF_DSQG_W_SITES", raising=False)
        if question:
            monkeypatch.setenv("DWARF_DSQG_W_QUESTION", "1")
            monkeypatch.setenv("DWARF_DSQG_W_K_QUESTION", "4")
        else:
            monkeypatch.delenv("DWARF_DSQG_W_QUESTION", raising=False)
        if hisa_l3:
            monkeypatch.setenv("DWARF_DSQG_W_HISA_L3", "1")
            monkeypatch.setenv("DWARF_DSQG_W_K_HISA_EVIDENCE", "4")
            monkeypatch.setenv("DWARF_DSQG_W_K_L3_SKIP", "2")
        else:
            monkeypatch.delenv("DWARF_DSQG_W_HISA_L3", raising=False)
    else:
        monkeypatch.delenv("DWARF_DSQG_W", raising=False)
        monkeypatch.delenv("DWARF_DSQG_W_SITES", raising=False)
        monkeypatch.delenv("DWARF_DSQG_W_QUESTION", raising=False)
        monkeypatch.delenv("DWARF_DSQG_W_HISA_L3", raising=False)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    try:
        spec = importlib.util.spec_from_file_location(
            f"trainer_dsqg_w_{int(dsqg_w)}_{os.getpid()}_{id(monkeypatch)}",
            TRAINER,
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        for path in [str(ROOT / "kernels"), str(ROOT)]:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def make_model(mod):
    return mod.TriadicJ96Dsr(
        vocab_size=128,
        embedding_dim=mod.EMBEDDING_DIM,
        num_heads=mod.NUM_HEADS,
        ffn_dim=64,
        seq_len=32,
        dsr_layer=mod.DSR_LAYER,
        dropout=0.0,
        num_chunks=mod.NUM_CHUNKS,
        top_k_chunks=mod.TOP_K_CHUNKS,
    ).eval()


def test_dsqg_w_final_recomposer_is_disabled_by_default(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=False)
    model = make_model(mod)
    x = torch.randn(2, 8, mod.EMBEDDING_DIM)

    assert mod.DSQG_W_ENABLED is False
    assert model.dsqg_w_enabled is False
    assert model.dsqg_w is None
    out = model._apply_dsqg_w_recomposer(x)
    assert out is x
    assert model.dsqg_w_last_telemetry == {}


def test_dsqg_w_final_recomposer_uses_local_long_null_only_initial_path(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True)
    model = make_model(mod)
    x = torch.randn(2, 8, mod.EMBEDDING_DIM)

    assert mod.DSQG_W_ENABLED is True
    assert model.dsqg_w_enabled is True
    assert model.dsqg_w is not None
    assert model.dsqg_w_config.k_question == 0
    assert model.dsqg_w_config.k_hisa_evidence == 0
    assert model.dsqg_w_config.k_chunk == 0
    assert model.dsqg_w_config.k_l3_skip == 0

    out = model._apply_dsqg_w_recomposer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out - x).abs().max().item() < 1e-2
    telemetry = model.dsqg_w_last_telemetry
    assert telemetry["dsqg_w_valid_candidate_count"].item() <= model.dsqg_w_config.max_candidates
    assert telemetry["dsqg_w_candidate_fraction_question"].item() == 0.0
    assert telemetry["dsqg_w_candidate_fraction_hisa_evidence"].item() == 0.0
    assert telemetry["dsqg_w_candidate_fraction_chunk_rep"].item() == 0.0
    assert telemetry["dsqg_w_candidate_fraction_l3_skip"].item() == 0.0


def test_dsqg_w_forward_hidden_applies_recomposer_before_final_norm(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True)
    model = make_model(mod)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in model.blocks])
    model.norm = torch.nn.Identity()
    idx = torch.randint(0, 128, (1, 8), dtype=torch.long)
    calls = []

    def fake_apply(x, **kwargs):
        calls.append(x.shape)
        return x

    model._apply_dsqg_w_recomposer = fake_apply  # type: ignore[method-assign]
    _ = model.forward_hidden(idx)

    assert calls == [torch.Size([1, 8, mod.EMBEDDING_DIM])]


def test_dsqg_w_question_candidate_indices_are_threaded_into_provider(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, question=True)
    model = make_model(mod)
    x = torch.randn(2, 8, mod.EMBEDDING_DIM)
    question_indices = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.long)

    assert mod.DSQG_W_QUESTION_ENABLED is True
    assert model.dsqg_w_config.k_question == 4

    out = model._apply_dsqg_w_recomposer(x, question_indices=question_indices)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out - x).abs().max().item() < 1e-2
    telemetry = model.dsqg_w_last_telemetry
    assert telemetry["dsqg_w_candidate_fraction_question"].item() > 0.0
    assert telemetry["dsqg_w_question_mass"].item() > 0.0
    assert telemetry["dsqg_w_candidate_fraction_hisa_evidence"].item() == 0.0
    assert telemetry["dsqg_w_candidate_fraction_l3_skip"].item() == 0.0


def test_forward_accepts_optional_dsqg_w_question_indices(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, question=True)
    model = make_model(mod)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in model.blocks])
    model.norm = torch.nn.Identity()
    idx = torch.randint(0, 128, (1, 8), dtype=torch.long)
    question_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    seen = []

    def fake_apply(x, **kwargs):
        seen.append(kwargs.get("question_indices"))
        return x

    model._apply_dsqg_w_recomposer = fake_apply  # type: ignore[method-assign]
    _ = model(idx, dsqg_w_question_indices=question_indices)

    assert seen == [question_indices]


def test_dsqg_w_hisa_l3_candidate_indices_read_from_l3_states(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, question=True, hisa_l3=True)
    model = make_model(mod)
    final_states = torch.randn(2, 8, mod.EMBEDDING_DIM)
    l3_states = final_states + 0.25
    hisa_indices = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.long)
    l3_skip_indices = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)

    assert mod.DSQG_W_HISA_L3_ENABLED is True
    assert model.dsqg_w_config.k_hisa_evidence == 4
    assert model.dsqg_w_config.k_l3_skip == 2

    out = model._apply_dsqg_w_recomposer(
        final_states,
        l3_states=l3_states,
        hisa_evidence_indices=hisa_indices,
        l3_skip_indices=l3_skip_indices,
    )

    assert out.shape == final_states.shape
    assert torch.isfinite(out).all()
    assert (out - final_states).abs().max().item() < 1e-2
    telemetry = model.dsqg_w_last_telemetry
    assert telemetry["dsqg_w_candidate_fraction_hisa_evidence"].item() > 0.0
    assert telemetry["dsqg_w_candidate_fraction_l3_skip"].item() > 0.0
    assert telemetry["dsqg_w_hisa_evidence_mass"].item() > 0.0
    assert telemetry["dsqg_w_l3_skip_mass"].item() > 0.0
    assert telemetry["dsqg_w_l3_source_mass"].item() > 0.0


def test_forward_accepts_optional_dsqg_w_hisa_l3_indices(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, hisa_l3=True)
    model = make_model(mod)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in model.blocks])
    model.norm = torch.nn.Identity()
    idx = torch.randint(0, 128, (1, 8), dtype=torch.long)
    hisa_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    l3_skip_indices = torch.tensor([[0, 4]], dtype=torch.long)
    seen = []

    def fake_apply(x, **kwargs):
        seen.append((kwargs.get("l3_states"), kwargs.get("hisa_evidence_indices"), kwargs.get("l3_skip_indices")))
        return x

    model._apply_dsqg_w_recomposer = fake_apply  # type: ignore[method-assign]
    _ = model(
        idx,
        dsqg_w_hisa_evidence_indices=hisa_indices,
        dsqg_w_l3_skip_indices=l3_skip_indices,
    )

    l3_states, seen_hisa, seen_l3_skip = seen[0]
    assert l3_states is not None
    assert seen_hisa is hisa_indices
    assert seen_l3_skip is l3_skip_indices


def test_layer_summary_includes_configured_dsqg_w_sites(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, question=True, hisa_l3=True, sites="2,6,final")
    model = make_model(mod)

    summary = model.layer_summary()

    assert "DSQG-W-sites=layer_2,layer_6,final" in summary
    assert "L2:DSQG-C" in summary
    assert "L6:DSQG-C" in summary
    assert "FINAL:DSQG-W" in summary


def test_dsqg_w_sites_apply_layer_and_final_recomposers_in_order(monkeypatch) -> None:
    mod = load_trainer(monkeypatch, dsqg_w=True, question=True, sites="2,final")
    model = make_model(mod)
    model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in model.blocks])
    model.norm = torch.nn.Identity()
    idx = torch.randint(0, 128, (1, 8), dtype=torch.long)
    calls = []

    assert mod.DSQG_W_SITE_SPECS == (2, "final")
    assert model.dsqg_w_site_keys == ("layer_2", "final")
    assert set(model.dsqg_w_blocks.keys()) == {"layer_2", "final"}
    assert model.dsqg_w is model.dsqg_w_blocks["final"]

    def fake_apply(x, **kwargs):
        calls.append(kwargs.get("site_key"))
        return x

    model._apply_dsqg_w_recomposer = fake_apply  # type: ignore[method-assign]
    _ = model.forward_hidden(idx)

    assert calls == ["layer_2", "final"]
