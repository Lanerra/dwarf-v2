import math

import pytest
import torch

from kernels.dsqg_w.dsqg_w_mvp import (
    CandidateSource,
    CandidateType,
    CandidateProvider,
    DSQGWBlock,
    DSQGWConfig,
    DSQGWEvidenceBindingHub,
)


def _make_fixture(batch: int = 2, seq: int = 5, k: int = 6, d: int = 16):
    torch.manual_seed(701)
    x = torch.randn(batch, seq, d)
    cand_states = torch.randn(batch, seq, k, d)
    type_cycle = torch.tensor(
        [
            int(CandidateType.LOCAL),
            int(CandidateType.QUESTION),
            int(CandidateType.HISA_EVIDENCE),
            int(CandidateType.LONG_OFFSET),
            int(CandidateType.CHUNK_REP),
            int(CandidateType.L3_SKIP),
        ],
        dtype=torch.long,
    )
    source_cycle = torch.tensor(
        [
            int(CandidateSource.FINAL),
            int(CandidateSource.FINAL),
            int(CandidateSource.HISA),
            int(CandidateSource.FINAL),
            int(CandidateSource.SUMMARY),
            int(CandidateSource.L3),
        ],
        dtype=torch.long,
    )
    cand_types = type_cycle[:k].reshape(1, 1, k).expand(batch, seq, k).clone()
    cand_sources = source_cycle[:k].reshape(1, 1, k).expand(batch, seq, k).clone()
    cand_mask = torch.ones(batch, seq, k, dtype=torch.bool)
    distances = torch.arange(1, k + 1, dtype=torch.float32).reshape(1, 1, k).expand(batch, seq, k).clone()
    scores = torch.linspace(-1.0, 1.0, k, dtype=torch.float32).reshape(1, 1, k).expand(batch, seq, k).clone()
    return x, cand_states, cand_types, cand_sources, cand_mask, distances, scores


def _new_hub(d: int = 16, gate_init: float = -5.0):
    torch.manual_seed(702)
    return DSQGWEvidenceBindingHub(
        d=d,
        n_types=len(CandidateType),
        n_sources=len(CandidateSource),
        bottleneck=32,
        gate_init=gate_init,
        phase_bands=4,
        use_score_features=True,
    )


def test_evidence_binding_hub_is_candidate_permutation_invariant() -> None:
    x, cand_states, cand_types, cand_sources, cand_mask, distances, scores = _make_fixture()
    hub = _new_hub(d=x.shape[-1]).eval()

    y, telemetry, aux = hub(
        x,
        cand_states,
        cand_types,
        cand_sources,
        cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    perm = torch.tensor([2, 0, 5, 1, 4, 3], dtype=torch.long)
    y_perm, _, aux_perm = hub(
        x,
        cand_states[:, :, perm],
        cand_types[:, :, perm],
        cand_sources[:, :, perm],
        cand_mask[:, :, perm],
        candidate_distances=distances[:, :, perm],
        cand_scores=scores[:, :, perm],
        return_aux=True,
    )

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(y, y_perm, atol=1e-6, rtol=1e-6)
    assert torch.allclose(aux["bound_packet"], aux_perm["bound_packet"], atol=1e-6, rtol=1e-6)
    assert telemetry["dsqg_w_ebh_active_row_fraction"].item() == pytest.approx(1.0)


def test_evidence_binding_hub_packet_is_sensitive_to_shuffled_evidence_identity() -> None:
    x, cand_states, cand_types, cand_sources, cand_mask, distances, scores = _make_fixture()
    hub = _new_hub(d=x.shape[-1]).eval()

    _, _, aux = hub(
        x,
        cand_states,
        cand_types,
        cand_sources,
        cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    shuffled_states = cand_states.flip(dims=(2,))
    y_shuf, _, aux_shuf = hub(
        x,
        shuffled_states,
        cand_types,
        cand_sources,
        cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )

    packet_delta = (aux["bound_packet"] - aux_shuf["bound_packet"]).abs().mean().item()
    assert packet_delta > 1e-3
    assert torch.isfinite(y_shuf).all()


def test_evidence_binding_hub_zero_candidate_rows_are_identity() -> None:
    x, cand_states, cand_types, cand_sources, _cand_mask, distances, scores = _make_fixture()
    hub = _new_hub(d=x.shape[-1]).eval()
    empty_mask = torch.zeros(cand_types.shape, dtype=torch.bool)
    null_types = torch.full_like(cand_types, int(CandidateType.NULL))
    null_sources = torch.full_like(cand_sources, int(CandidateSource.NULL))

    y, telemetry, aux = hub(
        x,
        cand_states,
        null_types,
        null_sources,
        empty_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )

    assert torch.equal(y, x)
    assert telemetry["dsqg_w_ebh_active_row_fraction"].item() == pytest.approx(0.0)
    assert aux["candidate_weight_mass"].max().item() == pytest.approx(0.0)


def test_evidence_binding_hub_lane_masses_are_separable_and_gradients_flow() -> None:
    x, cand_states, cand_types, cand_sources, cand_mask, distances, scores = _make_fixture()
    cand_states = cand_states.clone().requires_grad_(True)
    hub = _new_hub(d=x.shape[-1], gate_init=-2.0)

    y, telemetry, aux = hub(
        x,
        cand_states,
        cand_types,
        cand_sources,
        cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    loss = y.square().mean() + aux["bound_packet"].square().mean()
    loss.backward()

    assert telemetry["dsqg_w_ebh_hisa_evidence_mass"].item() > 0.0
    assert telemetry["dsqg_w_ebh_l3_source_mass"].item() > 0.0
    assert cand_states.grad is not None
    assert cand_states.grad.norm().item() > 0.0
    grad_names = {
        name: param.grad.norm().item()
        for name, param in hub.named_parameters()
        if param.grad is not None and math.isfinite(param.grad.norm().item())
    }
    assert grad_names["value_proj.weight"] > 0.0
    assert grad_names["read_mix.weight"] > 0.0
    assert grad_names["delta_proj.3.weight"] > 0.0
    assert grad_names["bind_gate.weight"] > 0.0


def test_default_dsqg_w_config_has_no_evidence_binding_hub() -> None:
    cfg = DSQGWConfig(d=16, n_heads=4)
    block = DSQGWBlock.from_config(cfg)

    assert block.evidence_binding_hub is None


def test_dsqg_w_config_can_enable_evidence_binding_hub() -> None:
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_gate_init=-7.0,
        ebh_phase_bands=3,
        ebh_score_features=False,
    )
    block = DSQGWBlock.from_config(cfg)

    assert isinstance(block.evidence_binding_hub, DSQGWEvidenceBindingHub)
    assert block.evidence_binding_hub.use_score_features is False
    assert block.evidence_binding_hub.phase_bands == 3


def test_materialized_dsqg_w_forward_applies_evidence_binding_hub_and_backprops() -> None:
    torch.manual_seed(703)
    x, cand_states, cand_types, cand_sources, cand_mask, distances, scores = _make_fixture(batch=1, seq=6, k=6, d=16)
    x = x.requires_grad_(True)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        bottleneck=32,
        gate_init=-5.0,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_gate_init=-2.0,
    )
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block(
        x,
        cand_states,
        cand_types,
        cand_sources,
        cand_mask,
        cand_scores=scores,
        candidate_distances=distances,
    )
    out.square().mean().backward()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_ebh_enabled"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_active_row_fraction"].item() == pytest.approx(1.0)
    assert block.evidence_binding_hub is not None
    assert block.evidence_binding_hub.value_proj.weight.grad is not None
    assert block.evidence_binding_hub.value_proj.weight.grad.norm().item() > 0.0
    assert block.evidence_binding_hub.bind_gate.weight.grad is not None
    assert block.evidence_binding_hub.bind_gate.weight.grad.norm().item() > 0.0


def test_sourcewise_dsqg_w_forward_applies_evidence_binding_hub() -> None:
    torch.manual_seed(704)
    x = torch.randn(1, 6, 16, requires_grad=True)
    l3 = (x.detach() * 1.25).clone().requires_grad_(True)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        bottleneck=32,
        gate_init=-5.0,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_gate_init=-2.0,
    )
    positions = torch.arange(6)
    question = torch.tensor([[0, 3]])
    hisa = torch.stack([(positions - i).clamp_min(0) for i in [1, 2]], dim=-1).unsqueeze(0)
    scores = torch.arange(12, dtype=x.dtype).reshape(1, 6, 2)
    l3_skip = (positions - 4).clamp_min(0).reshape(1, 6, 1)
    candidate_batch = CandidateProvider(cfg).build_metadata(
        x.detach(),
        l3_states=l3.detach(),
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=scores,
        l3_skip_indices=l3_skip,
    )
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block.forward_sourcewise(
        x,
        candidate_batch.cand_token_indices,
        candidate_batch.cand_types,
        candidate_batch.cand_sources,
        candidate_batch.cand_mask,
        l3_states=l3,
        cand_scores=candidate_batch.cand_scores,
        candidate_distances=candidate_batch.candidate_distances,
    )
    out.square().mean().backward()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_ebh_enabled"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_sourcewise_ebh_materialized"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_triton_sourcewise_semantic_bypass"].item() == pytest.approx(0.0)
    assert block.evidence_binding_hub is not None
    assert block.evidence_binding_hub.value_proj.weight.grad is not None
    assert block.evidence_binding_hub.value_proj.weight.grad.norm().item() > 0.0


def test_evidence_binding_hub_sourcewise_packet_matches_materialized_raw_layout() -> None:
    torch.manual_seed(705)
    x = torch.randn(1, 7, 16)
    l3 = torch.randn(1, 7, 16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        bottleneck=32,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_gate_init=-2.0,
    )
    positions = torch.arange(7)
    batch = CandidateProvider(cfg).build_metadata(
        x,
        l3_states=l3,
        question_indices=torch.tensor([[0, 4]]),
        hisa_evidence_indices=torch.stack([(positions - i).clamp_min(0) for i in [1, 3]], dim=-1).unsqueeze(0),
        hisa_evidence_scores=torch.linspace(-1.0, 1.0, steps=14).reshape(1, 7, 2),
        l3_skip_indices=(positions - 5).clamp_min(0).reshape(1, 7, 1),
    )
    block = DSQGWBlock.from_config(cfg).eval()
    assert block.evidence_binding_hub is not None
    cand_states = block._materialize_sourcewise_candidate_states(
        x,
        batch.cand_token_indices,
        batch.cand_sources,
        batch.cand_mask,
        l3_states=l3,
    )

    y_mat, _, aux_mat = block.evidence_binding_hub(
        x,
        cand_states,
        batch.cand_types,
        batch.cand_sources,
        batch.cand_mask,
        candidate_distances=batch.candidate_distances,
        cand_scores=batch.cand_scores,
        return_aux=True,
    )
    y_packet, telemetry, aux_packet = block.evidence_binding_hub.forward_sourcewise_packet(
        x,
        batch.cand_token_indices,
        batch.cand_types,
        batch.cand_sources,
        batch.cand_mask,
        l3_states=l3,
        candidate_distances=batch.candidate_distances,
        cand_scores=batch.cand_scores,
        return_aux=True,
    )

    assert torch.allclose(y_packet, y_mat, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aux_packet["bound_packet"], aux_mat["bound_packet"], atol=1e-5, rtol=1e-5)
    assert telemetry["dsqg_w_ebh_packet_sourcewise"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_packet_triton"].item() == pytest.approx(0.0)


def test_sourcewise_ebh_packet_flag_avoids_ebh_materialization_and_backprops(monkeypatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_EBH_SOURCEWISE_PACKET", "1")
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "0")
    torch.manual_seed(706)
    x = torch.randn(1, 6, 16, requires_grad=True)
    l3 = torch.randn(1, 6, 16, requires_grad=True)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        bottleneck=32,
        gate_init=-5.0,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_gate_init=-2.0,
    )
    positions = torch.arange(6)
    batch = CandidateProvider(cfg).build_metadata(
        x.detach(),
        l3_states=l3.detach(),
        question_indices=torch.tensor([[0, 3]]),
        hisa_evidence_indices=torch.stack([(positions - i).clamp_min(0) for i in [1, 2]], dim=-1).unsqueeze(0),
        hisa_evidence_scores=torch.randn(1, 6, 2),
        l3_skip_indices=(positions - 4).clamp_min(0).reshape(1, 6, 1),
    )
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block.forward_sourcewise(
        x,
        batch.cand_token_indices,
        batch.cand_types,
        batch.cand_sources,
        batch.cand_mask,
        l3_states=l3,
        cand_scores=batch.cand_scores,
        candidate_distances=batch.candidate_distances,
    )
    out.square().mean().backward()

    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_sourcewise_ebh_materialized"].item() == pytest.approx(0.0)
    assert telemetry["dsqg_w_ebh_packet_sourcewise"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_packet_semantic_approx"].item() == pytest.approx(0.0)
    assert x.grad is not None and x.grad.norm().item() > 0.0
    assert l3.grad is not None and l3.grad.norm().item() > 0.0
    assert block.evidence_binding_hub is not None
    assert block.evidence_binding_hub.value_proj.weight.grad is not None
    assert block.evidence_binding_hub.bind_gate.weight.grad is not None


def test_sourcewise_ebh_packet_score_features_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_EBH_SOURCEWISE_PACKET", "1")
    x = torch.randn(1, 5, 16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=4,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=1,
        k_l3_skip=0,
        k_chunk=0,
        bottleneck=32,
        use_evidence_binding_hub=True,
        ebh_bottleneck=32,
        ebh_score_features=False,
    )
    positions = torch.arange(5)
    batch = CandidateProvider(cfg).build_metadata(
        x,
        question_indices=torch.tensor([[0, 2]]),
        hisa_evidence_indices=(positions - 1).clamp_min(0).reshape(1, 5, 1),
        hisa_evidence_scores=torch.randn(1, 5, 1),
    )
    block = DSQGWBlock.from_config(cfg)
    out, telemetry = block.forward_sourcewise(
        x,
        batch.cand_token_indices,
        batch.cand_types,
        batch.cand_sources,
        batch.cand_mask,
        cand_scores=batch.cand_scores,
        candidate_distances=batch.candidate_distances,
    )

    assert torch.isfinite(out).all()
    assert block.evidence_binding_hub is not None
    assert block.evidence_binding_hub.use_score_features is False
    assert telemetry["dsqg_w_sourcewise_ebh_materialized"].item() == pytest.approx(0.0)
