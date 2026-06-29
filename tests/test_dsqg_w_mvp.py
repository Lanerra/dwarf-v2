from __future__ import annotations

import inspect

import pytest
import torch

from kernels.dsqg_w.dsqg_w_mvp import (
    CandidateProvider,
    CandidateSource,
    CandidateType,
    DSQGWBlock,
    DSQGWConfig,
    answer_masked_loss,
    conditional_copy_unlikelihood_loss,
    entropy_floor_loss,
    local_mass_cap_loss,
)


def make_hidden(batch: int = 2, seq: int = 9, d: int = 16) -> torch.Tensor:
    torch.manual_seed(17)
    return torch.randn(batch, seq, d)


def test_candidate_provider_keeps_candidates_bounded_causal_and_nonempty() -> None:
    x = make_hidden(seq=12)
    cfg = DSQGWConfig(d=16, n_heads=4, max_candidates=6)
    provider = CandidateProvider(cfg)

    batch = provider.build(
        x,
        question_indices=torch.tensor([[0, 2, 6, 10], [1, 3, 4, 11]]),
        hisa_evidence_indices=torch.tensor([
            [[0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4], [0, 1, 4]],
            [[0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5]],
        ]),
    )

    assert batch.cand_states.shape == (2, 12, 6, 16)
    assert batch.cand_types.shape == (2, 12, 6)
    assert batch.cand_sources.shape == (2, 12, 6)
    assert batch.cand_mask.shape == (2, 12, 6)
    assert batch.cand_token_indices.shape == (2, 12, 6)
    assert batch.cand_mask.any(dim=-1).all()

    query_positions = torch.arange(12).reshape(1, 12, 1)
    valid_token_indices = batch.cand_token_indices.masked_select(batch.cand_mask)
    valid_query_positions = query_positions.expand_as(batch.cand_token_indices).masked_select(batch.cand_mask)
    assert torch.le(valid_token_indices, valid_query_positions).all()
    assert batch.valid_candidate_count.max().item() <= cfg.max_candidates
    assert batch.telemetry["dsqg_w_candidate_invalid_rate"].item() >= 0.0


def test_candidate_provider_deduplicates_by_token_and_source_with_semantic_priority() -> None:
    x = make_hidden(batch=1, seq=6)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(1,),
        k_question=1,
        k_hisa_evidence=0,
        k_chunk=0,
    )
    provider = CandidateProvider(cfg)

    batch = provider.build(x, question_indices=torch.tensor([[3]]))

    # At t=4, local offset 1 and long offset 1 both point to token 3 in FINAL,
    # and question_indices also points to token 3 in FINAL. QUESTION must win.
    t4_tokens = batch.cand_token_indices[0, 4][batch.cand_mask[0, 4]]
    t4_sources = batch.cand_sources[0, 4][batch.cand_mask[0, 4]]
    t4_types = batch.cand_types[0, 4][batch.cand_mask[0, 4]]
    final_token3 = (t4_tokens == 3) & (t4_sources == int(CandidateSource.FINAL))
    assert final_token3.sum().item() == 1
    assert t4_types[final_token3].item() == int(CandidateType.QUESTION)
    assert batch.telemetry["dsqg_w_candidate_duplicate_rate"].item() > 0.0


def test_candidate_provider_fast_path_matches_reference_candidate_layout() -> None:
    x = make_hidden(batch=2, seq=11, d=16)
    l3 = x + 0.125
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=10,
        bottleneck=32,
        local_offsets=(1, 2, 4),
        long_offsets=(4, 8),
        k_question=3,
        k_hisa_evidence=2,
        k_chunk=0,
        k_l3_skip=2,
    )
    provider = CandidateProvider(cfg)
    question = torch.tensor([[0, 3, 7], [1, 4, 9]])
    positions = torch.arange(x.shape[1])
    hisa = torch.stack([
        torch.stack([(positions - 1).clamp_min(0), (positions - 3).clamp_min(0)], dim=-1),
        torch.stack([(positions - 2).clamp_min(0), (positions - 5).clamp_min(0)], dim=-1),
    ], dim=0)
    l3_skip = torch.stack([
        torch.stack([(positions - 6).clamp_min(0), (positions - 8).clamp_min(0)], dim=-1),
        torch.stack([(positions - 4).clamp_min(0), (positions - 7).clamp_min(0)], dim=-1),
    ], dim=0)

    reference = provider._build_reference(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        l3_skip_indices=l3_skip,
    )
    fast = provider.build(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        l3_skip_indices=l3_skip,
    )

    torch.testing.assert_close(fast.cand_states, reference.cand_states)
    torch.testing.assert_close(fast.cand_types, reference.cand_types)
    torch.testing.assert_close(fast.cand_sources, reference.cand_sources)
    torch.testing.assert_close(fast.cand_mask, reference.cand_mask)
    torch.testing.assert_close(fast.cand_token_indices, reference.cand_token_indices)
    torch.testing.assert_close(fast.valid_candidate_count, reference.valid_candidate_count)


def test_candidate_provider_build_uses_vectorized_fast_path_not_batch_token_python_loop() -> None:
    source = inspect.getsource(CandidateProvider.build)
    assert "for b in range" not in source
    assert "for t in range" not in source


def test_dsqg_w_block_shape_no_nan_identityish_init_and_required_telemetry() -> None:
    torch.manual_seed(23)
    x = make_hidden(seq=7)
    cfg = DSQGWConfig(d=16, n_heads=4, max_candidates=10, bottleneck=32, gate_init=-5.0)
    provider = CandidateProvider(cfg)
    cands = provider.build(x, question_indices=torch.tensor([[0, 2], [1, 3]]))
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block(
        x,
        cands.cand_states,
        cands.cand_types,
        cands.cand_sources,
        cands.cand_mask,
        return_routing=True,
    )

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert (out - x).abs().max().item() < 1e-2
    for key in [
        "dsqg_w_entropy",
        "dsqg_w_local_mass",
        "dsqg_w_question_mass",
        "dsqg_w_hisa_evidence_mass",
        "dsqg_w_long_offset_mass",
        "dsqg_w_chunk_rep_mass",
        "dsqg_w_l3_source_mass",
        "dsqg_w_final_source_mass",
        "dsqg_w_null_mass",
        "dsqg_w_valid_candidate_count",
        "dsqg_w_gate_mean",
        "dsqg_w_gate_min",
        "dsqg_w_gate_max",
        "dsqg_w_delta_norm",
        "dsqg_w_x_norm",
        "dsqg_w_delta_to_x_ratio",
        "dsqg_w_read_norm",
        "dsqg_w_typed_read_norms",
        "read_mix_weight_norm",
        "dsqg_w_probs",
    ]:
        assert key in telemetry


def test_dsqg_w_width_cell_disabled_preserves_legacy_block_output() -> None:
    torch.manual_seed(41)
    x = make_hidden(batch=1, seq=6, d=16)
    base_cfg = DSQGWConfig(d=16, n_heads=4, max_candidates=8, bottleneck=32, gate_init=-5.0)
    width_disabled_cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        bottleneck=32,
        gate_init=-5.0,
        use_width_cell=False,
    )
    provider = CandidateProvider(base_cfg)
    cands = provider.build(
        x,
        question_indices=torch.tensor([[0, 2, 4]]),
        hisa_evidence_indices=torch.tensor([[[0, 0], [0, 0], [0, 1], [0, 2], [1, 3], [2, 4]]]),
    )
    legacy = DSQGWBlock.from_config(base_cfg).eval()
    disabled = DSQGWBlock.from_config(width_disabled_cfg).eval()
    disabled.load_state_dict(legacy.state_dict())

    out_legacy, telemetry_legacy = legacy(
        x, cands.cand_states, cands.cand_types, cands.cand_sources, cands.cand_mask
    )
    out_disabled, telemetry_disabled = disabled(
        x, cands.cand_states, cands.cand_types, cands.cand_sources, cands.cand_mask
    )

    torch.testing.assert_close(out_disabled, out_legacy, atol=0.0, rtol=0.0)
    assert "dsqg_w_width_gate_mean" not in telemetry_legacy
    assert "dsqg_w_width_gate_mean" not in telemetry_disabled


def test_dsqg_w_width_cell_is_near_identity_when_closed_and_reports_width_telemetry() -> None:
    torch.manual_seed(43)
    x = make_hidden(batch=1, seq=7, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=10,
        bottleneck=32,
        gate_init=-5.0,
        use_width_cell=True,
        width_bottleneck=8,
        width_gate_init=-12.0,
    )
    provider = CandidateProvider(cfg)
    cands = provider.build(
        x,
        question_indices=torch.tensor([[0, 2, 5]]),
        hisa_evidence_indices=torch.tensor([[[0, 0], [0, 0], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5]]]),
        l3_skip_indices=torch.tensor([[[0], [0], [1], [2], [3], [4], [5]]]),
    )
    block = DSQGWBlock.from_config(cfg).eval()

    out, telemetry = block(x, cands.cand_states, cands.cand_types, cands.cand_sources, cands.cand_mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_width_gate_mean"].item() < 1e-4
    assert telemetry["dsqg_w_width_entropy"].item() > 0.0
    assert telemetry["dsqg_w_width_delta_norm"].item() >= 0.0


def test_dsqg_w_width_cell_pair_bias_can_directionally_route_question_to_evidence() -> None:
    torch.manual_seed(47)
    x = make_hidden(batch=1, seq=5, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        bottleneck=32,
        gate_init=-5.0,
        use_width_cell=True,
        width_bottleneck=8,
        width_gate_init=6.0,
    )
    provider = CandidateProvider(cfg)
    cands = provider.build(
        x,
        question_indices=torch.tensor([[0, 2]]),
        hisa_evidence_indices=torch.tensor([[[0], [0], [1], [2], [3]]]),
    )
    block = DSQGWBlock.from_config(cfg).eval()
    assert block.width_cell is not None

    with torch.no_grad():
        block.width_cell.type_pair_bias.zero_()
        block.width_cell.type_pair_bias[
            int(CandidateType.QUESTION), int(CandidateType.HISA_EVIDENCE)
        ].fill_(8.0)

    _, telemetry = block(
        x,
        cands.cand_states,
        cands.cand_types,
        cands.cand_sources,
        cands.cand_mask,
    )

    assert telemetry["dsqg_w_width_question_to_hisa_evidence_mass"].item() > 0.70
    assert telemetry["dsqg_w_width_self_mass"].item() < 0.60


def test_dsqg_w_block_is_causal_under_future_token_changes() -> None:
    torch.manual_seed(29)
    x_a = make_hidden(batch=1, seq=10, d=16)
    x_b = x_a.clone()
    cut = 5
    x_b[:, cut + 1 :] = torch.randn_like(x_b[:, cut + 1 :]) * 11.0
    cfg = DSQGWConfig(d=16, n_heads=4, max_candidates=12, bottleneck=32, gate_init=-5.0)
    provider = CandidateProvider(cfg)
    block = DSQGWBlock.from_config(cfg).eval()

    kwargs = {"question_indices": torch.tensor([[0, 2, 4]])}
    cands_a = provider.build(x_a, **kwargs)
    cands_b = provider.build(x_b, **kwargs)

    out_a, _ = block(x_a, cands_a.cand_states, cands_a.cand_types, cands_a.cand_sources, cands_a.cand_mask)
    out_b, _ = block(x_b, cands_b.cand_states, cands_b.cand_types, cands_b.cand_sources, cands_b.cand_mask)

    torch.testing.assert_close(out_a[:, : cut + 1], out_b[:, : cut + 1], atol=1e-6, rtol=1e-6)


def test_dsqg_w_block_backward_gives_operator_gradients() -> None:
    torch.manual_seed(31)
    x = make_hidden(seq=8).requires_grad_(True)
    cfg = DSQGWConfig(d=16, n_heads=4, max_candidates=10, bottleneck=32)
    provider = CandidateProvider(cfg)
    cands = provider.build(x.detach(), question_indices=torch.tensor([[0, 1], [0, 2]]))
    cands.cand_states.requires_grad_(True)
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block(
        x,
        cands.cand_states,
        cands.cand_types,
        cands.cand_sources,
        cands.cand_mask,
        return_routing=True,
    )
    loss = out.square().mean() + 0.01 * local_mass_cap_loss(
        telemetry["dsqg_w_probs"], cands.cand_types, cands.cand_mask, cap=0.35
    )
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert cands.cand_states.grad is not None and torch.isfinite(cands.cand_states.grad).all()
    grad_names = {name for name, param in block.named_parameters() if param.grad is not None and param.grad.abs().sum() > 0}
    assert "q_proj.weight" in grad_names
    assert "read_mix.weight" in grad_names
    assert "fuse.0.weight" in grad_names
    assert "fuse.2.weight" in grad_names


def test_answer_and_copy_conflict_losses_are_answer_position_only_and_differentiable() -> None:
    torch.manual_seed(37)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]])
    answer_mask = torch.tensor([[False, True, False, False], [False, False, True, False]])
    bad_copy_mask = torch.zeros_like(logits, dtype=torch.bool)
    bad_copy_mask[0, 1, 5] = True
    bad_copy_mask[1, 2, 6] = True

    ce = answer_masked_loss(logits, labels, answer_mask)
    copy = conditional_copy_unlikelihood_loss(logits, labels, answer_mask, bad_copy_mask, margin=0.25)
    total = ce + 0.1 * copy
    total.backward()

    assert ce.item() > 0.0
    assert copy.item() > 0.0
    assert logits.grad is not None
    assert logits.grad[~answer_mask].abs().sum().item() == pytest.approx(0.0)


def test_local_mass_and_entropy_losses_accept_answer_masks() -> None:
    probs = torch.tensor(
        [[[[0.80, 0.10], [0.10, 0.30], [0.10, 0.60]], [[0.34, 0.20], [0.33, 0.20], [0.33, 0.60]]]],
        requires_grad=True,
    )
    cand_types = torch.tensor([[[int(CandidateType.LOCAL), int(CandidateType.QUESTION), int(CandidateType.HISA_EVIDENCE)], [int(CandidateType.LOCAL), int(CandidateType.QUESTION), int(CandidateType.HISA_EVIDENCE)]]])
    cand_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    answer_mask = torch.tensor([[True, False]])

    local = local_mass_cap_loss(probs, cand_types, cand_mask, answer_mask=answer_mask, cap=0.35)
    entropy = entropy_floor_loss(probs, answer_mask=answer_mask, floor=1.2)
    (local + entropy).backward()

    assert local.item() > 0.0
    assert entropy.item() > 0.0
    assert probs.grad is not None
    assert torch.isfinite(probs.grad).all()


def test_dsqg_w_reference_does_not_allocate_dense_t_by_t_attention() -> None:
    source = inspect.getsource(DSQGWBlock.forward)
    forbidden_fragments = ["T, T", "T,T", "torch.tril", "causal_mask", "attn_mask", "@ k.transpose"]
    assert not any(fragment in source for fragment in forbidden_fragments)
