from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from kernels.dsqg_w.dsqg_w_mvp import (
    CandidateEvidenceBit,
    CandidateLayout,
    CandidateProvider,
    CandidateSource,
    CandidateType,
    DSQGWBlock,
    DSQGWConfig,
    DSQGWEvidenceBindingHub,
    DSQGWEvidencePriorComposer,
    DSQGWTypedCandidateMixer,
    DSQGWWidthCell,
    answer_masked_loss,
    conditional_copy_unlikelihood_loss,
    entropy_floor_loss,
    local_mass_cap_loss,
    width_pair_transfer_loss,
)


def test_dsqg_w_mvp_public_surface_uses_split_module_canonical_objects() -> None:
    import kernels.dsqg_w as pkg
    import kernels.dsqg_w.block as block_module
    import kernels.dsqg_w.candidate_batch as candidate_batch
    import kernels.dsqg_w.candidate_provider as candidate_provider
    import kernels.dsqg_w.candidate_types as candidate_types
    import kernels.dsqg_w.config as config
    import kernels.dsqg_w.dsqg_w_mvp as mvp
    import kernels.dsqg_w.ebh_packet as ebh_packet
    import kernels.dsqg_w.evidence_prior as evidence_prior
    import kernels.dsqg_w.losses as losses
    import kernels.dsqg_w.sourcewise_gather as sourcewise_gather
    import kernels.dsqg_w.sourcewise_read as sourcewise_read
    import kernels.dsqg_w.typed_mixer as typed_mixer
    import kernels.dsqg_w.width_cell as width_cell

    assert mvp.DSQGWConfig is config.DSQGWConfig
    assert pkg.DSQGWConfig is config.DSQGWConfig
    assert mvp.CandidateType is candidate_types.CandidateType
    assert mvp.CandidateSource is candidate_types.CandidateSource
    assert mvp.CandidateEvidenceBit is candidate_types.CandidateEvidenceBit
    assert mvp.Candidate is candidate_batch.Candidate
    assert mvp.CandidateBatch is candidate_batch.CandidateBatch
    assert mvp.CandidateLayout is candidate_batch.CandidateLayout
    assert pkg.CandidateLayout is candidate_batch.CandidateLayout
    assert mvp.CandidateProvider is candidate_provider.CandidateProvider
    assert mvp.DSQGWWidthCell is width_cell.DSQGWWidthCell
    assert mvp.DSQGWTypedCandidateMixer is typed_mixer.DSQGWTypedCandidateMixer
    assert mvp.DSQGWEvidencePriorComposer is evidence_prior.DSQGWEvidencePriorComposer
    assert mvp.DSQGWEvidenceBindingHub is ebh_packet.DSQGWEvidenceBindingHub
    assert mvp.DSQGWBlock is block_module.DSQGWBlock
    assert mvp.DSQGWEvidenceBindingHub.__module__ == "kernels.dsqg_w.ebh_packet"
    assert mvp.DSQGWBlock.__module__ == "kernels.dsqg_w.block"
    assert mvp._DSQGWSourcewiseCandidateStateGather is sourcewise_gather._DSQGWSourcewiseCandidateStateGather
    assert mvp._dsqg_w_candidate_state_gather_kernel is sourcewise_gather._dsqg_w_candidate_state_gather_kernel
    assert mvp._dsqg_w_candidate_state_gather_backward_kernel is sourcewise_gather._dsqg_w_candidate_state_gather_backward_kernel
    assert mvp._DSQGWSourcewiseTritonCompactRead is sourcewise_read._DSQGWSourcewiseTritonCompactRead
    assert mvp._dsqg_w_sourcewise_read_slots_kernel is sourcewise_read._dsqg_w_sourcewise_read_slots_kernel
    assert mvp._dsqg_w_sourcewise_read_slots_backward_kernel is sourcewise_read._dsqg_w_sourcewise_read_slots_backward_kernel
    assert sourcewise_read.CandidateSource is candidate_types.CandidateSource
    assert mvp.width_pair_transfer_loss is width_cell.width_pair_transfer_loss
    assert mvp.answer_masked_loss is losses.answer_masked_loss
    assert mvp.conditional_copy_unlikelihood_loss is losses.conditional_copy_unlikelihood_loss
    assert mvp.local_mass_cap_loss is losses.local_mass_cap_loss
    assert mvp.entropy_floor_loss is losses.entropy_floor_loss
    assert mvp.candidate_recall is losses.candidate_recall

    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=4,
        use_width_cell=True,
        width_bottleneck=8,
        use_typed_mixer=True,
        typed_mixer_bottleneck=8,
        use_evidence_prior=True,
    )
    block = DSQGWBlock.from_config(cfg)

    assert isinstance(cfg, config.DSQGWConfig)
    assert isinstance(block.width_cell, width_cell.DSQGWWidthCell)
    assert isinstance(block.typed_mixer, typed_mixer.DSQGWTypedCandidateMixer)
    assert isinstance(block.evidence_prior, evidence_prior.DSQGWEvidencePriorComposer)
    assert DSQGWBlock._materialize_sourcewise_candidate_states.__globals__["_DSQGWSourcewiseCandidateStateGather"] is sourcewise_gather._DSQGWSourcewiseCandidateStateGather
    assert DSQGWBlock._forward_sourcewise_triton.__globals__["_DSQGWSourcewiseTritonCompactRead"] is sourcewise_read._DSQGWSourcewiseTritonCompactRead
    assert DSQGWBlock._forward_sourcewise_triton.__globals__["_dsqg_w_sourcewise_read_slots_kernel"] is sourcewise_read._dsqg_w_sourcewise_read_slots_kernel


def test_sourcewise_delete_after_test_recompute_candidates_remain_isolated() -> None:
    import kernels.dsqg_w.dsqg_w_mvp as mvp
    import kernels.dsqg_w.sourcewise_read as sourcewise_read

    assert mvp._DSQGWSourcewiseTritonRecompute is sourcewise_read._DSQGWSourcewiseTritonRecompute
    assert mvp._dsqg_w_sourcewise_functional_recompute is sourcewise_read._dsqg_w_sourcewise_functional_recompute
    assert mvp._dsqg_w_sourcewise_read_slots_recompute is sourcewise_read._dsqg_w_sourcewise_read_slots_recompute

    live_sourcewise_sources = "\n".join(
        [
            inspect.getsource(mvp.DSQGWBlock.forward_sourcewise),
            inspect.getsource(mvp.DSQGWBlock._forward_sourcewise_triton),
            inspect.getsource(mvp.DSQGWEvidenceBindingHub.forward_sourcewise_packet),
            inspect.getsource(mvp.DSQGWEvidenceBindingHub._forward_sourcewise_packet_triton_accum),
        ]
    )
    assert "_DSQGWSourcewiseTritonRecompute.apply" not in live_sourcewise_sources
    assert "_dsqg_w_sourcewise_read_slots_recompute(" not in live_sourcewise_sources
    assert "_dsqg_w_sourcewise_functional_recompute(" not in live_sourcewise_sources


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
    t4_bits = batch.evidence_bits[0, 4][batch.cand_mask[0, 4]]
    collapsed_bits = int(t4_bits[final_token3].item())
    assert collapsed_bits & int(CandidateEvidenceBit.QUESTION)
    assert collapsed_bits & int(CandidateEvidenceBit.LOCAL)
    assert collapsed_bits & int(CandidateEvidenceBit.LONG_OFFSET)
    assert batch.evidence_count[0, 4][batch.cand_mask[0, 4]][final_token3].item() == 3
    assert batch.telemetry["dsqg_w_candidate_multi_evidence_fraction"].item() > 0.0
    assert batch.telemetry["dsqg_w_candidate_duplicate_rate"].item() > 0.0


def test_candidate_provider_quota_cap_preserves_non_hisa_candidates_when_enabled() -> None:
    x = make_hidden(batch=1, seq=5)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=4,
        local_offsets=(),
        long_offsets=(),
        k_question=1,
        k_hisa_evidence=4,
        k_l3_skip=1,
        k_chunk=0,
        use_candidate_quotas=True,
        quota_hisa_max=2,
    )
    provider = CandidateProvider(cfg)
    positions = torch.arange(5)
    hisa_indices = torch.stack(
        [positions, (positions - 1).clamp_min(0), (positions - 2).clamp_min(0), (positions - 3).clamp_min(0)],
        dim=-1,
    ).unsqueeze(0)
    hisa_scores = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=x.dtype).reshape(1, 1, 4).expand(1, 5, 4)
    batch = provider.build(
        x,
        question_indices=torch.tensor([[0]]),
        hisa_evidence_indices=hisa_indices,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=(positions - 1).clamp_min(0).view(1, 5, 1),
    )

    valid_types = batch.cand_types[0, 4][batch.cand_mask[0, 4]]
    hisa_family_ids = torch.tensor(
        [
            int(CandidateType.HISA_EVIDENCE),
            int(CandidateType.HISA_EVIDENCE_REP0),
            int(CandidateType.HISA_EVIDENCE_REP1),
            int(CandidateType.HISA_EVIDENCE_REP2),
            int(CandidateType.HISA_EVIDENCE_REP3),
        ],
        dtype=valid_types.dtype,
    )
    hisa_family = torch.isin(valid_types, hisa_family_ids)
    assert hisa_family.sum().item() <= 2
    assert (valid_types == int(CandidateType.QUESTION)).any()
    assert batch.telemetry["dsqg_w_candidate_quota_hisa_clipped_fraction"].item() > 0.0


def test_candidate_provider_preserves_and_prioritizes_dsr_scores_without_offsets() -> None:
    x = make_hidden(batch=1, seq=5)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=2,
        local_offsets=(),
        long_offsets=(),
        k_question=0,
        k_hisa_evidence=3,
        k_chunk=0,
        k_l3_skip=0,
    )
    provider = CandidateProvider(cfg)
    hisa_indices = torch.tensor([[[0, 1, 2], [0, 1, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]])
    hisa_scores = torch.tensor([[[0.0, 1.0, 9.0], [0.0, 8.0, 3.0], [0.0, 4.0, 2.0], [1.0, 7.0, 6.0], [2.0, 5.0, 4.0]]])

    batch = provider.build(
        x,
        hisa_evidence_indices=hisa_indices,
        hisa_evidence_scores=hisa_scores,
    )

    assert batch.cand_scores is not None
    assert batch.cand_token_indices[0, 3].tolist() == [2, 3]
    assert batch.cand_scores[0, 3].tolist() == [7.0, 6.0]
    assert batch.cand_sources[0, 3].tolist() == [int(CandidateSource.HISA), int(CandidateSource.HISA)]
    assert batch.telemetry["dsqg_w_candidate_fraction_local"].item() == 0.0
    assert batch.telemetry["dsqg_w_candidate_score_mean"].item() > 0.0


def test_dsqg_w_block_uses_candidate_scores_as_measurable_routing_bias() -> None:
    torch.manual_seed(101)
    x = make_hidden(batch=1, seq=4, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=2,
        local_offsets=(),
        long_offsets=(),
        k_question=0,
        k_hisa_evidence=2,
        k_chunk=0,
        k_l3_skip=0,
        bottleneck=32,
        gate_init=-2.5,
    )
    provider = CandidateProvider(cfg)
    indices = torch.tensor([[[0, -1], [0, 1], [0, 2], [1, 3]]])
    low_then_high = torch.tensor([[[0.0, 0.0], [0.0, 8.0], [0.0, 8.0], [0.0, 8.0]]])
    cands = provider.build(x, hisa_evidence_indices=indices, hisa_evidence_scores=low_then_high)
    block = DSQGWBlock.from_config(cfg)

    _, telemetry = block(
        x,
        cands.cand_states,
        cands.cand_types,
        cands.cand_sources,
        cands.cand_mask,
        cand_scores=cands.cand_scores,
        return_routing=True,
    )

    assert telemetry["dsqg_w_candidate_score_bias_norm"].item() > 0.0
    probs = telemetry["dsqg_w_probs"].mean(dim=-1)
    assert cands.cand_scores[0, 3, 0].item() > cands.cand_scores[0, 3, 1].item()
    assert probs[0, 3, 0].item() > probs[0, 3, 1].item()


def test_evidence_prior_composer_zero_init_is_centered_noop_with_telemetry() -> None:
    composer = DSQGWEvidencePriorComposer(n_types=len(CandidateType), n_sources=len(CandidateSource), clip=2.0)
    cand_types = torch.tensor([[[int(CandidateType.HISA_EVIDENCE), int(CandidateType.QUESTION), int(CandidateType.LOCAL)]]])
    cand_sources = torch.tensor([[[int(CandidateSource.HISA), int(CandidateSource.FINAL), int(CandidateSource.FINAL)]]])
    cand_mask = torch.ones_like(cand_types, dtype=torch.bool)
    cand_scores = torch.tensor([[[5.0, 0.0, -1.0]]])
    evidence_bits = torch.tensor([[[
        int(CandidateEvidenceBit.HISA) | int(CandidateEvidenceBit.QUESTION),
        int(CandidateEvidenceBit.QUESTION),
        int(CandidateEvidenceBit.LOCAL),
    ]]])
    evidence_count = torch.tensor([[[2, 1, 1]]])
    distances = torch.tensor([[[4, 2, 1]]])

    prior, telemetry = composer(
        cand_types,
        cand_sources,
        cand_mask,
        raw_hisa_scores=cand_scores,
        evidence_bits=evidence_bits,
        evidence_count=evidence_count,
        candidate_distances=distances,
    )

    assert torch.equal(prior, torch.zeros_like(prior))
    assert telemetry["dsqg_w_prior_norm"].item() == 0.0
    assert telemetry["dsqg_w_prior_clip_fraction"].item() == 0.0
    assert telemetry["dsqg_w_prior_multi_evidence_fraction"].item() > 0.0


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
    assert telemetry["dsqg_w_width_hisa_evidence_to_question_mass"].item() > 0.0
    assert telemetry["dsqg_w_width_self_mass"].item() < 0.60


def test_dsqg_w_width_transfer_loss_treats_typed_hisa_reps_as_evidence_family() -> None:
    cand_types = torch.tensor([[[
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE_REP0),
        int(CandidateType.HISA_EVIDENCE_REP1),
    ]]])
    cand_mask = torch.ones_like(cand_types, dtype=torch.bool)
    probs = torch.zeros(1, 1, 3, 3)
    probs[0, 0, 0, 1] = 0.90  # QUESTION reads HISA rep.
    probs[0, 0, 0, 0] = 0.10
    probs[0, 0, 1, 0] = 0.80  # HISA rep reads QUESTION.
    probs[0, 0, 1, 1] = 0.20
    probs[0, 0, 2, 0] = 0.70  # Another HISA rep reads QUESTION.
    probs[0, 0, 2, 2] = 0.30

    loss = width_pair_transfer_loss(probs, cand_types, cand_mask)

    assert loss.item() < 0.25


def test_dsqg_w_width_cell_relation_features_can_affect_lateral_routing() -> None:
    cand_states = torch.zeros(1, 1, 3, 16)
    cand_states[0, 0, 0, 0] = 1.0
    cand_states[0, 0, 1, 0] = 1.0
    cand_states[0, 0, 2, 0] = -1.0
    cand_types = torch.tensor([[[
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE),
        int(CandidateType.LOCAL),
    ]]])
    cand_sources = torch.tensor([[[
        int(CandidateSource.FINAL),
        int(CandidateSource.HISA),
        int(CandidateSource.FINAL),
    ]]])
    cand_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=3,
        use_width_cell=True,
        width_bottleneck=8,
        width_gate_init=6.0,
    )
    block = DSQGWBlock.from_config(cfg).eval()
    assert block.width_cell is not None
    with torch.no_grad():
        block.width_cell.q_proj.weight.zero_()
        block.width_cell.k_proj.weight.zero_()
        block.width_cell.type_pair_bias.zero_()
        block.width_cell.source_pair_bias.zero_()
        block.width_cell.self_bias.fill_(-8.0)
        block.width_cell.rel_diff_proj.weight.zero_()
        block.width_cell.rel_prod_proj.weight.zero_()
        block.width_cell.rel_prod_proj.weight[0, 0] = 1.0
        block.width_cell.rel_diff_score.zero_()
        block.width_cell.rel_prod_score.zero_()
        block.width_cell.rel_prod_score[0] = 8.0

    _, telemetry = block.width_cell(cand_states, cand_types, cand_sources, cand_mask)

    assert telemetry["dsqg_w_width_question_to_hisa_evidence_mass"].item() > 0.80
    assert telemetry["dsqg_w_width_self_mass"].item() < 0.20


def test_dsqg_w_width_cell_relation_scores_are_split_and_nonzero_at_init() -> None:
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=3,
        use_width_cell=True,
        width_bottleneck=8,
    )
    block = DSQGWBlock.from_config(cfg)
    assert block.width_cell is not None

    assert hasattr(block.width_cell, "rel_diff_score")
    assert hasattr(block.width_cell, "rel_prod_score")
    assert block.width_cell.rel_diff_score.shape == (8,)
    assert block.width_cell.rel_prod_score.shape == (8,)
    assert block.width_cell.rel_diff_score.detach().abs().sum().item() > 0.0
    assert block.width_cell.rel_prod_score.detach().abs().sum().item() > 0.0


def test_candidate_provider_can_label_hisa_evidence_slots_as_query_representatives() -> None:
    x = make_hidden(batch=1, seq=6, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        local_offsets=(),
        long_offsets=(),
        k_question=0,
        k_hisa_evidence=4,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(cfg)
    hisa = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 2, 0], [0, 1, 2, 3], [1, 2, 3, 4]]])

    batch = provider.build(x, hisa_evidence_indices=hisa)

    valid_types = batch.cand_types[batch.cand_mask].tolist()
    assert int(CandidateType.HISA_EVIDENCE_REP0) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP1) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP2) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP3) in valid_types
    assert int(CandidateType.HISA_EVIDENCE) not in valid_types
    assert batch.telemetry["dsqg_w_candidate_fraction_hisa_evidence_rep0"].item() > 0.0


def test_specialized_metadata_fast_path_preserves_typed_hisa_representatives() -> None:
    x = make_hidden(batch=1, seq=6, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=4,
        k_l3_skip=1,
        k_chunk=0,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(cfg)
    positions = torch.arange(6)
    question = torch.tensor([[0, 3]])
    hisa = torch.stack([(positions - i).clamp_min(0) for i in [1, 2, 3, 4]], dim=-1).unsqueeze(0)
    scores = torch.arange(24, dtype=x.dtype).reshape(1, 6, 4)
    l3_skip = (positions - 5).clamp_min(0).reshape(1, 6, 1)

    batch = provider.build_metadata(
        x,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=scores,
        l3_skip_indices=l3_skip,
    )

    valid_types = batch.cand_types[batch.cand_mask].tolist()
    assert batch.telemetry["dsqg_w_candidate_specialized_metadata"].item() == 1.0
    assert int(CandidateType.HISA_EVIDENCE_REP0) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP1) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP2) in valid_types
    assert int(CandidateType.HISA_EVIDENCE_REP3) in valid_types


def test_typed_candidate_mixer_is_bounded_and_near_identity_when_closed() -> None:
    torch.manual_seed(53)
    x = make_hidden(batch=1, seq=7, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=10,
        bottleneck=32,
        gate_init=-5.0,
        use_typed_mixer=True,
        typed_mixer_bottleneck=8,
        typed_mixer_gate_init=-12.0,
        typed_hisa_reps=True,
        k_hisa_evidence=4,
    )
    provider = CandidateProvider(cfg)
    cands = provider.build(
        x,
        question_indices=torch.tensor([[0, 2, 5]]),
        hisa_evidence_indices=torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 2, 0], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]]),
    )
    block = DSQGWBlock.from_config(cfg).eval()
    assert isinstance(block.typed_mixer, DSQGWTypedCandidateMixer)

    out, telemetry = block(x, cands.cand_states, cands.cand_types, cands.cand_sources, cands.cand_mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_typed_mixer_gate_mean"].item() < 1e-4
    assert telemetry["dsqg_w_typed_mixer_entropy"].item() > 0.0
    assert telemetry["dsqg_w_typed_mixer_delta_norm"].item() >= 0.0


def test_forced_lateral_gates_use_constant_gate_values_and_keep_masked_candidates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(54)
    cand_states = torch.randn(1, 3, 5, 16)
    cand_types = torch.tensor([[[1, 2, 3, 0, 0], [1, 4, 5, 0, 0], [1, 2, 5, 6, 0]]])
    cand_sources = torch.tensor([[[1, 2, 2, 0, 0], [1, 2, 3, 0, 0], [1, 2, 3, 3, 0]]])
    cand_mask = torch.tensor([[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]], dtype=torch.bool)

    monkeypatch.setenv("DWARF_DSQG_W_FORCE_TYPED_MIXER_GATE", "0.7")
    monkeypatch.setenv("DWARF_DSQG_W_FORCE_WIDTH_GATE", "0.7")

    typed = DSQGWTypedCandidateMixer(d=16, n_heads=4, n_types=8, bottleneck=8, gate_init=-12.0).eval()
    mixed, typed_tel = typed(cand_states, cand_types, cand_mask)

    width = DSQGWWidthCell(d=16, n_heads=4, n_types=8, n_sources=4, bottleneck=8, gate_init=-12.0).eval()
    widened, width_tel = width(mixed, cand_types, cand_sources, cand_mask)

    assert typed_tel["dsqg_w_typed_mixer_forced_gate"].item() == 1.0
    assert width_tel["dsqg_w_width_forced_gate"].item() == 1.0
    assert typed_tel["dsqg_w_typed_mixer_gate_mean"].item() == pytest.approx(0.7)
    assert width_tel["dsqg_w_width_gate_mean"].item() == pytest.approx(0.7)
    assert torch.allclose(mixed.masked_select(~cand_mask[..., None]), cand_states.masked_select(~cand_mask[..., None]))
    assert torch.allclose(widened.masked_select(~cand_mask[..., None]), mixed.masked_select(~cand_mask[..., None]))


def test_forced_ebh_gate_uses_constant_value_and_zero_evidence_stays_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(63)
    x = torch.randn(2, 4, 16)
    cand_states = torch.randn(2, 4, 3, 16)
    cand_types = torch.zeros(2, 4, 3, dtype=torch.long)
    cand_sources = torch.zeros(2, 4, 3, dtype=torch.long)
    cand_mask = torch.zeros(2, 4, 3, dtype=torch.bool)

    monkeypatch.setenv("DWARF_DSQG_W_FORCE_EBH_GATE", "0.7")

    hub = DSQGWEvidenceBindingHub(d=16, n_types=8, n_sources=4, bottleneck=8, gate_init=-12.0).eval()
    out, telemetry, aux = hub(x, cand_states, cand_types, cand_sources, cand_mask, return_aux=True)

    assert torch.equal(out, x)
    assert telemetry["dsqg_w_ebh_forced_gate"].item() == 1.0
    assert telemetry["dsqg_w_ebh_bind_gate_mean"].item() == pytest.approx(0.7)
    assert telemetry["dsqg_w_ebh_active_row_fraction"].item() == 0.0
    assert torch.allclose(aux["bind_gate"], torch.full_like(aux["bind_gate"], 0.7))


def test_ebh_pair_mixer_is_bounded_k2_and_preserves_invalid_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_FORCE_EBH_PAIR_GATE", "0.7")
    torch.manual_seed(71)
    hub = DSQGWEvidenceBindingHub(
        d=16,
        n_types=len(CandidateType),
        n_sources=len(CandidateSource),
        bottleneck=8,
        gate_init=-2.0,
        phase_bands=2,
        use_pair_mixer=True,
        pair_rank=8,
    )
    x = torch.randn(2, 3, 16)
    cand_states = torch.randn(2, 3, 5, 16)
    cand_types = torch.tensor(
        [
            [[2, 6, 7, 1, 0], [2, 8, 1, 0, 0], [6, 7, 2, 1, 0]],
            [[2, 6, 1, 0, 0], [7, 2, 6, 1, 0], [2, 1, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    cand_sources = torch.tensor(
        [
            [[1, 3, 3, 1, 0], [1, 3, 1, 0, 0], [3, 3, 1, 1, 0]],
            [[1, 3, 1, 0, 0], [3, 1, 3, 1, 0], [1, 1, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    cand_mask = cand_types != int(CandidateType.NULL)

    out, telemetry, aux = hub(x, cand_states, cand_types, cand_sources, cand_mask, return_aux=True)

    assert out.shape == x.shape
    assert aux["aligned_candidates"].shape == cand_states.shape
    assert aux["ebh_pair_probs"].shape == (2, 3, 5, 5)
    valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
    assert torch.equal(
        aux["ebh_pair_probs"].masked_select(~valid_pair),
        torch.zeros_like(aux["ebh_pair_probs"]).masked_select(~valid_pair),
    )
    assert torch.equal(
        aux["aligned_candidates"].masked_select(~cand_mask[..., None]),
        torch.zeros_like(aux["aligned_candidates"]).masked_select(~cand_mask[..., None]),
    )
    assert telemetry["dsqg_w_ebh_pair_mixer_enabled"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_pair_gate_mean"].item() == pytest.approx(0.7)
    assert telemetry["dsqg_w_ebh_pair_forced_gate"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_pair_entropy"].item() > 0.0


def test_sourcewise_path_materializes_semantic_candidate_machinery_instead_of_rejecting() -> None:
    torch.manual_seed(57)
    x = make_hidden(batch=1, seq=6, d=16).requires_grad_(True)
    l3 = (x * 1.25).clone()
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        bottleneck=32,
        gate_init=-2.5,
        local_offsets=(),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=4,
        k_l3_skip=1,
        use_width_cell=True,
        width_bottleneck=8,
        width_gate_init=-4.0,
        use_typed_mixer=True,
        typed_mixer_bottleneck=8,
        typed_mixer_gate_init=-4.0,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(cfg)
    positions = torch.arange(6)
    question = torch.tensor([[0, 3]])
    hisa = torch.stack([(positions - i).clamp_min(0) for i in [1, 2, 3, 4]], dim=-1).unsqueeze(0)
    scores = torch.arange(24, dtype=x.dtype).reshape(1, 6, 4)
    l3_skip = (positions - 5).clamp_min(0).reshape(1, 6, 1)
    metadata = provider.build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=scores,
        l3_skip_indices=l3_skip,
    )
    block = DSQGWBlock.from_config(cfg)

    out, telemetry = block.forward_sourcewise(
        x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=l3,
        cand_scores=metadata.cand_scores,
        return_routing=True,
    )
    out.square().mean().backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert telemetry["dsqg_w_sourcewise_semantic_materialized"].item() == 1.0
    assert telemetry["dsqg_w_triton_sourcewise_semantic_bypass"].item() == 0.0
    assert "dsqg_w_width_gate_mean" in telemetry
    assert "dsqg_w_typed_mixer_gate_mean" in telemetry


def test_sourcewise_lane_b_routes_through_ebh_packet_before_w_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_EBH_SOURCEWISE_PACKET", "1")
    torch.manual_seed(58)
    x = make_hidden(batch=1, seq=5, d=16)
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        bottleneck=32,
        local_offsets=(),
        long_offsets=(),
        k_question=1,
        k_hisa_evidence=2,
        k_l3_skip=1,
        use_evidence_binding_hub=True,
        ebh_score_features=True,
    )
    provider = CandidateProvider(cfg)
    positions = torch.arange(5)
    metadata = provider.build_metadata(
        x,
        question_indices=torch.tensor([[0]]),
        hisa_evidence_indices=torch.stack([(positions - 1).clamp_min(0), (positions - 2).clamp_min(0)], dim=-1).unsqueeze(0),
        hisa_evidence_scores=torch.ones(1, 5, 2),
        l3_skip_indices=(positions - 3).clamp_min(0).reshape(1, 5, 1),
    )
    block = DSQGWBlock.from_config(cfg).eval()
    assert block.evidence_binding_hub is not None
    calls = {"packet": 0}

    def fake_packet(x_arg, *args, **kwargs):
        calls["packet"] += 1
        return x_arg, {
            "dsqg_w_ebh_packet_sourcewise": x_arg.new_tensor(1.0),
            "dsqg_w_ebh_packet_triton": x_arg.new_tensor(1.0),
        }

    monkeypatch.setattr(block.evidence_binding_hub, "forward_sourcewise_packet", fake_packet)
    out, telemetry = block.forward_sourcewise(
        x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        cand_scores=metadata.cand_scores,
    )

    assert calls["packet"] == 1
    assert out.shape == x.shape
    assert telemetry["dsqg_w_sourcewise_ebh_materialized"].item() == pytest.approx(0.0)
    assert telemetry["dsqg_w_ebh_packet_sourcewise"].item() == pytest.approx(1.0)
    assert telemetry["dsqg_w_ebh_packet_triton"].item() == pytest.approx(1.0)


def test_ebh_packet_triton_accum_guard_stays_score_on_lane_b_only() -> None:
    source = inspect.getsource(DSQGWEvidenceBindingHub.forward_sourcewise_packet)
    assert "self._forward_sourcewise_packet_triton_accum" in source
    assert "DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM" in source
    assert "self.use_score_features" in source
    assert "not self.use_pair_mixer" in source


def test_query_conditioned_type_bias_can_route_scores_to_hisa_rep_candidate() -> None:
    torch.manual_seed(59)
    x = torch.zeros(1, 5, 16)
    x[:, :, 0] = 1.0
    cfg = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=6,
        bottleneck=32,
        gate_init=-5.0,
        local_offsets=(1,),
        long_offsets=(),
        k_question=0,
        k_hisa_evidence=1,
        use_query_type_bias=True,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(cfg)
    cands = provider.build(x, hisa_evidence_indices=torch.tensor([[[0], [0], [1], [2], [3]]]))
    block = DSQGWBlock.from_config(cfg).eval()
    block.norm_x = nn.Identity()
    with torch.no_grad():
        block.q_proj.weight.zero_()
        block.k_proj.weight.zero_()
        block.type_bias.zero_()
        block.source_bias.zero_()
        block.role_key.weight.zero_()
        block.source_key.weight.zero_()
        block.query_type_bias.weight.zero_()
        for head in range(cfg.n_heads):
            out_idx = int(CandidateType.HISA_EVIDENCE_REP0) * cfg.n_heads + head
            block.query_type_bias.weight[out_idx, 0] = 8.0

    _, telemetry = block(
        x,
        cands.cand_states,
        cands.cand_types,
        cands.cand_sources,
        cands.cand_mask,
        return_routing=True,
    )

    assert telemetry["dsqg_w_hisa_evidence_rep0_mass"].item() > 0.80
    assert telemetry["dsqg_w_query_type_bias_norm"].item() > 0.0


def test_width_pair_transfer_loss_rewards_question_evidence_lateral_mass() -> None:
    probs = torch.tensor(
        [[[
            [0.20, 0.70, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.20, 0.60],
        ]]],
        requires_grad=True,
    )
    cand_types = torch.tensor([[[
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE),
        int(CandidateType.LOCAL),
    ]]])
    cand_mask = torch.ones(1, 1, 3, dtype=torch.bool)

    loss = width_pair_transfer_loss(probs, cand_types, cand_mask)
    loss.backward()

    expected = -0.5 * (torch.log(torch.tensor(0.70)) + torch.log(torch.tensor(0.60)))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert probs.grad is not None
    assert probs.grad[0, 0, 0, 1].item() < 0.0
    assert probs.grad[0, 0, 1, 0].item() < 0.0
    # Balanced geometric pressure penalizes the weaker reverse direction more strongly.
    assert abs(probs.grad[0, 0, 1, 0].item()) > abs(probs.grad[0, 0, 0, 1].item())


def test_width_pair_transfer_loss_penalizes_one_way_collapse() -> None:
    cand_types = torch.tensor([[[
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE),
        int(CandidateType.LOCAL),
    ]]])
    cand_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    balanced = torch.tensor([[[
        [0.20, 0.45, 0.35],
        [0.45, 0.20, 0.35],
        [0.20, 0.20, 0.60],
    ]]])
    collapsed = torch.tensor([[[
        [0.01, 0.89, 0.10],
        [0.01, 0.89, 0.10],
        [0.20, 0.20, 0.60],
    ]]])

    balanced_loss = width_pair_transfer_loss(balanced, cand_types, cand_mask)
    collapsed_loss = width_pair_transfer_loss(collapsed, cand_types, cand_mask)

    assert collapsed_loss.item() > balanced_loss.item()


def test_width_pair_transfer_loss_adds_entropy_floor_penalty() -> None:
    cand_types = torch.tensor([[[
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE),
        int(CandidateType.LOCAL),
    ]]])
    cand_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    high_entropy = torch.tensor([[[
        [0.275, 0.45, 0.275],
        [0.45, 0.275, 0.275],
        [0.30, 0.30, 0.40],
    ]]])
    low_entropy = torch.tensor([[[
        [0.01, 0.45, 0.54],
        [0.45, 0.01, 0.54],
        [0.01, 0.01, 0.98],
    ]]])

    high = width_pair_transfer_loss(
        high_entropy,
        cand_types,
        cand_mask,
        entropy_floor=1.0,
        entropy_weight=0.5,
    )
    low = width_pair_transfer_loss(
        low_entropy,
        cand_types,
        cand_mask,
        entropy_floor=1.0,
        entropy_weight=0.5,
    )
    no_floor_high = width_pair_transfer_loss(high_entropy, cand_types, cand_mask)
    no_floor_low = width_pair_transfer_loss(low_entropy, cand_types, cand_mask)

    assert no_floor_high.item() == pytest.approx(no_floor_low.item(), rel=1e-5)
    assert low.item() > high.item()


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
