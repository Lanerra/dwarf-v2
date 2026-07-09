from __future__ import annotations

import pytest
import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateProvider, CandidateSource, CandidateType, DSQGWBlock, DSQGWConfig


def _sourcewise_fixture():
    torch.manual_seed(20260701)
    config = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1, 2),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    final_states = torch.randn(2, 7, 16, requires_grad=True)
    l3_states = torch.randn(2, 7, 16, requires_grad=True)
    positions = torch.arange(7)
    question_indices = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    hisa_indices = torch.stack(
        [(positions - 1).clamp_min(0), (positions - 3).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    hisa_scores = torch.linspace(-0.3, 0.5, steps=14).reshape(1, 7, 2).expand(2, -1, -1).contiguous()
    l3_skip_indices = (positions - 2).clamp_min(0).view(1, 7, 1).expand(2, -1, -1).contiguous()
    return config, final_states, l3_states, question_indices, hisa_indices, hisa_scores, l3_skip_indices


def test_sourcewise_accumulation_matches_dense_forward_and_backward(monkeypatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "0")
    config, x, l3, question, hisa, hisa_scores, l3_skip = _sourcewise_fixture()
    dense_provider = CandidateProvider(config)
    sourcewise_provider = CandidateProvider(config)
    dense_block = DSQGWBlock.from_config(config)
    sourcewise_block = DSQGWBlock.from_config(config)
    sourcewise_block.load_state_dict(dense_block.state_dict())
    dense_x = x.detach().clone().requires_grad_(True)
    dense_l3 = l3.detach().clone().requires_grad_(True)
    source_x = x.detach().clone().requires_grad_(True)
    source_l3 = l3.detach().clone().requires_grad_(True)

    dense_candidates = dense_provider.build(
        dense_x,
        l3_states=dense_l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    metadata = sourcewise_provider.build_metadata(
        source_x,
        l3_states=source_l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    dense_out, dense_telemetry = dense_block(
        dense_x,
        dense_candidates.cand_states,
        dense_candidates.cand_types,
        dense_candidates.cand_sources,
        dense_candidates.cand_mask,
        cand_scores=dense_candidates.cand_scores,
        return_routing=True,
    )
    source_out, source_telemetry = sourcewise_block.forward_sourcewise(
        source_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=source_l3,
        cand_scores=metadata.cand_scores,
        return_routing=True,
    )

    assert metadata.cand_states.numel() == 0
    assert torch.equal(metadata.cand_token_indices, dense_candidates.cand_token_indices)
    assert torch.equal(metadata.cand_types, dense_candidates.cand_types)
    assert torch.equal(metadata.cand_sources, dense_candidates.cand_sources)
    assert torch.equal(metadata.cand_mask, dense_candidates.cand_mask)
    assert torch.allclose(source_out, dense_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(source_telemetry["dsqg_w_probs"], dense_telemetry["dsqg_w_probs"], atol=1e-6, rtol=1e-6)
    for key in ("dsqg_w_read_norm", "dsqg_w_hisa_source_mass", "dsqg_w_l3_source_mass"):
        assert torch.allclose(source_telemetry[key], dense_telemetry[key], atol=1e-6, rtol=1e-6)

    dense_loss = dense_out.square().mean() + dense_telemetry["dsqg_w_probs"].square().mean()
    source_loss = source_out.square().mean() + source_telemetry["dsqg_w_probs"].square().mean()
    dense_loss.backward()
    source_loss.backward()

    assert torch.allclose(source_x.grad, dense_x.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(source_l3.grad, dense_l3.grad, atol=1e-5, rtol=1e-5)


def test_sourcewise_path_does_not_build_candidate_state_or_projected_kv_surfaces(monkeypatch) -> None:
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "0")
    config, x, l3, question, hisa, hisa_scores, l3_skip = _sourcewise_fixture()
    provider = CandidateProvider(config)
    block = DSQGWBlock.from_config(config)

    def forbidden_gather(*args, **kwargs):
        raise AssertionError("build_metadata must not gather/materialize candidate states")

    monkeypatch.setattr(provider, "_gather_states", forbidden_gather)
    metadata = provider.build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )

    projected_input_shapes = []
    original_k = block.k_proj.forward
    original_v = block.v_proj.forward

    def k_spy(arg):
        projected_input_shapes.append(tuple(arg.shape))
        assert arg.ndim == 3
        return original_k(arg)

    def v_spy(arg):
        projected_input_shapes.append(tuple(arg.shape))
        assert arg.ndim == 3
        return original_v(arg)

    monkeypatch.setattr(block.k_proj, "forward", k_spy)
    monkeypatch.setattr(block.v_proj, "forward", v_spy)

    out, telemetry = block.forward_sourcewise(
        x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=l3,
        cand_scores=metadata.cand_scores,
    )

    assert metadata.cand_states.numel() == 0
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_sourcewise"].item() == 1.0
    assert projected_input_shapes
    assert all(len(shape) == 3 for shape in projected_input_shapes)


def test_dsr_selected_metadata_specialization_matches_generic_path(monkeypatch) -> None:
    torch.manual_seed(202607020)
    config = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        typed_hisa_reps=False,
    )
    provider = CandidateProvider(config)
    final_states = torch.randn(2, 7, 16)
    l3_states = torch.randn(2, 7, 16)
    positions = torch.arange(7)
    question = torch.tensor([[0, 3, 3, 9], [0, 2, 5, 9]], dtype=torch.long)
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 1).clamp_min(0),
            (positions - 3).clamp_min(0),
            torch.full_like(positions, 99),
        ],
        dim=-1,
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    hisa_scores = torch.tensor([0.1, 0.9, -0.2, 4.0], dtype=final_states.dtype).reshape(1, 1, 4).expand(2, 7, -1)
    l3_skip = torch.stack(
        [(positions - 2).clamp_min(0), (positions - 2).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()

    monkeypatch.setenv("DWARF_DSQG_W_SPECIALIZED_METADATA", "0")
    generic = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    monkeypatch.setenv("DWARF_DSQG_W_SPECIALIZED_METADATA", "1")
    specialized = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )

    assert specialized.cand_states.numel() == 0
    assert specialized.cand_token_indices.shape[-1] == 11

    def valid_rows(batch):
        return torch.stack(
            [batch.cand_token_indices, batch.cand_types, batch.cand_sources],
            dim=-1,
        )[batch.cand_mask]

    assert torch.equal(valid_rows(specialized), valid_rows(generic))
    assert torch.allclose(specialized.cand_scores[specialized.cand_mask], generic.cand_scores[generic.cand_mask])
    assert torch.equal(specialized.evidence_bits[specialized.cand_mask], generic.evidence_bits[generic.cand_mask])
    assert torch.equal(specialized.evidence_count[specialized.cand_mask], generic.evidence_count[generic.cand_mask])
    assert torch.equal(
        specialized.candidate_distances[specialized.cand_mask],
        generic.candidate_distances[generic.cand_mask],
    )
    assert specialized.valid_candidate_count.float().mean() == generic.valid_candidate_count.float().mean()
    assert specialized.active_source_ids == generic.active_source_ids
    assert specialized.telemetry["dsqg_w_candidate_specialized_metadata"].item() == 1.0
    assert specialized.telemetry["dsqg_w_candidate_slot_count"].item() == 11.0
    assert generic.telemetry.get("dsqg_w_candidate_specialized_metadata", torch.tensor(0.0)).item() == 0.0
    assert generic.telemetry["dsqg_w_candidate_slot_count"].item() == 16.0
    assert set(specialized.active_source_ids) == {
        int(CandidateSource.FINAL),
        int(CandidateSource.HISA),
        int(CandidateSource.L3),
        int(CandidateSource.NULL),
    }


def test_candidate_provider_reuses_static_metadata_cache(monkeypatch) -> None:
    torch.manual_seed(20260708)
    config = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(config)
    final_states = torch.randn(2, 7, 16)
    l3_states = torch.randn(2, 7, 16)
    positions = torch.arange(7)
    question = torch.tensor([[0, 3, 3, 9], [0, 2, 5, 9]], dtype=torch.long)
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 3).clamp_min(0),
            (positions - 4).clamp_min(0),
            torch.full_like(positions, 99),
        ],
        dim=-1,
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    hisa_scores = torch.tensor([0.9, 0.1, -0.2, 4.0], dtype=final_states.dtype).reshape(1, 1, 4).expand(2, 7, -1)
    l3_skip = torch.stack(
        [(positions - 2).clamp_min(0), (positions - 5).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()

    monkeypatch.setenv("DWARF_DSQG_W_SPECIALIZED_METADATA", "1")
    first = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    positions_cached = next(iter(provider._positions_cache.values()))
    priority_cached = next(iter(provider._priority_table_cache.values()))
    arange_cached = provider._arange_cache[(4, str(final_states.device))]
    rep_types_cached = provider._typed_hisa_rep_type_ids_cache[str(final_states.device)]

    second = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )

    assert next(iter(provider._positions_cache.values())).data_ptr() == positions_cached.data_ptr()
    assert next(iter(provider._priority_table_cache.values())).data_ptr() == priority_cached.data_ptr()
    assert provider._arange_cache[(4, str(final_states.device))].data_ptr() == arange_cached.data_ptr()
    assert provider._typed_hisa_rep_type_ids_cache[str(final_states.device)].data_ptr() == rep_types_cached.data_ptr()
    assert torch.equal(second.cand_token_indices, first.cand_token_indices)
    assert torch.equal(second.cand_types, first.cand_types)
    assert torch.equal(second.cand_sources, first.cand_sources)
    assert torch.equal(second.cand_mask, first.cand_mask)
    assert torch.allclose(second.cand_scores, first.cand_scores)
    assert second.telemetry["dsqg_w_candidate_specialized_metadata"].item() == 1.0


def test_candidate_diagnostic_telemetry_is_opt_in(monkeypatch) -> None:
    torch.manual_seed(202607081)
    config = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(config)
    final_states = torch.randn(2, 7, 16)
    l3_states = torch.randn(2, 7, 16)
    positions = torch.arange(7)
    question = torch.tensor([[0, 3, 3, 9], [0, 2, 5, 9]], dtype=torch.long)
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 3).clamp_min(0),
            (positions - 4).clamp_min(0),
            torch.full_like(positions, 99),
        ],
        dim=-1,
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    hisa_scores = torch.tensor([0.9, 0.1, -0.2, 4.0], dtype=final_states.dtype).reshape(1, 1, 4).expand(2, 7, -1)
    l3_skip = torch.stack(
        [(positions - 2).clamp_min(0), (positions - 5).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()

    monkeypatch.delenv("DWARF_DSQG_W_CANDIDATE_DIAGNOSTICS", raising=False)
    normal = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    normal_keys = set(normal.telemetry)
    assert {
        "dsqg_w_candidate_duplicate_rate",
        "dsqg_w_candidate_invalid_rate",
        "dsqg_w_valid_candidate_count",
        "dsqg_w_static_source_count",
        "dsqg_w_candidate_specialized_metadata",
        "dsqg_w_candidate_slot_count",
    }.issubset(normal_keys)
    assert "dsqg_w_candidate_score_mean" not in normal_keys
    assert "dsqg_w_candidate_fraction_question" not in normal_keys
    assert "dsqg_w_candidate_multi_evidence_fraction" not in normal_keys

    monkeypatch.setenv("DWARF_DSQG_W_CANDIDATE_DIAGNOSTICS", "1")
    diagnostic = provider.build_metadata(
        final_states,
        l3_states=l3_states,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    diagnostic_keys = set(diagnostic.telemetry)
    assert torch.equal(diagnostic.cand_token_indices, normal.cand_token_indices)
    assert torch.equal(diagnostic.cand_types, normal.cand_types)
    assert torch.equal(diagnostic.cand_sources, normal.cand_sources)
    assert torch.equal(diagnostic.cand_mask, normal.cand_mask)
    assert torch.equal(diagnostic.evidence_bits, normal.evidence_bits)
    assert torch.equal(diagnostic.evidence_count, normal.evidence_count)
    assert torch.equal(diagnostic.candidate_distances, normal.candidate_distances)
    assert "dsqg_w_candidate_score_mean" in diagnostic_keys
    assert "dsqg_w_candidate_fraction_question" in diagnostic_keys
    assert "dsqg_w_candidate_multi_evidence_fraction" in diagnostic_keys


def test_grouped_slot_materialization_matches_general_source_group_path(monkeypatch) -> None:
    torch.manual_seed(20260705)
    config = DSQGWConfig(
        d=16,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        typed_hisa_reps=True,
    )
    provider = CandidateProvider(config)
    block = DSQGWBlock.from_config(config)
    x = torch.randn(2, 7, 16, requires_grad=True)
    l3 = torch.randn(2, 7, 16, requires_grad=True)
    positions = torch.arange(7)
    question = torch.tensor([[0, 3, 3, 9], [0, 2, 5, 9]], dtype=torch.long)
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 3).clamp_min(0),
            (positions - 5).clamp_min(0),
            torch.full_like(positions, 99),
        ],
        dim=-1,
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    hisa_scores = torch.linspace(-0.5, 0.5, steps=2 * 7 * 4).reshape(2, 7, 4)
    l3_skip = torch.stack(
        [(positions - 2).clamp_min(0), torch.full_like(positions, 99)], dim=-1
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    metadata = provider.build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )

    monkeypatch.setenv("DWARF_DSQG_W_GROUPED_SLOT_MATERIALIZE", "0")
    general_x = x.detach().clone().requires_grad_(True)
    general_l3 = l3.detach().clone().requires_grad_(True)
    general = block._materialize_sourcewise_candidate_states(
        general_x,
        metadata.cand_token_indices,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=general_l3,
    )

    monkeypatch.setenv("DWARF_DSQG_W_GROUPED_SLOT_MATERIALIZE", "1")
    grouped_x = x.detach().clone().requires_grad_(True)
    grouped_l3 = l3.detach().clone().requires_grad_(True)
    grouped = block._materialize_sourcewise_candidate_states(
        grouped_x,
        metadata.cand_token_indices,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=grouped_l3,
    )

    assert torch.allclose(grouped, general, atol=0.0, rtol=0.0)
    general.square().mean().backward()
    grouped.square().mean().backward()
    assert torch.allclose(grouped_x.grad, general_x.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(grouped_l3.grad, general_l3.grad, atol=0.0, rtol=0.0)


def _require_cuda_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton DSQG-W sourcewise tests")
    pytest.importorskip("triton")


def _cuda_sourcewise_metadata(config: DSQGWConfig, *, seq_len: int):
    device = torch.device("cuda")
    x = torch.randn(1, seq_len, config.d, device=device)
    l3 = torch.randn(1, seq_len, config.d, device=device)
    positions = torch.arange(seq_len, device=device)
    question_base = torch.tensor([[0, 3, 7, 11]], device=device, dtype=torch.long)
    question = question_base[:, : config.k_question].contiguous()
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 5).clamp_min(0),
            (positions - 9).clamp_min(0),
            (positions - 17).clamp_min(0),
        ],
        dim=-1,
    )[:, : config.k_hisa_evidence].unsqueeze(0).contiguous()
    hisa_scores = torch.linspace(-0.4, 0.4, steps=seq_len * config.k_hisa_evidence, device=device).reshape(
        1, seq_len, config.k_hisa_evidence
    )
    l3_skip = torch.stack(
        [(positions - 2).clamp_min(0), (positions - 6).clamp_min(0)], dim=-1
    )[:, : config.k_l3_skip].unsqueeze(0).contiguous()
    metadata = CandidateProvider(config).build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    return x, l3, metadata


def test_triton_candidate_state_gather_matches_eager_materialization_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607052)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        typed_hisa_reps=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=16)
    block = DSQGWBlock.from_config(config).cuda()

    monkeypatch.setenv("DWARF_DSQG_W_TRITON_CAND_STATE_GATHER", "0")
    eager_x = x.detach().clone().requires_grad_(True)
    eager_l3 = l3.detach().clone().requires_grad_(True)
    eager = block._materialize_sourcewise_candidate_states(
        eager_x,
        metadata.cand_token_indices,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=eager_l3,
    )

    monkeypatch.setenv("DWARF_DSQG_W_TRITON_CAND_STATE_GATHER", "1")
    triton_x = x.detach().clone().requires_grad_(True)
    triton_l3 = l3.detach().clone().requires_grad_(True)
    triton_out = block._materialize_sourcewise_candidate_states(
        triton_x,
        metadata.cand_token_indices,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=triton_l3,
    )

    torch.cuda.synchronize()
    assert torch.allclose(triton_out, eager, atol=0.0, rtol=0.0)
    eager.square().mean().backward()
    triton_out.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.allclose(triton_x.grad, eager_x.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(triton_l3.grad, eager_l3.grad, atol=0.0, rtol=0.0)


def test_triton_transformed_compact_read_matches_materialized_allopen_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607053)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_width_cell=True,
        width_bottleneck=16,
        width_gate_init=-1.5,
        width_entropy_floor=1.5,
        width_entropy_weight=0.25,
        use_typed_mixer=True,
        typed_mixer_bottleneck=16,
        typed_mixer_gate_init=-1.5,
        use_query_type_bias=True,
        typed_hisa_reps=True,
        use_evidence_prior=True,
        evidence_prior_init_scale=0.0,
        use_candidate_quotas=True,
        quota_hisa_max=4,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=16)
    base_block = DSQGWBlock.from_config(config).cuda().train()
    fused_block = DSQGWBlock.from_config(config).cuda().train()
    fused_block.load_state_dict(base_block.state_dict())

    monkeypatch.setenv("DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ", "0")
    base_x = x.detach().clone().requires_grad_(True)
    base_l3 = l3.detach().clone().requires_grad_(True)
    base_out, base_tel = base_block.forward_sourcewise(
        base_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=base_l3,
        cand_scores=metadata.cand_scores,
        evidence_bits=metadata.evidence_bits,
        evidence_count=metadata.evidence_count,
        candidate_distances=metadata.candidate_distances,
    )
    base_loss = base_out.square().mean() + 0.001 * base_tel["dsqg_w_width_aux_loss"]
    base_loss.backward()

    monkeypatch.setenv("DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ", "1")
    monkeypatch.setenv("DWARF_DSQG_W_MATERIALIZED_COMPACT_READ_BACKWARD", "triton")
    monkeypatch.setenv("DWARF_DSQG_W_SOURCEWISE_WIDTH_CELL_FUSION", "1")
    fused_x = x.detach().clone().requires_grad_(True)
    fused_l3 = l3.detach().clone().requires_grad_(True)
    fused_out, fused_tel = fused_block.forward_sourcewise(
        fused_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=fused_l3,
        cand_scores=metadata.cand_scores,
        evidence_bits=metadata.evidence_bits,
        evidence_count=metadata.evidence_count,
        candidate_distances=metadata.candidate_distances,
    )
    fused_loss = fused_out.square().mean() + 0.001 * fused_tel["dsqg_w_width_aux_loss"]
    fused_loss.backward()
    torch.cuda.synchronize()

    assert fused_tel["dsqg_w_triton_transformed_compact_read"].item() == 1.0
    assert fused_tel["dsqg_w_sourcewise_width_cell_fusion"].item() == 1.0
    assert fused_tel["dsqg_w_sourcewise_semantic_materialized"].item() == 0.0
    assert torch.allclose(fused_out, base_out, atol=2e-4, rtol=2e-4)
    assert torch.allclose(fused_x.grad, base_x.grad, atol=5e-4, rtol=5e-4)
    assert torch.allclose(fused_l3.grad, base_l3.grad, atol=5e-4, rtol=5e-4)
    base_params = dict(base_block.named_parameters())
    fused_params = dict(fused_block.named_parameters())
    for name in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "role_key.weight",
        "source_key.weight",
        "width_cell.q_proj.weight",
        "typed_mixer.q_proj.weight",
    ):
        base_grad = base_params[name].grad
        fused_grad = fused_params[name].grad
        assert base_grad is not None and fused_grad is not None
        assert torch.allclose(fused_grad, base_grad, atol=8e-4, rtol=8e-4), name



def test_projected_width_control_uses_triton_sourcewise_without_semantic_materialization(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607052)
    config = DSQGWConfig(
        d=128,
        n_heads=4,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_width_cell=True,
        width_bottleneck=16,
        width_gate_init=-1.5,
        width_entropy_floor=1.5,
        width_entropy_weight=0.25,
        use_typed_mixer=True,
        typed_mixer_bottleneck=16,
        typed_mixer_gate_init=-1.5,
        use_query_type_bias=True,
        typed_hisa_reps=True,
        use_evidence_prior=True,
        evidence_prior_init_scale=0.0,
        use_candidate_quotas=True,
        quota_hisa_max=4,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=16)
    block = DSQGWBlock.from_config(config).cuda().train()
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
    monkeypatch.setenv("DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL", "1")
    monkeypatch.setenv("DWARF_DSQG_W_PROJECTED_WIDTH_BIAS_SCALE", "3.0")

    x_req = x.detach().clone().requires_grad_(True)
    l3_req = l3.detach().clone().requires_grad_(True)
    out, telemetry = block.forward_sourcewise(
        x_req,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=l3_req,
        cand_scores=metadata.cand_scores,
        evidence_bits=metadata.evidence_bits,
        evidence_count=metadata.evidence_count,
        candidate_distances=metadata.candidate_distances,
    )
    loss = out.square().mean() + 0.001 * telemetry["dsqg_w_width_aux_loss"]
    loss.backward()
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_triton_sourcewise"].item() == 1.0
    assert telemetry["dsqg_w_projected_width_control"].item() == 1.0
    assert telemetry["dsqg_w_projected_width_semantic_control"].item() == 1.0
    assert telemetry["dsqg_w_sourcewise_semantic_materialized"].item() == 0.0
    assert telemetry["dsqg_w_typed_mixer_projected_bypass"].item() == 1.0
    assert x_req.grad is not None and torch.isfinite(x_req.grad).all()
    assert l3_req.grad is not None and torch.isfinite(l3_req.grad).all()
    width_grad = block.width_cell.q_proj.weight.grad
    assert width_grad is not None
    assert torch.isfinite(width_grad).all()
    assert width_grad.abs().sum() > 0


def test_triton_sourcewise_matches_eager_sourcewise_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607011)
    config = DSQGWConfig(
        d=512,
        n_heads=8,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=32)
    eager_block = DSQGWBlock.from_config(config).cuda().eval()
    triton_block = DSQGWBlock.from_config(config).cuda().eval()
    triton_block.load_state_dict(eager_block.state_dict())

    with torch.no_grad():
        eager_out, eager_telemetry = eager_block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=metadata.cand_scores,
            return_routing=True,
        )
        monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
        triton_out, triton_telemetry = triton_block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=metadata.cand_scores,
            return_routing=True,
        )

    torch.cuda.synchronize()
    assert metadata.cand_states.numel() == 0
    assert triton_telemetry["dsqg_w_triton_sourcewise"].item() == 1.0
    assert torch.allclose(triton_out, eager_out, atol=2e-5, rtol=2e-5)
    assert torch.allclose(triton_telemetry["dsqg_w_probs"], eager_telemetry["dsqg_w_probs"], atol=2e-5, rtol=2e-5)
    for key in ("dsqg_w_read_norm", "dsqg_w_hisa_source_mass", "dsqg_w_l3_source_mass"):
        assert torch.allclose(triton_telemetry[key], eager_telemetry[key], atol=2e-5, rtol=2e-5)


def test_triton_sourcewise_avoids_forbidden_candidate_surfaces_with_padded_hd(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607012)
    config = DSQGWConfig(
        d=20,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=9)
    block = DSQGWBlock.from_config(config).cuda().eval()
    projected_input_shapes: list[tuple[int, ...]] = []
    original_k = block.k_proj.forward
    original_v = block.v_proj.forward

    def k_spy(arg):
        projected_input_shapes.append(tuple(arg.shape))
        assert arg.ndim == 3
        return original_k(arg)

    def v_spy(arg):
        projected_input_shapes.append(tuple(arg.shape))
        assert arg.ndim == 3
        return original_v(arg)

    def forbidden_gather(*args, **kwargs):
        raise AssertionError("Triton sourcewise must not use eager per-candidate gather rows")

    monkeypatch.setattr(block.k_proj, "forward", k_spy)
    monkeypatch.setattr(block.v_proj, "forward", v_spy)
    monkeypatch.setattr(block, "_gather_source_rows", forbidden_gather)
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")

    with torch.no_grad():
        out, telemetry = block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=metadata.cand_scores,
        )

    torch.cuda.synchronize()
    assert metadata.cand_states.numel() == 0
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert telemetry["dsqg_w_triton_sourcewise"].item() == 1.0
    assert projected_input_shapes
    assert all(len(shape) == 3 for shape in projected_input_shapes)


def test_triton_sourcewise_autograd_uses_compact_read_backward_not_full_recompute_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607013)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=9)
    eager_x = x.detach().clone().requires_grad_(True)
    eager_l3 = l3.detach().clone().requires_grad_(True)
    triton_x = x.detach().clone().requires_grad_(True)
    triton_l3 = l3.detach().clone().requires_grad_(True)
    eager_block = DSQGWBlock.from_config(config).cuda()
    triton_block = DSQGWBlock.from_config(config).cuda()
    triton_block.load_state_dict(eager_block.state_dict())

    import kernels.dsqg_w.dsqg_w_mvp as dsqg_w_mvp

    def forbidden_full_recompute(*args, **kwargs):
        raise AssertionError("Triton no-routing backward must not call the full PyTorch recompute helper")

    monkeypatch.setattr(dsqg_w_mvp, "_dsqg_w_sourcewise_functional_recompute", forbidden_full_recompute)

    eager_out, _ = eager_block.forward_sourcewise(
        eager_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=eager_l3,
        cand_scores=metadata.cand_scores,
    )
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
    triton_out, telemetry = triton_block.forward_sourcewise(
        triton_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=triton_l3,
        cand_scores=metadata.cand_scores,
    )

    assert torch.allclose(triton_out, eager_out, atol=2e-5, rtol=2e-5)
    assert telemetry["dsqg_w_triton_sourcewise_recompute_backward"].item() == 0.0
    assert telemetry["dsqg_w_triton_compact_read_backward"].item() == 1.0
    eager_loss = eager_out.square().mean()
    triton_loss = triton_out.square().mean()
    eager_loss.backward()
    triton_loss.backward()
    torch.cuda.synchronize()

    assert triton_x.grad is not None
    assert triton_l3.grad is not None
    assert triton_block.read_mix.weight.grad is not None
    assert triton_block.q_proj.weight.grad is not None
    assert triton_block.k_proj.weight.grad is not None
    assert triton_block.v_proj.weight.grad is not None
    assert triton_block.norm_z.weight.grad is not None
    assert triton_block.fuse[0].weight.grad is not None
    assert triton_block.fuse[2].weight.grad is not None
    assert triton_block.gate.grad is not None
    assert torch.allclose(triton_x.grad, eager_x.grad, atol=3e-4, rtol=3e-4)
    assert torch.allclose(triton_l3.grad, eager_l3.grad, atol=3e-4, rtol=3e-4)
    for name, triton_param, eager_param in (
        ("q_proj.weight", triton_block.q_proj.weight, eager_block.q_proj.weight),
        ("k_proj.weight", triton_block.k_proj.weight, eager_block.k_proj.weight),
        ("v_proj.weight", triton_block.v_proj.weight, eager_block.v_proj.weight),
        ("read_mix.weight", triton_block.read_mix.weight, eager_block.read_mix.weight),
        ("norm_z.weight", triton_block.norm_z.weight, eager_block.norm_z.weight),
        ("fuse.0.weight", triton_block.fuse[0].weight, eager_block.fuse[0].weight),
        ("fuse.2.weight", triton_block.fuse[2].weight, eager_block.fuse[2].weight),
        ("gate", triton_block.gate, eager_block.gate),
    ):
        assert triton_param.grad is not None, name
        assert eager_param.grad is not None, name
        assert torch.allclose(triton_param.grad, eager_param.grad, atol=3e-4, rtol=3e-4), name


def test_triton_sourcewise_default_backward_uses_true_kernel_and_no_backward_probs_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607015)
    config = DSQGWConfig(
        d=60,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=9)
    eager_x = x.detach().clone().requires_grad_(True)
    eager_l3 = l3.detach().clone().requires_grad_(True)
    triton_x = x.detach().clone().requires_grad_(True)
    triton_l3 = l3.detach().clone().requires_grad_(True)
    eager_block = DSQGWBlock.from_config(config).cuda()
    triton_block = DSQGWBlock.from_config(config).cuda()
    triton_block.load_state_dict(eager_block.state_dict())

    import kernels.dsqg_w.dsqg_w_mvp as dsqg_w_mvp

    def forbidden_full_recompute(*args, **kwargs):
        raise AssertionError("true compact-read backward must not call the whole-block PyTorch recompute helper")

    def forbidden_compact_python_vjp(*args, **kwargs):
        raise AssertionError("default compact-read backward must use the Triton VJP, not the PyTorch VJP fallback")

    monkeypatch.setattr(dsqg_w_mvp, "_dsqg_w_sourcewise_functional_recompute", forbidden_full_recompute)
    monkeypatch.setattr(dsqg_w_mvp, "_dsqg_w_sourcewise_compact_read_backward_pytorch", forbidden_compact_python_vjp, raising=False)
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
    monkeypatch.delenv("DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD", raising=False)

    eager_out, _ = eager_block.forward_sourcewise(
        eager_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=eager_l3,
        cand_scores=metadata.cand_scores,
    )
    triton_out, telemetry = triton_block.forward_sourcewise(
        triton_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=triton_l3,
        cand_scores=metadata.cand_scores,
    )

    assert torch.allclose(triton_out, eager_out, atol=3e-5, rtol=3e-5)
    assert telemetry["dsqg_w_triton_true_backward"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_v20_split_kernels"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_monolithic_kernel"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_query_kernel"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_source_kernel"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_probs_materialized"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_lse_saved"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_reduction_buffer_bytes"].item() == 0.0
    assert telemetry["dsqg_w_triton_schedule_block_hd"].item() == 16.0
    assert telemetry["dsqg_w_triton_schedule_num_warps"].item() >= 1.0
    assert telemetry["dsqg_w_triton_schedule_num_stages"].item() >= 1.0

    eager_out.float().square().mean().backward()
    triton_out.float().square().mean().backward()
    torch.cuda.synchronize()

    assert torch.allclose(triton_x.grad, eager_x.grad, atol=5e-4, rtol=5e-4)
    assert torch.allclose(triton_l3.grad, eager_l3.grad, atol=5e-4, rtol=5e-4)
    for name, triton_param, eager_param in (
        ("q_proj.weight", triton_block.q_proj.weight, eager_block.q_proj.weight),
        ("k_proj.weight", triton_block.k_proj.weight, eager_block.k_proj.weight),
        ("v_proj.weight", triton_block.v_proj.weight, eager_block.v_proj.weight),
        ("read_mix.weight", triton_block.read_mix.weight, eager_block.read_mix.weight),
        ("role_key.weight", triton_block.role_key.weight, eager_block.role_key.weight),
        ("source_key.weight", triton_block.source_key.weight, eager_block.source_key.weight),
        ("type_bias", triton_block.type_bias, eager_block.type_bias),
        ("source_bias", triton_block.source_bias, eager_block.source_bias),
        ("query_type_bias.weight", triton_block.query_type_bias.weight, eager_block.query_type_bias.weight),
        ("norm_z.weight", triton_block.norm_z.weight, eager_block.norm_z.weight),
        ("fuse.0.weight", triton_block.fuse[0].weight, eager_block.fuse[0].weight),
        ("fuse.2.weight", triton_block.fuse[2].weight, eager_block.fuse[2].weight),
        ("gate", triton_block.gate, eager_block.gate),
    ):
        assert triton_param.grad is not None, name
        assert eager_param.grad is not None, name
        assert torch.allclose(triton_param.grad, eager_param.grad, atol=8e-4, rtol=8e-4), name



def test_triton_sourcewise_query_only_backward_can_skip_source_kv_grads_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607053)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=8)
    block = DSQGWBlock.from_config(config).cuda()
    x_req = x.detach().clone().requires_grad_(True)
    l3_req = l3.detach().clone().requires_grad_(True)
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRADS", "0")

    out, telemetry = block.forward_sourcewise(
        x_req,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=l3_req,
        cand_scores=metadata.cand_scores,
    )
    out.float().square().mean().backward()
    torch.cuda.synchronize()

    assert telemetry["dsqg_w_triton_true_backward"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_source_grads"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_monolithic_kernel"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_query_kernel"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_source_kernel"].item() == 0.0
    assert block.q_proj.weight.grad is not None
    assert block.role_key.weight.grad is not None
    assert block.source_key.weight.grad is not None
    # Source K/V projection gradients are intentionally severed by this speed-control mode.
    assert block.k_proj.weight.grad is None or block.k_proj.weight.grad.abs().sum().item() == 0.0
    assert block.v_proj.weight.grad is None or block.v_proj.weight.grad.abs().sum().item() == 0.0
    assert x_req.grad is not None and torch.isfinite(x_req.grad).all()
    assert l3_req.grad is None or torch.isfinite(l3_req.grad).all()


def test_triton_sourcewise_v20_split_backward_is_opt_in_and_matches_eager_on_cuda(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607016)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=8)
    eager_x = x.detach().clone().requires_grad_(True)
    eager_l3 = l3.detach().clone().requires_grad_(True)
    triton_x = x.detach().clone().requires_grad_(True)
    triton_l3 = l3.detach().clone().requires_grad_(True)
    eager_block = DSQGWBlock.from_config(config).cuda()
    triton_block = DSQGWBlock.from_config(config).cuda()
    triton_block.load_state_dict(eager_block.state_dict())

    eager_out, _ = eager_block.forward_sourcewise(
        eager_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=eager_l3,
        cand_scores=metadata.cand_scores,
    )
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION", "v20_split")
    triton_out, telemetry = triton_block.forward_sourcewise(
        triton_x,
        metadata.cand_token_indices,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        l3_states=triton_l3,
        cand_scores=metadata.cand_scores,
    )

    assert torch.allclose(triton_out, eager_out, atol=3e-5, rtol=3e-5)
    assert telemetry["dsqg_w_triton_true_backward"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_v20_split_kernels"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_monolithic_kernel"].item() == 0.0
    assert telemetry["dsqg_w_triton_backward_query_kernel"].item() == 1.0
    assert telemetry["dsqg_w_triton_backward_source_kernel"].item() == 1.0
    assert telemetry["dsqg_w_triton_score_recompute_blocks"].item() == 2.0

    eager_out.float().square().mean().backward()
    triton_out.float().square().mean().backward()
    torch.cuda.synchronize()

    assert torch.allclose(triton_x.grad, eager_x.grad, atol=5e-4, rtol=5e-4)
    assert torch.allclose(triton_l3.grad, eager_l3.grad, atol=5e-4, rtol=5e-4)
    for name, triton_param, eager_param in (
        ("q_proj.weight", triton_block.q_proj.weight, eager_block.q_proj.weight),
        ("k_proj.weight", triton_block.k_proj.weight, eager_block.k_proj.weight),
        ("v_proj.weight", triton_block.v_proj.weight, eager_block.v_proj.weight),
        ("role_key.weight", triton_block.role_key.weight, eager_block.role_key.weight),
        ("source_key.weight", triton_block.source_key.weight, eager_block.source_key.weight),
        ("type_bias", triton_block.type_bias, eager_block.type_bias),
        ("source_bias", triton_block.source_bias, eager_block.source_bias),
    ):
        assert triton_param.grad is not None, name
        assert eager_param.grad is not None, name
        assert torch.allclose(triton_param.grad, eager_param.grad, atol=8e-4, rtol=8e-4), name


def test_triton_sourcewise_no_routing_and_fused_read_mix_do_not_materialize_forbidden_outputs(monkeypatch) -> None:
    _require_cuda_triton()
    torch.manual_seed(202607014)
    config = DSQGWConfig(
        d=32,
        n_heads=4,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=2,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
        use_query_type_bias=True,
    )
    x, l3, metadata = _cuda_sourcewise_metadata(config, seq_len=9)
    block = DSQGWBlock.from_config(config).cuda().eval()
    monkeypatch.setenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "1")

    with torch.no_grad():
        out, telemetry = block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=metadata.cand_scores,
            return_routing=False,
        )

    torch.cuda.synchronize()
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert "dsqg_w_probs" not in telemetry
    assert telemetry["dsqg_w_triton_probs_materialized"].item() == 0.0
    assert telemetry["dsqg_w_triton_read_accum_materialized"].item() == 0.0
    assert telemetry["dsqg_w_triton_compact_read_slots_materialized"].item() == 1.0
    assert telemetry["dsqg_w_triton_compact_read_slots"].item() == len(block.read_type_ids) + 1
    assert telemetry["dsqg_w_triton_score_recompute_blocks"].item() == 1.0


def test_vectorized_candidate_provider_skips_unused_summary_gather(monkeypatch) -> None:
    config = DSQGWConfig(
        d=8,
        n_heads=2,
        max_candidates=8,
        local_offsets=(1, 2),
        long_offsets=(),
        k_question=1,
        k_hisa_evidence=2,
        k_l3_skip=1,
        k_chunk=0,
    )
    provider = CandidateProvider(config)
    final_states = torch.randn(2, 6, 8)
    l3_states = torch.randn(2, 6, 8)
    question_indices = torch.tensor([[0], [1]])
    positions = torch.arange(6)
    hisa_indices = torch.stack([(positions - 1).clamp_min(0), (positions - 3).clamp_min(0)], dim=-1).unsqueeze(0).expand(2, -1, -1)
    l3_skip_indices = (positions - 2).clamp_min(0).view(1, 6, 1).expand(2, -1, -1)

    original = provider._gather_states
    gathered_bases: list[int] = []

    def spy(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        gathered_bases.append(id(states))
        return original(states, token_indices)

    monkeypatch.setattr(provider, "_gather_states", spy)

    batch = provider.build(
        final_states,
        l3_states=l3_states,
        question_indices=question_indices,
        hisa_evidence_indices=hisa_indices,
        l3_skip_indices=l3_skip_indices,
    )

    assert batch.cand_states.shape == (2, 6, 8, 8)
    assert torch.isfinite(batch.cand_states).all()
    assert gathered_bases == [id(final_states), id(l3_states)]


def test_vectorized_candidate_provider_reuses_final_gather_for_missing_l3_states(monkeypatch) -> None:
    config = DSQGWConfig(
        d=8,
        n_heads=2,
        max_candidates=8,
        local_offsets=(1,),
        long_offsets=(),
        k_question=0,
        k_hisa_evidence=2,
        k_l3_skip=0,
        k_chunk=0,
    )
    provider = CandidateProvider(config)
    final_states = torch.randn(1, 6, 8)
    positions = torch.arange(6)
    hisa_indices = torch.stack([(positions - 1).clamp_min(0), (positions - 2).clamp_min(0)], dim=-1).unsqueeze(0)

    original = provider._gather_states
    call_count = 0

    def spy(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original(states, token_indices)

    monkeypatch.setattr(provider, "_gather_states", spy)

    batch = provider.build(final_states, hisa_evidence_indices=hisa_indices)

    assert batch.cand_states.shape == (1, 6, 8, 8)
    assert torch.isfinite(batch.cand_states).all()
    assert call_count == 1


def test_dsqg_w_block_read_mix_matches_dense_typed_read_reference() -> None:
    torch.manual_seed(123)
    block = DSQGWBlock.from_config(
        DSQGWConfig(
            d=16,
            n_heads=4,
            max_candidates=6,
            local_offsets=(1,),
            long_offsets=(),
            k_question=1,
            k_hisa_evidence=1,
            k_l3_skip=1,
            k_chunk=0,
        )
    )
    probs = torch.rand(2, 5, 6, 4)
    probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-8)
    v = torch.randn(2, 5, 6, 4, 4)
    cand_types = torch.tensor(
        [
            [[1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0]],
            [[1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0], [1, 2, 3, 6, 0, 0]],
        ],
        dtype=torch.long,
    )
    cand_mask = cand_types != int(CandidateType.NULL)
    r_all = (probs[..., None] * v).sum(dim=2).reshape(2, 5, 16)

    typed_reads = []
    for type_id in range(block.n_types):
        type_mask = ((cand_types == type_id) & cand_mask)[:, :, :, None, None]
        p_type = probs[..., None].masked_fill(~type_mask, 0.0)
        typed_reads.append((p_type * v).sum(dim=2).reshape(2, 5, 16))
    expected = block.read_mix(torch.cat([r_all] + typed_reads, dim=-1))

    actual, norms = block._mix_typed_reads(r_all, probs, v, cand_types, cand_mask)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert len(norms) == block.n_types
    assert norms[int(CandidateType.LOCAL)].item() > 0.0
    assert norms[int(CandidateType.LONG_OFFSET)].item() == 0.0


def test_dsqg_w_compact_read_slot_batched_read_mix_matches_slot_linears(monkeypatch) -> None:
    torch.manual_seed(202607021)
    block = DSQGWBlock.from_config(
        DSQGWConfig(
            d=16,
            n_heads=4,
            max_candidates=6,
            local_offsets=(),
            long_offsets=(),
            k_question=1,
            k_hisa_evidence=1,
            k_l3_skip=1,
            k_chunk=0,
            null_fallback=True,
        )
    )
    read_slots = torch.randn(2, 5, len(block.read_type_ids) + 1, 16, requires_grad=True)

    expected = block._mix_compact_read_slots(read_slots, batched=False)
    monkeypatch.setenv("DWARF_DSQG_W_BATCHED_READ_MIX", "1")
    actual = block._mix_compact_read_slots(read_slots)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    grad = torch.randn_like(actual)
    expected.backward(grad, retain_graph=True)
    grad_slots_expected = read_slots.grad.detach().clone()
    grad_weight_expected = block.read_mix.weight.grad.detach().clone()
    read_slots.grad.zero_()
    block.read_mix.weight.grad.zero_()
    actual.backward(grad)
    assert torch.allclose(read_slots.grad, grad_slots_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(block.read_mix.weight.grad, grad_weight_expected, atol=1e-6, rtol=1e-6)


def test_dsqg_w_block_from_config_limits_read_mix_to_possible_candidate_types() -> None:
    block = DSQGWBlock.from_config(
        DSQGWConfig(
            d=16,
            n_heads=4,
            max_candidates=6,
            local_offsets=(1,),
            long_offsets=(),
            k_question=1,
            k_hisa_evidence=1,
            k_l3_skip=1,
            k_chunk=0,
            null_fallback=True,
        )
    )

    assert set(block.read_type_ids) == {
        int(CandidateType.LOCAL),
        int(CandidateType.QUESTION),
        int(CandidateType.HISA_EVIDENCE),
        int(CandidateType.L3_SKIP),
        int(CandidateType.NULL),
    }
    assert block._read_slot_count == len(block.read_type_ids) + 1
    assert block._read_slot_block >= block._read_slot_count
    assert block._read_slot_block & (block._read_slot_block - 1) == 0
    slot_map = block._type_slot_map_for(torch.empty((), device=block._type_slot_map.device))
    assert slot_map.dtype == torch.int32
    assert slot_map[int(CandidateType.LOCAL)].item() == block.read_type_ids.index(int(CandidateType.LOCAL)) + 1
    assert slot_map[int(CandidateType.LONG_OFFSET)].item() == -1
    assert "_type_slot_map" not in block.state_dict()
