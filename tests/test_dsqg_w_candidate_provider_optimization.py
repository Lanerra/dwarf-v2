from __future__ import annotations

import pytest
import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateProvider, CandidateType, DSQGWBlock, DSQGWConfig


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


def test_sourcewise_accumulation_matches_dense_forward_and_backward() -> None:
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


def test_triton_sourcewise_autograd_matches_eager_sourcewise_backward_on_cuda(monkeypatch) -> None:
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
    assert telemetry["dsqg_w_triton_sourcewise_recompute_backward"].item() == 1.0
    eager_loss = eager_out.square().mean()
    triton_loss = triton_out.square().mean()
    eager_loss.backward()
    triton_loss.backward()
    torch.cuda.synchronize()

    assert triton_x.grad is not None
    assert triton_l3.grad is not None
    assert triton_block.read_mix.weight.grad is not None
    assert triton_block.q_proj.weight.grad is not None
    assert torch.allclose(triton_x.grad, eager_x.grad, atol=3e-4, rtol=3e-4)
    assert torch.allclose(triton_l3.grad, eager_l3.grad, atol=3e-4, rtol=3e-4)
    assert torch.allclose(
        triton_block.read_mix.weight.grad,
        eager_block.read_mix.weight.grad,
        atol=3e-4,
        rtol=3e-4,
    )


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
    assert telemetry["dsqg_w_triton_read_mix_fused"].item() == 1.0


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
