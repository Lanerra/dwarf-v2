from __future__ import annotations

import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateProvider, CandidateType, DSQGWBlock, DSQGWConfig


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
