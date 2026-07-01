from __future__ import annotations

import torch
import pytest

from kernels.hierarchical_sparse_attn_v15_hisa import (
    HierarchicalSparseAttentionV15HISA,
    _build_stage2_token_indices,
    _pack_hisa_selected_tokens_for_dsqg_w,
)


def test_stage2_query_representative_selector_uses_routing_chosen_rows_not_rowmax() -> None:
    # Two chunks of four tokens.  Query chunk 1 attends to key chunk 0.
    # Row-max sees a larger score from query row 4 -> key token 0; the
    # representative path is routed to query row 7 and should choose key token 3.
    bsz, heads, seq_len, chunks, chunk_size, hd = 1, 1, 8, 2, 4, 2
    q = torch.zeros(bsz, heads, seq_len, hd)
    k = torch.zeros_like(q)
    k[0, 0, 0] = torch.tensor([1.0, 0.0])
    k[0, 0, 3] = torch.tensor([0.0, 1.0])
    q[0, 0, 4] = torch.tensor([10.0, 0.0])
    q[0, 0, 7] = torch.tensor([0.0, 5.0])
    top_k = torch.tensor([[[[-1], [0]]]], dtype=torch.long)

    rowmax_idx, _, _, _ = _build_stage2_token_indices(
        q,
        k,
        top_k,
        B=bsz,
        H=heads,
        N=seq_len,
        num_chunks=chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=1,
        stage2_rep_r=0,
    )
    routing = torch.zeros(bsz, heads, seq_len, chunks)
    routing[0, 0, 7, 0] = 1.0
    rep_idx, _, _, _ = _build_stage2_token_indices(
        q,
        k,
        top_k,
        B=bsz,
        H=heads,
        N=seq_len,
        num_chunks=chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=1,
        routing_weights=routing,
        stage2_rep_r=1,
    )

    assert rowmax_idx[0, 0, 1, 0, 0].item() == 0
    assert rep_idx[0, 0, 1, 0, 0].item() == 3


def test_stage2_query_representative_selector_requires_routing_weights() -> None:
    q = torch.zeros(1, 1, 4, 2)
    k = torch.zeros_like(q)
    top_k = torch.tensor([[[[-1], [0]]]], dtype=torch.long)

    try:
        _build_stage2_token_indices(
            q,
            k,
            top_k,
            B=1,
            H=1,
            N=4,
            num_chunks=2,
            chunk_size=2,
            hisa_top_m_tokens=1,
            stage2_rep_r=1,
        )
    except ValueError as exc:
        assert "routing_weights" in str(exc)
    else:
        raise AssertionError("stage2_rep_r must require routing_weights")


def test_hisa_stage2_rep_r_defaults_to_rep4_with_rowmax_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HISA_STAGE2_REP_R", raising=False)
    monkeypatch.delenv("DWARF_HISA_STAGE2_REP_R", raising=False)
    default_attn = HierarchicalSparseAttentionV15HISA(D=16, H=2, hd=8, num_chunks=2)
    assert default_attn.stage2_rep_r == 4

    monkeypatch.setenv("DWARF_HISA_STAGE2_REP_R", "0")
    rowmax_attn = HierarchicalSparseAttentionV15HISA(D=16, H=2, hd=8, num_chunks=2)
    assert rowmax_attn.stage2_rep_r == 0


def test_stage2_query_representative_scores_keep_pooled_max_for_duplicate_tokens() -> None:
    # Regression for the representative union/fill merge: if one representative
    # contributes a token with a low score, a later/higher representative score
    # for the same token must not be shadowed by that low duplicate.
    bsz = heads = 1
    seq_len, chunks, chunk_size, hd = 8, 2, 4, 4
    q = torch.zeros(bsz, heads, seq_len, hd)
    k = torch.zeros_like(q)
    k[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    k[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    k[0, 0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    q[0, 0, 4] = torch.tensor([5.0, 0.0, 0.0, 0.0])
    q[0, 0, 5] = torch.tensor([100.0, 101.0, 10.0, 0.0])
    top_k = torch.tensor([[[[-1], [0]]]], dtype=torch.long)
    routing = torch.zeros(bsz, heads, seq_len, chunks)
    routing[0, 0, 4:8, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])

    row_idx, row_scores, _, _ = _build_stage2_token_indices(
        q,
        k,
        top_k,
        B=bsz,
        H=heads,
        N=seq_len,
        num_chunks=chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=2,
        stage2_rep_r=0,
    )
    rep_idx, rep_scores, _, _ = _build_stage2_token_indices(
        q,
        k,
        top_k,
        B=bsz,
        H=heads,
        N=seq_len,
        num_chunks=chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=2,
        routing_weights=routing,
        stage2_rep_r=chunk_size,
    )

    assert rep_idx[0, 0, 1, 0].tolist() == row_idx[0, 0, 1, 0].tolist()
    assert rep_scores[0, 0, 1, 0].tolist() == pytest.approx(
        row_scores[0, 0, 1, 0].tolist()
    )


def test_stage2_query_representative_selector_preserves_partial_chunk_bounds() -> None:
    torch.manual_seed(123)
    bsz, heads, seq_len, chunks, chunk_size, hd = 1, 2, 10, 3, 4, 3
    top_k_slots = 2
    q = torch.randn(bsz, heads, chunks * chunk_size, hd)
    k = torch.randn_like(q)
    top_k = torch.tensor([[[[-1, -1], [0, 1], [1, 2]], [[-1, -1], [0, 1], [1, 2]]]])
    routing = torch.rand(bsz, heads, seq_len, chunks)

    idx, scores, m_actual, _ = _build_stage2_token_indices(
        q,
        k,
        top_k,
        B=bsz,
        H=heads,
        N=seq_len,
        num_chunks=chunks,
        chunk_size=chunk_size,
        hisa_top_m_tokens=3,
        routing_weights=routing,
        stage2_rep_r=2,
    )

    assert idx.shape == (bsz, heads, chunks, top_k_slots, m_actual)
    assert scores.shape == idx.shape
    valid = idx >= 0
    assert torch.isfinite(scores[valid]).all()
    assert (scores[~valid] == float("-inf")).all()
    if bool(valid.any()):
        assert (idx[valid] < seq_len).all()
        assert (idx[valid] >= 0).all()


def test_pack_hisa_selected_tokens_for_dsqg_w_is_bounded_causal_and_score_sorted() -> None:
    # HISA internally stores selected tokens as [B,H,C_query,K,M]. DSQG-W
    # needs per-token bounded candidates [B,T,J] so it composes actual retrieved
    # evidence instead of doing independent offset retrieval.
    token_idx = torch.tensor(
        [[[[[0, 2, 3]], [[1, 4, 6]]]]],
        dtype=torch.int32,
    )
    token_scores = torch.tensor(
        [[[[[0.1, 9.0, 2.0]], [[7.0, 1.0, 8.0]]]]],
        dtype=torch.float32,
    )

    indices, scores = _pack_hisa_selected_tokens_for_dsqg_w(
        token_idx,
        token_scores,
        seq_len=8,
        chunk_size=4,
        max_candidates=2,
    )

    assert indices.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    # Query chunk 0 owns positions 0..3. For t=2, token 3 is future-invalid,
    # so the highest valid retrieved evidence is token 2, then token 0.
    assert indices[0, 2].tolist() == [2, 0]
    assert scores[0, 2].tolist() == pytest.approx([9.0, 0.1])
    # Query chunk 1 owns positions 4..7. For t=7, token 6 outranks token 1.
    assert indices[0, 7].tolist() == [6, 1]
    assert scores[0, 7].tolist() == pytest.approx([8.0, 7.0])
