from __future__ import annotations

import torch

from kernels.hierarchical_sparse_attn_v15_hisa import _build_stage2_token_indices


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

    rowmax_idx, _, _ = _build_stage2_token_indices(
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
    rep_idx, _, _ = _build_stage2_token_indices(
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
