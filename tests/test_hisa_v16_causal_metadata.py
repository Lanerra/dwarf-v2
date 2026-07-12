import math

import torch

from kernels.hierarchical_sparse_attn_v16_hisa_causal import (
    HierarchicalSparseAttentionV16HISACausal,
    _build_causal_tile_metadata_blocked,
    _build_causal_tile_metadata_eager,
    _completed_chunk_representatives,
)


def _metadata_inputs():
    torch.manual_seed(20260711)
    batch, heads, seq_len, head_dim = 2, 2, 65, 8
    chunks, selector_tile = 4, 8
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn_like(query)
    valid_lengths = torch.tensor([65, 47])
    chunk_size = math.ceil(seq_len / chunks)
    route_logits = torch.matmul(query, key[:, :, ::chunk_size, :].transpose(-2, -1))
    return query, key, route_logits, valid_lengths, chunk_size, chunks, selector_tile


def _assert_same_metadata(actual, expected):
    assert torch.equal(actual.top_chunk_idx, expected.top_chunk_idx)
    assert torch.equal(actual.token_idx, expected.token_idx)
    assert torch.equal(actual.token_scores, expected.token_scores)
    assert torch.equal(actual.tile_starts, expected.tile_starts)
    assert torch.equal(actual.valid_lengths, expected.valid_lengths)
    assert actual.chunk_size == expected.chunk_size
    assert actual.selector_tile_size == expected.selector_tile_size


def test_blocked_metadata_matches_eager_oracle_with_mixed_valid_lengths():
    query, key, route_logits, valid_lengths, chunk_size, chunks, selector_tile = _metadata_inputs()
    kwargs = dict(
        num_chunks=chunks,
        chunk_size=chunk_size,
        top_k_chunks=3,
        top_m_tokens=7,
        selector_tile_size=selector_tile,
        valid_lengths=valid_lengths,
    )
    eager = _build_causal_tile_metadata_eager(query, key, route_logits, **kwargs)
    blocked = _build_causal_tile_metadata_blocked(query, key, route_logits, tile_block_size=4, **kwargs)
    _assert_same_metadata(blocked, eager)


def test_blocked_metadata_never_exports_future_or_invalid_tokens():
    query, key, route_logits, valid_lengths, chunk_size, chunks, selector_tile = _metadata_inputs()
    metadata = _build_causal_tile_metadata_blocked(
        query,
        key,
        route_logits,
        num_chunks=chunks,
        chunk_size=chunk_size,
        top_k_chunks=3,
        top_m_tokens=7,
        selector_tile_size=selector_tile,
        valid_lengths=valid_lengths,
        tile_block_size=4,
    )
    for tile, tile_start in enumerate(metadata.tile_starts.tolist()):
        token_ids = metadata.token_idx[:, :, tile]
        valid = token_ids >= 0
        starts = torch.full_like(token_ids, tile_start)
        lengths = valid_lengths[:, None, None, None].expand_as(token_ids)
        assert torch.all(token_ids[valid] < starts[valid])
        assert torch.all(token_ids[valid] < lengths[valid])


def test_v16_metadata_builder_defaults_to_eager_and_keeps_blocked_opt_in(monkeypatch):
    default = HierarchicalSparseAttentionV16HISACausal(D=16, H=2, hd=8, num_chunks=2)
    assert default.metadata_builder == "eager"
    assert default.metadata_tile_block_size == 8

    monkeypatch.setenv("DWARF_HISA_V16_METADATA_BUILDER", "blocked")
    monkeypatch.setenv("DWARF_HISA_V16_METADATA_TILE_BLOCK", "1")
    blocked = HierarchicalSparseAttentionV16HISACausal(D=16, H=2, hd=8, num_chunks=2)
    assert blocked.metadata_builder == "blocked"
    assert blocked.metadata_tile_block_size == 1


def test_blocked_builder_matches_eager_module_output_and_input_gradient():
    torch.manual_seed(23)
    eager = HierarchicalSparseAttentionV16HISACausal(D=16, H=2, hd=8, num_chunks=4, backend="eager")
    blocked = HierarchicalSparseAttentionV16HISACausal(D=16, H=2, hd=8, num_chunks=4, backend="eager")
    blocked.load_state_dict(eager.state_dict())
    eager.metadata_builder = "eager"
    blocked.metadata_builder = "blocked"
    blocked.metadata_tile_block_size = 4
    x_eager = torch.randn(2, 65, 16, requires_grad=True)
    x_blocked = x_eager.detach().clone().requires_grad_(True)
    valid_lengths = torch.tensor([65, 47])
    y_eager = eager(x_eager, valid_lengths=valid_lengths)
    y_blocked = blocked(x_blocked, valid_lengths=valid_lengths)
    torch.testing.assert_close(y_blocked, y_eager, atol=0, rtol=0)
    y_eager.square().mean().backward()
    y_blocked.square().mean().backward()
    torch.testing.assert_close(x_blocked.grad, x_eager.grad, atol=1e-9, rtol=1e-6)


def test_top2_blend_alpha_zero_is_exact_max_l2_rollback():
    _, key, _, valid_lengths, chunk_size, chunks, _ = _metadata_inputs()
    baseline = _completed_chunk_representatives(
        key, num_chunks=chunks, chunk_size=chunk_size, valid_lengths=valid_lengths
    )
    rollback = _completed_chunk_representatives(
        key,
        num_chunks=chunks,
        chunk_size=chunk_size,
        valid_lengths=valid_lengths,
        representative_mode="top2_blend",
        top2_blend_alpha=0.0,
    )
    torch.testing.assert_close(rollback, baseline, atol=0, rtol=0)
