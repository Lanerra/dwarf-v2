"""Strict-causal HISA V16: local evidence plus completed-chunk global evidence.

V15 shared routing/token metadata over an entire query chunk.  That makes an
otherwise causal final attention mask insufficient: later query rows and keys
inside the active chunk can alter an earlier row's candidate set or route prior.

V16 has an explicit control/data-plane split:

* local lane: every query attends to the strict recent window ``[q-W, q)``;
* global lane: query-tile metadata is chosen from the tile's first query and
  only chunks whose exclusive end precedes that tile start; and
* continuous route scores remain per-query, while the discrete selections are
  deliberately detached.

The eager path is the canonical semantic oracle and CPU fallback.  CUDA uses
tile-aligned Triton forward/backward kernels; the backward preserves the same
strict local/global candidate topology while accumulating irregular key/value
fan-in with FP32 atomics.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by CPU-only installations
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@dataclass(frozen=True)
class HISAMetadata:
    """Detached, tile-scoped discrete routing data for V16.

    ``top_chunk_idx`` has ``[B,H,T,K]`` and ``token_idx`` / ``token_scores``
    have ``[B,H,T,K,M]``.  Invalid entries are ``-1`` / ``-inf``.
    """

    top_chunk_idx: torch.Tensor
    token_idx: torch.Tensor
    token_scores: torch.Tensor
    tile_starts: torch.Tensor
    valid_lengths: torch.Tensor
    chunk_size: int
    selector_tile_size: int


@dataclass(frozen=True)
class HISASelectionCapture:
    """Ephemeral train-only route-logit loss surface for the evidence sidecar.

    The metadata is already detached by the selector. ``route_logits`` remains
    differentiable, so an auxiliary can supervise the pre-top-k distribution
    without changing either the discrete routing choice or normal outputs.
    """

    route_logits: torch.Tensor
    metadata: HISAMetadata


_DEFAULT_LOCAL_WINDOW: Final[int] = 64
_DEFAULT_SELECTOR_TILE: Final[int] = 16


def _next_pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _as_valid_lengths(
    valid_lengths: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_lengths is None:
        return torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    valid_lengths = valid_lengths.to(device=device, dtype=torch.long).reshape(-1)
    if valid_lengths.numel() != batch_size:
        raise ValueError(
            f"valid_lengths must have {batch_size} entries, got {valid_lengths.numel()}"
        )
    if bool((valid_lengths < 0).any()) or bool((valid_lengths > seq_len).any()):
        raise ValueError(f"valid_lengths must be in [0, {seq_len}]")
    return valid_lengths


def _to_heads(tensor: torch.Tensor, *, batch_size: int, seq_len: int, heads: int, head_dim: int) -> torch.Tensor:
    return tensor.reshape(batch_size, seq_len, heads, head_dim).transpose(1, 2).contiguous()


def _completed_chunk_representatives(
    key: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    valid_lengths: torch.Tensor,
    representative_mode: str = "max_l2",
    top2_blend_alpha: float = 0.25,
) -> torch.Tensor:
    """Return a salience representative for each physical chunk.

    A representative may inspect its full chunk because V16 consumes it only
    after that entire chunk is complete relative to a query-selector tile.
    """

    batch_size, heads, seq_len, head_dim = key.shape
    padded_len = num_chunks * chunk_size
    if padded_len != seq_len:
        key = F.pad(key, (0, 0, 0, padded_len - seq_len))
    chunks = key.reshape(batch_size, heads, num_chunks, chunk_size, head_dim)
    positions = torch.arange(padded_len, device=key.device)
    valid = positions.reshape(1, 1, num_chunks, chunk_size) < valid_lengths.reshape(batch_size, 1, 1, 1)
    energy = chunks.float().square().sum(dim=-1).masked_fill(~valid, float("-inf"))
    best = energy.argmax(dim=-1, keepdim=True)
    reps = torch.gather(
        chunks,
        dim=3,
        index=best.unsqueeze(-1).expand(-1, -1, -1, 1, head_dim),
    ).squeeze(3)
    if representative_mode == "top2_blend" and top2_blend_alpha != 0.0 and chunk_size >= 2:
        if not 0.0 <= top2_blend_alpha <= 1.0:
            raise ValueError("top2_blend_alpha must be in [0, 1]")
        top2 = energy.topk(2, dim=-1).indices
        salient = torch.gather(
            chunks,
            dim=3,
            index=top2.unsqueeze(-1).expand(-1, -1, -1, 2, head_dim),
        )
        reps = (1.0 - top2_blend_alpha) * salient[..., 0, :] + top2_blend_alpha * salient[..., 1, :]
    elif representative_mode != "max_l2" and representative_mode != "top2_blend":
        raise ValueError("representative_mode must be max_l2 or top2_blend")
    chunk_start = torch.arange(num_chunks, device=key.device) * chunk_size
    chunk_valid = chunk_start.reshape(1, num_chunks) < valid_lengths.reshape(batch_size, 1)
    return reps.masked_fill(~chunk_valid[:, None, :, None], 0.0)


def _build_causal_tile_metadata_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    route_logits: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    top_k_chunks: int,
    top_m_tokens: int,
    selector_tile_size: int,
    valid_lengths: torch.Tensor,
) -> HISAMetadata:
    """Build detached V16 metadata without future query/key dependence."""

    batch_size, heads, seq_len, head_dim = query.shape
    device = query.device
    n_tiles = math.ceil(seq_len / selector_tile_size)
    k_slots = max(1, int(top_k_chunks))
    m_slots = min(max(1, int(top_m_tokens)), chunk_size)
    top_chunks = torch.full((batch_size, heads, n_tiles, k_slots), -1, dtype=torch.long, device=device)
    token_idx = torch.full(
        (batch_size, heads, n_tiles, k_slots, m_slots),
        -1,
        dtype=torch.long,
        device=device,
    )
    token_scores = torch.full(
        (batch_size, heads, n_tiles, k_slots, m_slots),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    tile_starts = torch.arange(n_tiles, device=device, dtype=torch.long) * selector_tile_size
    chunk_starts = torch.arange(num_chunks, device=device, dtype=torch.long) * chunk_size
    chunk_ends = chunk_starts + chunk_size

    padded_len = num_chunks * chunk_size
    key_pad = F.pad(key, (0, 0, 0, padded_len - seq_len)) if padded_len > seq_len else key
    key_chunks = key_pad.reshape(batch_size, heads, num_chunks, chunk_size, head_dim)
    b_index = torch.arange(batch_size, device=device).reshape(batch_size, 1, 1)
    h_index = torch.arange(heads, device=device).reshape(1, heads, 1)
    within_chunk = torch.arange(chunk_size, device=device).reshape(1, 1, 1, chunk_size)

    # The discrete route/top-M controls are intentionally detached.  The route
    # scores themselves are gathered later from the differentiable logits.
    with torch.no_grad():
        for tile, tile_start_tensor in enumerate(tile_starts):
            tile_start = int(tile_start_tensor.item())
            if tile_start >= seq_len:
                break
            tile_has_query = tile_start < valid_lengths
            completed = chunk_ends < (tile_start + 1)  # exclusive end <= tile_start
            chunk_has_token = chunk_starts.reshape(1, num_chunks) < valid_lengths.reshape(batch_size, 1)
            eligible = completed.reshape(1, 1, num_chunks) & chunk_has_token[:, None, :] & tile_has_query[:, None, None]
            routing_at_start = route_logits[:, :, tile_start, :].detach().masked_fill(~eligible, float("-inf"))
            k_eff = min(k_slots, num_chunks)
            values, indices = routing_at_start.topk(k_eff, dim=-1)
            valid_selected = torch.isfinite(values)
            indices = torch.where(valid_selected, indices, torch.full_like(indices, -1))
            top_chunks[:, :, tile, :k_eff] = indices

            safe_indices = indices.clamp_min(0)
            selected_keys = key_chunks[b_index, h_index, safe_indices]
            selected_abs = safe_indices[..., None] * chunk_size + within_chunk
            selected_valid = (
                valid_selected[..., None]
                & (selected_abs < valid_lengths.reshape(batch_size, 1, 1, 1))
                & (selected_abs < tile_start)
            )
            q0 = query[:, :, tile_start, :].detach().unsqueeze(2).unsqueeze(3)
            scores = (q0 * selected_keys.detach()).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~selected_valid, float("-inf"))
            top_values, local_indices = scores.topk(m_slots, dim=-1)
            absolute = safe_indices[..., None] * chunk_size + local_indices
            finite = torch.isfinite(top_values) & valid_selected[..., None]
            token_idx[:, :, tile, :k_eff] = torch.where(finite, absolute, torch.full_like(absolute, -1))
            token_scores[:, :, tile, :k_eff] = torch.where(
                finite,
                top_values.float(),
                torch.full_like(top_values.float(), float("-inf")),
            )

    return HISAMetadata(
        top_chunk_idx=top_chunks,
        token_idx=token_idx,
        token_scores=token_scores,
        tile_starts=tile_starts,
        valid_lengths=valid_lengths.detach(),
        chunk_size=int(chunk_size),
        selector_tile_size=int(selector_tile_size),
    )


def _build_causal_tile_metadata_blocked(
    query: torch.Tensor,
    key: torch.Tensor,
    route_logits: torch.Tensor,
    *,
    num_chunks: int,
    chunk_size: int,
    top_k_chunks: int,
    top_m_tokens: int,
    selector_tile_size: int,
    valid_lengths: torch.Tensor,
    tile_block_size: int,
) -> HISAMetadata:
    """Build exact V16 tile metadata in independent vectorized tile blocks."""
    if tile_block_size < 1:
        raise ValueError("tile_block_size must be positive")
    batch_size, heads, seq_len, head_dim = query.shape
    device = query.device
    n_tiles = math.ceil(seq_len / selector_tile_size)
    k_slots = max(1, int(top_k_chunks))
    m_slots = min(max(1, int(top_m_tokens)), chunk_size)
    top_chunks = torch.full((batch_size, heads, n_tiles, k_slots), -1, dtype=torch.long, device=device)
    token_idx = torch.full(
        (batch_size, heads, n_tiles, k_slots, m_slots), -1, dtype=torch.long, device=device
    )
    token_scores = torch.full(
        (batch_size, heads, n_tiles, k_slots, m_slots), float("-inf"), dtype=torch.float32, device=device
    )
    tile_starts = torch.arange(n_tiles, device=device, dtype=torch.long) * selector_tile_size
    chunk_starts = torch.arange(num_chunks, device=device, dtype=torch.long) * chunk_size
    chunk_ends = chunk_starts + chunk_size
    padded_len = num_chunks * chunk_size
    key_pad = F.pad(key, (0, 0, 0, padded_len - seq_len)) if padded_len > seq_len else key
    key_chunks = key_pad.reshape(batch_size, heads, num_chunks, chunk_size, head_dim)
    b_index = torch.arange(batch_size, device=device).reshape(batch_size, 1, 1, 1)
    h_index = torch.arange(heads, device=device).reshape(1, heads, 1, 1)
    within_chunk = torch.arange(chunk_size, device=device).reshape(1, 1, 1, 1, chunk_size)
    chunk_has_token = chunk_starts.reshape(1, num_chunks) < valid_lengths.reshape(batch_size, 1)
    k_eff = min(k_slots, num_chunks)

    # Discrete selections remain detached. Blocking changes only eager launch
    # scheduling; each semantic tile still uses exactly its first query row.
    with torch.no_grad():
        for tile_begin in range(0, n_tiles, tile_block_size):
            starts = tile_starts[tile_begin : tile_begin + tile_block_size]
            block_tiles = starts.numel()
            tile_has_query = starts.reshape(1, block_tiles) < valid_lengths.reshape(batch_size, 1)
            completed = chunk_ends.reshape(1, num_chunks) < (starts.reshape(block_tiles, 1) + 1)
            eligible = (
                completed.reshape(1, 1, block_tiles, num_chunks)
                & chunk_has_token.reshape(batch_size, 1, 1, num_chunks)
                & tile_has_query.reshape(batch_size, 1, block_tiles, 1)
            )
            routing_at_start = route_logits[:, :, starts, :].detach().masked_fill(~eligible, float("-inf"))
            values, indices = routing_at_start.topk(k_eff, dim=-1)
            valid_selected = torch.isfinite(values)
            indices = torch.where(valid_selected, indices, torch.full_like(indices, -1))
            top_chunks[:, :, tile_begin : tile_begin + block_tiles, :k_eff] = indices

            safe_indices = indices.clamp_min(0)
            selected_keys = key_chunks[b_index, h_index, safe_indices]
            selected_abs = safe_indices[..., None] * chunk_size + within_chunk
            selected_valid = (
                valid_selected[..., None]
                & (selected_abs < valid_lengths.reshape(batch_size, 1, 1, 1, 1))
                & (selected_abs < starts.reshape(1, 1, block_tiles, 1, 1))
            )
            q0 = query[:, :, starts, :].detach().unsqueeze(3).unsqueeze(4)
            scores = (q0 * selected_keys.detach()).sum(dim=-1) / math.sqrt(head_dim)
            scores = scores.masked_fill(~selected_valid, float("-inf"))
            top_values, local_indices = scores.topk(m_slots, dim=-1)
            absolute = safe_indices[..., None] * chunk_size + local_indices
            finite = torch.isfinite(top_values) & valid_selected[..., None]
            token_idx[:, :, tile_begin : tile_begin + block_tiles, :k_eff] = torch.where(
                finite, absolute, torch.full_like(absolute, -1)
            )
            token_scores[:, :, tile_begin : tile_begin + block_tiles, :k_eff] = torch.where(
                finite, top_values.float(), torch.full_like(top_values.float(), float("-inf"))
            )
    return HISAMetadata(
        top_chunk_idx=top_chunks,
        token_idx=token_idx,
        token_scores=token_scores,
        tile_starts=tile_starts,
        valid_lengths=valid_lengths.detach(),
        chunk_size=int(chunk_size),
        selector_tile_size=int(selector_tile_size),
    )


def _eager_v16_attention_scalar(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    route_logits: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
    route_prior_scale: float,
) -> torch.Tensor:
    """Canonical scalar V16 attention reference for CPU/debugging."""

    batch_size, heads, seq_len, head_dim = query.shape
    rows: list[torch.Tensor] = []
    scale = 1.0 / math.sqrt(head_dim)
    for batch in range(batch_size):
        head_rows: list[torch.Tensor] = []
        valid_len = int(metadata.valid_lengths[batch].item())
        for head in range(heads):
            query_rows: list[torch.Tensor] = []
            for q_pos in range(seq_len):
                if q_pos >= valid_len:
                    query_rows.append(torch.zeros_like(value[batch, head, 0]))
                    continue
                local_start = max(0, q_pos - local_window)
                local_ids = torch.arange(local_start, q_pos, device=query.device, dtype=torch.long)
                tile = q_pos // metadata.selector_tile_size
                global_ids = metadata.token_idx[batch, head, tile].reshape(-1)
                global_chunks = metadata.top_chunk_idx[batch, head, tile].reshape(-1, 1).expand(
                    -1, metadata.token_idx.shape[-1]
                ).reshape(-1)
                # Local ownership wins on overlap.  This makes the two lanes
                # disjoint even when a selected completed chunk is nearby.
                global_keep = (
                    (global_ids >= 0)
                    & (global_ids < local_start)
                    & (global_ids < q_pos)
                    & (global_ids < valid_len)
                    & (global_chunks >= 0)
                )
                kept_global_ids = global_ids[global_keep]
                kept_global_chunks = global_chunks[global_keep]
                candidate_ids = torch.cat((local_ids, kept_global_ids), dim=0)
                if candidate_ids.numel() == 0:
                    query_rows.append(torch.zeros_like(value[batch, head, 0]))
                    continue
                candidate_keys = key[batch, head, candidate_ids]
                candidate_values = value[batch, head, candidate_ids]
                scores = torch.matmul(candidate_keys, query[batch, head, q_pos]) * scale
                if kept_global_ids.numel() > 0:
                    route_bias = route_logits[batch, head, q_pos, kept_global_chunks] * route_prior_scale
                    scores = torch.cat((scores[: local_ids.numel()], scores[local_ids.numel() :] + route_bias), dim=0)
                probabilities = torch.softmax(scores.float(), dim=0).to(candidate_values.dtype)
                query_rows.append(torch.matmul(probabilities, candidate_values))
            head_rows.append(torch.stack(query_rows, dim=0))
        rows.append(torch.stack(head_rows, dim=0))
    return torch.stack(rows, dim=0)


def _eager_v16_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    route_logits: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
    route_prior_scale: float,
) -> torch.Tensor:
    """Tile-vectorized V16 oracle used for CPU and diagnostic replay."""

    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    local_offsets = torch.arange(local_window, device=query.device, dtype=torch.long)
    for tile in range(metadata.top_chunk_idx.shape[2]):
        start = tile * metadata.selector_tile_size
        end = min(start + metadata.selector_tile_size, seq_len)
        if start >= end:
            break
        q_positions = torch.arange(start, end, device=query.device, dtype=torch.long)
        q_len = q_positions.numel()
        q = query[:, :, start:end, :]
        query_valid = q_positions.reshape(1, q_len) < metadata.valid_lengths.reshape(batch_size, 1)

        local_ids = q_positions.reshape(q_len, 1) - local_window + local_offsets.reshape(1, local_window)
        local_valid = (
            (local_ids >= 0)
            & (local_ids < seq_len)
            & (local_ids < metadata.valid_lengths.reshape(batch_size, 1, 1))
            & query_valid.reshape(batch_size, q_len, 1)
        )
        safe_local_ids = local_ids.clamp(0, max(seq_len - 1, 0))
        local_k = key[:, :, safe_local_ids, :]
        local_v = value[:, :, safe_local_ids, :]
        local_scores = (q.unsqueeze(-2) * local_k).sum(dim=-1) * scale
        local_scores = local_scores.masked_fill(~local_valid[:, None], float("-inf"))

        chunks = metadata.top_chunk_idx[:, :, tile, :]
        indices = metadata.token_idx[:, :, tile, :, :]
        m_slots = indices.shape[-1]
        global_ids = indices.reshape(batch_size, heads, -1)
        global_chunks = chunks.unsqueeze(-1).expand(-1, -1, -1, m_slots).reshape(batch_size, heads, -1)
        safe_global_ids = global_ids.clamp(0, max(seq_len - 1, 0))
        global_k = torch.gather(key, 2, safe_global_ids.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        global_v = torch.gather(value, 2, safe_global_ids.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        safe_global_chunks = global_chunks.clamp_min(0)
        route_ids = safe_global_chunks.unsqueeze(2).expand(-1, -1, q_len, -1)
        route_scores = torch.gather(route_logits[:, :, start:end, :], -1, route_ids)
        global_scores = torch.matmul(q, global_k.transpose(-2, -1)) * scale + route_prior_scale * route_scores
        global_valid = (
            (global_ids >= 0).unsqueeze(2)
            & (global_chunks >= 0).unsqueeze(2)
            & (global_ids.unsqueeze(2) < q_positions.reshape(1, 1, q_len, 1) - local_window)
            & (global_ids.unsqueeze(2) < metadata.valid_lengths.reshape(batch_size, 1, 1, 1))
            & query_valid[:, None, :, None]
        )
        global_scores = global_scores.masked_fill(~global_valid, float("-inf"))

        local_max = local_scores.max(dim=-1).values
        global_max = global_scores.max(dim=-1).values
        max_score = torch.maximum(local_max, global_max)
        safe_max = torch.where(torch.isfinite(max_score), max_score, torch.zeros_like(max_score))
        local_prob = torch.where(torch.isfinite(local_scores), torch.exp(local_scores.float() - safe_max.unsqueeze(-1).float()), torch.zeros_like(local_scores.float()))
        global_prob = torch.where(torch.isfinite(global_scores), torch.exp(global_scores.float() - safe_max.unsqueeze(-1).float()), torch.zeros_like(global_scores.float()))
        denominator = local_prob.sum(dim=-1) + global_prob.sum(dim=-1)
        local_acc = torch.einsum("bhqw,bhqwd->bhqd", local_prob.to(local_v.dtype), local_v)
        global_acc = torch.matmul(global_prob.to(global_v.dtype), global_v)
        tile_out = (local_acc + global_acc) / denominator.clamp_min(1.0).unsqueeze(-1).to(local_acc.dtype)
        output[:, :, start:end, :] = torch.where(query_valid[:, None, :, None], tile_out, torch.zeros_like(tile_out))
    return output


_COMPILED_REPLAY = None


def _replay_v16_attention(*args, **kwargs) -> torch.Tensor:
    """Optionally compile the vectorized replay outside trainer-level compile."""

    global _COMPILED_REPLAY
    if os.getenv("DWARF_HISA_V16_COMPILE_REPLAY", "0") == "1" and hasattr(torch, "compile"):
        if _COMPILED_REPLAY is None:
            _COMPILED_REPLAY = torch.compile(_eager_v16_attention, dynamic=False)
        return _COMPILED_REPLAY(*args, **kwargs)
    return _eager_v16_attention(*args, **kwargs)


if _TRITON_AVAILABLE:

    @triton.jit
    def _v16_hisa_forward_kernel(
        Q, K, V, ROUTE, TOP_CHUNK, TOKEN_IDX, VALID_LENS, OUT, LSE,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_rb, stride_rh, stride_rn, stride_rc,
        stride_cb, stride_ch, stride_ct, stride_ck,
        stride_ib, stride_ih, stride_it, stride_ik, stride_im,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_lseb, stride_lseh, stride_lsen,
        N, H: tl.constexpr, HD: tl.constexpr, K_VAL: tl.constexpr,
        M_VAL: tl.constexpr, M_PAD: tl.constexpr, SELECTOR_TILE: tl.constexpr,
        LOCAL_W: tl.constexpr, ROUTE_SCALE: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_LOCAL: tl.constexpr,
    ):
        bh = tl.program_id(0)
        tile = tl.program_id(1)
        q_block = tl.program_id(2)
        batch = bh // H
        head = bh % H
        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        qs = tile * SELECTOR_TILE + q_offsets
        qmask = (qs < N) & (q_offsets < SELECTOR_TILE)
        valid_len = tl.load(VALID_LENS + batch).to(tl.int32)
        qmask = qmask & (qs < valid_len)
        dims = tl.arange(0, HD)
        q_base = Q + batch * stride_qb + head * stride_qh
        k_base = K + batch * stride_kb + head * stride_kh
        v_base = V + batch * stride_vb + head * stride_vh
        q = tl.load(q_base + qs[:, None] * stride_qn + dims[None, :] * stride_qd, mask=qmask[:, None], other=0.0)
        running_max = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, HD], tl.float32)

        # The local candidate universe covers every per-row [q-W,q) window in
        # this launch block.  Masks select the correct strict causal slice.
        local_start = tile * SELECTOR_TILE + q_block * BLOCK_Q - LOCAL_W
        local_offsets = tl.arange(0, BLOCK_LOCAL)
        local_ids = local_start + local_offsets
        local_mask = (local_ids >= 0) & (local_ids < N) & (local_ids < valid_len)
        local_k = tl.load(
            k_base + local_ids[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=local_mask[:, None], other=0.0,
        )
        local_scores = tl.dot(q, tl.trans(local_k), input_precision="ieee") * (1.0 / tl.sqrt(HD * 1.0))
        local_selected = (
            qmask[:, None]
            & local_mask[None, :]
            & (local_ids[None, :] < qs[:, None])
            & (local_ids[None, :] >= qs[:, None] - LOCAL_W)
        )
        local_scores = tl.where(local_selected, local_scores, float("-inf"))
        next_max = tl.max(local_scores, axis=1)
        has_local = next_max > float("-inf")
        safe_max = tl.where(has_local, next_max, tl.zeros([BLOCK_Q], tl.float32))
        local_prob = tl.where(local_selected, tl.exp(local_scores - safe_max[:, None]), 0.0)
        running_max = tl.where(has_local, next_max, running_max)
        running_sum = tl.sum(local_prob, axis=1)
        local_v = tl.load(
            v_base + local_ids[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=local_mask[:, None], other=0.0,
        )
        acc += tl.dot(local_prob.to(q.dtype), local_v, input_precision="ieee")

        top_base = TOP_CHUNK + batch * stride_cb + head * stride_ch + tile * stride_ct
        index_base = TOKEN_IDX + batch * stride_ib + head * stride_ih + tile * stride_it
        route_base = ROUTE + batch * stride_rb + head * stride_rh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk_id = tl.load(top_base + slot * stride_ck).to(tl.int32)
            chunk_valid = chunk_id >= 0
            safe_chunk_id = tl.maximum(chunk_id, 0)
            ids = tl.load(
                index_base + slot * stride_ik + token_offsets * stride_im,
                mask=token_mask, other=-1,
            ).to(tl.int32)
            id_mask = token_mask & (ids >= 0) & (ids < N) & (ids < valid_len) & chunk_valid
            global_k = tl.load(
                k_base + ids[:, None] * stride_kn + dims[None, :] * stride_kd,
                mask=id_mask[:, None], other=0.0,
            )
            route = tl.load(
                route_base + qs * stride_rn + safe_chunk_id * stride_rc,
                mask=qmask & chunk_valid, other=0.0,
            ).to(tl.float32)
            global_scores = tl.dot(q, tl.trans(global_k), input_precision="ieee") * (1.0 / tl.sqrt(HD * 1.0)) + ROUTE_SCALE * route[:, None]
            # A local row owns its recent keys, preventing duplicate softmax
            # candidates when a selected completed chunk overlaps the window.
            global_selected = qmask[:, None] & id_mask[None, :] & (ids[None, :] < qs[:, None] - LOCAL_W)
            global_scores = tl.where(global_selected, global_scores, float("-inf"))
            block_max = tl.max(global_scores, axis=1)
            has_global = block_max > float("-inf")
            merged_max = tl.maximum(running_max, block_max)
            safe_merged = tl.where(
                (running_max > float("-inf")) | has_global,
                merged_max,
                tl.zeros([BLOCK_Q], tl.float32),
            )
            prior_scale = tl.where(running_max > float("-inf"), tl.exp(running_max - safe_merged), 0.0)
            global_prob = tl.where(global_selected, tl.exp(global_scores - safe_merged[:, None]), 0.0)
            running_sum = running_sum * prior_scale + tl.sum(global_prob, axis=1)
            acc = acc * prior_scale[:, None]
            global_v = tl.load(
                v_base + ids[:, None] * stride_vn + dims[None, :] * stride_vd,
                mask=id_mask[:, None], other=0.0,
            )
            acc += tl.dot(global_prob.to(q.dtype), global_v, input_precision="ieee")
            running_max = tl.where(
                (running_max > float("-inf")) | has_global,
                merged_max,
                running_max,
            )

        denom = tl.where(running_sum > 0.0, running_sum, 1.0)
        out = acc / denom[:, None]
        out_base = OUT + batch * stride_ob + head * stride_oh
        tl.store(
            out_base + qs[:, None] * stride_on + dims[None, :] * stride_od,
            out,
            mask=qmask[:, None],
        )
        lse = tl.where(
            running_sum > 0.0,
            running_max + tl.log(running_sum),
            float("-inf"),
        )
        tl.store(
            LSE + batch * stride_lseb + head * stride_lseh + qs * stride_lsen,
            lse,
            mask=qmask,
        )


    @triton.jit
    def _v16_hisa_backward_kernel(
        Q, K, V, O, DO, LSE, ROUTE, TOP_CHUNK, TOKEN_IDX, VALID_LENS,
        DQ, DK, DV, DROUTE,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_lseb, stride_lseh, stride_lsen,
        stride_rb, stride_rh, stride_rn, stride_rc,
        stride_cb, stride_ch, stride_ct, stride_ck,
        stride_ib, stride_ih, stride_it, stride_ik, stride_im,
        stride_dqb, stride_dqh, stride_dqn, stride_dqd,
        stride_dkb, stride_dkh, stride_dkn, stride_dkd,
        stride_dvb, stride_dvh, stride_dvn, stride_dvd,
        stride_drb, stride_drh, stride_drn, stride_drc,
        N, H: tl.constexpr, HD: tl.constexpr, K_VAL: tl.constexpr,
        M_VAL: tl.constexpr, M_PAD: tl.constexpr, SELECTOR_TILE: tl.constexpr,
        LOCAL_W: tl.constexpr, ROUTE_SCALE: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_LOCAL: tl.constexpr, MASK_ATOMICS: tl.constexpr,
    ):
        # The launch topology mirrors forward: a program uniquely owns its
        # query rows, but many programs can contribute to a source K/V row.
        bh = tl.program_id(0)
        tile = tl.program_id(1)
        q_block = tl.program_id(2)
        batch = bh // H
        head = bh % H
        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        qs = tile * SELECTOR_TILE + q_offsets
        valid_len = tl.load(VALID_LENS + batch).to(tl.int32)
        qmask = (qs < N) & (q_offsets < SELECTOR_TILE) & (qs < valid_len)
        dims = tl.arange(0, HD)
        scale = 1.0 / tl.sqrt(HD * 1.0)

        q_base = Q + batch * stride_qb + head * stride_qh
        k_base = K + batch * stride_kb + head * stride_kh
        v_base = V + batch * stride_vb + head * stride_vh
        o_base = O + batch * stride_ob + head * stride_oh
        do_base = DO + batch * stride_dob + head * stride_doh
        dq_base = DQ + batch * stride_dqb + head * stride_dqh
        dk_base = DK + batch * stride_dkb + head * stride_dkh
        dv_base = DV + batch * stride_dvb + head * stride_dvh

        q = tl.load(
            q_base + qs[:, None] * stride_qn + dims[None, :] * stride_qd,
            mask=qmask[:, None], other=0.0,
        )
        o = tl.load(
            o_base + qs[:, None] * stride_on + dims[None, :] * stride_od,
            mask=qmask[:, None], other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_base + qs[:, None] * stride_don + dims[None, :] * stride_dod,
            mask=qmask[:, None], other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + batch * stride_lseb + head * stride_lseh + qs * stride_lsen,
            mask=qmask, other=float("-inf"),
        )
        lse_valid = lse > float("-inf")
        safe_lse = tl.where(lse_valid, lse, 0.0)
        delta = tl.sum(do * o, axis=1)
        dq = tl.zeros([BLOCK_Q, HD], tl.float32)

        # Local lane: exactly the same strict recent-window mask as forward.
        local_start = tile * SELECTOR_TILE + q_block * BLOCK_Q - LOCAL_W
        local_offsets = tl.arange(0, BLOCK_LOCAL)
        local_ids = local_start + local_offsets
        local_mask = (local_ids >= 0) & (local_ids < N) & (local_ids < valid_len)
        safe_local_ids = tl.maximum(tl.minimum(local_ids, N - 1), 0)
        local_k = tl.load(
            k_base + safe_local_ids[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=local_mask[:, None], other=0.0,
        )
        local_selected = (
            qmask[:, None]
            & local_mask[None, :]
            & (local_ids[None, :] < qs[:, None])
            & (local_ids[None, :] >= qs[:, None] - LOCAL_W)
        )
        local_scores = tl.dot(q, tl.trans(local_k), input_precision="ieee") * scale
        local_scores = tl.where(local_selected, local_scores, float("-inf"))
        local_p = tl.where(
            local_selected & lse_valid[:, None],
            tl.exp(local_scores - safe_lse[:, None]),
            0.0,
        )
        local_v = tl.load(
            v_base + safe_local_ids[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=local_mask[:, None], other=0.0,
        ).to(tl.float32)
        local_dv_dot = tl.dot(do, tl.trans(local_v), input_precision="ieee")
        local_ds = local_p * (local_dv_dot - delta[:, None])
        dq += tl.dot(local_ds, local_k.to(tl.float32), input_precision="ieee") * scale
        local_dk = tl.dot(tl.trans(local_ds), q.to(tl.float32), input_precision="ieee") * scale
        local_dv = tl.dot(tl.trans(local_p), do, input_precision="ieee")
        local_write_mask = local_mask
        if MASK_ATOMICS:
            local_write_mask = local_mask & (tl.sum(local_selected.to(tl.int32), axis=0) > 0)
        tl.atomic_add(
            dk_base + safe_local_ids[:, None] * stride_dkn + dims[None, :] * stride_dkd,
            local_dk,
            mask=local_write_mask[:, None], sem="relaxed",
        )
        tl.atomic_add(
            dv_base + safe_local_ids[:, None] * stride_dvn + dims[None, :] * stride_dvd,
            local_dv,
            mask=local_write_mask[:, None], sem="relaxed",
        )

        top_base = TOP_CHUNK + batch * stride_cb + head * stride_ch + tile * stride_ct
        index_base = TOKEN_IDX + batch * stride_ib + head * stride_ih + tile * stride_it
        route_base = ROUTE + batch * stride_rb + head * stride_rh
        droute_base = DROUTE + batch * stride_drb + head * stride_drh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk_id = tl.load(top_base + slot * stride_ck).to(tl.int32)
            chunk_valid = chunk_id >= 0
            safe_chunk_id = tl.maximum(chunk_id, 0)
            ids = tl.load(
                index_base + slot * stride_ik + token_offsets * stride_im,
                mask=token_mask, other=-1,
            ).to(tl.int32)
            id_mask = token_mask & (ids >= 0) & (ids < N) & (ids < valid_len) & chunk_valid
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            global_k = tl.load(
                k_base + safe_ids[:, None] * stride_kn + dims[None, :] * stride_kd,
                mask=id_mask[:, None], other=0.0,
            )
            route = tl.load(
                route_base + qs * stride_rn + safe_chunk_id * stride_rc,
                mask=qmask & chunk_valid, other=0.0,
            ).to(tl.float32)
            global_selected = (
                qmask[:, None]
                & id_mask[None, :]
                & (ids[None, :] < qs[:, None] - LOCAL_W)
            )
            global_scores = (
                tl.dot(q, tl.trans(global_k), input_precision="ieee") * scale
                + ROUTE_SCALE * route[:, None]
            )
            global_scores = tl.where(global_selected, global_scores, float("-inf"))
            global_p = tl.where(
                global_selected & lse_valid[:, None],
                tl.exp(global_scores - safe_lse[:, None]),
                0.0,
            )
            global_v = tl.load(
                v_base + safe_ids[:, None] * stride_vn + dims[None, :] * stride_vd,
                mask=id_mask[:, None], other=0.0,
            ).to(tl.float32)
            global_dv_dot = tl.dot(do, tl.trans(global_v), input_precision="ieee")
            global_ds = global_p * (global_dv_dot - delta[:, None])
            dq += tl.dot(global_ds, global_k.to(tl.float32), input_precision="ieee") * scale
            global_dk = tl.dot(tl.trans(global_ds), q.to(tl.float32), input_precision="ieee") * scale
            global_dv = tl.dot(tl.trans(global_p), do, input_precision="ieee")
            global_write_mask = id_mask
            if MASK_ATOMICS:
                global_write_mask = id_mask & (tl.sum(global_selected.to(tl.int32), axis=0) > 0)
            tl.atomic_add(
                dk_base + safe_ids[:, None] * stride_dkn + dims[None, :] * stride_dkd,
                global_dk,
                mask=global_write_mask[:, None], sem="relaxed",
            )
            tl.atomic_add(
                dv_base + safe_ids[:, None] * stride_dvn + dims[None, :] * stride_dvd,
                global_dv,
                mask=global_write_mask[:, None], sem="relaxed",
            )
            # Tile metadata comes from topk, so valid chunk IDs are unique for a
            # tile.  This program owns each query row, making a direct store safe.
            tl.store(
                droute_base + qs * stride_drn + safe_chunk_id * stride_drc,
                ROUTE_SCALE * tl.sum(global_ds, axis=1),
                mask=qmask & chunk_valid,
            )

        tl.store(
            dq_base + qs[:, None] * stride_dqn + dims[None, :] * stride_dqd,
            dq,
            mask=qmask[:, None],
        )


class _V16HISATritonFn(torch.autograd.Function):
    """Fused strict-causal forward and tile-aligned Triton backward."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        route_logits: torch.Tensor,
        top_chunk_idx: torch.Tensor,
        token_idx: torch.Tensor,
        valid_lengths: torch.Tensor,
        selector_tile_size: int,
        local_window: int,
        route_prior_scale: float,
        requested_block_q: int,
        mask_atomics: bool,
    ) -> torch.Tensor:
        if not query.is_cuda or not _TRITON_AVAILABLE:
            metadata = HISAMetadata(
                top_chunk_idx=top_chunk_idx,
                token_idx=token_idx,
                token_scores=torch.empty(0, device=query.device),
                tile_starts=torch.arange(top_chunk_idx.shape[2], device=query.device) * selector_tile_size,
                valid_lengths=valid_lengths,
                chunk_size=0,
                selector_tile_size=selector_tile_size,
            )
            return _eager_v16_attention(
                query, key, value, route_logits, metadata,
                local_window=local_window, route_prior_scale=route_prior_scale,
            )
        batch_size, heads, seq_len, head_dim = query.shape
        k_val = top_chunk_idx.shape[-1]
        m_val = token_idx.shape[-1]
        m_pad = max(16, _next_pow2(m_val))
        block_q = int(requested_block_q) if requested_block_q > 0 else 16
        if block_q < 16 or (block_q & (block_q - 1)):
            raise ValueError("V16 BLOCK_Q must be a power of two >= 16")
        block_local = _next_pow2(local_window + block_q)
        if block_local > 256:
            raise ValueError("V16 local_window + BLOCK_Q must not exceed 256 for the fused forward")
        # Invalid valid-length rows are masked by the kernel and must retain the
        # eager oracle's zero output rather than uninitialized storage.
        out = torch.zeros_like(query)
        lse = torch.full(
            (batch_size, heads, seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        n_q_blocks = triton.cdiv(selector_tile_size, block_q)
        grid = (batch_size * heads, top_chunk_idx.shape[2], n_q_blocks)
        _v16_hisa_forward_kernel[grid](
            query, key, value, route_logits, top_chunk_idx, token_idx, valid_lengths, out, lse,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            route_logits.stride(0), route_logits.stride(1), route_logits.stride(2), route_logits.stride(3),
            top_chunk_idx.stride(0), top_chunk_idx.stride(1), top_chunk_idx.stride(2), top_chunk_idx.stride(3),
            token_idx.stride(0), token_idx.stride(1), token_idx.stride(2), token_idx.stride(3), token_idx.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N=seq_len, H=heads, HD=head_dim, K_VAL=k_val, M_VAL=m_val, M_PAD=m_pad,
            SELECTOR_TILE=selector_tile_size, LOCAL_W=local_window,
            ROUTE_SCALE=float(route_prior_scale), BLOCK_Q=block_q, BLOCK_LOCAL=block_local,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(
            query, key, value, route_logits, top_chunk_idx, token_idx, valid_lengths, out, lse,
        )
        ctx.selector_tile_size = selector_tile_size
        ctx.local_window = local_window
        ctx.route_prior_scale = route_prior_scale
        ctx.block_q = block_q
        ctx.mask_atomics = bool(mask_atomics)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, route_logits, top_chunk_idx, token_idx, valid_lengths, out, lse = ctx.saved_tensors
        batch_size, heads, seq_len, head_dim = query.shape
        k_val = top_chunk_idx.shape[-1]
        m_val = token_idx.shape[-1]
        m_pad = max(16, _next_pow2(m_val))
        block_q = ctx.block_q
        block_local = _next_pow2(ctx.local_window + block_q)
        grad_output = grad_output.contiguous()

        # Source gradients have irregular many-query fan-in, so the kernel uses
        # FP32 atomics.  dQ and dRoute are uniquely owned by a query-tile launch.
        dquery = torch.zeros_like(query, dtype=torch.float32)
        dkey = torch.zeros_like(key, dtype=torch.float32)
        dvalue = torch.zeros_like(value, dtype=torch.float32)
        droute = torch.zeros_like(route_logits, dtype=torch.float32)
        grid = (
            batch_size * heads,
            top_chunk_idx.shape[2],
            triton.cdiv(ctx.selector_tile_size, block_q),
        )
        _v16_hisa_backward_kernel[grid](
            query, key, value, out, grad_output, lse, route_logits,
            top_chunk_idx, token_idx, valid_lengths,
            dquery, dkey, dvalue, droute,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            route_logits.stride(0), route_logits.stride(1), route_logits.stride(2), route_logits.stride(3),
            top_chunk_idx.stride(0), top_chunk_idx.stride(1), top_chunk_idx.stride(2), top_chunk_idx.stride(3),
            token_idx.stride(0), token_idx.stride(1), token_idx.stride(2), token_idx.stride(3), token_idx.stride(4),
            dquery.stride(0), dquery.stride(1), dquery.stride(2), dquery.stride(3),
            dkey.stride(0), dkey.stride(1), dkey.stride(2), dkey.stride(3),
            dvalue.stride(0), dvalue.stride(1), dvalue.stride(2), dvalue.stride(3),
            droute.stride(0), droute.stride(1), droute.stride(2), droute.stride(3),
            N=seq_len, H=heads, HD=head_dim, K_VAL=k_val, M_VAL=m_val, M_PAD=m_pad,
            SELECTOR_TILE=ctx.selector_tile_size, LOCAL_W=ctx.local_window,
            ROUTE_SCALE=float(ctx.route_prior_scale), BLOCK_Q=block_q, BLOCK_LOCAL=block_local,
            MASK_ATOMICS=ctx.mask_atomics,
            num_warps=4, num_stages=2,
        )
        return (
            dquery.to(query.dtype),
            dkey.to(key.dtype),
            dvalue.to(value.dtype),
            droute.to(route_logits.dtype),
            None, None, None, None, None, None, None, None,
        )


class HierarchicalSparseAttentionV16HISACausal(nn.Module):
    """V16 strict-causal HISA module with an eager oracle and Triton core."""

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int = 32,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int = 32,
        *,
        local_window: int | None = None,
        selector_tile_size: int | None = None,
        temperature: float = 1.0,
        route_prior_scale: float = 1.0,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        if D != H * hd:
            raise ValueError(f"D={D} must equal H*hd={H * hd}")
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        self.H = int(H)
        self.num_heads = int(H)
        self.hd = int(hd)
        self.num_chunks = int(num_chunks)
        self.top_k_chunks = int(top_k_chunks)
        self.hisa_top_m_tokens = int(hisa_top_m_tokens)
        self.local_window = int(
            local_window if local_window is not None else os.getenv("DWARF_HISA_V16_LOCAL_WINDOW", _DEFAULT_LOCAL_WINDOW)
        )
        self.selector_tile_size = int(
            selector_tile_size if selector_tile_size is not None else os.getenv("DWARF_HISA_V16_SELECTOR_TILE", _DEFAULT_SELECTOR_TILE)
        )
        if self.local_window < 1 or self.selector_tile_size < 1:
            raise ValueError("local_window and selector_tile_size must be positive")
        self.temperature = float(temperature)
        self.route_prior_scale = float(route_prior_scale)
        self.backend = (backend or os.getenv("DWARF_HISA_V16_BACKEND", "triton")).lower()
        if self.backend not in {"triton", "eager", "auto"}:
            raise ValueError("backend must be one of triton, eager, auto")
        self.triton_block_q = int(os.getenv("DWARF_HISA_V16_BLOCK_Q", "16"))
        self.backward_impl = os.getenv("DWARF_HISA_V16_BWD", "atomic_masked").strip().lower()
        if self.backward_impl not in {"atomic", "atomic_masked"}:
            raise ValueError("DWARF_HISA_V16_BWD must be atomic or atomic_masked")
        self.metadata_builder = os.getenv("DWARF_HISA_V16_METADATA_BUILDER", "eager").strip().lower()
        if self.metadata_builder not in {"blocked", "eager"}:
            raise ValueError("DWARF_HISA_V16_METADATA_BUILDER must be blocked or eager")
        self.metadata_tile_block_size = int(os.getenv("DWARF_HISA_V16_METADATA_TILE_BLOCK", "8"))
        if self.metadata_tile_block_size < 1:
            raise ValueError("DWARF_HISA_V16_METADATA_TILE_BLOCK must be positive")
        self.representative_mode = os.getenv("DWARF_HISA_V16_REP_MODE", "max_l2").strip().lower()
        if self.representative_mode not in {"max_l2", "top2_blend"}:
            raise ValueError("DWARF_HISA_V16_REP_MODE must be max_l2 or top2_blend")
        self.top2_blend_alpha = float(os.getenv("DWARF_HISA_V16_REP_BLEND_ALPHA", "0.25"))
        if not 0.0 <= self.top2_blend_alpha <= 1.0:
            raise ValueError("DWARF_HISA_V16_REP_BLEND_ALPHA must be in [0, 1]")
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)
        self._routing_entropy: torch.Tensor | float = float("nan")
        # Intentionally not a buffer/state-dict entry: this is a one-forward
        # training side channel and must never surface in ordinary inference.
        self.hisa_evidence_capture: HISASelectionCapture | None = None

    def forward(
        self,
        x: torch.Tensor,
        kv_inject=None,
        *,
        valid_lengths: torch.Tensor | None = None,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HISAMetadata]:
        del kv_inject
        # Clear before doing any work so a failed/new forward cannot leak a
        # stale differentiable graph to the trainer.
        self.hisa_evidence_capture = None
        batch_size, seq_len, _ = x.shape
        valid_lengths = _as_valid_lengths(
            valid_lengths if valid_lengths is not None else getattr(self, "_causal_control_valid_lengths", None),
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )
        query = _to_heads(self.W_q(x), batch_size=batch_size, seq_len=seq_len, heads=self.H, head_dim=self.hd)
        key = _to_heads(self.W_k(x), batch_size=batch_size, seq_len=seq_len, heads=self.H, head_dim=self.hd)
        value = _to_heads(self.W_v(x), batch_size=batch_size, seq_len=seq_len, heads=self.H, head_dim=self.hd)
        chunk_size = math.ceil(seq_len / self.num_chunks)
        reps = _completed_chunk_representatives(
            key,
            num_chunks=self.num_chunks,
            chunk_size=chunk_size,
            valid_lengths=valid_lengths,
            representative_mode=self.representative_mode,
            top2_blend_alpha=self.top2_blend_alpha,
        )
        route_logits = torch.matmul(query, reps.transpose(-2, -1)) / (math.sqrt(self.hd) * self.temperature)
        metadata_kwargs = dict(
            num_chunks=self.num_chunks,
            chunk_size=chunk_size,
            top_k_chunks=self.top_k_chunks,
            top_m_tokens=self.hisa_top_m_tokens,
            selector_tile_size=self.selector_tile_size,
            valid_lengths=valid_lengths,
        )
        if self.metadata_builder == "blocked":
            metadata = _build_causal_tile_metadata_blocked(
                query, key, route_logits, tile_block_size=self.metadata_tile_block_size, **metadata_kwargs
            )
        else:
            metadata = _build_causal_tile_metadata_eager(query, key, route_logits, **metadata_kwargs)
        if self.training and os.getenv("DWARF_HISA_EVIDENCE_AUX", "0") == "1":
            self.hisa_evidence_capture = HISASelectionCapture(
                route_logits=route_logits,
                metadata=metadata,
            )
        with torch.no_grad():
            tile_scores = route_logits[:, :, metadata.tile_starts.clamp_max(seq_len - 1), :]
            self._routing_entropy = torch.softmax(tile_scores.float(), dim=-1).mul(
                torch.log_softmax(tile_scores.float(), dim=-1)
            ).sum(dim=-1).neg().mean().detach()
        use_triton = self.backend == "triton" or (self.backend == "auto" and x.is_cuda)
        if use_triton and x.is_cuda and _TRITON_AVAILABLE:
            attended = _V16HISATritonFn.apply(
                query,
                key,
                value,
                route_logits,
                metadata.top_chunk_idx,
                metadata.token_idx,
                valid_lengths,
                self.selector_tile_size,
                self.local_window,
                self.route_prior_scale,
                self.triton_block_q,
                self.backward_impl == "atomic_masked",
            )
        else:
            attended = _eager_v16_attention(
                query,
                key,
                value,
                route_logits,
                metadata,
                local_window=self.local_window,
                route_prior_scale=self.route_prior_scale,
            )
        out = self.W_o(attended.transpose(1, 2).reshape(batch_size, seq_len, self.H * self.hd))
        return (out, metadata) if return_metadata else out
