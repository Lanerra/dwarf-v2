"""Strict-causal HISA V18 under the V16/V17-compatible public class name.

Compared with the supplied V16 implementation, this revision uses a fixed
physical chunk size, tile-anchor routing logits, selected-only differentiable
route scores, an all-token metadata path with no redundant token-index tensor,
learnable centered route priors, counterfactual router supervision, an optional
exploration slot, normalized semantic chunk representatives, distinct local and
global K/V lanes, global-only low-rank value/key adapters, and magnitude-aware
EMA/NPCI injection. V18 additionally computes selected-route scores per selector
tile, can run the regular local lane through FlexAttention, and merges local and
irregular-global lanes exactly through their log-sum-exp normalizers. The legacy
combined Triton kernel remains available as a parity/fallback backend.

The eager implementation is the semantic oracle and CPU fallback. CUDA uses a
custom Triton forward/backward with FP32 source-gradient atomics.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    try:
        from torch.nn.attention.flex_attention import AuxRequest
    except Exception:  # PyTorch releases with the older return_lse surface
        AuxRequest = None
    _FLEX_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover - older PyTorch builds
    AuxRequest = None
    create_block_mask = None
    flex_attention = None
    _FLEX_ATTENTION_AVAILABLE = False


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only development
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _next_pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("softplus target must be finite and positive")
    return math.log(math.expm1(value))


def _to_heads(
    tensor: torch.Tensor,
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    # Triton consumes explicit strides; no full head-layout copy is needed.
    return tensor.reshape(batch_size, seq_len, heads, head_dim).permute(0, 2, 1, 3)


def _as_valid_lengths(
    lengths: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    result = lengths.to(device=device, dtype=torch.int32).reshape(-1)
    if result.numel() != batch_size:
        raise ValueError(f"valid_lengths must contain {batch_size} entries")
    valid = ((result >= 0) & (result <= seq_len)).all()
    if device.type == "cuda":
        torch._assert_async(valid, f"valid_lengths must be in [0,{seq_len}]")
    elif not bool(valid):
        raise ValueError(f"valid_lengths must be in [0,{seq_len}]")
    return result


def _magnitude_aware_rotate(
    x: torch.Tensor,
    delta: torch.Tensor,
    theta_h: torch.Tensor,
    *,
    strength_tau: float = 0.25,
) -> torch.Tensor:
    """Norm-preserving tangent rotation whose angle vanishes with delta norm."""
    tau = float(strength_tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("strength_tau must be finite and positive")
    xf, df = x.float(), delta.float()
    theta = theta_h.float().reshape(1, -1, 1, 1)
    norm = xf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = xf / norm
    perpendicular = df - (df * unit).sum(-1, keepdim=True) * unit
    perpendicular_norm = perpendicular.norm(dim=-1, keepdim=True)
    active = perpendicular_norm > norm * 1e-7
    direction = torch.where(
        active,
        perpendicular / perpendicular_norm.clamp_min(1e-20),
        torch.zeros_like(perpendicular),
    )
    strength = torch.tanh(perpendicular_norm / (tau * norm + 1e-12))
    angle = theta * strength
    rotated = torch.cos(angle) * xf + torch.sin(angle) * norm * direction
    return torch.where(active, rotated, xf).to(x.dtype)


@dataclass(frozen=True)
class HISAMetadata:
    top_chunk_idx: torch.Tensor          # int32 [B,H,T,K]
    token_idx: torch.Tensor              # empty or int32 [B,H,T,K,M]
    token_scores: torch.Tensor           # empty or fp32 [B,H,T,K,M]
    tile_starts: torch.Tensor            # int32 [T]
    valid_lengths: torch.Tensor          # int32 [B]
    chunk_size: int
    selector_tile_size: int
    enumerate_all: bool


@dataclass(frozen=True)
class HISASelectionCapture:
    """Ephemeral differentiable route surface for external analysis/losses."""
    anchor_logits: torch.Tensor
    metadata: HISAMetadata
    auxiliary_loss: torch.Tensor


def _chunk_layout(seq_len: int, chunk_size: int) -> tuple[int, int]:
    chunks = max(1, math.ceil(seq_len / chunk_size))
    return chunks, chunks * chunk_size


def _chunk_tensors(
    key: torch.Tensor,
    chunk_size: int,
    valid_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, heads, seq_len, head_dim = key.shape
    chunks, padded_len = _chunk_layout(seq_len, chunk_size)
    padded = F.pad(key, (0, 0, 0, padded_len - seq_len)) if padded_len > seq_len else key
    values = padded.reshape(batch_size, heads, chunks, chunk_size, head_dim)
    ids = torch.arange(padded_len, device=key.device, dtype=torch.int32).reshape(
        1, 1, chunks, chunk_size
    )
    token_valid = ids < valid_lengths.reshape(batch_size, 1, 1, 1)
    starts = torch.arange(chunks, device=key.device, dtype=torch.int32) * chunk_size
    chunk_valid = starts.reshape(1, chunks) < valid_lengths.reshape(batch_size, 1)
    return values, token_valid, chunk_valid


def _completed_chunk_representatives(
    key: torch.Tensor,
    *,
    chunk_size: int,
    valid_lengths: torch.Tensor,
    representative_mode: str,
    blend_alpha: float,
) -> torch.Tensor:
    chunks, token_valid, chunk_valid = _chunk_tensors(key, chunk_size, valid_lengths)
    values = chunks.float()
    valid_float = token_valid.unsqueeze(-1).to(values.dtype)
    mean = (values * valid_float).sum(3) / valid_float.sum(3).clamp_min(1.0)
    energy = values.square().sum(-1).masked_fill(~token_valid, float("-inf"))
    best = energy.argmax(-1, keepdim=True)
    max_vector = torch.gather(
        values,
        3,
        best.unsqueeze(-1).expand(-1, -1, -1, 1, values.shape[-1]),
    ).squeeze(3)
    if representative_mode == "max_l2":
        representative = max_vector
    elif representative_mode == "mean":
        representative = mean
    elif representative_mode == "mean_max_blend":
        representative = (1.0 - blend_alpha) * mean + blend_alpha * max_vector
    elif representative_mode == "top2_blend":
        count = min(2, chunks.shape[3])
        top = energy.topk(count, dim=-1).indices
        if count == 1:
            representative = max_vector
        else:
            salient = torch.gather(
                values,
                3,
                top.unsqueeze(-1).expand(-1, -1, -1, -1, values.shape[-1]),
            )
            representative = (
                (1.0 - blend_alpha) * salient[..., 0, :]
                + blend_alpha * salient[..., 1, :]
            )
    else:
        raise ValueError(
            "representative_mode must be max_l2, mean, mean_max_blend, or top2_blend"
        )
    representative = F.normalize(representative, dim=-1, eps=1e-6)
    return representative.masked_fill(
        ~chunk_valid[:, None, :, None],
        0.0,
    ).to(key.dtype)


def _eligibility(
    tile_starts: torch.Tensor,
    num_chunks: int,
    chunk_size: int,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    chunk_starts = torch.arange(
        num_chunks,
        device=tile_starts.device,
        dtype=torch.int32,
    ) * chunk_size
    chunk_ends = chunk_starts + chunk_size
    tile_valid = tile_starts.reshape(1, -1) < valid_lengths.reshape(-1, 1)
    chunk_valid = chunk_starts.reshape(1, -1) < valid_lengths.reshape(-1, 1)
    completed = chunk_ends.reshape(1, 1, -1) <= tile_starts.reshape(1, -1, 1)
    return completed & chunk_valid[:, None, :] & tile_valid[:, :, None]


def _inject_exploration_slot(
    indices: torch.Tensor,
    valid: torch.Tensor,
    eligible: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability <= 0.0 or indices.shape[-1] == 0:
        return indices, valid
    batch_size, heads, tiles, slots = indices.shape
    chunks = eligible.shape[-1]
    selected_counts = torch.zeros(
        batch_size,
        heads,
        tiles,
        chunks,
        dtype=torch.int32,
        device=indices.device,
    )
    # scatter_ can let an invalid -1 slot (clamped to zero) overwrite a valid
    # chunk-zero selection. Integer scatter_add preserves set membership.
    selected_counts.scatter_add_(
        -1,
        indices.clamp_min(0).long(),
        valid.to(torch.int32),
    )
    selected = selected_counts > 0
    candidate = eligible[:, None].expand(-1, heads, -1, -1) & ~selected
    random_scores = torch.rand(candidate.shape, device=indices.device).masked_fill(
        ~candidate,
        -1.0,
    )
    best_value, best_choice = random_scores.max(-1)
    replace = (
        torch.rand(batch_size, heads, tiles, device=indices.device) < probability
    ) & (best_value >= 0.0)
    result_indices = indices.clone()
    result_valid = valid.clone()
    result_indices[..., slots - 1] = torch.where(
        replace,
        best_choice.to(result_indices.dtype),
        result_indices[..., slots - 1],
    )
    result_valid[..., slots - 1] = torch.where(
        replace,
        torch.ones_like(result_valid[..., slots - 1]),
        result_valid[..., slots - 1],
    )
    return result_indices, result_valid


def _build_causal_tile_metadata(
    anchor_query: torch.Tensor,
    global_key: torch.Tensor,
    anchor_logits: torch.Tensor,
    *,
    chunk_size: int,
    top_k_chunks: int,
    top_m_tokens: int,
    selector_tile_size: int,
    valid_lengths: torch.Tensor,
    canonical_token_order: bool,
    exploration_probability: float,
) -> HISAMetadata:
    batch_size, heads, tiles, head_dim = anchor_query.shape
    seq_len = global_key.shape[2]
    num_chunks, padded_len = _chunk_layout(seq_len, chunk_size)
    k_slots = min(top_k_chunks, num_chunks)
    m_slots = min(top_m_tokens, chunk_size)
    enumerate_all = not canonical_token_order and m_slots == chunk_size
    tile_starts = torch.arange(
        tiles,
        device=global_key.device,
        dtype=torch.int32,
    ) * selector_tile_size
    eligible = _eligibility(tile_starts, num_chunks, chunk_size, valid_lengths)
    with torch.no_grad():
        masked_logits = anchor_logits.detach().masked_fill(
            ~eligible[:, None],
            float("-inf"),
        )
        values, indices = masked_logits.topk(k_slots, dim=-1)
        valid_selected = torch.isfinite(values)
        indices = torch.where(valid_selected, indices, torch.full_like(indices, -1))
        indices, valid_selected = _inject_exploration_slot(
            indices,
            valid_selected,
            eligible,
            exploration_probability,
        )
        top_chunks = indices.to(torch.int32)
        if enumerate_all:
            token_idx = torch.empty(0, dtype=torch.int32, device=global_key.device)
            token_scores = torch.empty(0, dtype=torch.float32, device=global_key.device)
        else:
            padded = (
                F.pad(global_key, (0, 0, 0, padded_len - seq_len))
                if padded_len > seq_len
                else global_key
            )
            key_chunks = padded.reshape(
                batch_size,
                heads,
                num_chunks,
                chunk_size,
                head_dim,
            )
            safe_indices = indices.clamp_min(0).long()
            batch_index = torch.arange(batch_size, device=global_key.device).reshape(
                batch_size, 1, 1, 1
            )
            head_index = torch.arange(heads, device=global_key.device).reshape(
                1, heads, 1, 1
            )
            selected_keys = key_chunks[batch_index, head_index, safe_indices]
            within = torch.arange(
                chunk_size,
                device=global_key.device,
                dtype=torch.int32,
            ).reshape(1, 1, 1, 1, -1)
            absolute = indices[..., None].to(torch.int32) * chunk_size + within
            token_valid = (
                valid_selected[..., None]
                & (absolute >= 0)
                & (absolute < valid_lengths.reshape(batch_size, 1, 1, 1, 1))
                & (absolute < tile_starts.reshape(1, 1, tiles, 1, 1))
            )
            scores = torch.einsum(
                "bhtd,bhtkmd->bhtkm",
                anchor_query.detach().float(),
                selected_keys.detach().float(),
            ) / math.sqrt(head_dim)
            scores = scores.masked_fill(~token_valid, float("-inf"))
            token_scores, local_indices = scores.topk(m_slots, dim=-1)
            absolute_selected = (
                indices[..., None].to(torch.int32) * chunk_size
                + local_indices.to(torch.int32)
            )
            finite = torch.isfinite(token_scores) & valid_selected[..., None]
            token_idx = torch.where(
                finite,
                absolute_selected,
                torch.full_like(absolute_selected, -1),
            ).to(torch.int32)
            token_scores = torch.where(
                finite,
                token_scores.float(),
                torch.full_like(token_scores.float(), float("-inf")),
            )
    return HISAMetadata(
        top_chunk_idx=top_chunks,
        token_idx=token_idx,
        token_scores=token_scores,
        tile_starts=tile_starts,
        valid_lengths=valid_lengths.detach(),
        chunk_size=int(chunk_size),
        selector_tile_size=int(selector_tile_size),
        enumerate_all=bool(enumerate_all),
    )


def _global_ids(
    metadata: HISAMetadata,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = metadata.top_chunk_idx
    if metadata.enumerate_all:
        within = torch.arange(
            metadata.chunk_size,
            device=chunks.device,
            dtype=torch.int32,
        ).reshape(1, 1, 1, 1, -1)
        ids = chunks[..., None] * metadata.chunk_size + within
    else:
        ids = metadata.token_idx
    valid = (chunks[..., None] >= 0) & (ids >= 0) & (ids < seq_len)
    return ids, valid


def _selected_route_scores(
    query_normalized: torch.Tensor,
    representatives: torch.Tensor,
    metadata: HISAMetadata,
    route_scale_by_head: torch.Tensor,
) -> torch.Tensor:
    """Score selected representatives without an N x K x HD gather.

    Selection is tile-scoped, so gather [B,H,T,K,HD] once and contract it with
    the [B,H,T,S,HD] query tiles. The previous implementation expanded selected
    representatives independently for every query row, creating a roughly
    selector_tile_size-times larger intermediate.
    """
    batch_size, heads, seq_len, head_dim = query_normalized.shape
    tiles = metadata.top_chunk_idx.shape[2]
    selector = metadata.selector_tile_size
    padded_len = tiles * selector
    query_padded = (
        F.pad(query_normalized, (0, 0, 0, padded_len - seq_len))
        if padded_len > seq_len
        else query_normalized
    )
    query_tiles = query_padded.reshape(
        batch_size, heads, tiles, selector, head_dim
    )
    chunks = metadata.top_chunk_idx
    safe = chunks.clamp_min(0).long()
    batch_index = torch.arange(batch_size, device=query_normalized.device).reshape(
        batch_size, 1, 1, 1
    )
    head_index = torch.arange(heads, device=query_normalized.device).reshape(
        1, heads, 1, 1
    )
    selected = representatives[batch_index, head_index, safe]
    scores = torch.einsum(
        "bhtsd,bhtkd->bhtsk",
        query_tiles.float(),
        selected.float(),
    )
    valid = chunks >= 0
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    mean = (scores * valid[..., None, :]).sum(-1, keepdim=True) / count[..., None]
    centered = torch.where(
        valid[..., None, :],
        scores - mean,
        torch.zeros_like(scores),
    )
    scaled = centered * route_scale_by_head.reshape(1, heads, 1, 1, 1).float()
    return scaled.reshape(batch_size, heads, padded_len, -1)[:, :, :seq_len].to(
        query_normalized.dtype
    )


def _router_auxiliary_loss(
    anchor_logits: torch.Tensor,
    anchor_query: torch.Tensor,
    global_key: torch.Tensor,
    metadata: HISAMetadata,
    *,
    samples: int,
    temperature: float,
) -> torch.Tensor:
    """Train anchor routing toward sampled token-level chunk evidence."""
    if samples <= 0:
        return anchor_logits.sum() * 0.0
    batch_size, heads, tiles, num_chunks = anchor_logits.shape
    eligibility_all = _eligibility(
        metadata.tile_starts,
        num_chunks,
        metadata.chunk_size,
        metadata.valid_lengths,
    )
    # At least one physical chunk can be complete only once tile_start reaches
    # chunk_size. Build that static candidate range directly instead of using
    # data-dependent nonzero(), which would split torch.compile graphs. Rows
    # whose valid length still makes a sampled tile ineligible are masked below.
    first_useful_tile = math.ceil(
        metadata.chunk_size / metadata.selector_tile_size
    )
    candidate_count = max(0, tiles - first_useful_tile)
    if candidate_count == 0:
        return anchor_logits.sum() * 0.0
    sample_count = min(int(samples), candidate_count)
    useful_tiles = torch.arange(
        first_useful_tile, tiles, device=anchor_logits.device, dtype=torch.long
    )
    if sample_count == candidate_count:
        tile_ids = useful_tiles
    else:
        permutation = torch.randperm(candidate_count, device=anchor_logits.device)
        tile_ids = useful_tiles[permutation[:sample_count]]
    key_chunks, token_valid, _ = _chunk_tensors(
        global_key,
        metadata.chunk_size,
        metadata.valid_lengths,
    )
    query = anchor_query[:, :, tile_ids].float()
    token_scores = torch.einsum(
        "bhsd,bhcmd->bhscm",
        query,
        key_chunks.float(),
    ) / math.sqrt(global_key.shape[-1])
    eligible = eligibility_all[:, tile_ids]
    mask = token_valid[:, :, None] & eligible[:, None, :, :, None]
    token_scores = token_scores.masked_fill(~mask, -1e9)
    oracle = torch.logsumexp(token_scores, dim=-1)
    route = anchor_logits[:, :, tile_ids].float().masked_fill(
        ~eligible[:, None],
        -1e9,
    )
    target = torch.softmax((oracle / float(temperature)).detach(), dim=-1)
    per_row = -(target * torch.log_softmax(route, dim=-1)).sum(-1)
    valid_rows = eligible.any(-1)[:, None].expand(-1, heads, -1)
    return (per_row * valid_rows).sum() / valid_rows.sum().clamp_min(1)


def _eligible_route_entropy(
    anchor_logits: torch.Tensor,
    metadata: HISAMetadata,
) -> torch.Tensor:
    eligible = _eligibility(
        metadata.tile_starts,
        anchor_logits.shape[-1],
        metadata.chunk_size,
        metadata.valid_lengths,
    )
    masked = anchor_logits.float().masked_fill(~eligible[:, None], -1e9)
    probability = torch.softmax(masked, dim=-1)
    entropy = -(probability * torch.log_softmax(masked, dim=-1)).sum(-1)
    valid = eligible.any(-1)[:, None].expand_as(entropy)
    return (entropy * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def _routing_quality_diagnostics(
    query: torch.Tensor,
    global_key: torch.Tensor,
    representatives: torch.Tensor,
    anchor_logits: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
    max_queries_per_batch: int,
) -> dict[str, torch.Tensor]:
    """Exact-on-sampled-row routing metrics without an N-by-N allocation."""
    device = query.device
    batch_size, heads, seq_len, head_dim = query.shape
    num_chunks = representatives.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    zeros = lambda: torch.zeros((), device=device, dtype=torch.float32)
    selected_hits, selected_targets = zeros(), zeros()
    candidate_hits, candidate_targets = zeros(), zeros()
    representative_misses, representative_total, representative_regret = zeros(), zeros(), zeros()
    overlap_removed, overlap_total = zeros(), zeros()
    sampled = 0
    all_ids, all_valid = _global_ids(metadata, seq_len)

    for batch in range(batch_size):
        valid_len = int(metadata.valid_lengths[batch].item())
        eligible_tiles = [
            tile
            for tile, start in enumerate(metadata.tile_starts.tolist())
            if start < valid_len and start // metadata.chunk_size > 0
        ]
        if not eligible_tiles:
            continue
        budget = min(max_queries_per_batch, len(eligible_tiles))
        chosen = (
            torch.linspace(0, len(eligible_tiles) - 1, steps=budget)
            .round()
            .long()
            .tolist()
        )
        for choice in chosen:
            tile = eligible_tiles[choice]
            q_pos = int(metadata.tile_starts[tile].item())
            eligible_count = min(num_chunks, q_pos // metadata.chunk_size)
            if eligible_count <= 0:
                continue
            k_eff = min(metadata.top_chunk_idx.shape[-1], eligible_count)
            chunk_ids = torch.arange(eligible_count, device=device)
            token_ids = (
                chunk_ids[:, None] * metadata.chunk_size
                + torch.arange(metadata.chunk_size, device=device)[None, :]
            )
            token_mask = token_ids < valid_len
            safe_ids = token_ids.clamp_max(seq_len - 1)
            keys = global_key[batch, :, safe_ids, :]
            token_scores = torch.einsum(
                "hd,hcmd->hcm",
                query[batch, :, q_pos],
                keys,
            ) * scale
            token_scores = token_scores.masked_fill(~token_mask[None], float("-inf"))
            chunk_scores = token_scores.max(-1).values
            ideal_chunks = torch.argsort(
                chunk_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., :k_eff]
            selected = metadata.top_chunk_idx[batch, :, tile]
            selected_hits += (
                ideal_chunks.unsqueeze(-1) == selected.unsqueeze(-2)
            ).any(-1).sum()
            selected_targets += heads * k_eff

            representative_scores = torch.einsum(
                "hd,hcd->hc",
                F.normalize(query[batch, :, q_pos].float(), dim=-1),
                representatives[batch, :, :eligible_count].float(),
            )
            exact_scores = chunk_scores
            representative_misses += (
                representative_scores.argmax(-1) != exact_scores.argmax(-1)
            ).sum()
            representative_regret += (
                exact_scores.max(-1).values
                - exact_scores.gather(
                    -1,
                    representative_scores.argmax(-1, keepdim=True),
                ).squeeze(-1)
            ).sum()
            representative_total += heads

            candidate_ids = all_ids[batch, :, tile].reshape(heads, -1)
            candidate_valid = all_valid[batch, :, tile].reshape(heads, -1)
            local_start = max(0, q_pos - local_window)
            removed = candidate_valid & (candidate_ids >= local_start)
            overlap_removed += removed.sum()
            overlap_total += candidate_valid.sum()
            kept = candidate_valid & (candidate_ids < local_start)

            flat_ids = token_ids[token_mask]
            global_eligible_ids = flat_ids[flat_ids < local_start]
            candidate_budget = min(
                int(kept.sum(-1).max().item()) if kept.numel() else 0,
                int(global_eligible_ids.numel()),
            )
            if candidate_budget > 0:
                oracle_scores = torch.einsum(
                    "hd,hnd->hn",
                    query[batch, :, q_pos],
                    global_key[batch, :, global_eligible_ids],
                ) * scale
                oracle = global_eligible_ids[
                    torch.argsort(
                        oracle_scores,
                        dim=-1,
                        descending=True,
                        stable=True,
                    )[..., :candidate_budget]
                ]
                candidate_hits += (
                    (oracle.unsqueeze(-1) == candidate_ids.unsqueeze(-2))
                    & kept.unsqueeze(-2)
                ).any(-1).sum()
                candidate_targets += heads * candidate_budget
            sampled += 1

    return {
        "sampled_queries": torch.tensor(sampled, device=device, dtype=torch.float32),
        "selected_chunk_recall": selected_hits / selected_targets.clamp_min(1),
        "final_candidate_recall": candidate_hits / candidate_targets.clamp_min(1),
        "representative_miss_rate": representative_misses / representative_total.clamp_min(1),
        "mean_representative_regret": representative_regret / representative_total.clamp_min(1),
        "local_overlap_removal_rate": overlap_removed / overlap_total.clamp_min(1),
    }


def _eager_local_lane(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    local_window: int,
    selector_tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict sliding-window lane returning normalized output and natural LSE."""
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, heads, seq_len), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    offsets = torch.arange(local_window, device=query.device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    for start in range(0, seq_len, selector_tile_size):
        end = min(start + selector_tile_size, seq_len)
        positions = torch.arange(start, end, device=query.device, dtype=torch.int32)
        q = query[:, :, start:end]
        q_valid = positions.reshape(1, -1) < valid_lengths.reshape(batch_size, 1)
        ids = positions[:, None] - local_window + offsets[None]
        valid = (
            (ids >= 0)
            & (ids < positions[:, None])
            & (ids < valid_lengths.reshape(batch_size, 1, 1))
            & q_valid[:, :, None]
        )
        safe = ids.clamp(0, max(seq_len - 1, 0)).long()
        keys = key[:, :, safe]
        values = value[:, :, safe]
        scores = torch.einsum("bhqd,bhqwd->bhqw", q, keys) * scale
        scores = scores.masked_fill(~valid[:, None], float("-inf"))
        maximum = scores.max(-1).values
        safe_max = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.where(
            torch.isfinite(scores),
            torch.exp(scores.float() - safe_max[..., None].float()),
            torch.zeros_like(scores.float()),
        )
        denominator = weights.sum(-1)
        lane = torch.einsum(
            "bhqw,bhqwd->bhqd", weights.to(values.dtype), values
        ) / denominator.clamp_min(1.0)[..., None].to(values.dtype)
        output[:, :, start:end] = torch.where(
            q_valid[:, None, :, None], lane, torch.zeros_like(lane)
        )
        lse[:, :, start:end] = torch.where(
            denominator > 0, safe_max.float() + denominator.log(),
            torch.full_like(safe_max.float(), float("-inf")),
        )
    return output, lse


def _eager_global_lane(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    route: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Irregular selected-chunk lane returning normalized output and LSE."""
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, heads, seq_len), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    all_ids, all_valid = _global_ids(metadata, seq_len)
    scale = 1.0 / math.sqrt(head_dim)
    for tile in range(metadata.top_chunk_idx.shape[2]):
        start = tile * metadata.selector_tile_size
        end = min(start + metadata.selector_tile_size, seq_len)
        if start >= end:
            break
        positions = torch.arange(start, end, device=query.device, dtype=torch.int32)
        q = query[:, :, start:end]
        q_valid = positions.reshape(1, -1) < metadata.valid_lengths.reshape(batch_size, 1)
        ids = all_ids[:, :, tile].reshape(batch_size, heads, -1)
        id_valid = all_valid[:, :, tile].reshape(batch_size, heads, -1)
        safe_ids = ids.clamp(0, max(seq_len - 1, 0)).long()
        keys = torch.gather(
            global_key, 2, safe_ids[..., None].expand(-1, -1, -1, head_dim)
        )
        values = torch.gather(
            global_value, 2, safe_ids[..., None].expand(-1, -1, -1, head_dim)
        )
        repeat = metadata.chunk_size if metadata.enumerate_all else metadata.token_idx.shape[-1]
        prior = route[:, :, start:end].repeat_interleave(repeat, dim=-1)
        scores = torch.matmul(q, keys.transpose(-2, -1)) * scale + prior
        valid = (
            id_valid[:, :, None]
            & (ids[:, :, None] < positions.reshape(1, 1, -1, 1) - local_window)
            & (ids[:, :, None] < metadata.valid_lengths.reshape(batch_size, 1, 1, 1))
            & q_valid[:, None, :, None]
        )
        scores = scores.masked_fill(~valid, float("-inf"))
        maximum = scores.max(-1).values
        safe_max = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.where(
            torch.isfinite(scores),
            torch.exp(scores.float() - safe_max[..., None].float()),
            torch.zeros_like(scores.float()),
        )
        denominator = weights.sum(-1)
        lane = torch.matmul(weights.to(values.dtype), values) / denominator.clamp_min(1.0)[..., None].to(values.dtype)
        output[:, :, start:end] = torch.where(
            q_valid[:, None, :, None], lane, torch.zeros_like(lane)
        )
        lse[:, :, start:end] = torch.where(
            denominator > 0, safe_max.float() + denominator.log(),
            torch.full_like(safe_max.float(), float("-inf")),
        )
    return output, lse


def _merge_attention_lanes(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    global_output: torch.Tensor,
    global_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly merge two independently normalized softmax lanes by LSE."""
    local_valid = torch.isfinite(local_lse)
    global_valid = torch.isfinite(global_lse)
    any_valid = local_valid | global_valid
    maximum = torch.maximum(local_lse, global_lse)
    safe_maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
    local_weight = torch.where(
        local_valid, torch.exp(local_lse - safe_maximum), torch.zeros_like(local_lse)
    )
    global_weight = torch.where(
        global_valid, torch.exp(global_lse - safe_maximum), torch.zeros_like(global_lse)
    )
    denominator = local_weight + global_weight
    output = (
        local_weight[..., None].to(local_output.dtype) * local_output
        + global_weight[..., None].to(global_output.dtype) * global_output
    ) / denominator.clamp_min(1.0)[..., None].to(local_output.dtype)
    combined_lse = torch.where(
        any_valid, safe_maximum + denominator.clamp_min(1e-30).log(),
        torch.full_like(safe_maximum, float("-inf")),
    )
    return output, combined_lse


def _eager_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    route: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
) -> torch.Tensor:
    """Tile-vectorized semantic oracle used on CPU and for CUDA parity tests."""
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    score_scale = 1.0 / math.sqrt(head_dim)
    global_ids_all, global_valid_all = _global_ids(metadata, seq_len)
    local_offsets = torch.arange(
        local_window,
        device=query.device,
        dtype=torch.int32,
    )
    for tile in range(metadata.top_chunk_idx.shape[2]):
        start = tile * metadata.selector_tile_size
        end = min((tile + 1) * metadata.selector_tile_size, seq_len)
        if start >= end:
            break
        query_positions = torch.arange(
            start,
            end,
            device=query.device,
            dtype=torch.int32,
        )
        query_tile = query[:, :, start:end]
        query_valid = query_positions.reshape(1, -1) < metadata.valid_lengths.reshape(
            batch_size, 1
        )
        local_ids = (
            query_positions[:, None] - local_window + local_offsets[None]
        )
        local_valid = (
            (local_ids >= 0)
            & (local_ids < query_positions[:, None])
            & (local_ids < metadata.valid_lengths.reshape(batch_size, 1, 1))
            & query_valid[:, :, None]
        )
        safe_local_ids = local_ids.clamp(0, max(seq_len - 1, 0)).long()
        local_keys = local_key[:, :, safe_local_ids]
        local_values = local_value[:, :, safe_local_ids]
        local_scores = torch.einsum(
            "bhqd,bhqwd->bhqw",
            query_tile,
            local_keys,
        ) * score_scale
        local_scores = local_scores.masked_fill(
            ~local_valid[:, None],
            float("-inf"),
        )

        ids = global_ids_all[:, :, tile].reshape(batch_size, heads, -1)
        id_valid = global_valid_all[:, :, tile].reshape(batch_size, heads, -1)
        safe_ids = ids.clamp(0, max(seq_len - 1, 0)).long()
        global_keys = torch.gather(
            global_key,
            2,
            safe_ids[..., None].expand(-1, -1, -1, head_dim),
        )
        global_values = torch.gather(
            global_value,
            2,
            safe_ids[..., None].expand(-1, -1, -1, head_dim),
        )
        repeat = (
            metadata.chunk_size
            if metadata.enumerate_all
            else metadata.token_idx.shape[-1]
        )
        route_prior = route[:, :, start:end].repeat_interleave(repeat, dim=-1)
        global_scores = (
            torch.matmul(query_tile, global_keys.transpose(-2, -1)) * score_scale
            + route_prior
        )
        global_valid = (
            id_valid[:, :, None]
            & (
                ids[:, :, None]
                < query_positions.reshape(1, 1, -1, 1) - local_window
            )
            & (ids[:, :, None] < metadata.valid_lengths.reshape(batch_size, 1, 1, 1))
            & query_valid[:, None, :, None]
        )
        global_scores = global_scores.masked_fill(
            ~global_valid,
            float("-inf"),
        )

        maximum = torch.maximum(
            local_scores.max(-1).values,
            global_scores.max(-1).values,
        )
        safe_maximum = torch.where(
            torch.isfinite(maximum),
            maximum,
            torch.zeros_like(maximum),
        )
        local_probability = torch.where(
            torch.isfinite(local_scores),
            torch.exp(local_scores.float() - safe_maximum[..., None].float()),
            torch.zeros_like(local_scores.float()),
        )
        global_probability = torch.where(
            torch.isfinite(global_scores),
            torch.exp(global_scores.float() - safe_maximum[..., None].float()),
            torch.zeros_like(global_scores.float()),
        )
        denominator = local_probability.sum(-1) + global_probability.sum(-1)
        accumulator = torch.einsum(
            "bhqw,bhqwd->bhqd",
            local_probability.to(local_values.dtype),
            local_values,
        ) + torch.matmul(
            global_probability.to(global_values.dtype),
            global_values,
        )
        tile_output = accumulator / denominator.clamp_min(1.0)[..., None].to(
            accumulator.dtype
        )
        output[:, :, start:end] = torch.where(
            query_valid[:, None, :, None],
            tile_output,
            torch.zeros_like(tile_output),
        )
    return output


if _TRITON_AVAILABLE:

    @triton.jit
    def _online_lane(scores, selected, values, running_max, running_sum, accumulator):
        lane_max = tl.max(scores, axis=1)
        has_lane = lane_max > float("-inf")
        merged_max = tl.maximum(running_max, lane_max)
        safe_max = tl.where(
            (running_max > float("-inf")) | has_lane,
            merged_max,
            0.0,
        )
        old_scale = tl.where(
            running_max > float("-inf"),
            tl.exp(running_max - safe_max),
            0.0,
        )
        probability = tl.where(selected, tl.exp(scores - safe_max[:, None]), 0.0)
        running_sum = running_sum * old_scale + tl.sum(probability, axis=1)
        accumulator = (
            accumulator * old_scale[:, None]
            + tl.dot(probability.to(values.dtype), values, input_precision="ieee")
        )
        running_max = tl.where(
            (running_max > float("-inf")) | has_lane,
            merged_max,
            running_max,
        )
        return running_max, running_sum, accumulator

    @triton.jit
    def _global_lane_forward_kernel(
        Q, GLOBAL_K, GLOBAL_V, ROUTE, CHUNKS, IDS, LENGTHS, OUT, LSE,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        srb, srh, srn, srk,
        scb, sch, sct, sck,
        sib, sih, sit, sik, sim,
        sob, soh, son, sod,
        slseb, slseh, slsen,
        N, H: tl.constexpr, HD: tl.constexpr,
        K_VAL: tl.constexpr, M_VAL: tl.constexpr, M_PAD: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ENUMERATE_ALL: tl.constexpr,
        SELECTOR_TILE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        tile = tl.program_id(1)
        query_block = tl.program_id(2)
        batch = batch_head // H
        head = batch_head % H
        query_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        query_positions = tile * SELECTOR_TILE + query_offsets
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_mask = (
            (query_offsets < SELECTOR_TILE)
            & (query_positions < N)
            & (query_positions < valid_length)
        )
        dimensions = tl.arange(0, HD)
        query = tl.load(
            Q + batch * sqb + head * sqh
            + query_positions[:, None] * sqn + dimensions[None, :] * sqd,
            mask=query_mask[:, None], other=0.0,
        )
        score_scale = 1.0 / tl.sqrt(HD * 1.0)
        running_max = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_Q], tl.float32)
        accumulator = tl.zeros([BLOCK_Q, HD], tl.float32)
        chunk_base = CHUNKS + batch * scb + head * sch + tile * sct
        id_base = IDS + batch * sib + head * sih + tile * sit
        route_base = ROUTE + batch * srb + head * srh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            chunk_valid = chunk >= 0
            if ENUMERATE_ALL:
                ids = chunk * CHUNK_SIZE + token_offsets
            else:
                ids = tl.load(
                    id_base + slot * sik + token_offsets * sim,
                    mask=token_mask, other=-1,
                ).to(tl.int32)
            id_mask = (
                token_mask & chunk_valid & (ids >= 0) & (ids < N)
                & (ids < valid_length)
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            keys = tl.load(
                GLOBAL_K + batch * sgkb + head * sgkh
                + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
                mask=id_mask[:, None], other=0.0,
            )
            prior = tl.load(
                route_base + query_positions * srn + slot * srk,
                mask=query_mask & chunk_valid, other=0.0,
            ).to(tl.float32)
            scores = (
                tl.dot(query, tl.trans(keys), input_precision="ieee") * score_scale
                + prior[:, None]
            )
            selected = (
                query_mask[:, None] & id_mask[None, :]
                & (ids[None, :] < query_positions[:, None] - LOCAL_WINDOW)
            )
            scores = tl.where(selected, scores, float("-inf"))
            values = tl.load(
                GLOBAL_V + batch * sgvb + head * sgvh
                + safe_ids[:, None] * sgvn + dimensions[None, :] * sgvd,
                mask=id_mask[:, None], other=0.0,
            )
            running_max, running_sum, accumulator = _online_lane(
                scores, selected, values, running_max, running_sum, accumulator
            )
        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        tl.store(
            OUT + batch * sob + head * soh
            + query_positions[:, None] * son + dimensions[None, :] * sod,
            accumulator / denominator[:, None], mask=query_mask[:, None],
        )
        lse = tl.where(
            running_sum > 0.0, running_max + tl.log(running_sum), float("-inf")
        )
        tl.store(
            LSE + batch * slseb + head * slseh + query_positions * slsen,
            lse, mask=query_mask,
        )


    @triton.jit
    def _global_lane_backward_kernel(
        Q, GLOBAL_K, GLOBAL_V, OUT, DOUT, DLSE, LSE,
        ROUTE, CHUNKS, IDS, LENGTHS,
        DQ, DGLOBAL_K, DGLOBAL_V, DROUTE,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        sob, soh, son, sod,
        sdob, sdoh, sdon, sdod,
        sdlb, sdlh, sdln,
        slseb, slseh, slsen,
        srb, srh, srn, srk,
        scb, sch, sct, sck,
        sib, sih, sit, sik, sim,
        sdqb, sdqh, sdqn, sdqd,
        sdgkb, sdgkh, sdgkn, sdgkd,
        sdgvb, sdgvh, sdgvn, sdgvd,
        sdrb, sdrh, sdrn, sdrk,
        N, H: tl.constexpr, HD: tl.constexpr,
        K_VAL: tl.constexpr, M_VAL: tl.constexpr, M_PAD: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ENUMERATE_ALL: tl.constexpr,
        SELECTOR_TILE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        BLOCK_Q: tl.constexpr, MASK_ATOMICS: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        tile = tl.program_id(1)
        query_block = tl.program_id(2)
        batch = batch_head // H
        head = batch_head % H
        query_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        query_positions = tile * SELECTOR_TILE + query_offsets
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_mask = (
            (query_offsets < SELECTOR_TILE)
            & (query_positions < N)
            & (query_positions < valid_length)
        )
        dimensions = tl.arange(0, HD)
        scale = 1.0 / tl.sqrt(HD * 1.0)
        query = tl.load(
            Q + batch * sqb + head * sqh
            + query_positions[:, None] * sqn + dimensions[None, :] * sqd,
            mask=query_mask[:, None], other=0.0,
        )
        output = tl.load(
            OUT + batch * sob + head * soh
            + query_positions[:, None] * son + dimensions[None, :] * sod,
            mask=query_mask[:, None], other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            DOUT + batch * sdob + head * sdoh
            + query_positions[:, None] * sdon + dimensions[None, :] * sdod,
            mask=query_mask[:, None], other=0.0,
        ).to(tl.float32)
        lse_gradient = tl.load(
            DLSE + batch * sdlb + head * sdlh + query_positions * sdln,
            mask=query_mask, other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + batch * slseb + head * slseh + query_positions * slsen,
            mask=query_mask, other=float("-inf"),
        )
        lse_valid = lse > float("-inf")
        safe_lse = tl.where(lse_valid, lse, 0.0)
        delta = tl.sum(output_gradient * output, axis=1)
        dquery = tl.zeros([BLOCK_Q, HD], tl.float32)
        chunk_base = CHUNKS + batch * scb + head * sch + tile * sct
        id_base = IDS + batch * sib + head * sih + tile * sit
        route_base = ROUTE + batch * srb + head * srh
        droute_base = DROUTE + batch * sdrb + head * sdrh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            chunk_valid = chunk >= 0
            if ENUMERATE_ALL:
                ids = chunk * CHUNK_SIZE + token_offsets
            else:
                ids = tl.load(
                    id_base + slot * sik + token_offsets * sim,
                    mask=token_mask, other=-1,
                ).to(tl.int32)
            id_mask = (
                token_mask & chunk_valid & (ids >= 0) & (ids < N)
                & (ids < valid_length)
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            keys = tl.load(
                GLOBAL_K + batch * sgkb + head * sgkh
                + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
                mask=id_mask[:, None], other=0.0,
            )
            prior = tl.load(
                route_base + query_positions * srn + slot * srk,
                mask=query_mask & chunk_valid, other=0.0,
            ).to(tl.float32)
            selected = (
                query_mask[:, None] & id_mask[None, :]
                & (ids[None, :] < query_positions[:, None] - LOCAL_WINDOW)
            )
            scores = (
                tl.dot(query, tl.trans(keys), input_precision="ieee") * scale
                + prior[:, None]
            )
            scores = tl.where(selected, scores, float("-inf"))
            probability = tl.where(
                selected & lse_valid[:, None],
                tl.exp(scores - safe_lse[:, None]), 0.0,
            )
            values = tl.load(
                GLOBAL_V + batch * sgvb + head * sgvh
                + safe_ids[:, None] * sgvn + dimensions[None, :] * sgvd,
                mask=id_mask[:, None], other=0.0,
            ).to(tl.float32)
            dscore = probability * (
                tl.dot(output_gradient, tl.trans(values), input_precision="ieee")
                - delta[:, None] + lse_gradient[:, None]
            )
            dquery += tl.dot(dscore, keys.to(tl.float32), input_precision="ieee") * scale
            dkey = tl.dot(tl.trans(dscore), query.to(tl.float32), input_precision="ieee") * scale
            dvalue = tl.dot(tl.trans(probability), output_gradient, input_precision="ieee")
            write = id_mask
            if MASK_ATOMICS:
                write = id_mask & (tl.sum(selected.to(tl.int32), axis=0) > 0)
            tl.atomic_add(
                DGLOBAL_K + batch * sdgkb + head * sdgkh
                + safe_ids[:, None] * sdgkn + dimensions[None, :] * sdgkd,
                dkey, mask=write[:, None], sem="relaxed",
            )
            tl.atomic_add(
                DGLOBAL_V + batch * sdgvb + head * sdgvh
                + safe_ids[:, None] * sdgvn + dimensions[None, :] * sdgvd,
                dvalue, mask=write[:, None], sem="relaxed",
            )
            tl.store(
                droute_base + query_positions * sdrn + slot * sdrk,
                tl.sum(dscore, axis=1), mask=query_mask & chunk_valid,
            )
        tl.store(
            DQ + batch * sdqb + head * sdqh
            + query_positions[:, None] * sdqn + dimensions[None, :] * sdqd,
            dquery, mask=query_mask[:, None],
        )


    @triton.jit
    def _hisa_forward_kernel(
        Q, LOCAL_K, LOCAL_V, GLOBAL_K, GLOBAL_V,
        ROUTE, CHUNKS, IDS, LENGTHS, OUT, LSE,
        sqb, sqh, sqn, sqd,
        slkb, slkh, slkn, slkd,
        slvb, slvh, slvn, slvd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        srb, srh, srn, srk,
        scb, sch, sct, sck,
        sib, sih, sit, sik, sim,
        sob, soh, son, sod,
        slseb, slseh, slsen,
        N, H: tl.constexpr, HD: tl.constexpr,
        K_VAL: tl.constexpr, M_VAL: tl.constexpr, M_PAD: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ENUMERATE_ALL: tl.constexpr,
        SELECTOR_TILE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_HISTORY: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        tile = tl.program_id(1)
        query_block = tl.program_id(2)
        batch = batch_head // H
        head = batch_head % H
        query_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        block_start = tile * SELECTOR_TILE + query_block * BLOCK_Q
        query_positions = tile * SELECTOR_TILE + query_offsets
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_mask = (
            (query_offsets < SELECTOR_TILE)
            & (query_positions < N)
            & (query_positions < valid_length)
        )
        dimensions = tl.arange(0, HD)
        query = tl.load(
            Q
            + batch * sqb
            + head * sqh
            + query_positions[:, None] * sqn
            + dimensions[None, :] * sqd,
            mask=query_mask[:, None],
            other=0.0,
        )
        score_scale = 1.0 / tl.sqrt(HD * 1.0)
        running_max = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_Q], tl.float32)
        accumulator = tl.zeros([BLOCK_Q, HD], tl.float32)

        # W-token history ending immediately before this query block.
        history_offsets = tl.arange(0, BLOCK_HISTORY)
        history_ids = block_start - LOCAL_WINDOW + history_offsets
        history_mask = (
            (history_offsets < LOCAL_WINDOW)
            & (history_ids >= 0)
            & (history_ids < N)
            & (history_ids < valid_length)
        )
        history_keys = tl.load(
            LOCAL_K
            + batch * slkb
            + head * slkh
            + history_ids[:, None] * slkn
            + dimensions[None, :] * slkd,
            mask=history_mask[:, None],
            other=0.0,
        )
        history_scores = (
            tl.dot(query, tl.trans(history_keys), input_precision="ieee")
            * score_scale
        )
        history_selected = (
            query_mask[:, None]
            & history_mask[None, :]
            & (history_ids[None, :] >= query_positions[:, None] - LOCAL_WINDOW)
            & (history_ids[None, :] < query_positions[:, None])
            & (history_ids[None, :] < block_start)
        )
        history_scores = tl.where(
            history_selected,
            history_scores,
            float("-inf"),
        )
        history_values = tl.load(
            LOCAL_V
            + batch * slvb
            + head * slvh
            + history_ids[:, None] * slvn
            + dimensions[None, :] * slvd,
            mask=history_mask[:, None],
            other=0.0,
        )
        running_max, running_sum, accumulator = _online_lane(
            history_scores,
            history_selected,
            history_values,
            running_max,
            running_sum,
            accumulator,
        )

        # Small intra-block strict-causal lane.
        intra_offsets = tl.arange(0, BLOCK_Q)
        intra_ids = block_start + intra_offsets
        intra_mask = (intra_ids < N) & (intra_ids < valid_length)
        intra_keys = tl.load(
            LOCAL_K
            + batch * slkb
            + head * slkh
            + intra_ids[:, None] * slkn
            + dimensions[None, :] * slkd,
            mask=intra_mask[:, None],
            other=0.0,
        )
        intra_scores = (
            tl.dot(query, tl.trans(intra_keys), input_precision="ieee")
            * score_scale
        )
        intra_selected = (
            query_mask[:, None]
            & intra_mask[None, :]
            & (intra_ids[None, :] < query_positions[:, None])
            & (intra_ids[None, :] >= query_positions[:, None] - LOCAL_WINDOW)
        )
        intra_scores = tl.where(intra_selected, intra_scores, float("-inf"))
        intra_values = tl.load(
            LOCAL_V
            + batch * slvb
            + head * slvh
            + intra_ids[:, None] * slvn
            + dimensions[None, :] * slvd,
            mask=intra_mask[:, None],
            other=0.0,
        )
        running_max, running_sum, accumulator = _online_lane(
            intra_scores,
            intra_selected,
            intra_values,
            running_max,
            running_sum,
            accumulator,
        )

        chunk_base = CHUNKS + batch * scb + head * sch + tile * sct
        id_base = IDS + batch * sib + head * sih + tile * sit
        route_base = ROUTE + batch * srb + head * srh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            chunk_valid = chunk >= 0
            if ENUMERATE_ALL:
                ids = chunk * CHUNK_SIZE + token_offsets
            else:
                ids = tl.load(
                    id_base + slot * sik + token_offsets * sim,
                    mask=token_mask,
                    other=-1,
                ).to(tl.int32)
            id_mask = (
                token_mask
                & chunk_valid
                & (ids >= 0)
                & (ids < N)
                & (ids < valid_length)
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            global_keys = tl.load(
                GLOBAL_K
                + batch * sgkb
                + head * sgkh
                + safe_ids[:, None] * sgkn
                + dimensions[None, :] * sgkd,
                mask=id_mask[:, None],
                other=0.0,
            )
            prior = tl.load(
                route_base + query_positions * srn + slot * srk,
                mask=query_mask & chunk_valid,
                other=0.0,
            ).to(tl.float32)
            global_scores = (
                tl.dot(query, tl.trans(global_keys), input_precision="ieee")
                * score_scale
                + prior[:, None]
            )
            global_selected = (
                query_mask[:, None]
                & id_mask[None, :]
                & (ids[None, :] < query_positions[:, None] - LOCAL_WINDOW)
            )
            global_scores = tl.where(
                global_selected,
                global_scores,
                float("-inf"),
            )
            global_values = tl.load(
                GLOBAL_V
                + batch * sgvb
                + head * sgvh
                + safe_ids[:, None] * sgvn
                + dimensions[None, :] * sgvd,
                mask=id_mask[:, None],
                other=0.0,
            )
            running_max, running_sum, accumulator = _online_lane(
                global_scores,
                global_selected,
                global_values,
                running_max,
                running_sum,
                accumulator,
            )

        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        tl.store(
            OUT
            + batch * sob
            + head * soh
            + query_positions[:, None] * son
            + dimensions[None, :] * sod,
            accumulator / denominator[:, None],
            mask=query_mask[:, None],
        )
        lse = tl.where(
            running_sum > 0.0,
            running_max + tl.log(running_sum),
            float("-inf"),
        )
        tl.store(
            LSE
            + batch * slseb
            + head * slseh
            + query_positions * slsen,
            lse,
            mask=query_mask,
        )

    @triton.jit
    def _hisa_backward_kernel(
        Q, LOCAL_K, LOCAL_V, GLOBAL_K, GLOBAL_V,
        OUT, DOUT, LSE, ROUTE, CHUNKS, IDS, LENGTHS,
        DQ, DLOCAL_K, DLOCAL_V, DGLOBAL_K, DGLOBAL_V, DROUTE,
        sqb, sqh, sqn, sqd,
        slkb, slkh, slkn, slkd,
        slvb, slvh, slvn, slvd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        sob, soh, son, sod,
        sdob, sdoh, sdon, sdod,
        slseb, slseh, slsen,
        srb, srh, srn, srk,
        scb, sch, sct, sck,
        sib, sih, sit, sik, sim,
        sdqb, sdqh, sdqn, sdqd,
        sdlkb, sdlkh, sdlkn, sdlkd,
        sdlvb, sdlvh, sdlvn, sdlvd,
        sdgkb, sdgkh, sdgkn, sdgkd,
        sdgvb, sdgvh, sdgvn, sdgvd,
        sdrb, sdrh, sdrn, sdrk,
        N, H: tl.constexpr, HD: tl.constexpr,
        K_VAL: tl.constexpr, M_VAL: tl.constexpr, M_PAD: tl.constexpr,
        CHUNK_SIZE: tl.constexpr, ENUMERATE_ALL: tl.constexpr,
        SELECTOR_TILE: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_HISTORY: tl.constexpr,
        MASK_ATOMICS: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        tile = tl.program_id(1)
        query_block = tl.program_id(2)
        batch = batch_head // H
        head = batch_head % H
        query_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        block_start = tile * SELECTOR_TILE + query_block * BLOCK_Q
        query_positions = tile * SELECTOR_TILE + query_offsets
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_mask = (
            (query_offsets < SELECTOR_TILE)
            & (query_positions < N)
            & (query_positions < valid_length)
        )
        dimensions = tl.arange(0, HD)
        score_scale = 1.0 / tl.sqrt(HD * 1.0)
        query = tl.load(
            Q
            + batch * sqb
            + head * sqh
            + query_positions[:, None] * sqn
            + dimensions[None, :] * sqd,
            mask=query_mask[:, None],
            other=0.0,
        )
        output = tl.load(
            OUT
            + batch * sob
            + head * soh
            + query_positions[:, None] * son
            + dimensions[None, :] * sod,
            mask=query_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            DOUT
            + batch * sdob
            + head * sdoh
            + query_positions[:, None] * sdon
            + dimensions[None, :] * sdod,
            mask=query_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE
            + batch * slseb
            + head * slseh
            + query_positions * slsen,
            mask=query_mask,
            other=float("-inf"),
        )
        lse_valid = lse > float("-inf")
        safe_lse = tl.where(lse_valid, lse, 0.0)
        delta = tl.sum(output_gradient * output, axis=1)
        dquery = tl.zeros([BLOCK_Q, HD], tl.float32)

        history_offsets = tl.arange(0, BLOCK_HISTORY)
        history_ids = block_start - LOCAL_WINDOW + history_offsets
        history_mask = (
            (history_offsets < LOCAL_WINDOW)
            & (history_ids >= 0)
            & (history_ids < N)
            & (history_ids < valid_length)
        )
        safe_history_ids = tl.maximum(tl.minimum(history_ids, N - 1), 0)
        history_keys = tl.load(
            LOCAL_K
            + batch * slkb
            + head * slkh
            + safe_history_ids[:, None] * slkn
            + dimensions[None, :] * slkd,
            mask=history_mask[:, None],
            other=0.0,
        )
        history_selected = (
            query_mask[:, None]
            & history_mask[None, :]
            & (history_ids[None, :] >= query_positions[:, None] - LOCAL_WINDOW)
            & (history_ids[None, :] < query_positions[:, None])
            & (history_ids[None, :] < block_start)
        )
        history_scores = (
            tl.dot(query, tl.trans(history_keys), input_precision="ieee")
            * score_scale
        )
        history_scores = tl.where(
            history_selected,
            history_scores,
            float("-inf"),
        )
        history_probability = tl.where(
            history_selected & lse_valid[:, None],
            tl.exp(history_scores - safe_lse[:, None]),
            0.0,
        )
        history_values = tl.load(
            LOCAL_V
            + batch * slvb
            + head * slvh
            + safe_history_ids[:, None] * slvn
            + dimensions[None, :] * slvd,
            mask=history_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        history_dscore = history_probability * (
            tl.dot(output_gradient, tl.trans(history_values), input_precision="ieee")
            - delta[:, None]
        )
        dquery += (
            tl.dot(
                history_dscore,
                history_keys.to(tl.float32),
                input_precision="ieee",
            )
            * score_scale
        )
        dhistory_key = (
            tl.dot(
                tl.trans(history_dscore),
                query.to(tl.float32),
                input_precision="ieee",
            )
            * score_scale
        )
        dhistory_value = tl.dot(
            tl.trans(history_probability),
            output_gradient,
            input_precision="ieee",
        )
        history_write = history_mask
        if MASK_ATOMICS:
            history_write = history_mask & (
                tl.sum(history_selected.to(tl.int32), axis=0) > 0
            )
        tl.atomic_add(
            DLOCAL_K
            + batch * sdlkb
            + head * sdlkh
            + safe_history_ids[:, None] * sdlkn
            + dimensions[None, :] * sdlkd,
            dhistory_key,
            mask=history_write[:, None],
            sem="relaxed",
        )
        tl.atomic_add(
            DLOCAL_V
            + batch * sdlvb
            + head * sdlvh
            + safe_history_ids[:, None] * sdlvn
            + dimensions[None, :] * sdlvd,
            dhistory_value,
            mask=history_write[:, None],
            sem="relaxed",
        )

        intra_offsets = tl.arange(0, BLOCK_Q)
        intra_ids = block_start + intra_offsets
        intra_mask = (intra_ids < N) & (intra_ids < valid_length)
        safe_intra_ids = tl.maximum(tl.minimum(intra_ids, N - 1), 0)
        intra_keys = tl.load(
            LOCAL_K
            + batch * slkb
            + head * slkh
            + safe_intra_ids[:, None] * slkn
            + dimensions[None, :] * slkd,
            mask=intra_mask[:, None],
            other=0.0,
        )
        intra_selected = (
            query_mask[:, None]
            & intra_mask[None, :]
            & (intra_ids[None, :] < query_positions[:, None])
            & (intra_ids[None, :] >= query_positions[:, None] - LOCAL_WINDOW)
        )
        intra_scores = (
            tl.dot(query, tl.trans(intra_keys), input_precision="ieee")
            * score_scale
        )
        intra_scores = tl.where(intra_selected, intra_scores, float("-inf"))
        intra_probability = tl.where(
            intra_selected & lse_valid[:, None],
            tl.exp(intra_scores - safe_lse[:, None]),
            0.0,
        )
        intra_values = tl.load(
            LOCAL_V
            + batch * slvb
            + head * slvh
            + safe_intra_ids[:, None] * slvn
            + dimensions[None, :] * slvd,
            mask=intra_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        intra_dscore = intra_probability * (
            tl.dot(output_gradient, tl.trans(intra_values), input_precision="ieee")
            - delta[:, None]
        )
        dquery += (
            tl.dot(
                intra_dscore,
                intra_keys.to(tl.float32),
                input_precision="ieee",
            )
            * score_scale
        )
        dintra_key = (
            tl.dot(
                tl.trans(intra_dscore),
                query.to(tl.float32),
                input_precision="ieee",
            )
            * score_scale
        )
        dintra_value = tl.dot(
            tl.trans(intra_probability),
            output_gradient,
            input_precision="ieee",
        )
        intra_write = intra_mask
        if MASK_ATOMICS:
            intra_write = intra_mask & (
                tl.sum(intra_selected.to(tl.int32), axis=0) > 0
            )
        tl.atomic_add(
            DLOCAL_K
            + batch * sdlkb
            + head * sdlkh
            + safe_intra_ids[:, None] * sdlkn
            + dimensions[None, :] * sdlkd,
            dintra_key,
            mask=intra_write[:, None],
            sem="relaxed",
        )
        tl.atomic_add(
            DLOCAL_V
            + batch * sdlvb
            + head * sdlvh
            + safe_intra_ids[:, None] * sdlvn
            + dimensions[None, :] * sdlvd,
            dintra_value,
            mask=intra_write[:, None],
            sem="relaxed",
        )

        chunk_base = CHUNKS + batch * scb + head * sch + tile * sct
        id_base = IDS + batch * sib + head * sih + tile * sit
        route_base = ROUTE + batch * srb + head * srh
        droute_base = DROUTE + batch * sdrb + head * sdrh
        token_offsets = tl.arange(0, M_PAD)
        token_mask = token_offsets < M_VAL
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            chunk_valid = chunk >= 0
            if ENUMERATE_ALL:
                ids = chunk * CHUNK_SIZE + token_offsets
            else:
                ids = tl.load(
                    id_base + slot * sik + token_offsets * sim,
                    mask=token_mask,
                    other=-1,
                ).to(tl.int32)
            id_mask = (
                token_mask
                & chunk_valid
                & (ids >= 0)
                & (ids < N)
                & (ids < valid_length)
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            global_keys = tl.load(
                GLOBAL_K
                + batch * sgkb
                + head * sgkh
                + safe_ids[:, None] * sgkn
                + dimensions[None, :] * sgkd,
                mask=id_mask[:, None],
                other=0.0,
            )
            prior = tl.load(
                route_base + query_positions * srn + slot * srk,
                mask=query_mask & chunk_valid,
                other=0.0,
            ).to(tl.float32)
            global_selected = (
                query_mask[:, None]
                & id_mask[None, :]
                & (ids[None, :] < query_positions[:, None] - LOCAL_WINDOW)
            )
            global_scores = (
                tl.dot(query, tl.trans(global_keys), input_precision="ieee")
                * score_scale
                + prior[:, None]
            )
            global_scores = tl.where(
                global_selected,
                global_scores,
                float("-inf"),
            )
            global_probability = tl.where(
                global_selected & lse_valid[:, None],
                tl.exp(global_scores - safe_lse[:, None]),
                0.0,
            )
            global_values = tl.load(
                GLOBAL_V
                + batch * sgvb
                + head * sgvh
                + safe_ids[:, None] * sgvn
                + dimensions[None, :] * sgvd,
                mask=id_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            global_dscore = global_probability * (
                tl.dot(output_gradient, tl.trans(global_values), input_precision="ieee")
                - delta[:, None]
            )
            dquery += (
                tl.dot(
                    global_dscore,
                    global_keys.to(tl.float32),
                    input_precision="ieee",
                )
                * score_scale
            )
            dglobal_key = (
                tl.dot(
                    tl.trans(global_dscore),
                    query.to(tl.float32),
                    input_precision="ieee",
                )
                * score_scale
            )
            dglobal_value = tl.dot(
                tl.trans(global_probability),
                output_gradient,
                input_precision="ieee",
            )
            global_write = id_mask
            if MASK_ATOMICS:
                global_write = id_mask & (
                    tl.sum(global_selected.to(tl.int32), axis=0) > 0
                )
            tl.atomic_add(
                DGLOBAL_K
                + batch * sdgkb
                + head * sdgkh
                + safe_ids[:, None] * sdgkn
                + dimensions[None, :] * sdgkd,
                dglobal_key,
                mask=global_write[:, None],
                sem="relaxed",
            )
            tl.atomic_add(
                DGLOBAL_V
                + batch * sdgvb
                + head * sdgvh
                + safe_ids[:, None] * sdgvn
                + dimensions[None, :] * sdgvd,
                dglobal_value,
                mask=global_write[:, None],
                sem="relaxed",
            )
            # Route slots are unique coordinates even when two slots happen to
            # reference the same chunk during exploration.
            tl.store(
                droute_base + query_positions * sdrn + slot * sdrk,
                tl.sum(global_dscore, axis=1),
                mask=query_mask & chunk_valid,
            )

        tl.store(
            DQ
            + batch * sdqb
            + head * sdqh
            + query_positions[:, None] * sdqn
            + dimensions[None, :] * sdqd,
            dquery,
            mask=query_mask[:, None],
        )


class _GlobalHISATritonFn(torch.autograd.Function):
    """Irregular global lane with explicit output and LSE gradients."""

    @staticmethod
    def forward(
        ctx,
        query,
        global_key,
        global_value,
        route,
        chunks,
        token_idx,
        valid_lengths,
        chunk_size,
        enumerate_all,
        selector_tile_size,
        local_window,
        requested_block_q,
        mask_atomics,
    ):
        batch_size, heads, seq_len, head_dim = query.shape
        metadata = HISAMetadata(
            top_chunk_idx=chunks,
            token_idx=token_idx,
            token_scores=torch.empty(0, device=query.device),
            tile_starts=(
                torch.arange(chunks.shape[2], device=query.device, dtype=torch.int32)
                * int(selector_tile_size)
            ),
            valid_lengths=valid_lengths,
            chunk_size=int(chunk_size),
            selector_tile_size=int(selector_tile_size),
            enumerate_all=bool(enumerate_all),
        )
        if not query.is_cuda or not _TRITON_AVAILABLE:
            return _eager_global_lane(
                query, global_key, global_value, route, metadata,
                local_window=int(local_window),
            )
        if not _is_power_of_two(head_dim):
            raise ValueError("Triton HISA requires a power-of-two head dimension")
        block_q = int(requested_block_q) if requested_block_q > 0 else 16
        if block_q < 16 or not _is_power_of_two(block_q):
            raise ValueError("HISA BLOCK_Q must be a power of two >=16")
        slots = chunks.shape[-1]
        m_slots = int(chunk_size) if enumerate_all else token_idx.shape[-1]
        m_pad = max(16, _next_pow2(m_slots))
        if m_pad > 256:
            raise ValueError("HISA global lane supports at most 256 tokens per chunk")
        storage = token_idx if token_idx.numel() else torch.empty(1, dtype=torch.int32, device=query.device)
        id_strides = storage.stride() if storage.ndim == 5 else (0, 0, 0, 0, 0)
        output = torch.zeros_like(query)
        lse = torch.full(
            (batch_size, heads, seq_len), float("-inf"),
            device=query.device, dtype=torch.float32,
        )
        grid = (
            batch_size * heads, chunks.shape[2],
            triton.cdiv(int(selector_tile_size), block_q),
        )
        _global_lane_forward_kernel[grid](
            query, global_key, global_value, route, chunks, storage, valid_lengths,
            output, lse,
            *query.stride(), *global_key.stride(), *global_value.stride(),
            *route.stride(), *chunks.stride(), *id_strides,
            *output.stride(), *lse.stride(),
            N=seq_len, H=heads, HD=head_dim, K_VAL=slots,
            M_VAL=m_slots, M_PAD=m_pad, CHUNK_SIZE=int(chunk_size),
            ENUMERATE_ALL=bool(enumerate_all),
            SELECTOR_TILE=int(selector_tile_size), LOCAL_WINDOW=int(local_window),
            BLOCK_Q=block_q, num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(
            query, global_key, global_value, route, chunks, storage,
            valid_lengths, output, lse,
        )
        ctx.chunk_size = int(chunk_size)
        ctx.enumerate_all = bool(enumerate_all)
        ctx.selector_tile_size = int(selector_tile_size)
        ctx.local_window = int(local_window)
        ctx.block_q = block_q
        ctx.mask_atomics = bool(mask_atomics)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        (
            query, global_key, global_value, route, chunks, storage,
            valid_lengths, output, lse,
        ) = ctx.saved_tensors
        batch_size, heads, seq_len, head_dim = query.shape
        grad_output = (
            torch.zeros_like(output) if grad_output is None else grad_output.contiguous()
        )
        grad_lse = (
            torch.zeros_like(lse) if grad_lse is None else grad_lse.contiguous()
        )
        dquery = torch.zeros_like(query, dtype=torch.float32)
        dglobal_key = torch.zeros_like(global_key, dtype=torch.float32)
        dglobal_value = torch.zeros_like(global_value, dtype=torch.float32)
        droute = torch.zeros_like(route, dtype=torch.float32)
        m_slots = ctx.chunk_size if ctx.enumerate_all else storage.shape[-1]
        m_pad = max(16, _next_pow2(m_slots))
        id_strides = storage.stride() if storage.ndim == 5 else (0, 0, 0, 0, 0)
        grid = (
            batch_size * heads, chunks.shape[2],
            triton.cdiv(ctx.selector_tile_size, ctx.block_q),
        )
        _global_lane_backward_kernel[grid](
            query, global_key, global_value, output, grad_output, grad_lse, lse,
            route, chunks, storage, valid_lengths,
            dquery, dglobal_key, dglobal_value, droute,
            *query.stride(), *global_key.stride(), *global_value.stride(),
            *output.stride(), *grad_output.stride(), *grad_lse.stride(),
            *lse.stride(), *route.stride(), *chunks.stride(), *id_strides,
            *dquery.stride(), *dglobal_key.stride(), *dglobal_value.stride(),
            *droute.stride(),
            N=seq_len, H=heads, HD=head_dim, K_VAL=chunks.shape[-1],
            M_VAL=m_slots, M_PAD=m_pad, CHUNK_SIZE=ctx.chunk_size,
            ENUMERATE_ALL=ctx.enumerate_all,
            SELECTOR_TILE=ctx.selector_tile_size, LOCAL_WINDOW=ctx.local_window,
            BLOCK_Q=ctx.block_q, MASK_ATOMICS=ctx.mask_atomics,
            num_warps=4, num_stages=2,
        )
        return (
            dquery.to(query.dtype),
            dglobal_key.to(global_key.dtype),
            dglobal_value.to(global_value.dtype),
            droute.to(route.dtype),
            None, None, None, None, None, None, None, None, None,
        )


class _HISATritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        global_key: torch.Tensor,
        global_value: torch.Tensor,
        route: torch.Tensor,
        chunks: torch.Tensor,
        token_idx: torch.Tensor,
        valid_lengths: torch.Tensor,
        chunk_size: int,
        enumerate_all: bool,
        selector_tile_size: int,
        local_window: int,
        requested_block_q: int,
        mask_atomics: bool,
    ) -> torch.Tensor:
        batch_size, heads, seq_len, head_dim = query.shape
        if not query.is_cuda or not _TRITON_AVAILABLE:
            metadata = HISAMetadata(
                top_chunk_idx=chunks,
                token_idx=token_idx,
                token_scores=torch.empty(0, device=query.device),
                tile_starts=(
                    torch.arange(chunks.shape[2], device=query.device, dtype=torch.int32)
                    * selector_tile_size
                ),
                valid_lengths=valid_lengths,
                chunk_size=int(chunk_size),
                selector_tile_size=int(selector_tile_size),
                enumerate_all=bool(enumerate_all),
            )
            return _eager_attention(
                query,
                local_key,
                local_value,
                global_key,
                global_value,
                route,
                metadata,
                local_window=local_window,
            )
        if not _is_power_of_two(head_dim):
            raise ValueError("Triton HISA requires a power-of-two head dimension")
        block_q = int(requested_block_q) if requested_block_q > 0 else 16
        if block_q < 16 or not _is_power_of_two(block_q):
            raise ValueError("HISA BLOCK_Q must be a power of two >=16")
        block_history = max(16, _next_pow2(local_window))
        if block_history > 256:
            raise ValueError("Triton HISA currently supports local_window <=256")
        k_slots = chunks.shape[-1]
        m_slots = int(chunk_size) if enumerate_all else token_idx.shape[-1]
        m_pad = max(16, _next_pow2(m_slots))
        if m_pad > 256:
            raise ValueError("Triton HISA currently supports at most 256 tokens per selected chunk")
        storage = (
            token_idx
            if token_idx.numel()
            else torch.empty(1, dtype=torch.int32, device=query.device)
        )
        id_strides = storage.stride() if storage.ndim == 5 else (0, 0, 0, 0, 0)
        output = torch.zeros_like(query)
        lse = torch.full(
            (batch_size, heads, seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        grid = (
            batch_size * heads,
            chunks.shape[2],
            triton.cdiv(selector_tile_size, block_q),
        )
        _hisa_forward_kernel[grid](
            query, local_key, local_value, global_key, global_value,
            route, chunks, storage, valid_lengths, output, lse,
            *query.stride(), *local_key.stride(), *local_value.stride(),
            *global_key.stride(), *global_value.stride(), *route.stride(),
            *chunks.stride(), *id_strides, *output.stride(), *lse.stride(),
            N=seq_len, H=heads, HD=head_dim,
            K_VAL=k_slots, M_VAL=m_slots, M_PAD=m_pad,
            CHUNK_SIZE=int(chunk_size), ENUMERATE_ALL=bool(enumerate_all),
            SELECTOR_TILE=int(selector_tile_size), LOCAL_WINDOW=int(local_window),
            BLOCK_Q=block_q, BLOCK_HISTORY=block_history,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(
            query, local_key, local_value, global_key, global_value,
            route, chunks, storage, valid_lengths, output, lse,
        )
        ctx.chunk_size = int(chunk_size)
        ctx.enumerate_all = bool(enumerate_all)
        ctx.selector_tile_size = int(selector_tile_size)
        ctx.local_window = int(local_window)
        ctx.block_q = block_q
        ctx.mask_atomics = bool(mask_atomics)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query, local_key, local_value, global_key, global_value,
            route, chunks, storage, valid_lengths, output, lse,
        ) = ctx.saved_tensors
        batch_size, heads, seq_len, head_dim = query.shape
        grad_output = grad_output.contiguous()
        dquery = torch.zeros_like(query, dtype=torch.float32)
        dlocal_key = torch.zeros_like(local_key, dtype=torch.float32)
        dlocal_value = torch.zeros_like(local_value, dtype=torch.float32)
        dglobal_key = torch.zeros_like(global_key, dtype=torch.float32)
        dglobal_value = torch.zeros_like(global_value, dtype=torch.float32)
        droute = torch.zeros_like(route, dtype=torch.float32)
        m_slots = ctx.chunk_size if ctx.enumerate_all else storage.shape[-1]
        m_pad = max(16, _next_pow2(m_slots))
        id_strides = storage.stride() if storage.ndim == 5 else (0, 0, 0, 0, 0)
        block_history = max(16, _next_pow2(ctx.local_window))
        grid = (
            batch_size * heads,
            chunks.shape[2],
            triton.cdiv(ctx.selector_tile_size, ctx.block_q),
        )
        _hisa_backward_kernel[grid](
            query, local_key, local_value, global_key, global_value,
            output, grad_output, lse, route, chunks, storage, valid_lengths,
            dquery, dlocal_key, dlocal_value, dglobal_key, dglobal_value, droute,
            *query.stride(), *local_key.stride(), *local_value.stride(),
            *global_key.stride(), *global_value.stride(),
            *output.stride(), *grad_output.stride(), *lse.stride(), *route.stride(),
            *chunks.stride(), *id_strides,
            *dquery.stride(), *dlocal_key.stride(), *dlocal_value.stride(),
            *dglobal_key.stride(), *dglobal_value.stride(), *droute.stride(),
            N=seq_len, H=heads, HD=head_dim,
            K_VAL=chunks.shape[-1], M_VAL=m_slots, M_PAD=m_pad,
            CHUNK_SIZE=ctx.chunk_size, ENUMERATE_ALL=ctx.enumerate_all,
            SELECTOR_TILE=ctx.selector_tile_size, LOCAL_WINDOW=ctx.local_window,
            BLOCK_Q=ctx.block_q, BLOCK_HISTORY=block_history,
            MASK_ATOMICS=ctx.mask_atomics,
            num_warps=4, num_stages=2,
        )
        return (
            dquery.to(query.dtype),
            dlocal_key.to(local_key.dtype),
            dlocal_value.to(local_value.dtype),
            dglobal_key.to(global_key.dtype),
            dglobal_value.to(global_value.dtype),
            droute.to(route.dtype),
            None, None, None, None, None, None, None, None, None,
        )


class HierarchicalSparseAttentionV16HISACausal(nn.Module):
    """V18 implementation retaining the supplied V16/V17 public class names."""

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int = 32,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int = 64,
        *,
        chunk_size: int | None = None,
        local_window: int | None = None,
        selector_tile_size: int | None = None,
        temperature: float = 1.0,
        route_prior_scale: float = 0.1,
        backend: str | None = None,
        token_selection_mode: str | None = None,
        representative_mode: str = "mean_max_blend",
        representative_blend_alpha: float = 0.5,
        exploration_probability: float = 0.05,
        route_aux_weight: float = 0.01,
        route_aux_samples: int = 4,
        route_aux_temperature: float = 1.0,
        global_adapter_rank: int = 16,
        npci_theta_max: float = 0.25,
        max_seq_len: int | None = None,
        local_backend: str = "flex",
        collect_routing_diagnostics: bool | None = None,
        diagnostic_max_queries: int | None = None,
    ) -> None:
        super().__init__()
        D, H, hd = int(D), int(H), int(hd)
        if D != H * hd:
            raise ValueError(f"D={D} must equal H*hd={H * hd}")
        if not _is_power_of_two(hd):
            raise ValueError("HISA head dimension must be a power of two")
        if top_k_chunks < 1 or hisa_top_m_tokens < 1:
            raise ValueError("top_k_chunks and hisa_top_m_tokens must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(route_prior_scale) or route_prior_scale <= 0:
            raise ValueError("route_prior_scale must be finite and positive")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0,1]")
        if route_aux_weight < 0 or route_aux_samples < 0:
            raise ValueError("route auxiliary weight/samples must be non-negative")
        if not math.isfinite(route_aux_temperature) or route_aux_temperature <= 0:
            raise ValueError("route_aux_temperature must be finite and positive")
        if representative_mode not in {
            "max_l2", "mean", "mean_max_blend", "top2_blend"
        }:
            raise ValueError("unsupported representative_mode")
        if not 0.0 <= representative_blend_alpha <= 1.0:
            raise ValueError("representative_blend_alpha must be in [0,1]")
        if global_adapter_rank < 0:
            raise ValueError("global_adapter_rank must be non-negative")
        if not math.isfinite(npci_theta_max) or npci_theta_max <= 0:
            raise ValueError("npci_theta_max must be finite and positive")
        local_backend = str(local_backend).lower()
        if local_backend not in {"flex", "combined"}:
            raise ValueError("local_backend must be flex or combined")
        if max_seq_len is not None and int(max_seq_len) < 1:
            raise ValueError("max_seq_len must be positive when supplied")

        resolved_chunk_size = (
            int(chunk_size)
            if chunk_size is not None
            else int(os.getenv("DWARF_HISA_CHUNK_SIZE", "64"))
        )
        resolved_local_window = (
            int(local_window)
            if local_window is not None
            else int(os.getenv("DWARF_HISA_V16_LOCAL_WINDOW", "64"))
        )
        resolved_selector_tile = (
            int(selector_tile_size)
            if selector_tile_size is not None
            else int(os.getenv("DWARF_HISA_V16_SELECTOR_TILE", "16"))
        )
        if resolved_chunk_size < 1 or resolved_local_window < 1 or resolved_selector_tile < 1:
            raise ValueError("chunk_size, local_window, and selector_tile_size must be positive")

        self.D = D
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.legacy_num_chunks = int(num_chunks)
        self.chunk_size = resolved_chunk_size
        self.top_k_chunks = int(top_k_chunks)
        self.hisa_top_m_tokens = int(hisa_top_m_tokens)
        self.local_window = resolved_local_window
        self.selector_tile_size = resolved_selector_tile
        self.temperature = float(temperature)
        self.backend = (backend or os.getenv("DWARF_HISA_V16_BACKEND", "auto")).lower()
        if self.backend not in {"auto", "eager", "triton"}:
            raise ValueError("backend must be auto, eager, or triton")
        self.token_selection_mode = (
            token_selection_mode
            or os.getenv("DWARF_HISA_V16_TOKEN_SELECTION", "auto")
        ).lower()
        if self.token_selection_mode not in {"auto", "canonical"}:
            raise ValueError("token_selection_mode must be auto or canonical")
        self.representative_mode = representative_mode
        self.representative_blend_alpha = float(representative_blend_alpha)
        self.exploration_probability = float(exploration_probability)
        self.route_aux_weight = float(route_aux_weight)
        self.route_aux_samples = int(route_aux_samples)
        self.route_aux_temperature = float(route_aux_temperature)
        self.global_adapter_rank = int(global_adapter_rank)
        self.npci_theta_max = float(npci_theta_max)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.local_backend = local_backend
        self._local_block_mask = None
        self._local_block_mask_key: tuple[str, int | None, int, int] | None = None
        self.triton_block_q = int(os.getenv("DWARF_HISA_V16_BLOCK_Q", "16"))
        self.backward_impl = os.getenv(
            "DWARF_HISA_V16_BWD",
            "atomic_masked",
        ).lower()
        if self.backward_impl not in {"atomic", "atomic_masked"}:
            raise ValueError("DWARF_HISA_V16_BWD must be atomic or atomic_masked")
        if collect_routing_diagnostics is None:
            collect_routing_diagnostics = (
                os.getenv("DWARF_HISA_V16_ROUTING_DIAGNOSTICS", "0") == "1"
            )
        self.collect_routing_diagnostics = bool(collect_routing_diagnostics)
        self.diagnostic_max_queries = int(diagnostic_max_queries or 8)
        if self.diagnostic_max_queries < 1:
            raise ValueError("diagnostic_max_queries must be positive")

        # Q/K/V and residual-content gate share one input GEMM.
        self.qkvg_proj = nn.Linear(D, 4 * D, bias=True)
        self.W_o = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.qkvg_proj.bias.zero_()
        if self.global_adapter_rank:
            rank = self.global_adapter_rank
            self.global_k_down = nn.Linear(D, rank, bias=False)
            self.global_k_up = nn.Linear(rank, D, bias=False)
            self.global_v_down = nn.Linear(D, rank, bias=False)
            self.global_v_up = nn.Linear(rank, D, bias=False)
            nn.init.zeros_(self.global_k_up.weight)
            nn.init.zeros_(self.global_v_up.weight)
        else:
            self.global_k_down = None
            self.global_k_up = None
            self.global_v_down = None
            self.global_v_up = None
        self.route_prior_raw = nn.Parameter(
            torch.full((H,), _inverse_softplus(route_prior_scale))
        )
        raw_theta = math.atanh(min(0.01 / max(self.npci_theta_max, 1e-6), 0.99))
        self.npci_theta_k = nn.Parameter(torch.full((H,), raw_theta))
        self.npci_theta_v = nn.Parameter(torch.full((H,), raw_theta))
        self._routing_entropy: torch.Tensor | float = float("nan")
        self._routing_diagnostics: dict[str, torch.Tensor] = {}
        self.hisa_evidence_capture: HISASelectionCapture | None = None
        self._routing_auxiliary_loss: torch.Tensor | None = None
        self._last_token_selection_path = ""

    @property
    def route_prior_scale(self) -> torch.Tensor:
        return F.softplus(self.route_prior_raw)

    def routing_auxiliary_loss(self, *, clear: bool = False) -> torch.Tensor | None:
        value = self._routing_auxiliary_loss
        if clear:
            self._routing_auxiliary_loss = None
        return value

    def semantic_config(self) -> dict[str, object]:
        return {
            "implementation": "hisa-v18-split-lse",
            "chunk_size": self.chunk_size,
            "top_k_chunks": self.top_k_chunks,
            "top_m_tokens": self.hisa_top_m_tokens,
            "local_window": self.local_window,
            "selector_tile_size": self.selector_tile_size,
            "temperature": self.temperature,
            "representative_mode": self.representative_mode,
            "representative_blend_alpha": self.representative_blend_alpha,
            "exploration_probability": self.exploration_probability,
            "route_aux_weight": self.route_aux_weight,
            "route_aux_samples": self.route_aux_samples,
            "route_aux_temperature": self.route_aux_temperature,
            "global_adapter_rank": self.global_adapter_rank,
            "npci_theta_max": self.npci_theta_max,
            "token_selection_mode": self.token_selection_mode,
            "selected_route_scoring": "tile_gather",
            "lane_merge": "exact_lse",
        }

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "local_backend": self.local_backend,
            "triton_block_q": self.triton_block_q,
            "backward_impl": self.backward_impl,
            "collect_routing_diagnostics": self.collect_routing_diagnostics,
            "flex_attention_available": _FLEX_ATTENTION_AVAILABLE,
            "flex_lse_api": "aux_request" if AuxRequest is not None else "return_lse",
        }


    def _ensure_local_block_mask(
        self, device: torch.device, seq_len: int
    ):
        if not _FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
        key = (device.type, device.index, int(seq_len), self.local_window)
        if self._local_block_mask is not None and self._local_block_mask_key == key:
            return self._local_block_mask
        window = int(self.local_window)

        def local_mask(_batch, _head, query_index, key_index):
            return (key_index < query_index) & (key_index >= query_index - window)

        self._local_block_mask = create_block_mask(
            local_mask,
            B=None,
            H=None,
            Q_LEN=int(seq_len),
            KV_LEN=int(seq_len),
            device=device,
            BLOCK_SIZE=128,
        )
        self._local_block_mask_key = key
        return self._local_block_mask

    def prepare_runtime(
        self, device: torch.device | str, seq_len: int | None = None
    ) -> None:
        """Prebuild non-state runtime metadata before trainer compilation."""
        if self.local_backend != "flex" or not _FLEX_ATTENTION_AVAILABLE:
            return
        length = int(seq_len if seq_len is not None else (self.max_seq_len or 0))
        if length > 0:
            self._ensure_local_block_mask(torch.device(device), length)

    def _local_lane(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.is_cuda and _FLEX_ATTENTION_AVAILABLE:
            block_mask = self._ensure_local_block_mask(query.device, query.shape[2])
            if AuxRequest is not None:
                output, auxiliary = flex_attention(
                    query,
                    local_key,
                    local_value,
                    block_mask=block_mask,
                    return_aux=AuxRequest(lse=True),
                )
                lse = auxiliary.lse
            else:
                # Compatibility with the pre-AuxRequest FlexAttention API.
                output, lse = flex_attention(
                    query, local_key, local_value,
                    block_mask=block_mask, return_lse=True,
                )
            positions = torch.arange(query.shape[2], device=query.device)
            valid = positions.reshape(1, -1) < valid_lengths.reshape(-1, 1)
            output = torch.where(
                valid[:, None, :, None], output, torch.zeros_like(output)
            )
            lse = torch.where(
                valid[:, None], lse, torch.full_like(lse, float("-inf"))
            )
            return output, lse
        return _eager_local_lane(
            query, local_key, local_value, valid_lengths,
            local_window=self.local_window,
            selector_tile_size=self.selector_tile_size,
        )

    def reset_global_adapters_(self) -> None:
        if self.global_adapter_rank:
            assert self.global_k_up is not None and self.global_v_up is not None
            nn.init.zeros_(self.global_k_up.weight)
            nn.init.zeros_(self.global_v_up.weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        fused_weight_key = prefix + "qkvg_proj.weight"
        fused_bias_key = prefix + "qkvg_proj.bias"
        if fused_weight_key not in state_dict:
            query_weight = state_dict.get(prefix + "W_q.weight")
            key_weight = state_dict.get(prefix + "W_k.weight")
            value_weight = state_dict.get(prefix + "W_v.weight")
            if query_weight is not None and key_weight is not None and value_weight is not None:
                parent_prefix = prefix[:-len("attn.")] if prefix.endswith("attn.") else prefix
                gate_weight = state_dict.get(parent_prefix + "gate_proj.weight")
                if gate_weight is None:
                    gate_weight = state_dict.get(parent_prefix + "global_gate.weight")
                if gate_weight is None:
                    gate_weight = torch.zeros_like(query_weight)
                query_bias = state_dict.get(
                    prefix + "W_q.bias",
                    query_weight.new_zeros(query_weight.shape[0]),
                )
                key_bias = state_dict.get(
                    prefix + "W_k.bias",
                    key_weight.new_zeros(key_weight.shape[0]),
                )
                value_bias = state_dict.get(
                    prefix + "W_v.bias",
                    value_weight.new_zeros(value_weight.shape[0]),
                )
                gate_bias = state_dict.get(parent_prefix + "gate_proj.bias")
                if gate_bias is None:
                    gate_bias = state_dict.get(parent_prefix + "global_gate.bias")
                if gate_bias is None:
                    gate_bias = query_weight.new_zeros(query_weight.shape[0])
                state_dict[fused_weight_key] = torch.cat(
                    (query_weight, key_weight, value_weight, gate_weight),
                    dim=0,
                )
                state_dict[fused_bias_key] = torch.cat(
                    (query_bias, key_bias, value_bias, gate_bias),
                    dim=0,
                )
                for name in (
                    "W_q.weight", "W_q.bias", "W_k.weight", "W_k.bias",
                    "W_v.weight", "W_v.bias",
                ):
                    state_dict.pop(prefix + name, None)
                for gate_name in ("gate_proj.weight", "gate_proj.bias", "global_gate.weight", "global_gate.bias"):
                    state_dict.pop(parent_prefix + gate_name, None)

        # Strictly load old checkpoints by supplying deterministic defaults for
        # newly introduced semantic parameters/adapters.
        current = self.state_dict()
        for local_key in (
            "route_prior_raw", "npci_theta_k", "npci_theta_v",
            "global_k_down.weight", "global_k_up.weight",
            "global_v_down.weight", "global_v_up.weight",
        ):
            full_key = prefix + local_key
            if local_key in current and full_key not in state_dict:
                state_dict[full_key] = current[local_key].detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_inject: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        valid_lengths: torch.Tensor | None = None,
        return_metadata: bool = False,
        return_auxiliary: bool = False,
    ):
        self.hisa_evidence_capture = None
        self._routing_auxiliary_loss = None
        batch_size, seq_len, _ = x.shape
        lengths = _as_valid_lengths(
            valid_lengths
            if valid_lengths is not None
            else getattr(self, "_causal_control_valid_lengths", None),
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )
        query_flat, key_flat, value_flat, gate = self.qkvg_proj(x).split(
            self.D,
            dim=-1,
        )
        query = _to_heads(query_flat, batch_size, seq_len, self.H, self.hd)
        local_key = _to_heads(key_flat, batch_size, seq_len, self.H, self.hd)
        local_value = _to_heads(value_flat, batch_size, seq_len, self.H, self.hd)
        global_key = local_key
        global_value = local_value
        if self.global_adapter_rank:
            assert self.global_k_down is not None and self.global_k_up is not None
            assert self.global_v_down is not None and self.global_v_up is not None
            global_key = global_key + _to_heads(
                self.global_k_up(self.global_k_down(x)),
                batch_size,
                seq_len,
                self.H,
                self.hd,
            )
            global_value = global_value + _to_heads(
                self.global_v_up(self.global_v_down(x)),
                batch_size,
                seq_len,
                self.H,
                self.hd,
            )
        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            if key_delta.shape != global_key.shape or value_delta.shape != global_value.shape:
                raise ValueError("kv_inject must contain [B,H,N,HD] tensors")
            global_key = _magnitude_aware_rotate(
                global_key,
                key_delta,
                self.npci_theta_max * torch.tanh(self.npci_theta_k),
            )
            global_value = _magnitude_aware_rotate(
                global_value,
                value_delta,
                self.npci_theta_max * torch.tanh(self.npci_theta_v),
            )

        representatives = _completed_chunk_representatives(
            global_key,
            chunk_size=self.chunk_size,
            valid_lengths=lengths,
            representative_mode=self.representative_mode,
            blend_alpha=self.representative_blend_alpha,
        )
        query_normalized = F.normalize(query.float(), dim=-1, eps=1e-6).to(query.dtype)
        tiles = math.ceil(seq_len / self.selector_tile_size)
        tile_starts_long = (
            torch.arange(tiles, device=x.device, dtype=torch.long)
            * self.selector_tile_size
        )
        anchor_query = query_normalized[:, :, tile_starts_long.clamp_max(seq_len - 1)]
        anchor_logits = torch.matmul(
            anchor_query.float(),
            representatives.float().transpose(-2, -1),
        ) / self.temperature
        canonical = self.token_selection_mode == "canonical" or return_metadata
        metadata = _build_causal_tile_metadata(
            anchor_query,
            global_key,
            anchor_logits,
            chunk_size=self.chunk_size,
            top_k_chunks=self.top_k_chunks,
            top_m_tokens=self.hisa_top_m_tokens,
            selector_tile_size=self.selector_tile_size,
            valid_lengths=lengths,
            canonical_token_order=canonical,
            exploration_probability=(
                self.exploration_probability if self.training else 0.0
            ),
        )
        self._last_token_selection_path = (
            "enumerate_all" if metadata.enumerate_all else "canonical_topk"
        )
        route = _selected_route_scores(
            query_normalized,
            representatives,
            metadata,
            self.route_prior_scale,
        )
        if self.training and self.route_aux_weight > 0.0:
            auxiliary = _router_auxiliary_loss(
                anchor_logits,
                anchor_query,
                global_key,
                metadata,
                samples=self.route_aux_samples,
                temperature=self.route_aux_temperature,
            ) * self.route_aux_weight
            self._routing_auxiliary_loss = auxiliary
            self.hisa_evidence_capture = HISASelectionCapture(
                anchor_logits=anchor_logits,
                metadata=metadata,
                auxiliary_loss=auxiliary,
            )

        with torch.no_grad():
            self._routing_entropy = _eligible_route_entropy(
                anchor_logits,
                metadata,
            ).detach()
            diagnostics: dict[str, torch.Tensor] = {
                "routing_entropy": self._routing_entropy,
                "selected_route_rms": route.float().square().mean().sqrt().detach(),
                "route_prior_scale_mean": self.route_prior_scale.mean().detach(),
                "enumerate_all": torch.tensor(
                    float(metadata.enumerate_all),
                    device=x.device,
                ),
            }
            if self.collect_routing_diagnostics:
                diagnostics.update(
                    _routing_quality_diagnostics(
                        query,
                        global_key,
                        representatives,
                        anchor_logits,
                        metadata,
                        local_window=self.local_window,
                        max_queries_per_batch=self.diagnostic_max_queries,
                    )
                )
            self._routing_diagnostics = diagnostics

        use_triton = self.backend == "triton" or (
            self.backend == "auto" and x.is_cuda and _TRITON_AVAILABLE
        )
        split_lanes = self.local_backend == "flex" and (
            _FLEX_ATTENTION_AVAILABLE or not x.is_cuda
        )
        if split_lanes:
            local_output, local_lse = self._local_lane(
                query, local_key, local_value, lengths
            )
            if use_triton and x.is_cuda and _TRITON_AVAILABLE:
                global_output, global_lse = _GlobalHISATritonFn.apply(
                    query,
                    global_key,
                    global_value,
                    route,
                    metadata.top_chunk_idx,
                    metadata.token_idx,
                    lengths,
                    self.chunk_size,
                    metadata.enumerate_all,
                    self.selector_tile_size,
                    self.local_window,
                    self.triton_block_q,
                    self.backward_impl == "atomic_masked",
                )
            else:
                global_output, global_lse = _eager_global_lane(
                    query, global_key, global_value, route, metadata,
                    local_window=self.local_window,
                )
            attended, combined_lse = _merge_attention_lanes(
                local_output, local_lse, global_output, global_lse
            )
            with torch.no_grad():
                valid_local = torch.isfinite(local_lse)
                valid_global = torch.isfinite(global_lse)
                maximum = torch.maximum(local_lse, global_lse)
                safe = torch.where(
                    valid_local | valid_global, maximum, torch.zeros_like(maximum)
                )
                local_weight = torch.where(
                    valid_local, torch.exp(local_lse - safe), torch.zeros_like(local_lse)
                )
                global_weight = torch.where(
                    valid_global, torch.exp(global_lse - safe), torch.zeros_like(global_lse)
                )
                denominator = local_weight + global_weight
                valid_rows = denominator > 0.0
                safe_denominator = denominator.clamp_min(1.0)
                valid_count = valid_rows.sum().clamp_min(1)
                self._routing_diagnostics["global_attention_mass"] = (
                    torch.where(
                        valid_rows, global_weight / safe_denominator,
                        torch.zeros_like(global_weight),
                    ).sum() / valid_count
                ).detach()
                self._routing_diagnostics["local_attention_mass"] = (
                    torch.where(
                        valid_rows, local_weight / safe_denominator,
                        torch.zeros_like(local_weight),
                    ).sum() / valid_count
                ).detach()
                self._routing_diagnostics["combined_lse_finite_rate"] = (
                    torch.isfinite(combined_lse).float().mean().detach()
                )
        elif use_triton and x.is_cuda and _TRITON_AVAILABLE:
            attended = _HISATritonFn.apply(
                query, local_key, local_value, global_key, global_value, route,
                metadata.top_chunk_idx, metadata.token_idx, lengths,
                self.chunk_size, metadata.enumerate_all, self.selector_tile_size,
                self.local_window, self.triton_block_q,
                self.backward_impl == "atomic_masked",
            )
        else:
            attended = _eager_attention(
                query, local_key, local_value, global_key, global_value, route,
                metadata, local_window=self.local_window,
            )
        merged = attended.permute(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            self.D,
        )
        output = self.W_o(merged) * torch.sigmoid(gate)
        auxiliary = self._routing_auxiliary_loss
        if auxiliary is None:
            auxiliary = output.sum() * 0.0
        if return_metadata and return_auxiliary:
            return output, metadata, auxiliary
        if return_metadata:
            return output, metadata
        if return_auxiliary:
            return output, auxiliary
        return output


HierarchicalSparseAttentionV17HISACausal = HierarchicalSparseAttentionV16HISACausal
HierarchicalSparseAttentionV18HISACausal = HierarchicalSparseAttentionV16HISACausal


if __name__ == "__main__":
    torch.manual_seed(7)
    module = HierarchicalSparseAttentionV16HISACausal(
        D=32,
        H=4,
        hd=8,
        chunk_size=8,
        top_k_chunks=2,
        hisa_top_m_tokens=8,
        local_window=8,
        selector_tile_size=4,
        backend="eager",
        route_aux_samples=2,
    ).double()
    values = torch.randn(2, 23, 32, dtype=torch.float64, requires_grad=True)
    output, metadata, auxiliary = module(
        values,
        return_metadata=True,
        return_auxiliary=True,
    )
    (output.square().mean() + auxiliary).backward()
    assert torch.isfinite(values.grad).all()
    module.eval()
    with torch.no_grad():
        baseline = module(values.detach())
        changed = values.detach().clone()
        changed[:, 15:] += torch.randn_like(changed[:, 15:]) * 10
        perturbed = module(changed)
        assert torch.equal(baseline[:, :15], perturbed[:, :15])
    print("CPU HISA forward/backward/causality smoke test: PASS")
