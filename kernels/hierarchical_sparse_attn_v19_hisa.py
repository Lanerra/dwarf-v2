"""Strict-causal hierarchical sparse attention used by DWARF models.

The module combines a local causal lane with token-routed completed chunks.
FlexAttention implements the local lane on CUDA, and Triton implements the
irregular global lane. A PyTorch reference path supports CPU execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


_COMPILED_FLEX_ATTENTION = (
    torch.compile(flex_attention, mode="default", dynamic=False)
    if _FLEX_ATTENTION_AVAILABLE
    else None
)


@torch.compiler.disable
def _isolated_flex_attention_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlexAttention outside the surrounding compiled autograd graph."""
    if _COMPILED_FLEX_ATTENTION is None:
        raise RuntimeError("compiled FlexAttention callable is unavailable")
    if AuxRequest is not None:
        output, auxiliary = _COMPILED_FLEX_ATTENTION(
            query,
            key,
            value,
            block_mask=block_mask,
            return_aux=AuxRequest(lse=True),
        )
        lse = auxiliary.lse
    else:
        output, lse = _COMPILED_FLEX_ATTENTION(
            query,
            key,
            value,
            block_mask=block_mask,
            return_lse=True,
        )
    return output, lse


class _StableRMSNormalizeFn(torch.autograd.Function):
    """FP32 RMS normalization with an explicit backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, eps: float) -> torch.Tensor:
        inverse_rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + float(eps))
        ctx.save_for_backward(value, inverse_rms)
        return value * inverse_rms

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        value, inverse_rms = ctx.saved_tensors
        gradient = grad_output.float()
        projection = (gradient * value).mean(-1, keepdim=True)
        grad_value = gradient * inverse_rms - value * inverse_rms.pow(3) * projection
        return grad_value.to(value.dtype), None


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only development
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def hisa_runtime_capabilities() -> dict[str, bool]:
    """Return availability of the production HISA execution dependencies."""
    return {
        "flex_attention": bool(_FLEX_ATTENTION_AVAILABLE),
        "triton": bool(_TRITON_AVAILABLE),
    }


def _next_pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _representative_mix_shift(heads: int, target_mean: float) -> float:
    """Shift the canonical head spread to a requested initial sigmoid mean."""
    target = min(max(float(target_mean), 1e-4), 1.0 - 1e-4)
    if target == 0.5:
        return 0.0
    base = torch.linspace(-2.0, 2.0, int(heads), dtype=torch.float64)
    lower, upper = -30.0, 30.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        observed = float(torch.sigmoid(base + midpoint).mean())
        if observed < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _resolve_attention_execution(
    *,
    backend: str,
    is_cuda: bool,
    triton_available: bool,
    flex_available: bool,
) -> bool:
    """Resolve the global attention backend."""
    if is_cuda and not flex_available:
        raise RuntimeError(
            "HISA requires FlexAttention for CUDA execution"
        )
    if backend == "triton":
        if not is_cuda:
            raise RuntimeError(
                "HISA backend='triton' requires CUDA; use backend='eager' "
                "explicitly for the CPU reference path"
            )
        if not triton_available:
            raise RuntimeError(
                "HISA backend='triton' was requested, but Triton is unavailable"
            )
        use_triton = True
    else:
        use_triton = backend == "auto" and is_cuda and triton_available
    return use_triton


def _validate_triton_geometry(
    *,
    head_dim: int,
    local_window: int,
    selected_tokens_per_chunk: int,
    block_q: int,
) -> None:
    """Fail before metadata/projection work when resolved Triton geometry is invalid."""
    if head_dim < 16 or not _is_power_of_two(head_dim):
        raise ValueError(
            "Triton HISA requires a power-of-two head dimension >=16"
        )
    if block_q < 16 or not _is_power_of_two(block_q):
        raise ValueError("HISA BLOCK_Q must be a power of two >=16")
    if max(16, _next_pow2(selected_tokens_per_chunk)) > 256:
        raise ValueError(
            "Triton HISA supports at most 256 selected tokens per chunk"
        )

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


@torch.no_grad()
def _rotation_diagnostics(
    x: torch.Tensor,
    delta: torch.Tensor,
    theta_h: torch.Tensor,
    *,
    label: str,
    strength_tau: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Measure the actual tokenwise NPCI rotation on diagnostic forwards only."""
    xf, df = x.float(), delta.float()
    norm = xf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = xf / norm
    perpendicular = df - (df * unit).sum(-1, keepdim=True) * unit
    perpendicular_norm = perpendicular.norm(dim=-1, keepdim=True)
    strength = torch.tanh(
        perpendicular_norm / (float(strength_tau) * norm + 1e-12)
    ).squeeze(-1)
    physical_abs = theta_h.detach().float().abs()
    actual_abs = physical_abs.reshape(1, -1, 1) * strength
    metrics = {
        f"npci_{label}_actual_abs_angle_p50": torch.quantile(actual_abs, 0.50),
        f"npci_{label}_actual_abs_angle_p90": torch.quantile(actual_abs, 0.90),
        f"npci_{label}_actual_abs_angle_p99": torch.quantile(actual_abs, 0.99),
        f"npci_{label}_strength_saturation_fraction": (strength > 0.9).float().mean(),
    }
    for head, value in enumerate(physical_abs):
        metrics[f"npci_{label}_physical_abs_theta_head{head:02d}"] = value
    return {name: value.detach() for name, value in metrics.items()}


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
    query_chunk_idx: torch.Tensor | None = None  # int32 [B,H,N,K] for token routing


@dataclass(frozen=True)
class HISASelectionCapture:
    """Ephemeral routing evidence produced only on explicit metadata requests."""
    anchor_logits: torch.Tensor
    metadata: HISAMetadata
    auxiliary_loss: torch.Tensor
    sampled_anchor_logits: torch.Tensor | None = None
    sampled_tile_ids: torch.Tensor | None = None


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
    blend_alpha: float | torch.Tensor,
) -> torch.Tensor:
    chunks, token_valid, chunk_valid = _chunk_tensors(key, chunk_size, valid_lengths)
    values = chunks.float()
    directions = F.normalize(values, dim=-1, eps=1e-6)
    valid_float = token_valid.unsqueeze(-1).to(values.dtype)
    # Average normalized token directions for cosine-based routing.
    mean = (directions * valid_float).sum(3) / valid_float.sum(3).clamp_min(1.0)
    mean = F.normalize(mean, dim=-1, eps=1e-6)
    energy = values.square().sum(-1).masked_fill(~token_valid, float("-inf"))
    best = energy.argmax(-1, keepdim=True)
    max_vector = torch.gather(
        directions,
        3,
        best.unsqueeze(-1).expand(-1, -1, -1, 1, values.shape[-1]),
    ).squeeze(3)
    if torch.is_tensor(blend_alpha):
        mixture = blend_alpha.to(device=values.device, dtype=values.dtype).reshape(
            1, values.shape[1], 1, 1
        )
    else:
        mixture = float(blend_alpha)
    representative = (1.0 - mixture) * mean + mixture * max_vector
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
    local_window: int,
) -> torch.Tensor:
    chunk_starts = torch.arange(
        num_chunks,
        device=tile_starts.device,
        dtype=torch.int32,
    ) * chunk_size
    chunk_ends = chunk_starts + chunk_size
    tile_valid = tile_starts.reshape(1, -1) < valid_lengths.reshape(-1, 1)
    chunk_valid = chunk_starts.reshape(1, -1) < valid_lengths.reshape(-1, 1)
    # Route only chunks that are fully outside the local lane.
    globally_accessible_end = tile_starts - int(local_window)
    completed_and_global = (
        chunk_ends.reshape(1, 1, -1)
        <= globally_accessible_end.reshape(1, -1, 1)
    )
    return completed_and_global & chunk_valid[:, None, :] & tile_valid[:, :, None]


def _inject_exploration_slot(
    indices: torch.Tensor,
    valid: torch.Tensor,
    eligible: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the final route with a uniform unseen eligible chunk.

    Causal chunk eligibility is always a contiguous prefix. Sampling a rank in
    that prefix and skipping the at-most-K selected IDs avoids constructing the
    former ``[B,H,T,C]`` count, candidate, and random-score tensors.
    """
    if probability <= 0.0 or indices.shape[-1] == 0:
        return indices, valid
    batch_size, heads, tiles, slots = indices.shape
    eligible_count = eligible.sum(-1, dtype=torch.int64)[:, None].expand(
        batch_size, heads, tiles
    )
    selected_count = valid.sum(-1, dtype=torch.int64)
    unseen_count = eligible_count - selected_count
    has_candidate = unseen_count > 0

    selected_sorted = torch.where(
        valid,
        indices.to(torch.int64),
        eligible_count[..., None],
    ).sort(dim=-1).values
    random_rank = torch.floor(
        torch.rand(batch_size, heads, tiles, device=indices.device)
        * unseen_count.clamp_min(1).to(torch.float32)
    ).to(torch.int64)
    choice = random_rank
    # Map the rank in the unseen set back into the contiguous eligible prefix.
    # K is a fixed, very small architecture constant, so this loop has stable
    # shape and O(B*H*T*K) memory/work.
    for slot in range(slots):
        selected_id = selected_sorted[..., slot]
        choice = choice + (selected_id <= choice).to(choice.dtype)

    replace = (
        torch.rand(batch_size, heads, tiles, device=indices.device) < probability
    ) & has_candidate
    result_indices = indices.clone()
    result_valid = valid.clone()
    result_indices[..., slots - 1] = torch.where(
        replace,
        choice.to(result_indices.dtype),
        result_indices[..., slots - 1],
    )
    result_valid[..., slots - 1] = torch.where(
        replace,
        torch.ones_like(result_valid[..., slots - 1]),
        result_valid[..., slots - 1],
    )
    return result_indices, result_valid


def _build_causal_tile_metadata(
    anchor_logits: torch.Tensor,
    *,
    chunk_size: int,
    top_k_chunks: int,
    selector_tile_size: int,
    local_window: int,
    valid_lengths: torch.Tensor,
    exploration_probability: float,
    preserve_anchor_logits: bool = True,
) -> HISAMetadata:
    _, _, tiles, num_chunks = anchor_logits.shape
    k_slots = min(top_k_chunks, num_chunks)
    tile_starts = torch.arange(
        tiles,
        device=anchor_logits.device,
        dtype=torch.int32,
    ) * selector_tile_size
    eligible = _eligibility(
        tile_starts,
        num_chunks,
        chunk_size,
        valid_lengths,
        local_window,
    )
    with torch.no_grad():
        detached_logits = anchor_logits.detach()
        masked_logits = (
            detached_logits.masked_fill(~eligible[:, None], float("-inf"))
            if preserve_anchor_logits
            else detached_logits.masked_fill_(~eligible[:, None], float("-inf"))
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
        token_idx = torch.empty(0, dtype=torch.int32, device=anchor_logits.device)
        token_scores = torch.empty(0, dtype=torch.float32, device=anchor_logits.device)
    return HISAMetadata(
        top_chunk_idx=top_chunks,
        token_idx=token_idx,
        token_scores=token_scores,
        tile_starts=tile_starts,
        valid_lengths=valid_lengths.detach(),
        chunk_size=int(chunk_size),
        selector_tile_size=int(selector_tile_size),
        enumerate_all=True,
    )


def _pack_token_metadata(
    token_metadata: HISAMetadata,
    *,
    pack_size: int,
) -> HISAMetadata:
    """Deduplicate exact per-token chunk choices into bounded physical packs.

    The semantic selection remains ``query_chunk_idx`` with K chunks per token.
    ``top_chunk_idx`` is only the physical union consumed by the tiled attention
    kernel; per-query route masking prevents a token from using another token's
    union member.
    """
    if pack_size < 1:
        raise ValueError("pack_size must be positive")
    if token_metadata.selector_tile_size != 1:
        raise ValueError("token metadata must have selector_tile_size=1")
    if not token_metadata.enumerate_all:
        raise ValueError("physical token packing requires enumerate_all metadata")

    token_chunks = token_metadata.top_chunk_idx
    batch_size, heads, seq_len, slots = token_chunks.shape
    tiles = math.ceil(seq_len / pack_size)
    padded_len = tiles * pack_size
    packed = F.pad(
        token_chunks,
        (0, 0, 0, padded_len - seq_len),
        value=-1,
    ).reshape(batch_size, heads, tiles, pack_size * slots)

    # Sorting followed by an integer rank compacts each fixed-width set without
    # host synchronization or a variable-size unique operation. The physical
    # width remains pack_size*K, giving Triton one stable launch shape.
    sentinel = token_chunks.clamp_min(0).amax().to(token_chunks.dtype) + 1
    sorted_chunks = torch.where(
        packed >= 0,
        packed,
        sentinel.expand_as(packed),
    ).sort(dim=-1).values
    unique = (sorted_chunks != sentinel) & torch.cat(
        (
            torch.ones_like(sorted_chunks[..., :1], dtype=torch.bool),
            sorted_chunks[..., 1:] != sorted_chunks[..., :-1],
        ),
        dim=-1,
    )
    ranks = unique.cumsum(-1) - 1
    union = sentinel.expand_as(sorted_chunks).clone()
    union.scatter_reduce_(
        -1,
        ranks.clamp_min(0),
        torch.where(unique, sorted_chunks, sentinel.expand_as(sorted_chunks)),
        reduce="amin",
        include_self=True,
    )
    union = torch.where(union == sentinel, torch.full_like(union, -1), union)
    tile_starts = (
        torch.arange(tiles, device=token_chunks.device, dtype=torch.int32)
        * pack_size
    )
    return HISAMetadata(
        top_chunk_idx=union.to(torch.int32),
        token_idx=token_metadata.token_idx,
        token_scores=token_metadata.token_scores,
        tile_starts=tile_starts,
        valid_lengths=token_metadata.valid_lengths,
        chunk_size=token_metadata.chunk_size,
        selector_tile_size=int(pack_size),
        enumerate_all=True,
        query_chunk_idx=token_chunks,
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


def _strict_local_key_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    distance = query_positions - key_positions
    return (distance >= 1) & (distance <= local_window)


def _strict_local_or_boundary_key_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    local_window: int,
    chunk_size: int,
) -> torch.Tensor:
    """Cover the one partial chunk excluded from semantic whole-chunk routing.

    Whole-chunk routing may only use chunks ending before ``q-local_window``.
    Unless that boundary is chunk aligned, a causal prefix of the next chunk
    would otherwise be in neither the local nor global lane.  Treat that prefix
    as a deterministic local bridge; it never consumes one of the semantic K
    route slots and it never overlaps a globally eligible whole chunk.
    """
    distance = query_positions - key_positions
    local = (distance >= 1) & (distance <= int(local_window))
    cutoff = query_positions - int(local_window)
    has_boundary = cutoff > 0
    boundary_start = (cutoff // int(chunk_size)) * int(chunk_size)
    boundary = (
        has_boundary
        & (key_positions >= boundary_start)
        & (key_positions < cutoff)
    )
    return local | boundary


def _strict_global_key_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    return query_positions - key_positions > local_window


def _selected_route_scores(
    query_normalized: torch.Tensor,
    representatives: torch.Tensor,
    metadata: HISAMetadata,
    route_scale_by_head: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Score each query's chunks within its deduplicated physical pack."""
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
    if metadata.query_chunk_idx is None:
        valid = (chunks >= 0)[..., None, :].expand_as(scores)
    else:
        semantic_chunks = F.pad(
            metadata.query_chunk_idx,
            (0, 0, 0, padded_len - seq_len),
            value=-1,
        ).reshape(batch_size, heads, tiles, selector, -1)
        valid = (
            semantic_chunks[..., None] == chunks[..., None, None, :]
        ).any(dim=-2) & (chunks[..., None, :] >= 0)
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    mean = (scores * valid).sum(-1, keepdim=True) / count
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    scaled = (
        centered
        * route_scale_by_head.reshape(1, heads, 1, 1, 1).float()
        / float(temperature)
    )
    scaled = torch.where(valid, scaled, torch.full_like(scaled, float("-inf")))
    return scaled.reshape(batch_size, heads, padded_len, -1)[:, :, :seq_len].to(
        query_normalized.dtype
    )


def _resolve_route_aux_tile_ids(
    metadata: HISAMetadata,
    *,
    samples: int,
    local_window: int,
    tile_ids: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    tiles = metadata.top_chunk_idx.shape[2]
    first_useful_tile = math.ceil(
        (metadata.chunk_size + int(local_window)) / metadata.selector_tile_size
    )
    candidate_count = max(0, tiles - first_useful_tile)
    sample_count = min(int(samples), candidate_count)
    if sample_count <= 0:
        return torch.empty(0, device=device, dtype=torch.int64)
    useful_tiles = torch.arange(
        first_useful_tile, tiles, device=device, dtype=torch.int64
    )
    if tile_ids is None:
        if sample_count == candidate_count:
            return useful_tiles
        permutation = torch.randperm(candidate_count, device=device)
        return useful_tiles[permutation[:sample_count]]

    if not torch.is_tensor(tile_ids):
        raise TypeError("route_aux_tile_ids must be a tensor")
    if tile_ids.ndim != 1:
        raise ValueError("route_aux_tile_ids must be one-dimensional")
    if tile_ids.dtype != torch.int64:
        raise TypeError("route_aux_tile_ids must use int64")
    if tile_ids.device != device:
        raise ValueError("route_aux_tile_ids must be on the HISA input device")
    if tile_ids.numel() != sample_count:
        raise ValueError(
            f"route_aux_tile_ids must contain exactly {sample_count} IDs"
        )
    # The canonical trainer creates these IDs from a deterministic validated
    # recipe. Avoid a data-dependent scalar guard inside its compiled graph.
    if torch.compiler.is_compiling():
        return tile_ids
    ordered = tile_ids.sort().values
    unique = (
        torch.ones((), dtype=torch.bool, device=device)
        if ordered.numel() <= 1
        else (ordered[1:] != ordered[:-1]).all()
    )
    valid_ids = (tile_ids >= first_useful_tile) & (tile_ids < tiles)
    valid = valid_ids.all() & unique
    message = "route_aux_tile_ids must be unique and in the eligible tile range"
    if tile_ids.is_cuda:
        torch._assert_async(valid, message)
    elif not bool(valid):
        if bool(((tile_ids < 0) | (tile_ids >= tiles)).any()):
            raise ValueError("route_aux_tile_ids contains an out-of-range ID")
        if bool((tile_ids < first_useful_tile).any()):
            raise ValueError("route_aux_tile_ids contains an ineligible early tile")
        raise ValueError("route_aux_tile_ids must be unique")
    return tile_ids


def _router_auxiliary_loss(
    anchor_query: torch.Tensor,
    representatives: torch.Tensor,
    global_key: torch.Tensor,
    metadata: HISAMetadata,
    route_scale_by_head: torch.Tensor,
    *,
    samples: int,
    target_temperature: float,
    routing_temperature: float,
    local_window: int,
    oracle_temperature: float,
    tile_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train sampled effective route priors toward token-level evidence.

    Only the sampled query rows participate in the differentiable selector
    matmul. Hard top-k selection still uses the detached full routing surface.
    """
    resolved_ids = _resolve_route_aux_tile_ids(
        metadata,
        samples=samples,
        local_window=local_window,
        tile_ids=tile_ids,
        device=anchor_query.device,
    )
    if resolved_ids.numel() == 0:
        empty = anchor_query.new_empty(
            anchor_query.shape[0], anchor_query.shape[1], 0, representatives.shape[2]
        )
        return anchor_query.new_zeros(()), empty, resolved_ids

    _, heads, _, _ = anchor_query.shape
    query = anchor_query[:, :, resolved_ids].float()
    sampled_anchor_logits = torch.matmul(
        query,
        representatives.float().transpose(-2, -1),
    ) / float(routing_temperature)
    sampled_tile_starts = metadata.tile_starts[resolved_ids]
    eligible = _eligibility(
        sampled_tile_starts,
        representatives.shape[2],
        metadata.chunk_size,
        metadata.valid_lengths,
        local_window,
    )

    # Oracle evidence is a detached target. Construct it without autograd so
    # token-level chunk evidence does not retain a second global-key graph.
    with torch.no_grad():
        key_chunks, token_valid, _ = _chunk_tensors(
            global_key,
            metadata.chunk_size,
            metadata.valid_lengths,
        )
        key_directions = F.normalize(key_chunks.float(), dim=-1, eps=1e-6)
        token_scores = torch.einsum(
            "bhsd,bhcmd->bhscm",
            query.detach(),
            key_directions,
        ) / float(oracle_temperature)
        mask = token_valid[:, :, None] & eligible[:, None, :, :, None]
        token_scores = token_scores.masked_fill(~mask, -1e9)
        token_count = mask.sum(-1).clamp_min(1).to(token_scores.dtype)
        oracle = float(oracle_temperature) * (
            torch.logsumexp(token_scores, dim=-1) - token_count.log()
        )
        target = torch.softmax(oracle / float(target_temperature), dim=-1)

    safe_anchor_logits = torch.where(
        eligible[:, None],
        sampled_anchor_logits,
        torch.zeros_like(sampled_anchor_logits),
    )
    route = (
        safe_anchor_logits
        * route_scale_by_head.reshape(1, heads, 1, 1).float()
    ).masked_fill(~eligible[:, None], -1e9)
    per_row = -(target * torch.log_softmax(route, dim=-1)).sum(-1)
    valid_rows = eligible.any(-1)[:, None].expand(-1, heads, -1)
    loss = (per_row * valid_rows).sum() / valid_rows.sum().clamp_min(1)
    return loss, sampled_anchor_logits, resolved_ids


def _eligible_route_entropy(
    anchor_logits: torch.Tensor,
    metadata: HISAMetadata,
    local_window: int,
    *,
    tile_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = anchor_logits if tile_ids is None else anchor_logits[:, :, tile_ids]
    tile_starts = (
        metadata.tile_starts
        if tile_ids is None
        else metadata.tile_starts[tile_ids]
    )
    eligible = _eligibility(
        tile_starts,
        logits.shape[-1],
        metadata.chunk_size,
        metadata.valid_lengths,
        local_window,
    )
    masked = logits.float().masked_fill(~eligible[:, None], -1e9)
    probability = torch.softmax(masked, dim=-1)
    entropy = -(probability * torch.log_softmax(masked, dim=-1)).sum(-1)
    valid = eligible.any(-1)[:, None].expand_as(entropy)
    return (entropy * valid).sum() / valid.sum().clamp_min(1)


def _eager_local_lane(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    local_window: int,
    chunk_size: int,
    boundary_bridge: bool,
    selector_tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict sliding-window lane returning normalized output and natural LSE."""
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, heads, seq_len), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    lane_span = local_window + chunk_size - 1 if boundary_bridge else local_window
    offsets = torch.arange(lane_span, device=query.device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    for start in range(0, seq_len, selector_tile_size):
        end = min(start + selector_tile_size, seq_len)
        positions = torch.arange(start, end, device=query.device, dtype=torch.int32)
        q = query[:, :, start:end]
        q_valid = positions.reshape(1, -1) < valid_lengths.reshape(batch_size, 1)
        ids = positions[:, None] - lane_span + offsets[None]
        lane_mask = (
            _strict_local_or_boundary_key_mask(
                positions[:, None], ids, local_window, chunk_size
            )
            if boundary_bridge
            else _strict_local_key_mask(positions[:, None], ids, local_window)
        )
        valid = (
            (ids >= 0)
            & lane_mask
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
            & torch.isfinite(prior)
            & _strict_global_key_mask(
                positions.reshape(1, 1, -1, 1),
                ids[:, :, None],
                local_window,
            )
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


class _MergeAttentionLanesFn(torch.autograd.Function):
    """Exact two-lane merge with an explicit FP32 backward."""

    @staticmethod
    def forward(ctx, local_output, local_lse, global_output, global_lse):
        local_valid = torch.isfinite(local_lse)
        global_valid = torch.isfinite(global_lse)
        both_valid = local_valid & global_valid
        local_only = local_valid & ~global_valid
        global_only = global_valid & ~local_valid

        global_weight = torch.where(
            both_valid,
            torch.sigmoid(global_lse.float() - local_lse.float()),
            global_only.to(torch.float32),
        )
        local_weight = torch.where(
            both_valid,
            1.0 - global_weight,
            local_only.to(torch.float32),
        )
        output_float = (
            local_weight[..., None] * local_output.float()
            + global_weight[..., None] * global_output.float()
        )
        output = output_float.to(local_output.dtype)
        combined_lse = torch.where(
            both_valid,
            torch.logaddexp(local_lse.float(), global_lse.float()),
            torch.where(
                local_only,
                local_lse.float(),
                torch.where(
                    global_only,
                    global_lse.float(),
                    torch.full_like(local_lse.float(), float("-inf")),
                ),
            ),
        )
        ctx.save_for_backward(
            local_output,
            global_output,
            local_weight,
            global_weight,
        )
        ctx.local_lse_dtype = local_lse.dtype
        ctx.global_lse_dtype = global_lse.dtype
        return output, combined_lse, global_weight

    @staticmethod
    def backward(ctx, grad_output, grad_combined_lse, grad_global_weight):
        local_output, global_output, local_weight, global_weight = ctx.saved_tensors
        if grad_output is None:
            grad_output_float = torch.zeros_like(local_output, dtype=torch.float32)
        else:
            grad_output_float = grad_output.float()
        if grad_combined_lse is None:
            grad_combined_float = torch.zeros_like(local_weight)
        else:
            grad_combined_float = grad_combined_lse.float()
        if grad_global_weight is None:
            grad_global_weight_float = torch.zeros_like(global_weight)
        else:
            grad_global_weight_float = grad_global_weight.float()

        output_float = (
            local_weight[..., None] * local_output.float()
            + global_weight[..., None] * global_output.float()
        )
        grad_local_output = (
            grad_output_float * local_weight[..., None]
        ).to(local_output.dtype)
        grad_global_output = (
            grad_output_float * global_weight[..., None]
        ).to(global_output.dtype)
        grad_local_lse = (
            grad_output_float
            * local_weight[..., None]
            * (local_output.float() - output_float)
        ).sum(-1) + grad_combined_float * local_weight
        grad_global_lse = (
            grad_output_float
            * global_weight[..., None]
            * (global_output.float() - output_float)
        ).sum(-1) + grad_combined_float * global_weight
        mass_gradient = (
            grad_global_weight_float * local_weight * global_weight
        )
        grad_local_lse = grad_local_lse - mass_gradient
        grad_global_lse = grad_global_lse + mass_gradient
        return (
            grad_local_output,
            grad_local_lse.to(ctx.local_lse_dtype),
            grad_global_output,
            grad_global_lse.to(ctx.global_lse_dtype),
        )


def _merge_attention_lanes(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    global_output: torch.Tensor,
    global_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two independently normalized attention lanes."""
    output, combined_lse, _ = _MergeAttentionLanesFn.apply(
        local_output,
        local_lse,
        global_output,
        global_lse,
    )
    return output, combined_lse


def _merge_attention_lanes_with_mass(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    global_output: torch.Tensor,
    global_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge lanes and return exact global softmax mass."""
    return _MergeAttentionLanesFn.apply(
        local_output,
        local_lse,
        global_output,
        global_lse,
    )



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
                & (prior[:, None] > float("-inf"))
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
        BLOCK_Q: tl.constexpr,
        MASK_ATOMICS: tl.constexpr,
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
                & (prior[:, None] > float("-inf"))
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
            batch_size * heads,
            chunks.shape[2],
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
            BLOCK_Q=block_q,
            num_warps=4,
            num_stages=2,
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
        # Absent global rows have zero gradient contribution.
        lane_valid = torch.isfinite(lse)
        grad_output = torch.where(
            lane_valid[..., None],
            grad_output,
            torch.zeros_like(grad_output),
        ).contiguous()
        grad_lse = torch.where(
            lane_valid & torch.isfinite(grad_lse),
            grad_lse,
            torch.zeros_like(grad_lse),
        ).contiguous()
        dquery = torch.zeros_like(query, dtype=torch.float32)
        dglobal_key = torch.zeros_like(global_key, dtype=torch.float32)
        dglobal_value = torch.zeros_like(global_value, dtype=torch.float32)
        droute = torch.zeros_like(route, dtype=torch.float32)
        m_slots = ctx.chunk_size if ctx.enumerate_all else storage.shape[-1]
        m_pad = max(16, _next_pow2(m_slots))
        id_strides = storage.stride() if storage.ndim == 5 else (0, 0, 0, 0, 0)
        grid = (
            batch_size * heads,
            chunks.shape[2],
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
            MASK_ATOMICS=ctx.mask_atomics,
            BLOCK_Q=ctx.block_q,
            num_warps=4,
            num_stages=2,
        )
        return (
            dquery.to(query.dtype),
            dglobal_key.to(global_key.dtype),
            dglobal_value.to(global_value.dtype),
            droute.to(route.dtype),
            None, None, None, None, None, None, None, None, None,
        )


@torch.compiler.disable
def _global_hisa_triton_apply(*args):
    """Keep the custom Triton autograd boundary outside AOTAutograd."""
    return _GlobalHISATritonFn.apply(*args)


class HierarchicalSparseAttentionV19HISACausal(nn.Module):
    """Strict-causal local and routed-chunk attention for DWARF models."""

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int | None = None,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int | None = None,
        *,
        chunk_size: int | None = None,
        local_window: int | None = None,
        selector_tile_size: int | None = None,
        temperature: float = 1.0,
        route_prior_scale: float = 0.1,
        route_prior_max_scale: float = 2.0,
        global_lane_bias_limit: float = 4.0,
        backend: str | None = None,
        token_selection_mode: str | None = None,
        chunk_selection_scope: str | None = None,
        token_routing_pack_size: int | None = None,
        representative_mode: str = "mean_max_blend",
        representative_blend_alpha: float = 0.5,
        exploration_probability: float = 0.05,
        route_aux_weight: float = 0.01,
        route_aux_samples: int = 4,
        route_aux_temperature: float = 1.0,
        route_aux_oracle_temperature: float = 0.2,
        global_adapter_rank: int = 16,
        binding_rank: int = 64,
        npci_theta_max: float = 0.25,
        max_seq_len: int | None = None,
        local_backend: str = "flex",
        boundary_bridge: bool = True,
        triton_block_q: int | None = None,
        backward_impl: str | None = None,
        collect_routing_diagnostics: bool | None = None,
        diagnostic_max_queries: int | None = None,
        route_from_base_global_key: bool = False,
    ) -> None:
        super().__init__()
        D, H, hd = int(D), int(H), int(hd)
        if D < 1 or H < 1 or hd < 1:
            raise ValueError("D, H, and hd must be positive")
        if D != H * hd:
            raise ValueError(f"D={D} must equal H*hd={H * hd}")
        if top_k_chunks < 1:
            raise ValueError("top_k_chunks must be positive")
        if hisa_top_m_tokens is not None and int(hisa_top_m_tokens) < 1:
            raise ValueError("hisa_top_m_tokens must be positive when supplied")
        if num_chunks is not None and int(num_chunks) < 1:
            raise ValueError("num_chunks must be positive when supplied")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(route_prior_scale) or route_prior_scale <= 0:
            raise ValueError("route_prior_scale must be finite and positive")
        if (
            not math.isfinite(route_prior_max_scale)
            or route_prior_max_scale <= route_prior_scale
        ):
            raise ValueError(
                "route_prior_max_scale must be finite and exceed route_prior_scale"
            )
        if not math.isfinite(global_lane_bias_limit) or global_lane_bias_limit <= 0:
            raise ValueError("global_lane_bias_limit must be finite and positive")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0,1]")
        if route_aux_weight < 0 or route_aux_samples < 0:
            raise ValueError("route auxiliary weight/samples must be non-negative")
        if not math.isfinite(route_aux_temperature) or route_aux_temperature <= 0:
            raise ValueError("route_aux_temperature must be finite and positive")
        if (
            not math.isfinite(route_aux_oracle_temperature)
            or route_aux_oracle_temperature <= 0
        ):
            raise ValueError(
                "route_aux_oracle_temperature must be finite and positive"
            )
        if representative_mode != "mean_max_blend":
            raise ValueError("representative_mode must be 'mean_max_blend'")
        if not 0.0 <= representative_blend_alpha <= 1.0:
            raise ValueError("representative_blend_alpha must be in [0,1]")
        if global_adapter_rank < 0:
            raise ValueError("global_adapter_rank must be non-negative")
        if binding_rank < 0:
            raise ValueError("binding_rank must be non-negative")
        if not math.isfinite(npci_theta_max) or npci_theta_max <= 0:
            raise ValueError("npci_theta_max must be finite and positive")
        local_backend = str(local_backend).lower()
        if local_backend != "flex":
            raise ValueError("local_backend must be 'flex'")
        if not isinstance(boundary_bridge, bool):
            raise TypeError("boundary_bridge must be bool")
        if not boundary_bridge:
            raise ValueError("boundary_bridge must be enabled")
        if not isinstance(route_from_base_global_key, bool):
            raise TypeError("route_from_base_global_key must be bool")
        if max_seq_len is not None and int(max_seq_len) < 1:
            raise ValueError("max_seq_len must be positive when supplied")

        resolved_chunk_size = (
            int(chunk_size)
            if chunk_size is not None
            else 64
        )
        resolved_local_window = (
            int(local_window)
            if local_window is not None
            else 64
        )
        resolved_selector_tile = (
            int(selector_tile_size)
            if selector_tile_size is not None
            else 16
        )
        if (
            resolved_chunk_size < 1
            or resolved_local_window < 1
            or resolved_selector_tile < 1
        ):
            raise ValueError(
                "chunk_size, local_window, and selector_tile_size must be positive"
            )
        resolved_top_m = (
            resolved_chunk_size
            if hisa_top_m_tokens is None
            else int(hisa_top_m_tokens)
        )
        if resolved_top_m != resolved_chunk_size:
            raise ValueError(
                "HISA V19 enumerates complete selected chunks; "
                "hisa_top_m_tokens must equal chunk_size"
            )
        if num_chunks is not None and max_seq_len is not None:
            expected_chunks = math.ceil(int(max_seq_len) / resolved_chunk_size)
            if int(num_chunks) != expected_chunks:
                raise ValueError(
                    "num_chunks is a compatibility hint and must equal "
                    "ceil(max_seq_len/chunk_size) when both are supplied"
                )

        self.D = D
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.num_chunks_compatibility_hint = (
            None if num_chunks is None else int(num_chunks)
        )
        self.chunk_size = resolved_chunk_size
        self.top_k_chunks = int(top_k_chunks)
        self.hisa_top_m_tokens = resolved_top_m
        self.local_window = resolved_local_window
        self.selector_tile_size = resolved_selector_tile
        self.temperature = float(temperature)
        self.backend = (backend or "auto").lower()
        if self.backend not in {"auto", "eager", "triton"}:
            raise ValueError("backend must be auto, eager, or triton")
        self.token_selection_mode = (
            token_selection_mode
            or "auto"
        ).lower()
        if self.token_selection_mode != "auto":
            raise ValueError("token_selection_mode must be 'auto'")
        self.chunk_selection_scope = (
            chunk_selection_scope
            or "token"
        ).lower()
        if self.chunk_selection_scope != "token":
            raise ValueError("chunk_selection_scope must be 'token'")
        self.token_routing_pack_size = int(
            token_routing_pack_size
            if token_routing_pack_size is not None
            else 4
        )
        if (
            self.token_routing_pack_size < 1
            or self.token_routing_pack_size > 16
            or not _is_power_of_two(self.token_routing_pack_size)
        ):
            raise ValueError(
                "token_routing_pack_size must be a power of two in [1,16]"
            )
        self.representative_mode = representative_mode
        self.representative_blend_alpha = float(representative_blend_alpha)
        self.exploration_probability = float(exploration_probability)
        self.route_aux_weight = float(route_aux_weight)
        self.route_aux_samples = int(route_aux_samples)
        self.route_aux_temperature = float(route_aux_temperature)
        self.route_aux_oracle_temperature = float(route_aux_oracle_temperature)
        self.route_prior_max_scale = float(route_prior_max_scale)
        self.global_lane_bias_limit = float(global_lane_bias_limit)
        self.global_adapter_rank = int(global_adapter_rank)
        self.binding_rank = int(binding_rank)
        self.npci_theta_max = float(npci_theta_max)
        self.route_from_base_global_key = route_from_base_global_key
        self.route_source = (
            "base_global_k" if route_from_base_global_key else "rotated_global_k"
        )
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.local_backend = local_backend
        self.boundary_bridge = boundary_bridge
        self._local_block_mask = None
        self._local_block_mask_key: (
            tuple[str, int | None, int, int, int, bool] | None
        ) = None
        self.triton_block_q = int(
            triton_block_q
            if triton_block_q is not None
            else 16
        )
        self.backward_impl = (
            backward_impl
            if backward_impl is not None
            else "atomic_masked"
        ).lower()
        if self.backward_impl not in {"atomic", "atomic_masked"}:
            raise ValueError("backward_impl must be atomic or atomic_masked")
        if collect_routing_diagnostics is None:
            collect_routing_diagnostics = False
        if not isinstance(collect_routing_diagnostics, bool):
            raise TypeError("collect_routing_diagnostics must be bool")
        self.collect_routing_diagnostics = collect_routing_diagnostics
        self.diagnostic_max_queries = (
            8 if diagnostic_max_queries is None else int(diagnostic_max_queries)
        )
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
            nn.init.normal_(self.global_k_up.weight, mean=0.0, std=0.002)
            nn.init.normal_(self.global_v_up.weight, mean=0.0, std=0.002)
        else:
            self.global_k_down = None
            self.global_k_up = None
            self.global_v_down = None
            self.global_v_up = None
        initial_route_fraction = route_prior_scale / route_prior_max_scale
        initial_route_raw = math.log(
            initial_route_fraction / (1.0 - initial_route_fraction)
        )
        self.route_prior_raw = nn.Parameter(torch.full((H,), initial_route_raw))
        # Preserve the canonical per-head spread while making the public alpha
        # control the exact initial mean blend. The canonical alpha=0.5 uses an
        # exact zero shift and therefore retains the released state fingerprint.
        mix_shift = _representative_mix_shift(H, self.representative_blend_alpha)
        self.representative_mix_raw = nn.Parameter(
            torch.linspace(-2.0, 2.0, H) + mix_shift
        )
        neutral_global_bias = 0.0
        self.global_lane_logit_bias = nn.Parameter(
            torch.full((H,), neutral_global_bias)
        )
        if self.binding_rank:
            self.bind_query = nn.Linear(D, self.binding_rank, bias=False)
            self.bind_evidence = nn.Linear(D, self.binding_rank, bias=False)
            self.bind_output = nn.Linear(self.binding_rank, D, bias=False)
            self.binding_gain_raw = nn.Parameter(torch.tensor(-2.1972246))
        else:
            self.bind_query = None
            self.bind_evidence = None
            self.bind_output = None
            self.register_parameter("binding_gain_raw", None)
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
        return self.route_prior_max_scale * torch.sigmoid(self.route_prior_raw)

    @property
    def representative_mix(self) -> torch.Tensor:
        return torch.sigmoid(self.representative_mix_raw)

    @property
    def bounded_global_lane_logit_bias(self) -> torch.Tensor:
        limit = float(self.global_lane_bias_limit)
        return limit * torch.tanh(self.global_lane_logit_bias / limit)

    def routing_auxiliary_loss(self, *, clear: bool = False) -> torch.Tensor | None:
        value = self._routing_auxiliary_loss
        if clear:
            self._routing_auxiliary_loss = None
        return value

    def semantic_config(self) -> dict[str, object]:
        return {
            "implementation": "hisa-v19-accessible-routes-binding",
            "chunk_size": self.chunk_size,
            "num_chunks_compatibility_hint": self.num_chunks_compatibility_hint,
            "top_k_chunks": self.top_k_chunks,
            "selected_tokens_per_chunk": self.hisa_top_m_tokens,
            "selected_token_policy": "enumerate_complete_selected_chunks",
            "local_window": self.local_window,
            "boundary_bridge": self.boundary_bridge,
            "semantic_selection_scope": "per_token",
            "reference_lane_tile_size": self.selector_tile_size,
            "routing_temperature": self.temperature,
            "routing_temperature_applies_to": (
                "selected_route_priors_and_sampled_router_auxiliary"
            ),
            "routing_key_source": self.route_source,
            "global_attention_key_source": "post_packet_rotated_global_k",
            "router_auxiliary_oracle_key_source": self.route_source,
            "representative_mode": self.representative_mode,
            "representative_blend_alpha_initial_mean": (
                self.representative_blend_alpha
            ),
            "exploration_probability": self.exploration_probability,
            "exploration_policy": "uniform_unseen_eligible_chunk",
            "route_aux_weight": self.route_aux_weight,
            "route_aux_samples": self.route_aux_samples,
            "route_aux_temperature": self.route_aux_temperature,
            "route_aux_oracle_temperature": self.route_aux_oracle_temperature,
            "route_auxiliary_target": "sampled_effective_route_prior",
            "route_prior_max_scale": self.route_prior_max_scale,
            "global_lane_bias_limit": self.global_lane_bias_limit,
            "global_adapter_rank": self.global_adapter_rank,
            "binding_rank": self.binding_rank,
            "binding_source": "global_lane_weighted_by_exact_merged_mass",
            "representative_mix": "learned_per_head_normalized_mean_max",
            "global_lane_confidence": "learned_per_head_common_logit",
            "npci_theta_max": self.npci_theta_max,
            "token_selection_mode": self.token_selection_mode,
            "chunk_selection_scope": self.chunk_selection_scope,
            "token_routing_pack_size": self.token_routing_pack_size,
            "selected_route_scoring": "token_selection_physical_union",
            "lane_merge": "exact_lse",
            "selector_complexity": "O(N*ceil(N/chunk_size)*D)",
            "incremental_kv_cache": "not_implemented_by_this_module",
            "max_seq_len": self.max_seq_len,
        }

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "local_backend": self.local_backend,
            "triton_block_q": self.triton_block_q,
            "backward_impl": self.backward_impl,
            "collect_routing_diagnostics": self.collect_routing_diagnostics,
            "diagnostic_max_queries": self.diagnostic_max_queries,
            "selector_autograd": "sampled_auxiliary_rows_only",
            "hard_selector": "detached_full_token_chunk_surface",
            "exploration_workspace": "O(B*H*N*K)_beyond_hard_selector",
            "flex_attention_available": _FLEX_ATTENTION_AVAILABLE,
            "triton_available": _TRITON_AVAILABLE,
            "flex_lse_api": "aux_request" if AuxRequest is not None else "return_lse",
        }

    def _ensure_local_block_mask(
        self, device: torch.device, seq_len: int
    ):
        if not _FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
        key = (
            device.type,
            device.index,
            int(seq_len),
            self.local_window,
            self.chunk_size,
            self.boundary_bridge,
        )
        if self._local_block_mask is not None and self._local_block_mask_key == key:
            return self._local_block_mask
        window = int(self.local_window)
        chunk = int(self.chunk_size)
        boundary_bridge = bool(self.boundary_bridge)

        def local_mask(_batch, _head, query_index, key_index):
            # FlexAttention sees query[1:], so map its compact query index back
            # to the model sequence.  This keeps every Flex row non-empty while
            # preserving exact strict causality; model position zero is restored
            # explicitly as a zero/-inf row below.
            query_index = query_index + 1
            if boundary_bridge:
                return _strict_local_or_boundary_key_mask(
                    query_index, key_index, window, chunk
                )
            return _strict_local_key_mask(query_index, key_index, window)

        self._local_block_mask = create_block_mask(
            local_mask,
            B=None,
            H=None,
            Q_LEN=max(1, int(seq_len) - 1),
            KV_LEN=int(seq_len),
            device=device,
            BLOCK_SIZE=128,
        )
        self._local_block_mask_key = key
        return self._local_block_mask

    def prepare_runtime(
        self, device: torch.device | str, seq_len: int | None = None
    ) -> None:
        """Prebuild non-state runtime metadata before compilation."""
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
            if query.shape[2] == 1:
                return (
                    torch.zeros_like(query),
                    torch.full(
                        query.shape[:3],
                        float("-inf"),
                        device=query.device,
                        dtype=torch.float32,
                    ),
                )
            block_mask = self._ensure_local_block_mask(query.device, query.shape[2])
            output_tail, lse_tail = _isolated_flex_attention_lse(
                query[:, :, 1:],
                local_key,
                local_value,
                block_mask,
            )
            output = torch.cat(
                (torch.zeros_like(query[:, :, :1]), output_tail), dim=2
            )
            lse = torch.cat(
                (
                    torch.full_like(lse_tail[:, :, :1], float("-inf")),
                    lse_tail,
                ),
                dim=2,
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
            chunk_size=self.chunk_size,
            boundary_bridge=self.boundary_bridge,
            selector_tile_size=self.selector_tile_size,
        )

    def reset_global_adapters_(self) -> None:
        if self.global_adapter_rank:
            if self.global_k_up is None or self.global_v_up is None:
                raise RuntimeError("global adapter projections are incomplete")
            # A small nonzero residual lets both adapter factors learn immediately.
            nn.init.normal_(self.global_k_up.weight, mean=0.0, std=0.002)
            nn.init.normal_(self.global_v_up.weight, mean=0.0, std=0.002)

    def _binding_correction(
        self,
        x: torch.Tensor,
        binding_evidence_heads: torch.Tensor,
        binding_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the query/evidence semantic binding correction."""
        if self.bind_query is None or self.bind_evidence is None or self.bind_output is None:
            raise RuntimeError("binding correction modules are incomplete")
        binding_evidence = binding_evidence_heads.permute(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.D
        )
        evidence_float = binding_evidence.float()
        evidence_normalized = _StableRMSNormalizeFn.apply(evidence_float, 1e-6)
        query_features = self.bind_query(x)
        evidence_features = self.bind_evidence(
            evidence_normalized.to(query_features.dtype)
        )
        interaction = F.silu(query_features) * evidence_features
        bound = self.bind_output(interaction)
        return (
            torch.sigmoid(self.binding_gain_raw).to(bound.dtype)
            * binding_confidence.to(bound.dtype)
            * bound
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_inject: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_metadata: bool = False,
        return_auxiliary: bool = False,
    ):
        compiling = torch.compiler.is_compiling()
        collect_diagnostics = bool(
            collect_diagnostics or self.collect_routing_diagnostics
        )
        emit_diagnostics = collect_diagnostics and not compiling
        if not compiling:
            self.hisa_evidence_capture = None
            self._routing_auxiliary_loss = None
            self._routing_diagnostics = {}
        rotation_diagnostics: dict[str, torch.Tensor] = {}
        route_comparison_diagnostics: dict[str, torch.Tensor] = {}

        use_triton = _resolve_attention_execution(
            backend=self.backend,
            is_cuda=x.is_cuda,
            triton_available=_TRITON_AVAILABLE,
            flex_available=_FLEX_ATTENTION_AVAILABLE,
        )
        if use_triton:
            _validate_triton_geometry(
                head_dim=self.hd,
                local_window=self.local_window,
                selected_tokens_per_chunk=self.chunk_size,
                block_q=self.triton_block_q,
            )

        batch_size, seq_len, _ = x.shape
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds configured HISA bound "
                f"{self.max_seq_len}"
            )
        lengths = _as_valid_lengths(
            valid_lengths
            if valid_lengths is not None
            else getattr(self, "_causal_control_valid_lengths", None),
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )
        query_flat, key_flat, value_flat, gate = self.qkvg_proj(x).split(
            self.D, dim=-1
        )
        query = _to_heads(query_flat, batch_size, seq_len, self.H, self.hd)
        local_key = _to_heads(key_flat, batch_size, seq_len, self.H, self.hd)
        local_value = _to_heads(value_flat, batch_size, seq_len, self.H, self.hd)
        global_key = local_key
        global_value = local_value
        if self.global_adapter_rank:
            if self.global_k_down is None or self.global_k_up is None:
                raise RuntimeError("global key adapter modules are incomplete")
            if self.global_v_down is None or self.global_v_up is None:
                raise RuntimeError("global value adapter modules are incomplete")
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
        base_global_key = global_key
        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            if (
                key_delta.shape != global_key.shape
                or value_delta.shape != global_value.shape
            ):
                raise ValueError("kv_inject must contain [B,H,N,HD] tensors")
            physical_theta_k = self.npci_theta_max * torch.tanh(self.npci_theta_k)
            physical_theta_v = self.npci_theta_max * torch.tanh(self.npci_theta_v)
            if emit_diagnostics:
                rotation_diagnostics.update(
                    _rotation_diagnostics(
                        global_key, key_delta, physical_theta_k, label="k"
                    )
                )
                rotation_diagnostics.update(
                    _rotation_diagnostics(
                        global_value, value_delta, physical_theta_v, label="v"
                    )
                )
            global_key = _magnitude_aware_rotate(
                global_key,
                key_delta,
                physical_theta_k,
            )
            global_value = _magnitude_aware_rotate(
                global_value,
                value_delta,
                physical_theta_v,
            )

        routing_key = base_global_key if self.route_from_base_global_key else global_key
        representatives = _completed_chunk_representatives(
            routing_key,
            chunk_size=self.chunk_size,
            valid_lengths=lengths,
            blend_alpha=self.representative_mix,
        )
        query_normalized = F.normalize(query.float(), dim=-1, eps=1e-6).to(
            query.dtype
        )
        selection_query = query_normalized

        # Hard top-k routing has no useful gradient. Keep the complete token/chunk
        # selector out of autograd; the auxiliary below recomputes only its sampled
        # query rows with gradients enabled.
        with torch.no_grad():
            selection_logits = torch.matmul(
                selection_query.float(),
                representatives.float().transpose(-2, -1),
            )
        retain_selection_logits = emit_diagnostics or (
            return_metadata and not compiling
        )
        selection_metadata = _build_causal_tile_metadata(
            selection_logits,
            chunk_size=self.chunk_size,
            top_k_chunks=self.top_k_chunks,
            selector_tile_size=1,
            local_window=self.local_window,
            valid_lengths=lengths,
            exploration_probability=(
                self.exploration_probability if self.training else 0.0
            ),
            preserve_anchor_logits=retain_selection_logits,
        )
        if emit_diagnostics and self.route_from_base_global_key:
            with torch.no_grad():
                attention_representatives = _completed_chunk_representatives(
                    global_key,
                    chunk_size=self.chunk_size,
                    valid_lengths=lengths,
                    blend_alpha=self.representative_mix,
                )
                attention_logits = torch.matmul(
                    selection_query.float(),
                    attention_representatives.float().transpose(-2, -1),
                )
                route_deterministic = _build_causal_tile_metadata(
                    selection_query.float()
                    @ representatives.float().transpose(-2, -1),
                    chunk_size=self.chunk_size,
                    top_k_chunks=self.top_k_chunks,
                    selector_tile_size=1,
                    local_window=self.local_window,
                    valid_lengths=lengths,
                    exploration_probability=0.0,
                ).top_chunk_idx
                attention_deterministic = _build_causal_tile_metadata(
                    attention_logits,
                    chunk_size=self.chunk_size,
                    top_k_chunks=self.top_k_chunks,
                    selector_tile_size=1,
                    local_window=self.local_window,
                    valid_lengths=lengths,
                    exploration_probability=0.0,
                ).top_chunk_idx
                route_valid = route_deterministic >= 0
                attention_valid = attention_deterministic >= 0
                intersection = (
                    (
                        route_deterministic[..., :, None]
                        == attention_deterministic[..., None, :]
                    )
                    & route_valid[..., :, None]
                    & attention_valid[..., None, :]
                ).any(-1).sum(-1).float()
                union = (
                    route_valid.sum(-1).float()
                    + attention_valid.sum(-1).float()
                    - intersection
                )
                route_comparison_diagnostics["route_vs_rotated_topk_jaccard"] = (
                    torch.where(union > 0, intersection / union.clamp_min(1.0), 1.0)
                    .mean()
                    .detach()
                )
        analysis_selection_logits = (
            selection_logits if retain_selection_logits else None
        )
        del selection_logits

        metadata = selection_metadata
        if selection_metadata.enumerate_all and self.token_routing_pack_size > 1:
            metadata = _pack_token_metadata(
                selection_metadata, pack_size=self.token_routing_pack_size
            )
        if not compiling:
            self._last_token_selection_path = (
                "enumerate_all_packed"
                if metadata.query_chunk_idx is not None
                else ("enumerate_all" if metadata.enumerate_all else "selected_tokens")
            )

        route = _selected_route_scores(
            query_normalized,
            representatives,
            metadata,
            self.route_prior_scale,
            temperature=self.temperature,
        )

        # Correct the lane prior for the number of available local/global keys.
        # Values are exact integers below 2**24. Float32 avoids a PyTorch
        # 2.13 Inductor symbolic-range bug on integer arange/subtraction.
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        local_count = positions.clamp(max=float(self.local_window))
        if self.boundary_bridge:
            cutoff = (positions - float(self.local_window)).clamp_min(0.0)
            local_count = local_count + torch.remainder(
                cutoff, float(self.chunk_size)
            )
        tokens_per_chunk = (
            self.chunk_size
            if metadata.enumerate_all
            else int(metadata.token_idx.shape[-1])
        )
        global_count = torch.isfinite(route).sum(-1).float() * float(tokens_per_chunk)
        count_correction = torch.where(
            (global_count > 0.0) & (local_count.reshape(1, 1, -1) > 0.0),
            torch.log(
                local_count.reshape(1, 1, -1).clamp_min(1.0)
                / global_count.clamp_min(1.0)
            ),
            torch.zeros_like(global_count),
        )
        route = route + (
            count_correction[..., None]
            + self.bounded_global_lane_logit_bias.reshape(1, self.H, 1, 1).float()
        ).to(route.dtype)

        auxiliary = torch.zeros((), device=x.device, dtype=torch.float32)
        sampled_anchor_logits: torch.Tensor | None = None
        sampled_tile_ids: torch.Tensor | None = None
        if self.training and self.route_aux_weight > 0.0:
            auxiliary_raw, sampled_anchor_logits, sampled_tile_ids = (
                _router_auxiliary_loss(
                    selection_query,
                    representatives,
                    routing_key,
                    selection_metadata,
                    self.route_prior_scale,
                    samples=self.route_aux_samples,
                    target_temperature=self.route_aux_temperature,
                    routing_temperature=self.temperature,
                    local_window=self.local_window,
                    oracle_temperature=self.route_aux_oracle_temperature,
                    tile_ids=route_aux_tile_ids,
                )
            )
            auxiliary = auxiliary_raw * self.route_aux_weight

        if not compiling:
            self._routing_auxiliary_loss = auxiliary
            if return_metadata:
                if analysis_selection_logits is None:
                    raise RuntimeError("HISA selector capture was not retained")
                self.hisa_evidence_capture = HISASelectionCapture(
                    anchor_logits=analysis_selection_logits,
                    metadata=selection_metadata,
                    auxiliary_loss=auxiliary,
                    sampled_anchor_logits=sampled_anchor_logits,
                    sampled_tile_ids=sampled_tile_ids,
                )

        if emit_diagnostics:
            if analysis_selection_logits is None:
                raise RuntimeError("HISA diagnostic selector was not retained")
            with torch.no_grad():
                first_useful = self.chunk_size + self.local_window
                candidate_count = max(0, seq_len - first_useful)
                diagnostic_count = min(
                    self.diagnostic_max_queries, candidate_count
                )
                if diagnostic_count > 0:
                    if diagnostic_count == candidate_count:
                        diagnostic_ids = torch.arange(
                            first_useful,
                            seq_len,
                            device=x.device,
                            dtype=torch.int64,
                        )
                    else:
                        diagnostic_ids = torch.linspace(
                            first_useful,
                            seq_len - 1,
                            diagnostic_count,
                            device=x.device,
                        ).round().to(torch.int64)
                    self._routing_entropy = _eligible_route_entropy(
                        analysis_selection_logits,
                        selection_metadata,
                        self.local_window,
                        tile_ids=diagnostic_ids,
                    ).detach()
                else:
                    self._routing_entropy = torch.zeros((), device=x.device)

                finite_route = torch.isfinite(route)
                finite_route_count = finite_route.sum().clamp_min(1)
                physical_chunks = (metadata.top_chunk_idx >= 0).sum(-1).float()
                semantic_chunks = (
                    selection_metadata.top_chunk_idx >= 0
                ).sum(-1).float()
                semantic_valid = (
                    selection_metadata.tile_starts.reshape(1, 1, -1)
                    < lengths.reshape(batch_size, 1, 1)
                ).expand_as(semantic_chunks)
                physical_valid = (
                    metadata.tile_starts.reshape(1, 1, -1)
                    < lengths.reshape(batch_size, 1, 1)
                ).expand_as(physical_chunks)
                semantic_mean = (
                    torch.where(
                        semantic_valid,
                        semantic_chunks,
                        torch.zeros_like(semantic_chunks),
                    ).sum()
                    / semantic_valid.sum().clamp_min(1)
                )
                physical_mean = (
                    torch.where(
                        physical_valid,
                        physical_chunks,
                        torch.zeros_like(physical_chunks),
                    ).sum()
                    / physical_valid.sum().clamp_min(1)
                )
                self._routing_diagnostics = {
                    "route_source_base_global_k": torch.tensor(
                        float(self.route_from_base_global_key), device=x.device
                    ),
                    **rotation_diagnostics,
                    **route_comparison_diagnostics,
                    "routing_entropy": self._routing_entropy,
                    "routing_entropy_queries": torch.tensor(
                        float(diagnostic_count), device=x.device
                    ),
                    "selected_route_rms": torch.sqrt(
                        torch.where(
                            finite_route, route.float().square(), 0.0
                        ).sum()
                        / finite_route_count
                    ).detach(),
                    "route_prior_scale_mean": self.route_prior_scale.mean().detach(),
                    "route_prior_scale_min": self.route_prior_scale.min().detach(),
                    "route_prior_scale_max": self.route_prior_scale.max().detach(),
                    "global_lane_logit_bias_mean": (
                        self.bounded_global_lane_logit_bias.mean().detach()
                    ),
                    "global_lane_logit_bias_min": (
                        self.bounded_global_lane_logit_bias.min().detach()
                    ),
                    "global_lane_logit_bias_max": (
                        self.bounded_global_lane_logit_bias.max().detach()
                    ),
                    "representative_mean_mix": (
                        (1.0 - self.representative_mix).mean().detach()
                    ),
                    "representative_mean_mix_min": (
                        (1.0 - self.representative_mix).min().detach()
                    ),
                    "representative_mean_mix_max": (
                        (1.0 - self.representative_mix).max().detach()
                    ),
                    "semantic_chunks_per_query": semantic_mean.detach(),
                    "physical_union_chunks_per_pack": physical_mean.detach(),
                    "physical_union_inflation": (
                        physical_mean / semantic_mean.clamp_min(1.0)
                    ).detach(),
                    "physical_union_slot_fraction": (
                        physical_mean
                        / (
                            semantic_mean.clamp_min(1.0)
                            * (
                                metadata.selector_tile_size
                                if metadata.query_chunk_idx is not None
                                else 1
                            )
                        )
                    ).detach(),
                    "physical_routing_pack_size": torch.tensor(
                        float(metadata.selector_tile_size), device=x.device
                    ),
                    "enumerate_all": torch.tensor(
                        float(metadata.enumerate_all), device=x.device
                    ),
                }

        local_output, local_lse = self._local_lane(
            query, local_key, local_value, lengths
        )
        if use_triton:
            global_output, global_lse = _global_hisa_triton_apply(
                query,
                global_key,
                global_value,
                route,
                metadata.top_chunk_idx,
                metadata.token_idx,
                lengths,
                self.chunk_size,
                metadata.enumerate_all,
                metadata.selector_tile_size,
                self.local_window,
                self.triton_block_q,
                self.backward_impl == "atomic_masked",
            )
        else:
            global_output, global_lse = _eager_global_lane(
                query,
                global_key,
                global_value,
                route,
                metadata,
                local_window=self.local_window,
            )
        attended, combined_lse, head_global_mass = _merge_attention_lanes_with_mass(
            local_output, local_lse, global_output, global_lse
        )
        binding_evidence_heads = (
            global_output.float() * head_global_mass.unsqueeze(-1)
        )
        binding_confidence = head_global_mass.mean(1).unsqueeze(-1)

        if emit_diagnostics:
            with torch.no_grad():
                valid_rows = torch.isfinite(combined_lse)
                valid_count = valid_rows.sum().clamp_min(1)
                global_mass = torch.where(
                    valid_rows,
                    head_global_mass.float(),
                    torch.zeros_like(head_global_mass.float()),
                )
                self._routing_diagnostics["global_attention_mass"] = (
                    global_mass.sum() / valid_count
                ).detach()
                self._routing_diagnostics["local_attention_mass"] = (
                    torch.where(
                        valid_rows, 1.0 - global_mass, 0.0
                    ).sum()
                    / valid_count
                ).detach()
                self._routing_diagnostics["combined_lse_finite_rate"] = (
                    valid_rows.float().mean().detach()
                )

        merged = attended.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.D
        )
        projected = self.W_o(merged)
        output = projected * torch.sigmoid(gate)
        if self.binding_rank:
            # Scale remote evidence by its exact merged-softmax mass.
            output = output + self._binding_correction(
                x, binding_evidence_heads, binding_confidence
            )
        if return_metadata and return_auxiliary:
            return output, metadata, auxiliary
        if return_metadata:
            return output, metadata
        if return_auxiliary:
            return output, auxiliary
        return output
