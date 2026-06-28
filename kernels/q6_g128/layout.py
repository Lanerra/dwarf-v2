#!/usr/bin/env python3
"""DWARF q6_g128 cache layout.

Layout target:
- cache tensor shape: [B, H, N, D] with D=64
- q6_g128 group = two adjacent token vectors per (B,H), i.e. 2 * D = 128 values
- payload layout: [B, H, ceil(N/2), 96] uint8 bytes
- scales layout: [B, H, ceil(N/2)] f32
- decode-on-read path gathers causal offset token vectors without full-cache decode

This is still a PyTorch prototype, not a fused Triton/CUDA kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required; use /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python") from exc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "logs/q6_g128_dwarf_layout_summary"

from . import pack as q6

HEAD_DIM = 64
TOKENS_PER_GROUP = 2
VALUES_PER_GROUP = 128
PAYLOAD_BYTES_PER_GROUP = 96
SCALE_BYTES_PER_GROUP = 4
BYTES_PER_PAIR_GROUP = PAYLOAD_BYTES_PER_GROUP + SCALE_BYTES_PER_GROUP


class DwarfQ6G128CacheLayout:
    def __init__(
        self,
        *,
        payload: torch.Tensor,
        scales: torch.Tensor,
        batch: int,
        heads: int,
        seq_len: int,
        padded_seq_len: int,
        head_dim: int,
        seed: int,
    ) -> None:
        if payload.dtype != torch.uint8:
            raise TypeError(f"payload must be uint8, got {payload.dtype}")
        if scales.dtype != torch.float32:
            raise TypeError(f"scales must be float32, got {scales.dtype}")
        self.payload = payload
        self.scales = scales
        self.batch = int(batch)
        self.heads = int(heads)
        self.seq_len = int(seq_len)
        self.padded_seq_len = int(padded_seq_len)
        self.head_dim = int(head_dim)
        self.seed = int(seed)
        self.bits = q6.BITS
        self.group_size = q6.GROUP_SIZE
        self.tokens_per_group = TOKENS_PER_GROUP

    @property
    def pair_groups(self) -> int:
        return self.padded_seq_len // TOKENS_PER_GROUP

    def storage_report(self) -> dict[str, Any]:
        payload_bytes = int(self.payload.numel() * self.payload.element_size())
        scale_bytes = int(self.scales.numel() * self.scales.element_size())
        total = payload_bytes + scale_bytes
        values = self.batch * self.heads * self.seq_len * self.head_dim
        padded_values = self.batch * self.heads * self.padded_seq_len * self.head_dim
        return {
            "batch": self.batch,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "padded_seq_len": self.padded_seq_len,
            "head_dim": self.head_dim,
            "pair_groups": self.pair_groups,
            "bits": self.bits,
            "group_size": self.group_size,
            "tokens_per_group": self.tokens_per_group,
            "payload_bytes": payload_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": total,
            "num_values": values,
            "padded_values": padded_values,
            "effective_bytes_per_value": total / values,
            "compression_vs_bf16": (values * 2) / total,
            "compression_vs_fp32": (values * 4) / total,
        }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    return gen


def _validate_cache_tensor(x: torch.Tensor) -> tuple[int, int, int, int]:
    if not torch.is_floating_point(x):
        raise TypeError(f"expected floating cache tensor, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"expected [B,H,N,D] tensor, got shape {tuple(x.shape)}")
    b, h, n, d = (int(v) for v in x.shape)
    if d != HEAD_DIM:
        raise ValueError(f"q6_g128 DWARF cache layout currently requires D={HEAD_DIM}, got {d}")
    return b, h, n, d


def pack_q6_g128_cache_layout(x: torch.Tensor, *, seed: int, min_scale: float = 1e-8) -> DwarfQ6G128CacheLayout:
    b, h, n, d = _validate_cache_tensor(x)
    padded_n = int(math.ceil(n / TOKENS_PER_GROUP) * TOKENS_PER_GROUP)
    xf = x.float()
    if padded_n != n:
        pad = torch.zeros((b, h, padded_n - n, d), device=x.device, dtype=torch.float32)
        xf = torch.cat([xf, pad], dim=2)
    groups = xf.reshape(b, h, padded_n // TOKENS_PER_GROUP, VALUES_PER_GROUP)
    scales = (groups.abs().amax(dim=-1).clamp_min(min_scale) / q6.QMAX).to(torch.float32)
    scaled = groups / scales.unsqueeze(-1)
    floor = torch.floor(scaled)
    frac = (scaled - floor).clamp(0.0, 1.0)
    gen = _make_generator(x.device, seed)
    rnd = torch.rand(frac.shape, device=x.device, generator=gen, dtype=torch.float32)
    qvals = (floor + (rnd < frac).to(torch.float32)).clamp(q6.QMIN, q6.QMAX).to(torch.int16)
    payload = q6.pack_signed_q6(qvals.reshape(-1)).reshape(b, h, padded_n // TOKENS_PER_GROUP, PAYLOAD_BYTES_PER_GROUP)
    return DwarfQ6G128CacheLayout(
        payload=payload,
        scales=scales,
        batch=b,
        heads=h,
        seq_len=n,
        padded_seq_len=padded_n,
        head_dim=d,
        seed=seed,
    )


def _decode_pair_payload(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    # payload [...,96] -> decoded [...,2,64]
    orig = payload.shape[:-1]
    qvals = q6.unpack_signed_q6(payload.reshape(-1, PAYLOAD_BYTES_PER_GROUP), values=payload.reshape(-1).numel() // 3 * 4)
    qvals = qvals.reshape(*orig, VALUES_PER_GROUP).to(torch.float32)
    decoded = qvals * scales.reshape(*orig, 1)
    return decoded.reshape(*orig, TOKENS_PER_GROUP, HEAD_DIM)


def decode_full_cache_layout(layout: DwarfQ6G128CacheLayout) -> torch.Tensor:
    decoded = _decode_pair_payload(layout.payload, layout.scales)
    return decoded.reshape(layout.batch, layout.heads, layout.padded_seq_len, layout.head_dim)[:, :, : layout.seq_len, :]


def decode_full_cache_layout_to_bf16_scratch(
    layout: DwarfQ6G128CacheLayout,
    *,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    scratch = decode_full_cache_layout(layout).to(torch.bfloat16)
    if not return_report:
        return scratch
    storage = layout.storage_report()
    scratch_bytes = int(scratch.numel() * scratch.element_size())
    report = {
        "scratch_dtype": str(scratch.dtype),
        "scratch_bytes": scratch_bytes,
        "scratch_bytes_per_value": scratch_bytes / storage["num_values"],
        "resident_q6_bytes": storage["total_bytes"],
        "resident_q6_bytes_per_value": storage["effective_bytes_per_value"],
        "scratch_plus_resident_bytes": scratch_bytes + storage["total_bytes"],
        "scratch_plus_resident_bytes_per_value": (scratch_bytes + storage["total_bytes"]) / storage["num_values"],
    }
    return scratch, report


def full_decode_scratch_then_gather(
    layout: DwarfQ6G128CacheLayout,
    offsets: list[int],
    *,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    scratch, scratch_report = decode_full_cache_layout_to_bf16_scratch(layout, return_report=True)
    gathered, mask = bf16_causal_gather(scratch, offsets)
    if not return_report:
        return gathered, mask
    idx, valid = causal_offset_index(layout.seq_len, offsets, device=layout.payload.device)
    report = {
        **scratch_report,
        "offset_count": len(offsets),
        "valid_token_reads": int(valid.sum().item()),
        "valid_vector_reads": int(valid.sum().item()) * layout.batch * layout.heads,
    }
    return gathered, mask, report


def causal_offset_index(seq_len: int, offsets: list[int], *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    device = device or torch.device("cpu")
    qpos = torch.arange(seq_len, device=device, dtype=torch.long).reshape(seq_len, 1)
    off = torch.tensor(offsets, device=device, dtype=torch.long).reshape(1, len(offsets))
    idx = qpos - off
    valid = idx >= 0
    return idx.clamp_min(0), valid


def _decode_report(layout: DwarfQ6G128CacheLayout, offsets: list[int], pair_idx: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    unique_pairs = torch.unique(pair_idx[valid]).numel() if bool(valid.any().item()) else 0
    valid_reads = int(valid.sum().item())
    unique_layout_groups = int(unique_pairs) * layout.batch * layout.heads
    return {
        "query_tokens": layout.seq_len,
        "offset_count": len(offsets),
        "valid_token_reads": valid_reads,
        "valid_vector_reads": valid_reads * layout.batch * layout.heads,
        "unique_pair_groups_touched": int(unique_pairs),
        "unique_layout_groups_touched": unique_layout_groups,
        "bytes_per_pair_group": BYTES_PER_PAIR_GROUP,
        "unique_group_bytes_touched": unique_layout_groups * BYTES_PER_PAIR_GROUP,
        "naive_read_bytes": valid_reads * layout.batch * layout.heads * BYTES_PER_PAIR_GROUP,
        "decoded_values_per_token": layout.head_dim,
    }


def decode_causal_offset_tokens(
    layout: DwarfQ6G128CacheLayout,
    offsets: list[int],
    *,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    device = layout.payload.device
    idx, valid = causal_offset_index(layout.seq_len, offsets, device=device)
    pair_idx = torch.div(idx, TOKENS_PER_GROUP, rounding_mode="floor")
    token_half = torch.remainder(idx, TOKENS_PER_GROUP)

    selected_payload = layout.payload[:, :, pair_idx, :]
    selected_scales = layout.scales[:, :, pair_idx]
    decoded_pairs = _decode_pair_payload(selected_payload, selected_scales)  # [B,H,N,K,2,D]
    gather_idx = token_half.reshape(1, 1, layout.seq_len, len(offsets), 1, 1).expand(
        layout.batch, layout.heads, layout.seq_len, len(offsets), 1, layout.head_dim
    )
    gathered = torch.gather(decoded_pairs, dim=4, index=gather_idx).squeeze(4)
    gathered = gathered * valid.reshape(1, 1, layout.seq_len, len(offsets), 1).to(gathered.dtype)
    if not return_report:
        return gathered, valid
    return gathered, valid, _decode_report(layout, offsets, pair_idx, valid)


def decode_causal_offset_tokens_unique_pairs(
    layout: DwarfQ6G128CacheLayout,
    offsets: list[int],
    *,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    device = layout.payload.device
    idx, valid = causal_offset_index(layout.seq_len, offsets, device=device)
    pair_idx = torch.div(idx, TOKENS_PER_GROUP, rounding_mode="floor")
    token_half = torch.remainder(idx, TOKENS_PER_GROUP)
    if bool(valid.any().item()):
        unique_pairs = torch.unique(pair_idx[valid])
    else:
        unique_pairs = torch.zeros((1,), device=device, dtype=torch.long)
    lookup = torch.full((layout.pair_groups,), -1, device=device, dtype=torch.long)
    lookup[unique_pairs] = torch.arange(unique_pairs.numel(), device=device, dtype=torch.long)
    local_pair_idx = lookup[pair_idx].clamp_min(0)
    decoded_unique = _decode_pair_payload(layout.payload[:, :, unique_pairs, :], layout.scales[:, :, unique_pairs])
    decoded_pairs = decoded_unique[:, :, local_pair_idx, :, :]  # [B,H,N,K,2,D]
    gather_idx = token_half.reshape(1, 1, layout.seq_len, len(offsets), 1, 1).expand(
        layout.batch, layout.heads, layout.seq_len, len(offsets), 1, layout.head_dim
    )
    gathered = torch.gather(decoded_pairs, dim=4, index=gather_idx).squeeze(4)
    gathered = gathered * valid.reshape(1, 1, layout.seq_len, len(offsets), 1).to(gathered.dtype)
    if not return_report:
        return gathered, valid
    report = _decode_report(layout, offsets, pair_idx, valid)
    report["decoded_pair_instances"] = int(unique_pairs.numel()) * layout.batch * layout.heads
    report["naive_pair_instances"] = int(valid.sum().item()) * layout.batch * layout.heads
    return gathered, valid, report


def decode_causal_offset_tokens_tiled(
    layout: DwarfQ6G128CacheLayout,
    offsets: list[int],
    *,
    tile_tokens: int,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if tile_tokens <= 0:
        raise ValueError("tile_tokens must be positive")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    device = layout.payload.device
    full_idx, full_valid = causal_offset_index(layout.seq_len, offsets, device=device)
    out = torch.zeros(
        (layout.batch, layout.heads, layout.seq_len, len(offsets), layout.head_dim),
        device=device,
        dtype=torch.float32,
    )
    decoded_pair_instances = 0
    tile_count = int(math.ceil(layout.seq_len / tile_tokens))
    for start in range(0, layout.seq_len, tile_tokens):
        end = min(layout.seq_len, start + tile_tokens)
        idx = full_idx[start:end]
        valid = full_valid[start:end]
        pair_idx = torch.div(idx, TOKENS_PER_GROUP, rounding_mode="floor")
        token_half = torch.remainder(idx, TOKENS_PER_GROUP)
        if bool(valid.any().item()):
            unique_pairs = torch.unique(pair_idx[valid])
        else:
            unique_pairs = torch.zeros((1,), device=device, dtype=torch.long)
        decoded_pair_instances += int(unique_pairs.numel()) * layout.batch * layout.heads
        lookup = torch.full((layout.pair_groups,), -1, device=device, dtype=torch.long)
        lookup[unique_pairs] = torch.arange(unique_pairs.numel(), device=device, dtype=torch.long)
        local_pair_idx = lookup[pair_idx].clamp_min(0)
        decoded_unique = _decode_pair_payload(layout.payload[:, :, unique_pairs, :], layout.scales[:, :, unique_pairs])
        decoded_pairs = decoded_unique[:, :, local_pair_idx, :, :]
        gather_idx = token_half.reshape(1, 1, end - start, len(offsets), 1, 1).expand(
            layout.batch, layout.heads, end - start, len(offsets), 1, layout.head_dim
        )
        gathered = torch.gather(decoded_pairs, dim=4, index=gather_idx).squeeze(4)
        out[:, :, start:end, :, :] = gathered * valid.reshape(1, 1, end - start, len(offsets), 1).to(gathered.dtype)
    if not return_report:
        return out, full_valid
    pair_idx_full = torch.div(full_idx, TOKENS_PER_GROUP, rounding_mode="floor")
    report = _decode_report(layout, offsets, pair_idx_full, full_valid)
    report["tile_tokens"] = int(tile_tokens)
    report["tile_count"] = tile_count
    report["decoded_pair_instances"] = decoded_pair_instances
    report["naive_pair_instances"] = int(full_valid.sum().item()) * layout.batch * layout.heads
    report["global_unique_pair_instances"] = report["unique_pair_groups_touched"] * layout.batch * layout.heads
    return out, full_valid, report


def make_workload(shape: tuple[int, int, int, int], name: str, device: torch.device, seed: int) -> torch.Tensor:
    gen = _make_generator(device, seed)
    if name == "gaussian":
        return torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
    if name == "nonstationary":
        base = torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
        ramp = torch.linspace(0.25, 6.0, shape[2], device=device, dtype=torch.float32).reshape(1, 1, shape[2], 1)
        return base * ramp
    if name == "adversarial":
        base = torch.randn(shape, device=device, generator=gen, dtype=torch.float32) * 0.5
        flat = base.reshape(-1)
        step = max(1, flat.numel() // 4096)
        idx = torch.arange(0, flat.numel(), step, device=device)[:4096]
        signs = torch.where((torch.arange(idx.numel(), device=device) % 2) == 0, 1.0, -1.0)
        flat[idx] = signs * torch.linspace(4.0, 32.0, idx.numel(), device=device)
        return base
    raise ValueError(f"unknown workload {name}")


def bf16_causal_gather(x: torch.Tensor, offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    idx, valid = causal_offset_index(x.shape[2], offsets, device=x.device)
    gathered = x[:, :, idx, :]
    return gathered * valid.reshape(1, 1, x.shape[2], len(offsets), 1).to(gathered.dtype), valid


def decode_cache_layout_token_window_to_bf16_scratch(
    layout: DwarfQ6G128CacheLayout,
    *,
    start: int,
    end: int,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Decode a contiguous token window from resident q6 cache into BF16 scratch.

    The resident layout stores one q6_g128 group per two adjacent D=64 token
    vectors.  Window boundaries may be odd, so this decodes the covering pair
    groups and then slices the requested token interval.  The returned scratch
    contains only `[start:end)` token vectors, not the whole sequence cache.
    """
    start = int(start)
    end = int(end)
    if start < 0 or end > layout.seq_len or start >= end:
        raise ValueError(f"expected non-empty token window within [0,{layout.seq_len}], got [{start},{end})")
    pair_start = start // TOKENS_PER_GROUP
    pair_end = int(math.ceil(end / TOKENS_PER_GROUP))
    decoded_pairs = _decode_pair_payload(layout.payload[:, :, pair_start:pair_end, :], layout.scales[:, :, pair_start:pair_end])
    decoded_tokens = decoded_pairs.reshape(layout.batch, layout.heads, (pair_end - pair_start) * TOKENS_PER_GROUP, layout.head_dim)
    local_start = start - pair_start * TOKENS_PER_GROUP
    scratch = decoded_tokens[:, :, local_start : local_start + (end - start), :].to(torch.bfloat16).contiguous()
    if not return_report:
        return scratch
    storage = layout.storage_report()
    scratch_bytes = int(scratch.numel() * scratch.element_size())
    decoded_pair_temp_bytes = int(decoded_pairs.numel() * decoded_pairs.element_size())
    report = {
        "window_start": start,
        "window_end": end,
        "window_tokens": end - start,
        "pair_start": pair_start,
        "pair_end": pair_end,
        "pair_groups_decoded": pair_end - pair_start,
        "covering_pair_tokens": (pair_end - pair_start) * TOKENS_PER_GROUP,
        "scratch_dtype": str(scratch.dtype),
        "scratch_bytes": scratch_bytes,
        "scratch_bytes_per_value": scratch_bytes / (layout.batch * layout.heads * (end - start) * layout.head_dim),
        "scratch_bytes_per_full_cache_value": scratch_bytes / storage["num_values"],
        "decoded_pair_temp_bytes_pytorch_only": decoded_pair_temp_bytes,
    }
    return scratch, report


def bf16_causal_gather_from_window(
    scratch: torch.Tensor,
    offsets: list[int],
    *,
    query_start: int,
    window_start: int,
    query_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather causal offset reads for a global query tile from a local BF16 scratch window."""
    if scratch.dtype != torch.bfloat16:
        raise TypeError(f"scratch must be bfloat16, got {scratch.dtype}")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    query_start = int(query_start)
    window_start = int(window_start)
    query_tokens = int(query_tokens)
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    device = scratch.device
    qpos = torch.arange(query_start, query_start + query_tokens, device=device, dtype=torch.long).reshape(query_tokens, 1)
    off = torch.tensor(offsets, device=device, dtype=torch.long).reshape(1, len(offsets))
    global_idx = qpos - off
    local_idx = global_idx - window_start
    valid = (global_idx >= 0) & (local_idx >= 0) & (local_idx < scratch.shape[2])
    gathered = scratch[:, :, local_idx.clamp(0, scratch.shape[2] - 1), :]
    return gathered * valid.reshape(1, 1, query_tokens, len(offsets), 1).to(gathered.dtype), valid


def tile_scoped_full_decode_scratch_then_gather(
    layout: DwarfQ6G128CacheLayout,
    offsets: list[int],
    *,
    tile_tokens: int,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Decode only the key window needed by each query tile, then BF16-gather.

    This is the tile-scoped version of `full_decode_scratch_then_gather`: resident
    q6 stays compressed, each query tile gets a short-lived BF16 scratch window
    spanning `[tile_start - max(offsets), tile_end)`, and the existing causal BF16
    gather semantics are applied against that scratch.  This is still a PyTorch
    prototype; timings are implementation-shape evidence, not production kernel throughput.
    """
    tile_tokens = int(tile_tokens)
    if tile_tokens <= 0:
        raise ValueError("tile_tokens must be positive")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    if any(o < 0 for o in offsets):
        raise ValueError("offsets must be nonnegative")
    device = layout.payload.device
    full_idx, full_valid = causal_offset_index(layout.seq_len, offsets, device=device)
    out = torch.zeros(
        (layout.batch, layout.heads, layout.seq_len, len(offsets), layout.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    max_offset = max(offsets)
    tile_count = int(math.ceil(layout.seq_len / tile_tokens))
    peak_scratch_bytes = 0
    peak_scratch_tokens = 0
    total_scratch_bytes = 0
    total_window_tokens = 0
    decoded_pair_groups_total = 0
    decoded_pair_temp_bytes_total = 0
    max_pair_groups_per_tile = 0
    for start in range(0, layout.seq_len, tile_tokens):
        end = min(layout.seq_len, start + tile_tokens)
        window_start = max(0, start - max_offset)
        window_end = end
        scratch, scratch_report = decode_cache_layout_token_window_to_bf16_scratch(
            layout, start=window_start, end=window_end, return_report=True
        )
        gathered, valid = bf16_causal_gather_from_window(
            scratch,
            offsets,
            query_start=start,
            window_start=window_start,
            query_tokens=end - start,
        )
        if not torch.equal(valid, full_valid[start:end]):
            raise RuntimeError("tile-scoped scratch gather mask diverged from full causal mask")
        out[:, :, start:end, :, :] = gathered
        scratch_bytes = int(scratch_report["scratch_bytes"])
        peak_scratch_bytes = max(peak_scratch_bytes, scratch_bytes)
        peak_scratch_tokens = max(peak_scratch_tokens, int(scratch_report["window_tokens"]))
        total_scratch_bytes += scratch_bytes
        total_window_tokens += int(scratch_report["window_tokens"])
        decoded_pair_groups_total += int(scratch_report["pair_groups_decoded"])
        decoded_pair_temp_bytes_total += int(scratch_report["decoded_pair_temp_bytes_pytorch_only"])
        max_pair_groups_per_tile = max(max_pair_groups_per_tile, int(scratch_report["pair_groups_decoded"]))
    if not return_report:
        return out, full_valid
    storage = layout.storage_report()
    pair_idx_full = torch.div(full_idx, TOKENS_PER_GROUP, rounding_mode="floor")
    report = _decode_report(layout, offsets, pair_idx_full, full_valid)
    full_scratch_bytes = storage["num_values"] * 2
    report.update(
        {
            "tile_tokens": tile_tokens,
            "tile_count": tile_count,
            "resident_q6_bytes": storage["total_bytes"],
            "resident_q6_bytes_per_value": storage["effective_bytes_per_value"],
            "resident_compression_vs_bf16": storage["compression_vs_bf16"],
            "peak_scratch_tokens": peak_scratch_tokens,
            "peak_scratch_bytes": peak_scratch_bytes,
            "peak_scratch_bytes_per_full_cache_value": peak_scratch_bytes / storage["num_values"],
            "full_scratch_bytes": full_scratch_bytes,
            "peak_scratch_vs_full_scratch": peak_scratch_bytes / full_scratch_bytes,
            "scratch_plus_resident_peak_bytes": peak_scratch_bytes + storage["total_bytes"],
            "scratch_plus_resident_peak_bytes_per_value": (peak_scratch_bytes + storage["total_bytes"]) / storage["num_values"],
            "total_tile_scratch_bytes_decoded": total_scratch_bytes,
            "total_tile_window_tokens": total_window_tokens,
            "decoded_pair_groups_total": decoded_pair_groups_total,
            "decoded_pair_instances_total": decoded_pair_groups_total * layout.batch * layout.heads,
            "decoded_pair_temp_bytes_total_pytorch_only": decoded_pair_temp_bytes_total,
            "max_pair_groups_per_tile": max_pair_groups_per_tile,
        }
    )
    return out, full_valid, report


def error_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    diff = y.float() - x.float()
    denom = x.float().pow(2).mean().sqrt().clamp_min(1e-8)
    return {
        "finite": bool(torch.isfinite(y).all().item()),
        "max_abs_err": float(diff.abs().max().detach().cpu()),
        "rms_err": float(diff.pow(2).mean().sqrt().detach().cpu()),
        "relative_rms_err": float((diff.pow(2).mean().sqrt() / denom).detach().cpu()),
        "mean_abs_err": float(diff.abs().mean().detach().cpu()),
    }


def _time_callable(device: torch.device, repeats: int, fn) -> float:
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) * 1000.0 / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1_000_000.0 / repeats


def parse_shape(s: str) -> tuple[int, int, int, int]:
    parts = tuple(int(p) for p in s.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be B,H,N,D")
    return parts  # type: ignore[return-value]


def parse_offsets(s: str) -> list[int]:
    vals = [int(p) for p in s.split(",") if p]
    if not vals or any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("offsets must be comma-separated nonnegative integers")
    return vals


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    for shape in args.shape:
        for workload in args.workload:
            x = make_workload(shape, workload, device, args.seed)
            x_bf16 = x.to(torch.bfloat16)
            layout = pack_q6_g128_cache_layout(x, seed=args.seed)
            decoded_full = decode_full_cache_layout(layout)
            q_gather, mask, read_report = decode_causal_offset_tokens(layout, args.offsets, return_report=True)
            q_unique_gather, unique_mask, unique_report = decode_causal_offset_tokens_unique_pairs(
                layout, args.offsets, return_report=True
            )
            bf16_gather, _ = bf16_causal_gather(x_bf16, args.offsets)
            expected_from_full, _ = bf16_causal_gather(decoded_full, args.offsets)
            if not torch.equal(q_gather, expected_from_full):
                raise RuntimeError("decode-on-read path diverged from full layout decode gather")
            if not torch.equal(q_unique_gather, expected_from_full) or not torch.equal(mask, unique_mask):
                raise RuntimeError("unique-pair decode path diverged from full layout decode gather")
            q_us = _time_callable(device, args.repeats, lambda: decode_causal_offset_tokens(layout, args.offsets))
            q_unique_us = _time_callable(
                device, args.repeats, lambda: decode_causal_offset_tokens_unique_pairs(layout, args.offsets)
            )
            bf16_us = _time_callable(device, args.repeats, lambda: bf16_causal_gather(x_bf16, args.offsets))
            full_decode_us = _time_callable(device, args.repeats, lambda: decode_full_cache_layout(layout))
            storage = layout.storage_report()
            err = error_metrics(bf16_gather.float(), q_unique_gather.float())
            rows.append(
                {
                    "variant": "q6_g128_dwarf_cache_decode_on_read",
                    "shape": "x".join(str(v) for v in shape),
                    "workload": workload,
                    "device": str(device),
                    "offsets": ",".join(str(v) for v in args.offsets),
                    "offset_count": len(args.offsets),
                    "effective_bytes_per_value": storage["effective_bytes_per_value"],
                    "compression_vs_bf16": storage["compression_vs_bf16"],
                    "payload_bytes": storage["payload_bytes"],
                    "scale_bytes": storage["scale_bytes"],
                    "total_bytes": storage["total_bytes"],
                    "q6_decode_on_read_us": q_us,
                    "q6_unique_pair_decode_us": q_unique_us,
                    "bf16_gather_us": bf16_us,
                    "full_q6_decode_us": full_decode_us,
                    "q6_vs_bf16_time_ratio": q_us / bf16_us if bf16_us > 0 else None,
                    "q6_unique_vs_bf16_time_ratio": q_unique_us / bf16_us if bf16_us > 0 else None,
                    **read_report,
                    "unique_decoded_pair_instances": unique_report.get("decoded_pair_instances"),
                    "naive_pair_instances": unique_report.get("naive_pair_instances"),
                    **err,
                }
            )
    return rows


def write_rows(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    fields = list(rows[0].keys()) if rows else []
    with out_prefix.with_suffix(".tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--offsets", type=parse_offsets, default=parse_offsets("0,1,2,4,8,16,32,64"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(1, 8, 2048, 64), (1, 8, 8192, 64)]
    if not args.workload:
        args.workload = ["gaussian", "nonstationary", "adversarial"]
    rows = run_benchmark(args)
    write_rows(rows, args.out_prefix)
    print(f"wrote {len(rows)} q6_g128 DWARF-layout rows to {args.out_prefix}.json/.tsv")
    bad = [r for r in rows if not r["finite"]]
    if bad:
        raise SystemExit(f"non-finite q6 layout rows: {len(bad)}")


if __name__ == "__main__":
    main()
