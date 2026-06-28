#!/usr/bin/env python3
"""Triton q6_g128 decode/gather kernels.

This is the trainer-adjacent gate after the PyTorch tile-scoped prototype:
resident q6_g128 cache stays compressed, a Triton kernel decodes only the key
window for each query tile into BF16 scratch, then the existing causal BF16 sparse
gather semantics read from that scratch.

The gather remains the same BF16 scratch-read contract as the PyTorch reference;
this file isolates the decode-window step as the first Triton implementation path.
Timings are microbench evidence, not a full trainer throughput claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required; use /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python") from exc

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("triton is required for this microbench") from exc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "logs/q6_g128_triton_tile_scoped_scratch_summary"

from . import layout as layout_mod


@triton.jit
def _decode_q6_g128_window_kernel(
    payload_ptr,
    scales_ptr,
    scratch_ptr,
    total_values: tl.constexpr,
    window_start: tl.constexpr,
    window_tokens: tl.constexpr,
    heads: tl.constexpr,
    pair_groups: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_values

    # scratch is contiguous [B,H,W,64]. Decode flattened offset -> b,h,t,d.
    d = offs & 63
    tmp = offs >> 6
    t = tmp % window_tokens
    tmp = tmp // window_tokens
    h = tmp % heads
    b = tmp // heads

    global_t = t + window_start
    pair = global_t >> 1
    half = global_t & 1
    val_in_pair = half * 64 + d
    word_idx = val_in_pair >> 2
    lane = val_in_pair & 3

    bh = b * heads + h
    payload_base = (bh * pair_groups + pair) * 96 + word_idx * 3
    b0 = tl.load(payload_ptr + payload_base + 0, mask=mask, other=0).to(tl.uint32)
    b1 = tl.load(payload_ptr + payload_base + 1, mask=mask, other=0).to(tl.uint32)
    b2 = tl.load(payload_ptr + payload_base + 2, mask=mask, other=0).to(tl.uint32)
    word = b0 | (b1 << 8) | (b2 << 16)
    code = ((word >> (lane * 6)) & 0x3F).to(tl.int32)
    signed = tl.where(code >= 32, code - 64, code).to(tl.float32)
    scale = tl.load(scales_ptr + bh * pair_groups + pair, mask=mask, other=0.0).to(tl.float32)
    val = signed * scale
    tl.store(scratch_ptr + offs, val, mask=mask)


@triton.jit
def _decode_q6_g128_direct_gather_kernel(
    payload_ptr,
    scales_ptr,
    offsets_ptr,
    out_ptr,
    total_values: tl.constexpr,
    seq_len: tl.constexpr,
    offset_count: tl.constexpr,
    heads: tl.constexpr,
    pair_groups: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_values

    # out is contiguous [B,H,N,K,64]. Decode flattened offset -> b,h,q,k,d.
    d = offs & 63
    tmp = offs >> 6
    k = tmp % offset_count
    tmp = tmp // offset_count
    q = tmp % seq_len
    tmp = tmp // seq_len
    h = tmp % heads
    b = tmp // heads

    offset = tl.load(offsets_ptr + k, mask=mask, other=0).to(tl.int32)
    src_t = q.to(tl.int32) - offset
    valid = src_t >= 0
    safe_t = tl.maximum(src_t, 0)
    pair = safe_t >> 1
    half = safe_t & 1
    val_in_pair = half * 64 + d
    word_idx = val_in_pair >> 2
    lane = val_in_pair & 3

    bh = b * heads + h
    payload_base = (bh * pair_groups + pair) * 96 + word_idx * 3
    load_mask = mask & valid
    b0 = tl.load(payload_ptr + payload_base + 0, mask=load_mask, other=0).to(tl.uint32)
    b1 = tl.load(payload_ptr + payload_base + 1, mask=load_mask, other=0).to(tl.uint32)
    b2 = tl.load(payload_ptr + payload_base + 2, mask=load_mask, other=0).to(tl.uint32)
    word = b0 | (b1 << 8) | (b2 << 16)
    code = ((word >> (lane * 6)) & 0x3F).to(tl.int32)
    signed = tl.where(code >= 32, code - 64, code).to(tl.float32)
    scale = tl.load(scales_ptr + bh * pair_groups + pair, mask=load_mask, other=0.0).to(tl.float32)
    val = tl.where(valid, signed * scale, 0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def _require_cuda_layout(layout: Any) -> None:
    if not layout.payload.is_cuda or not layout.scales.is_cuda:
        raise ValueError("Triton q6_g128 microbench requires a CUDA layout")
    if layout.head_dim != layout_mod.HEAD_DIM:
        raise ValueError(f"expected head_dim={layout_mod.HEAD_DIM}, got {layout.head_dim}")
    if not layout.payload.is_contiguous() or not layout.scales.is_contiguous():
        raise ValueError("payload and scales must be contiguous")


def triton_decode_cache_layout_token_window_to_bf16_scratch(
    layout: Any,
    *,
    start: int,
    end: int,
    block: int = 256,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Triton decode of a q6_g128 token window into contiguous BF16 scratch."""
    _require_cuda_layout(layout)
    start = int(start)
    end = int(end)
    if start < 0 or end > layout.seq_len or start >= end:
        raise ValueError(f"expected non-empty token window within [0,{layout.seq_len}], got [{start},{end})")
    window_tokens = end - start
    scratch = torch.empty(
        (layout.batch, layout.heads, window_tokens, layout.head_dim),
        device=layout.payload.device,
        dtype=torch.bfloat16,
    )
    total_values = scratch.numel()
    grid = (triton.cdiv(total_values, block),)
    _decode_q6_g128_window_kernel[grid](
        layout.payload,
        layout.scales,
        scratch,
        total_values,
        start,
        window_tokens,
        layout.heads,
        layout.pair_groups,
        BLOCK=block,
    )
    if not return_report:
        return scratch
    storage = layout.storage_report()
    scratch_bytes = int(scratch.numel() * scratch.element_size())
    pair_start = start // layout_mod.TOKENS_PER_GROUP
    pair_end = (end + layout_mod.TOKENS_PER_GROUP - 1) // layout_mod.TOKENS_PER_GROUP
    report = {
        "window_start": start,
        "window_end": end,
        "window_tokens": window_tokens,
        "pair_start": pair_start,
        "pair_end": pair_end,
        "pair_groups_decoded": pair_end - pair_start,
        "scratch_dtype": str(scratch.dtype),
        "scratch_bytes": scratch_bytes,
        "scratch_bytes_per_value": scratch_bytes / (layout.batch * layout.heads * window_tokens * layout.head_dim),
        "scratch_bytes_per_full_cache_value": scratch_bytes / storage["num_values"],
        "triton_block": int(block),
    }
    return scratch, report


def triton_direct_decode_gather(
    layout: Any,
    offsets: list[int],
    *,
    block: int = 256,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Decode resident q6_g128 directly into gathered BF16 output.

    This removes the BF16 scratch tensor and the separate BF16 gather-from-scratch
    operation. It is still a read-path probe, not a production trainer kernel,
    because output allocation and causal-mask construction remain outside the kernel.
    """
    _require_cuda_layout(layout)
    if not offsets:
        raise ValueError("offsets must be non-empty")
    if any(o < 0 for o in offsets):
        raise ValueError("offsets must be nonnegative")
    device = layout.payload.device
    full_idx, full_valid = layout_mod.causal_offset_index(layout.seq_len, offsets, device=device)
    offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    out = torch.empty(
        (layout.batch, layout.heads, layout.seq_len, len(offsets), layout.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    total_values = out.numel()
    grid = (triton.cdiv(total_values, block),)
    _decode_q6_g128_direct_gather_kernel[grid](
        layout.payload,
        layout.scales,
        offsets_t,
        out,
        total_values,
        layout.seq_len,
        len(offsets),
        layout.heads,
        layout.pair_groups,
        BLOCK=block,
    )
    if not return_report:
        return out, full_valid
    storage = layout.storage_report()
    pair_idx_full = torch.div(full_idx, layout_mod.TOKENS_PER_GROUP, rounding_mode="floor")
    report = layout_mod._decode_report(layout, offsets, pair_idx_full, full_valid)
    full_scratch_bytes = storage["num_values"] * 2
    report.update(
        {
            "resident_q6_bytes": storage["total_bytes"],
            "resident_q6_bytes_per_value": storage["effective_bytes_per_value"],
            "resident_compression_vs_bf16": storage["compression_vs_bf16"],
            "peak_scratch_tokens": 0,
            "peak_scratch_bytes": 0,
            "peak_scratch_bytes_per_full_cache_value": 0.0,
            "full_scratch_bytes": full_scratch_bytes,
            "peak_scratch_vs_full_scratch": 0.0,
            "scratch_plus_resident_peak_bytes": storage["total_bytes"],
            "scratch_plus_resident_peak_bytes_per_value": storage["total_bytes"] / storage["num_values"],
            "total_tile_scratch_bytes_decoded": 0,
            "total_tile_window_tokens": 0,
            "decoded_pair_groups_total": int(full_valid.sum().item()),
            "decoded_pair_instances_total": int(full_valid.sum().item()) * layout.batch * layout.heads,
            "triton_block": int(block),
        }
    )
    return out, full_valid, report


def triton_tile_scoped_decode_scratch_then_gather(
    layout: Any,
    offsets: list[int],
    *,
    tile_tokens: int,
    block: int = 256,
    return_report: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Triton decode each tile's key window, then use BF16 gather-from-window semantics."""
    _require_cuda_layout(layout)
    tile_tokens = int(tile_tokens)
    if tile_tokens <= 0:
        raise ValueError("tile_tokens must be positive")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    if any(o < 0 for o in offsets):
        raise ValueError("offsets must be nonnegative")
    device = layout.payload.device
    full_idx, full_valid = layout_mod.causal_offset_index(layout.seq_len, offsets, device=device)
    out = torch.empty(
        (layout.batch, layout.heads, layout.seq_len, len(offsets), layout.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    max_offset = max(offsets)
    tile_count = (layout.seq_len + tile_tokens - 1) // tile_tokens
    peak_scratch_bytes = 0
    peak_scratch_tokens = 0
    total_scratch_bytes = 0
    total_window_tokens = 0
    decoded_pair_groups_total = 0
    for start in range(0, layout.seq_len, tile_tokens):
        end = min(layout.seq_len, start + tile_tokens)
        window_start = max(0, start - max_offset)
        window_end = end
        scratch, scratch_report = triton_decode_cache_layout_token_window_to_bf16_scratch(
            layout, start=window_start, end=window_end, block=block, return_report=True
        )
        gathered, valid = layout_mod.bf16_causal_gather_from_window(
            scratch,
            offsets,
            query_start=start,
            window_start=window_start,
            query_tokens=end - start,
        )
        if not torch.equal(valid, full_valid[start:end]):
            raise RuntimeError("Triton tile-scoped scratch gather mask diverged from full causal mask")
        out[:, :, start:end, :, :] = gathered
        scratch_bytes = int(scratch_report["scratch_bytes"])
        peak_scratch_bytes = max(peak_scratch_bytes, scratch_bytes)
        peak_scratch_tokens = max(peak_scratch_tokens, int(scratch_report["window_tokens"]))
        total_scratch_bytes += scratch_bytes
        total_window_tokens += int(scratch_report["window_tokens"])
        decoded_pair_groups_total += int(scratch_report["pair_groups_decoded"])
    if not return_report:
        return out, full_valid
    storage = layout.storage_report()
    pair_idx_full = torch.div(full_idx, layout_mod.TOKENS_PER_GROUP, rounding_mode="floor")
    report = layout_mod._decode_report(layout, offsets, pair_idx_full, full_valid)
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
            "triton_block": int(block),
        }
    )
    return out, full_valid, report


def _time_callable(device: torch.device, repeats: int, fn) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) * 1000.0 / repeats)


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


def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise SystemExit("Triton tile-scoped scratch microbench requires CUDA")
    rows: list[dict[str, Any]] = []
    for shape in args.shape:
        for workload in args.workload:
            x = layout_mod.make_workload(shape, workload, device, args.seed)
            x_bf16 = x.to(torch.bfloat16)
            layout = layout_mod.pack_q6_g128_cache_layout(x, seed=args.seed)
            reference, reference_mask = layout_mod.tile_scoped_full_decode_scratch_then_gather(
                layout, args.offsets, tile_tokens=max(args.tile_tokens)
            )
            bf16_gather, _ = layout_mod.bf16_causal_gather(x_bf16, args.offsets)
            bf16_us = _time_callable(device, args.repeats, lambda: layout_mod.bf16_causal_gather(x_bf16, args.offsets))
            for tile_tokens in args.tile_tokens:
                q_gather, mask, q_report = triton_tile_scoped_decode_scratch_then_gather(
                    layout, args.offsets, tile_tokens=tile_tokens, block=args.block, return_report=True
                )
                # Compare to the PyTorch reference for the exact same tile size too; the
                # reference is semantically tile-size invariant, but this catches boundary drift.
                expected, expected_mask = layout_mod.tile_scoped_full_decode_scratch_then_gather(
                    layout, args.offsets, tile_tokens=tile_tokens
                )
                if not torch.equal(mask, expected_mask) or not torch.equal(q_gather, expected):
                    max_diff = (q_gather.float() - expected.float()).abs().max().item()
                    raise RuntimeError(
                        f"Triton tile-scoped scratch diverged from PyTorch reference at tile={tile_tokens}; max_diff={max_diff}"
                    )
                if not torch.equal(mask, reference_mask) or not torch.equal(q_gather, reference):
                    max_diff = (q_gather.float() - reference.float()).abs().max().item()
                    raise RuntimeError(
                        f"Triton tile-scoped scratch diverged from tile-invariant reference at tile={tile_tokens}; max_diff={max_diff}"
                    )
                pytorch_tile_us = _time_callable(
                    device,
                    args.repeats,
                    lambda tile_tokens=tile_tokens: layout_mod.tile_scoped_full_decode_scratch_then_gather(
                        layout, args.offsets, tile_tokens=tile_tokens
                    ),
                )
                triton_us = _time_callable(
                    device,
                    args.repeats,
                    lambda tile_tokens=tile_tokens: triton_tile_scoped_decode_scratch_then_gather(
                        layout, args.offsets, tile_tokens=tile_tokens, block=args.block
                    ),
                )
                err = layout_mod.error_metrics(bf16_gather.float(), q_gather.float())
                rows.append(
                    {
                        "variant": "q6_g128_triton_tile_scoped_bf16_scratch_decode",
                        "shape": "x".join(str(v) for v in shape),
                        "workload": workload,
                        "device": str(device),
                        "offsets": ",".join(str(v) for v in args.offsets),
                        "offset_count": len(args.offsets),
                        "tile_tokens": tile_tokens,
                        "triton_tile_e2e_us": triton_us,
                        "pytorch_tile_reference_us": pytorch_tile_us,
                        "bf16_gather_us": bf16_us,
                        "triton_vs_bf16_ratio": triton_us / bf16_us if bf16_us > 0 else None,
                        "triton_vs_pytorch_tile_ratio": triton_us / pytorch_tile_us if pytorch_tile_us > 0 else None,
                        **q_report,
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
    parser.add_argument("--tile-tokens", type=int, action="append", default=[])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(1, 8, 2048, 64), (1, 8, 8192, 64)]
    if not args.workload:
        args.workload = ["gaussian", "nonstationary", "adversarial"]
    if not args.tile_tokens:
        args.tile_tokens = [1024, 2048, 4096, 8192]
    if any(t <= 0 for t in args.tile_tokens):
        raise SystemExit("--tile-tokens values must be positive")
    rows = run_probe(args)
    write_rows(rows, args.out_prefix)
    print(f"wrote {len(rows)} Triton q6_g128 tile-scoped scratch rows to {args.out_prefix}.json/.tsv")
    bad = [r for r in rows if not r["finite"]]
    if bad:
        raise SystemExit(f"non-finite Triton tile-scoped scratch rows: {len(bad)}")


if __name__ == "__main__":
    main()
