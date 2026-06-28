#!/usr/bin/env python3
"""q6_g128 block-absmax stochastic beta=0 packing helpers.

This is the first integration-oriented artifact after the Phase 3 tensor probe. It implements
actual 6-bit signed packing (4 quantized values -> 3 bytes), f32 block scales, deterministic
seeded stochastic rounding, decode, byte accounting, and a small benchmark CLI.
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
DEFAULT_OUT = ROOT / "logs/q6_g128_prototype_summary"

BITS = 6
GROUP_SIZE = 128
QMAX = 31
QMIN = -31
SCALE_BYTES = 4


class Q6G128PackedTensor:
    def __init__(
        self,
        *,
        packed: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: tuple[int, ...],
        num_values: int,
        padded_values: int,
        seed: int,
    ) -> None:
        if packed.dtype != torch.uint8:
            raise TypeError(f"packed must be uint8, got {packed.dtype}")
        if scales.dtype != torch.float32:
            raise TypeError(f"scales must be float32, got {scales.dtype}")
        self.packed = packed
        self.scales = scales
        self.orig_shape = tuple(orig_shape)
        self.num_values = int(num_values)
        self.padded_values = int(padded_values)
        self.seed = int(seed)
        self.bits = BITS
        self.group_size = GROUP_SIZE

    def storage_report(self) -> dict[str, Any]:
        payload_bytes = int(self.packed.numel() * self.packed.element_size())
        scale_bytes = int(self.scales.numel() * self.scales.element_size())
        total = payload_bytes + scale_bytes
        return {
            "num_values": self.num_values,
            "padded_values": self.padded_values,
            "groups": int(self.scales.numel()),
            "bits": self.bits,
            "group_size": self.group_size,
            "payload_bytes": payload_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": total,
            "effective_bytes_per_value": total / self.num_values,
            "compression_vs_bf16": (self.num_values * 2) / total,
            "compression_vs_fp32": (self.num_values * 4) / total,
        }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    return gen


def q6_g128_byte_accounting(num_values: int) -> dict[str, Any]:
    groups = math.ceil(num_values / GROUP_SIZE)
    padded_values = groups * GROUP_SIZE
    payload_bytes = padded_values * BITS // 8
    scale_bytes = groups * SCALE_BYTES
    total = payload_bytes + scale_bytes
    return {
        "num_values": int(num_values),
        "padded_values": int(padded_values),
        "groups": int(groups),
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "payload_bytes": int(payload_bytes),
        "scale_bytes": int(scale_bytes),
        "padding_values": int(padded_values - num_values),
        "total_bytes": int(total),
        "effective_bytes_per_value": total / num_values,
        "compression_vs_bf16": (num_values * 2) / total,
        "compression_vs_fp32": (num_values * 4) / total,
    }


def pack_signed_q6(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.int64)
    if flat.numel() % 4 != 0:
        raise ValueError("q6 packing requires a multiple of 4 values")
    if bool(((flat < -32) | (flat > 31)).any().item()):
        raise ValueError("signed q6 values must be in [-32, 31]")
    codes = torch.bitwise_and(flat, 0x3F)
    c = codes.reshape(-1, 4)
    words = c[:, 0] | (c[:, 1] << 6) | (c[:, 2] << 12) | (c[:, 3] << 18)
    out = torch.empty((words.numel(), 3), device=values.device, dtype=torch.uint8)
    out[:, 0] = torch.bitwise_and(words, 0xFF).to(torch.uint8)
    out[:, 1] = torch.bitwise_and(words >> 8, 0xFF).to(torch.uint8)
    out[:, 2] = torch.bitwise_and(words >> 16, 0xFF).to(torch.uint8)
    return out.reshape(-1)


def unpack_signed_q6(packed: torch.Tensor, *, values: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if packed.numel() % 3 != 0:
        raise ValueError("q6 packed byte stream length must be a multiple of 3")
    b = packed.reshape(-1, 3).to(torch.int64)
    words = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    codes = torch.stack(
        [
            torch.bitwise_and(words, 0x3F),
            torch.bitwise_and(words >> 6, 0x3F),
            torch.bitwise_and(words >> 12, 0x3F),
            torch.bitwise_and(words >> 18, 0x3F),
        ],
        dim=1,
    ).reshape(-1)[:values]
    signed = torch.where(codes >= 32, codes - 64, codes)
    return signed.to(torch.int16)


def quantize_pack_q6_g128(x: torch.Tensor, *, seed: int, min_scale: float = 1e-8) -> Q6G128PackedTensor:
    if not torch.is_floating_point(x):
        raise TypeError(f"expected floating input, got {x.dtype}")
    orig_shape = tuple(x.shape)
    xf = x.float().reshape(-1)
    num_values = int(xf.numel())
    groups = math.ceil(num_values / GROUP_SIZE)
    padded_values = groups * GROUP_SIZE
    if padded_values != num_values:
        xf = torch.nn.functional.pad(xf, (0, padded_values - num_values))
    block = xf.reshape(groups, GROUP_SIZE)
    scales = (block.abs().amax(dim=1).clamp_min(min_scale) / QMAX).to(torch.float32)
    scaled = block / scales.reshape(groups, 1)
    floor = torch.floor(scaled)
    frac = (scaled - floor).clamp(0.0, 1.0)
    gen = _make_generator(x.device, seed)
    rnd = torch.rand(frac.shape, device=x.device, generator=gen, dtype=torch.float32)
    q = (floor + (rnd < frac).to(torch.float32)).clamp(QMIN, QMAX).to(torch.int16)
    packed = pack_signed_q6(q.reshape(-1))
    return Q6G128PackedTensor(
        packed=packed,
        scales=scales,
        orig_shape=orig_shape,
        num_values=num_values,
        padded_values=padded_values,
        seed=seed,
    )


def decode_q6_g128(packed_tensor: Q6G128PackedTensor) -> torch.Tensor:
    q = unpack_signed_q6(packed_tensor.packed, values=packed_tensor.padded_values).to(torch.float32)
    block = q.reshape(-1, GROUP_SIZE) * packed_tensor.scales.reshape(-1, 1)
    return block.reshape(-1)[: packed_tensor.num_values].reshape(packed_tensor.orig_shape)


def make_workload(shape: tuple[int, ...], name: str, device: torch.device, seed: int) -> torch.Tensor:
    gen = _make_generator(device, seed)
    if name == "gaussian":
        return torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
    if name == "nonstationary":
        base = torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
        ramp = torch.linspace(0.25, 6.0, shape[1], device=device, dtype=torch.float32).reshape(1, shape[1], 1, 1)
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
        raise argparse.ArgumentTypeError("shape must be B,N,H,D")
    return parts  # type: ignore[return-value]


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    for shape in args.shape:
        for workload in args.workload:
            x = make_workload(shape, workload, device, args.seed)
            packed = quantize_pack_q6_g128(x, seed=args.seed)
            y = decode_q6_g128(packed)
            report = packed.storage_report()
            pack_us = _time_callable(device, args.repeats, lambda: quantize_pack_q6_g128(x, seed=args.seed))
            # Decode the same packed tensor to isolate read/decode path.
            decode_us = _time_callable(device, args.repeats, lambda: decode_q6_g128(packed))
            rows.append(
                {
                    "variant": "q6_g128_block_absmax_stochastic_beta0_packed",
                    "shape": "x".join(str(v) for v in shape),
                    "workload": workload,
                    "device": str(device),
                    "num_values": report["num_values"],
                    "payload_bytes": report["payload_bytes"],
                    "scale_bytes": report["scale_bytes"],
                    "total_bytes": report["total_bytes"],
                    "effective_bytes_per_value": report["effective_bytes_per_value"],
                    "compression_vs_bf16": report["compression_vs_bf16"],
                    "compression_vs_fp32": report["compression_vs_fp32"],
                    "pack_us": pack_us,
                    "decode_us": decode_us,
                    **error_metrics(x, y),
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
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(1, 2048, 8, 64), (1, 8192, 8, 64)]
    if not args.workload:
        args.workload = ["gaussian", "nonstationary", "adversarial"]
    rows = run_benchmark(args)
    write_rows(rows, args.out_prefix)
    print(f"wrote {len(rows)} q6_g128 packed prototype rows to {args.out_prefix}.json/.tsv")
    bad = [r for r in rows if not r["finite"]]
    if bad:
        raise SystemExit(f"non-finite decode rows: {len(bad)}")


if __name__ == "__main__":
    main()
