#!/usr/bin/env python3
"""Numerically compare canonical and overlay pure-DSQG attention kernels.

The harness runs the real D512/H8/HD64/N2048 DSQG attention geometry with
identical state and inputs. It is intentionally a parity gate, not a speed
benchmark: timing candidates must pass this script before trainer probes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _load_kernel(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import kernel: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _attention(module: Any, *, device: torch.device, seq_len: int) -> torch.nn.Module:
    offsets = list(module.ALL_OFFSETS[:32])
    j_small = sum(offset <= 28 for offset in offsets)
    j_large = sum(offset >= 48 for offset in offsets)
    if j_small + j_large != len(offsets):
        raise ValueError(f"unsupported offset split: small={j_small}, large={j_large}, offsets={len(offsets)}")
    return module.DSQGAttentionV19(
        embedding_dim=512,
        num_heads=8,
        offsets=offsets,
        j_small=j_small,
        j_large=j_large,
        seq_len=seq_len,
        dropout=0.0,
    ).to(device).train()


def _max_abs_and_rel(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return float(delta.max().item()), float((delta / denom).max().item())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DSQG Triton parity")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline_module = _load_kernel(args.baseline_kernel, "dsqg_kernel_baseline")
    candidate_module = _load_kernel(args.candidate_kernel, "dsqg_kernel_candidate")
    baseline = _attention(baseline_module, device=device, seq_len=args.seq_len)
    candidate = _attention(candidate_module, device=device, seq_len=args.seq_len)
    candidate.load_state_dict(baseline.state_dict(), strict=True)

    x = torch.randn(args.batch_size, args.seq_len, 512, device=device, dtype=torch.bfloat16)
    x_baseline = x.detach().clone().requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        baseline_out = baseline(x_baseline)
        candidate_out = candidate(x_candidate)
        baseline_loss = baseline_out.float().square().mean()
        candidate_loss = candidate_out.float().square().mean()
    baseline_loss.backward()
    candidate_loss.backward()
    torch.cuda.synchronize(device)

    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "output": (candidate_out.detach(), baseline_out.detach()),
        "input_grad": (x_candidate.grad.detach(), x_baseline.grad.detach()),
    }
    for name, baseline_parameter in baseline.named_parameters():
        candidate_parameter = dict(candidate.named_parameters())[name]
        if baseline_parameter.grad is None or candidate_parameter.grad is None:
            continue
        tensors[f"grad:{name}"] = (candidate_parameter.grad.detach(), baseline_parameter.grad.detach())

    checks: dict[str, dict[str, float | bool]] = {}
    passed = True
    for name, (actual, expected) in tensors.items():
        max_abs, max_rel = _max_abs_and_rel(actual, expected)
        allclose = bool(torch.allclose(actual.float(), expected.float(), atol=args.atol, rtol=args.rtol))
        checks[name] = {"max_abs": max_abs, "max_rel": max_rel, "allclose": allclose}
        passed = passed and allclose
    result = {
        "passed": passed,
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "atol": args.atol,
        "rtol": args.rtol,
        "checks": checks,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("kernel parity failed")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--candidate-kernel", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
