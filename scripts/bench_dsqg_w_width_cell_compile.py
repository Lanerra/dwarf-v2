#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateSource, CandidateType, DSQGWWidthCell


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_inputs(batch: int, seq_len: int, j: int, d: int, dtype: torch.dtype):
    device = torch.device("cuda")
    torch.manual_seed(20260705)
    cand_states = torch.randn(batch, seq_len, j, d, device=device, dtype=dtype, requires_grad=True)
    pattern_types = torch.tensor(
        [
            CandidateType.QUESTION,
            CandidateType.QUESTION,
            CandidateType.QUESTION,
            CandidateType.QUESTION,
            CandidateType.HISA_EVIDENCE_REP0,
            CandidateType.HISA_EVIDENCE_REP1,
            CandidateType.HISA_EVIDENCE_REP2,
            CandidateType.HISA_EVIDENCE_REP3,
            CandidateType.L3_SKIP,
            CandidateType.L3_SKIP,
            CandidateType.NULL,
        ],
        device=device,
        dtype=torch.long,
    )[:j]
    pattern_sources = torch.tensor(
        [
            CandidateSource.FINAL,
            CandidateSource.FINAL,
            CandidateSource.FINAL,
            CandidateSource.FINAL,
            CandidateSource.HISA,
            CandidateSource.HISA,
            CandidateSource.HISA,
            CandidateSource.HISA,
            CandidateSource.L3,
            CandidateSource.L3,
            CandidateSource.NULL,
        ],
        device=device,
        dtype=torch.long,
    )[:j]
    cand_types = pattern_types.view(1, 1, j).expand(batch, seq_len, j).contiguous()
    cand_sources = pattern_sources.view(1, 1, j).expand(batch, seq_len, j).contiguous()
    cand_mask = cand_types != int(CandidateType.NULL)
    return cand_states, cand_types, cand_sources, cand_mask


def run_case(name: str, module: torch.nn.Module, inputs, warmup: int, iters: int):
    times = []
    peak = 0
    for step in range(warmup + iters):
        cand_states, cand_types, cand_sources, cand_mask = inputs
        if cand_states.grad is not None:
            cand_states.grad = None
        for p in module.parameters():
            p.grad = None
        torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        out, telemetry = module(cand_states, cand_types, cand_sources, cand_mask)
        loss = out.square().mean() + telemetry["dsqg_w_width_aux_loss"] * 0.001
        loss.backward()
        sync(); dt = (time.perf_counter() - t0) * 1000.0
        if step >= warmup:
            times.append(dt)
            peak = max(peak, torch.cuda.max_memory_allocated())
    return {"case": name, "mean_ms": sum(times)/len(times), "min_ms": min(times), "peak_mb": peak/1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--j", type=int, default=11)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("runs/profiles/width_compile_bench.json"))
    args = ap.parse_args()
    dtype = torch.bfloat16
    inputs = make_inputs(args.batch, args.seq_len, args.j, args.d, dtype)
    eager = DSQGWWidthCell(d=args.d, n_heads=args.heads, n_types=11, n_sources=6, bottleneck=args.width, gate_init=-1.5, entropy_floor=1.5, entropy_weight=0.25).cuda().to(dtype=dtype)
    compiled = DSQGWWidthCell(d=args.d, n_heads=args.heads, n_types=11, n_sources=6, bottleneck=args.width, gate_init=-1.5, entropy_floor=1.5, entropy_weight=0.25).cuda().to(dtype=dtype)
    compiled.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled, mode="reduce-overhead", dynamic=False, fullgraph=False)
    rows = [run_case("eager", eager, inputs, args.warmup, args.iters), run_case("compiled", compiled, inputs, args.warmup, args.iters)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(args.out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
