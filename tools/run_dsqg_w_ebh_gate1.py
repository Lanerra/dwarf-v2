#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.dsqg_w.dsqg_w_mvp import (
    CandidateProvider,
    CandidateSource,
    CandidateType,
    DSQGWBlock,
    DSQGWConfig,
    DSQGWEvidenceBindingHub,
)


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _make_metadata(batch: int, seq_len: int, k: int, d_model: int, n_heads: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(710)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    l3 = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
    cfg = DSQGWConfig(
        d=d_model,
        n_heads=n_heads,
        max_candidates=k,
        local_offsets=(1, 2, 4, 8),
        long_offsets=(16, 32, 64, 128, 256, 512, 1024, 2048),
        k_question=4,
        k_hisa_evidence=min(8, max(1, k // 3)),
        k_chunk=4,
        k_l3_skip=4,
        typed_hisa_reps=True,
    )
    positions = torch.arange(seq_len, device=device)
    question = torch.stack(
        [torch.linspace(0, max(seq_len - 1, 0), steps=4, device=device).round().long() for _ in range(batch)],
        dim=0,
    )
    hisa_cols = []
    for offset in range(1, cfg.k_hisa_evidence + 1):
        hisa_cols.append((positions - offset).clamp_min(0))
    hisa = torch.stack(hisa_cols, dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
    hisa_scores = torch.linspace(-1.0, 1.0, cfg.k_hisa_evidence, device=device, dtype=dtype).reshape(1, 1, -1)
    hisa_scores = hisa_scores.expand(batch, seq_len, -1).contiguous()
    chunk_rep = ((positions // 16) * 16).clamp(max=max(seq_len - 1, 0)).reshape(1, seq_len, 1).expand(batch, -1, cfg.k_chunk)
    l3_skip = torch.stack([(positions - 4 - i).clamp_min(0) for i in range(cfg.k_l3_skip)], dim=-1)
    l3_skip = l3_skip.unsqueeze(0).expand(batch, -1, -1).contiguous()
    provider = CandidateProvider(cfg)
    metadata = provider.build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        chunk_rep_indices=chunk_rep,
        l3_skip_indices=l3_skip,
    )
    cand_states = provider._build_vectorized(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        chunk_rep_indices=chunk_rep,
        l3_skip_indices=l3_skip,
        materialize_states=True,
    ).cand_states
    return cfg, x, l3, metadata, cand_states


def _run_forward_ms(hub: DSQGWEvidenceBindingHub, x, cand_states, cand_types, cand_sources, cand_mask, distances, scores, iters: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            hub(x, cand_states, cand_types, cand_sources, cand_mask, candidate_distances=distances, cand_scores=scores)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            hub(x, cand_states, cand_types, cand_sources, cand_mask, candidate_distances=distances, cand_scores=scores)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-1 DSQG-W Evidence Binding Hub probe")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.d_model % args.n_heads != 0:
        raise ValueError("d-model must be divisible by n-heads")

    cfg, x, l3, metadata, cand_states = _make_metadata(args.batch, args.seq_len, args.k, args.d_model, args.n_heads, device, dtype)
    distances = metadata.candidate_distances
    scores = metadata.cand_scores
    if distances is None:
        positions = torch.arange(args.seq_len, device=device).reshape(1, args.seq_len, 1)
        distances = (positions - metadata.cand_token_indices.clamp_min(0)).clamp_min(0).to(dtype)

    hub = DSQGWEvidenceBindingHub(
        d=args.d_model,
        n_types=len(CandidateType),
        n_sources=len(CandidateSource),
        bottleneck=args.bottleneck,
        gate_init=-4.0,
        phase_bands=4,
        use_score_features=True,
    ).to(device=device, dtype=dtype).eval()

    y, telemetry, aux = hub(
        x,
        cand_states,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    perm = torch.randperm(metadata.cand_mask.shape[-1], device=device)
    y_perm, _, aux_perm = hub(
        x,
        cand_states[:, :, perm],
        metadata.cand_types[:, :, perm],
        metadata.cand_sources[:, :, perm],
        metadata.cand_mask[:, :, perm],
        candidate_distances=distances[:, :, perm],
        cand_scores=scores[:, :, perm] if scores is not None else None,
        return_aux=True,
    )
    shuffled_states = cand_states.flip(dims=(2,))
    y_shuf, _, aux_shuf = hub(
        x,
        shuffled_states,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    zero_mask = torch.zeros_like(metadata.cand_mask)
    y_zero, zero_telemetry, _ = hub(
        x,
        cand_states,
        torch.full_like(metadata.cand_types, int(CandidateType.NULL)),
        torch.full_like(metadata.cand_sources, int(CandidateSource.NULL)),
        zero_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )

    grad_states = cand_states.detach().clone().requires_grad_(True)
    train_hub = DSQGWEvidenceBindingHub(
        d=args.d_model,
        n_types=len(CandidateType),
        n_sources=len(CandidateSource),
        bottleneck=args.bottleneck,
        gate_init=-2.0,
        phase_bands=4,
        use_score_features=True,
    ).to(device=device, dtype=dtype)
    y_grad, _, aux_grad = train_hub(
        x.detach(),
        grad_states,
        metadata.cand_types,
        metadata.cand_sources,
        metadata.cand_mask,
        candidate_distances=distances,
        cand_scores=scores,
        return_aux=True,
    )
    grad_loss = y_grad.float().square().mean() + aux_grad["bound_packet"].float().square().mean()
    grad_loss.backward()

    scaling = []
    for k_scale in [4, 8, 16, 32, 64]:
        if k_scale > args.k:
            continue
        ms = _run_forward_ms(
            hub,
            x,
            cand_states[:, :, :k_scale],
            metadata.cand_types[:, :, :k_scale],
            metadata.cand_sources[:, :, :k_scale],
            metadata.cand_mask[:, :, :k_scale],
            distances[:, :, :k_scale],
            scores[:, :, :k_scale] if scores is not None else None,
            args.iters,
            args.warmup,
        )
        scaling.append({"k": k_scale, "forward_ms": ms, "tok_s": float(args.batch * args.seq_len / (ms / 1000.0))})

    sourcewise_cfg = DSQGWConfig(
        d=args.d_model,
        n_heads=args.n_heads,
        max_candidates=args.k,
        bottleneck=args.bottleneck,
        use_evidence_binding_hub=True,
        ebh_bottleneck=args.bottleneck,
        ebh_gate_init=-4.0,
        typed_hisa_reps=True,
    )
    block = DSQGWBlock.from_config(sourcewise_cfg).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        source_out, source_telemetry = block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=scores,
            candidate_distances=distances,
        )

    grad_norms = {
        name: float(param.grad.detach().float().norm().cpu().item())
        for name, param in train_hub.named_parameters()
        if param.grad is not None
    }
    type_mass = {
        ctype.name: float(telemetry.get(f"dsqg_w_ebh_{ctype.name.lower()}_mass", torch.tensor(0.0)).detach().float().cpu().item())
        for ctype in CandidateType
    }
    source_mass = {
        source.name: float(telemetry.get(f"dsqg_w_ebh_{source.name.lower()}_source_mass", torch.tensor(0.0)).detach().float().cpu().item())
        for source in CandidateSource
    }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "shape": {"batch": args.batch, "seq_len": args.seq_len, "k": args.k, "d_model": args.d_model},
        "complexity_contract": "O(B*T*K*D) candidate-local + O(B*T*(types+sources)*D) lane reductions; no O(K^2) candidate-candidate mixing.",
        "permutation_output_max_abs_diff": float((y - y_perm).detach().float().abs().max().cpu().item()),
        "permutation_packet_max_abs_diff": float((aux["bound_packet"] - aux_perm["bound_packet"]).detach().float().abs().max().cpu().item()),
        "shuffled_evidence_packet_mean_abs_delta": float((aux["bound_packet"] - aux_shuf["bound_packet"]).detach().float().abs().mean().cpu().item()),
        "shuffled_evidence_output_mean_abs_delta": float((y - y_shuf).detach().float().abs().mean().cpu().item()),
        "zero_candidate_identity_max_abs_delta": float((y_zero - x).detach().float().abs().max().cpu().item()),
        "zero_candidate_active_row_fraction": float(zero_telemetry["dsqg_w_ebh_active_row_fraction"].detach().float().cpu().item()),
        "candidate_state_grad_norm": float(grad_states.grad.detach().float().norm().cpu().item()),
        "grad_norms": grad_norms,
        "type_mass": type_mass,
        "source_mass": source_mass,
        "scaling": scaling,
        "sourcewise_output_finite": bool(torch.isfinite(source_out).all().item()),
        "sourcewise_ebh_materialized": float(source_telemetry["dsqg_w_sourcewise_ebh_materialized"].detach().float().cpu().item()),
        "telemetry": {k: float(v.detach().float().cpu().item()) for k, v in telemetry.items() if isinstance(v, torch.Tensor) and v.numel() == 1},
    }

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / f"dsqg_w_ebh_gate1_{stamp}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")

    md = [
        "# DSQG-W EBH Gate-1 Probe",
        "",
        "Gate-1 synthetic/current-candidate tensor probe; not a trainer or quality claim.",
        "",
        f"- Device: `{summary['device']}` / {summary['gpu_name']}",
        f"- Dtype: `{summary['dtype']}`",
        f"- Shape: `{summary['shape']}`",
        f"- Complexity: {summary['complexity_contract']}",
        "",
        "## Core checks",
        "",
        f"- Permutation output max abs diff: `{summary['permutation_output_max_abs_diff']:.6e}`",
        f"- Permutation packet max abs diff: `{summary['permutation_packet_max_abs_diff']:.6e}`",
        f"- Shuffled evidence packet mean abs delta: `{summary['shuffled_evidence_packet_mean_abs_delta']:.6e}`",
        f"- Shuffled evidence output mean abs delta: `{summary['shuffled_evidence_output_mean_abs_delta']:.6e}`",
        f"- Zero-candidate identity max abs delta: `{summary['zero_candidate_identity_max_abs_delta']:.6e}`",
        f"- Candidate-state grad norm: `{summary['candidate_state_grad_norm']:.6e}`",
        f"- Sourcewise EBH materialized: `{summary['sourcewise_ebh_materialized']}`",
        "",
        "## Scaling",
        "",
        "| K | forward ms | tok/s |",
        "|---:|---:|---:|",
    ]
    for row in scaling:
        md.append(f"| {row['k']} | {row['forward_ms']:.4f} | {row['tok_s']:.1f} |")
    md.extend(["", "## Type mass", ""])
    for name, value in type_mass.items():
        md.append(f"- {name}: `{value:.6f}`")
    md.extend(["", "## Source mass", ""])
    for name, value in source_mass.items():
        md.append(f"- {name}: `{value:.6f}`")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["perm_out", "perm_packet", "shuffle_packet", "shuffle_out", "zero_identity"],
            [
                summary["permutation_output_max_abs_diff"],
                summary["permutation_packet_max_abs_diff"],
                summary["shuffled_evidence_packet_mean_abs_delta"],
                summary["shuffled_evidence_output_mean_abs_delta"],
                summary["zero_candidate_identity_max_abs_delta"],
            ],
        )
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title("EBH Gate-1 core metrics")
        fig.tight_layout()
        fig.savefig(out_dir / "gate1_core_metrics.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["k"] for r in scaling], [r["forward_ms"] for r in scaling], marker="o")
        ax.set_xlabel("K candidates")
        ax.set_ylabel("forward ms")
        ax.set_title("EBH selected-candidate scaling")
        fig.tight_layout()
        fig.savefig(out_dir / "gate1_scaling.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(list(type_mass.keys()), list(type_mass.values()))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("EBH type lane mass")
        fig.tight_layout()
        fig.savefig(out_dir / "gate1_lane_mass.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "figure_error.txt").write_text(str(exc) + "\n")

    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, sort_keys=True, default=_json_safe))


if __name__ == "__main__":
    main()
