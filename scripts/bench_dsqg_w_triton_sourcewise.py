from __future__ import annotations

import json
import os
from contextlib import contextmanager

import torch

from kernels.dsqg_w.dsqg_w_mvp import CandidateProvider, DSQGWBlock, DSQGWConfig


@contextmanager
def env_flag(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def build_inputs(batch: int = 1, seq_len: int = 256):
    torch.manual_seed(20260701)
    device = torch.device("cuda")
    config = DSQGWConfig(
        d=512,
        n_heads=8,
        max_candidates=16,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        gate_init=-2.0,
        fuse_init_std=0.02,
        use_query_type_bias=True,
    )
    x = torch.randn(batch, seq_len, config.d, device=device)
    l3 = torch.randn(batch, seq_len, config.d, device=device)
    positions = torch.arange(seq_len, device=device)
    question = torch.tensor([[0, 3, 7, 11]], device=device, dtype=torch.long).expand(batch, -1).contiguous()
    hisa = torch.stack(
        [
            (positions - 1).clamp_min(0),
            (positions - 8).clamp_min(0),
            (positions - 32).clamp_min(0),
            (positions - 128).clamp_min(0),
        ],
        dim=-1,
    ).unsqueeze(0).expand(batch, -1, -1).contiguous()
    hisa_scores = torch.linspace(-0.5, 0.5, steps=batch * seq_len * config.k_hisa_evidence, device=device).reshape(
        batch, seq_len, config.k_hisa_evidence
    )
    l3_skip = torch.stack(
        [(positions - 16).clamp_min(0), (positions - 64).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(batch, -1, -1).contiguous()
    return config, x, l3, question, hisa, hisa_scores, l3_skip


def build_blocks(config: DSQGWConfig):
    dense_block = DSQGWBlock.from_config(config).cuda().eval()
    eager_block = DSQGWBlock.from_config(config).cuda().eval()
    triton_block = DSQGWBlock.from_config(config).cuda().eval()
    eager_block.load_state_dict(dense_block.state_dict())
    triton_block.load_state_dict(dense_block.state_dict())
    return dense_block, eager_block, triton_block


def bench_forward(name: str, fn, *, warmup: int = 5, iters: int = 20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return {
        "name": name,
        "ms": start.elapsed_time(end) / iters,
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "iters": iters,
        "warmup": warmup,
    }


def bench_fwd_bwd(name: str, fn, *, warmup: int = 1, iters: int = 5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return {
        "name": name,
        "ms": start.elapsed_time(end) / iters,
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "iters": iters,
        "warmup": warmup,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"Triton unavailable: {exc}") from exc

    config, x, l3, question, hisa, hisa_scores, l3_skip = build_inputs()
    provider = CandidateProvider(config)
    dense_candidates = provider.build(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    metadata = provider.build_metadata(
        x,
        l3_states=l3,
        question_indices=question,
        hisa_evidence_indices=hisa,
        hisa_evidence_scores=hisa_scores,
        l3_skip_indices=l3_skip,
    )
    dense_block, eager_block, triton_block = build_blocks(config)

    def dense_forward(return_routing: bool = False):
        return dense_block(
            x,
            dense_candidates.cand_states,
            dense_candidates.cand_types,
            dense_candidates.cand_sources,
            dense_candidates.cand_mask,
            cand_scores=dense_candidates.cand_scores,
            return_routing=return_routing,
        )

    def eager_forward(return_routing: bool = False):
        return eager_block.forward_sourcewise(
            x,
            metadata.cand_token_indices,
            metadata.cand_types,
            metadata.cand_sources,
            metadata.cand_mask,
            l3_states=l3,
            cand_scores=metadata.cand_scores,
            return_routing=return_routing,
        )

    def triton_forward(return_routing: bool = False):
        with env_flag("DWARF_DSQG_W_TRITON_SOURCEWISE", "1"):
            return triton_block.forward_sourcewise(
                x,
                metadata.cand_token_indices,
                metadata.cand_types,
                metadata.cand_sources,
                metadata.cand_mask,
                l3_states=l3,
                cand_scores=metadata.cand_scores,
                return_routing=return_routing,
            )

    def dense_fwd_bwd():
        dense_block.zero_grad(set_to_none=True)
        bx = x.detach().clone().requires_grad_(True)
        bl3 = l3.detach().clone().requires_grad_(True)
        candidates = provider.build(
            bx,
            l3_states=bl3,
            question_indices=question,
            hisa_evidence_indices=hisa,
            hisa_evidence_scores=hisa_scores,
            l3_skip_indices=l3_skip,
        )
        out, _ = dense_block(
            bx,
            candidates.cand_states,
            candidates.cand_types,
            candidates.cand_sources,
            candidates.cand_mask,
            cand_scores=candidates.cand_scores,
        )
        out.float().square().mean().backward()

    def eager_fwd_bwd():
        eager_block.zero_grad(set_to_none=True)
        bx = x.detach().clone().requires_grad_(True)
        bl3 = l3.detach().clone().requires_grad_(True)
        candidates = provider.build_metadata(
            bx,
            l3_states=bl3,
            question_indices=question,
            hisa_evidence_indices=hisa,
            hisa_evidence_scores=hisa_scores,
            l3_skip_indices=l3_skip,
        )
        out, _ = eager_block.forward_sourcewise(
            bx,
            candidates.cand_token_indices,
            candidates.cand_types,
            candidates.cand_sources,
            candidates.cand_mask,
            l3_states=bl3,
            cand_scores=candidates.cand_scores,
        )
        out.float().square().mean().backward()

    def triton_fwd_bwd():
        triton_block.zero_grad(set_to_none=True)
        bx = x.detach().clone().requires_grad_(True)
        bl3 = l3.detach().clone().requires_grad_(True)
        candidates = provider.build_metadata(
            bx,
            l3_states=bl3,
            question_indices=question,
            hisa_evidence_indices=hisa,
            hisa_evidence_scores=hisa_scores,
            l3_skip_indices=l3_skip,
        )
        with env_flag("DWARF_DSQG_W_TRITON_SOURCEWISE", "1"):
            out, _ = triton_block.forward_sourcewise(
                bx,
                candidates.cand_token_indices,
                candidates.cand_types,
                candidates.cand_sources,
                candidates.cand_mask,
                l3_states=bl3,
                cand_scores=candidates.cand_scores,
            )
        out.float().square().mean().backward()

    with torch.no_grad():
        dense_out, dense_tel = dense_forward(return_routing=True)
        eager_out, eager_tel = eager_forward(return_routing=True)
        triton_out, triton_tel = triton_forward(return_routing=True)
        triton_no_route_out, triton_no_route_tel = triton_forward(return_routing=False)
        torch.cuda.synchronize()

    result = {
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "B": x.shape[0],
            "N": x.shape[1],
            "D": config.d,
            "H": config.n_heads,
            "HD": config.d // config.n_heads,
            "J": metadata.cand_mask.shape[-1],
        },
        "features": {
            "sourcewise": True,
            "triton_full_recompute_backward": False,
            "triton_compact_read_backward": True,
            "triton_no_routing": True,
            "triton_compact_read_slots": True,
            "local_offsets": list(config.local_offsets),
            "long_offsets": list(config.long_offsets),
            "k_question": config.k_question,
            "k_hisa_evidence": config.k_hisa_evidence,
            "k_l3_skip": config.k_l3_skip,
            "query_type_bias": config.use_query_type_bias,
        },
        "parity": {
            "eager_vs_dense_out_max_abs_diff": float((eager_out - dense_out).abs().max().item()),
            "triton_vs_eager_out_max_abs_diff": float((triton_out - eager_out).abs().max().item()),
            "triton_no_route_vs_route_out_max_abs_diff": float((triton_no_route_out - triton_out).abs().max().item()),
            "triton_vs_dense_out_max_abs_diff": float((triton_out - dense_out).abs().max().item()),
            "triton_vs_eager_probs_max_abs_diff": float(
                (triton_tel["dsqg_w_probs"] - eager_tel["dsqg_w_probs"]).abs().max().item()
            ),
            "triton_read_norm": float(triton_tel["dsqg_w_read_norm"].item()),
            "eager_read_norm": float(eager_tel["dsqg_w_read_norm"].item()),
            "dense_read_norm": float(dense_tel["dsqg_w_read_norm"].item()),
        },
        "materialization": {
            "metadata_candidate_state_numel": int(metadata.cand_states.numel()),
            "dense_candidate_state_bytes": int(dense_candidates.cand_states.numel() * dense_candidates.cand_states.element_size()),
            "forbidden_projected_kv_surface_bytes_each": int(
                x.shape[0]
                * x.shape[1]
                * metadata.cand_mask.shape[-1]
                * config.n_heads
                * (config.d // config.n_heads)
                * x.element_size()
            ),
            "triton_no_route_probs_materialized": float(triton_no_route_tel["dsqg_w_triton_probs_materialized"].item()),
            "triton_read_accum_materialized": float(triton_no_route_tel["dsqg_w_triton_read_accum_materialized"].item()),
            "triton_read_mix_fused": float(triton_no_route_tel["dsqg_w_triton_read_mix_fused"].item()),
            "triton_compact_read_slots_materialized": float(
                triton_no_route_tel["dsqg_w_triton_compact_read_slots_materialized"].item()
            ),
            "triton_compact_read_slots": float(triton_no_route_tel["dsqg_w_triton_compact_read_slots"].item()),
            "triton_score_recompute_blocks": float(triton_no_route_tel["dsqg_w_triton_score_recompute_blocks"].item()),
        },
        "forward_timings": [
            bench_forward("dense_forward_no_routing", lambda: dense_forward(False)),
            bench_forward("eager_sourcewise_forward_no_routing", lambda: eager_forward(False)),
            bench_forward("triton_sourcewise_forward_no_routing", lambda: triton_forward(False)),
            bench_forward("triton_sourcewise_forward_with_routing", lambda: triton_forward(True), warmup=2, iters=5),
        ],
        "fwd_bwd_timings": [
            bench_fwd_bwd("dense_fwd_bwd", dense_fwd_bwd),
            bench_fwd_bwd("eager_sourcewise_fwd_bwd", eager_fwd_bwd),
            bench_fwd_bwd("triton_sourcewise_fwd_bwd", triton_fwd_bwd),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
