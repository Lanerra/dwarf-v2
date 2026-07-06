#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.dsqg_w.dsqg_w_mvp import CandidateProvider, DSQGWBlock, DSQGWConfig


@contextmanager
def set_env(overrides: dict[str, str | None]):
    old = {k: os.environ.get(k) for k in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def scalarize(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return [float(x) for x in value.detach().flatten()[:16].cpu().float().tolist()]
    if isinstance(value, dict):
        return {str(k): scalarize(v) for k, v in value.items()}
    return value


def allopen_config(
    *,
    width: bool,
    typed: bool,
    qtb: bool = True,
    prior: bool = True,
    ebh: bool = False,
    ebh_bottleneck: int = 64,
    ebh_score_features: bool = True,
) -> DSQGWConfig:
    return DSQGWConfig(
        d=512,
        n_heads=8,
        bottleneck=64,
        max_candidates=16,
        gate_init=-1.5,
        fuse_init_std=0.02,
        local_offsets=(),
        long_offsets=(),
        k_question=4,
        k_hisa_evidence=4,
        k_l3_skip=2,
        k_chunk=0,
        null_fallback=True,
        use_width_cell=width,
        width_bottleneck=64,
        width_gate_init=-1.5,
        width_entropy_floor=1.5,
        width_entropy_weight=0.25,
        use_typed_mixer=typed,
        typed_mixer_bottleneck=64,
        typed_mixer_gate_init=-1.5,
        use_query_type_bias=qtb,
        typed_hisa_reps=True,
        use_evidence_prior=prior,
        evidence_prior_clip=2.0,
        evidence_prior_init_scale=0.0,
        use_candidate_quotas=True,
        quota_hisa_max=4,
        use_evidence_binding_hub=ebh,
        ebh_bottleneck=ebh_bottleneck,
        ebh_gate_init=-2.0,
        ebh_phase_bands=3,
        ebh_score_features=ebh_score_features,
    )


def build_inputs(batch: int, seq_len: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(20260705)
    x = torch.randn(batch, seq_len, 512, device=device, dtype=dtype)
    l3 = torch.randn(batch, seq_len, 512, device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device)
    # Same shape family as the all-open trainer: four fixed question/cue slots,
    # four D/HISA selected-token slots, two L3 skip slots.
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
    # Non-constant score field, centered later by the DSQG-W block.
    score_base = torch.linspace(-0.5, 0.5, steps=seq_len * 4, device=device, dtype=torch.float32).reshape(1, seq_len, 4)
    hisa_scores = score_base.expand(batch, -1, -1).contiguous()
    l3_skip = torch.stack(
        [(positions - 16).clamp_min(0), (positions - 64).clamp_min(0)], dim=-1
    ).unsqueeze(0).expand(batch, -1, -1).contiguous()
    return x, l3, question, hisa, hisa_scores, l3_skip


def build_case(case: str) -> tuple[DSQGWConfig, dict[str, str | None], str]:
    env = {
        "DWARF_PROFILE_DSQG_W": "1",
        "DWARF_DSQG_W_FAST_TELEMETRY": "0",
        "DWARF_DSQG_W_TRITON_SOURCEWISE": "1",
        "DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD": "triton",
    }
    if case == "allopen_materialized":
        return allopen_config(width=True, typed=True), env, "sourcewise"
    if case == "allopen_fast_telemetry":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        return allopen_config(width=True, typed=True), env, "sourcewise"
    if case == "projected_width_trainable_control":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL"] = "1"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_BIAS_SCALE"] = "3.0"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_DETACH"] = "0"
        return allopen_config(width=True, typed=True), env, "sourcewise"
    if case == "projected_width_detached_control":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL"] = "1"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_BIAS_SCALE"] = "3.0"
        env["DWARF_DSQG_W_PROJECTED_WIDTH_DETACH"] = "1"
        return allopen_config(width=True, typed=True), env, "sourcewise"
    if case == "no_width_no_typed_triton":
        return allopen_config(width=False, typed=False), env, "sourcewise"
    if case == "width_only_materialized":
        return allopen_config(width=True, typed=False), env, "sourcewise"
    if case == "typed_only_materialized":
        return allopen_config(width=False, typed=True), env, "sourcewise"
    if case == "current_w_no_ebh_final_like":
        return allopen_config(width=False, typed=False), env, "sourcewise"
    if case == "current_w_width_typed_no_ebh":
        return allopen_config(width=True, typed=True), env, "sourcewise"
    if case == "ebh_materialized_oracle":
        return allopen_config(width=False, typed=False, ebh=True), env, "sourcewise"
    if case == "ebh_width_typed_materialized_oracle":
        return allopen_config(width=True, typed=True, ebh=True), env, "sourcewise"
    if case == "ebh_materialized_fast_telemetry":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        return allopen_config(width=False, typed=False, ebh=True), env, "sourcewise"
    if case == "ebh_materialized_no_score_features":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        return allopen_config(width=False, typed=False, ebh=True, ebh_score_features=False), env, "sourcewise"
    if case == "ebh_materialized_bottleneck32":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        return allopen_config(width=False, typed=False, ebh=True, ebh_bottleneck=32), env, "sourcewise"
    if case == "ebh_materialized_bottleneck128":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        return allopen_config(width=False, typed=False, ebh=True, ebh_bottleneck=128), env, "sourcewise"
    if case == "ebh_packet_sourcewise":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"] = "1"
        return allopen_config(width=False, typed=False, ebh=True), env, "sourcewise"
    if case == "ebh_packet_no_score_features":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"] = "1"
        return allopen_config(width=False, typed=False, ebh=True, ebh_score_features=False), env, "sourcewise"
    if case == "ebh_packet_bottleneck32":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"] = "1"
        return allopen_config(width=False, typed=False, ebh=True, ebh_bottleneck=32), env, "sourcewise"
    if case == "ebh_packet_width_typed_approx":
        env["DWARF_DSQG_W_FAST_TELEMETRY"] = "1"
        env["DWARF_DSQG_W_EBH_SOURCEWISE_PACKET"] = "1"
        return allopen_config(width=True, typed=True, ebh=True), env, "sourcewise"
    if case == "allopen_dense_materialized_build":
        env["DWARF_DSQG_W_TRITON_SOURCEWISE"] = "0"
        return allopen_config(width=True, typed=True), env, "dense"
    raise ValueError(f"unknown case {case!r}")


def tensor_accounting(batch: int, seq_len: int, j_count: int, d: int = 512, h: int = 8, read_slots: int = 12) -> dict[str, float]:
    bf16 = 2
    return {
        "candidate_states_mb_bf16": batch * seq_len * j_count * d * bf16 / 1024**2,
        "main_scores_or_probs_mb_bf16": batch * seq_len * j_count * h * bf16 / 1024**2,
        "width_pair_scores_or_probs_mb_bf16": batch * seq_len * j_count * j_count * bf16 / 1024**2,
        "compact_read_slots_mb_bf16": batch * seq_len * read_slots * d * bf16 / 1024**2,
    }


def run_once(
    *,
    block: DSQGWBlock,
    provider: CandidateProvider,
    mode: str,
    x: torch.Tensor,
    l3: torch.Tensor,
    question: torch.Tensor,
    hisa: torch.Tensor,
    hisa_scores: torch.Tensor,
    l3_skip: torch.Tensor,
    include_aux: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if mode == "dense":
        candidates = provider.build(
            x,
            l3_states=l3,
            question_indices=question,
            hisa_evidence_indices=hisa,
            hisa_evidence_scores=hisa_scores,
            l3_skip_indices=l3_skip,
        )
        out, telemetry = block(
            x,
            candidates.cand_states,
            candidates.cand_types,
            candidates.cand_sources,
            candidates.cand_mask,
            cand_scores=candidates.cand_scores,
            evidence_bits=candidates.evidence_bits,
            evidence_count=candidates.evidence_count,
            candidate_distances=candidates.candidate_distances,
        )
    else:
        candidates = provider.build_metadata(
            x,
            l3_states=l3,
            question_indices=question,
            hisa_evidence_indices=hisa,
            hisa_evidence_scores=hisa_scores,
            l3_skip_indices=l3_skip,
        )
        out, telemetry = block.forward_sourcewise(
            x,
            candidates.cand_token_indices,
            candidates.cand_types,
            candidates.cand_sources,
            candidates.cand_mask,
            l3_states=l3,
            cand_scores=candidates.cand_scores,
            evidence_bits=candidates.evidence_bits,
            evidence_count=candidates.evidence_count,
            candidate_distances=candidates.candidate_distances,
            needed_source_ids=candidates.active_source_ids,
        )
    loss = out.float().square().mean()
    if include_aux:
        aux = telemetry.get("dsqg_w_width_aux_loss")
        if torch.is_tensor(aux) and aux.requires_grad:
            loss = loss + aux.float() * 0.001
    return loss, telemetry


def time_case(
    case: str,
    *,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    profile: bool,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda")
    config, env, mode = build_case(case)
    x0, l30, question, hisa, hisa_scores, l3_skip = build_inputs(batch, seq_len, dtype, device)
    block = DSQGWBlock.from_config(config).to(device=device, dtype=dtype).train()
    provider = CandidateProvider(config)
    include_aux = config.use_width_cell

    def step() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        block.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        l3 = l30.detach().clone().requires_grad_(True)
        loss, telemetry = run_once(
            block=block,
            provider=provider,
            mode=mode,
            x=x,
            l3=l3,
            question=question,
            hisa=hisa,
            hisa_scores=hisa_scores,
            l3_skip=l3_skip,
            include_aux=include_aux,
        )
        loss.backward()
        return loss.detach(), telemetry

    with set_env(env):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        telemetry: dict[str, torch.Tensor] = {}
        losses: list[float] = []
        start.record()
        for _ in range(iters):
            loss, telemetry = step()
            losses.append(float(loss.cpu().item()))
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end) / max(iters, 1))
        peak_alloc_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
        peak_reserved_mb = float(torch.cuda.max_memory_reserved() / 1024**2)

        profiler_table = ""
        profiler_json = None
        if profile:
            trace_path = output_dir / f"{case}_torch_trace.json"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                step()
                torch.cuda.synchronize()
            profiler_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            prof.export_chrome_trace(str(trace_path))
            profiler_json = str(trace_path)

    tokens_per_step = batch * (seq_len - 1)
    tok_s_equiv = tokens_per_step / (elapsed_ms / 1000.0)
    telemetry_small = {
        k: scalarize(v)
        for k, v in telemetry.items()
        if k in {
            "dsqg_w_sourcewise_semantic_materialized",
            "dsqg_w_triton_sourcewise_semantic_bypass",
            "dsqg_w_sourcewise",
            "dsqg_w_triton_sourcewise",
            "dsqg_w_triton_compact_read_slots_materialized",
            "dsqg_w_triton_read_mix_fused",
            "dsqg_w_triton_compact_read_backward",
            "dsqg_w_batched_read_mix",
            "dsqg_w_valid_candidate_count",
            "dsqg_w_static_source_count",
            "dsqg_w_width_delta_norm",
            "dsqg_w_typed_mixer_delta_norm",
            "dsqg_w_delta_norm",
            "dsqg_w_projected_width_control",
            "dsqg_w_projected_width_bias_detached",
            "dsqg_w_projected_width_semantic_control",
            "dsqg_w_projected_width_bias_norm",
            "dsqg_w_typed_mixer_projected_bypass",
            "dsqg_w_ebh_enabled",
            "dsqg_w_ebh_delta_norm",
            "dsqg_w_ebh_bound_packet_norm",
            "dsqg_w_ebh_active_row_fraction",
            "dsqg_w_ebh_bind_gate_mean",
            "dsqg_w_sourcewise_ebh_materialized",
            "dsqg_w_ebh_packet_sourcewise",
            "dsqg_w_ebh_packet_triton",
            "dsqg_w_ebh_packet_semantic_approx",
        }
    }
    j_count = int(round(float(telemetry_small.get("dsqg_w_valid_candidate_count", config.k_question + config.k_hisa_evidence + config.k_l3_skip + 1))))
    return {
        "case": case,
        "mode": mode,
        "batch": batch,
        "seq_len": seq_len,
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "iters": iters,
        "ms_per_fwd_bwd": elapsed_ms,
        "tok_s_equiv_no_ga": tok_s_equiv,
        "tok_s_equiv_ga2_update_basis": tok_s_equiv * 2.0,
        "peak_alloc_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "loss_mean": sum(losses) / max(len(losses), 1),
        "telemetry": telemetry_small,
        "tensor_accounting": tensor_accounting(batch, seq_len, max(j_count, 1)),
        "profiler_table": profiler_table,
        "profiler_trace": profiler_json,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# DSQG-W All-Open Stage 1 Profile",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Shape: B={summary['batch']} N={summary['seq_len']} dtype={summary['dtype']}",
        "",
        "## Case Summary",
        "",
        "| case | ms fwd+bwd | tok/s equiv | GA2 basis | peak alloc MB | peak reserved MB | w_ebh_mat | EBH packet | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["cases"]:
        tel = row.get("telemetry", {})
        flags = ", ".join(f"{k}={v}" for k, v in tel.items() if "materialized" in k or "triton" in k or "sourcewise" in k or "approx" in k)
        w_ebh_mat = tel.get("dsqg_w_sourcewise_ebh_materialized", 0.0)
        ebh_packet = tel.get("dsqg_w_ebh_packet_sourcewise", 0.0)
        lines.append(
            f"| {row['case']} | {row['ms_per_fwd_bwd']:.3f} | {row['tok_s_equiv_no_ga']:.0f} | "
            f"{row['tok_s_equiv_ga2_update_basis']:.0f} | {row['peak_alloc_mb']:.1f} | {row['peak_reserved_mb']:.1f} | "
            f"{float(w_ebh_mat):.0f} | {float(ebh_packet):.0f} | `{flags}` |"
        )
    lines.extend(["", "## Profiler Tables", ""])
    for row in summary["cases"]:
        if row.get("profiler_table"):
            lines.extend([f"### {row['case']}", "", "```text", row["profiler_table"], "```", ""])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DSQG-W all-open sourcewise/materialized bottlenecks.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cases", default="current_w_no_ebh_final_like,current_w_width_typed_no_ebh,ebh_materialized_oracle,ebh_materialized_fast_telemetry,ebh_materialized_no_score_features,ebh_materialized_bottleneck32,ebh_materialized_bottleneck128,ebh_packet_sourcewise,ebh_packet_no_score_features,ebh_packet_bottleneck32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (ROOT / "runs" / "profiles" / f"dsqg_w_allopen_stage1_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    results = []
    for case in cases:
        print(f"[profile] case={case}", flush=True)
        results.append(
            time_case(
                case,
                batch=args.batch,
                seq_len=args.seq_len,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                profile=args.profile,
                output_dir=out_dir,
            )
        )
    summary = {
        "generated_at": dt.datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "cases": results,
    }
    json_path = out_dir / "profile_summary.json"
    md_path = out_dir / "profile_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_markdown(summary, md_path)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
