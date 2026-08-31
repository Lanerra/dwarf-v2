"""Strict-causal hierarchical sparse attention used by DWARF models.

The module combines a local causal lane with token-routed completed chunks.
FlexAttention implements the local lane on CUDA, and Triton implements the
irregular global lane. A PyTorch reference path supports CPU execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


def _install_repo_module_aliases() -> None:
    """Make repo-layout package and standalone import spellings identical."""
    source = Path(__file__).resolve()
    if source.parent.name != "kernels":
        return
    module = sys.modules[__name__]
    aliases = (
        "kernels.hierarchical_sparse_attn_v19_hisa",
        "hierarchical_sparse_attn_v19_hisa",
    )
    for alias in aliases:
        existing = sys.modules.get(alias)
        if existing is None:
            sys.modules[alias] = module
            continue
        if existing is module:
            continue
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == source:
            raise RuntimeError(
                "HISA V19 was loaded twice from the same source before module "
                "aliasing could be established"
            )
        raise RuntimeError(
            f"refusing to alias HISA V19 over a different module at {existing_file!r}"
        )


_install_repo_module_aliases()

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


_FLEX_COMPILE_MODE = os.getenv("DWARF_HISA_V19_FLEX_COMPILE_MODE", "default")
if _FLEX_COMPILE_MODE not in {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}:
    raise ValueError(
        "DWARF_HISA_V19_FLEX_COMPILE_MODE must be default, reduce-overhead, "
        "max-autotune, or max-autotune-no-cudagraphs"
    )
_COMPILED_FLEX_ATTENTION = (
    torch.compile(flex_attention, mode=_FLEX_COMPILE_MODE, dynamic=False)
    if _FLEX_ATTENTION_AVAILABLE
    else None
)

TRITON_DOT_MINIMUM_QUERY_BLOCK_SPECIALIZATION = "tl_dot_minimum_block_q_16"
_FLEX_EMPTY_ROW_POLICY = (
    "slice_strict_causal_empty_first_row_pending_cuda_backward_parity"
)


@torch.compiler.disable
def _isolated_flex_attention_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlexAttention outside the surrounding compiled autograd graph.

    PyTorch 2.12's AUTO FlexAttention selector can report NoValidChoices for
    valid BlockMask geometries, especially short/asymmetric Q/K lengths.  Pin
    the Triton backend, but keep physical tiles smaller than the 128-token
    sparse-mask granularity so FP32 fits Ada's CTA resource budget.
    """
    if _COMPILED_FLEX_ATTENTION is None:
        raise RuntimeError("compiled FlexAttention callable is unavailable")
    sparse_block = int(block_size)
    # BlockMask granularity and physical Triton attention tiles are independent.
    # A 128x128 physical tile at hd=64 exceeds Ada's per-CTA resource budget in
    # FP32 (~230 KiB requested on RTX 4090).  Keep the robust 128-token sparse
    # mask, but subdivide it into 64x64 physical tiles.  FlexAttention requires
    # the sparse Q/KV block sizes to be divisible by the physical tile sizes.
    tile = min(64, sparse_block)
    if sparse_block % tile != 0:
        raise RuntimeError(
            "HISA local Flex tile must divide the BlockMask granularity"
        )
    kernel_options = {
        "BACKEND": "TRITON",
        "BLOCK_M": tile,
        "BLOCK_N": tile,
        "bwd_BLOCK_M1": tile,
        "bwd_BLOCK_N1": tile,
        "bwd_BLOCK_M2": tile,
        "bwd_BLOCK_N2": tile,
        # Favor portability/resource headroom over an extra pipeline stage.
        "num_stages": 1,
        # Do not assert optional Flex mask fast-path contracts here.  Although
        # HISA's logical rows are nonempty and contiguous after slicing row 0,
        # PyTorch's sparse-block lowering has had version-specific edge cases
        # for asymmetric/non-multiple sequence geometry.  Defaults keep the
        # explicit safety checks until CUDA parity has established otherwise.
        "ROWS_GUARANTEED_SAFE": False,
        "BLOCKS_ARE_CONTIGUOUS": False,
    }
    if AuxRequest is not None:
        output, auxiliary = _COMPILED_FLEX_ATTENTION(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options=kernel_options,
            return_aux=AuxRequest(lse=True),
        )
        lse = auxiliary.lse
    else:
        output, lse = _COMPILED_FLEX_ATTENTION(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options=kernel_options,
            return_lse=True,
        )
    return output, lse



try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only development
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_TRITON_LIBRARY_API_AVAILABLE = bool(
    _TRITON_AVAILABLE
    and callable(getattr(torch.library, "triton_op", None))
    and callable(getattr(torch.library, "wrap_triton", None))
    and callable(getattr(torch.library, "register_autograd", None))
)
_GLOBAL_LANE_OPERATOR_MODE = (
    "triton_op_registered_autograd"
    if _TRITON_LIBRARY_API_AVAILABLE
    else "legacy_custom_autograd"
)
_DIRECT_GLOBAL_AUTOTUNE_ENABLED = (
    os.getenv("DWARF_HISA_V19_DIRECT_AUTOTUNE", "0") == "1"
)
_DIRECT_GLOBAL_LAYOUT_STRIDE_KEY = (
    "sqb", "sqh", "sqn", "sqd",
    "sgkb", "sgkh", "sgkn", "sgkd",
    "sgvb", "sgvh", "sgvn", "sgvd",
    "spvb", "spvh", "spvn", "spvd",
    "srb", "srh", "srn", "srk",
    "scb", "sch", "scn", "sck",
)
_DIRECT_GLOBAL_AUTOTUNE_KEY = (
    "B",
    "H",
    "N",
    "HD",
    "RANK",
    "RANK_PAD",
    "K_VAL",
    "CHUNK_SIZE",
    "CHUNK_PAD",
    "LOCAL_WINDOW",
    "HAS_EVIDENCE",
    "ARCH",
) + _DIRECT_GLOBAL_LAYOUT_STRIDE_KEY
_DIRECT_SOURCE_AUTOTUNE_KEY = _DIRECT_GLOBAL_AUTOTUNE_KEY + ("SOURCE_BLOCK",)


def _direct_global_autotune_contract() -> dict[str, dict[str, object]]:
    """Return the bounded, checkpoint-independent launch search contract."""
    return {
        "forward": {
            "key": _DIRECT_GLOBAL_AUTOTUNE_KEY,
            "configs": ((4, 2), (8, 2), (4, 3)),
        },
        "query_backward": {
            "key": _DIRECT_GLOBAL_AUTOTUNE_KEY,
            "configs": ((4, 2), (8, 2), (4, 1)),
        },
        "source_backward": {
            "key": _DIRECT_SOURCE_AUTOTUNE_KEY,
            "configs": ((4, 1), (8, 1)),
        },
    }



def _cuda_arch_code(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return int(major * 10 + minor)


def hisa_runtime_capabilities() -> dict[str, bool | str]:
    """Return availability of the production HISA execution dependencies."""
    return {
        "flex_attention": bool(_FLEX_ATTENTION_AVAILABLE),
        "triton": bool(_TRITON_AVAILABLE),
        "triton_library_operator": bool(_TRITON_LIBRARY_API_AVAILABLE),
        "global_lane_operator_mode": _GLOBAL_LANE_OPERATOR_MODE,
        "flex_empty_row_policy": _FLEX_EMPTY_ROW_POLICY,
        "streaming_selector": True,
        "hierarchical_parent_child_selector": True,
        "exact_page_reranker": True,
        "incremental_exact_page_cache": True,
        "aggregate_only_global_kernel": True,
        "source_block_and_atomic_backward": True,
        "selector_tile_size_role": "compatibility_and_diagnostics_only",
        "token_routing_pack_size_role": "compatibility_only",
    }


def hisa_integration_contract() -> dict[str, object]:
    """Expose the optimized execution semantics without checkpoint-visible state."""
    return {
        "global_lane_operator_mode": _GLOBAL_LANE_OPERATOR_MODE,
        "global_lane_arithmetic": "factorized_page_softmax_then_route_softmax",
        "global_lane_autotune_default": _DIRECT_GLOBAL_AUTOTUNE_ENABLED,
        "aggregate_only_specialization": True,
        "flex_empty_row_policy": _FLEX_EMPTY_ROW_POLICY,
        "hard_selector": "detached_streaming_top_m",
        "selector_autograd": "selected_K_plus_sampled_auxiliary_rows_only",
        "selector_workspace": "bounded_by_stream_block_and_candidate_width",
        "hierarchical_routing": "completed_parent_groups_then_child_candidates",
        "exact_page_rerank": "token_level_lse_over_top_m_candidates",
        "incremental_kv_cache": "exact_completed_pages_and_cached_addresses_O(N)",
        "backward_variants": (
            "source_block_stable",
            "source_block_counting",
            "atomic",
        ),
        "selector_tile_size_role": "compatibility_and_diagnostics_only",
        "token_routing_pack_size_role": "compatibility_only",
    }


HISA_ROUTE_SOURCE_POLICY_POST_PACKET = "post_packet_single_source"
HISA_ROUTE_SOURCE_POLICY_FROZEN_BASE = "frozen_base_route_decoupled"
HISA_ROUTE_SOURCE_POLICY_HYBRID_DUAL_SOURCE = "root_hybrid_dual_source"


@dataclass(frozen=True)
class HISARouteSourcePolicy:
    """Own every key/value source decision in the routed global lane."""

    name: str
    coarse_candidate_sources: tuple[str, ...]
    hard_rerank_key: str
    selected_prior_key: str
    route_aux_teacher_key: str
    global_attention_key: str = "post_packet_rotated_global_k"
    global_attention_value: str = "post_packet_rotated_global_v"

    @property
    def route_from_base_global_key(self) -> bool:
        return self.coarse_candidate_sources[0] == "base_global_k"

    @property
    def dual_source_candidate_union(self) -> bool:
        return len(self.coarse_candidate_sources) == 2

    @property
    def rerank_selected_priors_with_post_packet_representatives(self) -> bool:
        return self.selected_prior_key == "post_packet_rotated_global_k"

    def contract(self) -> dict[str, object]:
        return {
            "coarse_candidate_sources": self.coarse_candidate_sources,
            "hard_rerank_key": self.hard_rerank_key,
            "selected_prior_key": self.selected_prior_key,
            "route_aux_teacher_key": self.route_aux_teacher_key,
            "global_attention_key": self.global_attention_key,
            "global_attention_value": self.global_attention_value,
        }


HISA_ROUTE_SOURCE_POLICIES = {
    HISA_ROUTE_SOURCE_POLICY_POST_PACKET: HISARouteSourcePolicy(
        name=HISA_ROUTE_SOURCE_POLICY_POST_PACKET,
        coarse_candidate_sources=("post_packet_rotated_global_k",),
        hard_rerank_key="post_packet_rotated_global_k",
        selected_prior_key="post_packet_rotated_global_k",
        route_aux_teacher_key="post_packet_rotated_global_k",
    ),
    HISA_ROUTE_SOURCE_POLICY_FROZEN_BASE: HISARouteSourcePolicy(
        name=HISA_ROUTE_SOURCE_POLICY_FROZEN_BASE,
        coarse_candidate_sources=("base_global_k",),
        hard_rerank_key="base_global_k",
        selected_prior_key="base_global_k",
        route_aux_teacher_key="base_global_k",
    ),
    HISA_ROUTE_SOURCE_POLICY_HYBRID_DUAL_SOURCE: HISARouteSourcePolicy(
        name=HISA_ROUTE_SOURCE_POLICY_HYBRID_DUAL_SOURCE,
        coarse_candidate_sources=(
            "base_global_k",
            "post_packet_rotated_global_k",
        ),
        hard_rerank_key="post_packet_rotated_global_k",
        selected_prior_key="base_global_k",
        route_aux_teacher_key="post_packet_rotated_global_k",
    ),
}


def _resolve_route_source_policy(
    route_source_policy: str | None,
    *,
    route_from_base_global_key: bool | None,
    dual_source_candidate_union: bool | None,
    rerank_selected_priors_with_post_packet_representatives: bool | None,
) -> HISARouteSourcePolicy:
    """Resolve the new explicit policy while preserving the legacy constructor."""
    if route_source_policy is None:
        route_from_base = (
            False
            if route_from_base_global_key is None
            else route_from_base_global_key
        )
        dual_source = (
            True
            if dual_source_candidate_union is None
            else dual_source_candidate_union
        )
        rerank_priors = (
            False
            if rerank_selected_priors_with_post_packet_representatives is None
            else rerank_selected_priors_with_post_packet_representatives
        )
        if not route_from_base:
            return HISA_ROUTE_SOURCE_POLICIES[
                HISA_ROUTE_SOURCE_POLICY_POST_PACKET
            ]
        if dual_source and not rerank_priors:
            return HISA_ROUTE_SOURCE_POLICIES[
                HISA_ROUTE_SOURCE_POLICY_HYBRID_DUAL_SOURCE
            ]
        return HISARouteSourcePolicy(
            name="legacy_base_route_custom",
            coarse_candidate_sources=(
                ("base_global_k", "post_packet_rotated_global_k")
                if dual_source
                else ("base_global_k",)
            ),
            hard_rerank_key="post_packet_rotated_global_k",
            selected_prior_key=(
                "post_packet_rotated_global_k"
                if rerank_priors
                else "base_global_k"
            ),
            route_aux_teacher_key="post_packet_rotated_global_k",
        )

    policy = HISA_ROUTE_SOURCE_POLICIES.get(str(route_source_policy))
    if policy is None:
        choices = ", ".join(sorted(HISA_ROUTE_SOURCE_POLICIES))
        raise ValueError(f"route_source_policy must be one of: {choices}")
    expected = {
        "route_from_base_global_key": policy.route_from_base_global_key,
        "dual_source_candidate_union": policy.dual_source_candidate_union,
        "rerank_selected_priors_with_post_packet_representatives": (
            policy.rerank_selected_priors_with_post_packet_representatives
        ),
    }
    supplied = {
        "route_from_base_global_key": route_from_base_global_key,
        "dual_source_candidate_union": dual_source_candidate_union,
        "rerank_selected_priors_with_post_packet_representatives": (
            rerank_selected_priors_with_post_packet_representatives
        ),
    }
    conflicts = [
        name
        for name, value in supplied.items()
        if value is not None and value != expected[name]
    ]
    if conflicts:
        raise ValueError(
            "route source policy conflicts with legacy flags: "
            + ", ".join(conflicts)
        )
    return policy


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
    """Resolve the global backend without silently selecting a CUDA debug path."""
    if is_cuda and not flex_available:
        raise RuntimeError("HISA requires FlexAttention for CUDA execution")
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
        return True
    if backend == "auto" and is_cuda:
        if not triton_available:
            raise RuntimeError(
                "HISA backend='auto' on CUDA requires Triton. Refusing to fall "
                "back to the eager reference operator; select backend='eager' "
                "explicitly only for debugging."
            )
        return True
    return False


def _validate_triton_geometry(
    *,
    head_dim: int,
    tokens_per_chunk: int,
) -> None:
    """Validate only geometry consumed by the direct complete-page kernel."""
    if head_dim < 16 or not _is_power_of_two(head_dim):
        raise ValueError(
            "Triton HISA requires a power-of-two head dimension >=16"
        )
    if max(16, _next_pow2(tokens_per_chunk)) > 256:
        raise ValueError(
            "Triton HISA supports at most 256 tokens per complete chunk"
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
    return result.contiguous()


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
    valid_lengths: torch.Tensor,
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
    positions = torch.arange(x.shape[2], device=x.device).reshape(1, 1, -1)
    valid = (positions < valid_lengths.reshape(-1, 1, 1)).expand_as(strength)
    valid_actual_abs = actual_abs[valid]
    valid_strength = strength[valid]
    if valid_actual_abs.numel():
        p50 = torch.quantile(valid_actual_abs, 0.50)
        p90 = torch.quantile(valid_actual_abs, 0.90)
        p99 = torch.quantile(valid_actual_abs, 0.99)
        saturation = (valid_strength > 0.9).float().mean()
    else:
        p50 = p90 = p99 = saturation = torch.full(
            (), float("nan"), device=x.device, dtype=torch.float32
        )
    metrics = {
        f"npci_{label}_actual_abs_angle_p50": p50,
        f"npci_{label}_actual_abs_angle_p90": p90,
        f"npci_{label}_actual_abs_angle_p99": p99,
        f"npci_{label}_strength_saturation_fraction": saturation,
        f"npci_{label}_valid_token_head_count": valid.sum(),
    }
    for head, value in enumerate(physical_abs):
        metrics[f"npci_{label}_physical_abs_theta_head{head:02d}"] = value
    return {name: value.detach() for name, value in metrics.items()}


@torch.no_grad()
def _combined_lse_diagnostics(
    combined_lse: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify finite, structural -inf, and failed expected causal rows."""
    if combined_lse.ndim != 3:
        raise ValueError("combined LSE must use [B,H,N] shape")
    batch_size, heads, seq_len = combined_lse.shape
    positions = torch.arange(seq_len, device=combined_lse.device).reshape(1, 1, -1)
    expected = (
        (positions > 0)
        & (positions < valid_lengths.reshape(batch_size, 1, 1))
    ).expand(batch_size, heads, seq_len)
    finite = torch.isfinite(combined_lse)
    expected_count = expected.sum()
    finite_expected_count = (expected & finite).sum()
    finite_rate = torch.where(
        expected_count > 0,
        finite_expected_count.float() / expected_count.clamp_min(1).float(),
        torch.full((), float("nan"), device=combined_lse.device),
    )
    return {
        "combined_lse_expected_causal_row_count": expected_count.detach(),
        "combined_lse_finite_expected_causal_row_count": (
            finite_expected_count.detach()
        ),
        "combined_lse_unexpected_nonfinite_count": (
            expected & ~finite
        ).sum().detach(),
        "combined_lse_structural_neg_inf_count": (
            ~expected & torch.isneginf(combined_lse)
        ).sum().detach(),
        "combined_lse_finite_rate": finite_rate.detach(),
    }


@dataclass(frozen=True)
class HISAMetadata:
    top_chunk_idx: torch.Tensor          # int32 [B,H,T,K]
    tile_starts: torch.Tensor            # int32 [T]
    valid_lengths: torch.Tensor          # int32 [B]
    chunk_size: int
    selector_tile_size: int
    query_chunk_idx: torch.Tensor | None = None  # int32 [B,H,N,K] for token routing


@dataclass(frozen=True)
class HISASelectionCapture:
    """Ephemeral routing evidence produced only on explicit metadata requests."""
    anchor_logits: torch.Tensor
    metadata: HISAMetadata
    auxiliary_loss: torch.Tensor
    sampled_anchor_logits: torch.Tensor | None = None
    sampled_tile_ids: torch.Tensor | None = None
    sampled_teacher_mass: torch.Tensor | None = None


@dataclass(frozen=True)
class HISAAddressDiagnostics:
    """Optional expensive representative telemetry, absent on production paths."""

    max_token_positions: torch.Tensor
    max_token_ids: torch.Tensor | None
    mean_norm: torch.Tensor
    max_norm: torch.Tensor
    mean_max_cosine: torch.Tensor


@dataclass(frozen=True)
class HISAChunkAddresses:
    """Child addresses, coherence, and optional fixed-group parent summaries.

    ``values`` is [B,H,C,D] for one address or [B,H,C,A,D] for multiple
    addresses. ``coherence`` is the norm of the pre-normalized mean direction.
    Parent summaries are consumed only by the detached hierarchical selector.
    """

    values: torch.Tensor
    address_names: tuple[str, ...]
    chunk_valid: torch.Tensor
    coherence: torch.Tensor
    parent_values: torch.Tensor | None = None
    parent_coherence: torch.Tensor | None = None
    parent_group_size: int = 1
    diagnostics: HISAAddressDiagnostics | None = None


@dataclass(frozen=True)
class HISARouterAuxiliary:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    coverage_loss: torch.Tensor
    sampled_anchor_logits: torch.Tensor
    sampled_tile_ids: torch.Tensor
    teacher_mass: torch.Tensor
    teacher_global_mass: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class HISARouteEvidence:
    """Per-route normalized evidence and route log-partition values."""

    output: torch.Tensor       # [B,H,N,K,E]
    lse: torch.Tensor          # fp32 [B,H,N,K]


@dataclass
class HISAIncrementalState:
    """Inference cache for local history, exact pages, and semantic addresses.

    Child addresses and completed parent summaries are appended only when a page
    closes. The exact page K/V tensors still make this an O(N)-memory cache.
    """

    position: int
    local_key: torch.Tensor
    local_value: torch.Tensor
    local_positions: torch.Tensor
    pending_base_global_key: torch.Tensor
    pending_global_key: torch.Tensor
    pending_global_value: torch.Tensor
    completed_base_global_key: torch.Tensor
    completed_global_key: torch.Tensor
    completed_global_value: torch.Tensor
    completed_routing_addresses: HISAChunkAddresses
    completed_attention_addresses: HISAChunkAddresses


@torch.no_grad()
def _forced_route_coverage_diagnostics(
    route_indices: torch.Tensor,
    forced_route_chunk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure coverage of caller-supplied relevant chunks; IDs do not force routing."""
    if route_indices.ndim != 4:
        raise ValueError("route_indices must have shape [B,H,N,K]")
    batch_size, heads, seq_len, _ = route_indices.shape
    forced = forced_route_chunk_ids
    if forced.ndim == 3:
        if forced.shape[:2] != (batch_size, seq_len):
            raise ValueError("forced_route_chunk_ids must have shape [B,N,P]")
        forced = forced[:, None].expand(-1, heads, -1, -1)
    elif forced.ndim == 4:
        if forced.shape[:3] != (batch_size, heads, seq_len):
            raise ValueError("forced_route_chunk_ids must have shape [B,H,N,P]")
    else:
        raise ValueError(
            "forced_route_chunk_ids must have shape [B,N,P] or [B,H,N,P]"
        )
    forced = forced.to(device=route_indices.device, dtype=torch.int64)
    routes = route_indices.to(torch.int64)
    valid_target = forced >= 0
    covered = (
        forced[..., None] == routes[..., None, :]
    ).any(-1) & valid_target
    target_count = valid_target.sum().clamp_min(1)
    target_coverage = covered.sum().float() / target_count.float()
    valid_row = valid_target.any(-1)
    row_any = (covered.any(-1) & valid_row)
    row_all = (
        torch.where(valid_target, covered, torch.ones_like(covered)).all(-1)
        & valid_row
    )
    row_count = valid_row.sum().clamp_min(1)
    by_head_count = valid_target.sum((0, 2, 3)).clamp_min(1)
    by_head = covered.sum((0, 2, 3)).float() / by_head_count.float()
    return {
        "forced_route_target_coverage": target_coverage,
        "forced_route_any_row_coverage": row_any.sum().float() / row_count.float(),
        "forced_route_complete_row_coverage": row_all.sum().float() / row_count.float(),
        "forced_route_target_coverage_by_head": by_head,
        "forced_route_target_count": valid_target.sum().float(),
    }


@torch.no_grad()
def _direct_kernel_geometry_diagnostics(
    route: torch.Tensor,
    metadata: HISAMetadata,
) -> dict[str, torch.Tensor]:
    """Describe the realized one-program-per-query complete-page arithmetic."""
    if route.ndim != 4:
        raise ValueError("selected route scores must use [B,H,N,K_VAL] shape")
    batch_size, heads, seq_len, route_width = route.shape
    if metadata.top_chunk_idx.shape[-1] != route_width:
        raise ValueError("route width must equal semantic route width")
    query_programs = batch_size * heads * seq_len
    static_pages = query_programs * route_width
    selected_pages = torch.isfinite(route).sum()
    chunk_pad = max(16, _next_pow2(int(metadata.chunk_size)))
    static_pairs = static_pages * chunk_pad
    semantic_pairs = selected_pages * int(metadata.chunk_size)
    return {
        "direct_kernel_query_program_count": torch.tensor(
            query_programs, device=route.device
        ),
        "direct_kernel_valid_query_program_count": (
            metadata.valid_lengths.sum(dtype=torch.int64) * heads
        ).detach(),
        "direct_kernel_route_width": torch.tensor(route_width, device=route.device),
        "direct_kernel_static_page_slot_count": torch.tensor(
            static_pages, device=route.device
        ),
        "direct_kernel_selected_page_count": selected_pages.detach(),
        "direct_kernel_chunk_pad": torch.tensor(chunk_pad, device=route.device),
        "direct_kernel_static_dot_pair_count": torch.tensor(
            static_pairs, device=route.device
        ),
        "direct_kernel_semantic_dot_pair_count": semantic_pairs.detach(),
        "direct_kernel_semantic_to_static_pair_fraction": (
            semantic_pairs.float()
            / torch.tensor(static_pairs, device=route.device).clamp_min(1).float()
        ).detach(),
    }




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


def _masked_direction_mean(
    directions: torch.Tensor,
    token_valid: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_float = token_valid.unsqueeze(-1).to(directions.dtype)
    mean_pre = (directions * valid_float).sum(dim) / valid_float.sum(dim).clamp_min(1.0)
    coherence = mean_pre.norm(dim=-1)
    mean = F.normalize(mean_pre, dim=-1, eps=1e-6)
    return mean_pre, mean, coherence


def _completed_chunk_addresses(
    key: torch.Tensor,
    *,
    chunk_size: int,
    valid_lengths: torch.Tensor,
    blend_alpha: float | torch.Tensor,
    mode: str,
    hierarchy_group_size: int = 1,
    collect_diagnostics: bool = False,
    token_ids: torch.Tensor | None = None,
) -> HISAChunkAddresses:
    """Build child addresses and causal-safe fixed-group parent addresses."""
    chunks, token_valid, chunk_valid = _chunk_tensors(key, chunk_size, valid_lengths)
    values = chunks.float()
    directions = F.normalize(values, dim=-1, eps=1e-6)
    _, mean, coherence = _masked_direction_mean(directions, token_valid, dim=3)

    energy = values.square().sum(-1).masked_fill(~token_valid, float("-inf"))
    best = energy.argmax(-1, keepdim=True)
    max_vector = torch.gather(
        directions,
        3,
        best.unsqueeze(-1).expand(-1, -1, -1, 1, values.shape[-1]),
    ).squeeze(3)

    if mode in {"mean_max_blend", "mean_max_blend_ablation"}:
        if torch.is_tensor(blend_alpha):
            mixture = blend_alpha.to(device=values.device, dtype=values.dtype).reshape(
                1, values.shape[1], 1, 1
            )
        else:
            mixture = float(blend_alpha)
        address_values = F.normalize(
            (1.0 - mixture) * mean + mixture * max_vector,
            dim=-1,
            eps=1e-6,
        )
        address_names = ("mean_max_blend",)
    elif mode == "multi_address":
        address_values = torch.stack((mean, max_vector), dim=-2)
        address_names = ("mean", "max_energy")
    elif mode == "multi_landmark":
        midpoint = max(1, int(chunk_size) // 2)
        _, first_mean, _ = _masked_direction_mean(
            directions[:, :, :, :midpoint], token_valid[:, :, :, :midpoint], dim=3
        )
        _, second_mean, _ = _masked_direction_mean(
            directions[:, :, :, midpoint:], token_valid[:, :, :, midpoint:], dim=3
        )
        address_values = torch.stack(
            (mean, max_vector, first_mean, second_mean), dim=-2
        )
        address_names = ("mean", "max_energy", "first_half", "second_half")
    else:
        raise ValueError(
            "representative_mode must be mean_max_blend, mean_max_blend_ablation, "
            "multi_address, or multi_landmark"
        )

    if address_values.ndim == 4:
        address_values = address_values.masked_fill(
            ~chunk_valid[:, None, :, None], 0.0
        )
    else:
        address_values = address_values.masked_fill(
            ~chunk_valid[:, None, :, None, None], 0.0
        )
    valid_head_chunk = chunk_valid[:, None].expand(-1, key.shape[1], -1)
    coherence = coherence.masked_fill(~valid_head_chunk, 0.0)

    group_size = max(1, int(hierarchy_group_size))
    parent_values: torch.Tensor | None = None
    parent_coherence: torch.Tensor | None = None
    if group_size > 1:
        child_count = directions.shape[2]
        padded_children = math.ceil(child_count / group_size) * group_size
        child_pad = padded_children - child_count
        directions_parent = (
            F.pad(directions, (0, 0, 0, 0, 0, child_pad))
            if child_pad else directions
        )
        token_valid_parent = (
            F.pad(token_valid, (0, 0, 0, child_pad), value=False)
            if child_pad else token_valid
        )
        parent_count = padded_children // group_size
        directions_parent = directions_parent.reshape(
            values.shape[0], values.shape[1], parent_count,
            group_size * int(chunk_size), values.shape[-1]
        )
        token_valid_parent = token_valid_parent.reshape(
            values.shape[0], 1, parent_count, group_size * int(chunk_size)
        )
        _, parent_values, parent_coherence = _masked_direction_mean(
            directions_parent, token_valid_parent, dim=3
        )
        parent_starts = (
            torch.arange(parent_count, device=key.device, dtype=torch.int32)
            * group_size * int(chunk_size)
        )
        parent_valid = parent_starts.reshape(1, parent_count) < valid_lengths.reshape(-1, 1)
        parent_values = parent_values.masked_fill(
            ~parent_valid[:, None, :, None], 0.0
        )
        parent_coherence = parent_coherence.masked_fill(
            ~parent_valid[:, None], 0.0
        )

    diagnostics: HISAAddressDiagnostics | None = None
    if collect_diagnostics:
        max_energy = energy.max(-1).values.clamp_min(0.0)
        max_norm = torch.sqrt(max_energy)
        mean_max_cosine = (mean * max_vector).sum(-1)
        chunk_indices = torch.arange(
            values.shape[2], device=values.device, dtype=torch.int64
        ).reshape(1, 1, -1)
        max_positions = chunk_indices * int(chunk_size) + best.squeeze(-1).to(torch.int64)
        max_positions = torch.where(
            chunk_valid[:, None], max_positions, torch.full_like(max_positions, -1)
        )
        max_token_ids = None
        if token_ids is not None:
            if token_ids.shape != (key.shape[0], key.shape[2]):
                raise ValueError("token_ids must have shape [B,N]")
            if token_ids.device != key.device:
                raise ValueError("token_ids must be on the HISA input device")
            safe_positions = max_positions.clamp_min(0)
            expanded_ids = token_ids[:, None, :].expand(-1, key.shape[1], -1)
            max_token_ids = torch.gather(expanded_ids, 2, safe_positions)
            max_token_ids = torch.where(
                max_positions >= 0, max_token_ids, torch.full_like(max_token_ids, -1)
            )
        diagnostics = HISAAddressDiagnostics(
            max_token_positions=max_positions,
            max_token_ids=max_token_ids,
            mean_norm=mean.norm(dim=-1).masked_fill(~valid_head_chunk, 0.0),
            max_norm=max_norm.masked_fill(~valid_head_chunk, 0.0),
            mean_max_cosine=mean_max_cosine.masked_fill(~valid_head_chunk, 0.0),
        )

    return HISAChunkAddresses(
        values=address_values.float(),
        address_names=address_names,
        chunk_valid=chunk_valid,
        coherence=coherence.float(),
        parent_values=None if parent_values is None else parent_values.float(),
        parent_coherence=None if parent_coherence is None else parent_coherence.float(),
        parent_group_size=group_size,
        diagnostics=diagnostics,
    )


def _empty_incremental_address_cache(
    batch_size: int,
    heads: int,
    head_dim: int,
    *,
    mode: str,
    hierarchy_group_size: int,
    device: torch.device,
) -> HISAChunkAddresses:
    address_names_by_mode = {
        "mean_max_blend": ("mean_max_blend",),
        "mean_max_blend_ablation": ("mean_max_blend",),
        "multi_address": ("mean", "max_energy"),
        "multi_landmark": ("mean", "max_energy", "first_half", "second_half"),
    }
    address_names = address_names_by_mode[mode]
    if len(address_names) == 1:
        values = torch.empty(
            batch_size, heads, 0, head_dim, device=device, dtype=torch.float32
        )
    else:
        values = torch.empty(
            batch_size, heads, 0, len(address_names), head_dim,
            device=device, dtype=torch.float32,
        )
    group = max(1, int(hierarchy_group_size))
    parent_values = (
        torch.empty(batch_size, heads, 0, head_dim, device=device, dtype=torch.float32)
        if group > 1 else None
    )
    parent_coherence = (
        torch.empty(batch_size, heads, 0, device=device, dtype=torch.float32)
        if group > 1 else None
    )
    return HISAChunkAddresses(
        values=values,
        address_names=address_names,
        chunk_valid=torch.empty(batch_size, 0, device=device, dtype=torch.bool),
        coherence=torch.empty(
            batch_size, heads, 0, device=device, dtype=torch.float32
        ),
        parent_values=parent_values,
        parent_coherence=parent_coherence,
        parent_group_size=group,
        diagnostics=None,
    )


@torch.no_grad()
def _append_incremental_address_cache(
    cache: HISAChunkAddresses,
    page_key: torch.Tensor,
    all_pages: torch.Tensor,
    *,
    chunk_size: int,
    blend_alpha: float | torch.Tensor,
    mode: str,
) -> HISAChunkAddresses:
    """Append one child address and, when complete, one fixed-group parent."""
    if page_key.ndim != 4 or page_key.shape[2] != int(chunk_size):
        raise ValueError("completed incremental page must have shape [B,H,chunk_size,D]")
    batch_size, heads, _, head_dim = page_key.shape
    full_lengths = torch.full(
        (batch_size,), int(chunk_size), device=page_key.device, dtype=torch.int32
    )
    child = _completed_chunk_addresses(
        page_key,
        chunk_size=int(chunk_size),
        valid_lengths=full_lengths,
        blend_alpha=blend_alpha,
        mode=mode,
        hierarchy_group_size=1,
        collect_diagnostics=False,
    )
    if child.address_names != cache.address_names:
        raise RuntimeError("incremental address mode changed after cache creation")
    values = torch.cat((cache.values, child.values), dim=2)
    coherence = torch.cat((cache.coherence, child.coherence), dim=2)
    chunk_valid = torch.cat(
        (
            cache.chunk_valid,
            torch.ones(batch_size, 1, device=page_key.device, dtype=torch.bool),
        ),
        dim=1,
    )

    parent_values = cache.parent_values
    parent_coherence = cache.parent_coherence
    group = int(cache.parent_group_size)
    completed_count = values.shape[2]
    if group > 1 and completed_count % group == 0:
        if all_pages.ndim != 5 or all_pages.shape[2] != completed_count:
            raise ValueError("all_pages must include the newly completed page")
        group_pages = all_pages[:, :, completed_count - group:completed_count]
        directions = F.normalize(group_pages.float(), dim=-1, eps=1e-6)
        mean_pre = directions.mean(dim=(2, 3))
        parent = F.normalize(mean_pre, dim=-1, eps=1e-6)
        parent_strength = mean_pre.norm(dim=-1)
        if parent_values is None or parent_coherence is None:
            parent_values = torch.empty(
                batch_size, heads, 0, head_dim,
                device=page_key.device, dtype=torch.float32,
            )
            parent_coherence = torch.empty(
                batch_size, heads, 0,
                device=page_key.device, dtype=torch.float32,
            )
        parent_values = torch.cat((parent_values, parent[:, :, None]), dim=2)
        parent_coherence = torch.cat(
            (parent_coherence, parent_strength[:, :, None]), dim=2
        )

    return HISAChunkAddresses(
        values=values,
        address_names=cache.address_names,
        chunk_valid=chunk_valid,
        coherence=coherence,
        parent_values=parent_values,
        parent_coherence=parent_coherence,
        parent_group_size=group,
        diagnostics=None,
    )


def _score_chunk_addresses(
    query: torch.Tensor,
    addresses: torch.Tensor,
    *,
    reduction: str,
    temperature: float,
) -> torch.Tensor:
    if addresses.ndim == 4:
        return torch.matmul(query.float(), addresses.float().transpose(-2, -1))
    if addresses.ndim != 5:
        raise ValueError("addresses must have shape [B,H,C,D] or [B,H,C,A,D]")
    scores = torch.einsum("bhsd,bhcad->bhsca", query.float(), addresses.float())
    if reduction == "max":
        return scores.max(-1).values
    if reduction == "logsumexp":
        address_count = max(1, addresses.shape[-2])
        return float(temperature) * (
            torch.logsumexp(scores / float(temperature), dim=-1)
            - math.log(address_count)
        )
    raise ValueError("representative_score_reduction must be max or logsumexp")


def _normalized_routing_query(query: torch.Tensor) -> torch.Tensor:
    return F.normalize(query.float(), dim=-1, eps=1e-6)


def _coherence_score_correction(
    coherence: torch.Tensor,
    coherence_weight: torch.Tensor | float,
) -> torch.Tensor:
    if torch.is_tensor(coherence_weight):
        shape = (1, int(coherence_weight.numel())) + (1,) * max(1, coherence.ndim - 2)
        weight = coherence_weight.float().reshape(shape)
    else:
        weight = float(coherence_weight)
    return weight * torch.log(coherence.float().clamp_min(1e-4))


def _dense_routing_score_surface(
    query_normalized: torch.Tensor,
    addresses: HISAChunkAddresses | torch.Tensor,
    *,
    reduction: str = "max",
    temperature: float = 1.0,
    coherence: torch.Tensor | None = None,
    coherence_weight: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    if isinstance(addresses, HISAChunkAddresses):
        address_values = addresses.values
        coherence = addresses.coherence
    else:
        address_values = addresses
    with torch.autocast(device_type=query_normalized.device.type, enabled=False):
        scores = _score_chunk_addresses(
            query_normalized.float(), address_values.float(),
            reduction=reduction, temperature=temperature,
        ).float()
        if coherence is not None:
            scores = scores + _coherence_score_correction(
                coherence, coherence_weight
            )[:, :, None, :]
        return scores


def _gather_chunk_values(values: torch.Tensor, chunk_idx: torch.Tensor) -> torch.Tensor:
    safe = chunk_idx.clamp_min(0).long()
    batch_size, heads, seq_len, slots = safe.shape
    if values.ndim == 4:
        expanded = values[:, :, None].expand(
            batch_size, heads, seq_len, values.shape[2], values.shape[3]
        )
        index = safe[..., None].expand(
            batch_size, heads, seq_len, slots, values.shape[3]
        )
        return torch.gather(expanded, 3, index)
    if values.ndim == 5:
        expanded = values[:, :, None].expand(
            batch_size, heads, seq_len, values.shape[2], values.shape[3], values.shape[4]
        )
        index = safe[..., None, None].expand(
            batch_size, heads, seq_len, slots, values.shape[3], values.shape[4]
        )
        return torch.gather(expanded, 3, index)
    raise ValueError("chunk values must be rank 4 or 5")


def _selected_routing_scores(
    query_normalized: torch.Tensor,
    addresses: HISAChunkAddresses,
    chunk_idx: torch.Tensor,
    *,
    reduction: str,
    temperature: float,
    coherence_weight: torch.Tensor | float,
) -> torch.Tensor:
    selected = _gather_chunk_values(addresses.values, chunk_idx)
    if selected.ndim == 5:
        scores = (query_normalized.float()[..., None, :] * selected.float()).sum(-1)
    else:
        per_address = (
            query_normalized.float()[..., None, None, :] * selected.float()
        ).sum(-1)
        if reduction == "max":
            scores = per_address.max(-1).values
        elif reduction == "logsumexp":
            address_count = max(1, selected.shape[-2])
            scores = float(temperature) * (
                torch.logsumexp(per_address / float(temperature), dim=-1)
                - math.log(address_count)
            )
        else:
            raise ValueError("representative_score_reduction must be max or logsumexp")
    expanded_coherence = addresses.coherence[:, :, None].expand(
        query_normalized.shape[0], query_normalized.shape[1],
        query_normalized.shape[2], addresses.coherence.shape[-1]
    )
    selected_coherence = torch.gather(
        expanded_coherence, -1, chunk_idx.clamp_min(0).long()
    )
    scores = scores + _coherence_score_correction(selected_coherence, coherence_weight)
    return scores.masked_fill(chunk_idx < 0, float("-inf"))


def _representative_diagnostics(addresses: HISAChunkAddresses) -> dict[str, torch.Tensor]:
    diagnostics = addresses.diagnostics
    if diagnostics is None:
        raise RuntimeError("representative diagnostics were not collected")
    valid = addresses.chunk_valid[:, None].expand_as(addresses.coherence)
    count = valid.sum().clamp_min(1)

    def average(value: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, value.float(), 0.0).sum() / count

    primary = addresses.values.float() if addresses.values.ndim == 4 else addresses.values[..., 0, :].float()
    primary = F.normalize(primary, dim=-1, eps=1e-6)
    similarity = torch.einsum("bhcd,bhkd->bhck", primary, primary)
    pair_valid = valid[..., :, None] & valid[..., None, :]
    diagonal = torch.eye(
        primary.shape[2], device=primary.device, dtype=torch.bool
    ).reshape(1, 1, primary.shape[2], primary.shape[2])
    pair_valid = pair_valid & ~diagonal
    pair_count = pair_valid.sum().clamp_min(1)
    cross_similarity = torch.where(
        pair_valid, similarity, torch.zeros_like(similarity)
    ).sum() / pair_count
    result = {
        "representative_mean_norm": average(diagnostics.mean_norm),
        "representative_max_norm": average(diagnostics.max_norm),
        "representative_mean_max_cosine": average(diagnostics.mean_max_cosine),
        "representative_pre_normalization_norm": average(addresses.coherence),
        "representative_cross_chunk_similarity": cross_similarity,
        "representative_max_winner_positions": diagnostics.max_token_positions.detach(),
    }
    if diagnostics.max_token_ids is not None:
        result["representative_max_winner_token_ids"] = diagnostics.max_token_ids.detach()
    if addresses.parent_coherence is not None:
        parent_valid = addresses.parent_coherence > 0
        result["representative_parent_coherence"] = torch.where(
            parent_valid, addresses.parent_coherence,
            torch.zeros_like(addresses.parent_coherence),
        ).sum() / parent_valid.sum().clamp_min(1)
    return {name: value.detach() for name, value in result.items()}



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


def _entry_eligibility(
    *,
    seq_len: int,
    entry_count: int,
    entry_span: int,
    valid_lengths: torch.Tensor,
    local_window: int,
) -> torch.Tensor:
    query_positions = torch.arange(
        seq_len, device=valid_lengths.device, dtype=torch.int32
    ).reshape(1, seq_len, 1)
    entry_ends = (
        torch.arange(entry_count, device=valid_lengths.device, dtype=torch.int32) + 1
    ).reshape(1, 1, entry_count) * int(entry_span)
    query_valid = query_positions < valid_lengths.reshape(-1, 1, 1)
    return query_valid & (entry_ends <= query_positions - int(local_window))


@torch.no_grad()
def _streaming_topk_address_values(
    query_normalized: torch.Tensor,
    values: torch.Tensor,
    coherence: torch.Tensor,
    *,
    entry_span: int,
    valid_lengths: torch.Tensor,
    local_window: int,
    top_k: int,
    block_size: int,
    reduction: str,
    temperature: float,
    coherence_weight: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, heads, seq_len, _ = query_normalized.shape
    entry_count = values.shape[2]
    k = min(max(1, int(top_k)), entry_count)
    top_values = torch.full(
        (batch_size, heads, seq_len, k), float("-inf"),
        device=query_normalized.device, dtype=torch.float32,
    )
    top_indices = torch.full(
        (batch_size, heads, seq_len, k), -1,
        device=query_normalized.device, dtype=torch.int64,
    )
    eligibility = _entry_eligibility(
        seq_len=seq_len, entry_count=entry_count, entry_span=entry_span,
        valid_lengths=valid_lengths, local_window=local_window,
    )
    step = max(1, int(block_size))
    for start in range(0, entry_count, step):
        end = min(start + step, entry_count)
        block_scores = _score_chunk_addresses(
            query_normalized, values[:, :, start:end],
            reduction=reduction, temperature=temperature,
        ).float()
        block_scores = block_scores + _coherence_score_correction(
            coherence[:, :, start:end], coherence_weight
        )[:, :, None, :]
        block_scores = block_scores.masked_fill(
            ~eligibility[:, None, :, start:end], float("-inf")
        )
        block_ids = torch.arange(
            start, end, device=query_normalized.device, dtype=torch.int64
        ).reshape(1, 1, 1, -1).expand_as(block_scores)
        merged_values = torch.cat((top_values, block_scores), dim=-1)
        merged_indices = torch.cat((top_indices, block_ids), dim=-1)
        top_values, order = merged_values.topk(k, dim=-1)
        top_indices = torch.gather(merged_indices, -1, order)
    valid = torch.isfinite(top_values)
    top_indices = torch.where(valid, top_indices, torch.full_like(top_indices, -1))
    return top_values, top_indices, valid


def _deduplicate_fixed_candidates(
    candidates: torch.Tensor,
    *,
    entry_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sentinel = torch.full_like(candidates, int(entry_count))
    sortable = torch.where(
        (candidates >= 0) & (candidates < int(entry_count)), candidates, sentinel
    )
    ordered = sortable.sort(dim=-1).values
    unique = (ordered != sentinel) & torch.cat(
        (torch.ones_like(ordered[..., :1], dtype=torch.bool),
         ordered[..., 1:] != ordered[..., :-1]),
        dim=-1,
    )
    return torch.where(unique, ordered, sentinel), unique


@torch.no_grad()
def _hierarchical_candidate_metadata(
    query_normalized: torch.Tensor,
    addresses: HISAChunkAddresses,
    *,
    chunk_size: int,
    valid_lengths: torch.Tensor,
    local_window: int,
    candidate_k: int,
    parent_top_k: int,
    streaming_block_size: int,
    reduction: str,
    temperature: float,
    coherence_weight: torch.Tensor | float,
    hierarchical: bool,
) -> HISAMetadata:
    batch_size, heads, seq_len, _ = query_normalized.shape
    child_count = addresses.values.shape[2]
    use_hierarchy = bool(
        hierarchical and addresses.parent_values is not None
        and addresses.parent_coherence is not None
        and addresses.parent_group_size > 1
    )
    if not use_hierarchy:
        _, indices, valid = _streaming_topk_address_values(
            query_normalized, addresses.values, addresses.coherence,
            entry_span=int(chunk_size), valid_lengths=valid_lengths,
            local_window=local_window, top_k=candidate_k,
            block_size=streaming_block_size, reduction=reduction,
            temperature=temperature, coherence_weight=coherence_weight,
        )
        indices = torch.where(valid, indices, torch.full_like(indices, -1))
    else:
        group = int(addresses.parent_group_size)
        _, parent_indices, parent_valid = _streaming_topk_address_values(
            query_normalized, addresses.parent_values, addresses.parent_coherence,
            entry_span=group * int(chunk_size), valid_lengths=valid_lengths,
            local_window=local_window, top_k=parent_top_k,
            block_size=streaming_block_size, reduction="max", temperature=1.0,
            coherence_weight=coherence_weight,
        )
        child_offsets = torch.arange(
            group, device=query_normalized.device, dtype=torch.int64
        )
        parent_children = (
            parent_indices[..., None] * group + child_offsets
        ).reshape(batch_size, heads, seq_len, -1)
        parent_children_valid = parent_valid[..., None].expand(
            *parent_valid.shape, group
        ).reshape_as(parent_children)

        cutoff = (
            torch.arange(seq_len, device=query_normalized.device, dtype=torch.int64)
            - int(local_window)
        ).clamp_min(0)
        eligible_child_count = torch.div(
            cutoff, int(chunk_size), rounding_mode="floor"
        ).clamp(max=child_count)
        complete_parent_count = torch.div(
            eligible_child_count, group, rounding_mode="floor"
        )
        tail_base = complete_parent_count * group
        tail = tail_base[:, None] + child_offsets[None, :]
        tail_valid = tail < eligible_child_count[:, None]
        tail = tail.reshape(1, 1, seq_len, group).expand(
            batch_size, heads, seq_len, group
        )
        tail_valid = tail_valid.reshape(1, 1, seq_len, group).expand_as(tail)
        query_valid = (
            torch.arange(seq_len, device=query_normalized.device).reshape(1, 1, -1, 1)
            < valid_lengths.reshape(batch_size, 1, 1, 1)
        )
        candidate_ids = torch.cat((parent_children, tail), dim=-1)
        candidate_valid = torch.cat((parent_children_valid, tail_valid), dim=-1)
        candidate_valid = candidate_valid & query_valid & (candidate_ids < child_count)
        candidate_ids = torch.where(
            candidate_valid, candidate_ids, torch.full_like(candidate_ids, -1)
        )
        deduplicated, unique = _deduplicate_fixed_candidates(
            candidate_ids, entry_count=child_count
        )
        safe_ids = deduplicated.clamp(max=max(0, child_count - 1))
        candidate_scores = _selected_routing_scores(
            query_normalized, addresses, safe_ids,
            reduction=reduction, temperature=temperature,
            coherence_weight=coherence_weight,
        ).masked_fill(~unique, float("-inf"))
        k = min(max(1, int(candidate_k)), candidate_scores.shape[-1])
        values, order = candidate_scores.topk(k, dim=-1)
        indices = torch.gather(deduplicated, -1, order)
        valid = torch.isfinite(values) & (indices < child_count)
        indices = torch.where(valid, indices, torch.full_like(indices, -1))

    return HISAMetadata(
        top_chunk_idx=indices.to(torch.int32),
        tile_starts=torch.arange(seq_len, device=query_normalized.device, dtype=torch.int32),
        valid_lengths=valid_lengths.detach(), chunk_size=int(chunk_size),
        selector_tile_size=1,
    )


def _candidate_page_lse_reference(
    query: torch.Tensor,
    global_key: torch.Tensor,
    candidate_chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    chunk_size: int,
    local_window: int,
    query_block_size: int = 16,
) -> torch.Tensor:
    batch_size, heads, seq_len, head_dim = query.shape
    slots = candidate_chunks.shape[-1]
    result = torch.full(
        (batch_size, heads, seq_len, slots), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)
    within = torch.arange(int(chunk_size), device=query.device, dtype=torch.int64).reshape(1, 1, 1, -1)
    batch_index = torch.arange(batch_size, device=query.device).reshape(batch_size, 1, 1, 1)
    head_index = torch.arange(heads, device=query.device).reshape(1, heads, 1, 1)
    block = max(1, int(query_block_size))
    for start in range(0, seq_len, block):
        end = min(start + block, seq_len)
        positions = torch.arange(start, end, device=query.device, dtype=torch.int64)
        q = query[:, :, start:end]
        q_valid = positions.reshape(1, -1) < valid_lengths.reshape(batch_size, 1)
        for slot in range(slots):
            chunk = candidate_chunks[:, :, start:end, slot].to(torch.int64)
            ids = chunk[..., None] * int(chunk_size) + within
            safe_ids = ids.clamp(0, max(0, seq_len - 1))
            keys = global_key[batch_index, head_index, safe_ids]
            scores = torch.einsum("bhqd,bhqmd->bhqm", q, keys) * scale
            valid = (
                (chunk[..., None] >= 0)
                & (ids < valid_lengths.reshape(batch_size, 1, 1, 1))
                & (ids < positions.reshape(1, 1, -1, 1) - int(local_window))
                & q_valid[:, None, :, None]
            )
            scores = scores.masked_fill(~valid, float("-inf"))
            result[:, :, start:end, slot] = torch.logsumexp(scores.float(), dim=-1)
    return result


def _candidate_page_lse(
    query: torch.Tensor,
    global_key: torch.Tensor,
    candidate_chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    chunk_size: int,
    local_window: int,
) -> torch.Tensor:
    if query.is_cuda and _TRITON_AVAILABLE:
        return _selected_page_lse_triton_apply(
            query, global_key, candidate_chunks, valid_lengths,
            int(chunk_size), int(local_window),
        )
    return _candidate_page_lse_reference(
        query, global_key, candidate_chunks, valid_lengths,
        chunk_size=int(chunk_size), local_window=int(local_window),
    )


@torch.no_grad()
def _rerank_candidate_metadata(
    query: torch.Tensor,
    global_key: torch.Tensor,
    candidate_metadata: HISAMetadata,
    *,
    top_k: int,
    local_window: int,
    exact_rerank: bool,
) -> tuple[HISAMetadata, torch.Tensor | None]:
    candidate_chunks = candidate_metadata.top_chunk_idx
    if not exact_rerank or candidate_chunks.shape[-1] <= int(top_k):
        selected = candidate_chunks[..., : int(top_k)]
        return HISAMetadata(
            top_chunk_idx=selected, tile_starts=candidate_metadata.tile_starts,
            valid_lengths=candidate_metadata.valid_lengths,
            chunk_size=candidate_metadata.chunk_size, selector_tile_size=1,
        ), None
    exact_lse = _candidate_page_lse(
        query, global_key, candidate_chunks, candidate_metadata.valid_lengths,
        chunk_size=candidate_metadata.chunk_size, local_window=int(local_window),
    )
    k = min(int(top_k), exact_lse.shape[-1])
    values, order = exact_lse.topk(k, dim=-1)
    selected = torch.gather(candidate_chunks, -1, order)
    valid = torch.isfinite(values) & (selected >= 0)
    selected = torch.where(valid, selected, torch.full_like(selected, -1))
    return HISAMetadata(
        top_chunk_idx=selected.to(torch.int32),
        tile_starts=candidate_metadata.tile_starts,
        valid_lengths=candidate_metadata.valid_lengths,
        chunk_size=candidate_metadata.chunk_size, selector_tile_size=1,
    ), values


@torch.no_grad()
def _inject_streaming_exploration(
    metadata: HISAMetadata,
    query_normalized: torch.Tensor,
    addresses: HISAChunkAddresses,
    probability: float | torch.Tensor,
    *,
    local_window: int,
    policy: str,
    temperature: float,
    block_size: int,
    reduction: str,
    representative_temperature: float,
    coherence_weight: torch.Tensor | float,
) -> HISAMetadata:
    indices = metadata.top_chunk_idx.to(torch.int64)
    valid = indices >= 0
    if indices.shape[-1] == 0 or (
        not torch.is_tensor(probability) and float(probability) <= 0.0
    ):
        return metadata
    if policy == "uniform_unseen_ablation":
        eligible = _eligibility(
            metadata.tile_starts, addresses.values.shape[2], metadata.chunk_size,
            metadata.valid_lengths, int(local_window),
        )
        indices, valid = _inject_exploration_slot(
            indices, valid, eligible, probability,
            policy=policy, temperature=temperature,
        )
    elif policy == "tail_softmax":
        batch_size, heads, seq_len, _ = indices.shape
        best_value = torch.full(
            (batch_size, heads, seq_len), float("-inf"),
            device=indices.device, dtype=torch.float32,
        )
        best_index = torch.full_like(best_value, -1, dtype=torch.int64)
        entry_count = addresses.values.shape[2]
        eligibility = _entry_eligibility(
            seq_len=seq_len, entry_count=entry_count,
            entry_span=metadata.chunk_size,
            valid_lengths=metadata.valid_lengths, local_window=int(local_window),
        )
        step = max(1, int(block_size))
        for start in range(0, entry_count, step):
            end = min(start + step, entry_count)
            scores = _score_chunk_addresses(
                query_normalized, addresses.values[:, :, start:end],
                reduction=reduction, temperature=representative_temperature,
            ).float()
            scores = scores + _coherence_score_correction(
                addresses.coherence[:, :, start:end], coherence_weight
            )[:, :, None, :]
            block_ids = torch.arange(
                start, end, device=indices.device, dtype=torch.int64
            ).reshape(1, 1, 1, -1)
            already_selected = (block_ids[..., None] == indices[..., None, :]).any(-1)
            unseen = eligibility[:, None, :, start:end] & ~already_selected
            uniform = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            sampled = (scores / float(temperature) + gumbel).masked_fill(
                ~unseen, float("-inf")
            )
            block_value, block_offset = sampled.max(-1)
            replace_best = block_value > best_value
            best_value = torch.where(replace_best, block_value, best_value)
            best_index = torch.where(
                replace_best, block_offset.to(torch.int64) + start, best_index
            )
        has_candidate = torch.isfinite(best_value)
        replace = (
            torch.rand(batch_size, heads, seq_len, device=indices.device) < probability
        ) & has_candidate
        indices = indices.clone(); valid = valid.clone()
        indices[..., -1] = torch.where(replace, best_index, indices[..., -1])
        valid[..., -1] = torch.where(replace, torch.ones_like(replace), valid[..., -1])
    else:
        raise ValueError("exploration_policy must be tail_softmax or uniform_unseen_ablation")
    indices = torch.where(valid, indices, torch.full_like(indices, -1))
    return HISAMetadata(
        top_chunk_idx=indices.to(torch.int32), tile_starts=metadata.tile_starts,
        valid_lengths=metadata.valid_lengths, chunk_size=metadata.chunk_size,
        selector_tile_size=1,
    )


def _inject_exploration_slot(
    indices: torch.Tensor,
    valid: torch.Tensor,
    eligible: torch.Tensor,
    probability: float | torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    policy: str = "uniform_unseen_ablation",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the final route with an unseen eligible exploratory chunk.

    ``tail_softmax`` samples the best router tail rather than treating every
    unseen chunk identically. The released uniform policy remains available as
    an explicitly named ablation.
    """
    if indices.shape[-1] == 0 or (
        not torch.is_tensor(probability) and probability <= 0.0
    ):
        return indices, valid
    if policy not in {"tail_softmax", "uniform_unseen_ablation"}:
        raise ValueError(
            "exploration_policy must be tail_softmax or uniform_unseen_ablation"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("exploration_temperature must be finite and positive")
    batch_size, heads, tiles, slots = indices.shape
    eligible_count = eligible.sum(-1, dtype=torch.int64)[:, None].expand(
        batch_size, heads, tiles
    )
    selected_count = valid.sum(-1, dtype=torch.int64)
    unseen_count = eligible_count - selected_count
    has_candidate = unseen_count > 0

    if policy == "uniform_unseen_ablation":
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
        # Causal eligibility is a contiguous prefix. Map an unseen rank back to
        # a chunk ID while touching only the fixed, small K selection.
        for slot in range(slots):
            selected_id = selected_sorted[..., slot]
            choice = choice + (selected_id <= choice).to(choice.dtype)
    else:
        if logits is None or logits.shape != (
            batch_size,
            heads,
            tiles,
            eligible.shape[-1],
        ):
            raise ValueError("tail_softmax exploration requires matching router logits")
        # Inductor lowers this bounded count scatter to a CUDA atomic; Triton
        # supports int32 here but rejects int16 atomics at compile time.
        selected_mask = torch.zeros_like(logits, dtype=torch.int32)
        selected_mask.scatter_add_(
            -1,
            indices.clamp_min(0).long(),
            valid.to(selected_mask.dtype),
        )
        selected_mask = selected_mask.to(torch.bool)
        unseen = eligible[:, None].expand_as(logits) & ~selected_mask
        tail_logits = (logits.detach().float() / float(temperature)).masked_fill(
            ~unseen, -1e9
        )
        probabilities = torch.softmax(tail_logits, dim=-1) * unseen
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
            1e-12
        )
        safe_probabilities = torch.where(
            has_candidate[..., None],
            probabilities,
            F.one_hot(
                torch.zeros_like(has_candidate, dtype=torch.int64),
                num_classes=eligible.shape[-1],
            ).to(probabilities.dtype),
        )
        choice = torch.multinomial(
            safe_probabilities.reshape(-1, eligible.shape[-1]), 1
        ).reshape(batch_size, heads, tiles)

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


def _annealed_exploration_probability(
    initial: float,
    final: float,
    step: int | torch.Tensor,
    anneal_steps: int,
) -> float | torch.Tensor:
    if torch.is_tensor(step):
        if anneal_steps <= 0:
            return torch.full_like(step, float(initial), dtype=torch.float32)
        progress = step.float().clamp(0, int(anneal_steps)) / float(anneal_steps)
        return float(initial) + progress * (float(final) - float(initial))
    if anneal_steps <= 0:
        return float(initial)
    progress = min(max(int(step), 0), int(anneal_steps)) / float(anneal_steps)
    return float(initial) + progress * (float(final) - float(initial))



def _global_ids(
    metadata: HISAMetadata,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = metadata.top_chunk_idx
    within = torch.arange(
        metadata.chunk_size,
        device=chunks.device,
        dtype=torch.int32,
    ).reshape(1, 1, 1, 1, -1)
    ids = chunks[..., None] * metadata.chunk_size + within
    valid = (chunks[..., None] >= 0) & (ids >= 0) & (ids < seq_len)
    return ids, valid


def _semantic_route_chunks(metadata: HISAMetadata, seq_len: int) -> torch.Tensor:
    """Return the K semantic candidate IDs owned by each query row."""
    if metadata.query_chunk_idx is not None:
        return metadata.query_chunk_idx[:, :, :seq_len]
    return metadata.top_chunk_idx.repeat_interleave(
        metadata.selector_tile_size, dim=2
    )[:, :, :seq_len]


def _semantic_route_priors(
    route: torch.Tensor,
    metadata: HISAMetadata,
    semantic_chunks: torch.Tensor,
) -> torch.Tensor:
    """Gather semantic K priors from a possibly deduplicated physical pack."""
    seq_len = semantic_chunks.shape[2]
    physical_chunks = metadata.top_chunk_idx.repeat_interleave(
        metadata.selector_tile_size, dim=2
    )[:, :, :seq_len]
    physical_route = route[:, :, :seq_len]
    matches = (
        semantic_chunks[..., None] == physical_chunks[..., None, :]
    ) & (semantic_chunks[..., None] >= 0)
    return physical_route[..., None, :].masked_fill(~matches, float("-inf")).max(
        dim=-1
    ).values


def _route_identity_features(
    similarity: torch.Tensor,
    chunk_idx: torch.Tensor,
    *,
    route_scale_by_head: torch.Tensor,
    temperature: float,
    chunk_size: int,
    query_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build relative route identity and absolute abstention features.

    The relative prior remains centered across valid K routes. Absolute
    max/mean/entropy are shared by the binder null and the global lane's
    learned common offset, so K=1 retains query-dependent confidence.
    """
    valid = chunk_idx >= 0
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    mean = torch.where(valid, similarity, torch.zeros_like(similarity)).sum(
        -1, keepdim=True
    ) / count
    relative_prior = (
        (similarity - mean)
        * route_scale_by_head.reshape(1, -1, 1, 1).float()
        / float(temperature)
    )

    seq_len = similarity.shape[2]
    if query_positions is None:
        query_position = torch.arange(
            seq_len, device=similarity.device, dtype=torch.float32
        ).reshape(1, 1, seq_len, 1)
    else:
        query_position = query_positions.to(
            device=similarity.device, dtype=torch.float32
        ).reshape(1, 1, seq_len, 1)
    denominator = query_position + 1.0
    chunk_start = chunk_idx.clamp_min(0).float() * float(chunk_size)
    chunk_end = chunk_start + float(chunk_size)
    chunk_center = chunk_start + 0.5 * float(chunk_size - 1)
    relative_age = (query_position - chunk_end) / denominator
    relative_position = chunk_center / denominator
    features = torch.stack(
        (
            similarity.float(),
            relative_prior.float(),
            relative_age,
            relative_position,
        ),
        dim=-1,
    )
    features = torch.where(valid[..., None], features, torch.zeros_like(features))

    masked = similarity.float().masked_fill(~valid, float("-inf"))
    absolute_max = masked.max(-1).values
    absolute_max = torch.where(
        valid.any(-1), absolute_max, torch.zeros_like(absolute_max)
    )
    absolute_mean = mean.squeeze(-1).float()
    safe_logits = masked.masked_fill(~valid, -1e9)
    probability = torch.softmax(safe_logits, dim=-1)
    entropy = -(probability * torch.log_softmax(safe_logits, dim=-1)).sum(-1)
    entropy = torch.where(
        valid.any(-1), entropy, torch.zeros_like(entropy)
    )
    summary = torch.stack((absolute_max, absolute_mean, entropy), dim=-1)
    return features, summary


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


def _resolve_route_aux_tile_ids(
    metadata: HISAMetadata,
    *,
    samples: int,
    route_slots: int,
    local_window: int,
    tile_ids: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Sample only rows where more than K chunks can compete for K slots."""
    tiles = metadata.top_chunk_idx.shape[2]
    first_competitive_tile = math.ceil(
        ((int(route_slots) + 1) * metadata.chunk_size + int(local_window))
        / metadata.selector_tile_size
    )
    candidate_count = max(0, tiles - first_competitive_tile)
    sample_count = min(int(samples), candidate_count)
    if sample_count <= 0:
        return torch.empty(0, device=device, dtype=torch.int64)
    useful_tiles = torch.arange(
        first_competitive_tile, tiles, device=device, dtype=torch.int64
    )
    if tile_ids is None:
        if sample_count == candidate_count:
            return useful_tiles
        permutation = torch.randperm(candidate_count, device=device)
        return useful_tiles[permutation[:sample_count]]
    if not torch.is_tensor(tile_ids):
        raise TypeError("route_aux_tile_ids must be a tensor")
    if tile_ids.ndim != 1 or tile_ids.dtype != torch.int64:
        raise TypeError("route_aux_tile_ids must be one-dimensional int64")
    if tile_ids.device != device:
        raise ValueError("route_aux_tile_ids must be on the HISA input device")
    if tile_ids.numel() != sample_count:
        raise ValueError(f"route_aux_tile_ids must contain exactly {sample_count} IDs")
    if torch.compiler.is_compiling():
        return tile_ids
    ordered = tile_ids.sort().values
    unique = torch.ones((), dtype=torch.bool, device=device) if ordered.numel() <= 1 else (ordered[1:] != ordered[:-1]).all()
    valid_ids = (tile_ids >= first_competitive_tile) & (tile_ids < tiles)
    valid = valid_ids.all() & unique
    message = "route_aux_tile_ids must be unique and in the competitive tile range"
    if tile_ids.is_cuda:
        torch._assert_async(valid, message)
    elif not bool(valid):
        if bool(((tile_ids < 0) | (tile_ids >= tiles)).any()):
            raise ValueError("route_aux_tile_ids contains an out-of-range ID")
        if bool((tile_ids < first_competitive_tile).any()):
            raise ValueError("route_aux_tile_ids contains a row without more eligible chunks than final route slots")
        raise ValueError("route_aux_tile_ids must be unique")
    return tile_ids


def _sampled_router_teacher_targets(
    query: torch.Tensor,
    teacher_key: torch.Tensor,
    *,
    sampled_positions: torch.Tensor,
    chunk_size: int,
    valid_lengths: torch.Tensor,
    local_window: int,
    mode: str,
    attention_scale: float,
    cosine_temperature: float,
    target_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditional global chunk mass and total global-vs-local mass."""
    key_chunks, token_valid, _ = _chunk_tensors(teacher_key, chunk_size, valid_lengths)
    eligible = _eligibility(
        sampled_positions.to(torch.int32), key_chunks.shape[2], chunk_size,
        valid_lengths, local_window,
    )
    chunk_count, token_count = key_chunks.shape[2], key_chunks.shape[3]
    token_positions = torch.arange(
        chunk_count * token_count, device=query.device, dtype=torch.int32
    ).reshape(1, 1, 1, chunk_count, token_count)
    query_positions = sampled_positions.to(torch.int32).reshape(1, 1, -1, 1, 1)
    query_valid = sampled_positions.reshape(1, -1) < valid_lengths.reshape(-1, 1)
    all_past = token_valid[:, :, None] & (token_positions < query_positions) & query_valid[:, None, :, None, None]
    global_mask = all_past & eligible[:, None, :, :, None]
    if mode == "dense_attention":
        token_scores = torch.einsum("bhsd,bhcmd->bhscm", query.float(), key_chunks.float()) * float(attention_scale)
    elif mode == "cosine_ablation":
        query_direction = F.normalize(query.float(), dim=-1, eps=1e-6)
        key_direction = F.normalize(key_chunks.float(), dim=-1, eps=1e-6)
        token_scores = torch.einsum("bhsd,bhcmd->bhscm", query_direction, key_direction) / float(cosine_temperature)
    else:
        raise ValueError("route_aux_teacher must be dense_attention or cosine_ablation")
    masked = token_scores.masked_fill(~all_past, -1e9)
    all_probability = torch.softmax(masked.reshape(*masked.shape[:-2], -1), dim=-1).reshape_as(masked)
    all_probability = all_probability * all_past
    unconditional_chunk_mass = (all_probability * global_mask).sum(-1)
    teacher_global_mass = unconditional_chunk_mass.sum(-1).clamp(0.0, 1.0)
    conditional_mass = unconditional_chunk_mass / teacher_global_mass[..., None].clamp_min(1e-12)
    conditional_mass = conditional_mass * eligible[:, None]
    if target_temperature != 1.0:
        softened = torch.log(conditional_mass.clamp_min(1e-12)) / float(target_temperature)
        conditional_mass = torch.softmax(
            softened.masked_fill(~eligible[:, None], -1e9), dim=-1
        ) * eligible[:, None]
    conditional_mass = conditional_mass / conditional_mass.sum(-1, keepdim=True).clamp_min(1e-12)
    return conditional_mass, teacher_global_mass



def _teacher_weighted_coverage_loss(
    route_logits: torch.Tensor,
    teacher_mass: torch.Tensor,
    eligible: torch.Tensor,
    *,
    route_slots: int,
) -> torch.Tensor:
    """Teacher-conditional expected top-K coverage; no unconditional repulsion."""
    expanded_eligible = (
        eligible[:, None] if eligible.ndim == route_logits.ndim - 1 else eligible
    )
    probabilities = torch.softmax(
        route_logits.float().masked_fill(~expanded_eligible, -1e9), dim=-1
    ) * expanded_eligible
    covered = 1.0 - (1.0 - probabilities).clamp_min(0.0).pow(int(route_slots))
    covered_teacher_mass = (teacher_mass.float() * covered).sum(-1)
    valid = expanded_eligible.any(-1).expand_as(covered_teacher_mass)
    loss = -torch.log(covered_teacher_mass.clamp_min(1e-12))
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def _rms_match_global_key(
    global_key: torch.Tensor,
    local_key: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(global_key.shape[2], device=global_key.device)
    valid = positions.reshape(1, 1, -1, 1) < valid_lengths.reshape(-1, 1, 1, 1)
    # Match each token independently. Sequence- or chunk-wide RMS statistics
    # would let a future suffix rescale keys used by an earlier causal query.
    local_rms = torch.sqrt(local_key.float().square().mean(-1, keepdim=True))
    global_rms = torch.sqrt(global_key.float().square().mean(-1, keepdim=True))
    scale = local_rms / global_rms.clamp_min(1e-6)
    matched = (global_key.float() * scale).to(global_key.dtype)
    return torch.where(valid, matched, global_key)


def _router_auxiliary_loss(
    query_normalized: torch.Tensor,
    addresses: HISAChunkAddresses,
    attention_query: torch.Tensor,
    teacher_key: torch.Tensor,
    metadata: HISAMetadata,
    route_scale_by_head: torch.Tensor,
    *,
    samples: int,
    route_slots: int,
    target_temperature: float,
    routing_temperature: float,
    local_window: int,
    oracle_temperature: float,
    teacher_mode: str,
    coverage_weight: float,
    representative_reduction: str,
    representative_temperature: float,
    coherence_weight: torch.Tensor | float,
    tile_ids: torch.Tensor | None = None,
) -> HISARouterAuxiliary:
    resolved_ids = _resolve_route_aux_tile_ids(
        metadata, samples=samples, route_slots=route_slots,
        local_window=local_window, tile_ids=tile_ids,
        device=query_normalized.device,
    )
    chunk_count = addresses.values.shape[2]
    if resolved_ids.numel() == 0:
        empty = query_normalized.new_empty(query_normalized.shape[0], query_normalized.shape[1], 0, chunk_count)
        empty_global = query_normalized.new_empty(query_normalized.shape[0], query_normalized.shape[1], 0)
        zero = query_normalized.new_zeros(())
        return HISARouterAuxiliary(
            loss=zero, cross_entropy=zero, coverage_loss=zero,
            sampled_anchor_logits=empty, sampled_tile_ids=resolved_ids,
            teacher_mass=empty, teacher_global_mass=empty_global,
            eligible=torch.empty(query_normalized.shape[0], 0, chunk_count, device=query_normalized.device, dtype=torch.bool),
        )
    sampled_anchor_logits = _dense_routing_score_surface(
        query_normalized[:, :, resolved_ids], addresses,
        reduction=representative_reduction,
        temperature=representative_temperature,
        coherence_weight=coherence_weight,
    ).float() / float(routing_temperature)
    sampled_tile_starts = metadata.tile_starts[resolved_ids]
    eligible = _eligibility(
        sampled_tile_starts, chunk_count, metadata.chunk_size,
        metadata.valid_lengths, local_window,
    )
    with torch.no_grad():
        target, teacher_global_mass = _sampled_router_teacher_targets(
            attention_query[:, :, resolved_ids].detach(), teacher_key,
            sampled_positions=sampled_tile_starts,
            chunk_size=metadata.chunk_size,
            valid_lengths=metadata.valid_lengths,
            local_window=local_window, mode=teacher_mode,
            attention_scale=1.0 / math.sqrt(attention_query.shape[-1]),
            cosine_temperature=oracle_temperature,
            target_temperature=target_temperature,
        )
    safe_anchor_logits = torch.where(
        eligible[:, None], sampled_anchor_logits, torch.zeros_like(sampled_anchor_logits)
    )
    route = (
        safe_anchor_logits * route_scale_by_head.reshape(1, query_normalized.shape[1], 1, 1).float()
    ).masked_fill(~eligible[:, None], -1e9)
    per_row = -(target * torch.log_softmax(route, dim=-1)).sum(-1)
    valid_rows = (eligible.any(-1)[:, None] & (teacher_global_mass > 0.0)).expand_as(per_row)
    cross_entropy = (per_row * valid_rows).sum() / valid_rows.sum().clamp_min(1)
    if float(coverage_weight) > 0.0:
        coverage_loss = _teacher_weighted_coverage_loss(route, target, eligible, route_slots=int(route_slots))
    else:
        coverage_loss = route.new_zeros(())
    loss = cross_entropy + float(coverage_weight) * coverage_loss
    return HISARouterAuxiliary(
        loss=loss, cross_entropy=cross_entropy, coverage_loss=coverage_loss,
        sampled_anchor_logits=sampled_anchor_logits,
        sampled_tile_ids=resolved_ids, teacher_mass=target,
        teacher_global_mass=teacher_global_mass, eligible=eligible,
    )



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
    selector_tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict sliding-window lane returning normalized output and natural LSE."""
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, heads, seq_len), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    lane_span = local_window + chunk_size - 1
    offsets = torch.arange(lane_span, device=query.device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    for start in range(0, seq_len, selector_tile_size):
        end = min(start + selector_tile_size, seq_len)
        positions = torch.arange(start, end, device=query.device, dtype=torch.int32)
        q = query[:, :, start:end]
        q_valid = positions.reshape(1, -1) < valid_lengths.reshape(batch_size, 1)
        ids = positions[:, None] - lane_span + offsets[None]
        lane_mask = _strict_local_or_boundary_key_mask(
            positions[:, None], ids, local_window, chunk_size
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
        repeat = metadata.chunk_size
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


def _eager_route_evidence(
    query: torch.Tensor,
    global_key: torch.Tensor,
    evidence_value: torch.Tensor,
    route: torch.Tensor,
    metadata: HISAMetadata,
    global_lse: torch.Tensor,
    *,
    local_window: int,
) -> HISARouteEvidence:
    """Reference per-route attention contract over semantic candidate IDs.

    This intentionally follows ``query_chunk_idx`` rather than physical packed
    union slots. It is an eager oracle and integration seam, not a replacement
    for the production Triton aggregate lane.
    """
    batch_size, heads, seq_len, head_dim = query.shape
    if evidence_value.shape[:3] != (batch_size, heads, seq_len):
        raise ValueError("evidence_value must have shape [B,H,N,E]")
    semantic_chunks = _semantic_route_chunks(metadata, seq_len)
    semantic_prior = _semantic_route_priors(route, metadata, semantic_chunks)
    within = torch.arange(
        metadata.chunk_size, device=query.device, dtype=torch.int32
    ).reshape(1, 1, 1, -1)
    scale = 1.0 / math.sqrt(head_dim)
    batch_index = torch.arange(batch_size, device=query.device).reshape(
        batch_size, 1, 1, 1
    )
    head_index = torch.arange(heads, device=query.device).reshape(1, heads, 1, 1)
    slots = semantic_chunks.shape[-1]
    output = torch.zeros(
        batch_size,
        heads,
        seq_len,
        slots,
        evidence_value.shape[-1],
        device=evidence_value.device,
        dtype=evidence_value.dtype,
    )
    lse = torch.full(
        (batch_size, heads, seq_len, slots),
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )
    # Bound eager/reference temporaries by the physical query-pack width rather
    # than materializing [B,H,N,chunk_size,HD] once per semantic route.
    for start in range(0, seq_len, metadata.selector_tile_size):
        end = min(start + metadata.selector_tile_size, seq_len)
        positions = torch.arange(
            start, end, device=query.device, dtype=torch.int32
        ).reshape(1, 1, -1, 1)
        query_tile = query[:, :, start:end]
        for slot in range(slots):
            chunk = semantic_chunks[:, :, start:end, slot]
            prior = semantic_prior[:, :, start:end, slot]
            ids = chunk[..., None] * metadata.chunk_size + within
            safe_ids = ids.clamp(0, max(seq_len - 1, 0)).long()
            keys = global_key[batch_index, head_index, safe_ids]
            values = evidence_value[batch_index, head_index, safe_ids]
            scores = (
                torch.einsum("bhqd,bhqmd->bhqm", query_tile, keys) * scale
                + prior[..., None]
            )
            valid = (
                (chunk[..., None] >= 0)
                & (ids >= 0)
                & (ids < seq_len)
                & (ids < metadata.valid_lengths.reshape(batch_size, 1, 1, 1))
                & _strict_global_key_mask(positions, ids, local_window)
                & (positions < metadata.valid_lengths.reshape(batch_size, 1, 1, 1))
                & torch.isfinite(prior[..., None])
            )
            scores = scores.masked_fill(~valid, float("-inf"))
            maximum = scores.max(-1).values
            safe_max = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            weights = torch.where(
                torch.isfinite(scores),
                torch.exp(scores.float() - safe_max[..., None].float()),
                torch.zeros_like(scores.float()),
            )
            denominator = weights.sum(-1)
            normalized = torch.einsum(
                "bhqm,bhqme->bhqe", weights.to(values.dtype), values
            ) / denominator.clamp_min(1.0)[..., None].to(values.dtype)
            output[:, :, start:end, slot] = torch.where(
                (denominator > 0)[..., None],
                normalized,
                torch.zeros_like(normalized),
            )
            lse[:, :, start:end, slot] = torch.where(
                denominator > 0,
                safe_max.float() + denominator.log(),
                torch.full_like(safe_max.float(), float("-inf")),
            )
    return HISARouteEvidence(output=output, lse=lse)


def _eager_global_lane_with_route_evidence(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    route: torch.Tensor,
    metadata: HISAMetadata,
    *,
    local_window: int,
    evidence_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, HISARouteEvidence]:
    """Preserve the existing aggregate exactly and additionally retain K slots."""
    output, lse = _eager_global_lane(
        query,
        global_key,
        global_value,
        route,
        metadata,
        local_window=local_window,
    )
    evidence = _eager_route_evidence(
        query,
        global_key,
        global_value if evidence_value is None else evidence_value,
        route,
        metadata,
        lse,
        local_window=local_window,
    )
    return output, lse, evidence


def _direct_global_hisa_reference(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    projected_value: torch.Tensor,
    route: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    chunk_size: int,
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager oracle for aggregate output plus native low-rank route slots."""
    metadata = HISAMetadata(
        top_chunk_idx=chunks,
        tile_starts=torch.arange(
            query.shape[2], device=query.device, dtype=torch.int32
        ),
        valid_lengths=valid_lengths,
        chunk_size=int(chunk_size),
        selector_tile_size=1,
    )
    output, lse, evidence = _eager_global_lane_with_route_evidence(
        query,
        global_key,
        global_value,
        route,
        metadata,
        local_window=int(local_window),
        evidence_value=projected_value,
    )
    return output, lse, evidence.output, evidence.lse


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
    def _selected_page_lse_kernel(
        Q, GLOBAL_K, CHUNKS, LENGTHS, OUT,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        scb, sch, scn, sck,
        sob, soh, son, sok,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        K_VAL: tl.constexpr, CHUNK_SIZE: tl.constexpr,
        CHUNK_PAD: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
    ):
        program = tl.program_id(0)
        slot = program % K_VAL
        query_program = program // K_VAL
        query_position = query_program % N
        batch_head = query_program // N
        batch = batch_head // H
        head = batch_head % H
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_valid = query_position < valid_length
        chunk = tl.load(
            CHUNKS + batch * scb + head * sch + query_position * scn + slot * sck,
            mask=query_valid, other=-1,
        ).to(tl.int32)
        dimensions = tl.arange(0, HD)
        token_offsets = tl.arange(0, CHUNK_PAD)
        query = tl.load(
            Q + batch * sqb + head * sqh + query_position * sqn + dimensions * sqd,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        ids = chunk * CHUNK_SIZE + token_offsets
        selected = (
            query_valid & (chunk >= 0) & (token_offsets < CHUNK_SIZE)
            & (ids >= 0) & (ids < N) & (ids < valid_length)
            & (ids < query_position - LOCAL_WINDOW)
        )
        safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
        keys = tl.load(
            GLOBAL_K + batch * sgkb + head * sgkh
            + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
            mask=selected[:, None], other=0.0,
        ).to(tl.float32)
        score = tl.sum(keys * query[None, :], axis=1) / tl.sqrt(HD * 1.0)
        score = tl.where(selected, score, float("-inf"))
        maximum = tl.max(score, axis=0)
        safe_max = tl.where(maximum > float("-inf"), maximum, 0.0)
        probability = tl.where(selected, tl.exp(score - safe_max), 0.0)
        denominator = tl.sum(probability, axis=0)
        lse = tl.where(denominator > 0.0, maximum + tl.log(denominator), float("-inf"))
        tl.store(
            OUT + batch * sob + head * soh + query_position * son + slot * sok,
            lse, mask=query_valid,
        )

    @triton.jit
    def _direct_global_lane_forward_kernel(
        Q, GLOBAL_K, GLOBAL_V, PROJECTED_V, ROUTE, CHUNKS, LENGTHS,
        OUT, LSE, ROUTE_OUT, ROUTE_LSE,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        spvb, spvh, spvn, spvd,
        srb, srh, srn, srk,
        scb, sch, scn, sck,
        sob, soh, son, sod,
        slseb, slseh, slsen,
        srob, sroh, sron, srok, srod,
        srlb, srlh, srln, srlk,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        RANK: tl.constexpr, RANK_PAD: tl.constexpr,
        K_VAL: tl.constexpr, CHUNK_SIZE: tl.constexpr,
        CHUNK_PAD: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        HAS_EVIDENCE: tl.constexpr, ARCH: tl.constexpr,
    ):
        program = tl.program_id(0)
        query_position = program % N
        batch_head = program // N
        batch = batch_head // H
        head = batch_head % H
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_valid = query_position < valid_length
        dimensions = tl.arange(0, HD)
        projected_dimensions = tl.arange(0, RANK_PAD)
        token_offsets = tl.arange(0, CHUNK_PAD)
        query = tl.load(
            Q + batch * sqb + head * sqh + query_position * sqn + dimensions * sqd,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(HD * 1.0)
        running_max = tl.full([], float("-inf"), tl.float32)
        running_sum = tl.zeros([], tl.float32)
        accumulator = tl.zeros([HD], tl.float32)
        chunk_base = CHUNKS + batch * scb + head * sch + query_position * scn
        route_base = ROUTE + batch * srb + head * srh + query_position * srn
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            prior = tl.load(
                route_base + slot * srk,
                mask=query_valid & (chunk >= 0), other=float("-inf"),
            ).to(tl.float32)
            ids = chunk * CHUNK_SIZE + token_offsets
            selected = (
                query_valid & (chunk >= 0) & (token_offsets < CHUNK_SIZE)
                & (ids >= 0) & (ids < N) & (ids < valid_length)
                & (ids < query_position - LOCAL_WINDOW)
                & (prior > float("-inf"))
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            keys = tl.load(
                GLOBAL_K + batch * sgkb + head * sgkh
                + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
                mask=selected[:, None], other=0.0,
            ).to(tl.float32)
            values = tl.load(
                GLOBAL_V + batch * sgvb + head * sgvh
                + safe_ids[:, None] * sgvn + dimensions[None, :] * sgvd,
                mask=selected[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * scale + prior
            scores = tl.where(selected, scores, float("-inf"))
            route_max = tl.max(scores, axis=0)
            route_safe_max = tl.where(route_max > float("-inf"), route_max, 0.0)
            route_probability = tl.where(selected, tl.exp(scores - route_safe_max), 0.0)
            route_sum = tl.sum(route_probability, axis=0)
            route_denominator = tl.where(route_sum > 0.0, route_sum, 1.0)
            page_output = tl.sum(route_probability[:, None] * values, axis=0) / route_denominator
            route_slot_lse = tl.where(
                route_sum > 0.0, route_max + tl.log(route_sum), float("-inf")
            )
            if HAS_EVIDENCE:
                projected_values = tl.load(
                    PROJECTED_V + batch * spvb + head * spvh
                    + safe_ids[:, None] * spvn
                    + projected_dimensions[None, :] * spvd,
                    mask=selected[:, None] & (projected_dimensions[None, :] < RANK),
                    other=0.0,
                ).to(tl.float32)
                route_projected = tl.sum(
                    route_probability[:, None] * projected_values, axis=0
                ) / route_denominator
                tl.store(
                    ROUTE_OUT + batch * srob + head * sroh
                    + query_position * sron + slot * srok
                    + projected_dimensions * srod,
                    route_projected,
                    mask=query_valid & (projected_dimensions < RANK),
                )
                tl.store(
                    ROUTE_LSE + batch * srlb + head * srlh
                    + query_position * srln + slot * srlk,
                    route_slot_lse, mask=query_valid,
                )
            has_route = route_sum > 0.0
            merged_max = tl.maximum(running_max, route_slot_lse)
            safe_merged_max = tl.where(
                (running_max > float("-inf")) | has_route, merged_max, 0.0
            )
            old_scale = tl.where(
                running_max > float("-inf"),
                tl.exp(running_max - safe_merged_max), 0.0,
            )
            page_scale = tl.where(
                has_route, tl.exp(route_slot_lse - safe_merged_max), 0.0
            )
            running_sum = running_sum * old_scale + page_scale
            accumulator = accumulator * old_scale + page_scale * page_output
            running_max = tl.where(
                (running_max > float("-inf")) | has_route, merged_max, running_max
            )
        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        tl.store(
            OUT + batch * sob + head * soh + query_position * son + dimensions * sod,
            accumulator / denominator, mask=query_valid,
        )
        lse = tl.where(
            running_sum > 0.0, running_max + tl.log(running_sum), float("-inf")
        )
        tl.store(
            LSE + batch * slseb + head * slseh + query_position * slsen,
            lse, mask=query_valid,
        )

    @triton.jit
    def _direct_global_lane_query_backward_kernel(
        Q, GLOBAL_K, GLOBAL_V, PROJECTED_V, OUT, DOUT, DLSE, LSE,
        ROUTE_OUT, DROUTE_OUT, DROUTE_LSE, ROUTE_LSE,
        ROUTE, CHUNKS, LENGTHS, DQ, DROUTE, DELTA,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        spvb, spvh, spvn, spvd,
        sob, soh, son, sod,
        sdob, sdoh, sdon, sdod,
        sdlb, sdlh, sdln,
        slseb, slseh, slsen,
        srob, sroh, sron, srok, srod,
        sdrob, sdroh, sdron, sdrok, sdrod,
        sdrlb, sdrlh, sdrln, sdrlk,
        srlb, srlh, srln, srlk,
        srb, srh, srn, srk,
        scb, sch, scn, sck,
        sdqb, sdqh, sdqn, sdqd,
        sdrb, sdrh, sdrn, sdrk,
        sdeb, sdeh, sden,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        RANK: tl.constexpr, RANK_PAD: tl.constexpr,
        K_VAL: tl.constexpr, CHUNK_SIZE: tl.constexpr,
        CHUNK_PAD: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        HAS_EVIDENCE: tl.constexpr, ARCH: tl.constexpr,
    ):
        program = tl.program_id(0)
        query_position = program % N
        batch_head = program // N
        batch = batch_head // H
        head = batch_head % H
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_valid = query_position < valid_length
        dimensions = tl.arange(0, HD)
        projected_dimensions = tl.arange(0, RANK_PAD)
        token_offsets = tl.arange(0, CHUNK_PAD)
        query = tl.load(
            Q + batch * sqb + head * sqh + query_position * sqn + dimensions * sqd,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        output = tl.load(
            OUT + batch * sob + head * soh + query_position * son + dimensions * sod,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            DOUT + batch * sdob + head * sdoh + query_position * sdon + dimensions * sdod,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + batch * slseb + head * slseh + query_position * slsen,
            mask=query_valid, other=float("-inf"),
        ).to(tl.float32)
        lse_gradient = tl.load(
            DLSE + batch * sdlb + head * sdlh + query_position * sdln,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        lse_valid = lse > float("-inf")
        safe_lse = tl.where(lse_valid, lse, 0.0)
        delta = tl.sum(output_gradient * output, axis=0)
        tl.store(
            DELTA + batch * sdeb + head * sdeh + query_position * sden,
            delta, mask=query_valid,
        )
        dquery = tl.zeros([HD], tl.float32)
        scale = 1.0 / tl.sqrt(HD * 1.0)
        chunk_base = CHUNKS + batch * scb + head * sch + query_position * scn
        route_base = ROUTE + batch * srb + head * srh + query_position * srn
        droute_base = DROUTE + batch * sdrb + head * sdrh + query_position * sdrn
        for slot in range(K_VAL):
            chunk = tl.load(chunk_base + slot * sck).to(tl.int32)
            prior = tl.load(
                route_base + slot * srk,
                mask=query_valid & (chunk >= 0), other=float("-inf"),
            ).to(tl.float32)
            ids = chunk * CHUNK_SIZE + token_offsets
            selected = (
                query_valid & (chunk >= 0) & (token_offsets < CHUNK_SIZE)
                & (ids >= 0) & (ids < N) & (ids < valid_length)
                & (ids < query_position - LOCAL_WINDOW)
                & (prior > float("-inf")) & lse_valid
            )
            safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
            keys = tl.load(
                GLOBAL_K + batch * sgkb + head * sgkh
                + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
                mask=selected[:, None], other=0.0,
            ).to(tl.float32)
            values = tl.load(
                GLOBAL_V + batch * sgvb + head * sgvh
                + safe_ids[:, None] * sgvn + dimensions[None, :] * sgvd,
                mask=selected[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * scale + prior
            probability = tl.where(selected, tl.exp(scores - safe_lse), 0.0)
            dscore = probability * (
                tl.sum(values * output_gradient[None, :], axis=1)
                - delta + lse_gradient
            )
            if HAS_EVIDENCE:
                route_slot_lse = tl.load(
                    ROUTE_LSE + batch * srlb + head * srlh
                    + query_position * srln + slot * srlk,
                    mask=query_valid, other=float("-inf"),
                ).to(tl.float32)
                route_lse_valid = route_slot_lse > float("-inf")
                safe_route_lse = tl.where(route_lse_valid, route_slot_lse, 0.0)
                route_output = tl.load(
                    ROUTE_OUT + batch * srob + head * sroh
                    + query_position * sron + slot * srok
                    + projected_dimensions * srod,
                    mask=query_valid & (projected_dimensions < RANK), other=0.0,
                ).to(tl.float32)
                route_output_gradient = tl.load(
                    DROUTE_OUT + batch * sdrob + head * sdroh
                    + query_position * sdron + slot * sdrok
                    + projected_dimensions * sdrod,
                    mask=query_valid & (projected_dimensions < RANK), other=0.0,
                ).to(tl.float32)
                route_lse_gradient = tl.load(
                    DROUTE_LSE + batch * sdrlb + head * sdrlh
                    + query_position * sdrln + slot * sdrlk,
                    mask=query_valid & route_lse_valid, other=0.0,
                ).to(tl.float32)
                projected_values = tl.load(
                    PROJECTED_V + batch * spvb + head * spvh
                    + safe_ids[:, None] * spvn
                    + projected_dimensions[None, :] * spvd,
                    mask=selected[:, None] & (projected_dimensions[None, :] < RANK),
                    other=0.0,
                ).to(tl.float32)
                route_probability = tl.where(
                    selected & route_lse_valid,
                    tl.exp(scores - safe_route_lse), 0.0,
                )
                route_delta = tl.sum(route_output_gradient * route_output, axis=0)
                route_dscore = route_probability * (
                    tl.sum(projected_values * route_output_gradient[None, :], axis=1)
                    - route_delta + route_lse_gradient
                )
                dscore += route_dscore
            dquery += tl.sum(dscore[:, None] * keys, axis=0) * scale
            tl.store(
                droute_base + slot * sdrk, tl.sum(dscore, axis=0),
                mask=query_valid & (chunk >= 0),
            )
        tl.store(
            DQ + batch * sdqb + head * sdqh + query_position * sdqn + dimensions * sdqd,
            dquery, mask=query_valid,
        )

    @triton.jit
    def _direct_global_lane_source_block_backward_kernel(
        Q, GLOBAL_K, GLOBAL_V, PROJECTED_V, DOUT, DLSE, LSE, DELTA,
        ROUTE_OUT, DROUTE_OUT, DROUTE_LSE, ROUTE_LSE,
        ROUTE, CHUNKS, LENGTHS, REVERSE_EDGES, CHUNK_OFFSETS,
        DGLOBAL_K, DGLOBAL_V, DPROJECTED_V,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        spvb, spvh, spvn, spvd,
        sdob, sdoh, sdon, sdod,
        sdlb, sdlh, sdln,
        slseb, slseh, slsen,
        sdeb, sdeh, sden,
        srob, sroh, sron, srok, srod,
        sdrob, sdroh, sdron, sdrok, sdrod,
        sdrlb, sdrlh, sdrln, sdrlk,
        srlb, srlh, srln, srlk,
        srb, srh, srn, srk,
        scb, sch, scn, sck,
        sreb, sreh, sree,
        soffb, soffh, soffc,
        sdgkb, sdgkh, sdgkn, sdgkd,
        sdgvb, sdgvh, sdgvn, sdgvd,
        sdpvb, sdpvh, sdpvn, sdpvd,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        RANK: tl.constexpr, RANK_PAD: tl.constexpr,
        K_VAL: tl.constexpr, CHUNK_SIZE: tl.constexpr,
        NUM_CHUNKS: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        SOURCE_BLOCK: tl.constexpr, BLOCKS_PER_CHUNK: tl.constexpr,
        HAS_EVIDENCE: tl.constexpr, ARCH: tl.constexpr,
    ):
        program = tl.program_id(0)
        block_in_chunk = program % BLOCKS_PER_CHUNK
        chunk_program = program // BLOCKS_PER_CHUNK
        source_chunk = chunk_program % NUM_CHUNKS
        batch_head = chunk_program // NUM_CHUNKS
        batch = batch_head // H
        head = batch_head % H
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        local_offsets = block_in_chunk * SOURCE_BLOCK + tl.arange(0, SOURCE_BLOCK)
        key_positions = source_chunk * CHUNK_SIZE + local_offsets
        source_valid = (
            (local_offsets < CHUNK_SIZE) & (key_positions < N)
            & (key_positions < valid_length)
        )
        dimensions = tl.arange(0, HD)
        projected_dimensions = tl.arange(0, RANK_PAD)
        keys = tl.load(
            GLOBAL_K + batch * sgkb + head * sgkh
            + key_positions[:, None] * sgkn + dimensions[None, :] * sgkd,
            mask=source_valid[:, None], other=0.0,
        ).to(tl.float32)
        values = tl.load(
            GLOBAL_V + batch * sgvb + head * sgvh
            + key_positions[:, None] * sgvn + dimensions[None, :] * sgvd,
            mask=source_valid[:, None], other=0.0,
        ).to(tl.float32)
        if HAS_EVIDENCE:
            projected_values = tl.load(
                PROJECTED_V + batch * spvb + head * spvh
                + key_positions[:, None] * spvn
                + projected_dimensions[None, :] * spvd,
                mask=source_valid[:, None] & (projected_dimensions[None, :] < RANK),
                other=0.0,
            ).to(tl.float32)
        offset_base = CHUNK_OFFSETS + batch * soffb + head * soffh
        edge_start = tl.load(offset_base + source_chunk * soffc).to(tl.int32)
        edge_end = tl.load(offset_base + (source_chunk + 1) * soffc).to(tl.int32)
        edge_base = REVERSE_EDGES + batch * sreb + head * sreh
        dkeys = tl.zeros([SOURCE_BLOCK, HD], tl.float32)
        dvalues = tl.zeros([SOURCE_BLOCK, HD], tl.float32)
        if HAS_EVIDENCE:
            dprojected = tl.zeros([SOURCE_BLOCK, RANK_PAD], tl.float32)
        scale = 1.0 / tl.sqrt(HD * 1.0)
        for reverse_offset in tl.range(edge_start, edge_end, num_stages=1, loop_unroll_factor=1):
            edge = tl.load(edge_base + reverse_offset * sree).to(tl.int32)
            query_position = edge // K_VAL
            slot = edge - query_position * K_VAL
            routed_chunk = tl.load(
                CHUNKS + batch * scb + head * sch
                + query_position * scn + slot * sck
            ).to(tl.int32)
            prior = tl.load(
                ROUTE + batch * srb + head * srh
                + query_position * srn + slot * srk
            ).to(tl.float32)
            base_selected = (
                (routed_chunk == source_chunk) & (query_position < valid_length)
                & (prior > float("-inf"))
            )
            selected = source_valid & base_selected & (key_positions < query_position - LOCAL_WINDOW)
            query = tl.load(
                Q + batch * sqb + head * sqh
                + query_position * sqn + dimensions * sqd,
                mask=base_selected, other=0.0,
            ).to(tl.float32)
            output_gradient = tl.load(
                DOUT + batch * sdob + head * sdoh
                + query_position * sdon + dimensions * sdod,
                mask=base_selected, other=0.0,
            ).to(tl.float32)
            lse = tl.load(
                LSE + batch * slseb + head * slseh + query_position * slsen,
                mask=base_selected, other=float("-inf"),
            ).to(tl.float32)
            lse_gradient = tl.load(
                DLSE + batch * sdlb + head * sdlh + query_position * sdln,
                mask=base_selected, other=0.0,
            ).to(tl.float32)
            delta = tl.load(
                DELTA + batch * sdeb + head * sdeh + query_position * sden,
                mask=base_selected, other=0.0,
            ).to(tl.float32)
            lse_valid = lse > float("-inf")
            selected = selected & lse_valid
            safe_lse = tl.where(lse_valid, lse, 0.0)
            score = tl.sum(keys * query[None, :], axis=1) * scale + prior
            probability = tl.where(selected, tl.exp(score - safe_lse), 0.0)
            dscore = probability * (
                tl.sum(output_gradient[None, :] * values, axis=1)
                - delta + lse_gradient
            )
            if HAS_EVIDENCE:
                route_slot_lse = tl.load(
                    ROUTE_LSE + batch * srlb + head * srlh
                    + query_position * srln + slot * srlk,
                    mask=base_selected, other=float("-inf"),
                ).to(tl.float32)
                route_lse_valid = route_slot_lse > float("-inf")
                safe_route_lse = tl.where(route_lse_valid, route_slot_lse, 0.0)
                route_output = tl.load(
                    ROUTE_OUT + batch * srob + head * sroh
                    + query_position * sron + slot * srok
                    + projected_dimensions * srod,
                    mask=base_selected & (projected_dimensions < RANK), other=0.0,
                ).to(tl.float32)
                route_output_gradient = tl.load(
                    DROUTE_OUT + batch * sdrob + head * sdroh
                    + query_position * sdron + slot * sdrok
                    + projected_dimensions * sdrod,
                    mask=base_selected & (projected_dimensions < RANK), other=0.0,
                ).to(tl.float32)
                route_lse_gradient = tl.load(
                    DROUTE_LSE + batch * sdrlb + head * sdrlh
                    + query_position * sdrln + slot * sdrlk,
                    mask=base_selected & route_lse_valid, other=0.0,
                ).to(tl.float32)
                route_probability = tl.where(
                    selected & route_lse_valid, tl.exp(score - safe_route_lse), 0.0
                )
                route_delta = tl.sum(route_output_gradient * route_output, axis=0)
                route_dscore = route_probability * (
                    tl.sum(route_output_gradient[None, :] * projected_values, axis=1)
                    - route_delta + route_lse_gradient
                )
                dscore += route_dscore
                dprojected += route_probability[:, None] * route_output_gradient[None, :]
            dkeys += dscore[:, None] * query[None, :] * scale
            dvalues += probability[:, None] * output_gradient[None, :]
        tl.store(
            DGLOBAL_K + batch * sdgkb + head * sdgkh
            + key_positions[:, None] * sdgkn + dimensions[None, :] * sdgkd,
            dkeys, mask=source_valid[:, None],
        )
        tl.store(
            DGLOBAL_V + batch * sdgvb + head * sdgvh
            + key_positions[:, None] * sdgvn + dimensions[None, :] * sdgvd,
            dvalues, mask=source_valid[:, None],
        )
        if HAS_EVIDENCE:
            tl.store(
                DPROJECTED_V + batch * sdpvb + head * sdpvh
                + key_positions[:, None] * sdpvn
                + projected_dimensions[None, :] * sdpvd,
                dprojected,
                mask=source_valid[:, None] & (projected_dimensions[None, :] < RANK),
            )

    @triton.jit
    def _direct_global_lane_atomic_source_backward_kernel(
        Q, GLOBAL_K, GLOBAL_V, PROJECTED_V, DOUT, DLSE, LSE, DELTA,
        ROUTE_OUT, DROUTE_OUT, DROUTE_LSE, ROUTE_LSE,
        ROUTE, CHUNKS, LENGTHS, DGLOBAL_K, DGLOBAL_V, DPROJECTED_V,
        sqb, sqh, sqn, sqd,
        sgkb, sgkh, sgkn, sgkd,
        sgvb, sgvh, sgvn, sgvd,
        spvb, spvh, spvn, spvd,
        sdob, sdoh, sdon, sdod,
        sdlb, sdlh, sdln,
        slseb, slseh, slsen,
        sdeb, sdeh, sden,
        srob, sroh, sron, srok, srod,
        sdrob, sdroh, sdron, sdrok, sdrod,
        sdrlb, sdrlh, sdrln, sdrlk,
        srlb, srlh, srln, srlk,
        srb, srh, srn, srk,
        scb, sch, scn, sck,
        sdgkb, sdgkh, sdgkn, sdgkd,
        sdgvb, sdgvh, sdgvn, sdgvd,
        sdpvb, sdpvh, sdpvn, sdpvd,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
        RANK: tl.constexpr, RANK_PAD: tl.constexpr,
        K_VAL: tl.constexpr, CHUNK_SIZE: tl.constexpr,
        CHUNK_PAD: tl.constexpr, LOCAL_WINDOW: tl.constexpr,
        HAS_EVIDENCE: tl.constexpr, ARCH: tl.constexpr,
    ):
        program = tl.program_id(0)
        slot = program % K_VAL
        query_program = program // K_VAL
        query_position = query_program % N
        batch_head = query_program // N
        batch = batch_head // H
        head = batch_head % H
        valid_length = tl.load(LENGTHS + batch).to(tl.int32)
        query_valid = query_position < valid_length
        chunk = tl.load(
            CHUNKS + batch * scb + head * sch + query_position * scn + slot * sck,
            mask=query_valid, other=-1,
        ).to(tl.int32)
        prior = tl.load(
            ROUTE + batch * srb + head * srh + query_position * srn + slot * srk,
            mask=query_valid & (chunk >= 0), other=float("-inf"),
        ).to(tl.float32)
        dimensions = tl.arange(0, HD)
        projected_dimensions = tl.arange(0, RANK_PAD)
        token_offsets = tl.arange(0, CHUNK_PAD)
        ids = chunk * CHUNK_SIZE + token_offsets
        selected = (
            query_valid & (chunk >= 0) & (token_offsets < CHUNK_SIZE)
            & (ids >= 0) & (ids < N) & (ids < valid_length)
            & (ids < query_position - LOCAL_WINDOW)
            & (prior > float("-inf"))
        )
        safe_ids = tl.maximum(tl.minimum(ids, N - 1), 0)
        query = tl.load(
            Q + batch * sqb + head * sqh + query_position * sqn + dimensions * sqd,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            GLOBAL_K + batch * sgkb + head * sgkh
            + safe_ids[:, None] * sgkn + dimensions[None, :] * sgkd,
            mask=selected[:, None], other=0.0,
        ).to(tl.float32)
        values = tl.load(
            GLOBAL_V + batch * sgvb + head * sgvh
            + safe_ids[:, None] * sgvn + dimensions[None, :] * sgvd,
            mask=selected[:, None], other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            DOUT + batch * sdob + head * sdoh
            + query_position * sdon + dimensions * sdod,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + batch * slseb + head * slseh + query_position * slsen,
            mask=query_valid, other=float("-inf"),
        ).to(tl.float32)
        lse_gradient = tl.load(
            DLSE + batch * sdlb + head * sdlh + query_position * sdln,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            DELTA + batch * sdeb + head * sdeh + query_position * sden,
            mask=query_valid, other=0.0,
        ).to(tl.float32)
        lse_valid = lse > float("-inf")
        selected = selected & lse_valid
        safe_lse = tl.where(lse_valid, lse, 0.0)
        score = tl.sum(keys * query[None, :], axis=1) / tl.sqrt(HD * 1.0) + prior
        probability = tl.where(selected, tl.exp(score - safe_lse), 0.0)
        dscore = probability * (
            tl.sum(values * output_gradient[None, :], axis=1) - delta + lse_gradient
        )
        if HAS_EVIDENCE:
            route_slot_lse = tl.load(
                ROUTE_LSE + batch * srlb + head * srlh
                + query_position * srln + slot * srlk,
                mask=query_valid, other=float("-inf"),
            ).to(tl.float32)
            route_lse_valid = route_slot_lse > float("-inf")
            safe_route_lse = tl.where(route_lse_valid, route_slot_lse, 0.0)
            route_output = tl.load(
                ROUTE_OUT + batch * srob + head * sroh
                + query_position * sron + slot * srok
                + projected_dimensions * srod,
                mask=query_valid & (projected_dimensions < RANK), other=0.0,
            ).to(tl.float32)
            route_output_gradient = tl.load(
                DROUTE_OUT + batch * sdrob + head * sdroh
                + query_position * sdron + slot * sdrok
                + projected_dimensions * sdrod,
                mask=query_valid & (projected_dimensions < RANK), other=0.0,
            ).to(tl.float32)
            route_lse_gradient = tl.load(
                DROUTE_LSE + batch * sdrlb + head * sdrlh
                + query_position * sdrln + slot * sdrlk,
                mask=query_valid & route_lse_valid, other=0.0,
            ).to(tl.float32)
            projected_values = tl.load(
                PROJECTED_V + batch * spvb + head * spvh
                + safe_ids[:, None] * spvn
                + projected_dimensions[None, :] * spvd,
                mask=selected[:, None] & (projected_dimensions[None, :] < RANK),
                other=0.0,
            ).to(tl.float32)
            route_probability = tl.where(
                selected & route_lse_valid, tl.exp(score - safe_route_lse), 0.0
            )
            route_delta = tl.sum(route_output_gradient * route_output, axis=0)
            route_dscore = route_probability * (
                tl.sum(projected_values * route_output_gradient[None, :], axis=1)
                - route_delta + route_lse_gradient
            )
            dscore += route_dscore
            tl.atomic_add(
                DPROJECTED_V + batch * sdpvb + head * sdpvh
                + safe_ids[:, None] * sdpvn
                + projected_dimensions[None, :] * sdpvd,
                route_probability[:, None] * route_output_gradient[None, :],
                mask=selected[:, None] & (projected_dimensions[None, :] < RANK),
            )
        tl.atomic_add(
            DGLOBAL_K + batch * sdgkb + head * sdgkh
            + safe_ids[:, None] * sdgkn + dimensions[None, :] * sdgkd,
            dscore[:, None] * query[None, :] / tl.sqrt(HD * 1.0),
            mask=selected[:, None],
        )
        tl.atomic_add(
            DGLOBAL_V + batch * sdgvb + head * sdgvh
            + safe_ids[:, None] * sdgvn + dimensions[None, :] * sdgvd,
            probability[:, None] * output_gradient[None, :],
            mask=selected[:, None],
        )

    @triton.jit
    def _reverse_edge_count_kernel(
        FLAT_CHUNKS, COUNTS,
        TOTAL: tl.constexpr, EDGES: tl.constexpr,
        NUM_CHUNKS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        row = offsets // EDGES
        chunk = tl.load(FLAT_CHUNKS + offsets, mask=valid, other=-1).to(tl.int32)
        selected = valid & (chunk >= 0) & (chunk < NUM_CHUNKS)
        tl.atomic_add(COUNTS + row * NUM_CHUNKS + chunk, 1, mask=selected)

    @triton.jit
    def _reverse_edge_scatter_kernel(
        FLAT_CHUNKS, OFFSETS, CURSORS, REVERSE_EDGES,
        TOTAL: tl.constexpr, EDGES: tl.constexpr,
        NUM_CHUNKS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        edge = offsets % EDGES
        row = offsets // EDGES
        chunk = tl.load(FLAT_CHUNKS + offsets, mask=valid, other=-1).to(tl.int32)
        selected = valid & (chunk >= 0) & (chunk < NUM_CHUNKS)
        cursor = tl.atomic_add(
            CURSORS + row * NUM_CHUNKS + chunk, 1, mask=selected
        )
        base = tl.load(
            OFFSETS + row * (NUM_CHUNKS + 1) + chunk,
            mask=selected, other=0,
        ).to(tl.int32)
        tl.store(REVERSE_EDGES + row * EDGES + base + cursor, edge, mask=selected)


if _TRITON_AVAILABLE:
    _direct_forward_autotuned_kernel = triton.autotune(
        configs=[
            triton.Config({}, num_warps=warps, num_stages=stages)
            for warps, stages in _direct_global_autotune_contract()["forward"]["configs"]
        ],
        key=list(_DIRECT_GLOBAL_AUTOTUNE_KEY),
        restore_value=["OUT", "LSE", "ROUTE_OUT", "ROUTE_LSE"],
        cache_results=True,
    )(_direct_global_lane_forward_kernel)
    _direct_query_backward_autotuned_kernel = triton.autotune(
        configs=[
            triton.Config({}, num_warps=warps, num_stages=stages)
            for warps, stages in _direct_global_autotune_contract()["query_backward"]["configs"]
        ],
        key=list(_DIRECT_GLOBAL_AUTOTUNE_KEY),
        reset_to_zero=["DQ", "DROUTE", "DELTA"],
        cache_results=True,
    )(_direct_global_lane_query_backward_kernel)
    _direct_source_backward_autotuned_kernel = triton.autotune(
        configs=[
            triton.Config({}, num_warps=warps, num_stages=stages)
            for warps, stages in _direct_global_autotune_contract()["source_backward"]["configs"]
        ],
        key=list(_DIRECT_SOURCE_AUTOTUNE_KEY),
        reset_to_zero=["DGLOBAL_K", "DGLOBAL_V", "DPROJECTED_V"],
        cache_results=True,
    )(_direct_global_lane_source_block_backward_kernel)
else:  # pragma: no cover - CPU-only imports
    _direct_forward_autotuned_kernel = None
    _direct_query_backward_autotuned_kernel = None
    _direct_source_backward_autotuned_kernel = None


_BACKWARD_SOURCE_BLOCK_STABLE = 0
_BACKWARD_SOURCE_BLOCK_COUNTING = 1
_BACKWARD_ATOMIC = 2


def _backward_impl_code(value: str | int) -> int:
    if isinstance(value, int):
        if value not in {
            _BACKWARD_SOURCE_BLOCK_STABLE,
            _BACKWARD_SOURCE_BLOCK_COUNTING,
            _BACKWARD_ATOMIC,
        }:
            raise ValueError("invalid HISA backward implementation code")
        return value
    normalized = str(value).lower()
    aliases = {
        "source_owned": _BACKWARD_SOURCE_BLOCK_STABLE,
        "source_block_stable": _BACKWARD_SOURCE_BLOCK_STABLE,
        "source_block_counting": _BACKWARD_SOURCE_BLOCK_COUNTING,
        "atomic": _BACKWARD_ATOMIC,
        "atomic_masked": _BACKWARD_ATOMIC,
    }
    if normalized not in aliases:
        raise ValueError(
            "backward_impl must be source_owned/source_block_stable, "
            "source_block_counting, atomic, or atomic_masked"
        )
    return aliases[normalized]


def _stable_chunk_reverse_edges(
    chunks: torch.Tensor,
    *,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic q-major reverse edges and source-chunk offsets."""
    flat = chunks.reshape(*chunks.shape[:2], -1)
    sentinel = torch.full_like(flat, int(num_chunks))
    sortable = torch.where((flat >= 0) & (flat < int(num_chunks)), flat, sentinel)
    sorted_chunks, reverse_edges = torch.sort(sortable, dim=-1, stable=True)
    boundaries = torch.arange(
        int(num_chunks) + 1, device=chunks.device, dtype=sorted_chunks.dtype
    ).reshape(1, 1, -1).expand(*chunks.shape[:2], -1)
    offsets = torch.searchsorted(
        sorted_chunks.contiguous(), boundaries.contiguous(), right=False
    )
    return reverse_edges.to(torch.int32), offsets.to(torch.int32)


def _counting_chunk_reverse_edges(
    chunks: torch.Tensor,
    *,
    num_chunks: int,
    use_wrap_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(E+C) bounded-ID CSR builder; edge order is not bitwise deterministic."""
    if not chunks.is_cuda or not _TRITON_AVAILABLE:
        return _stable_chunk_reverse_edges(chunks, num_chunks=num_chunks)
    batch_size, heads = chunks.shape[:2]
    flat = chunks.reshape(batch_size * heads, -1).contiguous().to(torch.int32)
    edges = flat.shape[1]
    counts = torch.zeros(
        batch_size * heads, int(num_chunks), device=chunks.device, dtype=torch.int32
    )
    block = 256
    total = flat.numel()
    count_kernel = (
        torch.library.wrap_triton(_reverse_edge_count_kernel)
        if use_wrap_triton else _reverse_edge_count_kernel
    )
    count_kernel[(math.ceil(total / block),)](
        flat, counts,
        TOTAL=total, EDGES=edges, NUM_CHUNKS=int(num_chunks), BLOCK=block,
        num_warps=4,
    )
    offsets = torch.cat(
        (torch.zeros_like(counts[:, :1]), torch.cumsum(counts, dim=-1)), dim=-1
    ).to(torch.int32)
    cursors = torch.zeros_like(counts)
    reverse_edges = torch.empty_like(flat)
    scatter_kernel = (
        torch.library.wrap_triton(_reverse_edge_scatter_kernel)
        if use_wrap_triton else _reverse_edge_scatter_kernel
    )
    scatter_kernel[(math.ceil(total / block),)](
        flat, offsets, cursors, reverse_edges,
        TOTAL=total, EDGES=edges, NUM_CHUNKS=int(num_chunks), BLOCK=block,
        num_warps=4,
    )
    return (
        reverse_edges.reshape(batch_size, heads, edges),
        offsets.reshape(batch_size, heads, int(num_chunks) + 1),
    )


def _validate_direct_global_inputs(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    projected_value: torch.Tensor | None,
    route: torch.Tensor,
    chunks: torch.Tensor,
) -> tuple[int, int, int, int, int, bool]:
    if query.ndim != 4:
        raise ValueError("query must have shape [B,H,N,HD]")
    batch_size, heads, seq_len, head_dim = query.shape
    if global_key.shape != query.shape or global_value.shape != query.shape:
        raise ValueError("global key/value must match query [B,H,N,HD]")
    has_evidence = projected_value is not None
    rank = 1
    if has_evidence:
        assert projected_value is not None
        if projected_value.ndim != 4 or projected_value.shape[:3] != query.shape[:3]:
            raise ValueError("projected_value must have shape [B,H,N,R]")
        rank = int(projected_value.shape[-1])
        if rank < 1:
            raise ValueError("projected route evidence rank must be positive")
    if chunks.shape[:3] != (batch_size, heads, seq_len):
        raise ValueError("chunks must have shape [B,H,N,K]")
    if route.shape != chunks.shape:
        raise ValueError("route and chunks must have identical [B,H,N,K] shape")
    if head_dim < 16 or not _is_power_of_two(head_dim):
        raise ValueError("Triton HISA requires a power-of-two head dimension >=16")
    return batch_size, heads, seq_len, head_dim, rank, has_evidence


def _selected_page_lse_triton_apply(
    query: torch.Tensor,
    global_key: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    chunk_size: int,
    local_window: int,
) -> torch.Tensor:
    if not query.is_cuda or not _TRITON_AVAILABLE:
        return _candidate_page_lse_reference(
            query, global_key, chunks, valid_lengths,
            chunk_size=int(chunk_size), local_window=int(local_window),
        )
    batch_size, heads, seq_len, head_dim = query.shape
    output = torch.full(
        (*chunks.shape,), float("-inf"), device=query.device, dtype=torch.float32
    )
    lengths = valid_lengths.reshape(-1).contiguous()
    chunk_pad = max(16, _next_pow2(int(chunk_size)))
    selected_lse_kernel = (
        torch.library.wrap_triton(_selected_page_lse_kernel)
        if _TRITON_LIBRARY_API_AVAILABLE else _selected_page_lse_kernel
    )
    selected_lse_kernel[(batch_size * heads * seq_len * chunks.shape[-1],)](
        query, global_key, chunks, lengths, output,
        *query.stride(), *global_key.stride(), *chunks.stride(), *output.stride(),
        B=batch_size, N=seq_len, H=heads, HD=head_dim,
        K_VAL=chunks.shape[-1], CHUNK_SIZE=int(chunk_size),
        CHUNK_PAD=chunk_pad, LOCAL_WINDOW=int(local_window),
        num_warps=4, num_stages=2,
    )
    return output


def _direct_global_forward_impl(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    projected_value: torch.Tensor | None,
    route: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    chunk_size: int,
    local_window: int,
    *,
    use_wrap_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, heads, seq_len, head_dim, rank, has_evidence = (
        _validate_direct_global_inputs(
            query, global_key, global_value, projected_value, route, chunks
        )
    )
    chunk_pad = max(16, _next_pow2(int(chunk_size)))
    rank_pad = _next_pow2(rank)
    if chunk_pad > 256:
        raise ValueError("HISA global lane supports at most 256 tokens per chunk")
    lengths = valid_lengths.reshape(-1).contiguous()
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, heads, seq_len), float("-inf"),
        device=query.device, dtype=torch.float32,
    )
    projected_kernel = projected_value if projected_value is not None else global_value[..., :1]
    if has_evidence:
        route_output = torch.zeros(
            batch_size, heads, seq_len, chunks.shape[-1], rank,
            device=projected_kernel.device, dtype=projected_kernel.dtype,
        )
        route_lse = torch.full(
            (batch_size, heads, seq_len, chunks.shape[-1]), float("-inf"),
            device=query.device, dtype=torch.float32,
        )
        route_output_strides = route_output.stride()
        route_lse_strides = route_lse.stride()
    else:
        route_output = torch.empty(1, device=query.device, dtype=query.dtype)
        route_lse = torch.empty(1, device=query.device, dtype=torch.float32)
        route_output_strides = (0, 0, 0, 0, 0)
        route_lse_strides = (0, 0, 0, 0)
    selected_kernel = (
        _direct_forward_autotuned_kernel
        if _DIRECT_GLOBAL_AUTOTUNE_ENABLED else _direct_global_lane_forward_kernel
    )
    if selected_kernel is None:
        raise RuntimeError("direct forward Triton kernel is unavailable")
    cache_entries_before = len(selected_kernel.cache) if _DIRECT_GLOBAL_AUTOTUNE_ENABLED else 0
    kernel = torch.library.wrap_triton(selected_kernel) if use_wrap_triton else selected_kernel
    grid = (batch_size * heads * seq_len,)
    launch_kwargs = {
        "B": batch_size, "N": seq_len, "H": heads, "HD": head_dim,
        "RANK": rank, "RANK_PAD": rank_pad, "K_VAL": chunks.shape[-1],
        "CHUNK_SIZE": int(chunk_size), "CHUNK_PAD": chunk_pad,
        "LOCAL_WINDOW": int(local_window), "HAS_EVIDENCE": has_evidence,
        "ARCH": _cuda_arch_code(query.device),
    }
    if not _DIRECT_GLOBAL_AUTOTUNE_ENABLED:
        launch_kwargs.update(num_warps=4, num_stages=2)

    def launch() -> None:
        kernel[grid](
            query, global_key, global_value, projected_kernel, route, chunks, lengths,
            output, lse, route_output, route_lse,
            *query.stride(), *global_key.stride(), *global_value.stride(),
            *projected_kernel.stride(), *route.stride(), *chunks.stride(),
            *output.stride(), *lse.stride(),
            *route_output_strides, *route_lse_strides,
            **launch_kwargs,
        )
    launch()
    if _DIRECT_GLOBAL_AUTOTUNE_ENABLED and len(selected_kernel.cache) > cache_entries_before:
        output.zero_(); lse.fill_(float("-inf"))
        if has_evidence:
            route_output.zero_(); route_lse.fill_(float("-inf"))
        launch()
    return output, lse, route_output, route_lse


def _prepare_direct_global_gradients(
    output: torch.Tensor,
    lse: torch.Tensor,
    route_output: torch.Tensor,
    route_lse: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_lse: torch.Tensor | None,
    grad_route_output: torch.Tensor | None,
    grad_route_lse: torch.Tensor | None,
    *,
    has_evidence: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = torch.zeros_like(output) if grad_output is None else grad_output.contiguous()
    grad_lse = torch.zeros_like(lse) if grad_lse is None else grad_lse.contiguous()
    lane_valid = torch.isfinite(lse)
    grad_output = torch.where(
        lane_valid[..., None], grad_output, torch.zeros_like(grad_output)
    ).contiguous()
    grad_lse = torch.where(
        lane_valid & torch.isfinite(grad_lse), grad_lse, torch.zeros_like(grad_lse)
    ).contiguous()
    if not has_evidence:
        return (
            grad_output, grad_lse,
            torch.zeros_like(route_output), torch.zeros_like(route_lse),
        )
    grad_route_output = (
        torch.zeros_like(route_output)
        if grad_route_output is None else grad_route_output.contiguous()
    )
    grad_route_lse = (
        torch.zeros_like(route_lse)
        if grad_route_lse is None else grad_route_lse.contiguous()
    )
    route_valid = torch.isfinite(route_lse)
    grad_route_output = torch.where(
        route_valid[..., None], grad_route_output, torch.zeros_like(grad_route_output)
    ).contiguous()
    grad_route_lse = torch.where(
        route_valid & torch.isfinite(grad_route_lse),
        grad_route_lse, torch.zeros_like(grad_route_lse),
    ).contiguous()
    return grad_output, grad_lse, grad_route_output, grad_route_lse


def _direct_global_backward_impl(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    projected_value: torch.Tensor | None,
    route: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    route_output: torch.Tensor,
    route_lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor,
    grad_route_output: torch.Tensor,
    grad_route_lse: torch.Tensor,
    chunk_size: int,
    local_window: int,
    backward_mode: int,
    source_block_size: int,
    *,
    use_wrap_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    batch_size, heads, seq_len, head_dim, rank, has_evidence = (
        _validate_direct_global_inputs(
            query, global_key, global_value, projected_value, route, chunks
        )
    )
    mode = _backward_impl_code(backward_mode)
    lengths = valid_lengths.reshape(-1).contiguous()
    projected_kernel = projected_value if projected_value is not None else global_value[..., :1]
    dquery = torch.zeros_like(query)
    droute = torch.zeros_like(route)
    delta = torch.zeros_like(lse, dtype=torch.float32)
    chunk_pad = max(16, _next_pow2(int(chunk_size)))
    rank_pad = _next_pow2(rank)
    route_output_strides = route_output.stride() if has_evidence else (0, 0, 0, 0, 0)
    route_lse_strides = route_lse.stride() if has_evidence else (0, 0, 0, 0)
    grad_route_output_strides = grad_route_output.stride() if has_evidence else (0, 0, 0, 0, 0)
    grad_route_lse_strides = grad_route_lse.stride() if has_evidence else (0, 0, 0, 0)

    selected_query_kernel = (
        _direct_query_backward_autotuned_kernel
        if _DIRECT_GLOBAL_AUTOTUNE_ENABLED else _direct_global_lane_query_backward_kernel
    )
    if selected_query_kernel is None:
        raise RuntimeError("direct query-backward Triton kernel is unavailable")
    query_kernel = torch.library.wrap_triton(selected_query_kernel) if use_wrap_triton else selected_query_kernel
    query_kwargs = {
        "B": batch_size, "N": seq_len, "H": heads, "HD": head_dim,
        "RANK": rank, "RANK_PAD": rank_pad, "K_VAL": chunks.shape[-1],
        "CHUNK_SIZE": int(chunk_size), "CHUNK_PAD": chunk_pad,
        "LOCAL_WINDOW": int(local_window), "HAS_EVIDENCE": has_evidence,
        "ARCH": _cuda_arch_code(query.device),
    }
    if not _DIRECT_GLOBAL_AUTOTUNE_ENABLED:
        query_kwargs.update(num_warps=4, num_stages=2)
    query_kernel[(batch_size * heads * seq_len,)](
        query, global_key, global_value, projected_kernel,
        output, grad_output, grad_lse, lse,
        route_output, grad_route_output, grad_route_lse, route_lse,
        route, chunks, lengths, dquery, droute, delta,
        *query.stride(), *global_key.stride(), *global_value.stride(),
        *projected_kernel.stride(), *output.stride(), *grad_output.stride(),
        *grad_lse.stride(), *lse.stride(),
        *route_output_strides, *grad_route_output_strides,
        *grad_route_lse_strides, *route_lse_strides,
        *route.stride(), *chunks.stride(), *dquery.stride(),
        *droute.stride(), *delta.stride(),
        **query_kwargs,
    )

    if mode == _BACKWARD_ATOMIC:
        dglobal_key_acc = torch.zeros_like(global_key, dtype=torch.float32)
        dglobal_value_acc = torch.zeros_like(global_value, dtype=torch.float32)
        dprojected_acc = (
            torch.zeros_like(projected_kernel, dtype=torch.float32)
            if has_evidence else torch.empty(1, device=query.device, dtype=torch.float32)
        )
        dprojected_strides = dprojected_acc.stride() if has_evidence else (0, 0, 0, 0)
        atomic_kernel = (
            torch.library.wrap_triton(_direct_global_lane_atomic_source_backward_kernel)
            if use_wrap_triton else _direct_global_lane_atomic_source_backward_kernel
        )
        atomic_kernel[(batch_size * heads * seq_len * chunks.shape[-1],)](
            query, global_key, global_value, projected_kernel,
            grad_output, grad_lse, lse, delta,
            route_output, grad_route_output, grad_route_lse, route_lse,
            route, chunks, lengths,
            dglobal_key_acc, dglobal_value_acc, dprojected_acc,
            *query.stride(), *global_key.stride(), *global_value.stride(),
            *projected_kernel.stride(), *grad_output.stride(), *grad_lse.stride(),
            *lse.stride(), *delta.stride(),
            *route_output_strides, *grad_route_output_strides,
            *grad_route_lse_strides, *route_lse_strides,
            *route.stride(), *chunks.stride(),
            *dglobal_key_acc.stride(), *dglobal_value_acc.stride(),
            *dprojected_strides,
            B=batch_size, N=seq_len, H=heads, HD=head_dim,
            RANK=rank, RANK_PAD=rank_pad, K_VAL=chunks.shape[-1],
            CHUNK_SIZE=int(chunk_size), CHUNK_PAD=chunk_pad,
            LOCAL_WINDOW=int(local_window), HAS_EVIDENCE=has_evidence,
            ARCH=_cuda_arch_code(query.device), num_warps=4, num_stages=1,
        )
        dglobal_key = dglobal_key_acc.to(global_key.dtype)
        dglobal_value = dglobal_value_acc.to(global_value.dtype)
        dprojected_value = dprojected_acc.to(projected_kernel.dtype) if has_evidence else None
    else:
        num_chunks = math.ceil(seq_len / int(chunk_size))
        if mode == _BACKWARD_SOURCE_BLOCK_COUNTING:
            reverse_edges, chunk_offsets = _counting_chunk_reverse_edges(
                chunks, num_chunks=num_chunks, use_wrap_triton=use_wrap_triton
            )
        else:
            reverse_edges, chunk_offsets = _stable_chunk_reverse_edges(
                chunks, num_chunks=num_chunks
            )
        dglobal_key = torch.zeros_like(global_key)
        dglobal_value = torch.zeros_like(global_value)
        dprojected_kernel = (
            torch.zeros_like(projected_kernel)
            if has_evidence else torch.empty(1, device=query.device, dtype=query.dtype)
        )
        dprojected_strides = dprojected_kernel.stride() if has_evidence else (0, 0, 0, 0)
        source_block = int(source_block_size)
        if source_block < 1 or not _is_power_of_two(source_block) or source_block > 32:
            raise ValueError("source_block_size must be a power of two in [1,32]")
        blocks_per_chunk = math.ceil(int(chunk_size) / source_block)
        selected_source_kernel = (
            _direct_source_backward_autotuned_kernel
            if _DIRECT_GLOBAL_AUTOTUNE_ENABLED
            else _direct_global_lane_source_block_backward_kernel
        )
        if selected_source_kernel is None:
            raise RuntimeError("direct source-block backward Triton kernel is unavailable")
        source_kernel = torch.library.wrap_triton(selected_source_kernel) if use_wrap_triton else selected_source_kernel
        source_kwargs = {
            "B": batch_size, "N": seq_len, "H": heads, "HD": head_dim,
            "RANK": rank, "RANK_PAD": rank_pad, "K_VAL": chunks.shape[-1],
            "CHUNK_SIZE": int(chunk_size), "NUM_CHUNKS": num_chunks,
            "LOCAL_WINDOW": int(local_window), "SOURCE_BLOCK": source_block,
            "BLOCKS_PER_CHUNK": blocks_per_chunk,
            "HAS_EVIDENCE": has_evidence, "ARCH": _cuda_arch_code(query.device),
        }
        if not _DIRECT_GLOBAL_AUTOTUNE_ENABLED:
            source_kwargs.update(num_warps=4, num_stages=1)
        source_kernel[(batch_size * heads * num_chunks * blocks_per_chunk,)](
            query, global_key, global_value, projected_kernel,
            grad_output, grad_lse, lse, delta,
            route_output, grad_route_output, grad_route_lse, route_lse,
            route, chunks, lengths, reverse_edges, chunk_offsets,
            dglobal_key, dglobal_value, dprojected_kernel,
            *query.stride(), *global_key.stride(), *global_value.stride(),
            *projected_kernel.stride(), *grad_output.stride(), *grad_lse.stride(),
            *lse.stride(), *delta.stride(),
            *route_output_strides, *grad_route_output_strides,
            *grad_route_lse_strides, *route_lse_strides,
            *route.stride(), *chunks.stride(), *reverse_edges.stride(),
            *chunk_offsets.stride(), *dglobal_key.stride(),
            *dglobal_value.stride(), *dprojected_strides,
            **source_kwargs,
        )
        dprojected_value = dprojected_kernel if has_evidence else None
    return dquery, dglobal_key, dglobal_value, dprojected_value, droute


if _TRITON_LIBRARY_API_AVAILABLE:

    @torch.library.triton_op("dwarf_hisa_v19::direct_global_lane_evidence", mutates_args={})
    def _direct_global_hisa_evidence_op(
        query: torch.Tensor, global_key: torch.Tensor, global_value: torch.Tensor,
        projected_value: torch.Tensor, route: torch.Tensor, chunks: torch.Tensor,
        valid_lengths: torch.Tensor, chunk_size: int, local_window: int,
        backward_mode: int, source_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _direct_global_forward_impl(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, chunk_size, local_window, use_wrap_triton=True,
        )

    @torch.library.triton_op("dwarf_hisa_v19::direct_global_lane_evidence_backward", mutates_args={})
    def _direct_global_hisa_evidence_backward_op(
        query: torch.Tensor, global_key: torch.Tensor, global_value: torch.Tensor,
        projected_value: torch.Tensor, route: torch.Tensor, chunks: torch.Tensor,
        valid_lengths: torch.Tensor, output: torch.Tensor, lse: torch.Tensor,
        route_output: torch.Tensor, route_lse: torch.Tensor,
        grad_output: torch.Tensor, grad_lse: torch.Tensor,
        grad_route_output: torch.Tensor, grad_route_lse: torch.Tensor,
        chunk_size: int, local_window: int, backward_mode: int,
        source_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = _direct_global_backward_impl(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse,
            grad_output, grad_lse, grad_route_output, grad_route_lse,
            chunk_size, local_window, backward_mode, source_block_size,
            use_wrap_triton=True,
        )
        assert result[3] is not None
        return result[0], result[1], result[2], result[3], result[4]

    def _setup_evidence_context(ctx, inputs, output) -> None:
        (*tensors, chunk_size, local_window, backward_mode, source_block_size) = inputs
        output_tensor, lse, route_output, route_lse = output
        ctx.save_for_backward(*tensors, output_tensor, lse, route_output, route_lse)
        ctx.chunk_size = int(chunk_size); ctx.local_window = int(local_window)
        ctx.backward_mode = int(backward_mode); ctx.source_block_size = int(source_block_size)

    def _evidence_registered_backward(ctx, grad_output, grad_lse, grad_route_output, grad_route_lse):
        (
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse,
        ) = ctx.saved_tensors
        grads = _prepare_direct_global_gradients(
            output, lse, route_output, route_lse,
            grad_output, grad_lse, grad_route_output, grad_route_lse,
            has_evidence=True,
        )
        dquery, dkey, dvalue, dprojected, droute = _direct_global_hisa_evidence_backward_op(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse,
            *grads, ctx.chunk_size, ctx.local_window,
            ctx.backward_mode, ctx.source_block_size,
        )
        return dquery, dkey, dvalue, dprojected, droute, None, None, None, None, None, None

    torch.library.register_autograd(
        "dwarf_hisa_v19::direct_global_lane_evidence",
        _evidence_registered_backward, setup_context=_setup_evidence_context,
    )

    @torch.library.triton_op("dwarf_hisa_v19::direct_global_lane_aggregate", mutates_args={})
    def _direct_global_hisa_aggregate_op(
        query: torch.Tensor, global_key: torch.Tensor, global_value: torch.Tensor,
        route: torch.Tensor, chunks: torch.Tensor, valid_lengths: torch.Tensor,
        chunk_size: int, local_window: int, backward_mode: int,
        source_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse, _, _ = _direct_global_forward_impl(
            query, global_key, global_value, None, route, chunks, valid_lengths,
            chunk_size, local_window, use_wrap_triton=True,
        )
        return output, lse

    @torch.library.triton_op("dwarf_hisa_v19::direct_global_lane_aggregate_backward", mutates_args={})
    def _direct_global_hisa_aggregate_backward_op(
        query: torch.Tensor, global_key: torch.Tensor, global_value: torch.Tensor,
        route: torch.Tensor, chunks: torch.Tensor, valid_lengths: torch.Tensor,
        output: torch.Tensor, lse: torch.Tensor,
        grad_output: torch.Tensor, grad_lse: torch.Tensor,
        chunk_size: int, local_window: int, backward_mode: int,
        source_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dummy_output = torch.empty(1, device=query.device, dtype=query.dtype)
        dummy_lse = torch.empty(1, device=query.device, dtype=torch.float32)
        grads = _prepare_direct_global_gradients(
            output, lse, dummy_output, dummy_lse,
            grad_output, grad_lse, None, None, has_evidence=False,
        )
        result = _direct_global_backward_impl(
            query, global_key, global_value, None, route, chunks, valid_lengths,
            output, lse, dummy_output, dummy_lse, *grads,
            chunk_size, local_window, backward_mode, source_block_size,
            use_wrap_triton=True,
        )
        return result[0], result[1], result[2], result[4]

    def _setup_aggregate_context(ctx, inputs, output) -> None:
        (*tensors, chunk_size, local_window, backward_mode, source_block_size) = inputs
        output_tensor, lse = output
        ctx.save_for_backward(*tensors, output_tensor, lse)
        ctx.chunk_size = int(chunk_size); ctx.local_window = int(local_window)
        ctx.backward_mode = int(backward_mode); ctx.source_block_size = int(source_block_size)

    def _aggregate_registered_backward(ctx, grad_output, grad_lse):
        query, global_key, global_value, route, chunks, valid_lengths, output, lse = ctx.saved_tensors
        dquery, dkey, dvalue, droute = _direct_global_hisa_aggregate_backward_op(
            query, global_key, global_value, route, chunks, valid_lengths,
            output, lse, grad_output, grad_lse,
            ctx.chunk_size, ctx.local_window, ctx.backward_mode, ctx.source_block_size,
        )
        return dquery, dkey, dvalue, droute, None, None, None, None, None, None

    torch.library.register_autograd(
        "dwarf_hisa_v19::direct_global_lane_aggregate",
        _aggregate_registered_backward, setup_context=_setup_aggregate_context,
    )


class _LegacyDirectGlobalHISAEvidenceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, global_key, global_value, projected_value, route, chunks,
                valid_lengths, chunk_size, local_window, backward_mode, source_block_size):
        output, lse, route_output, route_lse = _direct_global_forward_impl(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, chunk_size, local_window, use_wrap_triton=False,
        )
        ctx.save_for_backward(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse,
        )
        ctx.chunk_size=int(chunk_size); ctx.local_window=int(local_window)
        ctx.backward_mode=int(backward_mode); ctx.source_block_size=int(source_block_size)
        return output, lse, route_output, route_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse, grad_route_output, grad_route_lse):
        (
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse,
        ) = ctx.saved_tensors
        grads = _prepare_direct_global_gradients(
            output, lse, route_output, route_lse,
            grad_output, grad_lse, grad_route_output, grad_route_lse,
            has_evidence=True,
        )
        result = _direct_global_backward_impl(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, output, lse, route_output, route_lse, *grads,
            ctx.chunk_size, ctx.local_window, ctx.backward_mode,
            ctx.source_block_size, use_wrap_triton=False,
        )
        return (*result, None, None, None, None, None, None)


class _LegacyDirectGlobalHISAAggregateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, global_key, global_value, route, chunks, valid_lengths,
                chunk_size, local_window, backward_mode, source_block_size):
        output, lse, _, _ = _direct_global_forward_impl(
            query, global_key, global_value, None, route, chunks, valid_lengths,
            chunk_size, local_window, use_wrap_triton=False,
        )
        ctx.save_for_backward(
            query, global_key, global_value, route, chunks, valid_lengths, output, lse
        )
        ctx.chunk_size=int(chunk_size); ctx.local_window=int(local_window)
        ctx.backward_mode=int(backward_mode); ctx.source_block_size=int(source_block_size)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        query, global_key, global_value, route, chunks, valid_lengths, output, lse = ctx.saved_tensors
        dummy_output = torch.empty(1, device=query.device, dtype=query.dtype)
        dummy_lse = torch.empty(1, device=query.device, dtype=torch.float32)
        grads = _prepare_direct_global_gradients(
            output, lse, dummy_output, dummy_lse,
            grad_output, grad_lse, None, None, has_evidence=False,
        )
        result = _direct_global_backward_impl(
            query, global_key, global_value, None, route, chunks, valid_lengths,
            output, lse, dummy_output, dummy_lse, *grads,
            ctx.chunk_size, ctx.local_window, ctx.backward_mode,
            ctx.source_block_size, use_wrap_triton=False,
        )
        return result[0], result[1], result[2], result[4], None, None, None, None, None, None


def _direct_global_hisa_triton_apply(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    route: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    chunk_size: int,
    local_window: int,
    *,
    projected_value: torch.Tensor | None = None,
    backward_impl: str | int = "source_block_stable",
    source_block_size: int = 8,
):
    if not query.is_cuda or not _TRITON_AVAILABLE:
        metadata = HISAMetadata(
            top_chunk_idx=chunks,
            tile_starts=torch.arange(query.shape[2], device=query.device, dtype=torch.int32),
            valid_lengths=valid_lengths, chunk_size=int(chunk_size), selector_tile_size=1,
        )
        if projected_value is None:
            return _eager_global_lane(
                query, global_key, global_value, route, metadata,
                local_window=int(local_window),
            )
        return _direct_global_hisa_reference(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, int(chunk_size), int(local_window),
        )
    backward_mode = _backward_impl_code(backward_impl)
    if projected_value is None:
        if _TRITON_LIBRARY_API_AVAILABLE:
            return _direct_global_hisa_aggregate_op(
                query, global_key, global_value, route, chunks, valid_lengths,
                int(chunk_size), int(local_window), backward_mode,
                int(source_block_size),
            )
        return _LegacyDirectGlobalHISAAggregateFn.apply(
            query, global_key, global_value, route, chunks, valid_lengths,
            int(chunk_size), int(local_window), backward_mode,
            int(source_block_size),
        )
    if _TRITON_LIBRARY_API_AVAILABLE:
        return _direct_global_hisa_evidence_op(
            query, global_key, global_value, projected_value, route, chunks,
            valid_lengths, int(chunk_size), int(local_window), backward_mode,
            int(source_block_size),
        )
    return _LegacyDirectGlobalHISAEvidenceFn.apply(
        query, global_key, global_value, projected_value, route, chunks,
        valid_lengths, int(chunk_size), int(local_window), backward_mode,
        int(source_block_size),
    )


def benchmark_direct_global_backward_variants(
    query: torch.Tensor,
    global_key: torch.Tensor,
    global_value: torch.Tensor,
    route: torch.Tensor,
    chunks: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    chunk_size: int,
    local_window: int,
    projected_value: torch.Tensor | None = None,
    source_block_size: int = 8,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float]:
    """Benchmark the real stable-CSR, counting-CSR, and atomic backward paths."""
    if not query.is_cuda:
        raise RuntimeError("HISA backward benchmarking requires CUDA")
    timings: dict[str, float] = {}
    for name in ("source_block_stable", "source_block_counting", "atomic"):
        def run() -> None:
            q=query.detach().requires_grad_(True)
            k=global_key.detach().requires_grad_(True)
            v=global_value.detach().requires_grad_(True)
            r=route.detach().requires_grad_(True)
            p=None if projected_value is None else projected_value.detach().requires_grad_(True)
            result=_direct_global_hisa_triton_apply(
                q,k,v,r,chunks,valid_lengths,chunk_size,local_window,
                projected_value=p,backward_impl=name,
                source_block_size=source_block_size,
            )
            outputs=result if isinstance(result, tuple) else (result,)
            loss=sum(t.float().nan_to_num().square().mean() for t in outputs)
            loss.backward()
        for _ in range(int(warmup)):
            run()
        torch.cuda.synchronize()
        start_event=torch.cuda.Event(enable_timing=True)
        end_event=torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(int(repeats)):
            run()
        end_event.record(); torch.cuda.synchronize()
        timings[name]=float(start_event.elapsed_time(end_event))/max(1,int(repeats))
    return timings


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
        global_route_confidence_limit: float = 2.0,
        backend: str | None = None,
        token_selection_mode: str | None = None,
        chunk_selection_scope: str | None = None,
        token_routing_pack_size: int | None = None,
        representative_mode: str = "multi_landmark",
        representative_blend_alpha: float = 0.5,
        representative_score_reduction: str = "logsumexp",
        representative_lse_temperature: float = 0.5,
        coherence_score_max_weight: float = 1.0,
        coherence_score_initial_weight: float = 0.25,
        routing_candidate_multiplier: int = 2,
        routing_stream_block_size: int = 16,
        hierarchical_routing: bool = True,
        hierarchy_group_size: int = 4,
        parent_top_k: int | None = None,
        exact_page_rerank: bool = True,
        dual_source_candidate_union: bool | None = None,
        exploration_probability: float = 0.05,
        exploration_policy: str = "tail_softmax",
        exploration_temperature: float = 1.0,
        exploration_final_probability: float | None = 0.0,
        exploration_anneal_steps: int = 10_000,
        route_aux_weight: float = 0.01,
        route_aux_samples: int = 4,
        route_aux_temperature: float = 1.0,
        route_aux_oracle_temperature: float = 0.2,
        route_aux_teacher: str = "dense_attention",
        route_aux_coverage_weight: float = 0.0,
        global_mass_aux_weight: float = 0.01,
        binding_null_aux_weight: float = 0.002,
        require_auxiliary_return: bool = True,
        global_adapter_rank: int = 16,
        global_key_calibration: str = "none",
        binding_rank: int = 64,
        count_correction_scale_limit: float = 2.0,
        npci_theta_max: float = 0.25,
        max_seq_len: int | None = None,
        local_backend: str = "flex",
        boundary_bridge: bool = True,
        local_block_size: int = 128,
        local_mask_cache_size: int = 4,
        triton_block_q: int | None = 16,
        triton_query_pack_specialization: str | None = (
            TRITON_DOT_MINIMUM_QUERY_BLOCK_SPECIALIZATION
        ),
        backward_impl: str | None = None,
        source_block_size: int = 8,
        collect_routing_diagnostics: bool | None = None,
        diagnostic_max_queries: int | None = None,
        route_from_base_global_key: bool | None = None,
        rerank_selected_priors_with_post_packet_representatives: bool | None = None,
        route_source_policy: str | None = None,
    ) -> None:
        super().__init__()
        D, H, hd = int(D), int(H), int(hd)
        if D < 1 or H < 1 or hd < 1 or D != H * hd:
            raise ValueError("D, H, and hd must be positive and D must equal H*hd")
        if int(top_k_chunks) < 1:
            raise ValueError("top_k_chunks must be positive")
        resolved_chunk_size = 64 if chunk_size is None else int(chunk_size)
        resolved_local_window = 64 if local_window is None else int(local_window)
        resolved_selector_tile = 16 if selector_tile_size is None else int(selector_tile_size)
        if min(resolved_chunk_size, resolved_local_window, resolved_selector_tile) < 1:
            raise ValueError("chunk_size, local_window, and selector_tile_size must be positive")
        resolved_top_m = resolved_chunk_size if hisa_top_m_tokens is None else int(hisa_top_m_tokens)
        if resolved_top_m != resolved_chunk_size:
            raise ValueError("complete-page HISA requires hisa_top_m_tokens == chunk_size")
        if num_chunks is not None and int(num_chunks) < 1:
            raise ValueError("num_chunks must be positive when supplied")
        if num_chunks is not None and max_seq_len is not None:
            expected = math.ceil(int(max_seq_len) / resolved_chunk_size)
            if int(num_chunks) != expected:
                raise ValueError("num_chunks compatibility hint disagrees with max_seq_len/chunk_size")
        if not math.isfinite(temperature) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(route_prior_scale) or float(route_prior_scale) <= 0:
            raise ValueError("route_prior_scale must be finite and positive")
        if not math.isfinite(route_prior_max_scale) or float(route_prior_max_scale) <= float(route_prior_scale):
            raise ValueError("route_prior_max_scale must exceed route_prior_scale")
        for name, value in {
            "global_lane_bias_limit": global_lane_bias_limit,
            "global_route_confidence_limit": global_route_confidence_limit,
            "coherence_score_max_weight": coherence_score_max_weight,
            "count_correction_scale_limit": count_correction_scale_limit,
            "npci_theta_max": npci_theta_max,
        }.items():
            if not math.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= float(coherence_score_initial_weight) <= float(coherence_score_max_weight):
            raise ValueError("coherence_score_initial_weight must be in [0,max_weight]")
        if representative_mode not in {
            "mean_max_blend", "mean_max_blend_ablation", "multi_address", "multi_landmark"
        }:
            raise ValueError("unsupported representative_mode")
        if not 0.0 <= float(representative_blend_alpha) <= 1.0:
            raise ValueError("representative_blend_alpha must be in [0,1]")
        if representative_score_reduction not in {"max", "logsumexp"}:
            raise ValueError("representative_score_reduction must be max or logsumexp")
        if not math.isfinite(representative_lse_temperature) or float(representative_lse_temperature) <= 0:
            raise ValueError("representative_lse_temperature must be finite and positive")
        if int(routing_candidate_multiplier) < 1 or int(routing_stream_block_size) < 1:
            raise ValueError("routing candidate multiplier and stream block size must be positive")
        if int(hierarchy_group_size) < 1:
            raise ValueError("hierarchy_group_size must be positive")
        if parent_top_k is not None and int(parent_top_k) < 1:
            raise ValueError("parent_top_k must be positive when supplied")
        for name, value in {
            "hierarchical_routing": hierarchical_routing,
            "exact_page_rerank": exact_page_rerank,
            "require_auxiliary_return": require_auxiliary_return,
            "boundary_bridge": boundary_bridge,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        for name, value in {
            "dual_source_candidate_union": dual_source_candidate_union,
            "route_from_base_global_key": route_from_base_global_key,
            "rerank_selected_priors_with_post_packet_representatives": (
                rerank_selected_priors_with_post_packet_representatives
            ),
        }.items():
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        if not boundary_bridge:
            raise ValueError("boundary_bridge must remain enabled for complete causal coverage")
        if exploration_final_probability is None:
            exploration_final_probability = exploration_probability
        if not 0.0 <= float(exploration_probability) <= 1.0 or not 0.0 <= float(exploration_final_probability) <= 1.0:
            raise ValueError("exploration probabilities must be in [0,1]")
        if int(exploration_anneal_steps) < 0:
            raise ValueError("exploration_anneal_steps must be non-negative")
        if exploration_policy not in {"tail_softmax", "uniform_unseen_ablation"}:
            raise ValueError("unsupported exploration_policy")
        if not math.isfinite(exploration_temperature) or float(exploration_temperature) <= 0:
            raise ValueError("exploration_temperature must be finite and positive")
        for name, value in {
            "route_aux_weight": route_aux_weight,
            "route_aux_coverage_weight": route_aux_coverage_weight,
            "global_mass_aux_weight": global_mass_aux_weight,
            "binding_null_aux_weight": binding_null_aux_weight,
        }.items():
            if float(value) < 0 or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite and non-negative")
        if int(route_aux_samples) < 0:
            raise ValueError("route_aux_samples must be non-negative")
        if int(route_aux_samples) == 0 and (
            float(route_aux_weight) > 0.0 or float(global_mass_aux_weight) > 0.0
        ):
            raise ValueError(
                "route_aux_samples must be positive when route or global-mass "
                "teacher auxiliary weights are enabled"
            )
        if not math.isfinite(route_aux_temperature) or float(route_aux_temperature) <= 0:
            raise ValueError("route_aux_temperature must be finite and positive")
        if not math.isfinite(route_aux_oracle_temperature) or float(route_aux_oracle_temperature) <= 0:
            raise ValueError("route_aux_oracle_temperature must be finite and positive")
        if route_aux_teacher not in {"dense_attention", "cosine_ablation"}:
            raise ValueError("unsupported route_aux_teacher")
        if global_key_calibration not in {"none", "rms_match_local"}:
            raise ValueError("global_key_calibration must be none or rms_match_local")
        if int(global_adapter_rank) < 0 or int(binding_rank) < 0:
            raise ValueError("adapter and binding ranks must be non-negative")
        if max_seq_len is not None and int(max_seq_len) < 1:
            raise ValueError("max_seq_len must be positive when supplied")
        local_backend = str(local_backend).lower()
        if local_backend != "flex":
            raise ValueError("local_backend must be 'flex'")
        if int(local_block_size) < 16 or not _is_power_of_two(int(local_block_size)):
            raise ValueError("local_block_size must be a power of two >=16")
        if int(local_mask_cache_size) < 1:
            raise ValueError("local_mask_cache_size must be positive")
        if int(source_block_size) < 1 or int(source_block_size) > 32 or not _is_power_of_two(int(source_block_size)):
            raise ValueError("source_block_size must be a power of two in [1,32]")

        self.D, self.H, self.num_heads, self.hd = D, H, H, hd
        self.num_chunks_compatibility_hint = None if num_chunks is None else int(num_chunks)
        self.chunk_size = resolved_chunk_size
        self.top_k_chunks = int(top_k_chunks)
        self.hisa_top_m_tokens = resolved_top_m
        self.local_window = resolved_local_window
        self.selector_tile_size = resolved_selector_tile
        self.temperature = float(temperature)
        self.backend = (backend or "auto").lower()
        if self.backend not in {"auto", "eager", "triton"}:
            raise ValueError("backend must be auto, eager, or triton")

        # Retained only so existing trainer configurations continue to parse.
        self.token_selection_mode = (token_selection_mode or "auto").lower()
        self.chunk_selection_scope = (chunk_selection_scope or "token").lower()
        if self.token_selection_mode != "auto" or self.chunk_selection_scope != "token":
            raise ValueError("legacy selection controls only accept auto/token")
        self.token_routing_pack_size = 4 if token_routing_pack_size is None else int(token_routing_pack_size)
        if self.token_routing_pack_size < 1 or self.token_routing_pack_size > 16 or not _is_power_of_two(self.token_routing_pack_size):
            raise ValueError("token_routing_pack_size must be a power of two in [1,16]")

        self.representative_mode = representative_mode
        self.representative_blend_alpha = float(representative_blend_alpha)
        self.representative_score_reduction = representative_score_reduction
        self.representative_lse_temperature = float(representative_lse_temperature)
        self.coherence_score_max_weight = float(coherence_score_max_weight)
        self.routing_candidate_multiplier = int(routing_candidate_multiplier)
        self.routing_candidate_count = max(
            self.top_k_chunks, self.top_k_chunks * self.routing_candidate_multiplier
        )
        self.routing_stream_block_size = int(routing_stream_block_size)
        self.hierarchical_routing = bool(hierarchical_routing)
        self.hierarchy_group_size = int(hierarchy_group_size)
        default_parent_top_k = max(
            1, math.ceil(self.routing_candidate_count / self.hierarchy_group_size) + 1
        )
        self.parent_top_k = default_parent_top_k if parent_top_k is None else int(parent_top_k)
        self.exact_page_rerank = bool(exact_page_rerank)
        resolved_route_policy = _resolve_route_source_policy(
            route_source_policy,
            route_from_base_global_key=route_from_base_global_key,
            dual_source_candidate_union=dual_source_candidate_union,
            rerank_selected_priors_with_post_packet_representatives=(
                rerank_selected_priors_with_post_packet_representatives
            ),
        )
        self.route_source_policy = resolved_route_policy.name
        self.route_source_contract = resolved_route_policy.contract()
        self.route_from_base_global_key = (
            resolved_route_policy.route_from_base_global_key
        )
        self.dual_source_candidate_union = (
            resolved_route_policy.dual_source_candidate_union
        )
        self.hard_rerank_from_base_global_key = (
            resolved_route_policy.hard_rerank_key == "base_global_k"
        )
        self.route_aux_teacher_from_base_global_key = (
            resolved_route_policy.route_aux_teacher_key == "base_global_k"
        )
        self.rerank_selected_priors_with_post_packet_representatives = (
            resolved_route_policy.rerank_selected_priors_with_post_packet_representatives
        )

        self.exploration_probability = float(exploration_probability)
        self.exploration_policy = exploration_policy
        self.exploration_temperature = float(exploration_temperature)
        self.exploration_final_probability = float(exploration_final_probability)
        self.exploration_anneal_steps = int(exploration_anneal_steps)
        self.route_aux_weight = float(route_aux_weight)
        self.route_aux_samples = int(route_aux_samples)
        self.route_aux_temperature = float(route_aux_temperature)
        self.route_aux_oracle_temperature = float(route_aux_oracle_temperature)
        self.route_aux_teacher = route_aux_teacher
        self.route_aux_coverage_weight = float(route_aux_coverage_weight)
        self.global_mass_aux_weight = float(global_mass_aux_weight)
        self.binding_null_aux_weight = float(binding_null_aux_weight)
        self.require_auxiliary_return = bool(require_auxiliary_return)

        self.route_prior_max_scale = float(route_prior_max_scale)
        self.global_lane_bias_limit = float(global_lane_bias_limit)
        self.global_route_confidence_limit = float(global_route_confidence_limit)
        self.global_adapter_rank = int(global_adapter_rank)
        self.global_key_calibration = global_key_calibration
        self.binding_rank = int(binding_rank)
        self.count_correction_scale_limit = float(count_correction_scale_limit)
        self.register_buffer(
            "binding_state_version", torch.tensor(3, dtype=torch.int32),
            persistent=bool(self.binding_rank),
        )
        self.npci_theta_max = float(npci_theta_max)
        self.route_source = "base_global_k" if self.route_from_base_global_key else "rotated_global_k"
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.local_backend = local_backend
        self.boundary_bridge = boundary_bridge
        self.local_block_size = int(local_block_size)
        self.local_mask_cache_size = int(local_mask_cache_size)
        self._local_block_mask_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._local_count_cache: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
        self._local_block_mask = None
        self._local_block_mask_key = None

        self.triton_block_q = 16 if triton_block_q is None else int(triton_block_q)
        self.triton_query_pack_specialization = triton_query_pack_specialization
        if self.triton_block_q < 1 or not _is_power_of_two(self.triton_block_q):
            raise ValueError("compatibility BLOCK_Q must be a positive power of two")
        if self.triton_query_pack_specialization not in {None, TRITON_DOT_MINIMUM_QUERY_BLOCK_SPECIALIZATION}:
            raise ValueError("unsupported compatibility query-pack specialization")
        self.backward_impl = (backward_impl or "source_block_stable").lower()
        self.backward_impl_code = _backward_impl_code(self.backward_impl)
        self.resolved_backward_impl = {
            _BACKWARD_SOURCE_BLOCK_STABLE: "source_block_stable",
            _BACKWARD_SOURCE_BLOCK_COUNTING: "source_block_counting",
            _BACKWARD_ATOMIC: "atomic",
        }[self.backward_impl_code]
        self.source_block_size = int(source_block_size)

        if collect_routing_diagnostics is None:
            collect_routing_diagnostics = False
        if not isinstance(collect_routing_diagnostics, bool):
            raise TypeError("collect_routing_diagnostics must be bool")
        self.collect_routing_diagnostics = collect_routing_diagnostics
        self.diagnostic_max_queries = 8 if diagnostic_max_queries is None else int(diagnostic_max_queries)
        if self.diagnostic_max_queries < 1:
            raise ValueError("diagnostic_max_queries must be positive")

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
            self.global_k_down = self.global_k_up = None
            self.global_v_down = self.global_v_up = None

        initial_route_fraction = float(route_prior_scale) / float(route_prior_max_scale)
        self.route_prior_raw = nn.Parameter(
            torch.full((H,), math.log(initial_route_fraction / (1.0 - initial_route_fraction)))
        )
        mix_shift = _representative_mix_shift(H, self.representative_blend_alpha)
        self.representative_mix_raw = nn.Parameter(torch.linspace(-2.0, 2.0, H) + mix_shift)
        if self.representative_mode != "mean_max_blend":
            self.representative_mix_raw.requires_grad_(False)
        coherence_fraction = min(
            max(float(coherence_score_initial_weight) / self.coherence_score_max_weight, 1e-4),
            1.0 - 1e-4,
        )
        self.representative_coherence_raw = nn.Parameter(
            torch.full((H,), math.log(coherence_fraction / (1.0 - coherence_fraction)))
        )
        self.global_lane_logit_bias = nn.Parameter(torch.zeros(H))
        self.global_route_confidence_weight = nn.Parameter(torch.zeros(H, 3))
        self.global_route_confidence_bias = nn.Parameter(torch.zeros(H))
        self.count_correction_raw = nn.Parameter(torch.zeros(H))

        if self.binding_rank:
            self.binding_head_rank = max(1, math.ceil(self.binding_rank / H))
            self.binding_feature_dim = H * self.binding_head_rank
            self.bind_query = nn.Linear(D, self.binding_feature_dim, bias=False)
            self.bind_evidence_weight = nn.Parameter(torch.empty(H, hd, self.binding_head_rank))
            self.bind_route_identity = nn.Linear(4, self.binding_head_rank, bias=True)
            self.bind_route_score = nn.Linear(self.binding_head_rank, 1, bias=False)
            self.bind_null = nn.Parameter(torch.empty(H, self.binding_head_rank))
            self.bind_null_prior = nn.Parameter(torch.zeros(H))
            self.bind_abstention = nn.Linear(3, 1, bias=True)
            self.bind_output = nn.Linear(self.binding_feature_dim, D, bias=False)
            self.binding_gain_raw = nn.Parameter(torch.tensor(math.log(0.25 / 0.75)))
            self.reset_binding_parameters_()
        else:
            self.binding_head_rank = self.binding_feature_dim = 0
            self.bind_query = None
            self.register_parameter("bind_evidence_weight", None)
            self.bind_route_identity = self.bind_route_score = None
            self.register_parameter("bind_null", None)
            self.register_parameter("bind_null_prior", None)
            self.bind_abstention = self.bind_output = None
            self.register_parameter("binding_gain_raw", None)

        raw_theta = math.atanh(min(0.01 / max(self.npci_theta_max, 1e-6), 0.99))
        self.npci_theta_k = nn.Parameter(torch.full((H,), raw_theta))
        self.npci_theta_v = nn.Parameter(torch.full((H,), raw_theta))
        self._routing_entropy: torch.Tensor | float = float("nan")
        self._routing_diagnostics: dict[str, torch.Tensor] = {}
        self.hisa_evidence_capture: HISASelectionCapture | None = None
        self._last_token_selection_path = ""

    @property
    def route_prior_scale(self) -> torch.Tensor:
        return self.route_prior_max_scale * torch.sigmoid(self.route_prior_raw)

    @property
    def effective_route_prior_scale(self) -> torch.Tensor:
        """Checkpoint-preserving route scale after compatibility temperature."""
        return self.route_prior_scale / float(self.temperature)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # New scalar/head calibrators have neutral/default migrations and do not
        # invalidate attention checkpoints that predate this optimized kernel.
        for name in ("representative_coherence_raw", "count_correction_raw"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        if self.binding_rank:
            legacy_key = prefix + "bind_evidence.weight"
            version_key = prefix + "binding_state_version"
            if legacy_key in state_dict:
                error_msgs.append(
                    "legacy pooled HISA binder checkpoint detected; binder v3 "
                    "must be initialized rather than shape-migrated"
                )
            elif version_key not in state_dict:
                error_msgs.append("route-slot HISA binder checkpoint is missing binding_state_version=3")
            else:
                version = int(state_dict[version_key].item())
                if version != 3:
                    error_msgs.append(
                        f"unsupported route-slot HISA binder state version {version}; "
                        "expected 3 because mass-grounded scoring changes semantics"
                    )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def load_legacy_binding_v2_backbone_(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ):
        """Load a v2 checkpoint while explicitly discarding its incompatible binder.

        All non-binder tensors are loaded normally. Binder-v3 parameters are reset
        to their mass-grounded initialization and the migrated state is marked v3.
        This is intentionally opt-in; ordinary ``load_state_dict`` rejects v2.
        """
        if not self.binding_rank:
            raise RuntimeError("legacy binder migration requires binding_rank > 0")
        version = state_dict.get("binding_state_version")
        if version is None or int(version.item()) != 2:
            raise ValueError("state_dict is not a route-slot HISA binder-v2 checkpoint")
        self.reset_binding_parameters_()
        current = self.state_dict()
        migrated = OrderedDict(state_dict)
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            migrated._metadata = metadata  # type: ignore[attr-defined]
        binder_prefixes = (
            "bind_query.",
            "bind_route_identity.",
            "bind_route_score.",
            "bind_abstention.",
            "bind_output.",
        )
        binder_exact = {
            "binding_state_version",
            "bind_evidence_weight",
            "bind_null",
            "bind_null_prior",
            "binding_gain_raw",
        }
        for key in list(migrated):
            if key == "bind_evidence.weight":
                del migrated[key]
        for key, value in current.items():
            if key in binder_exact or key.startswith(binder_prefixes):
                migrated[key] = value.detach().clone()
        return self.load_state_dict(migrated, strict=strict)


    @property
    def representative_mix(self) -> torch.Tensor:
        return torch.sigmoid(self.representative_mix_raw)

    @property
    def representative_blend_for_addresses(self) -> float | torch.Tensor:
        if self.representative_mode == "mean_max_blend_ablation":
            return float(self.representative_blend_alpha)
        return self.representative_mix

    @property
    def representative_coherence_weight(self) -> torch.Tensor:
        return self.coherence_score_max_weight * torch.sigmoid(
            self.representative_coherence_raw
        )

    @property
    def count_correction_scale(self) -> torch.Tensor:
        return self.count_correction_scale_limit * torch.sigmoid(
            self.count_correction_raw
        )

    @property
    def bounded_global_lane_logit_bias(self) -> torch.Tensor:
        limit = float(self.global_lane_bias_limit)
        return limit * torch.tanh(self.global_lane_logit_bias / limit)

    def _route_confidence_offset(
        self,
        absolute_route_summary: torch.Tensor,
        semantic_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """Map selected absolute route quality to one bounded logit per query."""
        if absolute_route_summary.shape[-1] != 3:
            raise ValueError("absolute route summary must end in max/mean/entropy")
        if semantic_chunks.shape[:-1] != absolute_route_summary.shape[:-1]:
            raise ValueError("semantic chunks and absolute route summary must align")
        confidence_summary = absolute_route_summary.to(
            dtype=self.global_route_confidence_weight.dtype
        )
        raw = torch.einsum(
            "bhnf,hf->bhn",
            confidence_summary,
            self.global_route_confidence_weight,
        ) + self.global_route_confidence_bias.reshape(1, self.H, 1)
        limit = float(self.global_route_confidence_limit)
        bounded = limit * torch.tanh(raw / limit)
        has_route = (semantic_chunks >= 0).any(-1)
        return torch.where(has_route, bounded, torch.zeros_like(bounded))

    def _quality_candidate_config(self) -> dict[str, object]:
        """Versioned active semantics only; compatibility diagnostics are excluded."""
        return {
            "format": "hisa-v19-optimized-quality-v3",
            "model_dim": self.D,
            "heads": self.H,
            "head_dim": self.hd,
            "chunk_size": self.chunk_size,
            "top_k_chunks": self.top_k_chunks,
            "local_window": self.local_window,
            "boundary_bridge": self.boundary_bridge,
            "route_source_policy": self.route_source_policy,
            "route_source_contract": self.route_source_contract,
            "routing_key_source": self.route_source,
            "routing_temperature": self.temperature,
            "routing_candidate_count": self.routing_candidate_count,
            "streaming_selector_block": self.routing_stream_block_size,
            "hierarchical_routing": self.hierarchical_routing,
            "hierarchy_group_size": self.hierarchy_group_size,
            "parent_top_k": self.parent_top_k,
            "exact_page_rerank": self.exact_page_rerank,
            "dual_source_candidate_union": self.dual_source_candidate_union,
            "representative_mode": self.representative_mode,
            "representative_score_reduction": self.representative_score_reduction,
            "representative_lse_temperature": self.representative_lse_temperature,
            "representative_blend_alpha_initial_mean": self.representative_blend_alpha,
            "coherence_score_max_weight": self.coherence_score_max_weight,
            "route_prior_max_scale": self.route_prior_max_scale,
            "global_lane_bias_limit": self.global_lane_bias_limit,
            "global_route_confidence_limit": self.global_route_confidence_limit,
            "count_correction_scale_limit": self.count_correction_scale_limit,
            "global_adapter_rank": self.global_adapter_rank,
            "global_key_calibration": self.global_key_calibration,
            "binding_rank": self.binding_rank,
            "binding_state_version": 3,
            "npci_theta_max": self.npci_theta_max,
            "rerank_selected_priors_with_post_packet_representatives": self.rerank_selected_priors_with_post_packet_representatives,
            "route_aux_teacher": self.route_aux_teacher,
            "route_aux_weight": self.route_aux_weight,
            "route_aux_samples": self.route_aux_samples,
            "route_aux_temperature": self.route_aux_temperature,
            "route_aux_oracle_temperature": self.route_aux_oracle_temperature,
            "route_aux_coverage_weight": self.route_aux_coverage_weight,
            "global_mass_aux_weight": self.global_mass_aux_weight,
            "binding_null_aux_weight": self.binding_null_aux_weight,
            "exploration_probability": self.exploration_probability,
            "exploration_final_probability": self.exploration_final_probability,
            "exploration_anneal_steps": self.exploration_anneal_steps,
            "exploration_policy": self.exploration_policy,
            "exploration_temperature": self.exploration_temperature,
        }

    def _quality_candidate_fingerprint(self) -> str:
        payload = json.dumps(
            self._quality_candidate_config(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def semantic_config(self) -> dict[str, object]:
        config = dict(self._quality_candidate_config())
        config.update({
            "implementation": "hisa-v19-streaming-hierarchical-route-slots-null-binding-v3",
            "selected_token_policy": "enumerate_complete_selected_chunks",
            "semantic_selection_scope": "per_token",
            "hard_selector": "detached_streaming_parent_child_top_m_exact_page_lse_top_k",
            "selector_autograd": "selected_K_plus_sampled_aux_rows_only",
            "representative_coherence": "pre_normalized_mean_norm_monotonic_penalty",
            "global_attention_key_source": "post_packet_rotated_global_k",
            "route_source_contract": self.route_source_contract,
            "route_auxiliary_target": "conditional_global_chunk_mass_plus_total_global_mass",
            "route_auxiliary_transport": "explicit_forward_return_required",
            "binding_source": "per_head_per_route_normalized_low_rank_evidence",
            "binding_aggregation": "mass_grounded_route_null_softmax_with_separate_payload",
            "binding_mass": "exact_merged_attention_log_mass",
            "binding_route_identity_features": "absolute_similarity,relative_centered_prior,relative_age,relative_position",
            "binding_abstention_features": "absolute_max,absolute_mean,route_entropy",
            "lane_merge": "exact_lse",
            "selected_attention_arithmetic": "direct_complete_page_scan",
            "incremental_cache": "exact_completed_page_reference_api_O(N)_KV",
            "quality_candidate_fingerprint_sha256": self._quality_candidate_fingerprint(),
            "max_seq_len": self.max_seq_len,
        })
        return config

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "local_backend": self.local_backend,
            "local_block_size": self.local_block_size,
            "local_flex_backend": "TRITON_explicit_tiles",
            "local_flex_compile_mode": _FLEX_COMPILE_MODE,
            "local_mask_cache_size": self.local_mask_cache_size,
            "backward_impl": self.resolved_backward_impl,
            "source_block_size": self.source_block_size,
            "direct_kernel_schedule": "autotuned_candidate" if _DIRECT_GLOBAL_AUTOTUNE_ENABLED else "static_default",
            "aggregate_only_triton_specialization": True,
            "factorized_page_route_softmax": True,
            "reverse_edge_options": ("stable_sort", "bounded_id_counting", "atomic_query_owned"),
            "collect_routing_diagnostics": self.collect_routing_diagnostics,
            "diagnostic_max_queries": self.diagnostic_max_queries,
            "selector_workspace": "O(B*H*N*M)+stream_block; no differentiable B*H*N*C surface",
            "slot_evidence_temporary_bound": "O(B*H*N*K*binding_head_rank)",
            "flex_attention_available": _FLEX_ATTENTION_AVAILABLE,
            "triton_available": _TRITON_AVAILABLE,
            "triton_library_operator": _TRITON_LIBRARY_API_AVAILABLE,
            "compatibility_only": {
                "selector_tile_size": self.selector_tile_size,
                "token_routing_pack_size": self.token_routing_pack_size,
                "triton_block_q": self.triton_block_q,
                "triton_query_pack_specialization": self.triton_query_pack_specialization,
                "num_chunks_hint": self.num_chunks_compatibility_hint,
                "hisa_top_m_tokens": self.hisa_top_m_tokens,
            },
        }


    def _ensure_local_block_mask(self, device: torch.device, seq_len: int):
        if not _FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
        key = (
            device.type, device.index, int(seq_len), self.local_window,
            self.chunk_size, self.boundary_bridge, self.local_block_size,
        )
        cached = self._local_block_mask_cache.get(key)
        if cached is not None:
            self._local_block_mask_cache.move_to_end(key)
            self._local_block_mask = cached
            self._local_block_mask_key = key
            return cached
        window, chunk = int(self.local_window), int(self.chunk_size)

        def local_mask(_batch, _head, query_index, key_index):
            query_index = query_index + 1
            return _strict_local_or_boundary_key_mask(
                query_index, key_index, window, chunk
            )

        mask = create_block_mask(
            local_mask, B=None, H=None,
            Q_LEN=max(1, int(seq_len) - 1), KV_LEN=int(seq_len),
            device=device, BLOCK_SIZE=self.local_block_size,
        )
        self._local_block_mask_cache[key] = mask
        self._local_block_mask_cache.move_to_end(key)
        while len(self._local_block_mask_cache) > self.local_mask_cache_size:
            self._local_block_mask_cache.popitem(last=False)
        self._local_block_mask = mask
        self._local_block_mask_key = key
        return mask

    def _local_count_geometry(self, device: torch.device, seq_len: int) -> torch.Tensor:
        key = (device.type, device.index, int(seq_len), self.local_window, self.chunk_size)
        if not torch.compiler.is_compiling():
            cached = self._local_count_cache.get(key)
            if cached is not None:
                self._local_count_cache.move_to_end(key)
                return cached
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        local_count = positions.clamp(max=float(self.local_window))
        cutoff = (positions - float(self.local_window)).clamp_min(0.0)
        local_count = local_count + torch.remainder(cutoff, float(self.chunk_size))
        if not torch.compiler.is_compiling():
            self._local_count_cache[key] = local_count
            self._local_count_cache.move_to_end(key)
            while len(self._local_count_cache) > self.local_mask_cache_size:
                self._local_count_cache.popitem(last=False)
        return local_count

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
                self.local_block_size,
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
            selector_tile_size=self.selector_tile_size,
        )

    def initialize_global_adapter_up_(self) -> None:
        if self.global_adapter_rank:
            if self.global_k_up is None or self.global_v_up is None:
                raise RuntimeError("global adapter projections are incomplete")
            # A small nonzero residual lets both adapter factors learn immediately.
            nn.init.normal_(self.global_k_up.weight, mean=0.0, std=0.002)
            nn.init.normal_(self.global_v_up.weight, mean=0.0, std=0.002)

    def reset_global_adapters_(self) -> None:
        if self.global_adapter_rank:
            if self.global_k_down is None or self.global_v_down is None:
                raise RuntimeError("global adapter projections are incomplete")
            self.global_k_down.reset_parameters()
            self.global_v_down.reset_parameters()
            self.initialize_global_adapter_up_()

    @torch.no_grad()
    def reset_binding_parameters_(self) -> None:
        """Initialize binder v3 as a useful mass-weighted low-rank value path."""
        if not self.binding_rank:
            return
        if (
            self.bind_query is None or self.bind_route_identity is None
            or self.bind_route_score is None or self.bind_abstention is None
            or self.bind_output is None or self.bind_evidence_weight is None
            or self.bind_null is None or self.bind_null_prior is None
            or self.binding_gain_raw is None
        ):
            raise RuntimeError("binding reset modules are incomplete")
        nn.init.zeros_(self.bind_query.weight)
        nn.init.zeros_(self.bind_route_identity.weight)
        nn.init.zeros_(self.bind_route_identity.bias)
        nn.init.zeros_(self.bind_route_score.weight)
        nn.init.zeros_(self.bind_null)
        nn.init.zeros_(self.bind_null_prior)
        nn.init.zeros_(self.bind_abstention.weight)
        nn.init.zeros_(self.bind_abstention.bias)
        for head in range(self.H):
            nn.init.orthogonal_(self.bind_evidence_weight[head])
        nn.init.zeros_(self.bind_output.weight)
        for head in range(self.H):
            row_start = head * self.hd
            row_end = row_start + self.hd
            column_start = head * self.binding_head_rank
            column_end = column_start + self.binding_head_rank
            self.bind_output.weight[row_start:row_end, column_start:column_end].copy_(
                self.bind_evidence_weight[head]
            )
        self.binding_gain_raw.fill_(math.log(0.25 / 0.75))

    def _binding_correction(
        self,
        x: torch.Tensor,
        route_evidence_features: torch.Tensor,
        route_log_mass: torch.Tensor,
        route_identity_features: torch.Tensor,
        absolute_route_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mass-grounded slot binding with separate scoring and payload paths."""
        if (
            self.bind_query is None or self.bind_route_identity is None
            or self.bind_route_score is None or self.bind_null is None
            or self.bind_null_prior is None or self.bind_abstention is None
            or self.bind_output is None or self.binding_gain_raw is None
        ):
            raise RuntimeError("binding correction modules are incomplete")
        batch_size, seq_len, _ = x.shape
        expected_prefix = (batch_size, self.H, seq_len)
        if route_evidence_features.shape[:3] != expected_prefix:
            raise ValueError("route evidence must have shape [B,H,N,K,R]")
        if route_evidence_features.shape[-1] != self.binding_head_rank:
            raise ValueError("route evidence has the wrong head-local binding rank")
        if route_log_mass.shape != route_evidence_features.shape[:-1]:
            raise ValueError("route_log_mass must have shape [B,H,N,K]")
        if route_identity_features.shape != (*route_log_mass.shape, 4):
            raise ValueError("route identity features must have shape [B,H,N,K,4]")
        if absolute_route_summary.shape != (*expected_prefix, 3):
            raise ValueError("absolute route summary must have shape [B,H,N,3]")

        query_features = torch.tanh(
            self.bind_query(x).reshape(
                batch_size, seq_len, self.H, self.binding_head_rank
            ).permute(0, 2, 1, 3)
        )
        evidence = route_evidence_features.to(query_features.dtype)
        identity = self.bind_route_identity(
            route_identity_features.to(query_features.dtype)
        )
        score_content = F.layer_norm(
            evidence + identity, (self.binding_head_rank,)
        )
        score_interaction = query_features[..., None, :] * score_content + identity
        learned_route_adjustment = self.bind_route_score(
            score_interaction
        ).squeeze(-1).float()
        valid_route = torch.isfinite(route_log_mass)
        route_logits = route_log_mass.float() + learned_route_adjustment
        route_logits = route_logits.masked_fill(~valid_route, float("-inf"))

        finite_log_mass = torch.where(
            valid_route, route_log_mass.float(), torch.full_like(route_log_mass.float(), float("-inf"))
        )
        route_mass_sum = torch.exp(finite_log_mass).sum(-1).clamp(0.0, 1.0)
        base_null_log_mass = torch.log((1.0 - route_mass_sum).clamp_min(1e-30))
        null_interaction = query_features * self.bind_null.reshape(
            1, self.H, 1, self.binding_head_rank
        ).to(query_features.dtype)
        learned_null_adjustment = (
            self.bind_route_score(null_interaction).squeeze(-1).float()
            + self.bind_null_prior.reshape(1, self.H, 1).float()
            + self.bind_abstention(
                absolute_route_summary.to(query_features.dtype)
            ).squeeze(-1).float()
        )
        null_logits = base_null_log_mass + learned_null_adjustment
        all_logits = torch.cat((route_logits, null_logits[..., None]), dim=-1)
        all_weights = torch.softmax(all_logits, dim=-1)
        route_weights = all_weights[..., :-1]
        null_probability = all_weights[..., -1]
        null_log_odds = null_logits - torch.logsumexp(route_logits, dim=-1)

        payload = evidence * (1.0 + 0.1 * query_features[..., None, :]) + identity
        per_head = (
            route_weights[..., None].to(payload.dtype) * payload
        ).sum(dim=3)
        combined = per_head.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.binding_feature_dim
        )
        bound = self.bind_output(combined)
        correction = torch.sigmoid(self.binding_gain_raw).to(bound.dtype) * bound
        return correction, null_probability, null_log_odds

    def forward(
        self,
        x: torch.Tensor,
        kv_inject: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        forced_route_chunk_ids: torch.Tensor | None = None,
        exploration_step: int | torch.Tensor | None = None,
        exploration_probability_override: float | torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_metadata: bool = False,
        return_auxiliary: bool = False,
    ):
        compiling = torch.compiler.is_compiling()
        emit_diagnostics = bool(
            (
                collect_diagnostics
                or self.collect_routing_diagnostics
                or forced_route_chunk_ids is not None
            )
            and not compiling
        )
        if forced_route_chunk_ids is not None and compiling:
            raise RuntimeError(
                "forced_route_chunk_ids is a diagnostics-only input and is not "
                "supported inside torch.compile"
            )
        if not compiling:
            self.hisa_evidence_capture = None
            self._routing_diagnostics = {}

        if x.ndim != 3 or x.shape[-1] != self.D:
            raise ValueError(f"x must have shape [B,N,{self.D}]")
        batch_size, seq_len, _ = x.shape
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds configured HISA bound {self.max_seq_len}"
            )
        lengths = _as_valid_lengths(
            valid_lengths if valid_lengths is not None
            else getattr(self, "_causal_control_valid_lengths", None),
            batch_size=batch_size, seq_len=seq_len, device=x.device,
        )
        use_triton = _resolve_attention_execution(
            backend=self.backend, is_cuda=x.is_cuda,
            triton_available=_TRITON_AVAILABLE,
            flex_available=_FLEX_ATTENTION_AVAILABLE,
        )
        if use_triton:
            _validate_triton_geometry(head_dim=self.hd, tokens_per_chunk=self.chunk_size)

        teacher_aux_active = bool(
            self.training and self.route_aux_samples > 0
            and (self.route_aux_weight > 0.0 or self.global_mass_aux_weight > 0.0)
        )
        binding_aux_active = bool(
            self.training and self.binding_rank and self.binding_null_aux_weight > 0.0
        )
        any_aux_active = teacher_aux_active or binding_aux_active
        if any_aux_active and self.require_auxiliary_return and not return_auxiliary:
            raise RuntimeError(
                "HISA auxiliary losses are enabled during training; call with "
                "return_auxiliary=True and add the returned scalar to the model loss, "
                "or explicitly disable the auxiliary weights."
            )

        exploration_probability: float | torch.Tensor = 0.0
        if self.training:
            if exploration_probability_override is not None:
                exploration_probability = exploration_probability_override
            elif (
                self.exploration_probability != self.exploration_final_probability
                and self.exploration_anneal_steps > 0
                and exploration_step is None
            ):
                raise RuntimeError(
                    "exploration annealing is configured but exploration_step was not "
                    "supplied; pass a device scalar/int step or a resolved "
                    "exploration_probability_override"
                )
            elif exploration_step is None:
                exploration_probability = self.exploration_probability
            else:
                exploration_probability = _annealed_exploration_probability(
                    self.exploration_probability,
                    self.exploration_final_probability,
                    exploration_step,
                    self.exploration_anneal_steps,
                )
            if torch.is_tensor(exploration_probability):
                if exploration_probability.numel() != 1:
                    raise ValueError("exploration probability override must be scalar")
            elif not 0.0 <= float(exploration_probability) <= 1.0:
                raise ValueError("exploration probability override must be in [0,1]")

        query_flat, key_flat, value_flat, gate = self.qkvg_proj(x).split(self.D, dim=-1)
        query = _to_heads(query_flat, batch_size, seq_len, self.H, self.hd)
        local_key = _to_heads(key_flat, batch_size, seq_len, self.H, self.hd)
        local_value = _to_heads(value_flat, batch_size, seq_len, self.H, self.hd)
        global_key, global_value = local_key, local_value
        if self.global_adapter_rank:
            if any(module is None for module in (
                self.global_k_down, self.global_k_up,
                self.global_v_down, self.global_v_up,
            )):
                raise RuntimeError("global adapter modules are incomplete")
            assert self.global_k_down is not None and self.global_k_up is not None
            assert self.global_v_down is not None and self.global_v_up is not None
            global_key = global_key + _to_heads(
                self.global_k_up(self.global_k_down(x)),
                batch_size, seq_len, self.H, self.hd,
            )
            global_value = global_value + _to_heads(
                self.global_v_up(self.global_v_down(x)),
                batch_size, seq_len, self.H, self.hd,
            )
        base_global_key = global_key
        rotation_diagnostics: dict[str, torch.Tensor] = {}
        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            if key_delta.shape != global_key.shape or value_delta.shape != global_value.shape:
                raise ValueError("kv_inject must contain [B,H,N,HD] tensors")
            theta_k = self.npci_theta_max * torch.tanh(self.npci_theta_k)
            theta_v = self.npci_theta_max * torch.tanh(self.npci_theta_v)
            if emit_diagnostics:
                rotation_diagnostics.update(_rotation_diagnostics(
                    global_key, key_delta, theta_k,
                    valid_lengths=lengths, label="k",
                ))
                rotation_diagnostics.update(_rotation_diagnostics(
                    global_value, value_delta, theta_v,
                    valid_lengths=lengths, label="v",
                ))
            global_key = _magnitude_aware_rotate(global_key, key_delta, theta_k)
            global_value = _magnitude_aware_rotate(global_value, value_delta, theta_v)
        if self.global_key_calibration == "rms_match_local":
            global_key = _rms_match_global_key(global_key, local_key, lengths)

        query_normalized = _normalized_routing_query(query)
        routing_key = base_global_key if self.route_from_base_global_key else global_key
        hierarchy_size = self.hierarchy_group_size if self.hierarchical_routing else 1
        routing_addresses = _completed_chunk_addresses(
            routing_key,
            chunk_size=self.chunk_size,
            valid_lengths=lengths,
            blend_alpha=self.representative_blend_for_addresses,
            mode=self.representative_mode,
            hierarchy_group_size=hierarchy_size,
            collect_diagnostics=emit_diagnostics,
            token_ids=token_ids if emit_diagnostics else None,
        )
        deterministic_candidates = _hierarchical_candidate_metadata(
            query_normalized,
            routing_addresses,
            chunk_size=self.chunk_size,
            valid_lengths=lengths,
            local_window=self.local_window,
            candidate_k=self.routing_candidate_count,
            parent_top_k=self.parent_top_k,
            streaming_block_size=self.routing_stream_block_size,
            reduction=self.representative_score_reduction,
            temperature=self.representative_lse_temperature,
            coherence_weight=self.representative_coherence_weight.detach(),
            hierarchical=self.hierarchical_routing,
        )

        attention_addresses: HISAChunkAddresses | None = None
        if self.route_from_base_global_key and (
            self.dual_source_candidate_union
            or self.rerank_selected_priors_with_post_packet_representatives
        ):
            attention_addresses = _completed_chunk_addresses(
                global_key,
                chunk_size=self.chunk_size,
                valid_lengths=lengths,
                blend_alpha=self.representative_blend_for_addresses,
                mode=self.representative_mode,
                hierarchy_group_size=hierarchy_size,
                collect_diagnostics=False,
            )
        if (
            self.route_from_base_global_key
            and self.dual_source_candidate_union
            and self.exact_page_rerank
            and attention_addresses is not None
        ):
            attention_candidates = _hierarchical_candidate_metadata(
                query_normalized,
                attention_addresses,
                chunk_size=self.chunk_size,
                valid_lengths=lengths,
                local_window=self.local_window,
                candidate_k=self.routing_candidate_count,
                parent_top_k=self.parent_top_k,
                streaming_block_size=self.routing_stream_block_size,
                reduction=self.representative_score_reduction,
                temperature=self.representative_lse_temperature,
                coherence_weight=self.representative_coherence_weight.detach(),
                hierarchical=self.hierarchical_routing,
            )
            concatenated = torch.cat(
                (deterministic_candidates.top_chunk_idx,
                 attention_candidates.top_chunk_idx), dim=-1
            ).to(torch.int64)
            deduplicated, unique = _deduplicate_fixed_candidates(
                concatenated, entry_count=routing_addresses.values.shape[2]
            )
            union_ids = torch.where(
                unique, deduplicated, torch.full_like(deduplicated, -1)
            ).to(torch.int32)
            deterministic_candidates = HISAMetadata(
                top_chunk_idx=union_ids,
                tile_starts=deterministic_candidates.tile_starts,
                valid_lengths=lengths,
                chunk_size=self.chunk_size,
                selector_tile_size=1,
            )

        hard_rerank_key = (
            base_global_key
            if self.hard_rerank_from_base_global_key
            else global_key
        )
        deterministic_metadata, exact_candidate_lse = _rerank_candidate_metadata(
            query,
            hard_rerank_key,
            deterministic_candidates,
            top_k=self.top_k_chunks,
            local_window=self.local_window,
            exact_rerank=self.exact_page_rerank,
        )
        metadata = _inject_streaming_exploration(
            deterministic_metadata,
            query_normalized,
            routing_addresses,
            exploration_probability,
            local_window=self.local_window,
            policy=self.exploration_policy,
            temperature=self.exploration_temperature,
            block_size=self.routing_stream_block_size,
            reduction=self.representative_score_reduction,
            representative_temperature=self.representative_lse_temperature,
            coherence_weight=self.representative_coherence_weight.detach(),
        )
        if not compiling:
            self._last_token_selection_path = (
                "hierarchical_parent_child_top_m_exact_page_lse_top_k"
                if self.hierarchical_routing else
                "streaming_flat_top_m_exact_page_lse_top_k"
            )

        prior_addresses = routing_addresses
        if self.rerank_selected_priors_with_post_packet_representatives:
            if attention_addresses is None:
                attention_addresses = _completed_chunk_addresses(
                    global_key,
                    chunk_size=self.chunk_size,
                    valid_lengths=lengths,
                    blend_alpha=self.representative_blend_for_addresses,
                    mode=self.representative_mode,
                    hierarchy_group_size=1,
                    collect_diagnostics=False,
                )
            prior_addresses = attention_addresses
        selected_similarity = _selected_routing_scores(
            query_normalized,
            prior_addresses,
            metadata.top_chunk_idx,
            reduction=self.representative_score_reduction,
            temperature=self.representative_lse_temperature,
            coherence_weight=self.representative_coherence_weight,
        )
        valid_route = torch.isfinite(selected_similarity)
        valid_count = valid_route.sum(-1, keepdim=True).clamp_min(1)
        selected_mean = torch.where(
            valid_route, selected_similarity, torch.zeros_like(selected_similarity)
        ).sum(-1, keepdim=True) / valid_count
        centered_similarity = torch.where(
            valid_route, selected_similarity - selected_mean,
            torch.zeros_like(selected_similarity),
        )
        route = (
            centered_similarity
            * self.effective_route_prior_scale.reshape(1, self.H, 1, 1)
        ).masked_fill(~valid_route, float("-inf"))
        semantic_chunks = metadata.top_chunk_idx.to(torch.int64)
        semantic_similarity = torch.where(
            valid_route, selected_similarity, torch.zeros_like(selected_similarity)
        )
        route_identity_features, absolute_route_summary = _route_identity_features(
            semantic_similarity,
            semantic_chunks,
            route_scale_by_head=self.route_prior_scale,
            temperature=self.temperature,
            chunk_size=self.chunk_size,
        )
        route_confidence_offset = self._route_confidence_offset(
            absolute_route_summary, semantic_chunks
        )
        route = route + route_confidence_offset[..., None].to(route.dtype)

        local_count = self._local_count_geometry(x.device, seq_len)
        global_count = torch.isfinite(route).sum(-1).float() * float(self.chunk_size)
        raw_count_correction = torch.where(
            (global_count > 0.0) & (local_count.reshape(1, 1, -1) > 0.0),
            torch.log(
                local_count.reshape(1, 1, -1).clamp_min(1.0)
                / global_count.clamp_min(1.0)
            ),
            torch.zeros_like(global_count),
        )
        count_correction = (
            raw_count_correction
            * self.count_correction_scale.reshape(1, self.H, 1)
        )
        route_common_correction = (
            count_correction
            + self.bounded_global_lane_logit_bias.reshape(1, self.H, 1).float()
        )
        route = route + route_common_correction[..., None].to(route.dtype)

        auxiliary = torch.zeros((), device=x.device, dtype=torch.float32)
        router_auxiliary: HISARouterAuxiliary | None = None
        if teacher_aux_active:
            route_aux_teacher_key = (
                base_global_key
                if self.route_aux_teacher_from_base_global_key
                else global_key
            )
            router_auxiliary = _router_auxiliary_loss(
                query_normalized,
                routing_addresses,
                query,
                route_aux_teacher_key,
                deterministic_metadata,
                self.route_prior_scale,
                samples=self.route_aux_samples,
                route_slots=self.top_k_chunks,
                target_temperature=self.route_aux_temperature,
                routing_temperature=self.temperature,
                local_window=self.local_window,
                oracle_temperature=self.route_aux_oracle_temperature,
                teacher_mode=self.route_aux_teacher,
                coverage_weight=self.route_aux_coverage_weight,
                representative_reduction=self.representative_score_reduction,
                representative_temperature=self.representative_lse_temperature,
                coherence_weight=self.representative_coherence_weight,
                tile_ids=route_aux_tile_ids,
            )
            auxiliary = auxiliary + self.route_aux_weight * router_auxiliary.loss

        retain_selector = emit_diagnostics or (return_metadata and not compiling)
        analysis_selection_logits = None
        if retain_selector:
            with torch.no_grad():
                analysis_selection_logits = _dense_routing_score_surface(
                    query_normalized,
                    routing_addresses,
                    reduction=self.representative_score_reduction,
                    temperature=self.representative_lse_temperature,
                    coherence_weight=self.representative_coherence_weight.detach(),
                ).detach()

        local_output, local_lse = self._local_lane(
            query, local_key, local_value, lengths
        )
        route_evidence: HISARouteEvidence | None = None
        evidence_value: torch.Tensor | None = None
        has_possible_global = seq_len > self.local_window + self.chunk_size
        if self.binding_rank and has_possible_global:
            if self.bind_evidence_weight is None:
                raise RuntimeError("binding evidence projection is absent")
            evidence_value = torch.einsum(
                "bhnd,hdr->bhnr", global_value, self.bind_evidence_weight
            )
        if not has_possible_global:
            global_output = torch.zeros_like(query)
            global_lse = torch.full(
                (batch_size, self.H, seq_len), float("-inf"),
                device=x.device, dtype=torch.float32,
            )
        elif use_triton:
            if evidence_value is None:
                global_output, global_lse = _direct_global_hisa_triton_apply(
                    query, global_key, global_value, route,
                    metadata.top_chunk_idx, lengths,
                    self.chunk_size, self.local_window,
                    backward_impl=self.resolved_backward_impl,
                    source_block_size=self.source_block_size,
                )
            else:
                global_output, global_lse, route_output, route_lse = (
                    _direct_global_hisa_triton_apply(
                        query, global_key, global_value, route,
                        metadata.top_chunk_idx, lengths,
                        self.chunk_size, self.local_window,
                        projected_value=evidence_value,
                        backward_impl=self.resolved_backward_impl,
                        source_block_size=self.source_block_size,
                    )
                )
                route_evidence = HISARouteEvidence(output=route_output, lse=route_lse)
        else:
            if evidence_value is None:
                global_output, global_lse = _eager_global_lane(
                    query, global_key, global_value, route, metadata,
                    local_window=self.local_window,
                )
            else:
                global_output, global_lse, route_evidence = _eager_global_lane_with_route_evidence(
                    query, global_key, global_value, route, metadata,
                    local_window=self.local_window, evidence_value=evidence_value,
                )

        attended, combined_lse, head_global_mass = _merge_attention_lanes_with_mass(
            local_output, local_lse, global_output, global_lse
        )
        if router_auxiliary is not None and self.global_mass_aux_weight > 0.0:
            sampled_ids = router_auxiliary.sampled_tile_ids
            if sampled_ids.numel() > 0:
                predicted_global = head_global_mass[:, :, sampled_ids].float().clamp(1e-6, 1.0 - 1e-6)
                predicted_global_logits = torch.logit(predicted_global)
                target_global = router_auxiliary.teacher_global_mass.detach().float().clamp(0.0, 1.0)
                global_mass_loss = F.binary_cross_entropy_with_logits(
                    predicted_global_logits, target_global, reduction="mean"
                )
                auxiliary = auxiliary + self.global_mass_aux_weight * global_mass_loss
            else:
                global_mass_loss = auxiliary.new_zeros(())
        else:
            global_mass_loss = auxiliary.new_zeros(())

        merged = attended.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.D)
        projected = self.W_o(merged)
        binding_correction = torch.zeros_like(projected)
        null_probability: torch.Tensor | None = None
        null_log_odds: torch.Tensor | None = None
        if self.binding_rank:
            if route_evidence is None:
                # Prefixes with no complete global page have an exact null binder.
                null_probability = torch.ones(
                    batch_size, self.H, seq_len,
                    device=x.device, dtype=torch.float32,
                )
                null_log_odds = torch.full_like(null_probability, 20.0)
            else:
                route_log_mass = torch.where(
                    torch.isfinite(route_evidence.lse)
                    & torch.isfinite(combined_lse.float()[..., None]),
                    route_evidence.lse.float() - combined_lse.float()[..., None],
                    torch.full_like(route_evidence.lse.float(), float("-inf")),
                )
                binding_correction, null_probability, null_log_odds = self._binding_correction(
                    x,
                    route_evidence.output,
                    route_log_mass,
                    route_identity_features,
                    absolute_route_summary,
                )
            if (
                self.training and self.binding_null_aux_weight > 0.0
                and null_log_odds is not None
            ):
                positions = torch.arange(seq_len, device=x.device).reshape(1, 1, -1)
                valid_rows = (positions > 0) & (positions < lengths.reshape(batch_size, 1, 1))
                target_null = (1.0 - head_global_mass.detach().float()).clamp(0.0, 1.0)
                safe_null_log_odds = torch.nan_to_num(
                    null_log_odds.float(), nan=0.0, posinf=20.0, neginf=-20.0
                ).clamp(-20.0, 20.0)
                per_row_null = F.binary_cross_entropy_with_logits(
                    safe_null_log_odds,
                    target_null,
                    reduction="none",
                )
                null_loss = torch.where(
                    valid_rows, per_row_null, torch.zeros_like(per_row_null)
                ).sum() / valid_rows.sum().clamp_min(1)
                auxiliary = auxiliary + self.binding_null_aux_weight * null_loss
            else:
                null_loss = auxiliary.new_zeros(())
        else:
            null_loss = auxiliary.new_zeros(())

        # Binding is part of the same content branch and no longer bypasses the
        # residual-content gate.
        output = (projected + binding_correction) * torch.sigmoid(gate)

        if not compiling and return_metadata:
            if analysis_selection_logits is None:
                raise RuntimeError("HISA selector capture was not retained")
            self.hisa_evidence_capture = HISASelectionCapture(
                anchor_logits=analysis_selection_logits,
                metadata=metadata,
                auxiliary_loss=auxiliary,
                sampled_anchor_logits=(
                    None if router_auxiliary is None
                    else router_auxiliary.sampled_anchor_logits
                ),
                sampled_tile_ids=(
                    None if router_auxiliary is None
                    else router_auxiliary.sampled_tile_ids
                ),
                sampled_teacher_mass=(
                    None if router_auxiliary is None
                    else router_auxiliary.teacher_mass
                ),
            )

        if emit_diagnostics:
            with torch.no_grad():
                positions = torch.arange(seq_len, device=x.device).reshape(1, 1, -1)
                valid_rows = (
                    (positions > 0)
                    & (positions < lengths.reshape(batch_size, 1, 1))
                    & torch.isfinite(combined_lse)
                )
                valid_count_diag = valid_rows.sum().clamp_min(1)
                global_mass_diag = torch.where(
                    valid_rows, head_global_mass.float(),
                    torch.zeros_like(head_global_mass.float()),
                )
                finite_route = torch.isfinite(route)
                route_count = finite_route.sum().clamp_min(1)
                diagnostics = {
                    **rotation_diagnostics,
                    **_representative_diagnostics(routing_addresses),
                    **_combined_lse_diagnostics(combined_lse, lengths),
                    **_direct_kernel_geometry_diagnostics(route, metadata),
                    "global_attention_mass": global_mass_diag.sum() / valid_count_diag,
                    "local_attention_mass": torch.where(
                        valid_rows, 1.0 - global_mass_diag, 0.0
                    ).sum() / valid_count_diag,
                    "selected_route_rms": torch.sqrt(
                        torch.where(finite_route, route.float().square(), 0.0).sum()
                        / route_count
                    ),
                    "route_prior_scale_mean": self.route_prior_scale.mean(),
                    "representative_coherence_weight_mean": self.representative_coherence_weight.mean(),
                    "count_correction_scale_mean": self.count_correction_scale.mean(),
                    "global_lane_logit_bias_mean": self.bounded_global_lane_logit_bias.mean(),
                    "route_confidence_offset_mean": torch.where(
                        (semantic_chunks >= 0).any(-1), route_confidence_offset,
                        torch.zeros_like(route_confidence_offset),
                    ).sum() / (semantic_chunks >= 0).any(-1).sum().clamp_min(1),
                    "exploration_probability_effective": torch.as_tensor(
                        exploration_probability, device=x.device, dtype=torch.float32
                    ),
                    "routing_candidate_count": torch.tensor(
                        float(deterministic_candidates.top_chunk_idx.shape[-1]), device=x.device
                    ),
                    "hierarchical_routing_enabled": torch.tensor(
                        float(self.hierarchical_routing), device=x.device
                    ),
                    "exact_page_rerank_enabled": torch.tensor(
                        float(self.exact_page_rerank), device=x.device
                    ),
                    "router_auxiliary_loss": (
                        auxiliary.new_zeros(()) if router_auxiliary is None
                        else router_auxiliary.loss
                    ),
                    "global_mass_auxiliary_loss": global_mass_loss,
                    "binding_null_auxiliary_loss": null_loss,
                }
                if exact_candidate_lse is not None:
                    finite_exact = torch.isfinite(exact_candidate_lse)
                    diagnostics["exact_candidate_page_lse_mean"] = torch.where(
                        finite_exact, exact_candidate_lse,
                        torch.zeros_like(exact_candidate_lse),
                    ).sum() / finite_exact.sum().clamp_min(1)
                if analysis_selection_logits is not None:
                    first_competitive = (
                        (self.top_k_chunks + 1) * self.chunk_size + self.local_window
                    )
                    ids = torch.arange(
                        min(seq_len, first_competitive), seq_len,
                        device=x.device, dtype=torch.int64,
                    )[: self.diagnostic_max_queries]
                    self._routing_entropy = (
                        _eligible_route_entropy(
                            analysis_selection_logits, metadata,
                            self.local_window, tile_ids=ids,
                        ) if ids.numel() else torch.zeros((), device=x.device)
                    )
                    diagnostics["routing_entropy"] = self._routing_entropy
                if forced_route_chunk_ids is not None:
                    diagnostics.update(
                        _forced_route_coverage_diagnostics(
                            metadata.top_chunk_idx, forced_route_chunk_ids
                        )
                    )
                self._routing_diagnostics = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in diagnostics.items()
                }

        if return_metadata and return_auxiliary:
            return output, metadata, auxiliary
        if return_metadata:
            return output, metadata
        if return_auxiliary:
            return output, auxiliary
        return output


    def init_incremental_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HISAIncrementalState:
        """Create an empty exact-page inference cache for one-token decoding."""
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        resolved_device = parameter.device if device is None else torch.device(device)
        resolved_dtype = parameter.dtype if dtype is None else dtype
        empty_tokens = torch.empty(
            int(batch_size), self.H, 0, self.hd,
            device=resolved_device, dtype=resolved_dtype,
        )
        empty_pages = torch.empty(
            int(batch_size), self.H, 0, self.chunk_size, self.hd,
            device=resolved_device, dtype=resolved_dtype,
        )
        hierarchy_size = self.hierarchy_group_size if self.hierarchical_routing else 1
        routing_addresses = _empty_incremental_address_cache(
            int(batch_size), self.H, self.hd,
            mode=self.representative_mode,
            hierarchy_group_size=hierarchy_size,
            device=resolved_device,
        )
        attention_addresses = _empty_incremental_address_cache(
            int(batch_size), self.H, self.hd,
            mode=self.representative_mode,
            hierarchy_group_size=hierarchy_size,
            device=resolved_device,
        )
        return HISAIncrementalState(
            position=0,
            local_key=empty_tokens,
            local_value=empty_tokens.clone(),
            local_positions=torch.empty(0, device=resolved_device, dtype=torch.int64),
            pending_base_global_key=empty_tokens.clone(),
            pending_global_key=empty_tokens.clone(),
            pending_global_value=empty_tokens.clone(),
            completed_base_global_key=empty_pages,
            completed_global_key=empty_pages.clone(),
            completed_global_value=empty_pages.clone(),
            completed_routing_addresses=routing_addresses,
            completed_attention_addresses=attention_addresses,
        )

    def _incremental_candidate_chunks(
        self,
        query: torch.Tensor,
        addresses: HISAChunkAddresses,
        global_key_pages: torch.Tensor,
        *,
        eligible_count: int,
        secondary_addresses: HISAChunkAddresses | None = None,
    ) -> torch.Tensor:
        """Apply parent→child top-M and exact reranking to one incremental row."""
        query_normalized = _normalized_routing_query(query)
        batch_size, heads = query_normalized.shape[:2]
        child_count = addresses.values.shape[2]
        if eligible_count <= 0 or child_count == 0:
            return torch.full(
                (batch_size, heads, 1, self.top_k_chunks), -1,
                device=query_normalized.device, dtype=torch.int32,
            )

        def coarse_candidates(source: HISAChunkAddresses) -> torch.Tensor:
            if source.values.shape[2] != child_count:
                raise ValueError("incremental address sources must have equal child counts")
            coarse = _dense_routing_score_surface(
                query_normalized,
                source,
                reduction=self.representative_score_reduction,
                temperature=self.representative_lse_temperature,
                coherence_weight=self.representative_coherence_weight,
            )
            chunk_ids = torch.arange(
                child_count, device=query_normalized.device, dtype=torch.int64
            ).reshape(1, 1, 1, child_count)
            coarse = coarse.masked_fill(
                chunk_ids >= int(eligible_count), float("-inf")
            )

            if (
                self.hierarchical_routing
                and source.parent_values is not None
                and source.parent_coherence is not None
                and source.parent_group_size > 1
            ):
                group = int(source.parent_group_size)
                eligible_parents = int(eligible_count) // group
                if eligible_parents > 0:
                    parent_scores = _score_chunk_addresses(
                        query_normalized,
                        source.parent_values,
                        reduction="max",
                        temperature=1.0,
                    ) + _coherence_score_correction(
                        source.parent_coherence,
                        self.representative_coherence_weight,
                    )[:, :, None, :]
                    parent_ids = torch.arange(
                        source.parent_values.shape[2],
                        device=query_normalized.device,
                        dtype=torch.int64,
                    ).reshape(1, 1, 1, -1)
                    parent_scores = parent_scores.masked_fill(
                        parent_ids >= eligible_parents, float("-inf")
                    )
                    parent_k = min(self.parent_top_k, eligible_parents)
                    _, selected_parents = parent_scores.topk(parent_k, dim=-1)
                    offsets = torch.arange(
                        group, device=query_normalized.device, dtype=torch.int64
                    )
                    parent_children = (
                        selected_parents[..., None] * group + offsets
                    ).reshape(batch_size, heads, 1, -1)
                else:
                    parent_children = torch.empty(
                        batch_size, heads, 1, 0,
                        device=query_normalized.device, dtype=torch.int64,
                    )
                tail_start = eligible_parents * group
                tail = torch.arange(
                    tail_start, int(eligible_count),
                    device=query_normalized.device, dtype=torch.int64,
                ).reshape(1, 1, 1, -1).expand(batch_size, heads, 1, -1)
                candidates = torch.cat((parent_children, tail), dim=-1)
                deduplicated, unique = _deduplicate_fixed_candidates(
                    candidates, entry_count=child_count
                )
                safe = deduplicated.clamp(max=child_count - 1)
                candidate_scores = torch.gather(coarse, -1, safe)
                candidate_scores = candidate_scores.masked_fill(
                    ~unique, float("-inf")
                )
                candidate_k = min(
                    self.routing_candidate_count, candidate_scores.shape[-1]
                )
                values, order = candidate_scores.topk(candidate_k, dim=-1)
                selected = torch.gather(deduplicated, -1, order)
                return torch.where(
                    torch.isfinite(values), selected, torch.full_like(selected, -1)
                )

            candidate_k = min(self.routing_candidate_count, int(eligible_count))
            values, selected = coarse.topk(candidate_k, dim=-1)
            return torch.where(
                torch.isfinite(values), selected, torch.full_like(selected, -1)
            )

        primary = coarse_candidates(addresses)
        if secondary_addresses is not None:
            secondary = coarse_candidates(secondary_addresses)
            union, unique = _deduplicate_fixed_candidates(
                torch.cat((primary, secondary), dim=-1), entry_count=child_count
            )
            coarse_candidates = torch.where(
                unique, union, torch.full_like(union, -1)
            )
        else:
            coarse_candidates = primary

        if not self.exact_page_rerank or coarse_candidates.shape[-1] <= self.top_k_chunks:
            selected = coarse_candidates[..., : self.top_k_chunks]
            if selected.shape[-1] < self.top_k_chunks:
                selected = F.pad(
                    selected, (0, self.top_k_chunks - selected.shape[-1]), value=-1
                )
            return selected.to(torch.int32)

        slots = coarse_candidates.shape[-1]
        safe = coarse_candidates.clamp_min(0)
        expanded_pages = global_key_pages[:, :, None].expand(
            batch_size, heads, 1, child_count, self.chunk_size, self.hd
        )
        gather_index = safe[..., None, None].expand(
            batch_size, heads, 1, slots, self.chunk_size, self.hd
        )
        selected_pages = torch.gather(expanded_pages, 3, gather_index)
        token_scores = torch.einsum(
            "bhqd,bhqkmd->bhqkm", query.float(), selected_pages.float()
        ) / math.sqrt(self.hd)
        exact_lse = torch.logsumexp(token_scores, dim=-1).masked_fill(
            coarse_candidates < 0, float("-inf")
        )
        k = min(self.top_k_chunks, slots)
        values, order = exact_lse.topk(k, dim=-1)
        selected = torch.gather(coarse_candidates, -1, order)
        selected = torch.where(
            torch.isfinite(values), selected, torch.full_like(selected, -1)
        )
        if k < self.top_k_chunks:
            selected = F.pad(selected, (0, self.top_k_chunks - k), value=-1)
        return selected.to(torch.int32)

    @torch.no_grad()
    def forward_incremental(
        self,
        x_t: torch.Tensor,
        state: HISAIncrementalState,
        kv_inject: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, HISAIncrementalState]:
        """Decode one token using cached local history and completed exact pages.

        The state retains exact completed-page K/V, so memory grows with context.
        It eliminates prefix reprojection/re-addressing of token activations but is
        intentionally a correctness/reference cache rather than an O(1) cache.
        """
        if self.training:
            raise RuntimeError("forward_incremental is inference-only; call eval()")
        if x_t.ndim != 3 or x_t.shape[1] != 1 or x_t.shape[2] != self.D:
            raise ValueError(f"x_t must have shape [B,1,{self.D}]")
        batch_size = x_t.shape[0]
        if state.local_key.shape[:2] != (batch_size, self.H):
            raise ValueError("incremental state batch/head geometry does not match x_t")
        if x_t.device != state.local_key.device:
            raise ValueError("x_t and incremental state must be on the same device")
        position = int(state.position)
        if self.max_seq_len is not None and position >= self.max_seq_len:
            raise ValueError("incremental position exceeds configured max_seq_len")

        query_flat, key_flat, value_flat, gate = self.qkvg_proj(x_t).split(self.D, dim=-1)
        query = _to_heads(query_flat, batch_size, 1, self.H, self.hd)
        local_key_t = _to_heads(key_flat, batch_size, 1, self.H, self.hd)
        local_value_t = _to_heads(value_flat, batch_size, 1, self.H, self.hd)
        global_key_t, global_value_t = local_key_t, local_value_t
        if self.global_adapter_rank:
            assert self.global_k_down is not None and self.global_k_up is not None
            assert self.global_v_down is not None and self.global_v_up is not None
            global_key_t = global_key_t + _to_heads(
                self.global_k_up(self.global_k_down(x_t)),
                batch_size, 1, self.H, self.hd,
            )
            global_value_t = global_value_t + _to_heads(
                self.global_v_up(self.global_v_down(x_t)),
                batch_size, 1, self.H, self.hd,
            )
        base_global_key_t = global_key_t
        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            if key_delta.shape != global_key_t.shape or value_delta.shape != global_value_t.shape:
                raise ValueError("incremental kv_inject must contain [B,H,1,HD] tensors")
            theta_k = self.npci_theta_max * torch.tanh(self.npci_theta_k)
            theta_v = self.npci_theta_max * torch.tanh(self.npci_theta_v)
            global_key_t = _magnitude_aware_rotate(global_key_t, key_delta, theta_k)
            global_value_t = _magnitude_aware_rotate(global_value_t, value_delta, theta_v)
        if self.global_key_calibration == "rms_match_local":
            one = torch.ones(batch_size, device=x_t.device, dtype=torch.int32)
            global_key_t = _rms_match_global_key(global_key_t, local_key_t, one)

        if state.local_key.shape[2] == 0:
            local_output = torch.zeros_like(query)
            local_lse = torch.full(
                (batch_size, self.H, 1), float("-inf"),
                device=x_t.device, dtype=torch.float32,
            )
        else:
            local_mask = _strict_local_or_boundary_key_mask(
                torch.tensor(position, device=x_t.device, dtype=torch.int64),
                state.local_positions,
                self.local_window,
                self.chunk_size,
            )
            scores = torch.einsum(
                "bhqd,bhld->bhql", query.float(), state.local_key.float()
            ) / math.sqrt(self.hd)
            scores = scores.masked_fill(~local_mask.reshape(1, 1, 1, -1), float("-inf"))
            local_lse = torch.logsumexp(scores, dim=-1)
            probability = torch.softmax(scores, dim=-1)
            probability = torch.where(
                torch.isfinite(scores), probability, torch.zeros_like(probability)
            )
            local_output = torch.einsum(
                "bhql,bhld->bhqd", probability.to(state.local_value.dtype),
                state.local_value,
            )

        completed_count = state.completed_global_key.shape[2]
        eligible_count = min(
            completed_count,
            max(0, (position - self.local_window) // self.chunk_size),
        )
        route_evidence: HISARouteEvidence | None = None
        route_identity_features: torch.Tensor | None = None
        absolute_route_summary: torch.Tensor | None = None
        if eligible_count <= 0:
            global_output = torch.zeros_like(query)
            global_lse = torch.full_like(local_lse, float("-inf"))
        else:
            addresses = state.completed_routing_addresses
            attention_addresses = state.completed_attention_addresses
            q_normalized = _normalized_routing_query(query)
            secondary_addresses = (
                attention_addresses
                if (
                    self.route_from_base_global_key
                    and self.dual_source_candidate_union
                    and self.exact_page_rerank
                )
                else None
            )
            selected_chunks = self._incremental_candidate_chunks(
                query,
                addresses,
                state.completed_global_key,
                eligible_count=eligible_count,
                secondary_addresses=secondary_addresses,
            )
            prior_addresses = (
                attention_addresses
                if (
                    self.rerank_selected_priors_with_post_packet_representatives
                    and self.route_from_base_global_key
                )
                else addresses
            )
            similarity = _selected_routing_scores(
                q_normalized,
                prior_addresses,
                selected_chunks,
                reduction=self.representative_score_reduction,
                temperature=self.representative_lse_temperature,
                coherence_weight=self.representative_coherence_weight,
            )
            valid_route = torch.isfinite(similarity)
            count = valid_route.sum(-1, keepdim=True).clamp_min(1)
            mean = torch.where(
                valid_route, similarity, torch.zeros_like(similarity)
            ).sum(-1, keepdim=True) / count
            centered = torch.where(
                valid_route, similarity - mean, torch.zeros_like(similarity)
            )
            route = (
                centered * self.effective_route_prior_scale.reshape(1, self.H, 1, 1)
            ).masked_fill(~valid_route, float("-inf"))
            semantic_similarity = torch.where(
                valid_route, similarity, torch.zeros_like(similarity)
            )
            route_identity_features, absolute_route_summary = _route_identity_features(
                semantic_similarity,
                selected_chunks.to(torch.int64),
                route_scale_by_head=self.route_prior_scale,
                temperature=self.temperature,
                chunk_size=self.chunk_size,
                query_positions=torch.tensor([position], device=x_t.device),
            )
            route = route + self._route_confidence_offset(
                absolute_route_summary, selected_chunks
            )[..., None].to(route.dtype)
            local_count = self._local_count_geometry(x_t.device, position + 1)[-1]
            global_count = torch.isfinite(route).sum(-1).float() * float(self.chunk_size)
            raw_count = torch.where(
                (global_count > 0.0) & (local_count > 0.0),
                torch.log(local_count.clamp_min(1.0) / global_count.clamp_min(1.0)),
                torch.zeros_like(global_count),
            )
            common = (
                raw_count * self.count_correction_scale.reshape(1, self.H, 1)
                + self.bounded_global_lane_logit_bias.reshape(1, self.H, 1)
            )
            route = route + common[..., None].to(route.dtype)

            safe = selected_chunks.clamp_min(0).to(torch.int64)
            slots = safe.shape[-1]
            expanded_k = state.completed_global_key[:, :, None].expand(
                batch_size, self.H, 1, completed_count, self.chunk_size, self.hd
            )
            expanded_v = state.completed_global_value[:, :, None].expand_as(expanded_k)
            page_index = safe[..., None, None].expand(
                batch_size, self.H, 1, slots, self.chunk_size, self.hd
            )
            selected_k = torch.gather(expanded_k, 3, page_index)
            selected_v = torch.gather(expanded_v, 3, page_index)
            token_scores = torch.einsum(
                "bhqd,bhqkmd->bhqkm", query.float(), selected_k.float()
            ) / math.sqrt(self.hd)
            token_scores = token_scores + route[..., None].float()
            token_scores = token_scores.masked_fill(
                (selected_chunks < 0)[..., None], float("-inf")
            )
            route_lse = torch.logsumexp(token_scores, dim=-1)
            global_lse = torch.logsumexp(
                token_scores.reshape(batch_size, self.H, 1, -1), dim=-1
            )
            global_probability = torch.softmax(
                token_scores.reshape(batch_size, self.H, 1, -1), dim=-1
            ).reshape_as(token_scores)
            global_probability = torch.where(
                torch.isfinite(token_scores), global_probability,
                torch.zeros_like(global_probability),
            )
            global_output = torch.einsum(
                "bhqkm,bhqkmd->bhqd",
                global_probability.to(selected_v.dtype), selected_v,
            )
            if self.binding_rank:
                assert self.bind_evidence_weight is not None
                projected_tokens = torch.einsum(
                    "bhqkmd,hdr->bhqkmr", selected_v, self.bind_evidence_weight
                )
                safe_route_lse = torch.where(
                    torch.isfinite(route_lse), route_lse, torch.zeros_like(route_lse)
                )
                route_probability = torch.where(
                    torch.isfinite(token_scores),
                    torch.exp(token_scores - safe_route_lse[..., None]),
                    torch.zeros_like(token_scores),
                )
                route_output = torch.einsum(
                    "bhqkm,bhqkmr->bhqkr",
                    route_probability.to(projected_tokens.dtype), projected_tokens,
                )
                route_evidence = HISARouteEvidence(
                    output=route_output, lse=route_lse.float()
                )

        attended, combined_lse, _head_global_mass = _merge_attention_lanes_with_mass(
            local_output, local_lse, global_output, global_lse
        )
        merged = attended.permute(0, 2, 1, 3).reshape(batch_size, 1, self.D)
        projected = self.W_o(merged)
        binding_correction = torch.zeros_like(projected)
        if self.binding_rank and route_evidence is not None:
            assert route_identity_features is not None and absolute_route_summary is not None
            route_log_mass = torch.where(
                torch.isfinite(route_evidence.lse)
                & torch.isfinite(combined_lse.float()[..., None]),
                route_evidence.lse.float() - combined_lse.float()[..., None],
                torch.full_like(route_evidence.lse.float(), float("-inf")),
            )
            binding_correction, _ = self._binding_correction(
                x_t,
                route_evidence.output,
                route_log_mass,
                route_identity_features,
                absolute_route_summary,
            )
        output = (projected + binding_correction) * torch.sigmoid(gate)

        local_key_history = torch.cat((state.local_key, local_key_t), dim=2)
        local_value_history = torch.cat((state.local_value, local_value_t), dim=2)
        local_positions = torch.cat((
            state.local_positions,
            torch.tensor([position], device=x_t.device, dtype=torch.int64),
        ))
        max_history = self.local_window + self.chunk_size - 1
        if local_key_history.shape[2] > max_history:
            local_key_history = local_key_history[:, :, -max_history:]
            local_value_history = local_value_history[:, :, -max_history:]
            local_positions = local_positions[-max_history:]

        pending_base_global_key = torch.cat(
            (state.pending_base_global_key, base_global_key_t), dim=2
        )
        pending_global_key = torch.cat(
            (state.pending_global_key, global_key_t), dim=2
        )
        pending_global_value = torch.cat(
            (state.pending_global_value, global_value_t), dim=2
        )
        completed_base = state.completed_base_global_key
        completed_key = state.completed_global_key
        completed_value = state.completed_global_value
        routing_address_cache = state.completed_routing_addresses
        attention_address_cache = state.completed_attention_addresses
        if pending_global_key.shape[2] == self.chunk_size:
            completed_base = torch.cat(
                (completed_base, pending_base_global_key.unsqueeze(2)), dim=2
            )
            completed_key = torch.cat(
                (completed_key, pending_global_key.unsqueeze(2)), dim=2
            )
            completed_value = torch.cat(
                (completed_value, pending_global_value.unsqueeze(2)), dim=2
            )
            attention_address_cache = _append_incremental_address_cache(
                attention_address_cache,
                pending_global_key,
                completed_key,
                chunk_size=self.chunk_size,
                blend_alpha=self.representative_blend_for_addresses,
                mode=self.representative_mode,
            )
            if self.route_from_base_global_key:
                routing_address_cache = _append_incremental_address_cache(
                    routing_address_cache,
                    pending_base_global_key,
                    completed_base,
                    chunk_size=self.chunk_size,
                    blend_alpha=self.representative_blend_for_addresses,
                    mode=self.representative_mode,
                )
            else:
                routing_address_cache = attention_address_cache
            pending_base_global_key = pending_base_global_key[:, :, :0]
            pending_global_key = pending_global_key[:, :, :0]
            pending_global_value = pending_global_value[:, :, :0]
        elif pending_global_key.shape[2] > self.chunk_size:
            raise RuntimeError("incremental pending page exceeded chunk_size")

        next_state = HISAIncrementalState(
            position=position + 1,
            local_key=local_key_history,
            local_value=local_value_history,
            local_positions=local_positions,
            pending_base_global_key=pending_base_global_key,
            pending_global_key=pending_global_key,
            pending_global_value=pending_global_value,
            completed_base_global_key=completed_base,
            completed_global_key=completed_key,
            completed_global_value=completed_value,
            completed_routing_addresses=routing_address_cache,
            completed_attention_addresses=attention_address_cache,
        )
        return output, next_state
