from __future__ import annotations

import os
from contextlib import nullcontext

import torch


def _dsqg_w_profile_enabled() -> bool:
    return os.getenv("DWARF_PROFILE_DSQG_W", "0") == "1"


def _dsqg_w_geometry_audit_enabled() -> bool:
    return os.getenv("DWARF_DSQG_W_GEOMETRY_AUDIT", "0") == "1" or _dsqg_w_profile_enabled()


def _dsqg_w_profile_range(name: str):
    if _dsqg_w_profile_enabled():
        return torch.profiler.record_function(f"dsqg_w/{name}")
    return nullcontext()


def _dsqg_w_geometry_telemetry(
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lightweight candidate geometry audit for fixed-offset/slab eligibility.

    Only called behind DWARF_DSQG_W_GEOMETRY_AUDIT/DWARF_PROFILE_DSQG_W because
    unique/mode checks are diagnostic and may synchronize. Outputs are detached
    tensors so telemetry never touches autograd.
    """

    if cand_token_indices.ndim != 3:
        return {}
    device = cand_token_indices.device
    dtype = torch.float32
    bsz, seq_len, j_count = cand_token_indices.shape
    if j_count == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "dsqg_w_geometry_fixed_slots": zero,
            "dsqg_w_geometry_fixed_slot_fraction": zero,
            "dsqg_w_geometry_mode_delta_fraction": zero,
            "dsqg_w_geometry_slab_candidate_slots": zero,
        }
    pos = torch.arange(seq_len, device=device, dtype=cand_token_indices.dtype).view(1, seq_len)
    fixed_slots = 0
    mode_fraction_total = 0.0
    slab_candidate_slots = 0
    for slot in range(int(j_count)):
        valid = cand_mask[:, :, slot] & (cand_token_indices[:, :, slot] >= 0)
        valid_count = int(valid.sum().detach().cpu().item())
        if valid_count <= 0:
            continue
        delta = (pos - cand_token_indices[:, :, slot]).masked_select(valid)
        source_vals = cand_sources[:, :, slot].masked_select(valid)
        type_vals = cand_types[:, :, slot].masked_select(valid)
        unique_delta, delta_counts = torch.unique(delta, return_counts=True)
        unique_source = torch.unique(source_vals)
        unique_type = torch.unique(type_vals)
        max_count = int(delta_counts.max().detach().cpu().item()) if delta_counts.numel() else 0
        mode_fraction_total += float(max_count) / float(valid_count)
        is_const_source_type = unique_source.numel() == 1 and unique_type.numel() == 1
        is_fixed = unique_delta.numel() == 1 and is_const_source_type
        if is_fixed:
            fixed_slots += 1
        # A deliberately permissive first-pass slab proxy: mostly-one-delta slots
        # with stable source/type are worth deeper Nsight/Triton treatment.
        if is_const_source_type and (float(max_count) / float(valid_count)) >= 0.95:
            slab_candidate_slots += 1
    denom = float(max(int(j_count), 1))
    return {
        "dsqg_w_geometry_fixed_slots": torch.tensor(float(fixed_slots), device=device, dtype=dtype),
        "dsqg_w_geometry_fixed_slot_fraction": torch.tensor(float(fixed_slots) / denom, device=device, dtype=dtype),
        "dsqg_w_geometry_mode_delta_fraction": torch.tensor(mode_fraction_total / denom, device=device, dtype=dtype),
        "dsqg_w_geometry_slab_candidate_slots": torch.tensor(float(slab_candidate_slots), device=device, dtype=dtype),
    }

__all__ = [
    "_dsqg_w_profile_enabled",
    "_dsqg_w_geometry_audit_enabled",
    "_dsqg_w_profile_range",
    "_dsqg_w_geometry_telemetry",
]
