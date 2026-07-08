from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .candidate_provider import CandidateProvider
from .candidate_types import CandidateSource, CandidateType
from .config import DSQGWConfig
from .ebh_packet import DSQGWEvidenceBindingHub
from .evidence_prior import DSQGWEvidencePriorComposer
from .gates import _forced_gate_value
from .instrumentation import _dsqg_w_profile_range
from .sourcewise_gather import _DSQGWSourcewiseCandidateStateGather
from .sourcewise_read import (
    _DSQGWSourcewiseTritonCompactRead,
    _TRITON_SOURCEWISE_AVAILABLE,
    _dsqg_w_sourcewise_read_slots_kernel,
    triton,
)
from .triton_schedule import _dsqg_w_triton_schedule
from .typed_mixer import DSQGWTypedCandidateMixer
from .width_cell import DSQGWWidthCell, _hisa_evidence_type_mask, width_pair_transfer_loss


def _materialized_triton_compact_read():
    # Legacy/materialized transformed compact-read now lives with sourcewise read
    # helpers so block.py no longer depends on the compatibility monolith. Keep
    # the import lazy to avoid unnecessary Triton/autograd setup on light imports.
    from .sourcewise_read import _DSQGWMaterializedTritonCompactRead

    return _DSQGWMaterializedTritonCompactRead


def _read_type_ids_from_config(config: DSQGWConfig) -> tuple[int, ...]:
    """Return a semantics-preserving superset of candidate types a config can emit."""
    type_ids: set[int] = set()
    if config.null_fallback:
        type_ids.add(int(CandidateType.NULL))
    if config.local_offsets:
        type_ids.add(int(CandidateType.LOCAL))
    if config.k_question > 0:
        type_ids.add(int(CandidateType.QUESTION))
    if config.k_hisa_evidence > 0:
        if config.typed_hisa_reps:
            type_ids.update(
                {
                    int(CandidateType.HISA_EVIDENCE_REP0),
                    int(CandidateType.HISA_EVIDENCE_REP1),
                    int(CandidateType.HISA_EVIDENCE_REP2),
                    int(CandidateType.HISA_EVIDENCE_REP3),
                    int(CandidateType.HISA_EVIDENCE),
                }
            )
        else:
            type_ids.add(int(CandidateType.HISA_EVIDENCE))
    if config.long_offsets:
        type_ids.add(int(CandidateType.LONG_OFFSET))
    if config.k_chunk > 0:
        type_ids.add(int(CandidateType.CHUNK_REP))
    if config.k_l3_skip > 0:
        type_ids.add(int(CandidateType.L3_SKIP))
    return tuple(sorted(type_id for type_id in type_ids if 0 <= type_id < config.n_types))

class DSQGWBlock(nn.Module):
    """Diagnostic DSQG-W semantic-width recomposer.

    Inputs:
      x:            [B, T, D]
      cand_states:  [B, T, J, D]
      cand_types:   [B, T, J]
      cand_sources: [B, T, J]
      cand_mask:    [B, T, J] bool
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_types: int,
        n_sources: int,
        bottleneck: int,
        max_candidates: int,
        local_type_id: int,
        gate_init: float = -5.0,
        fuse_init_std: float = 1e-4,
        use_width_cell: bool = False,
        width_bottleneck: int = 64,
        width_gate_init: float = -5.0,
        width_self_bias_init: float = 0.0,
        width_entropy_floor: float = 0.0,
        width_entropy_weight: float = 0.0,
        use_typed_mixer: bool = False,
        typed_mixer_bottleneck: int = 64,
        typed_mixer_gate_init: float = -5.0,
        use_query_type_bias: bool = False,
        use_evidence_prior: bool = False,
        evidence_prior_clip: float = 2.0,
        evidence_prior_init_scale: float = 0.0,
        use_evidence_binding_hub: bool = False,
        ebh_bottleneck: int = 256,
        ebh_gate_init: float = -5.0,
        ebh_phase_bands: int = 4,
        ebh_score_features: bool = True,
        ebh_pair_mixer: bool = False,
        ebh_pair_rank: int = 64,
        ebh_pair_gate_init: float = -2.5,
        read_type_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.dh = int(d // n_heads)
        self.n_types = int(n_types)
        self.n_sources = int(n_sources)
        self.max_candidates = int(max_candidates)
        self.local_type_id = int(local_type_id)
        self.read_type_ids = tuple(range(self.n_types)) if read_type_ids is None else tuple(int(t) for t in read_type_ids)
        self.width_cell = (
            DSQGWWidthCell(
                d=d,
                n_heads=n_heads,
                n_types=n_types,
                n_sources=n_sources,
                bottleneck=width_bottleneck,
                gate_init=width_gate_init,
                self_bias_init=width_self_bias_init,
                entropy_floor=width_entropy_floor,
                entropy_weight=width_entropy_weight,
            )
            if use_width_cell
            else None
        )
        self.typed_mixer = (
            DSQGWTypedCandidateMixer(
                d=d,
                n_heads=n_heads,
                n_types=n_types,
                bottleneck=typed_mixer_bottleneck,
                gate_init=typed_mixer_gate_init,
            )
            if use_typed_mixer
            else None
        )
        self.use_query_type_bias = bool(use_query_type_bias)
        self.evidence_prior = (
            DSQGWEvidencePriorComposer(
                n_types=n_types,
                n_sources=n_sources,
                clip=evidence_prior_clip,
                init_scale=evidence_prior_init_scale,
            )
            if use_evidence_prior
            else None
        )
        self.evidence_binding_hub = (
            DSQGWEvidenceBindingHub(
                d=d,
                n_types=n_types,
                n_sources=n_sources,
                bottleneck=ebh_bottleneck,
                gate_init=ebh_gate_init,
                phase_bands=ebh_phase_bands,
                use_score_features=ebh_score_features,
                use_pair_mixer=ebh_pair_mixer,
                pair_rank=ebh_pair_rank,
                pair_gate_init=ebh_pair_gate_init,
            )
            if use_evidence_binding_hub
            else None
        )

        self.norm_x = nn.LayerNorm(d)
        self.norm_c = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.role_key = nn.Embedding(n_types, d)
        self.source_key = nn.Embedding(n_sources, d)
        self.type_bias = nn.Parameter(torch.zeros(n_types, n_heads))
        self.query_type_bias = nn.Linear(d, n_types * n_heads, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(n_sources, n_heads))
        self.read_mix = nn.Linear((n_types + 1) * d, d, bias=False)
        self.norm_z = nn.LayerNorm(4 * d)
        self.fuse = nn.Sequential(
            nn.Linear(4 * d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )
        self.gate = nn.Parameter(torch.full((d,), float(gate_init)))
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=float(fuse_init_std))
        nn.init.zeros_(self.fuse[-1].bias)

    @classmethod
    def from_config(cls, config: DSQGWConfig) -> "DSQGWBlock":
        return cls(
            d=config.d,
            n_heads=config.n_heads,
            n_types=config.n_types,
            n_sources=config.n_sources,
            bottleneck=config.bottleneck,
            max_candidates=config.max_candidates,
            local_type_id=config.local_type_id,
            gate_init=config.gate_init,
            fuse_init_std=config.fuse_init_std,
            use_width_cell=config.use_width_cell,
            width_bottleneck=config.width_bottleneck,
            width_gate_init=config.width_gate_init,
            width_self_bias_init=config.width_self_bias_init,
            width_entropy_floor=config.width_entropy_floor,
            width_entropy_weight=config.width_entropy_weight,
            use_typed_mixer=config.use_typed_mixer,
            typed_mixer_bottleneck=config.typed_mixer_bottleneck,
            typed_mixer_gate_init=config.typed_mixer_gate_init,
            use_query_type_bias=config.use_query_type_bias,
            use_evidence_prior=config.use_evidence_prior,
            evidence_prior_clip=config.evidence_prior_clip,
            evidence_prior_init_scale=config.evidence_prior_init_scale,
            use_evidence_binding_hub=config.use_evidence_binding_hub,
            ebh_bottleneck=config.ebh_bottleneck,
            ebh_gate_init=config.ebh_gate_init,
            ebh_phase_bands=config.ebh_phase_bands,
            ebh_score_features=config.ebh_score_features,
            ebh_pair_mixer=config.ebh_pair_mixer,
            ebh_pair_rank=config.ebh_pair_rank,
            ebh_pair_gate_init=config.ebh_pair_gate_init,
            read_type_ids=_read_type_ids_from_config(config),
        )

    def _mix_typed_reads(
        self,
        r_all: torch.Tensor,
        probs: torch.Tensor,
        v: torch.Tensor,
        cand_types: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bsz, seq_len, _ = r_all.shape
        weight = self.read_mix.weight
        read_slots = [r_all]
        typed_read_norms = [r_all.new_tensor(0.0) for _ in range(self.n_types)]
        active_type_ids: list[int] = []
        for type_id in self.read_type_ids:
            if type_id < 0 or type_id >= self.n_types:
                continue
            type_mask = ((cand_types == type_id) & cand_mask).to(probs.dtype)
            r_type_h = torch.einsum("btjh,btjhd,btj->bthd", probs, v, type_mask)
            r_type = r_type_h.reshape(bsz, seq_len, self.d)
            read_slots.append(r_type)
            active_type_ids.append(int(type_id))
            if os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") != "1":
                typed_read_norms[type_id] = r_type.norm(dim=-1).mean()
        if os.getenv("DWARF_DSQG_W_DENSE_BATCHED_READ_MIX", "0") == "1" and len(read_slots) > 1:
            slices = [weight[:, : self.d]]
            for type_id in active_type_ids:
                start = (type_id + 1) * self.d
                slices.append(weight[:, start : start + self.d])
            weight_by_slot = torch.stack(slices, dim=0)
            slots_by_slot = torch.stack(read_slots, dim=2).reshape(-1, len(read_slots), self.d).transpose(0, 1)
            read = torch.bmm(slots_by_slot, weight_by_slot.transpose(1, 2)).sum(dim=0)
            return read.reshape(bsz, seq_len, self.d), typed_read_norms
        read = F.linear(r_all, weight[:, : self.d])
        for type_id, r_type in zip(active_type_ids, read_slots[1:]):
            start = (type_id + 1) * self.d
            read = read + F.linear(r_type, weight[:, start : start + self.d])
        return read, typed_read_norms

    def _mix_compact_read_slots(self, read_slots: torch.Tensor, *, batched: bool | None = None) -> torch.Tensor:
        if read_slots.ndim != 4:
            raise ValueError("read_slots must have shape [B, T, S, D]")
        if read_slots.shape[-1] != self.d:
            raise ValueError("read_slots last dim must match block d")
        expected_slots = len(self.read_type_ids) + 1
        if read_slots.shape[2] != expected_slots:
            raise ValueError(f"read_slots slot count {read_slots.shape[2]} does not match expected {expected_slots}")
        if batched is None:
            batched = os.getenv("DWARF_DSQG_W_BATCHED_READ_MIX", "0") == "1"
        weight = self.read_mix.weight
        if not batched:
            read = F.linear(read_slots[:, :, 0, :], weight[:, : self.d])
            for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
                if 0 <= int(type_id) < self.n_types:
                    start = (int(type_id) + 1) * self.d
                    read = read + F.linear(read_slots[:, :, slot_idx, :], weight[:, start : start + self.d])
            return read
        slices = [weight[:, : self.d]]
        for type_id in self.read_type_ids:
            start = (int(type_id) + 1) * self.d
            slices.append(weight[:, start : start + self.d])
        weight_by_slot = torch.stack(slices, dim=0)  # [S, D_out, D_in]
        slots_by_slot = read_slots.reshape(-1, expected_slots, self.d).transpose(0, 1)  # [S, B*T, D]
        mixed = torch.bmm(slots_by_slot, weight_by_slot.transpose(1, 2)).sum(dim=0)
        return mixed.reshape(read_slots.shape[0], read_slots.shape[1], self.d)

    @staticmethod
    def _gather_source_rows(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = states.shape[:2]
        batch_offsets = torch.arange(bsz, device=states.device, dtype=torch.long).reshape(bsz, 1) * seq_len
        flat_indices = (batch_offsets + token_indices.to(torch.long)).reshape(-1)
        return states.reshape(bsz * seq_len, *states.shape[2:]).index_select(0, flat_indices).reshape(
            bsz, seq_len, *states.shape[2:]
        )

    def _compose_evidence_prior_scores(
        self,
        cand_scores: torch.Tensor | None,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        evidence_bits: torch.Tensor | None = None,
        evidence_count: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.evidence_prior is None:
            return cand_scores, {}
        if evidence_bits is None:
            evidence_bits = CandidateProvider._candidate_type_evidence_bits(cand_types).masked_fill(~cand_mask, 0)
        if evidence_count is None:
            evidence_count = CandidateProvider._evidence_count_from_bits(evidence_bits).masked_fill(~cand_mask, 0)
        prior, telemetry = self.evidence_prior(
            cand_types,
            cand_sources,
            cand_mask,
            raw_hisa_scores=cand_scores,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
        )
        prior = prior.to(device=cand_types.device, dtype=cand_scores.dtype if cand_scores is not None else self.type_bias.dtype)
        combined = prior if cand_scores is None else cand_scores + prior.to(dtype=cand_scores.dtype)
        telemetry["dsqg_w_evidence_prior_enabled"] = prior.new_tensor(1.0).detach()
        return combined, telemetry

    @staticmethod
    def _gather_source_candidate_states(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """Gather [B,T,J] causal token indices from a [B,T,D] source surface."""
        if states.ndim != 3 or token_indices.ndim != 3:
            raise ValueError("expected states [B,T,D] and token_indices [B,T,J]")
        bsz, seq_len, d = states.shape
        if token_indices.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate token index shape must align with states [B,T]")
        gather_tokens = token_indices.clamp(0, max(seq_len - 1, 0)).to(torch.long)
        batch_offsets = torch.arange(bsz, device=states.device, dtype=torch.long).reshape(bsz, 1, 1) * seq_len
        flat_indices = (batch_offsets + gather_tokens).reshape(-1)
        gathered = states.reshape(bsz * seq_len, d).index_select(0, flat_indices)
        return gathered.reshape(bsz, seq_len, token_indices.shape[-1], d)

    def _materialize_sourcewise_candidate_states(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Materialize compact metadata only when semantic candidate machinery needs it.

        Sourcewise/Triton normally avoids [B,T,J,D] candidates.  Width-cell and
        typed-mixer semantics operate over candidate states themselves, so they
        must gather the same causal source surfaces instead of rejecting or
        bypassing those mechanisms.
        """
        final_states = x
        l3_base = l3_states if l3_states is not None else final_states
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
        }
        cand_states = x.new_zeros((*cand_token_indices.shape, x.shape[-1]))
        grouped_bases: dict[int, tuple[torch.Tensor, list[int]]] = {}
        for source_id, states in bases.items():
            if source_id == int(CandidateSource.NULL):
                continue
            cache_key = id(states)
            if cache_key not in grouped_bases:
                grouped_bases[cache_key] = (states, [])
            grouped_bases[cache_key][1].append(int(source_id))
        if (
            os.getenv("DWARF_DSQG_W_TRITON_CAND_STATE_GATHER", "1") != "0"
            and _TRITON_SOURCEWISE_AVAILABLE
            and triton is not None
            and x.is_cuda
            and chunk_rep_states is None
            and cand_token_indices.is_cuda
            and cand_sources.is_cuda
            and cand_mask.is_cuda
        ):
            return _DSQGWSourcewiseCandidateStateGather.apply(
                x,
                l3_states if l3_states is not None else x,
                cand_token_indices,
                cand_sources,
                cand_mask,
                l3_states is not None,
            )
        if os.getenv("DWARF_DSQG_W_GROUPED_SLOT_MATERIALIZE", "1") != "0":
            # The DSR-selected all-open geometry has slot-constant source groups:
            # question slots gather from final x, HISA/L3 slots gather from l3,
            # and the null slot is zero.  The older source-group implementation
            # gathered a full [B,T,J,D] tensor once for x and once for l3, then
            # masked away most slots.  That doubles candidate-state gather traffic
            # and its backward scatter for the common 4Q+4HISA+2L3+null packet.
            # When each slot belongs to a single source group, gather only those
            # slots for that base in one batched index_select.  If a diagnostic
            # geometry mixes sources within a slot, fall back to the fully general
            # path below.
            null_id = int(CandidateSource.NULL)
            grouped_slots: dict[int, tuple[torch.Tensor, list[int]]] = {}
            can_use_grouped_slots = True
            for slot_idx in range(cand_token_indices.shape[-1]):
                valid_j = cand_mask[:, :, slot_idx]
                source_j = cand_sources[:, :, slot_idx]
                if not bool(valid_j.any()):
                    continue
                if bool(((source_j == null_id) | ~valid_j).all()):
                    continue
                matched_key: int | None = None
                matched_states: torch.Tensor | None = None
                for cache_key, (states, source_ids) in grouped_bases.items():
                    source_match = torch.zeros_like(valid_j, dtype=torch.bool)
                    for source_id in source_ids:
                        source_match = source_match | (source_j == int(source_id))
                    if bool((source_match | ~valid_j).all()):
                        matched_key = int(cache_key)
                        matched_states = states
                        break
                if matched_key is None or matched_states is None:
                    can_use_grouped_slots = False
                    break
                if matched_key not in grouped_slots:
                    grouped_slots[matched_key] = (matched_states, [])
                grouped_slots[matched_key][1].append(int(slot_idx))
            if can_use_grouped_slots:
                cand_states = x.new_zeros((*cand_token_indices.shape, x.shape[-1]))
                for states, slot_indices in grouped_slots.values():
                    if not slot_indices:
                        continue
                    gathered = self._gather_source_candidate_states(
                        states,
                        cand_token_indices[:, :, slot_indices],
                    )
                    gathered = gathered * cand_mask[:, :, slot_indices, None].to(gathered.dtype)
                    cand_states[:, :, slot_indices, :] = gathered
                return cand_states
        if os.getenv("DWARF_DSQG_W_SLOT_MATERIALIZE", "0") == "1":
            slot_states: list[torch.Tensor] = []
            for j in range(cand_token_indices.shape[-1]):
                token_j = cand_token_indices[:, :, j]
                mask_j_all = cand_mask[:, :, j]
                out_j = x.new_zeros((*token_j.shape, x.shape[-1]))
                for states, source_ids in grouped_bases.values():
                    source_mask_j = torch.zeros_like(mask_j_all, dtype=torch.bool)
                    for source_id in source_ids:
                        source_mask_j = source_mask_j | (cand_sources[:, :, j] == int(source_id))
                    source_mask_j = source_mask_j & mask_j_all
                    if bool(source_mask_j.any()):
                        gathered_j = self._gather_source_rows(states, token_j.clamp(0, max(x.shape[1] - 1, 0)))
                        out_j = out_j + gathered_j * source_mask_j[..., None].to(gathered_j.dtype)
                slot_states.append(out_j)
            return torch.stack(slot_states, dim=2)
        for states, source_ids in grouped_bases.values():
            source_mask = torch.zeros_like(cand_mask, dtype=torch.bool)
            for source_id in source_ids:
                source_mask = source_mask | (cand_sources == int(source_id))
            source_mask = source_mask & cand_mask
            if bool(source_mask.any()):
                gathered = self._gather_source_candidate_states(states, cand_token_indices)
                cand_states = cand_states + gathered * source_mask[..., None].to(gathered.dtype)
        return cand_states


    def _sourcewise_width_cell_fused(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply width-cell transfer from source metadata without pre-materializing [B,T,J,D].

        This keeps the width-cell semantics identical to materializing candidate
        states first, but gathers source-normalized width projections from their
        native source surfaces.  The full-D candidate surface is only assembled at
        the residual output boundary required by downstream typed mixer/read code.
        """
        if self.width_cell is None:
            raise ValueError("sourcewise width-cell fusion requires width_cell")
        width_cell = self.width_cell
        bsz, seq_len, d = x.shape
        j_count = cand_mask.shape[-1]
        if d != width_cell.d:
            raise ValueError("sourcewise width-cell fusion d mismatch")
        final_states = x
        l3_base = l3_states if l3_states is not None else final_states
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        zero_base = torch.zeros_like(final_states)
        bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
            int(CandidateSource.NULL): zero_base,
        }

        width_dim = width_cell.width_dim
        use_triton_gather = (
            os.getenv("DWARF_DSQG_W_SOURCEWISE_WIDTH_TRITON_GATHER", "1") != "0"
            and _TRITON_SOURCEWISE_AVAILABLE
            and triton is not None
            and x.is_cuda
            and chunk_rep_states is None
            and cand_token_indices.is_cuda
            and cand_sources.is_cuda
            and cand_mask.is_cuda
        )
        if use_triton_gather:
            has_l3 = l3_states is not None
            weight = torch.cat(
                [
                    width_cell.q_proj.weight,
                    width_cell.k_proj.weight,
                    width_cell.v_proj.weight,
                    width_cell.rel_diff_proj.weight,
                    width_cell.rel_prod_proj.weight,
                ],
                dim=0,
            )
            final_proj = F.linear(width_cell.norm_c(final_states), weight).split(width_dim, dim=-1)
            if has_l3 and id(l3_base) != id(final_states):
                l3_proj = F.linear(width_cell.norm_c(l3_base), weight).split(width_dim, dim=-1)
            else:
                l3_proj = final_proj
            cand_states = _DSQGWSourcewiseCandidateStateGather.apply(
                final_states,
                l3_base,
                cand_token_indices,
                cand_sources,
                cand_mask,
                has_l3,
            )
            q, k, v, rel_diff, rel_prod = (
                _DSQGWSourcewiseCandidateStateGather.apply(
                    final_part,
                    l3_part,
                    cand_token_indices,
                    cand_sources,
                    cand_mask,
                    has_l3,
                )
                for final_part, l3_part in zip(final_proj, l3_proj)
            )
        else:
            cand_states = x.new_zeros((bsz, seq_len, j_count, d))
            q = x.new_zeros((bsz, seq_len, j_count, width_dim))
            k = x.new_zeros((bsz, seq_len, j_count, width_dim))
            v = x.new_zeros((bsz, seq_len, j_count, width_dim))
            rel_diff = x.new_zeros((bsz, seq_len, j_count, width_dim))
            rel_prod = x.new_zeros((bsz, seq_len, j_count, width_dim))

            projected_by_object: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for source_id, states in bases.items():
                source_mask = (cand_sources == int(source_id)) & cand_mask
                if not bool(source_mask.any()):
                    continue
                cache_key = id(states)
                projected = projected_by_object.get(cache_key)
                if projected is None:
                    states_n = width_cell.norm_c(states)
                    projected = F.linear(
                        states_n,
                        torch.cat(
                            [
                                width_cell.q_proj.weight,
                                width_cell.k_proj.weight,
                                width_cell.v_proj.weight,
                                width_cell.rel_diff_proj.weight,
                                width_cell.rel_prod_proj.weight,
                            ],
                            dim=0,
                        ),
                    ).split(width_dim, dim=-1)
                    projected_by_object[cache_key] = projected
                gathered_states = self._gather_source_candidate_states(states, cand_token_indices)
                mask_d = source_mask[..., None].to(gathered_states.dtype)
                cand_states = cand_states + gathered_states * mask_d
                for out_tensor, source_projection in zip((q, k, v, rel_diff, rel_prod), projected):
                    gathered_projection = self._gather_source_candidate_states(source_projection, cand_token_indices)
                    out_tensor.add_(gathered_projection * source_mask[..., None].to(gathered_projection.dtype))

        scores = torch.bmm(
            q.reshape(bsz * seq_len, j_count, width_dim),
            k.reshape(bsz * seq_len, j_count, width_dim).transpose(1, 2),
        ).reshape(bsz, seq_len, j_count, j_count) / math.sqrt(float(width_dim))
        rel_diff_hidden = torch.tanh(rel_diff[:, :, :, None, :] - rel_diff[:, :, None, :, :])
        rel_prod_hidden = torch.tanh(rel_prod[:, :, :, None, :] * rel_prod[:, :, None, :, :])
        scores = scores + (
            rel_diff_hidden * width_cell.rel_diff_score.reshape(1, 1, 1, 1, width_dim)
        ).sum(dim=-1) / math.sqrt(float(width_dim))
        scores = scores + (
            rel_prod_hidden * width_cell.rel_prod_score.reshape(1, 1, 1, 1, width_dim)
        ).sum(dim=-1) / math.sqrt(float(width_dim))
        scores = scores + width_cell.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        scores = scores + width_cell.source_pair_bias[cand_sources[:, :, :, None], cand_sources[:, :, None, :]]
        scores.diagonal(dim1=-2, dim2=-1).add_(width_cell.self_bias)

        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=3)
        probs = probs.masked_fill(~valid_pair, 0.0)

        lateral = torch.bmm(
            probs.reshape(bsz * seq_len, j_count, j_count),
            v.reshape(bsz * seq_len, j_count, width_dim),
        ).reshape(bsz, seq_len, j_count, width_dim)
        delta = width_cell.lateral_up(lateral)
        forced_gate = _forced_gate_value("DWARF_DSQG_W_FORCE_WIDTH_GATE", device=x.device, dtype=delta.dtype)
        if forced_gate is None:
            gate = torch.sigmoid(width_cell.gate).reshape(1, 1, 1, d)
            forced_gate_flag = x.new_tensor(0.0)
        else:
            gate = forced_gate.reshape(1, 1, 1, 1).expand(1, 1, 1, d)
            forced_gate_flag = x.new_tensor(1.0)
        out = cand_states + gate * delta * cand_mask[..., None].to(delta.dtype)

        p_mean = probs
        valid_targets = cand_mask.bool()
        p_safe = p_mean.clamp_min(1e-8)
        entropy_per_target = -(p_safe * p_safe.log()).sum(dim=-1)
        entropy = entropy_per_target.masked_select(valid_targets).mean()
        diag = torch.eye(j_count, device=x.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        self_mass = p_mean.masked_fill(~diag, 0.0).sum(dim=-1).masked_select(valid_targets).mean()

        def pair_mass(target_mask: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
            target_mask = target_mask & valid_targets
            source_mask = source_mask & cand_mask
            if not target_mask.any():
                return x.new_tensor(0.0)
            mass = p_mean.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
            return mass.masked_select(target_mask).mean()

        question_mask = cand_types == int(CandidateType.QUESTION)
        hisa_family_mask = _hisa_evidence_type_mask(cand_types)
        valid_delta_count = cand_mask.to(delta.dtype).sum().clamp_min(1.0)
        delta_norm = (delta.norm(dim=-1) * cand_mask.to(delta.dtype)).sum() / valid_delta_count
        transfer_aux_loss = width_pair_transfer_loss(p_mean, cand_types, cand_mask)
        entropy_penalty = torch.relu(entropy.new_tensor(width_cell.entropy_floor) - entropy)
        aux_loss = transfer_aux_loss + width_cell.entropy_weight * entropy_penalty
        telemetry = {
            "dsqg_w_width_entropy": entropy.detach(),
            "dsqg_w_width_self_mass": self_mass.detach(),
            "dsqg_w_width_gate_mean": gate.mean().detach(),
            "dsqg_w_width_gate_min": gate.min().detach(),
            "dsqg_w_width_gate_max": gate.max().detach(),
            "dsqg_w_width_forced_gate": forced_gate_flag.detach(),
            "dsqg_w_width_gate_logit_mean": width_cell.gate.detach().mean(),
            "dsqg_w_width_delta_norm": delta_norm.detach(),
            "dsqg_w_width_aux_loss": aux_loss,
            "dsqg_w_width_aux_loss_value": aux_loss.detach(),
            "dsqg_w_width_transfer_aux_loss": transfer_aux_loss.detach(),
            "dsqg_w_width_entropy_penalty": entropy_penalty.detach(),
            "dsqg_w_width_entropy_floor": entropy.new_tensor(width_cell.entropy_floor).detach(),
            "dsqg_w_width_entropy_weight": entropy.new_tensor(width_cell.entropy_weight).detach(),
            "dsqg_w_width_question_to_hisa_evidence_mass": pair_mass(question_mask, hisa_family_mask).detach(),
            "dsqg_w_width_hisa_evidence_to_question_mass": pair_mass(hisa_family_mask, question_mask).detach(),
            "dsqg_w_width_rel_diff_score_norm": width_cell.rel_diff_score.detach().norm(),
            "dsqg_w_width_rel_prod_score_norm": width_cell.rel_prod_score.detach().norm(),
            "dsqg_w_width_recompute_checkpoint": x.new_tensor(0.0).detach(),
            "dsqg_w_sourcewise_width_cell_fused": x.new_tensor(1.0).detach(),
        }
        return out, telemetry


    def _sourcewise_projected_width_score_bias(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Projected-space width control: width routing as a read score bias.

        This is intentionally not exact width-cell parity.  It keeps the width
        routing/scoring parameters trainable through the compact-read score-bias
        gradient, but avoids building transformed [B,T,J,D] candidates.
        """
        if self.width_cell is None:
            raise ValueError("projected width score bias requires width_cell")
        width_cell = self.width_cell
        bsz, seq_len, _ = x.shape
        j_count = cand_mask.shape[-1]
        width_dim = width_cell.width_dim
        final_states = x
        l3_base = l3_states if l3_states is not None else final_states
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        zero_base = torch.zeros_like(final_states)
        bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
            int(CandidateSource.NULL): zero_base,
        }
        use_triton_gather = (
            os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_TRITON_GATHER", "1") != "0"
            and _TRITON_SOURCEWISE_AVAILABLE
            and triton is not None
            and x.is_cuda
            and chunk_rep_states is None
            and cand_token_indices.is_cuda
            and cand_sources.is_cuda
            and cand_mask.is_cuda
        )
        weight = torch.cat(
            [
                width_cell.q_proj.weight,
                width_cell.k_proj.weight,
                width_cell.rel_diff_proj.weight,
                width_cell.rel_prod_proj.weight,
            ],
            dim=0,
        )
        if use_triton_gather:
            has_l3 = l3_states is not None
            final_proj = F.linear(width_cell.norm_c(final_states), weight).split(width_dim, dim=-1)
            if has_l3 and id(l3_base) != id(final_states):
                l3_proj = F.linear(width_cell.norm_c(l3_base), weight).split(width_dim, dim=-1)
            else:
                l3_proj = final_proj
            q, k, rel_diff, rel_prod = (
                _DSQGWSourcewiseCandidateStateGather.apply(
                    final_part,
                    l3_part,
                    cand_token_indices,
                    cand_sources,
                    cand_mask,
                    has_l3,
                )
                for final_part, l3_part in zip(final_proj, l3_proj)
            )
        else:
            q = x.new_zeros((bsz, seq_len, j_count, width_dim))
            k = x.new_zeros((bsz, seq_len, j_count, width_dim))
            rel_diff = x.new_zeros((bsz, seq_len, j_count, width_dim))
            rel_prod = x.new_zeros((bsz, seq_len, j_count, width_dim))
            projected_by_object: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for source_id, states in bases.items():
                source_mask = (cand_sources == int(source_id)) & cand_mask
                if not bool(source_mask.any()):
                    continue
                cache_key = id(states)
                projected = projected_by_object.get(cache_key)
                if projected is None:
                    projected = F.linear(width_cell.norm_c(states), weight).split(width_dim, dim=-1)
                    projected_by_object[cache_key] = projected
                for out_tensor, source_projection in zip((q, k, rel_diff, rel_prod), projected):
                    gathered_projection = self._gather_source_candidate_states(source_projection, cand_token_indices)
                    out_tensor.add_(gathered_projection * source_mask[..., None].to(gathered_projection.dtype))

        scores = torch.bmm(
            q.reshape(bsz * seq_len, j_count, width_dim),
            k.reshape(bsz * seq_len, j_count, width_dim).transpose(1, 2),
        ).reshape(bsz, seq_len, j_count, j_count) / math.sqrt(float(width_dim))
        rel_diff_hidden = torch.tanh(rel_diff[:, :, :, None, :] - rel_diff[:, :, None, :, :])
        rel_prod_hidden = torch.tanh(rel_prod[:, :, :, None, :] * rel_prod[:, :, None, :, :])
        scores = scores + (
            rel_diff_hidden * width_cell.rel_diff_score.reshape(1, 1, 1, 1, width_dim)
        ).sum(dim=-1) / math.sqrt(float(width_dim))
        scores = scores + (
            rel_prod_hidden * width_cell.rel_prod_score.reshape(1, 1, 1, 1, width_dim)
        ).sum(dim=-1) / math.sqrt(float(width_dim))
        scores = scores + width_cell.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        scores = scores + width_cell.source_pair_bias[cand_sources[:, :, :, None], cand_sources[:, :, None, :]]
        scores.diagonal(dim1=-2, dim2=-1).add_(width_cell.self_bias)
        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1).masked_fill(~valid_pair, 0.0)

        mode = os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_BIAS_MODE", "inbound").lower()
        if mode in {"outbound", "target"}:
            raw_bias = probs.mean(dim=-1)
        elif mode in {"symmetric", "sym"}:
            raw_bias = 0.5 * (probs.mean(dim=-1) + probs.mean(dim=-2))
        else:
            raw_bias = probs.mean(dim=-2)
        valid_denom = cand_mask.to(raw_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = raw_bias - (raw_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
        scale = float(os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_BIAS_SCALE", "1.0"))
        bias = (centered * scale).masked_fill(~cand_mask, 0.0).to(dtype=x.dtype)
        detach_bias = os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_DETACH", "0") == "1"
        if detach_bias:
            bias = bias.detach()

        p_safe = probs.clamp_min(1e-8)
        valid_targets = cand_mask.bool()
        entropy = (-(p_safe * p_safe.log()).sum(dim=-1)).masked_select(valid_targets).mean()
        diag = torch.eye(j_count, device=x.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        self_mass = probs.masked_fill(~diag, 0.0).sum(dim=-1).masked_select(valid_targets).mean()
        aux_loss = width_pair_transfer_loss(
            probs,
            cand_types,
            cand_mask,
            entropy_floor=width_cell.entropy_floor,
            entropy_weight=width_cell.entropy_weight,
        )
        telemetry = {
            "dsqg_w_projected_width_control": x.new_tensor(1.0).detach(),
            "dsqg_w_projected_width_bias_detached": x.new_tensor(1.0 if detach_bias else 0.0).detach(),
            "dsqg_w_projected_width_bias_scale": x.new_tensor(scale).detach(),
            "dsqg_w_projected_width_bias_norm": bias.masked_select(cand_mask).norm().detach() / cand_mask.float().sum().clamp_min(1.0),
            "dsqg_w_width_entropy": entropy.detach(),
            "dsqg_w_width_self_mass": self_mass.detach(),
            "dsqg_w_width_gate_mean": torch.sigmoid(width_cell.gate).mean().detach(),
            "dsqg_w_width_gate_logit_mean": width_cell.gate.detach().mean(),
            "dsqg_w_width_aux_loss": aux_loss,
            "dsqg_w_width_aux_loss_value": aux_loss.detach(),
            "dsqg_w_width_transfer_aux_loss": aux_loss.detach(),
            "dsqg_w_width_rel_diff_score_norm": width_cell.rel_diff_score.detach().norm(),
            "dsqg_w_width_rel_prod_score_norm": width_cell.rel_prod_score.detach().norm(),
        }
        return bias, telemetry

    def _source_projection_cache(
        self,
        x: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        needed_source_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        final_states = x
        l3_base = l3_states if l3_states is not None else final_states
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        zero_base = torch.zeros_like(final_states)
        bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
            int(CandidateSource.NULL): zero_base,
        }
        projected_by_object: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        needed = None if needed_source_ids is None else set(int(source_id) for source_id in needed_source_ids)
        for source_id, states in bases.items():
            if needed is not None and int(source_id) not in needed:
                continue
            cache_key = id(states)
            projected = projected_by_object.get(cache_key)
            if projected is None:
                states_n = self.norm_c(states)
                k_src = self.k_proj(states_n).reshape(*states.shape[:2], self.n_heads, self.dh)
                v_src = self.v_proj(states_n).reshape(*states.shape[:2], self.n_heads, self.dh)
                projected = (k_src, v_src)
                projected_by_object[cache_key] = projected
            out[source_id] = projected
        return out

    def _forward_sourcewise_triton(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        evidence_bits: torch.Tensor | None = None,
        evidence_count: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        return_routing: bool = False,
        needed_source_ids: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not _TRITON_SOURCEWISE_AVAILABLE or triton is None:
            raise NotImplementedError("DWARF_DSQG_W_TRITON_SOURCEWISE=1 requires Triton")
        if not x.is_cuda:
            raise NotImplementedError("DWARF_DSQG_W_TRITON_SOURCEWISE=1 requires CUDA tensors")
        triton_params = (
            self.norm_x.weight,
            self.norm_x.bias,
            self.norm_c.weight,
            self.norm_c.bias,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.role_key.weight,
            self.source_key.weight,
            self.type_bias,
            self.query_type_bias.weight,
            self.source_bias,
            self.read_mix.weight,
            self.norm_z.weight,
            self.norm_z.bias,
            self.fuse[0].weight,
            self.fuse[0].bias,
            self.fuse[2].weight,
            self.fuse[2].bias,
            self.gate,
        )
        needs_backward = torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (x, l3_states, chunk_rep_states, *triton_params)
            if tensor is not None
        )
        bsz, seq_len, d = x.shape
        j_count = cand_mask.shape[-1]
        h = self.n_heads
        dh = self.dh
        schedule = _dsqg_w_triton_schedule(dh, x.device)
        block_hd = schedule.block_hd
        if block_hd > 128:
            raise NotImplementedError("Triton DSQG-W sourcewise prototype supports head_dim <= 128")

        with _dsqg_w_profile_range("q_projection"):
            x_n = self.norm_x(x)
            q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh).contiguous()
        if needed_source_ids is None:
            needed_source_ids = tuple(
                int(source)
                for source in CandidateSource
                if bool(((cand_sources == int(source)) & cand_mask).any())
            )
        needed_with_final = tuple(sorted(set(int(s) for s in needed_source_ids) | {int(CandidateSource.FINAL)}))
        with _dsqg_w_profile_range("source_projection_cache"):
            projected_sources = self._source_projection_cache(
                x,
                l3_states=l3_states,
                chunk_rep_states=chunk_rep_states,
                needed_source_ids=needed_with_final,
            )
        k_final, v_final = projected_sources[int(CandidateSource.FINAL)]
        k_l3, v_l3 = projected_sources.get(
            int(CandidateSource.L3),
            projected_sources.get(int(CandidateSource.HISA), (k_final, v_final)),
        )
        k_summary, v_summary = projected_sources.get(int(CandidateSource.SUMMARY), (k_final, v_final))
        k_final = k_final.contiguous()
        v_final = v_final.contiguous()
        k_l3 = k_l3.contiguous()
        v_l3 = v_l3.contiguous()
        k_summary = k_summary.contiguous()
        v_summary = v_summary.contiguous()

        empty = torch.empty((0,), device=x.device, dtype=x.dtype)
        cand_scores, prior_telemetry = self._compose_evidence_prior_scores(
            cand_scores,
            cand_types,
            cand_sources,
            cand_mask,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
        )
        score_bias = empty
        candidate_score_bias_norm = x.new_tensor(0.0)
        use_score_bias = cand_scores is not None
        if cand_scores is not None:
            if cand_scores.shape != cand_mask.shape:
                raise ValueError("cand_scores must have shape [B,T,J]")
            score_bias = cand_scores.to(device=x.device, dtype=x.dtype)
            score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
            valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
            score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
            score_bias = score_bias.masked_fill(~cand_mask, 0.0).contiguous()
            candidate_score_bias_norm = score_bias.masked_select(cand_mask).norm() / cand_mask.float().sum().clamp_min(1.0)
        qtb = empty
        query_type_bias_norm = x.new_tensor(0.0)
        use_qtb = bool(self.use_query_type_bias)
        if self.use_query_type_bias:
            qtb = self.query_type_bias(x_n).reshape(bsz, seq_len, self.n_types, h).contiguous()
            query_type_bias_norm = qtb.norm(dim=-1).mean()

        read_slot_count = 1 + len(self.read_type_ids)
        read_slot_block = triton.next_power_of_2(read_slot_count)
        type_slot_map = torch.full((self.n_types,), -1, device=x.device, dtype=torch.int32)
        for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
            if 0 <= int(type_id) < self.n_types:
                type_slot_map[int(type_id)] = int(slot_idx)

        probs = torch.empty((bsz, seq_len, j_count, h), device=x.device, dtype=x.dtype) if return_routing else empty
        if needs_backward:
            read_slots = _DSQGWSourcewiseTritonCompactRead.apply(
                q,
                k_final,
                v_final,
                k_l3,
                v_l3,
                k_summary,
                v_summary,
                self.role_key.weight,
                self.source_key.weight,
                self.type_bias,
                self.source_bias,
                qtb,
                score_bias,
                cand_token_indices,
                cand_types,
                cand_sources,
                cand_mask,
                type_slot_map,
                use_qtb,
                use_score_bias,
                d,
                h,
                dh,
                self.n_types,
                read_slot_count,
                block_hd,
            )
            if return_routing:
                with torch.no_grad():
                    _dsqg_w_sourcewise_read_slots_kernel[(bsz * seq_len * h,)](
                        q.detach().contiguous(),
                        k_final.detach().contiguous(),
                        v_final.detach().contiguous(),
                        k_l3.detach().contiguous(),
                        v_l3.detach().contiguous(),
                        k_summary.detach().contiguous(),
                        v_summary.detach().contiguous(),
                        self.role_key.weight.detach().contiguous(),
                        self.source_key.weight.detach().contiguous(),
                        self.type_bias.detach().contiguous(),
                        self.source_bias.detach().contiguous(),
                        qtb.detach().contiguous() if use_qtb else empty,
                        score_bias.detach().contiguous() if use_score_bias else empty,
                        cand_token_indices.contiguous(),
                        cand_types.contiguous(),
                        cand_sources.contiguous(),
                        cand_mask.contiguous(),
                        type_slot_map.contiguous(),
                        torch.empty_like(read_slots),
                        empty,
                        probs,
                        B=bsz,
                        N=seq_len,
                        H=h,
                        HD=dh,
                        D=d,
                        J=j_count,
                        N_TYPES=self.n_types,
                        READ_SLOTS=read_slot_count,
                        MAX_READ_SLOTS=read_slot_block,
                        BLOCK_HD=block_hd,
                        USE_QTB=use_qtb,
                        USE_SCORE_BIAS=use_score_bias,
                        STORE_LSE=False,
                        STORE_PROBS=True,
                        num_warps=schedule.num_warps,
                        num_stages=schedule.num_stages,
                    )
        else:
            read_slots = torch.empty((bsz, seq_len, read_slot_count, d), device=x.device, dtype=x.dtype)
            _dsqg_w_sourcewise_read_slots_kernel[(bsz * seq_len * h,)](
                q,
                k_final,
                v_final,
                k_l3,
                v_l3,
                k_summary,
                v_summary,
                self.role_key.weight.contiguous(),
                self.source_key.weight.contiguous(),
                self.type_bias.contiguous(),
                self.source_bias.contiguous(),
                qtb,
                score_bias,
                cand_token_indices.contiguous(),
                cand_types.contiguous(),
                cand_sources.contiguous(),
                cand_mask.contiguous(),
                type_slot_map.contiguous(),
                read_slots,
                empty,
                probs,
                B=bsz,
                N=seq_len,
                H=h,
                HD=dh,
                D=d,
                J=j_count,
                N_TYPES=self.n_types,
                READ_SLOTS=read_slot_count,
                MAX_READ_SLOTS=read_slot_block,
                BLOCK_HD=block_hd,
                USE_QTB=use_qtb,
                USE_SCORE_BIAS=use_score_bias,
                STORE_LSE=False,
                STORE_PROBS=return_routing,
                num_warps=schedule.num_warps,
                num_stages=schedule.num_stages,
            )
        batched_read_mix = os.getenv("DWARF_DSQG_W_BATCHED_READ_MIX", "0") == "1"
        with _dsqg_w_profile_range("read_mix"):
            read = self._mix_compact_read_slots(read_slots, batched=batched_read_mix)
        with _dsqg_w_profile_range("fuse_norm_mlp_gate"):
            z = torch.cat([x, read, x * read, read - x], dim=-1)
            delta = self.fuse(self.norm_z(z))
            gate = torch.sigmoid(self.gate).reshape(1, 1, d)
            x_out = x + gate * delta

        if return_routing:
            p_mean = probs.mean(dim=-1)
            p_safe = p_mean.clamp_min(1e-8)
            entropy = -(p_safe * p_safe.log()).sum(dim=-1).mean()
        else:
            p_mean = None
            entropy = x.new_tensor(0.0)
        valid_counts = cand_mask.sum(dim=-1).float()
        delta_norm = delta.norm(dim=-1).mean()
        x_norm = x.norm(dim=-1).mean()
        read_norm = read.norm(dim=-1).mean()
        typed_read_norms = [x.new_tensor(0.0) for _ in range(self.n_types)]
        if return_routing:
            for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
                if 0 <= int(type_id) < self.n_types:
                    typed_read_norms[int(type_id)] = read_slots[:, :, slot_idx, :].norm(dim=-1).mean()
        true_backward = needs_backward and os.getenv("DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD", "triton").lower() != "pytorch"
        split_backward = true_backward and os.getenv(
            "DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION", "monolithic"
        ).lower() in {"1", "true", "split", "v20_split"}
        source_backward_grads = os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRADS", "1") != "0"
        source_backward_grad_every = max(1, int(os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRAD_EVERY", "1")))

        telemetry: dict[str, torch.Tensor] = {
            "dsqg_w_entropy": entropy.detach(),
            "dsqg_w_valid_candidate_count": valid_counts.mean().detach(),
            "dsqg_w_gate_mean": gate.mean().detach(),
            "dsqg_w_gate_min": gate.min().detach(),
            "dsqg_w_gate_max": gate.max().detach(),
            "dsqg_w_gate_logit_mean": self.gate.detach().mean(),
            "dsqg_w_delta_norm": delta_norm.detach(),
            "dsqg_w_x_norm": x_norm.detach(),
            "dsqg_w_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
            "dsqg_w_read_norm": read_norm.detach(),
            "dsqg_w_typed_read_norms": torch.stack(typed_read_norms).detach(),
            "read_mix_weight_norm": self.read_mix.weight.norm().detach(),
            "dsqg_w_query_type_bias_norm": query_type_bias_norm.detach(),
            "dsqg_w_candidate_score_bias_norm": candidate_score_bias_norm.detach(),
            "dsqg_w_sourcewise": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_sourcewise": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_sourcewise_recompute_backward": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_compact_read_backward": x.new_tensor(1.0 if needs_backward else 0.0).detach(),
            "dsqg_w_triton_probs_materialized": x.new_tensor(1.0 if return_routing else 0.0).detach(),
            "dsqg_w_triton_read_accum_materialized": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_read_mix_fused": x.new_tensor(0.0).detach(),
            "dsqg_w_batched_read_mix": x.new_tensor(1.0 if batched_read_mix else 0.0).detach(),
            "dsqg_w_triton_compact_read_slots_materialized": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_compact_read_slots": x.new_tensor(float(read_slot_count)).detach(),
            "dsqg_w_triton_score_recompute_blocks": x.new_tensor(2.0 if split_backward else 1.0).detach(),
            "dsqg_w_triton_true_backward": x.new_tensor(1.0 if true_backward else 0.0).detach(),
            "dsqg_w_triton_backward_v20_split_kernels": x.new_tensor(1.0 if split_backward else 0.0).detach(),
            "dsqg_w_triton_backward_monolithic_kernel": x.new_tensor(1.0 if true_backward and not split_backward and source_backward_grads else 0.0).detach(),
            "dsqg_w_triton_backward_query_kernel": x.new_tensor(1.0 if split_backward or not source_backward_grads else 0.0).detach(),
            "dsqg_w_triton_backward_source_kernel": x.new_tensor(1.0 if split_backward and source_backward_grads else 0.0).detach(),
            "dsqg_w_triton_backward_source_grads": x.new_tensor(1.0 if source_backward_grads else 0.0).detach(),
            "dsqg_w_triton_backward_source_grad_every": x.new_tensor(float(source_backward_grad_every)).detach(),
            "dsqg_w_triton_backward_probs_materialized": x.new_tensor(0.0 if true_backward else (1.0 if needs_backward else 0.0)).detach(),
            "dsqg_w_triton_backward_lse_saved": x.new_tensor(1.0 if needs_backward else 0.0).detach(),
            "dsqg_w_triton_backward_reduction_buffer_bytes": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_schedule_block_hd": x.new_tensor(float(schedule.block_hd)).detach(),
            "dsqg_w_triton_schedule_num_warps": x.new_tensor(float(schedule.num_warps)).detach(),
            "dsqg_w_triton_schedule_num_stages": x.new_tensor(float(schedule.num_stages)).detach(),
            "dsqg_w_static_source_count": x.new_tensor(float(len(needed_source_ids or ()))).detach(),
            "dsqg_w_static_source_set_used": x.new_tensor(1.0).detach(),
        }
        if return_routing:
            for ctype in CandidateType:
                mask = (cand_types == int(ctype)) & cand_mask
                mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
                telemetry[f"dsqg_w_{ctype.name.lower()}_mass"] = mass.detach()
            telemetry["dsqg_w_local_mass"] = telemetry[f"dsqg_w_{CandidateType.LOCAL.name.lower()}_mass"]
            telemetry["dsqg_w_question_mass"] = telemetry[f"dsqg_w_{CandidateType.QUESTION.name.lower()}_mass"]
            telemetry["dsqg_w_hisa_evidence_mass"] = telemetry[f"dsqg_w_{CandidateType.HISA_EVIDENCE.name.lower()}_mass"]
            telemetry["dsqg_w_long_offset_mass"] = telemetry[f"dsqg_w_{CandidateType.LONG_OFFSET.name.lower()}_mass"]
            telemetry["dsqg_w_chunk_rep_mass"] = telemetry[f"dsqg_w_{CandidateType.CHUNK_REP.name.lower()}_mass"]
            telemetry["dsqg_w_null_mass"] = telemetry[f"dsqg_w_{CandidateType.NULL.name.lower()}_mass"]
            for source in CandidateSource:
                mask = (cand_sources == int(source)) & cand_mask
                mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
                telemetry[f"dsqg_w_{source.name.lower()}_source_mass"] = mass.detach()
            telemetry["dsqg_w_l3_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.L3.name.lower()}_source_mass"]
            telemetry["dsqg_w_final_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.FINAL.name.lower()}_source_mass"]
        telemetry.update(prior_telemetry)
        if return_routing:
            telemetry["dsqg_w_probs"] = probs
        return x_out, telemetry

    def forward_sourcewise(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        evidence_bits: torch.Tensor | None = None,
        evidence_count: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        return_routing: bool = False,
        needed_source_ids: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, d = x.shape
        if d != self.d:
            raise ValueError(f"x last dim {d} does not match block d {self.d}")
        if cand_types.shape != cand_mask.shape or cand_sources.shape != cand_mask.shape:
            raise ValueError("candidate type/source/mask tensors must have shape [B,T,J]")
        if cand_token_indices.shape != cand_mask.shape:
            raise ValueError("candidate token indices must have shape [B,T,J]")
        if cand_mask.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate metadata shape mismatch")
        if not cand_mask.any(dim=-1).all():
            raise ValueError("DSQG-W received an all-invalid candidate row")
        j_count = cand_mask.shape[-1]
        if j_count > self.max_candidates:
            raise ValueError("candidate count exceeds DSQG-W max_candidates")
        if l3_states is not None and l3_states.shape != x.shape:
            raise ValueError("l3_states must match x shape")
        if chunk_rep_states is not None and chunk_rep_states.shape != x.shape:
            raise ValueError("chunk_rep_states must match x shape")
        ebh_sourcewise_packet = (
            self.evidence_binding_hub is not None
            and os.getenv("DWARF_DSQG_W_EBH_SOURCEWISE_PACKET", "0") == "1"
        )
        if ebh_sourcewise_packet:
            assert self.evidence_binding_hub is not None
            semantic_approx = self.width_cell is not None or self.typed_mixer is not None
            with _dsqg_w_profile_range("ebh_sourcewise_packet"):
                x_ebh, ebh_telemetry = self.evidence_binding_hub.forward_sourcewise_packet(
                    x,
                    cand_token_indices,
                    cand_types,
                    cand_sources,
                    cand_mask,
                    l3_states=l3_states,
                    chunk_rep_states=chunk_rep_states,
                    cand_scores=cand_scores,
                    candidate_distances=candidate_distances,
                )
            saved_ebh = self.evidence_binding_hub
            self.evidence_binding_hub = None
            try:
                x_out, telemetry = self.forward_sourcewise(
                    x_ebh,
                    cand_token_indices,
                    cand_types,
                    cand_sources,
                    cand_mask,
                    l3_states=l3_states,
                    chunk_rep_states=chunk_rep_states,
                    cand_scores=cand_scores,
                    evidence_bits=evidence_bits,
                    evidence_count=evidence_count,
                    candidate_distances=candidate_distances,
                    return_routing=return_routing,
                    needed_source_ids=needed_source_ids,
                )
            finally:
                self.evidence_binding_hub = saved_ebh
            telemetry.update(ebh_telemetry)
            telemetry["dsqg_w_sourcewise_ebh_materialized"] = x.new_tensor(0.0).detach()
            telemetry["dsqg_w_ebh_packet_sourcewise"] = x.new_tensor(1.0).detach()
            telemetry["dsqg_w_ebh_packet_triton"] = ebh_telemetry.get(
                "dsqg_w_ebh_packet_triton", x.new_tensor(0.0).detach()
            )
            telemetry["dsqg_w_ebh_packet_semantic_approx"] = x.new_tensor(1.0 if semantic_approx else 0.0).detach()
            return x_out, telemetry
        if self.width_cell is not None or self.typed_mixer is not None or self.evidence_binding_hub is not None:
            semantic_telemetry: dict[str, torch.Tensor] = {}
            projected_width_control = (
                self.width_cell is not None
                and os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_CONTROL", "0") == "1"
                and os.getenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "0") == "1"
                and x.is_cuda
                and _TRITON_SOURCEWISE_AVAILABLE
                and triton is not None
            )
            if projected_width_control:
                with _dsqg_w_profile_range("projected_width_score_bias"):
                    if os.getenv("DWARF_DSQG_W_PROJECTED_WIDTH_DETACH", "0") == "1":
                        with torch.no_grad():
                            width_bias, semantic_telemetry = self._sourcewise_projected_width_score_bias(
                                x.detach(),
                                cand_token_indices,
                                cand_types,
                                cand_sources,
                                cand_mask,
                                l3_states=l3_states.detach() if l3_states is not None else None,
                                chunk_rep_states=chunk_rep_states.detach() if chunk_rep_states is not None else None,
                            )
                    else:
                        width_bias, semantic_telemetry = self._sourcewise_projected_width_score_bias(
                            x,
                            cand_token_indices,
                            cand_types,
                            cand_sources,
                            cand_mask,
                            l3_states=l3_states,
                            chunk_rep_states=chunk_rep_states,
                        )
                cand_scores = width_bias if cand_scores is None else cand_scores.to(width_bias.dtype) + width_bias
                x_out, telemetry = self._forward_sourcewise_triton(
                    x,
                    cand_token_indices,
                    cand_types,
                    cand_sources,
                    cand_mask,
                    l3_states=l3_states,
                    chunk_rep_states=chunk_rep_states,
                    cand_scores=cand_scores,
                    evidence_bits=evidence_bits,
                    evidence_count=evidence_count,
                    candidate_distances=candidate_distances,
                    return_routing=return_routing,
                    needed_source_ids=needed_source_ids,
                )
                telemetry.update(semantic_telemetry)
                telemetry["dsqg_w_sourcewise_semantic_materialized"] = x.new_tensor(0.0).detach()
                telemetry["dsqg_w_sourcewise_width_cell_fusion"] = x.new_tensor(0.0).detach()
                telemetry["dsqg_w_projected_width_semantic_control"] = x.new_tensor(1.0).detach()
                telemetry["dsqg_w_typed_mixer_projected_bypass"] = x.new_tensor(1.0 if self.typed_mixer is not None else 0.0).detach()
                telemetry["dsqg_w_triton_sourcewise_semantic_bypass"] = x.new_tensor(1.0).detach()
                return x_out, telemetry
            sourcewise_width_fused = (
                self.width_cell is not None
                and os.getenv("DWARF_DSQG_W_SOURCEWISE_WIDTH_CELL_FUSION", "0") == "1"
            )
            if sourcewise_width_fused:
                width_recompute = (
                    os.getenv("DWARF_DSQG_W_SOURCEWISE_WIDTH_RECOMPUTE", "1") != "0"
                    and torch.is_grad_enabled()
                    and x.requires_grad
                    and os.getenv("DWARF_DSQG_W_WIDTH_AUX_WEIGHT", "0") in {"", "0", "0.0"}
                )
                if width_recompute:
                    if l3_states is not None:
                        def _sourcewise_width_out(x_arg, l3_arg):
                            with _dsqg_w_profile_range("sourcewise_width_cell_fused"):
                                out, _ = self._sourcewise_width_cell_fused(
                                    x_arg,
                                    cand_token_indices,
                                    cand_types,
                                    cand_sources,
                                    cand_mask,
                                    l3_states=l3_arg,
                                    chunk_rep_states=chunk_rep_states,
                                )
                            return out

                        cand_states = checkpoint(_sourcewise_width_out, x, l3_states, use_reentrant=False)
                    else:
                        def _sourcewise_width_out_no_l3(x_arg):
                            with _dsqg_w_profile_range("sourcewise_width_cell_fused"):
                                out, _ = self._sourcewise_width_cell_fused(
                                    x_arg,
                                    cand_token_indices,
                                    cand_types,
                                    cand_sources,
                                    cand_mask,
                                    l3_states=None,
                                    chunk_rep_states=chunk_rep_states,
                                )
                            return out

                        cand_states = checkpoint(_sourcewise_width_out_no_l3, x, use_reentrant=False)
                    if os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") == "1":
                        zero = x.new_tensor(0.0)
                        gate = torch.sigmoid(self.width_cell.gate).reshape(-1)
                        semantic_telemetry = {
                            "dsqg_w_width_entropy": zero.detach(),
                            "dsqg_w_width_self_mass": zero.detach(),
                            "dsqg_w_width_gate_mean": gate.mean().detach(),
                            "dsqg_w_width_gate_min": gate.min().detach(),
                            "dsqg_w_width_gate_max": gate.max().detach(),
                            "dsqg_w_width_gate_logit_mean": self.width_cell.gate.detach().mean(),
                            "dsqg_w_width_delta_norm": zero.detach(),
                            "dsqg_w_width_aux_loss_value": zero.detach(),
                            "dsqg_w_width_transfer_aux_loss": zero.detach(),
                            "dsqg_w_width_entropy_penalty": zero.detach(),
                            "dsqg_w_width_entropy_floor": zero.detach(),
                            "dsqg_w_width_entropy_weight": zero.detach(),
                            "dsqg_w_width_question_to_hisa_evidence_mass": zero.detach(),
                            "dsqg_w_width_hisa_evidence_to_question_mass": zero.detach(),
                            "dsqg_w_width_rel_diff_score_norm": self.width_cell.rel_diff_score.detach().norm(),
                            "dsqg_w_width_rel_prod_score_norm": self.width_cell.rel_prod_score.detach().norm(),
                            "dsqg_w_width_recompute_checkpoint": zero.detach(),
                            "dsqg_w_sourcewise_width_cell_fused": x.new_tensor(1.0).detach(),
                            "dsqg_w_sourcewise_width_fast_telemetry": x.new_tensor(1.0).detach(),
                        }
                    else:
                        with torch.no_grad():
                            with _dsqg_w_profile_range("sourcewise_width_cell_fused_telemetry"):
                                _, semantic_telemetry = self._sourcewise_width_cell_fused(
                                    x.detach(),
                                    cand_token_indices,
                                    cand_types,
                                    cand_sources,
                                    cand_mask,
                                    l3_states=l3_states.detach() if l3_states is not None else None,
                                    chunk_rep_states=chunk_rep_states.detach() if chunk_rep_states is not None else None,
                                )
                    semantic_telemetry["dsqg_w_sourcewise_width_recompute_checkpoint"] = x.new_tensor(1.0).detach()
                else:
                    with _dsqg_w_profile_range("sourcewise_width_cell_fused"):
                        cand_states, semantic_telemetry = self._sourcewise_width_cell_fused(
                            x,
                            cand_token_indices,
                            cand_types,
                            cand_sources,
                            cand_mask,
                            l3_states=l3_states,
                            chunk_rep_states=chunk_rep_states,
                        )
                    semantic_telemetry["dsqg_w_sourcewise_width_recompute_checkpoint"] = x.new_tensor(0.0).detach()
                if self.typed_mixer is not None:
                    cand_states, typed_mixer_telemetry = self.typed_mixer(cand_states, cand_types, cand_mask)
                    semantic_telemetry.update(typed_mixer_telemetry)
            else:
                cand_states = self._materialize_sourcewise_candidate_states(
                    x,
                    cand_token_indices,
                    cand_sources,
                    cand_mask,
                    l3_states=l3_states,
                    chunk_rep_states=chunk_rep_states,
                )
            x_out, telemetry = self.forward(
                x,
                cand_states,
                cand_types,
                cand_sources,
                cand_mask,
                cand_scores=cand_scores,
                evidence_bits=evidence_bits,
                evidence_count=evidence_count,
                candidate_distances=candidate_distances,
                return_routing=return_routing,
                semantic_transforms_applied=sourcewise_width_fused,
                precomputed_semantic_telemetry=semantic_telemetry if sourcewise_width_fused else None,
            )
            telemetry["dsqg_w_sourcewise_semantic_materialized"] = x.new_tensor(0.0 if sourcewise_width_fused else 1.0).detach()
            telemetry["dsqg_w_sourcewise_width_cell_fusion"] = x.new_tensor(1.0 if sourcewise_width_fused else 0.0).detach()
            telemetry["dsqg_w_sourcewise_ebh_materialized"] = x.new_tensor(
                1.0 if self.evidence_binding_hub is not None and not sourcewise_width_fused else 0.0
            ).detach()
            telemetry["dsqg_w_triton_sourcewise_semantic_bypass"] = x.new_tensor(0.0).detach()
            return x_out, telemetry
        if os.getenv("DWARF_DSQG_W_TRITON_SOURCEWISE", "0") == "1":
            return self._forward_sourcewise_triton(
                x,
                cand_token_indices,
                cand_types,
                cand_sources,
                cand_mask,
                l3_states=l3_states,
                chunk_rep_states=chunk_rep_states,
                cand_scores=cand_scores,
                evidence_bits=evidence_bits,
                evidence_count=evidence_count,
                candidate_distances=candidate_distances,
                return_routing=return_routing,
                needed_source_ids=needed_source_ids,
            )

        h = self.n_heads
        dh = self.dh
        x_n = self.norm_x(x)
        q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh)
        if needed_source_ids is None:
            needed_source_ids = tuple(
                int(source)
                for source in CandidateSource
                if bool(((cand_sources == int(source)) & cand_mask).any())
            )
        projected_sources = self._source_projection_cache(
            x,
            l3_states=l3_states,
            chunk_rep_states=chunk_rep_states,
            needed_source_ids=needed_source_ids,
        )
        gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
        cand_scores, prior_telemetry = self._compose_evidence_prior_scores(
            cand_scores,
            cand_types,
            cand_sources,
            cand_mask,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
        )

        score_bias = None
        candidate_score_bias_norm = x.new_tensor(0.0)
        if cand_scores is not None:
            if cand_scores.shape != cand_mask.shape:
                raise ValueError("cand_scores must have shape [B,T,J]")
            score_bias = cand_scores.to(device=x.device, dtype=x.dtype)
            score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
            valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
            score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
            score_bias = score_bias.masked_fill(~cand_mask, 0.0)
            candidate_score_bias_norm = score_bias.masked_select(cand_mask).norm() / cand_mask.float().sum().clamp_min(1.0)
        qtb = None
        query_type_bias_norm = x.new_tensor(0.0)
        if self.use_query_type_bias:
            qtb = self.query_type_bias(x_n).reshape(bsz, seq_len, self.n_types, h)
            query_type_bias_norm = qtb.norm(dim=-1).mean()

        score_parts: list[torch.Tensor] = []
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            k_j = x.new_zeros((bsz, seq_len, h, dh))
            for source_id, (k_src, _) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = self._gather_source_rows(k_src, token_j)
                    k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
            role = self.role_key(cand_types[:, :, j]).reshape(bsz, seq_len, h, dh)
            source = self.source_key(source_j).reshape(bsz, seq_len, h, dh)
            score_j = (q * (k_j + role + source)).sum(dim=-1) / math.sqrt(float(dh))
            score_j = score_j + self.type_bias[cand_types[:, :, j]]
            if score_bias is not None:
                score_j = score_j + score_bias[:, :, j, None]
            if qtb is not None:
                score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, h)).squeeze(2)
            score_j = score_j + self.source_bias[source_j]
            score_j = score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min)
            score_parts.append(score_j)
        scores = torch.stack(score_parts, dim=2)
        probs = F.softmax(scores, dim=2)

        r_all_h = x.new_zeros((bsz, seq_len, h, dh))
        typed_reads_h = {
            type_id: x.new_zeros((bsz, seq_len, h, dh))
            for type_id in self.read_type_ids
            if 0 <= type_id < self.n_types
        }
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            v_j = x.new_zeros((bsz, seq_len, h, dh))
            for source_id, (_, v_src) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = self._gather_source_rows(v_src, token_j)
                    v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
            contrib = probs[:, :, j, :, None] * v_j
            r_all_h = r_all_h + contrib
            for type_id in typed_reads_h:
                type_mask = ((cand_types[:, :, j] == int(type_id)) & cand_mask[:, :, j])[:, :, None, None]
                typed_reads_h[type_id] = typed_reads_h[type_id] + contrib * type_mask.to(contrib.dtype)

        r_all = r_all_h.reshape(bsz, seq_len, d)
        weight = self.read_mix.weight
        read = F.linear(r_all, weight[:, : self.d])
        typed_read_norms = [r_all.new_tensor(0.0) for _ in range(self.n_types)]
        for type_id, r_type_h in typed_reads_h.items():
            r_type = r_type_h.reshape(bsz, seq_len, d)
            start = (int(type_id) + 1) * self.d
            read = read + F.linear(r_type, weight[:, start : start + self.d])
            typed_read_norms[int(type_id)] = r_type.norm(dim=-1).mean()

        z = torch.cat([x, read, x * read, read - x], dim=-1)
        delta = self.fuse(self.norm_z(z))
        gate = torch.sigmoid(self.gate).reshape(1, 1, d)
        x_out = x + gate * delta

        p_mean = probs.mean(dim=-1)
        p_safe = p_mean.clamp_min(1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1).mean()
        valid_counts = cand_mask.sum(dim=-1).float()
        delta_norm = delta.norm(dim=-1).mean()
        x_norm = x.norm(dim=-1).mean()
        read_norm = read.norm(dim=-1).mean()
        telemetry: dict[str, torch.Tensor] = {
            "dsqg_w_entropy": entropy.detach(),
            "dsqg_w_valid_candidate_count": valid_counts.mean().detach(),
            "dsqg_w_gate_mean": gate.mean().detach(),
            "dsqg_w_gate_min": gate.min().detach(),
            "dsqg_w_gate_max": gate.max().detach(),
            "dsqg_w_gate_logit_mean": self.gate.detach().mean(),
            "dsqg_w_delta_norm": delta_norm.detach(),
            "dsqg_w_x_norm": x_norm.detach(),
            "dsqg_w_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
            "dsqg_w_read_norm": read_norm.detach(),
            "dsqg_w_typed_read_norms": torch.stack(typed_read_norms).detach(),
            "read_mix_weight_norm": self.read_mix.weight.norm().detach(),
            "dsqg_w_query_type_bias_norm": query_type_bias_norm.detach(),
            "dsqg_w_candidate_score_bias_norm": candidate_score_bias_norm.detach(),
            "dsqg_w_sourcewise": x.new_tensor(1.0).detach(),
        }
        for ctype in CandidateType:
            mask = (cand_types == int(ctype)) & cand_mask
            mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
            telemetry[f"dsqg_w_{ctype.name.lower()}_mass"] = mass.detach()
        telemetry["dsqg_w_local_mass"] = telemetry[f"dsqg_w_{CandidateType.LOCAL.name.lower()}_mass"]
        telemetry["dsqg_w_question_mass"] = telemetry[f"dsqg_w_{CandidateType.QUESTION.name.lower()}_mass"]
        telemetry["dsqg_w_hisa_evidence_mass"] = telemetry[f"dsqg_w_{CandidateType.HISA_EVIDENCE.name.lower()}_mass"]
        telemetry["dsqg_w_long_offset_mass"] = telemetry[f"dsqg_w_{CandidateType.LONG_OFFSET.name.lower()}_mass"]
        telemetry["dsqg_w_chunk_rep_mass"] = telemetry[f"dsqg_w_{CandidateType.CHUNK_REP.name.lower()}_mass"]
        telemetry["dsqg_w_null_mass"] = telemetry[f"dsqg_w_{CandidateType.NULL.name.lower()}_mass"]
        for source in CandidateSource:
            mask = (cand_sources == int(source)) & cand_mask
            mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
            telemetry[f"dsqg_w_{source.name.lower()}_source_mass"] = mass.detach()
        telemetry["dsqg_w_l3_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.L3.name.lower()}_source_mass"]
        telemetry["dsqg_w_final_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.FINAL.name.lower()}_source_mass"]
        telemetry.update(prior_telemetry)
        if return_routing:
            telemetry["dsqg_w_probs"] = probs
        return x_out, telemetry

    def forward(
        self,
        x: torch.Tensor,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        cand_scores: torch.Tensor | None = None,
        evidence_bits: torch.Tensor | None = None,
        evidence_count: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        return_routing: bool = False,
        semantic_transforms_applied: bool = False,
        precomputed_semantic_telemetry: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, d = x.shape
        b2, t2, j_count, d2 = cand_states.shape
        if (bsz, seq_len, d) != (b2, t2, d2):
            raise ValueError("x and cand_states shape mismatch")
        if d != self.d:
            raise ValueError(f"x last dim {d} does not match block d {self.d}")
        if j_count > self.max_candidates:
            raise ValueError("candidate count exceeds DSQG-W max_candidates")
        if not cand_mask.any(dim=-1).all():
            raise ValueError("DSQG-W received an all-invalid candidate row")

        width_telemetry: dict[str, torch.Tensor] = {}
        typed_mixer_telemetry: dict[str, torch.Tensor] = {}
        ebh_telemetry: dict[str, torch.Tensor] = {}
        if precomputed_semantic_telemetry is not None:
            width_telemetry.update(precomputed_semantic_telemetry)
        if self.width_cell is not None and not semantic_transforms_applied:
            width_recompute = (
                os.getenv("DWARF_DSQG_W_WIDTH_CELL_RECOMPUTE", "1") != "0"
                and torch.is_grad_enabled()
                and cand_states.requires_grad
                and os.getenv("DWARF_DSQG_W_WIDTH_AUX_WEIGHT", "0") in {"", "0", "0.0"}
            )
            if width_recompute:
                width_input = cand_states

                def _width_out(states, types, sources, mask):
                    out, _ = self.width_cell(states, types, sources, mask)
                    return out

                cand_states = checkpoint(
                    _width_out,
                    cand_states,
                    cand_types,
                    cand_sources,
                    cand_mask,
                    use_reentrant=False,
                )
                # Telemetry is not on the training loss path when aux weight is
                # zero.  Compute it without saving the full width-cell graph; this
                # keeps the Stage-3 all-open path from carrying both width-cell
                # activations and the downstream compact-read graph at N=2048.
                with torch.no_grad():
                    _, width_telemetry = self.width_cell(width_input.detach(), cand_types, cand_sources, cand_mask)
                width_telemetry["dsqg_w_width_recompute_checkpoint"] = cand_states.new_tensor(1.0).detach()
            else:
                cand_states, width_telemetry = self.width_cell(cand_states, cand_types, cand_sources, cand_mask)
                width_telemetry["dsqg_w_width_recompute_checkpoint"] = cand_states.new_tensor(0.0).detach()
        if self.typed_mixer is not None and not semantic_transforms_applied:
            cand_states, typed_mixer_telemetry = self.typed_mixer(cand_states, cand_types, cand_mask)
        if self.evidence_binding_hub is not None and not semantic_transforms_applied:
            x, ebh_telemetry = self.evidence_binding_hub(
                x,
                cand_states,
                cand_types,
                cand_sources,
                cand_mask,
                candidate_distances=candidate_distances,
                cand_scores=cand_scores,
            )

        h = self.n_heads
        dh = self.dh
        x_n = self.norm_x(x)
        c_n = self.norm_c(cand_states)

        q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh)
        k = self.k_proj(c_n).reshape(bsz, seq_len, j_count, h, dh)
        v = self.v_proj(c_n).reshape(bsz, seq_len, j_count, h, dh)

        if (
            os.getenv("DWARF_DSQG_W_TRITON_TRANSFORMED_COMPACT_READ", "0") == "1"
            and _TRITON_SOURCEWISE_AVAILABLE
            and triton is not None
            and x.is_cuda
            and not return_routing
        ):
            schedule = _dsqg_w_triton_schedule(dh, x.device)
            read_slot_count = 1 + len(self.read_type_ids)
            type_slot_map = torch.full((self.n_types,), -1, device=x.device, dtype=torch.int32)
            for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
                if 0 <= int(type_id) < self.n_types:
                    type_slot_map[int(type_id)] = int(slot_idx)

            cand_scores, prior_telemetry = self._compose_evidence_prior_scores(
                cand_scores,
                cand_types,
                cand_sources,
                cand_mask,
                evidence_bits=evidence_bits,
                evidence_count=evidence_count,
                candidate_distances=candidate_distances,
            )
            empty = torch.empty((0,), device=x.device, dtype=x.dtype)
            score_bias = empty
            candidate_score_bias_norm = x.new_tensor(0.0)
            use_score_bias = cand_scores is not None
            if cand_scores is not None:
                if cand_scores.shape != cand_mask.shape:
                    raise ValueError("cand_scores must have shape [B,T,J]")
                score_bias = cand_scores.to(device=x.device, dtype=x.dtype)
                score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
                valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
                score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
                score_bias = score_bias.masked_fill(~cand_mask, 0.0).contiguous()
                candidate_score_bias_norm = score_bias.masked_select(cand_mask).norm() / cand_mask.float().sum().clamp_min(1.0)
            qtb = empty
            query_type_bias_norm = x.new_tensor(0.0)
            use_qtb = bool(self.use_query_type_bias)
            if self.use_query_type_bias:
                qtb = self.query_type_bias(x_n).reshape(bsz, seq_len, self.n_types, h).contiguous()
                query_type_bias_norm = qtb.norm(dim=-1).mean()

            read_slots = _materialized_triton_compact_read().apply(
                q,
                k,
                v,
                self.role_key.weight,
                self.source_key.weight,
                self.type_bias,
                self.source_bias,
                qtb,
                score_bias,
                cand_types,
                cand_sources,
                cand_mask,
                type_slot_map,
                use_qtb,
                use_score_bias,
                d,
                h,
                dh,
                self.n_types,
                read_slot_count,
                schedule.block_hd,
            )
            read = self._mix_compact_read_slots(read_slots)
            z = torch.cat([x, read, x * read, read - x], dim=-1)
            delta = self.fuse(self.norm_z(z))
            gate = torch.sigmoid(self.gate).reshape(1, 1, d)
            x_out = x + gate * delta

            fast_telemetry = os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") == "1"
            valid_counts = cand_mask.sum(dim=-1).float()
            delta_norm = x.new_tensor(0.0) if fast_telemetry else delta.norm(dim=-1).mean()
            x_norm = x.norm(dim=-1).mean()
            read_norm = x.new_tensor(0.0) if fast_telemetry else read.norm(dim=-1).mean()
            typed_read_norms = [x.new_tensor(0.0) for _ in range(self.n_types)]
            if not fast_telemetry:
                for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
                    if 0 <= int(type_id) < self.n_types:
                        typed_read_norms[int(type_id)] = read_slots[:, :, slot_idx, :].norm(dim=-1).mean()
            zero_mass = x.new_tensor(0.0)
            telemetry: dict[str, torch.Tensor] = {
                "dsqg_w_entropy": zero_mass.detach(),
                "dsqg_w_valid_candidate_count": valid_counts.mean().detach(),
                "dsqg_w_gate_mean": gate.mean().detach(),
                "dsqg_w_gate_min": gate.min().detach(),
                "dsqg_w_gate_max": gate.max().detach(),
                "dsqg_w_gate_logit_mean": self.gate.detach().mean(),
                "dsqg_w_delta_norm": delta_norm.detach(),
                "dsqg_w_x_norm": x_norm.detach(),
                "dsqg_w_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
                "dsqg_w_read_norm": read_norm.detach(),
                "dsqg_w_typed_read_norms": torch.stack(typed_read_norms).detach(),
                "read_mix_weight_norm": self.read_mix.weight.norm().detach(),
                "dsqg_w_query_type_bias_norm": query_type_bias_norm.detach(),
                "dsqg_w_candidate_score_bias_norm": candidate_score_bias_norm.detach(),
                "dsqg_w_triton_transformed_compact_read": x.new_tensor(1.0).detach(),
                "dsqg_w_triton_transformed_compact_read_backward": x.new_tensor(1.0).detach(),
                "dsqg_w_triton_compact_read_slots_materialized": x.new_tensor(1.0).detach(),
            }
            for ctype in CandidateType:
                telemetry[f"dsqg_w_{ctype.name.lower()}_mass"] = zero_mass.detach()
            telemetry["dsqg_w_local_mass"] = telemetry[f"dsqg_w_{CandidateType.LOCAL.name.lower()}_mass"]
            telemetry["dsqg_w_question_mass"] = telemetry[f"dsqg_w_{CandidateType.QUESTION.name.lower()}_mass"]
            telemetry["dsqg_w_hisa_evidence_mass"] = telemetry[f"dsqg_w_{CandidateType.HISA_EVIDENCE.name.lower()}_mass"]
            telemetry["dsqg_w_long_offset_mass"] = telemetry[f"dsqg_w_{CandidateType.LONG_OFFSET.name.lower()}_mass"]
            telemetry["dsqg_w_chunk_rep_mass"] = telemetry[f"dsqg_w_{CandidateType.CHUNK_REP.name.lower()}_mass"]
            telemetry["dsqg_w_null_mass"] = telemetry[f"dsqg_w_{CandidateType.NULL.name.lower()}_mass"]
            for source in CandidateSource:
                telemetry[f"dsqg_w_{source.name.lower()}_source_mass"] = zero_mass.detach()
            telemetry["dsqg_w_l3_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.L3.name.lower()}_source_mass"]
            telemetry["dsqg_w_final_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.FINAL.name.lower()}_source_mass"]
            telemetry.update(prior_telemetry)
            telemetry.update(width_telemetry)
            telemetry.update(typed_mixer_telemetry)
            telemetry.update(ebh_telemetry)
            return x_out, telemetry

        # Avoid materializing role/source embeddings as [B,T,J,H,DH].  They are
        # candidate-type/source constants, so score them against q once for the
        # tiny type/source vocabularies and gather the resulting [B,T,J,H]
        # logits.  This removes two large embedding tensors and their expensive
        # scatter-style backward from the typed/width materialized path.
        inv_sqrt_dh = 1.0 / math.sqrt(float(dh))
        scores = torch.einsum("bthd,btjhd->btjh", q, k)
        role_table = self.role_key.weight.reshape(self.n_types, h, dh)
        source_table = self.source_key.weight.reshape(self.n_sources, h, dh)
        role_scores = torch.einsum("bthd,rhd->btrh", q, role_table)
        source_scores = torch.einsum("bthd,rhd->btrh", q, source_table)
        scores = scores + role_scores.gather(2, cand_types[:, :, :, None].expand(-1, -1, -1, h))
        scores = scores + source_scores.gather(2, cand_sources[:, :, :, None].expand(-1, -1, -1, h))
        scores = scores * inv_sqrt_dh
        scores = scores + self.type_bias[cand_types]
        cand_scores, prior_telemetry = self._compose_evidence_prior_scores(
            cand_scores,
            cand_types,
            cand_sources,
            cand_mask,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
        )
        candidate_score_bias_norm = x.new_tensor(0.0)
        if cand_scores is not None:
            if cand_scores.shape != cand_mask.shape:
                raise ValueError("cand_scores must have shape [B,T,J]")
            score_bias = cand_scores.to(device=x.device, dtype=scores.dtype)
            score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
            valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
            score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
            score_bias = score_bias.masked_fill(~cand_mask, 0.0)
            scores = scores + score_bias[:, :, :, None]
            candidate_score_bias_norm = score_bias.masked_select(cand_mask).norm() / cand_mask.float().sum().clamp_min(1.0)
        query_type_bias_norm = x.new_tensor(0.0)
        if self.use_query_type_bias:
            qtb = self.query_type_bias(x_n).reshape(bsz, seq_len, self.n_types, h)
            scores = scores + qtb.gather(
                2,
                cand_types[:, :, :, None].expand(-1, -1, -1, h),
            )
            query_type_bias_norm = qtb.norm(dim=-1).mean()
        scores = scores + self.source_bias[cand_sources]
        scores = scores.masked_fill(~cand_mask[:, :, :, None], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=2)

        r_all_h = torch.einsum("btjh,btjhd->bthd", probs, v)
        r_all = r_all_h.reshape(bsz, seq_len, d)

        read, typed_read_norms = self._mix_typed_reads(r_all, probs, v, cand_types, cand_mask)
        z = torch.cat([x, read, x * read, read - x], dim=-1)
        delta = self.fuse(self.norm_z(z))
        gate = torch.sigmoid(self.gate).reshape(1, 1, d)
        x_out = x + gate * delta

        fast_telemetry = os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") == "1"
        if fast_telemetry:
            p_mean = None
            entropy = x.new_tensor(0.0)
        else:
            p_mean = probs.mean(dim=-1)
            p_safe = p_mean.clamp_min(1e-8)
            entropy = -(p_safe * p_safe.log()).sum(dim=-1).mean()
        valid_counts = cand_mask.sum(dim=-1).float()
        delta_norm = x.new_tensor(0.0) if fast_telemetry else delta.norm(dim=-1).mean()
        x_norm = x.norm(dim=-1).mean()
        read_norm = x.new_tensor(0.0) if fast_telemetry else read.norm(dim=-1).mean()

        telemetry: dict[str, torch.Tensor] = {
            "dsqg_w_entropy": entropy.detach(),
            "dsqg_w_valid_candidate_count": valid_counts.mean().detach(),
            "dsqg_w_gate_mean": gate.mean().detach(),
            "dsqg_w_gate_min": gate.min().detach(),
            "dsqg_w_gate_max": gate.max().detach(),
            "dsqg_w_gate_logit_mean": self.gate.detach().mean(),
            "dsqg_w_delta_norm": delta_norm.detach(),
            "dsqg_w_x_norm": x_norm.detach(),
            "dsqg_w_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
            "dsqg_w_read_norm": read_norm.detach(),
            "dsqg_w_typed_read_norms": torch.stack(typed_read_norms).detach(),
            "read_mix_weight_norm": self.read_mix.weight.norm().detach(),
            "dsqg_w_query_type_bias_norm": query_type_bias_norm.detach(),
            "dsqg_w_candidate_score_bias_norm": candidate_score_bias_norm.detach(),
        }
        zero_mass = x.new_tensor(0.0)
        if fast_telemetry:
            for ctype in CandidateType:
                telemetry[f"dsqg_w_{ctype.name.lower()}_mass"] = zero_mass.detach()
        else:
            assert p_mean is not None
            for ctype in CandidateType:
                mask = (cand_types == int(ctype)) & cand_mask
                mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
                telemetry[f"dsqg_w_{ctype.name.lower()}_mass"] = mass.detach()
        telemetry["dsqg_w_local_mass"] = telemetry[f"dsqg_w_{CandidateType.LOCAL.name.lower()}_mass"]
        telemetry["dsqg_w_question_mass"] = telemetry[f"dsqg_w_{CandidateType.QUESTION.name.lower()}_mass"]
        telemetry["dsqg_w_hisa_evidence_mass"] = telemetry[f"dsqg_w_{CandidateType.HISA_EVIDENCE.name.lower()}_mass"]
        telemetry["dsqg_w_long_offset_mass"] = telemetry[f"dsqg_w_{CandidateType.LONG_OFFSET.name.lower()}_mass"]
        telemetry["dsqg_w_chunk_rep_mass"] = telemetry[f"dsqg_w_{CandidateType.CHUNK_REP.name.lower()}_mass"]
        telemetry["dsqg_w_null_mass"] = telemetry[f"dsqg_w_{CandidateType.NULL.name.lower()}_mass"]
        if fast_telemetry:
            for source in CandidateSource:
                telemetry[f"dsqg_w_{source.name.lower()}_source_mass"] = zero_mass.detach()
        else:
            assert p_mean is not None
            for source in CandidateSource:
                mask = (cand_sources == int(source)) & cand_mask
                mass = p_mean.masked_fill(~mask, 0.0).sum(dim=-1).mean()
                telemetry[f"dsqg_w_{source.name.lower()}_source_mass"] = mass.detach()
        telemetry["dsqg_w_l3_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.L3.name.lower()}_source_mass"]
        telemetry["dsqg_w_final_source_mass"] = telemetry[f"dsqg_w_{CandidateSource.FINAL.name.lower()}_source_mass"]
        telemetry.update(prior_telemetry)
        telemetry.update(width_telemetry)
        telemetry.update(typed_mixer_telemetry)
        telemetry.update(ebh_telemetry)

        if return_routing:
            telemetry["dsqg_w_probs"] = probs
        return x_out, telemetry


__all__ = ["DSQGWBlock"]
