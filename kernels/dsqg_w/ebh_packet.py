from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate_types import CandidateSource, CandidateType
from .gates import _forced_gate_value
from .sourcewise_gather import _DSQGWSourcewiseCandidateStateGather, _TRITON_SOURCEWISE_AVAILABLE, triton
from .width_cell import _hisa_evidence_type_mask


class DSQGWEvidenceBindingHub(nn.Module):
    """Sparse query-local evidence binder for DSQG-W.

    This is the TPJ-like path for testing whether D/HISA-selected evidence can
    be aligned into the current query token's semantic frame and injected as an
    owned residual packet.  It consumes an already-bounded candidate set and only
    performs candidate-local work plus all/type/source lane reductions by
    default.  Optional bounded pair mixing uses low-rank [B,T,K,K] scores only
    before lane reduction; it deliberately avoids materializing [B,T,K,K,D].
    """

    def __init__(
        self,
        *,
        d: int,
        n_types: int,
        n_sources: int,
        bottleneck: int,
        gate_init: float = -5.0,
        phase_bands: int = 4,
        max_distance: int = 8192,
        use_score_features: bool = True,
        use_pair_mixer: bool = False,
        pair_rank: int = 64,
        pair_gate_init: float = -2.5,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("d must be positive")
        if bottleneck <= 0:
            raise ValueError("evidence binding bottleneck must be positive")
        if phase_bands <= 0:
            raise ValueError("evidence binding phase_bands must be positive")
        if max_distance <= 0:
            raise ValueError("evidence binding max_distance must be positive")
        if pair_rank <= 0:
            raise ValueError("evidence binding pair_rank must be positive")
        self.d = int(d)
        self.n_types = int(n_types)
        self.n_sources = int(n_sources)
        self.phase_bands = int(phase_bands)
        self.max_distance = int(max_distance)
        self.use_score_features = bool(use_score_features)
        self.use_pair_mixer = bool(use_pair_mixer)
        self.pair_rank = int(pair_rank)
        self.eps = float(eps)

        self.norm_x = nn.LayerNorm(d)
        self.norm_c = nn.LayerNorm(d)
        self.value_proj = nn.Linear(d, d, bias=False)
        self.query_proj = nn.Linear(d, d, bias=False)
        self.type_value = nn.Embedding(n_types, d)
        self.source_value = nn.Embedding(n_sources, d)
        self.phase_proj = nn.Linear(2 * self.phase_bands, d, bias=False)
        self.score_proj = nn.Linear(1, d, bias=False)
        if self.use_pair_mixer:
            self.pair_q_proj = nn.Linear(d, self.pair_rank, bias=False)
            self.pair_k_proj = nn.Linear(d, self.pair_rank, bias=False)
            self.pair_v_proj = nn.Linear(d, self.pair_rank, bias=False)
            self.pair_out_proj = nn.Linear(self.pair_rank, d, bias=False)
            self.pair_type_bias = nn.Parameter(torch.zeros(n_types, n_types))
            self.pair_source_bias = nn.Parameter(torch.zeros(n_sources, n_sources))
            self.pair_gate = nn.Parameter(torch.full((d,), float(pair_gate_init)))
        else:
            self.pair_q_proj = None
            self.pair_k_proj = None
            self.pair_v_proj = None
            self.pair_out_proj = None
            self.register_parameter("pair_type_bias", None)
            self.register_parameter("pair_source_bias", None)
            self.register_parameter("pair_gate", None)
        lane_width = (1 + self.n_types + self.n_sources) * d
        self.read_mix = nn.Linear(lane_width, d, bias=False)
        self.packet_norm = nn.LayerNorm(d)
        self.delta_proj = nn.Sequential(
            nn.Linear(4 * d, bottleneck),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(bottleneck, d),
        )
        self.bind_gate = nn.Linear(2 * d, d)

        nn.init.zeros_(self.bind_gate.weight)
        nn.init.constant_(self.bind_gate.bias, float(gate_init))
        nn.init.normal_(self.delta_proj[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_proj[-1].bias)

    def _bind_gate_value(self, gate_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.bind_gate(gate_input)
        forced_gate = _forced_gate_value("DWARF_DSQG_W_FORCE_EBH_GATE", device=gate_input.device, dtype=gate_logits.dtype)
        if forced_gate is None:
            return torch.sigmoid(gate_logits), gate_input.new_tensor(0.0)
        return torch.full_like(gate_logits, forced_gate.item()), gate_input.new_tensor(1.0)

    def _apply_pair_mixer(
        self,
        aligned: torch.Tensor,
        safe_types: torch.Tensor,
        safe_sources: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        if not self.use_pair_mixer:
            return aligned, {}, None
        if (
            self.pair_q_proj is None
            or self.pair_k_proj is None
            or self.pair_v_proj is None
            or self.pair_out_proj is None
            or self.pair_type_bias is None
            or self.pair_source_bias is None
            or self.pair_gate is None
        ):
            raise RuntimeError("EBH pair mixer was enabled without pair parameters")
        bsz, seq_len, j_count, _ = aligned.shape
        q = self.pair_q_proj(aligned)
        k = self.pair_k_proj(aligned)
        v = self.pair_v_proj(aligned)
        scores = torch.bmm(
            q.reshape(bsz * seq_len, j_count, self.pair_rank),
            k.reshape(bsz * seq_len, j_count, self.pair_rank).transpose(1, 2),
        ).reshape(bsz, seq_len, j_count, j_count) / math.sqrt(float(self.pair_rank))
        scores = scores + self.pair_type_bias[safe_types[:, :, :, None], safe_types[:, :, None, :]]
        scores = scores + self.pair_source_bias[safe_sources[:, :, :, None], safe_sources[:, :, None, :]]
        valid_pair = valid[:, :, :, None] & valid[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1).masked_fill(~valid_pair, 0.0)
        mixed = torch.bmm(
            probs.reshape(bsz * seq_len, j_count, j_count),
            v.reshape(bsz * seq_len, j_count, self.pair_rank),
        ).reshape(bsz, seq_len, j_count, self.pair_rank)
        delta = self.pair_out_proj(mixed)
        forced_gate = _forced_gate_value("DWARF_DSQG_W_FORCE_EBH_PAIR_GATE", device=aligned.device, dtype=aligned.dtype)
        if forced_gate is None:
            gate = torch.sigmoid(self.pair_gate).reshape(1, 1, 1, self.d)
            forced_gate_flag = aligned.new_tensor(0.0)
        else:
            gate = forced_gate.reshape(1, 1, 1, 1).expand(1, 1, 1, self.d)
            forced_gate_flag = aligned.new_tensor(1.0)
        valid_f = valid[..., None].to(delta.dtype)
        out = (aligned + gate * delta * valid_f).masked_fill(~valid[..., None], 0.0)

        valid_targets = valid.bool()
        p_safe = probs.clamp_min(1e-8)
        entropy_per_target = -(p_safe * p_safe.log()).sum(dim=-1)
        entropy = entropy_per_target.masked_select(valid_targets).mean() if valid_targets.any() else aligned.new_tensor(0.0)
        diag = torch.eye(j_count, device=aligned.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        self_mass = probs.masked_fill(~diag, 0.0).sum(dim=-1)
        self_mass = self_mass.masked_select(valid_targets).mean() if valid_targets.any() else aligned.new_tensor(0.0)
        valid_delta_count = valid.to(delta.dtype).sum().clamp_min(1.0)
        delta_norm = (delta.norm(dim=-1) * valid.to(delta.dtype)).sum() / valid_delta_count
        question_mask = safe_types == int(CandidateType.QUESTION)
        hisa_family_mask = _hisa_evidence_type_mask(safe_types)

        def pair_mass(target_mask: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
            target_mask = target_mask & valid_targets
            source_mask = source_mask & valid
            if not bool(target_mask.any()):
                return aligned.new_tensor(0.0)
            mass = probs.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
            selected = mass.masked_select(target_mask)
            return selected.mean() if selected.numel() else aligned.new_tensor(0.0)

        telemetry = {
            "dsqg_w_ebh_pair_mixer_enabled": aligned.new_tensor(1.0).detach(),
            "dsqg_w_ebh_pair_gate_mean": gate.mean().detach(),
            "dsqg_w_ebh_pair_forced_gate": forced_gate_flag.detach(),
            "dsqg_w_ebh_pair_gate_logit_mean": self.pair_gate.detach().mean(),
            "dsqg_w_ebh_pair_entropy": entropy.detach(),
            "dsqg_w_ebh_pair_self_mass": self_mass.detach(),
            "dsqg_w_ebh_pair_delta_norm": delta_norm.detach(),
            "dsqg_w_ebh_pair_question_to_hisa_mass": pair_mass(question_mask, hisa_family_mask).detach(),
            "dsqg_w_ebh_pair_hisa_to_question_mass": pair_mass(hisa_family_mask, question_mask).detach(),
        }
        return out, telemetry, probs.detach()

    def forward(
        self,
        x: torch.Tensor,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        candidate_distances: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if x.ndim != 3 or cand_states.ndim != 4:
            raise ValueError("x must be [B,T,D] and cand_states must be [B,T,J,D]")
        bsz, seq_len, d = x.shape
        b2, t2, j_count, d2 = cand_states.shape
        if (bsz, seq_len, d) != (b2, t2, d2):
            raise ValueError("x and cand_states shape mismatch")
        if d != self.d:
            raise ValueError(f"x last dim {d} does not match evidence binding d {self.d}")
        if cand_types.shape != (bsz, seq_len, j_count) or cand_sources.shape != cand_types.shape:
            raise ValueError("candidate type/source tensors must have shape [B,T,J]")
        if cand_mask.shape != cand_types.shape:
            raise ValueError("candidate mask must have shape [B,T,J]")

        device = x.device
        dtype = x.dtype
        valid = cand_mask.to(device=device, dtype=torch.bool)
        safe_types = cand_types.to(device=device, dtype=torch.long).clamp(0, self.n_types - 1)
        safe_sources = cand_sources.to(device=device, dtype=torch.long).clamp(0, self.n_sources - 1)
        weights = valid.to(dtype=dtype)
        mass = weights.sum(dim=-1, keepdim=True)
        norm_weights = torch.where(mass > 0, weights / mass.clamp_min(self.eps), torch.zeros_like(weights))

        x_n = self.norm_x(x)
        c_n = self.norm_c(cand_states)
        aligned = self.value_proj(c_n)
        aligned = aligned + self.type_value(safe_types).to(dtype=dtype)
        aligned = aligned + self.source_value(safe_sources).to(dtype=dtype)
        aligned = aligned + self.phase_proj(
            self._phase_features(candidate_distances, bsz, seq_len, j_count, device, dtype)
        )
        if self.use_score_features:
            aligned = aligned + self.score_proj(
                self._score_features(cand_scores, valid, bsz, seq_len, j_count, device, dtype)[..., None]
            )
        query_context = self.query_proj(x_n)
        aligned = aligned + query_context[:, :, None, :]
        aligned = aligned.masked_fill(~valid[..., None], 0.0)
        aligned, pair_telemetry, pair_probs = self._apply_pair_mixer(aligned, safe_types, safe_sources, valid)

        all_read = (aligned * norm_weights[..., None]).sum(dim=2)
        type_reads, type_mass = self._lane_reads(aligned, weights, safe_types, self.n_types)
        source_reads, source_mass = self._lane_reads(aligned, weights, safe_sources, self.n_sources)
        lane_cat = torch.cat(
            [
                all_read,
                type_reads.reshape(bsz, seq_len, self.n_types * d),
                source_reads.reshape(bsz, seq_len, self.n_sources * d),
            ],
            dim=-1,
        )
        bound_packet = self.packet_norm(self.read_mix(lane_cat))
        gate_input = torch.cat([x_n, bound_packet], dim=-1)
        gate, forced_gate_flag = self._bind_gate_value(gate_input)
        delta_input = torch.cat([x, bound_packet, x * bound_packet, bound_packet - x], dim=-1)
        delta = self.delta_proj(delta_input)
        has_evidence = (mass.squeeze(-1) > 0).to(dtype=dtype)[..., None]
        out = x + has_evidence * gate * delta

        telemetry = self._telemetry(
            x,
            bound_packet,
            delta,
            gate,
            weights,
            type_mass,
            source_mass,
            has_evidence,
            forced_gate_flag,
        )
        telemetry.update(pair_telemetry)
        if return_aux:
            aux = {
                "bound_packet": bound_packet,
                "aligned_candidates": aligned,
                "all_read": all_read,
                "type_reads": type_reads,
                "source_reads": source_reads,
                "candidate_weight_mass": mass.squeeze(-1).detach(),
                "normalized_candidate_weights": norm_weights.detach(),
                "type_weight_mass": type_mass.detach(),
                "source_weight_mass": source_mass.detach(),
                "bind_gate": gate,
                "delta": delta,
            }
            if pair_probs is not None:
                aux["ebh_pair_probs"] = pair_probs
            return out, telemetry, aux
        return out, telemetry

    @staticmethod
    def _gather_source_rows(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = states.shape
        safe = token_indices.to(device=states.device, dtype=torch.long).clamp(0, max(seq_len - 1, 0))
        return torch.gather(states, 1, safe[:, :, None].expand(bsz, seq_len, d))

    def forward_sourcewise_packet(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Build the EBH lane packet directly from sourcewise metadata.

        This preserves the materialized EBH lane math for raw source candidate
        states, but accumulates one slot at a time into all/type/source lanes
        instead of constructing a persistent [B,T,J,D] candidate-state tensor.
        When upstream semantic transforms (width cell / typed mixer) are active,
        callers must label the path as an approximation because those transforms
        are not included in this packet.
        """
        if x.ndim != 3:
            raise ValueError("x must be [B,T,D]")
        bsz, seq_len, d = x.shape
        if d != self.d:
            raise ValueError(f"x last dim {d} does not match evidence binding d {self.d}")
        if cand_mask.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate metadata shape mismatch")
        if cand_token_indices.shape != cand_mask.shape or cand_types.shape != cand_mask.shape:
            raise ValueError("candidate token/type tensors must have shape [B,T,J]")
        if cand_sources.shape != cand_mask.shape:
            raise ValueError("candidate source tensor must have shape [B,T,J]")
        if l3_states is not None and l3_states.shape != x.shape:
            raise ValueError("l3_states must match x shape")
        if chunk_rep_states is not None and chunk_rep_states.shape != x.shape:
            raise ValueError("chunk_rep_states must match x shape")

        device = x.device
        dtype = x.dtype
        j_count = cand_mask.shape[-1]
        use_triton_lane_accum = (
            os.getenv("DWARF_DSQG_W_EBH_TRITON_LANE_ACCUM", "0") == "1"
            and self.use_score_features
            and not self.use_pair_mixer
            and _TRITON_SOURCEWISE_AVAILABLE
            and triton is not None
            and x.is_cuda
            and cand_token_indices.is_cuda
            and cand_sources.is_cuda
            and cand_mask.is_cuda
            and chunk_rep_states is None
        )
        if use_triton_lane_accum:
            fast_result = self._forward_sourcewise_packet_triton_accum(
                x,
                cand_token_indices,
                cand_types,
                cand_sources,
                cand_mask,
                l3_states=l3_states,
                candidate_distances=candidate_distances,
                cand_scores=cand_scores,
                return_aux=return_aux,
            )
            if fast_result is not None:
                return fast_result

        valid = cand_mask.to(device=device, dtype=torch.bool)
        safe_types = cand_types.to(device=device, dtype=torch.long).clamp(0, self.n_types - 1)
        safe_sources = cand_sources.to(device=device, dtype=torch.long).clamp(0, self.n_sources - 1)
        weights = valid.to(dtype=dtype)
        mass = weights.sum(dim=-1, keepdim=True)
        type_mass = x.new_zeros((bsz, seq_len, self.n_types))
        source_mass = x.new_zeros((bsz, seq_len, self.n_sources))
        for slot_idx in range(j_count):
            w_j = weights[:, :, slot_idx]
            type_mass.scatter_add_(2, safe_types[:, :, slot_idx, None], w_j[:, :, None])
            source_mass.scatter_add_(2, safe_sources[:, :, slot_idx, None], w_j[:, :, None])

        x_n = self.norm_x(x)
        query_context = self.query_proj(x_n)
        final_base = self.value_proj(self.norm_c(x))
        l3_source = l3_states if l3_states is not None else x
        l3_base = final_base if l3_source is x else self.value_proj(self.norm_c(l3_source))
        summary_source = chunk_rep_states if chunk_rep_states is not None else x
        summary_base = final_base if summary_source is x else self.value_proj(self.norm_c(summary_source))
        null_base = self.value_proj(self.norm_c(torch.zeros_like(x)))
        projected_bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_base,
            int(CandidateSource.QUESTION_CACHE): final_base,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
            int(CandidateSource.NULL): null_base,
        }

        score_features = None
        if self.use_score_features:
            score_features = self._score_features(cand_scores, valid, bsz, seq_len, j_count, device, dtype)

        all_sum = x.new_zeros((bsz, seq_len, d))
        type_sums = x.new_zeros((bsz, seq_len, self.n_types, d))
        source_sums = x.new_zeros((bsz, seq_len, self.n_sources, d))
        gather_tokens = cand_token_indices.to(device=device, dtype=torch.long).clamp(0, max(seq_len - 1, 0))
        for slot_idx in range(j_count):
            valid_j = valid[:, :, slot_idx]
            if not bool(valid_j.any()):
                continue
            source_j = safe_sources[:, :, slot_idx]
            token_j = gather_tokens[:, :, slot_idx]
            value_j = x.new_zeros((bsz, seq_len, d))
            for source_id, base in projected_bases.items():
                source_mask = (source_j == int(source_id)) & valid_j
                if bool(source_mask.any()):
                    gathered = self._gather_source_rows(base, token_j)
                    value_j = value_j + gathered * source_mask[:, :, None].to(dtype)
            type_j = safe_types[:, :, slot_idx]
            aligned_j = value_j
            aligned_j = aligned_j + self.type_value(type_j).to(dtype=dtype)
            aligned_j = aligned_j + self.source_value(source_j).to(dtype=dtype)
            aligned_j = aligned_j + self.phase_proj(
                self._phase_features(
                    None if candidate_distances is None else candidate_distances[:, :, slot_idx : slot_idx + 1],
                    bsz,
                    seq_len,
                    1,
                    device,
                    dtype,
                ).squeeze(2)
            )
            if score_features is not None:
                aligned_j = aligned_j + self.score_proj(score_features[:, :, slot_idx, None])
            aligned_j = aligned_j + query_context
            aligned_j = aligned_j * valid_j[:, :, None].to(dtype)
            all_sum = all_sum + aligned_j
            type_sums.scatter_add_(2, type_j[:, :, None, None].expand(-1, -1, 1, d), aligned_j[:, :, None, :])
            source_sums.scatter_add_(2, source_j[:, :, None, None].expand(-1, -1, 1, d), aligned_j[:, :, None, :])

        all_read = torch.where(mass > 0, all_sum / mass.clamp_min(self.eps), torch.zeros_like(all_sum))
        type_reads = torch.where(
            type_mass[:, :, :, None] > 0,
            type_sums / type_mass[:, :, :, None].clamp_min(self.eps),
            torch.zeros_like(type_sums),
        )
        source_reads = torch.where(
            source_mass[:, :, :, None] > 0,
            source_sums / source_mass[:, :, :, None].clamp_min(self.eps),
            torch.zeros_like(source_sums),
        )
        lane_cat = torch.cat(
            [
                all_read,
                type_reads.reshape(bsz, seq_len, self.n_types * d),
                source_reads.reshape(bsz, seq_len, self.n_sources * d),
            ],
            dim=-1,
        )
        bound_packet = self.packet_norm(self.read_mix(lane_cat))
        gate_input = torch.cat([x_n, bound_packet], dim=-1)
        gate, forced_gate_flag = self._bind_gate_value(gate_input)
        delta_input = torch.cat([x, bound_packet, x * bound_packet, bound_packet - x], dim=-1)
        delta = self.delta_proj(delta_input)
        has_evidence = (mass.squeeze(-1) > 0).to(dtype=dtype)[..., None]
        out = x + has_evidence * gate * delta

        telemetry = self._telemetry(
            x,
            bound_packet,
            delta,
            gate,
            weights,
            type_mass,
            source_mass,
            has_evidence,
            forced_gate_flag,
        )
        telemetry["dsqg_w_ebh_packet_sourcewise"] = x.new_tensor(1.0).detach()
        telemetry["dsqg_w_ebh_packet_triton"] = x.new_tensor(0.0).detach()
        telemetry["dsqg_w_ebh_score_features_enabled"] = x.new_tensor(1.0 if self.use_score_features else 0.0).detach()
        telemetry["dsqg_w_ebh_packet_no_score_legacy"] = x.new_tensor(0.0 if self.use_score_features else 1.0).detach()
        telemetry["dsqg_w_ebh_pair_mixer_legacy"] = x.new_tensor(1.0 if self.use_pair_mixer else 0.0).detach()
        if return_aux:
            aux = {
                "bound_packet": bound_packet,
                "all_read": all_read,
                "type_reads": type_reads,
                "source_reads": source_reads,
                "candidate_weight_mass": mass.squeeze(-1).detach(),
                "type_weight_mass": type_mass.detach(),
                "source_weight_mass": source_mass.detach(),
                "bind_gate": gate,
                "delta": delta,
            }
            return out, telemetry, aux
        return out, telemetry

    def _forward_sourcewise_packet_triton_accum(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Fast EBH packet path using the existing Triton sourcewise gather seam.

        The slow sourcewise packet path avoids persistent candidate materialization
        but still performs Python slot/source loops and scatter-add lane
        accumulation.  This path gathers already-projected source surfaces once
        into candidate order, then reuses the vectorized EBH lane reducer from the
        materialized path.  It is exact for raw sourcewise packets without real
        chunk/summary states; callers fall back before entering this helper when
        `chunk_rep_states` is present.
        """
        if not (_TRITON_SOURCEWISE_AVAILABLE and triton is not None and x.is_cuda):
            return None
        bsz, seq_len, d = x.shape
        j_count = cand_mask.shape[-1]
        device = x.device
        dtype = x.dtype

        valid = cand_mask.to(device=device, dtype=torch.bool)
        safe_types = cand_types.to(device=device, dtype=torch.long).clamp(0, self.n_types - 1)
        safe_sources = cand_sources.to(device=device, dtype=torch.long).clamp(0, self.n_sources - 1)
        weights = valid.to(dtype=dtype)
        mass = weights.sum(dim=-1, keepdim=True)
        norm_weights = torch.where(mass > 0, weights / mass.clamp_min(self.eps), torch.zeros_like(weights))

        x_n = self.norm_x(x)
        query_context = self.query_proj(x_n)
        final_base = self.value_proj(self.norm_c(x))
        l3_source = l3_states if l3_states is not None else x
        l3_base = final_base if l3_source is x else self.value_proj(self.norm_c(l3_source))
        values = _DSQGWSourcewiseCandidateStateGather.apply(
            final_base,
            l3_base,
            cand_token_indices,
            safe_sources,
            valid,
            l3_source is not x,
        )

        aligned = values
        aligned = aligned + self.type_value(safe_types).to(dtype=dtype)
        aligned = aligned + self.source_value(safe_sources).to(dtype=dtype)
        aligned = aligned + self.phase_proj(self._phase_features(candidate_distances, bsz, seq_len, j_count, device, dtype))
        if self.use_score_features:
            aligned = aligned + self.score_proj(
                self._score_features(cand_scores, valid, bsz, seq_len, j_count, device, dtype)[..., None]
            )
        aligned = aligned + query_context[:, :, None, :]
        aligned = aligned.masked_fill(~valid[..., None], 0.0)

        all_read = (aligned * norm_weights[..., None]).sum(dim=2)
        type_reads, type_mass = self._lane_reads(aligned, weights, safe_types, self.n_types)
        source_reads, source_mass = self._lane_reads(aligned, weights, safe_sources, self.n_sources)
        lane_cat = torch.cat(
            [
                all_read,
                type_reads.reshape(bsz, seq_len, self.n_types * d),
                source_reads.reshape(bsz, seq_len, self.n_sources * d),
            ],
            dim=-1,
        )
        bound_packet = self.packet_norm(self.read_mix(lane_cat))
        gate_input = torch.cat([x_n, bound_packet], dim=-1)
        gate, forced_gate_flag = self._bind_gate_value(gate_input)
        delta_input = torch.cat([x, bound_packet, x * bound_packet, bound_packet - x], dim=-1)
        delta = self.delta_proj(delta_input)
        has_evidence = (mass.squeeze(-1) > 0).to(dtype=dtype)[..., None]
        out = x + has_evidence * gate * delta

        telemetry = self._telemetry(
            x,
            bound_packet,
            delta,
            gate,
            weights,
            type_mass,
            source_mass,
            has_evidence,
            forced_gate_flag,
        )
        telemetry["dsqg_w_ebh_packet_sourcewise"] = x.new_tensor(1.0).detach()
        telemetry["dsqg_w_ebh_packet_triton"] = x.new_tensor(1.0).detach()
        telemetry["dsqg_w_ebh_score_features_enabled"] = x.new_tensor(1.0).detach()
        telemetry["dsqg_w_ebh_packet_no_score_legacy"] = x.new_tensor(0.0).detach()
        telemetry["dsqg_w_ebh_pair_mixer_legacy"] = x.new_tensor(0.0).detach()
        if return_aux:
            aux = {
                "bound_packet": bound_packet,
                "all_read": all_read,
                "type_reads": type_reads,
                "source_reads": source_reads,
                "candidate_weight_mass": mass.squeeze(-1).detach(),
                "type_weight_mass": type_mass.detach(),
                "source_weight_mass": source_mass.detach(),
                "bind_gate": gate,
                "delta": delta,
            }
            return out, telemetry, aux
        return out, telemetry

    def _phase_features(
        self,
        candidate_distances: torch.Tensor | None,
        bsz: int,
        seq_len: int,
        j_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if candidate_distances is None:
            distances = torch.zeros((bsz, seq_len, j_count), device=device, dtype=dtype)
        else:
            if candidate_distances.shape != (bsz, seq_len, j_count):
                raise ValueError("candidate_distances must have shape [B,T,J]")
            distances = candidate_distances.to(device=device, dtype=dtype).clamp_min(0.0)
        phase = torch.log1p(distances) / math.log1p(float(self.max_distance))
        bands = torch.arange(1, self.phase_bands + 1, device=device, dtype=dtype)
        angles = phase[..., None] * bands * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _score_features(
        self,
        cand_scores: torch.Tensor | None,
        valid: torch.Tensor,
        bsz: int,
        seq_len: int,
        j_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if cand_scores is None:
            return torch.zeros((bsz, seq_len, j_count), device=device, dtype=dtype)
        if cand_scores.shape != (bsz, seq_len, j_count):
            raise ValueError("cand_scores must have shape [B,T,J]")
        scores = torch.nan_to_num(cand_scores.to(device=device, dtype=dtype), nan=0.0, neginf=0.0, posinf=0.0)
        scores = scores.masked_fill(~valid, 0.0)
        denom = valid.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = scores.sum(dim=-1, keepdim=True) / denom
        var = ((scores - mean).masked_fill(~valid, 0.0).square().sum(dim=-1, keepdim=True) / denom).clamp_min(1e-6)
        return ((scores - mean) / var.sqrt()).clamp(-5.0, 5.0).masked_fill(~valid, 0.0)

    def _lane_reads(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        lane_ids: torch.Tensor,
        n_lanes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lane_mask = F.one_hot(lane_ids, num_classes=n_lanes).to(dtype=values.dtype)
        lane_weights = weights[..., None] * lane_mask
        lane_mass = lane_weights.sum(dim=2)
        lane_norm = torch.where(
            lane_mass[:, :, None, :] > 0,
            lane_weights / lane_mass[:, :, None, :].clamp_min(self.eps),
            torch.zeros_like(lane_weights),
        )
        reads = torch.einsum("btjd,btjl->btld", values, lane_norm)
        return reads, lane_mass

    def _telemetry(
        self,
        x: torch.Tensor,
        bound_packet: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        weights: torch.Tensor,
        type_mass: torch.Tensor,
        source_mass: torch.Tensor,
        has_evidence: torch.Tensor,
        forced_gate_flag: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            valid_count = weights.sum(dim=-1)
            fast_telemetry = os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") == "1"
            delta_norm = x.new_tensor(0.0) if fast_telemetry else delta.norm(dim=-1).mean()
            gated_delta_norm = x.new_tensor(0.0) if fast_telemetry else (has_evidence * gate * delta).norm(dim=-1).mean()
            x_norm = x.norm(dim=-1).mean()
            packet_norm = x.new_tensor(0.0) if fast_telemetry else bound_packet.norm(dim=-1).mean()
            telemetry = {
                "dsqg_w_ebh_enabled": x.new_tensor(1.0).detach(),
                "dsqg_w_ebh_valid_candidate_count": valid_count.mean().detach(),
                "dsqg_w_ebh_candidate_weight_mass": weights.sum(dim=-1).mean().detach(),
                "dsqg_w_ebh_active_row_fraction": has_evidence.squeeze(-1).float().mean().detach(),
                "dsqg_w_ebh_bind_gate_mean": gate.mean().detach(),
                "dsqg_w_ebh_bind_gate_min": gate.min().detach(),
                "dsqg_w_ebh_bind_gate_max": gate.max().detach(),
                "dsqg_w_ebh_forced_gate": forced_gate_flag.detach(),
                "dsqg_w_ebh_bind_gate_logit_mean": self.bind_gate.bias.detach().mean(),
                "dsqg_w_ebh_delta_norm": delta_norm.detach(),
                "dsqg_w_ebh_x_norm": x_norm.detach(),
                "dsqg_w_ebh_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
                "dsqg_w_ebh_gated_delta_to_x_ratio": (gated_delta_norm / x_norm.clamp_min(1e-8)).detach(),
                "dsqg_w_ebh_bound_packet_norm": packet_norm.detach(),
            }
            for ctype in CandidateType:
                if int(ctype) < type_mass.shape[-1]:
                    telemetry[f"dsqg_w_ebh_{ctype.name.lower()}_mass"] = type_mass[..., int(ctype)].mean().detach()
            for source in CandidateSource:
                if int(source) < source_mass.shape[-1]:
                    telemetry[f"dsqg_w_ebh_{source.name.lower()}_source_mass"] = source_mass[..., int(source)].mean().detach()
            return telemetry


__all__ = ["DSQGWEvidenceBindingHub"]
