from __future__ import annotations

# Behavior-preserving extraction boundary for DSQG-W candidate construction.
# The compatibility monolith still exports its public surface; this module owns
# the standalone CandidateProvider implementation for split-module imports.
import os

import torch

from .candidate_batch import Candidate, CandidateBatch
from .candidate_types import (
    CandidateEvidenceBit,
    CandidateSource,
    CandidateType,
    _CANDIDATE_PRIORITY,
)
from .config import DSQGWConfig
from .instrumentation import _dsqg_w_geometry_audit_enabled, _dsqg_w_geometry_telemetry, _dsqg_w_profile_range


class CandidateProvider:

    """Bounded causal heterogeneous candidate construction for DSQG-W.

    This is intentionally diagnostic and explicit, not fused.  It constructs only
    O(B*T*J) candidate tensors and rejects future-token routes.  The default MVP
    supports LOCAL, QUESTION, HISA_EVIDENCE/L3_SKIP, LONG_OFFSET, CHUNK_REP, and
    NULL fallback candidates.
    """

    _CANDIDATE_TYPE_EVIDENCE_BITS_CACHE: dict[str, torch.Tensor] = {}

    def __init__(self, config: DSQGWConfig):
        self.config = config
        self._positions_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._priority_table_cache: dict[str, torch.Tensor] = {}
        self._arange_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._typed_hisa_rep_type_ids_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _candidate_diagnostics_enabled() -> bool:
        """Return whether expensive candidate-distribution telemetry is enabled.

        Normal training needs the path/promotion markers, not per-type/source
        histograms or score summaries. Keep the diagnostic surface opt-in so the
        hot metadata path avoids masked_selects and enum loops unless explicitly
        requested for audits.
        """

        return os.getenv("DWARF_DSQG_W_CANDIDATE_DIAGNOSTICS", "0") == "1"

    @staticmethod
    def _device_key(device: torch.device) -> str:
        return str(device)

    def _positions(self, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (int(bsz), int(seq_len), self._device_key(device))
        cached = self._positions_cache.get(key)
        if cached is None:
            cached = torch.arange(seq_len, device=device, dtype=torch.long).reshape(1, seq_len, 1).expand(bsz, -1, -1)
            self._positions_cache[key] = cached
        return cached

    def _arange(self, count: int, device: torch.device) -> torch.Tensor:
        key = (int(count), self._device_key(device))
        cached = self._arange_cache.get(key)
        if cached is None:
            cached = torch.arange(int(count), device=device, dtype=torch.long)
            self._arange_cache[key] = cached
        return cached

    def _priority_table(self, device: torch.device) -> torch.Tensor:
        key = self._device_key(device)
        cached = self._priority_table_cache.get(key)
        if cached is None:
            cached = torch.full((max(self.config.n_types, len(CandidateType)),), 99, device=device, dtype=torch.long)
            for ctype, priority_value in _CANDIDATE_PRIORITY.items():
                cached[int(ctype)] = int(priority_value)
            self._priority_table_cache[key] = cached
        return cached

    def _typed_hisa_rep_type_ids(self, device: torch.device) -> torch.Tensor:
        key = self._device_key(device)
        cached = self._typed_hisa_rep_type_ids_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                [
                    int(CandidateType.HISA_EVIDENCE_REP0),
                    int(CandidateType.HISA_EVIDENCE_REP1),
                    int(CandidateType.HISA_EVIDENCE_REP2),
                    int(CandidateType.HISA_EVIDENCE_REP3),
                ],
                device=device,
                dtype=torch.long,
            )
            self._typed_hisa_rep_type_ids_cache[key] = cached
        return cached

    def build(
        self,
        final_states: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        question_indices: torch.Tensor | list[list[int]] | None = None,
        hisa_evidence_indices: torch.Tensor | None = None,
        hisa_evidence_scores: torch.Tensor | None = None,
        chunk_rep_indices: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        l3_skip_indices: torch.Tensor | None = None,
    ) -> CandidateBatch:
        if not self._can_use_vectorized(
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            chunk_rep_indices=chunk_rep_indices,
            l3_skip_indices=l3_skip_indices,
        ):
            return self._build_reference(
                final_states,
                l3_states=l3_states,
                question_indices=question_indices,
                hisa_evidence_indices=hisa_evidence_indices,
                hisa_evidence_scores=hisa_evidence_scores,
                chunk_rep_indices=chunk_rep_indices,
                chunk_rep_states=chunk_rep_states,
                l3_skip_indices=l3_skip_indices,
            )
        return self._build_vectorized(
            final_states,
            l3_states=l3_states,
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            chunk_rep_indices=chunk_rep_indices,
            chunk_rep_states=chunk_rep_states,
            l3_skip_indices=l3_skip_indices,
            materialize_states=True,
        )

    def build_metadata(
        self,
        final_states: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        question_indices: torch.Tensor | None = None,
        hisa_evidence_indices: torch.Tensor | None = None,
        hisa_evidence_scores: torch.Tensor | None = None,
        chunk_rep_indices: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        l3_skip_indices: torch.Tensor | None = None,
    ) -> CandidateBatch:
        """Build compact candidate metadata without materializing [B,T,J,D] states."""
        if not self._can_use_vectorized(
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            chunk_rep_indices=chunk_rep_indices,
            l3_skip_indices=l3_skip_indices,
        ):
            raise ValueError("build_metadata requires tensor candidate indices/scores")
        if self._can_use_dsr_selected_metadata_fast_path(
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            chunk_rep_indices=chunk_rep_indices,
            l3_skip_indices=l3_skip_indices,
        ):
            return self._build_dsr_selected_metadata_fast(
                final_states,
                l3_states=l3_states,
                question_indices=question_indices,
                hisa_evidence_indices=hisa_evidence_indices,
                hisa_evidence_scores=hisa_evidence_scores,
                l3_skip_indices=l3_skip_indices,
            )
        return self._build_vectorized(
            final_states,
            l3_states=l3_states,
            question_indices=question_indices,
            hisa_evidence_indices=hisa_evidence_indices,
            hisa_evidence_scores=hisa_evidence_scores,
            chunk_rep_indices=chunk_rep_indices,
            chunk_rep_states=chunk_rep_states,
            l3_skip_indices=l3_skip_indices,
            materialize_states=False,
        )

    @staticmethod
    def _can_use_vectorized(**kwargs) -> bool:
        return all(value is None or isinstance(value, torch.Tensor) for value in kwargs.values())

    def _can_use_dsr_selected_metadata_fast_path(
        self,
        *,
        question_indices: torch.Tensor | None,
        hisa_evidence_indices: torch.Tensor | None,
        hisa_evidence_scores: torch.Tensor | None,
        chunk_rep_indices: torch.Tensor | None,
        l3_skip_indices: torch.Tensor | None,
    ) -> bool:
        if os.getenv("DWARF_DSQG_W_SPECIALIZED_METADATA", "1") == "0":
            return False
        return (
            not self.config.local_offsets
            and not self.config.long_offsets
            and self.config.k_chunk <= 0
            and chunk_rep_indices is None
            and question_indices is not None
            and hisa_evidence_indices is not None
            and hisa_evidence_scores is not None
            and l3_skip_indices is not None
            and self.config.k_hisa_evidence > 0
            and self.config.k_question > 0
            and self.config.k_l3_skip > 0
        )

    @staticmethod
    def _dedupe_same_source_group(
        tokens: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep at most one candidate per token inside a same-source/type group."""
        same_key = (tokens.unsqueeze(-1) == tokens.unsqueeze(-2)) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        k_count = tokens.shape[-1]
        order = torch.arange(k_count, device=tokens.device, dtype=torch.long)
        score_j = scores.unsqueeze(-2)
        max_score_same_key = score_j.masked_fill(~same_key, float("-inf")).amax(dim=-1)
        earlier_same_key = same_key & (order.reshape(1, 1, 1, k_count) < order.reshape(1, 1, k_count, 1))
        earlier_higher_or_equal_score = earlier_same_key & (score_j >= scores.unsqueeze(-1))
        keep = valid & (scores >= max_score_same_key) & ~earlier_higher_or_equal_score.any(dim=-1)
        duplicate_count = (valid & earlier_same_key.any(dim=-1)).sum()
        return keep, duplicate_count

    @staticmethod
    def _is_hisa_type(types: torch.Tensor) -> torch.Tensor:
        return (
            (types == int(CandidateType.HISA_EVIDENCE))
            | (types == int(CandidateType.HISA_EVIDENCE_REP0))
            | (types == int(CandidateType.HISA_EVIDENCE_REP1))
            | (types == int(CandidateType.HISA_EVIDENCE_REP2))
            | (types == int(CandidateType.HISA_EVIDENCE_REP3))
        )

    @classmethod
    def _candidate_type_evidence_bits(cls, types: torch.Tensor) -> torch.Tensor:
        key = str(types.device)
        table = cls._CANDIDATE_TYPE_EVIDENCE_BITS_CACHE.get(key)
        max_type_id = max(int(ctype) for ctype in CandidateType)
        if table is None or table.numel() <= max_type_id or table.device != types.device:
            table = torch.zeros((max_type_id + 1,), device=types.device, dtype=torch.long)
            hisa_type_ids = {
                int(CandidateType.HISA_EVIDENCE),
                int(CandidateType.HISA_EVIDENCE_REP0),
                int(CandidateType.HISA_EVIDENCE_REP1),
                int(CandidateType.HISA_EVIDENCE_REP2),
                int(CandidateType.HISA_EVIDENCE_REP3),
            }
            for type_id in hisa_type_ids:
                if 0 <= type_id < table.numel():
                    table[type_id] = int(CandidateEvidenceBit.HISA)
            for ctype, bit in (
                (CandidateType.QUESTION, CandidateEvidenceBit.QUESTION),
                (CandidateType.L3_SKIP, CandidateEvidenceBit.L3_SKIP),
                (CandidateType.LOCAL, CandidateEvidenceBit.LOCAL),
                (CandidateType.LONG_OFFSET, CandidateEvidenceBit.LONG_OFFSET),
                (CandidateType.CHUNK_REP, CandidateEvidenceBit.CHUNK_REP),
                (CandidateType.NULL, CandidateEvidenceBit.NULL),
            ):
                table[int(ctype)] = table[int(ctype)] | int(bit)
            cls._CANDIDATE_TYPE_EVIDENCE_BITS_CACHE[key] = table
        return table[types.clamp_min(0).clamp_max(table.numel() - 1)]

    @staticmethod
    def _evidence_count_from_bits(bits: torch.Tensor) -> torch.Tensor:
        count = torch.zeros_like(bits, dtype=torch.long)
        for bit in CandidateEvidenceBit:
            count = count + ((bits & int(bit)) != 0).to(torch.long)
        return count

    @classmethod
    def _collapse_evidence_by_token_source(
        cls,
        tokens: torch.Tensor,
        sources: torch.Tensor,
        evidence_bits: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        same_key = (
            (tokens.unsqueeze(-1) == tokens.unsqueeze(-2))
            & (sources.unsqueeze(-1) == sources.unsqueeze(-2))
            & valid.unsqueeze(-1)
            & valid.unsqueeze(-2)
        )
        collapsed = torch.zeros_like(evidence_bits, dtype=torch.long)
        for bit in CandidateEvidenceBit:
            bit_present = ((evidence_bits & int(bit)) != 0).unsqueeze(-2)
            has_bit = (same_key & bit_present).any(dim=-1)
            collapsed = torch.where(has_bit, collapsed | int(bit), collapsed)
        collapsed = collapsed.masked_fill(~valid, 0)
        return collapsed, cls._evidence_count_from_bits(collapsed).masked_fill(~valid, 0)

    def _apply_candidate_quotas(
        self,
        sort_key: torch.Tensor,
        keep: torch.Tensor,
        types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = torch.zeros((), device=sort_key.device, dtype=torch.long)
        if not self.config.use_candidate_quotas or self.config.quota_hisa_max <= 0:
            return keep, clipped
        hisa = self._is_hisa_type(types) & keep
        if not bool(hisa.any()):
            return keep, clipped
        cap = min(int(self.config.quota_hisa_max), int(sort_key.shape[-1]))
        hisa_order = sort_key.masked_fill(~hisa, float("inf")).argsort(dim=-1)[..., :cap]
        allowed_hisa = torch.zeros_like(keep, dtype=torch.bool).scatter(-1, hisa_order, True) & hisa
        capped_keep = keep & (~hisa | allowed_hisa)
        clipped = (hisa & ~allowed_hisa).sum()
        return capped_keep, clipped

    @staticmethod
    def _evidence_telemetry(
        *,
        raw_types: torch.Tensor,
        raw_sources: torch.Tensor,
        raw_valid: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        evidence_count: torch.Tensor,
        quota_clipped: torch.Tensor,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        device = cand_mask.device
        valid_slots = cand_mask.to(torch.float32).sum().clamp_min(1.0)
        raw_slots = raw_valid.to(torch.float32).sum().clamp_min(1.0)
        valid_rows = cand_mask.any(dim=-1)
        row_denom = valid_rows.to(torch.float32).sum().clamp_min(1.0)
        multi = (evidence_count > 1) & cand_mask
        hisa_slots = CandidateProvider._is_hisa_type(cand_types) & cand_mask
        hisa_frac_by_row = hisa_slots.to(torch.float32).sum(dim=-1) / cand_mask.to(torch.float32).sum(dim=-1).clamp_min(1.0)
        question_rows = ((cand_types == int(CandidateType.QUESTION)) & cand_mask).any(dim=-1)
        out: dict[str, torch.Tensor] = {
            "dsqg_w_candidate_multi_evidence_fraction": (multi.to(torch.float32).sum() / valid_slots).to(device=device, dtype=dtype),
            "dsqg_w_candidate_evidence_count_mean": (
                evidence_count.to(torch.float32).masked_select(cand_mask).mean() if cand_mask.any() else torch.zeros((), device=device)
            ).to(device=device, dtype=dtype),
            "dsqg_w_candidate_hisa_monopoly_row_fraction": ((hisa_frac_by_row >= 0.75) & valid_rows).to(torch.float32).sum().to(device=device, dtype=dtype) / row_denom.to(device=device, dtype=dtype),
            "dsqg_w_candidate_missing_question_row_fraction": (valid_rows & ~question_rows).to(torch.float32).sum().to(device=device, dtype=dtype) / row_denom.to(device=device, dtype=dtype),
            "dsqg_w_candidate_quota_hisa_clipped_fraction": quota_clipped.to(device=device, dtype=dtype) / raw_slots.to(device=device, dtype=dtype),
        }
        for source in CandidateSource:
            raw_mask = (raw_sources == int(source)) & raw_valid
            post_mask = (cand_sources == int(source)) & cand_mask
            out[f"dsqg_w_candidate_pre_source_fraction_{source.name.lower()}"] = (raw_mask.to(torch.float32).sum() / raw_slots).to(device=device, dtype=dtype)
            out[f"dsqg_w_candidate_post_source_fraction_{source.name.lower()}"] = (post_mask.to(torch.float32).sum() / valid_slots).to(device=device, dtype=dtype)
        for ctype in CandidateType:
            raw_mask = (raw_types == int(ctype)) & raw_valid
            out[f"dsqg_w_candidate_pre_fraction_{ctype.name.lower()}"] = (raw_mask.to(torch.float32).sum() / raw_slots).to(device=device, dtype=dtype)
        return out

    def _build_dsr_selected_metadata_fast(
        self,
        final_states: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        question_indices: torch.Tensor,
        hisa_evidence_indices: torch.Tensor,
        hisa_evidence_scores: torch.Tensor,
        l3_skip_indices: torch.Tensor,
    ) -> CandidateBatch:
        if final_states.ndim != 3:
            raise ValueError("final_states must have shape [B, T, D]")
        bsz, seq_len, d = final_states.shape
        if d != self.config.d:
            raise ValueError(f"final_states last dim {d} does not match config.d {self.config.d}")
        if l3_states is not None and l3_states.shape != final_states.shape:
            raise ValueError("l3_states must match final_states shape")
        device = final_states.device
        with _dsqg_w_profile_range("candidate_metadata/normalize_inputs"):
            positions = self._positions(bsz, seq_len, device)
            q_idx = self._normalize_index_tensor(question_indices, bsz, seq_len, self.config.k_question, device)
            h_idx = self._normalize_index_tensor(hisa_evidence_indices, bsz, seq_len, self.config.k_hisa_evidence, device)
            h_scores = self._normalize_score_tensor(hisa_evidence_scores, bsz, seq_len, 0 if h_idx is None else h_idx.shape[-1], device)
            s_idx = self._normalize_index_tensor(l3_skip_indices, bsz, seq_len, self.config.k_l3_skip, device)
        if q_idx is None or h_idx is None or h_scores is None or s_idx is None:
            return self._build_vectorized(
                final_states,
                l3_states=l3_states,
                question_indices=question_indices,
                hisa_evidence_indices=hisa_evidence_indices,
                hisa_evidence_scores=hisa_evidence_scores,
                l3_skip_indices=l3_skip_indices,
                materialize_states=False,
            )

        def group(tokens: torch.Tensor, ctype: int | torch.Tensor, source: int, scores: torch.Tensor | None = None):
            valid = (tokens >= 0) & (tokens <= positions)
            score_values = torch.zeros(tokens.shape, device=device, dtype=final_states.dtype) if scores is None else torch.nan_to_num(
                scores.to(device=device, dtype=final_states.dtype), nan=0.0, neginf=0.0, posinf=0.0
            )
            keep, dup = self._dedupe_same_source_group(tokens, score_values, valid)
            if torch.is_tensor(ctype):
                types = ctype.to(device=device, dtype=torch.long).expand_as(tokens)
            else:
                types = torch.full(tokens.shape, int(ctype), device=device, dtype=torch.long)
            sources = torch.full(tokens.shape, int(source), device=device, dtype=torch.long)
            return tokens, types, sources, keep, score_values, dup

        with _dsqg_w_profile_range("candidate_metadata/group_concat"):
            if self.config.typed_hisa_reps:
                rep_type_ids = self._typed_hisa_rep_type_ids(device)
                slot_ids = self._arange(h_idx.shape[-1], device)
                h_types = rep_type_ids[slot_ids.clamp_max(rep_type_ids.numel() - 1)].reshape(1, 1, -1).expand_as(h_idx)
            else:
                h_types = int(CandidateType.HISA_EVIDENCE)

            groups = [
                group(h_idx, h_types, int(CandidateSource.HISA), h_scores),
                group(q_idx, int(CandidateType.QUESTION), int(CandidateSource.FINAL), None),
                group(s_idx, int(CandidateType.L3_SKIP), int(CandidateSource.L3), None),
            ]
            tokens = torch.cat([g[0] for g in groups], dim=-1)
            types = torch.cat([g[1] for g in groups], dim=-1)
            sources = torch.cat([g[2] for g in groups], dim=-1)
            valid = torch.cat([g[3] for g in groups], dim=-1)
            scores = torch.cat([g[4] for g in groups], dim=-1)
            duplicate_count = sum((g[5] for g in groups), torch.zeros((), device=device, dtype=torch.long))
            had_valid = valid.any(dim=-1, keepdim=True)
            active_source_ids = {
                int(CandidateSource.FINAL),
                int(CandidateSource.HISA),
                int(CandidateSource.L3),
            }
            if self.config.null_fallback:
                active_source_ids.add(int(CandidateSource.NULL))
                null_tokens = positions
                null_valid = ~had_valid
                tokens = torch.cat([tokens, null_tokens], dim=-1)
                types = torch.cat([types, torch.full_like(null_tokens, int(CandidateType.NULL))], dim=-1)
                sources = torch.cat([sources, torch.full_like(null_tokens, int(CandidateSource.NULL))], dim=-1)
                valid = torch.cat([valid, null_valid], dim=-1)
                scores = torch.cat([scores, final_states.new_zeros(null_tokens.shape)], dim=-1)
        if tokens.shape[-1] == 0 or not valid.any(dim=-1).all():
            raise RuntimeError("CandidateProvider produced an all-invalid DSQG-W candidate row")

        with _dsqg_w_profile_range("candidate_metadata/collapse_sort"):
            raw_valid = valid
            raw_types = types
            raw_sources = sources
            # The DSR-selected fast path emits only disjoint source groups
            # (FINAL/QUESTION, HISA, L3, optional NULL) after same-source dedupe.
            # Collapsing by token+source is therefore an identity operation on
            # evidence bits, so avoid the O(R^2) same-key tensor and per-bit loop.
            evidence_bits_all = self._candidate_type_evidence_bits(types).masked_fill(~valid, 0)
            evidence_count_all = (evidence_bits_all != 0).to(torch.long).masked_fill(~valid, 0)
            candidate_distances_all = (positions - tokens).clamp_min(0).masked_fill(~valid, 0)

            priority_table = self._priority_table(device)
            priority = priority_table[types.clamp_min(0)]
            sort_stride_source = max(self.config.n_sources, len(CandidateSource)) + 1
            sort_stride_token = (seq_len + 1) * sort_stride_source
            sort_key = (
                priority.to(torch.float32) * float(sort_stride_token * 1_000_000)
                - scores.to(torch.float32) * float(sort_stride_token)
                + tokens.clamp_min(0).to(torch.float32) * float(sort_stride_source)
                + sources.clamp_min(0).to(torch.float32)
            )
            keep = valid
            keep, quota_clipped = self._apply_candidate_quotas(sort_key, keep, types)
            sort_key = sort_key.masked_fill(~keep, float("inf"))
            order_idx = sort_key.argsort(dim=-1)
            order_valid = torch.ones_like(order_idx, dtype=torch.bool)
            j_max = min(int(self.config.max_candidates), int(tokens.shape[-1]))
            order_idx = order_idx[..., :j_max]
            order_valid = order_valid[..., :j_max]

        with _dsqg_w_profile_range("candidate_metadata/gather_mask_finalize"):
            cand_token_indices = tokens.gather(-1, order_idx)
            cand_types = types.gather(-1, order_idx)
            cand_sources = sources.gather(-1, order_idx)
            cand_scores = scores.gather(-1, order_idx)
            evidence_bits = evidence_bits_all.gather(-1, order_idx)
            evidence_count = evidence_count_all.gather(-1, order_idx)
            candidate_distances = candidate_distances_all.gather(-1, order_idx)
            cand_mask = keep.gather(-1, order_idx) & order_valid
            valid_count = cand_mask.sum(dim=-1).to(torch.long)
            cand_token_indices = cand_token_indices.masked_fill(~cand_mask, -1)
            cand_types = cand_types.masked_fill(~cand_mask, int(CandidateType.NULL))
            cand_sources = cand_sources.masked_fill(~cand_mask, int(CandidateSource.NULL))
            cand_scores = cand_scores.masked_fill(~cand_mask, 0.0)
            evidence_bits = evidence_bits.masked_fill(~cand_mask, 0)
            evidence_count = evidence_count.masked_fill(~cand_mask, 0)
            candidate_distances = candidate_distances.masked_fill(~cand_mask, 0)

        with _dsqg_w_profile_range("candidate_metadata/telemetry"):
            raw_count = max(int(tokens.shape[-1]), 1)
            invalid_count = (~valid).sum()
            denom = torch.tensor(float(raw_count * bsz * seq_len), device=device, dtype=final_states.dtype).clamp_min(1.0)
            telemetry = {
                "dsqg_w_candidate_duplicate_rate": duplicate_count.to(final_states.dtype) / denom,
                "dsqg_w_candidate_invalid_rate": invalid_count.to(final_states.dtype) / denom,
                "dsqg_w_valid_candidate_count": valid_count.float().mean(),
                "dsqg_w_static_source_count": final_states.new_tensor(float(len(active_source_ids))),
                "dsqg_w_candidate_specialized_metadata": final_states.new_tensor(1.0),
                "dsqg_w_candidate_slot_count": final_states.new_tensor(float(j_max)),
            }
            if self._candidate_diagnostics_enabled():
                selected_scores = cand_scores.masked_select(cand_mask) if cand_mask.any() else None
                telemetry["dsqg_w_candidate_score_mean"] = (
                    selected_scores.mean() if selected_scores is not None else final_states.new_tensor(0.0)
                )
                telemetry["dsqg_w_candidate_score_max"] = (
                    selected_scores.max() if selected_scores is not None else final_states.new_tensor(0.0)
                )
                mask_denom = cand_mask.float().sum().clamp_min(1.0)
                for ctype in CandidateType:
                    mask = (cand_types == int(ctype)) & cand_mask
                    telemetry[f"dsqg_w_candidate_fraction_{ctype.name.lower()}"] = mask.float().sum() / mask_denom
                telemetry.update(
                    self._evidence_telemetry(
                        raw_types=raw_types,
                        raw_sources=raw_sources,
                        raw_valid=raw_valid,
                        cand_types=cand_types,
                        cand_sources=cand_sources,
                        cand_mask=cand_mask,
                        evidence_count=evidence_count,
                        quota_clipped=quota_clipped,
                        dtype=final_states.dtype,
                    )
                )
            if _dsqg_w_geometry_audit_enabled():
                telemetry.update(_dsqg_w_geometry_telemetry(cand_token_indices, cand_types, cand_sources, cand_mask))
        return CandidateBatch(
            cand_states=final_states.new_empty((0,)),
            cand_types=cand_types,
            cand_sources=cand_sources,
            cand_mask=cand_mask,
            cand_token_indices=cand_token_indices,
            valid_candidate_count=valid_count,
            cand_scores=cand_scores,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
            telemetry=telemetry,
            active_source_ids=tuple(sorted(active_source_ids)),
        )

    def _build_vectorized(
        self,
        final_states: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        question_indices: torch.Tensor | None = None,
        hisa_evidence_indices: torch.Tensor | None = None,
        hisa_evidence_scores: torch.Tensor | None = None,
        chunk_rep_indices: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        l3_skip_indices: torch.Tensor | None = None,
        materialize_states: bool = True,
    ) -> CandidateBatch:
        if final_states.ndim != 3:
            raise ValueError("final_states must have shape [B, T, D]")
        bsz, seq_len, d = final_states.shape
        if d != self.config.d:
            raise ValueError(f"final_states last dim {d} does not match config.d {self.config.d}")
        if l3_states is not None and l3_states.shape != final_states.shape:
            raise ValueError("l3_states must match final_states shape")
        if chunk_rep_states is not None and chunk_rep_states.shape != final_states.shape:
            raise ValueError("chunk_rep_states must match final_states shape")

        device = final_states.device
        positions = self._positions(bsz, seq_len, device)
        raw_tokens: list[torch.Tensor] = []
        raw_types: list[torch.Tensor] = []
        raw_sources: list[torch.Tensor] = []
        raw_valids: list[torch.Tensor] = []
        raw_scores: list[torch.Tensor] = []
        active_source_ids: set[int] = set()

        def add(
            tokens: torch.Tensor,
            ctype: int | torch.Tensor,
            source: int,
            valid: torch.Tensor | None = None,
            score: torch.Tensor | None = None,
        ) -> None:
            active_source_ids.add(int(source))
            tokens = tokens.to(device=device, dtype=torch.long)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(-1)
            if tokens.shape[0] == 1 and bsz != 1:
                tokens = tokens.expand(bsz, -1, -1)
            if tokens.shape[1] == 1 and seq_len != 1:
                tokens = tokens.expand(-1, seq_len, -1)
            if tokens.shape[:2] != (bsz, seq_len):
                raise ValueError("candidate index tensor must broadcast to [B,T,K]")
            valid_mask = (tokens >= 0) & (tokens <= positions)
            if valid is not None:
                valid_mask = valid_mask & valid.to(device=device, dtype=torch.bool)
            raw_tokens.append(tokens)
            if isinstance(ctype, torch.Tensor):
                type_values = ctype.to(device=device, dtype=torch.long)
                if type_values.ndim == 1:
                    type_values = type_values.reshape(1, 1, -1)
                elif type_values.ndim == 2:
                    type_values = type_values.unsqueeze(-1)
                if type_values.shape[0] == 1 and bsz != 1:
                    type_values = type_values.expand(bsz, -1, -1)
                if type_values.shape[1] == 1 and seq_len != 1:
                    type_values = type_values.expand(-1, seq_len, -1)
                if type_values.shape != tokens.shape:
                    raise ValueError("candidate type tensor must broadcast to candidate token tensor shape")
                raw_types.append(type_values)
            else:
                raw_types.append(torch.full(tokens.shape, int(ctype), device=device, dtype=torch.long))
            raw_sources.append(torch.full(tokens.shape, int(source), device=device, dtype=torch.long))
            raw_valids.append(valid_mask)
            if score is None:
                raw_scores.append(torch.zeros(tokens.shape, device=device, dtype=final_states.dtype))
            else:
                score_values = score.to(device=device, dtype=final_states.dtype)
                if score_values.ndim == 2:
                    score_values = score_values.unsqueeze(-1)
                if score_values.shape[0] == 1 and bsz != 1:
                    score_values = score_values.expand(bsz, -1, -1)
                if score_values.shape[1] == 1 and seq_len != 1:
                    score_values = score_values.expand(-1, seq_len, -1)
                if score_values.shape != tokens.shape:
                    raise ValueError("candidate score tensor must broadcast to candidate token tensor shape")
                raw_scores.append(torch.nan_to_num(score_values, nan=0.0, neginf=0.0, posinf=0.0))

        for offset in self.config.local_offsets:
            add(positions - int(offset), int(CandidateType.LOCAL), int(CandidateSource.FINAL))
        q_idx = self._normalize_index_tensor(question_indices, bsz, seq_len, self.config.k_question, device)
        if q_idx is not None:
            add(q_idx, int(CandidateType.QUESTION), int(CandidateSource.FINAL))
        h_idx = self._normalize_index_tensor(hisa_evidence_indices, bsz, seq_len, self.config.k_hisa_evidence, device)
        h_scores = self._normalize_score_tensor(hisa_evidence_scores, bsz, seq_len, 0 if h_idx is None else h_idx.shape[-1], device)
        if h_idx is not None:
            if self.config.typed_hisa_reps:
                rep_type_ids = self._typed_hisa_rep_type_ids(device)
                n_rep = min(h_idx.shape[-1], rep_type_ids.numel())
                if n_rep > 0:
                    type_ids = rep_type_ids[:n_rep].reshape(1, 1, n_rep)
                    add(h_idx[..., :n_rep], type_ids, int(CandidateSource.HISA), score=None if h_scores is None else h_scores[..., :n_rep])
                if h_idx.shape[-1] > n_rep:
                    add(h_idx[..., n_rep:], int(CandidateType.HISA_EVIDENCE), int(CandidateSource.HISA), score=None if h_scores is None else h_scores[..., n_rep:])
            else:
                add(h_idx, int(CandidateType.HISA_EVIDENCE), int(CandidateSource.HISA), score=h_scores)
        for offset in self.config.long_offsets:
            add(positions - int(offset), int(CandidateType.LONG_OFFSET), int(CandidateSource.FINAL))
        c_idx = self._normalize_index_tensor(chunk_rep_indices, bsz, seq_len, self.config.k_chunk, device)
        if c_idx is not None:
            add(c_idx, int(CandidateType.CHUNK_REP), int(CandidateSource.SUMMARY))
        s_idx = self._normalize_index_tensor(l3_skip_indices, bsz, seq_len, self.config.k_l3_skip, device)
        if s_idx is not None:
            add(s_idx, int(CandidateType.L3_SKIP), int(CandidateSource.L3))

        if raw_tokens:
            tokens = torch.cat(raw_tokens, dim=-1)
            types = torch.cat(raw_types, dim=-1)
            sources = torch.cat(raw_sources, dim=-1)
            valid = torch.cat(raw_valids, dim=-1)
            scores = torch.cat(raw_scores, dim=-1)
        else:
            tokens = positions.new_empty((bsz, seq_len, 0))
            types = positions.new_empty((bsz, seq_len, 0))
            sources = positions.new_empty((bsz, seq_len, 0))
            valid = torch.zeros((bsz, seq_len, 0), device=device, dtype=torch.bool)
            scores = final_states.new_empty((bsz, seq_len, 0))

        had_valid = valid.any(dim=-1, keepdim=True)
        if self.config.null_fallback:
            active_source_ids.add(int(CandidateSource.NULL))
            null_tokens = positions
            null_valid = ~had_valid
            tokens = torch.cat([tokens, null_tokens], dim=-1)
            types = torch.cat([types, torch.full_like(null_tokens, int(CandidateType.NULL))], dim=-1)
            sources = torch.cat([sources, torch.full_like(null_tokens, int(CandidateSource.NULL))], dim=-1)
            valid = torch.cat([valid, null_valid], dim=-1)
            scores = torch.cat([scores, final_states.new_zeros(null_tokens.shape)], dim=-1)

        if tokens.shape[-1] == 0 or not valid.any(dim=-1).all():
            raise RuntimeError("CandidateProvider produced an all-invalid DSQG-W candidate row")

        priority_table = self._priority_table(device)
        priority = priority_table[types.clamp_min(0)]
        raw_count = max(int(tokens.shape[-1]), 1)
        invalid_count = (~valid).sum()

        same_key = (tokens.unsqueeze(-1) == tokens.unsqueeze(-2)) & (sources.unsqueeze(-1) == sources.unsqueeze(-2))
        same_key = same_key & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        r_count = tokens.shape[-1]
        order = self._arange(r_count, device)
        priority_j = priority.unsqueeze(-2)
        min_priority = priority_j.masked_fill(~same_key, 99).amin(dim=-1)
        score_j = scores.unsqueeze(-2)
        max_score_same_key = score_j.masked_fill(~same_key, float("-inf")).amax(dim=-1)
        earlier_same_key = same_key & (order.reshape(1, 1, 1, r_count) < order.reshape(1, 1, r_count, 1))
        duplicate_mask = valid & earlier_same_key.any(dim=-1)
        same_min_priority = earlier_same_key & (priority_j == priority.unsqueeze(-1))
        same_min_priority_higher_or_equal_score = same_min_priority & (score_j >= scores.unsqueeze(-1))
        keep = valid & (priority == min_priority) & (scores >= max_score_same_key) & ~same_min_priority_higher_or_equal_score.any(dim=-1)
        duplicate_count = duplicate_mask.sum()
        raw_valid = valid
        raw_types = types
        raw_sources = sources
        evidence_bits_all, evidence_count_all = self._collapse_evidence_by_token_source(
            tokens,
            sources,
            self._candidate_type_evidence_bits(types),
            valid,
        )
        candidate_distances_all = (positions - tokens).clamp_min(0).masked_fill(~valid, 0)

        sort_stride_source = max(self.config.n_sources, len(CandidateSource)) + 1
        sort_stride_token = (seq_len + 1) * sort_stride_source
        sort_key = (
            priority.to(torch.float64) * float(sort_stride_token * 1_000_000)
            - scores.to(torch.float64) * float(sort_stride_token)
            + tokens.clamp_min(0).to(torch.float64) * float(sort_stride_source)
            + sources.clamp_min(0).to(torch.float64)
        )
        keep, quota_clipped = self._apply_candidate_quotas(sort_key, keep, types)
        sort_key = sort_key.masked_fill(~keep, float("inf"))
        order_idx = sort_key.argsort(dim=-1)
        order_valid = torch.ones_like(order_idx, dtype=torch.bool)
        j_max = self.config.max_candidates
        if order_idx.shape[-1] < j_max:
            pad = order_idx.new_zeros((*order_idx.shape[:-1], j_max - order_idx.shape[-1]))
            order_idx = torch.cat([order_idx, pad], dim=-1)
            order_valid = torch.cat([order_valid, torch.zeros_like(pad, dtype=torch.bool)], dim=-1)
        order_idx = order_idx[..., :j_max]
        order_valid = order_valid[..., :j_max]

        cand_token_indices = tokens.gather(-1, order_idx)
        cand_types = types.gather(-1, order_idx)
        cand_sources = sources.gather(-1, order_idx)
        cand_scores = scores.gather(-1, order_idx)
        evidence_bits = evidence_bits_all.gather(-1, order_idx)
        evidence_count = evidence_count_all.gather(-1, order_idx)
        candidate_distances = candidate_distances_all.gather(-1, order_idx)
        cand_mask = keep.gather(-1, order_idx) & order_valid
        valid_count = cand_mask.sum(dim=-1).to(torch.long)
        cand_token_indices = cand_token_indices.masked_fill(~cand_mask, -1)
        cand_types = cand_types.masked_fill(~cand_mask, int(CandidateType.NULL))
        cand_sources = cand_sources.masked_fill(~cand_mask, int(CandidateSource.NULL))
        cand_scores = cand_scores.masked_fill(~cand_mask, 0.0)
        evidence_bits = evidence_bits.masked_fill(~cand_mask, 0)
        evidence_count = evidence_count.masked_fill(~cand_mask, 0)
        candidate_distances = candidate_distances.masked_fill(~cand_mask, 0)

        if materialize_states:
            gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
            final_source = (cand_sources == int(CandidateSource.FINAL)) | (cand_sources == int(CandidateSource.QUESTION_CACHE))
            l3_source = (cand_sources == int(CandidateSource.L3)) | (cand_sources == int(CandidateSource.HISA))
            summary_source = cand_sources == int(CandidateSource.SUMMARY)
            final_needed = bool(self.config.local_offsets or self.config.long_offsets or q_idx is not None)
            l3_needed = h_idx is not None or s_idx is not None
            summary_needed = c_idx is not None

            final_gather = self._gather_states(final_states, gather_tokens) if final_needed else None
            cand_states = torch.zeros((bsz, seq_len, j_max, d), device=device, dtype=final_states.dtype)
            if final_gather is not None:
                cand_states = torch.where(final_source[..., None], final_gather, cand_states)
            if l3_needed:
                l3_base = l3_states if l3_states is not None else final_states
                l3_gather = final_gather if l3_base is final_states and final_gather is not None else self._gather_states(l3_base, gather_tokens)
                cand_states = torch.where(l3_source[..., None], l3_gather, cand_states)
            if summary_needed:
                summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
                summary_gather = final_gather if summary_base is final_states and final_gather is not None else self._gather_states(summary_base, gather_tokens)
                cand_states = torch.where(summary_source[..., None], summary_gather, cand_states)
            cand_states = cand_states * cand_mask[..., None].to(cand_states.dtype)
        else:
            cand_states = final_states.new_empty((0,))

        denom = torch.tensor(float(raw_count * bsz * seq_len), device=device, dtype=final_states.dtype).clamp_min(1.0)
        telemetry = {
            "dsqg_w_candidate_duplicate_rate": duplicate_count.to(final_states.dtype) / denom,
            "dsqg_w_candidate_invalid_rate": invalid_count.to(final_states.dtype) / denom,
            "dsqg_w_valid_candidate_count": valid_count.float().mean(),
            "dsqg_w_static_source_count": final_states.new_tensor(float(len(active_source_ids))),
            "dsqg_w_candidate_specialized_metadata": final_states.new_tensor(0.0),
            "dsqg_w_candidate_slot_count": final_states.new_tensor(float(j_max)),
        }
        if self._candidate_diagnostics_enabled():
            selected_scores = cand_scores.masked_select(cand_mask) if cand_mask.any() else None
            telemetry["dsqg_w_candidate_score_mean"] = (
                selected_scores.mean() if selected_scores is not None else final_states.new_tensor(0.0)
            )
            telemetry["dsqg_w_candidate_score_max"] = (
                selected_scores.max() if selected_scores is not None else final_states.new_tensor(0.0)
            )
            mask_denom = cand_mask.float().sum().clamp_min(1.0)
            for ctype in CandidateType:
                mask = (cand_types == int(ctype)) & cand_mask
                telemetry[f"dsqg_w_candidate_fraction_{ctype.name.lower()}"] = mask.float().sum() / mask_denom
            telemetry.update(
                self._evidence_telemetry(
                    raw_types=raw_types,
                    raw_sources=raw_sources,
                    raw_valid=raw_valid,
                    cand_types=cand_types,
                    cand_sources=cand_sources,
                    cand_mask=cand_mask,
                    evidence_count=evidence_count,
                    quota_clipped=quota_clipped,
                    dtype=final_states.dtype,
                )
            )
        if _dsqg_w_geometry_audit_enabled():
            telemetry.update(
                _dsqg_w_geometry_telemetry(
                    cand_token_indices,
                    cand_types,
                    cand_sources,
                    cand_mask,
                )
            )

        return CandidateBatch(
            cand_states=cand_states,
            cand_types=cand_types,
            cand_sources=cand_sources,
            cand_mask=cand_mask,
            cand_token_indices=cand_token_indices,
            valid_candidate_count=valid_count,
            cand_scores=cand_scores,
            evidence_bits=evidence_bits,
            evidence_count=evidence_count,
            candidate_distances=candidate_distances,
            telemetry=telemetry,
            active_source_ids=tuple(sorted(active_source_ids)),
        )

    @staticmethod
    def _normalize_index_tensor(
        indices: torch.Tensor | None,
        bsz: int,
        seq_len: int,
        limit: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if indices is None or limit <= 0:
            return None
        values = indices.to(device=device, dtype=torch.long)
        if values.ndim == 1:
            values = values[:limit].reshape(1, 1, -1).expand(bsz, seq_len, -1)
        elif values.ndim == 2:
            values = values[:, :limit].reshape(values.shape[0], 1, -1).expand(-1, seq_len, -1)
        elif values.ndim == 3:
            values = values[:, :, :limit]
        else:
            raise ValueError("candidate index tensors must be rank 1, 2, or 3")
        if values.shape[0] == 1 and bsz != 1:
            values = values.expand(bsz, -1, -1)
        if values.shape[1] == 1 and seq_len != 1:
            values = values.expand(-1, seq_len, -1)
        if values.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate index tensor must broadcast to [B,T,K]")
        return values

    @staticmethod
    def _normalize_score_tensor(
        scores: torch.Tensor | None,
        bsz: int,
        seq_len: int,
        limit: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if scores is None or limit <= 0:
            return None
        values = scores.to(device=device)
        if values.ndim == 1:
            values = values[:limit].reshape(1, 1, -1).expand(bsz, seq_len, -1)
        elif values.ndim == 2:
            values = values[:, :limit].reshape(values.shape[0], 1, -1).expand(-1, seq_len, -1)
        elif values.ndim == 3:
            values = values[:, :, :limit]
        else:
            raise ValueError("candidate score tensors must be rank 1, 2, or 3")
        if values.shape[0] == 1 and bsz != 1:
            values = values.expand(bsz, -1, -1)
        if values.shape[1] == 1 and seq_len != 1:
            values = values.expand(-1, seq_len, -1)
        if values.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate score tensor must broadcast to [B,T,K]")
        return values[..., :limit]

    @staticmethod
    def _gather_states(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = states.shape
        batch_offsets = torch.arange(bsz, device=states.device, dtype=torch.long).reshape(bsz, 1, 1) * seq_len
        flat_indices = (batch_offsets + token_indices.to(torch.long)).reshape(-1)
        return states.reshape(bsz * seq_len, d).index_select(0, flat_indices).reshape(*token_indices.shape, d)

    def _build_reference(
        self,
        final_states: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        question_indices: torch.Tensor | list[list[int]] | None = None,
        hisa_evidence_indices: torch.Tensor | None = None,
        hisa_evidence_scores: torch.Tensor | None = None,
        chunk_rep_indices: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        l3_skip_indices: torch.Tensor | None = None,
    ) -> CandidateBatch:
        if final_states.ndim != 3:
            raise ValueError("final_states must have shape [B, T, D]")
        bsz, seq_len, d = final_states.shape
        if d != self.config.d:
            raise ValueError(f"final_states last dim {d} does not match config.d {self.config.d}")
        if l3_states is not None and l3_states.shape != final_states.shape:
            raise ValueError("l3_states must match final_states shape")
        if chunk_rep_states is not None and chunk_rep_states.shape != final_states.shape:
            raise ValueError("chunk_rep_states must match final_states shape")

        device = final_states.device
        source_states: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_states if l3_states is not None else final_states,
            int(CandidateSource.HISA): l3_states if l3_states is not None else final_states,
            int(CandidateSource.SUMMARY): chunk_rep_states if chunk_rep_states is not None else final_states,
            int(CandidateSource.NULL): torch.zeros_like(final_states),
        }

        j_max = self.config.max_candidates
        cand_states = final_states.new_zeros((bsz, seq_len, j_max, d))
        cand_types = torch.full((bsz, seq_len, j_max), int(CandidateType.NULL), device=device, dtype=torch.long)
        cand_sources = torch.full((bsz, seq_len, j_max), int(CandidateSource.NULL), device=device, dtype=torch.long)
        cand_mask = torch.zeros((bsz, seq_len, j_max), device=device, dtype=torch.bool)
        cand_token_indices = torch.full((bsz, seq_len, j_max), -1, device=device, dtype=torch.long)
        valid_count = torch.zeros((bsz, seq_len), device=device, dtype=torch.long)

        raw_count = 0
        invalid_count = 0
        duplicate_count = 0

        for b in range(bsz):
            for t in range(seq_len):
                dedup: dict[tuple[int, int], Candidate] = {}

                def consider(token_index: int, source: int, ctype: int, offset: int | None = None) -> None:
                    nonlocal raw_count, invalid_count, duplicate_count
                    raw_count += 1
                    valid = 0 <= int(token_index) <= t
                    if not valid:
                        invalid_count += 1
                        return
                    cand = Candidate(int(token_index), int(source), int(ctype), None if offset is None else int(offset), True)
                    key = (cand.token_index, cand.source_layer)
                    prev = dedup.get(key)
                    if prev is not None:
                        duplicate_count += 1
                        if _CANDIDATE_PRIORITY[cand.candidate_type] < _CANDIDATE_PRIORITY[prev.candidate_type]:
                            dedup[key] = cand
                    else:
                        dedup[key] = cand

                for offset in self.config.local_offsets:
                    consider(t - int(offset), int(CandidateSource.FINAL), int(CandidateType.LOCAL), int(offset))

                for q_idx in self._indices_for_position(question_indices, b, t, self.config.k_question):
                    # QUESTION intentionally uses FINAL source so cue/local duplicates
                    # are de-duplicated by token/layer with QUESTION priority.
                    consider(q_idx, int(CandidateSource.FINAL), int(CandidateType.QUESTION), None)

                for h_idx in self._indices_for_position(hisa_evidence_indices, b, t, self.config.k_hisa_evidence):
                    consider(h_idx, int(CandidateSource.HISA), int(CandidateType.HISA_EVIDENCE), None)

                for offset in self.config.long_offsets:
                    consider(t - int(offset), int(CandidateSource.FINAL), int(CandidateType.LONG_OFFSET), int(offset))

                for c_idx in self._indices_for_position(chunk_rep_indices, b, t, self.config.k_chunk):
                    consider(c_idx, int(CandidateSource.SUMMARY), int(CandidateType.CHUNK_REP), None)

                for l3_idx in self._indices_for_position(l3_skip_indices, b, t, self.config.k_l3_skip):
                    consider(l3_idx, int(CandidateSource.L3), int(CandidateType.L3_SKIP), None)

                if not dedup and self.config.null_fallback:
                    consider(t, int(CandidateSource.NULL), int(CandidateType.NULL), None)

                ordered = sorted(
                    dedup.values(),
                    key=lambda c: (_CANDIDATE_PRIORITY[c.candidate_type], c.token_index, c.source_layer),
                )[:j_max]
                valid_count[b, t] = len(ordered)
                for j, cand in enumerate(ordered):
                    cand_mask[b, t, j] = True
                    cand_types[b, t, j] = cand.candidate_type
                    cand_sources[b, t, j] = cand.source_layer
                    cand_token_indices[b, t, j] = cand.token_index
                    cand_states[b, t, j] = source_states[cand.source_layer][b, cand.token_index]

        if not cand_mask.any(dim=-1).all():
            raise RuntimeError("CandidateProvider produced an all-invalid DSQG-W candidate row")

        denom = max(raw_count, 1)
        telemetry = {
            "dsqg_w_candidate_duplicate_rate": final_states.new_tensor(float(duplicate_count) / float(denom)),
            "dsqg_w_candidate_invalid_rate": final_states.new_tensor(float(invalid_count) / float(denom)),
            "dsqg_w_valid_candidate_count": valid_count.float().mean(),
            "dsqg_w_static_source_count": final_states.new_tensor(float(torch.unique(cand_sources[cand_mask]).numel())),
        }
        if self._candidate_diagnostics_enabled():
            for ctype in CandidateType:
                mask = (cand_types == int(ctype)) & cand_mask
                telemetry[f"dsqg_w_candidate_fraction_{ctype.name.lower()}"] = mask.float().sum() / cand_mask.float().sum().clamp_min(1.0)
        if _dsqg_w_geometry_audit_enabled():
            telemetry.update(
                _dsqg_w_geometry_telemetry(
                    cand_token_indices,
                    cand_types,
                    cand_sources,
                    cand_mask,
                )
            )

        return CandidateBatch(
            cand_states=cand_states,
            cand_types=cand_types,
            cand_sources=cand_sources,
            cand_mask=cand_mask,
            cand_token_indices=cand_token_indices,
            valid_candidate_count=valid_count,
            telemetry=telemetry,
            active_source_ids=tuple(int(v) for v in sorted(torch.unique(cand_sources[cand_mask]).detach().cpu().tolist())),
        )

    @staticmethod
    def _indices_for_position(
        indices: torch.Tensor | list[list[int]] | None,
        batch_index: int,
        query_position: int,
        limit: int,
    ) -> list[int]:
        if indices is None or limit <= 0:
            return []
        if isinstance(indices, torch.Tensor):
            if indices.ndim == 1:
                vals = indices.detach().cpu().tolist()
            elif indices.ndim == 2:
                vals = indices[batch_index].detach().cpu().tolist()
            elif indices.ndim == 3:
                vals = indices[batch_index, query_position].detach().cpu().tolist()
            else:
                raise ValueError("candidate index tensors must be rank 1, 2, or 3")
        else:
            vals = indices[batch_index]
        out: list[int] = []
        for val in vals:
            idx = int(val)
            if idx < 0:
                continue
            if idx <= query_position:
                out.append(idx)
            if len(out) >= limit:
                break
        return out

__all__ = ["CandidateProvider"]
