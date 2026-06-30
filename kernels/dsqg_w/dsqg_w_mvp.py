from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateType(IntEnum):
    NULL = 0
    LOCAL = 1
    QUESTION = 2
    HISA_EVIDENCE = 3
    LONG_OFFSET = 4
    CHUNK_REP = 5
    L3_SKIP = 6
    HISA_EVIDENCE_REP0 = 7
    HISA_EVIDENCE_REP1 = 8
    HISA_EVIDENCE_REP2 = 9
    HISA_EVIDENCE_REP3 = 10


class CandidateSource(IntEnum):
    NULL = 0
    FINAL = 1
    L3 = 2
    HISA = 3
    SUMMARY = 4
    QUESTION_CACHE = 5


# Lower is better.  This matches the DWARF v2 proposal: semantic evidence/cues
# replace duplicate local/long routes instead of letting them inflate mass.
_CANDIDATE_PRIORITY: dict[int, int] = {
    int(CandidateType.HISA_EVIDENCE): 0,
    int(CandidateType.HISA_EVIDENCE_REP0): 0,
    int(CandidateType.HISA_EVIDENCE_REP1): 0,
    int(CandidateType.HISA_EVIDENCE_REP2): 0,
    int(CandidateType.HISA_EVIDENCE_REP3): 0,
    int(CandidateType.QUESTION): 1,
    int(CandidateType.CHUNK_REP): 2,
    int(CandidateType.L3_SKIP): 3,
    int(CandidateType.LONG_OFFSET): 4,
    int(CandidateType.LOCAL): 5,
    int(CandidateType.NULL): 6,
}


@dataclass(frozen=True)
class DSQGWConfig:
    d: int
    n_heads: int
    n_types: int = len(CandidateType)
    n_sources: int = len(CandidateSource)
    bottleneck: int = 256
    max_candidates: int = 32
    gate_init: float = -5.0
    fuse_init_std: float = 1e-4
    local_offsets: tuple[int, ...] = (1, 2, 4, 8)
    long_offsets: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048)
    k_question: int = 4
    k_hisa_evidence: int = 8
    k_chunk: int = 4
    k_l3_skip: int = 4
    null_fallback: bool = True
    local_type_id: int = int(CandidateType.LOCAL)
    use_width_cell: bool = False
    width_bottleneck: int = 64
    width_gate_init: float = -5.0
    width_self_bias_init: float = 0.0
    width_entropy_floor: float = 0.0
    width_entropy_weight: float = 0.0
    use_typed_mixer: bool = False
    typed_mixer_bottleneck: int = 64
    typed_mixer_gate_init: float = -5.0
    use_query_type_bias: bool = False
    typed_hisa_reps: bool = False

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive")
        if self.n_heads <= 0 or self.d % self.n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.fuse_init_std < 0.0:
            raise ValueError("fuse_init_std must be non-negative")
        if self.n_types < len(CandidateType):
            raise ValueError("n_types must cover all CandidateType values")
        if self.n_sources < len(CandidateSource):
            raise ValueError("n_sources must cover all CandidateSource values")
        if self.width_bottleneck <= 0:
            raise ValueError("width_bottleneck must be positive")
        if self.width_entropy_floor < 0.0:
            raise ValueError("width_entropy_floor must be non-negative")
        if self.width_entropy_weight < 0.0:
            raise ValueError("width_entropy_weight must be non-negative")
        if self.typed_mixer_bottleneck <= 0:
            raise ValueError("typed_mixer_bottleneck must be positive")


@dataclass(frozen=True)
class Candidate:
    token_index: int
    source_layer: int
    candidate_type: int
    offset: int | None
    valid: bool = True


@dataclass(frozen=True)
class CandidateBatch:
    cand_states: torch.Tensor
    cand_types: torch.Tensor
    cand_sources: torch.Tensor
    cand_mask: torch.Tensor
    cand_token_indices: torch.Tensor
    valid_candidate_count: torch.Tensor
    cand_scores: torch.Tensor | None = None
    telemetry: dict[str, torch.Tensor] = field(default_factory=dict)


class CandidateProvider:
    """Bounded causal heterogeneous candidate construction for DSQG-W.

    This is intentionally diagnostic and explicit, not fused.  It constructs only
    O(B*T*J) candidate tensors and rejects future-token routes.  The default MVP
    supports LOCAL, QUESTION, HISA_EVIDENCE/L3_SKIP, LONG_OFFSET, CHUNK_REP, and
    NULL fallback candidates.
    """

    def __init__(self, config: DSQGWConfig):
        self.config = config

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
        )

    @staticmethod
    def _can_use_vectorized(**kwargs) -> bool:
        return all(value is None or isinstance(value, torch.Tensor) for value in kwargs.values())

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
        positions = torch.arange(seq_len, device=device, dtype=torch.long).reshape(1, seq_len, 1).expand(bsz, -1, -1)
        raw_tokens: list[torch.Tensor] = []
        raw_types: list[torch.Tensor] = []
        raw_sources: list[torch.Tensor] = []
        raw_valids: list[torch.Tensor] = []
        raw_scores: list[torch.Tensor] = []

        def add(
            tokens: torch.Tensor,
            ctype: int | torch.Tensor,
            source: int,
            valid: torch.Tensor | None = None,
            score: torch.Tensor | None = None,
        ) -> None:
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
                rep_types = [
                    int(CandidateType.HISA_EVIDENCE_REP0),
                    int(CandidateType.HISA_EVIDENCE_REP1),
                    int(CandidateType.HISA_EVIDENCE_REP2),
                    int(CandidateType.HISA_EVIDENCE_REP3),
                ]
                n_rep = min(h_idx.shape[-1], len(rep_types))
                if n_rep > 0:
                    type_ids = torch.tensor(rep_types[:n_rep], device=device, dtype=torch.long).reshape(1, 1, n_rep)
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
            null_tokens = positions
            null_valid = ~had_valid
            tokens = torch.cat([tokens, null_tokens], dim=-1)
            types = torch.cat([types, torch.full_like(null_tokens, int(CandidateType.NULL))], dim=-1)
            sources = torch.cat([sources, torch.full_like(null_tokens, int(CandidateSource.NULL))], dim=-1)
            valid = torch.cat([valid, null_valid], dim=-1)
            scores = torch.cat([scores, final_states.new_zeros(null_tokens.shape)], dim=-1)

        if tokens.shape[-1] == 0 or not valid.any(dim=-1).all():
            raise RuntimeError("CandidateProvider produced an all-invalid DSQG-W candidate row")

        priority_table = torch.full((max(self.config.n_types, len(CandidateType)),), 99, device=device, dtype=torch.long)
        for ctype, priority_value in _CANDIDATE_PRIORITY.items():
            priority_table[int(ctype)] = int(priority_value)
        priority = priority_table[types.clamp_min(0)]
        raw_count = max(int(tokens.shape[-1]), 1)
        invalid_count = (~valid).sum()

        same_key = (tokens.unsqueeze(-1) == tokens.unsqueeze(-2)) & (sources.unsqueeze(-1) == sources.unsqueeze(-2))
        same_key = same_key & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        r_count = tokens.shape[-1]
        order = torch.arange(r_count, device=device, dtype=torch.long)
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

        sort_stride_source = max(self.config.n_sources, len(CandidateSource)) + 1
        sort_stride_token = (seq_len + 1) * sort_stride_source
        sort_key = (
            priority.to(torch.float64) * float(sort_stride_token * 1_000_000)
            - scores.to(torch.float64) * float(sort_stride_token)
            + tokens.clamp_min(0).to(torch.float64) * float(sort_stride_source)
            + sources.clamp_min(0).to(torch.float64)
        )
        sort_key = sort_key.masked_fill(~keep, float("inf"))
        order_idx = sort_key.argsort(dim=-1)
        j_max = self.config.max_candidates
        if order_idx.shape[-1] < j_max:
            pad = order_idx.new_zeros((*order_idx.shape[:-1], j_max - order_idx.shape[-1]))
            order_idx = torch.cat([order_idx, pad], dim=-1)
        order_idx = order_idx[..., :j_max]

        cand_token_indices = tokens.gather(-1, order_idx)
        cand_types = types.gather(-1, order_idx)
        cand_sources = sources.gather(-1, order_idx)
        cand_scores = scores.gather(-1, order_idx)
        cand_mask = keep.gather(-1, order_idx)
        valid_count = cand_mask.sum(dim=-1).to(torch.long)
        cand_token_indices = cand_token_indices.masked_fill(~cand_mask, -1)
        cand_types = cand_types.masked_fill(~cand_mask, int(CandidateType.NULL))
        cand_sources = cand_sources.masked_fill(~cand_mask, int(CandidateSource.NULL))
        cand_scores = cand_scores.masked_fill(~cand_mask, 0.0)

        gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
        final_gather = self._gather_states(final_states, gather_tokens)
        l3_base = l3_states if l3_states is not None else final_states
        l3_gather = self._gather_states(l3_base, gather_tokens)
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        summary_gather = self._gather_states(summary_base, gather_tokens)
        cand_states = torch.zeros((bsz, seq_len, j_max, d), device=device, dtype=final_states.dtype)
        final_source = (cand_sources == int(CandidateSource.FINAL)) | (cand_sources == int(CandidateSource.QUESTION_CACHE))
        l3_source = (cand_sources == int(CandidateSource.L3)) | (cand_sources == int(CandidateSource.HISA))
        summary_source = cand_sources == int(CandidateSource.SUMMARY)
        cand_states = torch.where(final_source[..., None], final_gather, cand_states)
        cand_states = torch.where(l3_source[..., None], l3_gather, cand_states)
        cand_states = torch.where(summary_source[..., None], summary_gather, cand_states)
        cand_states = cand_states * cand_mask[..., None].to(cand_states.dtype)

        denom = torch.tensor(float(raw_count * bsz * seq_len), device=device, dtype=final_states.dtype).clamp_min(1.0)
        telemetry = {
            "dsqg_w_candidate_duplicate_rate": duplicate_count.to(final_states.dtype) / denom,
            "dsqg_w_candidate_invalid_rate": invalid_count.to(final_states.dtype) / denom,
            "dsqg_w_valid_candidate_count": valid_count.float().mean(),
            "dsqg_w_candidate_score_mean": cand_scores.masked_select(cand_mask).mean() if cand_mask.any() else final_states.new_tensor(0.0),
            "dsqg_w_candidate_score_max": cand_scores.masked_select(cand_mask).max() if cand_mask.any() else final_states.new_tensor(0.0),
        }
        mask_denom = cand_mask.float().sum().clamp_min(1.0)
        for ctype in CandidateType:
            mask = (cand_types == int(ctype)) & cand_mask
            telemetry[f"dsqg_w_candidate_fraction_{ctype.name.lower()}"] = mask.float().sum() / mask_denom

        return CandidateBatch(
            cand_states=cand_states,
            cand_types=cand_types,
            cand_sources=cand_sources,
            cand_mask=cand_mask,
            cand_token_indices=cand_token_indices,
            valid_candidate_count=valid_count,
            cand_scores=cand_scores,
            telemetry=telemetry,
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
        }
        for ctype in CandidateType:
            mask = (cand_types == int(ctype)) & cand_mask
            telemetry[f"dsqg_w_candidate_fraction_{ctype.name.lower()}"] = mask.float().sum() / cand_mask.float().sum().clamp_min(1.0)

        return CandidateBatch(
            cand_states=cand_states,
            cand_types=cand_types,
            cand_sources=cand_sources,
            cand_mask=cand_mask,
            cand_token_indices=cand_token_indices,
            valid_candidate_count=valid_count,
            telemetry=telemetry,
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


def width_pair_transfer_loss(
    probs: torch.Tensor,
    cand_types: torch.Tensor,
    cand_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
    entropy_floor: float = 0.0,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """Directional width-transfer loss for QUESTION <-> HISA_EVIDENCE lateral mass."""
    if probs.ndim == 5:
        p = probs.mean(dim=-1)
    elif probs.ndim == 4:
        p = probs
    else:
        raise ValueError("probs must have shape [B,T,J,J] or [B,T,J,J,H]")
    if cand_types.shape != cand_mask.shape or p.shape[:3] != cand_types.shape or p.shape[3] != cand_types.shape[2]:
        raise ValueError("candidate tensors must align with probs [B,T,J,J]")
    valid_targets = cand_mask.bool()

    def direction_mass(target_type: CandidateType, source_type: CandidateType) -> torch.Tensor:
        target_mask = (cand_types == int(target_type)) & valid_targets
        source_mask = (cand_types == int(source_type)) & cand_mask.bool()
        if not target_mask.any():
            return p.sum() * 0.0
        mass = p.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
        selected = mass.masked_select(target_mask)
        if selected.numel() == 0:
            return p.sum() * 0.0
        return selected.mean()

    q_to_hisa = direction_mass(CandidateType.QUESTION, CandidateType.HISA_EVIDENCE)
    hisa_to_q = direction_mass(CandidateType.HISA_EVIDENCE, CandidateType.QUESTION)
    transfer_loss = -0.5 * (
        torch.log(q_to_hisa.clamp_min(float(eps)))
        + torch.log(hisa_to_q.clamp_min(float(eps)))
    )
    if entropy_weight <= 0.0 or entropy_floor <= 0.0:
        return transfer_loss
    p_safe = p.clamp_min(float(eps))
    entropy = -(p_safe * p_safe.log()).sum(dim=-1).masked_select(valid_targets).mean()
    entropy_penalty = torch.relu(entropy.new_tensor(float(entropy_floor)) - entropy)
    return transfer_loss + float(entropy_weight) * entropy_penalty


class DSQGWWidthCell(nn.Module):
    """Bounded candidate-to-candidate semantic transfer over the DSQG-W width axis."""

    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        n_types: int,
        n_sources: int,
        bottleneck: int,
        gate_init: float = -5.0,
        self_bias_init: float = 0.0,
        entropy_floor: float = 0.0,
        entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.width_dim = int(bottleneck)
        if self.width_dim % self.n_heads != 0:
            raise ValueError("width bottleneck must be divisible by n_heads")
        self.n_types = int(n_types)
        self.n_sources = int(n_sources)
        self.entropy_floor = float(entropy_floor)
        self.entropy_weight = float(entropy_weight)

        self.norm_c = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, self.width_dim, bias=False)
        self.k_proj = nn.Linear(d, self.width_dim, bias=False)
        self.v_proj = nn.Linear(d, self.width_dim, bias=False)
        self.lateral_up = nn.Linear(self.width_dim, d, bias=False)
        self.type_pair_bias = nn.Parameter(torch.zeros(n_types, n_types))
        self.source_pair_bias = nn.Parameter(torch.zeros(n_sources, n_sources))
        self.self_bias = nn.Parameter(torch.tensor(float(self_bias_init)))
        self.gate = nn.Parameter(torch.full((d,), float(gate_init)))

    def forward(
        self,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, j_count, d = cand_states.shape
        if d != self.d:
            raise ValueError(f"cand_states last dim {d} does not match width-cell d {self.d}")
        if cand_types.shape != (bsz, seq_len, j_count) or cand_sources.shape != cand_types.shape:
            raise ValueError("candidate type/source tensors must have shape [B,T,J]")
        if cand_mask.shape != cand_types.shape:
            raise ValueError("candidate mask must have shape [B,T,J]")
        if not cand_mask.any(dim=-1).all():
            raise ValueError("DSQG-W width cell received an all-invalid candidate row")

        c_n = self.norm_c(cand_states)
        q = self.q_proj(c_n)
        k = self.k_proj(c_n)
        v = self.v_proj(c_n)

        scores = (q[:, :, :, None, :] * k[:, :, None, :, :]).sum(dim=-1) / math.sqrt(float(self.width_dim))
        scores = scores + self.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        scores = scores + self.source_pair_bias[cand_sources[:, :, :, None], cand_sources[:, :, None, :]]
        eye = torch.eye(j_count, device=cand_states.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        scores = scores + eye.to(scores.dtype) * self.self_bias

        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=3)
        probs = probs.masked_fill(~valid_pair, 0.0)

        lateral = (probs[..., None] * v[:, :, None, :, :]).sum(dim=3)
        delta = self.lateral_up(lateral)
        gate = torch.sigmoid(self.gate).reshape(1, 1, 1, d)
        out = cand_states + gate * delta * cand_mask[..., None].to(delta.dtype)

        p_mean = probs
        valid_targets = cand_mask.bool()
        p_safe = p_mean.clamp_min(1e-8)
        entropy_per_target = -(p_safe * p_safe.log()).sum(dim=-1)
        entropy = entropy_per_target.masked_select(valid_targets).mean()
        diag = torch.eye(j_count, device=cand_states.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        self_mass = p_mean.masked_fill(~diag, 0.0).sum(dim=-1).masked_select(valid_targets).mean()

        def pair_mass(target_type: CandidateType, source_type: CandidateType) -> torch.Tensor:
            target_mask = (cand_types == int(target_type)) & valid_targets
            source_mask = (cand_types == int(source_type)) & cand_mask
            if not target_mask.any():
                return cand_states.new_tensor(0.0)
            mass = p_mean.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
            return mass.masked_select(target_mask).mean()

        delta_norm = delta.masked_select(cand_mask[..., None]).reshape(-1, d).norm(dim=-1).mean()
        transfer_aux_loss = width_pair_transfer_loss(p_mean, cand_types, cand_mask)
        entropy_penalty = torch.relu(entropy.new_tensor(self.entropy_floor) - entropy)
        aux_loss = transfer_aux_loss + self.entropy_weight * entropy_penalty
        telemetry = {
            "dsqg_w_width_entropy": entropy.detach(),
            "dsqg_w_width_self_mass": self_mass.detach(),
            "dsqg_w_width_gate_mean": gate.mean().detach(),
            "dsqg_w_width_gate_min": gate.min().detach(),
            "dsqg_w_width_gate_max": gate.max().detach(),
            "dsqg_w_width_delta_norm": delta_norm.detach(),
            "dsqg_w_width_aux_loss": aux_loss,
            "dsqg_w_width_aux_loss_value": aux_loss.detach(),
            "dsqg_w_width_transfer_aux_loss": transfer_aux_loss.detach(),
            "dsqg_w_width_entropy_penalty": entropy_penalty.detach(),
            "dsqg_w_width_entropy_floor": entropy.new_tensor(self.entropy_floor).detach(),
            "dsqg_w_width_entropy_weight": entropy.new_tensor(self.entropy_weight).detach(),
            "dsqg_w_width_question_to_hisa_evidence_mass": pair_mass(
                CandidateType.QUESTION, CandidateType.HISA_EVIDENCE
            ).detach(),
            "dsqg_w_width_hisa_evidence_to_question_mass": pair_mass(
                CandidateType.HISA_EVIDENCE, CandidateType.QUESTION
            ).detach(),
        }
        return out, telemetry


class DSQGWTypedCandidateMixer(nn.Module):
    """Small typed candidate-set mixer applied before DSQG-W query scoring.

    This is bounded to the candidate axis J.  It never attends over sequence
    positions; it only lets the already-built causal candidate set exchange
    typed evidence before the main query-conditioned scoring step.
    """

    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        n_types: int,
        bottleneck: int,
        gate_init: float = -5.0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.mix_dim = int(bottleneck)
        if self.mix_dim % self.n_heads != 0:
            raise ValueError("typed mixer bottleneck must be divisible by n_heads")
        self.n_types = int(n_types)

        self.norm_c = nn.LayerNorm(d)
        self.type_embed = nn.Embedding(n_types, d)
        self.q_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.k_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.v_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.out_proj = nn.Linear(self.mix_dim, d, bias=False)
        self.type_pair_bias = nn.Parameter(torch.zeros(n_types, n_types))
        self.gate = nn.Parameter(torch.full((d,), float(gate_init)))

    def forward(
        self,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, j_count, d = cand_states.shape
        if d != self.d:
            raise ValueError(f"cand_states last dim {d} does not match typed mixer d {self.d}")
        if cand_types.shape != (bsz, seq_len, j_count) or cand_mask.shape != cand_types.shape:
            raise ValueError("candidate type/mask tensors must have shape [B,T,J]")
        if not cand_mask.any(dim=-1).all():
            raise ValueError("typed candidate mixer received an all-invalid candidate row")

        c = self.norm_c(cand_states + self.type_embed(cand_types))
        q = self.q_proj(c)
        k = self.k_proj(c)
        v = self.v_proj(c)
        scores = (q[:, :, :, None, :] * k[:, :, None, :, :]).sum(dim=-1) / math.sqrt(float(self.mix_dim))
        scores = scores + self.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = probs.masked_fill(~valid_pair, 0.0)

        mixed = (probs[..., None] * v[:, :, None, :, :]).sum(dim=3)
        delta = self.out_proj(mixed)
        gate = torch.sigmoid(self.gate).reshape(1, 1, 1, d)
        out = cand_states + gate * delta * cand_mask[..., None].to(delta.dtype)

        valid_targets = cand_mask.bool()
        p_safe = probs.clamp_min(1e-8)
        entropy = (-(p_safe * p_safe.log()).sum(dim=-1)).masked_select(valid_targets).mean()
        delta_norm = delta.masked_select(cand_mask[..., None]).reshape(-1, d).norm(dim=-1).mean()
        telemetry = {
            "dsqg_w_typed_mixer_entropy": entropy.detach(),
            "dsqg_w_typed_mixer_gate_mean": gate.mean().detach(),
            "dsqg_w_typed_mixer_gate_min": gate.min().detach(),
            "dsqg_w_typed_mixer_gate_max": gate.max().detach(),
            "dsqg_w_typed_mixer_delta_norm": delta_norm.detach(),
        }
        return out, telemetry


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
        )

    def forward(
        self,
        x: torch.Tensor,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        cand_scores: torch.Tensor | None = None,
        return_routing: bool = False,
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
        if self.width_cell is not None:
            cand_states, width_telemetry = self.width_cell(cand_states, cand_types, cand_sources, cand_mask)
        typed_mixer_telemetry: dict[str, torch.Tensor] = {}
        if self.typed_mixer is not None:
            cand_states, typed_mixer_telemetry = self.typed_mixer(cand_states, cand_types, cand_mask)

        h = self.n_heads
        dh = self.dh
        x_n = self.norm_x(x)
        c_n = self.norm_c(cand_states)

        q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh)
        k = self.k_proj(c_n).reshape(bsz, seq_len, j_count, h, dh)
        v = self.v_proj(c_n).reshape(bsz, seq_len, j_count, h, dh)
        role = self.role_key(cand_types).reshape(bsz, seq_len, j_count, h, dh)
        source = self.source_key(cand_sources).reshape(bsz, seq_len, j_count, h, dh)
        k_eff = k + role + source

        scores = (q[:, :, None, :, :] * k_eff).sum(dim=-1) / math.sqrt(float(dh))
        scores = scores + self.type_bias[cand_types]
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

        r_all_h = (probs[..., None] * v).sum(dim=2)
        r_all = r_all_h.reshape(bsz, seq_len, d)

        typed_reads = []
        typed_read_norms = []
        for type_id in range(self.n_types):
            type_mask = ((cand_types == type_id) & cand_mask)[:, :, :, None, None]
            p_type = probs[..., None].masked_fill(~type_mask, 0.0)
            r_type_h = (p_type * v).sum(dim=2)
            r_type = r_type_h.reshape(bsz, seq_len, d)
            typed_reads.append(r_type)
            typed_read_norms.append(r_type.norm(dim=-1).mean())

        r_cat = torch.cat([r_all] + typed_reads, dim=-1)
        read = self.read_mix(r_cat)
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
            "dsqg_w_delta_norm": delta_norm.detach(),
            "dsqg_w_x_norm": x_norm.detach(),
            "dsqg_w_delta_to_x_ratio": (delta_norm / x_norm.clamp_min(1e-8)).detach(),
            "dsqg_w_read_norm": read_norm.detach(),
            "dsqg_w_typed_read_norms": torch.stack(typed_read_norms).detach(),
            "read_mix_weight_norm": self.read_mix.weight.norm().detach(),
            "dsqg_w_query_type_bias_norm": query_type_bias_norm.detach(),
            "dsqg_w_candidate_score_bias_norm": candidate_score_bias_norm.detach(),
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
        telemetry.update(width_telemetry)
        telemetry.update(typed_mixer_telemetry)

        if return_routing:
            telemetry["dsqg_w_probs"] = probs
        return x_out, telemetry


def answer_masked_loss(logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or answer_mask.shape != labels.shape:
        raise ValueError("expected logits [B,T,V], labels [B,T], answer_mask [B,T]")
    selected = answer_mask.bool()
    if not selected.any():
        raise ValueError("answer_mask selects no positions")
    return F.cross_entropy(logits[selected], labels[selected])


def conditional_copy_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_mask: torch.Tensor,
    bad_copy_token_mask: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Margin anti-copy loss for lexical-gap answer positions.

    bad_copy_token_mask is [B,T,V] and should be true only for source/evidence
    copy competitors that are not valid gold answer tokens.  Gold-token entries
    are ignored defensively here as well.
    """
    if bad_copy_token_mask.shape != logits.shape:
        raise ValueError("bad_copy_token_mask must match logits shape [B,T,V]")
    if labels.shape != logits.shape[:2] or answer_mask.shape != labels.shape:
        raise ValueError("labels and answer_mask must have shape [B,T]")
    mask = bad_copy_token_mask.bool().clone()
    mask.scatter_(2, labels.unsqueeze(-1), False)
    position_mask = answer_mask.bool() & mask.any(dim=-1)
    if not position_mask.any():
        return logits.sum() * 0.0
    bad_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).amax(dim=-1)
    gold_logits = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return F.softplus(bad_logits[position_mask] - gold_logits[position_mask] + float(margin)).mean()


def _mean_head_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim == 4:
        return probs.mean(dim=-1)
    if probs.ndim == 3:
        return probs
    raise ValueError("probs must have shape [B,T,J] or [B,T,J,H]")


def local_mass_cap_loss(
    probs: torch.Tensor,
    cand_types: torch.Tensor,
    cand_mask: torch.Tensor,
    *,
    answer_mask: torch.Tensor | None = None,
    cap: float = 0.35,
    local_type_id: int = int(CandidateType.LOCAL),
) -> torch.Tensor:
    p_mean = _mean_head_probs(probs)
    local_mask = (cand_types == int(local_type_id)) & cand_mask.bool()
    local_mass = p_mean.masked_fill(~local_mask, 0.0).sum(dim=-1)
    if answer_mask is not None:
        selected = answer_mask.bool()
        if not selected.any():
            return probs.sum() * 0.0
        local_mass = local_mass[selected]
    return F.relu(local_mass - float(cap)).square().mean()


def entropy_floor_loss(
    probs: torch.Tensor,
    *,
    answer_mask: torch.Tensor | None = None,
    floor: float = 1.25,
) -> torch.Tensor:
    p_mean = _mean_head_probs(probs)
    p_safe = p_mean.clamp_min(1e-8)
    entropy = -(p_safe * p_safe.log()).sum(dim=-1)
    if answer_mask is not None:
        selected = answer_mask.bool()
        if not selected.any():
            return probs.sum() * 0.0
        entropy = entropy[selected]
    return F.relu(float(floor) - entropy).square().mean()


def candidate_recall(
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_mask: torch.Tensor,
    gold_evidence_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute candidate recall overall and by type for evidence-token audits.

    gold_evidence_indices is [B,T,K] or [B,K].  Values <0 are ignored.  Recall is
    the fraction of query rows with at least one valid gold evidence token present.
    """
    if gold_evidence_indices.ndim == 2:
        gold = gold_evidence_indices[:, None, :].expand(-1, cand_token_indices.shape[1], -1)
    elif gold_evidence_indices.ndim == 3:
        gold = gold_evidence_indices
    else:
        raise ValueError("gold_evidence_indices must be [B,K] or [B,T,K]")
    gold = gold.to(device=cand_token_indices.device)
    gold_valid = gold >= 0
    if not gold_valid.any():
        zero = cand_token_indices.float().sum() * 0.0
        return {"dsqg_w_gold_evidence_candidate_recall": zero.detach()}
    present = ((cand_token_indices[:, :, :, None] == gold[:, :, None, :]) & cand_mask[:, :, :, None] & gold_valid[:, :, None, :]).any(dim=(2, 3))
    has_gold = gold_valid.any(dim=-1)
    denom = has_gold.float().sum().clamp_min(1.0)
    out: dict[str, torch.Tensor] = {
        "dsqg_w_gold_evidence_candidate_recall": (present & has_gold).float().sum() / denom,
    }
    for ctype in CandidateType:
        type_present = (
            (cand_token_indices[:, :, :, None] == gold[:, :, None, :])
            & cand_mask[:, :, :, None]
            & (cand_types[:, :, :, None] == int(ctype))
            & gold_valid[:, :, None, :]
        ).any(dim=(2, 3))
        out[f"dsqg_w_gold_evidence_candidate_recall_{ctype.name.lower()}"] = (type_present & has_gold).float().sum() / denom
    return out
