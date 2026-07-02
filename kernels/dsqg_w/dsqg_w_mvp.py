from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import IntEnum
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_SOURCEWISE_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_SOURCEWISE_AVAILABLE = False


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
class _DSQGWTritonSchedule:
    """Centralized launch schedule for DSQG-W Triton row/head kernels.

    Mirrors V20's discipline of deriving launch shape from head dimension and
    SM family in one place. Values are adapted to DSQG-W's one-row/one-head
    programs rather than copied from V20's BLOCK_N x HD tile kernels.
    """

    block_hd: int
    num_warps: int
    num_stages: int


def _next_pow2_int(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _dsqg_w_triton_schedule(head_dim: int, device: torch.device | None = None) -> _DSQGWTritonSchedule:
    block_hd = _next_pow2_int(int(head_dim))
    if block_hd <= 64:
        base_warps = 1
    elif block_hd <= 128:
        base_warps = 2
    else:
        base_warps = 4

    num_stages = 2
    if device is not None and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(device)
        except Exception:
            major, minor = (0, 0)
        if major >= 9:
            base_warps = min(max(base_warps, 2), 4)
            num_stages = 3
        elif major == 8 and minor == 9:
            num_stages = 2
    return _DSQGWTritonSchedule(block_hd=block_hd, num_warps=base_warps, num_stages=num_stages)


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


if _TRITON_SOURCEWISE_AVAILABLE:

    @triton.jit
    def _dsqg_w_sourcewise_score_read_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        read_ptr,
        read_mix_weight_ptr,
        probs_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        OUT_BLOCK: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        STORE_PROBS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        out_pid = tl.program_id(1)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        out_offs = out_pid * OUT_BLOCK + tl.arange(0, OUT_BLOCK)
        out_mask = out_offs < D
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        read_out = tl.zeros((int(OUT_BLOCK),), tl.float32)

        row_j_base = (b * N + n) * J
        max_score = -float("inf")
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            max_score = tl.maximum(max_score, score)

        denom = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            denom += tl.where(valid, tl.exp(score - max_score), 0.0)

        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            p = tl.where(valid, tl.exp(score - max_score) / denom, 0.0)
            if STORE_PROBS and out_pid == 0:
                tl.store(probs_ptr + ((b * N + n) * J + j) * H + h, p)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            contrib = p * v
            in_cols = h * HD + offs
            all_w = tl.load(
                read_mix_weight_ptr + out_offs[:, None] * ((N_TYPES + 1) * D) + in_cols[None, :],
                mask=out_mask[:, None] & hd_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            read_out += tl.sum(all_w * contrib[None, :], axis=1)
            typed_slot = ctype + 1
            typed_cols = typed_slot * D + h * HD + offs
            type_w = tl.load(
                read_mix_weight_ptr + out_offs[:, None] * ((N_TYPES + 1) * D) + typed_cols[None, :],
                mask=out_mask[:, None] & hd_mask[None, :] & valid & (ctype >= 0) & (ctype < N_TYPES),
                other=0.0,
            ).to(tl.float32)
            read_out += tl.sum(type_w * contrib[None, :], axis=1)
        tl.atomic_add(read_ptr + (b * N + n) * D + out_offs, read_out, sem="relaxed", mask=out_mask)


    @triton.jit
    def _dsqg_w_sourcewise_read_slots_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        read_slots_ptr,
        lse_ptr,
        probs_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        MAX_READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        STORE_LSE: tl.constexpr,
        STORE_PROBS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)

        row_j_base = (b * N + n) * J
        max_score = -float("inf")
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            max_score = tl.maximum(max_score, score)

        denom = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            denom += tl.where(valid, tl.exp(score - max_score), 0.0)

        if STORE_LSE:
            tl.store(lse_ptr + (b * N + n) * H + h, max_score + tl.log(denom))

        slot_ids = tl.arange(0, MAX_READ_SLOTS)
        acc = tl.zeros((int(MAX_READ_SLOTS), int(BLOCK_HD)), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            p = tl.where(valid, tl.exp(score - max_score) / denom, 0.0)
            if STORE_PROBS:
                tl.store(probs_ptr + ((b * N + n) * J + j) * H + h, p)
            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            contrib = p * v
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            add_slot = (slot_ids[:, None] == 0) | (slot_ids[:, None] == type_slot)
            active_slot = slot_ids[:, None] < READ_SLOTS
            acc += tl.where(add_slot & active_slot, contrib[None, :], 0.0)

        store_base = ((b * N + n) * READ_SLOTS + slot_ids[:, None]) * D + h * HD + offs[None, :]
        tl.store(read_slots_ptr + store_base, acc, mask=(slot_ids[:, None] < READ_SLOTS) & hd_mask[None, :])


    @triton.jit
    def _dsqg_w_sourcewise_read_slots_backward_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        lse_ptr,
        grad_slots_ptr,
        grad_q_ptr,
        grad_k_final_ptr,
        grad_v_final_ptr,
        grad_k_l3_ptr,
        grad_v_l3_ptr,
        grad_k_summary_ptr,
        grad_v_summary_ptr,
        grad_role_key_ptr,
        grad_source_key_ptr,
        grad_type_bias_ptr,
        grad_source_bias_ptr,
        grad_qtb_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        COMPUTE_QUERY: tl.constexpr,
        COMPUTE_SOURCE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + (b * N + n) * H + h).to(tl.float32)
        row_j_base = (b * N + n) * J

        sum_p_dp = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            sum_p_dp += p * dp

        grad_q = tl.zeros((int(BLOCK_HD),), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            ds = tl.where(valid, p * (dp - sum_p_dp), 0.0)
            d_k_eff = ds * q * inv_sqrt
            d_v = p * dcontrib
            if COMPUTE_QUERY:
                grad_q += ds * (k + role + src_role) * inv_sqrt

            final_src = (source_id == 1) | (source_id == 5)
            l3_src = (source_id == 2) | (source_id == 3)
            summary_src = source_id == 4
            if COMPUTE_SOURCE:
                tl.atomic_add(grad_k_final_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & final_src)
                tl.atomic_add(grad_v_final_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & final_src)
                tl.atomic_add(grad_k_l3_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & l3_src)
                tl.atomic_add(grad_v_l3_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & l3_src)
                tl.atomic_add(grad_k_summary_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & summary_src)
                tl.atomic_add(grad_v_summary_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & summary_src)
            if COMPUTE_QUERY:
                tl.atomic_add(grad_role_key_ptr + ctype * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
                tl.atomic_add(grad_source_key_ptr + source_id * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
                tl.atomic_add(grad_type_bias_ptr + ctype * H + h, ds, sem="relaxed", mask=valid)
                tl.atomic_add(grad_source_bias_ptr + source_id * H + h, ds, sem="relaxed", mask=valid)
                if USE_QTB:
                    tl.atomic_add(grad_qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, ds, sem="relaxed", mask=valid)

        if COMPUTE_QUERY:
            tl.store(grad_q_ptr + q_base, grad_q, mask=hd_mask)


def _dsqg_w_sourcewise_functional_recompute(
    x: torch.Tensor,
    l3_states: torch.Tensor | None,
    chunk_rep_states: torch.Tensor | None,
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
    cand_scores: torch.Tensor | None,
    *,
    d: int,
    n_heads: int,
    dh: int,
    n_types: int,
    read_type_ids: tuple[int, ...],
    use_query_type_bias: bool,
    norm_x_weight: torch.Tensor,
    norm_x_bias: torch.Tensor,
    norm_c_weight: torch.Tensor,
    norm_c_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    role_key_weight: torch.Tensor,
    source_key_weight: torch.Tensor,
    type_bias: torch.Tensor,
    query_type_bias_weight: torch.Tensor,
    source_bias: torch.Tensor,
    read_mix_weight: torch.Tensor,
    norm_z_weight: torch.Tensor,
    norm_z_bias: torch.Tensor,
    fuse0_weight: torch.Tensor,
    fuse0_bias: torch.Tensor,
    fuse2_weight: torch.Tensor,
    fuse2_bias: torch.Tensor,
    gate_param: torch.Tensor,
) -> torch.Tensor:
    """PyTorch sourcewise recompute used only by Triton custom backward."""
    bsz, seq_len, _ = x.shape
    j_count = cand_mask.shape[-1]
    x_n = F.layer_norm(x, (d,), norm_x_weight, norm_x_bias)
    q = F.linear(x_n, q_proj_weight).reshape(bsz, seq_len, n_heads, dh)

    final_states = x
    l3_base = l3_states if l3_states is not None else final_states
    summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
    zero_base = torch.zeros_like(final_states)
    source_bases: dict[int, torch.Tensor] = {
        int(CandidateSource.FINAL): final_states,
        int(CandidateSource.QUESTION_CACHE): final_states,
        int(CandidateSource.L3): l3_base,
        int(CandidateSource.HISA): l3_base,
        int(CandidateSource.SUMMARY): summary_base,
        int(CandidateSource.NULL): zero_base,
    }
    projected_by_object: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for source_id, states in source_bases.items():
        if not bool(((cand_sources == int(source_id)) & cand_mask).any()):
            continue
        cache_key = id(states)
        projected = projected_by_object.get(cache_key)
        if projected is None:
            states_n = F.layer_norm(states, (d,), norm_c_weight, norm_c_bias)
            k_src = F.linear(states_n, k_proj_weight).reshape(bsz, seq_len, n_heads, dh)
            v_src = F.linear(states_n, v_proj_weight).reshape(bsz, seq_len, n_heads, dh)
            projected = (k_src, v_src)
            projected_by_object[cache_key] = projected
        projected_sources[source_id] = projected

    gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
    score_bias = None
    if cand_scores is not None:
        score_bias = cand_scores.to(device=x.device, dtype=x.dtype)
        score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
        valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
        score_bias = score_bias.masked_fill(~cand_mask, 0.0)
    qtb = None
    if use_query_type_bias:
        qtb = F.linear(x_n, query_type_bias_weight).reshape(bsz, seq_len, n_types, n_heads)

    score_parts: list[torch.Tensor] = []
    batch_offsets = torch.arange(bsz, device=x.device, dtype=torch.long).reshape(bsz, 1) * seq_len
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        k_j = x.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (k_src, _) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = k_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
        role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, n_heads, dh)
        source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, n_heads, dh)
        score_j = (q * (k_j + role + source)).sum(dim=-1) / math.sqrt(float(dh))
        score_j = score_j + type_bias[cand_types[:, :, j]]
        if score_bias is not None:
            score_j = score_j + score_bias[:, :, j, None]
        if qtb is not None:
            score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, n_heads)).squeeze(2)
        score_j = score_j + source_bias[source_j]
        score_j = score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min)
        score_parts.append(score_j)
    scores = torch.stack(score_parts, dim=2)
    probs = F.softmax(scores, dim=2)

    r_all_h = x.new_zeros((bsz, seq_len, n_heads, dh))
    typed_reads_h = {
        type_id: x.new_zeros((bsz, seq_len, n_heads, dh))
        for type_id in read_type_ids
        if 0 <= int(type_id) < n_types
    }
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        v_j = x.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (_, v_src) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = v_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
        contrib = probs[:, :, j, :, None] * v_j
        r_all_h = r_all_h + contrib
        for type_id in typed_reads_h:
            type_mask = ((cand_types[:, :, j] == int(type_id)) & cand_mask[:, :, j])[:, :, None, None]
            typed_reads_h[type_id] = typed_reads_h[type_id] + contrib * type_mask.to(contrib.dtype)

    r_all = r_all_h.reshape(bsz, seq_len, d)
    read = F.linear(r_all, read_mix_weight[:, :d])
    for type_id, r_type_h in typed_reads_h.items():
        r_type = r_type_h.reshape(bsz, seq_len, d)
        start = (int(type_id) + 1) * d
        read = read + F.linear(r_type, read_mix_weight[:, start : start + d])
    z = torch.cat([x, read, x * read, read - x], dim=-1)
    z_n = F.layer_norm(z, (4 * d,), norm_z_weight, norm_z_bias)
    hidden = F.gelu(F.linear(z_n, fuse0_weight, fuse0_bias))
    delta = F.linear(hidden, fuse2_weight, fuse2_bias)
    gate = torch.sigmoid(gate_param).reshape(1, 1, d)
    return x + gate * delta


def _dsqg_w_sourcewise_read_slots_recompute(
    q: torch.Tensor,
    k_final: torch.Tensor,
    v_final: torch.Tensor,
    k_l3: torch.Tensor,
    v_l3: torch.Tensor,
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    role_key_weight: torch.Tensor,
    source_key_weight: torch.Tensor,
    type_bias: torch.Tensor,
    source_bias: torch.Tensor,
    qtb: torch.Tensor | None,
    score_bias: torch.Tensor | None,
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
    type_slot_map: torch.Tensor,
    *,
    d: int,
    n_heads: int,
    dh: int,
    read_slots: int,
) -> torch.Tensor:
    """Compact [B,N,S,D] read-slot recompute for the read-only Triton autograd node."""
    bsz, seq_len, j_count = cand_mask.shape
    gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
    batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).reshape(bsz, 1) * seq_len
    projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        int(CandidateSource.FINAL): (k_final, v_final),
        int(CandidateSource.QUESTION_CACHE): (k_final, v_final),
        int(CandidateSource.L3): (k_l3, v_l3),
        int(CandidateSource.HISA): (k_l3, v_l3),
        int(CandidateSource.SUMMARY): (k_summary, v_summary),
    }

    score_parts: list[torch.Tensor] = []
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        k_j = q.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (k_src, _) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = k_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
        role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, n_heads, dh)
        source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, n_heads, dh)
        score_j = (q * (k_j + role + source)).sum(dim=-1) / math.sqrt(float(dh))
        score_j = score_j + type_bias[cand_types[:, :, j]]
        score_j = score_j + source_bias[source_j]
        if score_bias is not None:
            score_j = score_j + score_bias[:, :, j, None]
        if qtb is not None:
            score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, n_heads)).squeeze(2)
        score_j = score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min)
        score_parts.append(score_j)
    scores = torch.stack(score_parts, dim=2)
    probs = F.softmax(scores, dim=2)

    slots_h = q.new_zeros((bsz, seq_len, read_slots, n_heads, dh))
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        v_j = q.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (_, v_src) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = v_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
        contrib = probs[:, :, j, :, None] * v_j
        slots_h[:, :, 0] = slots_h[:, :, 0] + contrib
        type_slots = type_slot_map[cand_types[:, :, j]].to(torch.long)
        for slot in range(1, read_slots):
            slot_mask = ((type_slots == slot) & cand_mask[:, :, j])[:, :, None, None]
            slots_h[:, :, slot] = slots_h[:, :, slot] + contrib * slot_mask.to(contrib.dtype)
    return slots_h.reshape(bsz, seq_len, read_slots, d)


class _DSQGWSourcewiseTritonCompactRead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k_final,
        v_final,
        k_l3,
        v_l3,
        k_summary,
        v_summary,
        role_key_weight,
        source_key_weight,
        type_bias,
        source_bias,
        qtb,
        score_bias,
        cand_token_indices,
        cand_types,
        cand_sources,
        cand_mask,
        type_slot_map,
        use_qtb: bool,
        use_score_bias: bool,
        d: int,
        n_heads: int,
        dh: int,
        n_types: int,
        read_slots: int,
        block_hd: int,
    ):
        ctx.use_qtb = bool(use_qtb)
        ctx.use_score_bias = bool(use_score_bias)
        ctx.d = int(d)
        ctx.n_heads = int(n_heads)
        ctx.dh = int(dh)
        ctx.read_slots = int(read_slots)
        bsz, seq_len = q.shape[:2]
        schedule = _dsqg_w_triton_schedule(dh, q.device)
        read_slots_out = torch.empty((bsz, seq_len, int(read_slots), int(d)), device=q.device, dtype=q.dtype)
        lse_out = torch.empty((bsz, seq_len, int(n_heads)), device=q.device, dtype=torch.float32)
        ctx.save_for_backward(
            q,
            k_final,
            v_final,
            k_l3,
            v_l3,
            k_summary,
            v_summary,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse_out,
        )
        empty = torch.empty((0,), device=q.device, dtype=q.dtype)
        _dsqg_w_sourcewise_read_slots_kernel[(bsz * seq_len * int(n_heads),)](
            q.contiguous(),
            k_final.contiguous(),
            v_final.contiguous(),
            k_l3.contiguous(),
            v_l3.contiguous(),
            k_summary.contiguous(),
            v_summary.contiguous(),
            role_key_weight.contiguous(),
            source_key_weight.contiguous(),
            type_bias.contiguous(),
            source_bias.contiguous(),
            qtb.contiguous() if bool(use_qtb) else empty,
            score_bias.contiguous() if bool(use_score_bias) else empty,
            cand_token_indices.contiguous(),
            cand_types.contiguous(),
            cand_sources.contiguous(),
            cand_mask.contiguous(),
            type_slot_map.contiguous(),
            read_slots_out,
            lse_out,
            empty,
            B=bsz,
            N=seq_len,
            H=int(n_heads),
            HD=int(dh),
            D=int(d),
            J=cand_mask.shape[-1],
            N_TYPES=int(n_types),
            READ_SLOTS=int(read_slots),
            MAX_READ_SLOTS=int(triton.next_power_of_2(int(read_slots))),
            BLOCK_HD=schedule.block_hd,
            USE_QTB=bool(use_qtb),
            USE_SCORE_BIAS=bool(use_score_bias),
            STORE_LSE=True,
            STORE_PROBS=False,
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
        )
        return read_slots_out

    @staticmethod
    def backward(ctx, grad_read_slots):
        saved = ctx.saved_tensors
        (
            q,
            k_final,
            v_final,
            k_l3,
            v_l3,
            k_summary,
            v_summary,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse,
        ) = saved
        bsz, seq_len, j_count = cand_mask.shape
        h = ctx.n_heads
        dh = ctx.dh
        d = ctx.d
        if os.getenv("DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD", "triton").lower() != "pytorch":
            grad_q = torch.zeros_like(q)
            grad_k_final = torch.zeros_like(k_final)
            grad_v_final = torch.zeros_like(v_final)
            grad_k_l3 = torch.zeros_like(k_l3)
            grad_v_l3 = torch.zeros_like(v_l3)
            grad_k_summary = torch.zeros_like(k_summary)
            grad_v_summary = torch.zeros_like(v_summary)
            grad_role_key = torch.zeros_like(role_key_weight)
            grad_source_key = torch.zeros_like(source_key_weight)
            grad_type_bias = torch.zeros_like(type_bias)
            grad_source_bias = torch.zeros_like(source_bias)
            grad_qtb = torch.zeros_like(qtb) if ctx.use_qtb else None
            empty = torch.empty((0,), device=q.device, dtype=q.dtype)
            schedule = _dsqg_w_triton_schedule(dh, q.device)
            grid = (bsz * seq_len * h,)

            def launch_split_kernel(*, compute_query: bool, compute_source: bool) -> None:
                _dsqg_w_sourcewise_read_slots_backward_kernel[grid](
                    q.contiguous(),
                    k_final.contiguous(),
                    v_final.contiguous(),
                    k_l3.contiguous(),
                    v_l3.contiguous(),
                    k_summary.contiguous(),
                    v_summary.contiguous(),
                    role_key_weight.contiguous(),
                    source_key_weight.contiguous(),
                    type_bias.contiguous(),
                    source_bias.contiguous(),
                    qtb.contiguous() if ctx.use_qtb else empty,
                    score_bias.contiguous() if ctx.use_score_bias else empty,
                    cand_token_indices.contiguous(),
                    cand_types.contiguous(),
                    cand_sources.contiguous(),
                    cand_mask.contiguous(),
                    type_slot_map.contiguous(),
                    lse.contiguous(),
                    grad_read_slots.contiguous(),
                    grad_q,
                    grad_k_final,
                    grad_v_final,
                    grad_k_l3,
                    grad_v_l3,
                    grad_k_summary,
                    grad_v_summary,
                    grad_role_key,
                    grad_source_key,
                    grad_type_bias,
                    grad_source_bias,
                    grad_qtb if grad_qtb is not None else empty,
                    B=bsz,
                    N=seq_len,
                    H=h,
                    HD=dh,
                    D=d,
                    J=j_count,
                    N_TYPES=type_bias.shape[0],
                    READ_SLOTS=ctx.read_slots,
                    BLOCK_HD=schedule.block_hd,
                    USE_QTB=ctx.use_qtb,
                    USE_SCORE_BIAS=ctx.use_score_bias,
                    COMPUTE_QUERY=compute_query,
                    COMPUTE_SOURCE=compute_source,
                    num_warps=schedule.num_warps,
                    num_stages=schedule.num_stages,
                )

            # V20-style organization can be enabled for profiling, but keep the
            # fused monolithic launch as the default until split scheduling wins in
            # full trainer windows rather than only as a code-organization pattern.
            split_backward = os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION", "monolithic").lower() in {
                "1",
                "true",
                "split",
                "v20_split",
            }
            if split_backward:
                launch_split_kernel(compute_query=True, compute_source=False)
                launch_split_kernel(compute_query=False, compute_source=True)
            else:
                launch_split_kernel(compute_query=True, compute_source=True)
            return (
                grad_q,
                grad_k_final,
                grad_v_final,
                grad_k_l3,
                grad_v_l3,
                grad_k_summary,
                grad_v_summary,
                grad_role_key,
                grad_source_key,
                grad_type_bias,
                grad_source_bias,
                grad_qtb,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
        batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).reshape(bsz, 1) * seq_len
        inv_sqrt = 1.0 / math.sqrt(float(dh))
        projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            int(CandidateSource.FINAL): (k_final, v_final),
            int(CandidateSource.QUESTION_CACHE): (k_final, v_final),
            int(CandidateSource.L3): (k_l3, v_l3),
            int(CandidateSource.HISA): (k_l3, v_l3),
            int(CandidateSource.SUMMARY): (k_summary, v_summary),
        }

        score_parts: list[torch.Tensor] = []
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            k_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (k_src, _) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = k_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
            role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, h, dh)
            source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, h, dh)
            score_j = (q * (k_j + role + source)).sum(dim=-1) * inv_sqrt
            score_j = score_j + type_bias[cand_types[:, :, j]] + source_bias[source_j]
            if ctx.use_score_bias:
                score_j = score_j + score_bias[:, :, j, None]
            if ctx.use_qtb:
                score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, h)).squeeze(2)
            score_parts.append(score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min))
        scores = torch.stack(score_parts, dim=2)
        probs = F.softmax(scores, dim=2)

        grad_slots_h = grad_read_slots.reshape(bsz, seq_len, ctx.read_slots, h, dh)
        dp_parts: list[torch.Tensor] = []
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            v_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (_, v_src) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = v_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
            type_slots = type_slot_map[cand_types[:, :, j]].to(torch.long)
            dcontrib = grad_slots_h[:, :, 0]
            for slot in range(1, ctx.read_slots):
                dcontrib = dcontrib + grad_slots_h[:, :, slot] * (type_slots == slot)[:, :, None, None].to(grad_slots_h.dtype)
            dp_parts.append((dcontrib * v_j).sum(dim=-1))
        dp = torch.stack(dp_parts, dim=2)
        ds = probs * (dp - (dp * probs).sum(dim=2, keepdim=True))
        ds = ds.masked_fill(~cand_mask[:, :, :, None], 0.0)

        grad_q = torch.zeros_like(q)
        grad_k_final = torch.zeros_like(k_final)
        grad_v_final = torch.zeros_like(v_final)
        grad_k_l3 = torch.zeros_like(k_l3)
        grad_v_l3 = torch.zeros_like(v_l3)
        grad_k_summary = torch.zeros_like(k_summary)
        grad_v_summary = torch.zeros_like(v_summary)
        grad_role_key = torch.zeros_like(role_key_weight)
        grad_source_key = torch.zeros_like(source_key_weight)
        grad_type_bias = torch.zeros_like(type_bias)
        grad_source_bias = torch.zeros_like(source_bias)
        grad_qtb = torch.zeros_like(qtb) if ctx.use_qtb else None
        k_grads: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): grad_k_final,
            int(CandidateSource.QUESTION_CACHE): grad_k_final,
            int(CandidateSource.L3): grad_k_l3,
            int(CandidateSource.HISA): grad_k_l3,
            int(CandidateSource.SUMMARY): grad_k_summary,
        }
        v_grads: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): grad_v_final,
            int(CandidateSource.QUESTION_CACHE): grad_v_final,
            int(CandidateSource.L3): grad_v_l3,
            int(CandidateSource.HISA): grad_v_l3,
            int(CandidateSource.SUMMARY): grad_v_summary,
        }

        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            ctype_j = cand_types[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            type_slots = type_slot_map[ctype_j].to(torch.long)
            dcontrib = grad_slots_h[:, :, 0]
            for slot in range(1, ctx.read_slots):
                dcontrib = dcontrib + grad_slots_h[:, :, slot] * (type_slots == slot)[:, :, None, None].to(grad_slots_h.dtype)
            d_v_j = probs[:, :, j, :, None] * dcontrib

            k_eff_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (k_src, _) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = k_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    k_eff_j = k_eff_j + gathered * source_mask[:, :, None, None].to(k_eff_j.dtype)
            role = F.embedding(ctype_j, role_key_weight).reshape(bsz, seq_len, h, dh)
            source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, h, dh)
            k_eff_j = k_eff_j + role + source
            d_k_eff = ds[:, :, j, :, None] * q * inv_sqrt
            grad_q = grad_q + ds[:, :, j, :, None] * k_eff_j * inv_sqrt

            for source_id in k_grads:
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    mask = source_mask[:, :, None, None].to(d_k_eff.dtype)
                    k_add = (d_k_eff * mask).reshape(bsz * seq_len, h, dh).to(k_grads[source_id].dtype)
                    v_add = (d_v_j * mask).reshape(bsz * seq_len, h, dh).to(v_grads[source_id].dtype)
                    k_grads[source_id].reshape(bsz * seq_len, h, dh).index_add_(0, flat_indices, k_add)
                    v_grads[source_id].reshape(bsz * seq_len, h, dh).index_add_(0, flat_indices, v_add)

            grad_role_key.index_add_(0, ctype_j.reshape(-1), d_k_eff.reshape(bsz * seq_len, d).to(grad_role_key.dtype))
            grad_source_key.index_add_(0, source_j.reshape(-1), d_k_eff.reshape(bsz * seq_len, d).to(grad_source_key.dtype))
            ctype_flat = ctype_j.reshape(-1)
            source_flat = source_j.reshape(-1)
            ds_flat = ds[:, :, j, :].reshape(bsz * seq_len, h)
            for head_idx in range(h):
                grad_type_bias[:, head_idx].index_add_(0, ctype_flat, ds_flat[:, head_idx].to(grad_type_bias.dtype))
                grad_source_bias[:, head_idx].index_add_(0, source_flat, ds_flat[:, head_idx].to(grad_source_bias.dtype))
            if grad_qtb is not None:
                grad_qtb.scatter_add_(2, ctype_j[:, :, None, None].expand(-1, -1, 1, h), ds[:, :, j, None, :].to(grad_qtb.dtype))

        grad_list: list[torch.Tensor | None] = [
            grad_q,
            grad_k_final,
            grad_v_final,
            grad_k_l3,
            grad_v_l3,
            grad_k_summary,
            grad_v_summary,
            grad_role_key,
            grad_source_key,
            grad_type_bias,
            grad_source_bias,
        ]
        return (
            *grad_list,
            grad_qtb,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DSQGWSourcewiseTritonRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask, l3_present: bool, chunk_present: bool, scores_present: bool, *params):
        ctx.block = block
        ctx.l3_present = bool(l3_present)
        ctx.chunk_present = bool(chunk_present)
        ctx.scores_present = bool(scores_present)
        ctx.save_for_backward(x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask, *params)
        out, _ = block._forward_sourcewise_triton(
            x,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            l3_states=l3_states if l3_present else None,
            chunk_rep_states=chunk_rep_states if chunk_present else None,
            cand_scores=cand_scores if scores_present else None,
            return_routing=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask = saved[:8]
        params = saved[8:]
        block = ctx.block
        x_req = x.detach().requires_grad_(True)
        l3_req = l3_states.detach().requires_grad_(True) if ctx.l3_present else None
        chunk_req = chunk_rep_states.detach().requires_grad_(True) if ctx.chunk_present else None
        param_reqs = [p.detach().requires_grad_(True) for p in params]
        with torch.enable_grad():
            out = _dsqg_w_sourcewise_functional_recompute(
                x_req,
                l3_req,
                chunk_req,
                cand_token_indices,
                cand_types,
                cand_sources,
                cand_mask,
                cand_scores if ctx.scores_present else None,
                d=block.d,
                n_heads=block.n_heads,
                dh=block.dh,
                n_types=block.n_types,
                read_type_ids=block.read_type_ids,
                use_query_type_bias=block.use_query_type_bias,
                norm_x_weight=param_reqs[0],
                norm_x_bias=param_reqs[1],
                norm_c_weight=param_reqs[2],
                norm_c_bias=param_reqs[3],
                q_proj_weight=param_reqs[4],
                k_proj_weight=param_reqs[5],
                v_proj_weight=param_reqs[6],
                role_key_weight=param_reqs[7],
                source_key_weight=param_reqs[8],
                type_bias=param_reqs[9],
                query_type_bias_weight=param_reqs[10],
                source_bias=param_reqs[11],
                read_mix_weight=param_reqs[12],
                norm_z_weight=param_reqs[13],
                norm_z_bias=param_reqs[14],
                fuse0_weight=param_reqs[15],
                fuse0_bias=param_reqs[16],
                fuse2_weight=param_reqs[17],
                fuse2_bias=param_reqs[18],
                gate_param=param_reqs[19],
            )
            grad_inputs = torch.autograd.grad(
                out,
                [x_req] + ([l3_req] if l3_req is not None else []) + ([chunk_req] if chunk_req is not None else []) + param_reqs,
                grad_out,
                allow_unused=True,
            )
        idx = 0
        grad_x = grad_inputs[idx]; idx += 1
        grad_l3 = grad_inputs[idx] if ctx.l3_present else None
        if ctx.l3_present:
            idx += 1
        grad_chunk = grad_inputs[idx] if ctx.chunk_present else None
        if ctx.chunk_present:
            idx += 1
        grad_params = list(grad_inputs[idx:])
        while len(grad_params) < len(params):
            grad_params.append(None)
        return (None, grad_x, grad_l3, grad_chunk, None, None, None, None, None, None, None, None, *grad_params)


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
        read = F.linear(r_all, weight[:, : self.d])
        typed_read_norms = [r_all.new_tensor(0.0) for _ in range(self.n_types)]
        for type_id in self.read_type_ids:
            if type_id < 0 or type_id >= self.n_types:
                continue
            type_mask = ((cand_types == type_id) & cand_mask)[:, :, :, None, None]
            p_type = probs[..., None].masked_fill(~type_mask, 0.0)
            r_type_h = (p_type * v).sum(dim=2)
            r_type = r_type_h.reshape(bsz, seq_len, self.d)
            start = (type_id + 1) * self.d
            read = read + F.linear(r_type, weight[:, start : start + self.d])
            typed_read_norms[type_id] = r_type.norm(dim=-1).mean()
        return read, typed_read_norms

    @staticmethod
    def _gather_source_rows(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = states.shape[:2]
        batch_offsets = torch.arange(bsz, device=states.device, dtype=torch.long).reshape(bsz, 1) * seq_len
        flat_indices = (batch_offsets + token_indices.to(torch.long)).reshape(-1)
        return states.reshape(bsz * seq_len, *states.shape[2:]).index_select(0, flat_indices).reshape(
            bsz, seq_len, *states.shape[2:]
        )

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
        return_routing: bool = False,
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

        x_n = self.norm_x(x)
        q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh).contiguous()
        needed_source_ids = tuple(
            int(source)
            for source in CandidateSource
            if bool(((cand_sources == int(source)) & cand_mask).any())
        )
        needed_with_final = tuple(sorted(set(needed_source_ids) | {int(CandidateSource.FINAL)}))
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
        weight = self.read_mix.weight
        read = F.linear(read_slots[:, :, 0, :], weight[:, : self.d])
        for slot_idx, type_id in enumerate(self.read_type_ids, start=1):
            if 0 <= int(type_id) < self.n_types:
                start = (int(type_id) + 1) * self.d
                read = read + F.linear(read_slots[:, :, slot_idx, :], weight[:, start : start + self.d])
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
            "dsqg_w_sourcewise": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_sourcewise": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_sourcewise_recompute_backward": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_compact_read_backward": x.new_tensor(1.0 if needs_backward else 0.0).detach(),
            "dsqg_w_triton_probs_materialized": x.new_tensor(1.0 if return_routing else 0.0).detach(),
            "dsqg_w_triton_read_accum_materialized": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_read_mix_fused": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_compact_read_slots_materialized": x.new_tensor(1.0).detach(),
            "dsqg_w_triton_compact_read_slots": x.new_tensor(float(read_slot_count)).detach(),
            "dsqg_w_triton_score_recompute_blocks": x.new_tensor(2.0 if split_backward else 1.0).detach(),
            "dsqg_w_triton_true_backward": x.new_tensor(1.0 if true_backward else 0.0).detach(),
            "dsqg_w_triton_backward_v20_split_kernels": x.new_tensor(1.0 if split_backward else 0.0).detach(),
            "dsqg_w_triton_backward_monolithic_kernel": x.new_tensor(1.0 if true_backward and not split_backward else 0.0).detach(),
            "dsqg_w_triton_backward_query_kernel": x.new_tensor(1.0 if split_backward else 0.0).detach(),
            "dsqg_w_triton_backward_source_kernel": x.new_tensor(1.0 if split_backward else 0.0).detach(),
            "dsqg_w_triton_backward_probs_materialized": x.new_tensor(0.0 if true_backward else (1.0 if needs_backward else 0.0)).detach(),
            "dsqg_w_triton_backward_lse_saved": x.new_tensor(1.0 if needs_backward else 0.0).detach(),
            "dsqg_w_triton_backward_reduction_buffer_bytes": x.new_tensor(0.0).detach(),
            "dsqg_w_triton_schedule_block_hd": x.new_tensor(float(schedule.block_hd)).detach(),
            "dsqg_w_triton_schedule_num_warps": x.new_tensor(float(schedule.num_warps)).detach(),
            "dsqg_w_triton_schedule_num_stages": x.new_tensor(float(schedule.num_stages)).detach(),
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
        return_routing: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.width_cell is not None or self.typed_mixer is not None:
            raise NotImplementedError("sourcewise DSQG-W does not implement width_cell or typed_mixer")
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
                return_routing=return_routing,
            )

        h = self.n_heads
        dh = self.dh
        x_n = self.norm_x(x)
        q = self.q_proj(x_n).reshape(bsz, seq_len, h, dh)
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

        read, typed_read_norms = self._mix_typed_reads(r_all, probs, v, cand_types, cand_mask)
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
