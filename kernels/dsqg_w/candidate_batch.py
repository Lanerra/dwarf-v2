from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Candidate:
    token_index: int
    source_layer: int
    candidate_type: int
    offset: int | None
    valid: bool = True


@dataclass(frozen=True)
class CandidateLayout:
    """Static slot contract for common DSQG-W candidate layouts.

    Dynamic candidate metadata should normally be limited to token indices,
    scores/distances, and validity.  For DSR/HISA layouts the semantic meaning
    of slot ``j`` is static, so type/source/group live here as [J] tensors and
    can be expanded as zero-stride views only for compatibility with legacy
    call sites.
    """

    slot_type: torch.Tensor
    slot_source: torch.Tensor
    slot_group: torch.Tensor
    read_type_ids: torch.Tensor
    active_sources: tuple[int, ...]
    has_scores: bool
    has_distances: bool

    def __post_init__(self) -> None:
        if self.slot_type.ndim != 1 or self.slot_source.ndim != 1 or self.slot_group.ndim != 1:
            raise ValueError("CandidateLayout slot tensors must be rank-1 [J]")
        if self.slot_type.shape != self.slot_source.shape or self.slot_type.shape != self.slot_group.shape:
            raise ValueError("CandidateLayout slot_type/source/group must have matching [J] shape")
        if self.read_type_ids.ndim != 1:
            raise ValueError("CandidateLayout read_type_ids must be rank-1")

    @property
    def slot_count(self) -> int:
        return int(self.slot_type.shape[0])

    def expand_slot_type(self, batch: int, seq_len: int) -> torch.Tensor:
        return self.slot_type.reshape(1, 1, self.slot_count).expand(int(batch), int(seq_len), self.slot_count)

    def expand_slot_source(self, batch: int, seq_len: int) -> torch.Tensor:
        return self.slot_source.reshape(1, 1, self.slot_count).expand(int(batch), int(seq_len), self.slot_count)

    def expand_slot_group(self, batch: int, seq_len: int) -> torch.Tensor:
        return self.slot_group.reshape(1, 1, self.slot_count).expand(int(batch), int(seq_len), self.slot_count)


@dataclass(frozen=True)
class CandidateBatch:
    cand_states: torch.Tensor
    cand_types: torch.Tensor
    cand_sources: torch.Tensor
    cand_mask: torch.Tensor
    cand_token_indices: torch.Tensor
    valid_candidate_count: torch.Tensor
    cand_scores: torch.Tensor | None = None
    evidence_bits: torch.Tensor | None = None
    evidence_count: torch.Tensor | None = None
    candidate_distances: torch.Tensor | None = None
    telemetry: dict[str, torch.Tensor] = field(default_factory=dict)
    active_source_ids: tuple[int, ...] = field(default_factory=tuple)
    candidate_layout: CandidateLayout | None = None


__all__ = ["Candidate", "CandidateBatch", "CandidateLayout"]
