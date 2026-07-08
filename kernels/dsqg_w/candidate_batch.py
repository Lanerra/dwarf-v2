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


__all__ = ["Candidate", "CandidateBatch"]
