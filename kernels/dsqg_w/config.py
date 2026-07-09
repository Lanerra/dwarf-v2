from __future__ import annotations

from dataclasses import dataclass

from .candidate_types import CandidateSource, CandidateType


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
    use_evidence_prior: bool = False
    evidence_prior_clip: float = 2.0
    evidence_prior_init_scale: float = 0.0
    use_evidence_binding_hub: bool = False
    ebh_bottleneck: int = 256
    ebh_gate_init: float = -5.0
    ebh_phase_bands: int = 4
    ebh_score_features: bool = True
    ebh_sourcewise_packet: bool = False
    ebh_triton_lane_accum: bool = False
    ebh_pair_mixer: bool = False
    ebh_pair_rank: int = 64
    ebh_pair_gate_init: float = -2.5
    use_candidate_workspace: bool = False
    candidate_workspace_dim: int = 64
    candidate_workspace_phase_bands: int = 4
    candidate_workspace_score_features: bool = True
    candidate_workspace_pair_transfer: bool = False
    candidate_workspace_pair_gate_init: float = -2.5
    use_candidate_quotas: bool = False
    quota_hisa_max: int = 0
    read_type_ids: tuple[int, ...] | None = None

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
        if self.evidence_prior_clip <= 0.0:
            raise ValueError("evidence_prior_clip must be positive")
        if self.ebh_bottleneck <= 0:
            raise ValueError("ebh_bottleneck must be positive")
        if self.ebh_phase_bands <= 0:
            raise ValueError("ebh_phase_bands must be positive")
        if self.ebh_pair_rank <= 0:
            raise ValueError("ebh_pair_rank must be positive")
        if self.candidate_workspace_dim <= 0:
            raise ValueError("candidate_workspace_dim must be positive")
        if self.candidate_workspace_phase_bands <= 0:
            raise ValueError("candidate_workspace_phase_bands must be positive")
        if self.quota_hisa_max < 0:
            raise ValueError("quota_hisa_max must be non-negative")
        if self.read_type_ids is not None:
            for type_id in self.read_type_ids:
                if not 0 <= int(type_id) < self.n_types:
                    raise ValueError("read_type_ids entries must be valid candidate type ids")


__all__ = ["DSQGWConfig"]
