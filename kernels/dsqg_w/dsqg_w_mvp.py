from __future__ import annotations

# Compatibility shim for the historical DSQG-W monolith.
#
# The implementation now lives in split modules under kernels.dsqg_w. This file
# intentionally preserves old public/private import paths while avoiding duplicate
# active definitions in the monolith. Do not add implementation code here; add it
# to the canonical split module and re-export only if legacy callers need it.

from .candidate_types import (
    CandidateEvidenceBit,
    CandidateSource,
    CandidateType,
    _CANDIDATE_PRIORITY,
)
from .triton_schedule import (
    _DSQGWTritonSchedule,
    _dsqg_w_triton_schedule,
    _next_pow2_int,
)
from .instrumentation import (
    _dsqg_w_geometry_audit_enabled,
    _dsqg_w_geometry_telemetry,
    _dsqg_w_profile_enabled,
    _dsqg_w_profile_range,
)
from .gates import _forced_gate_value
from .sourcewise_gather import (
    _DSQGWSourcewiseCandidateStateGather,
    _dsqg_w_candidate_state_gather_backward_kernel,
    _dsqg_w_candidate_state_gather_kernel,
)
from .config import DSQGWConfig
from .candidate_batch import Candidate, CandidateBatch, CandidateLayout
from .candidate_provider import CandidateProvider
from .width_cell import (
    DSQGWWidthCell,
    _hisa_evidence_type_mask,
    width_pair_transfer_loss,
)
from .evidence_prior import DSQGWEvidencePriorComposer
from .candidate_workspace import CandidateWorkspace, CandidateWorkspaceOutput
from .typed_mixer import DSQGWTypedCandidateMixer
from .ebh_packet import DSQGWEvidenceBindingHub
from .sourcewise_read import (
    _DSQGWMaterializedTritonCompactRead,
    _DSQGWSourcewiseTritonCompactRead,
    _DSQGWSourcewiseTritonRecompute,
    _TRITON_SOURCEWISE_AVAILABLE,
    _dsqg_w_materialized_read_slots_recompute,
    _dsqg_w_sourcewise_functional_recompute,
    _dsqg_w_sourcewise_read_slots_backward_kernel,
    _dsqg_w_sourcewise_read_slots_kernel,
    _dsqg_w_sourcewise_read_slots_recompute,
    _dsqg_w_sourcewise_score_read_kernel,
    triton,
    tl,
)
from .block import DSQGWBlock, _read_type_ids_from_config
from .losses import (
    answer_masked_loss,
    candidate_recall,
    conditional_copy_unlikelihood_loss,
    entropy_floor_loss,
    local_mass_cap_loss,
    _mean_head_probs,
)

__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateLayout",
    "CandidateWorkspace",
    "CandidateWorkspaceOutput",
    "CandidateEvidenceBit",
    "CandidateProvider",
    "CandidateSource",
    "CandidateType",
    "DSQGWBlock",
    "DSQGWConfig",
    "DSQGWEvidenceBindingHub",
    "DSQGWEvidencePriorComposer",
    "DSQGWTypedCandidateMixer",
    "DSQGWWidthCell",
    "answer_masked_loss",
    "candidate_recall",
    "conditional_copy_unlikelihood_loss",
    "entropy_floor_loss",
    "local_mass_cap_loss",
    "width_pair_transfer_loss",
]
