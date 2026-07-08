from __future__ import annotations

# Keep package import light. Public DSQG-W symbols are loaded lazily from the
# compatibility monolith so existing submodule imports keep their behavior while
# the split modules are populated incrementally.
_LEGACY_PUBLIC_EXPORTS = {
    "DSQGWConfig",
    "CandidateProvider",
    "Candidate",
    "CandidateBatch",
    "CandidateType",
    "CandidateSource",
    "CandidateEvidenceBit",
    "DSQGWBlock",
    "DSQGWWidthCell",
    "DSQGWTypedCandidateMixer",
    "DSQGWEvidenceBindingHub",
    "DSQGWEvidencePriorComposer",
    "width_pair_transfer_loss",
    "answer_masked_loss",
    "conditional_copy_unlikelihood_loss",
    "local_mass_cap_loss",
    "entropy_floor_loss",
    "candidate_recall",
}

__all__ = sorted(_LEGACY_PUBLIC_EXPORTS)


def __getattr__(name: str):
    if name in _LEGACY_PUBLIC_EXPORTS:
        from . import dsqg_w_mvp as _dsqg_w_mvp

        return getattr(_dsqg_w_mvp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
