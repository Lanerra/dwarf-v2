from __future__ import annotations

# Legacy alias: materialized EBH remains a guarded method on the canonical hub
# class until a later behavior-preserving extraction.
from ..ebh_packet import DSQGWEvidenceBindingHub

MaterializedEvidenceBindingHub = DSQGWEvidenceBindingHub

__all__ = ["DSQGWEvidenceBindingHub", "MaterializedEvidenceBindingHub"]
