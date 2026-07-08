from __future__ import annotations

# Legacy alias: the guarded EBH pair mixer remains implemented on the canonical
# hub class until a later behavior-preserving extraction.
from ..ebh_packet import DSQGWEvidenceBindingHub

PairMixerEvidenceBindingHub = DSQGWEvidenceBindingHub

__all__ = ["DSQGWEvidenceBindingHub", "PairMixerEvidenceBindingHub"]
