from __future__ import annotations

from enum import IntEnum


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


class CandidateEvidenceBit(IntEnum):
    HISA = 1 << 0
    QUESTION = 1 << 1
    L3_SKIP = 1 << 2
    LOCAL = 1 << 3
    LONG_OFFSET = 1 << 4
    CHUNK_REP = 1 << 5
    NULL = 1 << 6


# Lower is better. This matches the DWARF v2 proposal: semantic evidence/cues
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


__all__ = [
    "CandidateType",
    "CandidateSource",
    "CandidateEvidenceBit",
    "_CANDIDATE_PRIORITY",
]
