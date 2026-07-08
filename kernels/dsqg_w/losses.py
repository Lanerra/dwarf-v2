from __future__ import annotations

import torch
import torch.nn.functional as F

from .candidate_types import CandidateType


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
    copy competitors that are not valid gold answer tokens. Gold-token entries
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

    gold_evidence_indices is [B,T,K] or [B,K]. Values <0 are ignored. Recall is
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


__all__ = [
    "answer_masked_loss",
    "conditional_copy_unlikelihood_loss",
    "local_mass_cap_loss",
    "entropy_floor_loss",
    "candidate_recall",
    "_mean_head_probs",
]
