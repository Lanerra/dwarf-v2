from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate_types import CandidateType
from .gates import _forced_gate_value


def _hisa_evidence_type_mask(cand_types: torch.Tensor) -> torch.Tensor:
    """Return mask for all concrete HISA evidence candidate type IDs.

    Typed HISA representatives are semantically still HISA evidence for width
    transfer objectives/telemetry. Keeping this predicate centralized prevents
    aux losses from silently going inactive when typed_hisa_reps=True.
    """

    return (
        (cand_types == int(CandidateType.HISA_EVIDENCE))
        | (cand_types == int(CandidateType.HISA_EVIDENCE_REP0))
        | (cand_types == int(CandidateType.HISA_EVIDENCE_REP1))
        | (cand_types == int(CandidateType.HISA_EVIDENCE_REP2))
        | (cand_types == int(CandidateType.HISA_EVIDENCE_REP3))
    )


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

    def direction_mass(target_mask: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        target_mask = target_mask & valid_targets
        source_mask = source_mask & cand_mask.bool()
        if not target_mask.any():
            return p.sum() * 0.0
        mass = p.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
        selected = mass.masked_select(target_mask)
        if selected.numel() == 0:
            return p.sum() * 0.0
        return selected.mean()

    question_mask = cand_types == int(CandidateType.QUESTION)
    hisa_family_mask = _hisa_evidence_type_mask(cand_types)
    q_to_hisa = direction_mass(question_mask, hisa_family_mask)
    hisa_to_q = direction_mass(hisa_family_mask, question_mask)
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
        self.rel_diff_proj = nn.Linear(d, self.width_dim, bias=False)
        self.rel_prod_proj = nn.Linear(d, self.width_dim, bias=False)
        self.rel_diff_score = nn.Parameter(torch.empty(self.width_dim))
        self.rel_prod_score = nn.Parameter(torch.empty(self.width_dim))
        self.lateral_up = nn.Linear(self.width_dim, d, bias=False)
        self.type_pair_bias = nn.Parameter(torch.zeros(n_types, n_types))
        self.source_pair_bias = nn.Parameter(torch.zeros(n_sources, n_sources))
        self.self_bias = nn.Parameter(torch.tensor(float(self_bias_init)))
        self.gate = nn.Parameter(torch.full((d,), float(gate_init)))
        nn.init.normal_(self.rel_diff_score, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_prod_score, mean=0.0, std=0.02)

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
        q, k, v, rel_diff, rel_prod = F.linear(
            c_n,
            torch.cat(
                [
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                    self.rel_diff_proj.weight,
                    self.rel_prod_proj.weight,
                ],
                dim=0,
            ),
        ).split(self.width_dim, dim=-1)

        scores = torch.bmm(
            q.reshape(bsz * seq_len, j_count, self.width_dim),
            k.reshape(bsz * seq_len, j_count, self.width_dim).transpose(1, 2),
        ).reshape(bsz, seq_len, j_count, j_count) / math.sqrt(float(self.width_dim))
        rel_diff_hidden = torch.tanh(rel_diff[:, :, :, None, :] - rel_diff[:, :, None, :, :])
        rel_prod_hidden = torch.tanh(rel_prod[:, :, :, None, :] * rel_prod[:, :, None, :, :])
        scores = scores + (
            rel_diff_hidden * self.rel_diff_score.reshape(1, 1, 1, 1, self.width_dim)
        ).sum(dim=-1) / math.sqrt(float(self.width_dim))
        scores = scores + (
            rel_prod_hidden * self.rel_prod_score.reshape(1, 1, 1, 1, self.width_dim)
        ).sum(dim=-1) / math.sqrt(float(self.width_dim))
        scores = scores + self.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        scores = scores + self.source_pair_bias[cand_sources[:, :, :, None], cand_sources[:, :, None, :]]
        scores.diagonal(dim1=-2, dim2=-1).add_(self.self_bias)

        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=3)
        probs = probs.masked_fill(~valid_pair, 0.0)

        lateral = torch.bmm(
            probs.reshape(bsz * seq_len, j_count, j_count),
            v.reshape(bsz * seq_len, j_count, self.width_dim),
        ).reshape(bsz, seq_len, j_count, self.width_dim)
        delta = self.lateral_up(lateral)
        forced_gate = _forced_gate_value("DWARF_DSQG_W_FORCE_WIDTH_GATE", device=cand_states.device, dtype=delta.dtype)
        if forced_gate is None:
            gate = torch.sigmoid(self.gate).reshape(1, 1, 1, d)
            forced_gate_flag = cand_states.new_tensor(0.0)
        else:
            gate = forced_gate.reshape(1, 1, 1, 1).expand(1, 1, 1, d)
            forced_gate_flag = cand_states.new_tensor(1.0)
        out = cand_states + gate * delta * cand_mask[..., None].to(delta.dtype)

        p_mean = probs
        valid_targets = cand_mask.bool()
        p_safe = p_mean.clamp_min(1e-8)
        entropy_per_target = -(p_safe * p_safe.log()).sum(dim=-1)
        entropy = entropy_per_target.masked_select(valid_targets).mean()
        diag = torch.eye(j_count, device=cand_states.device, dtype=torch.bool).reshape(1, 1, j_count, j_count)
        self_mass = p_mean.masked_fill(~diag, 0.0).sum(dim=-1).masked_select(valid_targets).mean()

        def pair_mass(target_mask: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
            target_mask = target_mask & valid_targets
            source_mask = source_mask & cand_mask
            if not target_mask.any():
                return cand_states.new_tensor(0.0)
            mass = p_mean.masked_fill(~source_mask[:, :, None, :], 0.0).sum(dim=-1)
            return mass.masked_select(target_mask).mean()

        question_mask = cand_types == int(CandidateType.QUESTION)
        hisa_family_mask = _hisa_evidence_type_mask(cand_types)

        # Avoid masked_select(cand_mask[..., None]) here: at trainer shape (BS=16,
        # N=2048, J=11, D=512) it materializes a multi-GB boolean-expanded copy
        # purely for telemetry and can OOM before the width-cell gate is measured.
        valid_delta_count = cand_mask.to(delta.dtype).sum().clamp_min(1.0)
        delta_norm = (delta.norm(dim=-1) * cand_mask.to(delta.dtype)).sum() / valid_delta_count
        transfer_aux_loss = width_pair_transfer_loss(p_mean, cand_types, cand_mask)
        entropy_penalty = torch.relu(entropy.new_tensor(self.entropy_floor) - entropy)
        aux_loss = transfer_aux_loss + self.entropy_weight * entropy_penalty
        telemetry = {
            "dsqg_w_width_entropy": entropy.detach(),
            "dsqg_w_width_self_mass": self_mass.detach(),
            "dsqg_w_width_gate_mean": gate.mean().detach(),
            "dsqg_w_width_gate_min": gate.min().detach(),
            "dsqg_w_width_gate_max": gate.max().detach(),
            "dsqg_w_width_forced_gate": forced_gate_flag.detach(),
            "dsqg_w_width_gate_logit_mean": self.gate.detach().mean(),
            "dsqg_w_width_delta_norm": delta_norm.detach(),
            "dsqg_w_width_aux_loss": aux_loss,
            "dsqg_w_width_aux_loss_value": aux_loss.detach(),
            "dsqg_w_width_transfer_aux_loss": transfer_aux_loss.detach(),
            "dsqg_w_width_entropy_penalty": entropy_penalty.detach(),
            "dsqg_w_width_entropy_floor": entropy.new_tensor(self.entropy_floor).detach(),
            "dsqg_w_width_entropy_weight": entropy.new_tensor(self.entropy_weight).detach(),
            "dsqg_w_width_question_to_hisa_evidence_mass": pair_mass(question_mask, hisa_family_mask).detach(),
            "dsqg_w_width_hisa_evidence_to_question_mass": pair_mass(hisa_family_mask, question_mask).detach(),
            "dsqg_w_width_rel_diff_score_norm": self.rel_diff_score.detach().norm(),
            "dsqg_w_width_rel_prod_score_norm": self.rel_prod_score.detach().norm(),
        }
        return out, telemetry


__all__ = [
    "DSQGWWidthCell",
    "_hisa_evidence_type_mask",
    "width_pair_transfer_loss",
]
