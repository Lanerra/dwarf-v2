from __future__ import annotations

import torch
import torch.nn as nn

from .candidate_types import CandidateEvidenceBit


class DSQGWEvidencePriorComposer(nn.Module):
    """Bounded scalar evidence prior for DSQG-W candidate scoring.

    The module is intentionally scalar and row-centered so it can feed the
    existing cand_scores path without changing kernel ABI. All learned weights
    default to zero, making the composer a strict no-op until enabled scales
    learn away from zero.
    """

    def __init__(self, *, n_types: int, n_sources: int, clip: float = 2.0, init_scale: float = 0.0) -> None:
        super().__init__()
        if clip <= 0.0:
            raise ValueError("clip must be positive")
        self.n_types = int(n_types)
        self.n_sources = int(n_sources)
        self.clip = float(clip)
        self.type_bias = nn.Parameter(torch.zeros(n_types))
        self.source_bias = nn.Parameter(torch.zeros(n_sources))
        self.feature_scale = nn.Parameter(torch.full((5,), float(init_scale)))

    def forward(
        self,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        raw_hisa_scores: torch.Tensor | None = None,
        evidence_bits: torch.Tensor | None = None,
        evidence_count: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cand_types.shape != cand_mask.shape or cand_sources.shape != cand_mask.shape:
            raise ValueError("candidate type/source/mask tensors must align")
        dtype = self.type_bias.dtype
        device = cand_types.device
        valid = cand_mask.bool()
        types = cand_types.clamp(0, self.n_types - 1)
        sources = cand_sources.clamp(0, self.n_sources - 1)
        prior = self.type_bias[types].to(device=device, dtype=dtype) + self.source_bias[sources].to(device=device, dtype=dtype)

        if raw_hisa_scores is None:
            score = torch.zeros_like(prior)
        else:
            score = torch.nan_to_num(raw_hisa_scores.to(device=device, dtype=dtype), nan=0.0, neginf=0.0, posinf=0.0)
        valid_f = valid.to(dtype)
        denom = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = score.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / denom
        var = ((score - mean).masked_fill(~valid, 0.0).square().sum(dim=-1, keepdim=True) / denom).clamp_min(1e-6)
        score_z = ((score - mean) / var.sqrt()).clamp(-5.0, 5.0).masked_fill(~valid, 0.0)

        masked_score = score.masked_fill(~valid, -float("inf"))
        order = masked_score.argsort(dim=-1, descending=True)
        ranks = torch.zeros_like(order)
        rank_values = torch.arange(order.shape[-1], device=device, dtype=order.dtype).expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        rank_feature = torch.rsqrt(ranks.to(dtype) + 1.0).masked_fill(~valid, 0.0)

        if evidence_count is None:
            count_feature = torch.zeros_like(prior)
        else:
            count_feature = torch.log1p(evidence_count.to(device=device, dtype=dtype)).masked_fill(~valid, 0.0)

        if evidence_bits is None:
            qh_feature = torch.zeros_like(prior)
            multi_fraction = prior.new_tensor(0.0)
        else:
            bits = evidence_bits.to(device=device, dtype=torch.long)
            qh = ((bits & int(CandidateEvidenceBit.QUESTION)) != 0) & ((bits & int(CandidateEvidenceBit.HISA)) != 0) & valid
            qh_feature = qh.to(dtype)
            if evidence_count is not None and valid.any():
                multi_fraction = ((evidence_count.to(device=device) > 1) & valid).to(dtype).sum() / valid_f.sum().clamp_min(1.0)
            else:
                multi_fraction = prior.new_tensor(0.0)

        if candidate_distances is None:
            distance_feature = torch.zeros_like(prior)
        else:
            distance_feature = torch.log1p(candidate_distances.to(device=device, dtype=dtype)).masked_fill(~valid, 0.0)

        features = (score_z, rank_feature, count_feature, qh_feature, -distance_feature)
        for idx, feature in enumerate(features):
            prior = prior + self.feature_scale[idx].to(dtype=dtype, device=device) * feature

        centered = prior - prior.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / denom
        centered = centered.masked_fill(~valid, 0.0)
        clipped = centered.clamp(-self.clip, self.clip).masked_fill(~valid, 0.0)
        valid_values = clipped.masked_select(valid)
        if valid_values.numel() == 0:
            zero = clipped.sum() * 0.0
            telemetry = {
                "dsqg_w_prior_norm": zero.detach(),
                "dsqg_w_prior_abs_mean": zero.detach(),
                "dsqg_w_prior_std": zero.detach(),
                "dsqg_w_prior_clip_fraction": zero.detach(),
                "dsqg_w_prior_multi_evidence_fraction": zero.detach(),
            }
        else:
            clipped_any = ((centered.abs() > self.clip) & valid).to(dtype).sum() / valid_f.sum().clamp_min(1.0)
            telemetry = {
                "dsqg_w_prior_norm": (valid_values.norm() / valid_values.numel()).detach(),
                "dsqg_w_prior_abs_mean": valid_values.abs().mean().detach(),
                "dsqg_w_prior_std": valid_values.std(unbiased=False).detach(),
                "dsqg_w_prior_clip_fraction": clipped_any.detach(),
                "dsqg_w_prior_multi_evidence_fraction": multi_fraction.detach(),
                "dsqg_w_prior_feature_scale_norm": self.feature_scale.detach().norm(),
                "dsqg_w_prior_type_bias_norm": self.type_bias.detach().norm(),
                "dsqg_w_prior_source_bias_norm": self.source_bias.detach().norm(),
            }
        return clipped.to(dtype=raw_hisa_scores.dtype if raw_hisa_scores is not None else torch.float32), telemetry


__all__ = ["DSQGWEvidencePriorComposer"]
