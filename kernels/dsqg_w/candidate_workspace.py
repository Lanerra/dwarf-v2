from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate_types import CandidateSource
from .config import DSQGWConfig


@dataclass(frozen=True)
class CandidateWorkspaceOutput:
    workspace: torch.Tensor
    score_bias: torch.Tensor
    telemetry: dict[str, torch.Tensor] = field(default_factory=dict)


class CandidateWorkspace(nn.Module):
    """Low-rank per-token evidence workspace for sourcewise DSQG-W.

    This module is deliberately not a second D-width attention stack.  It projects
    source surfaces once to ``w``, gathers selected evidence as [B,T,J,w], adds
    static slot/type/source and score/distance features, then emits a scalar
    candidate score bias for the sourcewise reader.
    """

    def __init__(
        self,
        *,
        d: int,
        n_types: int,
        n_sources: int,
        workspace_dim: int,
        phase_bands: int = 4,
        max_distance: int = 8192,
        use_score_features: bool = True,
        use_query_scores: bool = True,
        use_pair_transfer: bool = False,
        pair_gate_init: float = -2.5,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("d must be positive")
        if n_types <= 0 or n_sources <= 0:
            raise ValueError("n_types and n_sources must be positive")
        if workspace_dim <= 0:
            raise ValueError("workspace_dim must be positive")
        if phase_bands <= 0:
            raise ValueError("phase_bands must be positive")
        self.d = int(d)
        self.n_types = int(n_types)
        self.n_sources = int(n_sources)
        self.workspace_dim = int(workspace_dim)
        self.phase_bands = int(phase_bands)
        self.max_distance = int(max_distance)
        self.use_score_features = bool(use_score_features)
        self.use_query_scores = bool(use_query_scores)
        self.use_pair_transfer = bool(use_pair_transfer)
        self.eps = 1e-6

        self.source_norm = nn.LayerNorm(d)
        self.source_proj = nn.Linear(d, workspace_dim, bias=False)
        self.type_embed = nn.Embedding(n_types, workspace_dim)
        self.source_embed = nn.Embedding(n_sources, workspace_dim)
        self.phase_proj = nn.Linear(2 * phase_bands, workspace_dim, bias=False)
        self.score_proj = nn.Linear(1, workspace_dim, bias=False)
        self.workspace_norm = nn.LayerNorm(workspace_dim)
        self.score_head = nn.Linear(workspace_dim, 1, bias=False)
        self.query_proj = nn.Linear(d, workspace_dim, bias=False)
        self.query_key = nn.Linear(workspace_dim, workspace_dim, bias=False)
        if self.use_pair_transfer:
            self.pair_q = nn.Linear(workspace_dim, workspace_dim, bias=False)
            self.pair_k = nn.Linear(workspace_dim, workspace_dim, bias=False)
            self.pair_v = nn.Linear(workspace_dim, workspace_dim, bias=False)
            self.pair_gate = nn.Parameter(torch.tensor(float(pair_gate_init)))
        else:
            self.pair_q = None
            self.pair_k = None
            self.pair_v = None
            self.register_parameter("pair_gate", None)

    @classmethod
    def from_config(cls, config: DSQGWConfig) -> "CandidateWorkspace":
        return cls(
            d=config.d,
            n_types=config.n_types,
            n_sources=config.n_sources,
            workspace_dim=config.candidate_workspace_dim,
            phase_bands=config.candidate_workspace_phase_bands,
            use_score_features=config.candidate_workspace_score_features,
            use_query_scores=config.candidate_workspace_query_scores,
            use_pair_transfer=config.candidate_workspace_pair_transfer,
            pair_gate_init=config.candidate_workspace_pair_gate_init,
        )

    @staticmethod
    def _gather_source_rows(states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, width = states.shape
        safe = token_indices.to(device=states.device, dtype=torch.long).clamp(0, max(seq_len - 1, 0))
        if safe.ndim == 2:
            return torch.gather(states, 1, safe[:, :, None].expand(bsz, seq_len, width))
        if safe.ndim == 3:
            j_count = safe.shape[-1]
            expanded_states = states[:, :, None, :].expand(bsz, seq_len, j_count, width)
            return torch.gather(expanded_states, 1, safe[..., None].expand(bsz, seq_len, j_count, width))
        raise ValueError("token_indices must have shape [B,T] or [B,T,J]")

    def _project_sources(
        self,
        x: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        needed_source_ids: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        final_states = x
        l3_base = l3_states if l3_states is not None else final_states
        summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
        zero_base = torch.zeros_like(final_states)
        bases: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): final_states,
            int(CandidateSource.QUESTION_CACHE): final_states,
            int(CandidateSource.L3): l3_base,
            int(CandidateSource.HISA): l3_base,
            int(CandidateSource.SUMMARY): summary_base,
            int(CandidateSource.NULL): zero_base,
        }
        needed = None if needed_source_ids is None else {int(source_id) for source_id in needed_source_ids}
        projected_by_object: dict[int, torch.Tensor] = {}
        projected: dict[int, torch.Tensor] = {}
        for source_id, states in bases.items():
            if needed is not None and source_id not in needed:
                continue
            cache_key = id(states)
            source_w = projected_by_object.get(cache_key)
            if source_w is None:
                source_w = self.source_proj(self.source_norm(states))
                projected_by_object[cache_key] = source_w
            projected[source_id] = source_w
        return projected

    def _phase_features(
        self,
        candidate_distances: torch.Tensor | None,
        bsz: int,
        seq_len: int,
        j_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if candidate_distances is None:
            distances = torch.zeros((bsz, seq_len, j_count), device=device, dtype=dtype)
        else:
            if candidate_distances.shape != (bsz, seq_len, j_count):
                raise ValueError("candidate_distances must have shape [B,T,J]")
            distances = candidate_distances.to(device=device, dtype=dtype).clamp_min(0.0)
        phase = torch.log1p(distances) / math.log1p(float(max(self.max_distance, 1)))
        bands = torch.arange(1, self.phase_bands + 1, device=device, dtype=dtype)
        angles = phase[..., None] * bands * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _score_features(
        self,
        cand_scores: torch.Tensor | None,
        valid: torch.Tensor,
        bsz: int,
        seq_len: int,
        j_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if cand_scores is None:
            return torch.zeros((bsz, seq_len, j_count), device=device, dtype=dtype)
        if cand_scores.shape != (bsz, seq_len, j_count):
            raise ValueError("cand_scores must have shape [B,T,J]")
        scores = torch.nan_to_num(cand_scores.to(device=device, dtype=dtype), nan=0.0, neginf=0.0, posinf=0.0)
        scores = scores.masked_fill(~valid, 0.0)
        denom = valid.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = scores.sum(dim=-1, keepdim=True) / denom
        var = ((scores - mean).masked_fill(~valid, 0.0).square().sum(dim=-1, keepdim=True) / denom).clamp_min(1e-6)
        return ((scores - mean) / var.sqrt()).clamp(-5.0, 5.0).masked_fill(~valid, 0.0)

    def _apply_pair_transfer(
        self,
        workspace: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_pair_transfer:
            return workspace, workspace.new_tensor(0.0)
        assert self.pair_q is not None and self.pair_k is not None and self.pair_v is not None and self.pair_gate is not None
        q = self.pair_q(workspace)
        k = self.pair_k(workspace)
        v = self.pair_v(workspace)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.workspace_dim))
        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1).masked_fill(~valid_pair, 0.0)
        transfer = torch.matmul(probs, v)
        gate = torch.sigmoid(self.pair_gate)
        return workspace + gate * transfer * cand_mask[..., None].to(transfer.dtype), gate.detach()

    def forward_sourcewise(
        self,
        x: torch.Tensor,
        cand_token_indices: torch.Tensor,
        cand_types: torch.Tensor,
        cand_sources: torch.Tensor,
        cand_mask: torch.Tensor,
        *,
        l3_states: torch.Tensor | None = None,
        chunk_rep_states: torch.Tensor | None = None,
        cand_scores: torch.Tensor | None = None,
        candidate_distances: torch.Tensor | None = None,
        needed_source_ids: tuple[int, ...] | None = None,
    ) -> CandidateWorkspaceOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,D]")
        bsz, seq_len, d = x.shape
        if d != self.d:
            raise ValueError(f"x last dim {d} does not match workspace d {self.d}")
        if cand_token_indices.shape != cand_mask.shape or cand_types.shape != cand_mask.shape or cand_sources.shape != cand_mask.shape:
            raise ValueError("candidate metadata tensors must have shape [B,T,J]")
        if cand_mask.shape[:2] != (bsz, seq_len):
            raise ValueError("candidate metadata shape mismatch")
        if l3_states is not None and l3_states.shape != x.shape:
            raise ValueError("l3_states must match x shape")
        if chunk_rep_states is not None and chunk_rep_states.shape != x.shape:
            raise ValueError("chunk_rep_states must match x shape")

        device = x.device
        dtype = x.dtype
        j_count = cand_mask.shape[-1]
        valid = cand_mask.to(device=device, dtype=torch.bool)
        safe_types = cand_types.to(device=device, dtype=torch.long).clamp(0, self.n_types - 1)
        safe_sources = cand_sources.to(device=device, dtype=torch.long).clamp(0, self.n_sources - 1)
        gather_tokens = cand_token_indices.to(device=device, dtype=torch.long).clamp(0, max(seq_len - 1, 0))
        if needed_source_ids is None:
            needed_source_ids = tuple(
                int(source_id)
                for source_id in CandidateSource
                if bool(((safe_sources == int(source_id)) & valid).any())
            )
        projected_sources = self._project_sources(
            x,
            l3_states=l3_states,
            chunk_rep_states=chunk_rep_states,
            needed_source_ids=needed_source_ids,
        )
        workspace = x.new_zeros((bsz, seq_len, j_count, self.workspace_dim))
        for source_id, source_w in projected_sources.items():
            source_mask = (safe_sources == int(source_id)) & valid
            if bool(source_mask.any()):
                gathered = self._gather_source_rows(source_w, gather_tokens.reshape(bsz, seq_len, j_count))
                workspace = workspace + gathered * source_mask[..., None].to(dtype)
        workspace = workspace + self.type_embed(safe_types).to(dtype=dtype)
        workspace = workspace + self.source_embed(safe_sources).to(dtype=dtype)
        workspace = workspace + self.phase_proj(
            self._phase_features(candidate_distances, bsz, seq_len, j_count, device, dtype)
        )
        if self.use_score_features:
            workspace = workspace + self.score_proj(
                self._score_features(cand_scores, valid, bsz, seq_len, j_count, device, dtype)[..., None]
            )
        workspace = workspace.masked_fill(~valid[..., None], 0.0)
        workspace, pair_gate = self._apply_pair_transfer(workspace, valid)
        workspace = workspace.masked_fill(~valid[..., None], 0.0)
        workspace_n = self.workspace_norm(workspace)
        base_score_bias = self.score_head(workspace_n).squeeze(-1)
        query_score = workspace.new_zeros((bsz, seq_len, j_count))
        if self.use_query_scores:
            query_w = self.query_proj(self.source_norm(x))
            key_w = self.query_key(workspace_n)
            query_score = (query_w[:, :, None, :] * key_w).sum(dim=-1) / math.sqrt(float(self.workspace_dim))
            query_score = query_score.masked_fill(~valid, 0.0)
        score_bias = base_score_bias + query_score
        score_bias = score_bias.masked_fill(~valid, 0.0)
        denom = valid.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = score_bias - (score_bias.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / denom)
        score_bias = centered.masked_fill(~valid, 0.0)
        bias_norm = score_bias.masked_select(valid).norm() / valid.to(dtype).sum().clamp_min(1.0)
        query_score_norm = query_score.masked_select(valid).norm() / valid.to(dtype).sum().clamp_min(1.0)
        workspace_norm = workspace.norm(dim=-1).masked_select(valid).mean() if valid.any() else x.new_tensor(0.0)
        telemetry = {
            "dsqg_w_candidate_workspace_enabled": x.new_tensor(1.0).detach(),
            "dsqg_w_candidate_workspace_dim": x.new_tensor(float(self.workspace_dim)).detach(),
            "dsqg_w_candidate_workspace_score_bias_norm": bias_norm.detach(),
            "dsqg_w_candidate_workspace_query_conditioned": x.new_tensor(1.0 if self.use_query_scores else 0.0).detach(),
            "dsqg_w_candidate_workspace_query_score_norm": query_score_norm.detach(),
            "dsqg_w_candidate_workspace_norm": workspace_norm.detach(),
            "dsqg_w_candidate_workspace_pair_transfer": x.new_tensor(1.0 if self.use_pair_transfer else 0.0).detach(),
            "dsqg_w_candidate_workspace_pair_gate": pair_gate.detach(),
            "dsqg_w_candidate_workspace_materialized_d_candidates": x.new_tensor(0.0).detach(),
        }
        return CandidateWorkspaceOutput(workspace=workspace, score_bias=score_bias.to(dtype=x.dtype), telemetry=telemetry)


__all__ = ["CandidateWorkspace", "CandidateWorkspaceOutput"]
