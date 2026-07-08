from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import _forced_gate_value


class DSQGWTypedCandidateMixer(nn.Module):
    """Small typed candidate-set mixer applied before DSQG-W query scoring.

    This is bounded to the candidate axis J. It never attends over sequence
    positions; it only lets the already-built causal candidate set exchange
    typed evidence before the main query-conditioned scoring step.
    """

    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        n_types: int,
        bottleneck: int,
        gate_init: float = -5.0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = int(d)
        self.n_heads = int(n_heads)
        self.mix_dim = int(bottleneck)
        if self.mix_dim % self.n_heads != 0:
            raise ValueError("typed mixer bottleneck must be divisible by n_heads")
        self.n_types = int(n_types)

        self.norm_c = nn.LayerNorm(d)
        self.type_embed = nn.Embedding(n_types, d)
        self.q_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.k_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.v_proj = nn.Linear(d, self.mix_dim, bias=False)
        self.out_proj = nn.Linear(self.mix_dim, d, bias=False)
        self.type_pair_bias = nn.Parameter(torch.zeros(n_types, n_types))
        self.gate = nn.Parameter(torch.full((d,), float(gate_init)))

    def forward(
        self,
        cand_states: torch.Tensor,
        cand_types: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, j_count, d = cand_states.shape
        if d != self.d:
            raise ValueError(f"cand_states last dim {d} does not match typed mixer d {self.d}")
        if cand_types.shape != (bsz, seq_len, j_count) or cand_mask.shape != cand_types.shape:
            raise ValueError("candidate type/mask tensors must have shape [B,T,J]")
        if not cand_mask.any(dim=-1).all():
            raise ValueError("typed candidate mixer received an all-invalid candidate row")

        if os.getenv("DWARF_DSQG_W_TYPED_MIXER_TYPE_EMBED", "1") == "1":
            c = self.norm_c(cand_states + self.type_embed(cand_types))
            type_embed_active = True
        else:
            c = self.norm_c(cand_states)
            type_embed_active = False
        q, k, v = F.linear(
            c,
            torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0),
        ).split(self.mix_dim, dim=-1)
        scores = torch.bmm(
            q.reshape(bsz * seq_len, j_count, self.mix_dim),
            k.reshape(bsz * seq_len, j_count, self.mix_dim).transpose(1, 2),
        ).reshape(bsz, seq_len, j_count, j_count) / math.sqrt(float(self.mix_dim))
        if os.getenv("DWARF_DSQG_W_TYPED_MIXER_PAIR_BIAS", "1") == "1":
            scores = scores + self.type_pair_bias[cand_types[:, :, :, None], cand_types[:, :, None, :]]
        valid_pair = cand_mask[:, :, :, None] & cand_mask[:, :, None, :]
        scores = scores.masked_fill(~valid_pair, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = probs.masked_fill(~valid_pair, 0.0)

        mixed = torch.bmm(
            probs.reshape(bsz * seq_len, j_count, j_count),
            v.reshape(bsz * seq_len, j_count, self.mix_dim),
        ).reshape(bsz, seq_len, j_count, self.mix_dim)
        delta = self.out_proj(mixed)
        forced_gate = _forced_gate_value("DWARF_DSQG_W_FORCE_TYPED_MIXER_GATE", device=cand_states.device, dtype=delta.dtype)
        if forced_gate is None:
            gate = torch.sigmoid(self.gate).reshape(1, 1, 1, d)
            forced_gate_flag = cand_states.new_tensor(0.0)
        else:
            gate = forced_gate.reshape(1, 1, 1, 1).expand(1, 1, 1, d)
            forced_gate_flag = cand_states.new_tensor(1.0)
        out = cand_states + gate * delta * cand_mask[..., None].to(delta.dtype)

        fast_telemetry = os.getenv("DWARF_DSQG_W_FAST_TELEMETRY", "0") == "1"
        if fast_telemetry:
            entropy = cand_states.new_tensor(0.0)
            delta_norm = cand_states.new_tensor(0.0)
        else:
            valid_targets = cand_mask.bool()
            p_safe = probs.clamp_min(1e-8)
            entropy = (-(p_safe * p_safe.log()).sum(dim=-1)).masked_select(valid_targets).mean()
            valid_delta_count = cand_mask.to(delta.dtype).sum().clamp_min(1.0)
            delta_norm = (delta.norm(dim=-1) * cand_mask.to(delta.dtype)).sum() / valid_delta_count
        telemetry = {
            "dsqg_w_typed_mixer_entropy": entropy.detach(),
            "dsqg_w_typed_mixer_gate_mean": gate.mean().detach(),
            "dsqg_w_typed_mixer_gate_min": gate.min().detach(),
            "dsqg_w_typed_mixer_gate_max": gate.max().detach(),
            "dsqg_w_typed_mixer_forced_gate": forced_gate_flag.detach(),
            "dsqg_w_typed_mixer_gate_logit_mean": self.gate.detach().mean(),
            "dsqg_w_typed_mixer_delta_norm": delta_norm.detach(),
            "dsqg_w_typed_mixer_type_embed_active": cand_states.new_tensor(1.0 if type_embed_active else 0.0).detach(),
        }
        return out, telemetry


__all__ = ["DSQGWTypedCandidateMixer"]
