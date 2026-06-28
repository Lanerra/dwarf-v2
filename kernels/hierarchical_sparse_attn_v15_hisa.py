"""
Hierarchical Sparse Attention V15.1-HISA — corrected compact token refinement.

This version fixes the main V15 Stage-2 bug: token refinement is now scoped by
query chunk and selected chunk instead of using one global per-position mask.
The Triton kernels consume compact token indices [B, H, C_query, K, M] and only
load/dot the selected M tokens for each selected chunk, so the top-m refinement
also reduces QK/V work instead of computing a full key chunk and masking it
afterward.

Stage 1: select top-k chunks per query chunk, with the self chunk guaranteed.
Stage 2: within each selected chunk, select top-m key tokens for that query
         chunk. Selection metadata is built under no_grad; the train-time
         routing-gradient path is preserved by adding log(routing_weight) in
         the attention kernel.

V15.1 changes:
  1. Query tiling: kernels now tile the query dimension (BLOCK_Q <= 128, third
     grid axis) instead of mapping the entire chunk (chunk_size = ceil(N/C),
     which grows with N for fixed C) to one program. Removes the register-spill
     / launch-failure cliff at long context.
  2. Stage-2 scoring streams over query-row blocks with a running max instead
     of materializing [B,H,K,chunk,chunk] scores. Memory only; the selection
     and its O(N^2*K/C) compute are semantically unchanged.
  3. Train/eval consistency: the log(routing_weight) score bias used to train
     routing is now also applied at eval by default (routing_bias_in_eval=True)
     so the evaluated function matches the trained one. Temperature likewise
     no longer silently switches to 1.0 at eval. Set routing_bias_in_eval=False
     or DWARF_HISA_ROUTING_BIAS_IN_EVAL=0 for the legacy/no-bias eval behavior.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl



def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _env_flag(name: str, default: bool, *aliases: str) -> bool:
    """Parse a boolean environment flag with conservative true/false spellings."""
    raw = None
    raw_name = name
    for key in (name, *aliases):
        if key in os.environ:
            raw = os.environ[key]
            raw_name = key
            break
    if raw is None or raw == "":
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{raw_name} must be a boolean flag, got {raw!r}")


def _compute_chunk_representatives(
    K_pad: torch.Tensor,
    num_chunks: int,
    valid_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build Stage-1 chunk representatives without mean-pooling away a standout key.

    We select the highest-L2 token within each chunk and use that exact token
    vector as the chunk representative. This preserves a real token direction
    instead of constructing a synthetic per-dimension max vector, and it avoids
    the 1/chunk_size signal dilution of mean pooling.
    """
    B, H, N_padded, hd = K_pad.shape
    assert N_padded % num_chunks == 0, "K_pad must be padded to chunk boundaries"

    chunk_size = N_padded // num_chunks
    K_chunks = K_pad.reshape(B, H, num_chunks, chunk_size, hd)

    token_energy = K_chunks.float().square().sum(dim=-1)
    if valid_lengths is not None:
        valid_lengths = valid_lengths.to(device=K_pad.device, dtype=torch.long).reshape(B)
        positions = torch.arange(N_padded, device=K_pad.device).reshape(1, 1, N_padded)
        valid = positions < valid_lengths.reshape(B, 1, 1)
        valid = valid.reshape(B, 1, num_chunks, chunk_size).expand(-1, H, -1, -1)
        token_energy = token_energy.masked_fill(~valid, float('-inf'))
    best_idx = token_energy.argmax(dim=3, keepdim=True)
    gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, -1, 1, hd)
    reps = torch.gather(K_chunks, dim=3, index=gather_idx).squeeze(3)
    if valid_lengths is not None:
        chunk_starts = torch.arange(num_chunks, device=K_pad.device) * chunk_size
        chunk_has_valid = chunk_starts.reshape(1, num_chunks) < valid_lengths.reshape(B, 1)
        reps = reps.masked_fill(~chunk_has_valid[:, None, :, None], 0.0)
    return reps


def _compute_routing(
    Q: torch.Tensor,
    chunk_reps: torch.Tensor,
    *,
    seq_len: int,
    num_chunks: int,
    chunk_size: int,
    hd: int,
    temperature: float,
    training: bool,
) -> tuple[torch.Tensor, float]:
    """
    Compute routing logits and weights with causal masking and temperature.

    Returns (routing_weights, effective_temperature).
    routing_weights shape: (B, H, N, num_chunks)
    """
    device = Q.device

    routing_logits = torch.matmul(Q, chunk_reps.transpose(-2, -1)) / math.sqrt(hd)

    positions = torch.arange(seq_len, device=device)
    chunk_starts = torch.arange(num_chunks, device=device) * chunk_size
    causal_ok = chunk_starts.unsqueeze(0) < positions.unsqueeze(1)
    routing_logits = routing_logits.masked_fill(~causal_ok[None, None], float("-inf"))

    # Fix 6 (consistency): the temperature is part of the function being
    # trained, so it now applies in eval as well. Previously eval silently
    # switched to temp=1.0, which (for temperature != 1.0) evaluated a
    # different routing distribution than the one trained. `training` is kept
    # in the signature for call-site compatibility but no longer changes math.
    del training
    temp = temperature
    routing_weights = F.softmax(routing_logits / temp, dim=-1)
    routing_weights = torch.nan_to_num(routing_weights, nan=0.0)

    return routing_weights, temp


def _prepare_stage2_selected_chunks(
    top_k_packed: torch.Tensor,
    c_q: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Consume the full packed Stage-1 selection for query chunk c_q.

    Padding remains represented as -1 and is handled by the returned valid mask.
    """
    selected = top_k_packed[:, :, c_q, :]
    valid = selected >= 0
    ci = selected.clamp(0, num_chunks - 1)
    return selected, valid, ci


def _build_stage1_top_k_packed(
    routing_weights: torch.Tensor,
    *,
    seq_len: int,
    chunk_size: int,
    num_chunks: int,
    top_k_chunks: int,
    valid_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build the packed Stage-1 chunk selection for every query chunk, including
    chunk 0. The self chunk is guaranteed in the packed selection.
    """
    B, H, _, _ = routing_weights.shape
    device = routing_weights.device

    top_k_list = []
    for c_q in range(num_chunks):
        q_start = c_q * chunk_size
        q_end = min(q_start + chunk_size, seq_len)
        if q_start >= seq_len:
            top_k_list.append(torch.full((B, H, top_k_chunks), -1, dtype=torch.long, device=device))
            continue

        n_valid_with_self = c_q + 1
        w_c_full = routing_weights[:, :, q_start:q_end, :n_valid_with_self]
        if valid_lengths is None:
            w_mean_full = w_c_full.mean(dim=2)
        else:
            valid_lengths = valid_lengths.to(device=device, dtype=torch.long).reshape(B)
            q_pos = torch.arange(q_start, q_end, device=device).reshape(1, 1, -1, 1)
            q_valid = q_pos < valid_lengths.reshape(B, 1, 1, 1)
            denom = q_valid.sum(dim=2).clamp_min(1)
            w_mean_full = (w_c_full * q_valid).sum(dim=2) / denom

        n_others = c_q
        if n_others > 0 and top_k_chunks > 1:
            w_others = w_mean_full.clone()
            w_others[:, :, c_q] = float("-inf")
            topk_others = min(top_k_chunks - 1, n_others)
            _, idx_others = w_others.topk(topk_others, dim=-1)
            idx = torch.cat([
                idx_others,
                torch.full((B, H, 1), c_q, dtype=torch.long, device=device),
            ], dim=-1)
        else:
            idx = torch.full((B, H, 1), c_q, dtype=torch.long, device=device)

        if idx.shape[-1] < top_k_chunks:
            pad = torch.full((B, H, top_k_chunks - idx.shape[-1]), -1, dtype=torch.long, device=device)
            idx = torch.cat([idx, pad], dim=-1)

        if valid_lengths is not None:
            chunk_has_query = (q_start < valid_lengths.reshape(B, 1, 1))
            idx = torch.where(chunk_has_query, idx, torch.full_like(idx, -1))

        top_k_list.append(idx)

    return torch.stack(top_k_list, dim=2)


def _build_stage2_token_indices(
    Q_pad: torch.Tensor,
    K_pad: torch.Tensor,
    top_k_packed: torch.Tensor,
    *,
    B: int,
    H: int,
    N: int,
    num_chunks: int,
    chunk_size: int,
    hisa_top_m_tokens: int,
    collect_stats: bool = False,
    stage2_q_block: int = 256,
    routing_weights: torch.Tensor | None = None,
    stage2_rep_r: int = 0,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Build query-scoped compact Stage-2 token indices.

    Returns:
      token_idx_packed: int32 [B, H, C_query, K, M], where invalid slots are -1.
      m_actual:         min(max(hisa_top_m_tokens, 1), chunk_size)
      selected_fraction: scalar tensor with fraction of valid compact slots, or NaN when telemetry is disabled.

    Unlike the old global [B,H,N] token mask, this metadata is scoped to the
    query chunk and the selected chunk. It therefore cannot be polluted by token
    choices made by later query chunks, and the Triton kernel can gather only M
    keys/values instead of computing a full key chunk.

    Scoring streams over query-row blocks of size `stage2_q_block` with a
    running max instead of materializing the full [B,H,K,chunk,chunk] score
    tensor; peak memory per query chunk drops from O(chunk^2) to
    O(stage2_q_block * chunk). NOTE: total *compute* is unchanged and still
    O(N^2 * K / C) over the sequence — this stage is the only superlinear
    component of the architecture. max() is order-independent, so selection is
    semantically identical to the dense version (up to GEMM-shape-dependent
    last-ulp accumulation differences in the dot products themselves).
    """
    device = Q_pad.device
    C = num_chunks
    K_sel = top_k_packed.shape[-1]
    hd = Q_pad.shape[-1]
    m_actual = min(max(int(hisa_top_m_tokens), 1), chunk_size)
    q_block = max(1, int(stage2_q_block))

    token_idx = torch.full(
        (B, H, C, K_sel, m_actual),
        -1,
        dtype=torch.int32,
        device=device,
    )

    K_reshaped = K_pad.view(B, H, C, chunk_size, hd)
    Q_reshaped = Q_pad.view(B, H, C, chunk_size, hd)

    b_idx_3d = torch.arange(B, device=device).view(B, 1, 1)
    h_idx_3d = torch.arange(H, device=device).view(1, H, 1)
    q_offsets = torch.arange(chunk_size, device=device)
    k_offsets = torch.arange(chunk_size, device=device)

    for c_q in range(C):
        q_start = c_q * chunk_size
        if q_start >= N:
            break

        selected, valid_chunks, ci = _prepare_stage2_selected_chunks(
            top_k_packed, c_q=c_q, num_chunks=C
        )
        del selected

        # [B,H,K,chunk,hd]
        k_slices = K_reshaped[b_idx_3d, h_idx_3d, ci]
        # [B,H,chunk,hd]
        q_slice = Q_reshaped[:, :, c_q, :]

        # Key-side validity is block-invariant; compute once per query chunk.
        k_abs = ci[..., None] * chunk_size + k_offsets.view(1, 1, 1, chunk_size)
        k_valid = (k_abs < N) & valid_chunks[..., None]
        q_abs_all = q_start + q_offsets

        if stage2_rep_r > 0:
            if routing_weights is None:
                raise ValueError("routing_weights is required when stage2_rep_r > 0")
            q_end = min(q_start + chunk_size, N)
            q_len = max(q_end - q_start, 0)
            rep_r = min(max(int(stage2_rep_r), 1), max(q_len, 1))
            per_rep_m = max(1, math.ceil(m_actual / rep_r))

            # Pick pair-specific query representatives: for each selected key
            # chunk j, choose rows in query chunk c_q with the largest routing
            # weight to j. This is the linear Stage-2 selector path:
            # O(K * rep_r * chunk) instead of O(K * chunk^2) per query chunk.
            rw_q = routing_weights[:, :, q_start:q_end, :]  # [B,H,q_len,C]
            if q_len == 0:
                token_scores = torch.full(
                    (B, H, K_sel, chunk_size), float("-inf"),
                    device=device, dtype=Q_pad.dtype,
                )
            else:
                gather_idx = ci[..., None, None].expand(B, H, K_sel, q_len, 1)
                rw_sel = torch.gather(
                    rw_q.unsqueeze(2).expand(B, H, K_sel, q_len, C),
                    dim=-1,
                    index=gather_idx,
                ).squeeze(-1)
                rw_sel = rw_sel.masked_fill(~valid_chunks[..., None], float("-inf"))
                rep_vals, rep_idx = rw_sel.topk(rep_r, dim=-1)
                rep_valid = torch.isfinite(rep_vals) & valid_chunks[..., None]

                q_exp = q_slice.unsqueeze(2).expand(B, H, K_sel, chunk_size, hd)
                q_reps = torch.gather(
                    q_exp,
                    dim=3,
                    index=rep_idx[..., None].expand(B, H, K_sel, rep_r, hd),
                )
                # [B,H,K,rep_r,chunk]
                rep_scores = torch.matmul(
                    q_reps,
                    k_slices.transpose(-2, -1),
                ) / math.sqrt(hd)
                rep_abs = q_start + rep_idx
                causal_rep = (
                    k_valid.unsqueeze(-2)
                    & rep_valid[..., None]
                    & (k_abs.unsqueeze(-2) < rep_abs[..., None])
                )
                rep_scores = rep_scores.masked_fill(~causal_rep, float("-inf"))

                # Conservative union: take top ceil(M/r) tokens per query rep,
                # deduplicate by scatter-reducing into token_scores, then fill
                # any duplicate-created holes from pooled representative scores.
                top_rep_vals, top_rep_idx = rep_scores.topk(
                    min(per_rep_m, chunk_size), dim=-1
                )
                union_scores = torch.full(
                    (B, H, K_sel, chunk_size), float("-inf"),
                    device=device, dtype=Q_pad.dtype,
                )
                union_scores.scatter_reduce_(
                    dim=-1,
                    index=top_rep_idx.reshape(B, H, K_sel, -1),
                    src=top_rep_vals.reshape(B, H, K_sel, -1),
                    reduce="amax",
                    include_self=True,
                )
                pooled_scores = rep_scores.max(dim=-2).values
                token_scores = torch.where(
                    torch.isfinite(union_scores), union_scores, pooled_scores
                )
        else:
            # Running max of causal-masked scores over query rows, streamed in
            # blocks. One compact token set per selected chunk for this query
            # chunk; max over query rows preserves tokens that are important to
            # any row in the query chunk while keeping the kernel metadata compact.
            token_scores = torch.full(
                (B, H, K_sel, chunk_size), float("-inf"),
                device=device, dtype=Q_pad.dtype,
            )
            for qb0 in range(0, chunk_size, q_block):
                q_abs_blk = q_abs_all[qb0:qb0 + q_block]
                if int(q_abs_blk[0]) >= N:
                    break  # rows are ascending; everything past here is padding
                q_blk = q_slice[:, :, qb0:qb0 + q_block]
                # [B,H,K,qb,chunk]
                s_blk = torch.matmul(
                    q_blk.unsqueeze(2),
                    k_slices.transpose(-2, -1),
                ) / math.sqrt(hd)

                causal_blk = (
                    k_valid.unsqueeze(-2)
                    & (q_abs_blk < N).view(1, 1, 1, -1, 1)
                    & (k_abs.unsqueeze(-2) < q_abs_blk.view(1, 1, 1, -1, 1))
                )
                s_blk = s_blk.masked_fill(~causal_blk, float("-inf"))
                token_scores = torch.maximum(token_scores, s_blk.max(dim=-2).values)

        top_vals, top_m_idx = token_scores.topk(m_actual, dim=-1)
        flat_pos = ci[..., None] * chunk_size + top_m_idx
        finite = torch.isfinite(top_vals)
        valid_flat = finite & valid_chunks[..., None] & (flat_pos < N)
        flat_pos = torch.where(valid_flat, flat_pos, torch.full_like(flat_pos, -1))
        token_idx[:, :, c_q, :, :] = flat_pos.to(torch.int32)

    if collect_stats:
        selected_fraction = (token_idx >= 0).float().mean().detach()
    else:
        selected_fraction = torch.full((), float('nan'), device=device)
    return token_idx, m_actual, selected_fraction


# ---------------------------------------------------------------------------
# Triton forward: compact HISA token refinement within selected chunks
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_fwd_hisa(
    Q, K, V, ROUTING_W, TOP_K_IDX, TOKEN_IDX, OUT, LSE_OUT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_ib, stride_ih, stride_ip,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lseb, stride_lseh, stride_lsen,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_HD: tl.constexpr,
    M_VAL: tl.constexpr, M_PAD: tl.constexpr,
    APPLY_RW_BIAS: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    qt = tl.program_id(2)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    q_offsets = qt * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qs = q_start + q_offsets
    qm = (qs < N) & (q_offsets < CHUNK_SIZE)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    ms = tl.arange(0, M_PAD)
    mm = ms < M_VAL

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    q_c = tl.load(
        q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )

    mi = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    li = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_idx_base = TOKEN_IDX + b * stride_ib + h * stride_ih

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + (c_q * K_VAL + ki) * stride_tk).to(tl.int32)
        chunk_valid = chunk_idx >= 0
        safe_chunk_idx = tl.maximum(chunk_idx, 0)
        idx_off = ((c_q * K_VAL + ki) * M_VAL + ms) * stride_ip
        ks = tl.load(token_idx_base + idx_off, mask=mm, other=-1).to(tl.int32)
        km = (ks >= 0) & (ks < N) & mm & chunk_valid

        k_block = tl.load(
            k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )

        q_f = q_c.to(tl.float32)
        k_f = k_block.to(tl.float32)
        s = tl.dot(q_f, tl.trans(k_f), input_precision="tf32") * sc

        selected = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :] & chunk_valid
        s = tl.where(selected, s, float("-inf"))

        if APPLY_RW_BIAS:
            rw = tl.load(
                ROUTING_W + b * stride_rb + h * stride_rh + qs * stride_rn + safe_chunk_idx * stride_rc,
                mask=qm & chunk_valid,
                other=1e-8,
            ).to(tl.float32)
            log_rw = tl.log(tl.maximum(rw, 1e-8))
            s = tl.where(selected, s + log_rw[:, None], float("-inf"))

        m_new = tl.max(s, axis=1)
        has_prev = mi > float("-inf")
        has_curr = m_new > float("-inf")
        has_any = has_prev | has_curr
        mn_raw = tl.maximum(mi, m_new)
        mn = tl.where(has_any, mn_raw, tl.zeros_like(mn_raw))
        cor = tl.where(
            has_prev,
            tl.math.exp2((mi - mn) * 1.4426950408889634),
            tl.zeros_like(mi),
        )
        p_raw = tl.math.exp2((s - mn[:, None]) * 1.4426950408889634)
        p = tl.where(selected, p_raw, 0.0)

        li = tl.where(has_any, li * cor + tl.sum(p, axis=1), li)
        mi = tl.where(has_any, mn_raw, mi)

        v_block = tl.load(
            v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )
        acc = acc * cor[:, None] + tl.dot(
            p.to(tl.float32), v_block.to(tl.float32), input_precision="tf32"
        )

    ls = tl.where(li > 0.0, li, 1.0)
    acc = acc / ls[:, None]
    safe_mi = tl.where(mi > float("-inf"), mi, tl.zeros_like(mi))
    lse = tl.where(mi > float("-inf"), safe_mi + tl.log(ls), float("-inf"))

    o_base = OUT + b * stride_ob + h * stride_oh
    tl.store(
        o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=qm[:, None] & dm[None, :],
    )
    lse_base = LSE_OUT + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + qs * stride_lsen, lse, mask=qm)


# ---------------------------------------------------------------------------
# Triton backward: dQ direct, dK/dV atomic, dRouting direct
# ---------------------------------------------------------------------------

@triton.jit
def _dsr_bwd_hisa(
    Q, K, V, O, DO, LSE_OUT, ROUTING_W, TOP_K_IDX, TOKEN_IDX,
    DQ, DK, DV, DRW,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lseb, stride_lseh, stride_lsen,
    stride_rb, stride_rh, stride_rn, stride_rc,
    stride_tb, stride_th, stride_tk,
    stride_ib, stride_ih, stride_ip,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_drb, stride_drh, stride_drn, stride_drc,
    N, H: tl.constexpr, HD: tl.constexpr,
    C: tl.constexpr, K_VAL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_HD: tl.constexpr,
    M_VAL: tl.constexpr, M_PAD: tl.constexpr,
    APPLY_RW_BIAS: tl.constexpr,
):
    bh = tl.program_id(0)
    c_q = tl.program_id(1)
    qt = tl.program_id(2)
    b = bh // H
    h = bh % H

    q_start = c_q * CHUNK_SIZE
    sc = 1.0 / tl.sqrt(HD * 1.0)

    q_offsets = qt * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qs = q_start + q_offsets
    qm = (qs < N) & (q_offsets < CHUNK_SIZE)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    ms = tl.arange(0, M_PAD)
    mm = ms < M_VAL

    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh

    q_c = tl.load(
        q_base + qs[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    do_c = tl.load(
        do_base + qs[:, None] * stride_don + ds[None, :] * stride_dod,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    o_c = tl.load(
        o_base + qs[:, None] * stride_on + ds[None, :] * stride_od,
        mask=qm[:, None] & dm[None, :],
        other=0.0,
    )
    lse_c = tl.load(
        LSE_OUT + b * stride_lseb + h * stride_lseh + qs * stride_lsen,
        mask=qm,
        other=0.0,
    )

    D_c = tl.sum(do_c.to(tl.float32) * o_c.to(tl.float32), axis=1)
    dq_c = tl.zeros([BLOCK_Q, BLOCK_HD], tl.float32)

    top_k_base = TOP_K_IDX + b * stride_tb + h * stride_th
    token_idx_base = TOKEN_IDX + b * stride_ib + h * stride_ih

    for ki in range(K_VAL):
        chunk_idx = tl.load(top_k_base + (c_q * K_VAL + ki) * stride_tk).to(tl.int32)
        chunk_valid = chunk_idx >= 0
        safe_chunk_idx = tl.maximum(chunk_idx, 0)
        idx_off = ((c_q * K_VAL + ki) * M_VAL + ms) * stride_ip
        ks = tl.load(token_idx_base + idx_off, mask=mm, other=-1).to(tl.int32)
        km = (ks >= 0) & (ks < N) & mm & chunk_valid

        k_block = tl.load(
            k_base + ks[:, None] * stride_kn + ds[None, :] * stride_kd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )
        v_block = tl.load(
            v_base + ks[:, None] * stride_vn + ds[None, :] * stride_vd,
            mask=km[:, None] & dm[None, :],
            other=0.0,
        )

        q_f = q_c.to(tl.float32)
        k_f = k_block.to(tl.float32)
        s = tl.dot(q_f, tl.trans(k_f), input_precision="tf32") * sc

        selected = (ks[None, :] < qs[:, None]) & qm[:, None] & km[None, :] & chunk_valid

        if APPLY_RW_BIAS:
            rw = tl.load(
                ROUTING_W + b * stride_rb + h * stride_rh + qs * stride_rn + safe_chunk_idx * stride_rc,
                mask=qm & chunk_valid,
                other=1e-8,
            ).to(tl.float32)
            rw_safe = tl.maximum(rw, 1e-8)
            log_rw = tl.log(rw_safe)
            s = tl.where(selected, s + log_rw[:, None], float("-inf"))
        else:
            rw_safe = tl.full([BLOCK_Q], 1.0, tl.float32)
            s = tl.where(selected, s, float("-inf"))

        has_lse = lse_c > float("-inf")
        safe_lse = tl.where(has_lse, lse_c, tl.zeros_like(lse_c))
        alpha = tl.where(selected & has_lse[:, None], tl.exp(s - safe_lse[:, None]), 0.0)

        do_f = do_c.to(tl.float32)
        v_f = v_block.to(tl.float32)
        dot_rv = tl.dot(do_f, tl.trans(v_f), input_precision="tf32")
        ds_matrix = alpha * (dot_rv - D_c[:, None])

        dq_c += tl.dot(ds_matrix, k_f, input_precision="tf32") * sc

        dk_block = tl.dot(tl.trans(ds_matrix), q_f, input_precision="tf32") * sc
        tl.atomic_add(
            DK + b * stride_dkb + h * stride_dkh + ks[:, None] * stride_dkn + ds[None, :] * stride_dkd,
            tl.where(km[:, None] & dm[None, :], dk_block, 0.0),
            mask=km[:, None] & dm[None, :],
        )

        dv_block = tl.dot(tl.trans(alpha), do_f, input_precision="tf32")
        tl.atomic_add(
            DV + b * stride_dvb + h * stride_dvh + ks[:, None] * stride_dvn + ds[None, :] * stride_dvd,
            tl.where(km[:, None] & dm[None, :], dv_block, 0.0),
            mask=km[:, None] & dm[None, :],
        )

        if APPLY_RW_BIAS:
            drw_accum = tl.sum(tl.where(selected, ds_matrix, 0.0), axis=1)
            drw_chunk = drw_accum / rw_safe
            tl.store(
                DRW + b * stride_drb + h * stride_drh + qs * stride_drn + safe_chunk_idx * stride_drc,
                tl.where(qm & chunk_valid, drw_chunk, 0.0),
                mask=qm & chunk_valid,
            )

    dq_base = DQ + b * stride_dqb + h * stride_dqh
    tl.store(
        dq_base + qs[:, None] * stride_dqn + ds[None, :] * stride_dqd,
        dq_c.to(tl.bfloat16),
        mask=qm[:, None] & dm[None, :],
    )


# ---------------------------------------------------------------------------
# Autograd function: Triton forward + Triton backward
# ---------------------------------------------------------------------------

class _DSRHISAAttendFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        routing_weights,
        top_k_packed,
        token_idx_packed,
        chunk_size,
        apply_rw_bias,
        N_orig=None,
        pad_len=None,
        hisa_top_m_tokens=None,
        effective_temp=None,
    ):
        B, H, N, hd = Q.shape
        C = routing_weights.shape[-1]
        k_val = top_k_packed.shape[-1]
        m_val = token_idx_packed.shape[-1]
        device = Q.device

        if N_orig is None:
            N_orig = N
        if pad_len is None:
            pad_len = 0
        if hisa_top_m_tokens is None:
            hisa_top_m_tokens = m_val
        if effective_temp is None:
            effective_temp = 1.0

        BLOCK_HD = _next_pow2(hd)
        # Fix 1: tile the query dimension instead of mapping the whole chunk to
        # one program. chunk_size = ceil(N / C) grows linearly with N when C is
        # fixed, and a [chunk_size, BLOCK_HD] fp32 q tile blows past the
        # register file at long context (256KB/program at N=32768, C=32),
        # causing local-memory spills or launch failures. BLOCK_Q caps the
        # per-program tile; a third grid axis covers the rest of the chunk.
        BLOCK_Q = max(16, min(_next_pow2(chunk_size), 128))
        n_q_tiles = triton.cdiv(chunk_size, BLOCK_Q)
        M_PAD = max(16, _next_pow2(m_val))

        out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
        lse_out = torch.full((B, H, N), float("-inf"), dtype=torch.float32, device=device)

        top_k_flat = top_k_packed.contiguous().reshape(B, H, -1).to(torch.int32)
        token_idx_flat = token_idx_packed.contiguous().reshape(B, H, -1).to(torch.int32)

        if C > 1:
            _nw = 4
            _ns = 2
            grid = (B * H, C, n_q_tiles)

            _dsr_fwd_hisa[grid](
                Q, K, V, routing_weights, top_k_flat, token_idx_flat, out, lse_out,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                routing_weights.stride(0), routing_weights.stride(1),
                routing_weights.stride(2), routing_weights.stride(3),
                top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_Q=BLOCK_Q, BLOCK_HD=BLOCK_HD,
                M_VAL=m_val, M_PAD=M_PAD,
                APPLY_RW_BIAS=1 if apply_rw_bias else 0,
                num_warps=_nw, num_stages=_ns,
            )

        replay_mode = os.environ.get("HISA_RECOMPUTE", "none").lower()

        if replay_mode in ("out_lse", "all"):
            ctx.save_for_backward(Q, K, V, routing_weights, top_k_flat, token_idx_flat)
        else:
            ctx.save_for_backward(Q, K, V, routing_weights, out, lse_out, top_k_flat, token_idx_flat)

        ctx.chunk_size = chunk_size
        ctx.apply_rw_bias = apply_rw_bias
        ctx.C = C
        ctx.k_val = k_val
        ctx.m_val = m_val
        ctx.M_PAD = M_PAD
        ctx.BLOCK_Q = BLOCK_Q
        ctx.n_q_tiles = n_q_tiles
        ctx.BLOCK_HD = BLOCK_HD
        ctx.replay_mode = replay_mode
        ctx.N_orig = N_orig
        ctx.pad_len = pad_len
        ctx.hisa_top_m_tokens = hisa_top_m_tokens
        ctx.effective_temp = effective_temp
        return out

    @staticmethod
    def backward(ctx, grad_output):
        replay_mode = ctx.replay_mode
        saved = ctx.saved_tensors

        if replay_mode in ("out_lse", "all"):
            Q, K, V, routing_weights, top_k_flat, token_idx_flat = saved
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
            m_val = ctx.m_val
            M_PAD = ctx.M_PAD
            BLOCK_Q = ctx.BLOCK_Q
            n_q_tiles = ctx.n_q_tiles
            BLOCK_HD = ctx.BLOCK_HD
            device = Q.device

            out = torch.zeros(B, H, N, hd, dtype=Q.dtype, device=device)
            lse_out = torch.full((B, H, N), float("-inf"), dtype=torch.float32, device=device)

            if C > 1:
                _nw = 4
                _ns = 2
                grid = (B * H, C, n_q_tiles)
                _dsr_fwd_hisa[grid](
                    Q, K, V, routing_weights, top_k_flat, token_idx_flat, out, lse_out,
                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    routing_weights.stride(0), routing_weights.stride(1),
                    routing_weights.stride(2), routing_weights.stride(3),
                    top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                    token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                    N=N, H=H, HD=hd,
                    C=C, K_VAL=k_val,
                    CHUNK_SIZE=chunk_size, BLOCK_Q=BLOCK_Q, BLOCK_HD=BLOCK_HD,
                    M_VAL=m_val, M_PAD=M_PAD,
                    APPLY_RW_BIAS=1 if ctx.apply_rw_bias else 0,
                    num_warps=_nw, num_stages=_ns,
                )
        else:
            Q, K, V, routing_weights, out, lse_out, top_k_flat, token_idx_flat = saved
            B, H, N, hd = Q.shape
            chunk_size = ctx.chunk_size
            C = ctx.C
            k_val = ctx.k_val
            m_val = ctx.m_val
            M_PAD = ctx.M_PAD
            BLOCK_Q = ctx.BLOCK_Q
            n_q_tiles = ctx.n_q_tiles
            BLOCK_HD = ctx.BLOCK_HD
            device = Q.device

        grad_output = grad_output.contiguous()

        dQ = torch.zeros_like(Q)
        dK = torch.zeros(B, H, N, hd, device=device, dtype=torch.float32)
        dV = torch.zeros(B, H, N, hd, device=device, dtype=torch.float32)
        dRW = torch.zeros_like(routing_weights)

        if C > 1:
            _nw = 4
            _ns = 2
            grid = (B * H, C, n_q_tiles)

            _dsr_bwd_hisa[grid](
                Q, K, V, out, grad_output, lse_out, routing_weights, top_k_flat,
                token_idx_flat, dQ, dK, dV, dRW,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                grad_output.stride(0), grad_output.stride(1),
                grad_output.stride(2), grad_output.stride(3),
                lse_out.stride(0), lse_out.stride(1), lse_out.stride(2),
                routing_weights.stride(0), routing_weights.stride(1),
                routing_weights.stride(2), routing_weights.stride(3),
                top_k_flat.stride(0), top_k_flat.stride(1), top_k_flat.stride(2),
                token_idx_flat.stride(0), token_idx_flat.stride(1), token_idx_flat.stride(2),
                dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
                dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
                dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
                dRW.stride(0), dRW.stride(1), dRW.stride(2), dRW.stride(3),
                N=N, H=H, HD=hd,
                C=C, K_VAL=k_val,
                CHUNK_SIZE=chunk_size, BLOCK_Q=BLOCK_Q, BLOCK_HD=BLOCK_HD,
                M_VAL=m_val, M_PAD=M_PAD,
                APPLY_RW_BIAS=1 if ctx.apply_rw_bias else 0,
                num_warps=_nw, num_stages=_ns,
            )

        return (
            dQ,
            dK.to(K.dtype),
            dV.to(V.dtype),
            dRW,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Module — HISA token-level refinement
# ---------------------------------------------------------------------------

class HierarchicalSparseAttentionV15HISA(nn.Module):

    def __init__(
        self,
        D: int,
        H: int,
        hd: int,
        num_chunks: int = 32,
        top_k_chunks: int = 4,
        hisa_top_m_tokens: int = 32,
        routing_bias_in_eval: bool | None = None,
        stage2_q_block: int = 256,
    ):
        super().__init__()
        self.H = H
        self.num_heads = H
        self.hd = hd
        self.num_chunks = num_chunks
        self.top_k_chunks = top_k_chunks
        self.hisa_top_m_tokens = hisa_top_m_tokens
        # Fix 6: training adds log(routing_weight) to attention scores (the
        # NSA-style routing-gradient path); the old code dropped that term at
        # eval, so the evaluated function was not the trained function. By
        # default we keep the bias at eval for train/eval consistency, but make
        # the eval policy centrally configurable for inference/eval probes.
        # Constructor args take precedence over env so scripts can pin a policy.
        if routing_bias_in_eval is None:
            routing_bias_in_eval = _env_flag(
                "DWARF_HISA_ROUTING_BIAS_IN_EVAL",
                True,
                "HISA_ROUTING_BIAS_IN_EVAL",
            )
        self.routing_bias_in_eval = bool(routing_bias_in_eval)
        # Query-row block size for streamed Stage-2 token scoring (memory knob;
        # does not change selection semantics).
        self.stage2_q_block = int(stage2_q_block)
        # Experimental linear Stage-2 selector: when >0, choose this many
        # routing-weight query representatives per selected chunk instead of
        # row-maxing over every query row. Default 0 preserves V15.1 behavior.
        self.stage2_rep_r = int(os.getenv(
            "HISA_STAGE2_REP_R",
            os.getenv("DWARF_HISA_STAGE2_REP_R", "0"),
        ))
        self.temperature = 1.0
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)
        self._routing_entropy: torch.Tensor | float = float("nan")
        self._stage2_selected_fraction: torch.Tensor | float = float("nan")
        self.collect_telemetry = os.getenv("HISA_TELEMETRY", "0") == "1"

    def forward(self, x: torch.Tensor, kv_inject=None) -> torch.Tensor:
        # kv_inject is accepted for API compatibility with DSQG blocks, but HISA
        # intentionally does not consume NPCI K/V deltas in this architecture.
        del kv_inject

        B, N, _ = x.shape
        H, hd = self.H, self.hd
        C = self.num_chunks
        k = self.top_k_chunks
        m = self.hisa_top_m_tokens
        chunk_size = math.ceil(N / C)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, N, H, hd).transpose(1, 2)

        Q = to_heads(self.W_q(x))
        K = to_heads(self.W_k(x))
        V = to_heads(self.W_v(x))

        pad_len = chunk_size * C - N
        V_pad = F.pad(V, (0, 0, 0, pad_len)) if pad_len > 0 else V
        K_pad = F.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
        Q_pad = F.pad(Q, (0, 0, 0, pad_len)) if pad_len > 0 else Q
        causal_control_valid_lengths = getattr(self, '_causal_control_valid_lengths', None)
        if causal_control_valid_lengths is not None:
            causal_control_valid_lengths = causal_control_valid_lengths.to(device=x.device, dtype=torch.long).reshape(B)
        chunk_reps = _compute_chunk_representatives(
            K_pad,
            num_chunks=C,
            valid_lengths=causal_control_valid_lengths,
        )

        routing_weights, effective_temp = _compute_routing(
            Q,
            chunk_reps,
            seq_len=N,
            num_chunks=C,
            chunk_size=chunk_size,
            hd=hd,
            temperature=self.temperature,
            training=self.training,
        )
        routing_weights_pad = F.pad(routing_weights, (0, 0, 0, pad_len)) if pad_len > 0 else routing_weights

        with torch.no_grad():
            w = routing_weights.clamp(min=1e-8)
            self._routing_entropy = (-(w * w.log()).sum(dim=-1).mean()).detach()

            top_k_packed = _build_stage1_top_k_packed(
                routing_weights,
                seq_len=N,
                chunk_size=chunk_size,
                num_chunks=C,
                top_k_chunks=k,
                valid_lengths=causal_control_valid_lengths,
            )

            token_idx_packed, m_actual, selected_fraction = _build_stage2_token_indices(
                Q_pad,
                K_pad,
                top_k_packed,
                B=B,
                H=H,
                N=N,
                num_chunks=C,
                chunk_size=chunk_size,
                hisa_top_m_tokens=m,
                collect_stats=self.collect_telemetry,
                stage2_q_block=self.stage2_q_block,
                routing_weights=routing_weights,
                stage2_rep_r=self.stage2_rep_r,
            )
            self._stage2_selected_fraction = selected_fraction

        apply_rw_bias = self.training or self.routing_bias_in_eval
        out = _DSRHISAAttendFn.apply(
            Q_pad,
            K_pad,
            V_pad,
            routing_weights_pad,
            top_k_packed,
            token_idx_packed,
            chunk_size,
            apply_rw_bias,
            N,
            pad_len,
            m_actual,
            effective_temp,
        )

        out_flat = out[:, :, :N, :].transpose(1, 2).reshape(B, N, H * hd)
        return self.W_o(out_flat)
