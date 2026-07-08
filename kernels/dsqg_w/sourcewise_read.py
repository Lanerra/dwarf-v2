from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from .candidate_types import CandidateSource
from .triton_schedule import _dsqg_w_triton_schedule

try:
    import triton
    import triton.language as tl

    _TRITON_SOURCEWISE_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_SOURCEWISE_AVAILABLE = False

if _TRITON_SOURCEWISE_AVAILABLE:

    @triton.jit
    def _dsqg_w_sourcewise_score_read_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        read_ptr,
        read_mix_weight_ptr,
        probs_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        OUT_BLOCK: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        STORE_PROBS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        out_pid = tl.program_id(1)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        out_offs = out_pid * OUT_BLOCK + tl.arange(0, OUT_BLOCK)
        out_mask = out_offs < D
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        read_out = tl.zeros((int(OUT_BLOCK),), tl.float32)

        row_j_base = (b * N + n) * J
        max_score = -float("inf")
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            max_score = tl.maximum(max_score, score)

        denom = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            denom += tl.where(valid, tl.exp(score - max_score), 0.0)

        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))

            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            p = tl.where(valid, tl.exp(score - max_score) / denom, 0.0)
            if STORE_PROBS and out_pid == 0:
                tl.store(probs_ptr + ((b * N + n) * J + j) * H + h, p)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            contrib = p * v
            in_cols = h * HD + offs
            all_w = tl.load(
                read_mix_weight_ptr + out_offs[:, None] * ((N_TYPES + 1) * D) + in_cols[None, :],
                mask=out_mask[:, None] & hd_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            read_out += tl.sum(all_w * contrib[None, :], axis=1)
            typed_slot = ctype + 1
            typed_cols = typed_slot * D + h * HD + offs
            type_w = tl.load(
                read_mix_weight_ptr + out_offs[:, None] * ((N_TYPES + 1) * D) + typed_cols[None, :],
                mask=out_mask[:, None] & hd_mask[None, :] & valid & (ctype >= 0) & (ctype < N_TYPES),
                other=0.0,
            ).to(tl.float32)
            read_out += tl.sum(type_w * contrib[None, :], axis=1)
        tl.atomic_add(read_ptr + (b * N + n) * D + out_offs, read_out, sem="relaxed", mask=out_mask)


    @triton.jit
    def _dsqg_w_sourcewise_read_slots_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        read_slots_ptr,
        lse_ptr,
        probs_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        MAX_READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        STORE_LSE: tl.constexpr,
        STORE_PROBS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)

        row_j_base = (b * N + n) * J
        max_score = -float("inf")
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            max_score = tl.maximum(max_score, score)

        denom = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            denom += tl.where(valid, tl.exp(score - max_score), 0.0)

        if STORE_LSE:
            tl.store(lse_ptr + (b * N + n) * H + h, max_score + tl.log(denom))

        slot_ids = tl.arange(0, MAX_READ_SLOTS)
        acc = tl.zeros((int(MAX_READ_SLOTS), int(BLOCK_HD)), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            p = tl.where(valid, tl.exp(score - max_score) / denom, 0.0)
            if STORE_PROBS:
                tl.store(probs_ptr + ((b * N + n) * J + j) * H + h, p)
            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            contrib = p * v
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            add_slot = (slot_ids[:, None] == 0) | (slot_ids[:, None] == type_slot)
            active_slot = slot_ids[:, None] < READ_SLOTS
            acc += tl.where(add_slot & active_slot, contrib[None, :], 0.0)

        store_base = ((b * N + n) * READ_SLOTS + slot_ids[:, None]) * D + h * HD + offs[None, :]
        tl.store(read_slots_ptr + store_base, acc, mask=(slot_ids[:, None] < READ_SLOTS) & hd_mask[None, :])



    @triton.jit
    def _dsqg_w_sourcewise_read_slots_backward_kernel(
        q_ptr,
        k_final_ptr,
        v_final_ptr,
        k_l3_ptr,
        v_l3_ptr,
        k_summary_ptr,
        v_summary_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_token_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        lse_ptr,
        grad_slots_ptr,
        grad_q_ptr,
        grad_k_final_ptr,
        grad_v_final_ptr,
        grad_k_l3_ptr,
        grad_v_l3_ptr,
        grad_k_summary_ptr,
        grad_v_summary_ptr,
        grad_role_key_ptr,
        grad_source_key_ptr,
        grad_type_bias_ptr,
        grad_source_bias_ptr,
        grad_qtb_ptr,
        grad_score_bias_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
        COMPUTE_QUERY: tl.constexpr,
        COMPUTE_SOURCE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + (b * N + n) * H + h).to(tl.float32)
        row_j_base = (b * N + n) * J

        sum_p_dp = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            sum_p_dp += p * dp

        grad_q = tl.zeros((int(BLOCK_HD),), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            tok = tl.load(cand_token_ptr + meta)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            tok = tl.maximum(0, tl.minimum(tok, N - 1))
            src_base = ((b * N + tok) * H + h) * HD + offs
            k = tl.zeros((int(BLOCK_HD),), tl.float32)
            k += tl.load(k_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            k += tl.load(k_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            k += tl.load(k_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            v = tl.zeros((int(BLOCK_HD),), tl.float32)
            v += tl.load(v_final_ptr + src_base, mask=hd_mask & valid & ((source_id == 1) | (source_id == 5)), other=0.0).to(tl.float32)
            v += tl.load(v_l3_ptr + src_base, mask=hd_mask & valid & ((source_id == 2) | (source_id == 3)), other=0.0).to(tl.float32)
            v += tl.load(v_summary_ptr + src_base, mask=hd_mask & valid & (source_id == 4), other=0.0).to(tl.float32)
            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            ds = tl.where(valid, p * (dp - sum_p_dp), 0.0)
            d_k_eff = ds * q * inv_sqrt
            d_v = p * dcontrib
            if COMPUTE_QUERY:
                grad_q += ds * (k + role + src_role) * inv_sqrt

            final_src = (source_id == 1) | (source_id == 5)
            l3_src = (source_id == 2) | (source_id == 3)
            summary_src = source_id == 4
            if COMPUTE_SOURCE:
                tl.atomic_add(grad_k_final_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & final_src)
                tl.atomic_add(grad_v_final_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & final_src)
                tl.atomic_add(grad_k_l3_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & l3_src)
                tl.atomic_add(grad_v_l3_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & l3_src)
                tl.atomic_add(grad_k_summary_ptr + src_base, d_k_eff, sem="relaxed", mask=hd_mask & valid & summary_src)
                tl.atomic_add(grad_v_summary_ptr + src_base, d_v, sem="relaxed", mask=hd_mask & valid & summary_src)
            if COMPUTE_QUERY:
                tl.atomic_add(grad_role_key_ptr + ctype * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
                tl.atomic_add(grad_source_key_ptr + source_id * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
                tl.atomic_add(grad_type_bias_ptr + ctype * H + h, ds, sem="relaxed", mask=valid)
                tl.atomic_add(grad_source_bias_ptr + source_id * H + h, ds, sem="relaxed", mask=valid)
                if USE_QTB:
                    tl.atomic_add(grad_qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, ds, sem="relaxed", mask=valid)
                if USE_SCORE_BIAS:
                    tl.atomic_add(grad_score_bias_ptr + meta, ds, sem="relaxed", mask=valid)

        if COMPUTE_QUERY:
            tl.store(grad_q_ptr + q_base, grad_q, mask=hd_mask)

    @triton.jit
    def _dsqg_w_materialized_read_slots_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        read_slots_ptr,
        lse_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        MAX_READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        row_j_base = (b * N + n) * J

        max_score = -float("inf")
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            kv_base = (((b * N + n) * J + j) * H + h) * HD + offs
            k = tl.load(k_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            max_score = tl.maximum(max_score, score)

        denom = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            kv_base = (((b * N + n) * J + j) * H + h) * HD + offs
            k = tl.load(k_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            denom += tl.where(valid, tl.exp(score - max_score), 0.0)

        tl.store(lse_ptr + (b * N + n) * H + h, max_score + tl.log(denom))

        slot_ids = tl.arange(0, MAX_READ_SLOTS)
        acc = tl.zeros((int(MAX_READ_SLOTS), int(BLOCK_HD)), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            kv_base = (((b * N + n) * J + j) * H + h) * HD + offs
            k = tl.load(k_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
            p = tl.where(valid, tl.exp(score - max_score) / denom, 0.0)
            v = tl.load(v_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            contrib = p * v
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            add_slot = (slot_ids[:, None] == 0) | (slot_ids[:, None] == type_slot)
            active_slot = slot_ids[:, None] < READ_SLOTS
            acc += tl.where(add_slot & active_slot, contrib[None, :], 0.0)

        store_base = ((b * N + n) * READ_SLOTS + slot_ids[:, None]) * D + h * HD + offs[None, :]
        tl.store(read_slots_ptr + store_base, acc, mask=(slot_ids[:, None] < READ_SLOTS) & hd_mask[None, :])


    @triton.jit
    def _dsqg_w_materialized_read_slots_backward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        role_key_ptr,
        source_key_ptr,
        type_bias_ptr,
        source_bias_ptr,
        qtb_ptr,
        score_bias_ptr,
        cand_type_ptr,
        cand_source_ptr,
        cand_mask_ptr,
        type_slot_map_ptr,
        lse_ptr,
        grad_slots_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        grad_role_key_ptr,
        grad_source_key_ptr,
        grad_type_bias_ptr,
        grad_source_bias_ptr,
        grad_qtb_ptr,
        grad_score_bias_ptr,
        B: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        J: tl.constexpr,
        N_TYPES: tl.constexpr,
        READ_SLOTS: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        USE_QTB: tl.constexpr,
        USE_SCORE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        row = pid // H
        n = row % N
        b = row // N
        offs = tl.arange(0, BLOCK_HD)
        hd_mask = offs < HD
        inv_sqrt = 1.0 / tl.sqrt(HD + 0.0)
        q_base = ((b * N + n) * H + h) * HD + offs
        q = tl.load(q_ptr + q_base, mask=hd_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + (b * N + n) * H + h).to(tl.float32)
        row_j_base = (b * N + n) * J

        sum_p_dp = 0.0
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            kv_base = (((b * N + n) * J + j) * H + h) * HD + offs
            k = tl.load(k_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            v = tl.load(v_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            sum_p_dp += p * dp

        grad_q = tl.zeros((int(BLOCK_HD),), tl.float32)
        for j in tl.static_range(0, J):
            meta = row_j_base + j
            valid = tl.load(cand_mask_ptr + meta).to(tl.int1)
            ctype = tl.load(cand_type_ptr + meta)
            source_id = tl.load(cand_source_ptr + meta)
            kv_base = (((b * N + n) * J + j) * H + h) * HD + offs
            k = tl.load(k_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + kv_base, mask=hd_mask & valid, other=0.0).to(tl.float32)
            role = tl.load(role_key_ptr + ctype * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            src_role = tl.load(source_key_ptr + source_id * D + h * HD + offs, mask=hd_mask & valid, other=0.0).to(tl.float32)
            score = tl.sum(q * (k + role + src_role), axis=0) * inv_sqrt
            score += tl.load(type_bias_ptr + ctype * H + h, mask=valid, other=0.0).to(tl.float32)
            score += tl.load(source_bias_ptr + source_id * H + h, mask=valid, other=0.0).to(tl.float32)
            if USE_SCORE_BIAS:
                score += tl.load(score_bias_ptr + meta, mask=valid, other=0.0).to(tl.float32)
            if USE_QTB:
                score += tl.load(qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, mask=valid, other=0.0).to(tl.float32)
            p = tl.where(valid, tl.exp(score - lse), 0.0)

            grad_slot0 = tl.load(grad_slots_ptr + ((b * N + n) * READ_SLOTS + 0) * D + h * HD + offs, mask=hd_mask, other=0.0).to(tl.float32)
            type_slot = tl.load(type_slot_map_ptr + ctype, mask=valid & (ctype >= 0) & (ctype < N_TYPES), other=-1)
            grad_type = tl.load(
                grad_slots_ptr + ((b * N + n) * READ_SLOTS + type_slot) * D + h * HD + offs,
                mask=hd_mask & valid & (type_slot > 0) & (type_slot < READ_SLOTS),
                other=0.0,
            ).to(tl.float32)
            dcontrib = grad_slot0 + grad_type
            dp = tl.sum(dcontrib * v, axis=0)
            ds = tl.where(valid, p * (dp - sum_p_dp), 0.0)
            d_k_eff = ds * q * inv_sqrt
            d_v = p * dcontrib
            grad_q += ds * (k + role + src_role) * inv_sqrt
            tl.store(grad_k_ptr + kv_base, d_k_eff, mask=hd_mask & valid)
            tl.store(grad_v_ptr + kv_base, d_v, mask=hd_mask & valid)
            tl.atomic_add(grad_role_key_ptr + ctype * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
            tl.atomic_add(grad_source_key_ptr + source_id * D + h * HD + offs, d_k_eff, sem="relaxed", mask=hd_mask & valid)
            tl.atomic_add(grad_type_bias_ptr + ctype * H + h, ds, sem="relaxed", mask=valid)
            tl.atomic_add(grad_source_bias_ptr + source_id * H + h, ds, sem="relaxed", mask=valid)
            if USE_QTB:
                tl.atomic_add(grad_qtb_ptr + ((b * N + n) * N_TYPES + ctype) * H + h, ds, sem="relaxed", mask=valid)
            if USE_SCORE_BIAS:
                tl.atomic_add(grad_score_bias_ptr + meta, ds, sem="relaxed", mask=valid)

        tl.store(grad_q_ptr + q_base, grad_q, mask=hd_mask)
else:
    _dsqg_w_sourcewise_score_read_kernel = None
    _dsqg_w_sourcewise_read_slots_kernel = None
    _dsqg_w_sourcewise_read_slots_backward_kernel = None
    _dsqg_w_materialized_read_slots_kernel = None
    _dsqg_w_materialized_read_slots_backward_kernel = None

def _dsqg_w_sourcewise_functional_recompute(
    x: torch.Tensor,
    l3_states: torch.Tensor | None,
    chunk_rep_states: torch.Tensor | None,
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
    cand_scores: torch.Tensor | None,
    *,
    d: int,
    n_heads: int,
    dh: int,
    n_types: int,
    read_type_ids: tuple[int, ...],
    use_query_type_bias: bool,
    norm_x_weight: torch.Tensor,
    norm_x_bias: torch.Tensor,
    norm_c_weight: torch.Tensor,
    norm_c_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    role_key_weight: torch.Tensor,
    source_key_weight: torch.Tensor,
    type_bias: torch.Tensor,
    query_type_bias_weight: torch.Tensor,
    source_bias: torch.Tensor,
    read_mix_weight: torch.Tensor,
    norm_z_weight: torch.Tensor,
    norm_z_bias: torch.Tensor,
    fuse0_weight: torch.Tensor,
    fuse0_bias: torch.Tensor,
    fuse2_weight: torch.Tensor,
    fuse2_bias: torch.Tensor,
    gate_param: torch.Tensor,
) -> torch.Tensor:
    """PyTorch sourcewise recompute used only by Triton custom backward."""
    bsz, seq_len, _ = x.shape
    j_count = cand_mask.shape[-1]
    x_n = F.layer_norm(x, (d,), norm_x_weight, norm_x_bias)
    q = F.linear(x_n, q_proj_weight).reshape(bsz, seq_len, n_heads, dh)

    final_states = x
    l3_base = l3_states if l3_states is not None else final_states
    summary_base = chunk_rep_states if chunk_rep_states is not None else final_states
    zero_base = torch.zeros_like(final_states)
    source_bases: dict[int, torch.Tensor] = {
        int(CandidateSource.FINAL): final_states,
        int(CandidateSource.QUESTION_CACHE): final_states,
        int(CandidateSource.L3): l3_base,
        int(CandidateSource.HISA): l3_base,
        int(CandidateSource.SUMMARY): summary_base,
        int(CandidateSource.NULL): zero_base,
    }
    projected_by_object: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for source_id, states in source_bases.items():
        if not bool(((cand_sources == int(source_id)) & cand_mask).any()):
            continue
        cache_key = id(states)
        projected = projected_by_object.get(cache_key)
        if projected is None:
            states_n = F.layer_norm(states, (d,), norm_c_weight, norm_c_bias)
            k_src = F.linear(states_n, k_proj_weight).reshape(bsz, seq_len, n_heads, dh)
            v_src = F.linear(states_n, v_proj_weight).reshape(bsz, seq_len, n_heads, dh)
            projected = (k_src, v_src)
            projected_by_object[cache_key] = projected
        projected_sources[source_id] = projected

    gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
    score_bias = None
    if cand_scores is not None:
        score_bias = cand_scores.to(device=x.device, dtype=x.dtype)
        score_bias = torch.nan_to_num(score_bias, nan=0.0, neginf=0.0, posinf=0.0)
        valid_denom = cand_mask.to(score_bias.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        score_bias = score_bias - (score_bias.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True) / valid_denom)
        score_bias = score_bias.masked_fill(~cand_mask, 0.0)
    qtb = None
    if use_query_type_bias:
        qtb = F.linear(x_n, query_type_bias_weight).reshape(bsz, seq_len, n_types, n_heads)

    score_parts: list[torch.Tensor] = []
    batch_offsets = torch.arange(bsz, device=x.device, dtype=torch.long).reshape(bsz, 1) * seq_len
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        k_j = x.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (k_src, _) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = k_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
        role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, n_heads, dh)
        source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, n_heads, dh)
        score_j = (q * (k_j + role + source)).sum(dim=-1) / math.sqrt(float(dh))
        score_j = score_j + type_bias[cand_types[:, :, j]]
        if score_bias is not None:
            score_j = score_j + score_bias[:, :, j, None]
        if qtb is not None:
            score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, n_heads)).squeeze(2)
        score_j = score_j + source_bias[source_j]
        score_j = score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min)
        score_parts.append(score_j)
    scores = torch.stack(score_parts, dim=2)
    probs = F.softmax(scores, dim=2)

    r_all_h = x.new_zeros((bsz, seq_len, n_heads, dh))
    typed_reads_h = {
        type_id: x.new_zeros((bsz, seq_len, n_heads, dh))
        for type_id in read_type_ids
        if 0 <= int(type_id) < n_types
    }
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        v_j = x.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (_, v_src) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = v_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
        contrib = probs[:, :, j, :, None] * v_j
        r_all_h = r_all_h + contrib
        for type_id in typed_reads_h:
            type_mask = ((cand_types[:, :, j] == int(type_id)) & cand_mask[:, :, j])[:, :, None, None]
            typed_reads_h[type_id] = typed_reads_h[type_id] + contrib * type_mask.to(contrib.dtype)

    r_all = r_all_h.reshape(bsz, seq_len, d)
    read = F.linear(r_all, read_mix_weight[:, :d])
    for type_id, r_type_h in typed_reads_h.items():
        r_type = r_type_h.reshape(bsz, seq_len, d)
        start = (int(type_id) + 1) * d
        read = read + F.linear(r_type, read_mix_weight[:, start : start + d])
    z = torch.cat([x, read, x * read, read - x], dim=-1)
    z_n = F.layer_norm(z, (4 * d,), norm_z_weight, norm_z_bias)
    hidden = F.gelu(F.linear(z_n, fuse0_weight, fuse0_bias))
    delta = F.linear(hidden, fuse2_weight, fuse2_bias)
    gate = torch.sigmoid(gate_param).reshape(1, 1, d)
    return x + gate * delta


def _dsqg_w_sourcewise_read_slots_recompute(
    q: torch.Tensor,
    k_final: torch.Tensor,
    v_final: torch.Tensor,
    k_l3: torch.Tensor,
    v_l3: torch.Tensor,
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    role_key_weight: torch.Tensor,
    source_key_weight: torch.Tensor,
    type_bias: torch.Tensor,
    source_bias: torch.Tensor,
    qtb: torch.Tensor | None,
    score_bias: torch.Tensor | None,
    cand_token_indices: torch.Tensor,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
    type_slot_map: torch.Tensor,
    *,
    d: int,
    n_heads: int,
    dh: int,
    read_slots: int,
) -> torch.Tensor:
    """Compact [B,N,S,D] read-slot recompute for the read-only Triton autograd node."""
    bsz, seq_len, j_count = cand_mask.shape
    gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
    batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).reshape(bsz, 1) * seq_len
    projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        int(CandidateSource.FINAL): (k_final, v_final),
        int(CandidateSource.QUESTION_CACHE): (k_final, v_final),
        int(CandidateSource.L3): (k_l3, v_l3),
        int(CandidateSource.HISA): (k_l3, v_l3),
        int(CandidateSource.SUMMARY): (k_summary, v_summary),
    }

    score_parts: list[torch.Tensor] = []
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        k_j = q.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (k_src, _) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = k_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
        role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, n_heads, dh)
        source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, n_heads, dh)
        score_j = (q * (k_j + role + source)).sum(dim=-1) / math.sqrt(float(dh))
        score_j = score_j + type_bias[cand_types[:, :, j]]
        score_j = score_j + source_bias[source_j]
        if score_bias is not None:
            score_j = score_j + score_bias[:, :, j, None]
        if qtb is not None:
            score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, n_heads)).squeeze(2)
        score_j = score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min)
        score_parts.append(score_j)
    scores = torch.stack(score_parts, dim=2)
    probs = F.softmax(scores, dim=2)

    slots_h = q.new_zeros((bsz, seq_len, read_slots, n_heads, dh))
    for j in range(j_count):
        token_j = gather_tokens[:, :, j]
        source_j = cand_sources[:, :, j]
        flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
        v_j = q.new_zeros((bsz, seq_len, n_heads, dh))
        for source_id, (_, v_src) in projected_sources.items():
            source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
            if bool(source_mask.any()):
                gathered = v_src.reshape(bsz * seq_len, n_heads, dh).index_select(0, flat_indices).reshape(bsz, seq_len, n_heads, dh)
                v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
        contrib = probs[:, :, j, :, None] * v_j
        slots_h[:, :, 0] = slots_h[:, :, 0] + contrib
        type_slots = type_slot_map[cand_types[:, :, j]].to(torch.long)
        for slot in range(1, read_slots):
            slot_mask = ((type_slots == slot) & cand_mask[:, :, j])[:, :, None, None]
            slots_h[:, :, slot] = slots_h[:, :, slot] + contrib * slot_mask.to(contrib.dtype)
    return slots_h.reshape(bsz, seq_len, read_slots, d)

class _DSQGWSourcewiseTritonCompactRead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k_final,
        v_final,
        k_l3,
        v_l3,
        k_summary,
        v_summary,
        role_key_weight,
        source_key_weight,
        type_bias,
        source_bias,
        qtb,
        score_bias,
        cand_token_indices,
        cand_types,
        cand_sources,
        cand_mask,
        type_slot_map,
        use_qtb: bool,
        use_score_bias: bool,
        d: int,
        n_heads: int,
        dh: int,
        n_types: int,
        read_slots: int,
        block_hd: int,
    ):
        if not _TRITON_SOURCEWISE_AVAILABLE or triton is None or _dsqg_w_sourcewise_read_slots_kernel is None:
            raise NotImplementedError("DSQG-W sourcewise compact read requires Triton")
        ctx.use_qtb = bool(use_qtb)
        ctx.use_score_bias = bool(use_score_bias)
        ctx.d = int(d)
        ctx.n_heads = int(n_heads)
        ctx.dh = int(dh)
        ctx.read_slots = int(read_slots)
        bsz, seq_len = q.shape[:2]
        schedule = _dsqg_w_triton_schedule(dh, q.device)
        read_slots_out = torch.empty((bsz, seq_len, int(read_slots), int(d)), device=q.device, dtype=q.dtype)
        lse_out = torch.empty((bsz, seq_len, int(n_heads)), device=q.device, dtype=torch.float32)
        ctx.save_for_backward(
            q,
            k_final,
            v_final,
            k_l3,
            v_l3,
            k_summary,
            v_summary,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse_out,
        )
        empty = torch.empty((0,), device=q.device, dtype=q.dtype)
        _dsqg_w_sourcewise_read_slots_kernel[(bsz * seq_len * int(n_heads),)](
            q.contiguous(),
            k_final.contiguous(),
            v_final.contiguous(),
            k_l3.contiguous(),
            v_l3.contiguous(),
            k_summary.contiguous(),
            v_summary.contiguous(),
            role_key_weight.contiguous(),
            source_key_weight.contiguous(),
            type_bias.contiguous(),
            source_bias.contiguous(),
            qtb.contiguous() if bool(use_qtb) else empty,
            score_bias.contiguous() if bool(use_score_bias) else empty,
            cand_token_indices.contiguous(),
            cand_types.contiguous(),
            cand_sources.contiguous(),
            cand_mask.contiguous(),
            type_slot_map.contiguous(),
            read_slots_out,
            lse_out,
            empty,
            B=bsz,
            N=seq_len,
            H=int(n_heads),
            HD=int(dh),
            D=int(d),
            J=cand_mask.shape[-1],
            N_TYPES=int(n_types),
            READ_SLOTS=int(read_slots),
            MAX_READ_SLOTS=int(triton.next_power_of_2(int(read_slots))),
            BLOCK_HD=schedule.block_hd,
            USE_QTB=bool(use_qtb),
            USE_SCORE_BIAS=bool(use_score_bias),
            STORE_LSE=True,
            STORE_PROBS=False,
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
        )
        return read_slots_out

    @staticmethod
    def backward(ctx, grad_read_slots):
        saved = ctx.saved_tensors
        (
            q,
            k_final,
            v_final,
            k_l3,
            v_l3,
            k_summary,
            v_summary,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse,
        ) = saved
        bsz, seq_len, j_count = cand_mask.shape
        h = ctx.n_heads
        dh = ctx.dh
        d = ctx.d
        if os.getenv("DWARF_DSQG_W_TRITON_COMPACT_READ_BACKWARD", "triton").lower() != "pytorch":
            grad_q = torch.zeros_like(q)
            grad_k_final = torch.zeros_like(k_final)
            grad_v_final = torch.zeros_like(v_final)
            grad_k_l3 = torch.zeros_like(k_l3)
            grad_v_l3 = torch.zeros_like(v_l3)
            grad_k_summary = torch.zeros_like(k_summary)
            grad_v_summary = torch.zeros_like(v_summary)
            grad_role_key = torch.zeros_like(role_key_weight)
            grad_source_key = torch.zeros_like(source_key_weight)
            grad_type_bias = torch.zeros_like(type_bias)
            grad_source_bias = torch.zeros_like(source_bias)
            grad_qtb = torch.zeros_like(qtb) if ctx.use_qtb else None
            grad_score_bias = torch.zeros_like(score_bias) if ctx.use_score_bias else None
            empty = torch.empty((0,), device=q.device, dtype=q.dtype)
            schedule = _dsqg_w_triton_schedule(dh, q.device)
            grid = (bsz * seq_len * h,)

            def launch_split_kernel(*, compute_query: bool, compute_source: bool) -> None:
                _dsqg_w_sourcewise_read_slots_backward_kernel[grid](
                    q.contiguous(),
                    k_final.contiguous(),
                    v_final.contiguous(),
                    k_l3.contiguous(),
                    v_l3.contiguous(),
                    k_summary.contiguous(),
                    v_summary.contiguous(),
                    role_key_weight.contiguous(),
                    source_key_weight.contiguous(),
                    type_bias.contiguous(),
                    source_bias.contiguous(),
                    qtb.contiguous() if ctx.use_qtb else empty,
                    score_bias.contiguous() if ctx.use_score_bias else empty,
                    cand_token_indices.contiguous(),
                    cand_types.contiguous(),
                    cand_sources.contiguous(),
                    cand_mask.contiguous(),
                    type_slot_map.contiguous(),
                    lse.contiguous(),
                    grad_read_slots.contiguous(),
                    grad_q,
                    grad_k_final,
                    grad_v_final,
                    grad_k_l3,
                    grad_v_l3,
                    grad_k_summary,
                    grad_v_summary,
                    grad_role_key,
                    grad_source_key,
                    grad_type_bias,
                    grad_source_bias,
                    grad_qtb if grad_qtb is not None else empty,
                    grad_score_bias if grad_score_bias is not None else empty,
                    B=bsz,
                    N=seq_len,
                    H=h,
                    HD=dh,
                    D=d,
                    J=j_count,
                    N_TYPES=type_bias.shape[0],
                    READ_SLOTS=ctx.read_slots,
                    BLOCK_HD=schedule.block_hd,
                    USE_QTB=ctx.use_qtb,
                    USE_SCORE_BIAS=ctx.use_score_bias,
                    COMPUTE_QUERY=compute_query,
                    COMPUTE_SOURCE=compute_source,
                    num_warps=schedule.num_warps,
                    num_stages=schedule.num_stages,
                )

            # V20-style organization can be enabled for profiling, but keep the
            # fused monolithic launch as the default until split scheduling wins in
            # full trainer windows rather than only as a code-organization pattern.
            split_backward = os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_ORGANIZATION", "monolithic").lower() in {
                "1",
                "true",
                "split",
                "v20_split",
            }
            source_grads = os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRADS", "1") != "0"
            source_grad_every = max(1, int(os.getenv("DWARF_DSQG_W_TRITON_BACKWARD_SOURCE_GRAD_EVERY", "1")))
            if source_grads and source_grad_every > 1:
                source_grad_counter = int(getattr(_DSQGWSourcewiseTritonCompactRead, "_source_grad_counter", 0))
                source_grads = (source_grad_counter % source_grad_every) == 0
                setattr(_DSQGWSourcewiseTritonCompactRead, "_source_grad_counter", source_grad_counter + 1)
            if not source_grads:
                launch_split_kernel(compute_query=True, compute_source=False)
            elif split_backward:
                launch_split_kernel(compute_query=True, compute_source=False)
                launch_split_kernel(compute_query=False, compute_source=True)
            else:
                launch_split_kernel(compute_query=True, compute_source=True)
            return (
                grad_q,
                grad_k_final,
                grad_v_final,
                grad_k_l3,
                grad_v_l3,
                grad_k_summary,
                grad_v_summary,
                grad_role_key,
                grad_source_key,
                grad_type_bias,
                grad_source_bias,
                grad_qtb,
                grad_score_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        gather_tokens = cand_token_indices.clamp(0, max(seq_len - 1, 0))
        batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).reshape(bsz, 1) * seq_len
        inv_sqrt = 1.0 / math.sqrt(float(dh))
        projected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            int(CandidateSource.FINAL): (k_final, v_final),
            int(CandidateSource.QUESTION_CACHE): (k_final, v_final),
            int(CandidateSource.L3): (k_l3, v_l3),
            int(CandidateSource.HISA): (k_l3, v_l3),
            int(CandidateSource.SUMMARY): (k_summary, v_summary),
        }

        score_parts: list[torch.Tensor] = []
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            k_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (k_src, _) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = k_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    k_j = k_j + gathered * source_mask[:, :, None, None].to(k_j.dtype)
            role = F.embedding(cand_types[:, :, j], role_key_weight).reshape(bsz, seq_len, h, dh)
            source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, h, dh)
            score_j = (q * (k_j + role + source)).sum(dim=-1) * inv_sqrt
            score_j = score_j + type_bias[cand_types[:, :, j]] + source_bias[source_j]
            if ctx.use_score_bias:
                score_j = score_j + score_bias[:, :, j, None]
            if ctx.use_qtb:
                score_j = score_j + qtb.gather(2, cand_types[:, :, j, None, None].expand(-1, -1, 1, h)).squeeze(2)
            score_parts.append(score_j.masked_fill(~cand_mask[:, :, j, None], torch.finfo(score_j.dtype).min))
        scores = torch.stack(score_parts, dim=2)
        probs = F.softmax(scores, dim=2)

        grad_slots_h = grad_read_slots.reshape(bsz, seq_len, ctx.read_slots, h, dh)
        dp_parts: list[torch.Tensor] = []
        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            v_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (_, v_src) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = v_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    v_j = v_j + gathered * source_mask[:, :, None, None].to(v_j.dtype)
            type_slots = type_slot_map[cand_types[:, :, j]].to(torch.long)
            dcontrib = grad_slots_h[:, :, 0]
            for slot in range(1, ctx.read_slots):
                dcontrib = dcontrib + grad_slots_h[:, :, slot] * (type_slots == slot)[:, :, None, None].to(grad_slots_h.dtype)
            dp_parts.append((dcontrib * v_j).sum(dim=-1))
        dp = torch.stack(dp_parts, dim=2)
        ds = probs * (dp - (dp * probs).sum(dim=2, keepdim=True))
        ds = ds.masked_fill(~cand_mask[:, :, :, None], 0.0)

        grad_q = torch.zeros_like(q)
        grad_k_final = torch.zeros_like(k_final)
        grad_v_final = torch.zeros_like(v_final)
        grad_k_l3 = torch.zeros_like(k_l3)
        grad_v_l3 = torch.zeros_like(v_l3)
        grad_k_summary = torch.zeros_like(k_summary)
        grad_v_summary = torch.zeros_like(v_summary)
        grad_role_key = torch.zeros_like(role_key_weight)
        grad_source_key = torch.zeros_like(source_key_weight)
        grad_type_bias = torch.zeros_like(type_bias)
        grad_source_bias = torch.zeros_like(source_bias)
        grad_qtb = torch.zeros_like(qtb) if ctx.use_qtb else None
        grad_score_bias = torch.zeros_like(score_bias) if ctx.use_score_bias else None
        k_grads: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): grad_k_final,
            int(CandidateSource.QUESTION_CACHE): grad_k_final,
            int(CandidateSource.L3): grad_k_l3,
            int(CandidateSource.HISA): grad_k_l3,
            int(CandidateSource.SUMMARY): grad_k_summary,
        }
        v_grads: dict[int, torch.Tensor] = {
            int(CandidateSource.FINAL): grad_v_final,
            int(CandidateSource.QUESTION_CACHE): grad_v_final,
            int(CandidateSource.L3): grad_v_l3,
            int(CandidateSource.HISA): grad_v_l3,
            int(CandidateSource.SUMMARY): grad_v_summary,
        }

        for j in range(j_count):
            token_j = gather_tokens[:, :, j]
            source_j = cand_sources[:, :, j]
            ctype_j = cand_types[:, :, j]
            flat_indices = (batch_offsets + token_j.to(torch.long)).reshape(-1)
            type_slots = type_slot_map[ctype_j].to(torch.long)
            dcontrib = grad_slots_h[:, :, 0]
            for slot in range(1, ctx.read_slots):
                dcontrib = dcontrib + grad_slots_h[:, :, slot] * (type_slots == slot)[:, :, None, None].to(grad_slots_h.dtype)
            d_v_j = probs[:, :, j, :, None] * dcontrib

            k_eff_j = q.new_zeros((bsz, seq_len, h, dh))
            for source_id, (k_src, _) in projected_sources.items():
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    gathered = k_src.reshape(bsz * seq_len, h, dh).index_select(0, flat_indices).reshape(bsz, seq_len, h, dh)
                    k_eff_j = k_eff_j + gathered * source_mask[:, :, None, None].to(k_eff_j.dtype)
            role = F.embedding(ctype_j, role_key_weight).reshape(bsz, seq_len, h, dh)
            source = F.embedding(source_j, source_key_weight).reshape(bsz, seq_len, h, dh)
            k_eff_j = k_eff_j + role + source
            d_k_eff = ds[:, :, j, :, None] * q * inv_sqrt
            grad_q = grad_q + ds[:, :, j, :, None] * k_eff_j * inv_sqrt

            for source_id in k_grads:
                source_mask = (source_j == int(source_id)) & cand_mask[:, :, j]
                if bool(source_mask.any()):
                    mask = source_mask[:, :, None, None].to(d_k_eff.dtype)
                    k_add = (d_k_eff * mask).reshape(bsz * seq_len, h, dh).to(k_grads[source_id].dtype)
                    v_add = (d_v_j * mask).reshape(bsz * seq_len, h, dh).to(v_grads[source_id].dtype)
                    k_grads[source_id].reshape(bsz * seq_len, h, dh).index_add_(0, flat_indices, k_add)
                    v_grads[source_id].reshape(bsz * seq_len, h, dh).index_add_(0, flat_indices, v_add)

            grad_role_key.index_add_(0, ctype_j.reshape(-1), d_k_eff.reshape(bsz * seq_len, d).to(grad_role_key.dtype))
            grad_source_key.index_add_(0, source_j.reshape(-1), d_k_eff.reshape(bsz * seq_len, d).to(grad_source_key.dtype))
            ctype_flat = ctype_j.reshape(-1)
            source_flat = source_j.reshape(-1)
            ds_flat = ds[:, :, j, :].reshape(bsz * seq_len, h)
            for head_idx in range(h):
                grad_type_bias[:, head_idx].index_add_(0, ctype_flat, ds_flat[:, head_idx].to(grad_type_bias.dtype))
                grad_source_bias[:, head_idx].index_add_(0, source_flat, ds_flat[:, head_idx].to(grad_source_bias.dtype))
            if grad_qtb is not None:
                grad_qtb.scatter_add_(2, ctype_j[:, :, None, None].expand(-1, -1, 1, h), ds[:, :, j, None, :].to(grad_qtb.dtype))
            if grad_score_bias is not None:
                grad_score_bias[:, :, j] = ds[:, :, j, :].sum(dim=-1).to(grad_score_bias.dtype)

        grad_list: list[torch.Tensor | None] = [
            grad_q,
            grad_k_final,
            grad_v_final,
            grad_k_l3,
            grad_v_l3,
            grad_k_summary,
            grad_v_summary,
            grad_role_key,
            grad_source_key,
            grad_type_bias,
            grad_source_bias,
        ]
        return (
            *grad_list,
            grad_qtb,
            grad_score_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DSQGWSourcewiseTritonRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask, l3_present: bool, chunk_present: bool, scores_present: bool, *params):
        ctx.block = block
        ctx.l3_present = bool(l3_present)
        ctx.chunk_present = bool(chunk_present)
        ctx.scores_present = bool(scores_present)
        ctx.save_for_backward(x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask, *params)
        out, _ = block._forward_sourcewise_triton(
            x,
            cand_token_indices,
            cand_types,
            cand_sources,
            cand_mask,
            l3_states=l3_states if l3_present else None,
            chunk_rep_states=chunk_rep_states if chunk_present else None,
            cand_scores=cand_scores if scores_present else None,
            return_routing=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, l3_states, chunk_rep_states, cand_scores, cand_token_indices, cand_types, cand_sources, cand_mask = saved[:8]
        params = saved[8:]
        block = ctx.block
        x_req = x.detach().requires_grad_(True)
        l3_req = l3_states.detach().requires_grad_(True) if ctx.l3_present else None
        chunk_req = chunk_rep_states.detach().requires_grad_(True) if ctx.chunk_present else None
        param_reqs = [p.detach().requires_grad_(True) for p in params]
        with torch.enable_grad():
            out = _dsqg_w_sourcewise_functional_recompute(
                x_req,
                l3_req,
                chunk_req,
                cand_token_indices,
                cand_types,
                cand_sources,
                cand_mask,
                cand_scores if ctx.scores_present else None,
                d=block.d,
                n_heads=block.n_heads,
                dh=block.dh,
                n_types=block.n_types,
                read_type_ids=block.read_type_ids,
                use_query_type_bias=block.use_query_type_bias,
                norm_x_weight=param_reqs[0],
                norm_x_bias=param_reqs[1],
                norm_c_weight=param_reqs[2],
                norm_c_bias=param_reqs[3],
                q_proj_weight=param_reqs[4],
                k_proj_weight=param_reqs[5],
                v_proj_weight=param_reqs[6],
                role_key_weight=param_reqs[7],
                source_key_weight=param_reqs[8],
                type_bias=param_reqs[9],
                query_type_bias_weight=param_reqs[10],
                source_bias=param_reqs[11],
                read_mix_weight=param_reqs[12],
                norm_z_weight=param_reqs[13],
                norm_z_bias=param_reqs[14],
                fuse0_weight=param_reqs[15],
                fuse0_bias=param_reqs[16],
                fuse2_weight=param_reqs[17],
                fuse2_bias=param_reqs[18],
                gate_param=param_reqs[19],
            )
            grad_inputs = torch.autograd.grad(
                out,
                [x_req] + ([l3_req] if l3_req is not None else []) + ([chunk_req] if chunk_req is not None else []) + param_reqs,
                grad_out,
                allow_unused=True,
            )
        idx = 0
        grad_x = grad_inputs[idx]; idx += 1
        grad_l3 = grad_inputs[idx] if ctx.l3_present else None
        if ctx.l3_present:
            idx += 1
        grad_chunk = grad_inputs[idx] if ctx.chunk_present else None
        if ctx.chunk_present:
            idx += 1
        grad_params = list(grad_inputs[idx:])
        while len(grad_params) < len(params):
            grad_params.append(None)
        return (None, grad_x, grad_l3, grad_chunk, None, None, None, None, None, None, None, None, *grad_params)

def _dsqg_w_materialized_read_slots_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    role_key_weight: torch.Tensor,
    source_key_weight: torch.Tensor,
    type_bias: torch.Tensor,
    source_bias: torch.Tensor,
    qtb: torch.Tensor | None,
    score_bias: torch.Tensor | None,
    cand_types: torch.Tensor,
    cand_sources: torch.Tensor,
    cand_mask: torch.Tensor,
    type_slot_map: torch.Tensor,
    *,
    d: int,
    n_heads: int,
    dh: int,
    read_slots: int,
) -> torch.Tensor:
    """Compact [B,N,S,D] read-slot recompute for transformed/materialized candidates."""
    bsz, seq_len, j_count = cand_mask.shape
    inv_sqrt = 1.0 / math.sqrt(float(dh))
    role = F.embedding(cand_types, role_key_weight).reshape(bsz, seq_len, j_count, n_heads, dh)
    source = F.embedding(cand_sources, source_key_weight).reshape(bsz, seq_len, j_count, n_heads, dh)
    scores = (q[:, :, None, :, :] * (k + role + source)).sum(dim=-1) * inv_sqrt
    scores = scores + type_bias[cand_types] + source_bias[cand_sources]
    if score_bias is not None:
        scores = scores + score_bias[:, :, :, None]
    if qtb is not None:
        scores = scores + qtb.gather(2, cand_types[:, :, :, None].expand(-1, -1, -1, n_heads))
    scores = scores.masked_fill(~cand_mask[:, :, :, None], torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=2)
    contrib = probs[:, :, :, :, None] * v
    slots_h = q.new_zeros((bsz, seq_len, read_slots, n_heads, dh))
    slots_h[:, :, 0] = contrib.sum(dim=2)
    type_slots = type_slot_map[cand_types].to(torch.long)
    for slot in range(1, read_slots):
        slot_mask = ((type_slots == slot) & cand_mask)[:, :, :, None, None]
        slots_h[:, :, slot] = (contrib * slot_mask.to(contrib.dtype)).sum(dim=2)
    return slots_h.reshape(bsz, seq_len, read_slots, d)


class _DSQGWMaterializedTritonCompactRead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        role_key_weight,
        source_key_weight,
        type_bias,
        source_bias,
        qtb,
        score_bias,
        cand_types,
        cand_sources,
        cand_mask,
        type_slot_map,
        use_qtb: bool,
        use_score_bias: bool,
        d: int,
        n_heads: int,
        dh: int,
        n_types: int,
        read_slots: int,
        block_hd: int,
    ):
        ctx.use_qtb = bool(use_qtb)
        ctx.use_score_bias = bool(use_score_bias)
        ctx.d = int(d)
        ctx.n_heads = int(n_heads)
        ctx.dh = int(dh)
        ctx.read_slots = int(read_slots)
        bsz, seq_len = q.shape[:2]
        schedule = _dsqg_w_triton_schedule(dh, q.device)
        read_slots_out = torch.empty((bsz, seq_len, int(read_slots), int(d)), device=q.device, dtype=q.dtype)
        lse_out = torch.empty((bsz, seq_len, int(n_heads)), device=q.device, dtype=torch.float32)
        ctx.save_for_backward(
            q,
            k,
            v,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse_out,
        )
        empty = torch.empty((0,), device=q.device, dtype=q.dtype)
        _dsqg_w_materialized_read_slots_kernel[(bsz * seq_len * int(n_heads),)](
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            role_key_weight.contiguous(),
            source_key_weight.contiguous(),
            type_bias.contiguous(),
            source_bias.contiguous(),
            qtb.contiguous() if bool(use_qtb) else empty,
            score_bias.contiguous() if bool(use_score_bias) else empty,
            cand_types.contiguous(),
            cand_sources.contiguous(),
            cand_mask.contiguous(),
            type_slot_map.contiguous(),
            read_slots_out,
            lse_out,
            B=bsz,
            N=seq_len,
            H=int(n_heads),
            HD=int(dh),
            D=int(d),
            J=cand_mask.shape[-1],
            N_TYPES=int(n_types),
            READ_SLOTS=int(read_slots),
            MAX_READ_SLOTS=int(triton.next_power_of_2(int(read_slots))),
            BLOCK_HD=schedule.block_hd,
            USE_QTB=bool(use_qtb),
            USE_SCORE_BIAS=bool(use_score_bias),
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
        )
        return read_slots_out

    @staticmethod
    def backward(ctx, grad_read_slots):
        (
            q,
            k,
            v,
            role_key_weight,
            source_key_weight,
            type_bias,
            source_bias,
            qtb,
            score_bias,
            cand_types,
            cand_sources,
            cand_mask,
            type_slot_map,
            lse,
        ) = ctx.saved_tensors
        materialized_backward_impl = os.getenv(
            "DWARF_DSQG_W_MATERIALIZED_COMPACT_READ_BACKWARD",
            "pytorch",
        ).lower()
        if materialized_backward_impl in {"1", "true", "triton"}:
            bsz, seq_len, _ = cand_mask.shape
            h = ctx.n_heads
            dh = ctx.dh
            d = ctx.d
            grad_q = torch.zeros_like(q)
            grad_k = torch.zeros_like(k)
            grad_v = torch.zeros_like(v)
            grad_role = torch.zeros_like(role_key_weight)
            grad_source = torch.zeros_like(source_key_weight)
            grad_type_bias = torch.zeros_like(type_bias)
            grad_source_bias = torch.zeros_like(source_bias)
            grad_qtb = torch.zeros_like(qtb) if ctx.use_qtb else None
            grad_score_bias = torch.zeros_like(score_bias) if ctx.use_score_bias else None
            empty = torch.empty((0,), device=q.device, dtype=q.dtype)
            schedule = _dsqg_w_triton_schedule(dh, q.device)
            _dsqg_w_materialized_read_slots_backward_kernel[(bsz * seq_len * h,)](
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                role_key_weight.contiguous(),
                source_key_weight.contiguous(),
                type_bias.contiguous(),
                source_bias.contiguous(),
                qtb.contiguous() if ctx.use_qtb else empty,
                score_bias.contiguous() if ctx.use_score_bias else empty,
                cand_types.contiguous(),
                cand_sources.contiguous(),
                cand_mask.contiguous(),
                type_slot_map.contiguous(),
                lse.contiguous(),
                grad_read_slots.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                grad_role,
                grad_source,
                grad_type_bias,
                grad_source_bias,
                grad_qtb if grad_qtb is not None else empty,
                grad_score_bias if grad_score_bias is not None else empty,
                B=bsz,
                N=seq_len,
                H=h,
                HD=dh,
                D=d,
                J=cand_mask.shape[-1],
                N_TYPES=type_bias.shape[0],
                READ_SLOTS=ctx.read_slots,
                BLOCK_HD=schedule.block_hd,
                USE_QTB=ctx.use_qtb,
                USE_SCORE_BIAS=ctx.use_score_bias,
                num_warps=schedule.num_warps,
                num_stages=schedule.num_stages,
            )
            return (
                grad_q,
                grad_k,
                grad_v,
                grad_role,
                grad_source,
                grad_type_bias,
                grad_source_bias,
                grad_qtb,
                grad_score_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)
        role_req = role_key_weight.detach().requires_grad_(True)
        source_req = source_key_weight.detach().requires_grad_(True)
        type_bias_req = type_bias.detach().requires_grad_(True)
        source_bias_req = source_bias.detach().requires_grad_(True)
        qtb_req = qtb.detach().requires_grad_(True) if ctx.use_qtb else None
        score_bias_req = score_bias.detach().requires_grad_(True) if ctx.use_score_bias else None
        grad_targets = [q_req, k_req, v_req, role_req, source_req, type_bias_req, source_bias_req]
        if qtb_req is not None:
            grad_targets.append(qtb_req)
        if score_bias_req is not None:
            grad_targets.append(score_bias_req)
        with torch.enable_grad():
            read_slots = _dsqg_w_materialized_read_slots_recompute(
                q_req,
                k_req,
                v_req,
                role_req,
                source_req,
                type_bias_req,
                source_bias_req,
                qtb_req,
                score_bias_req,
                cand_types,
                cand_sources,
                cand_mask,
                type_slot_map,
                d=ctx.d,
                n_heads=ctx.n_heads,
                dh=ctx.dh,
                read_slots=ctx.read_slots,
            )
            grads = torch.autograd.grad(read_slots, grad_targets, grad_read_slots, allow_unused=True)
        grad_iter = iter(grads)
        grad_q = next(grad_iter)
        grad_k = next(grad_iter)
        grad_v = next(grad_iter)
        grad_role = next(grad_iter)
        grad_source = next(grad_iter)
        grad_type_bias = next(grad_iter)
        grad_source_bias = next(grad_iter)
        grad_qtb = next(grad_iter) if ctx.use_qtb else None
        grad_score_bias = next(grad_iter) if ctx.use_score_bias else None
        return (
            grad_q,
            grad_k,
            grad_v,
            grad_role,
            grad_source,
            grad_type_bias,
            grad_source_bias,
            grad_qtb,
            grad_score_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

__all__ = [
    "_DSQGWSourcewiseTritonCompactRead",
    "_DSQGWSourcewiseTritonRecompute",
    "_dsqg_w_sourcewise_score_read_kernel",
    "_dsqg_w_sourcewise_read_slots_kernel",
    "_dsqg_w_sourcewise_read_slots_backward_kernel",
    "_dsqg_w_sourcewise_functional_recompute",
    "_dsqg_w_sourcewise_read_slots_recompute",
    "_DSQGWMaterializedTritonCompactRead",
    "_dsqg_w_materialized_read_slots_recompute",
]
