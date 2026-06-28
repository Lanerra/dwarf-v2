#!/usr/bin/env python3
"""q6_g128 fused/direct-consume DSQG forward kernels.

This Stage-A probe removes the gathered K/V tensors from the q6 attention read
boundary.  The Triton kernel consumes packed q6_g128 K/V directly and produces
only the DSQG attention result [B,H,N,64] plus a small [B,H,N] LSE tensor and an
[N] causal-valid-count diagnostic.

Fused semantics covered here:
- signed symmetric q6_g128 decode with the existing token-pair cache layout;
- BF16 rounding at the same point as the direct-gather oracle;
- q dot decoded K;
- q dot scale_embed;
- per-offset/per-head pos_bias;
- causal invalid-offset masking, including all-invalid rows;
- softmax over J held entirely in registers (no [B,H,N,J] allocation);
- sparse MOVT rotations using gated phase_base/phase_gain, y_pre, z_pre,
  plane_shift, and R_PLANES=4;
- weighted V reduction to BF16 [B,H,N,64].

Deliberately outside this Stage-A kernel boundary:
- q6 packing/stochastic rounding (the existing deterministic packer is reused);
- NPCI and q/k/v projection (the kernel consumes post-projection/post-NPCI q/k/v);
- if_gain, output projection, output gate, and dropout;
- backward/STE gradients (designed separately for Stage C).

The oracle is the existing q6 direct-decode-gather K/V path followed by the same
PyTorch DSQG consumer used in the q6 trainer clone.  The probe also checks that
the direct-gather oracle remains exactly equal to the q6 full-scratch oracle.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

# Triton 3.5+ compatibility for module-scope constants referenced by JIT code.
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "torch is required; use /home/dlewis3/Desktop/AI/DWARF/.venv/bin/python"
    ) from exc

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("triton is required for the q6_g128 fused direct-consume probe") from exc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "logs/q6_g128_fused_direct_consume_summary"

from . import layout as layout_mod
from . import decode as direct_mod

HEAD_DIM = 64
R_PLANES = 4
PAYLOAD_BYTES_PER_PAIR = 96

# Actual first DSQG group from the D512/L10 trainer: 17 dense/small offsets,
# followed by 15 sparse/large offsets that receive MOVT.
DEFAULT_GROUP_A_OFFSETS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 19, 21, 23, 28,
    48, 64, 96, 121, 161, 192, 212, 245, 273, 295, 342, 375, 384, 413, 441,
]
DEFAULT_GROUP_A_J_SMALL = 17


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


@triton.jit
def _decode_q6_g128_token_rows(
    payload_ptr,
    scales_ptr,
    bh,
    token_idx,
    row_valid,
    ds,
    dim_valid,
    PAIR_GROUPS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    HD: tl.constexpr,
):
    """Decode [BLOCK_N,HD] q6 rows and round exactly through BF16 storage."""
    safe_t = tl.maximum(token_idx, 0)
    pair = safe_t >> 1
    half = safe_t & 1
    val_in_pair = half[:, None] * HD + ds[None, :]
    word_idx = val_in_pair >> 2
    lane = val_in_pair & 3

    payload_base = (
        (bh * PAIR_GROUPS + pair[:, None]) * PAYLOAD_BYTES
        + word_idx * 3
    )
    load_mask = row_valid[:, None] & dim_valid[None, :]
    b0 = tl.load(payload_ptr + payload_base + 0, mask=load_mask, other=0).to(tl.uint32)
    b1 = tl.load(payload_ptr + payload_base + 1, mask=load_mask, other=0).to(tl.uint32)
    b2 = tl.load(payload_ptr + payload_base + 2, mask=load_mask, other=0).to(tl.uint32)
    word = b0 | (b1 << 8) | (b2 << 16)
    code = ((word >> (lane * 6)) & 0x3F).to(tl.int32)
    signed = tl.where(code >= 32, code - 64, code).to(tl.float32)
    scale = tl.load(
        scales_ptr + bh * PAIR_GROUPS + pair,
        mask=row_valid,
        other=0.0,
    ).to(tl.float32)

    # The existing direct-gather oracle writes decoded q6 values to BF16 and the
    # PyTorch consumer immediately casts them back to FP32.  Preserve that exact
    # quantization boundary rather than dotting unrounded FP32 decoded values.
    decoded = signed * scale[:, None]
    return decoded.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _q6_g128_dsqg_direct_consume_kernel(
    Q,
    K_PAYLOAD,
    K_SCALES,
    V_PAYLOAD,
    V_SCALES,
    POS_BIAS,
    SCALE_EMBED,
    PHASE_BASE,
    PHASE_GAIN,
    Y_PRE,
    Z_PRE,
    OFFSETS,
    OUT,
    LSE,
    VALID_COUNTS,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ln,
    stride_pbi,
    stride_pbh,
    stride_sei,
    stride_sed,
    stride_phi,
    stride_phh,
    stride_pgi,
    stride_pgh,
    stride_yb,
    stride_yh,
    stride_yn,
    stride_zb,
    stride_zh,
    stride_zn,
    N,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HD: tl.constexpr,
    PAIR_GROUPS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr,
    J_SMALL_VAL: tl.constexpr,
    J_LARGE_VAL: tl.constexpr,
    J_PAD: tl.constexpr,
    R_PLANES_VAL: tl.constexpr,
    PLANE_SHIFT: tl.constexpr,
):
    """Decode packed q6 K/V directly into one DSQG attention result tile."""
    bh = tl.program_id(0)
    block_n = tl.program_id(1)
    b = bh // H_Q
    h = bh % H_Q
    kv_h = h // KV_GROUP
    kv_bh = b * H_KV + kv_h

    ns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb = Q + b * stride_qb + h * stride_qh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(
        qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
        mask=nm[:, None] & dm[None, :],
        other=0.0,
    ).to(tl.float32)

    # J=32 in the trainer.  Keeping this [BLOCK_N,J_PAD] matrix in registers is
    # both smaller and numerically closer to the current offline PyTorch softmax
    # than an online recurrence.  No global [B,H,N,J] tensor is created.
    scores = tl.full([BLOCK_N, J_PAD], float("-inf"), tl.float32)
    valid_count = tl.zeros([BLOCK_N], tl.int32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)

        kt = _decode_q6_g128_token_rows(
            K_PAYLOAD,
            K_SCALES,
            kv_bh,
            kp,
            valid,
            ds,
            dm,
            PAIR_GROUPS=PAIR_GROUPS,
            PAYLOAD_BYTES=PAYLOAD_BYTES,
            HD=HD,
        )
        se_i = tl.load(
            SCALE_EMBED + i * stride_sei + ds * stride_sed,
            mask=dm,
            other=0.0,
        ).to(tl.float32)
        s = tl.sum(q * kt, axis=1) * sc
        s += tl.sum(q * se_i[None, :], axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh).to(tl.float32)
        s = tl.where(valid, s, float("-inf"))
        scores = tl.where((js == i)[None, :], s[:, None], scores)
        valid_count += valid.to(tl.int32)

    max_score = tl.max(scores, axis=1)
    all_invalid = max_score == float("-inf")
    safe_max = tl.where(all_invalid, 0.0, max_score)
    exp_scores = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
    exp_scores = tl.where((js < J_VAL)[None, :], exp_scores, 0.0)
    denom = tl.sum(exp_scores, axis=1)
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    probs = exp_scores / safe_denom[:, None]
    lse = tl.where(
        all_invalid,
        0.0,
        safe_max + tl.log2(safe_denom) * 0.6931471805599453,
    )

    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # Small offsets are passthrough values.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)
        vt = _decode_q6_g128_token_rows(
            V_PAYLOAD,
            V_SCALES,
            kv_bh,
            kp,
            valid,
            ds,
            dm,
            PAIR_GROUPS=PAIR_GROUPS,
            PAYLOAD_BYTES=PAYLOAD_BYTES,
            HD=HD,
        )
        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt

    # Large offsets receive the exact R=4 sequential/disjoint Givens MOVT used by
    # the trainer clone.  phase_base/phase_gain are already phase_gate-gated.
    for slot in range(J_LARGE_VAL):
        i = J_SMALL_VAL + slot
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)
        vt = _decode_q6_g128_token_rows(
            V_PAYLOAD,
            V_SCALES,
            kv_bh,
            kp,
            valid,
            ds,
            dm,
            PAIR_GROUPS=PAIR_GROUPS,
            PAYLOAD_BYTES=PAYLOAD_BYTES,
            HD=HD,
        )
        vt_rot = vt
        for r in range(R_PLANES_VAL):
            ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
            ch_b = ch_a + 1
            mask_a = ds == ch_a
            mask_b = ds == ch_b

            y_r = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0).to(tl.float32)
            z_r = tl.load(zb + tl.maximum(kp, 0) * stride_zn + r, mask=valid, other=0.0).to(tl.float32)
            pb_r = tl.load(PHASE_BASE + slot * stride_phi + h * stride_phh + r).to(tl.float32)
            pg_r = tl.load(PHASE_GAIN + slot * stride_pgi + h * stride_pgh + r).to(tl.float32)
            theta = pb_r + pg_r * y_r * z_r
            theta = tl.where(valid, theta, 0.0)
            cos_t = tl.cos(theta)
            sin_t = tl.sin(theta)

            v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
            v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
            vt_rot = tl.where(
                mask_a[None, :],
                (cos_t * v_a - sin_t * v_b)[:, None],
                vt_rot,
            )
            vt_rot = tl.where(
                mask_b[None, :],
                (sin_t * v_a + cos_t * v_b)[:, None],
                vt_rot,
            )

        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt_rot

    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(
        ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=nm[:, None] & dm[None, :],
    )
    tl.store(lb + ns * stride_ln, lse, mask=nm)

    # Causal validity is independent of B/H.  Only bh=0 writes the diagnostic.
    tl.store(VALID_COUNTS + ns, valid_count, mask=nm & (bh == 0))


@triton.jit
def _q6_g128_dsqg_direct_consume_pair_q_kernel(
    Q,
    K_PAYLOAD,
    K_SCALES,
    V_PAYLOAD,
    V_SCALES,
    POS_BIAS,
    SCALE_EMBED,
    PHASE_BASE,
    PHASE_GAIN,
    Y_PRE,
    Z_PRE,
    OFFSETS,
    OUT,
    LSE,
    VALID_COUNTS,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ln,
    stride_pbi,
    stride_pbh,
    stride_sei,
    stride_sed,
    stride_phi,
    stride_phh,
    stride_pgi,
    stride_pgh,
    stride_yb,
    stride_yh,
    stride_yn,
    stride_zb,
    stride_zh,
    stride_zn,
    N,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HD: tl.constexpr,
    PAIR_GROUPS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr,
    J_SMALL_VAL: tl.constexpr,
    J_LARGE_VAL: tl.constexpr,
    J_PAD: tl.constexpr,
    R_PLANES_VAL: tl.constexpr,
    PLANE_SHIFT: tl.constexpr,
):
    """Stage-F.2 forward slice: one program computes two Q heads sharing one KV head.

    This reuses q6 K/V decode within the fused forward for KV_GROUP=2. Backward
    still uses the Stage-F.1 per-query-head fused core.
    """
    bh_kv = tl.program_id(0)
    block_n = tl.program_id(1)
    b = bh_kv // H_KV
    kv_h = bh_kv % H_KV
    h0 = kv_h * 2
    h1 = h0 + 1

    ns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb0 = Q + b * stride_qb + h0 * stride_qh
    qb1 = Q + b * stride_qb + h1 * stride_qh
    yb0 = Y_PRE + b * stride_yb + h0 * stride_yh
    yb1 = Y_PRE + b * stride_yb + h1 * stride_yh
    zb0 = Z_PRE + b * stride_zb + h0 * stride_zh
    zb1 = Z_PRE + b * stride_zb + h1 * stride_zh

    q0 = tl.load(qb0 + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    q1 = tl.load(qb1 + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    scores0 = tl.full([BLOCK_N, J_PAD], float("-inf"), tl.float32)
    scores1 = tl.full([BLOCK_N, J_PAD], float("-inf"), tl.float32)
    valid_count = tl.zeros([BLOCK_N], tl.int32)

    for i in range(J_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)
        kt = _decode_q6_g128_token_rows(
            K_PAYLOAD, K_SCALES, bh_kv, kp, valid, ds, dm,
            PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
        )
        se_i = tl.load(SCALE_EMBED + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.float32)
        s0 = tl.sum(q0 * kt, axis=1) * sc
        s0 += tl.sum(q0 * se_i[None, :], axis=1) * sc
        s0 += tl.load(POS_BIAS + i * stride_pbi + h0 * stride_pbh).to(tl.float32)
        s0 = tl.where(valid, s0, float("-inf"))
        scores0 = tl.where((js == i)[None, :], s0[:, None], scores0)
        s1 = tl.sum(q1 * kt, axis=1) * sc
        s1 += tl.sum(q1 * se_i[None, :], axis=1) * sc
        s1 += tl.load(POS_BIAS + i * stride_pbi + h1 * stride_pbh).to(tl.float32)
        s1 = tl.where(valid, s1, float("-inf"))
        scores1 = tl.where((js == i)[None, :], s1[:, None], scores1)
        valid_count += valid.to(tl.int32)

    max0 = tl.max(scores0, axis=1)
    max1 = tl.max(scores1, axis=1)
    invalid0 = max0 == float("-inf")
    invalid1 = max1 == float("-inf")
    safe0 = tl.where(invalid0, 0.0, max0)
    safe1 = tl.where(invalid1, 0.0, max1)
    exp0 = tl.exp2((scores0 - safe0[:, None]) * 1.4426950408889634)
    exp1 = tl.exp2((scores1 - safe1[:, None]) * 1.4426950408889634)
    exp0 = tl.where((js < J_VAL)[None, :], exp0, 0.0)
    exp1 = tl.where((js < J_VAL)[None, :], exp1, 0.0)
    den0 = tl.sum(exp0, axis=1)
    den1 = tl.sum(exp1, axis=1)
    safe_den0 = tl.where(den0 > 0.0, den0, 1.0)
    safe_den1 = tl.where(den1 > 0.0, den1, 1.0)
    probs0 = exp0 / safe_den0[:, None]
    probs1 = exp1 / safe_den1[:, None]
    lse0 = tl.where(invalid0, 0.0, safe0 + tl.log2(safe_den0) * 0.6931471805599453)
    lse1 = tl.where(invalid1, 0.0, safe1 + tl.log2(safe_den1) * 0.6931471805599453)

    acc0 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    acc1 = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)
        vt = _decode_q6_g128_token_rows(
            V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
            PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
        )
        p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
        p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
        acc0 += p0[:, None] * vt
        acc1 += p1[:, None] * vt

    for slot in range(J_LARGE_VAL):
        i = J_SMALL_VAL + slot
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns.to(tl.int32) - delta
        valid = nm & (kp >= 0) & (kp < N)
        vt = _decode_q6_g128_token_rows(
            V_PAYLOAD, V_SCALES, bh_kv, kp, valid, ds, dm,
            PAIR_GROUPS=PAIR_GROUPS, PAYLOAD_BYTES=PAYLOAD_BYTES, HD=HD,
        )
        vt0 = vt
        vt1 = vt
        for r in range(R_PLANES_VAL):
            ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
            ch_b = ch_a + 1
            mask_a = ds == ch_a
            mask_b = ds == ch_b
            z_idx = tl.maximum(kp, 0)

            y0 = tl.load(yb0 + ns * stride_yn + r, mask=nm, other=0.0).to(tl.float32)
            z0 = tl.load(zb0 + z_idx * stride_zn + r, mask=valid, other=0.0).to(tl.float32)
            pb0 = tl.load(PHASE_BASE + slot * stride_phi + h0 * stride_phh + r).to(tl.float32)
            pg0 = tl.load(PHASE_GAIN + slot * stride_pgi + h0 * stride_pgh + r).to(tl.float32)
            theta0 = tl.where(valid, pb0 + pg0 * y0 * z0, 0.0)
            c0 = tl.cos(theta0)
            s0 = tl.sin(theta0)
            va = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
            vb = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
            vt0 = tl.where(mask_a[None, :], (c0 * va - s0 * vb)[:, None], vt0)
            vt0 = tl.where(mask_b[None, :], (s0 * va + c0 * vb)[:, None], vt0)

            y1 = tl.load(yb1 + ns * stride_yn + r, mask=nm, other=0.0).to(tl.float32)
            z1 = tl.load(zb1 + z_idx * stride_zn + r, mask=valid, other=0.0).to(tl.float32)
            pb1 = tl.load(PHASE_BASE + slot * stride_phi + h1 * stride_phh + r).to(tl.float32)
            pg1 = tl.load(PHASE_GAIN + slot * stride_pgi + h1 * stride_pgh + r).to(tl.float32)
            theta1 = tl.where(valid, pb1 + pg1 * y1 * z1, 0.0)
            c1 = tl.cos(theta1)
            s1 = tl.sin(theta1)
            vt1 = tl.where(mask_a[None, :], (c1 * va - s1 * vb)[:, None], vt1)
            vt1 = tl.where(mask_b[None, :], (s1 * va + c1 * vb)[:, None], vt1)

        p0 = tl.sum(probs0 * (js == i)[None, :].to(tl.float32), axis=1)
        p1 = tl.sum(probs1 * (js == i)[None, :].to(tl.float32), axis=1)
        acc0 += p0[:, None] * vt0
        acc1 += p1[:, None] * vt1

    ob0 = OUT + b * stride_ob + h0 * stride_oh
    ob1 = OUT + b * stride_ob + h1 * stride_oh
    lb0 = LSE + b * stride_lb + h0 * stride_lh
    lb1 = LSE + b * stride_lb + h1 * stride_lh
    tl.store(ob0 + ns[:, None] * stride_on + ds[None, :] * stride_od, acc0.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(ob1 + ns[:, None] * stride_on + ds[None, :] * stride_od, acc1.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(lb0 + ns * stride_ln, lse0, mask=nm)
    tl.store(lb1 + ns * stride_ln, lse1, mask=nm)
    tl.store(VALID_COUNTS + ns, valid_count, mask=nm & (bh_kv == 0))


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _require_layout_pair(k_layout: Any, v_layout: Any) -> tuple[int, int, int, int]:
    direct_mod._require_cuda_layout(k_layout)
    direct_mod._require_cuda_layout(v_layout)
    k_shape = (k_layout.batch, k_layout.heads, k_layout.seq_len, k_layout.head_dim)
    v_shape = (v_layout.batch, v_layout.heads, v_layout.seq_len, v_layout.head_dim)
    if k_shape != v_shape:
        raise ValueError(f"K/V q6 layouts differ: {k_shape} vs {v_shape}")
    if k_layout.pair_groups != v_layout.pair_groups:
        raise ValueError("K/V q6 pair-group counts differ")
    if k_layout.head_dim != HEAD_DIM:
        raise ValueError(f"fused q6 direct-consume requires D={HEAD_DIM}")
    return k_shape


def triton_q6_g128_dsqg_direct_consume(
    q: torch.Tensor,
    k_layout: Any,
    v_layout: Any,
    offsets_dev: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    gated_phase_base: torch.Tensor,
    gated_phase_gain: torch.Tensor,
    y_pre: torch.Tensor,
    z_pre: torch.Tensor,
    *,
    j_small: int,
    plane_shift: int,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    return_report: bool = False,
    pair_q_heads: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
]:
    """Run full q6 decode + DSQG consume without a gathered K/V allocation.

    Returns pre-if_gain BF16 attention output, FP32 LSE, and per-query valid
    offset counts.  if_gain remains outside this boundary to preserve the
    trainer's BF16-output-then-FP32-gain operation order.
    """
    b, h_kv, n, d = _require_layout_pair(k_layout, v_layout)
    if q.ndim != 4:
        raise ValueError(f"q must be [B,Hq,N,D], got {tuple(q.shape)}")
    bq, h_q, nq, dq = (int(v) for v in q.shape)
    if (bq, nq, dq) != (b, n, d):
        raise ValueError(f"q shape {tuple(q.shape)} is incompatible with K/V layout {(b, h_kv, n, d)}")
    if h_q % h_kv != 0:
        raise ValueError(f"query heads Hq={h_q} must be divisible by KV heads Hkv={h_kv}")
    kv_group = h_q // h_kv
    _require_tensor("q", q, shape=(b, h_q, n, d), dtype=torch.bfloat16)
    if not q.is_contiguous():
        q = q.contiguous()

    if offsets_dev.dtype != torch.int32 or offsets_dev.ndim != 1 or not offsets_dev.is_cuda:
        raise TypeError("offsets_dev must be contiguous CUDA int32 [J]")
    offsets_dev = offsets_dev.contiguous()
    j_val = int(offsets_dev.numel())
    j_small = int(j_small)
    j_large = j_val - j_small
    if j_val <= 0:
        raise ValueError("offsets must be non-empty")
    if not (0 <= j_small <= j_val):
        raise ValueError(f"j_small={j_small} must be in [0,{j_val}]")
    if j_val > 64:
        raise ValueError("Stage-A fused direct-consume currently caps J at 64")
    if block_n not in (16, 32, 64, 128):
        raise ValueError("block_n must be one of 16,32,64,128")
    if plane_shift < 0 or plane_shift > (d // R_PLANES) - 2:
        raise ValueError(f"plane_shift={plane_shift} does not fit R={R_PLANES}, D={d}")

    _require_tensor("pos_bias", pos_bias, shape=(j_val, h_q), dtype=torch.float32)
    _require_tensor("scale_embed", scale_embed, shape=(j_val, d), dtype=torch.float32)
    phase_rows = max(j_large, 1)
    _require_tensor(
        "gated_phase_base", gated_phase_base, shape=(phase_rows, h_q, R_PLANES), dtype=torch.float32
    )
    _require_tensor(
        "gated_phase_gain", gated_phase_gain, shape=(phase_rows, h_q, R_PLANES), dtype=torch.float32
    )
    _require_tensor("y_pre", y_pre, shape=(b, h_q, n, R_PLANES), dtype=torch.float32)
    _require_tensor("z_pre", z_pre, shape=(b, h_q, n, R_PLANES), dtype=torch.float32)

    pos_bias = pos_bias.contiguous()
    scale_embed = scale_embed.contiguous()
    gated_phase_base = gated_phase_base.contiguous()
    gated_phase_gain = gated_phase_gain.contiguous()
    y_pre = y_pre.contiguous()
    z_pre = z_pre.contiguous()

    out = torch.empty((b, h_q, n, d), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, h_q, n), device=q.device, dtype=torch.float32)
    valid_counts = torch.empty((n,), device=q.device, dtype=torch.int32)

    block_hd = _next_pow2(d)
    j_pad = max(16, _next_pow2(j_val))
    use_pair_q = bool(pair_q_heads and kv_group == 2 and h_q == h_kv * 2)
    if use_pair_q:
        grid = (b * h_kv, triton.cdiv(n, block_n))
        _q6_g128_dsqg_direct_consume_pair_q_kernel[grid](
            q,
            k_layout.payload,
            k_layout.scales,
            v_layout.payload,
            v_layout.scales,
            pos_bias,
            scale_embed,
            gated_phase_base,
            gated_phase_gain,
            y_pre,
            z_pre,
            offsets_dev,
            out,
            lse,
            valid_counts,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            gated_phase_base.stride(0), gated_phase_base.stride(1),
            gated_phase_gain.stride(0), gated_phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            N=n,
            H_Q=h_q,
            H_KV=h_kv,
            HD=d,
            PAIR_GROUPS=k_layout.pair_groups,
            PAYLOAD_BYTES=PAYLOAD_BYTES_PER_PAIR,
            BLOCK_N=block_n,
            BLOCK_HD=block_hd,
            J_VAL=j_val,
            J_SMALL_VAL=j_small,
            J_LARGE_VAL=j_large,
            J_PAD=j_pad,
            R_PLANES_VAL=R_PLANES,
            PLANE_SHIFT=int(plane_shift),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    else:
        grid = (b * h_q, triton.cdiv(n, block_n))
        _q6_g128_dsqg_direct_consume_kernel[grid](
            q,
            k_layout.payload,
            k_layout.scales,
            v_layout.payload,
            v_layout.scales,
            pos_bias,
            scale_embed,
            gated_phase_base,
            gated_phase_gain,
            y_pre,
            z_pre,
            offsets_dev,
            out,
            lse,
            valid_counts,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            pos_bias.stride(0), pos_bias.stride(1),
            scale_embed.stride(0), scale_embed.stride(1),
            gated_phase_base.stride(0), gated_phase_base.stride(1),
            gated_phase_gain.stride(0), gated_phase_gain.stride(1),
            y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
            z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
            N=n,
            H_Q=h_q,
            H_KV=h_kv,
            KV_GROUP=kv_group,
            HD=d,
            PAIR_GROUPS=k_layout.pair_groups,
            PAYLOAD_BYTES=PAYLOAD_BYTES_PER_PAIR,
            BLOCK_N=block_n,
            BLOCK_HD=block_hd,
            J_VAL=j_val,
            J_SMALL_VAL=j_small,
            J_LARGE_VAL=j_large,
            J_PAD=j_pad,
            R_PLANES_VAL=R_PLANES,
            PLANE_SHIFT=int(plane_shift),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )

    if not return_report:
        return out, lse, valid_counts

    k_storage = k_layout.storage_report()
    v_storage = v_layout.storage_report()
    one_gather_bytes = b * h_q * n * j_val * d * torch.tensor([], dtype=torch.bfloat16).element_size()
    report = {
        "read_implementation": "q6_triton_fused_direct_consume",
        "scratch_mode": "direct_q6_decode_to_dsqg_output",
        "softmax_mode": "register_resident_offline_j",
        "head_dim": d,
        "j_val": j_val,
        "j_small": j_small,
        "j_large": j_large,
        "r_planes": R_PLANES,
        "plane_shift": int(plane_shift),
        "block_n": int(block_n),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
        "resident_q6_bytes": int(k_storage["total_bytes"] + v_storage["total_bytes"]),
        "resident_q6_compression_vs_bf16": float((2 * b * h_q * n * d * torch.tensor([], dtype=torch.bfloat16).element_size()) / max(1, int(k_storage["total_bytes"] + v_storage["total_bytes"]))),
        "num_query_heads": int(h_q),
        "num_kv_heads": int(h_kv),
        "kv_group_size": int(kv_group),
        "pair_q_forward_enabled": bool(use_pair_q),
        "pair_q_forward_core": "triton_pair_query_head_forward_reuse" if use_pair_q else "disabled",
        "attention_output_shape": "x".join(str(v) for v in out.shape),
        "attention_output_bytes": int(out.numel() * out.element_size()),
        "lse_bytes": int(lse.numel() * lse.element_size()),
        "valid_count_bytes": int(valid_counts.numel() * valid_counts.element_size()),
        "materialized_k_gather_bytes": 0,
        "materialized_v_gather_bytes": 0,
        "materialized_gather_bytes": 0,
        "oracle_one_gather_bytes": int(one_gather_bytes),
        "oracle_kv_gather_bytes": int(2 * one_gather_bytes),
        "avoided_kv_gather_bytes": int(2 * one_gather_bytes),
        "includes_movt": True,
        "includes_scale_embed": True,
        "includes_pos_bias": True,
        "includes_causal_mask": True,
        "includes_if_gain": False,
        "includes_out_proj_gate_dropout": False,
        "includes_backward": False,
    }
    return out, lse, valid_counts, report


def _rms_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)


def _make_phase_probes(device: torch.device, plane_shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.zeros((R_PLANES, HEAD_DIM), device=device, dtype=torch.float32)
    k = torch.zeros_like(q)
    segment = HEAD_DIM // R_PLANES
    for r in range(R_PLANES):
        a = r * segment + plane_shift
        b = a + 1
        angle = 2.0 * math.pi * r / R_PLANES
        q[r, a] = math.cos(angle)
        q[r, b] = math.sin(angle)
        k[r, a] = math.cos(angle + math.pi / 2.0)
        k[r, b] = math.sin(angle + math.pi / 2.0)
    return q, k


def _rotate_sparse_values_oracle(
    values: torch.Tensor,
    y_pre: torch.Tensor,
    z_pre: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    gated_phase_base: torch.Tensor,
    gated_phase_gain: torch.Tensor,
    *,
    j_small: int,
    plane_shift: int,
) -> torch.Tensor:
    j_val = int(values.shape[3])
    if j_small >= j_val:
        return values
    out = values.clone()
    segment = HEAD_DIM // R_PLANES
    for i in range(j_small, j_val):
        slot = i - j_small
        kp = idx[:, i]
        valid_i = valid[:, i]
        for r in range(R_PLANES):
            ch_a = r * segment + plane_shift
            ch_b = ch_a + 1
            z_i = z_pre[:, :, kp, r]
            theta = (
                gated_phase_base[slot, :, r].reshape(1, -1, 1)
                + gated_phase_gain[slot, :, r].reshape(1, -1, 1)
                * y_pre[:, :, :, r]
                * z_i
            )
            theta = torch.where(valid_i.reshape(1, 1, -1), theta, torch.zeros_like(theta))
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            old_a = out[:, :, :, i, ch_a].clone()
            old_b = out[:, :, :, i, ch_b].clone()
            out[:, :, :, i, ch_a] = cos_t * old_a - sin_t * old_b
            out[:, :, :, i, ch_b] = sin_t * old_a + cos_t * old_b
    return out


def pytorch_dsqg_consumer_oracle(
    q: torch.Tensor,
    k_gather: torch.Tensor,
    v_gather: torch.Tensor,
    valid: torch.Tensor,
    offsets: list[int],
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    gated_phase_base: torch.Tensor,
    gated_phase_gain: torch.Tensor,
    y_pre: torch.Tensor,
    z_pre: torch.Tensor,
    *,
    j_small: int,
    plane_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current trainer-clone PyTorch DSQG consume semantics, pre-if_gain."""
    b, h, n, d = q.shape
    idx, expected_valid = layout_mod.causal_offset_index(n, offsets, device=q.device)
    if not torch.equal(valid, expected_valid):
        raise RuntimeError("oracle gather mask differs from causal_offset_index")

    qf = q.float()
    sc = 1.0 / math.sqrt(float(d))
    scores = torch.einsum("bhnd,bhnjd->bhnj", qf, k_gather.float()) * sc
    scores = scores + torch.einsum("bhnd,jd->bhnj", qf, scale_embed.float()) * sc
    scores = scores + pos_bias.float().transpose(0, 1).reshape(1, h, 1, len(offsets))
    mask = valid.reshape(1, 1, n, len(offsets))
    scores = scores.masked_fill(~mask, float("-inf"))

    max_scores = scores.amax(dim=-1, keepdim=True)
    all_invalid = ~torch.isfinite(max_scores)
    safe_max = torch.where(all_invalid, torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - safe_max).masked_fill(~mask, 0.0)
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = exp_scores / denom
    lse = torch.where(
        all_invalid,
        torch.zeros_like(max_scores),
        safe_max + torch.log(denom),
    ).squeeze(-1)

    v_rot = _rotate_sparse_values_oracle(
        v_gather.float(),
        y_pre,
        z_pre,
        idx,
        valid,
        gated_phase_base,
        gated_phase_gain,
        j_small=j_small,
        plane_shift=plane_shift,
    )
    out = torch.sum(probs.unsqueeze(-1) * v_rot, dim=3).to(dtype=q.dtype)
    return out, lse


class ProbeInputs:
    def __init__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_layout: Any,
        v_layout: Any,
        offsets: list[int],
        offsets_dev: torch.Tensor,
        pos_bias: torch.Tensor,
        scale_embed: torch.Tensor,
        gated_phase_base: torch.Tensor,
        gated_phase_gain: torch.Tensor,
        y_pre: torch.Tensor,
        z_pre: torch.Tensor,
        if_gain: torch.Tensor,
        j_small: int,
        plane_shift: int,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.k_layout = k_layout
        self.v_layout = v_layout
        self.offsets = offsets
        self.offsets_dev = offsets_dev
        self.pos_bias = pos_bias
        self.scale_embed = scale_embed
        self.gated_phase_base = gated_phase_base
        self.gated_phase_gain = gated_phase_gain
        self.y_pre = y_pre
        self.z_pre = z_pre
        self.if_gain = if_gain
        self.j_small = int(j_small)
        self.plane_shift = int(plane_shift)


def build_probe_inputs(
    shape: tuple[int, int, int, int],
    workload: str,
    offsets: list[int],
    *,
    j_small: int,
    plane_shift: int,
    device: torch.device,
    seed: int,
) -> ProbeInputs:
    b, h, n, d = shape
    if d != HEAD_DIM:
        raise ValueError(f"probe requires D={HEAD_DIM}, got {d}")
    if not (0 <= j_small <= len(offsets)):
        raise ValueError("invalid j_small")
    j_large = len(offsets) - j_small

    # Match trainer projection precision: q/k/v are BF16 before q6 packing.
    q = layout_mod.make_workload(shape, workload, device, seed).to(torch.bfloat16).contiguous()
    k = layout_mod.make_workload(shape, workload, device, seed + 1).to(torch.bfloat16).contiguous()
    v = layout_mod.make_workload(shape, workload, device, seed + 2).to(torch.bfloat16).contiguous()
    k_layout = layout_mod.pack_q6_g128_cache_layout(k, seed=seed + 101)
    v_layout = layout_mod.pack_q6_g128_cache_layout(v, seed=seed + 202)
    offsets_dev = torch.tensor(offsets, device=device, dtype=torch.int32)

    # Actual DSQG-like deterministic score parameters.
    alphas = torch.linspace(0.2, 2.0, h, device=device, dtype=torch.float32)
    delta_vals = torch.tensor(
        [math.log1p(int(delta)) for delta in offsets], device=device, dtype=torch.float32
    )
    pos_bias = (-delta_vals[:, None] * alphas[None, :]).contiguous()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 303)
    scale_embed = (
        0.15
        + 0.01
        * torch.randn((len(offsets), d), device=device, generator=gen, dtype=torch.float32)
    ).contiguous()
    phase_rows = max(j_large, 1)
    phase_base = (
        0.01
        * torch.randn((phase_rows, h, R_PLANES), device=device, generator=gen, dtype=torch.float32)
    )
    phase_gain = (
        0.001
        * torch.randn((phase_rows, h, R_PLANES), device=device, generator=gen, dtype=torch.float32)
    )
    phase_gate = torch.randn((phase_rows,), device=device, generator=gen, dtype=torch.float32) * 0.1
    phase_gate = torch.sigmoid(phase_gate)[:, None, None]
    gated_phase_base = (phase_base * phase_gate).contiguous()
    gated_phase_gain = (phase_gain * phase_gate).contiguous()

    query_probes, key_probes = _make_phase_probes(device, plane_shift)
    probe_scale = 1.0 / math.sqrt(float(d))
    y_pre = (
        torch.einsum("bhnd,rd->bhnr", _rms_normalize_last(q), query_probes) * probe_scale
    ).contiguous()
    z_pre = (
        torch.einsum("bhnd,rd->bhnr", _rms_normalize_last(k), key_probes) * probe_scale
    ).contiguous()
    if_gain = (
        1.0 + 0.05 * torch.randn((h,), device=device, generator=gen, dtype=torch.float32)
    ).contiguous()

    return ProbeInputs(
        q=q,
        k=k,
        v=v,
        k_layout=k_layout,
        v_layout=v_layout,
        offsets=offsets,
        offsets_dev=offsets_dev,
        pos_bias=pos_bias,
        scale_embed=scale_embed,
        gated_phase_base=gated_phase_base,
        gated_phase_gain=gated_phase_gain,
        y_pre=y_pre,
        z_pre=z_pre,
        if_gain=if_gain,
        j_small=j_small,
        plane_shift=plane_shift,
    )


def direct_gather_pytorch_oracle(inputs: ProbeInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_g, valid_k = direct_mod.triton_direct_decode_gather(inputs.k_layout, inputs.offsets)
    v_g, valid_v = direct_mod.triton_direct_decode_gather(inputs.v_layout, inputs.offsets)
    if not torch.equal(valid_k, valid_v):
        raise RuntimeError("direct-gather K/V masks differ")
    out, lse = pytorch_dsqg_consumer_oracle(
        inputs.q,
        k_g,
        v_g,
        valid_k,
        inputs.offsets,
        inputs.pos_bias,
        inputs.scale_embed,
        inputs.gated_phase_base,
        inputs.gated_phase_gain,
        inputs.y_pre,
        inputs.z_pre,
        j_small=inputs.j_small,
        plane_shift=inputs.plane_shift,
    )
    return out, lse, valid_k


def full_scratch_pytorch_oracle(inputs: ProbeInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_g, valid_k = layout_mod.full_decode_scratch_then_gather(inputs.k_layout, inputs.offsets)
    v_g, valid_v = layout_mod.full_decode_scratch_then_gather(inputs.v_layout, inputs.offsets)
    if not torch.equal(valid_k, valid_v):
        raise RuntimeError("full-scratch K/V masks differ")
    out, lse = pytorch_dsqg_consumer_oracle(
        inputs.q,
        k_g,
        v_g,
        valid_k,
        inputs.offsets,
        inputs.pos_bias,
        inputs.scale_embed,
        inputs.gated_phase_base,
        inputs.gated_phase_gain,
        inputs.y_pre,
        inputs.z_pre,
        j_small=inputs.j_small,
        plane_shift=inputs.plane_shift,
    )
    return out, lse, valid_k


def fused_direct_consume(inputs: ProbeInputs, *, block_n: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
]:
    return triton_q6_g128_dsqg_direct_consume(
        inputs.q,
        inputs.k_layout,
        inputs.v_layout,
        inputs.offsets_dev,
        inputs.pos_bias,
        inputs.scale_embed,
        inputs.gated_phase_base,
        inputs.gated_phase_gain,
        inputs.y_pre,
        inputs.z_pre,
        j_small=inputs.j_small,
        plane_shift=inputs.plane_shift,
        block_n=block_n,
        return_report=True,
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _error_metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    diff = actual.float() - expected.float()
    denom = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return {
        "max_abs_diff": float(diff.abs().max().detach().cpu()),
        "mean_abs_diff": float(diff.abs().mean().detach().cpu()),
        "relative_rms_diff": float((diff.square().mean().sqrt() / denom).detach().cpu()),
        "bitwise_equal": bool(torch.equal(expected, actual)),
        "exact_element_fraction": float((expected == actual).float().mean().detach().cpu()),
    }


def _time_cuda_us(device: torch.device, repeats: int, fn: Callable[[], Any]) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def _peak_allocated_delta_bytes(device: torch.device, fn: Callable[[], Any]) -> tuple[int, Any]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = int(torch.cuda.memory_allocated(device))
    result = fn()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    return max(0, peak - before), result


def run_case(
    *,
    shape: tuple[int, int, int, int],
    workload: str,
    offsets: list[int],
    j_small: int,
    plane_shift: int,
    device: torch.device,
    seed: int,
    block_n: int,
    repeats: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    inputs = build_probe_inputs(
        shape,
        workload,
        offsets,
        j_small=j_small,
        plane_shift=plane_shift,
        device=device,
        seed=seed,
    )

    # Warm compilation before memory/timing measurements.
    fused_out, fused_lse, fused_counts, fused_report = fused_direct_consume(
        inputs, block_n=block_n
    )
    direct_out, direct_lse, direct_valid = direct_gather_pytorch_oracle(inputs)
    full_out, full_lse, full_valid = full_scratch_pytorch_oracle(inputs)

    if not torch.equal(direct_valid, full_valid):
        raise RuntimeError("direct-gather and full-scratch causal masks differ")
    direct_full_out = _error_metrics(full_out, direct_out)
    direct_full_lse = _error_metrics(full_lse, direct_lse)
    if not direct_full_out["bitwise_equal"]:
        raise RuntimeError(
            "existing q6 direct-gather oracle diverged from full-scratch oracle: "
            f"max_diff={direct_full_out['max_abs_diff']}"
        )

    expected_counts = direct_valid.sum(dim=1, dtype=torch.int32)
    causal_parity = bool(torch.equal(fused_counts, expected_counts))
    fused_err = _error_metrics(direct_out, fused_out)
    fused_lse_err = _error_metrics(direct_lse, fused_lse)

    # Post-if_gain remains outside the kernel, but verify that retaining that
    # boundary does not change parity accounting.
    gain = inputs.if_gain.reshape(1, shape[1], 1, 1)
    direct_post_gain = direct_out * gain
    fused_post_gain = fused_out * gain
    post_gain_err = _error_metrics(direct_post_gain, fused_post_gain)

    fused_us = _time_cuda_us(
        device,
        repeats,
        lambda: fused_direct_consume(inputs, block_n=block_n),
    )
    oracle_us = _time_cuda_us(device, repeats, lambda: direct_gather_pytorch_oracle(inputs))

    fused_peak, _ = _peak_allocated_delta_bytes(
        device, lambda: fused_direct_consume(inputs, block_n=block_n)
    )
    oracle_peak, _ = _peak_allocated_delta_bytes(device, lambda: direct_gather_pytorch_oracle(inputs))

    allowed = bool(torch.allclose(fused_out.float(), direct_out.float(), atol=atol, rtol=rtol))
    no_gather_allocation_observed = fused_peak < int(fused_report["oracle_kv_gather_bytes"])
    finite = bool(torch.isfinite(fused_out.float()).all().item()) and bool(
        torch.isfinite(fused_lse).all().item()
    )

    row = {
        "variant": "q6_g128_fused_direct_consume_stage_a",
        "shape": "x".join(str(v) for v in shape),
        "workload": workload,
        "device": str(device),
        "offsets": ",".join(str(v) for v in offsets),
        "offset_count": len(offsets),
        "j_small": int(j_small),
        "j_large": len(offsets) - int(j_small),
        "seed": int(seed),
        "finite": finite,
        "causal_valid_count_parity": causal_parity,
        "semantic_allclose": allowed,
        "atol": float(atol),
        "rtol": float(rtol),
        "fused_output_max_abs_diff": fused_err["max_abs_diff"],
        "fused_output_mean_abs_diff": fused_err["mean_abs_diff"],
        "fused_output_relative_rms_diff": fused_err["relative_rms_diff"],
        "fused_output_bitwise_equal": fused_err["bitwise_equal"],
        "fused_output_exact_element_fraction": fused_err["exact_element_fraction"],
        "fused_lse_max_abs_diff": fused_lse_err["max_abs_diff"],
        "post_if_gain_max_abs_diff": post_gain_err["max_abs_diff"],
        "direct_vs_full_output_max_abs_diff": direct_full_out["max_abs_diff"],
        "direct_vs_full_lse_max_abs_diff": direct_full_lse["max_abs_diff"],
        "fused_us": fused_us,
        "direct_gather_pytorch_oracle_us": oracle_us,
        "fused_vs_oracle_time_ratio": fused_us / oracle_us if oracle_us > 0 else None,
        "fused_cuda_peak_allocated_delta_bytes": int(fused_peak),
        "oracle_cuda_peak_allocated_delta_bytes": int(oracle_peak),
        "fused_vs_oracle_peak_ratio": fused_peak / oracle_peak if oracle_peak > 0 else None,
        "no_gather_allocation_observed": no_gather_allocation_observed,
        "fused_peak_vs_oracle_kv_gather_bytes": (
            fused_peak / fused_report["oracle_kv_gather_bytes"]
            if fused_report["oracle_kv_gather_bytes"] > 0
            else None
        ),
        "pre_if_gain_output_bytes": _tensor_bytes(fused_out),
        "post_if_gain_output_bytes": _tensor_bytes(fused_post_gain),
        **fused_report,
    }
    return row


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(v) for v in text.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be B,H,N,D")
    return parts  # type: ignore[return-value]


def parse_offsets(text: str) -> list[int]:
    offsets = [int(v) for v in text.split(",") if v.strip()]
    if not offsets or any(v < 0 for v in offsets):
        raise argparse.ArgumentTypeError("offsets must be non-empty nonnegative integers")
    return offsets


def write_rows(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n"
    )
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_prefix.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument(
        "--offsets",
        type=parse_offsets,
        default=list(DEFAULT_GROUP_A_OFFSETS),
    )
    parser.add_argument("--j-small", type=int, default=DEFAULT_GROUP_A_J_SMALL)
    parser.add_argument("--plane-shift", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    # Arithmetic order differs between Triton reductions/transcendentals and the
    # eager PyTorch oracle.  Report bitwise equality separately; this tolerance is
    # only the Stage-A acceptance gate and must not be silently loosened.
    parser.add_argument("--atol", type=float, default=0.015625)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise SystemExit("q6_g128 fused direct-consume Stage-A probe requires CUDA")
    if not args.shape:
        # Odd N catches token-pair boundary mistakes; N=2048 is trainer-shaped.
        args.shape = [(1, 2, 257, 64), (1, 8, 2048, 64)]
    if not args.workload:
        args.workload = ["gaussian", "adversarial"]
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    rows: list[dict[str, Any]] = []
    for shape in args.shape:
        for workload in args.workload:
            row = run_case(
                shape=shape,
                workload=workload,
                offsets=args.offsets,
                j_small=args.j_small,
                plane_shift=args.plane_shift,
                device=device,
                seed=args.seed,
                block_n=args.block_n,
                repeats=args.repeats,
                atol=args.atol,
                rtol=args.rtol,
            )
            rows.append(row)
            print(
                "[FUSED_Q6] "
                f"shape={row['shape']} workload={workload} "
                f"max_diff={row['fused_output_max_abs_diff']:.6g} "
                f"bitwise={row['fused_output_bitwise_equal']} "
                f"causal={row['causal_valid_count_parity']} "
                f"fused_us={row['fused_us']:.1f} oracle_us={row['direct_gather_pytorch_oracle_us']:.1f} "
                f"peak={row['fused_cuda_peak_allocated_delta_bytes'] / 1e6:.3f}MB "
                f"oracle_peak={row['oracle_cuda_peak_allocated_delta_bytes'] / 1e6:.3f}MB "
                f"gather_bytes={row['materialized_gather_bytes']}"
            )

    write_rows(rows, args.out_prefix)
    print(f"wrote {len(rows)} rows to {args.out_prefix}.json/.tsv")

    bad = [
        row
        for row in rows
        if (
            not row["finite"]
            or not row["causal_valid_count_parity"]
            or not row["semantic_allclose"]
            or not row["no_gather_allocation_observed"]
            or row["direct_vs_full_output_max_abs_diff"] != 0.0
            or (args.require_bitwise and not row["fused_output_bitwise_equal"])
        )
    ]
    if bad:
        raise SystemExit(f"Stage-A fused q6 direct-consume gate failed for {len(bad)} row(s)")


if __name__ == "__main__":
    main()
