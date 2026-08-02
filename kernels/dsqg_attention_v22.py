"""Dynamic Sparse Query-Gather (DSQG) attention V22.

V22 uses one online-softmax traversal over all positive causal offsets, native
grouped-query attention (GQA), support-aware projection cropping, centered FP32
scale embeddings, analytic positional bias, and the retained causal
Norm-Preserving Coupled Injection (NPCI) rotation. Backward computes its softmax
response mean directly instead of feeding a rounded output back into D.

``DSQGAttentionV19``, ``DSQGAttentionV20``, and ``DSQGAttentionV21`` are public
compatibility aliases for the canonical ``DSQGAttentionV22`` class.
"""


from __future__ import annotations

import math
import os

# Triton 3.5+ compatibility for module-scope constants referenced by @jit kernels.
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # CPU-only semantic-oracle/test installations
    _TRITON_AVAILABLE = False

    class _KernelStub:
        def __init__(self, function):
            self.function = function

        def __getitem__(self, _grid):
            raise RuntimeError("Triton is unavailable; use the eager DSQG oracle")

    class _JitStub:
        def __call__(self, function):
            return _KernelStub(function)

    class _TritonStub:
        jit = _JitStub()

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    class _TLStub:
        @staticmethod
        def constexpr(value=None):
            return value

    triton = _TritonStub()
    tl = _TLStub()

_LOG2E = tl.constexpr(1.4426950408889634)
NPCI_THETA_MAX = 0.25
NPCI_THETA_INIT = 0.01


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


ALL_OFFSETS = [
    1,2,3,4,5,6,7,8,9,10,13,15,16,19,21,23,28,
    48,64,96,121,161,192,212,245,273,295,342,375,384,
    413,441,473,512,549,579,593,631,653,694,716,768,
    826,846,900,936,970,1000,1024,1074,1108,1144,1166,
    1190,1218,1244,1288,1322,1385,1423,1451,1497,1522,
    1550,1581,1603,1617,1634,1651,1661,1710,1743,1780,
    1810,1820,1852,1860,1876,1886,1897,1903,1916,1926,
    1929,1941,1965,1983,2006,2011,2029,2037,2044,2068,
    2097,2113,2199,
]


def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _strict_int(name: str, value, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= {minimum}, got bool")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            f"{name} must be an integer >= {minimum}, got {type(value).__name__}"
        ) from exc
    if result < minimum:
        raise ValueError(f"{name}={result} must be >= {minimum}")
    return result


def _canonicalize_offsets(
    offsets: list[int] | tuple[int, ...],
) -> list[int]:
    """Validate and order positive unique causal offsets."""
    values: list[int] = []
    for offset in offsets:
        values.append(_strict_int("DSQG offset", offset, minimum=1))
    if not values:
        raise ValueError("DSQG requires at least one offset")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate DSQG offsets are not supported: {values}")

    return sorted(values)


def npci_rotate(
    x: torch.Tensor,
    x_delta: torch.Tensor,
    theta_h: torch.Tensor,
    *,
    strength_tau: float = 0.25,
) -> torch.Tensor:
    """Apply the retained magnitude-aware causal injection rotation."""
    tau = float(strength_tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("strength_tau must be finite and positive")
    original_dtype = x.dtype
    xf, df = x.float(), x_delta.float()
    theta = theta_h.float().reshape(1, -1, 1, 1)
    norm = xf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = xf / norm
    perpendicular = df - (df * unit).sum(dim=-1, keepdim=True) * unit
    perpendicular_norm = perpendicular.norm(dim=-1, keepdim=True)
    active = perpendicular_norm > norm * 1e-7
    direction = torch.where(
        active,
        perpendicular / perpendicular_norm.clamp_min(1e-20),
        torch.zeros_like(perpendicular),
    )
    strength = torch.tanh(perpendicular_norm / (tau * norm + 1e-12))
    angle = theta * strength
    rotated = torch.cos(angle) * xf + torch.sin(angle) * norm * direction
    return torch.where(active, rotated, xf).to(original_dtype)


def _raw_npci_theta_from_effective(theta: float) -> float:
    theta = float(theta)
    limit = float(NPCI_THETA_MAX)
    if not (0.0 <= abs(theta) < limit):
        raise ValueError(f"NPCI_THETA_INIT={theta} must satisfy abs(theta) < {limit}")
    return math.atanh(theta / limit)


# ===========================================================================
# Online Triton forward/backward
# ===========================================================================

@triton.jit
def _fwd_v22_online(
    Q, K, V, POS_BIAS, SCALE_EMBED, OUT, LSE, OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    """One online-softmax traversal over every configured causal offset."""
    bh = tl.program_id(0)
    block = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kv_head = head // KV_HEAD_GROUP_SIZE
    positions = QUERY_START + block * BLOCK_N + tl.arange(0, BLOCK_N)
    qmask = positions < N
    dims = tl.arange(0, BLOCK_HD)
    dmask = dims < HD
    scale = 1.0 / tl.sqrt(HD * 1.0)

    qbase = Q + batch * stride_qb + head * stride_qh
    kbase = K + batch * stride_kb + kv_head * stride_kh
    vbase = V + batch * stride_vb + kv_head * stride_vh
    query = tl.load(
        qbase + positions[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=qmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)

    running_max = tl.full([BLOCK_N], float('-inf'), tl.float32)
    running_sum = tl.zeros([BLOCK_N], tl.float32)
    accumulator = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for logical_index in range(J_VAL):
        delta = tl.load(OFFSETS + logical_index).to(tl.int32)
        source = positions - delta
        valid = qmask & (source >= 0) & (source < N)
        key = tl.load(
            kbase + source[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        virtual_key = tl.load(
            SCALE_EMBED + logical_index * stride_sei + dims * stride_sed,
            mask=dmask, other=0.0,
        ).to(tl.float32)
        score = tl.sum(query * (key + virtual_key[None, :]), axis=1) * scale
        score += tl.load(POS_BIAS + logical_index * stride_pbi + head * stride_pbh)
        score = tl.where(valid, score, float('-inf'))

        merged_max = tl.maximum(running_max, score)
        has_old = running_max > float('-inf')
        safe_max = tl.where(has_old | valid, merged_max, 0.0)
        old_scale = tl.where(
            has_old, tl.exp2((running_max - safe_max) * _LOG2E), 0.0
        )
        new_weight = tl.where(
            valid, tl.exp2((score - safe_max) * _LOG2E), 0.0
        )
        value = tl.load(
            vbase + source[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale[:, None] + new_weight[:, None] * value
        running_sum = running_sum * old_scale + new_weight
        running_max = tl.where(has_old | valid, merged_max, running_max)

    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    output = accumulator / denominator[:, None]
    lse = tl.where(
        running_sum > 0.0,
        running_max + tl.log2(running_sum) * (1.0 / _LOG2E),
        float('-inf'),
    )
    tl.store(
        OUT + batch * stride_ob + head * stride_oh
        + positions[:, None] * stride_on + dims[None, :] * stride_od,
        output, mask=qmask[:, None] & dmask[None, :],
    )
    tl.store(
        LSE + batch * stride_lb + head * stride_lh + positions * stride_ln,
        lse, mask=qmask,
    )


@triton.jit
def _bwd_dq_v22(
    Q, K, V, POS_BIAS, SCALE_EMBED, DO, LSE, DELTA,
    DQ, OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    """Compute exact D and dQ together without saving a FP32 forward output."""
    bh = tl.program_id(0)
    block = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kv_head = head // KV_HEAD_GROUP_SIZE
    positions = QUERY_START + block * BLOCK_N + tl.arange(0, BLOCK_N)
    qmask = positions < N
    dims = tl.arange(0, BLOCK_HD)
    dmask = dims < HD
    scale = 1.0 / tl.sqrt(HD * 1.0)

    query = tl.load(
        Q + batch * stride_qb + head * stride_qh
        + positions[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=qmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DO + batch * stride_dob + head * stride_doh
        + positions[:, None] * stride_don + dims[None, :] * stride_dod,
        mask=qmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)
    lse = tl.load(
        LSE + batch * stride_lb + head * stride_lh + positions * stride_ln,
        mask=qmask, other=float('-inf'),
    ).to(tl.float32)

    probability_key = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    response_key = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    delta_row = tl.zeros([BLOCK_N], tl.float32)

    for logical_index in range(J_VAL):
        offset = tl.load(OFFSETS + logical_index).to(tl.int32)
        source = positions - offset
        valid = qmask & (source >= 0) & (source < N)
        key = tl.load(
            K + batch * stride_kb + kv_head * stride_kh
            + source[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        value = tl.load(
            V + batch * stride_vb + kv_head * stride_vh
            + source[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        virtual_key = tl.load(
            SCALE_EMBED + logical_index * stride_sei + dims * stride_sed,
            mask=dmask, other=0.0,
        ).to(tl.float32)
        score = tl.sum(query * (key + virtual_key[None, :]), axis=1) * scale
        score += tl.load(POS_BIAS + logical_index * stride_pbi + head * stride_pbh)
        score = tl.where(valid, score, float('-inf'))
        probability = tl.where(
            valid & (lse > float('-inf')),
            tl.exp2((score - lse) * _LOG2E),
            0.0,
        )
        response = tl.sum(dout * value, axis=1)
        effective_key = key + virtual_key[None, :]
        probability_key += probability[:, None] * effective_key
        response_key += (probability * response)[:, None] * effective_key
        delta_row += probability * response

    dq = (response_key - delta_row[:, None] * probability_key) * scale
    tl.store(
        DELTA + batch * stride_db + head * stride_dh + positions * stride_dn,
        delta_row, mask=qmask,
    )
    tl.store(
        DQ + batch * stride_dqb + head * stride_dqh
        + positions[:, None] * stride_dqn + dims[None, :] * stride_dqd,
        dq.to(tl.bfloat16), mask=qmask[:, None] & dmask[None, :],
    )

@triton.jit
def _bwd_dkdv_v22(
    Q, K, V, POS_BIAS, SCALE_EMBED, DO, LSE, DELTA,
    DK, DV, DPOS_BIAS, DSCALE_EMBED, OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    NATIVE_GQA: tl.constexpr,
):
    bh = tl.program_id(0)
    block = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kv_head = head // KV_HEAD_GROUP_SIZE
    source = block * BLOCK_N + tl.arange(0, BLOCK_N)
    smask = source < N
    dims = tl.arange(0, BLOCK_HD)
    dmask = dims < HD
    scale = 1.0 / tl.sqrt(HD * 1.0)

    key = tl.load(
        K + batch * stride_kb + kv_head * stride_kh
        + source[:, None] * stride_kn + dims[None, :] * stride_kd,
        mask=smask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)
    value = tl.load(
        V + batch * stride_vb + kv_head * stride_vh
        + source[:, None] * stride_vn + dims[None, :] * stride_vd,
        mask=smask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for logical_index in range(J_VAL):
        offset = tl.load(OFFSETS + logical_index).to(tl.int32)
        positions = source + offset
        valid = smask & (positions < N)
        query = tl.load(
            Q + batch * stride_qb + head * stride_qh
            + positions[:, None] * stride_qn + dims[None, :] * stride_qd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        dout = tl.load(
            DO + batch * stride_dob + head * stride_doh
            + positions[:, None] * stride_don + dims[None, :] * stride_dod,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            LSE + batch * stride_lb + head * stride_lh + positions * stride_ln,
            mask=valid, other=float('-inf'),
        ).to(tl.float32)
        delta_row = tl.load(
            DELTA + batch * stride_db + head * stride_dh + positions * stride_dn,
            mask=valid, other=0.0,
        ).to(tl.float32)
        virtual_key = tl.load(
            SCALE_EMBED + logical_index * stride_sei + dims * stride_sed,
            mask=dmask, other=0.0,
        ).to(tl.float32)
        score = tl.sum(query * (key + virtual_key[None, :]), axis=1) * scale
        score += tl.load(POS_BIAS + logical_index * stride_pbi + head * stride_pbh)
        score = tl.where(valid, score, float('-inf'))
        probability = tl.where(
            valid & (lse > float('-inf')),
            tl.exp2((score - lse) * _LOG2E),
            0.0,
        )
        dscore = probability * (tl.sum(dout * value, axis=1) - delta_row)
        dk += dscore[:, None] * query * scale
        dv += probability[:, None] * dout
        tl.atomic_add(
            DPOS_BIAS + logical_index * stride_pbi + head * stride_pbh,
            tl.sum(tl.where(valid, dscore, 0.0)), sem="relaxed",
        )
        dscale = tl.sum(dscore[:, None] * query, axis=0) * scale
        tl.atomic_add(
            DSCALE_EMBED + logical_index * stride_sei + dims * stride_sed,
            dscale, mask=dmask, sem="relaxed",
        )

    if NATIVE_GQA:
        tl.atomic_add(
            DK + batch * stride_dkb + kv_head * stride_dkh
            + source[:, None] * stride_dkn + dims[None, :] * stride_dkd,
            dk, mask=smask[:, None] & dmask[None, :], sem="relaxed",
        )
        tl.atomic_add(
            DV + batch * stride_dvb + kv_head * stride_dvh
            + source[:, None] * stride_dvn + dims[None, :] * stride_dvd,
            dv, mask=smask[:, None] & dmask[None, :], sem="relaxed",
        )
    else:
        tl.store(
            DK + batch * stride_dkb + head * stride_dkh
            + source[:, None] * stride_dkn + dims[None, :] * stride_dkd,
            dk.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
        )
        tl.store(
            DV + batch * stride_dvb + head * stride_dvh
            + source[:, None] * stride_dvn + dims[None, :] * stride_dvd,
            dv.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
        )

class _DSQGV22Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed, j_val, offsets_dev, query_start):
        batch, heads, seq_len, head_dim = q.shape
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape[0] != batch or k.shape[2:] != (seq_len, head_dim):
            raise ValueError(
                "native GQA requires q/k/v to share batch, sequence, and head dimensions"
            )
        kv_heads = k.shape[1]
        if heads % kv_heads:
            raise ValueError(
                f"query heads H={heads} must be divisible by KV heads H_KV={kv_heads}"
            )
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise TypeError("Triton DSQG V22 expects BF16 q/k/v tensors")
        if not q.is_cuda:
            raise RuntimeError("Triton DSQG V22 requires CUDA tensors")
        if pos_bias.shape != (j_val, heads):
            raise ValueError(f"pos_bias must have shape {(j_val, heads)}")
        if scale_embed.shape != (j_val, head_dim):
            raise ValueError(f"scale_embed must have shape {(j_val, head_dim)}")

        query_start = max(0, min(int(query_start), seq_len))
        kv_head_group_size = heads // kv_heads
        native_gqa = heads != kv_heads
        pos_bias = pos_bias.contiguous()
        scale_embed = scale_embed.contiguous()
        offsets_dev = offsets_dev.contiguous()

        capability = torch.cuda.get_device_capability()
        sm90 = capability[0] >= 9
        sm89 = capability == (8, 9)
        if head_dim <= 64:
            block_n, num_warps, num_stages = (
                (128, 8, 3) if sm90 else (64, 8, 2) if sm89 else (64, 4, 2)
            )
        elif head_dim <= 128:
            block_n, num_warps, num_stages = (
                (128, 8, 3) if sm90 else (64, 4, 2) if sm89 else (32, 4, 2)
            )
        elif head_dim <= 256:
            block_n, num_warps, num_stages = (
                (32, 4, 3) if sm90 else (32, 4, 2)
            )
        else:
            block_n, num_warps, num_stages = (
                (16, 4, 3) if sm90 else (16, 4, 2) if sm89 else (8, 4, 2)
            )
        block_hd = _next_pow2(head_dim)

        output = torch.zeros_like(q)
        lse = torch.full(
            (batch, heads, seq_len), float('-inf'), device=q.device, dtype=torch.float32
        )
        if query_start < seq_len:
            grid = (
                batch * heads,
                triton.cdiv(seq_len - query_start, block_n),
            )
            _fwd_v22_online[grid](
                q, k, v, pos_bias, scale_embed, output, lse, offsets_dev,
                *q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride(),
                *pos_bias.stride(), *scale_embed.stride(),
                H=heads, N=seq_len, HD=head_dim,
                BLOCK_N=block_n, BLOCK_HD=block_hd, J_VAL=j_val,
                KV_HEAD_GROUP_SIZE=kv_head_group_size, QUERY_START=query_start,
                num_warps=num_warps, num_stages=num_stages,
            )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed, lse, offsets_dev)
        ctx.block_n = block_n
        ctx.block_hd = block_hd
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        ctx.j_val = int(j_val)
        ctx.kv_head_group_size = kv_head_group_size
        ctx.native_gqa = native_gqa
        ctx.query_start = query_start
        return output

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pos_bias, scale_embed, lse, offsets_dev = ctx.saved_tensors
        batch, heads, seq_len, head_dim = q.shape
        block_n = ctx.block_n
        block_hd = ctx.block_hd
        num_warps = ctx.num_warps
        num_stages = ctx.num_stages
        dout = dout.contiguous()

        dq = torch.zeros_like(q)
        if ctx.native_gqa:
            dk_accum = torch.zeros_like(k, dtype=torch.float32)
            dv_accum = torch.zeros_like(v, dtype=torch.float32)
        else:
            dk_accum = torch.zeros_like(k)
            dv_accum = torch.zeros_like(v)
        dpos_bias = torch.zeros_like(pos_bias, dtype=torch.float32)
        dscale_embed = torch.zeros_like(scale_embed, dtype=torch.float32)
        delta = torch.zeros(
            (batch, heads, seq_len), device=q.device, dtype=torch.float32
        )

        if ctx.query_start < seq_len:
            query_grid = (
                batch * heads,
                triton.cdiv(seq_len - ctx.query_start, block_n),
            )
            _bwd_dq_v22[query_grid](
                q, k, v, pos_bias, scale_embed, dout, lse, delta,
                dq, offsets_dev,
                *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
                *lse.stride(), *delta.stride(), *dq.stride(),
                *pos_bias.stride(), *scale_embed.stride(),
                H=heads, N=seq_len, HD=head_dim,
                BLOCK_N=block_n, BLOCK_HD=block_hd, J_VAL=ctx.j_val,
                KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                QUERY_START=ctx.query_start,
                num_warps=num_warps, num_stages=num_stages,
            )

        kv_grid = (batch * heads, triton.cdiv(seq_len, block_n))
        _bwd_dkdv_v22[kv_grid](
            q, k, v, pos_bias, scale_embed, dout, lse, delta,
            dk_accum, dv_accum, dpos_bias, dscale_embed, offsets_dev,
            *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
            *lse.stride(), *delta.stride(),
            *dk_accum.stride(), *dv_accum.stride(),
            *pos_bias.stride(), *scale_embed.stride(),
            H=heads, N=seq_len, HD=head_dim,
            BLOCK_N=block_n, BLOCK_HD=block_hd, J_VAL=ctx.j_val,
            KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
            NATIVE_GQA=ctx.native_gqa,
            num_warps=num_warps, num_stages=num_stages,
        )

        dk = dk_accum.to(k.dtype) if ctx.native_gqa else dk_accum
        dv = dv_accum.to(v.dtype) if ctx.native_gqa else dv_accum
        return dq, dk, dv, dpos_bias, dscale_embed, None, None, None


# ===========================================================================
# Eager semantic oracle + public API
# ===========================================================================


def _eager_dsqg_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    offsets: list[int] | tuple[int, ...] | torch.Tensor,
    *,
    query_start: int = 0,
) -> torch.Tensor:
    """Differentiable clarity-first oracle for the V22 online semantics."""
    if isinstance(offsets, torch.Tensor):
        offsets_list = [int(value) for value in offsets.detach().cpu().tolist()]
    else:
        offsets_list = [int(value) for value in offsets]
    batch, query_heads, seq_len, head_dim = q.shape
    if v.shape != k.shape:
        raise ValueError("k and v must have identical shapes")
    if k.shape[0] != batch or k.shape[2:] != (seq_len, head_dim):
        raise ValueError("q/k/v must share batch, sequence, and head dimensions")
    kv_heads = k.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if pos_bias.shape != (len(offsets_list), query_heads):
        raise ValueError("pos_bias must have [J,Hq]")
    if scale_embed.shape != (len(offsets_list), head_dim):
        raise ValueError("scale_embed must have [J,HD]")

    head_map = torch.arange(query_heads, device=q.device) // (query_heads // kv_heads)
    key_expanded = k[:, head_map]
    value_expanded = v[:, head_map]
    positions = torch.arange(seq_len, device=q.device)
    scale = 1.0 / math.sqrt(float(head_dim))
    score_columns: list[torch.Tensor] = []
    value_columns: list[torch.Tensor] = []
    valid_columns: list[torch.Tensor] = []

    for logical_index, offset in enumerate(offsets_list):
        source = positions - offset
        valid = source >= 0
        safe_source = source.clamp_min(0)
        selected_key = key_expanded[:, :, safe_source, :]
        selected_value = value_expanded[:, :, safe_source, :]
        score = (
            q.float()
            * (selected_key.float() + scale_embed[logical_index].float())
        ).sum(-1) * scale
        score = score + pos_bias[logical_index].float().reshape(1, query_heads, 1)
        score = score.masked_fill(~valid.reshape(1, 1, seq_len), float('-inf'))
        score_columns.append(score)
        value_columns.append(selected_value.float())
        valid_columns.append(valid)

    scores = torch.stack(score_columns, dim=-1)
    values = torch.stack(value_columns, dim=-2)
    valid_mask = torch.stack(valid_columns, dim=-1).reshape(1, 1, seq_len, -1)
    live_rows = positions >= int(query_start)
    valid_mask = valid_mask & live_rows.reshape(1, 1, seq_len, 1)
    any_valid = valid_mask.any(-1, keepdim=True)
    safe_scores = torch.where(valid_mask, scores, torch.full_like(scores, float('-inf')))
    safe_scores = torch.where(any_valid, safe_scores, torch.zeros_like(safe_scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(valid_mask, probabilities, torch.zeros_like(probabilities))
    return (probabilities.unsqueeze(-1) * values).sum(-2).to(q.dtype)


def dsqg_attention_v22(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    offsets_dev: torch.Tensor,
    query_start: int = 0,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch V22 to Triton on CUDA or to the eager semantic oracle."""
    backend = str(backend).lower()
    if backend not in {"auto", "triton", "eager"}:
        raise ValueError("backend must be auto, triton, or eager")
    use_triton = backend == "triton" or (
        backend == "auto" and q.is_cuda and _TRITON_AVAILABLE
    )
    if not use_triton:
        return _eager_dsqg_attention(
            q, k, v, pos_bias, scale_embed, offsets_dev,
            query_start=int(query_start),
        )
    if not q.is_cuda or not _TRITON_AVAILABLE:
        raise RuntimeError("Triton DSQG V22 was requested but CUDA/Triton is unavailable")

    original_dtype = q.dtype
    q_bf16 = q if q.dtype == torch.bfloat16 else q.to(torch.bfloat16)
    k_bf16 = k if k.dtype == torch.bfloat16 else k.to(torch.bfloat16)
    v_bf16 = v if v.dtype == torch.bfloat16 else v.to(torch.bfloat16)
    output = _DSQGV22Fn.apply(
        q_bf16,
        k_bf16,
        v_bf16,
        pos_bias.float(),
        scale_embed.float(),
        int(offsets_dev.numel()),
        offsets_dev,
        int(query_start),
    )
    return output.to(original_dtype)


# ===========================================================================
# Canonical V22 module
# ===========================================================================


class DSQGAttentionV22(nn.Module):
    """Online-only DSQG attention with fused projections and causal safeguards."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets,
        seq_len=2048,
        dropout=0.1,
        pos_bias_scale=None,
        *,
        backend: str = "auto",
        scale_embed_init_std: float = 0.01,
        npci_strength_tau: float = 0.25,
        support_crop_projections: bool = True,
        support_crop_min_offset: int = 64,
    ):
        super().__init__()
        dimension = _strict_int("embedding_dim", embedding_dim, minimum=1)
        heads = _strict_int("num_heads", num_heads, minimum=1)
        if dimension % heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = heads
        self.head_dim = dimension // heads
        if self.head_dim < 16:
            raise ValueError("DSQG Triton paths require head_dim >= 16")
        self.seq_len = _strict_int("seq_len", seq_len, minimum=1)
        self.backend = str(backend).lower()
        if self.backend not in {"auto", "eager", "triton"}:
            raise ValueError("backend must be auto, eager, or triton")


        if not math.isfinite(float(scale_embed_init_std)) or scale_embed_init_std < 0:
            raise ValueError("scale_embed_init_std must be finite and non-negative")
        if not math.isfinite(float(npci_strength_tau)) or npci_strength_tau <= 0:
            raise ValueError("npci_strength_tau must be finite and positive")
        crop_min = _strict_int(
            "support_crop_min_offset", support_crop_min_offset
        )
        if not isinstance(support_crop_projections, bool):
            raise TypeError("support_crop_projections must be bool")
        self.scale_embed_init_std = float(scale_embed_init_std)
        self.npci_strength_tau = float(npci_strength_tau)
        self.support_crop_projections = support_crop_projections
        self.support_crop_min_offset = crop_min

        canonical_offsets = _canonicalize_offsets(tuple(offsets))
        self.offsets = tuple(canonical_offsets)
        self.j_val = len(canonical_offsets)
        self.minimum_offset = min(self.offsets)
        self.register_buffer(
            "offsets_dev",
            torch.tensor(canonical_offsets, dtype=torch.int32),
            persistent=False,
        )

        if pos_bias_scale is None:
            pos_bias_scale = _env_float("DWARF_DSQG_POS_BIAS_SCALE", 1.0)
        if not math.isfinite(float(pos_bias_scale)):
            raise ValueError("pos_bias_scale must be finite")
        self.register_buffer(
            "pos_bias_scale",
            torch.tensor(float(pos_bias_scale), dtype=torch.float32),
            persistent=True,
        )

        # Q/K/V and the residual-content gate share one input GEMM.
        self.qkvg_proj = nn.Linear(dimension, 4 * dimension, bias=True)
        self.out_proj = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.qkvg_proj.bias.zero_()

        initial_slopes = torch.linspace(0.2, 2.0, heads)
        self.pos_bias_log_slope = nn.Parameter(
            torch.log(torch.expm1(initial_slopes))
        )
        self.pos_bias_residual = nn.Parameter(torch.zeros(self.j_val, heads))
        self.scale_embed = nn.Parameter(
            torch.randn(self.j_val, self.head_dim) * self.scale_embed_init_std
        )
        with torch.no_grad():
            self.scale_embed.sub_(self.scale_embed.mean(0, keepdim=True))
        self.if_gain = nn.Parameter(torch.ones(heads))


        raw_theta = _raw_npci_theta_from_effective(NPCI_THETA_INIT)
        self.npci_theta_k = nn.Parameter(torch.full((heads,), raw_theta))
        self.npci_theta_v = nn.Parameter(torch.full((heads,), raw_theta))
        self.dropout = nn.Dropout(float(dropout))

    @property
    def pos_bias(self) -> torch.Tensor:
        log_distance = torch.log1p(
            self.offsets_dev.to(
                device=self.pos_bias_residual.device,
                dtype=self.pos_bias_residual.dtype,
            )
        ).reshape(-1, 1)
        slope = F.softplus(self.pos_bias_log_slope).reshape(1, -1)
        return (
            -log_distance * slope + self.pos_bias_residual
        ) * self.pos_bias_scale

    @property
    def centered_scale_embed(self) -> torch.Tensor:
        # Keep centering and the virtual-key path in FP32 for Triton/autograd.
        values = self.scale_embed.float()
        return values - values.mean(dim=0, keepdim=True)

    def support_statistics(self, model_length: int | None = None) -> dict[str, object]:
        length = int(model_length if model_length is not None else self.seq_len - 1)
        support = [max(0, length - offset) for offset in self.offsets]
        possible = max(1, length * len(self.offsets))
        return {
            "model_length": length,
            "offsets": self.offsets,
            "minimum_support": min(support),
            "maximum_support": max(support),
            "mean_valid_fraction": sum(support) / possible,
            "dead_offsets": tuple(
                offset for offset, count in zip(self.offsets, support, strict=True)
                if count == 0
            ),
        }

    def semantic_config(self) -> dict[str, object]:
        return {
            "implementation": "dsqg-v22-online",
            "offsets": self.offsets,
            "offset_loop": "single_j_val",
            "positional_bias": "analytic_log_plus_residual",
            "pos_bias_scale": float(self.pos_bias_scale.detach().cpu()),
            "scale_embed": "centered_fp32_virtual_key",
            "npci": "retained_magnitude_aware_causal_injection_rotation",
            "npci_strength_tau": self.npci_strength_tau,
        }

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "online_softmax": True,
            "query_support_start": self.minimum_offset,
            "support_crop_projections": self.support_crop_projections,
            "support_crop_min_offset": self.support_crop_min_offset,
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Supplied V19/V20 checkpoints stored separate qkv and gate projections.
        fused_weight = prefix + "qkvg_proj.weight"
        fused_bias = prefix + "qkvg_proj.bias"
        old_qkv_weight = prefix + "qkv_proj.weight"
        old_qkv_bias = prefix + "qkv_proj.bias"
        old_gate_weight = prefix + "gate_proj.weight"
        old_gate_bias = prefix + "gate_proj.bias"
        state_dict.pop(prefix + "out_proj.bias", None)
        if fused_weight not in state_dict and old_qkv_weight in state_dict:
            qkv_weight = state_dict.pop(old_qkv_weight)
            gate_weight = state_dict.pop(
                old_gate_weight,
                self.qkvg_proj.weight.detach()[3 * self.qkvg_proj.in_features :].clone(),
            )
            state_dict[fused_weight] = torch.cat((qkv_weight, gate_weight), dim=0)
            qkv_bias = state_dict.pop(
                old_qkv_bias,
                qkv_weight.new_zeros(3 * self.qkvg_proj.in_features),
            )
            gate_bias = state_dict.pop(
                old_gate_bias,
                qkv_bias.new_zeros(self.qkvg_proj.in_features),
            )
            state_dict[fused_bias] = torch.cat((qkv_bias, gate_bias), dim=0)

        old_pos_key = prefix + "pos_bias"
        residual_key = prefix + "pos_bias_residual"
        slope_key = prefix + "pos_bias_log_slope"
        if old_pos_key in state_dict and residual_key not in state_dict:
            old_pos = state_dict.pop(old_pos_key)
            state_dict[slope_key] = self.pos_bias_log_slope.detach().clone()
            log_distance = torch.log1p(
                self.offsets_dev.to(dtype=old_pos.dtype)
            ).reshape(-1, 1)
            analytic = -log_distance * F.softplus(
                self.pos_bias_log_slope.detach().to(old_pos.dtype)
            ).reshape(1, -1)
            state_dict[residual_key] = old_pos - analytic
        current = self.state_dict()
        for local_name in (
            "pos_bias_log_slope",
            "pos_bias_residual",
            "pos_bias_scale",
        ):
            key = prefix + local_name
            if key not in state_dict:
                state_dict[key] = current[local_name].detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_inject: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dimension = x.shape
        heads, head_dim = self.num_heads, self.head_dim
        query_start = min(self.minimum_offset, seq_len)
        key_end = max(0, seq_len - query_start)
        if query_start >= seq_len:
            return torch.zeros_like(x)

        crop = (
            self.support_crop_projections
            and query_start >= self.support_crop_min_offset
        )
        if crop:
            weight = self.qkvg_proj.weight
            bias = self.qkvg_proj.bias
            suffix = x[:, query_start:]
            prefix = x[:, :key_end]
            q_suffix = F.linear(suffix, weight[:dimension], bias[:dimension])
            gate_suffix = F.linear(
                suffix, weight[3 * dimension :], bias[3 * dimension :]
            )
            kv_prefix = F.linear(
                prefix, weight[dimension : 3 * dimension],
                bias[dimension : 3 * dimension],
            )
            k_prefix, v_prefix = kv_prefix.split(dimension, dim=-1)
            q_flat = x.new_zeros(batch, seq_len, dimension)
            k_flat = x.new_zeros(batch, seq_len, dimension)
            v_flat = x.new_zeros(batch, seq_len, dimension)
            content_gate = x.new_zeros(batch, seq_len, dimension)
            q_flat[:, query_start:] = q_suffix
            content_gate[:, query_start:] = gate_suffix
            k_flat[:, :key_end] = k_prefix
            v_flat[:, :key_end] = v_prefix
        else:
            q_flat, k_flat, v_flat, content_gate = self.qkvg_proj(x).split(
                dimension, dim=-1
            )

        query = q_flat.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)
        key = k_flat.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)
        value = v_flat.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3)
        if kv_inject is not None:
            key_delta, value_delta = kv_inject
            full_shape = key.shape
            prefix_shape = (batch, heads, key_end, head_dim)
            if key_delta.shape == full_shape and value_delta.shape == full_shape:
                key_delta_live = key_delta[:, :, :key_end]
                value_delta_live = value_delta[:, :, :key_end]
            elif key_delta.shape == prefix_shape and value_delta.shape == prefix_shape:
                key_delta_live = key_delta
                value_delta_live = value_delta
            else:
                raise ValueError(
                    "kv_inject must contain full [B,H,N,HD] tensors or the "
                    "exact live source prefix [B,H,N-min_offset,HD]"
                )
            rotated_key = npci_rotate(
                key[:, :, :key_end],
                key_delta_live,
                NPCI_THETA_MAX * torch.tanh(self.npci_theta_k),
                strength_tau=self.npci_strength_tau,
            )
            rotated_value = npci_rotate(
                value[:, :, :key_end],
                value_delta_live,
                NPCI_THETA_MAX * torch.tanh(self.npci_theta_v),
                strength_tau=self.npci_strength_tau,
            )
            if key_end == seq_len:
                key, value = rotated_key, rotated_value
            else:
                key = torch.cat((rotated_key, key[:, :, key_end:]), dim=2)
                value = torch.cat((rotated_value, value[:, :, key_end:]), dim=2)

        output = dsqg_attention_v22(
            query,
            key,
            value,
            self.pos_bias,
            self.centered_scale_embed,
            self.offsets_dev,
            query_start,
            backend=self.backend,
        )
        output = output * self.if_gain.reshape(1, heads, 1, 1)
        flattened = output.permute(0, 2, 1, 3).reshape(batch, seq_len, dimension)
        return self.dropout(
            self.out_proj(flattened * torch.sigmoid(content_gate))
        )


# Intentional public compatibility aliases.
DSQGAttentionV19 = DSQGAttentionV22
DSQGAttentionV20 = DSQGAttentionV22
DSQGAttentionV21 = DSQGAttentionV22


if __name__ == "__main__":
    torch.manual_seed(5)
    smoke_kwargs = dict(
        embedding_dim=64,
        num_heads=4,
        offsets=[1, 29, 32, 47, 48],
        seq_len=64,
        dropout=0.0,
        backend="eager",
    )
    module = DSQGAttentionV22(**smoke_kwargs).double()
    assert module.offsets == (1, 29, 32, 47, 48)
    expected_keys = {
        "pos_bias_scale",
        "pos_bias_log_slope",
        "pos_bias_residual",
        "scale_embed",
        "if_gain",
        "npci_theta_k",
        "npci_theta_v",
        "qkvg_proj.weight",
        "qkvg_proj.bias",
        "out_proj.weight",
    }
    state = module.state_dict()
    assert set(state) == expected_keys
    restored = DSQGAttentionV22(**smoke_kwargs).double()
    incompatible = restored.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    for key, value in state.items():
        assert torch.equal(value, restored.state_dict()[key])

    values = torch.randn(2, 61, 64, dtype=torch.float64, requires_grad=True)
    output = module(values)
    assert output.dtype == values.dtype
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
    output.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    module.eval()
    with torch.no_grad():
        baseline = module(values.detach())
        changed = values.detach().clone()
        changed[:, 40:] += torch.randn_like(changed[:, 40:]) * 10
        perturbed = module(changed)
        assert torch.equal(baseline[:, :40], perturbed[:, :40])
    print(
        "CPU DSQG V22 forward/backward/causality/offsets/state-dict smoke test: PASS"
    )
