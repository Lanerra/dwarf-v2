"""Dynamic Sparse Query-Gather attention with bounded causal routing."""


from __future__ import annotations

import bisect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

    class _KernelStub:
        def __init__(self, function):
            self.function = function

        def __getitem__(self, _grid):
            raise RuntimeError("Triton is unavailable; use the eager DSQG oracle")

    class _JitStub:
        def __call__(self, function):
            return _KernelStub(function)

    class _ConfigStub:
        def __init__(self, kwargs, num_warps=4, num_stages=3):
            self.kwargs = kwargs
            self.num_warps = num_warps
            self.num_stages = num_stages

    class _TritonStub:
        jit = _JitStub()
        Config = _ConfigStub

        @staticmethod
        def autotune(configs, key, **_kwargs):
            def decorate(kernel):
                kernel.configs = configs
                kernel.keys = key
                return kernel

            return decorate

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    class _TLStub:
        @staticmethod
        def constexpr(value=None):
            return value

    triton = _TritonStub()
    tl = _TLStub()


def dsqg_triton_available() -> bool:
    """Return whether the production Triton DSQG backend is importable."""
    return bool(_TRITON_AVAILABLE)


_LOG2E = tl.constexpr(1.4426950408889634)

_DSQG_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
]
_DSQG_MIN_AUTOTUNE_BLOCK_N = min(config.kwargs["BLOCK_N"] for config in _DSQG_AUTOTUNE_CONFIGS)


ALL_OFFSETS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15,
    16, 19, 21, 23, 28, 48, 64, 96, 121, 161, 192, 212,
    245, 273, 295, 342, 375, 384, 413, 441, 473, 512, 549, 579,
    593, 631, 653, 694, 716, 768, 826, 846, 900, 936, 970, 1000,
    1024, 1074, 1108, 1144, 1166, 1190, 1218, 1244, 1288, 1322, 1385, 1423,
    1451, 1497, 1522, 1550, 1581, 1603, 1617, 1634, 1651, 1661, 1710, 1743,
    1780, 1810, 1820, 1852, 1860, 1876, 1886, 1897, 1903, 1916, 1926, 1929,
    1941, 1965, 1983, 2006, 2011, 2029, 2037, 2044, 2068, 2097, 2113, 2199,
]


def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive tensors."""
    return value + torch.log(-torch.expm1(-value))


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


@triton.autotune(
    configs=_DSQG_AUTOTUNE_CONFIGS,
    key=[
        "BATCH", "H", "N", "HD", "J_VAL", "KV_HEAD_GROUP_SIZE",
        "QUERY_START",
    ],
    reset_to_zero=["OUT"],
    restore_value=["LSE"],
    cache_results=True,
)
@triton.jit
def _fwd_v23_online(
    Q, K, V, POS_BIAS, SCALE_EMBED, NULL_KEY, NULL_BIAS, OUT, LSE, OFFSETS,
    LOG_VALID_COUNT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_nkh, stride_nkd,
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
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

    null_key = tl.load(
        NULL_KEY + head * stride_nkh + dims * stride_nkd,
        mask=dmask, other=0.0,
    ).to(tl.float32)
    null_score = tl.sum(query * null_key[None, :], axis=1) * scale
    # Calibrate the null score against the number of valid offsets. The
    # position-only term is precomputed once per module instead of traversing
    # every offset a second time in each forward kernel.
    null_score += tl.load(LOG_VALID_COUNT + positions, mask=qmask, other=0.0)
    null_score += tl.load(NULL_BIAS + head)
    running_max = tl.where(qmask, null_score, float('-inf'))
    running_sum = tl.where(qmask, 1.0, 0.0)
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


@triton.autotune(
    configs=_DSQG_AUTOTUNE_CONFIGS,
    key=[
        "BATCH", "H", "N", "HD", "J_VAL", "KV_HEAD_GROUP_SIZE",
        "QUERY_START",
    ],
    reset_to_zero=["DQ", "DELTA", "DNULL_KEY", "DNULL_BIAS"],
    cache_results=True,
)
@triton.jit
def _bwd_dq_v23(
    Q, K, V, POS_BIAS, SCALE_EMBED, NULL_KEY, NULL_BIAS, DO, LSE, DELTA,
    DQ, DNULL_KEY, DNULL_BIAS, OFFSETS, LOG_VALID_COUNT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_nkh, stride_nkd,
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr,
    GRAD_BLOCKS: tl.constexpr,
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

    null_key = tl.load(
        NULL_KEY + head * stride_nkh + dims * stride_nkd,
        mask=dmask, other=0.0,
    ).to(tl.float32)
    null_score = tl.sum(query * null_key[None, :], axis=1) * scale
    null_score += tl.load(LOG_VALID_COUNT + positions, mask=qmask, other=0.0)
    null_score += tl.load(NULL_BIAS + head)
    null_probability = tl.where(
        qmask & (lse > float('-inf')),
        tl.exp2((null_score - lse) * _LOG2E),
        0.0,
    )
    probability_key = null_probability[:, None] * null_key[None, :]
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
    dnull_score = -null_probability * delta_row
    dnull_key = tl.sum(dnull_score[:, None] * query, axis=0) * scale
    partial = (batch * H + head) * GRAD_BLOCKS + block
    tl.store(
        DNULL_KEY + partial * HD + dims,
        dnull_key, mask=dmask,
    )
    tl.store(
        DNULL_BIAS + partial,
        tl.sum(tl.where(qmask, dnull_score, 0.0)),
    )
    tl.store(
        DELTA + batch * stride_db + head * stride_dh + positions * stride_dn,
        delta_row, mask=qmask,
    )
    # DQ has the BF16 Triton-input dtype. This is the single final rounding at
    # the custom-autograd boundary; all accumulation above remains FP32.
    tl.store(
        DQ + batch * stride_dqb + head * stride_dqh
        + positions[:, None] * stride_dqn + dims[None, :] * stride_dqd,
        dq.to(tl.bfloat16), mask=qmask[:, None] & dmask[None, :],
    )

@triton.autotune(
    configs=_DSQG_AUTOTUNE_CONFIGS,
    key=[
        "BATCH", "H", "N", "HD", "J_VAL", "KV_HEAD_GROUP_SIZE",
        "QUERY_START", "SOURCE_END",
    ],
    reset_to_zero=["DK", "DV", "DPOS_BIAS", "DSCALE_EMBED"],
    cache_results=True,
)
@triton.jit
def _bwd_dkdv_v23(
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
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr, SOURCE_END: tl.constexpr,
    GRAD_BLOCKS: tl.constexpr,
):
    bhkv = tl.program_id(0)
    block = tl.program_id(1)
    kv_heads = H // KV_HEAD_GROUP_SIZE
    batch = bhkv // kv_heads
    kv_head = bhkv % kv_heads
    first_head = kv_head * KV_HEAD_GROUP_SIZE
    source = block * BLOCK_N + tl.arange(0, BLOCK_N)
    smask = source < SOURCE_END
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

    for group_index in range(KV_HEAD_GROUP_SIZE):
        head = first_head + group_index
        for logical_index in range(J_VAL):
            offset = tl.load(OFFSETS + logical_index).to(tl.int32)
            positions = source + offset
            valid = smask & (positions >= QUERY_START) & (positions < N)
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
            score += tl.load(
                POS_BIAS + logical_index * stride_pbi + head * stride_pbh
            )
            score = tl.where(valid, score, float('-inf'))
            probability = tl.where(
                valid & (lse > float('-inf')),
                tl.exp2((score - lse) * _LOG2E),
                0.0,
            )
            dscore = probability * (tl.sum(dout * value, axis=1) - delta_row)
            dk += dscore[:, None] * query * scale
            dv += probability[:, None] * dout
            partial = ((batch * H + head) * GRAD_BLOCKS + block) * J_VAL + logical_index
            tl.store(
                DPOS_BIAS + partial,
                tl.sum(tl.where(valid, dscore, 0.0)),
            )
            dscale = tl.sum(dscore[:, None] * query, axis=0) * scale
            tl.store(
                DSCALE_EMBED + partial * HD + dims,
                dscale, mask=dmask,
            )

    # DK/DV have the BF16 Triton-input dtype. The grouped reductions above
    # accumulate in FP32 before these final stores.
    tl.store(
        DK + batch * stride_dkb + kv_head * stride_dkh
        + source[:, None] * stride_dkn + dims[None, :] * stride_dkd,
        dk.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
    )
    tl.store(
        DV + batch * stride_dvb + kv_head * stride_dvh
        + source[:, None] * stride_dvn + dims[None, :] * stride_dvd,
        dv.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
    )

def _support_band_ranges(
    offsets: tuple[int, ...],
    seq_len: int,
    query_start: int,
    source_end: int,
    boundaries: tuple[int, ...] = (64, 256, 512, 1024, 1536),
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return exact disjoint query/source bands with bounded active offset sets."""
    if not offsets:
        return (), ()
    cutoffs = sorted({0, *(int(v) for v in boundaries if 0 < int(v) < seq_len), seq_len})
    query_bands: list[tuple[int, int, int]] = []
    start = max(0, int(query_start))
    for end in cutoffs[1:]:
        band_start = max(start, cutoffs[cutoffs.index(end) - 1])
        band_end = min(seq_len, end)
        if band_start < band_end:
            active = bisect.bisect_left(offsets, band_end)
            if active > 0:
                query_bands.append((band_start, band_end, active))
        if band_end > start:
            start = band_end
    if start < seq_len:
        query_bands.append((start, seq_len, len(offsets)))

    remaining_ranges = list(zip(cutoffs[:-1], cutoffs[1:]))
    source_bands: list[tuple[int, int, int]] = []
    for remaining_start, remaining_end in reversed(remaining_ranges):
        source_start = max(0, seq_len - remaining_end)
        source_stop = min(int(source_end), seq_len - remaining_start)
        if source_start < source_stop:
            active = bisect.bisect_left(offsets, remaining_end)
            if active > 0:
                source_bands.append((source_start, source_stop, active))
    source_bands.sort()
    return tuple(query_bands), tuple(source_bands)


def _band_launch_geometry(active_offsets: int) -> tuple[int, int, int]:
    """Conservative fixed launch geometry; benchmarkable without autotune replay."""
    if active_offsets <= 12:
        return 128, 4, 2
    if active_offsets <= 24:
        return 64, 4, 2
    return 32, 8, 2


@triton.jit
def _fwd_v23_banded(
    Q, K, V, POS_BIAS, SCALE_EMBED, NULL_KEY, NULL_BIAS, OUT, LSE, OFFSETS,
    LOG_VALID_COUNT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_nkh, stride_nkd,
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_ACTIVE: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr, QUERY_END: tl.constexpr,
):
    bh = tl.program_id(0)
    block = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kv_head = head // KV_HEAD_GROUP_SIZE
    positions = QUERY_START + block * BLOCK_N + tl.arange(0, BLOCK_N)
    qmask = positions < QUERY_END
    dims = tl.arange(0, BLOCK_HD)
    dmask = dims < HD
    scale = 1.0 / tl.sqrt(HD * 1.0)
    query = tl.load(
        Q + batch * stride_qb + head * stride_qh
        + positions[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=qmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)
    null_key = tl.load(
        NULL_KEY + head * stride_nkh + dims * stride_nkd,
        mask=dmask, other=0.0,
    ).to(tl.float32)
    null_score = tl.sum(query * null_key[None, :], axis=1) * scale
    null_score += tl.load(LOG_VALID_COUNT + positions, mask=qmask, other=0.0)
    null_score += tl.load(NULL_BIAS + head)
    running_max = tl.where(qmask, null_score, float('-inf'))
    running_sum = tl.where(qmask, 1.0, 0.0)
    accumulator = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    for logical_index in range(J_ACTIVE):
        delta = tl.load(OFFSETS + logical_index).to(tl.int32)
        source = positions - delta
        valid = qmask & (source >= 0) & (source < N)
        key = tl.load(
            K + batch * stride_kb + kv_head * stride_kh
            + source[:, None] * stride_kn + dims[None, :] * stride_kd,
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
        old_scale = tl.where(has_old, tl.exp2((running_max - safe_max) * _LOG2E), 0.0)
        new_weight = tl.where(valid, tl.exp2((score - safe_max) * _LOG2E), 0.0)
        value = tl.load(
            V + batch * stride_vb + kv_head * stride_vh
            + source[:, None] * stride_vn + dims[None, :] * stride_vd,
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
def _bwd_dq_v23_banded(
    Q, K, V, POS_BIAS, SCALE_EMBED, NULL_KEY, NULL_BIAS, DO, LSE, DELTA,
    DQ, DNULL_KEY, DNULL_BIAS, OFFSETS, LOG_VALID_COUNT,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_nkh, stride_nkd,
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_ACTIVE: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr, QUERY_END: tl.constexpr,
    GRAD_BLOCKS: tl.constexpr,
    GRAD_BLOCK_OFFSET: tl.constexpr,
):
    bh = tl.program_id(0)
    block = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kv_head = head // KV_HEAD_GROUP_SIZE
    positions = QUERY_START + block * BLOCK_N + tl.arange(0, BLOCK_N)
    qmask = positions < QUERY_END
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
    null_key = tl.load(
        NULL_KEY + head * stride_nkh + dims * stride_nkd,
        mask=dmask, other=0.0,
    ).to(tl.float32)
    null_score = tl.sum(query * null_key[None, :], axis=1) * scale
    null_score += tl.load(LOG_VALID_COUNT + positions, mask=qmask, other=0.0)
    null_score += tl.load(NULL_BIAS + head)
    null_probability = tl.where(
        qmask & (lse > float('-inf')),
        tl.exp2((null_score - lse) * _LOG2E), 0.0,
    )
    probability_key = null_probability[:, None] * null_key[None, :]
    response_key = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
    delta_row = tl.zeros([BLOCK_N], tl.float32)
    for logical_index in range(J_ACTIVE):
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
            tl.exp2((score - lse) * _LOG2E), 0.0,
        )
        response = tl.sum(dout * value, axis=1)
        effective_key = key + virtual_key[None, :]
        probability_key += probability[:, None] * effective_key
        response_key += (probability * response)[:, None] * effective_key
        delta_row += probability * response
    dq = (response_key - delta_row[:, None] * probability_key) * scale
    dnull_score = -null_probability * delta_row
    partial = (batch * H + head) * GRAD_BLOCKS + GRAD_BLOCK_OFFSET + block
    tl.store(
        DNULL_KEY + partial * HD + dims,
        tl.sum(dnull_score[:, None] * query, axis=0) * scale,
        mask=dmask,
    )
    tl.store(
        DNULL_BIAS + partial,
        tl.sum(tl.where(qmask, dnull_score, 0.0)),
    )
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
def _bwd_dkdv_v23_banded(
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
    BATCH: tl.constexpr, H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_ACTIVE: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr, SOURCE_START: tl.constexpr,
    SOURCE_END: tl.constexpr,
    GRAD_BLOCKS: tl.constexpr,
    GRAD_BLOCK_OFFSET: tl.constexpr, GRAD_J: tl.constexpr,
    COMPACT_PARTIALS: tl.constexpr = False, GRAD_ENTRIES: tl.constexpr = 0,
):
    bhkv = tl.program_id(0)
    block = tl.program_id(1)
    kv_heads = H // KV_HEAD_GROUP_SIZE
    batch = bhkv // kv_heads
    kv_head = bhkv % kv_heads
    first_head = kv_head * KV_HEAD_GROUP_SIZE
    source = SOURCE_START + block * BLOCK_N + tl.arange(0, BLOCK_N)
    smask = source < SOURCE_END
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
    for group_index in range(KV_HEAD_GROUP_SIZE):
        head = first_head + group_index
        for logical_index in range(J_ACTIVE):
            offset = tl.load(OFFSETS + logical_index).to(tl.int32)
            positions = source + offset
            valid = smask & (positions >= QUERY_START) & (positions < N)
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
                tl.exp2((score - lse) * _LOG2E), 0.0,
            )
            dscore = probability * (tl.sum(dout * value, axis=1) - delta_row)
            dk += dscore[:, None] * query * scale
            dv += probability[:, None] * dout
            if COMPACT_PARTIALS:
                partial = (
                    (batch * H + head) * GRAD_ENTRIES
                    + GRAD_BLOCK_OFFSET + block * J_ACTIVE + logical_index
                )
            else:
                partial = (
                    ((batch * H + head) * GRAD_BLOCKS + GRAD_BLOCK_OFFSET + block)
                    * GRAD_J + logical_index
                )
            tl.store(
                DPOS_BIAS + partial,
                tl.sum(tl.where(valid, dscore, 0.0)),
            )
            tl.store(
                DSCALE_EMBED + partial * HD + dims,
                tl.sum(dscore[:, None] * query, axis=0) * scale,
                mask=dmask,
            )
    tl.store(
        DK + batch * stride_dkb + kv_head * stride_dkh
        + source[:, None] * stride_dkn + dims[None, :] * stride_dkd,
        dk.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
    )
    tl.store(
        DV + batch * stride_dvb + kv_head * stride_dvh
        + source[:, None] * stride_dvn + dims[None, :] * stride_dvd,
        dv.to(tl.bfloat16), mask=smask[:, None] & dmask[None, :],
    )


@triton.jit
def _compact_scale_reduce_v23(
    PARTIALS, OUT,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, J: tl.constexpr,
    ENTRIES: tl.constexpr, ACTIVES: tl.constexpr, BLOCKS: tl.constexpr,
    STARTS: tl.constexpr, REDUCE_BLOCK: tl.constexpr,
):
    logical_index = tl.program_id(0)
    rows, dims = tl.arange(0, REDUCE_BLOCK), tl.arange(0, D)
    accumulator = tl.zeros((REDUCE_BLOCK, D), tl.float32)
    for band in tl.static_range(len(ACTIVES)):
        active = ACTIVES[band]
        blocks = BLOCKS[band]
        start = STARTS[band]
        if logical_index < active:
            for base in range(tl.cdiv(B * H * blocks, REDUCE_BLOCK)):
                row = base * REDUCE_BLOCK + rows
                batch_head = row // blocks
                block = row % blocks
                index = (
                    batch_head * ENTRIES + start + block * active + logical_index
                )
                accumulator += tl.load(
                    PARTIALS + index[:, None] * D + dims[None, :],
                    mask=row[:, None] < B * H * blocks, other=0.0,
                )
    tl.store(OUT + logical_index * D + dims, tl.sum(accumulator, axis=0))


@triton.jit
def _compact_bias_reduce_v23(
    PARTIALS, OUT,
    B: tl.constexpr, H: tl.constexpr, J: tl.constexpr,
    ENTRIES: tl.constexpr, ACTIVES: tl.constexpr, BLOCKS: tl.constexpr,
    STARTS: tl.constexpr, REDUCE_BLOCK: tl.constexpr,
):
    head, logical_index = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, REDUCE_BLOCK)
    accumulator = tl.zeros((REDUCE_BLOCK,), tl.float32)
    for band in tl.static_range(len(ACTIVES)):
        active = ACTIVES[band]
        blocks = BLOCKS[band]
        start = STARTS[band]
        if logical_index < active:
            for base in range(tl.cdiv(B * blocks, REDUCE_BLOCK)):
                row = base * REDUCE_BLOCK + rows
                batch = row // blocks
                block = row % blocks
                index = (
                    (batch * H + head) * ENTRIES
                    + start + block * active + logical_index
                )
                accumulator += tl.load(
                    PARTIALS + index, mask=row < B * blocks, other=0.0
                )
    tl.store(OUT + logical_index * H + head, tl.sum(accumulator, axis=0))


def _reduce_compact_shared_gradients_v23(
    partial_pos_bias: torch.Tensor,
    partial_scale_embed: torch.Tensor,
    *,
    batch: int, heads: int, j_val: int, head_dim: int, entries: int,
    actives: tuple[int, ...], blocks: tuple[int, ...], starts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    dpos_bias = torch.empty(
        (j_val, heads), device=partial_pos_bias.device, dtype=torch.float32
    )
    dscale_embed = torch.empty(
        (j_val, head_dim), device=partial_scale_embed.device, dtype=torch.float32
    )
    _compact_bias_reduce_v23[(heads, j_val)](
        partial_pos_bias, dpos_bias,
        B=batch, H=heads, J=j_val, ENTRIES=entries,
        ACTIVES=actives, BLOCKS=blocks, STARTS=starts, REDUCE_BLOCK=256,
        num_warps=4,
    )
    _compact_scale_reduce_v23[(j_val,)](
        partial_scale_embed, dscale_embed,
        B=batch, H=heads, D=head_dim, J=j_val, ENTRIES=entries,
        ACTIVES=actives, BLOCKS=blocks, STARTS=starts, REDUCE_BLOCK=64,
        num_warps=4,
    )
    return dpos_bias, dscale_embed


class _DSQGV23Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        pos_bias,
        scale_embed,
        null_key,
        null_bias,
        log_valid_count,
        j_val,
        offsets_dev,
        query_start,
        source_end=None,
        offsets_host=None,
        support_band_execution=True,
    ):
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
            raise TypeError("Triton DSQG V23 expects BF16 q/k/v tensors")
        if not q.is_cuda:
            raise RuntimeError("Triton DSQG V23 requires CUDA tensors")
        if pos_bias.shape != (j_val, heads):
            raise ValueError(f"pos_bias must have shape {(j_val, heads)}")
        if scale_embed.shape != (j_val, head_dim):
            raise ValueError(f"scale_embed must have shape {(j_val, head_dim)}")
        if null_key.shape != (heads, head_dim):
            raise ValueError(f"null_key must have shape {(heads, head_dim)}")
        if null_bias.shape != (heads,):
            raise ValueError(f"null_bias must have shape {(heads,)}")
        if log_valid_count.ndim != 1 or log_valid_count.numel() < seq_len:
            raise ValueError("log_valid_count must cover every query position")

        query_start = max(0, min(int(query_start), seq_len))
        source_end = seq_len if source_end is None else int(source_end)
        source_end = max(0, min(source_end, seq_len))
        kv_head_group_size = heads // kv_heads
        pos_bias = pos_bias.contiguous()
        scale_embed = scale_embed.contiguous()
        null_key = null_key.contiguous()
        null_bias = null_bias.contiguous()
        log_valid_count = log_valid_count[:seq_len].float().contiguous()
        offsets_dev = offsets_dev.contiguous()
        block_hd = _next_pow2(head_dim)

        use_bands = bool(support_band_execution and offsets_host is not None)
        offsets_tuple = (
            tuple(int(value) for value in offsets_host)
            if offsets_host is not None else ()
        )
        query_bands, source_bands = (
            _support_band_ranges(
                offsets_tuple, seq_len, query_start, source_end
            )
            if use_bands else ((), ())
        )
        output = torch.zeros_like(q)
        lse = torch.full(
            (batch, heads, seq_len), float('-inf'), device=q.device, dtype=torch.float32
        )
        if use_bands:
            for band_start, band_end, active in query_bands:
                block_n, warps, stages = _band_launch_geometry(active)
                grid = (
                    batch * heads,
                    triton.cdiv(band_end - band_start, block_n),
                )
                _fwd_v23_banded[grid](
                    q, k, v, pos_bias, scale_embed, null_key, null_bias,
                    output, lse, offsets_dev, log_valid_count,
                    *q.stride(), *k.stride(), *v.stride(),
                    *output.stride(), *lse.stride(),
                    *pos_bias.stride(), *scale_embed.stride(), *null_key.stride(),
                    BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                    BLOCK_N=block_n, BLOCK_HD=block_hd, J_ACTIVE=active,
                    KV_HEAD_GROUP_SIZE=kv_head_group_size,
                    QUERY_START=band_start, QUERY_END=band_end,
                    num_warps=warps, num_stages=stages,
                )
        elif query_start < seq_len:
            grid = lambda meta: (
                batch * heads,
                triton.cdiv(seq_len - query_start, meta["BLOCK_N"]),
            )
            _fwd_v23_online[grid](
                q, k, v, pos_bias, scale_embed, null_key, null_bias,
                output, lse, offsets_dev, log_valid_count,
                *q.stride(), *k.stride(), *v.stride(),
                *output.stride(), *lse.stride(),
                *pos_bias.stride(), *scale_embed.stride(), *null_key.stride(),
                BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                BLOCK_HD=block_hd, J_VAL=j_val,
                KV_HEAD_GROUP_SIZE=kv_head_group_size,
                QUERY_START=query_start,
            )

        ctx.save_for_backward(
            q, k, v, pos_bias, scale_embed, null_key, null_bias, lse,
            offsets_dev, log_valid_count,
        )
        ctx.block_hd = block_hd
        ctx.j_val = int(j_val)
        ctx.kv_head_group_size = kv_head_group_size
        ctx.query_start = query_start
        ctx.source_end = source_end
        ctx.query_bands = query_bands
        ctx.source_bands = source_bands
        ctx.use_bands = use_bands
        ctx.input_count = len(ctx.needs_input_grad)
        return output

    @staticmethod
    def backward(ctx, dout):
        (
            q, k, v, pos_bias, scale_embed, null_key, null_bias, lse,
            offsets_dev, log_valid_count,
        ) = ctx.saved_tensors
        batch, heads, seq_len, head_dim = q.shape
        dout = dout.contiguous()
        dq = torch.zeros_like(q)
        dk_accum = torch.zeros_like(k)
        dv_accum = torch.zeros_like(v)
        dpos_bias = torch.zeros_like(pos_bias, dtype=torch.float32)
        dscale_embed = torch.zeros_like(scale_embed, dtype=torch.float32)
        dnull_key = torch.zeros_like(null_key, dtype=torch.float32)
        dnull_bias = torch.zeros_like(null_bias, dtype=torch.float32)
        delta = torch.zeros(
            (batch, heads, seq_len), device=q.device, dtype=torch.float32
        )

        if ctx.query_start < seq_len:
            if ctx.use_bands:
                query_blocks = sum(
                    triton.cdiv(stop - start, _band_launch_geometry(active)[0])
                    for start, stop, active in ctx.query_bands
                )
                partial_null_key = torch.zeros(
                    (batch, heads, query_blocks, head_dim), device=q.device, dtype=torch.float32
                )
                partial_null_bias = torch.zeros(
                    (batch, heads, query_blocks), device=q.device, dtype=torch.float32
                )
                query_block_offset = 0
                for band_start, band_end, active in ctx.query_bands:
                    block_n, warps, stages = _band_launch_geometry(active)
                    grid = (
                        batch * heads,
                        triton.cdiv(band_end - band_start, block_n),
                    )
                    _bwd_dq_v23_banded[grid](
                        q, k, v, pos_bias, scale_embed, null_key, null_bias,
                        dout, lse, delta, dq, partial_null_key, partial_null_bias, offsets_dev,
                        log_valid_count,
                        *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
                        *lse.stride(), *delta.stride(), *dq.stride(),
                        *pos_bias.stride(), *scale_embed.stride(), *null_key.stride(),
                        BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                        BLOCK_N=block_n, BLOCK_HD=ctx.block_hd, J_ACTIVE=active,
                        KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                        QUERY_START=band_start, QUERY_END=band_end,
                        GRAD_BLOCKS=query_blocks, GRAD_BLOCK_OFFSET=query_block_offset,
                        num_warps=warps, num_stages=stages,
                    )
                    query_block_offset += grid[1]
                dnull_key = partial_null_key.sum(dim=(0, 2))
                dnull_bias = partial_null_bias.sum(dim=(0, 2))
            else:
                # Autotuning may choose any tile: reserve and zero enough slots
                # for the smallest tile, and reduce untouched slots as zeros.
                query_blocks = triton.cdiv(seq_len - ctx.query_start, _DSQG_MIN_AUTOTUNE_BLOCK_N)
                partial_null_key = torch.zeros(
                    (batch, heads, query_blocks, head_dim), device=q.device, dtype=torch.float32
                )
                partial_null_bias = torch.zeros(
                    (batch, heads, query_blocks), device=q.device, dtype=torch.float32
                )
                query_grid = lambda meta: (
                    batch * heads,
                    triton.cdiv(seq_len - ctx.query_start, meta["BLOCK_N"]),
                )
                _bwd_dq_v23[query_grid](
                    q, k, v, pos_bias, scale_embed, null_key, null_bias,
                    dout, lse, delta, dq, partial_null_key, partial_null_bias, offsets_dev,
                    log_valid_count,
                    *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
                    *lse.stride(), *delta.stride(), *dq.stride(),
                    *pos_bias.stride(), *scale_embed.stride(), *null_key.stride(),
                    BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                    BLOCK_HD=ctx.block_hd, J_VAL=ctx.j_val,
                    KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                    QUERY_START=ctx.query_start,
                    GRAD_BLOCKS=query_blocks,
                )
                dnull_key = partial_null_key.sum(dim=(0, 2))
                dnull_bias = partial_null_bias.sum(dim=(0, 2))

        if ctx.source_end > 0:
            kv_heads = heads // ctx.kv_head_group_size
            if ctx.use_bands:
                source_blocks = sum(
                    triton.cdiv(stop - start, _band_launch_geometry(active)[0])
                    for start, stop, active in ctx.source_bands
                )
                # Qualified combined-v4 path: allocate only live
                # (band, block, active-offset) partials instead of the rectangular
                # [all_blocks, J] workspace, then reduce them in a fixed order.
                compact = bool(head_dim == ctx.block_hd)
                compact_blocks = tuple(
                    triton.cdiv(stop - start, _band_launch_geometry(active)[0])
                    for start, stop, active in ctx.source_bands
                )
                compact_actives = tuple(active for _, _, active in ctx.source_bands)
                compact_starts_list: list[int] = []
                entries = 0
                for block_count, active in zip(
                    compact_blocks, compact_actives, strict=True
                ):
                    compact_starts_list.append(entries)
                    entries += block_count * active
                compact_starts = tuple(compact_starts_list)
                if compact:
                    partial_pos_bias = torch.empty(
                        (batch, heads, entries), device=q.device, dtype=torch.float32
                    )
                    partial_scale_embed = torch.empty(
                        (batch, heads, entries, head_dim),
                        device=q.device, dtype=torch.float32,
                    )
                else:
                    partial_pos_bias = torch.zeros(
                        (batch, heads, source_blocks, ctx.j_val),
                        device=q.device, dtype=torch.float32,
                    )
                    partial_scale_embed = torch.zeros(
                        (batch, heads, source_blocks, ctx.j_val, head_dim),
                        device=q.device, dtype=torch.float32,
                    )
                source_block_offset = 0
                for source_start, source_stop, active in ctx.source_bands:
                    block_n, warps, stages = _band_launch_geometry(active)
                    grid = (
                        batch * kv_heads,
                        triton.cdiv(source_stop - source_start, block_n),
                    )
                    _bwd_dkdv_v23_banded[grid](
                        q, k, v, pos_bias, scale_embed, dout, lse, delta,
                        dk_accum, dv_accum, partial_pos_bias, partial_scale_embed, offsets_dev,
                        *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
                        *lse.stride(), *delta.stride(),
                        *dk_accum.stride(), *dv_accum.stride(),
                        *pos_bias.stride(), *scale_embed.stride(),
                        BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                        BLOCK_N=block_n, BLOCK_HD=ctx.block_hd, J_ACTIVE=active,
                        KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                        QUERY_START=ctx.query_start,
                        SOURCE_START=source_start, SOURCE_END=source_stop,
                        GRAD_BLOCKS=source_blocks, GRAD_BLOCK_OFFSET=source_block_offset,
                        GRAD_J=ctx.j_val, COMPACT_PARTIALS=compact,
                        GRAD_ENTRIES=entries, num_warps=warps, num_stages=stages,
                    )
                    source_block_offset += grid[1] * active if compact else grid[1]
                if compact:
                    dpos_bias, dscale_embed = _reduce_compact_shared_gradients_v23(
                        partial_pos_bias, partial_scale_embed,
                        batch=batch, heads=heads, j_val=ctx.j_val, head_dim=head_dim,
                        entries=entries, actives=compact_actives, blocks=compact_blocks,
                        starts=compact_starts,
                    )
                else:
                    dpos_bias = partial_pos_bias.sum(dim=(0, 2)).transpose(0, 1).contiguous()
                    dscale_embed = partial_scale_embed.sum(dim=(0, 1, 2))
            else:
                source_blocks = triton.cdiv(ctx.source_end, _DSQG_MIN_AUTOTUNE_BLOCK_N)
                partial_pos_bias = torch.zeros(
                    (batch, heads, source_blocks, ctx.j_val), device=q.device, dtype=torch.float32
                )
                partial_scale_embed = torch.zeros(
                    (batch, heads, source_blocks, ctx.j_val, head_dim), device=q.device, dtype=torch.float32
                )
                kv_grid = lambda meta: (
                    batch * kv_heads,
                    triton.cdiv(ctx.source_end, meta["BLOCK_N"]),
                )
                _bwd_dkdv_v23[kv_grid](
                    q, k, v, pos_bias, scale_embed, dout, lse, delta,
                    dk_accum, dv_accum, partial_pos_bias, partial_scale_embed, offsets_dev,
                    *q.stride(), *k.stride(), *v.stride(), *dout.stride(),
                    *lse.stride(), *delta.stride(),
                    *dk_accum.stride(), *dv_accum.stride(),
                    *pos_bias.stride(), *scale_embed.stride(),
                    BATCH=batch, H=heads, N=seq_len, HD=head_dim,
                    BLOCK_HD=ctx.block_hd, J_VAL=ctx.j_val,
                    KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                    QUERY_START=ctx.query_start, SOURCE_END=ctx.source_end,
                    GRAD_BLOCKS=source_blocks,
                )
                dpos_bias = partial_pos_bias.sum(dim=(0, 2)).transpose(0, 1).contiguous()
                dscale_embed = partial_scale_embed.sum(dim=(0, 1, 2))

        gradients = (
            dq, dk_accum, dv_accum, dpos_bias, dscale_embed,
            dnull_key.to(null_key.dtype), dnull_bias.to(null_bias.dtype),
            None, None, None, None, None, None, None,
        )
        return gradients[:ctx.input_count]


def _eager_dsqg_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    null_key: torch.Tensor,
    null_bias: torch.Tensor,
    offsets: list[int] | tuple[int, ...] | torch.Tensor,
    log_valid_count: torch.Tensor | None = None,
    *,
    query_start: int = 0,
) -> torch.Tensor:
    """Differentiable reference implementation."""
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
    if null_key.shape != (query_heads, head_dim):
        raise ValueError("null_key must have [Hq,HD]")
    if null_bias.shape != (query_heads,):
        raise ValueError("null_bias must have [Hq]")

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

    # The zero-valued null source lets attention abstain from retrieval.
    if log_valid_count is None:
        valid_count = torch.stack(valid_columns, dim=-1).sum(-1).clamp_min(1)
        log_count = valid_count.float().log()
    else:
        if log_valid_count.ndim != 1 or log_valid_count.numel() < seq_len:
            raise ValueError("log_valid_count must cover every query position")
        log_count = log_valid_count[:seq_len].to(device=q.device, dtype=torch.float32)
    null_score = (
        q.float() * null_key.float().reshape(1, query_heads, 1, head_dim)
    ).sum(-1) * scale
    null_score = (
        null_score
        + null_bias.float().reshape(1, query_heads, 1)
        + log_count.reshape(1, 1, seq_len)
    )
    score_columns.append(null_score)
    value_columns.append(torch.zeros_like(value_expanded.float()))
    valid_columns.append(torch.ones_like(positions, dtype=torch.bool))

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


def _dsqg_attention_v23_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    null_key: torch.Tensor,
    null_bias: torch.Tensor,
    offsets_dev: torch.Tensor,
    log_valid_count: torch.Tensor,
    query_start: int = 0,
    *,
    source_end: int | None = None,
    backend: str = "auto",
    offsets_host: tuple[int, ...] | None = None,
    support_band_execution: bool = True,
) -> torch.Tensor:
    """Select the eager or Triton implementation."""
    backend = str(backend).lower()
    if backend not in {"auto", "triton", "eager"}:
        raise ValueError("backend must be auto, triton, or eager")
    if backend == "eager":
        use_triton = False
    elif backend == "triton":
        if not q.is_cuda or not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton DSQG V23 was requested but CUDA/Triton is unavailable"
            )
        use_triton = True
    elif q.is_cuda:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "CUDA DSQG backend='auto' requires Triton; use backend='eager' "
                "only for an explicit reference/debug run"
            )
        use_triton = True
    else:
        use_triton = False

    if not use_triton:
        eager_offsets = offsets_host if offsets_host is not None else offsets_dev
        return _eager_dsqg_attention(
            q, k, v, pos_bias, scale_embed, null_key, null_bias, eager_offsets,
            log_valid_count, query_start=int(query_start),
        )

    original_dtype = q.dtype
    q_bf16 = q if q.dtype == torch.bfloat16 else q.to(torch.bfloat16)
    k_bf16 = k if k.dtype == torch.bfloat16 else k.to(torch.bfloat16)
    v_bf16 = v if v.dtype == torch.bfloat16 else v.to(torch.bfloat16)
    output = _DSQGV23Fn.apply(
        q_bf16,
        k_bf16,
        v_bf16,
        pos_bias.float(),
        scale_embed.float(),
        null_key.float(),
        null_bias.float(),
        log_valid_count,
        int(offsets_dev.numel()),
        offsets_dev,
        int(query_start),
        source_end,
        offsets_host,
        bool(support_band_execution),
    )
    return output.to(original_dtype)


def dsqg_attention_v23(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    null_key: torch.Tensor,
    null_bias: torch.Tensor,
    offsets_dev: torch.Tensor,
    query_start: int = 0,
    *,
    backend: str = "auto",
    support_band_execution: bool = True,
) -> torch.Tensor:
    """Dispatch V23 without the module-only source-projection crop.

    The Triton path computes with BF16 Q/K/V and follows PyTorch's BF16
    attention-gradient contract. Use ``backend="eager"`` for an FP32 reference.
    """
    positions = torch.arange(
        q.shape[2], device=offsets_dev.device, dtype=offsets_dev.dtype
    ).reshape(-1, 1)
    valid_count = (positions >= offsets_dev.reshape(1, -1)).sum(-1).clamp_min(1)
    log_valid_count = valid_count.to(torch.float32).log()
    return _dsqg_attention_v23_dispatch(
        q,
        k,
        v,
        pos_bias,
        scale_embed,
        null_key,
        null_bias,
        offsets_dev,
        log_valid_count,
        query_start,
        backend=backend,
        offsets_host=tuple(int(value) for value in offsets_dev.detach().cpu().tolist()),
        support_band_execution=support_band_execution,
    )


class DSQGAttentionV23(nn.Module):
    """Online-only DSQG attention with bounded non-content routing terms."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        offsets: list[int] | tuple[int, ...],
        seq_len: int = 2048,
        dropout: float = 0.1,
        pos_bias_scale: float | None = None,
        *,
        backend: str = "auto",
        scale_embed_init_std: float = 0.01,
        scale_embed_max_norm: float = 1.0,
        null_key_max_norm: float = 1.0,
        null_bias_limit: float = 6.0,
        pos_bias_max_slope: float = 0.75,
        pos_bias_residual_limit: float = 1.5,
        support_crop_projections: bool = True,
        support_crop_min_offset: int = 64,
        support_band_execution: bool = True,
        diagnostic_max_queries: int = 8,
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
        if not math.isfinite(float(scale_embed_max_norm)) or scale_embed_max_norm <= 0:
            raise ValueError("scale_embed_max_norm must be finite and positive")
        if not math.isfinite(float(null_key_max_norm)) or null_key_max_norm <= 0:
            raise ValueError("null_key_max_norm must be finite and positive")
        if not math.isfinite(float(null_bias_limit)) or null_bias_limit <= 0:
            raise ValueError("null_bias_limit must be finite and positive")
        if not math.isfinite(float(pos_bias_max_slope)) or pos_bias_max_slope <= 0:
            raise ValueError("pos_bias_max_slope must be finite and positive")
        if (
            not math.isfinite(float(pos_bias_residual_limit))
            or pos_bias_residual_limit <= 0
        ):
            raise ValueError("pos_bias_residual_limit must be finite and positive")
        crop_min = _strict_int(
            "support_crop_min_offset", support_crop_min_offset
        )
        if not isinstance(support_crop_projections, bool):
            raise TypeError("support_crop_projections must be bool")
        if not isinstance(support_band_execution, bool):
            raise TypeError("support_band_execution must be bool")
        diagnostic_max = _strict_int(
            "diagnostic_max_queries", diagnostic_max_queries, minimum=1
        )
        self.scale_embed_init_std = float(scale_embed_init_std)
        self.support_crop_projections = support_crop_projections
        self.support_crop_min_offset = crop_min
        self.support_band_execution = support_band_execution
        self.diagnostic_max_queries = diagnostic_max
        self._routing_diagnostics: dict[str, torch.Tensor] = {}
        self.register_buffer(
            "scale_embed_max_norm",
            torch.tensor(float(scale_embed_max_norm), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "null_key_max_norm",
            torch.tensor(float(null_key_max_norm), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "null_bias_limit",
            torch.tensor(float(null_bias_limit), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "pos_bias_max_slope",
            torch.tensor(float(pos_bias_max_slope), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "pos_bias_residual_limit",
            torch.tensor(float(pos_bias_residual_limit), dtype=torch.float32),
            persistent=True,
        )

        canonical_offsets = [
            offset
            for offset in _canonicalize_offsets(tuple(offsets))
            if offset < self.seq_len
        ]
        if not canonical_offsets:
            raise ValueError(
                f"DSQG has no live offsets below the configured seq_len={self.seq_len}"
            )
        self.offsets = tuple(canonical_offsets)
        self.j_val = len(canonical_offsets)
        self.minimum_offset = min(self.offsets)
        self.register_buffer(
            "offsets_dev",
            torch.tensor(canonical_offsets, dtype=torch.int32),
            persistent=False,
        )
        positions = torch.arange(self.seq_len, dtype=torch.int32).reshape(-1, 1)
        offset_table = torch.tensor(canonical_offsets, dtype=torch.int32).reshape(1, -1)
        valid_count = (positions >= offset_table).sum(-1).clamp_min(1)
        self.register_buffer(
            "log_valid_count",
            valid_count.to(torch.float32).log(),
            persistent=False,
        )

        if pos_bias_scale is None:
            pos_bias_scale = 0.25
        if not math.isfinite(float(pos_bias_scale)) or float(pos_bias_scale) < 0.0:
            raise ValueError("pos_bias_scale must be finite and non-negative")
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

        # Geometric initialization provides both distance-neutral and selective heads.
        largest_initial_slope = min(0.5, float(pos_bias_max_slope) * 0.8)
        smallest_initial_slope = min(0.02, largest_initial_slope)
        initial_slopes = torch.exp(
            torch.linspace(
                math.log(smallest_initial_slope),
                math.log(largest_initial_slope),
                heads,
            )
        )
        maximum = torch.tensor(float(pos_bias_max_slope))
        unbounded_slopes = maximum * torch.atanh(
            (initial_slopes / maximum).clamp_max(1.0 - 1e-6)
        )
        self.pos_bias_log_slope = nn.Parameter(
            _inverse_softplus(unbounded_slopes)
        )
        self.pos_bias_residual = nn.Parameter(torch.zeros(self.j_val, heads))
        self.scale_embed = nn.Parameter(
            torch.randn(self.j_val, self.head_dim) * self.scale_embed_init_std
        )
        with torch.no_grad():
            self.scale_embed.sub_(self.scale_embed.mean(0, keepdim=True))
        self.null_key = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.null_bias = nn.Parameter(torch.zeros(heads))
        # The bounded gain initializes exactly to one.
        self.if_gain = nn.Parameter(torch.ones(heads))
        self.dropout = nn.Dropout(float(dropout))

    @property
    def bounded_if_gain(self) -> torch.Tensor:
        return 1.0 + torch.tanh(self.if_gain - 1.0)

    @property
    def pos_bias_slope(self) -> torch.Tensor:
        maximum = self.pos_bias_max_slope.to(
            device=self.pos_bias_log_slope.device,
            dtype=self.pos_bias_log_slope.dtype,
        )
        return maximum * torch.tanh(F.softplus(self.pos_bias_log_slope) / maximum)

    @property
    def bounded_pos_bias_residual(self) -> torch.Tensor:
        limit = self.pos_bias_residual_limit.to(
            device=self.pos_bias_residual.device,
            dtype=self.pos_bias_residual.dtype,
        )
        return limit * torch.tanh(self.pos_bias_residual / limit)

    @property
    def pos_bias(self) -> torch.Tensor:
        log_distance = torch.log1p(
            self.offsets_dev.to(
                device=self.pos_bias_residual.device,
                dtype=self.pos_bias_residual.dtype,
            )
        ).reshape(-1, 1)
        slope = self.pos_bias_slope.reshape(1, -1)
        return (
            -log_distance * slope + self.bounded_pos_bias_residual
        ) * self.pos_bias_scale

    @property
    def centered_scale_embed(self) -> torch.Tensor:
        # Center and bound virtual keys in FP32.
        values = self.scale_embed.float()
        centered = values - values.mean(dim=0, keepdim=True)
        norm = centered.norm(dim=-1, keepdim=True)
        maximum = self.scale_embed_max_norm.to(centered.device)
        radial_scale = torch.where(
            norm > 1e-12,
            maximum * torch.tanh(norm / maximum) / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        bounded = centered * radial_scale
        bounded = bounded - bounded.mean(dim=0, keepdim=True)
        peak = bounded.norm(dim=-1).amax()
        common_scale = torch.clamp(
            maximum / peak.clamp_min(1e-12), max=1.0
        )
        return bounded * common_scale

    @property
    def bounded_null_key(self) -> torch.Tensor:
        values = self.null_key.float()
        norm = values.norm(dim=-1, keepdim=True)
        maximum = self.null_key_max_norm.to(values.device)
        radial_scale = torch.where(
            norm > 1e-12,
            maximum * torch.tanh(norm / maximum) / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        return values * radial_scale

    @property
    def bounded_null_bias(self) -> torch.Tensor:
        limit = self.null_bias_limit.to(
            device=self.null_bias.device, dtype=self.null_bias.dtype
        )
        return limit * torch.tanh(self.null_bias / limit)

    def semantic_config(self) -> dict[str, object]:
        return {
            "implementation": "dsqg-v23-bounded-routing",
            "offsets": self.offsets,
            "offset_loop": "single_j_val_with_precomputed_null_count",
            "positional_bias": "bounded_analytic_log_plus_bounded_residual",
            "pos_bias_scale": float(self.pos_bias_scale.detach().cpu()),
            "pos_bias_max_slope": float(self.pos_bias_max_slope.detach().cpu()),
            "pos_bias_residual_limit": float(
                self.pos_bias_residual_limit.detach().cpu()
            ),
            "scale_embed": "exact_centered_fp32_smooth_norm_capped_virtual_key",
            "scale_embed_max_norm": float(self.scale_embed_max_norm.detach().cpu()),
            "null_candidate": "count_calibrated_bounded_key_bias_zero_value",
            "null_neutral_prior_mass": 0.5,
            "null_key_max_norm": float(self.null_key_max_norm.detach().cpu()),
            "null_bias_limit": float(self.null_bias_limit.detach().cpu()),
            "if_gain": "bounded_0_to_2",
        }

    @torch.no_grad()
    def routing_diagnostics(self) -> dict[str, torch.Tensor]:
        scale_norm = self.centered_scale_embed.norm(dim=-1)
        residual = self.bounded_pos_bias_residual
        diagnostics = {
            "pos_slope_min": self.pos_bias_slope.min(),
            "pos_slope_mean": self.pos_bias_slope.mean(),
            "pos_slope_max": self.pos_bias_slope.max(),
            "pos_residual_rms": residual.square().mean().sqrt(),
            "scale_embed_norm_mean": scale_norm.mean(),
            "scale_embed_norm_max": scale_norm.max(),
            "null_key_norm_mean": self.bounded_null_key.norm(dim=-1).mean(),
            "null_key_norm_max": self.bounded_null_key.norm(dim=-1).max(),
            "null_bias_min": self.bounded_null_bias.min(),
            "null_bias_mean": self.bounded_null_bias.mean(),
            "null_bias_max": self.bounded_null_bias.max(),
            "if_gain_mean": self.bounded_if_gain.mean(),
        }
        diagnostics.update(self._routing_diagnostics)
        return diagnostics

    @torch.no_grad()
    def _sampled_live_diagnostics(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        content_gate: torch.Tensor,
        query_start: int,
    ) -> dict[str, torch.Tensor]:
        """Measure actual routing mass on a bounded set of query positions."""
        _, heads, seq_len, head_dim = query.shape
        candidate_count = max(0, seq_len - int(query_start))
        sample_count = min(self.diagnostic_max_queries, candidate_count)
        if sample_count <= 0:
            zero = torch.zeros((), device=query.device, dtype=torch.float32)
            return {
                "live_query_samples": zero,
                "null_attention_mass": zero,
                "local_offset_mass": zero,
                "mid_offset_mass": zero,
                "long_offset_mass": zero,
                "content_gate_mean": zero,
            }
        if sample_count == candidate_count:
            sample_ids = torch.arange(
                query_start, seq_len, device=query.device, dtype=torch.long
            )
        else:
            sample_ids = torch.linspace(
                query_start, seq_len - 1, sample_count, device=query.device
            ).round().to(torch.long)
        sampled_query = query[:, :, sample_ids].float()
        scale = 1.0 / math.sqrt(float(head_dim))
        virtual_keys = self.centered_scale_embed
        positional_bias = self.pos_bias
        null_key = self.bounded_null_key
        null_bias = self.bounded_null_bias
        score_columns: list[torch.Tensor] = []
        valid_columns: list[torch.Tensor] = []
        for logical_index, offset in enumerate(self.offsets):
            source = sample_ids - int(offset)
            valid = source >= 0
            selected_key = key[:, :, source.clamp_min(0)].float()
            score = (
                sampled_query
                * (selected_key + virtual_keys[logical_index].float())
            ).sum(-1) * scale
            score = score + positional_bias[logical_index].float().reshape(1, heads, 1)
            score_columns.append(
                score.masked_fill(~valid.reshape(1, 1, -1), float("-inf"))
            )
            valid_columns.append(valid)
        null_score = (
            sampled_query
            * null_key.float().reshape(1, heads, 1, head_dim)
        ).sum(-1) * scale
        null_score = (
            null_score
            + null_bias.float().reshape(1, heads, 1)
            + self.log_valid_count[sample_ids].float().reshape(1, 1, -1)
        )
        scores = torch.stack((*score_columns, null_score), dim=-1)
        probability = torch.softmax(scores, dim=-1)
        offset_probability = probability[..., :-1]
        offsets = torch.tensor(self.offsets, device=query.device)
        local = offsets <= 8
        middle = (offsets > 8) & (offsets < 64)
        long = offsets >= 64

        def mass(mask: torch.Tensor) -> torch.Tensor:
            # Summing an empty selected dimension yields exact zeros without a
            # CUDA host synchronization.
            return offset_probability[..., mask].sum(-1).mean()

        gate = torch.sigmoid(content_gate[:, sample_ids]).float().mean()
        return {
            "live_query_samples": torch.tensor(
                float(sample_count), device=query.device
            ),
            "null_attention_mass": probability[..., -1].mean(),
            "local_offset_mass": mass(local),
            "mid_offset_mass": mass(middle),
            "long_offset_mass": mass(long),
            "content_gate_mean": gate,
        }

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "online_softmax": True,
            "kernel_schedule": (
                "support_banded_static"
                if self.support_band_execution
                else "triton_autotune"
            ),
            "autotune_axes": (
                ()
                if self.support_band_execution
                else ("BLOCK_N", "num_warps", "num_stages")
            ),
            "support_band_execution": self.support_band_execution,
            "compact_shared_gradient_partials": True,
            "support_band_boundaries": (64, 256, 512, 1024, 1536),
            "dkdv_schedule": "kv_head_owned",
            "query_support_start": self.minimum_offset,
            "source_gradient_end": "runtime_seq_len_minus_minimum_offset",
            "configured_sequence_bound": self.seq_len,
            "support_crop_projections": self.support_crop_projections,
            "support_crop_min_offset": self.support_crop_min_offset,
            "null_count_calibration": "precomputed_position_table",
            "triton_qkv_gradient_storage": "bf16_input_dtype_with_fp32_accumulation",
            "diagnostic_max_queries": self.diagnostic_max_queries,
        }

    def forward(
        self, x: torch.Tensor, *, collect_diagnostics: bool = False
    ) -> torch.Tensor:
        compiling = torch.compiler.is_compiling()
        if not compiling:
            self._routing_diagnostics = {}
        batch, seq_len, dimension = x.shape
        if seq_len > self.seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds configured sequence bound "
                f"{self.seq_len}"
            )
        heads, head_dim = self.num_heads, self.head_dim
        query_start = min(self.minimum_offset, seq_len)
        key_end = max(0, seq_len - query_start)
        if query_start >= seq_len:
            return x * 0.0

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

        output = _dsqg_attention_v23_dispatch(
            query,
            key,
            value,
            self.pos_bias,
            self.centered_scale_embed,
            self.bounded_null_key,
            self.bounded_null_bias,
            self.offsets_dev,
            self.log_valid_count,
            query_start,
            source_end=key_end,
            backend=self.backend,
            offsets_host=self.offsets,
            support_band_execution=self.support_band_execution,
        )
        if collect_diagnostics and not compiling:
            self._routing_diagnostics = {
                key: value.detach()
                for key, value in self._sampled_live_diagnostics(
                    query, key, content_gate, query_start
                ).items()
            }
        output = output * self.bounded_if_gain.reshape(1, heads, 1, 1)
        flattened = output.permute(0, 2, 1, 3).reshape(batch, seq_len, dimension)
        return self.dropout(
            self.out_proj(flattened * torch.sigmoid(content_gate))
        )
