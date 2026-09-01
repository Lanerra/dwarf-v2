"""Fused causal EMA recurrences for DWARF."""

from __future__ import annotations

import math

import torch

__all__ = (
    "bounded_ema_factor",
    "causal_ema_scan3",
    "inverse_bounded_ema_factor",
    "causal_ema_triton_available",
    "causal_ema_execution_config",
)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional on CPU
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

EMA_FLOOR = 1e-5
EMA_CEILING = 0.5
BLOCK_D = 64


def causal_ema_triton_available() -> bool:
    """Return whether the production Triton EMA backend is importable."""
    return bool(_TRITON_AVAILABLE)


def causal_ema_execution_config() -> dict[str, object]:
    return {
        "backend": "triton_autotune" if _TRITON_AVAILABLE else "cpu_reference",
        "dimension_blocks": (32, 64),
        "serial_sequence_scan": True,
        "fp32_recurrence_accumulation": True,
    }


def bounded_ema_factor(raw: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained parameter into the supported EMA range."""
    return EMA_FLOOR + (EMA_CEILING - EMA_FLOOR) * torch.sigmoid(raw)


def inverse_bounded_ema_factor(ema_factor: float) -> float:
    """Return the unconstrained parameter for an EMA factor."""
    ema_factor = float(ema_factor)
    if not EMA_FLOOR < ema_factor < EMA_CEILING:
        raise ValueError(
            f"ema_factor must be strictly inside ({EMA_FLOOR}, {EMA_CEILING})"
        )
    probability = (ema_factor - EMA_FLOOR) / (EMA_CEILING - EMA_FLOOR)
    return math.log(probability / (1.0 - probability))


if _TRITON_AVAILABLE:

    _EMA_AUTOTUNE_CONFIGS = [
        triton.Config({"BD": 32}, num_warps=2, num_stages=1),
        triton.Config({"BD": 32}, num_warps=4, num_stages=1),
        triton.Config({"BD": 64}, num_warps=4, num_stages=1),
    ]

    @triton.autotune(
        configs=_EMA_AUTOTUNE_CONFIGS,
        key=["N", "D"],
        cache_results=True,
    )
    @triton.jit
    def _ema3_fwd_serial(
        X,
        Y,
        A,
        N,
        D: tl.constexpr,
        sxb,
        sxn,
        sxd,
        syb,
        syn,
        syk,
        syd,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        batch, dblock = pid // ndb, pid % ndb
        dims = dblock * BD + tl.arange(0, BD)
        dmask = dims < D
        a0 = tl.load(A).to(tl.float32)
        a1 = tl.load(A + 1).to(tl.float32)
        a2 = tl.load(A + 2).to(tl.float32)
        q0, q1, q2 = 1.0 - a0, 1.0 - a1, 1.0 - a2
        s0 = tl.zeros([BD], tl.float32)
        s1 = tl.zeros([BD], tl.float32)
        s2 = tl.zeros([BD], tl.float32)
        for token in tl.range(0, N, num_stages=1):
            value = tl.load(
                X + batch * sxb + token * sxn + dims * sxd,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            tl.store(Y + batch * syb + token * syn + dims * syd, s0, mask=dmask)
            tl.store(
                Y + batch * syb + token * syn + syk + dims * syd, s1, mask=dmask
            )
            tl.store(
                Y + batch * syb + token * syn + 2 * syk + dims * syd,
                s2,
                mask=dmask,
            )
            s0 = a0 * value + q0 * s0
            s1 = a1 * value + q1 * s1
            s2 = a2 * value + q2 * s2

    @triton.autotune(
        configs=_EMA_AUTOTUNE_CONFIGS,
        key=["N", "D", "COMPUTE_DA"],
        reset_to_zero=["DA"],
        cache_results=True,
    )
    @triton.jit
    def _ema3_bwd_serial(
        X,
        Y,
        DY,
        DX,
        A,
        DA,
        N,
        D: tl.constexpr,
        sxb,
        sxn,
        sxd,
        syb,
        syn,
        syk,
        syd,
        sgb,
        sgn,
        sgk,
        sgd,
        sdb,
        sdn,
        sdd,
        BD: tl.constexpr,
        COMPUTE_DA: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        batch, dblock = pid // ndb, pid % ndb
        dims = dblock * BD + tl.arange(0, BD)
        dmask = dims < D
        a0 = tl.load(A).to(tl.float32)
        a1 = tl.load(A + 1).to(tl.float32)
        a2 = tl.load(A + 2).to(tl.float32)
        q0, q1, q2 = 1.0 - a0, 1.0 - a1, 1.0 - a2
        l0 = tl.zeros([BD], tl.float32)
        l1 = tl.zeros([BD], tl.float32)
        l2 = tl.zeros([BD], tl.float32)
        if COMPUTE_DA:
            da0 = tl.zeros([BD], tl.float32)
            da1 = tl.zeros([BD], tl.float32)
            da2 = tl.zeros([BD], tl.float32)
        for reverse_offset in tl.range(0, N, num_stages=1):
            token = N - 1 - reverse_offset
            g0 = tl.load(
                DY + batch * sgb + (token + 1) * sgn + dims * sgd,
                mask=(token + 1 < N) & dmask,
                other=0.0,
            ).to(tl.float32)
            g1 = tl.load(
                DY + batch * sgb + (token + 1) * sgn + sgk + dims * sgd,
                mask=(token + 1 < N) & dmask,
                other=0.0,
            ).to(tl.float32)
            g2 = tl.load(
                DY + batch * sgb + (token + 1) * sgn + 2 * sgk + dims * sgd,
                mask=(token + 1 < N) & dmask,
                other=0.0,
            ).to(tl.float32)
            l0 = g0 + q0 * l0
            l1 = g1 + q1 * l1
            l2 = g2 + q2 * l2
            p0 = tl.load(
                Y + batch * syb + token * syn + dims * syd,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            p1 = tl.load(
                Y + batch * syb + token * syn + syk + dims * syd,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            p2 = tl.load(
                Y + batch * syb + token * syn + 2 * syk + dims * syd,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                X + batch * sxb + token * sxn + dims * sxd,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                DX + batch * sdb + token * sdn + dims * sdd,
                a0 * l0 + a1 * l1 + a2 * l2,
                mask=dmask,
            )
            if COMPUTE_DA:
                da0 += tl.where(dmask, l0 * (value - p0), 0.0)
                da1 += tl.where(dmask, l1 * (value - p1), 0.0)
                da2 += tl.where(dmask, l2 * (value - p2), 0.0)
        if COMPUTE_DA:
            tl.atomic_add(DA, tl.sum(da0, axis=0), sem="relaxed")
            tl.atomic_add(DA + 1, tl.sum(da1, axis=0), sem="relaxed")
            tl.atomic_add(DA + 2, tl.sum(da2, axis=0), sem="relaxed")


class _CausalEMA3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ema_factors):
        x = x.contiguous()
        batch, seq_len, width = x.shape
        alpha = (
            ema_factors.detach()
            .to(device=x.device, dtype=torch.float32)
            .reshape(3)
            .contiguous()
        )
        output = torch.empty((batch, seq_len, 3, width), device=x.device, dtype=x.dtype)
        grid = lambda meta: (batch * triton.cdiv(width, meta["BD"]),)
        _ema3_fwd_serial[grid](
            x,
            output,
            alpha,
            seq_len,
            width,
            *x.stride(),
            *output.stride(),
        )
        ctx.save_for_backward(x, output, ema_factors, alpha)
        return output

    @staticmethod
    def backward(ctx, dy):
        x, output, ema_factors, alpha = ctx.saved_tensors
        dy = dy.contiguous()
        batch, seq_len, width = x.shape
        dx = torch.empty_like(x)
        need_da = bool(ctx.needs_input_grad[1])
        da = torch.zeros_like(alpha)
        grid = lambda meta: (batch * triton.cdiv(width, meta["BD"]),)
        _ema3_bwd_serial[grid](
            x,
            output,
            dy,
            dx,
            alpha,
            da,
            seq_len,
            width,
            *x.stride(),
            *output.stride(),
            *dy.stride(),
            *dx.stride(),
            COMPUTE_DA=need_da,
        )
        if not need_da:
            return dx, None
        return dx, da.to(ema_factors.dtype)


def _reference(
    x: torch.Tensor,
    ema_factors: torch.Tensor,
) -> torch.Tensor:
    accumulator_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    alpha = ema_factors.to(accumulator_dtype).reshape(1, 3, 1)
    state = torch.zeros(
        x.shape[0], 3, x.shape[2], device=x.device, dtype=accumulator_dtype
    )
    xf = x.to(accumulator_dtype)
    rows = []
    for token in range(x.shape[1]):
        rows.append(state)
        state = alpha * xf[:, token, None] + (1.0 - alpha) * state
    if not rows:
        return x.new_empty((x.shape[0], 0, 3, x.shape[2]))
    return torch.stack(rows, dim=1).to(x.dtype)


def causal_ema_scan3(
    x: torch.Tensor,
    ema_factors: torch.Tensor,
) -> torch.Tensor:
    """Apply three lagged causal EMA recurrences to ``x[B,N,D]``.

    Returns the preceding states as ``[B,N,3,D]``, with zeros at sequence
    boundaries.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,N,D], got {tuple(x.shape)}")
    if ema_factors.ndim != 1 or ema_factors.numel() != 3:
        raise ValueError("ema_factors must be a one-dimensional tensor of length 3")
    if ema_factors.device != x.device:
        raise ValueError("x and ema_factors must be on the same device")
    if not x.is_cuda:
        return _reference(x, ema_factors)
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "CUDA causal EMA execution requires Triton; the serial Python "
            "reference is CPU-only"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("CUDA causal EMA expects FP16, BF16, or FP32 input")
    return _CausalEMA3Fn.apply(x, ema_factors)
