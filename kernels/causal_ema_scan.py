"""Causal EMA scan with a compact reverse-adjoint backward.

Forward recurrence::

    y[t] = alpha * x[t] + (1 - alpha) * y[t - 1],  y[-1] = 0

Reverse recurrence::

    lambda[t] = dy[t] + (1 - alpha) * lambda[t + 1]
    dx[t]     = alpha * lambda[t]
    dalpha    = sum_t lambda[t] * (x[t] - y[t - 1])

The CUDA path uses three backward stages rather than a separate forward-
sensitivity scan and seven-field auxiliary tensor. CPU-only installations
remain importable and use an exact differentiable reference implementation.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only development
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

BLOCK_D = 64
BLOCK_T = 64


def bounded_ema_factor(
    raw: torch.Tensor,
    *,
    floor: float = 1e-5,
    ceiling: float = 0.5,
) -> torch.Tensor:
    """Smoothly map an unconstrained scalar into ``(floor, ceiling)``."""
    floor, ceiling = float(floor), float(ceiling)
    if not (math.isfinite(floor) and math.isfinite(ceiling) and 0 < floor < ceiling < 1):
        raise ValueError("EMA bounds must satisfy 0 < floor < ceiling < 1")
    return floor + (ceiling - floor) * torch.sigmoid(raw)


def inverse_bounded_ema_factor(
    alpha: float,
    *,
    floor: float = 1e-5,
    ceiling: float = 0.5,
) -> float:
    """Return the raw sigmoid parameter corresponding to ``alpha``."""
    alpha, floor, ceiling = float(alpha), float(floor), float(ceiling)
    if not floor < alpha < ceiling:
        raise ValueError(f"alpha must be strictly inside ({floor}, {ceiling})")
    probability = (alpha - floor) / (ceiling - floor)
    return math.log(probability / (1.0 - probability))


def _num_stages() -> int:
    if not torch.cuda.is_available():
        return 2
    major, minor = torch.cuda.get_device_capability()
    return 4 if ((major == 9 and minor == 0) or major > 9) else 2


if _TRITON_AVAILABLE:

    @triton.jit
    def _fwd_local(
        X, Y, A, END,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        sxb, sxn, sxd, syb, syn, syd,
        BT: tl.constexpr, BD: tl.constexpr,
    ):
        pid, tb = tl.program_id(0), tl.program_id(1)
        ndb = tl.cdiv(D, BD)
        b, db = pid // ndb, pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        alpha = tl.load(A).to(tl.float32)
        decay = 1.0 - alpha
        state = tl.zeros([BD], tl.float32)
        n0 = tb * BT
        for token_offset in range(BT):
            token = n0 + token_offset
            valid = (token < N) & dim_mask
            value = tl.load(
                X + b * sxb + token * sxn + dims * sxd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            state = tl.where(token < N, alpha * value + decay * state, state)
            tl.store(Y + b * syb + token * syn + dims * syd, state, mask=valid)
        offset = (pid * NT + tb) * BD + tl.arange(0, BD)
        tl.store(END + offset, state, mask=dim_mask)

    @triton.jit
    def _fwd_carry(
        END, CARRY, A,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        BT: tl.constexpr, BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        db = pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        decay = 1.0 - tl.load(A).to(tl.float32)
        carry = tl.zeros([BD], tl.float32)
        for tb in range(NT):
            offset = (pid * NT + tb) * BD + tl.arange(0, BD)
            tl.store(CARRY + offset, carry, mask=dim_mask)
            local_end = tl.load(END + offset, mask=dim_mask, other=0.0).to(tl.float32)
            valid_len = tl.minimum(BT, N - tb * BT)
            power = tl.full((), 1.0, tl.float32)
            for token_offset in range(BT):
                power = tl.where(token_offset < valid_len, power * decay, power)
            carry = local_end + power * carry

    @triton.jit
    def _fwd_apply(
        Y, CARRY, A,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        syb, syn, syd,
        BT: tl.constexpr, BD: tl.constexpr,
    ):
        pid, tb = tl.program_id(0), tl.program_id(1)
        ndb = tl.cdiv(D, BD)
        b, db = pid // ndb, pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        decay = 1.0 - tl.load(A).to(tl.float32)
        offset = (pid * NT + tb) * BD + tl.arange(0, BD)
        carry = tl.load(CARRY + offset, mask=dim_mask, other=0.0).to(tl.float32)
        power = decay
        n0 = tb * BT
        for token_offset in range(BT):
            token = n0 + token_offset
            valid = (token < N) & dim_mask
            value = tl.load(
                Y + b * syb + token * syn + dims * syd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                Y + b * syb + token * syn + dims * syd,
                value + power * carry,
                mask=valid,
            )
            power *= decay

    @triton.jit
    def _bwd_local(
        DY, A, START,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        sgb, sgn, sgd,
        BT: tl.constexpr, BD: tl.constexpr,
    ):
        pid, tb = tl.program_id(0), tl.program_id(1)
        ndb = tl.cdiv(D, BD)
        b, db = pid // ndb, pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        decay = 1.0 - tl.load(A).to(tl.float32)
        state = tl.zeros([BD], tl.float32)
        n0 = tb * BT
        for reverse_offset in range(BT):
            token = n0 + BT - 1 - reverse_offset
            valid = (token < N) & dim_mask
            gradient = tl.load(
                DY + b * sgb + token * sgn + dims * sgd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            state = tl.where(token < N, gradient + decay * state, state)
        offset = (pid * NT + tb) * BD + tl.arange(0, BD)
        tl.store(START + offset, state, mask=dim_mask)

    @triton.jit
    def _bwd_carry(
        START, AFTER, A,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        BT: tl.constexpr, BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        db = pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        decay = 1.0 - tl.load(A).to(tl.float32)
        future = tl.zeros([BD], tl.float32)
        for block_offset in range(NT):
            tb = NT - 1 - block_offset
            offset = (pid * NT + tb) * BD + tl.arange(0, BD)
            tl.store(AFTER + offset, future, mask=dim_mask)
            local = tl.load(START + offset, mask=dim_mask, other=0.0).to(tl.float32)
            valid_len = tl.minimum(BT, N - tb * BT)
            power = tl.full((), 1.0, tl.float32)
            for token_offset in range(BT):
                power = tl.where(token_offset < valid_len, power * decay, power)
            future = local + power * future

    @triton.jit
    def _bwd_apply(
        X, Y, DY, DX, AFTER, A, DA,
        N: tl.constexpr, D: tl.constexpr, NT: tl.constexpr,
        sxb, sxn, sxd, syb, syn, syd,
        sgb, sgn, sgd, sdb, sdn, sdd,
        BT: tl.constexpr, BD: tl.constexpr,
        COMPUTE_DA: tl.constexpr,
    ):
        pid, tb = tl.program_id(0), tl.program_id(1)
        ndb = tl.cdiv(D, BD)
        b, db = pid // ndb, pid % ndb
        dims = db * BD + tl.arange(0, BD)
        dim_mask = dims < D
        alpha = tl.load(A).to(tl.float32)
        decay = 1.0 - alpha
        offset = (pid * NT + tb) * BD + tl.arange(0, BD)
        state = tl.load(AFTER + offset, mask=dim_mask, other=0.0).to(tl.float32)
        if COMPUTE_DA:
            da_vector = tl.zeros([BD], tl.float32)
        n0 = tb * BT
        for reverse_offset in range(BT):
            token = n0 + BT - 1 - reverse_offset
            valid = (token < N) & dim_mask
            gradient = tl.load(
                DY + b * sgb + token * sgn + dims * sgd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                X + b * sxb + token * sxn + dims * sxd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            previous = tl.load(
                Y + b * syb + (token - 1) * syn + dims * syd,
                mask=(token > 0) & valid,
                other=0.0,
            ).to(tl.float32)
            state = tl.where(token < N, gradient + decay * state, state)
            tl.store(
                DX + b * sdb + token * sdn + dims * sdd,
                alpha * state,
                mask=valid,
            )
            if COMPUTE_DA:
                da_vector += tl.where(valid, state * (value - previous), 0.0)
        if COMPUTE_DA:
            tl.atomic_add(
                DA,
                tl.sum(tl.where(dim_mask, da_vector, 0.0), axis=0),
                sem="relaxed",
            )


class _CausalEMAFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha_input: torch.Tensor,
        floor: float,
        ceiling: float,
    ) -> torch.Tensor:
        x = x.contiguous()
        batch, seq_len, width = x.shape
        alpha = alpha_input.detach().clamp(floor, ceiling).to(
            device=x.device,
            dtype=torch.float32,
        ).reshape(1)
        output = torch.empty_like(x)
        dimension_blocks = triton.cdiv(width, BLOCK_D)
        time_blocks = triton.cdiv(seq_len, BLOCK_T)
        programs = batch * dimension_blocks
        states = torch.empty(
            (programs, time_blocks, BLOCK_D),
            device=x.device,
            dtype=torch.float32,
        )
        carries = torch.empty_like(states)
        stages = _num_stages()
        _fwd_local[(programs, time_blocks)](
            x, output, alpha, states,
            seq_len, width, time_blocks,
            *x.stride(), *output.stride(),
            BT=BLOCK_T, BD=BLOCK_D,
            num_warps=4, num_stages=stages,
        )
        _fwd_carry[(programs,)](
            states, carries, alpha,
            seq_len, width, time_blocks,
            BT=BLOCK_T, BD=BLOCK_D,
            num_warps=4, num_stages=1,
        )
        _fwd_apply[(programs, time_blocks)](
            output, carries, alpha,
            seq_len, width, time_blocks,
            *output.stride(),
            BT=BLOCK_T, BD=BLOCK_D,
            num_warps=4, num_stages=stages,
        )
        ctx.save_for_backward(x, output, alpha_input, alpha)
        ctx.floor = float(floor)
        ctx.ceiling = float(ceiling)
        ctx.time_blocks = int(time_blocks)
        return output

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, output, alpha_input, alpha = ctx.saved_tensors
        dy = dy.contiguous()
        batch, seq_len, width = x.shape
        dimension_blocks = triton.cdiv(width, BLOCK_D)
        time_blocks = ctx.time_blocks
        programs = batch * dimension_blocks
        states = torch.empty(
            (programs, time_blocks, BLOCK_D),
            device=x.device,
            dtype=torch.float32,
        )
        carries = torch.empty_like(states)
        dx = torch.empty_like(dy)
        need_da = bool(ctx.needs_input_grad[1])
        da = torch.zeros_like(alpha)
        stages = _num_stages()
        _bwd_local[(programs, time_blocks)](
            dy, alpha, states,
            seq_len, width, time_blocks,
            *dy.stride(),
            BT=BLOCK_T, BD=BLOCK_D,
            num_warps=4, num_stages=stages,
        )
        _bwd_carry[(programs,)](
            states, carries, alpha,
            seq_len, width, time_blocks,
            BT=BLOCK_T, BD=BLOCK_D,
            num_warps=4, num_stages=1,
        )
        _bwd_apply[(programs, time_blocks)](
            x, output, dy, dx, carries, alpha, da,
            seq_len, width, time_blocks,
            *x.stride(), *output.stride(), *dy.stride(), *dx.stride(),
            BT=BLOCK_T, BD=BLOCK_D,
            COMPUTE_DA=need_da,
            num_warps=4, num_stages=stages,
        )
        if not need_da:
            return dx, None, None, None
        da = da.reshape_as(alpha_input).to(alpha_input.dtype)
        # This projected clamp derivative matters only for legacy callers that
        # pass an already-bounded parameter directly. New callers normally use
        # bounded_ema_factor(raw), so alpha stays strictly inside the interval.
        blocked = (
            ((alpha_input <= ctx.floor) & (da > 0))
            | ((alpha_input >= ctx.ceiling) & (da < 0))
        )
        return dx, torch.where(blocked, torch.zeros_like(da), da), None, None



if _TRITON_AVAILABLE:

    @triton.jit
    def _reset_fwd_serial(
        X, RESET, Y, A,
        N, D: tl.constexpr,
        sxb, sxn, sxd, srb, srn, syb, syn, syd,
        BD: tl.constexpr,
    ):
        """Reset-aware causal EMA. One program owns one B x D tile.

        Resets are sparse in packed language-model streams, so this path keeps
        the state on-chip and avoids the former CUDA-to-eager fallback. It is
        intentionally a serial-in-time fallback; the no-reset path retains the
        faster block-associative scan above.
        """
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        batch = pid // ndb
        dblock = pid % ndb
        dims = dblock * BD + tl.arange(0, BD)
        dmask = dims < D
        alpha = tl.load(A).to(tl.float32)
        decay = 1.0 - alpha
        state = tl.zeros([BD], tl.float32)
        for token in tl.range(0, N, num_stages=1):
            reset = tl.load(RESET + batch * srb + token * srn).to(tl.int1)
            state = tl.where(reset, tl.zeros([BD], tl.float32), state)
            value = tl.load(
                X + batch * sxb + token * sxn + dims * sxd,
                mask=dmask, other=0.0,
            ).to(tl.float32)
            state = alpha * value + decay * state
            tl.store(
                Y + batch * syb + token * syn + dims * syd,
                state, mask=dmask,
            )


    @triton.jit
    def _reset_bwd_serial(
        X, Y, DY, RESET, DX, A, DA,
        N, D: tl.constexpr,
        sxb, sxn, sxd, syb, syn, syd,
        sgb, sgn, sgd, srb, srn, sdb, sdn, sdd,
        BD: tl.constexpr, COMPUTE_DA: tl.constexpr,
    ):
        """Exact reverse adjoint for the reset-aware recurrence."""
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        batch = pid // ndb
        dblock = pid % ndb
        dims = dblock * BD + tl.arange(0, BD)
        dmask = dims < D
        alpha = tl.load(A).to(tl.float32)
        decay = 1.0 - alpha
        adjoint = tl.zeros([BD], tl.float32)
        if COMPUTE_DA:
            da_vector = tl.zeros([BD], tl.float32)
        for reverse_offset in tl.range(0, N, num_stages=1):
            token = N - 1 - reverse_offset
            reset_next = tl.load(
                RESET + batch * srb + (token + 1) * srn,
                mask=token + 1 < N, other=1,
            ).to(tl.int1)
            future = tl.where(reset_next, tl.zeros([BD], tl.float32), adjoint)
            gradient = tl.load(
                DY + batch * sgb + token * sgn + dims * sgd,
                mask=dmask, other=0.0,
            ).to(tl.float32)
            adjoint = gradient + decay * future
            value = tl.load(
                X + batch * sxb + token * sxn + dims * sxd,
                mask=dmask, other=0.0,
            ).to(tl.float32)
            reset_here = tl.load(RESET + batch * srb + token * srn).to(tl.int1)
            previous = tl.load(
                Y + batch * syb + (token - 1) * syn + dims * syd,
                mask=(token > 0) & (~reset_here) & dmask, other=0.0,
            ).to(tl.float32)
            tl.store(
                DX + batch * sdb + token * sdn + dims * sdd,
                alpha * adjoint, mask=dmask,
            )
            if COMPUTE_DA:
                da_vector += tl.where(dmask, adjoint * (value - previous), 0.0)
        if COMPUTE_DA:
            tl.atomic_add(DA, tl.sum(da_vector, axis=0), sem="relaxed")


class _CausalEMAResetFn(torch.autograd.Function):
    """CUDA reset-aware EMA without falling back to an eager Python loop."""

    @staticmethod
    def forward(ctx, x, alpha_input, reset_mask, floor, ceiling):
        x = x.contiguous()
        reset_mask = reset_mask.to(device=x.device, dtype=torch.bool).contiguous()
        batch, seq_len, width = x.shape
        if reset_mask.shape != (batch, seq_len):
            raise ValueError(f"reset_mask must have shape {(batch, seq_len)}")
        alpha = alpha_input.detach().clamp(float(floor), float(ceiling)).to(
            device=x.device, dtype=torch.float32
        ).reshape(1)
        output = torch.empty_like(x)
        programs = batch * triton.cdiv(width, BLOCK_D)
        _reset_fwd_serial[(programs,)](
            x, reset_mask, output, alpha,
            seq_len, D=width,
            *x.stride(), *reset_mask.stride(), *output.stride(),
            BD=BLOCK_D, num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(x, output, alpha_input, alpha, reset_mask)
        ctx.floor = float(floor)
        ctx.ceiling = float(ceiling)
        return output

    @staticmethod
    def backward(ctx, dy):
        x, output, alpha_input, alpha, reset_mask = ctx.saved_tensors
        dy = dy.contiguous()
        batch, seq_len, width = x.shape
        dx = torch.empty_like(dy)
        need_da = bool(ctx.needs_input_grad[1])
        da = torch.zeros_like(alpha)
        programs = batch * triton.cdiv(width, BLOCK_D)
        _reset_bwd_serial[(programs,)](
            x, output, dy, reset_mask, dx, alpha, da,
            seq_len, D=width,
            *x.stride(), *output.stride(), *dy.stride(),
            *reset_mask.stride(), *dx.stride(),
            BD=BLOCK_D, COMPUTE_DA=need_da,
            num_warps=4, num_stages=1,
        )
        if not need_da:
            return dx, None, None, None, None
        da = da.reshape_as(alpha_input).to(alpha_input.dtype)
        blocked = (
            ((alpha_input <= ctx.floor) & (da > 0))
            | ((alpha_input >= ctx.ceiling) & (da < 0))
        )
        d_alpha = torch.where(blocked, torch.zeros_like(da), da)
        return dx, d_alpha, None, None, None

def _reference(
    x: torch.Tensor,
    alpha_input: torch.Tensor,
    floor: float,
    ceiling: float,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    accumulator_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    alpha = alpha_input.clamp(floor, ceiling).to(accumulator_dtype)
    decay = 1.0 - alpha
    state = torch.zeros(
        x.shape[0],
        x.shape[2],
        device=x.device,
        dtype=accumulator_dtype,
    )
    xf = x.to(accumulator_dtype)
    if reset_mask is not None:
        if reset_mask.shape != x.shape[:2]:
            raise ValueError(f"reset_mask must have shape {tuple(x.shape[:2])}")
        reset_mask = reset_mask.to(device=x.device, dtype=torch.bool)
    rows = []
    for token in range(x.shape[1]):
        if reset_mask is not None:
            state = torch.where(reset_mask[:, token, None], torch.zeros_like(state), state)
        state = alpha * xf[:, token] + decay * state
        rows.append(state)
    return torch.stack(rows, dim=1).to(x.dtype) if rows else torch.empty_like(x)


def causal_ema_scan(
    x: torch.Tensor,
    ema_factor: torch.Tensor,
    floor: float = 1e-5,
    ceiling: float = 0.5,
    *,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a scalar-alpha causal EMA to ``x[B,N,D]``.

    New callers should pass ``bounded_ema_factor(raw_parameter)``. CUDA uses the
    fast block-associative scan for ordinary streams and an exact serial-state
    Triton kernel when boundary resets are requested. CPU uses the eager oracle.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,N,D], got {tuple(x.shape)}")
    if ema_factor.numel() != 1:
        raise ValueError("ema_factor must contain one scalar")
    if not (0 < float(floor) < float(ceiling) < 1):
        raise ValueError("EMA bounds must satisfy 0 < floor < ceiling < 1")
    if not x.is_cuda or not _TRITON_AVAILABLE:
        return _reference(x, ema_factor, float(floor), float(ceiling), reset_mask)
    if reset_mask is not None:
        return _CausalEMAResetFn.apply(
            x, ema_factor, reset_mask, float(floor), float(ceiling)
        )
    return _CausalEMAFn.apply(x, ema_factor, float(floor), float(ceiling))


if __name__ == "__main__":
    torch.manual_seed(1)
    x = torch.randn(2, 9, 7, dtype=torch.float64, requires_grad=True)
    raw = torch.tensor(
        inverse_bounded_ema_factor(0.05),
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda values, parameter: causal_ema_scan(values, bounded_ema_factor(parameter)),
        (x, raw),
    )
    print("CPU EMA gradcheck: PASS")
