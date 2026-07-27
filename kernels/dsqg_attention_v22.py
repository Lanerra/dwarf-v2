"""DSQG V22 under the checkpoint-compatible V19/V20/V21 public class names.

The CUDA path retains grouped sparse causal attention and four-plane MOVT, but
adds the production revisions used by the accompanying trainer:

* one FlashAttention-style ``D = sum(dO * O)`` precomputation per backward;
* FP32 native-GQA K/V accumulation;
* sparse rotated-channel corrections instead of full-head MOVT rewrites;
* fused Q/K/V/content-gate projection and a bias-free output projection;
* centered scale embeddings and analytic-log-plus-residual positional bias;
* magnitude-aware NPCI; and
* a differentiable eager oracle for CPU testing and CUDA parity checks;
* an online-softmax baseline forward with scale embeddings folded into virtual keys;
* support-aware query/key launch bounds and optional projection cropping; and
* a fused Triton phase-probe projection with an exact custom backward.

``DSQGAttentionV19`` and ``DSQGAttentionV20`` remain aliases for import
compatibility. Supplied V19/V20 module state dictionaries are migrated during
strict loading; the revised state dictionary itself intentionally reflects the
new fused and factorized parameters.
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

R_PLANES = 4
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


def _next_pow2(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _normalize_plane_shift(r_planes: int, hd: int, plane_shift: int = 0) -> int:
    """Validate the within-segment MOVT channel-pair shift used for staggering."""
    if isinstance(plane_shift, bool):
        raise TypeError("plane_shift must be an integer, got bool")
    try:
        shift = int(plane_shift.__index__())
    except AttributeError as exc:
        raise TypeError(f"plane_shift must be an integer, got {type(plane_shift).__name__}") from exc
    except TypeError as exc:
        raise TypeError(f"plane_shift must be an integer, got {type(plane_shift).__name__}") from exc
    if hd < 2 * r_planes:
        raise ValueError(f"head_dim={hd} must be >= 2*r_planes={2 * r_planes}")
    segment = max(2, hd // r_planes)
    if shift < 0 or shift > segment - 2:
        raise ValueError(
            f"plane_shift={shift} must keep each MOVT pair inside its hd/r segment; "
            f"allowed range is [0, {segment - 2}] for head_dim={hd}, r_planes={r_planes}"
        )
    return shift


def _phase_plane_channels(r_planes: int, hd: int, plane_shift: int = 0) -> list[tuple[int, int]]:
    """Spread MOVT plane pairs across the head dimension, with optional layer stagger."""
    shift = _normalize_plane_shift(r_planes, hd, plane_shift)
    stride = max(2, hd // r_planes)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for r in range(r_planes):
        a = min(r * stride + shift, hd - 2)
        b = a + 1
        if a in used or b in used:
            # Extremely small/odd head sizes can collide with the simple spread.
            # Fall back to the next free adjacent pair while preserving determinism.
            found = False
            for cand in range(0, hd - 1):
                if cand not in used and cand + 1 not in used:
                    a, b = cand, cand + 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not assign {r_planes} disjoint MOVT planes in head_dim={hd}")
        used.add(a); used.add(b)
        pairs.append((a, b))
    return pairs


def _canonicalize_offsets(offsets: list[int], j_small: int, j_large: int) -> tuple[list[int], int, int]:
    """Return offsets sorted into the kernel-required [small | large] partition."""
    vals = []
    for d in offsets:
        if isinstance(d, bool):
            raise TypeError(f"DSQG offsets must be integer positive causal deltas, got bool: {d!r}")
        try:
            ivalue = d.__index__()
        except AttributeError as exc:
            raise TypeError(
                f"DSQG offsets must be integer positive causal deltas, got {type(d).__name__}: {d!r}"
            ) from exc
        except TypeError as exc:
            raise TypeError(
                f"DSQG offsets must be integer positive causal deltas, got {type(d).__name__}: {d!r}"
            ) from exc
        if ivalue <= 0:
            raise ValueError(f"DSQG offsets must be positive causal deltas; got {ivalue}")
        vals.append(int(ivalue))
    if len(set(vals)) != len(vals):
        raise ValueError(f"Duplicate DSQG offsets are not supported: {vals}")
    middle = [d for d in vals if not (d <= 28 or d >= 48)]
    if middle:
        raise ValueError(f"Offsets must be <=28 or >=48 for the DSQG split; got middle offsets {middle}")
    small = sorted(d for d in vals if d <= 28)
    large = sorted(d for d in vals if d >= 48)
    if len(small) != int(j_small) or len(large) != int(j_large):
        raise ValueError(
            f"j_small/j_large mismatch after canonicalization: "
            f"declared ({j_small}, {j_large}) vs actual ({len(small)}, {len(large)})"
        )
    return small + large, len(small), len(large)


def _rms_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize along the last dimension; cheaper than full L2 normalize."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def calibrated_movt_phase_gain_std(
    *,
    head_dim: int,
    target_dynamic_rms: float,
    gate_logit: float = 0.0,
) -> float:
    """Return a phase-gain std targeting the initial content-angle RMS.

    RMS-normalized Q/K projected onto unit probes and scaled by 1/sqrt(HD)
    produce RMS(y*z) approximately 1/HD. Thus target ≈ gate*gain_std/HD.
    """
    head_dim = int(head_dim)
    target = float(target_dynamic_rms)
    gate_logit = float(gate_logit)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(
            f"target_dynamic_rms must be finite and positive, got {target_dynamic_rms}"
        )
    if not math.isfinite(gate_logit):
        raise ValueError(f"gate_logit must be finite, got {gate_logit}")
    gate = 1.0 / (1.0 + math.exp(-gate_logit))
    return target * head_dim / gate


def npci_rotate(
    x: torch.Tensor,
    x_delta: torch.Tensor,
    theta_h: torch.Tensor,
    *,
    strength_tau: float = 0.25,
) -> torch.Tensor:
    """Magnitude-aware norm-preserving NPCI rotation.

    The original path normalized every non-zero perpendicular delta, so a gate
    could alter direction but could not reliably reduce the intervention angle.
    The effective angle now scales with relative delta magnitude and becomes an
    exact no-op at zero.
    """
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


def _orthogonal_phase_probes(r_planes: int, hd: int, plane_shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic orthogonal q/k probes with a pi/2 phase offset per plane."""
    assert hd >= 2 * r_planes
    q = torch.zeros(r_planes, hd, dtype=torch.float32)
    k = torch.zeros(r_planes, hd, dtype=torch.float32)
    for r, (a, b) in enumerate(_phase_plane_channels(r_planes, hd, plane_shift=plane_shift)):
        angle = 2.0 * math.pi * r / max(r_planes, 1)
        q[r, a] = math.cos(angle)
        q[r, b] = math.sin(angle)
        k[r, a] = math.cos(angle + math.pi / 2.0)
        k[r, b] = math.sin(angle + math.pi / 2.0)
    return q, k




if _TRITON_AVAILABLE:

    @triton.jit
    def _phase_probe_forward_kernel(
        X, PROBES, OUT,
        sxb, sxh, sxn, sxd, spr, spd, sob, soh, son, sor,
        N, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr,
        START: tl.constexpr, END: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    ):
        bh = tl.program_id(0)
        block = tl.program_id(1)
        batch = bh // H
        head = bh % H
        positions = START + block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = positions < END
        dims = tl.arange(0, BLOCK_HD)
        dmask = dims < HD
        values = tl.load(
            X + batch * sxb + head * sxh
            + positions[:, None] * sxn + dims[None, :] * sxd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        inverse_norm = tl.rsqrt(
            tl.sum(values * values, axis=1) + 1.0e-6 * HD
        )
        for plane in range(R):
            probe = tl.load(
                PROBES + plane * spr + dims * spd,
                mask=dmask, other=0.0,
            ).to(tl.float32)
            projected = tl.sum(values * probe[None, :], axis=1) * inverse_norm
            tl.store(
                OUT + batch * sob + head * soh
                + positions * son + plane * sor,
                projected, mask=valid,
            )


    @triton.jit
    def _phase_probe_backward_kernel(
        X, PROBES, DOUT, DX, DPROBES,
        sxb, sxh, sxn, sxd, spr, spd,
        sgb, sgh, sgn, sgr, sdb, sdh, sdn, sdd,
        N, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr,
        START: tl.constexpr, END: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
        COMPUTE_DPROBES: tl.constexpr,
    ):
        bh = tl.program_id(0)
        block = tl.program_id(1)
        batch = bh // H
        head = bh % H
        positions = START + block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = positions < END
        dims = tl.arange(0, BLOCK_HD)
        dmask = dims < HD
        values = tl.load(
            X + batch * sxb + head * sxh
            + positions[:, None] * sxn + dims[None, :] * sxd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        norm_sq = tl.sum(values * values, axis=1) + 1.0e-6 * HD
        inverse_norm = tl.rsqrt(norm_sq)
        inverse_norm_cubed = inverse_norm * inverse_norm * inverse_norm
        dx = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)
        for plane in range(R):
            probe = tl.load(
                PROBES + plane * spr + dims * spd,
                mask=dmask, other=0.0,
            ).to(tl.float32)
            projected_raw = tl.sum(values * probe[None, :], axis=1)
            gradient = tl.load(
                DOUT + batch * sgb + head * sgh
                + positions * sgn + plane * sgr,
                mask=valid, other=0.0,
            ).to(tl.float32)
            dx += gradient[:, None] * (
                probe[None, :] * inverse_norm[:, None]
                - projected_raw[:, None] * values
                * inverse_norm_cubed[:, None]
            )
            if COMPUTE_DPROBES:
                dprobe = tl.sum(
                    gradient[:, None] * inverse_norm[:, None] * values,
                    axis=0,
                )
                tl.atomic_add(
                    DPROBES + plane * spr + dims * spd,
                    dprobe, mask=dmask, sem="relaxed",
                )
        tl.store(
            DX + batch * sdb + head * sdh
            + positions[:, None] * sdn + dims[None, :] * sdd,
            dx, mask=valid[:, None] & dmask[None, :],
        )


class _PhaseProbeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, probes, start: int, end: int):
        batch, heads, seq_len, head_dim = values.shape
        start = max(0, min(int(start), seq_len))
        end = max(start, min(int(end), seq_len))
        output = torch.zeros(
            batch, heads, seq_len, probes.shape[0],
            device=values.device, dtype=torch.float32,
        )
        if end > start:
            block_n = 64 if head_dim <= 128 else 32
            block_hd = _next_pow2(head_dim)
            grid = (batch * heads, triton.cdiv(end - start, block_n))
            _phase_probe_forward_kernel[grid](
                values, probes, output,
                *values.stride(), *probes.stride(), *output.stride(),
                N=seq_len, H=heads, HD=head_dim, R=probes.shape[0],
                START=start, END=end, BLOCK_N=block_n, BLOCK_HD=block_hd,
                num_warps=4, num_stages=2,
            )
        ctx.save_for_backward(values, probes)
        ctx.start = start
        ctx.end = end
        return output

    @staticmethod
    def backward(ctx, grad_output):
        values, probes = ctx.saved_tensors
        batch, heads, seq_len, head_dim = values.shape
        grad_output = grad_output.contiguous()
        dx = torch.zeros_like(values, dtype=torch.float32)
        need_probes = bool(ctx.needs_input_grad[1])
        dprobes = torch.zeros_like(probes, dtype=torch.float32)
        if ctx.end > ctx.start:
            block_n = 64 if head_dim <= 128 else 32
            block_hd = _next_pow2(head_dim)
            grid = (batch * heads, triton.cdiv(ctx.end - ctx.start, block_n))
            _phase_probe_backward_kernel[grid](
                values, probes, grad_output, dx, dprobes,
                *values.stride(), *probes.stride(),
                *grad_output.stride(), *dx.stride(),
                N=seq_len, H=heads, HD=head_dim, R=probes.shape[0],
                START=ctx.start, END=ctx.end,
                BLOCK_N=block_n, BLOCK_HD=block_hd,
                COMPUTE_DPROBES=need_probes,
                num_warps=4, num_stages=2,
            )
        return dx.to(values.dtype), (dprobes.to(probes.dtype) if need_probes else None), None, None


def phase_probe_features(
    values: torch.Tensor,
    probes: torch.Tensor,
    *,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Project RMS-normalized head vectors onto a small probe bank.

    CUDA uses a fused custom kernel and backward; CPU/eager validation uses the
    algebraically identical PyTorch expression. Rows outside [start,end) are zero.
    """
    seq_len = values.shape[2]
    start = max(0, min(int(start), seq_len))
    end = seq_len if end is None else max(start, min(int(end), seq_len))
    if values.is_cuda and _TRITON_AVAILABLE:
        return _PhaseProbeFn.apply(values, probes, start, end)
    result = torch.zeros(
        *values.shape[:3], probes.shape[0],
        device=values.device, dtype=torch.float32,
    )
    if end > start:
        selected = values[:, :, start:end].float()
        inverse = torch.rsqrt(
            selected.square().sum(-1, keepdim=True) + 1.0e-6 * values.shape[-1]
        )
        result[:, :, start:end] = (
            torch.einsum("bhnd,rd->bhnr", selected, probes.float()) * inverse
        )
    return result


def _raw_npci_theta_from_effective(theta: float) -> float:
    theta = float(theta)
    limit = float(NPCI_THETA_MAX)
    if not (0.0 <= abs(theta) < limit):
        raise ValueError(f"NPCI_THETA_INIT={theta} must satisfy abs(theta) < {limit}")
    return math.atanh(theta / limit)


GROUPED_MODE_IDS = {
    'baseline': 0,
    'overlap_slab': 1,
    'overlap_slab_bwd': 3,
}
_UNIMPLEMENTED_GROUPED_MODES = {'packed_kv'}


def _resolve_grouped_mode(mode: str) -> int:
    if not isinstance(mode, str):
        raise TypeError(f"grouped_mode must be a string, got {type(mode).__name__}")
    if mode in _UNIMPLEMENTED_GROUPED_MODES:
        raise NotImplementedError(
            f"grouped_mode='{mode}' is advertised by older configs but is not implemented in this kernel"
        )
    if mode not in GROUPED_MODE_IDS:
        allowed = ', '.join(sorted(GROUPED_MODE_IDS.keys()))
        raise ValueError(f"Unknown grouped_mode='{mode}'. Allowed: {allowed}")
    return GROUPED_MODE_IDS[mode]


def _normalize_int_arg(name: str, value, *, minimum: int) -> int:
    """Normalize integer configuration args without accepting bool/float coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= {minimum}, got bool")
    try:
        ivalue = value.__index__()
    except AttributeError as exc:
        raise TypeError(f"{name} must be an integer >= {minimum}, got {type(value).__name__}") from exc
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer >= {minimum}, got {type(value).__name__}") from exc
    if ivalue < minimum:
        raise ValueError(f"{name}={ivalue} must be >= {minimum}")
    return int(ivalue)


def _normalize_grouping_args(
    k_group_size,
    max_group_size,
    max_group_spread,
    grouped_mode_id: int | None = None,
) -> tuple[int, int, int | None]:
    """Validate sparse grouping knobs and return canonical integer values.

    k_group_size is the compiled kernel capacity: kernels iterate exactly this
    many slots per sparse group. max_group_size is only a planner cap: it may be
    smaller than k_group_size to reduce slab waste, but it can never be larger.
    """
    k_group_size = _normalize_int_arg('k_group_size', k_group_size, minimum=1)

    if max_group_size is None:
        max_group_size = k_group_size
    else:
        max_group_size = _normalize_int_arg('max_group_size', max_group_size, minimum=1)
        if max_group_size > k_group_size:
            raise ValueError(
                f"max_group_size={max_group_size} cannot exceed k_group_size={k_group_size}; "
                "overlap-slab kernels iterate only K_GROUP_SIZE slots per group"
            )

    if grouped_mode_id in (
        GROUPED_MODE_IDS['overlap_slab'],
        GROUPED_MODE_IDS['overlap_slab_bwd'],
    ) and max_group_spread is None:
        # Keep slab tiles bounded for the first streaming/tiling implementation.
        # Larger spreads can be enabled explicitly after profiling/tuning.
        max_group_spread = 64

    if max_group_spread is not None:
        max_group_spread = _normalize_int_arg('max_group_spread', max_group_spread, minimum=0)

    return k_group_size, max_group_size, max_group_spread


def _plan_sparse_groups(
    offsets: list[int],
    j_small: int,
    k_group_size: int,
    max_group_size: int | None = None,
    max_group_spread: int | None = None,
    slab_block_n: int = 64,
) -> dict:
    """Plan sparse grouping for overlap-slab scheduling.

    This is schedule metadata only; logical index semantics remain unchanged.
    k_group_size is kernel capacity; max_group_size is a planner cap that may
    be smaller than, but never larger than, k_group_size.

    Cost model (fixed in V20.1): the slab kernel loads BLOCK_N + spread_g K
    columns per group per query block, so the DP minimizes
    sum_g (slab_block_n + spread_g). Under this cost, merging glen offsets
    pays whenever spread < (glen - 1) * slab_block_n, instead of the old
    spread + k_group_size/glen proxy under which merging only paid for
    spread < ~4 and production offset lattices degenerated to all-singleton
    plans (slab mode then strictly loses to baseline rowwise traversal).
    slab_block_n should approximate the launch-time BLOCK_N (64 on sm89).
    """
    k_group_size, max_group_size, max_group_spread = _normalize_grouping_args(
        k_group_size=k_group_size,
        max_group_size=max_group_size,
        max_group_spread=max_group_spread,
    )

    sparse = list(range(j_small, len(offsets)))
    sparse.sort(key=lambda i: offsets[i])
    n_sparse = len(sparse)

    if n_sparse == 0:
        return {
            'sparse_order': [],
            'group_start': [],
            'group_len': [],
            'group_dmin': [],
            'group_dmax': [],
            'group_i': [],
            'group_rel': [],
            'max_group_len': 0,
            'max_group_spread': 0,
        }

    gmax = max_group_size

    # DP over sorted sparse offsets.
    inf = 10**18
    dp = [inf] * (n_sparse + 1)
    prev = [None] * (n_sparse + 1)
    dp[0] = 0
    for i0 in range(n_sparse):
        if dp[i0] >= inf:
            continue
        dmin = offsets[sparse[i0]]
        dmax = dmin
        for glen in range(1, gmax + 1):
            i1 = i0 + glen
            if i1 > n_sparse:
                break
            d = offsets[sparse[i1 - 1]]
            dmin = d if d < dmin else dmin
            dmax = d if d > dmax else dmax
            spread = dmax - dmin
            if max_group_spread is not None and spread > max_group_spread:
                break

            # Traffic-based cost: K columns the slab kernel loads for this
            # group per query block (see docstring).
            cost = float(slab_block_n) + spread
            cand = dp[i0] + cost
            if cand < dp[i1]:
                dp[i1] = cand
                prev[i1] = (i0, i1, dmin, dmax)

    if prev[n_sparse] is None:
        # Fallback: fixed-size chunking of sorted sparse indices.
        sparse_order = sparse
        groups = [(i0, min(i0 + gmax, n_sparse))
                  for i0 in range(0, n_sparse, gmax)]
    else:
        groups_rev = []
        cur = n_sparse
        while cur > 0:
            step = prev[cur]
            if step is None:
                raise RuntimeError('Sparse grouping DP reconstruction failed')
            i0, i1, _dmin, _dmax = step
            groups_rev.append((i0, i1))
            cur = i0
        groups = list(reversed(groups_rev))
        sparse_order = []
        for i0, i1 in groups:
            sparse_order.extend(sparse[i0:i1])

    group_start = []
    group_len = []
    group_dmin = []
    group_dmax = []
    group_i = []
    group_rel = []
    max_len = 0
    max_spread_obs = 0

    cursor = 0
    for i0, i1 in groups:
        idxs = sparse[i0:i1]
        deltas = [offsets[i] for i in idxs]
        dmin = min(deltas)
        dmax = max(deltas)
        rel = [dmax - d for d in deltas]
        spread = dmax - dmin

        group_start.append(cursor)
        group_len.append(len(idxs))
        group_dmin.append(dmin)
        group_dmax.append(dmax)
        group_i.append(idxs)
        group_rel.append(rel)
        max_len = max(max_len, len(idxs))
        max_spread_obs = max(max_spread_obs, spread)
        cursor += len(idxs)

    return {
        'sparse_order': sparse_order,
        'group_start': group_start,
        'group_len': group_len,
        'group_dmin': group_dmin,
        'group_dmax': group_dmax,
        'group_i': group_i,
        'group_rel': group_rel,
        'max_group_len': max_len,
        'max_group_spread': max_spread_obs,
    }


def _verify_sparse_plan(offsets: list[int], j_small: int, plan: dict) -> None:
    sparse_expected = list(range(j_small, len(offsets)))
    sparse_order = list(plan['sparse_order'])

    if sorted(sparse_order) != sparse_expected:
        raise AssertionError('Sparse plan does not include each sparse logical index exactly once')

    for gi, start in enumerate(plan['group_start']):
        glen = plan['group_len'][gi]
        dmax = plan['group_dmax'][gi]
        for t in range(glen):
            idx = sparse_order[start + t]
            rel = plan['group_rel'][gi][t]
            delta = offsets[idx]
            if rel != dmax - delta:
                raise AssertionError(
                    f'Invalid rel mapping at group={gi}, t={t}: rel={rel}, dmax={dmax}, delta={delta}'
                )


# ===========================================================================
# Forward Kernel — with K block grouping on sparse cluster
# ===========================================================================

@triton.jit
def _fwd_v22_online(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    """One-pass online softmax over DSQG offsets.

    Scale embeddings are folded into offset-specific virtual keys, so each
    offset performs one q dot (k + e_offset) reduction. The weighted value is
    merged immediately, eliminating the [BLOCK_N,J_PAD] score/probability slabs
    and the second value traversal used by the older offline-softmax kernel.
    """
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
    ybase = Y_PRE + batch * stride_yb + head * stride_yh
    zbase = Z_PRE + batch * stride_zb + kv_head * stride_zh
    query = tl.load(
        qbase + positions[:, None] * stride_qn + dims[None, :] * stride_qd,
        mask=qmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)

    running_max = tl.full([BLOCK_N], float('-inf'), tl.float32)
    running_sum = tl.zeros([BLOCK_N], tl.float32)
    accumulator = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    for logical_index in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + logical_index).to(tl.int32)
        source = positions - delta
        valid = qmask & (source >= 0) & (source < N)
        key = tl.load(
            kbase + source[:, None] * stride_kn + dims[None, :] * stride_kd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        virtual = tl.load(
            SE + logical_index * stride_sei + dims * stride_sed,
            mask=dmask, other=0.0,
        ).to(tl.bfloat16).to(tl.float32)
        score = tl.sum(query * (key + virtual[None, :]), axis=1) * scale
        score += tl.load(POS_BIAS + logical_index * stride_pbi + head * stride_pbh)
        score = tl.where(valid, score, float('-inf'))
        merged_max = tl.maximum(running_max, score)
        has_old = running_sum > 0.0
        has_new = valid
        safe_max = tl.where(has_old | has_new, merged_max, 0.0)
        old_scale = tl.where(has_old, tl.exp2((running_max - safe_max) * _LOG2E), 0.0)
        new_weight = tl.where(valid, tl.exp2((score - safe_max) * _LOG2E), 0.0)
        value = tl.load(
            vbase + source[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale[:, None] + new_weight[:, None] * value
        running_sum = running_sum * old_scale + new_weight
        running_max = tl.where(has_old | has_new, merged_max, running_max)

    for group_start in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for group_offset in range(K_GROUP_SIZE):
            slot = group_start + group_offset
            if slot < J_LARGE_VAL:
                logical_index = J_SMALL_VAL + slot
                delta = tl.load(OFFSETS + logical_index).to(tl.int32)
                source = positions - delta
                valid = qmask & (source >= 0) & (source < N)
                key = tl.load(
                    kbase + source[:, None] * stride_kn + dims[None, :] * stride_kd,
                    mask=valid[:, None] & dmask[None, :], other=0.0,
                ).to(tl.float32)
                virtual = tl.load(
                    SE + logical_index * stride_sei + dims * stride_sed,
                    mask=dmask, other=0.0,
                ).to(tl.bfloat16).to(tl.float32)
                score = tl.sum(query * (key + virtual[None, :]), axis=1) * scale
                score += tl.load(POS_BIAS + logical_index * stride_pbi + head * stride_pbh)
                score = tl.where(valid, score, float('-inf'))
                merged_max = tl.maximum(running_max, score)
                has_old = running_sum > 0.0
                has_new = valid
                safe_max = tl.where(has_old | has_new, merged_max, 0.0)
                old_scale = tl.where(has_old, tl.exp2((running_max - safe_max) * _LOG2E), 0.0)
                new_weight = tl.where(valid, tl.exp2((score - safe_max) * _LOG2E), 0.0)
                value = tl.load(
                    vbase + source[:, None] * stride_vn + dims[None, :] * stride_vd,
                    mask=valid[:, None] & dmask[None, :], other=0.0,
                ).to(tl.float32)
                accumulator = accumulator * old_scale[:, None] + new_weight[:, None] * value

                phase_index = logical_index - J_SMALL_VAL
                for plane in range(R_PLANES_VAL):
                    channel_a = plane * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    channel_b = channel_a + 1
                    mask_a = dims == channel_a
                    mask_b = dims == channel_b
                    query_phase = tl.load(
                        ybase + positions * stride_yn + plane, mask=qmask, other=0.0
                    )
                    key_phase = tl.load(
                        zbase + source * stride_zn + plane, mask=valid, other=0.0
                    )
                    phase_base = tl.load(
                        PHASE_BASE + phase_index * stride_phi + head * stride_phh + plane
                    )
                    phase_gain = tl.load(
                        PHASE_GAIN + phase_index * stride_pgi + head * stride_pgh + plane
                    )
                    angle = phase_base + phase_gain * query_phase * key_phase
                    cosine = tl.cos(angle)
                    sine = tl.sin(angle)
                    value_a = tl.sum(value * mask_a[None, :].to(tl.float32), axis=1)
                    value_b = tl.sum(value * mask_b[None, :].to(tl.float32), axis=1)
                    correction = (
                        ((cosine - 1.0) * value_a - sine * value_b)[:, None]
                        * mask_a[None, :].to(tl.float32)
                        + (sine * value_a + (cosine - 1.0) * value_b)[:, None]
                        * mask_b[None, :].to(tl.float32)
                    )
                    accumulator += new_weight[:, None] * correction
                running_sum = running_sum * old_scale + new_weight
                running_max = tl.where(has_old | has_new, merged_max, running_max)

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
        output.to(tl.bfloat16), mask=qmask[:, None] & dmask[None, :],
    )
    tl.store(
        LSE + batch * stride_lb + head * stride_lh + positions * stride_ln,
        lse, mask=qmask,
    )


@triton.jit
def _fwd_v18_grouped(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,  # K block group size (e.g., 3)
    KV_HEAD_GROUP_SIZE: tl.constexpr,  # query heads per KV head for native GQA
    QUERY_START: tl.constexpr,
):
    """
    V18 forward kernel with explicit K block grouping.
    
    K block grouping strategy:
    - Offsets J_SMALL_VAL to J_VAL are grouped into K_GROUP_SIZE-sized groups
    - K blocks within each group are stored contiguously in memory
    - Coalesced loads for grouped K blocks
    
    Dense path (J_SMALL = 0-16):
    - 2 K block GEMMs, diagonal extraction into scores[:, 0:21]
    
    Sparse path (J_SMALL to J_VAL = 17-95):
    - K block loads grouped by K_GROUP_SIZE
    - Sequential dots from warm K
    - Loaded into scores[:, 21:96]
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_kv = h // KV_HEAD_GROUP_SIZE
    n0 = QUERY_START + blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    sc = 1.0 / (HD ** 0.5)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h_kv * stride_kh
    vb = V + b * stride_vb + h_kv * stride_vh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h_kv * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    # Offset-specific scale embeddings are virtual keys.
    scores = tl.full([BLOCK_N, J_PAD], float('-inf'), tl.float32)

    # Dense cluster (unchanged traversal). These are one-key-per-query diagonal
    # row reductions; a tiled tl.dot would compute off-diagonal pairs we discard.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)
        s = tl.sum(q * (kt + se_i[None, :]), axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        scores = tl.where((js == i)[None, :], s[:, None], scores)

    # Sparse cluster — explicit K-block grouped traversal via SPARSE_ORDER.
    # Baseline grouping still has one key per query row, so rowwise reductions
    # are cheaper than materializing a mostly-unused QK tile.
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                               mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)
                s = tl.sum(q * (kt + se_i[None, :]), axis=1) * sc
                s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                scores = tl.where((js == i)[None, :], s[:, None], scores)

    # ── Offline softmax over J dimension ──
    mi = tl.max(scores, axis=1)
    all_invalid = mi == float('-inf')
    safe_mi = tl.where(all_invalid, 0.0, mi)
    exp_s = tl.exp2((scores - safe_mi[:, None]) * _LOG2E)
    exp_s = tl.where(js[None, :] < J_VAL, exp_s, 0.0)
    li = tl.sum(exp_s, axis=1)
    ls = tl.where(li > 0.0, li, 1.0)
    probs = exp_s / ls[:, None]
    lse_val = tl.where(all_invalid, 0.0, safe_mi + tl.log2(ls) * (1.0 / _LOG2E))

    # ═══ Pass 2: weighted V sum with MOVT rotation ═══
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # Dense cluster — no rotation (passthrough)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt

    # Sparse cluster — MOVT rotation (R_PLANES Givens)
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)

                # MOVT touches only 2*R channels. Accumulate the ordinary
                # value once and add sparse channel corrections per plane.
                pi_idx = i - J_SMALL_VAL
                acc += p_i[:, None] * vt
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    correction = (
                        (rotated_a - v_a)[:, None] * mask_a[None, :].to(tl.float32)
                        + (rotated_b - v_b)[:, None] * mask_b[None, :].to(tl.float32)
                    )
                    acc += p_i[:, None] * correction


    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


@triton.jit
def _fwd_v18_overlap_slab(
    Q, K, V, POS_BIAS, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE, OUT, LSE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,  # maximum sparse offsets per slab group
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    """
    V18 forward kernel with explicit K block grouping.
    
    K block grouping strategy:
    - Offsets J_SMALL_VAL to J_VAL are grouped into K_GROUP_SIZE-sized groups
    - K blocks within each group are stored contiguously in memory
    - Coalesced loads for grouped K blocks
    
    Dense path (J_SMALL = 0-16):
    - 2 K block GEMMs, diagonal extraction into scores[:, 0:21]
    
    Sparse path (J_SMALL to J_VAL = 17-95):
    - K block loads grouped by K_GROUP_SIZE
    - Sequential dots from warm K
    - Loaded into scores[:, 21:96]
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = QUERY_START + blk * BLOCK_N

    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    sc = 1.0 / (HD ** 0.5)
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    # ── Tensor Core: batch scale_embed scoring (BF16 for tensor core util) ──
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # SE is stored FP32; rounded to BF16 below for the tensor-core dot
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # ═══ Pass 1: compute all J scores with K block grouping ═══
    scores = tl.where(js[None, :] < J_VAL, se_all, float('-inf'))

    # Dense cluster (unchanged traversal)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        scores = tl.where((js == i)[None, :], s[:, None] + scores, scores)

    # Sparse cluster — real overlap-slab scoring for K.
    # For group G: slab_base0 = n0 - dmax, col(row, i) = row + dmax - delta.
    # Therefore slab_base0 + col = n0 + row - delta = ns - delta exactly.
    # We tile slab columns so qk_tile is [BLOCK_N, SLAB_TILE_N], avoiding a huge
    # [BLOCK_N, BLOCK_SLAB_N] accumulator while still loading each K slab tile
    # once per group and serving every offset whose diagonal lies in that tile.
    rows = tl.arange(0, BLOCK_N)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = n0 - dmax
        real_slab_len = BLOCK_N + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            kpos = slab_base0 + cols
            k_tile_t = tl.load(
                kb + kpos[None, :] * stride_kn + ds[:, None] * stride_kd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (kpos[None, :] >= 0) & (kpos[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(q, k_tile_t, input_precision="tf32") * sc

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = dmax - delta
                diag_col = rows + rel
                kp = ns - delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (kp >= 0) & (kp < N) & nm & valid_slot & in_tile

                s = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                s += tl.load(POS_BIAS + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                scores = tl.where(((js == i) & valid_slot)[None, :] & in_tile[:, None], s[:, None] + scores, scores)

    # ── Offline softmax over J dimension ──
    mi = tl.max(scores, axis=1)
    all_invalid = mi == float('-inf')
    safe_mi = tl.where(all_invalid, 0.0, mi)
    exp_s = tl.exp2((scores - safe_mi[:, None]) * _LOG2E)
    exp_s = tl.where(js[None, :] < J_VAL, exp_s, 0.0)
    li = tl.sum(exp_s, axis=1)
    ls = tl.where(li > 0.0, li, 1.0)
    probs = exp_s / ls[:, None]
    lse_val = tl.where(all_invalid, 0.0, safe_mi + tl.log2(ls) * (1.0 / _LOG2E))

    # ═══ Pass 2: weighted V sum with MOVT rotation ═══
    acc = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # Dense cluster — no rotation (passthrough)
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)
        acc += p_i[:, None] * vt

    # Sparse cluster — MOVT rotation (R_PLANES Givens)
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = tl.load(SPARSE_ORDER + slot).to(tl.int32)
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                p_i = tl.sum(probs * (js == i)[None, :].to(tl.float32), axis=1)

                # MOVT touches only 2*R channels. Accumulate the ordinary
                # value once and add sparse channel corrections per plane.
                pi_idx = i - J_SMALL_VAL
                acc += p_i[:, None] * vt
                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    correction = (
                        (rotated_a - v_a)[:, None] * mask_a[None, :].to(tl.float32)
                        + (rotated_b - v_b)[:, None] * mask_b[None, :].to(tl.float32)
                    )
                    acc += p_i[:, None] * correction


    ob = OUT + b * stride_ob + h * stride_oh
    lb = LSE + b * stride_lb + h * stride_lh
    tl.store(ob + ns[:, None] * stride_on + ds[None, :] * stride_od,
             acc.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])
    tl.store(lb + ns * stride_ln, lse_val, mask=nm)


# ===========================================================================
# D computation — sum(dO * O, dim=-1)
# ===========================================================================

@triton.jit
def _compute_D_v18_grouped(
    DO, O, D,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_db, stride_dh, stride_dn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = QUERY_START + blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    do = tl.load(DO + b * stride_dob + h * stride_doh
                 + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    o = tl.load(O + b * stride_ob + h * stride_oh
                + ns[:, None] * stride_on + ds[None, :] * stride_od,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    tl.store(D + b * stride_db + h * stride_dh + ns * stride_dn,
             tl.sum(do * o, axis=1), mask=nm)


# ===========================================================================
# Backward: dQ + atomic dPOS_BIAS + Tensor Core dSCALE_EMBED + dY_PRE
# ===========================================================================

@triton.jit
def _bwd_dq_v18_grouped(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, DELTA,
    DQ, DPB, DSE, DY_PRE,
    OFFSETS,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dyb, stride_dyh, stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    KV_HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_START: tl.constexpr, FOLD_SE: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_kv = h // KV_HEAD_GROUP_SIZE
    n0 = QUERY_START + blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h_kv * stride_kh
    vb = V + b * stride_vb + h_kv * stride_vh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h_kv * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    do = tl.load(dob + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + ns * stride_ln, mask=nm, other=0.0)
    Dval = tl.load(DELTA + b * stride_db + h * stride_dh + ns * stride_dn,
                   mask=nm, other=0.0).to(tl.float32)

    dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # dy_pre accumulators
    dy_pre_0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_1 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_2 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_3 = tl.zeros([BLOCK_N], tl.float32)

    # Dense cluster. Rowwise reductions here are diagonal attention terms; using
    # tl.dot would add off-diagonal work instead of reusing useful K/V data.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed,
                           mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)
        if FOLD_SE:
            s = tl.sum(q * (kt + se_i_vec[None, :]), axis=1) * sc
        else:
            s = (tl.sum(q * kt, axis=1) + tl.sum(q * se_i_vec[None, :], axis=1)) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lse > float('-inf')),
                         tl.exp2((s - lse) * _LOG2E), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v = alpha * (dot_rv - Dval)
        dq += ds_v[:, None] * (kt + se_i_vec[None, :]) * sc

        tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                      tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

        dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
        tl.atomic_add(DSE + i * stride_sei + ds * stride_sed,
                      dse_i, mask=dm, sem="relaxed")

    # Sparse cluster — grouped traversal
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                kp = ns - delta
                val = (kp >= 0) & (kp < N) & nm

                kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                # Recompute score with the offset embedding folded into K.
                se_i_vec = tl.load(
                    SE + i * stride_sei + ds * stride_sed,
                    mask=dm, other=0.0,
                ).to(tl.bfloat16).to(tl.float32)
                if FOLD_SE:
                    s = tl.sum(q * (kt + se_i_vec[None, :]), axis=1) * sc
                else:
                    s = (tl.sum(q * kt, axis=1) + tl.sum(q * se_i_vec[None, :], axis=1)) * sc
                s += tl.load(PB + i * stride_pbi + h * stride_pbh)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lse > float('-inf')),
                                 tl.exp2((s - lse) * _LOG2E), 0.0)

                # Start from the ordinary value dot and add only the
                # rotated-channel corrections.
                pi_idx = i - J_SMALL_VAL
                dot_rv = tl.sum(do * vt, axis=1)

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    don_a = tl.sum(do * mask_a[None, :].to(tl.float32), axis=1)
                    don_b = tl.sum(do * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    dot_rv += don_a * (rotated_a - v_a) + don_b * (rotated_b - v_b)
                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    contrib = dth_r * pg_r * z_p
                    dy_pre_0 = tl.where(r == 0, dy_pre_0 + contrib, dy_pre_0)
                    dy_pre_1 = tl.where(r == 1, dy_pre_1 + contrib, dy_pre_1)
                    dy_pre_2 = tl.where(r == 2, dy_pre_2 + contrib, dy_pre_2)
                    dy_pre_3 = tl.where(r == 3, dy_pre_3 + contrib, dy_pre_3)

                ds_v = alpha * (dot_rv - Dval)
                dq += ds_v[:, None] * (kt + se_i_vec[None, :]) * sc

                tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                              tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")
                dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
                tl.atomic_add(
                    DSE + i * stride_sei + ds * stride_sed,
                    dse_i, mask=dm, sem="relaxed",
                )

    # Store dy_pre
    dyb = DY_PRE + b * stride_dyb + h * stride_dyh
    tl.store(dyb + ns * stride_dyn + 0, tl.where(nm, dy_pre_0, 0.0), mask=nm)
    tl.store(dyb + ns * stride_dyn + 1, tl.where(nm, dy_pre_1, 0.0), mask=nm)
    if R_PLANES_VAL >= 3:
        tl.store(dyb + ns * stride_dyn + 2, tl.where(nm, dy_pre_2, 0.0), mask=nm)
    if R_PLANES_VAL >= 4:
        tl.store(dyb + ns * stride_dyn + 3, tl.where(nm, dy_pre_3, 0.0), mask=nm)


@triton.jit
def _bwd_dq_v18_overlap_slab(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, DELTA,
    DQ, DPB, DSE, DY_PRE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_lb, stride_lh, stride_ln,
    stride_db, stride_dh, stride_dn,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_pbi, stride_pbh,
    stride_sei, stride_sed,
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dyb, stride_dyh, stride_dyn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_VAL: tl.constexpr, J_SMALL_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
    QUERY_START: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    n0 = QUERY_START + blk * BLOCK_N
    ns = n0 + tl.arange(0, BLOCK_N)
    nm = ns < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    qb = Q + b * stride_qb + h * stride_qh
    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    q = tl.load(qb + ns[:, None] * stride_qn + ds[None, :] * stride_qd,
                mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    do = tl.load(dob + ns[:, None] * stride_don + ds[None, :] * stride_dod,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + b * stride_lb + h * stride_lh + ns * stride_ln, mask=nm, other=0.0)
    Dval = tl.load(DELTA + b * stride_db + h * stride_dh + ns * stride_dn,
                   mask=nm, other=0.0).to(tl.float32)

    # Pre-compute scale_embed scores via Tensor Core (BF16 for tensor core util)
    SE_T = tl.load(
        SE + ds[:, None] * stride_sed + js[None, :] * stride_sei,
        mask=dm[:, None] & (js[None, :] < J_VAL),
        other=0.0
    )  # SE is stored FP32; rounded to BF16 below for the tensor-core dot
    se_all = tl.dot(q.to(tl.bfloat16), SE_T.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    dq = tl.zeros([BLOCK_N, BLOCK_HD], tl.float32)

    # dy_pre accumulators
    dy_pre_0 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_1 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_2 = tl.zeros([BLOCK_N], tl.float32)
    dy_pre_3 = tl.zeros([BLOCK_N], tl.float32)

    # Accumulator for batched dscale_embed via Tensor Core
    DSV_all = tl.zeros([J_PAD, BLOCK_N], tl.float32)

    # Dense cluster
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        kp = ns - delta
        val = (kp >= 0) & (kp < N) & nm

        kt = tl.load(kb + kp[:, None] * stride_kn + ds[None, :] * stride_kd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

        s = tl.sum(q * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
        s += se_score_i
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lse > float('-inf')),
                         tl.exp2((s - lse) * _LOG2E), 0.0)

        dot_rv = tl.sum(do * vt, axis=1)
        ds_v = alpha * (dot_rv - Dval)
        dq += ds_v[:, None] * kt * sc
        se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)  # bf16-round: match fwd SE dot
        dq += ds_v[:, None] * se_i_vec[None, :] * sc

        tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                      tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

        row_mask = js == i
        DSV_all = tl.where(row_mask[:, None], ds_v[None, :], DSV_all)

    # Sparse cluster — real overlap-slab K reuse for dQ.
    # For group G: slab_base0 = n0 - dmax, col(row, i) = row + dmax - delta.
    # qk_tile scores each Q row against a contiguous K slab tile loaded once.
    # Per-offset ds_v values are scattered into dsv_tile on the same diagonals;
    # a single dot(dsv_tile, K_tile) then accumulates the K-score dQ term.
    rows = tl.arange(0, BLOCK_N)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = n0 - dmax
        real_slab_len = BLOCK_N + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            kpos = slab_base0 + cols
            k_tile_t = tl.load(
                kb + kpos[None, :] * stride_kn + ds[:, None] * stride_kd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (kpos[None, :] >= 0) & (kpos[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(q, k_tile_t, input_precision="tf32") * sc
            dsv_tile = tl.zeros([BLOCK_N, SLAB_TILE_N], tl.float32)

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = dmax - delta
                diag_col = rows + rel
                kp = ns - delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (kp >= 0) & (kp < N) & nm & valid_slot & in_tile

                s = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                s += tl.load(PB + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                se_score_i = tl.sum(se_all * (js == i)[None, :].to(tl.float32), axis=1)
                s += se_score_i
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lse > float('-inf')),
                                 tl.exp2((s - lse) * _LOG2E), 0.0)

                vt = tl.load(vb + kp[:, None] * stride_vn + ds[None, :] * stride_vd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)

                # Start from the ordinary value dot and add only the
                # rotated-channel corrections.
                pi_idx = i - J_SMALL_VAL
                dot_rv = tl.sum(do * vt, axis=1)

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_p = tl.load(yb + ns * stride_yn + r, mask=nm, other=0.0)
                    z_p = tl.load(zb + kp * stride_zn + r, mask=val, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r, mask=valid_slot, other=0.0)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r, mask=valid_slot, other=0.0)
                    theta_r = pb_r + pg_r * y_p * z_p
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    don_a = tl.sum(do * mask_a[None, :].to(tl.float32), axis=1)
                    don_b = tl.sum(do * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    dot_rv += don_a * (rotated_a - v_a) + don_b * (rotated_b - v_b)
                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    contrib = dth_r * pg_r * z_p
                    dy_pre_0 = tl.where(r == 0, dy_pre_0 + contrib, dy_pre_0)
                    dy_pre_1 = tl.where(r == 1, dy_pre_1 + contrib, dy_pre_1)
                    dy_pre_2 = tl.where(r == 2, dy_pre_2 + contrib, dy_pre_2)
                    dy_pre_3 = tl.where(r == 3, dy_pre_3 + contrib, dy_pre_3)

                ds_v = alpha * (dot_rv - Dval)

                # dQ contribution through scale_embed is per-offset and unchanged.
                se_i_vec = tl.load(SE + i * stride_sei + ds * stride_sed, mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)  # bf16-round: match fwd SE dot
                dq += ds_v[:, None] * se_i_vec[None, :] * sc

                tl.atomic_add(DPB + i * stride_pbi + h * stride_pbh,
                              tl.sum(tl.where(val.to(tl.int1), ds_v, 0.0)), sem="relaxed")

                # Sparse dscale_embed is accumulated immediately because this tiled
                # path only has the true per-row ds_v inside the tile containing
                # each diagonal. Dense offsets still use DSV_all + final batched dot.
                dse_i = tl.sum(ds_v[:, None] * q, axis=0) * sc
                tl.atomic_add(DSE + i * stride_sei + ds * stride_sed,
                              dse_i, mask=valid_slot & dm, sem="relaxed")
                dsv_tile += ds_v[:, None] * (cols[None, :] == diag_col[:, None]).to(tl.float32)

            dq += tl.dot(dsv_tile, tl.trans(k_tile_t), input_precision="tf32") * sc

    # ── Tensor Core: batched dscale_embed (BF16 for tensor core util) ──
    dse_all = tl.dot(DSV_all.to(tl.bfloat16), q.to(tl.bfloat16)) * sc  # BF16->FP32 accum

    # Store dQ
    tl.store(DQ + b * stride_dqb + h * stride_dqh
             + ns[:, None] * stride_dqn + ds[None, :] * stride_dqd,
             dq.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])

    # Store dSE — atomic. Use a real mask, not just zeroed values: for HD values
    # that are not powers of two (e.g. D768/H16 -> HD=48, BLOCK_HD=64), padded
    # lanes would otherwise address past the end of each scale_embed row and can
    # trigger CUDA illegal memory access.
    tl.atomic_add(DSE + js[:, None] * stride_sei + ds[None, :] * stride_sed,
                  dse_all, mask=(js[:, None] < J_VAL) & dm[None, :], sem="relaxed")

    # Store dy_pre
    dyb = DY_PRE + b * stride_dyb + h * stride_dyh
    tl.store(dyb + ns * stride_dyn + 0, tl.where(nm, dy_pre_0, 0.0), mask=nm)
    tl.store(dyb + ns * stride_dyn + 1, tl.where(nm, dy_pre_1, 0.0), mask=nm)
    if R_PLANES_VAL >= 3:
        tl.store(dyb + ns * stride_dyn + 2, tl.where(nm, dy_pre_2, 0.0), mask=nm)
    if R_PLANES_VAL >= 4:
        tl.store(dyb + ns * stride_dyn + 3, tl.where(nm, dy_pre_3, 0.0), mask=nm)


# ===========================================================================
# Backward: dK + dV + atomic d_phase_base + atomic d_phase_gain + dZ_PRE
# ===========================================================================

@triton.jit
def _bwd_dkdv_v18_grouped(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, DELTA,
    DK, DV,
    DPHASE_BASE, DPHASE_GAIN,
    DZ_PRE,
    OFFSETS,
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
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_dphi, stride_dphh,
    stride_dpgi, stride_dpgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dzb, stride_dzh, stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_SMALL_VAL: tl.constexpr, J_LARGE_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr, J_PAD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    KV_HEAD_GROUP_SIZE: tl.constexpr,
    NATIVE_GQA: tl.constexpr, FOLD_SE: tl.constexpr,
):
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_kv = h // KV_HEAD_GROUP_SIZE
    m0 = blk * BLOCK_M
    ms = m0 + tl.arange(0, BLOCK_M)
    mm = ms < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    js = tl.arange(0, J_PAD)
    sc = 1.0 / (HD ** 0.5)

    kb = K + b * stride_kb + h_kv * stride_kh
    vb = V + b * stride_vb + h_kv * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h_kv * stride_zh

    kt = tl.load(kb + ms[:, None] * stride_kn + ds[None, :] * stride_kd,
                  mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vt = tl.load(vb + ms[:, None] * stride_vn + ds[None, :] * stride_vd,
                  mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre_0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_1 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_2 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_3 = tl.zeros([BLOCK_M], tl.float32)

    # Dense cluster. Each key row maps to sparse query diagonals, so row
    # reductions avoid useless off-diagonal tile work.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_ = ms + delta
        val = (np_ >= 0) & (np_ < N) & mm

        qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                      mask=val, other=0.0)
        Dn = tl.load(DELTA + b * stride_db + h * stride_dh + np_ * stride_dn,
                     mask=val, other=0.0).to(tl.float32)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)
        if FOLD_SE:
            s = tl.sum(qn * (kt + se_i[None, :]), axis=1) * sc
        else:
            s = (tl.sum(qn * kt, axis=1) + tl.sum(qn * se_i[None, :], axis=1)) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        s = tl.where(val.to(tl.int1), s, float('-inf'))

        alpha = tl.where(val & (lsen > float('-inf')),
                        tl.exp2((s - lsen) * _LOG2E), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v = alpha * (dot_rv - Dn)
        dk += ds_v[:, None] * qn * sc
        dv += alpha[:, None] * don

    # Sparse cluster — grouped traversal
    for g0 in range(0, J_LARGE_VAL, K_GROUP_SIZE):
        for gi in range(K_GROUP_SIZE):
            slot = g0 + gi
            if slot < J_LARGE_VAL:
                i = J_SMALL_VAL + slot  # direct index, no indirection for baseline
                delta = tl.load(OFFSETS + i).to(tl.int32)
                np_ = ms + delta
                val = (np_ >= 0) & (np_ < N) & mm

                qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                              mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                               mask=val, other=0.0)
                Dn = tl.load(DELTA + b * stride_db + h * stride_dh + np_ * stride_dn,
                             mask=val, other=0.0).to(tl.float32)

                se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                                mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)
                if FOLD_SE:
                    s = tl.sum(qn * (kt + se_i[None, :]), axis=1) * sc
                else:
                    s = (tl.sum(qn * kt, axis=1) + tl.sum(qn * se_i[None, :], axis=1)) * sc
                s += tl.load(PB + i * stride_pbi + h * stride_pbh)
                s = tl.where(val.to(tl.int1), s, float('-inf'))

                alpha = tl.where(val & (lsen > float('-inf')),
                                 tl.exp2((s - lsen) * _LOG2E), 0.0)

                pi_idx = i - J_SMALL_VAL

                dz_pre_0_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_1_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_2_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_3_local = tl.zeros([BLOCK_M], tl.float32)

                dv_c = alpha[:, None] * don
                dot_rv = tl.sum(don * vt, axis=1)

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_r = tl.load(yb + np_ * stride_yn + r, mask=val, other=0.0)
                    z_r = tl.load(zb + ms * stride_zn + r, mask=mm, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r)
                    theta_r = pb_r + pg_r * y_r * z_r
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    don_a = tl.sum(don * mask_a[None, :].to(tl.float32), axis=1)
                    don_b = tl.sum(don * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    dot_rv += don_a * (rotated_a - v_a) + don_b * (rotated_b - v_b)
                    dv_c += (
                        (alpha * ((cos_r - 1.0) * don_a + sin_r * don_b))[:, None]
                        * mask_a[None, :].to(tl.float32)
                        + (alpha * (-sin_r * don_a + (cos_r - 1.0) * don_b))[:, None]
                        * mask_b[None, :].to(tl.float32)
                    )

                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    tl.atomic_add(DPHASE_BASE + pi_idx * stride_dphi + h * stride_dphh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r, 0.0)), sem="relaxed")
                    tl.atomic_add(DPHASE_GAIN + pi_idx * stride_dpgi + h * stride_dpgh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r * y_r * z_r, 0.0)), sem="relaxed")

                    contrib = tl.where(val.to(tl.int1), dth_r * pg_r * y_r, 0.0)
                    dz_pre_0_local = tl.where(r == 0, dz_pre_0_local + contrib, dz_pre_0_local)
                    dz_pre_1_local = tl.where(r == 1, dz_pre_1_local + contrib, dz_pre_1_local)
                    dz_pre_2_local = tl.where(r == 2, dz_pre_2_local + contrib, dz_pre_2_local)
                    dz_pre_3_local = tl.where(r == 3, dz_pre_3_local + contrib, dz_pre_3_local)

                ds_v = alpha * (dot_rv - Dn)
                dk += ds_v[:, None] * qn * sc
                dv += dv_c

                dz_pre_0 += dz_pre_0_local
                dz_pre_1 += dz_pre_1_local
                dz_pre_2 += dz_pre_2_local
                dz_pre_3 += dz_pre_3_local

    if NATIVE_GQA:
        tl.atomic_add(DK + b * stride_dkb + h_kv * stride_dkh
                      + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
                      dk, mask=mm[:, None] & dm[None, :], sem="relaxed")
        tl.atomic_add(DV + b * stride_dvb + h_kv * stride_dvh
                      + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
                      dv, mask=mm[:, None] & dm[None, :], sem="relaxed")

        dzb = DZ_PRE + b * stride_dzb + h_kv * stride_dzh
        tl.atomic_add(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre_0, 0.0), mask=mm, sem="relaxed")
        tl.atomic_add(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre_1, 0.0), mask=mm, sem="relaxed")
        if R_PLANES_VAL >= 3:
            tl.atomic_add(dzb + ms * stride_dzn + 2, tl.where(mm, dz_pre_2, 0.0), mask=mm, sem="relaxed")
        if R_PLANES_VAL >= 4:
            tl.atomic_add(dzb + ms * stride_dzn + 3, tl.where(mm, dz_pre_3, 0.0), mask=mm, sem="relaxed")
    else:
        tl.store(DK + b * stride_dkb + h * stride_dkh
                 + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
                 dk.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])
        tl.store(DV + b * stride_dvb + h * stride_dvh
                 + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
                 dv.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])

        dzb = DZ_PRE + b * stride_dzb + h * stride_dzh
        tl.store(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre_0, 0.0), mask=mm)
        tl.store(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre_1, 0.0), mask=mm)
        if R_PLANES_VAL >= 3:
            tl.store(dzb + ms * stride_dzn + 2, tl.where(mm, dz_pre_2, 0.0), mask=mm)
        if R_PLANES_VAL >= 4:
            tl.store(dzb + ms * stride_dzn + 3, tl.where(mm, dz_pre_3, 0.0), mask=mm)


@triton.jit
def _bwd_dkdv_v18_overlap_slab(
    Q, K, V, PB, SE, PHASE_BASE, PHASE_GAIN, Y_PRE, Z_PRE,
    DO, LSE, DELTA,
    DK, DV,
    DPHASE_BASE, DPHASE_GAIN,
    DZ_PRE,
    OFFSETS, SPARSE_ORDER, GROUP_START, GROUP_LEN, GROUP_DMIN, GROUP_DMAX,
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
    stride_phi, stride_phh,
    stride_pgi, stride_pgh,
    stride_dphi, stride_dphh,
    stride_dpgi, stride_dpgh,
    stride_yb, stride_yh, stride_yn,
    stride_zb, stride_zh, stride_zn,
    stride_dzb, stride_dzh, stride_dzn,
    H: tl.constexpr, N, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HD: tl.constexpr,
    J_SMALL_VAL: tl.constexpr,
    R_PLANES_VAL: tl.constexpr, PLANE_SHIFT: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SLAB_N: tl.constexpr,
    SLAB_TILE_N: tl.constexpr,
):
    """dK/dV backward with sparse query-slab tiling for overlap_slab_bwd mode.

    The dense offsets keep the direct write-once path.  Sparse offsets are
    grouped by offset spread; each group loads contiguous Q slabs and reuses the
    QK tile plus a dscore slab to accumulate the K-score dK term with one
    dot(dscore_slab, Q_slab) per tile.
    """
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    m0 = blk * BLOCK_M
    ms = m0 + tl.arange(0, BLOCK_M)
    mm = ms < N
    ds = tl.arange(0, BLOCK_HD)
    dm = ds < HD
    sc = 1.0 / (HD ** 0.5)

    kb = K + b * stride_kb + h * stride_kh
    vb = V + b * stride_vb + h * stride_vh
    qb = Q + b * stride_qb + h * stride_qh
    dob = DO + b * stride_dob + h * stride_doh
    yb = Y_PRE + b * stride_yb + h * stride_yh
    zb = Z_PRE + b * stride_zb + h * stride_zh

    kt = tl.load(kb + ms[:, None] * stride_kn + ds[None, :] * stride_kd,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vt = tl.load(vb + ms[:, None] * stride_vn + ds[None, :] * stride_vd,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_HD], tl.float32)
    dz_pre_0 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_1 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_2 = tl.zeros([BLOCK_M], tl.float32)
    dz_pre_3 = tl.zeros([BLOCK_M], tl.float32)

    # Dense cluster: direct write-once accumulation.
    for i in range(J_SMALL_VAL):
        delta = tl.load(OFFSETS + i).to(tl.int32)
        np_ = ms + delta
        val = (np_ >= 0) & (np_ < N) & mm

        qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                     mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                      mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
        lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                      mask=val, other=0.0)
        Dn = tl.load(DELTA + b * stride_db + h * stride_dh + np_ * stride_dn,
                     mask=val, other=0.0).to(tl.float32)

        se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                       mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)  # bf16-round: match fwd SE dot
        s = tl.sum(qn * kt, axis=1) * sc
        s += tl.load(PB + i * stride_pbi + h * stride_pbh)
        s += tl.sum(qn * se_i[None, :], axis=1) * sc
        s = tl.where(val.to(tl.int1), s, float('-inf'))
        alpha = tl.where(val & (lsen > float('-inf')),
                         tl.exp2((s - lsen) * _LOG2E), 0.0)

        dot_rv = tl.sum(don * vt, axis=1)
        ds_v = alpha * (dot_rv - Dn)
        dk += ds_v[:, None] * qn * sc
        dv += alpha[:, None] * don

    rows = tl.arange(0, BLOCK_M)
    slab_tile_cols = tl.arange(0, SLAB_TILE_N)

    # Sparse cluster: query-slab reuse for qk and dK score path.
    for g in range(NUM_GROUPS):
        start = tl.load(GROUP_START + g).to(tl.int32)
        glen = tl.load(GROUP_LEN + g).to(tl.int32)
        dmin = tl.load(GROUP_DMIN + g).to(tl.int32)
        dmax = tl.load(GROUP_DMAX + g).to(tl.int32)
        slab_base0 = m0 + dmin
        real_slab_len = BLOCK_M + (dmax - dmin)

        for c0 in range(0, BLOCK_SLAB_N, SLAB_TILE_N):
            cols = c0 + slab_tile_cols
            qpos_tile = slab_base0 + cols
            q_tile_t = tl.load(
                qb + qpos_tile[None, :] * stride_qn + ds[:, None] * stride_qd,
                mask=dm[:, None] & (cols[None, :] < real_slab_len) & (qpos_tile[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            qk_tile = tl.dot(kt, q_tile_t, input_precision="tf32") * sc
            dsv_tile = tl.zeros([BLOCK_M, SLAB_TILE_N], tl.float32)

            for gi in range(K_GROUP_SIZE):
                valid_slot = gi < glen
                i = tl.load(SPARSE_ORDER + start + gi, mask=valid_slot, other=0).to(tl.int32)
                delta = tl.load(OFFSETS + i, mask=valid_slot, other=0).to(tl.int32)
                rel = delta - dmin
                diag_col = rows + rel
                np_ = ms + delta
                in_tile = (diag_col >= c0) & (diag_col < c0 + SLAB_TILE_N)
                val = (np_ >= 0) & (np_ < N) & mm & valid_slot & in_tile

                s_qk = tl.sum(
                    qk_tile * (cols[None, :] == diag_col[:, None]).to(tl.float32),
                    axis=1,
                )
                qn = tl.load(qb + np_[:, None] * stride_qn + ds[None, :] * stride_qd,
                             mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                don = tl.load(dob + np_[:, None] * stride_don + ds[None, :] * stride_dod,
                              mask=val[:, None] & dm[None, :], other=0.0).to(tl.float32)
                lsen = tl.load(LSE + b * stride_lb + h * stride_lh + np_ * stride_ln,
                               mask=val, other=0.0)
                Dn = tl.load(DELTA + b * stride_db + h * stride_dh + np_ * stride_dn,
                             mask=val, other=0.0).to(tl.float32)

                se_i = tl.load(SE + i * stride_sei + ds * stride_sed,
                               mask=dm, other=0.0).to(tl.bfloat16).to(tl.float32)  # bf16-round: match fwd SE dot
                s = s_qk
                s += tl.load(PB + i * stride_pbi + h * stride_pbh, mask=valid_slot, other=0.0)
                s += tl.sum(qn * se_i[None, :], axis=1) * sc
                s = tl.where(val.to(tl.int1), s, float('-inf'))
                alpha = tl.where(val & (lsen > float('-inf')),
                                 tl.exp2((s - lsen) * _LOG2E), 0.0)

                pi_idx = i - J_SMALL_VAL
                dz_pre_0_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_1_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_2_local = tl.zeros([BLOCK_M], tl.float32)
                dz_pre_3_local = tl.zeros([BLOCK_M], tl.float32)

                dv_c = alpha[:, None] * don
                dot_rv = tl.sum(don * vt, axis=1)

                for r in range(R_PLANES_VAL):
                    ch_a = r * (HD // R_PLANES_VAL) + PLANE_SHIFT
                    ch_b = ch_a + 1
                    mask_a = (ds == ch_a)
                    mask_b = (ds == ch_b)

                    y_r = tl.load(yb + np_ * stride_yn + r, mask=val, other=0.0)
                    z_r = tl.load(zb + ms * stride_zn + r, mask=mm, other=0.0)

                    pb_r = tl.load(PHASE_BASE + pi_idx * stride_phi + h * stride_phh + r, mask=valid_slot, other=0.0)
                    pg_r = tl.load(PHASE_GAIN + pi_idx * stride_pgi + h * stride_pgh + r, mask=valid_slot, other=0.0)
                    theta_r = pb_r + pg_r * y_r * z_r
                    cos_r = tl.cos(theta_r)
                    sin_r = tl.sin(theta_r)

                    v_a = tl.sum(vt * mask_a[None, :].to(tl.float32), axis=1)
                    v_b = tl.sum(vt * mask_b[None, :].to(tl.float32), axis=1)
                    don_a = tl.sum(don * mask_a[None, :].to(tl.float32), axis=1)
                    don_b = tl.sum(don * mask_b[None, :].to(tl.float32), axis=1)
                    rotated_a = cos_r * v_a - sin_r * v_b
                    rotated_b = sin_r * v_a + cos_r * v_b
                    dot_rv += don_a * (rotated_a - v_a) + don_b * (rotated_b - v_b)
                    dv_c += (
                        (alpha * ((cos_r - 1.0) * don_a + sin_r * don_b))[:, None]
                        * mask_a[None, :].to(tl.float32)
                        + (alpha * (-sin_r * don_a + (cos_r - 1.0) * don_b))[:, None]
                        * mask_b[None, :].to(tl.float32)
                    )

                    dth_r = alpha * (don_a * (-v_a * sin_r - v_b * cos_r)
                                     + don_b * (v_a * cos_r - v_b * sin_r))

                    tl.atomic_add(DPHASE_BASE + pi_idx * stride_dphi + h * stride_dphh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r, 0.0)), mask=valid_slot, sem="relaxed")
                    tl.atomic_add(DPHASE_GAIN + pi_idx * stride_dpgi + h * stride_dpgh + r,
                                  tl.sum(tl.where(val.to(tl.int1), dth_r * y_r * z_r, 0.0)), mask=valid_slot, sem="relaxed")

                    contrib = tl.where(val.to(tl.int1), dth_r * pg_r * y_r, 0.0)
                    dz_pre_0_local = tl.where(r == 0, dz_pre_0_local + contrib, dz_pre_0_local)
                    dz_pre_1_local = tl.where(r == 1, dz_pre_1_local + contrib, dz_pre_1_local)
                    dz_pre_2_local = tl.where(r == 2, dz_pre_2_local + contrib, dz_pre_2_local)
                    dz_pre_3_local = tl.where(r == 3, dz_pre_3_local + contrib, dz_pre_3_local)

                ds_v = alpha * (dot_rv - Dn)
                dsv_tile += ds_v[:, None] * (cols[None, :] == diag_col[:, None]).to(tl.float32)
                dv += dv_c
                dz_pre_0 += dz_pre_0_local
                dz_pre_1 += dz_pre_1_local
                dz_pre_2 += dz_pre_2_local
                dz_pre_3 += dz_pre_3_local

            dk += tl.dot(dsv_tile, tl.trans(q_tile_t), input_precision="tf32") * sc

    tl.store(DK + b * stride_dkb + h * stride_dkh
             + ms[:, None] * stride_dkn + ds[None, :] * stride_dkd,
             dk.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])
    tl.store(DV + b * stride_dvb + h * stride_dvh
             + ms[:, None] * stride_dvn + ds[None, :] * stride_dvd,
             dv.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])

    dzb = DZ_PRE + b * stride_dzb + h * stride_dzh
    tl.store(dzb + ms * stride_dzn + 0, tl.where(mm, dz_pre_0, 0.0), mask=mm)
    tl.store(dzb + ms * stride_dzn + 1, tl.where(mm, dz_pre_1, 0.0), mask=mm)
    if R_PLANES_VAL >= 3:
        tl.store(dzb + ms * stride_dzn + 2, tl.where(mm, dz_pre_2, 0.0), mask=mm)
    if R_PLANES_VAL >= 4:
        tl.store(dzb + ms * stride_dzn + 3, tl.where(mm, dz_pre_3, 0.0), mask=mm)


# ===========================================================================
# Autograd Function
# ===========================================================================

class _DSQGV18GroupedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                j_val, j_small, j_large,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                k_group_size, grouped_mode_id, verify_mode, max_group_spread_observed,
                plane_shift, query_start, online_softmax):
        B, H, N, HD = q.shape
        query_start = max(0, min(int(query_start), N))
        key_end = max(0, N - query_start)
        online_softmax = bool(online_softmax)
        Bk, H_KV, Nk, HDk = k.shape
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")
        if Bk != B or Nk != N or HDk != HD:
            raise ValueError(f"native GQA requires q/k/v to share B,N,HD: q={tuple(q.shape)} k={tuple(k.shape)}")
        if H % H_KV != 0:
            raise ValueError(f"num query heads H={H} must be divisible by KV heads H_KV={H_KV}")
        kv_head_group_size = H // H_KV
        native_gqa = H != H_KV
        if native_gqa and grouped_mode_id in (
            GROUPED_MODE_IDS['overlap_slab'],
            GROUPED_MODE_IDS['overlap_slab_bwd'],
        ):
            raise NotImplementedError("native GQA is currently implemented for grouped_mode='baseline' only")
        if y_pre.shape[:3] != (B, H, N):
            raise ValueError(f"y_pre must have shape [B,Hq,N,R]; got {tuple(y_pre.shape)} for q={tuple(q.shape)}")
        if z_pre.shape[:3] != (B, H_KV, N):
            raise ValueError(f"z_pre must have shape [B,Hkv,N,R]; got {tuple(z_pre.shape)} for k={tuple(k.shape)}")
        assert q.dtype == torch.bfloat16
        if not q.is_cuda:
            raise RuntimeError("DSQG V22 requires CUDA/Triton tensors")

        # Kernels consume explicit strides, so Q/K/V can stay as qkv_proj views.
        y_pre = y_pre.contiguous()
        z_pre = z_pre.contiguous()
        pos_bias = pos_bias.contiguous()
        scale_embed = scale_embed.contiguous()

        _cc = torch.cuda.get_device_capability()
        _sm90 = (_cc[0] == 9 and _cc[1] == 0) or _cc[0] > 9
        _sm89 = (_cc[0] == 8 and _cc[1] == 9)

        if HD <= 64:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 8, 2
            else:       BLOCK_N, _nw, _ns = 64, 4, 2
        elif HD <= 128:
            if _sm90:   BLOCK_N, _nw, _ns = 128, 8, 3
            elif _sm89: BLOCK_N, _nw, _ns = 64, 4, 2
            else:       BLOCK_N, _nw, _ns = 32, 4, 2
        elif HD <= 256:
            if _sm90:   BLOCK_N, _nw, _ns = 32, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 32, 4, 2
            else:       BLOCK_N, _nw, _ns = 16, 4, 2
        else:
            if _sm90:   BLOCK_N, _nw, _ns = 16, 4, 3
            elif _sm89: BLOCK_N, _nw, _ns = 16, 4, 2
            else:       BLOCK_N, _nw, _ns = 8, 4, 2

        BLOCK_HD = _next_pow2(HD)
        J_PAD = max(16, _next_pow2(j_val))
        if BLOCK_N < 16 or BLOCK_HD < 16 or J_PAD < 16:
            raise RuntimeError(
                f"Unsupported DSQG V22 launch shape for tl.dot: BLOCK_N={BLOCK_N}, "
                f"BLOCK_HD={BLOCK_HD}, J_PAD={J_PAD}; each must be >= 16"
            )

        # Rows before the minimum offset have no legal source. Keep them exact
        # zero and avoid launching query-owned kernels over that prefix.
        out = torch.zeros_like(q)
        lse = torch.full((B, H, N), float("-inf"), device=q.device, dtype=torch.float32)
        query_grid = (B * H, triton.cdiv(max(0, N - query_start), BLOCK_N))

        if grouped_mode_id in (
            GROUPED_MODE_IDS['overlap_slab'],
            GROUPED_MODE_IDS['overlap_slab_bwd'],
        ):
            _fwd_v18_overlap_slab[query_grid](
                q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
                y_pre, z_pre, out, lse,
                offsets_dev, sparse_order_dev, group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
                J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BLOCK_N + int(max_group_spread_observed)),
                SLAB_TILE_N=32, QUERY_START=query_start,
                num_warps=_nw, num_stages=_ns,
            )
        else:
            forward_kernel = _fwd_v22_online if online_softmax else _fwd_v18_grouped
            if online_softmax:
                forward_kernel[query_grid](
                    q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
                    y_pre, z_pre, out, lse, offsets_dev,
                    *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
                    *pos_bias.stride(), *scale_embed.stride(),
                    *phase_base.stride()[:2], *phase_gain.stride()[:2],
                    *y_pre.stride()[:3], *z_pre.stride()[:3],
                    H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
                    J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                    R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift,
                    K_GROUP_SIZE=k_group_size, KV_HEAD_GROUP_SIZE=kv_head_group_size,
                    QUERY_START=query_start, num_warps=_nw, num_stages=_ns,
                )
            else:
                forward_kernel[query_grid](
                    q, k, v, pos_bias, scale_embed, phase_base, phase_gain,
                    y_pre, z_pre, out, lse, offsets_dev,
                    *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
                    *pos_bias.stride(), *scale_embed.stride(),
                    *phase_base.stride()[:2], *phase_gain.stride()[:2],
                    *y_pre.stride()[:3], *z_pre.stride()[:3],
                    H=H, N=N, HD=HD, BLOCK_N=BLOCK_N, BLOCK_HD=BLOCK_HD,
                    J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                    R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                    K_GROUP_SIZE=k_group_size, KV_HEAD_GROUP_SIZE=kv_head_group_size,
                    QUERY_START=query_start, num_warps=_nw, num_stages=_ns,
                )

        ctx.save_for_backward(q, k, v, pos_bias, scale_embed,
                              phase_base, phase_gain, y_pre, z_pre,
                              out, lse, offsets_dev, sparse_order_dev,
                              group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev)
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HD = BLOCK_HD
        ctx.num_warps = _nw
        ctx.num_stages = _ns
        ctx.j_val = j_val
        ctx.j_small = j_small
        ctx.j_large = j_large
        ctx.J_PAD = J_PAD
        ctx.k_group_size = k_group_size
        ctx.kv_head_group_size = kv_head_group_size
        ctx.native_gqa = native_gqa
        ctx.grouped_mode_id = grouped_mode_id
        ctx.verify_mode = verify_mode
        ctx.max_group_spread_observed = max_group_spread_observed
        ctx.plane_shift = plane_shift
        ctx.query_start = query_start
        ctx.key_end = key_end
        ctx.online_softmax = online_softmax
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, k, v, pos_bias, scale_embed,
         phase_base, phase_gain, y_pre, z_pre,
         out, lse, offsets_dev, sparse_order_dev,
         group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev) = ctx.saved_tensors
        B, H, N, HD = q.shape
        H_KV = k.shape[1]
        BN = ctx.BLOCK_N
        BHD = ctx.BLOCK_HD
        NW = ctx.num_warps
        NS = ctx.num_stages
        j_val = ctx.j_val
        j_small = ctx.j_small
        j_large = ctx.j_large
        J_PAD = ctx.J_PAD
        k_group_size = ctx.k_group_size
        plane_shift = ctx.plane_shift

        dout = dout.contiguous()
        _dev = q.device
        query_grid = (B * H, triton.cdiv(max(0, N - ctx.query_start), BN))
        kv_grid = (B * H, triton.cdiv(max(0, ctx.key_end), BN))

        # FlashAttention-style delta = sum(dO * O, dim=-1), computed once.
        # The supplied dK/dV kernels recomputed this vector dot for every offset.
        delta = torch.zeros((B, H, N), device=_dev, dtype=torch.float32)
        _compute_D_v18_grouped[query_grid](
            dout, out, delta,
            *dout.stride(), *out.stride(), *delta.stride(),
            H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
            QUERY_START=ctx.query_start, num_warps=NW, num_stages=NS,
        )

        # dQ + atomic dpos_bias + atomic dscale_embed + dy_pre
        dq = torch.zeros_like(q)
        dy_pre = torch.zeros_like(y_pre)
        dpb = torch.zeros(j_val, H, device=_dev, dtype=torch.float32)
        dse = torch.zeros(j_val, HD, device=_dev, dtype=torch.float32)

        if ctx.grouped_mode_id == GROUPED_MODE_IDS['overlap_slab_bwd']:
            _bwd_dq_v18_overlap_slab[query_grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, delta,
                dq, dpb, dse, dy_pre,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
                J_VAL=j_val, J_SMALL_VAL=j_small,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BN + int(ctx.max_group_spread_observed)),
                SLAB_TILE_N=32, QUERY_START=ctx.query_start,
                num_warps=NW, num_stages=NS,
            )
        else:
            _bwd_dq_v18_grouped[query_grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, delta,
                dq, dpb, dse, dy_pre,
                offsets_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dy_pre.stride(0), dy_pre.stride(1), dy_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_N=BN, BLOCK_HD=BHD,
                J_VAL=j_val, J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                QUERY_START=ctx.query_start,
                FOLD_SE=ctx.online_softmax,
                num_warps=NW, num_stages=NS,
            )

        # dK + dV + atomic dphase_base + atomic dphase_gain + dz_pre
        if ctx.native_gqa:
            # Multiple query heads atomically contribute to each shared KV head.
            # Accumulate in FP32 and cast only after all atomics complete.
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            dz_pre = torch.zeros_like(z_pre)
        else:
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            dz_pre = torch.zeros_like(z_pre)
        d_phase_base_buf = torch.zeros_like(phase_base)
        d_phase_gain_buf = torch.zeros_like(phase_gain)

        if ctx.grouped_mode_id == GROUPED_MODE_IDS['overlap_slab_bwd']:
            _bwd_dkdv_v18_overlap_slab[kv_grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, delta,
                dk, dv,
                d_phase_base_buf, d_phase_gain_buf,
                dz_pre,
                offsets_dev, sparse_order_dev,
                group_start_dev, group_len_dev, group_dmin_dev, group_dmax_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                d_phase_base_buf.stride(0), d_phase_base_buf.stride(1),
                d_phase_gain_buf.stride(0), d_phase_gain_buf.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
                J_SMALL_VAL=j_small,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift,
                K_GROUP_SIZE=k_group_size,
                NUM_GROUPS=int(group_len_dev.numel()),
                BLOCK_SLAB_N=_next_pow2(BN + int(ctx.max_group_spread_observed)),
                SLAB_TILE_N=32,
                num_warps=NW, num_stages=NS,
            )
        else:
            _bwd_dkdv_v18_grouped[kv_grid](
                q, k, v, pos_bias, scale_embed,
                phase_base, phase_gain, y_pre, z_pre,
                dout, lse, delta,
                dk, dv,
                d_phase_base_buf, d_phase_gain_buf,
                dz_pre,
                offsets_dev,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                pos_bias.stride(0), pos_bias.stride(1),
                scale_embed.stride(0), scale_embed.stride(1),
                phase_base.stride(0), phase_base.stride(1),
                phase_gain.stride(0), phase_gain.stride(1),
                d_phase_base_buf.stride(0), d_phase_base_buf.stride(1),
                d_phase_gain_buf.stride(0), d_phase_gain_buf.stride(1),
                y_pre.stride(0), y_pre.stride(1), y_pre.stride(2),
                z_pre.stride(0), z_pre.stride(1), z_pre.stride(2),
                dz_pre.stride(0), dz_pre.stride(1), dz_pre.stride(2),
                H=H, N=N, HD=HD, BLOCK_M=BN, BLOCK_HD=BHD,
                J_SMALL_VAL=j_small, J_LARGE_VAL=j_large,
                R_PLANES_VAL=R_PLANES, PLANE_SHIFT=plane_shift, J_PAD=J_PAD,
                K_GROUP_SIZE=k_group_size,
                KV_HEAD_GROUP_SIZE=ctx.kv_head_group_size,
                NATIVE_GQA=ctx.native_gqa,
                FOLD_SE=ctx.online_softmax,
                num_warps=NW, num_stages=NS,
        )

        if ctx.native_gqa:
            dk = dk.to(dtype=k.dtype)
            dv = dv.to(dtype=v.dtype)

        return (dq, dk, dv,
                dpb, dse, d_phase_base_buf, d_phase_gain_buf, dy_pre, dz_pre,
                None, None, None,
                None, None,
                None, None, None, None,
                None, None, None,
                None, None, None, None)


# ===========================================================================
# Eager semantic oracle + public API
# ===========================================================================


def _rotate_sparse_value_eager(
    value: torch.Tensor,
    *,
    phase_base: torch.Tensor,
    phase_gain: torch.Tensor,
    query_phase: torch.Tensor,
    key_phase: torch.Tensor,
    plane_shift: int,
) -> torch.Tensor:
    """Apply the four disjoint MOVT planes without cloning the full value per plane."""
    head_dim = value.shape[-1]
    rotated = value
    for plane, (channel_a, channel_b) in enumerate(
        _phase_plane_channels(R_PLANES, head_dim, plane_shift)
    ):
        theta = (
            phase_base[..., plane].unsqueeze(-1)
            + phase_gain[..., plane].unsqueeze(-1)
            * query_phase[..., plane]
            * key_phase[..., plane]
        )
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        value_a = value[..., channel_a]
        value_b = value[..., channel_b]
        rotated_a = cosine * value_a - sine * value_b
        rotated_b = sine * value_a + cosine * value_b
        mask_a = F.one_hot(
            torch.tensor(channel_a, device=value.device),
            num_classes=head_dim,
        ).to(value.dtype)
        mask_b = F.one_hot(
            torch.tensor(channel_b, device=value.device),
            num_classes=head_dim,
        ).to(value.dtype)
        rotated = rotated + (
            (rotated_a - value_a).unsqueeze(-1) * mask_a
            + (rotated_b - value_b).unsqueeze(-1) * mask_b
        )
    return rotated


def _eager_dsqg_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    scale_embed: torch.Tensor,
    phase_base: torch.Tensor,
    phase_gain: torch.Tensor,
    y_pre: torch.Tensor,
    z_pre: torch.Tensor,
    offsets: list[int] | tuple[int, ...] | torch.Tensor,
    j_small: int,
    *,
    plane_shift: int = 0,
) -> torch.Tensor:
    """Differentiable CPU/CUDA oracle for the exact DSQG semantics.

    This path is intentionally clarity-first.  It is used for CPU tests and for
    CUDA parity checks against the Triton implementation, not for production
    training.
    """
    if isinstance(offsets, torch.Tensor):
        offsets_list = [int(value) for value in offsets.detach().cpu().tolist()]
    else:
        offsets_list = [int(value) for value in offsets]
    batch, query_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if v.shape != k.shape:
        raise ValueError("k and v must have identical shapes")
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if pos_bias.shape != (len(offsets_list), query_heads):
        raise ValueError("pos_bias must have [J,Hq]")
    if scale_embed.shape != (len(offsets_list), head_dim):
        raise ValueError("scale_embed must have [J,HD]")
    if y_pre.shape != (batch, query_heads, seq_len, R_PLANES):
        raise ValueError("y_pre must have [B,Hq,N,R]")
    if z_pre.shape != (batch, kv_heads, seq_len, R_PLANES):
        raise ValueError("z_pre must have [B,Hkv,N,R]")

    head_map = torch.arange(query_heads, device=q.device) // (query_heads // kv_heads)
    key_expanded = k[:, head_map]
    value_expanded = v[:, head_map]
    z_expanded = z_pre[:, head_map]
    positions = torch.arange(seq_len, device=q.device)
    scale = 1.0 / math.sqrt(float(head_dim))
    score_columns: list[torch.Tensor] = []
    value_columns: list[torch.Tensor] = []
    valid_columns: list[torch.Tensor] = []

    for logical_index, delta in enumerate(offsets_list):
        source = positions - delta
        valid = source >= 0
        safe_source = source.clamp_min(0)
        selected_key = key_expanded[:, :, safe_source, :]
        selected_value = value_expanded[:, :, safe_source, :]
        score = (
            q.float()
            * (selected_key.float() + scale_embed[logical_index].float())
        ).sum(-1) * scale
        score = score + pos_bias[logical_index].float().reshape(1, query_heads, 1)
        score = score.masked_fill(~valid.reshape(1, 1, seq_len), float("-inf"))

        if logical_index >= j_small:
            sparse_index = logical_index - j_small
            selected_z = z_expanded[:, :, safe_source, :]
            selected_value = _rotate_sparse_value_eager(
                selected_value.float(),
                phase_base=phase_base[sparse_index].float().reshape(1, query_heads, R_PLANES),
                phase_gain=phase_gain[sparse_index].float().reshape(1, query_heads, R_PLANES),
                query_phase=y_pre.float(),
                key_phase=selected_z.float(),
                plane_shift=plane_shift,
            ).to(selected_value.dtype)
        score_columns.append(score)
        value_columns.append(selected_value)
        valid_columns.append(valid)

    scores = torch.stack(score_columns, dim=-1)
    values = torch.stack(value_columns, dim=-2)
    valid_mask = torch.stack(valid_columns, dim=-1).reshape(1, 1, seq_len, -1)
    any_valid = valid_mask.any(-1, keepdim=True)
    safe_scores = torch.where(any_valid, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(valid_mask, probabilities, torch.zeros_like(probabilities))
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1.0)
    return (probabilities.to(values.dtype).unsqueeze(-1) * values).sum(-2).to(q.dtype)


def dsqg_attention_v18_grouped(
    q,
    k,
    v,
    pos_bias,
    scale_embed,
    phase_base,
    phase_gain,
    y_pre,
    z_pre,
    j_val,
    j_small,
    j_large,
    offsets_dev,
    sparse_order_dev,
    group_start_dev,
    group_len_dev,
    group_dmin_dev,
    group_dmax_dev,
    k_group_size,
    grouped_mode_id,
    verify_mode,
    max_group_spread_observed,
    plane_shift=0,
    query_start=0,
    *,
    backend: str = "auto",
    online_softmax: bool = True,
):
    """Dispatch to Triton in production and to the eager oracle for validation."""
    del verify_mode
    backend = str(backend).lower()
    if backend not in {"auto", "triton", "eager"}:
        raise ValueError("backend must be auto, triton, or eager")
    use_triton = backend == "triton" or (
        backend == "auto" and q.is_cuda and _TRITON_AVAILABLE
    )
    if not use_triton:
        return _eager_dsqg_attention(
            q,
            k,
            v,
            pos_bias,
            scale_embed,
            phase_base,
            phase_gain,
            y_pre,
            z_pre,
            offsets_dev,
            int(j_small),
            plane_shift=int(plane_shift),
        )
    if not q.is_cuda or not _TRITON_AVAILABLE:
        raise RuntimeError("Triton DSQG was requested but CUDA/Triton is unavailable")

    original_dtype = q.dtype
    q_bf16 = q if q.dtype == torch.bfloat16 else q.to(torch.bfloat16)
    k_bf16 = k if k.dtype == torch.bfloat16 else k.to(torch.bfloat16)
    v_bf16 = v if v.dtype == torch.bfloat16 else v.to(torch.bfloat16)
    output = _DSQGV18GroupedFn.apply(
        q_bf16,
        k_bf16,
        v_bf16,
        pos_bias.float(),
        scale_embed.float(),
        phase_base.float(),
        phase_gain.float(),
        y_pre.float(),
        z_pre.float(),
        int(j_val),
        int(j_small),
        int(j_large),
        offsets_dev,
        sparse_order_dev,
        group_start_dev,
        group_len_dev,
        group_dmin_dev,
        group_dmax_dev,
        int(k_group_size),
        int(grouped_mode_id),
        False,
        int(max_group_spread_observed),
        int(plane_shift),
        int(query_start),
        bool(online_softmax),
    )
    return output if original_dtype == torch.bfloat16 else output.to(original_dtype)


# ===========================================================================
# V22 module under the checkpoint-compatible V19/V20/V21 class names
# ===========================================================================


class DSQGAttentionV19(nn.Module):
    """Grouped DSQG with fused projections and explicit semantic safeguards."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        offsets,
        j_small,
        j_large,
        seq_len=2048,
        dropout=0.1,
        k_group_size=3,
        grouped_mode="baseline",
        verify_mode=False,
        max_group_size=None,
        max_group_spread=None,
        plane_shift=0,
        pos_bias_scale=None,
        movt_dynamic_rms_target=None,
        *,
        backend: str = "auto",
        train_phase_probes: bool = False,
        scale_embed_init_std: float = 0.01,
        npci_strength_tau: float = 0.25,
        online_softmax: bool = True,
        support_crop_projections: bool = True,
        support_crop_min_offset: int = 64,
    ):
        super().__init__()
        dimension = int(embedding_dim)
        heads = int(num_heads)
        if dimension % heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = heads
        self.head_dim = dimension // heads
        if self.head_dim < 16:
            raise ValueError("DSQG Triton paths require head_dim >= 16")
        self.seq_len = int(seq_len)
        self.backend = str(backend).lower()
        if self.backend not in {"auto", "eager", "triton"}:
            raise ValueError("backend must be auto, eager, or triton")
        if not math.isfinite(scale_embed_init_std) or scale_embed_init_std < 0:
            raise ValueError("scale_embed_init_std must be finite and non-negative")
        if not math.isfinite(npci_strength_tau) or npci_strength_tau <= 0:
            raise ValueError("npci_strength_tau must be finite and positive")
        if isinstance(support_crop_min_offset, bool) or int(support_crop_min_offset) < 0:
            raise ValueError("support_crop_min_offset must be a non-negative integer")
        self.scale_embed_init_std = float(scale_embed_init_std)
        self.npci_strength_tau = float(npci_strength_tau)
        self.train_phase_probes = bool(train_phase_probes)
        self.support_crop_projections = bool(support_crop_projections)
        self.support_crop_min_offset = int(support_crop_min_offset)
        self.movt_dynamic_rms_target = (
            None if movt_dynamic_rms_target is None else float(movt_dynamic_rms_target)
        )
        self.movt_phase_gain_init_std = (
            0.001
            if self.movt_dynamic_rms_target is None
            else calibrated_movt_phase_gain_std(
                head_dim=self.head_dim,
                target_dynamic_rms=self.movt_dynamic_rms_target,
                gate_logit=0.0,
            )
        )
        self.grouped_mode = str(grouped_mode)
        self.grouped_mode_id = _resolve_grouped_mode(self.grouped_mode)
        # The online kernel currently targets the production baseline traversal.
        # Overlap-slab modes retain their parity-validated offline forward.
        self.online_softmax = bool(online_softmax) and self.grouped_mode == "baseline"
        self.verify_mode = bool(verify_mode)
        k_group_size, max_group_size, max_group_spread = _normalize_grouping_args(
            k_group_size,
            max_group_size,
            max_group_spread,
            self.grouped_mode_id,
        )
        self.k_group_size = k_group_size
        self.max_group_size = max_group_size
        self.max_group_spread = max_group_spread
        self.plane_shift = _normalize_plane_shift(
            R_PLANES,
            self.head_dim,
            int(plane_shift),
        )
        canonical_offsets, j_small, j_large = _canonicalize_offsets(
            list(offsets),
            j_small,
            j_large,
        )
        self.offsets = tuple(canonical_offsets)
        self.j_val = len(canonical_offsets)
        self.j_small = int(j_small)
        self.j_large = int(j_large)
        if self.j_val < 1:
            raise ValueError("DSQG requires at least one offset")
        self.minimum_offset = min(self.offsets)

        self.register_buffer(
            "offsets_dev",
            torch.tensor(canonical_offsets, dtype=torch.int32),
            persistent=False,
        )
        plan = _plan_sparse_groups(
            offsets=canonical_offsets,
            j_small=self.j_small,
            k_group_size=k_group_size,
            max_group_size=max_group_size,
            max_group_spread=max_group_spread,
        )
        if self.verify_mode:
            _verify_sparse_plan(canonical_offsets, self.j_small, plan)
        self.register_buffer(
            "sparse_order_dev",
            torch.tensor(plan["sparse_order"], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "group_start_dev",
            torch.tensor(plan["group_start"], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "group_len_dev",
            torch.tensor(plan["group_len"], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "group_dmin_dev",
            torch.tensor(plan["group_dmin"], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "group_dmax_dev",
            torch.tensor(plan["group_dmax"], dtype=torch.int32),
            persistent=False,
        )
        self.max_group_len = int(plan["max_group_len"])
        self.max_group_spread_observed = int(plan["max_group_spread"])
        self.num_sparse_groups = int(len(plan["group_len"]))
        if pos_bias_scale is None:
            # Standalone callers retain the old environment fallback. Production
            # trainers should always pass an explicit, checkpointed value.
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
        self.phase_base = nn.Parameter(
            torch.randn(max(self.j_large, 1), heads, R_PLANES) * 0.01
        )
        self.phase_gain = nn.Parameter(
            torch.randn(max(self.j_large, 1), heads, R_PLANES)
            * self.movt_phase_gain_init_std
        )
        self.phase_gate = nn.Parameter(torch.zeros(max(self.j_large, 1)))
        query_probes, key_probes = _orthogonal_phase_probes(
            R_PLANES,
            self.head_dim,
            self.plane_shift,
        )
        self.query_probes = nn.Parameter(
            query_probes,
            requires_grad=self.train_phase_probes,
        )
        self.key_probes = nn.Parameter(
            key_probes,
            requires_grad=self.train_phase_probes,
        )
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
        return self.scale_embed - self.scale_embed.mean(dim=0, keepdim=True)

    def reset_phase_probes_(self) -> None:
        query, key = _orthogonal_phase_probes(
            R_PLANES,
            self.head_dim,
            self.plane_shift,
        )
        with torch.no_grad():
            self.query_probes.copy_(query.to(self.query_probes))
            self.key_probes.copy_(key.to(self.key_probes))

    def phase_probe_regularizer(self) -> torch.Tensor:
        if not self.train_phase_probes:
            return self.query_probes.sum() * 0.0
        query = F.normalize(self.query_probes.float(), dim=-1)
        key = F.normalize(self.key_probes.float(), dim=-1)
        identity = torch.eye(R_PLANES, device=query.device)
        q_gram = query @ query.t()
        k_gram = key @ key.t()
        # The deterministic initialization uses a quarter-turn Q/K relation.
        target_cross = torch.zeros_like(identity)
        return (
            (q_gram - identity).square().mean()
            + (k_gram - identity).square().mean()
            + (query @ key.t() - target_cross).square().mean()
        )

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
            "implementation": "dsqg-v22-fused",
            "offsets": self.offsets,
            "j_small": self.j_small,
            "j_large": self.j_large,
            "plane_shift": self.plane_shift,
            "positional_bias": "analytic_log_plus_residual",
            "pos_bias_scale": float(self.pos_bias_scale.detach().cpu()),
            "scale_embed": "centered_virtual_key",
            "phase_probes_trainable": self.train_phase_probes,
            "npci": "magnitude_aware",
            "npci_strength_tau": self.npci_strength_tau,
        }

    def execution_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "grouped_mode": self.grouped_mode,
            "online_softmax": self.online_softmax,
            "query_support_start": self.minimum_offset,
            "support_crop_projections": self.support_crop_projections,
            "support_crop_min_offset": self.support_crop_min_offset,
            "phase_probe_kernel": "fused_triton_or_eager",
            "k_group_size": self.k_group_size,
            "max_group_size": self.max_group_size,
            "max_group_spread": self.max_group_spread,
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
        # The revised output projection is bias-free so unsupported rows stay exact zero.
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

        # Convert the old unconstrained [J,H] table exactly into analytic base
        # plus residual.  Constant scale-embedding rows already represented a
        # softmax-null direction, so centering them preserves their behavior.
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
            # The supplied checkpoint stores the raw table; the old forward
            # multiplied that table by pos_bias_scale.  Keeping the same raw
            # value inside (analytic + residual) therefore preserves the exact
            # effective bias for every finite non-zero scale.
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
            # No offset has support at this runtime length. The bias-free branch
            # is exactly zero and none of its projection work is required.
            return torch.zeros_like(x)

        crop = (
            self.support_crop_projections
            and query_start >= self.support_crop_min_offset
        )
        if crop:
            # For a band whose minimum offset is d0, Q/gate rows before d0 and
            # K/V rows at or beyond N-d0 are provably outside the graph support.
            # Compute only the live suffix/prefix, then place them into full-stride
            # views expected by the shared sparse kernel.
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
                # Legacy banded blocks can construct the EMA/NPCI packet only
                # on the source prefix that any offset can actually consume.
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

        query_probe = F.normalize(self.query_probes.float(), dim=-1)
        key_probe = F.normalize(self.key_probes.float(), dim=-1)

        query_float = query.float()
        key_float = key.float()

        probe_epsilon = 1e-6 * head_dim

        query_inverse_norm = torch.rsqrt(
            query_float.square().sum(-1, keepdim=True) + probe_epsilon
        )
        key_inverse_norm = torch.rsqrt(
            key_float.square().sum(-1, keepdim=True) + probe_epsilon
        )

        y_pre = (
            torch.einsum("bhnd,rd->bhnr", query_float, query_probe)
            * query_inverse_norm
        ).contiguous()

        z_pre = (
            torch.einsum("bhnd,rd->bhnr", key_float, key_probe)
            * key_inverse_norm
        ).contiguous()

        phase_gate = torch.sigmoid(self.phase_gate)[:, None, None]
        output = dsqg_attention_v18_grouped(
            query,
            key,
            value,
            self.pos_bias,
            self.centered_scale_embed,
            self.phase_base * phase_gate,
            self.phase_gain * phase_gate,
            y_pre,
            z_pre,
            self.j_val,
            self.j_small,
            self.j_large,
            self.offsets_dev,
            self.sparse_order_dev,
            self.group_start_dev,
            self.group_len_dev,
            self.group_dmin_dev,
            self.group_dmax_dev,
            self.k_group_size,
            self.grouped_mode_id,
            self.verify_mode,
            self.max_group_spread_observed,
            self.plane_shift,
            query_start,
            backend=self.backend,
            online_softmax=self.online_softmax,
        )
        output = output * self.if_gain.reshape(1, heads, 1, 1)
        flattened = output.permute(0, 2, 1, 3).reshape(batch, seq_len, dimension)
        return self.dropout(
            self.out_proj(flattened * torch.sigmoid(content_gate))
        )


DSQGAttentionV20 = DSQGAttentionV19
DSQGAttentionV21 = DSQGAttentionV19
DSQGAttentionV22 = DSQGAttentionV19


if __name__ == "__main__":
    torch.manual_seed(5)
    module = DSQGAttentionV19(
        64,
        4,
        [1, 2, 4, 8, 16, 48],
        5,
        1,
        seq_len=64,
        dropout=0.0,
        backend="eager",
        train_phase_probes=True,
    ).double()
    values = torch.randn(2, 31, 64, dtype=torch.float64, requires_grad=True)
    output = module(values)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
    (output.square().mean() + 1e-3 * module.phase_probe_regularizer()).backward()
    assert torch.isfinite(values.grad).all()
    module.eval()
    with torch.no_grad():
        baseline = module(values.detach())
        changed = values.detach().clone()
        changed[:, 20:] += torch.randn_like(changed[:, 20:]) * 10
        perturbed = module(changed)
        assert torch.equal(baseline[:, :20], perturbed[:, :20])
    print("CPU DSQG forward/backward/causality smoke test: PASS")

