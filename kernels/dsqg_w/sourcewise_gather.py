from __future__ import annotations

import torch

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
    def _dsqg_w_candidate_state_gather_kernel(
        x_ptr,
        l3_ptr,
        token_ptr,
        source_ptr,
        mask_ptr,
        out_ptr,
        total: tl.constexpr,
        n: tl.constexpr,
        j_count: tl.constexpr,
        d: tl.constexpr,
        has_l3: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        active = offsets < total
        dim = offsets % d
        slot = (offsets // d) % j_count
        pos = (offsets // (d * j_count)) % n
        batch = offsets // (d * j_count * n)
        meta_offset = (batch * n + pos) * j_count + slot
        valid = active & (tl.load(mask_ptr + meta_offset, mask=active, other=0) != 0)
        source = tl.load(source_ptr + meta_offset, mask=active, other=0)
        token = tl.load(token_ptr + meta_offset, mask=active, other=0)
        token = tl.minimum(tl.maximum(token, 0), n - 1)
        source_l3 = (source == 2) | (source == 3)
        source_null = source == 0
        use_l3 = source_l3 & has_l3
        use_x = (~use_l3) & (~source_null)
        base_offset = (batch * n + token) * d + dim
        x_val = tl.load(x_ptr + base_offset, mask=valid & use_x, other=0.0)
        l3_val = tl.load(l3_ptr + base_offset, mask=valid & use_l3, other=0.0)
        # SUMMARY uses x on this fast path; callers with real chunk_rep_states
        # fall back before launching this kernel.
        val = x_val + l3_val
        tl.store(out_ptr + offsets, val, mask=active)


    @triton.jit
    def _dsqg_w_candidate_state_gather_backward_kernel(
        grad_out_ptr,
        token_ptr,
        source_ptr,
        mask_ptr,
        grad_x_ptr,
        grad_l3_ptr,
        total: tl.constexpr,
        n: tl.constexpr,
        j_count: tl.constexpr,
        d: tl.constexpr,
        has_l3: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        active = offsets < total
        dim = offsets % d
        slot = (offsets // d) % j_count
        pos = (offsets // (d * j_count)) % n
        batch = offsets // (d * j_count * n)
        meta_offset = (batch * n + pos) * j_count + slot
        valid = active & (tl.load(mask_ptr + meta_offset, mask=active, other=0) != 0)
        source = tl.load(source_ptr + meta_offset, mask=active, other=0)
        token = tl.load(token_ptr + meta_offset, mask=active, other=0)
        token = tl.minimum(tl.maximum(token, 0), n - 1)
        source_l3 = (source == 2) | (source == 3)
        source_null = source == 0
        use_l3 = source_l3 & has_l3
        use_x = (~use_l3) & (~source_null)
        base_offset = (batch * n + token) * d + dim
        grad = tl.load(grad_out_ptr + offsets, mask=valid, other=0.0)
        tl.atomic_add(grad_x_ptr + base_offset, grad, sem="relaxed", mask=valid & use_x)
        tl.atomic_add(grad_l3_ptr + base_offset, grad, sem="relaxed", mask=valid & use_l3)
else:
    _dsqg_w_candidate_state_gather_kernel = None
    _dsqg_w_candidate_state_gather_backward_kernel = None

class _DSQGWSourcewiseCandidateStateGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l3_states, cand_token_indices, cand_sources, cand_mask, has_l3: bool):
        if not _TRITON_SOURCEWISE_AVAILABLE or triton is None or _dsqg_w_candidate_state_gather_kernel is None:
            raise NotImplementedError("DSQG-W sourcewise candidate-state gather requires Triton")
        bsz, seq_len, d = x.shape
        j_count = cand_token_indices.shape[-1]
        out = torch.empty((bsz, seq_len, j_count, d), device=x.device, dtype=x.dtype)
        block = 256
        total = int(out.numel())
        l3_arg = l3_states if bool(has_l3) else x
        _dsqg_w_candidate_state_gather_kernel[(triton.cdiv(total, block),)](
            x.contiguous(),
            l3_arg.contiguous(),
            cand_token_indices.contiguous(),
            cand_sources.contiguous(),
            cand_mask.contiguous(),
            out,
            total=total,
            n=seq_len,
            j_count=j_count,
            d=d,
            has_l3=bool(has_l3),
            block=block,
        )
        ctx.save_for_backward(cand_token_indices, cand_sources, cand_mask)
        ctx.x_shape = tuple(x.shape)
        ctx.has_l3 = bool(has_l3)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        cand_token_indices, cand_sources, cand_mask = ctx.saved_tensors
        bsz, seq_len, d = ctx.x_shape
        j_count = cand_token_indices.shape[-1]
        grad_x = torch.zeros((bsz, seq_len, d), device=grad_out.device, dtype=grad_out.dtype)
        grad_l3 = torch.zeros_like(grad_x) if ctx.has_l3 else None
        block = 256
        total = int(grad_out.numel())
        grad_l3_arg = grad_l3 if grad_l3 is not None else grad_x
        _dsqg_w_candidate_state_gather_backward_kernel[(triton.cdiv(total, block),)](
            grad_out.contiguous(),
            cand_token_indices.contiguous(),
            cand_sources.contiguous(),
            cand_mask.contiguous(),
            grad_x,
            grad_l3_arg,
            total=total,
            n=seq_len,
            j_count=j_count,
            d=d,
            has_l3=ctx.has_l3,
            block=block,
        )
        return grad_x, grad_l3, None, None, None, None

__all__ = [
    "_DSQGWSourcewiseCandidateStateGather",
    "_dsqg_w_candidate_state_gather_kernel",
    "_dsqg_w_candidate_state_gather_backward_kernel",
]
