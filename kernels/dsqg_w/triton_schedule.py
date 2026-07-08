from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _DSQGWTritonSchedule:
    """Centralized launch schedule for DSQG-W Triton row/head kernels.

    Mirrors V20's discipline of deriving launch shape from head dimension and
    SM family in one place. Values are adapted to DSQG-W's one-row/one-head
    programs rather than copied from V20's BLOCK_N x HD tile kernels.
    """

    block_hd: int
    num_warps: int
    num_stages: int


def _next_pow2_int(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _dsqg_w_triton_schedule(head_dim: int, device: torch.device | None = None) -> _DSQGWTritonSchedule:
    block_hd = _next_pow2_int(int(head_dim))
    if block_hd <= 64:
        base_warps = 1
    elif block_hd <= 128:
        base_warps = 2
    else:
        base_warps = 4

    num_stages = 2
    if device is not None and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(device)
        except Exception:
            major, minor = (0, 0)
        if major >= 9:
            base_warps = min(max(base_warps, 2), 4)
            num_stages = 3
        elif major == 8 and minor == 9:
            num_stages = 2
    return _DSQGWTritonSchedule(block_hd=block_hd, num_warps=base_warps, num_stages=num_stages)

__all__ = [
    "_DSQGWTritonSchedule",
    "_next_pow2_int",
    "_dsqg_w_triton_schedule",
]
