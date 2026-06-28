"""q6_g128 kernels and layout helpers for DWARF DSQG attention.

This package consolidates the formerly Phase-3 verification modules used by the
q6_g128/KV-head-aware trainer path.  The public modules are:

- pack: signed q6 packing/unpacking and q6_g128 tensor packing
- layout: DWARF [B,H,N,64] q6_g128 cache layout
- decode: Triton q6 decode/gather kernels
- fused_consume: Triton DSQG direct-consume forward kernels, including Hq/Hkv support

The Stage-E/F backward Triton kernels are still trainer-integrated while the
long q6 run is active; see train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py.
"""

from .layout import DwarfQ6G128CacheLayout, pack_q6_g128_cache_layout
from .decode import triton_direct_decode_gather, triton_tile_scoped_decode_scratch_then_gather
from .fused_consume import triton_q6_g128_dsqg_direct_consume

__all__ = [
    "DwarfQ6G128CacheLayout",
    "pack_q6_g128_cache_layout",
    "triton_direct_decode_gather",
    "triton_tile_scoped_decode_scratch_then_gather",
    "triton_q6_g128_dsqg_direct_consume",
]
