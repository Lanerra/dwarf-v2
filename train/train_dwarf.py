#!/usr/bin/env python3
"""Self-contained public trainer for the canonical DWARF-v2 architecture.

The model is D512/H8/L12/FFN2048 with bounded-routing DSQG V23 blocks and
strict-causal HISA V19 global mixers at layers 3 and 9. The first global mixer
receives the causal-EMA interference packet; both mixers include the learned
semantic binder used by the current DWARF lineage.

The trainer accepts packed token rows and includes Muon+AdamW, WSD, deterministic
row selection, atomic resumable checkpoints, and the complete model definition.
Dataset preparation and evaluation remain out of scope.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_FILES = (
    "causal_ema_scan.py",
    "dsqg_attention_v23.py",
    "hierarchical_sparse_attn_v19_hisa.py",
)
KERNEL_DIR = next(
    (
        candidate
        for candidate in (
            SCRIPT_DIR,
            SCRIPT_DIR / "kernels",
            SCRIPT_DIR.parent / "kernels",
            SCRIPT_DIR.parent.parent / "kernels",
        )
        if all((candidate / name).is_file() for name in KERNEL_FILES)
    ),
    SCRIPT_DIR,
)
for directory in (SCRIPT_DIR, KERNEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from causal_ema_scan import (  # noqa: E402
    bounded_ema_factor,
    causal_ema_scan3,
    inverse_bounded_ema_factor,
)
from dsqg_attention_v23 import (  # noqa: E402
    ALL_OFFSETS,
    DSQGAttentionV23,
)
from hierarchical_sparse_attn_v19_hisa import (  # noqa: E402
    HierarchicalSparseAttentionV19HISACausal,
)

CHECKPOINT_KIND = "dwarf-58m-dsqgv23-hisav19-resume-v1"
ROUTE_AUX_RECIPE_SEED = 20_260_809
EXPECTED_PARAMETERS = 58_591_773
EXPECTED_TRAINABLE_PARAMETERS = 58_591_757
EXPECTED_STATE_FINGERPRINT = (
    "60db70dabe13746b81bbbd7256be0a7e34aa7e6d0e3ed3c7fb9ccc8b5250a69e"
)
EXPECTED_SEEDED_RNG_FINGERPRINT = (
    "366b83cd1650cf2de8ccd2ffe5fb3ef27344c37e53c9ee57bd6c0914dd44c9b6"
)
@dataclass(frozen=True)
class TrainRecipe:
    learning_rate: float = 3.0e-4
    batch_size: int = 15
    grad_accum_steps: int = 14
    steps: int = 4653
    warmup_steps: int = 233
    stable_steps: int = 3722
    decay_steps: int = 698
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip_muon: float = 1.0
    grad_clip_adamw: float = 1.0

    def __post_init__(self) -> None:
        if self.warmup_steps + self.stable_steps + self.decay_steps != self.steps:
            raise ValueError("WSD phases must sum to the total update count")

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def checkpoint_steps(self) -> set[int]:
        return {
            self.warmup_steps,
            math.ceil(self.steps * 0.25),
            math.ceil(self.steps * 0.50),
            math.ceil(self.steps * 0.75),
            self.steps,
        }


@dataclass(frozen=True)
class DwarfConfig:
    vocab_size: int = 32768
    embedding_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    seq_len: int = 2048
    num_layers: int = 12
    global_mixer_layers: tuple[int, ...] = (3, 9)
    dropout: float = 0.05
    min_offset_support: int = 64
    hisa_chunk_size: int = 32
    top_k_chunks: int = 4
    hisa_top_m_tokens: int = 32
    hisa_local_window: int = 64
    hisa_selector_tile: int = 16
    hisa_chunk_selection_scope: str = "token"
    hisa_token_routing_pack_size: int = 4
    hisa_global_adapter_rank: int = 64
    hisa_binding_rank: int = 64
    hisa_route_aux_weight: float = 0.02
    hisa_route_aux_samples: int = 8
    hisa_route_aux_temperature: float = 0.5
    hisa_route_aux_oracle_temperature: float = 0.3
    hisa_exploration_probability: float = 0.10
    hisa_backend: str = "triton"
    hisa_token_selection_mode: str = "auto"
    hisa_local_backend: str = "flex"
    hisa_boundary_bridge: bool = True
    hisa_triton_block_q: int = 16
    hisa_backward_impl: str = "atomic_masked"
    hisa_collect_routing_diagnostics: bool = False
    hisa_diagnostic_max_queries: int = 8
    ema_timescales: tuple[float, ...] = (16.0, 64.0, 256.0)
    eos_token_id: int = 1
    pad_token_id: int = 2
    eod_token_id: int = 4
    init_seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers < 4:
            raise ValueError("DWARF requires at least four blocks")
        if (
            not self.global_mixer_layers
            or tuple(sorted(set(self.global_mixer_layers)))
            != self.global_mixer_layers
            or self.global_mixer_layers[0] < 0
            or self.global_mixer_layers[-1] >= self.num_layers
        ):
            raise ValueError(
                "global_mixer_layers must be sorted unique in-range block indices"
            )
        if self.seq_len < 65:
            raise ValueError("seq_len must be at least 65")
        head_dim = self.embedding_dim // self.num_heads
        if head_dim & (head_dim - 1):
            raise ValueError("HISA requires a power-of-two head dimension")
        if self.hisa_chunk_selection_scope not in {"token", "tile"}:
            raise ValueError("HISA chunk selection scope must be token or tile")
        if self.hisa_token_routing_pack_size not in {1, 2, 4, 8, 16}:
            raise ValueError("HISA token routing pack size must be 1, 2, 4, 8, or 16")
        if self.hisa_backend not in {"auto", "eager", "triton"}:
            raise ValueError("HISA backend must be auto, eager, or triton")
        if self.hisa_token_selection_mode not in {"auto", "canonical"}:
            raise ValueError("HISA token selection mode must be auto or canonical")
        if self.hisa_local_backend not in {"flex", "combined"}:
            raise ValueError("HISA local backend must be flex or combined")
        if self.hisa_triton_block_q < 16 or self.hisa_triton_block_q & (
            self.hisa_triton_block_q - 1
        ):
            raise ValueError("HISA Triton BLOCK_Q must be a power of two >=16")
        if self.hisa_backward_impl not in {"atomic", "atomic_masked"}:
            raise ValueError("HISA backward implementation must be atomic or atomic_masked")
        if not isinstance(self.hisa_collect_routing_diagnostics, bool):
            raise TypeError("HISA routing diagnostics flag must be bool")
        if self.hisa_diagnostic_max_queries < 1:
            raise ValueError("HISA diagnostic max queries must be positive")

    @property
    def model_length(self) -> int:
        return self.seq_len - 1

    @property
    def swiglu_dim(self) -> int:
        return max(64, int(round((2.0 * self.ffn_dim / 3.0) / 64.0)) * 64)


RECIPE = TrainRecipe()


def build_offset_groups(config: DwarfConfig) -> tuple[tuple[int, ...], ...]:
    offsets = tuple(sorted(int(value) for value in ALL_OFFSETS))
    if len(offsets) != len(set(offsets)):
        raise ValueError("canonical DSQG offsets must be unique")
    offsets = tuple(
        offset
        for offset in offsets
        if config.model_length - offset >= config.min_offset_support
    )
    shared_local = tuple(offset for offset in offsets if offset <= 8)
    distributed = tuple(offset for offset in offsets if offset > 8)
    groups = tuple(
        tuple(sorted((*shared_local, *distributed[index::3])))
        for index in range(3)
    )
    if any(not group for group in groups):
        raise ValueError("offset pruning left an empty DSQG group")
    return groups


def route_aux_tile_ids_for_update(
    global_step: int,
    device: torch.device | str,
    config: DwarfConfig | None = None,
) -> torch.Tensor:
    """Derive one deterministic auxiliary sample shared by an optimizer update."""
    config = config or DwarfConfig()
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise ValueError("global optimizer step must be a positive integer")
    first_useful = math.ceil(config.hisa_chunk_size + config.hisa_local_window)
    candidate_count = config.model_length - first_useful
    sample_count = min(config.hisa_route_aux_samples, candidate_count)
    if sample_count != config.hisa_route_aux_samples:
        raise ValueError("production sequence has too few route-auxiliary tiles")
    material = f"{ROUTE_AUX_RECIPE_SEED}:{global_step}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randperm(candidate_count, generator=generator)[:sample_count]
    return (ids + first_useful).to(device=torch.device(device), dtype=torch.int64)


def _consume_retired_movt_rng(
    offsets: Iterable[int],
    num_heads: int,
    head_dim: int,
    *,
    reset_path: bool,
) -> None:
    """Preserve the released scratch-initialization RNG lineage only.

    MOVT no longer has model state or runtime semantics. The released constructor
    and reset path nevertheless consumed two normal draws before later parameters;
    keeping those draws here preserves seeded weights and the subsequent RNG stream.
    """
    shape = (max(sum(int(offset) >= 48 for offset in offsets), 1), num_heads, 4)
    if reset_path:
        nn.init.normal_(torch.empty(shape), mean=0.0, std=0.01)
        nn.init.normal_(torch.empty(shape), mean=0.0, std=0.02 * head_dim)
    else:
        torch.randn(shape)
        torch.randn(shape)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    locations = {
        "train/train_dwarf.py": Path(__file__).resolve(),
        "kernels/causal_ema_scan.py": Path(
            sys.modules["causal_ema_scan"].__file__
        ).resolve(),
        "kernels/dsqg_attention_v23.py": Path(
            sys.modules["dsqg_attention_v23"].__file__
        ).resolve(),
        "kernels/hierarchical_sparse_attn_v19_hisa.py": Path(
            sys.modules["hierarchical_sparse_attn_v19_hisa"].__file__
        ).resolve(),
    }
    return {name: _sha256(path) for name, path in locations.items()}


def validate_checkpoint_architecture(
    saved: dict[str, Any], current: dict[str, Any]
) -> None:
    if saved != current:
        raise ValueError("checkpoint architecture or source manifest does not match")


def assert_public_kernel_contracts() -> None:
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(17)
        kwargs = {
            "embedding_dim": 64,
            "num_heads": 4,
            "offsets": (29, 32, 47),
            "seq_len": 64,
            "dropout": 0.0,
            "backend": "eager",
        }
        module = DSQGAttentionV23(**kwargs).double()
        assert module.offsets == (29, 32, 47)
        restored = DSQGAttentionV23(**kwargs).double()
        incompatible = restored.load_state_dict(module.state_dict(), strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        values = torch.randn(2, 64, 64, dtype=torch.float64, requires_grad=True)
        output = module(values)
        assert torch.equal(output[:, :29], torch.zeros_like(output[:, :29]))
        output.square().mean().backward()
        assert torch.isfinite(values.grad).all()
    finally:
        torch.random.set_rng_state(state)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: DwarfConfig) -> None:
        super().__init__()
        hidden = config.swiglu_dim
        self.up_gate = nn.Linear(config.embedding_dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, config.embedding_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * up))


class InterferencePacket(nn.Module):
    def __init__(self, config: DwarfConfig) -> None:
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        raw = [inverse_bounded_ema_factor(1.0 / value) for value in config.ema_timescales]
        self.ema_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.mix_logits = nn.Parameter(torch.zeros(self.heads, len(raw)))
        self.gate_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.kv_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim, bias=False)
        with torch.no_grad():
            self.gate_proj.bias.fill_(-2.0)

    @property
    def ema_factors(self) -> torch.Tensor:
        return bounded_ema_factor(self.ema_raw)

    def forward(
        self,
        normalized: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scan_input = normalized.to(torch.bfloat16) if normalized.is_cuda else normalized
        scans = causal_ema_scan3(
            scan_input,
            self.ema_factors,
            reset_mask=reset_mask,
            lagged=True,
        )
        batch, seq_len, _ = normalized.shape
        stacked = scans.reshape(
            batch,
            seq_len,
            len(self.ema_factors),
            self.heads,
            self.head_dim,
        )
        mixture = torch.softmax(self.mix_logits.float(), dim=-1)
        pooled = (
            stacked
            * mixture.transpose(0, 1)
            .reshape(1, 1, len(self.ema_factors), self.heads, 1)
            .to(stacked.dtype)
        ).sum(2).reshape(batch, seq_len, -1)
        pooled_float = pooled.float()
        mean_square = pooled_float.square().mean(-1, keepdim=True)
        rms = (mean_square + 1e-6).sqrt()
        direction = pooled_float * torch.rsqrt(mean_square + 1e-6)
        confidence = torch.tanh(rms / 0.25)
        gate = torch.sigmoid(self.gate_proj(normalized).float())
        packet = (direction * confidence * gate).to(normalized.dtype)
        key_delta, value_delta = self.kv_proj(packet).chunk(2, dim=-1)
        key_delta = key_delta.reshape(batch, seq_len, self.heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        value_delta = value_delta.reshape(
            batch, seq_len, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        return key_delta, value_delta


class DSQGBlock(nn.Module):
    def __init__(
        self,
        config: DwarfConfig,
        offsets: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        self.attn = DSQGAttentionV23(
            config.embedding_dim,
            config.num_heads,
            offsets,
            seq_len=config.seq_len,
            dropout=config.dropout,
            pos_bias_scale=0.25,
            scale_embed_init_std=0.01,
            scale_embed_max_norm=0.25,
            null_key_max_norm=0.25,
            null_bias_limit=6.0,
            pos_bias_max_slope=0.75,
            pos_bias_residual_limit=1.5,
            support_crop_projections=True,
            support_crop_min_offset=64,
        )
        _consume_retired_movt_rng(
            offsets,
            config.num_heads,
            config.embedding_dim // config.num_heads,
            reset_path=False,
        )
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GlobalMixerBlock(nn.Module):
    def __init__(self, config: DwarfConfig, *, use_packet: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        self.attn = HierarchicalSparseAttentionV19HISACausal(
            D=config.embedding_dim,
            H=config.num_heads,
            hd=config.embedding_dim // config.num_heads,
            chunk_size=config.hisa_chunk_size,
            top_k_chunks=config.top_k_chunks,
            hisa_top_m_tokens=config.hisa_top_m_tokens,
            local_window=config.hisa_local_window,
            selector_tile_size=config.hisa_selector_tile,
            chunk_selection_scope=config.hisa_chunk_selection_scope,
            token_routing_pack_size=config.hisa_token_routing_pack_size,
            representative_mode="mean_max_blend",
            representative_blend_alpha=0.5,
            route_prior_scale=0.1,
            route_prior_max_scale=2.0,
            route_aux_weight=config.hisa_route_aux_weight,
            route_aux_samples=config.hisa_route_aux_samples,
            route_aux_temperature=config.hisa_route_aux_temperature,
            route_aux_oracle_temperature=config.hisa_route_aux_oracle_temperature,
            exploration_probability=config.hisa_exploration_probability,
            global_adapter_rank=config.hisa_global_adapter_rank,
            binding_rank=config.hisa_binding_rank,
            npci_theta_max=0.25,
            max_seq_len=config.model_length,
            backend=config.hisa_backend,
            token_selection_mode=config.hisa_token_selection_mode,
            local_backend=config.hisa_local_backend,
            boundary_bridge=config.hisa_boundary_bridge,
            triton_block_q=config.hisa_triton_block_q,
            backward_impl=config.hisa_backward_impl,
            collect_routing_diagnostics=config.hisa_collect_routing_diagnostics,
            diagnostic_max_queries=config.hisa_diagnostic_max_queries,
        )
        self.packet = InterferencePacket(config) if use_packet else None
        if self.packet is None:
            self.attn.npci_theta_k.requires_grad_(False)
            self.attn.npci_theta_v.requires_grad_(False)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        ema_reset_mask: torch.Tensor | None = None,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        kv_inject = (
            self.packet(normalized, ema_reset_mask)
            if self.packet is not None
            else None
        )
        attended, auxiliary = self.attn(
            normalized,
            kv_inject=kv_inject,
            valid_lengths=valid_lengths,
            route_aux_tile_ids=route_aux_tile_ids,
            collect_diagnostics=collect_diagnostics,
            return_auxiliary=True,
        )
        x = x + self.dropout(attended)
        return x + self.ffn(self.norm2(x)), auxiliary


class DwarfForCausalLM(nn.Module):
    def __init__(self, config: DwarfConfig | None = None) -> None:
        super().__init__()
        self.config = config or DwarfConfig()
        config = self.config
        self.offset_groups = build_offset_groups(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        global_layers = set(config.global_mixer_layers)
        layout: list[int | None] = []
        next_group = 0
        for layer_index in range(config.num_layers):
            if layer_index in global_layers:
                layout.append(None)
            else:
                layout.append(next_group)
                next_group = (next_group + 1) % len(self.offset_groups)
        blocks: list[nn.Module] = []
        global_index = 0
        for group_index in layout:
            if group_index is None:
                blocks.append(GlobalMixerBlock(config, use_packet=global_index == 0))
                global_index += 1
                continue
            blocks.append(DSQGBlock(config, self.offset_groups[group_index]))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        state = torch.random.get_rng_state()
        torch.manual_seed(self.config.init_seed + 20_001)
        try:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            for module in self.modules():
                if isinstance(module, DSQGAttentionV23):
                    _consume_retired_movt_rng(
                        module.offsets,
                        module.num_heads,
                        module.head_dim,
                        reset_path=True,
                    )
                    nn.init.normal_(
                        module.scale_embed,
                        mean=0.0,
                        std=module.scale_embed_init_std,
                    )
                    with torch.no_grad():
                        module.scale_embed.sub_(module.scale_embed.mean(0, keepdim=True))
                elif isinstance(module, HierarchicalSparseAttentionV19HISACausal):
                    module.reset_global_adapters_()
                elif isinstance(module, InterferencePacket):
                    with torch.no_grad():
                        module.gate_proj.bias.fill_(-2.0)
        finally:
            torch.random.set_rng_state(state)

    def prepare_runtime(self, device: torch.device | str) -> None:
        for module in self.modules():
            if isinstance(module, HierarchicalSparseAttentionV19HISACausal):
                module.prepare_runtime(device, self.config.model_length)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        ema_reset_mask: torch.Tensor | None = None,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_auxiliary: bool = False,
    ):
        x = self.embedding(input_ids)
        if x.is_cuda:
            x = x.to(torch.bfloat16)
        x = self.dropout(x)
        auxiliary = x.new_zeros(())
        for block in self.blocks:
            if isinstance(block, GlobalMixerBlock):
                x, block_auxiliary = block(
                    x,
                    ema_reset_mask,
                    valid_lengths,
                    route_aux_tile_ids,
                    collect_diagnostics,
                )
                auxiliary = auxiliary + block_auxiliary
            else:
                x = block(x)
        hidden = self.norm(x)
        if hidden.is_cuda:
            hidden = hidden.to(torch.bfloat16)
        return (hidden, auxiliary) if return_auxiliary else hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        ema_reset_mask: torch.Tensor | None = None,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_hidden: bool = False,
        return_auxiliary: bool = False,
    ):
        hidden, auxiliary = self.forward_hidden(
            input_ids,
            ema_reset_mask=ema_reset_mask,
            valid_lengths=valid_lengths,
            route_aux_tile_ids=route_aux_tile_ids,
            collect_diagnostics=collect_diagnostics,
            return_auxiliary=True,
        )
        output = hidden if return_hidden else self.lm_head(hidden)
        return (output, auxiliary) if return_auxiliary else output


def model_metadata(model: DwarfForCausalLM) -> dict[str, Any]:
    global_mixers = [
        (index, block)
        for index, block in enumerate(model.blocks)
        if isinstance(block, GlobalMixerBlock)
    ]
    layer_names = [
        "HISA" if isinstance(block, GlobalMixerBlock) else "DSQG"
        for block in model.blocks
    ]
    return {
        "format": "dwarf-58m-dsqgv23-hisav19-v1",
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "topology": {
            "layers": ",".join(layer_names),
            "global_mixer_layers": tuple(index for index, _ in global_mixers),
            "dsqg": "v23-bounded-routing-null-candidate",
            "hisa": "v19-accessible-routing-semantic-binder",
            "offset_groups": model.offset_groups,
        },
        "hisa": [
            {
                "layer": index,
                "ema_packet": block.packet is not None,
                "semantic": block.attn.semantic_config(),
                "execution": block.attn.execution_config(),
            }
            for index, block in global_mixers
        ],
        "sources": source_manifest(),
    }


def _state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _unique(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def make_parameter_groups(
    model: DwarfForCausalLM,
    recipe: TrainRecipe,
) -> dict[str, list[dict[str, Any]]]:
    special: dict[str, list[nn.Parameter]] = {
        "scale": [],
        "null": [],
        "npci": [],
        "route": [],
        "ema": [],
        "positional": [],
    }
    for module in model.modules():
        if isinstance(module, DSQGAttentionV23):
            special["scale"].append(module.scale_embed)
            special["null"].extend((module.null_key, module.null_bias))
            special["positional"].extend(
                (module.pos_bias_log_slope, module.pos_bias_residual)
            )
        elif isinstance(module, HierarchicalSparseAttentionV19HISACausal):
            special["route"].extend(
                (
                    module.route_prior_raw,
                    module.representative_mix_raw,
                    module.global_lane_logit_bias,
                )
            )
            if module.binding_gain_raw is not None:
                special["route"].append(module.binding_gain_raw)
            special["npci"].extend((module.npci_theta_k, module.npci_theta_v))
        elif isinstance(module, InterferencePacket):
            special["ema"].extend((module.ema_raw, module.mix_logits))
    special = {name: _unique(values) for name, values in special.items()}
    special_ids = {id(value) for values in special.values() for value in values}

    muon_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            if (
                weight.requires_grad
                and weight.ndim == 2
                and min(weight.shape) >= 32
                and id(weight) != id(model.embedding.weight)
                and id(weight) not in special_ids
            ):
                muon_ids.add(id(weight))

    muon: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in special_ids:
            continue
        if id(parameter) in muon_ids:
            muon.append(parameter)
        elif parameter.ndim >= 2 and not name.endswith("bias"):
            adam_decay.append(parameter)
        else:
            adam_no_decay.append(parameter)

    def group(
        name: str,
        parameters: list[nn.Parameter],
        lr: float,
        weight_decay: float,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "params": _unique(parameters),
            "lr": lr,
            "base_lr": lr,
            "weight_decay": weight_decay,
        }

    muon_groups = [
        group("muon_linear_weights", muon, recipe.learning_rate, recipe.weight_decay)
    ]
    adam_groups = [
        group("adam_decay", adam_decay, recipe.learning_rate, recipe.weight_decay),
        group("adam_no_decay", adam_no_decay, recipe.learning_rate, 0.0),
        group(
            "adam_scale_embed",
            special["scale"],
            recipe.learning_rate * 0.5,
            0.0,
        ),
        group("adam_null", special["null"], recipe.learning_rate, 0.0),
        group("adam_npci", special["npci"], recipe.learning_rate, 0.0),
        group("adam_route", special["route"], recipe.learning_rate * 2.0, 0.0),
        group("adam_ema", special["ema"], recipe.learning_rate, 0.0),
        group(
            "adam_positional",
            special["positional"],
            recipe.learning_rate * 0.5,
            0.0,
        ),
    ]
    muon_groups = [item for item in muon_groups if item["params"]]
    adam_groups = [item for item in adam_groups if item["params"]]
    assigned = [
        parameter
        for item in (*muon_groups, *adam_groups)
        for parameter in item["params"]
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(assigned) != len(expected) or {id(p) for p in assigned} != {
        id(p) for p in expected
    }:
        raise RuntimeError("optimizer partition is incomplete or overlapping")
    return {"muon": muon_groups, "adamw": adam_groups}


def optimizer_manifest(
    model: nn.Module,
    optimizers: Iterable[tuple[str, torch.optim.Optimizer]],
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    entries: list[dict[str, Any]] = []
    for optimizer_name, optimizer in optimizers:
        groups: list[dict[str, Any]] = []
        for group in optimizer.param_groups:
            parameters = []
            for parameter in group["params"]:
                name = names.get(id(parameter))
                if name is None:
                    raise RuntimeError("optimizer parameter is absent from model")
                parameters.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                    }
                )
            groups.append({"name": str(group.get("name", "")), "parameters": parameters})
        entries.append({"name": optimizer_name, "groups": groups})
    return {"format": "dwarf-optimizer-manifest-v1", "optimizers": entries}


def optimizer_manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class MultiOptimizer:
    def __init__(
        self,
        optimizers: Iterable[tuple[str, torch.optim.Optimizer]],
        clip_parameters: dict[str, list[nn.Parameter]],
        model: nn.Module | None = None,
    ) -> None:
        self.optimizers = list(optimizers)
        self.clip_parameters = clip_parameters
        self.manifest = (
            optimizer_manifest(model, self.optimizers) if model is not None else None
        )
        self.clip_telemetry = {
            "optimizer_updates": 0,
            "muon_clip_updates": 0,
            "adamw_clip_updates": 0,
        }

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            group
            for _, optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def zero_grad(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.step()

    @staticmethod
    def _partition_norm(
        parameters: list[nn.Parameter],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            device = parameters[0].device if parameters else torch.device("cpu")
            return torch.zeros((), device=device), []
        if len({gradient.device for gradient in gradients}) != 1:
            raise ValueError("gradient clipping partition spans multiple devices")
        component_norms = torch._foreach_norm(gradients, 2.0)
        total = torch.stack(
            [component.float().square() for component in component_norms]
        ).sum().sqrt()
        return total, gradients

    def clip(self, recipe: TrainRecipe) -> dict[str, Any]:
        names = ("muon", "adamw")
        limits = {
            "muon": float(recipe.grad_clip_muon),
            "adamw": float(recipe.grad_clip_adamw),
        }
        norms: dict[str, torch.Tensor] = {}
        gradients: dict[str, list[torch.Tensor]] = {}
        coefficients: dict[str, torch.Tensor] = {}
        for name in names:
            norms[name], gradients[name] = self._partition_norm(
                self.clip_parameters[name]
            )
            coefficients[name] = torch.clamp(
                limits[name] / (norms[name] + 1e-6), max=1.0
            )
        finite_muon, finite_adamw, clipped_muon, clipped_adamw = torch.stack(
            (
                torch.isfinite(norms["muon"]),
                torch.isfinite(norms["adamw"]),
                coefficients["muon"] < 1.0,
                coefficients["adamw"] < 1.0,
            )
        ).to(device="cpu", dtype=torch.bool).tolist()
        if not finite_muon:
            raise FloatingPointError("muon gradient norm is non-finite")
        if not finite_adamw:
            raise FloatingPointError("adamw gradient norm is non-finite")
        clipped = {"muon": clipped_muon, "adamw": clipped_adamw}
        for name in names:
            if gradients[name]:
                torch._foreach_mul_(
                    gradients[name], [coefficients[name]] * len(gradients[name])
                )
        self.clip_telemetry["optimizer_updates"] += 1
        for name in names:
            self.clip_telemetry[f"{name}_clip_updates"] += int(clipped[name])
        return {
            name: {
                "norm": norms[name],
                "coefficient": coefficients[name],
                "clipped": clipped[name],
            }
            for name in names
        } | {"host_sync_count": 1}

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "dwarf-muon-adamw-v2",
            "manifest": copy.deepcopy(self.manifest),
            "clip_telemetry": copy.deepcopy(self.clip_telemetry),
            "optimizers": [
                {"name": name, "state": optimizer.state_dict()}
                for name, optimizer in self.optimizers
            ],
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        expected_lr_factor: float | None = None,
        require_complete_state: bool = False,
    ) -> None:
        if not isinstance(state, dict) or state.get("kind") != "dwarf-muon-adamw-v2":
            raise ValueError("checkpoint optimizer kind does not match")
        if state.get("manifest") != self.manifest:
            raise ValueError("checkpoint optimizer manifest does not match")
        telemetry = state.get("clip_telemetry")
        if telemetry is None:
            telemetry = {
                "optimizer_updates": 0,
                "muon_clip_updates": 0,
                "adamw_clip_updates": 0,
            }
        keys = {"optimizer_updates", "muon_clip_updates", "adamw_clip_updates"}
        if (
            not isinstance(telemetry, dict)
            or not keys <= set(telemetry)
            or any(
                isinstance(telemetry[key], bool)
                or not isinstance(telemetry[key], int)
                or telemetry[key] < 0
                for key in keys
            )
            or any(
                telemetry[f"{name}_clip_updates"] > telemetry["optimizer_updates"]
                for name in ("muon", "adamw")
            )
        ):
            raise ValueError("checkpoint clip telemetry is invalid")
        saved = state.get("optimizers")
        if not isinstance(saved, list) or len(saved) != len(self.optimizers):
            raise ValueError("checkpoint optimizer count does not match")
        if expected_lr_factor is not None and (
            not math.isfinite(expected_lr_factor) or expected_lr_factor <= 0.0
        ):
            raise ValueError("checkpoint optimizer LR factor is invalid")
        for (name, optimizer), item in zip(self.optimizers, saved, strict=True):
            if not isinstance(item, dict) or item.get("name") != name:
                raise ValueError("checkpoint optimizer order does not match")
            if set(item) != {"name", "state"} or not isinstance(item["state"], dict):
                raise ValueError("checkpoint optimizer payload is invalid")
            saved_state = item["state"]
            current_state = optimizer.state_dict()
            if set(saved_state) != {"state", "param_groups"} or not isinstance(
                saved_state["state"], dict
            ) or not isinstance(saved_state["param_groups"], list):
                raise ValueError("checkpoint optimizer state is invalid")
            current_groups = current_state["param_groups"]
            saved_groups = saved_state["param_groups"]
            live_groups = optimizer.param_groups
            if len(saved_groups) != len(current_groups):
                raise ValueError("checkpoint optimizer parameter-group count does not match")
            parameter_ids: list[int] = []
            slot_parameters: dict[int, nn.Parameter] = {}
            for saved_group, current_group, live_group in zip(
                saved_groups, current_groups, live_groups, strict=True
            ):
                if not isinstance(saved_group, dict) or set(saved_group) != set(
                    current_group
                ):
                    raise ValueError("checkpoint optimizer group metadata does not match")
                saved_parameters = saved_group.get("params")
                current_parameters = current_group["params"]
                if (
                    not isinstance(saved_parameters, list)
                    or not all(isinstance(value, int) for value in saved_parameters)
                    or saved_parameters != current_parameters
                ):
                    raise ValueError(
                        "checkpoint optimizer serialized parameter sequence does not match"
                    )
                live_parameters = live_group.get("params")
                if not isinstance(live_parameters, list) or len(live_parameters) != len(
                    current_parameters
                ):
                    raise RuntimeError("live optimizer parameter sequence is invalid")
                for parameter_id, parameter in zip(
                    current_parameters, live_parameters, strict=True
                ):
                    if not isinstance(parameter, nn.Parameter):
                        raise RuntimeError("live optimizer slot is not a parameter")
                    slot_parameters[parameter_id] = parameter
                parameter_ids.extend(saved_parameters)
                saved_immutable = {
                    key: value
                    for key, value in saved_group.items()
                    if key not in {"params", "lr"}
                }
                current_immutable = {
                    key: value
                    for key, value in current_group.items()
                    if key not in {"params", "lr"}
                }
                if saved_immutable != current_immutable:
                    raise ValueError(
                        "checkpoint optimizer immutable metadata does not match"
                    )
                saved_lr = saved_group.get("lr")
                if (
                    isinstance(saved_lr, bool)
                    or not isinstance(saved_lr, (int, float))
                    or not math.isfinite(float(saved_lr))
                ):
                    raise ValueError("checkpoint optimizer learning rate is invalid")
                if expected_lr_factor is not None:
                    expected_lr = float(current_group["base_lr"]) * expected_lr_factor
                    if float(saved_lr) != expected_lr:
                        raise ValueError(
                            "checkpoint optimizer scheduled learning rate does not match"
                        )
            if len(parameter_ids) != len(set(parameter_ids)):
                raise ValueError("checkpoint optimizer parameter membership overlaps")
            parameter_id_set = set(parameter_ids)
            saved_state_ids = set(saved_state["state"])
            if saved_state_ids - parameter_id_set:
                raise ValueError("checkpoint optimizer state has unexpected parameter entries")
            if (saved_state_ids or require_complete_state) and (
                parameter_id_set - saved_state_ids
            ):
                raise ValueError("checkpoint optimizer state has missing parameter entries")
            for parameter_id, parameter_state in saved_state["state"].items():
                if not isinstance(parameter_state, dict) or not parameter_state:
                    raise ValueError("checkpoint optimizer parameter state is invalid")
                parameter = slot_parameters[parameter_id]
                for state_name, state_value in parameter_state.items():
                    if torch.is_tensor(state_value):
                        if state_value.ndim == 0:
                            continue
                        if state_value.shape != parameter.shape:
                            raise ValueError(
                                "checkpoint optimizer state tensor shape does not match "
                                f"parameter slot {parameter_id}: {state_name}"
                            )
                        if state_value.dtype != parameter.dtype:
                            raise ValueError(
                                "checkpoint optimizer state tensor dtype does not match "
                                f"parameter slot {parameter_id}: {state_name}"
                            )
                    elif isinstance(state_value, bool) or not isinstance(
                        state_value, (int, float)
                    ) or not math.isfinite(float(state_value)):
                        raise ValueError(
                            "checkpoint optimizer scalar state is invalid for "
                            f"parameter slot {parameter_id}: {state_name}"
                        )
        for (_, optimizer), item in zip(self.optimizers, saved, strict=True):
            optimizer.load_state_dict(item["state"])
        self.clip_telemetry = {key: telemetry[key] for key in keys}


def build_optimizer(
    model: DwarfForCausalLM,
    recipe: TrainRecipe = RECIPE,
) -> MultiOptimizer:
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("DWARF requires torch.optim.Muon (PyTorch >= 2.9)")
    groups = make_parameter_groups(model, recipe)
    muon = torch.optim.Muon(
        groups["muon"],
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        groups["adamw"],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=next(model.parameters()).is_cuda,
    )
    return MultiOptimizer(
        (("muon", muon), ("adamw", adamw)),
        {
            "muon": [p for item in groups["muon"] for p in item["params"]],
            "adamw": [p for item in groups["adamw"] for p in item["params"]],
        },
        model,
    )


def set_learning_rates(optimizer: MultiOptimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def wsd_multiplier(step_index: int, recipe: TrainRecipe = RECIPE) -> float:
    step_index = min(max(int(step_index), 0), recipe.steps - 1)
    if step_index < recipe.warmup_steps:
        return (step_index + 1) / recipe.warmup_steps
    decay_start = recipe.warmup_steps + recipe.stable_steps
    if step_index < decay_start:
        return 1.0
    progress = (step_index - decay_start + 1) / recipe.decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return recipe.min_lr_ratio + (1.0 - recipe.min_lr_ratio) * cosine


class ValidatedDatasetSource:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None,
        trust_expected_sha256: bool,
    ) -> None:
        self.path = Path(path).resolve()
        expected = None if expected_sha256 is None else expected_sha256.strip().lower()
        if expected is not None and (
            len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("dataset SHA-256 must be 64 lowercase hexadecimal characters")
        if trust_expected_sha256 and expected is None:
            raise ValueError("--trust-dataset-sha256 requires --dataset-sha256")
        self.descriptor = os.open(
            self.path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(self.descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("dataset must be a regular file")
            self.signature = self._signature(before)
            actual = (
                expected
                if trust_expected_sha256
                else _sha256(Path(f"/proc/self/fd/{self.descriptor}"))
            )
            if expected is not None and actual != expected:
                raise ValueError("dataset SHA-256 does not match")
            if self._signature(os.fstat(self.descriptor)) != self.signature:
                raise RuntimeError("dataset changed while it was validated")
            self.sha256 = actual
            self.hash_verification = (
                "trusted_expected_sha256"
                if trust_expected_sha256
                else "full_file_sha256"
            )
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = -1
            raise

    @staticmethod
    def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def assert_unchanged(self) -> None:
        if self.descriptor < 0:
            raise RuntimeError("dataset source is closed")
        if self._signature(os.fstat(self.descriptor)) != self.signature:
            raise RuntimeError("dataset changed during training")

    def load(self, *, seq_len: int) -> torch.Tensor:
        self.assert_unchanged()
        payload = torch.load(
            Path(f"/proc/self/fd/{self.descriptor}"),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.assert_unchanged()
        if isinstance(payload, dict):
            for key in ("train", "input_ids", "tokens", "data"):
                if key in payload:
                    payload = payload[key]
                    break
        if not torch.is_tensor(payload):
            raise TypeError("dataset must be a tensor or a dict containing packed tokens")
        if payload.ndim != 2 or payload.shape[1] != seq_len:
            raise ValueError(
                f"dataset must have shape [rows,{seq_len}], got {tuple(payload.shape)}"
            )
        if payload.dtype not in (torch.int32, torch.int64):
            raise TypeError("packed token IDs must be int32 or int64")
        return payload

    def identity(
        self,
        rows: int,
        tokenizer: dict[str, Any],
        stable_id: str | None,
    ) -> dict[str, Any]:
        self.assert_unchanged()
        return {
            "path": str(self.path),
            "stable_id": stable_id,
            "sha256": self.sha256,
            "hash_verification": self.hash_verification,
            "size": self.signature[2],
            "rows": rows,
            "tokenizer": copy.deepcopy(tokenizer),
        }

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def tokenizer_identity(path: str | Path, vocab_size: int) -> dict[str, Any]:
    from tokenizers import Tokenizer

    resolved = Path(path).resolve()
    tokenizer = Tokenizer.from_file(str(resolved))
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model")
    expected = {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2, "<|unk|>": 3, "<|eod|>": 4}
    observed = {token: tokenizer.token_to_id(token) for token in expected}
    if observed != expected:
        raise ValueError(f"tokenizer structural IDs do not match: {observed}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "vocab_size": vocab_size,
        "structural_token_ids": expected,
    }


def build_training_row_order(
    *,
    dataset_rows: int,
    stop_step: int,
    recipe: TrainRecipe = RECIPE,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    if isinstance(dataset_rows, bool) or not isinstance(dataset_rows, int):
        raise TypeError("dataset row count must be an integer")
    if isinstance(stop_step, bool) or not isinstance(stop_step, int):
        raise TypeError("stop step must be an integer")
    minimum_rows = stop_step * recipe.effective_batch
    if dataset_rows < minimum_rows:
        raise ValueError(
            f"dataset has {dataset_rows} rows; this run requires {minimum_rows}"
        )
    selected_steps = min(recipe.steps, dataset_rows // recipe.effective_batch)
    selected_rows = selected_steps * recipe.effective_batch
    return torch.arange(selected_rows, dtype=torch.int64), {
        "mode": "sequential_prefix",
        "rows": selected_rows,
        "steps": selected_steps,
    }


def _dataset_content_identity(identity: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("checkpoint dataset identity is invalid")
    result = copy.deepcopy(identity)
    result.pop("path", None)
    tokenizer = result.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("checkpoint tokenizer identity is invalid")
    tokenizer.pop("path", None)
    result.pop("selection", None)
    return result


def validate_dataset_identity(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    consumed_rows: int,
) -> None:
    if _dataset_content_identity(saved) != _dataset_content_identity(current):
        raise ValueError("checkpoint dataset content identity does not match")
    if isinstance(consumed_rows, bool) or not isinstance(consumed_rows, int):
        raise TypeError("consumed rows must be an integer")
    saved_selection = saved.get("selection")
    current_selection = current.get("selection")
    if not isinstance(saved_selection, dict) or not isinstance(current_selection, dict):
        raise ValueError("checkpoint dataset selection is invalid")
    if saved_selection.get("mode") != "sequential_prefix" or (
        current_selection.get("mode") != "sequential_prefix"
    ):
        raise ValueError("checkpoint dataset selection mode does not match")
    saved_rows = saved_selection.get("rows")
    current_rows = current_selection.get("rows")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < consumed_rows
        for value in (saved_rows, current_rows)
    ):
        raise ValueError("checkpoint dataset selection does not cover consumed rows")
    if current_rows < saved_rows:
        raise ValueError("checkpoint dataset selection cannot shrink on resume")


def preflight_training_rows(
    dataset: torch.Tensor,
    *,
    selected_rows: int,
    vocab_size: int,
    chunk_rows: int = 1024,
) -> dict[str, int]:
    if dataset.device.type != "cpu" or dataset.ndim != 2:
        raise ValueError("dataset preflight requires a two-dimensional CPU tensor")
    if not 0 < selected_rows <= len(dataset) or vocab_size < 1 or chunk_rows < 1:
        raise ValueError("dataset preflight bounds are invalid")
    minimum = vocab_size
    maximum = -1
    for start in range(0, selected_rows, chunk_rows):
        rows = dataset[start : min(start + chunk_rows, selected_rows)]
        local_min = int(rows.min())
        local_max = int(rows.max())
        if local_min < 0 or local_max >= vocab_size:
            raise ValueError(
                f"selected row contains a token outside vocabulary [0,{vocab_size})"
            )
        minimum = min(minimum, local_min)
        maximum = max(maximum, local_max)
    return {
        "rows": selected_rows,
        "tokens": selected_rows * dataset.shape[1],
        "minimum_token_id": minimum,
        "maximum_token_id": maximum,
    }


class _TrainingForwardCallable(nn.Module):
    def __init__(self, model: DwarfForCausalLM, *, diagnostics: bool) -> None:
        super().__init__()
        self.model = model
        self.diagnostics = diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        route_aux_tile_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            ema_reset_mask=None,
            valid_lengths=None,
            route_aux_tile_ids=route_aux_tile_ids,
            collect_diagnostics=self.diagnostics,
            return_hidden=True,
            return_auxiliary=True,
        )


def compiled_training_callables(
    model: DwarfForCausalLM,
) -> tuple[nn.Module, nn.Module]:
    normal = torch.compile(
        _TrainingForwardCallable(model, diagnostics=False),
        mode="default",
        dynamic=False,
    )
    diagnostic = torch.compile(
        _TrainingForwardCallable(model, diagnostics=True),
        mode="default",
        dynamic=False,
    )
    return normal, diagnostic


class BatchStager:
    def __init__(
        self,
        dataset: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.buffers = [
            torch.empty(
                (batch_size, dataset.shape[1]),
                dtype=dataset.dtype,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.stream = torch.cuda.Stream(device=device)
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.recorded = [False, False]
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dwarf-data")

    def _fill(self, indices: torch.Tensor, slot: int) -> int:
        if self.recorded[slot]:
            self.events[slot].synchronize()
        self.buffers[slot].copy_(self.dataset[indices])
        return slot

    def _to_device(self, slot: int) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            result = self.buffers[slot].to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
            self.events[slot].record(self.stream)
            self.recorded[slot] = True
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        result.record_stream(current)
        return result

    def batches(self, update_indices: torch.Tensor) -> Iterator[torch.Tensor]:
        microbatches = [
            update_indices[start : start + self.batch_size]
            for start in range(0, len(update_indices), self.batch_size)
        ]
        future: Future[int] = self.executor.submit(self._fill, microbatches[0], 0)
        for index in range(len(microbatches)):
            slot = future.result()
            if index + 1 < len(microbatches):
                next_slot = (index + 1) % 2
                future = self.executor.submit(
                    self._fill,
                    microbatches[index + 1],
                    next_slot,
                )
            yield self._to_device(slot)

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def _tensor_storage_key(value: torch.Tensor) -> tuple[Any, ...] | None:
    if value.layout != torch.strided:
        return None
    storage = value.untyped_storage()
    return (value.device.type, value.device.index, storage._cdata, storage.nbytes())


@dataclass
class PendingSnapshot:
    snapshot: dict[str, Any]
    copy_events: tuple[torch.cuda.Event, ...]
    source_keepalive: list[torch.Tensor]


def _walk_tensors(value: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def _cuda_device(value: torch.device | str | int) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("checkpoint stream keys must be CUDA devices")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def _async_stage_payload(
    payload: dict[str, Any],
    *,
    producer_streams: dict[torch.device | str | int, torch.cuda.Stream] | None,
) -> PendingSnapshot:
    tensors = list(_walk_tensors(payload))
    cuda_devices = {
        _cuda_device(tensor.device) for tensor in tensors if tensor.device.type == "cuda"
    }
    streams = {
        _cuda_device(device): stream for device, stream in (producer_streams or {}).items()
    }
    if cuda_devices != set(streams):
        raise ValueError("an explicit producer stream is required for each CUDA device")
    staging_streams: dict[torch.device, torch.cuda.Stream] = {}
    copy_events: dict[torch.device, torch.cuda.Event] = {}
    for device in cuda_devices:
        producer = streams[device]
        if _cuda_device(producer.device) != device:
            raise ValueError("checkpoint producer stream device does not match")
        ready = torch.cuda.Event()
        ready.record(producer)
        staging = torch.cuda.Stream(device=device)
        staging.wait_event(ready)
        staging_streams[device] = staging
        copy_events[device] = torch.cuda.Event()

    storage_memo: dict[tuple[Any, ...], torch.Tensor] = {}
    keepalive: list[torch.Tensor] = []

    def stage(value: Any) -> Any:
        if torch.is_tensor(value):
            detached = value.detach()
            storage_key = _tensor_storage_key(detached)
            if storage_key is None:
                if detached.device.type == "cuda":
                    raise ValueError("CUDA checkpoint tensors must use strided layout")
                return detached.clone()
            staged_storage = storage_memo.get(storage_key)
            if staged_storage is None:
                storage = detached.untyped_storage()
                source = torch.empty(0, device=detached.device, dtype=torch.uint8).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                if detached.device.type == "cpu":
                    staged_storage = source.clone()
                else:
                    device = _cuda_device(detached.device)
                    staged_storage = torch.empty(
                        storage.nbytes(), dtype=torch.uint8, device="cpu", pin_memory=True
                    )
                    with torch.cuda.stream(staging_streams[device]):
                        staged_storage.copy_(source, non_blocking=True)
                    source.record_stream(staging_streams[device])
                    keepalive.append(source)
                storage_memo[storage_key] = staged_storage
            flat = torch.empty(0, dtype=detached.dtype).set_(
                staged_storage.untyped_storage(),
                0,
                (staged_storage.numel() // detached.element_size(),),
                (1,),
            )
            return flat.as_strided(
                detached.size(), detached.stride(), detached.storage_offset()
            )
        if isinstance(value, dict):
            return {key: stage(item) for key, item in value.items()}
        if isinstance(value, list):
            return [stage(item) for item in value]
        if isinstance(value, tuple):
            return tuple(stage(item) for item in value)
        return copy.deepcopy(value)

    snapshot = stage(payload)
    for device, event in copy_events.items():
        event.record(staging_streams[device])
        streams[device].wait_event(event)
    return PendingSnapshot(snapshot, tuple(copy_events.values()), keepalive)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class AsyncCheckpointWriter:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dwarf-save")
        self.future: Future[None] | None = None
        self.lock = threading.Lock()

    @staticmethod
    def _save(pending: PendingSnapshot, path: Path) -> None:
        for event in pending.copy_events:
            event.synchronize()
        pending.source_keepalive.clear()
        atomic_torch_save(pending.snapshot, path)

    def submit(
        self,
        payload: dict[str, Any],
        path: Path,
        *,
        producer_streams: dict[
            torch.device | str | int, torch.cuda.Stream
        ] | None = None,
    ) -> None:
        with self.lock:
            if self.future is not None:
                self.future.result()
            pending = _async_stage_payload(payload, producer_streams=producer_streams)
            self.future = self.executor.submit(self._save, pending, path)

    def close(self) -> None:
        try:
            with self.lock:
                if self.future is not None:
                    self.future.result()
        finally:
            self.executor.shutdown(wait=True)


def checkpoint_payload(
    *,
    step: int,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "kind": CHECKPOINT_KIND,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture": architecture,
        "recipe": asdict(RECIPE),
        "dataset": dataset,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    return payload


def restore_checkpoint(
    path: str | Path,
    *,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("not a canonical DWARF resumable checkpoint")
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise ValueError("not a canonical DWARF resumable checkpoint")
    validate_checkpoint_architecture(checkpoint.get("architecture"), architecture)
    if checkpoint.get("recipe") != asdict(RECIPE):
        raise ValueError("checkpoint recipe does not match")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= RECIPE.steps:
        raise ValueError("checkpoint step is invalid")
    validate_dataset_identity(
        checkpoint.get("dataset"),
        dataset,
        consumed_rows=step * RECIPE.effective_batch,
    )
    saved_model = checkpoint.get("model")
    expected_model = model.state_dict()
    if not isinstance(saved_model, dict) or saved_model.keys() != expected_model.keys():
        raise ValueError("checkpoint model keys do not match")
    for name, expected in expected_model.items():
        saved = saved_model[name]
        if not torch.is_tensor(saved) or saved.shape != expected.shape or saved.dtype != expected.dtype:
            raise ValueError(f"checkpoint model tensor does not match: {name}")
    cuda_rng = checkpoint.get("cuda_rng")
    if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA RNG state does not match visible devices")
    if not all(torch.is_tensor(state) and state.dtype == torch.uint8 for state in cuda_rng):
        raise ValueError("checkpoint CUDA RNG state is invalid")
    if not torch.is_tensor(checkpoint.get("torch_rng")):
        raise ValueError("checkpoint CPU RNG state is invalid")
    if not isinstance(checkpoint.get("optimizer"), dict):
        raise ValueError("checkpoint optimizer state is invalid")
    model.load_state_dict(saved_model, strict=True)
    optimizer.load_state_dict(
        checkpoint["optimizer"],
        expected_lr_factor=wsd_multiplier(step - 1),
        require_complete_state=True,
    )
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(cuda_rng)
    return step


def amp_context():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def configure_compiled_backward_autocast() -> dict[str, str | bool]:
    try:
        from torch._functorch import config as functorch_config
    except ImportError as error:
        raise RuntimeError("compiled backward autocast policy is unavailable") from error
    previous = getattr(functorch_config, "backward_pass_autocast", None)
    if not isinstance(previous, str):
        raise RuntimeError("compiled backward autocast policy is unavailable")
    functorch_config.backward_pass_autocast = "off"
    observed = getattr(functorch_config, "backward_pass_autocast", None)
    if observed != "off":
        raise RuntimeError("compiled backward autocast policy could not be set")
    return {"previous": previous, "observed": observed, "required": True}


def _assert_finite(value: torch.Tensor, message: str) -> None:
    torch._assert_async(torch.isfinite(value).all(), message)


def nonfinite_gradient_report(model: nn.Module) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all()):
            continue
        report.append(
            {
                "parameter": name,
                "shape": list(gradient.shape),
                "nonfinite_values": int((~finite).sum()),
            }
        )
    return report


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DWARF training requires CUDA")
    stop_step = RECIPE.steps if args.stop_after is None else int(args.stop_after)
    if not 0 < stop_step <= RECIPE.steps:
        raise ValueError(f"--stop-after must be between 1 and {RECIPE.steps}")
    if args.save_every < 0 or args.log_every < 1:
        raise ValueError("invalid save/log interval")
    compile_policy = configure_compiled_backward_autocast()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = DwarfConfig()
    tokenizer = tokenizer_identity(args.tokenizer, config.vocab_size)
    source = ValidatedDatasetSource(
        args.dataset,
        expected_sha256=args.dataset_sha256,
        trust_expected_sha256=args.trust_dataset_sha256,
    )
    try:
        dataset = source.load(seq_len=config.seq_len)
        order, selection = build_training_row_order(
            dataset_rows=len(dataset), stop_step=stop_step
        )
        preflight = preflight_training_rows(
            dataset,
            selected_rows=len(order),
            vocab_size=config.vocab_size,
        )
        identity = source.identity(len(dataset), tokenizer, args.dataset_id)
        identity["selection"] = selection

        model = DwarfForCausalLM(config).to(device)
        model.prepare_runtime(device)
        architecture = model_metadata(model)
        if architecture["parameters"] != EXPECTED_PARAMETERS:
            raise RuntimeError("canonical DWARF parameter count changed")
        if architecture["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError("canonical DWARF trainable parameter count changed")
        optimizer = build_optimizer(model)
        start_step = 0
        if args.resume:
            start_step = restore_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                architecture=architecture,
                dataset=identity,
            )
        if start_step >= stop_step:
            raise ValueError("checkpoint is already at or beyond --stop-after")

        compiled, _ = compiled_training_callables(model)
        loss_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)
        stager = BatchStager(dataset, batch_size=RECIPE.batch_size, device=device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = AsyncCheckpointWriter()
    except BaseException:
        source.close()
        raise
    print(
        json.dumps(
            {
                "architecture": architecture,
                "recipe": asdict(RECIPE),
                "dataset": identity,
                "dataset_preflight": preflight,
                "start_step": start_step,
                "stop_step": stop_step,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "compiled": True,
                "compile_policy": compile_policy,
                "liger_fused_cross_entropy": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    compiled.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    interval_started = time.perf_counter()
    interval_start_step = start_step
    training_started = interval_started
    try:
        for step_index in range(start_step, stop_step):
            step = step_index + 1
            factor = wsd_multiplier(step_index)
            set_learning_rates(optimizer, factor)
            optimizer.zero_grad()
            begin = step_index * RECIPE.effective_batch
            update = order[begin : begin + RECIPE.effective_batch]
            loss_accumulator = torch.zeros((), device=device, dtype=torch.float32)
            language_accumulator = torch.zeros((), device=device, dtype=torch.float32)
            auxiliary_accumulator = torch.zeros((), device=device, dtype=torch.float32)
            route_aux_tile_ids = route_aux_tile_ids_for_update(step, device, config)
            for batch in stager.batches(update):
                input_ids, labels = batch[:, :-1], batch[:, 1:]
                with amp_context():
                    hidden, auxiliary = compiled(input_ids, route_aux_tile_ids)
                    language_loss = loss_fn(
                        model.lm_head.weight,
                        hidden.flatten(0, 1),
                        labels.flatten(),
                    )
                    loss = language_loss + auxiliary
                (loss / RECIPE.grad_accum_steps).backward()
                loss_accumulator += loss.detach().float() / RECIPE.grad_accum_steps
                language_accumulator += (
                    language_loss.detach().float() / RECIPE.grad_accum_steps
                )
                auxiliary_accumulator += (
                    auxiliary.detach().float() / RECIPE.grad_accum_steps
                )

            try:
                norms = optimizer.clip(RECIPE)
            except FloatingPointError as error:
                print(
                    json.dumps(
                        {
                            "status": "fail",
                            "reason": "nonfinite_gradients_before_clipping",
                            "step": step,
                            "parameters": nonfinite_gradient_report(model),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise error
            _assert_finite(loss_accumulator, "non-finite training loss")
            for name in ("muon", "adamw"):
                _assert_finite(
                    norms[name]["norm"], f"non-finite {name} gradient norm"
                )
            optimizer.step()

            if step % args.log_every == 0 or step in {1, stop_step}:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_seconds = now - interval_started
                interval_steps = step - interval_start_step
                targets = interval_steps * RECIPE.effective_batch * config.model_length
                event: dict[str, Any] = {
                    "step": step,
                    "loss": float(loss_accumulator),
                    "language_loss": float(language_accumulator),
                    "routing_auxiliary_loss": float(auxiliary_accumulator),
                    "lr_factor": factor,
                    "learning_rates": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "grad_norm_muon": float(norms["muon"]["norm"]),
                    "grad_norm_adamw": float(norms["adamw"]["norm"]),
                    "grad_clip_muon_coefficient": float(
                        norms["muon"]["coefficient"]
                    ),
                    "grad_clip_adamw_coefficient": float(
                        norms["adamw"]["coefficient"]
                    ),
                    "grad_clip_muon_clipped": norms["muon"]["clipped"],
                    "grad_clip_adamw_clipped": norms["adamw"]["clipped"],
                    "shifted_targets": step * RECIPE.effective_batch * config.model_length,
                    "interval_seconds": interval_seconds,
                    "shifted_targets_per_second": targets / interval_seconds,
                    "training_elapsed_seconds": now - training_started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                for module in model.modules():
                    if isinstance(module, HierarchicalSparseAttentionV19HISACausal):
                        event.update(
                            {
                                f"hisa_{key}": float(value)
                                for key, value in module._routing_diagnostics.items()
                                if torch.is_tensor(value) and value.numel() == 1
                            }
                        )
                print(json.dumps(event, sort_keys=True), flush=True)
                interval_started = now
                interval_start_step = step

            should_save = step in RECIPE.checkpoint_steps or step == stop_step
            should_save |= args.save_every > 0 and step % args.save_every == 0
            if should_save:
                source.assert_unchanged()
                if source_manifest() != architecture["sources"]:
                    raise RuntimeError("trainer or kernel source changed during training")
                writer.submit(
                    checkpoint_payload(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        architecture=architecture,
                        dataset=identity,
                    ),
                    output_dir / f"dwarf_step_{step:07d}.pt",
                    producer_streams={device: torch.cuda.current_stream(device)},
                )
    finally:
        try:
            stager.close()
        finally:
            try:
                writer.close()
                source.assert_unchanged()
            finally:
                source.close()


def self_test() -> None:
    assert_public_kernel_contracts()
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1234)
    model = DwarfForCausalLM()
    seeded_rng_fingerprint = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()
    torch.random.set_rng_state(rng_state)
    metadata = model_metadata(model)
    assert metadata["parameters"] == EXPECTED_PARAMETERS
    assert metadata["trainable_parameters"] == EXPECTED_TRAINABLE_PARAMETERS
    assert _state_fingerprint(model) == EXPECTED_STATE_FINGERPRINT
    assert seeded_rng_fingerprint == EXPECTED_SEEDED_RNG_FINGERPRINT
    assert len(model.blocks) == 12
    global_mixers = [block for block in model.blocks if isinstance(block, GlobalMixerBlock)]
    assert [index for index, block in enumerate(model.blocks) if isinstance(block, GlobalMixerBlock)] == [3, 9]
    assert isinstance(global_mixers[0].packet, InterferencePacket)
    assert global_mixers[1].packet is None
    assert global_mixers[0].attn.npci_theta_k.requires_grad
    assert not global_mixers[1].attn.npci_theta_k.requires_grad
    for global_mixer in global_mixers:
        assert not global_mixer.attn.collect_routing_diagnostics
        assert global_mixer.attn.chunk_selection_scope == "token"
        assert global_mixer.attn.token_routing_pack_size == 4
        assert global_mixer.attn.route_aux_weight == model.config.hisa_route_aux_weight
        assert global_mixer.attn.exploration_probability == model.config.hisa_exploration_probability
        assert global_mixer.attn.global_adapter_rank == model.config.hisa_global_adapter_rank
        assert global_mixer.attn.binding_rank == model.config.hisa_binding_rank
        assert global_mixer.attn.backend == model.config.hisa_backend
        assert global_mixer.attn.token_selection_mode == model.config.hisa_token_selection_mode
        assert global_mixer.attn.local_backend == model.config.hisa_local_backend
        assert global_mixer.attn.triton_block_q == model.config.hisa_triton_block_q
        assert global_mixer.attn.backward_impl == model.config.hisa_backward_impl
        assert global_mixer.attn.global_k_down is not None
        assert hasattr(global_mixer.attn, "npci_theta_k")
        assert hasattr(global_mixer.attn, "npci_theta_v")
    assert sum(isinstance(block, DSQGBlock) for block in model.blocks) == 10
    dsqg_layers = [block.attn for block in model.blocks if isinstance(block, DSQGBlock)]
    assert all(isinstance(module, DSQGAttentionV23) for module in dsqg_layers)
    assert all(not hasattr(module, "j_large") for module in dsqg_layers)
    assert all(not hasattr(module, "movt_enabled") for module in dsqg_layers)
    assert all(not hasattr(module, "npci_theta_k") for module in dsqg_layers)
    assert all(not hasattr(module, "npci_theta_v") for module in dsqg_layers)
    assert not any(
        token in name
        for name, _ in model.named_parameters()
        for token in ("phase_base", "phase_gain", "phase_gate", "query_probes", "key_probes")
    )
    groups = make_parameter_groups(model, RECIPE)
    assert [item["name"] for item in groups["muon"]] == ["muon_linear_weights"]
    assert [item["name"] for item in groups["adamw"]] == [
        "adam_decay",
        "adam_no_decay",
        "adam_scale_embed",
        "adam_null",
        "adam_npci",
        "adam_route",
        "adam_ema",
        "adam_positional",
    ]
    print(json.dumps({"status": "PASS", **metadata}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--tokenizer",
        default=str(SCRIPT_DIR.parent / "tokenizers" / "dwarf_bpe_v32768_tokenizer.json"),
    )
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--trust-dataset-sha256", action="store_true")
    parser.add_argument("--resume")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if not args.self_test and (not args.dataset or not args.output_dir):
        parser.error("--dataset and --output-dir are required for training")
    if args.trust_dataset_sha256 and not args.dataset_sha256:
        parser.error("--trust-dataset-sha256 requires --dataset-sha256")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        train(arguments)
