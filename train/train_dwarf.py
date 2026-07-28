#!/usr/bin/env python3
"""Minimal public trainer for the active DWARF-v2 architecture.

This reference implementation contains the active DWARF topology only:
triadic DSQG sparse blocks, a causal EMA interference injection at L2, and one
L3 global mixer.  The global mixer can be strict-causal V16 HISA or full causal
SDPA (`--global-mixer fa`), which is the topology used by DWARF-55M-Base.

The trainer accepts packed token rows and includes the validated Muon+AdamW,
WSD, checkpoint, and resume path used by the public recipes.  Dataset building,
evaluation, and campaign infrastructure intentionally remain out of scope.
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
    "dsqg_attention_v22.py",
    "hierarchical_sparse_attn_v18_hisa.py",
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
    causal_ema_scan,
    inverse_bounded_ema_factor,
)
from dsqg_attention_v22 import (  # noqa: E402
    ALL_OFFSETS,
    DSQGAttentionV19,
    R_PLANES,
)
from hierarchical_sparse_attn_v18_hisa import (  # noqa: E402
    HierarchicalSparseAttentionV16HISACausal,
)

EXPECTED_PARAMETERS = 55_348_728
EXPECTED_TRAINABLE_PARAMETERS = 55_344_120


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
    ffn_dim: int = 2400
    seq_len: int = 2048
    num_layers: int = 10
    dropout: float = 0.05
    min_offset_support: int = 64
    hisa_chunk_size: int = 64
    top_k_chunks: int = 4
    hisa_top_m_tokens: int = 64
    hisa_local_window: int = 64
    hisa_selector_tile: int = 16
    hisa_global_adapter_rank: int = 16
    ema_timescales: tuple[float, ...] = (16.0, 64.0, 256.0)
    init_seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers != 10:
            raise ValueError("the 55M DWARF topology has exactly ten blocks")
        if self.seq_len < 65:
            raise ValueError("seq_len must be at least 65")
        head_dim = self.embedding_dim // self.num_heads
        if head_dim & (head_dim - 1):
            raise ValueError("HISA requires a power-of-two head dimension")

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
    groups = tuple(tuple(offsets[index::3]) for index in range(3))
    if any(not group for group in groups):
        raise ValueError("offset pruning left an empty DSQG group")
    return groups


def _small_large_counts(offsets: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in offsets)
    small = sum(value <= 28 for value in values)
    large = sum(value >= 48 for value in values)
    if small + large != len(values):
        raise ValueError("DSQG offsets must remain in the supported kernel bands")
    return small, large


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    locations = {
        Path(__file__).name: Path(__file__).resolve(),
        "causal_ema_scan.py": Path(sys.modules["causal_ema_scan"].__file__).resolve(),
        "dsqg_attention_v22.py": Path(sys.modules["dsqg_attention_v22"].__file__).resolve(),
        "hierarchical_sparse_attn_v18_hisa.py": Path(
            sys.modules["hierarchical_sparse_attn_v18_hisa"].__file__
        ).resolve(),
    }
    return {name: _sha256(path) for name, path in locations.items()}


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
        self.mix_logits = nn.Parameter(torch.zeros(len(raw)))
        self.gate_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.kv_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim, bias=False)
        with torch.no_grad():
            self.gate_proj.bias.fill_(-2.0)

    @property
    def ema_factors(self) -> torch.Tensor:
        return bounded_ema_factor(self.ema_raw)

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scan_input = normalized.to(torch.bfloat16) if normalized.is_cuda else normalized
        scans = [causal_ema_scan(scan_input, factor.reshape(1)) for factor in self.ema_factors]
        mixture = torch.softmax(self.mix_logits.float(), dim=0)
        pooled = sum(
            weight.to(scans[0].dtype) * scan
            for weight, scan in zip(mixture, scans, strict=True)
        )
        pooled_float = pooled.float()
        rms = pooled_float.square().mean(-1, keepdim=True).sqrt()
        direction = pooled_float * torch.rsqrt(
            pooled_float.square().mean(-1, keepdim=True) + 1e-6
        )
        confidence = torch.tanh(rms / 0.25)
        gate = torch.sigmoid(self.gate_proj(normalized).float())
        packet = (direction * confidence * gate).to(normalized.dtype)
        key_delta, value_delta = self.kv_proj(packet).chunk(2, dim=-1)
        batch, seq_len, _ = normalized.shape
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
        *,
        plane_shift: int,
    ) -> None:
        super().__init__()
        small, large = _small_large_counts(offsets)
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        self.attn = DSQGAttentionV19(
            config.embedding_dim,
            config.num_heads,
            offsets,
            small,
            large,
            seq_len=config.seq_len,
            dropout=config.dropout,
            grouped_mode="baseline",
            plane_shift=plane_shift,
            pos_bias_scale=1.0,
            movt_dynamic_rms_target=0.01,
            train_phase_probes=False,
            scale_embed_init_std=0.01,
            npci_strength_tau=0.25,
            online_softmax=True,
            support_crop_projections=True,
            support_crop_min_offset=64,
        )
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GlobalMixerBlock(nn.Module):
    def __init__(self, config: DwarfConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        self.attn = HierarchicalSparseAttentionV16HISACausal(
            D=config.embedding_dim,
            H=config.num_heads,
            hd=config.embedding_dim // config.num_heads,
            chunk_size=config.hisa_chunk_size,
            top_k_chunks=config.top_k_chunks,
            hisa_top_m_tokens=config.hisa_top_m_tokens,
            local_window=config.hisa_local_window,
            selector_tile_size=config.hisa_selector_tile,
            representative_mode="mean_max_blend",
            representative_blend_alpha=0.5,
            route_prior_scale=0.1,
            route_aux_weight=0.01,
            route_aux_samples=4,
            route_aux_temperature=1.0,
            exploration_probability=0.05,
            global_adapter_rank=config.hisa_global_adapter_rank,
            npci_theta_max=0.25,
            max_seq_len=config.model_length,
            local_backend="flex",
            collect_routing_diagnostics=False,
        )
        self.packet = InterferencePacket(config)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        attended, auxiliary = self.attn(
            normalized,
            kv_inject=self.packet(normalized),
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
        layout: tuple[int | None, ...] = (0, 1, 2, None, 0, 1, 2, 0, 1, 2)
        blocks: list[nn.Module] = []
        dsqg_index = 0
        plane_segment = max(2, (config.embedding_dim // config.num_heads) // R_PLANES)
        for group_index in layout:
            if group_index is None:
                blocks.append(GlobalMixerBlock(config))
                continue
            plane_shift = 2 * (dsqg_index % max(1, plane_segment // 2))
            blocks.append(
                DSQGBlock(
                    config,
                    self.offset_groups[group_index],
                    plane_shift=plane_shift,
                )
            )
            dsqg_index += 1
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
                if isinstance(module, DSQGAttentionV19):
                    nn.init.normal_(module.phase_base, mean=0.0, std=0.01)
                    nn.init.normal_(
                        module.phase_gain,
                        mean=0.0,
                        std=module.movt_phase_gain_init_std,
                    )
                    nn.init.zeros_(module.phase_gate)
                    nn.init.normal_(
                        module.scale_embed,
                        mean=0.0,
                        std=module.scale_embed_init_std,
                    )
                    with torch.no_grad():
                        module.scale_embed.sub_(module.scale_embed.mean(0, keepdim=True))
                    module.reset_phase_probes_()
                elif isinstance(module, HierarchicalSparseAttentionV16HISACausal):
                    module.reset_global_adapters_()
                elif isinstance(module, InterferencePacket):
                    with torch.no_grad():
                        module.gate_proj.bias.fill_(-2.0)
        finally:
            torch.random.set_rng_state(state)

    def prepare_runtime(self, device: torch.device | str) -> None:
        for module in self.modules():
            if isinstance(module, HierarchicalSparseAttentionV16HISACausal):
                module.prepare_runtime(device, self.config.model_length)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        return_auxiliary: bool = False,
    ):
        x = self.embedding(input_ids)
        if x.is_cuda:
            x = x.to(torch.bfloat16)
        x = self.dropout(x)
        auxiliary = x.sum() * 0.0
        for block in self.blocks:
            if isinstance(block, GlobalMixerBlock):
                x, block_auxiliary = block(x)
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
        return_hidden: bool = False,
        return_auxiliary: bool = False,
    ):
        hidden, auxiliary = self.forward_hidden(input_ids, return_auxiliary=True)
        output = hidden if return_hidden else self.lm_head(hidden)
        return (output, auxiliary) if return_auxiliary else output


def model_metadata(model: DwarfForCausalLM) -> dict[str, Any]:
    return {
        "format": "dwarf-55m-v1",
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "topology": {
            "layers": "DSQG,DSQG,DSQG,HISA,DSQG,DSQG,DSQG,DSQG,DSQG,DSQG",
            "dsqg": "v22-baseline-online-softmax",
            "hisa": "v18-strict-causal",
            "offset_groups": model.offset_groups,
        },
        "sources": source_manifest(),
    }


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
        "phase": [],
        "npci": [],
        "route": [],
        "ema": [],
        "positional": [],
    }
    for module in model.modules():
        if isinstance(module, DSQGAttentionV19):
            special["scale"].append(module.scale_embed)
            special["phase"].extend(
                (module.phase_base, module.phase_gain, module.phase_gate)
            )
            special["npci"].extend((module.npci_theta_k, module.npci_theta_v))
            special["positional"].extend(
                (module.pos_bias_log_slope, module.pos_bias_residual)
            )
        elif isinstance(module, HierarchicalSparseAttentionV16HISACausal):
            special["route"].append(module.route_prior_raw)
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
            recipe.learning_rate * 4.0,
            0.0,
        ),
        group("adam_phase", special["phase"], recipe.learning_rate * 2.0, 0.0),
        group("adam_npci", special["npci"], recipe.learning_rate * 4.0, 0.0),
        group("adam_route", special["route"], recipe.learning_rate * 2.0, 0.0),
        group("adam_ema", special["ema"], recipe.learning_rate, 0.0),
        group("adam_positional", special["positional"], recipe.learning_rate, 0.0),
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


class MultiOptimizer:
    def __init__(
        self,
        optimizers: Iterable[tuple[str, torch.optim.Optimizer]],
        clip_parameters: dict[str, list[nn.Parameter]],
    ) -> None:
        self.optimizers = list(optimizers)
        self.clip_parameters = clip_parameters

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

    def clip(self, recipe: TrainRecipe) -> dict[str, torch.Tensor]:
        return {
            "muon": torch.nn.utils.clip_grad_norm_(
                self.clip_parameters["muon"], recipe.grad_clip_muon
            ),
            "adamw": torch.nn.utils.clip_grad_norm_(
                self.clip_parameters["adamw"], recipe.grad_clip_adamw
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "dwarf-muon-adamw-v1",
            "optimizers": [
                {"name": name, "state": optimizer.state_dict()}
                for name, optimizer in self.optimizers
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("kind") != "dwarf-muon-adamw-v1":
            raise ValueError("checkpoint optimizer kind does not match")
        saved = state.get("optimizers")
        if not isinstance(saved, list) or len(saved) != len(self.optimizers):
            raise ValueError("checkpoint optimizer count does not match")
        for (name, _), item in zip(self.optimizers, saved, strict=True):
            if not isinstance(item, dict) or item.get("name") != name:
                raise ValueError("checkpoint optimizer order does not match")
        for (_, optimizer), item in zip(self.optimizers, saved, strict=True):
            optimizer.load_state_dict(item["state"])


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


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return copy.deepcopy(value)


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

    def submit(self, payload: dict[str, Any], path: Path) -> None:
        snapshot = _tree_to_cpu(payload)
        with self.lock:
            if self.future is not None:
                self.future.result()
            self.future = self.executor.submit(atomic_torch_save, snapshot, path)

    def close(self) -> None:
        with self.lock:
            if self.future is not None:
                self.future.result()
        self.executor.shutdown(wait=True)


def checkpoint_payload(
    *,
    step: int,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "dwarf-55m-resume-v1",
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


def restore_checkpoint(
    path: str | Path,
    *,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "dwarf-55m-resume-v1":
        raise ValueError("not a 55M DWARF resumable checkpoint")
    if checkpoint.get("architecture") != architecture:
        raise ValueError("checkpoint architecture or source manifest does not match")
    if checkpoint.get("recipe") != asdict(RECIPE):
        raise ValueError("checkpoint recipe does not match")
    if checkpoint.get("dataset") != dataset:
        raise ValueError("checkpoint dataset identity does not match")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= RECIPE.steps:
        raise ValueError("checkpoint step is invalid")
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
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(cuda_rng)
    return step


def amp_context():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _assert_finite(value: torch.Tensor, message: str) -> None:
    torch._assert_async(torch.isfinite(value).all(), message)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DWARF training requires CUDA")
    stop_step = RECIPE.steps if args.stop_after is None else int(args.stop_after)
    if not 0 < stop_step <= RECIPE.steps:
        raise ValueError(f"--stop-after must be between 1 and {RECIPE.steps}")
    if args.save_every < 0 or args.log_every < 1:
        raise ValueError("invalid save/log interval")

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
        required_rows = RECIPE.effective_batch * stop_step
        if len(dataset) < required_rows:
            raise ValueError(
                f"dataset has {len(dataset)} rows; this run requires {required_rows}"
            )
        identity = source.identity(len(dataset), tokenizer, args.dataset_id)
        order = torch.arange(required_rows, dtype=torch.int64)
        identity["selection"] = {
            "mode": "sequential_prefix",
            "rows": required_rows,
        }

        model = DwarfForCausalLM(config).to(device)
        model.prepare_runtime(device)
        architecture = model_metadata(model)
        if architecture["parameters"] != EXPECTED_PARAMETERS:
            raise RuntimeError("55M parameter count changed")
        if architecture["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError("55M trainable parameter count changed")
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

        compiled: nn.Module = torch.compile(model, mode="default", dynamic=False)
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
                "start_step": start_step,
                "stop_step": stop_step,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "compiled": True,
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
            for batch in stager.batches(update):
                input_ids, labels = batch[:, :-1], batch[:, 1:]
                with amp_context():
                    hidden, auxiliary = compiled(
                        input_ids,
                        return_hidden=True,
                        return_auxiliary=True,
                    )
                    language_loss = loss_fn(
                        model.lm_head.weight,
                        hidden.flatten(0, 1),
                        labels.flatten(),
                    )
                    loss = language_loss + auxiliary
                (loss / RECIPE.grad_accum_steps).backward()
                loss_accumulator += loss.detach().float() / RECIPE.grad_accum_steps

            norms = optimizer.clip(RECIPE)
            _assert_finite(loss_accumulator, "non-finite training loss")
            for name, norm in norms.items():
                _assert_finite(norm, f"non-finite {name} gradient norm")
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
                    "lr_factor": factor,
                    "learning_rates": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "grad_norm_muon": float(norms["muon"]),
                    "grad_norm_adamw": float(norms["adamw"]),
                    "shifted_targets": step * RECIPE.effective_batch * config.model_length,
                    "interval_seconds": interval_seconds,
                    "shifted_targets_per_second": targets / interval_seconds,
                    "training_elapsed_seconds": now - training_started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                for module in model.modules():
                    if isinstance(module, HierarchicalSparseAttentionV16HISACausal):
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
    model = DwarfForCausalLM()
    metadata = model_metadata(model)
    assert metadata["parameters"] == EXPECTED_PARAMETERS
    assert metadata["trainable_parameters"] == EXPECTED_TRAINABLE_PARAMETERS
    assert len(model.blocks) == 10
    assert isinstance(model.blocks[3], GlobalMixerBlock)
    assert sum(isinstance(block, DSQGBlock) for block in model.blocks) == 9
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
