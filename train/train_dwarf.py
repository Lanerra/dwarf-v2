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
import contextlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = PROJECT_ROOT / "kernels"
if str(KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_DIR))

from causal_ema_scan import causal_ema_scan
from dsqg_attention_v20_bf16_se import (
    ALL_OFFSETS,
    DSQGAttentionV19,
    R_PLANES,
    calibrated_movt_phase_gain_std,
)
from hierarchical_sparse_attn_v16_hisa_causal import HierarchicalSparseAttentionV16HISACausal


def _parse_movt_dynamic_rms_target(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"0", "legacy", "none", "off"}:
        return None
    try:
        target = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "MOVT dynamic RMS target must be a positive float or legacy/off/none/0"
        ) from exc
    if not math.isfinite(target) or target <= 0.0:
        raise argparse.ArgumentTypeError(
            "MOVT dynamic RMS target must be finite and positive or legacy/off/none/0"
        )
    return target


@dataclass(frozen=True)
class TrainRecipe:
    name: str
    learning_rate: float
    batch_size: int
    grad_accum_steps: int
    steps: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    checkpoint_milestones: bool
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    phase_lr_mult: float = 10.0
    scale_embed_lr_mult: float = 8.0
    npci_theta_lr_mult: float = 8.0

    def __post_init__(self) -> None:
        if self.warmup_steps + self.stable_steps + self.decay_steps != self.steps:
            raise ValueError(f"{self.name}: WSD phases must sum to steps")

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def positions(self) -> int:
        return self.effective_batch * 2048 * self.steps

    @property
    def checkpoint_steps(self) -> set[int]:
        if not self.checkpoint_milestones:
            return {self.steps}
        return {
            math.ceil(self.steps * 0.25),
            math.ceil(self.steps * 0.50),
            math.ceil(self.steps * 0.75),
            self.steps,
        }


RECIPES = {
    "eb210-1b": TrainRecipe(
        name="eb210-1b",
        learning_rate=3.0e-4,
        batch_size=15,
        grad_accum_steps=14,
        steps=2325,
        warmup_steps=116,
        stable_steps=1860,
        decay_steps=349,
        checkpoint_milestones=True,
    ),
    "eb84-400m": TrainRecipe(
        name="eb84-400m",
        learning_rate=5.1e-4,
        batch_size=14,
        grad_accum_steps=6,
        steps=2300,
        warmup_steps=115,
        stable_steps=1840,
        decay_steps=345,
        checkpoint_milestones=False,
    ),
}


@dataclass(frozen=True)
class DwarfConfig:
    vocab_size: int = 32768
    embedding_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 1536
    seq_len: int = 2048
    num_layers: int = 10
    global_mixer: str = "hisa"
    num_chunks: int = 32
    top_k_chunks: int = 4
    hisa_top_m_tokens: int = 64
    dropout: float = 0.1
    scale_embed_init: float = 0.15
    movt_dynamic_rms_target: float | None = 0.01
    movt_phase_gain_init_std: float | None = None
    init_seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.num_heads <= 0:
            raise ValueError("embedding_dim and num_heads must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers != 10:
            raise ValueError("the public DWARF-v2 topology has exactly 10 layers")
        if self.global_mixer not in {"hisa", "fa"}:
            raise ValueError("global_mixer must be 'hisa' or 'fa'")
        if isinstance(self.init_seed, bool) or not isinstance(self.init_seed, int):
            raise TypeError("init_seed must be an integer")
        if self.movt_dynamic_rms_target is not None and (
            not math.isfinite(self.movt_dynamic_rms_target)
            or self.movt_dynamic_rms_target <= 0.0
        ):
            raise ValueError("movt_dynamic_rms_target must be finite and positive, or None")
        expected_gain_std = (
            0.001
            if self.movt_dynamic_rms_target is None
            else calibrated_movt_phase_gain_std(
                head_dim=self.embedding_dim // self.num_heads,
                target_dynamic_rms=self.movt_dynamic_rms_target,
                gate_logit=0.0,
            )
        )
        if self.movt_phase_gain_init_std is None:
            object.__setattr__(self, "movt_phase_gain_init_std", expected_gain_std)
        elif not math.isclose(self.movt_phase_gain_init_std, expected_gain_std, rel_tol=1e-12):
            raise ValueError(
                "movt_phase_gain_init_std must match the value derived from "
                "movt_dynamic_rms_target and head dimension"
            )


def _config_metadata(config: DwarfConfig) -> dict[str, object]:
    return asdict(config)


def _offset_groups() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    offsets = tuple(sorted(int(offset) for offset in ALL_OFFSETS))
    if len(offsets) != 96 or len(set(offsets)) != 96:
        raise ValueError("DWARF-v2 requires the canonical 96 unique DSQG offsets")
    if any(29 <= offset <= 47 for offset in offsets):
        raise ValueError("the canonical DSQG offset lattice must not contain 29..47")
    return offsets[:32], offsets[32:64], offsets[64:96]


def _small_large_counts(offsets: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(offset) for offset in offsets)
    small = sum(offset <= 28 for offset in values)
    large = sum(offset >= 48 for offset in values)
    if small + large != len(values):
        raise ValueError("DSQG offsets must be in the canonical small or large bands")
    return small, large


GROUP_A, GROUP_B, GROUP_C = _offset_groups()
J_SMALL_A, J_LARGE_A = _small_large_counts(GROUP_A)
J_SMALL_B, J_LARGE_B = _small_large_counts(GROUP_B)
J_SMALL_C, J_LARGE_C = _small_large_counts(GROUP_C)


class FFN(nn.Module):
    def __init__(self, embedding_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    """One triadic sparse-DSQG residual block."""

    def __init__(
        self,
        config: DwarfConfig,
        offsets: tuple[int, ...],
        j_small: int,
        j_large: int,
        *,
        interference: bool,
        plane_shift: int,
    ) -> None:
        super().__init__()
        self.interference = bool(interference)
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.attn = DSQGAttentionV19(
            config.embedding_dim,
            config.num_heads,
            offsets,
            j_small,
            j_large,
            seq_len=config.seq_len,
            dropout=config.dropout,
            plane_shift=plane_shift,
            movt_dynamic_rms_target=config.movt_dynamic_rms_target,
        )
        self.ffn = FFN(config.embedding_dim, config.ffn_dim, config.dropout)
        if self.interference:
            self.inter_norm = nn.LayerNorm(config.embedding_dim)
            self.inter_gate = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.inter_k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.inter_v_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.ema_factor = nn.Parameter(torch.full((1,), 0.020833))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv_inject = None
        if self.interference:
            normalized = self.inter_norm(x)
            batch, seq_len, _ = normalized.shape
            pooled = causal_ema_scan(normalized, self.ema_factor.abs() + 1e-5, floor=1e-5)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) / math.sqrt(pooled.shape[-1]) + 1e-6)
            interference = torch.sigmoid(self.inter_gate(normalized)) * pooled
            key_delta = self.inter_k_proj(interference).reshape(
                batch, seq_len, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous()
            value_delta = self.inter_v_proj(interference).reshape(
                batch, seq_len, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous()
            kv_inject = (key_delta, value_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        return x + self.ffn(self.norm2(x))


class FullCausalAttention(nn.Module):
    """Dense causal L3 mixer used by the published FA@L3 model topology."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = embedding_dim // num_heads
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self.W_q(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.W_k(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.W_v(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
        return self.W_o(attended.transpose(1, 2).reshape(batch, seq_len, -1))


class GlobalMixerBlock(nn.Module):
    def __init__(self, config: DwarfConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        if config.global_mixer == "hisa":
            self.attn: nn.Module = HierarchicalSparseAttentionV16HISACausal(
                D=config.embedding_dim,
                H=config.num_heads,
                hd=config.embedding_dim // config.num_heads,
                num_chunks=config.num_chunks,
                top_k_chunks=config.top_k_chunks,
                hisa_top_m_tokens=config.hisa_top_m_tokens,
            )
        else:
            self.attn = FullCausalAttention(config.embedding_dim, config.num_heads)
        self.gate_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.ffn = FFN(config.embedding_dim, config.ffn_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attn(self.norm1(x))
        x = x + self.dropout(attended * torch.sigmoid(self.gate_proj(x)))
        return x + self.ffn(self.norm2(x))


class DwarfForCausalLM(nn.Module):
    """Ten-layer DWARF-v2 causal language model without retired side paths."""

    def __init__(self, config: DwarfConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        layout = (
            (GROUP_A, J_SMALL_A, J_LARGE_A, False),
            (GROUP_B, J_SMALL_B, J_LARGE_B, False),
            (GROUP_C, J_SMALL_C, J_LARGE_C, True),
            None,
            (GROUP_A, J_SMALL_A, J_LARGE_A, False),
            (GROUP_B, J_SMALL_B, J_LARGE_B, False),
            (GROUP_C, J_SMALL_C, J_LARGE_C, False),
            (GROUP_A, J_SMALL_A, J_LARGE_A, False),
            (GROUP_B, J_SMALL_B, J_LARGE_B, False),
            (GROUP_C, J_SMALL_C, J_LARGE_C, False),
        )
        blocks: list[nn.Module] = []
        dsqg_index = 0
        for item in layout:
            if item is None:
                blocks.append(GlobalMixerBlock(config))
                continue
            offsets, j_small, j_large, interference = item
            plane_segment = max(2, (config.embedding_dim // config.num_heads) // R_PLANES)
            plane_shift = 2 * (dsqg_index % max(1, plane_segment // 2))
            blocks.append(
                DSQGBlock(
                    config,
                    offsets,
                    j_small,
                    j_large,
                    interference=interference,
                    plane_shift=plane_shift,
                )
            )
            dsqg_index += 1
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.manual_seed(self.config.init_seed + 20_001)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        torch.manual_seed(self.config.init_seed + 30_001)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        torch.manual_seed(self.config.init_seed + 40_001)
        for module in self.modules():
            if isinstance(module, DSQGAttentionV19):
                nn.init.normal_(module.phase_base, mean=0.0, std=0.01)
                module.reset_phase_probes_()
                nn.init.normal_(
                    module.phase_gain,
                    mean=0.0,
                    std=module.movt_phase_gain_init_std,
                )
                nn.init.zeros_(module.phase_gate)
                nn.init.constant_(module.scale_embed, self.config.scale_embed_init)
        torch.manual_seed(self.config.init_seed + 50_001)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.init_seed + 50_001)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids)
        return hidden if return_hidden else self.lm_head(hidden)


def build_model(**kwargs: object) -> DwarfForCausalLM:
    return DwarfForCausalLM(DwarfConfig(**kwargs))


def load_packed_dataset(path: str | Path, *, seq_len: int) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(payload, dict):
        for key in ("train", "input_ids", "tokens", "data"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError("dataset must be a tensor or a dict containing train, input_ids, tokens, or data")
    if payload.ndim != 2 or payload.shape[1] != seq_len:
        raise ValueError(f"dataset must have shape [rows, {seq_len}], got {tuple(payload.shape)}")
    if payload.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"dataset token IDs must be int32 or int64, got {payload.dtype}")
    return payload.contiguous()


def wsd_multiplier(step: int, recipe: TrainRecipe) -> float:
    step = min(max(int(step), 0), recipe.steps - 1)
    if step < recipe.warmup_steps:
        return (step + 1) / recipe.warmup_steps
    decay_start = recipe.warmup_steps + recipe.stable_steps
    if step < decay_start:
        return 1.0
    progress = (step - decay_start + 1) / recipe.decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return recipe.min_lr_ratio + (1.0 - recipe.min_lr_ratio) * cosine


def _unique_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _special_parameters(model: DwarfForCausalLM) -> dict[str, list[nn.Parameter]]:
    groups = {"scale_embed": [], "phase": [], "npci_theta": []}
    for module in model.modules():
        if not isinstance(module, DSQGAttentionV19):
            continue
        groups["scale_embed"].append(module.scale_embed)
        groups["phase"].extend(
            getattr(module, name)
            for name in ("phase_base", "phase_gain", "phase_gate", "query_probes", "key_probes")
        )
        groups["npci_theta"].extend((module.npci_theta_k, module.npci_theta_v))
    return {name: _unique_parameters(parameters) for name, parameters in groups.items()}


def make_parameter_groups(
    model: DwarfForCausalLM, recipe: TrainRecipe
) -> dict[str, list[dict[str, object]]]:
    """Put hidden matrices in Muon and all remaining parameters in AdamW."""

    special = _special_parameters(model)
    special_ids = {id(parameter) for values in special.values() for parameter in values}
    muon_hidden: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if id(parameter) in special_ids:
            continue
        is_embedding = name in {"embedding.weight", "lm_head.weight"}
        if parameter.ndim == 2 and not is_embedding and "norm" not in name.lower():
            muon_hidden.append(parameter)
        elif parameter.ndim >= 2 and not is_embedding:
            adamw_decay.append(parameter)
        else:
            adamw_no_decay.append(parameter)

    def group(name: str, parameters: list[nn.Parameter], lr: float, weight_decay: float) -> dict[str, object]:
        return {
            "name": name,
            "params": parameters,
            "lr": lr,
            "base_lr": lr,
            "weight_decay": weight_decay,
        }

    muon = [group("muon_hidden", muon_hidden, recipe.learning_rate, recipe.weight_decay)]
    adamw = [
        group("adamw_no_decay", adamw_no_decay, recipe.learning_rate, 0.0),
        group(
            "adamw_scale_embed",
            special["scale_embed"],
            recipe.learning_rate * recipe.scale_embed_lr_mult,
            0.0,
        ),
        group(
            "adamw_phase",
            special["phase"],
            recipe.learning_rate * recipe.phase_lr_mult,
            0.0,
        ),
        group(
            "adamw_npci_theta",
            special["npci_theta"],
            recipe.learning_rate * recipe.npci_theta_lr_mult,
            0.0,
        ),
    ]
    if adamw_decay:
        adamw.insert(0, group("adamw_decay", adamw_decay, recipe.learning_rate, recipe.weight_decay))
    if not muon_hidden or any(not item["params"] for item in (*muon, *adamw)):
        raise RuntimeError("optimizer parameter partition contains an empty required group")
    assigned = [parameter for item in (*muon, *adamw) for parameter in item["params"]]
    expected = list(model.parameters())
    if len(assigned) != len(expected) or {id(p) for p in assigned} != {id(p) for p in expected}:
        raise RuntimeError("optimizer parameter partition is incomplete or overlapping")
    return {"muon": muon, "adamw": adamw}


class MultiOptimizer:
    def __init__(self, optimizers: Iterable[tuple[str, torch.optim.Optimizer]]) -> None:
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return [group for _, optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "muon-adamw-v1",
            "optimizers": [
                {"name": name, "state": optimizer.state_dict()}
                for name, optimizer in self.optimizers
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "muon-adamw-v1":
            raise ValueError("checkpoint optimizer kind does not match")
        saved = state.get("optimizers")
        if not isinstance(saved, list) or len(saved) != len(self.optimizers):
            raise ValueError("checkpoint optimizer count does not match")
        for (name, optimizer), item in zip(self.optimizers, saved, strict=True):
            if not isinstance(item, dict) or item.get("name") != name:
                raise ValueError("checkpoint optimizer order does not match")
            optimizer.load_state_dict(item["state"])


def build_optimizer(model: DwarfForCausalLM, recipe: TrainRecipe) -> MultiOptimizer:
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("DWARF recipes require torch.optim.Muon (PyTorch >= 2.9)")
    groups = make_parameter_groups(model, recipe)
    muon = torch.optim.Muon(
        groups["muon"],
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(groups["adamw"], betas=(0.9, 0.95), eps=1e-8)
    return MultiOptimizer((('muon', muon), ('adamw', adamw)))


def _set_learning_rates(optimizer: MultiOptimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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


def _dataset_identity(path: str | Path, rows: int) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "rows": rows}


def _checkpoint_payload(
    *,
    step: int,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    config: DwarfConfig,
    recipe: TrainRecipe,
    dataset_identity: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "dwarf-resume-v1",
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "recipe": asdict(recipe),
        "dataset": dataset_identity,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
    }


def _restore_checkpoint(
    path: str | Path,
    *,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    config: DwarfConfig,
    recipe: TrainRecipe,
    dataset_identity: dict[str, object],
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "dwarf-resume-v1":
        raise ValueError("not a DWARF resumable checkpoint")
    if checkpoint.get("config") != asdict(config) or checkpoint.get("recipe") != asdict(recipe):
        raise ValueError("checkpoint configuration or recipe does not match")
    if checkpoint.get("dataset") != dataset_identity:
        raise ValueError("checkpoint dataset identity does not match")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= recipe.steps:
        raise ValueError("checkpoint step is invalid")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    cuda_rng = checkpoint.get("cuda_rng")
    if next(model.parameters()).is_cuda:
        if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_rng)
    elif cuda_rng is not None:
        raise ValueError("CUDA checkpoint cannot be resumed on CPU")
    return step


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_reference(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for a normal DWARF run; choose --device cpu only for small FA smoke tests")
    recipe = RECIPES[args.recipe]
    stop_step = recipe.steps if args.stop_after is None else args.stop_after
    if not 0 < stop_step <= recipe.steps:
        raise ValueError(f"--stop-after must be between 1 and {recipe.steps}")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = DwarfConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        seq_len=args.seq_len,
        global_mixer=args.global_mixer,
        num_chunks=args.num_chunks,
        top_k_chunks=args.top_k_chunks,
        hisa_top_m_tokens=args.hisa_top_m_tokens,
        dropout=args.dropout,
        movt_dynamic_rms_target=args.movt_dynamic_rms_target,
        init_seed=args.seed,
    )
    dataset = load_packed_dataset(args.dataset, seq_len=config.seq_len)
    required_rows = recipe.effective_batch * stop_step
    if len(dataset) < required_rows:
        raise ValueError(f"dataset has {len(dataset)} rows; this run requires at least {required_rows}")
    order_generator = torch.Generator(device="cpu").manual_seed(args.seed + 50_001)
    order = torch.randperm(len(dataset), generator=order_generator)
    model = DwarfForCausalLM(config).to(device)
    optimizer = build_optimizer(model, recipe)
    dataset_identity = _dataset_identity(args.dataset, len(dataset))
    start_step = 0
    if args.resume:
        start_step = _restore_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            config=config,
            recipe=recipe,
            dataset_identity=dataset_identity,
        )
    if start_step >= stop_step:
        raise ValueError("checkpoint is already at or beyond --stop-after")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32) if device.type == "cuda" else None
    train_model = model if args.no_compile else torch.compile(model, mode="default", dynamic=True)
    print(
        json.dumps(
            {
                "config": _config_metadata(config),
                "recipe": asdict(recipe),
                "dataset_rows": len(dataset),
                "start_step": start_step,
                "stop_step": stop_step,
                "device": str(device),
            },
            sort_keys=True,
        )
    )

    train_model.train()
    checkpoint_steps = recipe.checkpoint_steps
    for step_index in range(start_step, stop_step):
        step = step_index + 1
        factor = wsd_multiplier(step_index, recipe)
        _set_learning_rates(optimizer, factor)
        optimizer.zero_grad()
        loss_accumulator = torch.zeros((), device=device)
        update = order[step_index * recipe.effective_batch : step * recipe.effective_batch]
        for micro_step in range(recipe.grad_accum_steps):
            begin = micro_step * recipe.batch_size
            indices = update[begin : begin + recipe.batch_size]
            batch = dataset[indices].to(device=device, dtype=torch.long, non_blocking=True)
            input_ids, labels = batch[:, :-1], batch[:, 1:]
            with _amp_context(device):
                hidden = train_model(input_ids, return_hidden=True)
                if loss_fn is None:
                    logits = model.lm_head(hidden)
                    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
                else:
                    loss = loss_fn(
                        model.lm_head.weight,
                        hidden.flatten(0, 1),
                        labels.flatten(),
                    )
            (loss / recipe.grad_accum_steps).backward()
            loss_accumulator += loss.detach() / recipe.grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip_norm)
        loss_value = float(loss_accumulator)
        if not math.isfinite(loss_value) or not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite training telemetry at step {step}")
        optimizer.step()
        if step % args.log_every == 0 or step in {1, stop_step}:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "grad_norm": float(grad_norm),
                        "lr_factor": factor,
                        "positions": step * recipe.effective_batch * config.seq_len,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        should_save = step in checkpoint_steps or step == stop_step
        should_save |= args.save_every > 0 and step % args.save_every == 0
        if should_save:
            atomic_torch_save(
                _checkpoint_payload(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    recipe=recipe,
                    dataset_identity=dataset_identity,
                ),
                output_dir / f"dwarf_step_{step:07d}.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the public DWARF-v2 reference topology on packed token rows.")
    parser.add_argument("--dataset", required=True, help="local .pt packed-token tensor")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="eb210-1b")
    parser.add_argument("--stop-after", type=int, help="bounded optimizer-step stop for smoke or staged runs")
    parser.add_argument("--global-mixer", choices=("hisa", "fa"), default="hisa")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--top-k-chunks", type=int, default=4)
    parser.add_argument("--hisa-top-m-tokens", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--movt-dynamic-rms-target",
        type=_parse_movt_dynamic_rms_target,
        default=0.01,
        help="pre-prior MOVT content-angle RMS target, or legacy/off/none/0 for std=0.001",
    )
    parser.add_argument("--save-every", type=int, default=0, help="extra checkpoint interval; 0 uses recipe milestones")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_reference(parse_args())
