#!/usr/bin/env python3
"""Minimal public trainer for DWARF-v2.

This reference implementation contains the active DWARF topology only:
triadic DSQG sparse blocks, a causal EMA interference injection at L2, and one
L3 global mixer.  The global mixer can be strict-causal V16 HISA or full causal
SDPA (`--global-mixer fa`), which is the topology used by DWARF-55M-Base.

The trainer contains no experimental side paths, data-build scripts, launchers,
or evaluation tools.  It accepts a local packed integer tensor of shape [rows,
sequence_length] and writes model-only checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class DwarfConfig:
    vocab_size: int = 50282
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

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.num_heads <= 0:
            raise ValueError("embedding_dim and num_heads must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers != 10:
            raise ValueError("the public DWARF-v2 topology has exactly 10 layers")
        if self.global_mixer not in {"hisa", "fa"}:
            raise ValueError("global_mixer must be 'hisa' or 'fa'")
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
                module.reset_phase_probes_()
                nn.init.normal_(
                    module.phase_gain,
                    mean=0.0,
                    std=module.movt_phase_gain_init_std,
                )
                nn.init.zeros_(module.phase_gate)
                nn.init.constant_(module.scale_embed, self.config.scale_embed_init)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(input_ids))


def build_model(**kwargs: object) -> DwarfForCausalLM:
    return DwarfForCausalLM(DwarfConfig(**kwargs))


def load_packed_dataset(path: str | Path, *, seq_len: int) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("input_ids", "tokens", "data"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        raise TypeError("dataset must be a tensor or a dict containing input_ids, tokens, or data")
    if payload.ndim != 2 or payload.shape[1] != seq_len:
        raise ValueError(f"dataset must have shape [rows, {seq_len}], got {tuple(payload.shape)}")
    if payload.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"dataset token IDs must be int32 or int64, got {payload.dtype}")
    return payload.long().contiguous()


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_reference(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for a normal DWARF run; choose --device cpu only for small FA smoke tests")
    torch.manual_seed(args.seed)
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
    )
    dataset = load_packed_dataset(args.dataset, seq_len=config.seq_len)
    if len(dataset) < args.batch_size:
        raise ValueError("dataset has fewer rows than --batch-size")
    model = DwarfForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "config": _config_metadata(config),
                "dataset_rows": len(dataset),
                "device": str(device),
            },
            sort_keys=True,
        )
    )

    model.train()
    for step in range(1, args.max_steps + 1):
        indices = torch.randint(len(dataset), (args.batch_size,))
        batch = dataset[indices].to(device, non_blocking=True)
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        print(
            json.dumps(
                {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "tokens": int(labels.numel())},
                sort_keys=True,
            ),
            flush=True,
        )
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": _config_metadata(config),
                    "step": step,
                },
                output_dir / f"dwarf_step_{step}.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the public DWARF-v2 reference topology on packed token rows.")
    parser.add_argument("--dataset", required=True, help="local .pt tensor with shape [rows, --seq-len]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-mixer", choices=("hisa", "fa"), default="hisa")
    parser.add_argument("--vocab-size", type=int, default=50282)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_reference(parse_args())
