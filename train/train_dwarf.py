#!/usr/bin/env python3
"""Self-contained public trainer for the canonical DWARF-v2 architecture.

The model is D512/H8/L24/FFN2048 with bounded-routing DSQG V23 blocks and
strict-causal HISA V19 global mixers at layers 3, 10, and 17. Independent
causal-EMA K/V packets live at L3 and L17; L10 is ordinary HISA. At L17 only,
base global K supplies representatives, hard top-k routes, selected route
priors, and router auxiliary/oracle targets while packet-rotated K/V remain in
global token attention.

The trainer accepts packed token rows and includes Muon+AdamW, WSD, deterministic
row selection, atomic resumable checkpoints, and the complete model definition.
Dataset preparation and evaluation remain out of scope.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_FILES = (
    "causal_ema_scan.py",
    "dsqg_attention_v23.py",
    "hierarchical_sparse_attn_v19_hisa.py",
)
KERNEL_CANDIDATES = (
    SCRIPT_DIR,
    SCRIPT_DIR / "kernels",
    SCRIPT_DIR.parent / "kernels",
    SCRIPT_DIR.parent.parent / "kernels",
)
KERNEL_DIR = next(
    (
        candidate
        for candidate in KERNEL_CANDIDATES
        if all((candidate / name).is_file() for name in KERNEL_FILES)
    ),
    None,
)
if KERNEL_DIR is None:
    searched = ", ".join(str(path) for path in KERNEL_CANDIDATES)
    raise FileNotFoundError(
        "canonical DWARF kernel files were not found together; searched: " + searched
    )
for directory in (SCRIPT_DIR, KERNEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from causal_ema_scan import (  # noqa: E402
    bounded_ema_factor,
    causal_ema_scan3,
    causal_ema_triton_available,
    inverse_bounded_ema_factor,
)
from dsqg_attention_v23 import (  # noqa: E402
    ALL_OFFSETS,
    DSQGAttentionV23,
    dsqg_triton_available,
)
from hierarchical_sparse_attn_v19_hisa import (  # noqa: E402
    HierarchicalSparseAttentionV19HISACausal,
    hisa_runtime_capabilities,
)

CHECKPOINT_KIND = "dwarf-l24-l17-basek-route-public-packed-dsqgv23-hisav19-resume-v1"
CANONICAL_PARENT_CHECKPOINT_KIND = (
    "dwarf-l24-l17-basek-route-appendable-prefix-stable-dsqgv23-hisav19-resume-v1"
)
RELEASE_KIND = "dwarf-l24-l17-basek-route-dsqgv23-hisav19-weights-v1"
ROUTE_AUX_RECIPE_SEED = 20_260_809
EXPECTED_PARAMETERS = 100_179_313
EXPECTED_TRAINABLE_PARAMETERS = 100_179_297
EXPECTED_STATE_FINGERPRINT = (
    "5219e76a32c32db3b59697a0230eca730969d17b82696b23148a5bf9b935b06a"
)
EXPECTED_SEEDED_RNG_FINGERPRINT = (
    "799d503b1b60a6fcd0d39e034521df4c10c4eee699551f781bf9c9c20f573010"
)
CANONICAL_TOKENIZER_SHA256 = (
    "c695c9831c1af101ea17e95e37d82e47f079b3813d97a44b422daa0d0369d579"
)


@dataclass(frozen=True)
class TrainRecipe:
    learning_rate: float = 3.0e-4
    batch_size: int = 16
    grad_accum_steps: int = 8
    # One schedule identity spans the complete appendable 20B stable-LR trunk.
    steps: int = 76_331
    warmup_steps: int = 1_527
    stable_steps: int = 74_804
    decay_steps: int = 0
    min_lr_ratio: float = 1.0
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
        return {1_909, 3_817, 5_725, 7_633, 19_083, 38_166, 76_331}


@dataclass(frozen=True)
class DwarfConfig:
    vocab_size: int = 32768
    embedding_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    seq_len: int = 2048
    num_layers: int = 24
    global_mixer_layers: tuple[int, ...] = (3, 10, 17)
    ema_packet_layers: tuple[int, ...] = (3, 17)
    base_k_routing_layers: tuple[int, ...] = (17,)
    ema_packet_initial_coupling_fractions: tuple[float, ...] = (1.0, 0.3)
    dropout: float = 0.05
    min_offset_support: int = 64
    dsqg_backend: str = "triton"
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
        if len(self.ema_timescales) != 3:
            raise ValueError("DWARF requires exactly three EMA timescales")
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
        if (
            tuple(sorted(set(self.ema_packet_layers))) != self.ema_packet_layers
            or not set(self.ema_packet_layers).issubset(self.global_mixer_layers)
        ):
            raise ValueError("EMA packet layers must be sorted unique HISA layers")
        if (
            tuple(sorted(set(self.base_k_routing_layers)))
            != self.base_k_routing_layers
            or not set(self.base_k_routing_layers).issubset(self.ema_packet_layers)
        ):
            raise ValueError("base-K routing layers must be sorted packet-enabled HISA layers")
        if len(self.ema_packet_initial_coupling_fractions) != len(
            self.ema_packet_layers
        ) or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.ema_packet_initial_coupling_fractions
        ):
            raise ValueError(
                "EMA packet coupling fractions must align with packet layers and be in (0,1]"
            )
        if self.seq_len < 65:
            raise ValueError("seq_len must be at least 65")
        head_dim = self.embedding_dim // self.num_heads
        if head_dim & (head_dim - 1):
            raise ValueError("HISA requires a power-of-two head dimension")
        if self.dsqg_backend not in {"auto", "eager", "triton"}:
            raise ValueError("DSQG backend must be auto, eager, or triton")
        if self.hisa_top_m_tokens != self.hisa_chunk_size:
            raise ValueError(
                "canonical HISA enumerates complete chunks; top-M must equal chunk size"
            )
        if self.hisa_chunk_selection_scope != "token":
            raise ValueError("canonical HISA chunk selection scope must be token")
        if self.hisa_token_routing_pack_size not in {1, 2, 4, 8, 16}:
            raise ValueError("HISA token routing pack size must be 1, 2, 4, 8, or 16")
        if self.hisa_backend not in {"auto", "eager", "triton"}:
            raise ValueError("HISA backend must be auto, eager, or triton")
        if self.hisa_token_selection_mode != "auto":
            raise ValueError("canonical HISA token selection mode must be auto")
        if self.hisa_local_backend != "flex":
            raise ValueError("canonical HISA local backend must be flex")
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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_liger_loss_class():
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
    except Exception as error:
        raise RuntimeError(
            "DWARF training requires Liger fused linear cross-entropy"
        ) from error
    return LigerFusedLinearCrossEntropyLoss


def runtime_environment(device: torch.device | None = None) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            None if not torch.backends.cudnn.is_available()
            else torch.backends.cudnn.version()
        ),
        "triton": _package_version("triton"),
        "liger_kernel": _package_version("liger-kernel"),
        "tokenizers": _package_version("tokenizers"),
        "execution_policy": {
            "autocast_dtype": "bfloat16",
            "compile_mode": "default",
            "compile_dynamic": False,
            "compiled_backward_autocast": "off",
        },
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        resolved = torch.device(
            "cuda", torch.cuda.current_device() if device.index is None else device.index
        )
        properties = torch.cuda.get_device_properties(resolved)
        environment["gpu"] = {
            "logical_device": resolved.index,
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "visible_device_count": torch.cuda.device_count(),
        }
    return environment


def validate_training_runtime(
    device: torch.device, config: DwarfConfig
) -> tuple[type, dict[str, Any]]:
    """Fail before data hashing or model allocation when production backends are absent."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DWARF training requires CUDA")
    resolved_index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= resolved_index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {resolved_index} is not visible")
    torch.cuda.set_device(resolved_index)
    resolved = torch.device("cuda", resolved_index)
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # Compatibility with older supported PyTorch point releases.
        native_bf16 = torch.cuda.is_bf16_supported()
    if not native_bf16:
        raise RuntimeError("canonical DWARF training requires native CUDA BF16 support")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("DWARF requires torch.optim.Muon (PyTorch >= 2.9)")
    if config.dsqg_backend != "eager" and not dsqg_triton_available():
        raise RuntimeError("canonical CUDA DSQG execution requires Triton")
    if not causal_ema_triton_available():
        raise RuntimeError("canonical CUDA causal EMA execution requires Triton")
    hisa = hisa_runtime_capabilities()
    if not hisa["flex_attention"]:
        raise RuntimeError("canonical CUDA HISA execution requires FlexAttention")
    if config.hisa_backend != "eager" and not hisa["triton"]:
        raise RuntimeError("canonical CUDA HISA global execution requires Triton")
    loss_class = _load_liger_loss_class()
    return loss_class, runtime_environment(resolved)


def _environment_abi(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("checkpoint runtime environment is invalid")
    gpu = value.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("checkpoint GPU environment is invalid")
    return {
        "python": value.get("python"),
        "pytorch": value.get("pytorch"),
        "cuda_runtime": value.get("cuda_runtime"),
        "cudnn": value.get("cudnn"),
        "triton": value.get("triton"),
        "liger_kernel": value.get("liger_kernel"),
        "execution_policy": value.get("execution_policy"),
        # Device UUID, logical index, memory size, and visible-device count are
        # intentionally excluded. The numerical/runtime ABI is tied to the GPU
        # family and compute capability, not to unrelated visibility changes.
        "gpu_name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
    }


def validate_checkpoint_environment(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_mismatch: bool = False,
) -> None:
    saved_abi = _environment_abi(saved)
    current_abi = _environment_abi(current)
    if saved_abi == current_abi:
        return
    if allow_mismatch:
        return
    differences = {
        key: {"saved": saved_abi.get(key), "current": current_abi.get(key)}
        for key in sorted(set(saved_abi) | set(current_abi))
        if saved_abi.get(key) != current_abi.get(key)
    }
    raise ValueError(
        "checkpoint runtime environment does not match; pass "
        "--allow-environment-mismatch only after reviewing: "
        + json.dumps(differences, sort_keys=True)
    )


def _architecture_without_sources(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("checkpoint architecture metadata is invalid")
    result = copy.deepcopy(value)
    result.pop("sources", None)
    return result


def validate_checkpoint_architecture(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_source_mismatch: bool = False,
) -> None:
    if _architecture_without_sources(saved) != _architecture_without_sources(current):
        raise ValueError("checkpoint semantic architecture does not match")
    if saved.get("sources") != current.get("sources") and not allow_source_mismatch:
        raise ValueError(
            "checkpoint source manifest does not match; pass "
            "--allow-source-mismatch only after reviewing the source diff"
        )


def _require(condition: bool, message: str) -> None:
    """Raise an explicit release-test failure even under ``python -O``."""
    if not bool(condition):
        raise AssertionError(message)


def _require_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    message: str,
) -> None:
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, atol=atol, rtol=rtol
    ):
        maximum = (
            float((actual - expected).abs().max())
            if actual.shape == expected.shape and actual.numel()
            else float("inf")
        )
        raise AssertionError(f"{message}; maximum absolute difference={maximum}")


def _require_raises(
    error_type: type[BaseException], callable_, message: str
) -> None:
    try:
        callable_()
    except error_type:
        return
    except BaseException as error:
        raise AssertionError(
            f"{message}; raised {type(error).__name__} instead of {error_type.__name__}"
        ) from error
    raise AssertionError(f"{message}; no exception was raised")


def _independent_lagged_ema(
    x: torch.Tensor, factors: torch.Tensor
) -> torch.Tensor:
    accumulator_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    state = torch.zeros(
        x.shape[0], 3, x.shape[2], device=x.device, dtype=accumulator_dtype
    )
    alpha = factors.to(accumulator_dtype).reshape(1, 3, 1)
    rows: list[torch.Tensor] = []
    values = x.to(accumulator_dtype)
    for token in range(x.shape[1]):
        rows.append(state)
        state = alpha * values[:, token, None] + (1.0 - alpha) * state
    if not rows:
        return x.new_empty((x.shape[0], 0, 3, x.shape[2]))
    return torch.stack(rows, dim=1).to(x.dtype)


def assert_public_kernel_contracts() -> None:
    """Exercise CPU reference semantics without relying on optimized asserts."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(17)

        # Causal EMA: preceding-state semantics and exact CPU gradients.
        ema_x = torch.randn(2, 9, 7, dtype=torch.float64, requires_grad=True)
        ema_factors = torch.tensor(
            (0.07, 0.19, 0.41), dtype=torch.float64, requires_grad=True
        )
        ema_actual = causal_ema_scan3(ema_x, ema_factors)
        ema_expected = _independent_lagged_ema(ema_x, ema_factors)
        _require_close(
            ema_actual,
            ema_expected,
            atol=0.0,
            rtol=0.0,
            message="causal EMA preceding-state recurrence changed",
        )
        probe = torch.randn_like(ema_actual)
        actual_gradients = torch.autograd.grad(
            (ema_actual * probe).sum(), (ema_x, ema_factors), retain_graph=False
        )
        expected_gradients = torch.autograd.grad(
            (ema_expected * probe).sum(), (ema_x, ema_factors), retain_graph=False
        )
        for name, actual, expected in zip(
            ("input", "factor"), actual_gradients, expected_gradients, strict=True
        ):
            _require_close(
                actual,
                expected,
                atol=1e-12,
                rtol=1e-12,
                message=f"causal EMA {name} gradient changed",
            )

        # DSQG: state compatibility, support boundary, strict causality, and
        # bounded live telemetry.
        dsqg_kwargs = {
            "embedding_dim": 64,
            "num_heads": 4,
            "offsets": (29, 32, 47),
            "seq_len": 64,
            "dropout": 0.0,
            "backend": "eager",
            "diagnostic_max_queries": 5,
        }
        dsqg = DSQGAttentionV23(**dsqg_kwargs).double().eval()
        _require(dsqg.offsets == (29, 32, 47), "DSQG offsets changed")
        restored = DSQGAttentionV23(**dsqg_kwargs).double()
        incompatible = restored.load_state_dict(dsqg.state_dict(), strict=True)
        _require(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            "DSQG state-dict compatibility changed",
        )
        values = torch.randn(2, 64, 64, dtype=torch.float64, requires_grad=True)
        output = dsqg(values, collect_diagnostics=True)
        _require(
            torch.equal(output[:, :29], torch.zeros_like(output[:, :29])),
            "DSQG emitted values before minimum-offset support",
        )
        _require(
            torch.isfinite(output).all(), "DSQG CPU reference emitted non-finite values"
        )
        diagnostics = dsqg.routing_diagnostics()
        for key in (
            "null_attention_mass",
            "local_offset_mass",
            "mid_offset_mass",
            "long_offset_mass",
            "content_gate_mean",
        ):
            _require(key in diagnostics, f"DSQG diagnostic is absent: {key}")
            _require(
                torch.isfinite(diagnostics[key]).all(),
                f"DSQG diagnostic is non-finite: {key}",
            )
        split = 49
        changed = values.detach().clone()
        changed[:, split:] = torch.randn_like(changed[:, split:]) * 7.0
        prefix_a = dsqg(values.detach())[:, :split]
        prefix_b = dsqg(changed)[:, :split]
        _require_close(
            prefix_a,
            prefix_b,
            atol=0.0,
            rtol=0.0,
            message="DSQG future-token perturbation changed a causal prefix",
        )
        future_gradient = torch.autograd.grad(
            output[:, :split].square().sum(), values, retain_graph=False
        )[0][:, split:]
        _require(
            torch.count_nonzero(future_gradient) == 0,
            "DSQG causal prefix has a future-token gradient",
        )

        # HISA: local/global merge, sampled auxiliary, strict causality, and
        # formerly dead controls.
        hisa_kwargs = {
            "D": 64,
            "H": 4,
            "hd": 16,
            "top_k_chunks": 2,
            "hisa_top_m_tokens": 8,
            "chunk_size": 8,
            "local_window": 16,
            "selector_tile_size": 4,
            "token_routing_pack_size": 2,
            "exploration_probability": 0.0,
            "route_aux_weight": 0.02,
            "route_aux_samples": 2,
            "route_aux_temperature": 0.5,
            "route_aux_oracle_temperature": 0.3,
            "global_adapter_rank": 8,
            "binding_rank": 8,
            "max_seq_len": 64,
            "backend": "eager",
            "diagnostic_max_queries": 4,
        }
        hisa = HierarchicalSparseAttentionV19HISACausal(**hisa_kwargs).train()
        hisa_input = torch.randn(2, 64, 64, requires_grad=True)
        auxiliary_ids = torch.tensor((24, 47), dtype=torch.int64)
        hisa_output, hisa_auxiliary = hisa(
            hisa_input,
            route_aux_tile_ids=auxiliary_ids,
            collect_diagnostics=True,
            return_auxiliary=True,
        )
        _require(
            torch.isfinite(hisa_output).all() and torch.isfinite(hisa_auxiliary),
            "HISA CPU reference emitted a non-finite output or auxiliary",
        )
        _require(
            torch.equal(hisa_output[:, :1], torch.zeros_like(hisa_output[:, :1])),
            "strict-causal HISA position zero is not exactly zero",
        )
        (hisa_output.square().mean() + hisa_auxiliary).backward()
        _require(
            hisa_input.grad is not None and torch.isfinite(hisa_input.grad).all(),
            "HISA CPU backward produced an absent or non-finite input gradient",
        )
        _require(
            hisa.hisa_evidence_capture is None,
            "ordinary HISA diagnostics retained the full selector capture",
        )
        _require(
            "global_attention_mass" in hisa._routing_diagnostics,
            "HISA live routing diagnostics were not populated",
        )

        hisa.eval()
        causal_input = torch.randn(1, 64, 64, requires_grad=True)
        causal_changed = causal_input.detach().clone()
        causal_split = 51
        causal_changed[:, causal_split:] = torch.randn_like(
            causal_changed[:, causal_split:]
        ) * 11.0
        causal_a = hisa(causal_input)[:, :causal_split]
        causal_b = hisa(causal_changed)[:, :causal_split]
        _require_close(
            causal_a,
            causal_b,
            atol=0.0,
            rtol=0.0,
            message="HISA future-token perturbation changed a causal prefix",
        )
        hisa_future_gradient = torch.autograd.grad(
            causal_a.square().sum(), causal_input, retain_graph=False
        )[0][:, causal_split:]
        _require(
            torch.count_nonzero(hisa_future_gradient) == 0,
            "HISA causal prefix has a future-token gradient",
        )

        cold = HierarchicalSparseAttentionV19HISACausal(
            **hisa_kwargs, temperature=0.5
        ).eval()
        hot = HierarchicalSparseAttentionV19HISACausal(
            **hisa_kwargs, temperature=2.0
        ).eval()
        hot.load_state_dict(cold.state_dict(), strict=True)
        control_input = torch.randn(1, 64, 64)
        cold_output = cold(control_input)
        hot_output = hot(control_input)
        _require(
            not torch.allclose(cold_output, hot_output, atol=1e-8, rtol=1e-8),
            "HISA temperature is still a semantic no-op",
        )
        mean_low = float(
            HierarchicalSparseAttentionV19HISACausal(
                **hisa_kwargs, representative_blend_alpha=0.2
            ).representative_mix.detach().mean()
        )
        mean_high = float(
            HierarchicalSparseAttentionV19HISACausal(
                **hisa_kwargs, representative_blend_alpha=0.8
            ).representative_mix.detach().mean()
        )
        _require(
            mean_low < 0.25 and mean_high > 0.75,
            "HISA representative_blend_alpha does not control initialization",
        )
        bounded = HierarchicalSparseAttentionV19HISACausal(
            **{**hisa_kwargs, "max_seq_len": 32}
        )
        _require_raises(
            ValueError,
            lambda: bounded(torch.randn(1, 33, 64)),
            "HISA max_seq_len is not enforced",
        )
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

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scan_input = normalized.to(torch.bfloat16) if normalized.is_cuda else normalized
        scans = causal_ema_scan3(scan_input, self.ema_factors)
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
            backend=config.dsqg_backend,
            # Every canonical group contains offsets 1..8, so projection cropping
            # can never activate and only creates a misleading execution surface.
            support_crop_projections=False,
            support_crop_min_offset=64,
            diagnostic_max_queries=config.hisa_diagnostic_max_queries,
        )
        _consume_retired_movt_rng(
            offsets,
            config.num_heads,
            config.embedding_dim // config.num_heads,
            reset_path=False,
        )
        self.ffn = SwiGLUFFN(config)

    def forward(
        self, x: torch.Tensor, *, collect_diagnostics: bool = False
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), collect_diagnostics=collect_diagnostics
        )
        return x + self.ffn(self.norm2(x))


class GlobalMixerBlock(nn.Module):
    def __init__(
        self,
        config: DwarfConfig,
        *,
        use_packet: bool,
        packet_initial_coupling_fraction: float | None,
        route_from_base_global_key: bool,
    ) -> None:
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
            route_from_base_global_key=route_from_base_global_key,
        )
        self.packet = InterferencePacket(config) if use_packet else None
        self.packet_initial_coupling_fraction = packet_initial_coupling_fraction
        if (self.packet is None) != (packet_initial_coupling_fraction is None):
            raise ValueError("packet presence and coupling fraction must agree")
        if self.packet is None:
            self.attn.npci_theta_k.requires_grad_(False)
            self.attn.npci_theta_v.requires_grad_(False)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    @torch.no_grad()
    def reset_packet_coupling_(self) -> None:
        if self.packet is None:
            return
        fraction = float(self.packet_initial_coupling_fraction)
        physical_rotation = 0.01 * fraction
        raw = math.atanh(
            min(physical_rotation / max(self.attn.npci_theta_max, 1e-6), 0.99)
        )
        self.attn.npci_theta_k.fill_(raw)
        self.attn.npci_theta_v.fill_(raw)

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        kv_inject = self.packet(normalized) if self.packet is not None else None
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
        packet_fractions = dict(
            zip(
                config.ema_packet_layers,
                config.ema_packet_initial_coupling_fractions,
                strict=True,
            )
        )
        blocks: list[nn.Module] = []
        for layer_index, group_index in enumerate(layout):
            if group_index is None:
                fraction = packet_fractions.get(layer_index)
                blocks.append(
                    GlobalMixerBlock(
                        config,
                        use_packet=fraction is not None,
                        packet_initial_coupling_fraction=fraction,
                        route_from_base_global_key=(
                            layer_index in config.base_k_routing_layers
                        ),
                    )
                )
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
            for block in self.blocks:
                if isinstance(block, GlobalMixerBlock):
                    block.reset_packet_coupling_()
        finally:
            torch.random.set_rng_state(state)

    def prepare_runtime(self, device: torch.device | str) -> None:
        for module in self.modules():
            if isinstance(module, HierarchicalSparseAttentionV19HISACausal):
                module.prepare_runtime(device, self.config.model_length)

    def _forward_hidden_impl(
        self,
        input_ids: torch.Tensor,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_auxiliary: bool = False,
    ):
        x = self.embedding(input_ids)
        if x.is_cuda:
            x = x.to(torch.bfloat16)
        x = self.dropout(x)
        auxiliary = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            if isinstance(block, GlobalMixerBlock):
                x, block_auxiliary = block(
                    x,
                    valid_lengths,
                    route_aux_tile_ids,
                    collect_diagnostics,
                )
                auxiliary = auxiliary + block_auxiliary.float()
            else:
                x = block(x, collect_diagnostics=collect_diagnostics)
        hidden = self.norm(x)
        if hidden.is_cuda:
            hidden = hidden.to(torch.bfloat16)
        return (hidden, auxiliary) if return_auxiliary else hidden

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_auxiliary: bool = False,
    ):
        kwargs = {
            "valid_lengths": valid_lengths,
            "route_aux_tile_ids": route_aux_tile_ids,
            "collect_diagnostics": collect_diagnostics,
            "return_auxiliary": return_auxiliary,
        }
        if input_ids.is_cuda and not torch.is_autocast_enabled():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._forward_hidden_impl(input_ids, **kwargs)
        return self._forward_hidden_impl(input_ids, **kwargs)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_hidden: bool = False,
        return_auxiliary: bool = False,
    ):
        hidden, auxiliary = self._forward_hidden_impl(
            input_ids,
            valid_lengths=valid_lengths,
            route_aux_tile_ids=route_aux_tile_ids,
            collect_diagnostics=collect_diagnostics,
            return_auxiliary=True,
        )
        output = hidden if return_hidden else self.lm_head(hidden)
        return (output, auxiliary) if return_auxiliary else output

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        valid_lengths: torch.Tensor | None = None,
        route_aux_tile_ids: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        return_hidden: bool = False,
        return_auxiliary: bool = False,
    ):
        kwargs = {
            "valid_lengths": valid_lengths,
            "route_aux_tile_ids": route_aux_tile_ids,
            "collect_diagnostics": collect_diagnostics,
            "return_hidden": return_hidden,
            "return_auxiliary": return_auxiliary,
        }
        if input_ids.is_cuda and not torch.is_autocast_enabled():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._forward_impl(input_ids, **kwargs)
        return self._forward_impl(input_ids, **kwargs)


def model_metadata(model: DwarfForCausalLM) -> dict[str, Any]:
    global_mixers = [
        (index, block)
        for index, block in enumerate(model.blocks)
        if isinstance(block, GlobalMixerBlock)
    ]
    dsqg_mixers = [
        (index, block)
        for index, block in enumerate(model.blocks)
        if isinstance(block, DSQGBlock)
    ]
    layer_names = [
        "HISA" if isinstance(block, GlobalMixerBlock) else "DSQG"
        for block in model.blocks
    ]
    return {
        "format": "dwarf-l24-l17-basek-route-dsqgv23-hisav19-v1",
        "config": asdict(model.config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "topology": {
            "layers": ",".join(layer_names),
            "global_mixer_layers": tuple(index for index, _ in global_mixers),
            "ema_packet_layers": tuple(
                index for index, block in global_mixers if block.packet is not None
            ),
            "base_k_routing_layers": tuple(
                index
                for index, block in global_mixers
                if block.attn.route_from_base_global_key
            ),
            "routing_key_source_by_hisa_layer": {
                str(index): block.attn.route_source for index, block in global_mixers
            },
            "l17_route_contract": {
                "representatives": "base_global_k_after_ordinary_global_adapter",
                "hard_top_k": "base_global_k",
                "selected_route_priors": "base_global_k",
                "router_auxiliary": "base_global_k",
                "router_oracle": "base_global_k",
                "global_token_attention_key": "post_packet_rotated_global_k",
                "global_token_attention_value": "post_packet_rotated_global_v",
            },
            "dsqg": "v23-bounded-routing-null-candidate",
            "hisa": "v19-accessible-routing-semantic-binder",
            "offset_groups": model.offset_groups,
        },
        "complexity": {
            "dsqg_attention": "linear_in_sequence_length_for_fixed_offsets",
            "hisa_selected_attention": "linear_in_sequence_length_for_fixed_routing",
            "selector": "quadratic_in_sequence_length_at_fixed_chunk_size",
        },
        "cache": {
            "bounded_dsqg_history": True,
            "model_level_o1_kv_cache": False,
            "incremental_generation_api": False,
        },
        "dsqg": [
            {
                "layer": index,
                "semantic": block.attn.semantic_config(),
                "execution": block.attn.execution_config(),
            }
            for index, block in dsqg_mixers
        ],
        "hisa": [
            {
                "layer": index,
                "ema_packet": block.packet is not None,
                "ema_initial_coupling_fraction": (
                    block.packet_initial_coupling_fraction
                ),
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
    # Select exactly the requested run horizon. A later resume may safely extend
    # this same deterministic sequential prefix without rescanning unused rows.
    selected_steps = stop_step
    selected_rows = selected_steps * recipe.effective_batch
    return torch.arange(selected_rows, dtype=torch.int64), {
        "mode": "sequential_prefix",
        "rows": selected_rows,
        "steps": selected_steps,
        "dataset_order_requirement": "rows_must_be_pre_shuffled",
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
    # The selected horizon may grow or shrink across invocations. Both selections
    # are the same sequential prefix, and the consumed-row bound above is the only
    # condition needed for an exact resume.


def preflight_training_rows(
    dataset: torch.Tensor,
    *,
    selected_rows: int,
    vocab_size: int,
    pad_token_id: int | None = None,
    chunk_rows: int = 1024,
) -> dict[str, int]:
    if dataset.device.type != "cpu" or dataset.ndim != 2:
        raise ValueError("dataset preflight requires a two-dimensional CPU tensor")
    if not 0 < selected_rows <= len(dataset) or vocab_size < 1 or chunk_rows < 1:
        raise ValueError("dataset preflight bounds are invalid")
    minimum = vocab_size
    maximum = -1
    pad_tokens = 0
    for start in range(0, selected_rows, chunk_rows):
        rows = dataset[start : min(start + chunk_rows, selected_rows)]
        local_min_tensor, local_max_tensor = torch.aminmax(rows)
        local_min = int(local_min_tensor)
        local_max = int(local_max_tensor)
        if local_min < 0 or local_max >= vocab_size:
            raise ValueError(
                f"selected row contains a token outside vocabulary [0,{vocab_size})"
            )
        if pad_token_id is not None:
            pad_tokens += int((rows == int(pad_token_id)).sum())
        minimum = min(minimum, local_min)
        maximum = max(maximum, local_max)
    if pad_tokens:
        raise ValueError(
            f"selected packed rows contain {pad_tokens} pad tokens, but the canonical "
            "fused language-model loss has no padding mask"
        )
    return {
        "rows": selected_rows,
        "tokens": selected_rows * dataset.shape[1],
        "minimum_token_id": minimum,
        "maximum_token_id": maximum,
        "pad_tokens": pad_tokens,
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
        return self.model._forward_impl(
            input_ids,
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
    # Diagnostics intentionally remain eager: they mutate per-layer telemetry
    # dictionaries and run only on the final microbatch of logging updates.
    diagnostic = _TrainingForwardCallable(model, diagnostics=True)
    return normal, diagnostic


class BatchStager:
    """Double-buffer contiguous packed rows with overlapped H2D transfer."""

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
        shape = (batch_size, dataset.shape[1])
        self.host_buffers = [
            torch.empty(shape, dtype=dataset.dtype, pin_memory=True)
            for _ in range(2)
        ]
        self.device_buffers = [
            torch.empty(shape, dtype=torch.long, device=device)
            for _ in range(2)
        ]
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.copy_done = [torch.cuda.Event() for _ in range(2)]
        self.compute_done = [torch.cuda.Event() for _ in range(2)]
        self.copy_recorded = [False, False]
        self.compute_recorded = [False, False]
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dwarf-data"
        )

    def _fill_contiguous(self, row_start: int, slot: int) -> int:
        if self.copy_recorded[slot]:
            self.copy_done[slot].synchronize()
        source = self.dataset.narrow(0, row_start, self.batch_size)
        self.host_buffers[slot].copy_(source)
        return slot

    def _enqueue_copy(self, slot: int) -> None:
        with torch.cuda.stream(self.transfer_stream):
            if self.compute_recorded[slot]:
                self.transfer_stream.wait_event(self.compute_done[slot])
            self.device_buffers[slot].copy_(
                self.host_buffers[slot], non_blocking=True
            )
            self.copy_done[slot].record(self.transfer_stream)
            self.copy_recorded[slot] = True

    def _wait_for_copy(self, slot: int) -> torch.Tensor:
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.copy_done[slot])
        return self.device_buffers[slot]

    def batches(self, update_indices: torch.Tensor) -> Iterator[torch.Tensor]:
        if update_indices.device.type != "cpu" or update_indices.ndim != 1:
            raise ValueError("batch staging indices must be a one-dimensional CPU tensor")
        if len(update_indices) % self.batch_size:
            raise ValueError("update row count must be divisible by the microbatch size")
        if not len(update_indices):
            return
        first = int(update_indices[0])
        expected = torch.arange(first, first + len(update_indices), dtype=torch.int64)
        if not torch.equal(update_indices.to(torch.int64), expected):
            raise ValueError(
                "canonical BatchStager requires contiguous sequential row indices"
            )
        starts = list(range(first, first + len(update_indices), self.batch_size))

        initial_fills = [
            self.executor.submit(self._fill_contiguous, starts[index], index)
            for index in range(min(2, len(starts)))
        ]
        for future in initial_fills:
            self._enqueue_copy(future.result())

        for index in range(len(starts)):
            slot = index % 2
            batch = self._wait_for_copy(slot)
            next_index = index + 2
            # The previous H2D read of this host slot has completed, so refill it
            # while the consumer computes on the corresponding device buffer.
            next_fill = (
                self.executor.submit(
                    self._fill_contiguous, starts[next_index], slot
                )
                if next_index < len(starts)
                else None
            )
            yield batch

            # The consumer has now enqueued forward/backward work on the current
            # stream. Record when this persistent device slot becomes reusable.
            current = torch.cuda.current_stream(self.device)
            self.compute_done[slot].record(current)
            self.compute_recorded[slot] = True

            if next_fill is not None:
                next_fill.result()
                # This transfer waits for compute_done[slot] and can overlap the
                # other slot's next microbatch.
                self._enqueue_copy(slot)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.transfer_stream.synchronize()


def _tensor_storage_key(value: torch.Tensor) -> tuple[Any, ...] | None:
    if value.layout != torch.strided:
        return None
    storage = value.untyped_storage()
    return (
        value.device.type,
        value.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


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


def _resolved_cuda_device(device: torch.device | str) -> torch.device:
    value = torch.device(device)
    if value.type != "cuda":
        raise ValueError("expected a CUDA device")
    return torch.device(
        "cuda", torch.cuda.current_device() if value.index is None else value.index
    )


def checkpoint_payload(
    *,
    step: int,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
    device: torch.device,
    environment: dict[str, Any],
) -> dict[str, Any]:
    resolved = _resolved_cuda_device(device)
    return {
        "kind": CHECKPOINT_KIND,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture": architecture,
        "recipe": asdict(RECIPE),
        "dataset": dataset,
        "environment": copy.deepcopy(environment),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        # Save only the selected training device. Resume is then independent of
        # unrelated GPUs becoming visible, hidden, or reordered.
        "cuda_rng": torch.cuda.get_rng_state(resolved),
    }


def release_payload(
    *,
    step: int,
    model: DwarfForCausalLM,
    architecture: dict[str, Any],
    tokenizer: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    public_tokenizer = copy.deepcopy(tokenizer)
    public_tokenizer.pop("path", None)
    public_environment = copy.deepcopy(environment)
    gpu = public_environment.get("gpu")
    if isinstance(gpu, dict):
        for private_key in (
            "logical_device",
            "uuid",
            "visible_device_count",
            "total_memory",
        ):
            gpu.pop(private_key, None)
    return {
        "kind": RELEASE_KIND,
        "step": int(step),
        "model": model.state_dict(),
        "architecture": copy.deepcopy(architecture),
        "tokenizer": public_tokenizer,
        "environment": public_environment,
    }


def _stable_reconstruction_architecture(value: dict[str, Any]) -> dict[str, Any]:
    """Return model-defining metadata, excluding execution/source-only drift."""
    if not isinstance(value, dict):
        raise ValueError("checkpoint architecture metadata is invalid")
    topology = value.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("checkpoint architecture topology is invalid")
    required_topology = (
        "layers",
        "global_mixer_layers",
        "ema_packet_layers",
        "base_k_routing_layers",
        "routing_key_source_by_hisa_layer",
        "l17_route_contract",
        "dsqg",
        "hisa",
        "offset_groups",
    )
    return {
        "config": copy.deepcopy(value.get("config")),
        "parameters": value.get("parameters"),
        "trainable_parameters": value.get("trainable_parameters"),
        "topology": {key: copy.deepcopy(topology.get(key)) for key in required_topology},
    }


def reconstruct_model_from_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    allow_source_mismatch: bool = False,
) -> tuple[DwarfForCausalLM, dict[str, Any]]:
    """Strictly reconstruct model weights without resuming private training state."""
    if allow_source_mismatch and expected_sha256 is None:
        raise ValueError("allow_source_mismatch requires expected_sha256")
    checkpoint_path = Path(path).resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") not in {
        CHECKPOINT_KIND,
        CANONICAL_PARENT_CHECKPOINT_KIND,
        RELEASE_KIND,
    }:
        raise ValueError("not a reconstructable canonical DWARF checkpoint")
    saved_architecture = checkpoint.get("architecture")
    if not isinstance(saved_architecture, dict):
        raise ValueError("checkpoint architecture metadata is invalid")
    config_payload = copy.deepcopy(saved_architecture.get("config"))
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint model config is invalid")
    tuple_fields = (
        "global_mixer_layers",
        "ema_packet_layers",
        "base_k_routing_layers",
        "ema_packet_initial_coupling_fractions",
        "ema_timescales",
    )
    for field in tuple_fields:
        if isinstance(config_payload.get(field), list):
            config_payload[field] = tuple(config_payload[field])
    try:
        config = DwarfConfig(**config_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model config is invalid") from error
    model = DwarfForCausalLM(config)
    current_architecture = model_metadata(model)
    if _stable_reconstruction_architecture(saved_architecture) != (
        _stable_reconstruction_architecture(current_architecture)
    ):
        raise ValueError("checkpoint semantic architecture does not match")
    if (
        saved_architecture.get("sources") != current_architecture.get("sources")
        and not allow_source_mismatch
    ):
        raise ValueError(
            "checkpoint source manifest does not match; pass "
            "allow_source_mismatch only after reviewing the source diff"
        )
    saved_model = checkpoint.get("model")
    expected_model = model.state_dict()
    if not isinstance(saved_model, dict) or saved_model.keys() != expected_model.keys():
        raise ValueError("checkpoint model keys do not match")
    for name, expected in expected_model.items():
        saved = saved_model[name]
        if (
            not torch.is_tensor(saved)
            or saved.shape != expected.shape
            or saved.dtype != expected.dtype
        ):
            raise ValueError(f"checkpoint model tensor does not match: {name}")
    model.load_state_dict(saved_model, strict=True)
    model.eval()
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step is invalid")
    receipt = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "kind": checkpoint["kind"],
        "step": step,
        "loaded_tensors": len(saved_model),
        "parameters": current_architecture["parameters"],
        "trainable_parameters": current_architecture["trainable_parameters"],
        "architecture_format": current_architecture["format"],
        "source_mismatch_allowed": bool(allow_source_mismatch),
    }
    return model, receipt


def restore_checkpoint(
    path: str | Path,
    *,
    model: DwarfForCausalLM,
    optimizer: MultiOptimizer,
    architecture: dict[str, Any],
    dataset: dict[str, Any],
    device: torch.device,
    environment: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
    allow_environment_mismatch: bool = False,
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("not a canonical DWARF resumable checkpoint")
    kind = checkpoint.get("kind")
    if kind != CHECKPOINT_KIND:
        raise ValueError("not a canonical DWARF resumable checkpoint")
    validate_checkpoint_architecture(
        checkpoint.get("architecture"),
        architecture,
        allow_source_mismatch=allow_source_mismatch,
    )
    validate_checkpoint_environment(
        checkpoint.get("environment"),
        runtime_environment(device) if environment is None else environment,
        allow_mismatch=allow_environment_mismatch,
    )
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
        if (
            not torch.is_tensor(saved)
            or saved.shape != expected.shape
            or saved.dtype != expected.dtype
        ):
            raise ValueError(f"checkpoint model tensor does not match: {name}")
    resolved = _resolved_cuda_device(device)
    cuda_rng = checkpoint.get("cuda_rng")
    if not torch.is_tensor(cuda_rng) or cuda_rng.dtype != torch.uint8:
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
    torch.cuda.set_rng_state(cuda_rng, resolved)
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


def _capture_rng_state(device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor]:
    resolved = _resolved_cuda_device(device)
    return (
        random.getstate(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(resolved),
    )


def _restore_rng_state(
    state: tuple[Any, torch.Tensor, torch.Tensor], device: torch.device
) -> None:
    python_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, _resolved_cuda_device(device))


def warm_compiled_training_step(
    *,
    model: DwarfForCausalLM,
    compiled: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    config: DwarfConfig,
) -> None:
    """Compile/autotune the real training graph without advancing training RNG."""
    state = _capture_rng_state(device)
    try:
        synthetic = torch.randint(
            5,
            config.vocab_size,
            (RECIPE.batch_size, config.seq_len),
            device=device,
            dtype=torch.long,
        )
        input_ids, labels = synthetic[:, :-1], synthetic[:, 1:]
        route_ids = route_aux_tile_ids_for_update(1, device, config)
        model.zero_grad(set_to_none=True)
        with amp_context():
            hidden, auxiliary = compiled(input_ids, route_ids)
            language_loss = loss_fn(
                model.lm_head.weight, hidden.flatten(0, 1), labels.flatten()
            )
            loss = language_loss + auxiliary
        loss.backward()
        torch.cuda.synchronize(device)
        model.zero_grad(set_to_none=True)
    finally:
        _restore_rng_state(state, device)


@torch.no_grad()
def collect_model_diagnostics(
    *,
    model: DwarfForCausalLM,
    diagnostic: nn.Module,
    input_ids: torch.Tensor,
    route_aux_tile_ids: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], float, int, int]:
    """Run one RNG-neutral eager diagnostic forward and collect layer-qualified data."""
    state = _capture_rng_state(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with amp_context():
            diagnostic(input_ids, route_aux_tile_ids)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        metrics: dict[str, float] = {}
        for layer_index, block in enumerate(model.blocks):
            if isinstance(block, DSQGBlock):
                values = block.attn.routing_diagnostics()
                prefix = f"dsqg_l{layer_index:02d}_"
            elif isinstance(block, GlobalMixerBlock):
                values = block.attn._routing_diagnostics
                prefix = f"hisa_l{layer_index:02d}_"
            else:
                continue
            for key, value in values.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    metrics[prefix + key] = float(value)
            if isinstance(block, GlobalMixerBlock) and block.packet is not None:
                factors = bounded_ema_factor(block.packet.ema_raw.detach())
                mix = block.packet.mix_logits.detach().softmax(-1)
                entropy = -(mix * mix.clamp_min(1e-12).log()).sum(-1).mean()
                metrics[prefix + "ema_factor_fast"] = float(factors[0])
                metrics[prefix + "ema_factor_medium"] = float(factors[1])
                metrics[prefix + "ema_factor_slow"] = float(factors[2])
                metrics[prefix + "ema_mix_entropy"] = float(entropy)
                metrics[prefix + "npci_k_mean"] = float(
                    torch.tanh(block.attn.npci_theta_k.detach()).mean()
                    * block.attn.npci_theta_max
                )
                metrics[prefix + "npci_v_mean"] = float(
                    torch.tanh(block.attn.npci_theta_v.detach()).mean()
                    * block.attn.npci_theta_max
                )
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        return metrics, elapsed, allocated, reserved
    finally:
        _restore_rng_state(state, device)


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
    requested_device = torch.device(args.device)
    stop_step = RECIPE.steps if args.stop_after is None else int(args.stop_after)
    if not 0 < stop_step <= RECIPE.steps:
        raise ValueError(f"--stop-after must be between 1 and {RECIPE.steps}")
    if args.save_every < 0 or args.log_every < 1:
        raise ValueError("invalid save/log interval")

    config = DwarfConfig()
    loss_class, environment = validate_training_runtime(requested_device, config)
    device = _resolved_cuda_device(requested_device)
    compile_policy = configure_compiled_backward_autocast()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    tokenizer = tokenizer_identity(args.tokenizer, config.vocab_size)
    source = ValidatedDatasetSource(
        args.dataset,
        expected_sha256=args.dataset_sha256,
        trust_expected_sha256=args.trust_dataset_sha256,
    )
    stager: BatchStager | None = None
    writer: AsyncCheckpointWriter | None = None
    try:
        dataset = source.load(seq_len=config.seq_len)
        order, selection = build_training_row_order(
            dataset_rows=len(dataset), stop_step=stop_step
        )
        preflight = preflight_training_rows(
            dataset,
            selected_rows=len(order),
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
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
        if _state_fingerprint(model) != EXPECTED_STATE_FINGERPRINT:
            raise RuntimeError("canonical DWARF initial state fingerprint changed")

        optimizer = build_optimizer(model)
        start_step = 0
        if args.resume:
            start_step = restore_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                architecture=architecture,
                dataset=identity,
                device=device,
                environment=environment,
                allow_source_mismatch=args.allow_source_mismatch,
                allow_environment_mismatch=args.allow_environment_mismatch,
            )
        if start_step >= stop_step:
            raise ValueError("checkpoint is already at or beyond --stop-after")

        compiled, diagnostic = compiled_training_callables(model)
        loss_fn = loss_class(accum_dtype=torch.float32)
        stager = BatchStager(
            dataset, batch_size=RECIPE.batch_size, device=device
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = AsyncCheckpointWriter()

        compiled.train()
        diagnostic.train()
        warm_compiled_training_step(
            model=model,
            compiled=compiled,
            loss_fn=loss_fn,
            device=device,
            config=config,
        )

        print(
            json.dumps(
                {
                    "architecture": architecture,
                    "recipe": asdict(RECIPE),
                    "dataset": identity,
                    "dataset_preflight": preflight,
                    "environment": environment,
                    "start_step": start_step,
                    "stop_step": stop_step,
                    "device": str(device),
                    "compiled": True,
                    "compile_policy": compile_policy,
                    "compile_warmup": "complete_rng_neutral",
                    "liger_fused_cross_entropy": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        interval_started = time.perf_counter()
        interval_start_step = start_step
        wall_started = interval_started
        training_compute_elapsed = 0.0

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
            last_input_ids: torch.Tensor | None = None
            for batch in stager.batches(update):
                input_ids, labels = batch[:, :-1], batch[:, 1:]
                last_input_ids = input_ids
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

            logged = step % args.log_every == 0 or step in {1, stop_step}
            if logged:
                if last_input_ids is None:
                    raise RuntimeError("logging update contained no microbatches")
                torch.cuda.synchronize(device)
                training_now = time.perf_counter()
                interval_seconds = training_now - interval_started
                training_compute_elapsed += interval_seconds
                interval_steps = step - interval_start_step
                targets = (
                    interval_steps * RECIPE.effective_batch * config.model_length
                )
                training_peak_allocated = torch.cuda.max_memory_allocated(device)
                training_peak_reserved = torch.cuda.max_memory_reserved(device)
                (
                    diagnostics,
                    diagnostic_seconds,
                    diagnostic_peak_allocated,
                    diagnostic_peak_reserved,
                ) = collect_model_diagnostics(
                    model=model,
                    diagnostic=diagnostic,
                    input_ids=last_input_ids,
                    route_aux_tile_ids=route_aux_tile_ids,
                    device=device,
                )
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
                    "shifted_targets": (
                        step * RECIPE.effective_batch * config.model_length
                    ),
                    "interval_training_seconds": interval_seconds,
                    "shifted_targets_per_second": targets / interval_seconds,
                    "training_compute_elapsed_seconds": training_compute_elapsed,
                    "wall_elapsed_seconds": time.perf_counter() - wall_started,
                    "training_peak_allocated_bytes": training_peak_allocated,
                    "training_peak_reserved_bytes": training_peak_reserved,
                    "diagnostic_seconds": diagnostic_seconds,
                    "diagnostic_peak_allocated_bytes": diagnostic_peak_allocated,
                    "diagnostic_peak_reserved_bytes": diagnostic_peak_reserved,
                }
                event.update(diagnostics)
                print(json.dumps(event, sort_keys=True), flush=True)

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
                        device=device,
                        environment=environment,
                    ),
                    output_dir / f"dwarf_step_{step:07d}.pt",
                    producer_streams={
                        device: torch.cuda.current_stream(device)
                    },
                )

            if logged:
                torch.cuda.reset_peak_memory_stats(device)
                interval_started = time.perf_counter()
                interval_start_step = step

        if args.export_weights:
            source.assert_unchanged()
            writer.submit(
                release_payload(
                    step=stop_step,
                    model=model,
                    architecture=architecture,
                    tokenizer=tokenizer,
                    environment=environment,
                ),
                Path(args.export_weights),
                producer_streams={device: torch.cuda.current_stream(device)},
            )
    finally:
        try:
            if stager is not None:
                stager.close()
        finally:
            try:
                if writer is not None:
                    writer.close()
                source.assert_unchanged()
            finally:
                source.close()


def self_test() -> None:
    """Run deterministic CPU release checks for public model and kernel contracts."""
    assert_public_kernel_contracts()
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1234)
    model = DwarfForCausalLM()
    seeded_rng_fingerprint = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()
    torch.random.set_rng_state(rng_state)
    metadata = model_metadata(model)
    _require(
        metadata["parameters"] == EXPECTED_PARAMETERS,
        "canonical DWARF parameter count changed",
    )
    _require(
        metadata["trainable_parameters"] == EXPECTED_TRAINABLE_PARAMETERS,
        "canonical DWARF trainable parameter count changed",
    )
    _require(
        _state_fingerprint(model) == EXPECTED_STATE_FINGERPRINT,
        "canonical DWARF seeded state fingerprint changed",
    )
    _require(
        seeded_rng_fingerprint == EXPECTED_SEEDED_RNG_FINGERPRINT,
        "canonical DWARF seeded RNG lineage changed",
    )
    _require(len(model.blocks) == 24, "L24 DWARF layer count changed")
    global_mixers = [
        block for block in model.blocks if isinstance(block, GlobalMixerBlock)
    ]
    _require(
        [
            index
            for index, block in enumerate(model.blocks)
            if isinstance(block, GlobalMixerBlock)
        ]
        == [3, 10, 17],
        "L24 HISA layer placement changed",
    )
    packet_layers = [
        index
        for index, block in enumerate(model.blocks)
        if isinstance(block, GlobalMixerBlock) and block.packet is not None
    ]
    _require(packet_layers == [3, 17], "dual internal EMA packet placement changed")
    _require(
        [block.attn.route_source for block in global_mixers]
        == ["rotated_global_k", "rotated_global_k", "base_global_k"],
        "L3/L10/L17 routing-key source contract changed",
    )
    _require(
        RECIPE.steps == 76_331
        and RECIPE.effective_batch == 128
        and RECIPE.warmup_steps == 1_527
        and RECIPE.stable_steps == 74_804
        and RECIPE.decay_steps == 0
        and {1_909, 3_817, 5_725, 7_633}.issubset(RECIPE.checkpoint_steps),
        "canonical 20B stable-LR trunk schedule contract changed",
    )
    _require(global_mixers[1].packet is None, "L10 HISA gained an EMA packet")
    _require(
        global_mixers[0].packet is not global_mixers[2].packet,
        "L3 and L17 EMA packets became weight-tied",
    )
    _require(
        global_mixers[0].attn.npci_theta_k.requires_grad
        and global_mixers[2].attn.npci_theta_k.requires_grad,
        "packet-enabled NPCI parameter is frozen",
    )
    _require(
        not global_mixers[1].attn.npci_theta_k.requires_grad,
        "packet-disabled L10 NPCI parameter is trainable",
    )
    _require_close(
        torch.tanh(global_mixers[0].attn.npci_theta_k),
        torch.full_like(global_mixers[0].attn.npci_theta_k, 0.04),
        atol=1e-7,
        rtol=0.0,
        message="L3 initial NPCI coupling changed",
    )
    _require_close(
        torch.tanh(global_mixers[2].attn.npci_theta_k),
        torch.full_like(global_mixers[2].attn.npci_theta_k, 0.012),
        atol=1e-7,
        rtol=0.0,
        message="L17 low initial NPCI coupling changed",
    )
    for global_mixer in global_mixers:
        attention = global_mixer.attn
        _require(
            attention.chunk_selection_scope == "token",
            "canonical HISA routing scope changed",
        )
        _require(
            attention.token_routing_pack_size == 4,
            "canonical HISA routing pack changed",
        )
        _require(
            attention.route_aux_weight == model.config.hisa_route_aux_weight,
            "canonical HISA auxiliary weight changed",
        )
        _require(
            attention.exploration_probability
            == model.config.hisa_exploration_probability,
            "canonical HISA exploration probability changed",
        )
        _require(
            attention.global_adapter_rank == model.config.hisa_global_adapter_rank,
            "canonical HISA global adapter rank changed",
        )
        _require(
            attention.binding_rank == model.config.hisa_binding_rank,
            "canonical HISA binding rank changed",
        )
        _require(
            attention.backend == model.config.hisa_backend,
            "canonical HISA backend changed",
        )
        _require(
            attention.token_selection_mode == model.config.hisa_token_selection_mode,
            "canonical HISA token selection mode changed",
        )
        _require(
            attention.local_backend == model.config.hisa_local_backend,
            "canonical HISA local backend changed",
        )
        _require(
            attention.triton_block_q == model.config.hisa_triton_block_q,
            "canonical HISA Triton geometry changed",
        )
        _require(
            attention.backward_impl == model.config.hisa_backward_impl,
            "canonical HISA backward implementation changed",
        )
        _require(
            attention.global_k_down is not None,
            "canonical HISA global adapter is absent",
        )
    _require(
        sum(isinstance(block, DSQGBlock) for block in model.blocks) == 21,
        "L24 DSQG layer count changed",
    )
    dsqg_layers = [
        block.attn for block in model.blocks if isinstance(block, DSQGBlock)
    ]
    _require(
        all(module.backend == model.config.dsqg_backend for module in dsqg_layers),
        "canonical DSQG backend is not explicit",
    )
    _require(
        all(not module.support_crop_projections for module in dsqg_layers),
        "dead canonical DSQG projection cropping is still enabled",
    )
    _require(
        not any(
            token in name
            for name, _ in model.named_parameters()
            for token in (
                "phase_base",
                "phase_gain",
                "phase_gate",
                "query_probes",
                "key_probes",
            )
        ),
        "retired parameter families reappeared",
    )
    groups = make_parameter_groups(model, RECIPE)
    _require(
        [item["name"] for item in groups["muon"]]
        == ["muon_linear_weights"],
        "Muon optimizer partition changed",
    )
    _require(
        [item["name"] for item in groups["adamw"]]
        == [
            "adam_decay",
            "adam_no_decay",
            "adam_scale_embed",
            "adam_null",
            "adam_npci",
            "adam_route",
            "adam_ema",
            "adam_positional",
        ],
        "AdamW optimizer partition changed",
    )
    _require(
        metadata["complexity"]["selector"]
        == "quadratic_in_sequence_length_at_fixed_chunk_size",
        "release metadata overstates HISA selector complexity",
    )
    _require(
        metadata["cache"]["model_level_o1_kv_cache"] is False,
        "release metadata overstates model-level O(1) KV caching",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "parameters": metadata["parameters"],
                "trainable_parameters": metadata["trainable_parameters"],
                "state_fingerprint": EXPECTED_STATE_FINGERPRINT,
                "sources": metadata["sources"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--tokenizer",
        default=str(
            SCRIPT_DIR.parent / "tokenizers" / "dwarf_bpe_v32768_tokenizer.json"
        ),
    )
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--trust-dataset-sha256", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="allow resume after an explicitly reviewed source-only change",
    )
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="allow resume after an explicitly reviewed runtime ABI change",
    )

    parser.add_argument(
        "--export-weights",
        help="write a weights-only release artifact after the requested final step",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if not args.self_test and (not args.dataset or not args.output_dir):
        parser.error("--dataset and --output-dir are required for training")
    if args.trust_dataset_sha256 and not args.dataset_sha256:
        parser.error("--trust-dataset-sha256 requires --dataset-sha256")
    if args.allow_source_mismatch and not args.resume:
        parser.error("--allow-source-mismatch requires --resume")
    if args.allow_environment_mismatch and not args.resume:
        parser.error("--allow-environment-mismatch requires --resume")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        train(arguments)
