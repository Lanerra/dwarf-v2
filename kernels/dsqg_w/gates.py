from __future__ import annotations

import os

import torch


def _forced_gate_value(env_name: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    raw = os.getenv(env_name, "").strip()
    if raw == "":
        return None
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{env_name} must be in [0, 1], got {value}")
    return torch.tensor(value, device=device, dtype=dtype)

__all__ = ["_forced_gate_value"]
