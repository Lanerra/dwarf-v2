#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py"


def load_trainer(*, dsqg_w: bool, suffix: str):
    os.environ["DWARF_DISABLE_BNB"] = "1"
    os.environ["DWARF_LIGER"] = "0"
    os.environ["DWARF_TORCH_COMPILE"] = "0"
    os.environ["DWARF_Q6_G128"] = "0"
    os.environ["DWARF_DSQG_W_MAX_CANDIDATES"] = os.environ.get("DWARF_DSQG_W_MAX_CANDIDATES", "16")
    os.environ["DWARF_DSQG_W_BOTTLENECK"] = os.environ.get("DWARF_DSQG_W_BOTTLENECK", "64")
    if dsqg_w:
        os.environ["DWARF_DSQG_W"] = "1"
    else:
        os.environ.pop("DWARF_DSQG_W", None)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kernels"))
    try:
        spec = importlib.util.spec_from_file_location(f"dwarf_v2_trainer_{suffix}", TRAINER)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        for path in [str(ROOT / "kernels"), str(ROOT)]:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def make_model(mod, *, seed: int, vocab_size: int, ffn_dim: int, seq_len: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = mod.TriadicJ96Dsr(
        vocab_size=vocab_size,
        embedding_dim=mod.EMBEDDING_DIM,
        num_heads=mod.NUM_HEADS,
        ffn_dim=ffn_dim,
        seq_len=seq_len,
        dsr_layer=mod.DSR_LAYER,
        dropout=0.0,
        num_chunks=mod.NUM_CHUNKS,
        top_k_chunks=mod.TOP_K_CHUNKS,
    ).to(device).eval()
    return model


def tensor_stats(delta: torch.Tensor) -> dict[str, float]:
    d = delta.detach().float()
    return {
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rms": float(d.square().mean().sqrt().item()),
    }


def main() -> int:
    seed = int(os.environ.get("DWARF_DSQG_W_AUDIT_SEED", "20260628"))
    batch = int(os.environ.get("DWARF_DSQG_W_AUDIT_B", "1"))
    seq_len = int(os.environ.get("DWARF_DSQG_W_AUDIT_T", "64"))
    vocab_size = int(os.environ.get("DWARF_DSQG_W_AUDIT_VOCAB", "128"))
    ffn_dim = int(os.environ.get("DWARF_DSQG_W_AUDIT_FFN", "64"))
    device = torch.device(os.environ.get("DWARF_DSQG_W_AUDIT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))

    base_mod = load_trainer(dsqg_w=False, suffix="base")
    w_mod = load_trainer(dsqg_w=True, suffix="dsqgw")

    base = make_model(base_mod, seed=seed, vocab_size=vocab_size, ffn_dim=ffn_dim, seq_len=seq_len, device=device)
    with_w = make_model(w_mod, seed=seed, vocab_size=vocab_size, ffn_dim=ffn_dim, seq_len=seq_len, device=device)

    torch.manual_seed(seed + 1)
    idx = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)

    with torch.no_grad():
        base_trunk = base._forward_trunk(idx)
        w_trunk = with_w._forward_trunk(idx)
        w_recomposed = with_w._apply_dsqg_w_recomposer(w_trunk)
        base_logits = base.out(base.norm(base_trunk))
        w_logits = with_w.out(with_w.norm(w_recomposed))
        base_logp = F.log_softmax(base_logits.float(), dim=-1)
        w_logp = F.log_softmax(w_logits.float(), dim=-1)
        kl_w_to_base = F.kl_div(w_logp, base_logp.exp(), reduction="batchmean", log_target=False)
        kl_base_to_w = F.kl_div(base_logp, w_logp.exp(), reduction="batchmean", log_target=False)

    telemetry: dict[str, Any] = {}
    for key, value in with_w.dsqg_w_last_telemetry.items():
        if torch.is_tensor(value) and value.numel() == 1:
            telemetry[key] = float(value.detach().float().cpu().item())

    report = {
        "config": {
            "seed": seed,
            "batch": batch,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "ffn_dim": ffn_dim,
            "device": str(device),
            "dsqg_w_max_candidates": with_w.dsqg_w_config.max_candidates,
            "dsqg_w_bottleneck": with_w.dsqg_w_config.bottleneck,
            "dsqg_w_gate_init": with_w.dsqg_w_config.gate_init,
            "candidate_path": "LOCAL_LONG_NULL_ONLY",
        },
        "trunk_delta": tensor_stats(w_trunk - base_trunk),
        "recomposer_hidden_delta": tensor_stats(w_recomposed - w_trunk),
        "logit_delta": tensor_stats(w_logits - base_logits),
        "kl": {
            "w_to_base_batchmean": float(kl_w_to_base.detach().cpu().item()),
            "base_to_w_batchmean": float(kl_base_to_w.detach().cpu().item()),
        },
        "telemetry": telemetry,
        "thresholds": {
            "trunk_max_abs_required": 1e-6,
            "logit_max_abs_warn": 1e-2,
            "kl_warn": 1e-4,
        },
    }
    report["pass"] = bool(
        report["trunk_delta"]["max_abs"] <= report["thresholds"]["trunk_max_abs_required"]
        and report["logit_delta"]["max_abs"] <= report["thresholds"]["logit_max_abs_warn"]
        and report["kl"]["w_to_base_batchmean"] <= report["thresholds"]["kl_warn"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
