#!/usr/bin/env python3
"""Checkpoint-only causal and multiscale diagnostics for DWARF.

This tool never writes model/checkpoint state and never invokes the trainer.  It
reconstructs a checkpoint on a separately selected CUDA device, builds a
natural-text probe from cached LAMBADA contexts, and reports:

* full-model prefix invariance under suffix perturbation;
* DSQG selector distance-band use, effective slot counts, and learned scale /
  positional parameter structure;
* HISA's actual chunk/token selection metadata; and
* an offline, left-looking Daubechies-2 analysis-energy profile of projected
  K/V trajectories.

The Daubechies profile is a compressibility/scale diagnostic only.  It is not
an implementation of a wavelet memory layer and cannot establish a quality
win without a matched training experiment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
D4_LOW = (0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604)
D4_HIGH = (-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341)
DISTANCE_BANDS = (("local_1_28", 1, 28), ("near_29_127", 29, 127),
                  ("mid_128_511", 128, 511), ("far_512_plus", 512, 1 << 30))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _band_mass(offsets: list[int], mass: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, lo, hi in DISTANCE_BANDS:
        indices = [i for i, offset in enumerate(offsets) if lo <= offset <= hi]
        result[label] = float(mass[indices].sum().item()) if indices else 0.0
    return result


def _d4_energy_profile(values: torch.Tensor, max_levels: int = 8) -> list[dict[str, float | int]]:
    """Offline left-looking db2 analysis-energy profile over the time axis.

    The analysis filters use only present/past samples through left padding.
    This proves neither perfect reconstruction nor a usable streaming memory;
    it only measures how much projected K/V energy resides at each dyadic
    temporal detail scale.
    """
    if values.ndim != 4:
        raise ValueError(f"expected [B,H,N,D], got {tuple(values.shape)}")
    b, h, n, d = values.shape
    signal = values.detach().float().permute(0, 1, 3, 2).reshape(b * h * d, 1, n)
    low_filter = torch.tensor(D4_LOW, device=signal.device, dtype=signal.dtype).reshape(1, 1, -1)
    high_filter = torch.tensor(D4_HIGH, device=signal.device, dtype=signal.dtype).reshape(1, 1, -1)
    profile: list[dict[str, float | int]] = []
    for level in range(max_levels):
        if signal.shape[-1] < 4:
            break
        padded = F.pad(signal, (3, 0))
        low = F.conv1d(padded, low_filter, stride=2)
        high = F.conv1d(padded, high_filter, stride=2)
        low_energy = float(low.square().mean().item())
        high_energy = float(high.square().mean().item())
        profile.append({
            "level": level + 1,
            "input_length": int(signal.shape[-1]),
            "low_energy": low_energy,
            "detail_energy": high_energy,
            "detail_fraction": high_energy / max(low_energy + high_energy, 1e-20),
        })
        signal = low
    return profile


def _as_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, d = tensor.shape
    head_dim = d // num_heads
    return tensor.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _dsqg_selector_metrics(qkv: torch.Tensor, attention) -> dict[str, Any]:
    """Reconstruct pre-MOVT DSQG selector probabilities from captured QKV."""
    b, n, three_d = qkv.shape
    d = three_d // 3
    h = int(attention.num_heads)
    head_dim = d // h
    q_flat, k_flat, v_flat = qkv.split(d, dim=-1)
    q = _as_heads(q_flat, h).float()
    k = _as_heads(k_flat, h).float()
    v = _as_heads(v_flat, h)
    offsets = [int(value) for value in attention.offsets_dev.detach().cpu().tolist()]
    j = len(offsets)
    positions = torch.arange(n, device=q.device, dtype=torch.long).reshape(n, 1)
    offset_tensor = torch.tensor(offsets, device=q.device, dtype=torch.long).reshape(1, j)
    raw_idx = positions - offset_tensor
    valid = raw_idx >= 0
    safe_idx = raw_idx.clamp_min(0)
    k_gather = k[:, :, safe_idx, :]
    scores = torch.einsum("bhnd,bhnjd->bhnj", q, k_gather) / math.sqrt(float(head_dim))
    scores = scores + torch.einsum("bhnd,jd->bhnj", q, attention.scale_embed.float()) / math.sqrt(float(head_dim))
    pos_bias = (attention.pos_bias.float() * attention.pos_bias_scale.float()).transpose(0, 1)
    scores = scores + pos_bias.reshape(1, h, 1, j)
    scores = scores.masked_fill(~valid.reshape(1, 1, n, j), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    active_rows = valid.any(dim=-1)
    if not bool(active_rows.any()):
        raise ValueError("no query rows have a valid DSQG offset")
    # Some long-offset groups include an offset at or beyond the finite probe
    # length, so no row can expose every slot.  Measure the real selector over
    # all rows with at least one causal candidate and record full-slot coverage
    # separately rather than rejecting a legitimate finite-context path.
    active_probs = probs[:, :, active_rows, :]
    mean_prob = active_probs.mean(dim=(0, 2))
    entropy = -(active_probs.clamp_min(1e-20) * active_probs.clamp_min(1e-20).log()).sum(dim=-1)
    effective_slots = entropy.exp()
    scale_norm = attention.scale_embed.detach().float().norm(dim=-1)
    return {
        "seq_len": n,
        "active_query_rows": int(active_rows.sum().item()),
        "fully_available_query_rows": int(valid.all(dim=-1).sum().item()),
        "offsets": offsets,
        "mean_probability_per_head": mean_prob.detach().cpu().tolist(),
        "mean_probability": mean_prob.mean(dim=0).detach().cpu().tolist(),
        "mean_probability_distance_bands": _band_mass(offsets, mean_prob.mean(dim=0)),
        "per_head_distance_bands": [_band_mass(offsets, mean_prob[head]) for head in range(h)],
        "mean_selector_entropy": float(entropy.mean().item()),
        "mean_effective_slots": float(effective_slots.mean().item()),
        "per_head_effective_slots": effective_slots.mean(dim=(0, 2)).detach().cpu().tolist(),
        "scale_embed_norm_per_offset": scale_norm.detach().cpu().tolist(),
        "position_bias_per_head": pos_bias.detach().cpu().tolist(),
        "wavelet_energy_k": _d4_energy_profile(k),
        "wavelet_energy_v": _d4_energy_profile(v),
    }


def _hisa_summary(attention, metadata: Any | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "routing_entropy": float(torch.as_tensor(attention._routing_entropy).item()),
        "backend": str(getattr(attention, "backend", "legacy")),
        "num_chunks": int(attention.num_chunks),
        "top_k_chunks": int(attention.top_k_chunks),
        "top_m_tokens": int(attention.hisa_top_m_tokens),
    }
    if hasattr(attention, "stage2_rep_r"):
        result["stage2_rep_r"] = int(attention.stage2_rep_r)
    if metadata is not None:
        top_chunks = metadata.top_chunk_idx
        token_idx = metadata.token_idx
        valid_chunks = top_chunks[(top_chunks >= 0) & (top_chunks < attention.num_chunks)]
        if valid_chunks.numel() > 0:
            chunk_counts = torch.bincount(valid_chunks.reshape(-1), minlength=attention.num_chunks).float()
            chunk_mass = chunk_counts / chunk_counts.sum().clamp_min(1.0)
            chunk_entropy = -(chunk_mass.clamp_min(1e-20) * chunk_mass.clamp_min(1e-20).log()).sum()
            result["selected_chunk_mass"] = chunk_mass.detach().cpu().tolist()
            result["selected_chunk_effective_count"] = float(chunk_entropy.exp().item())
    else:
        top_chunks = attention._last_top_k_packed
        token_idx = attention._last_token_idx_packed
        result["stage2_selected_fraction"] = float(torch.as_tensor(attention._stage2_selected_fraction).item())
        result["chunk_size"] = int(attention._last_chunk_size)
    result["top_chunk_tensor_shape"] = list(top_chunks.shape)
    result["token_index_tensor_shape"] = list(token_idx.shape)
    valid_tokens = token_idx[token_idx >= 0]
    result["selected_token_count"] = int(valid_tokens.numel())
    if valid_tokens.numel() > 0:
        result["selected_token_min"] = int(valid_tokens.min().item())
        result["selected_token_max"] = int(valid_tokens.max().item())
    if metadata is None and token_idx.ndim == 5:
        # [B,H,query_chunk,top_k_chunk,top_m_token].  Compare selected tokens
        # with the final row of the corresponding query chunk: conservative,
        # causal chunk-level distance accounting rather than per-row claims.
        _, _, chunk_count, _, _ = token_idx.shape
        chunk_size = int(attention._last_chunk_size)
        query_ends = (torch.arange(chunk_count, device=token_idx.device) + 1) * chunk_size - 1
        distances = query_ends.reshape(1, 1, chunk_count, 1, 1) - token_idx
        valid = token_idx >= 0
        valid_distances = distances[valid]
        result["chunk_end_distance_bands"] = {
            label: int(((valid_distances >= lo) & (valid_distances <= hi)).sum().item())
            for label, lo, hi in DISTANCE_BANDS
        }
    return result


def _build_probe_ids(tokenizer, examples: list[dict[str, str]], *, seq_len: int) -> torch.Tensor:
    eod = tokenizer.tokenizer.token_to_id("<|endoftext|>")
    if eod is None:
        eod = tokenizer.tokenizer.token_to_id("<eos>")
    if eod is None:
        raise ValueError("tokenizer has no recognized EOD token")
    ids: list[int] = []
    for example in examples:
        ids.extend(tokenizer.encode(example["context"]))
        ids.append(int(eod))
        if len(ids) >= seq_len:
            break
    if len(ids) < seq_len:
        raise ValueError(f"only collected {len(ids)} natural-text probe tokens, expected {seq_len}")
    return torch.tensor([ids[:seq_len]], dtype=torch.long)


def _prefix_invariance(model, probe_ids: torch.Tensor, vocab_size: int, pivots: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for pivot in pivots:
            baseline = model(probe_ids)
            baseline_prefix = baseline[:, :pivot, :].float().cpu()
            del baseline
            altered = probe_ids.clone()
            altered[:, pivot:] = torch.randint(0, vocab_size, altered[:, pivot:].shape, device=altered.device)
            changed = model(altered)
            changed_prefix = changed[:, :pivot, :].float().cpu()
            delta = (baseline_prefix - changed_prefix).abs()
            results.append({
                "pivot": pivot,
                "max_abs_logit_delta": float(delta.max().item()),
                "mean_abs_logit_delta": float(delta.mean().item()),
                "exact_equal": bool(torch.equal(baseline_prefix, changed_prefix)),
            })
            del altered, changed, baseline_prefix, changed_prefix, delta
            torch.cuda.empty_cache()
    return results


def run(args: argparse.Namespace) -> Path:
    eval_script = _load_module(ROOT / "scripts" / "run_dolma3_20b_external_trio.py", "dwarf_external_trio")
    contract = Path(args.run_contract).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    arch = eval_script.arch_config_from_contract(contract)
    eval_external = eval_script.setup_legacy_imports()
    arch_name = f"wavelet_probe_{checkpoint.stem}".replace("-", "_")
    eval_external.ARCH_CONFIGS[arch_name] = arch
    eval_external.MAX_SEQ_LEN = int(args.seq_len)
    eval_external.TOKENIZER = arch["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("this diagnostic is intended for the separate CUDA evaluation lane")
    tokenizer = eval_external.load_tokenizer()
    examples = json.loads((Path(eval_external.CACHE_DIR) / "lambada.json").read_text(encoding="utf-8"))
    probe_ids = _build_probe_ids(tokenizer, examples[: args.contexts], seq_len=args.seq_len).to(device)
    model = eval_external.load_model_from_arch(arch_name, str(checkpoint), device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    captures: dict[str, torch.Tensor] = {}
    hooks = []
    dsqg_modules: list[tuple[int, Any]] = []
    hisa_module = None
    for layer_index, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if hasattr(attn, "qkv_proj"):
            dsqg_modules.append((layer_index, attn))
            hooks.append(attn.qkv_proj.register_forward_hook(
                lambda _module, _inputs, output, i=layer_index: captures.__setitem__(f"dsqg_qkv_{i}", output.detach())
            ))
        if hasattr(attn, "W_k") and hasattr(attn, "W_v"):
            hisa_module = attn
            hooks.append(attn.register_forward_pre_hook(
                lambda _module, inputs: captures.__setitem__("hisa_input", inputs[0].detach())
            ))
            hooks.append(attn.W_k.register_forward_hook(
                lambda _module, _inputs, output: captures.__setitem__("hisa_k", output.detach())
            ))
            hooks.append(attn.W_v.register_forward_hook(
                lambda _module, _inputs, output: captures.__setitem__("hisa_v", output.detach())
            ))
    try:
        with torch.no_grad():
            _ = model(probe_ids)
        dsqg = {
            f"layer_{layer_index}": _dsqg_selector_metrics(captures[f"dsqg_qkv_{layer_index}"], attn)
            for layer_index, attn in dsqg_modules
        }
        if hisa_module is None:
            raise RuntimeError("could not locate HISA block")
        hisa_k = _as_heads(captures["hisa_k"], int(hisa_module.H))
        hisa_v = _as_heads(captures["hisa_v"], int(hisa_module.H))
        hisa_metadata = None
        if hasattr(hisa_module, "backend"):
            _, hisa_metadata = hisa_module(captures["hisa_input"], return_metadata=True)
        hisa = _hisa_summary(hisa_module, hisa_metadata)
        hisa["wavelet_energy_k"] = _d4_energy_profile(hisa_k)
        hisa["wavelet_energy_v"] = _d4_energy_profile(hisa_v)
        prefix = _prefix_invariance(model, probe_ids, int(arch["eval_env"]["DWARF_VOCAB_SIZE"]), args.pivots)
    finally:
        for hook in hooks:
            hook.remove()

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "tool": "wavelet_checkpoint_diagnostics",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(checkpoint_payload["global_step"]),
        "run_contract": str(contract),
        "device": torch.cuda.get_device_name(0),
        "parameter_count": parameter_count,
        "probe": {
            "source": "cached_lambada_contexts_concatenated_with_native_eod",
            "contexts": args.contexts,
            "seq_len": args.seq_len,
            "pivots": args.pivots,
        },
        "prefix_invariance": prefix,
        "dsqg": dsqg,
        "hisa": hisa,
        "interpretation_limits": [
            "Prefix invariance validates the current checkpoint path, not a proposed wavelet module.",
            "Daubechies-2 detail energies are offline activation diagnostics, not a causal perfect-reconstruction implementation.",
            "Selector concentration and activation multiscale structure cannot establish PPL or semantic-transfer gains from a wavelet architecture.",
            "The probe concatenates natural LAMBADA contexts only to exercise a 2048-token path; it is not an IID held-out language-model metric.",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-contract", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--contexts", type=int, default=128)
    parser.add_argument("--pivots", type=int, nargs="+", default=[64, 512, 1024])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
