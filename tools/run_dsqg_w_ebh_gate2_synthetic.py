#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.dsqg_w.dsqg_w_mvp import CandidateSource, CandidateType, DSQGWEvidenceBindingHub


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class EBHSyntheticClassifier(nn.Module):
    def __init__(self, *, d: int, n_classes: int, bottleneck: int, gate_init: float) -> None:
        super().__init__()
        self.ebh = DSQGWEvidenceBindingHub(
            d=d,
            n_types=len(CandidateType),
            n_sources=len(CandidateSource),
            bottleneck=bottleneck,
            gate_init=gate_init,
            phase_bands=4,
            use_score_features=True,
        )
        self.classifier = nn.Linear(d, n_classes)

    def forward(self, x, cand_states, cand_types, cand_sources, cand_mask, distances, scores, *, return_aux: bool = False):
        y, telemetry, aux = self.ebh(
            x,
            cand_states,
            cand_types,
            cand_sources,
            cand_mask,
            candidate_distances=distances,
            cand_scores=scores,
            return_aux=True,
        )
        logits = self.classifier(y[:, 0, :])
        if return_aux:
            return logits, telemetry, aux
        return logits, telemetry


def _make_prototypes(n_classes: int, d: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(720)
    raw = torch.randn(n_classes, d, device=device)
    return F.normalize(raw, dim=-1) * 3.0


def make_batch(
    *,
    batch: int,
    k: int,
    d: int,
    n_classes: int,
    prototypes: torch.Tensor,
    device: torch.device,
    noise: float = 0.05,
    null: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, n_classes, (batch,), device=device)
    x = torch.zeros(batch, 1, d, device=device)
    x = x + noise * torch.randn_like(x)
    cand_states = torch.zeros(batch, 1, k, d, device=device)
    cand_types = torch.full((batch, 1, k), int(CandidateType.LOCAL), device=device, dtype=torch.long)
    cand_sources = torch.full((batch, 1, k), int(CandidateSource.FINAL), device=device, dtype=torch.long)
    cand_mask = torch.ones(batch, 1, k, device=device, dtype=torch.bool)
    distances = torch.arange(1, k + 1, device=device, dtype=torch.float32).reshape(1, 1, k).expand(batch, 1, k).clone()
    scores = torch.zeros(batch, 1, k, device=device)
    if null:
        cand_types.fill_(int(CandidateType.NULL))
        cand_sources.fill_(int(CandidateSource.NULL))
        cand_mask.zero_()
        return x, cand_states, cand_types, cand_sources, cand_mask, distances, scores, labels

    for b in range(batch):
        label = int(labels[b].item())
        correct_slot = int(torch.randint(0, k, (1,), device=device).item())
        wrong_classes = [(label + i + 1) % n_classes for i in range(k)]
        for j in range(k):
            cls = label if j == correct_slot else wrong_classes[j]
            cand_states[b, 0, j] = prototypes[cls] + noise * torch.randn(d, device=device)
            if j == correct_slot:
                cand_types[b, 0, j] = int(CandidateType.HISA_EVIDENCE)
                cand_sources[b, 0, j] = int(CandidateSource.HISA)
                scores[b, 0, j] = 2.0
            elif j % 3 == 0:
                cand_types[b, 0, j] = int(CandidateType.QUESTION)
                cand_sources[b, 0, j] = int(CandidateSource.FINAL)
                scores[b, 0, j] = -0.25
            elif j % 3 == 1:
                cand_types[b, 0, j] = int(CandidateType.L3_SKIP)
                cand_sources[b, 0, j] = int(CandidateSource.L3)
                scores[b, 0, j] = -0.5
            else:
                cand_types[b, 0, j] = int(CandidateType.CHUNK_REP)
                cand_sources[b, 0, j] = int(CandidateSource.SUMMARY)
                scores[b, 0, j] = -0.75
    return x, cand_states, cand_types, cand_sources, cand_mask, distances, scores, labels


def evaluate(model: EBHSyntheticClassifier, *, prototypes, batch: int, k: int, d: int, n_classes: int, device: torch.device, batches: int, mode: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_conf = 0.0
    total_gate = 0.0
    total_packet_delta = 0.0
    with torch.no_grad():
        for _ in range(batches):
            null = mode == "null"
            x, c, t, s, m, dist, scores, labels = make_batch(
                batch=batch, k=k, d=d, n_classes=n_classes, prototypes=prototypes, device=device, null=null
            )
            if mode == "shuffled":
                row_perm = torch.randperm(batch, device=device)
                c = c[row_perm]
                t = t[row_perm]
                s = s[row_perm]
                scores = scores[row_perm]
            logits, telemetry, aux = model(x, c, t, s, m, dist, scores, return_aux=True)
            loss = F.cross_entropy(logits, labels)
            probs = logits.softmax(dim=-1)
            total_loss += float(loss.item())
            total_acc += float((logits.argmax(dim=-1) == labels).float().mean().item())
            total_conf += float(probs.max(dim=-1).values.mean().item())
            total_gate += float(telemetry["dsqg_w_ebh_bind_gate_mean"].item())
            if mode == "shuffled":
                # Packet is already wrong-evidence packet here; track magnitude as a sanity signal.
                total_packet_delta += float(aux["bound_packet"].float().norm(dim=-1).mean().item())
    denom = float(max(1, batches))
    return {
        f"{mode}_loss": total_loss / denom,
        f"{mode}_acc": total_acc / denom,
        f"{mode}_max_conf": total_conf / denom,
        f"{mode}_bind_gate_mean": total_gate / denom,
        f"{mode}_packet_norm": total_packet_delta / denom if mode == "shuffled" else 0.0,
    }


def train_baseline(*, prototypes, batch: int, d: int, n_classes: int, device: torch.device, steps: int, lr: float) -> dict[str, float]:
    # No-evidence baseline: local x is label-independent noise, so this should stay near chance.
    clf = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    final_loss = 0.0
    final_acc = 0.0
    for _ in range(steps):
        x, *_rest, labels = make_batch(batch=batch, k=4, d=d, n_classes=n_classes, prototypes=prototypes, device=device)
        logits = clf(x[:, 0, :])
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        final_acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
    return {"baseline_final_loss": final_loss, "baseline_final_acc": final_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-2 synthetic semantic binding task for DSQG-W EBH")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gate-init", type=float, default=-1.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(721)
    prototypes = _make_prototypes(args.n_classes, args.d_model, device)
    model = EBHSyntheticClassifier(d=args.d_model, n_classes=args.n_classes, bottleneck=args.bottleneck, gate_init=args.gate_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        x, c, t, s, m, dist, scores, labels = make_batch(
            batch=args.batch, k=args.k, d=args.d_model, n_classes=args.n_classes, prototypes=prototypes, device=device
        )
        logits, telemetry, aux = model(x, c, t, s, m, dist, scores, return_aux=True)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "acc": float(acc),
                "bind_gate_mean": float(telemetry["dsqg_w_ebh_bind_gate_mean"].item()),
                "candidate_state_grad_norm": float(c.grad.float().norm().item()) if c.requires_grad and c.grad is not None else 0.0,
            })

    eval_correct = evaluate(model, prototypes=prototypes, batch=args.batch, k=args.k, d=args.d_model, n_classes=args.n_classes, device=device, batches=args.eval_batches, mode="correct")
    eval_shuffled = evaluate(model, prototypes=prototypes, batch=args.batch, k=args.k, d=args.d_model, n_classes=args.n_classes, device=device, batches=args.eval_batches, mode="shuffled")
    eval_null = evaluate(model, prototypes=prototypes, batch=args.batch, k=args.k, d=args.d_model, n_classes=args.n_classes, device=device, batches=args.eval_batches, mode="null")
    baseline = train_baseline(prototypes=prototypes, batch=args.batch, d=args.d_model, n_classes=args.n_classes, device=device, steps=max(50, args.steps // 4), lr=args.lr)

    x, c, t, s, m, dist, scores, labels = make_batch(
        batch=args.batch, k=args.k, d=args.d_model, n_classes=args.n_classes, prototypes=prototypes, device=device
    )
    c = c.detach().clone().requires_grad_(True)
    logits, telemetry, aux = model(x, c, t, s, m, dist, scores, return_aux=True)
    probe_loss = F.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    probe_loss.backward()
    grad_norms = {
        "candidate_states": float(c.grad.detach().float().norm().item()),
        "value_proj": float(model.ebh.value_proj.weight.grad.detach().float().norm().item()),
        "read_mix": float(model.ebh.read_mix.weight.grad.detach().float().norm().item()),
        "delta_proj": float(model.ebh.delta_proj[3].weight.grad.detach().float().norm().item()),
        "bind_gate": float(model.ebh.bind_gate.weight.grad.detach().float().norm().item()),
    }

    pass_gate = (
        eval_correct["correct_acc"] >= 0.85
        and eval_shuffled["shuffled_acc"] <= eval_correct["correct_acc"] - 0.25
        and eval_shuffled["shuffled_loss"] >= eval_correct["correct_loss"] + 0.2
        and eval_null["null_max_conf"] <= 0.65
        and all(v > 0.0 for v in grad_norms.values())
    )
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "shape": {"batch": args.batch, "k": args.k, "d_model": args.d_model, "n_classes": args.n_classes},
        "steps": args.steps,
        "history": history,
        **eval_correct,
        **eval_shuffled,
        **eval_null,
        **baseline,
        "grad_norms": grad_norms,
        "pass_gate2": bool(pass_gate),
        "pass_criteria": {
            "correct_acc_min": 0.85,
            "shuffled_acc_gap_min": 0.25,
            "shuffled_loss_gap_min": 0.2,
            "null_max_conf_max": 0.65,
            "nonzero_grad_norms": True,
        },
    }
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / f"dsqg_w_ebh_gate2_{stamp}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    md = [
        "# DSQG-W EBH Gate-2 Synthetic Semantic Binding",
        "",
        "Synthetic semantic-binding probe; not a real trainer quality claim.",
        "",
        f"- Device: `{summary['device']}` / {summary['gpu_name']}",
        f"- Shape: `{summary['shape']}`",
        f"- Steps: `{args.steps}`",
        f"- Pass Gate-2: `{summary['pass_gate2']}`",
        "",
        "## Metrics",
        "",
        f"- Correct evidence loss/acc: `{eval_correct['correct_loss']:.4f}` / `{eval_correct['correct_acc']:.4f}`",
        f"- Shuffled evidence loss/acc: `{eval_shuffled['shuffled_loss']:.4f}` / `{eval_shuffled['shuffled_acc']:.4f}`",
        f"- Null evidence loss/acc/conf: `{eval_null['null_loss']:.4f}` / `{eval_null['null_acc']:.4f}` / `{eval_null['null_max_conf']:.4f}`",
        f"- No-evidence baseline final loss/acc: `{baseline['baseline_final_loss']:.4f}` / `{baseline['baseline_final_acc']:.4f}`",
        "",
        "## Gradient norms",
    ]
    for name, value in grad_norms.items():
        md.append(f"- {name}: `{value:.6e}`")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, sort_keys=True, default=_json_safe))


if __name__ == "__main__":
    main()
