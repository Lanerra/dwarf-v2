#!/usr/bin/env python3
"""Run the ARC-Easy/PIQA/LAMBADA trio for a hardened Dolma-20B milestone checkpoint."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/home/dlewis3/Desktop/AI/DWARF")
TASKS = ["arc_easy", "piqa", "lambada"]
EXPECTED_PARAMETER_COUNT = 55_475_718


def setup_legacy_imports():
    for path in reversed((LEGACY_ROOT, LEGACY_ROOT / "analysis_shortlist", LEGACY_ROOT / "evals", LEGACY_ROOT / "tools")):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import analysis_shortlist.eval_external as eval_external  # type: ignore

    return eval_external


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")


def arch_config_from_contract(contract_path: Path | str) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    env = contract.get("env", {})
    trainer = Path(contract["trainer"])
    tokenizer = Path(env["DWARF_TOKENIZER"])
    if trainer.resolve() != (ROOT / "train" / "train_d512_l10_muon_olmo1_dolma3_20b_hardened.py").resolve():
        raise ValueError("external evaluation requires the isolated hardened Dolma-20B trainer")
    if env.get("DWARF_VOCAB_SIZE") != "50282":
        raise ValueError("external evaluation contract must use the 50,282-vocabulary tokenizer")
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    architecture_keys = {
        "DWARF_VOCAB_SIZE",
        "DWARF_SEQ_LEN",
        "DWARF_FFN_DIM",
        "DWARF_HISA_TOP_K",
        "DWARF_HISA_TOP_M",
        "DWARF_HISA_STAGE2_REP_R",
        "DWARF_HISA_IMPL",
        "DWARF_HISA_REP_MODE",
        "DWARF_HISA_REP_FRACTION",
        "DWARF_PURE_DSQG",
        "DWARF_Q6_G128",
        "DWARF_PRE_HISA_EMA",
        "DWARF_STAGGER_MOVT_PLANES",
        "DWARF_DISABLE_BNB",
    }
    eval_env = {key: str(value) for key, value in env.items() if key in architecture_keys}
    eval_env["DWARF_TORCH_COMPILE"] = "0"
    eval_env["DWARF_LIGER"] = "0"
    eval_env["DWARF_Q6_G128"] = "0"
    return {
        "arch": "triadic_j96_dsr",
        "D": 512,
        "H": 8,
        "FFN": int(env.get("DWARF_FFN_DIM", "1536")),
        "L": 10,
        "full_layer": 3,
        "train_script": str(trainer),
        "model_class": "TriadicJ96Dsr",
        "tokenizer": str(tokenizer),
        "eval_env": eval_env,
    }


def tokenizer_pad_id(tokenizer) -> int:
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(raw_tokenizer, "token_to_id"):
        for token in ("[PAD]", "<pad>", "<|padding|>", "<|pad|>", "<|endoftext|>", "[EOS]", "<eos>"):
            token_id = raw_tokenizer.token_to_id(token)
            if token_id is not None:
                return int(token_id)
    if hasattr(tokenizer, "pad_id"):
        return int(tokenizer.pad_id)
    raise ValueError("tokenizer has no recognized pad/EOS token")


def evaluate_lambada_target_strings(model, tokenizer, examples: list[dict[str, str]], device: str,
                                    max_examples: int | None = None) -> dict[str, float | int]:
    """Companion LAMBADA metric over every target token; preserve legacy one-token accuracy separately."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    exact_teacher_forced = 0
    teacher_forced_token_correct = 0
    skipped = 0
    selected_examples = examples if max_examples is None else examples[:max_examples]
    pad_id = tokenizer_pad_id(tokenizer)

    with torch.no_grad():
        for example in selected_examples:
            context_ids = list(tokenizer.encode(example["context"]))
            target_text = example["target"]
            if not target_text[:1].isspace():
                target_text = " " + target_text
            target_ids = list(tokenizer.encode(target_text))
            if not context_ids or not target_ids or len(target_ids) >= 2_048:
                skipped += 1
                continue

            context_ids = context_ids[-(2_048 - len(target_ids)):]
            sequence_ids = context_ids + target_ids
            padded_len = next((size for size in (32, 64, 128, 256, 512, 1024, 2048)
                               if size >= len(sequence_ids)), 2_048)
            input_ids = torch.tensor(
                [sequence_ids + [pad_id] * (padded_len - len(sequence_ids))],
                dtype=torch.long,
                device=device,
            )
            logits = model(input_ids)[0, len(context_ids) - 1:len(sequence_ids) - 1].float()
            targets = torch.tensor(target_ids, dtype=torch.long, device=device)
            total_nll += float(F.cross_entropy(logits, targets, reduction="sum").item())
            predictions = logits.argmax(dim=-1)
            correct_mask = predictions.eq(targets)
            teacher_forced_token_correct += int(correct_mask.sum().item())
            exact_teacher_forced += int(bool(correct_mask.all().item()))
            total_tokens += len(target_ids)
            total_examples += 1

    if total_tokens == 0:
        raise ValueError("no LAMBADA examples had usable context and target tokenization")
    mean_nll = total_nll / total_tokens
    return {
        "examples": total_examples,
        "target_tokens": total_tokens,
        "skipped_examples": skipped,
        "mean_target_nll": mean_nll,
        "target_token_ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
        "teacher_forced_target_token_accuracy": teacher_forced_token_correct / total_tokens,
        "exact_target_teacher_forced_accuracy": exact_teacher_forced / total_examples,
    }


def evaluate_lambada_full_target_greedy(model, tokenizer, examples: list[dict[str, str]], device: str,
                                        max_examples: int | None = None) -> dict[str, float | int]:
    """Generate every target token autoregressively; unlike the legacy scorer, supports multi-token words."""
    model.eval()
    exact = 0
    token_correct = 0
    total_tokens = 0
    total_examples = 0
    skipped = 0
    selected_examples = examples if max_examples is None else examples[:max_examples]
    pad_id = tokenizer_pad_id(tokenizer)

    with torch.no_grad():
        for example in selected_examples:
            context_ids = list(tokenizer.encode(example["context"]))
            target_text = example["target"]
            if not target_text[:1].isspace():
                target_text = " " + target_text
            target_ids = list(tokenizer.encode(target_text))
            if not context_ids or not target_ids or len(target_ids) >= 2_048:
                skipped += 1
                continue

            generated_ids = context_ids[-(2_048 - len(target_ids)):]
            prediction_ids: list[int] = []
            for _ in target_ids:
                padded_len = next((size for size in (32, 64, 128, 256, 512, 1024, 2048)
                                   if size >= len(generated_ids)), 2_048)
                input_ids = torch.tensor(
                    [generated_ids + [pad_id] * (padded_len - len(generated_ids))],
                    dtype=torch.long,
                    device=device,
                )
                prediction = int(model(input_ids)[0, len(generated_ids) - 1].argmax().item())
                prediction_ids.append(prediction)
                generated_ids.append(prediction)

            correct_mask = [prediction == target for prediction, target in zip(prediction_ids, target_ids)]
            token_correct += sum(correct_mask)
            exact += int(all(correct_mask))
            total_tokens += len(target_ids)
            total_examples += 1

    if total_tokens == 0:
        raise ValueError("no LAMBADA examples had usable context and target tokenization")
    return {
        "examples": total_examples,
        "target_tokens": total_tokens,
        "skipped_examples": skipped,
        "greedy_target_token_accuracy": token_correct / total_tokens,
        "exact_target_greedy_accuracy": exact / total_examples,
    }


def run_external_trio(*, contract_path: Path | str, checkpoint_path: Path | str, output_dir: Path | str,
                      max_examples: int | None = None) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    arch = arch_config_from_contract(contract_path)
    eval_external = setup_legacy_imports()
    arch_name = f"dwarf_dolma3_hardened_{checkpoint.stem}".replace("-", "_")
    eval_external.ARCH_CONFIGS[arch_name] = arch
    eval_external.MAX_SEQ_LEN = 2_048
    eval_external.TOKENIZER = arch["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = eval_external.load_tokenizer()
    load_started = time.time()
    model = eval_external.load_model_from_arch(arch_name, str(checkpoint), device)
    model.eval()
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"checkpoint architecture mismatch: expected {EXPECTED_PARAMETER_COUNT} parameters, got {n_params}")
    eval_started = time.time()
    results = eval_external.run_evaluation(model, tokenizer, device, tasks=TASKS, max_examples=max_examples)
    lambada_examples = json.loads((Path(eval_external.CACHE_DIR) / "lambada.json").read_text(encoding="utf-8"))
    lambada_target_diagnostic = evaluate_lambada_target_strings(
        model, tokenizer, lambada_examples, device, max_examples=max_examples,
    )
    lambada_full_target_greedy = evaluate_lambada_full_target_greedy(
        model, tokenizer, lambada_examples, device, max_examples=max_examples,
    )
    eval_external.print_summary(f"dolma3_20b_{checkpoint.stem}", results)
    payload = {
        "checkpoint": str(checkpoint),
        "contract": str(Path(contract_path).resolve()),
        "arch": arch_name,
        "parameter_count": n_params,
        "tasks": TASKS,
        "max_examples": max_examples,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "load_elapsed_s": time.time() - load_started,
        "eval_elapsed_s": time.time() - eval_started,
        "results": results,
        "lambada_target_diagnostic": lambada_target_diagnostic,
        "lambada_full_target_greedy": lambada_full_target_greedy,
    }
    output = Path(output_dir).resolve() / f"external_trio_{checkpoint.stem}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(output, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_external_trio(
        contract_path=args.run_contract,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
