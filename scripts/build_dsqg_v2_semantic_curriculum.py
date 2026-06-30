#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT / "tokenizers/olmo1_gpt_neox_dolma_v1_5_tokenizer.json"
DEFAULT_OUTPUT = ROOT / "datasets/dsqg_v2_semantic_curriculum_2048_20260629"

BUCKET_TO_ID = {
    "lexical_gap": 0,
    "copy_conflict": 1,
    "relation_bridge": 2,
    "retrieval_guardrail": 3,
}

ARCHITECTURE_NOTE = (
    "DSQG-W overlays semantic width on the DSQG-D retrieval/depth backbone; "
    "it is not a DSQG-D replacement."
)
MASK_ALIGNMENT = "token-column aligned; trainer uses loss_mask[:, 1:] for next-token targets"

TRAIN_FACTS = {
    "lexical_gap": [
        ("juvenile", "dog", "puppy", "dog"),
        ("juvenile", "cat", "kitten", "cat"),
        ("metal_symbol", "copper", "Cu", "copper"),
        ("metal_symbol", "iron", "Fe", "iron"),
        ("color", "banana", "yellow", "banana"),
        ("color", "grass", "green", "grass"),
    ],
    "copy_conflict": [
        ("juvenile", "horse", "foal", "horse"),
        ("juvenile", "cow", "calf", "cow"),
        ("metal_symbol", "gold", "Au", "gold"),
        ("color", "snow", "white", "snow"),
        ("color", "mars", "red", "mars"),
        ("metal_symbol", "sodium", "Na", "sodium"),
    ],
    "relation_bridge": [
        ("capital", "france", "paris", "country"),
        ("capital", "japan", "tokyo", "country"),
        ("tool", "write", "pencil", "action"),
        ("tool", "cut", "scissors", "action"),
        ("habitat", "camel", "desert", "animal"),
        ("habitat", "shark", "ocean", "animal"),
    ],
    "retrieval_guardrail": [
        ("codeword", "lumen", "silver", "key"),
        ("codeword", "brisk", "orange", "key"),
        ("codeword", "ember", "violet", "key"),
        ("codeword", "quartz", "blue", "key"),
        ("codeword", "cedar", "green", "key"),
        ("codeword", "raven", "black", "key"),
    ],
}

VAL_FACTS = {
    "lexical_gap": [
        ("juvenile", "sheep", "lamb", "sheep"),
        ("metal_symbol", "silver", "Ag", "silver"),
        ("color", "coal", "black", "coal"),
    ],
    "copy_conflict": [
        ("juvenile", "goose", "gosling", "goose"),
        ("metal_symbol", "lead", "Pb", "lead"),
        ("color", "sky", "blue", "sky"),
    ],
    "relation_bridge": [
        ("capital", "italy", "rome", "country"),
        ("tool", "measure", "ruler", "action"),
        ("habitat", "penguin", "ice", "animal"),
    ],
    "retrieval_guardrail": [
        ("codeword", "harbor", "teal", "key"),
        ("codeword", "summit", "white", "key"),
        ("codeword", "meadow", "gold", "key"),
    ],
}

TRAIN_TEMPLATES = {
    "lexical_gap": [
        "Relation {relation}. Evidence subject {subject}. Produce the mapped value.",
        "Semantic key {subject}; relation family {relation}; infer its paired answer.",
    ],
    "copy_conflict": [
        "Do not copy the nearby distractor {distractor}. Evidence subject {subject} requires relation {relation}.",
        "The local word {distractor} is a trap; use subject {subject} and relation {relation}.",
    ],
    "relation_bridge": [
        "Bridge source says {subject} belongs to {evidence}; relation requested is {relation}.",
        "Use the category clue {evidence} for {subject}; answer the {relation} query.",
    ],
    "retrieval_guardrail": [
        "Long key-value memory: key {subject} has value {answer}. Return the stored value.",
        "Remote evidence states code {subject} resolves to {answer}. Retrieve the value.",
    ],
}

VAL_TEMPLATES = {
    "lexical_gap": [
        "Heldout lexical clue: {subject}. Relation type: {relation}. Give only the mapped value.",
    ],
    "copy_conflict": [
        "Heldout trap: the close token says {distractor}; the evidence key is {subject}; relation {relation} decides.",
    ],
    "relation_bridge": [
        "Heldout bridge: {subject} is linked through {evidence}. Requested relation: {relation}.",
    ],
    "retrieval_guardrail": [
        "Heldout remote memory: {subject} maps to {answer}. State the value.",
    ],
}

DISTANCES_BY_BUCKET = {
    "lexical_gap": [32, 128],
    "copy_conflict": [32, 128],
    "relation_bridge": [128, 32],
    "retrieval_guardrail": [512, 128, 32],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    return [int(i) for i in ids]


def first_id(tokenizer: Tokenizer, text: str) -> int:
    return encode(tokenizer, text)[0]


def answer_columns(seq_len: int) -> list[int]:
    candidates = [160, 256, 384, 640, 768, 1024, 1280, 1536, 1792]
    usable = [col for col in candidates if 48 <= col < seq_len - 16]
    if usable:
        return usable
    return [max(32, min(seq_len - 16, int(seq_len * frac))) for frac in (0.45, 0.65, 0.82)]


def choose_distance(bucket: str, answer_col: int, evidence_len: int, idx: int) -> int:
    allowed = [d for d in DISTANCES_BY_BUCKET[bucket] if answer_col - d - evidence_len - 2 >= 8]
    if not allowed:
        allowed = [32]
    return int(allowed[idx % len(allowed)])


def place(row: torch.Tensor, ids: list[int], start: int) -> tuple[int, int]:
    if start < 0 or start + len(ids) > row.numel():
        raise ValueError(f"cannot place {len(ids)} ids at {start} in row length {row.numel()}")
    row[start : start + len(ids)] = torch.tensor(ids, dtype=row.dtype)
    return start, start + len(ids)


def make_record(
    *,
    tokenizer: Tokenizer,
    split: str,
    bucket: str,
    index: int,
    seq_len: int,
    rng: random.Random,
    pad_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if split == "train":
        facts_by_bucket = TRAIN_FACTS
        templates_by_bucket = TRAIN_TEMPLATES
    elif split == "val_same_family":
        # Same fact families as train, but held-out validation templates.  This
        # separates template generalization from strict unseen-family generalization.
        facts_by_bucket = TRAIN_FACTS
        templates_by_bucket = VAL_TEMPLATES
    else:
        facts_by_bucket = VAL_FACTS
        templates_by_bucket = VAL_TEMPLATES
    relation, subject, answer, evidence = facts_by_bucket[bucket][index % len(facts_by_bucket[bucket])]
    template = templates_by_bucket[bucket][index % len(templates_by_bucket[bucket])]
    distractor = subject if bucket != "copy_conflict" else f"not_{answer}"
    rendered = template.format(relation=relation, subject=subject, answer=answer, evidence=evidence, distractor=distractor)

    row = torch.full((seq_len,), int(pad_id), dtype=torch.int32)
    mask = torch.zeros((seq_len,), dtype=torch.bool)
    filler_id = first_id(tokenizer, " the")
    row[:] = int(filler_id)

    answer_ids = encode(tokenizer, " " + answer)
    subject_ids = encode(tokenizer, " " + subject)
    relation_ids = encode(tokenizer, " " + relation)
    evidence_ids = subject_ids
    header_ids = encode(tokenizer, f" DWARF semantic curriculum {bucket} split {split}. ")
    context_ids = encode(tokenizer, " " + rendered + " ")
    query_ids = encode(tokenizer, " Question: produce the answer. Answer:")
    eos_ids = [int(eos_id)]

    cols = answer_columns(seq_len)
    answer_col = cols[(index + rng.randrange(len(cols))) % len(cols)]
    if answer_col + len(answer_ids) + 1 >= seq_len:
        answer_col = seq_len - len(answer_ids) - 2
    distance = choose_distance(bucket, answer_col, len(evidence_ids), index)
    evidence_end_target = answer_col - distance - 1
    evidence_start = max(8 + len(header_ids), evidence_end_target - len(evidence_ids) + 1)
    evidence_end = evidence_start + len(evidence_ids)

    place(row, header_ids[: max(1, min(len(header_ids), evidence_start - 1))], 0)
    context_room = max(0, evidence_start - len(header_ids) - 2)
    if context_room > 0:
        place(row, context_ids[:context_room], len(header_ids) + 1)
    evidence_span = place(row, evidence_ids, evidence_start)
    relation_start = max(evidence_end + 2, answer_col - distance // 2)
    if relation_start + len(relation_ids) < answer_col - len(query_ids) - 2:
        relation_span = place(row, relation_ids, relation_start)
    else:
        relation_span = (-1, -1)

    distractor_span = (-1, -1)
    if bucket == "copy_conflict":
        distractor_ids = encode(tokenizer, " " + distractor)
        distractor_start = max(evidence_end + 1, answer_col - len(query_ids) - len(distractor_ids) - 4)
        if distractor_start + len(distractor_ids) < answer_col - len(query_ids) - 1:
            distractor_span = place(row, distractor_ids, distractor_start)

    query_start = max(evidence_end + 1, answer_col - len(query_ids))
    if query_start + len(query_ids) > answer_col:
        query_ids = query_ids[-max(1, answer_col - query_start) :]
        query_start = answer_col - len(query_ids)
    question_span = place(row, query_ids, query_start)
    answer_span = place(row, answer_ids, answer_col)
    mask[answer_span[0] : answer_span[1]] = True
    if answer_span[1] < seq_len:
        place(row, eos_ids, answer_span[1])
        if answer_span[1] + 1 < seq_len:
            row[answer_span[1] + 1 :] = int(pad_id)

    family_id = f"{bucket}:{relation}:{subject}:{answer}"
    template_id = f"{split}:{bucket}:{index % len(templates_by_bucket[bucket])}"
    non_pad = row[: max(answer_span[1] + 1, question_span[1])].tolist()
    prompt_hash = hashlib.sha256(bytes(str(non_pad), "utf-8")).hexdigest()
    record = {
        "id": f"{split}_{index:06d}_{bucket}_{subject}_{answer}",
        "split": split,
        "bucket": bucket,
        "source_id": BUCKET_TO_ID[bucket],
        "family_id": family_id,
        "template_id": template_id,
        "relation": relation,
        "subject": subject,
        "answer": answer,
        "evidence": evidence,
        "distance": distance,
        "answer_span": list(answer_span),
        "answer_token_count": int(mask.sum().item()),
        "evidence_span": list(evidence_span),
        "relation_span": list(relation_span),
        "question_span": list(question_span),
        "distractor_span": list(distractor_span),
        "prompt_hash": prompt_hash,
        "text_template": rendered,
        "decoded_prefix": tokenizer.decode([int(x) for x in non_pad[: min(len(non_pad), 220)]], skip_special_tokens=False),
    }
    return row, mask, record


def make_split(tokenizer: Tokenizer, *, split: str, size: int, seq_len: int, seed: int, pad_id: int, eos_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    buckets = list(BUCKET_TO_ID)
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    source_ids: list[int] = []
    records: list[dict[str, Any]] = []
    bucket_offsets: defaultdict[str, int] = defaultdict(int)
    for i in range(size):
        bucket = buckets[i % len(buckets)]
        local_index = bucket_offsets[bucket]
        bucket_offsets[bucket] += 1
        row, mask, record = make_record(
            tokenizer=tokenizer,
            split=split,
            bucket=bucket,
            index=local_index,
            seq_len=seq_len,
            rng=rng,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        rows.append(row)
        masks.append(mask)
        source_ids.append(BUCKET_TO_ID[bucket])
        records.append(record)
    order = list(range(size))
    rng.shuffle(order)
    return (
        torch.stack([rows[i] for i in order], dim=0).contiguous(),
        torch.stack([masks[i] for i in order], dim=0).contiguous(),
        torch.tensor([source_ids[i] for i in order], dtype=torch.int16),
        [records[i] for i in order],
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def split_summary(records: list[dict[str, Any]], loss_tokens: int, target_slots: int | None = None) -> dict[str, Any]:
    bucket_counts = Counter(str(r["bucket"]) for r in records)
    answer_tokens_by_bucket: dict[str, int] = defaultdict(int)
    distances_by_bucket: dict[str, list[int]] = defaultdict(list)
    for record in records:
        bucket = str(record["bucket"])
        answer_tokens_by_bucket[bucket] += int(record["answer_token_count"])
        if "distance" in record:
            distances_by_bucket[bucket].append(int(record["distance"]))
    return {
        "rows": len(records),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "answer_tokens_by_bucket": dict(sorted(answer_tokens_by_bucket.items())),
        "distance_values_by_bucket": {k: sorted(set(v)) for k, v in sorted(distances_by_bucket.items())},
        "real_loss_tokens": int(loss_tokens),
        "target_slots": int(target_slots) if target_slots is not None else None,
        "real_loss_fraction": (float(loss_tokens) / float(target_slots)) if target_slots else None,
    }


def audit_records(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    *,
    train_loss_tokens: int,
    val_loss_tokens: int,
    train_target_slots: int | None = None,
    val_target_slots: int | None = None,
    allow_family_overlap: bool = False,
) -> dict[str, Any]:
    train_hashes = {str(r["prompt_hash"]) for r in train_records}
    val_hashes = {str(r["prompt_hash"]) for r in val_records}
    train_ids = {str(r["id"]) for r in train_records}
    val_ids = {str(r["id"]) for r in val_records}
    train_families = {str(r["family_id"]) for r in train_records}
    val_families = {str(r["family_id"]) for r in val_records}
    train_templates = {str(r["template_id"]) for r in train_records}
    val_templates = {str(r["template_id"]) for r in val_records}
    leakage = {
        "prompt_hash_overlap_count": len(train_hashes & val_hashes),
        "record_id_overlap_count": len(train_ids & val_ids),
        "family_id_overlap_count": len(train_families & val_families),
        "template_id_overlap_count": len(train_templates & val_templates),
    }
    zero_answer_records = [r["id"] for r in train_records + val_records if int(r.get("answer_token_count", 0)) <= 0]
    fatal_leakage = dict(leakage)
    if allow_family_overlap:
        fatal_leakage["family_id_overlap_count"] = 0
    pass_flag = (
        all(v == 0 for v in fatal_leakage.values())
        and int(train_loss_tokens) > 0
        and int(val_loss_tokens) > 0
        and not zero_answer_records
    )
    return {
        "pass": bool(pass_flag),
        "leakage": leakage,
        "allow_family_overlap": bool(allow_family_overlap),
        "zero_answer_records": zero_answer_records[:20],
        "splits": {
            "train": split_summary(train_records, train_loss_tokens, train_target_slots),
            "val": split_summary(val_records, val_loss_tokens, val_target_slots),
        },
    }


def build_curriculum(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT,
    tokenizer_path: Path | str = DEFAULT_TOKENIZER,
    train_size: int = 4096,
    val_size: int = 512,
    seq_len: int = 2048,
    seed: int = 20260629,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = int(tokenizer.get_vocab_size())
    pad_id = 1
    eos_id = 50279
    if vocab_size <= eos_id:
        raise ValueError(f"expected OLMo vocab to include eos_id={eos_id}, got vocab_size={vocab_size}")

    train, train_mask, train_source_id, train_records = make_split(
        tokenizer, split="train", size=int(train_size), seq_len=int(seq_len), seed=int(seed), pad_id=pad_id, eos_id=eos_id
    )
    val, val_mask, val_source_id, val_records = make_split(
        tokenizer, split="val", size=int(val_size), seq_len=int(seq_len), seed=int(seed), pad_id=pad_id, eos_id=eos_id
    )
    val_same_family, val_same_family_mask, val_same_family_source_id, val_same_family_records = make_split(
        tokenizer,
        split="val_same_family",
        size=int(val_size),
        seq_len=int(seq_len),
        seed=int(seed),
        pad_id=pad_id,
        eos_id=eos_id,
    )
    train_loss_tokens = int(train_mask[:, 1:].sum().item())
    val_loss_tokens = int(val_mask[:, 1:].sum().item())
    val_same_family_loss_tokens = int(val_same_family_mask[:, 1:].sum().item())
    strict_audit = audit_records(
        train_records,
        val_records,
        train_loss_tokens=train_loss_tokens,
        val_loss_tokens=val_loss_tokens,
        train_target_slots=int(train_mask[:, 1:].numel()),
        val_target_slots=int(val_mask[:, 1:].numel()),
    )
    same_family_audit = audit_records(
        train_records,
        val_same_family_records,
        train_loss_tokens=train_loss_tokens,
        val_loss_tokens=val_same_family_loss_tokens,
        train_target_slots=int(train_mask[:, 1:].numel()),
        val_target_slots=int(val_same_family_mask[:, 1:].numel()),
        allow_family_overlap=True,
    )
    audit = {
        "pass": bool(strict_audit["pass"] and same_family_audit["pass"]),
        "validation_lanes": {
            "strict": strict_audit,
            "same_family": same_family_audit,
        },
        "splits": {
            "train": strict_audit["splits"]["train"],
            "val": strict_audit["splits"]["val"],
            "val_same_family": same_family_audit["splits"]["val"],
        },
        "zero_answer_records": sorted(
            set(strict_audit["zero_answer_records"]) | set(same_family_audit["zero_answer_records"])
        ),
    }
    if not audit["pass"]:
        raise ValueError(f"semantic curriculum audit failed: {json.dumps(audit['validation_lanes'], sort_keys=True)}")

    dataset_path = out / f"dsqg_v2_semantic_curriculum_{seq_len}_train{train_size}_val{val_size}.pt"
    same_family_dataset_path = out / f"dsqg_v2_semantic_curriculum_{seq_len}_train{train_size}_val{val_size}_same_family.pt"
    train_jsonl = out / "train_records.jsonl"
    val_jsonl = out / "val_records.jsonl"
    val_same_family_jsonl = out / "val_same_family_records.jsonl"
    manifest_path = out / "manifest.json"
    audit_path = out / "audit.json"
    samples_path = out / "decoded_samples.json"

    payload = {
        "train": train,
        "val": val,
        "val_same_family": val_same_family,
        "train_loss_mask": train_mask,
        "val_loss_mask": val_mask,
        "val_same_family_loss_mask": val_same_family_mask,
        "train_source_id": train_source_id,
        "val_source_id": val_source_id,
        "val_same_family_source_id": val_same_family_source_id,
        "vocab_size": vocab_size,
        "metadata": {
            "name": "dsqg_v2_semantic_curriculum",
            "seq_len": int(seq_len),
            "seed": int(seed),
            "active_val_lane": "strict",
            "mask_alignment": MASK_ALIGNMENT,
            "architecture_note": ARCHITECTURE_NOTE,
            "bucket_to_id": dict(BUCKET_TO_ID),
        },
    }
    same_family_payload = dict(payload)
    same_family_payload.update(
        {
            "val": val_same_family,
            "val_loss_mask": val_same_family_mask,
            "val_source_id": val_same_family_source_id,
            "metadata": {**payload["metadata"], "active_val_lane": "same_family"},
        }
    )
    # Keep the artifact compatible with the trainer's torch.load(..., weights_only=True).
    # PyTorch's restricted unpickler still rejects newer pickle protocol opcodes in
    # some versions, so use the default protocol instead of protocol 5 here.
    torch.save(payload, dataset_path)
    torch.save(same_family_payload, same_family_dataset_path)
    write_jsonl(train_jsonl, train_records)
    write_jsonl(val_jsonl, val_records)
    write_jsonl(val_same_family_jsonl, val_same_family_records)

    decoded_samples = {
        "train": [r["decoded_prefix"] for r in train_records[:4]],
        "val": [r["decoded_prefix"] for r in val_records[:4]],
        "val_same_family": [r["decoded_prefix"] for r in val_same_family_records[:4]],
    }
    samples_path.write_text(json.dumps(decoded_samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "name": "dsqg_v2_semantic_curriculum",
        "version": "20260629-small-answer-mask-v1",
        "created_by": str(Path(__file__).relative_to(ROOT)),
        "git_commit": git_commit(),
        "dataset_path": str(dataset_path),
        "same_family_dataset_path": str(same_family_dataset_path),
        "train_records_jsonl": str(train_jsonl),
        "val_records_jsonl": str(val_jsonl),
        "val_same_family_records_jsonl": str(val_same_family_jsonl),
        "audit_path": str(audit_path),
        "decoded_samples_path": str(samples_path),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": vocab_size,
            "pad_id": pad_id,
            "eos_id": eos_id,
        },
        "dataset_shape": {
            "seq_len": int(seq_len),
            "train_rows": int(train_size),
            "val_rows": int(val_size),
            "val_same_family_rows": int(val_size),
        },
        "bucket_to_id": dict(BUCKET_TO_ID),
        "mask_alignment": MASK_ALIGNMENT,
        "architecture_note": ARCHITECTURE_NOTE,
        "compatible_trainer": "train/train_d512_l10_muon_olmo1_base_v1_q6_g128_smoke.py",
        "intended_architecture": "DSQG-D backbone + DSQG-W semantic-width overlay at sites 2,6,final; width-cell gate -5 ladder candidate.",
        "audit_summary": audit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.update({
        "dataset_path": str(dataset_path),
        "same_family_dataset_path": str(same_family_dataset_path),
        "manifest_path": str(manifest_path),
        "train_records_jsonl": str(train_jsonl),
        "val_records_jsonl": str(val_jsonl),
        "val_same_family_records_jsonl": str(val_same_family_jsonl),
        "decoded_samples_path": str(samples_path),
        "dataset_sha256": sha256_file(dataset_path),
        "same_family_dataset_sha256": sha256_file(same_family_dataset_path),
    })
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "pass": True,
        "dataset_path": str(dataset_path),
        "same_family_dataset_path": str(same_family_dataset_path),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "train_records_jsonl": str(train_jsonl),
        "val_records_jsonl": str(val_jsonl),
        "val_same_family_records_jsonl": str(val_same_family_jsonl),
        "decoded_samples_path": str(samples_path),
        "train_real_loss_tokens": train_loss_tokens,
        "val_real_loss_tokens": val_loss_tokens,
        "val_same_family_real_loss_tokens": val_same_family_loss_tokens,
        "bucket_counts_train": audit["splits"]["train"]["bucket_counts"],
        "bucket_counts_val": audit["splits"]["val"]["bucket_counts"],
        "bucket_counts_val_same_family": audit["splits"]["val_same_family"]["bucket_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a small DWARF-v2 semantic answer-mask curriculum shard")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args(argv)
    report = build_curriculum(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
