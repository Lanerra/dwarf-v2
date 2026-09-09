"""Manifest-backed mixed-length data and resume primitives for canonical DWARF.

This module owns the padded-row seam only.  Architecture, optimizer, schedule and
HISA implementations remain in ``train_dwarf.py``.  It intentionally never rewrites
builder-produced shard bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

WIDTH = 2048
MODEL_LENGTH = WIDTH - 1
IGNORE_INDEX = -100
MIXED_CHECKPOINT_KIND = "dwarf-canonical-mixed-length-resume-v2"


def _json_object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"mixed-length manifest has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"mixed-length manifest {field} must be a positive integer")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


FileSignature = tuple[int, int, int, int, int]


def file_signature(path: str | Path) -> FileSignature:
    value = Path(path).stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@dataclass(frozen=True)
class MixedLengthRow:
    input_ids: torch.Tensor
    valid_length: int
    source_id: int
    document_length: int


@dataclass(frozen=True)
class MixedLengthShard:
    path: Path
    input_ids: torch.Tensor
    valid_lengths: torch.Tensor
    source_ids: torch.Tensor
    document_lengths: torch.Tensor
    loss_tokens: int
    signature: FileSignature | None
    sha256: str

    def assert_unchanged(self, *, full_hash: bool = True) -> None:
        if self.signature is None:
            return
        if file_signature(self.path) != self.signature or (
            full_hash and file_sha256(self.path) != self.sha256
        ):
            raise RuntimeError(
                f"mixed-length shard changed after validation: {self.path}"
            )

    @staticmethod
    def validate_tensors(
        tensors: Mapping[str, torch.Tensor],
        *,
        path: Path,
        vocab_size: int,
        pad_token_id: int,
        eod_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {"input_ids", "valid_lengths", "source_ids", "document_lengths"}
        absent = required - set(tensors)
        if absent:
            raise ValueError(f"{path}: missing required fields {sorted(absent)}")
        ids = tensors["input_ids"]
        lengths = tensors["valid_lengths"]
        source_ids = tensors["source_ids"]
        document_lengths = tensors["document_lengths"]
        if (
            not torch.is_tensor(ids)
            or ids.dtype != torch.int32
            or ids.ndim != 2
            or ids.shape[1] != WIDTH
        ):
            raise ValueError(f"{path}: input_ids must be int32[N,{WIDTH}]")
        for name, value, dtype in (
            ("valid_lengths", lengths, torch.int32),
            ("source_ids", source_ids, torch.int16),
            ("document_lengths", document_lengths, torch.int32),
        ):
            if (
                not torch.is_tensor(value)
                or value.dtype != dtype
                or value.ndim != 1
                or value.shape[0] != ids.shape[0]
            ):
                raise ValueError(
                    f"{path}: {name} has wrong dtype or is not aligned to input_ids"
                )
        if ids.numel() == 0:
            raise ValueError(f"{path}: empty shard")
        if bool(((ids < 0) | (ids >= vocab_size)).any()):
            raise ValueError(f"{path}: token id outside tokenizer vocabulary")
        if bool(((lengths < 1) | (lengths > MODEL_LENGTH)).any()):
            raise ValueError(f"{path}: valid_lengths must be in [1,{MODEL_LENGTH}]")
        if bool((document_lengths < 1).any()):
            raise ValueError(f"{path}: document_lengths must be positive")
        # ``valid_length`` counts shifted targets, including a natural terminal EOD.
        # Never invent or borrow an EOD; PAD may occur only after an actual terminal EOD.
        for row_index, raw_length in enumerate(lengths.tolist()):
            length = int(raw_length)
            row = ids[row_index]
            prefix = row[:length]
            if bool((prefix == pad_token_id).any()):
                raise ValueError(
                    f"{path}: row {row_index} has PAD before valid target boundary"
                )
            if bool((prefix == eod_token_id).any()):
                raise ValueError(
                    f"{path}: row {row_index} has internal EOD before terminal target"
                )
            terminal = int(row[length])
            if length < MODEL_LENGTH:
                if terminal != eod_token_id:
                    raise ValueError(
                        f"{path}: row {row_index} lacks terminal EOD at valid_lengths"
                    )
                if not bool((row[length + 1 :] == pad_token_id).all()):
                    raise ValueError(
                        f"{path}: row {row_index} has non-PAD after terminal EOD"
                    )
            elif terminal == pad_token_id:
                raise ValueError(
                    f"{path}: row {row_index} has PAD at full-row terminal"
                )
        return ids, lengths, source_ids, document_lengths

    @classmethod
    def load(
        cls,
        *,
        manifest_dir: Path,
        item: Mapping[str, Any],
        vocab_size: int,
        pad_token_id: int,
        eod_token_id: int,
    ) -> MixedLengthShard:
        allowed = {"path", "rows", "loss_tokens", "sha256", "bytes"}
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"manifest shard has unknown fields: {sorted(unknown)}")
        for field in ("path", "rows", "loss_tokens", "sha256", "bytes"):
            if field not in item:
                raise ValueError(f"manifest shard missing {field}")
        declared_sha256 = item["sha256"]
        declared_bytes = item["bytes"]
        if (
            not isinstance(declared_sha256, str)
            or len(declared_sha256) != 64
            or any(character not in "0123456789abcdef" for character in declared_sha256)
        ):
            raise ValueError(
                "manifest shard sha256 must be 64 lowercase hexadecimal characters"
            )
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 1
        ):
            raise ValueError("manifest shard bytes must be a positive integer")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest shard path must be a safe relative path")
        path = (manifest_dir / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        signature = file_signature(path)
        if signature[2] != declared_bytes:
            raise ValueError(f"{path}: declared byte count does not match")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != declared_sha256:
            raise ValueError(f"{path}: declared SHA-256 does not match")
        if file_signature(path) != signature:
            raise RuntimeError(f"{path}: shard changed while it was validated")
        loaded = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if file_signature(path) != signature:
            raise RuntimeError(f"{path}: shard changed while it was loaded")
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path}: shard is not a tensor dictionary")
        ids, lengths, source_ids, document_lengths = cls.validate_tensors(
            loaded,
            path=path,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eod_token_id=eod_token_id,
        )
        rows = int(item["rows"])
        loss_tokens = int(item["loss_tokens"])
        if rows != ids.shape[0] or loss_tokens != int(lengths.sum(dtype=torch.int64)):
            raise ValueError(f"{path}: declared rows/loss_tokens do not match tensors")
        return cls(
            path,
            ids,
            lengths,
            source_ids,
            document_lengths,
            loss_tokens,
            signature,
            observed_sha256,
        )


class MixedLengthDataset:
    """Appendable manifest-order view over immutable builder shards."""

    def __init__(
        self,
        shards: Sequence[MixedLengthShard],
        *,
        manifest: Mapping[str, Any],
        manifest_path: Path,
        manifest_signature: FileSignature | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        if not shards:
            raise ValueError("mixed-length dataset has no train shards")
        self.shards = tuple(shards)
        self.manifest = dict(manifest)
        self.manifest_path = manifest_path
        self._manifest_signature = manifest_signature
        self._manifest_sha256 = manifest_sha256
        offsets: list[int] = []
        total = 0
        for shard in self.shards:
            offsets.append(total)
            total += shard.input_ids.shape[0]
        self._offsets = tuple(offsets)
        self._rows = total
        self.total_targets = sum(shard.loss_tokens for shard in self.shards)

    def __len__(self) -> int:
        return self._rows

    def _locate(self, index: int) -> tuple[MixedLengthShard, int]:
        if index < 0 or index >= self._rows:
            raise IndexError(index)
        for offset, shard in reversed(
            tuple(zip(self._offsets, self.shards, strict=True))
        ):
            if index >= offset:
                return shard, index - offset
        raise AssertionError("row lookup offset invariant")

    def row(self, index: int) -> MixedLengthRow:
        shard, local = self._locate(index)
        return MixedLengthRow(
            input_ids=shard.input_ids[local],
            valid_length=int(shard.valid_lengths[local]),
            source_id=int(shard.source_ids[local]),
            document_length=int(shard.document_lengths[local]),
        )

    def target_count(self, index: int) -> int:
        return self.row(index).valid_length

    def assert_unchanged(self, *, full_hash: bool = True) -> None:
        """Use signatures per update; fully rehash at identity/checkpoint boundaries."""
        if self._manifest_signature is not None:
            if file_signature(self.manifest_path) != self._manifest_signature or (
                full_hash and file_sha256(self.manifest_path) != self._manifest_sha256
            ):
                raise RuntimeError("mixed-length manifest changed after validation")
        for shard in self.shards:
            shard.assert_unchanged(full_hash=full_hash)

    def identity(self) -> dict[str, Any]:
        self.assert_unchanged()
        return {
            "format": "dwarf-canonical-mixed-length-dataset-identity-v1",
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self._manifest_sha256,
            "rows": self._rows,
            "loss_tokens": self.total_targets,
            "shards": [
                {
                    "path": str(shard.path),
                    "rows": int(shard.input_ids.shape[0]),
                    "loss_tokens": shard.loss_tokens,
                    "sha256": shard.sha256,
                }
                for shard in self.shards
            ],
        }

    @classmethod
    def from_rows_for_test(
        cls, rows: Sequence[torch.Tensor], lengths: Sequence[int]
    ) -> MixedLengthDataset:
        ids = torch.stack(tuple(rows)).to(dtype=torch.int32)
        valid_lengths = torch.tensor(tuple(lengths), dtype=torch.int32)
        source_ids = torch.zeros(len(rows), dtype=torch.int16)
        document_lengths = valid_lengths.clone()
        shard = MixedLengthShard(
            Path("<test>"),
            ids,
            valid_lengths,
            source_ids,
            document_lengths,
            int(valid_lengths.sum()),
            None,
            "test",
        )
        return cls(
            (shard,), manifest={"format": "test"}, manifest_path=Path("<test-manifest>")
        )


def load_mixed_length_dataset(
    manifest_path: str | Path,
    *,
    vocab_size: int,
    pad_token_id: int,
    eod_token_id: int,
) -> MixedLengthDataset:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest_signature = file_signature(manifest_file)
    manifest_bytes = manifest_file.read_bytes()
    if file_signature(manifest_file) != manifest_signature:
        raise RuntimeError("mixed-length manifest changed while it was loaded")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(
        manifest_bytes, object_pairs_hook=_json_object_no_duplicate_keys
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("mixed-length manifest must be a JSON object")
    if manifest.get("format") != "dwarf-mixed-length-v1":
        raise ValueError("mixed-length manifest format must be dwarf-mixed-length-v1")
    required = {
        "train_shards",
        "source_names",
        "target_loss_tokens",
        "loss_tokens",
        "rows",
        "tokenizer",
        "tokenizer_sha256",
    }
    absent = required - set(manifest)
    if absent:
        raise ValueError(f"manifest missing required keys: {sorted(absent)}")
    for field in ("target_loss_tokens", "loss_tokens", "rows"):
        _positive_int(manifest[field], field=field)
    source_names = manifest["source_names"]
    if (
        not isinstance(source_names, list)
        or not source_names
        or not all(isinstance(name, str) and name for name in source_names)
        or len(source_names) != len(set(source_names))
    ):
        raise ValueError("manifest source_names must be a nonempty unique string array")
    if not isinstance(manifest["tokenizer"], str) or not manifest["tokenizer"]:
        raise ValueError("manifest tokenizer must be a nonempty path string")
    tokenizer_hash = manifest["tokenizer_sha256"]
    if (
        not isinstance(tokenizer_hash, str)
        or len(tokenizer_hash) != 64
        or any(character not in "0123456789abcdef" for character in tokenizer_hash)
    ):
        raise ValueError(
            "manifest tokenizer_sha256 must be 64 lowercase hexadecimal characters"
        )
    train_shards = manifest["train_shards"]
    if not isinstance(train_shards, list) or not train_shards:
        raise ValueError("manifest train_shards must be a nonempty array")
    if not all(isinstance(item, Mapping) for item in train_shards):
        raise ValueError("manifest train_shards entries must be objects")
    seen_shards: set[tuple[int, int]] = set()
    for item in train_shards:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest shard path must be a safe relative path")
        candidate = (manifest_file.parent / relative).resolve()
        try:
            candidate.relative_to(manifest_file.parent.resolve())
        except ValueError as error:
            raise ValueError(
                "manifest shard path escapes manifest directory"
            ) from error
        signature = file_signature(candidate)
        key = (signature[0], signature[1])
        if key in seen_shards:
            raise ValueError("manifest contains a duplicate train shard")
        seen_shards.add(key)
    shards = tuple(
        MixedLengthShard.load(
            manifest_dir=manifest_file.parent,
            item=item,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eod_token_id=eod_token_id,
        )
        for item in train_shards
    )
    dataset = MixedLengthDataset(
        shards,
        manifest=manifest,
        manifest_path=manifest_file,
        manifest_signature=manifest_signature,
        manifest_sha256=manifest_sha256,
    )
    if (
        int(manifest["rows"]) != len(dataset)
        or int(manifest["loss_tokens"]) != dataset.total_targets
    ):
        raise ValueError(
            "manifest aggregate rows/loss_tokens do not match train shards"
        )
    target = int(manifest["target_loss_tokens"])
    if dataset.total_targets < target or dataset.total_targets - target > MODEL_LENGTH:
        raise ValueError(
            "manifest loss_tokens must meet target with at most one-row excess"
        )
    return dataset


def build_labels(
    input_ids: torch.Tensor, valid_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[1] != WIDTH:
        raise ValueError(f"input_ids must have shape [B,{WIDTH}]")
    if valid_lengths.ndim != 1 or valid_lengths.shape[0] != input_ids.shape[0]:
        raise ValueError("valid_lengths must be aligned to input_ids")
    if bool(((valid_lengths < 0) | (valid_lengths > MODEL_LENGTH)).any()):
        raise ValueError(f"runtime valid_lengths must be in [0,{MODEL_LENGTH}]")
    labels = input_ids[:, 1:].to(dtype=torch.long).clone()
    positions = torch.arange(MODEL_LENGTH, device=input_ids.device).reshape(1, -1)
    labels.masked_fill_(
        positions
        >= valid_lengths.to(device=input_ids.device, dtype=torch.long).reshape(-1, 1),
        IGNORE_INDEX,
    )
    return input_ids[:, :-1].to(dtype=torch.long), labels


def masked_target_count(labels: torch.Tensor) -> int:
    return int((labels != IGNORE_INDEX).sum(dtype=torch.int64))


@dataclass(frozen=True)
class UpdatePlan:
    first_row: int
    active_rows: int
    physical_batch_size: int
    grad_accum_steps: int
    active_targets: int

    @property
    def active_row_indices(self) -> tuple[int, ...]:
        return tuple(range(self.first_row, self.first_row + self.active_rows))

    @property
    def padded_row_slots(self) -> int:
        return self.physical_batch_size * self.grad_accum_steps - self.active_rows


@dataclass(frozen=True)
class TargetBudgetPlan:
    target_budget: int
    achieved_targets: int
    excess_targets: int
    consumed_rows: tuple[int, ...]
    updates: tuple[UpdatePlan, ...]

    @classmethod
    def build(
        cls,
        dataset: MixedLengthDataset,
        *,
        target_budget: int,
        physical_batch_size: int,
        grad_accum_steps: int,
    ) -> TargetBudgetPlan:
        if target_budget < 1:
            raise ValueError("total target budget must be positive")
        if physical_batch_size < 1 or grad_accum_steps < 1:
            raise ValueError(
                "physical batch size and grad accumulation must be positive"
            )
        if target_budget > dataset.total_targets:
            raise ValueError("total target budget exceeds available active targets")
        achieved = 0
        count = 0
        while achieved < target_budget:
            achieved += dataset.target_count(count)
            count += 1
        excess = achieved - target_budget
        if excess > MODEL_LENGTH:
            raise AssertionError("whole-row budget overshoot exceeded one physical row")
        effective_rows = physical_batch_size * grad_accum_steps
        updates: list[UpdatePlan] = []
        before = 0
        while before < count:
            active_rows = min(effective_rows, count - before)
            active_targets = sum(
                dataset.target_count(index)
                for index in range(before, before + active_rows)
            )
            updates.append(
                UpdatePlan(
                    before,
                    active_rows,
                    physical_batch_size,
                    grad_accum_steps,
                    active_targets,
                )
            )
            before += active_rows
        return cls(target_budget, achieved, excess, tuple(range(count)), tuple(updates))

    def targets_before_update(self, update_index: int) -> int:
        if update_index < 0 or update_index > len(self.updates):
            raise IndexError(update_index)
        return sum(item.active_targets for item in self.updates[:update_index])

    def identity(self) -> dict[str, Any]:
        return {
            "format": "dwarf-canonical-mixed-length-target-plan-v1",
            "target_budget": self.target_budget,
            "achieved_targets": self.achieved_targets,
            "excess_targets": self.excess_targets,
            "updates": [
                {
                    "first_row": item.first_row,
                    "active_rows": item.active_rows,
                    "physical_batch_size": item.physical_batch_size,
                    "grad_accum_steps": item.grad_accum_steps,
                    "active_targets": item.active_targets,
                }
                for item in self.updates
            ],
        }


@dataclass(frozen=True)
class PhysicalBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    valid_lengths: torch.Tensor
    source_ids: torch.Tensor
    document_lengths: torch.Tensor


def collate_physical_batch(
    dataset: MixedLengthDataset,
    update: UpdatePlan,
    *,
    microbatch_index: int,
    pad_token_id: int,
) -> PhysicalBatch:
    if microbatch_index < 0 or microbatch_index >= update.grad_accum_steps:
        raise IndexError(microbatch_index)
    batch_size = update.physical_batch_size
    start = microbatch_index * batch_size
    stop = min(start + batch_size, update.active_rows)
    rows = [dataset.row(update.first_row + local) for local in range(start, stop)]
    raw = torch.full((batch_size, WIDTH), pad_token_id, dtype=torch.int32)
    lengths = torch.zeros(batch_size, dtype=torch.int32)
    source_ids = torch.full((batch_size,), -1, dtype=torch.int16)
    document_lengths = torch.zeros(batch_size, dtype=torch.int32)
    for slot, row in enumerate(rows):
        raw[slot].copy_(row.input_ids)
        lengths[slot] = row.valid_length
        source_ids[slot] = row.source_id
        document_lengths[slot] = row.document_length
    input_ids, labels = build_labels(raw, lengths)
    return PhysicalBatch(input_ids, labels, lengths, source_ids, document_lengths)


def active_cross_entropy(
    weight: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    fused_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
) -> torch.Tensor:
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2]:
        raise ValueError("hidden [B,T,D] and labels [B,T] must align")
    active = labels.reshape(-1) != IGNORE_INDEX
    if not bool(active.any()):
        return hidden.float().sum() * 0.0 + weight.float().sum() * 0.0
    active_hidden = hidden.reshape(-1, hidden.shape[-1])[active]
    active_labels = labels.reshape(-1)[active]
    if fused_loss is not None:
        return fused_loss(weight, active_hidden, active_labels)
    return F.cross_entropy(
        F.linear(active_hidden.float(), weight.float()), active_labels, reduction="mean"
    )


def normalized_microbatch_loss(
    language_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    *,
    active_targets: int,
    update_targets: int,
) -> torch.Tensor:
    if update_targets < 1 or active_targets < 0 or active_targets > update_targets:
        raise ValueError("invalid active-target normalization")
    if active_targets == 0:
        return language_loss * 0.0 + auxiliary_loss * 0.0
    return (language_loss + auxiliary_loss) * (
        float(active_targets) / float(update_targets)
    )


def mask_auxiliary_by_eligibility(
    auxiliary_loss: torch.Tensor, valid_lengths: torch.Tensor
) -> torch.Tensor:
    """Zero auxiliary only for an all-synthetic physical microbatch."""
    if valid_lengths.ndim != 1:
        raise ValueError("valid_lengths must be a rank-1 batch control")
    eligible = (valid_lengths.sum(dtype=torch.int64) > 0).to(
        device=auxiliary_loss.device, dtype=auxiliary_loss.dtype
    )
    return auxiliary_loss * eligible


def rng_payload() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def _validate_rng_payload(payload: Mapping[str, Any], *, require_cuda: bool) -> None:
    if not isinstance(payload, Mapping) or set(payload) not in (
        {"python", "torch_cpu"},
        {"python", "torch_cpu", "torch_cuda"},
    ):
        raise ValueError("mixed-length checkpoint RNG payload is invalid")
    cpu_state = payload.get("torch_cpu")
    if (
        not torch.is_tensor(cpu_state)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
    ):
        raise ValueError("mixed-length checkpoint CPU RNG state is invalid")
    cuda_states = payload.get("torch_cuda")
    if require_cuda:
        if (
            not isinstance(cuda_states, (list, tuple))
            or len(cuda_states) != torch.cuda.device_count()
            or not all(
                torch.is_tensor(state)
                and state.dtype == torch.uint8
                and state.ndim == 1
                for state in cuda_states
            )
        ):
            raise ValueError("mixed-length checkpoint CUDA RNG state is invalid")


def restore_rng(payload: Mapping[str, Any], *, require_cuda: bool = False) -> None:
    _validate_rng_payload(payload, require_cuda=require_cuda)
    random.setstate(payload["python"])
    torch.random.set_rng_state(payload["torch_cpu"])
    if require_cuda or "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(list(payload["torch_cuda"]))


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".partial",
        mode="w",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def mixed_checkpoint_payload(
    *,
    step: int,
    targets_completed: int,
    model: torch.nn.Module,
    optimizer: Any,
    architecture: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    plan: TargetBudgetPlan,
    run_config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": MIXED_CHECKPOINT_KIND,
        "step": step,
        "targets_completed": targets_completed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture": dict(architecture),
        "dataset": dict(dataset_identity),
        "target_plan": plan.identity(),
        "run_config": dict(run_config),
        "source_identity": dict(source_identity),
        "environment": dict(environment),
        "rng": rng_payload(),
    }


def restore_mixed_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Any,
    architecture: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    plan: TargetBudgetPlan,
    run_config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
    expected_lr_factor: float,
) -> tuple[int, int]:
    payload = torch.load(Path(path), weights_only=False, map_location="cpu")
    required = {
        "kind",
        "step",
        "targets_completed",
        "model",
        "optimizer",
        "architecture",
        "dataset",
        "target_plan",
        "run_config",
        "source_identity",
        "environment",
        "rng",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("mixed-length checkpoint schema does not match")
    if payload["kind"] != MIXED_CHECKPOINT_KIND:
        raise ValueError(
            "checkpoint kind does not match canonical mixed-length runtime"
        )
    for field, expected in (
        ("architecture", architecture),
        ("dataset", dataset_identity),
        ("target_plan", plan.identity()),
        ("run_config", run_config),
        ("source_identity", source_identity),
        ("environment", environment),
    ):
        if payload[field] != expected:
            raise ValueError(f"checkpoint {field} does not match active runtime")
    step = int(payload["step"])
    targets_completed = int(payload["targets_completed"])
    if (
        step < 0
        or step > len(plan.updates)
        or targets_completed != plan.targets_before_update(step)
    ):
        raise ValueError("checkpoint target-budget cursor is invalid")
    # All identities above are checked before either model or optimizer mutation.
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(
        payload["optimizer"],
        expected_lr_factor=expected_lr_factor,
        require_complete_state=True,
    )
    restore_rng(payload["rng"])
    return step, targets_completed
