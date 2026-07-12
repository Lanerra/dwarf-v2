from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PasskeyConfig:
    max_seq_len: int
    distances: list[int]
    trials: int
    batch_size: int
    words: list[str]
    filler_sentence: str
    intro_template: str
    retrieval_cue: str
    pad_id: int = 0


def _token_to_id(tokenizer: Any, token: str) -> int | None:
    raw = getattr(tokenizer, 'tokenizer', tokenizer)
    if hasattr(raw, 'token_to_id'):
        tok_id = raw.token_to_id(token)
        if tok_id is not None:
            return int(tok_id)
    vocab = getattr(raw, 'vocab', None)
    if isinstance(vocab, dict) and token in vocab:
        return int(vocab[token])
    return None


def resolve_tokenizer_pad_id(tokenizer: Any) -> int:
    """Resolve a tokenizer's real padding id without assuming legacy id0.

    OLMo's tokenizer uses id 0 for the ordinary token ``|||IP_ADDRESS|||``;
    treating it as padding contaminates prefix/padded evals. Prefer explicit pad
    tokens, then EOS as a safe padding fallback for causal eval, and fail closed
    if no known special token exists.
    """
    for token in ('[PAD]', '<pad>', '<|padding|>', '<|pad|>', '<|endoftext|>', '[EOS]', '<eos>'):
        tok_id = _token_to_id(tokenizer, token)
        if tok_id is not None:
            return tok_id
    raise ValueError('Tokenizer has no recognized pad/EOS token; refusing to use id 0')


@dataclass(frozen=True)
class PasskeyExample:
    distance: int
    target_word: str
    full_seq: list[int]
    cue_position: int
    candidate_token_ids: list[int]


def _amp_context(device: str):
    if str(device).startswith('cuda'):
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _temporary_eval(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@contextlib.contextmanager
def _temporary_causal_control_valid_lengths(
    model: torch.nn.Module,
    valid_lengths: list[int] | torch.Tensor | None,
    device: str,
):
    """Temporarily expose per-row prefix-valid lengths to HISA-style modules.

    Fixed-length passkey eval needs the model to see the training-time sequence
    geometry while HISA control metadata ignores pad/future rows. Modules that
    understand ``_causal_control_valid_lengths`` use it; other modules ignore it.
    """
    if valid_lengths is None:
        yield
        return
    valid = torch.as_tensor(valid_lengths, dtype=torch.long, device=device)
    touched: list[tuple[torch.nn.Module, bool, Any]] = []
    for module in model.modules():
        if isinstance(getattr(module, 'num_chunks', None), int) and getattr(module, 'num_chunks') > 1:
            had = hasattr(module, '_causal_control_valid_lengths')
            old = getattr(module, '_causal_control_valid_lengths', None)
            setattr(module, '_causal_control_valid_lengths', valid)
            touched.append((module, had, old))
    try:
        yield
    finally:
        for module, had, old in touched:
            if had:
                setattr(module, '_causal_control_valid_lengths', old)
            else:
                try:
                    delattr(module, '_causal_control_valid_lengths')
                except AttributeError:
                    pass


def _encode_candidate_words(tokenizer: Any, config: PasskeyConfig) -> dict[str, int]:
    """Return the one-token candidate the model is asked to emit after the cue.

    BPE-style tokenizers often have different ids for a bare word and for the
    same word as a continuation after a preceding space.  The passkey prompt
    scores logits after ``config.retrieval_cue`` (for example, after
    ``"the secret word is"``), so the candidate must be the first token added by
    ``retrieval_cue + ' ' + word``.  Falling back to bare ``encode(word)[0]``
    silently scores the wrong target for tokenizers whose vocabulary contains
    both ``apple`` and ``Ġapple``; worse, multi-token bare encodings can reduce
    the target to fragments such as ``ch`` or ``m``.
    """
    cue_ids = tokenizer.encode(config.retrieval_cue)
    word_token_ids: dict[str, int] = {}
    for word in config.words:
        candidate_id: int | None = None

        cue_plus_word = f"{config.retrieval_cue.rstrip()} {word}"
        continuation = tokenizer.encode(cue_plus_word)
        if len(continuation) > len(cue_ids) and continuation[:len(cue_ids)] == cue_ids:
            candidate_id = int(continuation[len(cue_ids)])

        if candidate_id is None:
            encoded = tokenizer.encode(' ' + word)
            if not encoded:
                encoded = tokenizer.encode(word)
            if not encoded:
                raise ValueError(f'Could not encode passkey word: {word}')
            candidate_id = int(encoded[0])

        word_token_ids[word] = candidate_id
    return word_token_ids


def build_passkey_examples(tokenizer: Any, config: PasskeyConfig) -> dict[int, list[PasskeyExample]]:
    filler_ids = tokenizer.encode(config.filler_sentence)
    cue_ids = tokenizer.encode(config.retrieval_cue)
    if not filler_ids:
        raise ValueError('Filler sentence encoded to an empty sequence')
    if not cue_ids:
        raise ValueError('Retrieval cue encoded to an empty sequence')

    word_token_ids = _encode_candidate_words(tokenizer, config)
    examples_by_distance: dict[int, list[PasskeyExample]] = {}

    for distance in config.distances:
        examples: list[PasskeyExample] = []
        for trial_idx in range(config.trials):
            target = config.words[trial_idx % len(config.words)]
            others = [word for word in config.words if word != target]
            intro_ids = tokenizer.encode(config.intro_template.format(word=target))
            available = config.max_seq_len - 1 - len(intro_ids) - len(cue_ids) - 1
            if distance > available:
                continue

            filler: list[int] = []
            while len(filler) < distance:
                filler.extend(filler_ids)

            full_seq = intro_ids + filler[:distance] + cue_ids
            if len(full_seq) >= config.max_seq_len:
                continue

            candidate_words = [target] + others[:9]
            examples.append(
                PasskeyExample(
                    distance=distance,
                    target_word=target,
                    full_seq=full_seq,
                    cue_position=len(full_seq) - 1,
                    candidate_token_ids=[word_token_ids[word] for word in candidate_words],
                )
            )
        examples_by_distance[distance] = examples

    return examples_by_distance


@torch.inference_mode()
def _logits_at_position(
    model: torch.nn.Module,
    ids: list[int],
    position: int,
    device: str,
    pad_to_length: int | None = None,
    pad_id: int = 0,
    causal_control_valid_length: int | None = None,
) -> torch.Tensor:
    if pad_to_length is not None:
        if pad_to_length < len(ids):
            raise ValueError(f'pad_to_length={pad_to_length} is shorter than sequence length={len(ids)}')
        ids = ids + [pad_id] * (pad_to_length - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with _temporary_causal_control_valid_lengths(
        model,
        None if causal_control_valid_length is None else [causal_control_valid_length],
        device,
    ):
        with _amp_context(device):
            logits = model(input_ids)
    return logits[0, position, :].float().detach().cpu()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hisa_num_chunks(model: torch.nn.Module) -> list[int]:
    chunks = set()
    for module in model.modules():
        value = getattr(module, 'num_chunks', None)
        if isinstance(value, int) and value > 1:
            chunks.add(value)
    return sorted(chunks)


def _kernel_compatible_prefix_length(
    model: torch.nn.Module,
    seq_len: int,
    max_seq_len: int,
) -> int:
    """Right-pad just enough for HISA kernels whose chunk arange needs pow2."""
    chunk_counts = _hisa_num_chunks(model)
    if not chunk_counts:
        return seq_len

    for padded_len in range(seq_len, max_seq_len + 1):
        if all(_is_power_of_two((padded_len + chunks - 1) // chunks) for chunks in chunk_counts):
            return padded_len

    raise ValueError(
        f'No kernel-compatible prefix length found for seq_len={seq_len} '
        f'within max_seq_len={max_seq_len}'
    )


@torch.inference_mode()
def _batched_logits_at_positions(
    model: torch.nn.Module,
    sequences: list[list[int]],
    positions: list[int],
    device: str,
    pad_to_length: int | None = None,
    pad_id: int = 0,
    causal_control_valid_lengths: list[int] | None = None,
) -> list[torch.Tensor]:
    if not sequences:
        return []
    max_len = max(len(seq) for seq in sequences)
    if pad_to_length is not None:
        if pad_to_length < max_len:
            raise ValueError(f'pad_to_length={pad_to_length} is shorter than max sequence length={max_len}')
        max_len = pad_to_length
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    pos = torch.tensor(positions, dtype=torch.long, device=device)
    with _temporary_causal_control_valid_lengths(model, causal_control_valid_lengths, device):
        with _amp_context(device):
            logits = model(input_ids)
    rows = torch.arange(input_ids.size(0), device=device)
    gathered = logits[rows, pos, :].float().detach().cpu()
    return [gathered[i] for i in range(gathered.size(0))]


@torch.inference_mode()
def legacy_padded_passkey_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    config: PasskeyConfig,
) -> dict[int, float]:
    examples_by_distance = build_passkey_examples(tokenizer, config)
    results: dict[int, float] = {}

    with _temporary_eval(model):
        for distance, examples in examples_by_distance.items():
            if not examples:
                results[distance] = 0.0
                continue

            correct = 0
            for example in examples:
                padded = example.full_seq + [config.pad_id] * (config.max_seq_len - len(example.full_seq))
                row_logits = _logits_at_position(model, padded, example.cue_position, device)
                candidate_logits = row_logits[example.candidate_token_ids]
                correct += int(candidate_logits.argmax().item() == 0)
            results[distance] = correct / len(examples)

    return results


@torch.inference_mode()
def passkey_accuracy_prefix_only(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    config: PasskeyConfig,
    kernel_compatible: bool = False,
) -> dict[int, float]:
    examples_by_distance = build_passkey_examples(tokenizer, config)
    results: dict[int, float] = {}

    with _temporary_eval(model):
        for distance, examples in examples_by_distance.items():
            if not examples:
                results[distance] = 0.0
                continue

            correct = 0
            examples_by_length: dict[int, list[PasskeyExample]] = {}
            for example in examples:
                examples_by_length.setdefault(len(example.full_seq), []).append(example)

            for same_length_examples in examples_by_length.values():
                sequences = [example.full_seq for example in same_length_examples]
                positions = [example.cue_position for example in same_length_examples]
                pad_to_length = None
                if kernel_compatible:
                    pad_to_length = _kernel_compatible_prefix_length(
                        model,
                        max(len(sequence) for sequence in sequences),
                        config.max_seq_len,
                    )
                rows = _batched_logits_at_positions(
                    model, sequences, positions, device, pad_to_length=pad_to_length, pad_id=config.pad_id
                )
                for example, row_logits in zip(same_length_examples, rows, strict=True):
                    candidate_logits = row_logits[example.candidate_token_ids]
                    correct += int(candidate_logits.argmax().item() == 0)

            results[distance] = correct / len(examples)

    return results


@torch.inference_mode()
def passkey_accuracy_kernel_compatible_prefix_only(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    config: PasskeyConfig,
) -> dict[int, float]:
    return passkey_accuracy_prefix_only(
        model,
        tokenizer,
        device,
        config,
        kernel_compatible=True,
    )


@torch.inference_mode()
def passkey_accuracy_fixed_length_causal_control(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    config: PasskeyConfig,
    *,
    fixed_length: int | None = None,
    batch_mode: str = 'singleton',
) -> dict[int, float]:
    """Score passkey at fixed training geometry with prefix-causal HISA control.

    ``batch_mode``:
    - ``singleton``: evaluate each example alone (most diagnostic/stable).
    - ``grouped``: batch same-length examples together.
    - ``repeated``: evaluate each example as row 0 of a repeated fixed-size batch;
      useful for checking batch-shape sensitivity without peer-content changes.
    """
    fixed_length = config.max_seq_len if fixed_length is None else int(fixed_length)
    if fixed_length > config.max_seq_len:
        raise ValueError(f'fixed_length={fixed_length} exceeds max_seq_len={config.max_seq_len}')
    if batch_mode not in {'singleton', 'grouped', 'repeated'}:
        raise ValueError(f"unknown passkey batch_mode={batch_mode!r}")

    examples_by_distance = build_passkey_examples(tokenizer, config)
    results: dict[int, float] = {}

    with _temporary_eval(model):
        for distance, examples in examples_by_distance.items():
            if not examples:
                results[distance] = 0.0
                continue

            correct = 0
            if batch_mode == 'singleton':
                for example in examples:
                    row_logits = _logits_at_position(
                        model,
                        example.full_seq,
                        example.cue_position,
                        device,
                        pad_to_length=fixed_length,
                        pad_id=config.pad_id,
                        causal_control_valid_length=len(example.full_seq),
                    )
                    candidate_logits = row_logits[example.candidate_token_ids]
                    correct += int(candidate_logits.argmax().item() == 0)
            elif batch_mode == 'repeated':
                repeat_n = max(1, int(config.batch_size))
                for example in examples:
                    rows = _batched_logits_at_positions(
                        model,
                        [example.full_seq] * repeat_n,
                        [example.cue_position] * repeat_n,
                        device,
                        pad_to_length=fixed_length,
                        pad_id=config.pad_id,
                        causal_control_valid_lengths=[len(example.full_seq)] * repeat_n,
                    )
                    candidate_logits = rows[0][example.candidate_token_ids]
                    correct += int(candidate_logits.argmax().item() == 0)
            else:
                examples_by_length: dict[int, list[PasskeyExample]] = {}
                for example in examples:
                    examples_by_length.setdefault(len(example.full_seq), []).append(example)
                for same_length_examples in examples_by_length.values():
                    sequences = [example.full_seq for example in same_length_examples]
                    positions = [example.cue_position for example in same_length_examples]
                    rows = _batched_logits_at_positions(
                        model,
                        sequences,
                        positions,
                        device,
                        pad_to_length=fixed_length,
                        pad_id=config.pad_id,
                        causal_control_valid_lengths=[len(example.full_seq) for example in same_length_examples],
                    )
                    for example, row_logits in zip(same_length_examples, rows, strict=True):
                        candidate_logits = row_logits[example.candidate_token_ids]
                        correct += int(candidate_logits.argmax().item() == 0)

            results[distance] = correct / len(examples)

    return results


@torch.inference_mode()
def passkey_prefix_consistency_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    config: PasskeyConfig,
    kernel_compatible: bool = False,
    eval_mode: str | None = None,
    fixed_length: int | None = None,
    batch_mode: str = 'grouped',
) -> dict[str, Any]:
    examples_by_distance = build_passkey_examples(tokenizer, config)
    filler_ids = tokenizer.encode(config.filler_sentence)

    max_pad_logit_delta = 0.0
    max_suffix_logit_delta = 0.0
    total_examples = 0

    with _temporary_eval(model):
        for examples in examples_by_distance.values():
            for example in examples:
                total_examples += 1
                # Compare at the fixed training geometry.  A short standalone
                # prefix changes the physical HISA chunk size (N/C), so it is
                # not a valid prefix-invariance control for a fixed-N model.
                # Every variant below exposes the true prefix length to HISA
                # metadata while retaining the same physical N.
                prefix_pad_to_length = config.max_seq_len
                prefix_logits = _logits_at_position(
                    model,
                    example.full_seq,
                    example.cue_position,
                    device,
                    pad_to_length=prefix_pad_to_length,
                    pad_id=config.pad_id,
                    causal_control_valid_length=len(example.full_seq),
                )

                padded = example.full_seq + [config.pad_id] * (config.max_seq_len - len(example.full_seq))
                padded_logits = _logits_at_position(
                    model,
                    padded,
                    example.cue_position,
                    device,
                    pad_to_length=config.max_seq_len,
                    pad_id=config.pad_id,
                    causal_control_valid_length=len(example.full_seq),
                )
                max_pad_logit_delta = max(
                    max_pad_logit_delta,
                    float((prefix_logits - padded_logits).abs().max().item()),
                )

                spare = config.max_seq_len - len(example.full_seq)
                if spare > 0:
                    suffix: list[int] = []
                    while len(suffix) < spare:
                        suffix.extend(filler_ids)
                    suffix_seq = example.full_seq + suffix[:spare]
                    suffix_logits = _logits_at_position(
                        model,
                        suffix_seq,
                        example.cue_position,
                        device,
                        pad_to_length=config.max_seq_len,
                        pad_id=config.pad_id,
                        causal_control_valid_length=len(example.full_seq),
                    )
                    max_suffix_logit_delta = max(
                        max_suffix_logit_delta,
                        float((prefix_logits - suffix_logits).abs().max().item()),
                    )

    if eval_mode is None:
        eval_mode = 'variable_prefix_kernel_compatible' if kernel_compatible else 'prefix_only'

    if eval_mode in {'kernel_compatible_prefix_only', 'variable_prefix_kernel_compatible'}:
        prefix_accuracy = passkey_accuracy_prefix_only(model, tokenizer, device, config, kernel_compatible=True)
        reported_mode = 'variable_prefix_kernel_compatible'
    elif eval_mode == 'prefix_only':
        prefix_accuracy = passkey_accuracy_prefix_only(model, tokenizer, device, config, kernel_compatible=False)
        reported_mode = 'prefix_only'
    elif eval_mode == 'fixed_length_causal_control':
        fixed_length = config.max_seq_len if fixed_length is None else int(fixed_length)
        prefix_accuracy = passkey_accuracy_fixed_length_causal_control(
            model,
            tokenizer,
            device,
            config,
            fixed_length=fixed_length,
            batch_mode=batch_mode,
        )
        reported_mode = f'fixed_{fixed_length}_causal_control_{batch_mode}'
    else:
        raise ValueError(f'unknown passkey eval_mode={eval_mode!r}')
    legacy_accuracy = legacy_padded_passkey_accuracy(model, tokenizer, device, config)

    return {
        'eval_mode': reported_mode,
        'legacy_eval_mode': 'kernel_compatible_prefix_only' if kernel_compatible else None,
        'prefix_consistent': (max_pad_logit_delta == 0.0 and max_suffix_logit_delta == 0.0),
        'max_pad_logit_delta': max_pad_logit_delta,
        'max_suffix_logit_delta': max_suffix_logit_delta,
        'examples_audited': total_examples,
        'batch_mode': batch_mode,
        'fixed_length': fixed_length,
        'prefix_accuracy': prefix_accuracy,
        'legacy_padded_accuracy': legacy_accuracy,
    }


def format_passkey_results(results: dict[int, float]) -> str:
    return '  '.join(f'd={distance}:{int(value * 100)}%' for distance, value in results.items())
