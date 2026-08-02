#!/usr/bin/env python3
"""Core semantic-transfer metric helpers for DWARF.

This module is intentionally model-agnostic: callers provide a tokenizer-like object
and a logits hook for a causal LM.  The helpers score only continuation spans, so
prompt/context tokens are excluded from target NLL.
"""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from . import semantic_choice_calibration
from . import content_free_pmi_mean_nll_v1 as protocol_v1


REQUIRED_BUILTIN_FAMILIES: tuple[str, ...] = (
    "definition_application",
    "entity_swap_relation",
    "paraphrase_cloze",
    "context_binding",
    "two_hop_lightweight",
    "cause_effect",
    "tool_affordance",
)


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Any: ...


LogitsHook = Callable[[list[int]], torch.Tensor]


@dataclass(frozen=True)
class SemanticTransferExample:
    family: str
    example_id: str
    context: str
    query: str
    target: str
    choices: list[str] | None
    paraphrase_group: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationScore:
    nll: float
    total_nll: float
    token_count: int
    token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultipleChoiceScore:
    predicted_choice: str
    choice_nlls: dict[str, float]
    choice_token_counts: dict[str, int]
    choice_token_ids: dict[str, list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticTransferResult:
    family: str
    example_id: str
    target_nll: float
    target_total_nll: float
    target_token_count: int
    exact_correct: bool | None
    choice_correct: bool | None
    predicted_choice: str | None
    target: str
    paraphrase_group: str | None
    choice_nlls: dict[str, float] = field(default_factory=dict)
    choice_token_ids: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SuiteSeed:
    family: str
    context: str
    query: str
    target: str
    choices: tuple[str, ...]
    holdout_context: str
    holdout_query: str
    holdout_target: str
    holdout_choices: tuple[str, ...]


def _build_builtin_v1_suite() -> list[SemanticTransferExample]:
    """Return the original deterministic synthetic semantic-transfer examples.

    Each required family has a base example and a held-out variant.  The held-out
    metadata is a simple leakage guard for downstream train/eval splits; it is not
    meant to be a benchmark claim.
    """

    seeds: tuple[_SuiteSeed, ...] = (
        _SuiteSeed(
            family="definition_application",
            context="In this note, a dax is a small animal that glows at night.",
            query="Which named animal glows at night?",
            target="dax",
            choices=("dax", "wug", "mip"),
            holdout_context="In this note, a wug is a tool that folds copper sheets.",
            holdout_query="Which named tool folds copper sheets?",
            holdout_target="wug",
            holdout_choices=("dax", "wug", "mip"),
        ),
        _SuiteSeed(
            family="entity_swap_relation",
            context="Mira gave the amber key to Sol. Tavi kept the blue key.",
            query="Who has the amber key?",
            target="Sol",
            choices=("Mira", "Sol", "Tavi"),
            holdout_context="Mira gave the amber key to Tavi. Sol kept the blue key.",
            holdout_query="Who has the amber key?",
            holdout_target="Tavi",
            holdout_choices=("Mira", "Sol", "Tavi"),
        ),
        _SuiteSeed(
            family="paraphrase_cloze",
            context="The pilot landed after the storm because the runway lights returned.",
            query="The pilot could land because the ___ returned.",
            target="runway lights",
            choices=("runway lights", "cargo doors", "radio jokes"),
            holdout_context="After the storm, the runway lights came back, so the pilot landed.",
            holdout_query="The pilot landed once the ___ came back.",
            holdout_target="runway lights",
            holdout_choices=("runway lights", "cargo doors", "radio jokes"),
        ),
        _SuiteSeed(
            family="context_binding",
            context="For this puzzle only, red means safe and green means stop.",
            query="In this puzzle, which color means safe?",
            target="red",
            choices=("red", "green", "blue"),
            holdout_context="For this puzzle only, blue means safe and red means stop.",
            holdout_query="In this puzzle, which color means safe?",
            holdout_target="blue",
            holdout_choices=("red", "green", "blue"),
        ),
        _SuiteSeed(
            family="two_hop_lightweight",
            context="Nia owns the silver box. The silver box contains the orchard map.",
            query="Who owns the box containing the orchard map?",
            target="Nia",
            choices=("Nia", "Oren", "Pax"),
            holdout_context="Oren owns the silver box. The silver box contains the harbor map.",
            holdout_query="Who owns the box containing the harbor map?",
            holdout_target="Oren",
            holdout_choices=("Nia", "Oren", "Pax"),
        ),
        _SuiteSeed(
            family="cause_effect",
            context="The glass cracked because hot tea was poured into the frozen cup.",
            query="What caused the glass to crack?",
            target="hot tea was poured into the frozen cup",
            choices=("hot tea was poured into the frozen cup", "the spoon was missing", "the lamp was dim"),
            holdout_context="The seedlings wilted because no one watered the tray for a week.",
            holdout_query="What caused the seedlings to wilt?",
            holdout_target="no one watered the tray for a week",
            holdout_choices=("no one watered the tray for a week", "the tray was painted", "the labels were neat"),
        ),
        _SuiteSeed(
            family="tool_affordance",
            context="A flarn is used to tighten tiny brass screws without scratching them.",
            query="Which tool should tighten tiny brass screws?",
            target="flarn",
            choices=("flarn", "glimmer pan", "paper comb"),
            holdout_context="A plicket is used to measure rain inside narrow jars.",
            holdout_query="Which tool should measure rain inside narrow jars?",
            holdout_target="plicket",
            holdout_choices=("flarn", "plicket", "paper comb"),
        ),
    )

    examples: list[SemanticTransferExample] = []
    for family_index, seed in enumerate(seeds, start=1):
        examples.append(
            SemanticTransferExample(
                family=seed.family,
                example_id=f"builtin_v1.{family_index:02d}.{seed.family}.base",
                context=seed.context,
                query=seed.query,
                target=seed.target,
                choices=list(seed.choices),
                paraphrase_group=f"builtin_v1.{seed.family}",
                metadata={
                    "suite": "builtin_v1",
                    "split": "eval",
                    "variant": "base",
                    "correct_choice": seed.target,
                },
            )
        )
        examples.append(
            SemanticTransferExample(
                family=seed.family,
                example_id=f"builtin_v1.{family_index:02d}.{seed.family}.holdout",
                context=seed.holdout_context,
                query=seed.holdout_query,
                target=seed.holdout_target,
                choices=list(seed.holdout_choices),
                paraphrase_group=f"builtin_v1.{seed.family}",
                metadata={
                    "suite": "builtin_v1",
                    "split": "holdout",
                    "variant": "entity_or_paraphrase_holdout",
                    "correct_choice": seed.holdout_target,
                },
            )
        )
    return examples


def _rotated_choices(correct: str, distractors: Sequence[str], correct_index: int, choice_count: int = 4) -> list[str]:
    """Return choices with the correct answer in a deterministic rotated slot."""

    unique_distractors: list[str] = []
    for distractor in distractors:
        if distractor != correct and distractor not in unique_distractors:
            unique_distractors.append(distractor)
    if len(unique_distractors) < choice_count - 1:
        raise ValueError(f"not enough distractors for correct choice {correct!r}")

    choices = unique_distractors[: choice_count - 1]
    choices.insert(correct_index % choice_count, correct)
    return choices


def _build_builtin_v2_suite(*, suite: str, seed: int, examples_per_family: int) -> list[SemanticTransferExample]:
    """Generate deterministic semantic-transfer v2 examples.

    The generator intentionally uses synthetic names and rotating bindings rather
    than global randomness.  ``seed`` changes the deterministic offsets used for
    lexical choices and answer positions, while stable IDs keep the exact suite
    auditable and reproducible.
    """

    names = ("Mira", "Sol", "Tavi", "Nia", "Oren", "Pax", "Rhea", "Juno", "Kato", "Lio")
    objects = ("amber key", "blue key", "copper token", "silver badge", "orchard map", "harbor pass")
    colors = ("red", "green", "blue", "yellow", "purple", "orange")
    roles = ("safe", "stop", "inspect", "deliver", "archive", "launch")
    animals = ("dax", "wug", "mip", "zorb", "fen", "luma", "narl", "vex")
    properties = (
        "glows at night",
        "sleeps under warm stones",
        "hums when rain starts",
        "carries seeds in a silver pouch",
        "changes color near copper",
        "counts echoes in caves",
    )
    tools = ("flarn", "plicket", "sprol", "tevver", "marn", "clisk", "vindle", "dorv")
    tool_jobs = (
        "tighten tiny brass screws",
        "measure rain inside narrow jars",
        "polish lenses without scratching them",
        "fold copper sheets into neat corners",
        "lift hot tiles from the kiln",
        "trim soft wax seals cleanly",
    )
    places = ("orchard", "harbor", "library", "market", "observatory", "garden")
    containers = ("silver box", "green crate", "wooden tube", "amber case", "blue locker", "brass drawer")
    effects = (
        "the glass cracked",
        "the seedlings wilted",
        "the bell rang",
        "the lantern dimmed",
        "the gate opened",
        "the ink faded",
    )
    causes = (
        "hot tea was poured into the frozen cup",
        "no one watered the tray for a week",
        "the wind pushed the rope against the clapper",
        "the battery was nearly empty",
        "the guard turned the brass wheel twice",
        "sunlight hit the old label all afternoon",
    )
    bad_causes = (
        "the spoon was missing",
        "the labels were neat",
        "the carpet was blue",
        "the ladder was short",
        "the window was round",
        "the pencils were sorted",
    )

    def offset(index: int, modulo: int, salt: int = 0) -> int:
        return (index + seed * 3 + salt) % modulo

    def make_example(
        *,
        family_index: int,
        family: str,
        index: int,
        context: str,
        query: str,
        target: str,
        distractors: Sequence[str],
        variant: str,
        split: str = "eval",
        anti_memorization: str | None = None,
    ) -> SemanticTransferExample:
        correct_index = (index + seed + family_index) % 4
        metadata: dict[str, Any] = {
            "suite": suite,
            "split": split,
            "variant": variant,
            "correct_choice": target,
            "generator_seed": seed,
            "family_index": family_index,
        }
        if anti_memorization is not None:
            metadata["anti_memorization"] = anti_memorization
        return SemanticTransferExample(
            family=family,
            example_id=f"{suite}.s{seed:04d}.{family_index:02d}.{family}.{index + 1:03d}.{variant}",
            context=context,
            query=query,
            target=target,
            choices=_rotated_choices(target, distractors, correct_index),
            paraphrase_group=f"{suite}.{family}.{index // 2:03d}",
            metadata=metadata,
        )

    def build_definition_application(family_index: int, index: int) -> SemanticTransferExample:
        word = animals[offset(index, len(animals))]
        prop = properties[offset(index, len(properties), 1)]
        distractors = [item for item in animals if item != word]
        context = f"In this field note, a {word} is the animal that {prop}."
        query = f"Which named animal {prop}?"
        return make_example(
            family_index=family_index,
            family="definition_application",
            index=index,
            context=context,
            query=query,
            target=word,
            distractors=distractors,
            variant="definition_apply_generated",
            split="heldout" if index % 5 == 4 else "eval",
        )

    def build_entity_swap_relation(family_index: int, index: int) -> SemanticTransferExample:
        giver = names[offset(index, len(names))]
        receiver = names[offset(index, len(names), 2)]
        keeper = names[offset(index, len(names), 5)]
        item = objects[offset(index, len(objects))]
        other = objects[offset(index, len(objects), 1)]
        if index % 2 == 0:
            context = f"{giver} usually keeps the {item}, but not today. {giver} gave the {item} to {receiver} instead. {keeper} kept the {other}."
            variant = "entity_swap_context_conflict"
            anti = "entity_swap_conflict"
        else:
            context = f"{receiver} passed the {item} to {keeper}. {giver} watched and kept the {other}."
            receiver = keeper
            variant = "entity_swap_relation_generated"
            anti = None
        return make_example(
            family_index=family_index,
            family="entity_swap_relation",
            index=index,
            context=context,
            query=f"Who has the {item}?",
            target=receiver,
            distractors=names,
            variant=variant,
            split="heldout" if index % 5 == 4 else "eval",
            anti_memorization=anti,
        )

    def build_paraphrase_cloze(family_index: int, index: int) -> SemanticTransferExample:
        effect = effects[offset(index, len(effects))]
        cause = causes[offset(index, len(causes))]
        distractors = [item for item in causes + bad_causes if item != cause]
        if index % 2 == 0:
            context = f"{effect.capitalize()} because {cause}."
            query = f"{effect.capitalize()} because ___."
        else:
            context = f"Because {cause}, {effect}."
            query = f"The reason that {effect} was ___."
        return make_example(
            family_index=family_index,
            family="paraphrase_cloze",
            index=index,
            context=context,
            query=query,
            target=cause,
            distractors=distractors,
            variant="paraphrase_cloze_generated",
            split="heldout" if index % 5 == 4 else "eval",
        )

    def build_context_binding(family_index: int, index: int) -> SemanticTransferExample:
        target_color = colors[offset(index, len(colors))]
        decoy_color = colors[offset(index, len(colors), 1)]
        role = roles[offset(index, len(roles))]
        prior_role = roles[offset(index, len(roles), 1)]
        if index % 2 == 0:
            context = f"Usually {target_color} means {prior_role} and {decoy_color} means {role}. In this puzzle only, {target_color} means {role} and {decoy_color} means {prior_role}."
            variant = "context_prior_conflict"
            anti = "context_prior_conflict"
        else:
            context = f"For this puzzle only, {target_color} means {role}, while {decoy_color} means {prior_role}."
            variant = "context_binding_generated"
            anti = None
        return make_example(
            family_index=family_index,
            family="context_binding",
            index=index,
            context=context,
            query=f"In this puzzle, which color means {role}?",
            target=target_color,
            distractors=colors,
            variant=variant,
            split="heldout" if index % 5 == 4 else "eval",
            anti_memorization=anti,
        )

    def build_context_binding_deconfounded(family_index: int, index: int) -> SemanticTransferExample:
        target_i = offset(index, len(colors))
        role_i = (index // len(colors) + seed) % len(roles)
        decoy_step = 1 + ((index // (len(colors) * len(roles))) % (len(colors) - 1))
        decoy_i = (target_i + decoy_step) % len(colors)
        prior_step = 1 + ((index + seed) % (len(roles) - 1))
        prior_role_i = (role_i + prior_step) % len(roles)
        target_color = colors[target_i]
        decoy_color = colors[decoy_i]
        role = roles[role_i]
        prior_role = roles[prior_role_i]
        prior_conflict = ((index // len(colors)) + seed) % 2 == 0
        if prior_conflict:
            context = (
                f"Puzzle card {index + 1}: Normally {target_color} means {prior_role} and {decoy_color} means {role}. "
                f"For this puzzle only, {target_color} means {role} and {decoy_color} means {prior_role}."
            )
            variant = "context_prior_conflict_deconfounded"
            anti = "context_prior_conflict"
        else:
            context = (
                f"Puzzle card {index + 1}: For this puzzle only, {target_color} means {role}, "
                f"while {decoy_color} means {prior_role}. Ignore ordinary color meanings."
            )
            variant = "context_binding_deconfounded"
            anti = None
        remaining = [color for color in colors if color not in {target_color, decoy_color}]
        rot = (index + seed) % len(remaining)
        distractors = [decoy_color] + remaining[rot:] + remaining[:rot]
        example = make_example(
            family_index=family_index,
            family="context_binding",
            index=index,
            context=context,
            query=f"In this puzzle, which color means {role}?\nAnswer:",
            target=target_color,
            distractors=distractors,
            variant=variant,
            split="heldout" if index % 5 == 4 else "eval",
            anti_memorization=anti,
        )
        metadata = dict(example.metadata)
        metadata.update(
            {
                "eval_revision": "context_binding_deconfounded_v3",
                "target_color": target_color,
                "local_decoy_color": decoy_color,
                "role": role,
                "prior_role": prior_role,
                "prior_conflict": prior_conflict,
                "answer_cue": "Answer:",
            }
        )
        return SemanticTransferExample(
            family=example.family,
            example_id=example.example_id,
            context=example.context,
            query=example.query,
            target=example.target,
            choices=example.choices,
            paraphrase_group=f"{suite}.context_binding.{index:03d}",
            metadata=metadata,
        )

    def build_two_hop_lightweight(family_index: int, index: int) -> SemanticTransferExample:
        owner = names[offset(index, len(names))]
        other_owner = names[offset(index, len(names), 3)]
        box = containers[offset(index, len(containers))]
        place = places[offset(index, len(places))]
        context = f"{owner} owns the {box}. The {box} contains the {place} map. {other_owner} owns an empty spare box."
        return make_example(
            family_index=family_index,
            family="two_hop_lightweight",
            index=index,
            context=context,
            query=f"Who owns the container with the {place} map?",
            target=owner,
            distractors=names,
            variant="two_hop_lightweight_generated",
            split="heldout" if index % 5 == 4 else "eval",
        )

    def build_cause_effect(family_index: int, index: int) -> SemanticTransferExample:
        effect = effects[offset(index, len(effects))]
        cause = causes[offset(index, len(causes))]
        distractors = [item for item in causes + bad_causes if item != cause]
        context = f"{effect.capitalize()} because {cause}. The report says the timing was not a coincidence."
        return make_example(
            family_index=family_index,
            family="cause_effect",
            index=index,
            context=context,
            query=f"What caused that {effect}?",
            target=cause,
            distractors=distractors,
            variant="cause_effect_generated",
            split="heldout" if index % 5 == 4 else "eval",
        )

    def build_tool_affordance(family_index: int, index: int) -> SemanticTransferExample:
        tool = tools[offset(index, len(tools))]
        job = tool_jobs[offset(index, len(tool_jobs), 1)]
        distractors = [item for item in tools if item != tool]
        context = f"A {tool} is used to {job}. Other tools in the kit have different jobs."
        return make_example(
            family_index=family_index,
            family="tool_affordance",
            index=index,
            context=context,
            query=f"Which tool should {job}?",
            target=tool,
            distractors=distractors,
            variant="tool_affordance_generated",
            split="heldout" if index % 5 == 4 else "eval",
        )

    builders = {
        "definition_application": build_definition_application,
        "entity_swap_relation": build_entity_swap_relation,
        "paraphrase_cloze": build_paraphrase_cloze,
        "context_binding": build_context_binding_deconfounded if suite.startswith("builtin_v3_deconfounded") else build_context_binding,
        "two_hop_lightweight": build_two_hop_lightweight,
        "cause_effect": build_cause_effect,
        "tool_affordance": build_tool_affordance,
    }

    examples: list[SemanticTransferExample] = []
    for family_index, family in enumerate(REQUIRED_BUILTIN_FAMILIES, start=1):
        builder = builders[family]
        for index in range(examples_per_family):
            examples.append(builder(family_index, index))
    return examples


def build_builtin_suite(
    suite: str = "builtin_v1",
    seed: int = 0,
    examples_per_family: int | None = None,
) -> list[SemanticTransferExample]:
    """Return a deterministic built-in semantic-transfer suite.

    ``builtin_v1`` preserves the original 14 hand-written examples.  The v2
    suites are generated deterministically with balanced answer positions and
    stable IDs; callers may override ``examples_per_family`` for smaller shape
    tests or bespoke dry runs.
    """

    if suite == "builtin_v1":
        return _build_builtin_v1_suite()
    if suite == "builtin_v2_mini":
        count = 20 if examples_per_family is None else examples_per_family
        return _build_builtin_v2_suite(suite=suite, seed=seed, examples_per_family=count)
    if suite == "builtin_v2_full":
        count = 50 if examples_per_family is None else examples_per_family
        return _build_builtin_v2_suite(suite=suite, seed=seed, examples_per_family=count)
    if suite == "builtin_v3_deconfounded_mini":
        count = 24 if examples_per_family is None else examples_per_family
        return _build_builtin_v2_suite(suite=suite, seed=seed, examples_per_family=count)
    if suite == "builtin_v3_deconfounded":
        count = 72 if examples_per_family is None else examples_per_family
        return _build_builtin_v2_suite(suite=suite, seed=seed, examples_per_family=count)
    raise ValueError(f"unknown builtin semantic-transfer suite: {suite}")


def _encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    return [int(token_id) for token_id in encoded]


def _normalise_logits(logits: torch.Tensor, expected_seq_len: int) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"logits hook returned batch size {logits.shape[0]}, expected 1")
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"logits hook must return [seq, vocab] or [1, seq, vocab], got shape {tuple(logits.shape)}")
    if logits.shape[0] < max(0, expected_seq_len - 1):
        raise ValueError(f"logits sequence length {logits.shape[0]} is too short for {expected_seq_len} input ids")
    return logits


def _target_start(full_ids: Sequence[int], prompt_ids: Sequence[int], target_ids: Sequence[int]) -> int:
    if len(full_ids) >= len(prompt_ids) and list(full_ids[: len(prompt_ids)]) == list(prompt_ids):
        return len(prompt_ids)
    if len(target_ids) <= len(full_ids) and list(full_ids[-len(target_ids) :]) == list(target_ids):
        return len(full_ids) - len(target_ids)
    raise ValueError(
        "could not identify continuation token span; tokenizer.encode(prompt + target) "
        "must preserve either the prompt prefix or target suffix"
    )


def score_target_continuation(
    prompt: str,
    target: str,
    tokenizer: TokenizerLike,
    logits_hook: LogitsHook,
) -> ContinuationScore:
    """Score target continuation NLL under causal-LM logits, excluding prompt tokens.

    ``logits_hook`` is called with token ids for ``prompt + target`` and should
    return logits where row ``i`` predicts token ``i + 1``.
    """

    if target == "":
        raise ValueError("empty target continuation cannot be scored")

    prompt_ids = _encode(tokenizer, prompt)
    target_alone_ids = _encode(tokenizer, target)
    full_ids = _encode(tokenizer, prompt + target)
    if not target_alone_ids:
        raise ValueError("empty target continuation cannot be scored")
    if len(full_ids) < 2:
        raise ValueError("at least two total tokens are required for causal continuation scoring")

    start = _target_start(full_ids, prompt_ids, target_alone_ids)
    if start >= len(full_ids):
        raise ValueError("empty target continuation cannot be scored")
    if start == 0:
        raise ValueError("cannot score first continuation token without a prompt/BOS token")

    target_ids = list(full_ids[start:])
    logits = _normalise_logits(logits_hook(list(full_ids)), len(full_ids))

    nll_values: list[float] = []
    for token_index in range(start, len(full_ids)):
        token_id = int(full_ids[token_index])
        if token_id >= logits.shape[-1]:
            raise ValueError(f"target token id {token_id} is outside logits vocab dimension {logits.shape[-1]}")
        row = logits[token_index - 1].unsqueeze(0)
        target_tensor = torch.tensor([token_id], dtype=torch.long, device=row.device)
        nll_values.append(float(F.cross_entropy(row, target_tensor, reduction="sum").detach().cpu().item()))

    total_nll = float(sum(nll_values))
    token_count = len(nll_values)
    return ContinuationScore(
        nll=total_nll / token_count,
        total_nll=total_nll,
        token_count=token_count,
        token_ids=target_ids,
    )


def _as_continuation(text: str) -> str:
    if text == "":
        return text
    if text[0].isspace():
        return text
    return " " + text


def score_multiple_choice(
    prompt: str,
    choices: Sequence[str],
    tokenizer: TokenizerLike,
    logits_hook: LogitsHook,
) -> MultipleChoiceScore:
    """Score choices as continuations and return the lowest mean-NLL choice."""

    if not choices:
        raise ValueError("multiple-choice scoring requires at least one choice")

    choice_nlls: dict[str, float] = {}
    choice_token_counts: dict[str, int] = {}
    choice_token_ids: dict[str, list[int]] = {}
    for choice in choices:
        scored = score_target_continuation(prompt, _as_continuation(choice), tokenizer, logits_hook)
        choice_nlls[str(choice)] = scored.nll
        choice_token_counts[str(choice)] = scored.token_count
        choice_token_ids[str(choice)] = list(scored.token_ids)

    predicted_choice = min(choice_nlls.items(), key=lambda item: (item[1], item[0]))[0]
    return MultipleChoiceScore(
        predicted_choice=predicted_choice,
        choice_nlls=choice_nlls,
        choice_token_counts=choice_token_counts,
        choice_token_ids=choice_token_ids,
    )


def format_prompt(context: str, query: str) -> str:
    return f"{context}\n{query}"


def evaluate_example(
    example: SemanticTransferExample,
    tokenizer: TokenizerLike,
    logits_hook: LogitsHook,
) -> SemanticTransferResult:
    """Evaluate one semantic-transfer example using caller-supplied logits."""

    prompt = format_prompt(example.context, example.query)
    target_score = score_target_continuation(prompt, _as_continuation(example.target), tokenizer, logits_hook)

    predicted_choice: str | None = None
    choice_correct: bool | None = None
    exact_correct: bool | None = None
    choice_nlls: dict[str, float] = {}
    choice_token_ids: dict[str, list[int]] = {}
    if example.choices is not None:
        choice_score = score_multiple_choice(prompt, example.choices, tokenizer, logits_hook)
        predicted_choice = choice_score.predicted_choice
        choice_nlls = choice_score.choice_nlls
        choice_token_ids = choice_score.choice_token_ids
        correct_choice = str(example.metadata.get("correct_choice", example.target))
        choice_correct = predicted_choice == correct_choice
        exact_correct = predicted_choice == example.target

    return SemanticTransferResult(
        family=example.family,
        example_id=example.example_id,
        target_nll=target_score.nll,
        target_total_nll=target_score.total_nll,
        target_token_count=target_score.token_count,
        exact_correct=exact_correct,
        choice_correct=choice_correct,
        predicted_choice=predicted_choice,
        target=example.target,
        paraphrase_group=example.paraphrase_group,
        choice_nlls=choice_nlls,
        choice_token_ids=choice_token_ids,
        metadata=dict(example.metadata),
    )


def evaluate_suite(
    examples: Iterable[SemanticTransferExample],
    tokenizer: TokenizerLike,
    logits_hook: LogitsHook,
) -> list[SemanticTransferResult]:
    return [evaluate_example(example, tokenizer, logits_hook) for example in examples]


def evaluator_source_hashes() -> dict[str, str]:
    """Return hashes for every implementation file affecting calibrated scores."""

    paths = {
        "semantic_transfer_eval.py": Path(__file__).resolve(),
        "semantic_choice_calibration.py": Path(
            semantic_choice_calibration.__file__
        ).resolve(),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def evaluator_protocol_manifest() -> dict[str, Any]:
    """Load the committed protocol manifest used to bind evaluator execution."""

    manifest = copy.deepcopy(protocol_v1.PROTOCOL_MANIFEST)
    manifest["manifest_source_sha256"] = hashlib.sha256(
        Path(protocol_v1.__file__).read_bytes()
    ).hexdigest()
    if manifest.get("protocol") != semantic_choice_calibration.CALIBRATION_PROTOCOL:
        raise RuntimeError("semantic evaluator protocol manifest revision mismatch")
    if manifest.get("scoring_revision") != "raw_plus_content_free_pmi_v1":
        raise RuntimeError("semantic evaluator scoring revision mismatch")
    hashes = manifest.get("evaluator_source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("semantic evaluator protocol manifest lacks source hashes")
    return manifest


def evaluate_suite_with_content_free_calibration(
    examples: Iterable[SemanticTransferExample],
    tokenizer: TokenizerLike,
    logits_hook: LogitsHook,
    *,
    prompts_by_family: Mapping[str, Sequence[str]] | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate raw choices and a separately reported prior-calibrated diagnostic.

    Raw continuation-NLL metrics remain unchanged.  Calibration subtracts each
    candidate's mean NLL under label-free prompts and is reported under a
    distinct key so answer production is never conflated with contextual
    discrimination.
    """

    protocol_manifest = evaluator_protocol_manifest()
    source_hashes = evaluator_source_hashes()
    expected_hashes = (
        protocol_manifest["evaluator_source_hashes"]
        if expected_source_hashes is None
        else expected_source_hashes
    )
    expected = {
        str(name): str(digest) for name, digest in expected_hashes.items()
    }
    if source_hashes != expected:
        raise RuntimeError(
            "semantic evaluator source hash mismatch: "
            f"expected {expected!r}, observed {source_hashes!r}"
        )

    materialized_examples = list(examples)
    results = evaluate_suite(materialized_examples, tokenizer, logits_hook)
    aggregate = aggregate_results(results)
    calibration = semantic_choice_calibration.build_content_free_choice_calibration(
        materialized_examples,
        lambda prompt, choice: score_target_continuation(
            prompt,
            _as_continuation(choice),
            tokenizer,
            logits_hook,
        ),
        prompts_by_family=prompts_by_family,
    )
    calibrated_results = semantic_choice_calibration.calibrate_choice_results(
        results,
        calibration,
    )
    aggregate["choice_calibration"] = (
        semantic_choice_calibration.aggregate_calibrated_choice_results(
            calibrated_results
        )
    )
    return {
        "scoring_revision": "raw_plus_content_free_pmi_v1",
        "evaluator_source_hashes": source_hashes,
        "protocol_manifest": protocol_manifest,
        "examples": len(results),
        "aggregate": aggregate,
        "rows": [result.to_dict() for result in results],
        "choice_calibration": calibration,
        "calibrated_rows": [result.to_dict() for result in calibrated_results],
    }


def _accuracy(values: Sequence[bool | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(1 for value in present if value) / len(present))


def _summarise(results: Sequence[SemanticTransferResult]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "mean_target_nll": None,
            "token_weighted_target_nll": None,
            "target_token_count": 0,
            "exact_accuracy": None,
            "choice_accuracy": None,
        }

    total_tokens = sum(result.target_token_count for result in results)
    total_nll = sum(result.target_total_nll for result in results)
    return {
        "count": len(results),
        "mean_target_nll": float(sum(result.target_nll for result in results) / len(results)),
        "token_weighted_target_nll": float(total_nll / total_tokens) if total_tokens else None,
        "target_token_count": int(total_tokens),
        "exact_accuracy": _accuracy([result.exact_correct for result in results]),
        "choice_accuracy": _accuracy([result.choice_correct for result in results]),
    }


def _paraphrase_consistency(results: Sequence[SemanticTransferResult]) -> dict[str, Any]:
    grouped: dict[str, list[SemanticTransferResult]] = defaultdict(list)
    for result in results:
        if result.paraphrase_group:
            grouped[result.paraphrase_group].append(result)

    groups: dict[str, bool] = {}
    for group, group_results in sorted(grouped.items()):
        predicted = [result.predicted_choice for result in group_results]
        groups[group] = (
            all(result.choice_correct is True for result in group_results)
            and all(choice is not None for choice in predicted)
            and len(set(predicted)) == 1
        )

    group_count = len(groups)
    consistent_group_count = sum(1 for value in groups.values() if value)
    return {
        "group_count": group_count,
        "consistent_group_count": consistent_group_count,
        "accuracy": float(consistent_group_count / group_count) if group_count else None,
        "groups": groups,
    }


def aggregate_results(results: Sequence[SemanticTransferResult]) -> dict[str, Any]:
    """Aggregate semantic-transfer results overall, by family, and by paraphrase group."""

    by_family_lists: dict[str, list[SemanticTransferResult]] = defaultdict(list)
    for result in results:
        by_family_lists[result.family].append(result)

    return {
        "overall": _summarise(list(results)),
        "by_family": {family: _summarise(items) for family, items in sorted(by_family_lists.items())},
        "paraphrase_consistency": _paraphrase_consistency(list(results)),
    }


__all__ = [
    "ContinuationScore",
    "LogitsHook",
    "MultipleChoiceScore",
    "REQUIRED_BUILTIN_FAMILIES",
    "SemanticTransferExample",
    "SemanticTransferResult",
    "TokenizerLike",
    "aggregate_results",
    "build_builtin_suite",
    "evaluate_example",
    "evaluate_suite",
    "evaluate_suite_with_content_free_calibration",
    "evaluator_protocol_manifest",
    "evaluator_source_hashes",
    "format_prompt",
    "score_multiple_choice",
    "score_target_continuation",
]
