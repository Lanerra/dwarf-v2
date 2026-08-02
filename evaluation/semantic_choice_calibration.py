#!/usr/bin/env python3
"""Prior-calibrated diagnostics for semantic multiple-choice evaluation.

Raw continuation NLL remains the operational answer-production metric.  This
module adds a separate content-free PMI-style diagnostic so vocabulary priors
cannot be mistaken for gains or losses in contextual discrimination.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


CALIBRATION_PROTOCOL = "content_free_pmi_mean_nll_v1"
DEFAULT_CONTENT_FREE_PROMPTS: dict[str, tuple[str, ...]] = {
    "context_binding": (
        "Puzzle card: No color-to-role mapping is provided.\n"
        "In this puzzle, which color means the requested role?\nAnswer:",
        "Puzzle card: The correct color is not stated.\n"
        "In this puzzle, which color is the answer?\nAnswer:",
    ),
}


class ChoiceResultLike(Protocol):
    family: str
    example_id: str
    choice_correct: bool | None
    predicted_choice: str | None
    target: str
    choice_nlls: Mapping[str, float]
    choice_token_ids: Mapping[str, Sequence[int]]
    metadata: Mapping[str, Any]


class ChoiceExampleLike(Protocol):
    family: str
    choices: Sequence[str] | None


@dataclass(frozen=True)
class CalibratedChoiceResult:
    family: str
    example_id: str
    target: str
    raw_predicted_choice: str
    raw_choice_correct: bool
    calibrated_predicted_choice: str
    calibrated_choice_correct: bool
    calibrated_choice_scores: dict[str, float]
    target_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_finite(value: Any, *, description: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def _family_calibrations(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    if calibration.get("protocol") != CALIBRATION_PROTOCOL:
        raise ValueError(
            f"choice calibration protocol must be {CALIBRATION_PROTOCOL!r}"
        )
    families = calibration.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("choice calibration families must be a mapping")
    return families


def build_content_free_choice_calibration(
    examples: Iterable[ChoiceExampleLike],
    score_continuation: Callable[[str, str], Any],
    *,
    prompts_by_family: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Score candidate priors under label-free prompts.

    ``score_continuation`` receives ``(prompt, choice)`` and may return either a
    ``(mean_nll, token_count, token_ids)`` tuple or an object exposing ``nll``,
    ``token_count``, and ``token_ids`` attributes. Correct-answer labels are
    never inspected.
    """

    prompts = (
        DEFAULT_CONTENT_FREE_PROMPTS
        if prompts_by_family is None
        else prompts_by_family
    )
    choices_by_family: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if example.family not in prompts or example.choices is None:
            continue
        choices_by_family[example.family].update(str(choice) for choice in example.choices)

    families: dict[str, Any] = {}
    for family, family_prompts_value in sorted(prompts.items()):
        if isinstance(family_prompts_value, (str, bytes)):
            raise ValueError(
                f"content-free prompts for {family!r} must be a sequence of prompts"
            )
        family_prompts = [str(prompt) for prompt in family_prompts_value]
        if not family_prompts or any(not prompt for prompt in family_prompts):
            raise ValueError(f"content-free prompts for {family!r} must be non-empty")
        choices = sorted(choices_by_family.get(family, set()))
        if not choices:
            continue

        baseline_nlls: dict[str, float] = {}
        token_counts: dict[str, int] = {}
        token_ids: dict[str, list[int]] = {}
        prompt_nlls: dict[str, list[float]] = {}
        for choice in choices:
            nll_values: list[float] = []
            observed_token_counts: set[int] = set()
            observed_token_ids: set[tuple[int, ...]] = set()
            for prompt in family_prompts:
                scored = score_continuation(prompt, choice)
                if isinstance(scored, tuple):
                    if len(scored) != 3:
                        raise ValueError(
                            "content-free scorer tuples must contain "
                            "(mean_nll, token_count, token_ids)"
                        )
                    nll, token_count, scored_token_ids = scored
                else:
                    try:
                        nll = scored.nll
                        token_count = scored.token_count
                        scored_token_ids = scored.token_ids
                    except AttributeError as exc:
                        raise ValueError(
                            "content-free scorer results must include token IDs"
                        ) from exc
                nll_values.append(
                    _require_finite(nll, description=f"baseline NLL for {family}/{choice}")
                )
                count = int(token_count)
                ids = tuple(int(token_id) for token_id in scored_token_ids)
                if len(ids) != count:
                    raise ValueError(
                        f"token IDs/count mismatch for {family}/{choice}"
                    )
                observed_token_counts.add(count)
                observed_token_ids.add(ids)
            if len(observed_token_counts) != 1 or next(iter(observed_token_counts)) <= 0:
                raise ValueError(
                    f"content-free token count changed across prompts for {family}/{choice}"
                )
            if len(observed_token_ids) != 1:
                raise ValueError(
                    f"content-free token IDs changed across prompts for {family}/{choice}"
                )
            baseline_nlls[choice] = float(statistics.mean(nll_values))
            token_counts[choice] = next(iter(observed_token_counts))
            token_ids[choice] = list(next(iter(observed_token_ids)))
            prompt_nlls[choice] = nll_values

        families[family] = {
            "prompts": family_prompts,
            "choice_baseline_nlls": baseline_nlls,
            "choice_token_counts": token_counts,
            "choice_token_ids": token_ids,
            "choice_prompt_nlls": prompt_nlls,
        }

    if not families:
        raise ValueError("no semantic choices matched the requested calibration families")
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "diagnostic_only": True,
        "raw_metrics_replaced": False,
        "families": families,
    }


def calibrate_choice_results(
    results: Iterable[ChoiceResultLike],
    calibration: Mapping[str, Any],
) -> list[CalibratedChoiceResult]:
    """Subtract content-free candidate NLLs and rescore calibrated families."""

    families = _family_calibrations(calibration)
    output: list[CalibratedChoiceResult] = []
    for result in results:
        family_calibration = families.get(result.family)
        if family_calibration is None:
            continue
        if result.predicted_choice is None:
            raise ValueError(
                f"raw predicted choice is absent for {result.example_id!r}"
            )
        baselines = family_calibration.get("choice_baseline_nlls")
        if not isinstance(baselines, Mapping):
            raise ValueError(f"choice baselines for {result.family!r} must be a mapping")
        baseline_token_ids = family_calibration.get("choice_token_ids")
        if not isinstance(baseline_token_ids, Mapping):
            raise ValueError(
                f"choice calibration token IDs for {result.family!r} must be a mapping"
            )
        raw_token_ids = getattr(result, "choice_token_ids", None)
        if not isinstance(raw_token_ids, Mapping):
            raise ValueError(
                f"raw choice token IDs are absent for {result.example_id!r}"
            )
        if len(result.choice_nlls) < 2:
            raise ValueError(
                f"calibrated result {result.example_id!r} requires at least two choices"
            )

        scores: dict[str, float] = {}
        for choice, nll in result.choice_nlls.items():
            if choice not in baselines:
                raise ValueError(
                    f"missing baseline for choice {choice!r} in family {result.family!r}"
                )
            if choice not in baseline_token_ids or choice not in raw_token_ids:
                raise ValueError(
                    f"missing token IDs for choice {choice!r} in {result.example_id!r}"
                )
            expected_ids = [int(token_id) for token_id in baseline_token_ids[choice]]
            observed_ids = [int(token_id) for token_id in raw_token_ids[choice]]
            if observed_ids != expected_ids:
                raise ValueError(
                    f"raw and calibration token IDs differ for "
                    f"{result.example_id}/{choice}: {observed_ids} != {expected_ids}"
                )
            raw_nll = _require_finite(
                nll, description=f"raw choice NLL for {result.example_id}/{choice}"
            )
            baseline = _require_finite(
                baselines[choice],
                description=f"baseline NLL for {result.family}/{choice}",
            )
            scores[str(choice)] = raw_nll - baseline

        correct_choice = str(result.metadata.get("correct_choice", result.target))
        if correct_choice not in scores:
            raise ValueError(
                f"correct choice {correct_choice!r} is absent for {result.example_id!r}"
            )
        if not isinstance(result.choice_correct, bool):
            raise ValueError(
                f"raw choice correctness is absent for {result.example_id!r}"
            )
        expected_raw_correct = result.predicted_choice == correct_choice
        if result.choice_correct != expected_raw_correct:
            raise ValueError(
                f"raw choice correctness is inconsistent for {result.example_id!r}"
            )
        predicted = min(scores.items(), key=lambda item: (item[1], item[0]))[0]
        best_distractor = min(
            score for choice, score in scores.items() if choice != correct_choice
        )
        output.append(
            CalibratedChoiceResult(
                family=str(result.family),
                example_id=str(result.example_id),
                target=correct_choice,
                raw_predicted_choice=str(result.predicted_choice),
                raw_choice_correct=expected_raw_correct,
                calibrated_predicted_choice=predicted,
                calibrated_choice_correct=predicted == correct_choice,
                calibrated_choice_scores=scores,
                target_margin=best_distractor - scores[correct_choice],
            )
        )
    return output


def _summarise_calibrated(
    results: Sequence[CalibratedChoiceResult],
) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "correct": 0,
            "accuracy": None,
            "raw_correct_same_rows": 0,
            "raw_accuracy_same_rows": None,
            "mean_target_margin": None,
            "median_target_margin": None,
            "prediction_counts": {},
            "target_counts": {},
        }
    count = len(results)
    correct = sum(result.calibrated_choice_correct for result in results)
    raw_correct = sum(result.raw_choice_correct for result in results)
    margins = [result.target_margin for result in results]
    return {
        "count": count,
        "correct": correct,
        "accuracy": float(correct / count),
        "raw_correct_same_rows": raw_correct,
        "raw_accuracy_same_rows": float(raw_correct / count),
        "mean_target_margin": float(statistics.mean(margins)),
        "median_target_margin": float(statistics.median(margins)),
        "prediction_counts": dict(
            sorted(Counter(result.calibrated_predicted_choice for result in results).items())
        ),
        "target_counts": dict(sorted(Counter(result.target for result in results).items())),
    }


def aggregate_calibrated_choice_results(
    results: Sequence[CalibratedChoiceResult],
) -> dict[str, Any]:
    """Aggregate calibrated metrics without replacing raw evaluator metrics."""

    by_family: dict[str, list[CalibratedChoiceResult]] = defaultdict(list)
    for result in results:
        by_family[result.family].append(result)
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "diagnostic_only": True,
        "raw_metrics_replaced": False,
        "overall": _summarise_calibrated(results),
        "by_family": {
            family: _summarise_calibrated(items)
            for family, items in sorted(by_family.items())
        },
    }


__all__ = [
    "CALIBRATION_PROTOCOL",
    "DEFAULT_CONTENT_FREE_PROMPTS",
    "CalibratedChoiceResult",
    "aggregate_calibrated_choice_results",
    "build_content_free_choice_calibration",
    "calibrate_choice_results",
]
