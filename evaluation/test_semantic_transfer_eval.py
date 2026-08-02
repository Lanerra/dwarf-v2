from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from evaluation import semantic_choice_calibration as evaluator


@dataclass(frozen=True)
class FakeSemanticResult:
    family: str
    example_id: str
    choice_correct: bool
    predicted_choice: str
    target: str
    choice_nlls: dict[str, float]
    choice_token_ids: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeExample:
    family: str
    choices: tuple[str, ...]


def make_result(
    *,
    example_id: str,
    target: str,
    baseline_nlls: dict[str, float],
    evidence: float = 0.5,
) -> FakeSemanticResult:
    choice_nlls = dict(baseline_nlls)
    choice_nlls[target] -= evidence
    predicted = min(choice_nlls, key=choice_nlls.get)
    return FakeSemanticResult(
        family="context_binding",
        example_id=example_id,
        choice_correct=predicted == target,
        predicted_choice=predicted,
        target=target,
        choice_nlls=choice_nlls,
        choice_token_ids={
            choice: [index]
            for index, choice in enumerate(baseline_nlls, start=1)
        },
        metadata={"correct_choice": target},
    )


def calibration(baseline_nlls: dict[str, float]) -> dict:
    return {
        "protocol": "content_free_pmi_mean_nll_v1",
        "families": {
            "context_binding": {
                "prompts": ["No answer is provided.\nAnswer:"],
                "choice_baseline_nlls": baseline_nlls,
                "choice_token_counts": {choice: 1 for choice in baseline_nlls},
                "choice_token_ids": {
                    choice: [index]
                    for index, choice in enumerate(baseline_nlls, start=1)
                },
            }
        },
    }


def test_content_free_calibration_separates_binding_evidence_from_token_prior() -> None:
    baselines = {"red": 1.0, "blue": 2.0, "green": 3.0, "orange": 4.0}
    results = [
        make_result(example_id=f"example-{target}", target=target, baseline_nlls=baselines)
        for target in baselines
    ]

    assert sum(result.choice_correct is True for result in results) == 1

    calibrated = evaluator.calibrate_choice_results(results, calibration(baselines))

    assert len(calibrated) == 4
    assert all(result.calibrated_choice_correct for result in calibrated)
    assert {result.calibrated_predicted_choice for result in calibrated} == set(baselines)
    assert all(result.target_margin == pytest.approx(0.5) for result in calibrated)


def test_aggregate_reports_raw_and_calibrated_accuracy_separately() -> None:
    baselines = {"red": 1.0, "blue": 2.0, "green": 3.0, "orange": 4.0}
    results = [
        make_result(example_id=f"example-{target}", target=target, baseline_nlls=baselines)
        for target in baselines
    ]
    calibrated = evaluator.calibrate_choice_results(results, calibration(baselines))

    aggregate = evaluator.aggregate_calibrated_choice_results(calibrated)

    assert aggregate["protocol"] == "content_free_pmi_mean_nll_v1"
    assert aggregate["overall"]["count"] == 4
    assert aggregate["overall"]["raw_accuracy_same_rows"] == pytest.approx(0.25)
    assert aggregate["overall"]["accuracy"] == pytest.approx(1.0)
    assert aggregate["by_family"]["context_binding"]["correct"] == 4


def test_calibration_requires_every_scored_choice_baseline() -> None:
    baselines = {"red": 1.0, "blue": 2.0, "green": 3.0}
    result = make_result(
        example_id="missing-orange",
        target="orange",
        baseline_nlls={**baselines, "orange": 4.0},
    )

    with pytest.raises(ValueError, match="missing baseline.*orange"):
        evaluator.calibrate_choice_results([result], calibration(baselines))


def test_calibration_rejects_nonfinite_baseline() -> None:
    baselines = {"red": 1.0, "blue": 2.0, "green": math.inf, "orange": 4.0}
    result = make_result(
        example_id="nonfinite-green",
        target="green",
        baseline_nlls={"red": 1.0, "blue": 2.0, "green": 3.0, "orange": 4.0},
    )

    with pytest.raises(ValueError, match="finite"):
        evaluator.calibrate_choice_results([result], calibration(baselines))


def test_uncalibrated_family_is_excluded_instead_of_silently_treated_as_raw() -> None:
    result = FakeSemanticResult(
        family="two_hop_lightweight",
        example_id="two-hop",
        choice_correct=True,
        predicted_choice="Mira",
        target="Mira",
        choice_nlls={"Mira": 1.0, "Sol": 2.0},
        metadata={"correct_choice": "Mira"},
    )

    calibrated = evaluator.calibrate_choice_results(
        [result],
        calibration({"red": 1.0, "blue": 2.0}),
    )

    assert calibrated == []


def test_content_free_baselines_average_prompts_without_reading_targets() -> None:
    examples = [FakeExample("context_binding", ("red", "blue"))]
    prompts = {"context_binding": ("neutral-one", "neutral-two")}
    calls: list[tuple[str, str]] = []

    def score(prompt: str, choice: str) -> tuple[float, int, list[int]]:
        calls.append((prompt, choice))
        prompt_offset = 1.0 if prompt == "neutral-one" else 3.0
        choice_offset = 0.0 if choice == "red" else 2.0
        return prompt_offset + choice_offset, 1, [1]

    built = evaluator.build_content_free_choice_calibration(
        examples,
        score,
        prompts_by_family=prompts,
    )

    family = built["families"]["context_binding"]
    assert built["protocol"] == evaluator.CALIBRATION_PROTOCOL
    assert built["diagnostic_only"] is True
    assert built["raw_metrics_replaced"] is False
    assert family["choice_baseline_nlls"] == {"blue": 4.0, "red": 2.0}
    assert family["choice_prompt_nlls"] == {
        "blue": [3.0, 5.0],
        "red": [1.0, 3.0],
    }
    assert family["choice_token_ids"] == {"blue": [1], "red": [1]}
    assert calls == [
        ("neutral-one", "blue"),
        ("neutral-two", "blue"),
        ("neutral-one", "red"),
        ("neutral-two", "red"),
    ]


def test_content_free_baselines_fail_closed_on_context_dependent_tokenization() -> None:
    examples = [FakeExample("context_binding", ("red", "blue"))]

    def score(prompt: str, choice: str) -> tuple[float, int, list[int]]:
        del choice
        count = 1 if prompt == "neutral-one" else 2
        return 1.0, count, list(range(count))

    with pytest.raises(ValueError, match="token count changed"):
        evaluator.build_content_free_choice_calibration(
            examples,
            score,
            prompts_by_family={
                "context_binding": ("neutral-one", "neutral-two")
            },
        )


def test_content_free_baselines_reject_same_length_different_token_ids() -> None:
    examples = [FakeExample("context_binding", ("red", "blue"))]

    def score(prompt: str, choice: str) -> tuple[float, int, list[int]]:
        del choice
        return 1.0, 1, [1 if prompt == "neutral-one" else 2]

    with pytest.raises(ValueError, match="token IDs changed"):
        evaluator.build_content_free_choice_calibration(
            examples,
            score,
            prompts_by_family={
                "context_binding": ("neutral-one", "neutral-two")
            },
        )


def test_content_free_prompt_collection_rejects_a_bare_string() -> None:
    examples = [FakeExample("context_binding", ("red", "blue"))]

    with pytest.raises(ValueError, match="sequence of prompts"):
        evaluator.build_content_free_choice_calibration(
            examples,
            lambda prompt, choice: (1.0, 1),
            prompts_by_family={"context_binding": "not-a-prompt-list"},
        )


def test_calibration_rejects_missing_raw_prediction() -> None:
    result = FakeSemanticResult(
        family="context_binding",
        example_id="missing-raw-prediction",
        choice_correct=False,
        predicted_choice=None,  # type: ignore[arg-type]
        target="red",
        choice_nlls={"red": 1.0, "blue": 2.0},
        choice_token_ids={"red": [1], "blue": [2]},
        metadata={"correct_choice": "red"},
    )

    with pytest.raises(ValueError, match="raw predicted choice"):
        evaluator.calibrate_choice_results(
            [result],
            calibration({"red": 1.0, "blue": 2.0}),
        )


def test_calibration_rejects_inconsistent_raw_correctness() -> None:
    result = FakeSemanticResult(
        family="context_binding",
        example_id="inconsistent-raw-correctness",
        choice_correct=False,
        predicted_choice="red",
        target="red",
        choice_nlls={"red": 1.0, "blue": 2.0},
        choice_token_ids={"red": [1], "blue": [2]},
        metadata={"correct_choice": "red"},
    )

    with pytest.raises(ValueError, match="raw choice correctness"):
        evaluator.calibrate_choice_results(
            [result],
            calibration({"red": 1.0, "blue": 2.0}),
        )


def test_calibration_rejects_raw_and_baseline_token_id_mismatch() -> None:
    result = FakeSemanticResult(
        family="context_binding",
        example_id="token-id-mismatch",
        choice_correct=True,
        predicted_choice="red",
        target="red",
        choice_nlls={"red": 1.0, "blue": 2.0},
        choice_token_ids={"red": [99], "blue": [2]},
        metadata={"correct_choice": "red"},
    )

    with pytest.raises(ValueError, match="token IDs"):
        evaluator.calibrate_choice_results(
            [result],
            calibration({"red": 1.0, "blue": 2.0}),
        )
