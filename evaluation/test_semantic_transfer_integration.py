from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from evaluation import semantic_transfer_eval as evaluator


class WhitespaceTokenizer:
    pad_token_id = 0
    vocabulary = {
        "ctx": 1,
        "neutral": 2,
        "Answer:": 3,
        "red": 4,
        "blue": 5,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocabulary[token] for token in text.split()]


@dataclass(frozen=True)
class FakeExample:
    family: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class FakeResult:
    family: str
    example_id: str
    target: str
    choice_nlls: dict[str, float]
    choice_token_ids: dict[str, list[int]]
    predicted_choice: str
    choice_correct: bool
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "example_id": self.example_id,
            "target": self.target,
        }


def test_calibrated_suite_keeps_raw_metrics_and_rows_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = [FakeExample("context_binding", ("red", "blue"))]
    raw_result = FakeResult(
        family="context_binding",
        example_id="one",
        target="blue",
        choice_nlls={"red": 1.0, "blue": 1.5},
        choice_token_ids={"red": [1], "blue": [2]},
        predicted_choice="red",
        choice_correct=False,
        metadata={"correct_choice": "blue"},
    )
    raw_aggregate = {
        "overall": {"choice_accuracy": 0.0},
        "by_family": {"context_binding": {"choice_accuracy": 0.0}},
    }

    monkeypatch.setattr(evaluator, "evaluate_suite", lambda *args, **kwargs: [raw_result])
    monkeypatch.setattr(evaluator, "aggregate_results", lambda results: raw_aggregate)
    monkeypatch.setattr(
        evaluator,
        "score_target_continuation",
        lambda prompt, target, tokenizer, logits_hook: evaluator.ContinuationScore(
            nll={" red": 1.0, " blue": 2.0}[target],
            total_nll={" red": 1.0, " blue": 2.0}[target],
            token_count=1,
            token_ids={" red": [1], " blue": [2]}[target],
        ),
    )

    output = evaluator.evaluate_suite_with_content_free_calibration(
        examples,
        tokenizer=object(),
        logits_hook=lambda ids: None,
        prompts_by_family={"context_binding": ("neutral",)},
    )

    assert output["rows"] == [raw_result.to_dict()]
    assert output["aggregate"]["overall"] == raw_aggregate["overall"]
    assert output["aggregate"]["by_family"] == raw_aggregate["by_family"]
    diagnostic = output["aggregate"]["choice_calibration"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["raw_metrics_replaced"] is False
    assert diagnostic["overall"]["raw_accuracy_same_rows"] == 0.0
    assert diagnostic["overall"]["accuracy"] == 1.0
    assert output["scoring_revision"] == "raw_plus_content_free_pmi_v1"
    assert output["evaluator_source_hashes"] == evaluator.evaluator_source_hashes()


def test_real_position_sensitive_scorer_runs_through_calibrated_wrapper() -> None:
    tokenizer = WhitespaceTokenizer()
    example = evaluator.SemanticTransferExample(
        family="context_binding",
        example_id="position-sensitive",
        context="ctx",
        query="Answer:",
        target="blue",
        choices=("red", "blue"),
        paraphrase_group=None,
        metadata={"correct_choice": "blue"},
    )
    observed_inputs: list[list[int]] = []

    def logits_hook(input_ids: list[int]) -> torch.Tensor:
        observed_inputs.append(list(input_ids))
        logits = torch.full((1, len(input_ids), 6), -10.0)
        # If scoring starts one row too early, this row predicts blue and the
        # raw forced-choice assertion below fails instead of passing by chance.
        logits[0, 0, tokenizer.vocabulary["blue"]] = 5.0
        if input_ids[0] == tokenizer.vocabulary["ctx"]:
            # Context narrows the red prior but does not overcome it raw.
            logits[0, 1, tokenizer.vocabulary["red"]] = 2.0
            logits[0, 1, tokenizer.vocabulary["blue"]] = 1.5
        elif input_ids[0] == tokenizer.vocabulary["neutral"]:
            # Content-free scoring exposes the stronger unconditional red prior.
            logits[0, 1, tokenizer.vocabulary["red"]] = 2.0
            logits[0, 1, tokenizer.vocabulary["blue"]] = 1.0
        else:  # pragma: no cover - guards the test fixture itself
            raise AssertionError(f"unexpected prompt IDs: {input_ids}")
        return logits

    output = evaluator.evaluate_suite_with_content_free_calibration(
        [example],
        tokenizer,
        logits_hook,
        prompts_by_family={"context_binding": ("neutral Answer:",)},
    )

    row = output["rows"][0]
    calibrated = output["calibrated_rows"][0]
    assert row["predicted_choice"] == "red"
    assert row["choice_correct"] is False
    assert row["choice_token_ids"] == {"red": [4], "blue": [5]}
    assert calibrated["calibrated_predicted_choice"] == "blue"
    assert calibrated["calibrated_choice_correct"] is True
    assert calibrated["target_margin"] > 0.0
    assert [1, 3, 4] in observed_inputs
    assert [1, 3, 5] in observed_inputs
    assert [2, 3, 4] in observed_inputs
    assert [2, 3, 5] in observed_inputs


def test_deconfounded_suite_has_balanced_context_labels_and_exact_family_sizes() -> None:
    examples = evaluator.build_builtin_suite("builtin_v3_deconfounded", seed=0)
    family_counts = Counter(example.family for example in examples)
    context_examples = [
        example for example in examples if example.family == "context_binding"
    ]

    assert len(examples) == 504
    assert family_counts == {
        family: 72 for family in evaluator.REQUIRED_BUILTIN_FAMILIES
    }
    assert Counter(example.target for example in context_examples) == {
        "red": 12,
        "green": 12,
        "blue": 12,
        "yellow": 12,
        "purple": 12,
        "orange": 12,
    }
    assert all(
        example.choices is not None and len(example.choices) == 4
        for example in context_examples
    )


def test_evaluator_source_hashes_bind_both_implementation_files() -> None:
    observed = evaluator.evaluator_source_hashes()
    expected_paths = {
        "semantic_transfer_eval.py": Path(evaluator.__file__),
        "semantic_choice_calibration.py": Path(
            evaluator.semantic_choice_calibration.__file__
        ),
    }

    assert set(observed) == set(expected_paths)
    for name, path in expected_paths.items():
        assert observed[name] == hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = evaluator.evaluator_protocol_manifest()
    assert manifest["protocol"] == "content_free_pmi_mean_nll_v1"
    assert manifest["scoring_revision"] == "raw_plus_content_free_pmi_v1"
    assert manifest["evaluator_source_hashes"] == observed
    manifest_path = Path(evaluator.protocol_v1.__file__)
    assert manifest["manifest_source_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_expected_source_hash_mismatch_fails_before_model_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "evaluate_suite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model scoring must not start")
        ),
    )

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        evaluator.evaluate_suite_with_content_free_calibration(
            [FakeExample("context_binding", ("red", "blue"))],
            tokenizer=object(),
            logits_hook=lambda ids: None,
            prompts_by_family={"context_binding": ("neutral",)},
            expected_source_hashes={
                "semantic_transfer_eval.py": "0" * 64,
                "semantic_choice_calibration.py": "0" * 64,
            },
        )
