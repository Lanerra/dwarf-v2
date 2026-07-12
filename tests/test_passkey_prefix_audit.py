import torch

from tools import passkey_eval


def test_prefix_audit_uses_fixed_geometry_and_causal_control_for_tail_comparisons(monkeypatch):
    config = passkey_eval.PasskeyConfig(
        max_seq_len=8,
        distances=[1],
        trials=1,
        batch_size=1,
        words=["apple"],
        filler_sentence=" filler",
        intro_template="secret {word}",
        retrieval_cue=" cue",
        pad_id=9,
    )
    example = passkey_eval.PasskeyExample(
        distance=1,
        target_word="apple",
        full_seq=[1, 2, 3],
        cue_position=2,
        candidate_token_ids=[4],
    )
    calls = []

    monkeypatch.setattr(passkey_eval, "build_passkey_examples", lambda *_: {1: [example]})
    monkeypatch.setattr(passkey_eval, "passkey_accuracy_fixed_length_causal_control", lambda *_args, **_kwargs: {1: 1.0})
    monkeypatch.setattr(passkey_eval, "legacy_padded_passkey_accuracy", lambda *_args, **_kwargs: {1: 1.0})

    def fake_logits(_model, ids, _position, _device, **kwargs):
        calls.append((list(ids), kwargs))
        control = kwargs.get("causal_control_valid_length")
        # A causal-control comparison must hide all tail-token identity changes.
        return torch.zeros(5) if control == len(example.full_seq) else torch.full((5,), float(len(ids)))

    monkeypatch.setattr(passkey_eval, "_logits_at_position", fake_logits)
    class Tokenizer:
        def encode(self, _text):
            return [7]

    audit = passkey_eval.passkey_prefix_consistency_audit(
        torch.nn.Module(), Tokenizer(), "cpu", config, eval_mode="fixed_length_causal_control"
    )

    assert audit["prefix_consistent"]
    assert audit["max_pad_logit_delta"] == 0.0
    assert audit["max_suffix_logit_delta"] == 0.0
    assert len(calls) == 3
    assert all(call[1]["pad_to_length"] == config.max_seq_len for call in calls)
    assert calls[0][1]["causal_control_valid_length"] == len(example.full_seq)
    assert calls[2][1]["causal_control_valid_length"] == len(example.full_seq)
