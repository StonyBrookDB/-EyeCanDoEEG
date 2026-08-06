"""Tests for eyecando.decode.stopping: softmax, request_next, and
should_decode.

Covers softmax's normalization and rank-preservation properties;
request_next() forwarding context/temperature to a (faked) CharLM and
returning None when no LM is wired in; and should_decode()'s core
decision logic -- decoding once softmax confidence crosses threshold,
resetting the accumulator (optionally seeded with an LM prior keyed on
context + the just-decoded character), leaving the accumulator
untouched when not decoding, and temperature's effect on confidence
without changing which symbol wins.
"""

import numpy as np

from eyecando.decode.accumulator import ScoreAccumulator
from eyecando.decode.stopping import request_next, should_decode, softmax


class _FakeCharLM:
    """Satisfies stopping.py's _CharLM protocol without loading a real model."""

    def __init__(self, character_set: list[str], canned_prior: np.ndarray) -> None:
        self.character_set = character_set
        self.canned_prior = canned_prior
        self.calls: list[tuple[str, float]] = []

    def prior(self, context: str, temperature: float = 1.0) -> np.ndarray:
        self.calls.append((context, temperature))
        return self.canned_prior / temperature


def test_softmax_sums_to_one() -> None:
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(probs.sum(), 1.0)


def test_softmax_is_monotonic_with_input() -> None:
    """argmax(softmax(x)) must always match argmax(x) -- softmax should
    never change which symbol wins, only normalize the scores."""
    x = np.array([0.1, 5.0, -2.0, 0.2])
    assert np.argmax(softmax(x)) == np.argmax(x)


def test_request_next_returns_none_without_lm() -> None:
    assert request_next(lm=None, idx=0, context="") is None


def test_request_next_forwards_context_and_temperature_to_lm_prior() -> None:
    canned = np.zeros(4)
    lm = _FakeCharLM(character_set=list("ABCD"), canned_prior=canned)
    result = request_next(lm, idx=2, context="AB", lm_temperature=2.0)
    assert lm.calls == [("AB", 2.0)]
    assert result is not None
    assert np.array_equal(result, canned / 2.0)


def _confident_accumulator(target_flat_idx: int) -> ScoreAccumulator:
    """One repetition strongly favoring the symbol at `target_flat_idx`."""
    acc = ScoreAccumulator()
    target_row, target_col = divmod(target_flat_idx, acc.n_cols)
    for row_id in range(1, 7):
        acc.push(row_id, 5.0 if row_id - 1 == target_row else -5.0)
    for col_id in range(7, 13):
        acc.push(col_id, 5.0 if col_id - 7 == target_col else -5.0)
    return acc


def test_should_decode_true_and_correct_index_above_threshold() -> None:
    acc = _confident_accumulator(target_flat_idx=17)
    decode, idx = should_decode(acc, threshold=0.5)
    assert decode
    assert idx == 17


def test_should_decode_false_below_threshold_when_scores_are_uniform() -> None:
    acc = ScoreAccumulator()  # all-zero scores -> uniform softmax -> low confidence
    decode, _ = should_decode(acc, threshold=0.5)
    assert not decode


def test_should_decode_resets_accumulator_when_decoding() -> None:
    acc = _confident_accumulator(target_flat_idx=0)
    decode, _ = should_decode(acc, threshold=0.5)
    assert decode
    assert np.all(acc.accum == 0)  # lm=None -> request_next() returns None -> resets to zero


def test_should_decode_seeds_prior_and_appends_decoded_char_to_context() -> None:
    """With an lm wired in, should_decode must call it with context +
    the just-decoded character (not the pre-decode context), scaled by
    lm_temperature, and seed the accumulator with the result."""
    character_set = [chr(ord("A") + i) for i in range(36)]
    canned = np.full(36, 0.25)
    lm = _FakeCharLM(character_set=character_set, canned_prior=canned)

    acc = _confident_accumulator(target_flat_idx=5)
    decode, idx = should_decode(acc, threshold=0.5, lm=lm, context="HI", lm_temperature=2.0)

    assert decode
    assert idx == 5
    assert lm.calls == [("HI" + character_set[5], 2.0)]
    assert np.allclose(acc.accum, (canned / 2.0).reshape(6, 6))


def test_should_decode_leaves_accumulator_untouched_when_not_decoding() -> None:
    acc = ScoreAccumulator()
    acc.push(flash_id=1, score=0.1)
    before = acc.accum.copy()
    decode, _ = should_decode(acc, threshold=0.99)
    assert not decode
    assert np.array_equal(acc.accum, before)


def test_higher_temperature_reduces_confidence_and_can_delay_decode() -> None:
    """A higher temperature should never make should_decode MORE willing to
    decode -- it softens the softmax, so anything that decodes at T=1
    should require at least as high a threshold, or fail to decode, at a
    higher T."""
    acc_t1 = _confident_accumulator(target_flat_idx=5)
    decode_t1, idx_t1 = should_decode(acc_t1, threshold=0.9, temperature=1.0)

    acc_t5 = _confident_accumulator(target_flat_idx=5)
    decode_t5, idx_t5 = should_decode(acc_t5, threshold=0.9, temperature=5.0)

    assert decode_t1
    assert not decode_t5


def test_temperature_does_not_change_which_symbol_wins() -> None:
    """Temperature scaling is monotonic -- the winning index must be the
    same regardless of temperature, even when the decode decision differs."""
    acc = _confident_accumulator(target_flat_idx=9)
    _, idx_default = should_decode(acc, threshold=0.0, temperature=1.0)
    acc2 = _confident_accumulator(target_flat_idx=9)
    _, idx_hot = should_decode(acc2, threshold=0.0, temperature=10.0)
    assert idx_default == idx_hot == 9
