"""Tests for LMDecoder. Loads the real figmtu/opt-350m-aac model once per
test session (module-scoped fixture) since a cold load takes real time;
these are integration tests against the real vendored algorithm, not
mocks.
"""

from __future__ import annotations

import string

import numpy as np
import pytest

from eyecando.decode.lm_decoder import LMDecoder

CHARACTER_SET = list(string.ascii_uppercase) + [" "]


@pytest.fixture(scope="module")
def decoder() -> LMDecoder:
    return LMDecoder(character_set=CHARACTER_SET)


def _assert_valid_log_prob_distribution(log_probs: np.ndarray) -> None:
    assert log_probs.shape == (len(CHARACTER_SET),)
    assert np.all(np.isfinite(log_probs))
    # These are natural-log probabilities -- every value must be <= 0
    # (allowing tiny float slop above exactly 0), and exponentiating must
    # recover a distribution that sums to 1. Neither check would catch a
    # regression to linear-space output on its own (a probability can also
    # be <= 1 and sum to 1) -- together they pin down log-space
    # specifically.
    assert np.all(log_probs <= 1e-9)
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)


def test_predict_log_probs_returns_valid_distribution_on_cold_start(decoder: LMDecoder) -> None:
    log_probs = decoder.predict_log_probs(evidence="")
    _assert_valid_log_prob_distribution(log_probs)


def test_predict_log_probs_returns_valid_distribution_mid_word(decoder: LMDecoder) -> None:
    log_probs = decoder.predict_log_probs(evidence="HELLO WORL")
    _assert_valid_log_prob_distribution(log_probs)

    # Sanity check the model is actually doing something sensible, not
    # just returning a valid-shaped distribution by accident: "WORL" is
    # missing exactly one letter to spell "WORLD".
    best_idx = int(np.argmax(log_probs))
    assert CHARACTER_SET[best_idx] == "D"


def test_predict_log_probs_output_is_aligned_to_character_set_order(decoder: LMDecoder) -> None:
    """LMDecoder must return log-probs in self.character_set's own order --
    achieved via predict_characters(..., sort_output=False), not by
    re-keying textslinger's own (potentially differently-ordered) output."""
    log_probs = decoder.predict_log_probs(evidence="HELLO WORL")
    d_idx = CHARACTER_SET.index("D")
    # "D" was shown to be the clear top prediction above -- confirm its
    # log-prob lands at its own character_set position, not position 0
    # (where a sorted-by-probability return would have put it).
    assert log_probs[d_idx] == log_probs.max()
    assert d_idx != 0
