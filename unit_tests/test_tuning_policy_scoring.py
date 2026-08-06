"""Tests for eyecando.tuning.policy_scoring.selection_score.

Covers the hard accuracy floor (rejecting below MIN_ACCURACY as -inf,
accepting exactly at it), the additive accuracy_weight/seconds_weight
formula, the documented 2%-accuracy-per-extra-repetition breakeven
between ACCURACY_WEIGHT and SECONDS_WEIGHT, and monotonicity in both
accuracy (higher is better) and repetitions (fewer is better) at a
fixed value of the other.
"""

import numpy as np

from eyecando.tuning.policy_scoring import (
    ACCURACY_WEIGHT,
    MIN_ACCURACY,
    SECONDS_WEIGHT,
    selection_score,
)


def test_rejects_accuracy_below_floor() -> None:
    assert selection_score(mean_reps=5.0, accuracy=MIN_ACCURACY - 0.01) == -np.inf


def test_accepts_accuracy_at_floor() -> None:
    assert np.isfinite(selection_score(mean_reps=5.0, accuracy=MIN_ACCURACY))


def test_additive_weighting_formula() -> None:
    mean_reps, accuracy = 5.0, 0.95
    expected = ACCURACY_WEIGHT * accuracy - SECONDS_WEIGHT * (mean_reps * 12 * 0.25)
    assert selection_score(mean_reps, accuracy) == expected


def test_stated_breakeven_holds() -> None:
    """An extra repetition (3 seconds) should be worth exactly 2% more
    accuracy -- the documented breakeven this weighting was derived from."""
    assert np.isclose(ACCURACY_WEIGHT * 0.02, SECONDS_WEIGHT * 3.0)


def test_more_accuracy_scores_higher_at_fixed_reps() -> None:
    low = selection_score(mean_reps=5.0, accuracy=0.9)
    high = selection_score(mean_reps=5.0, accuracy=0.95)
    assert high > low


def test_more_reps_scores_lower_at_fixed_accuracy() -> None:
    fast = selection_score(mean_reps=3.0, accuracy=0.9)
    slow = selection_score(mean_reps=8.0, accuracy=0.9)
    assert fast > slow
