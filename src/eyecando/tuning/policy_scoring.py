"""Stage 3 (live policy tuning) -- the selection_score utility function
scripts/simulate_dynamic_stopping.py uses to pick threshold/temperature/
lm_temperature.
"""

from __future__ import annotations

import numpy as np

from eyecando.pipeline.classifier import FLASHES_PER_ROUND, SECONDS_PER_FLASH

MIN_ACCURACY = 0.8  # P300 speller literature floor for "usable" per-selection accuracy

# Breakeven: an extra repetition (3 seconds -- FLASHES_PER_ROUND * SECONDS_PER_FLASH)
# is only worth it if it buys at least 2% more accuracy. ACCURACY_WEIGHT * 0.02 ==
# SECONDS_WEIGHT * 3, i.e. ACCURACY_WEIGHT / SECONDS_WEIGHT == 150.
ACCURACY_WEIGHT = 100.0  # utility points per unit of accuracy above the floor
SECONDS_WEIGHT = 100.0 / 150.0  # utility points lost per second of real elapsed time


def selection_score(
    mean_reps: float,
    accuracy: float,
    min_accuracy: float = MIN_ACCURACY,
    accuracy_weight: float = ACCURACY_WEIGHT,
    seconds_weight: float = SECONDS_WEIGHT,
) -> float:
    """Additive utility: accuracy_weight*accuracy - seconds_weight*seconds,
    hard-rejecting accuracy below `min_accuracy`.

    Deliberately not a ratio (unlike raw ITR, or accuracy/time): any
    ratio-shaped score structurally rewards shrinking the time spent once
    accuracy is "good enough," since the marginal value of more accuracy
    diminishes through the ratio while the cost of more time doesn't --
    that's true of ITR specifically (via the Wolpaw bits/selection term
    saturating well before accuracy=1) and would be just as true of a
    naive accuracy/time ratio. An additive score instead rewards every
    unit of accuracy above the floor at a constant, transparent rate, so
    "how many extra seconds is 1% more accuracy worth" is a real, tunable
    number instead of an emergent (and here, undesirable) side effect of
    the ratio's shape.

    Parameters
    ----------
    mean_reps : float
        Mean number of stimulus repetitions per selection under the
        policy being scored. Converted to elapsed seconds via
        ``mean_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH`` before being
        weighted by `seconds_weight`.
    accuracy : float
        Per-selection accuracy achieved under the policy being scored.
    min_accuracy : float
        Hard floor below which the policy is rejected outright regardless
        of how fast it is. Defaults to `MIN_ACCURACY`.
    accuracy_weight : float
        Utility points awarded per unit of accuracy above zero. Defaults
        to `ACCURACY_WEIGHT`.
    seconds_weight : float
        Utility points lost per second of real elapsed time. Defaults to
        `SECONDS_WEIGHT`, calibrated so that one extra repetition only
        pays for itself if it buys at least 2% more accuracy.

    Returns
    -------
    float
        The utility score, or ``-inf`` if `accuracy` is below
        `min_accuracy` -- callers doing a grid/argmax search over
        candidate policies can rely on the floor being enforced by this
        sentinel rather than needing a separate check.
    """
    if accuracy < min_accuracy:
        return -np.inf
    seconds = mean_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH
    return accuracy_weight * accuracy - seconds_weight * seconds
