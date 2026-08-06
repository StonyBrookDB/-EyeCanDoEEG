"""Tests for eyecando.pipeline.ea_schedules.UniformAccumulation (the
Schedule protocol's P1 implementation).

Covers merge_weights()'s output shape, the 1/(n_prior+n_batch) uniform
weighting (including the full-weight case when there's no prior), and
that a shared instance carries no mutable state across calls -- the
property that makes it safe to reuse as EuclideanAligner's default.
"""

import numpy as np

from eyecando.pipeline.ea_schedules import UniformAccumulation


def test_uniform_accumulation_returns_one_weight_per_batch_epoch() -> None:
    weights = UniformAccumulation().merge_weights(n_prior_epochs=0, n_batch_epochs=5)
    assert weights.shape == (5,)


def test_uniform_accumulation_first_new_batch_after_no_prior_gets_full_weight() -> None:
    weights = UniformAccumulation().merge_weights(n_prior_epochs=0, n_batch_epochs=5)
    np.testing.assert_allclose(weights, np.full(5, 1.0 / 5))
    assert weights.sum() == 1.0


def test_uniform_accumulation_equal_prior_and_batch_splits_evenly() -> None:
    weights = UniformAccumulation().merge_weights(n_prior_epochs=5, n_batch_epochs=5)
    np.testing.assert_allclose(weights, np.full(5, 0.1))
    assert weights.sum() == 0.5


def test_uniform_accumulation_weight_is_constant_across_the_batch() -> None:
    weights = UniformAccumulation().merge_weights(n_prior_epochs=30, n_batch_epochs=10)
    np.testing.assert_allclose(weights, np.full(10, 1.0 / 40))
    assert weights.sum() == 0.25


def test_uniform_accumulation_carries_no_mutable_state() -> None:
    """Same instance, called repeatedly -- must be pure with respect to
    its own state, since sharing one default instance across every
    EuclideanAligner() call relies on this."""
    schedule = UniformAccumulation()
    first = schedule.merge_weights(0, 1)
    schedule.merge_weights(100, 1)  # unrelated call, must not affect the next line
    second = schedule.merge_weights(0, 1)
    np.testing.assert_allclose(first, second)
    assert first.item() == second.item() == 1.0
