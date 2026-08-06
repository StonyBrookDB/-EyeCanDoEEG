"""Tests for ScoreAccumulator (eyecando.decode.accumulator).

Covers row/column push semantics (a flash's score lands on the whole
row or column, not one cell), accumulation across repeated pushes,
flattened scores() output, reset() behavior, and the prior-seeding
contract: shape/element-count validation, reshaping a flat prior into
the grid, taking a fresh prior per reset() call, and never mutating
the caller's original prior array.
"""

import numpy as np
import pytest

from eyecando.decode.accumulator import ScoreAccumulator


def test_constructor_stores_grid_size() -> None:
    acc = ScoreAccumulator(n_rows=6, n_cols=6)
    assert acc.n_rows == 6
    assert acc.n_cols == 6


def test_constructor_starts_at_zero() -> None:
    acc = ScoreAccumulator(n_rows=6, n_cols=6)
    assert acc.accum.shape == (6, 6)
    assert np.all(acc.accum == 0)


def test_push_row_flash_adds_to_whole_row() -> None:
    """flash_id 1-6 = rows -- a row flash's score should land on every
    symbol in that row, not just one cell."""
    acc = ScoreAccumulator()
    acc.push(flash_id=3, score=0.8)  # row index 2 (0-indexed)
    assert np.allclose(acc.accum[2, :], 0.8)
    assert np.allclose(acc.accum[:2, :], 0)
    assert np.allclose(acc.accum[3:, :], 0)


def test_push_col_flash_adds_to_whole_column() -> None:
    """flash_id 7-12 = columns -- a column flash's score should land on
    every symbol in that column."""
    acc = ScoreAccumulator()
    acc.push(flash_id=11, score=0.8)  # col index 4 (0-indexed)
    assert np.allclose(acc.accum[:, 4], 0.8)
    assert np.allclose(acc.accum[:, :4], 0)
    assert np.allclose(acc.accum[:, 5:], 0)


def test_push_accumulates_across_repeated_calls() -> None:
    acc = ScoreAccumulator()
    acc.push(flash_id=1, score=0.5)
    acc.push(flash_id=1, score=0.3)
    assert np.allclose(acc.accum[0, :], 0.8)


def test_scores_returns_flattened_grid() -> None:
    acc = ScoreAccumulator()
    acc.push(flash_id=1, score=1.0)
    acc.push(flash_id=7, score=1.0)
    scores = acc.scores()
    assert scores.shape == (36,)
    # symbol at row 0, col 0 got both a row and a column contribution.
    assert scores[0] == 2.0


def test_argmax_over_scores_identifies_true_target_symbol() -> None:
    """One full repetition: target is row index 2, col index 4 (flash_ids
    3 and 11). Target flashes get a strong positive score, all others a
    mild negative one -- argmax over the flattened grid should recover
    exactly that (row, col) pair."""
    acc = ScoreAccumulator()
    for row_id in range(1, 7):
        acc.push(row_id, 2.0 if row_id == 3 else -0.5)
    for col_id in range(7, 13):
        acc.push(col_id, 2.0 if col_id == 11 else -0.5)

    best_row, best_col = divmod(int(np.argmax(acc.scores())), 6)
    assert (best_row, best_col) == (2, 4)


def test_reset_clears_accumulated_scores() -> None:
    acc = ScoreAccumulator()
    acc.push(flash_id=1, score=5.0)
    acc.reset()
    assert acc.accum.shape == (6, 6)
    assert np.all(acc.accum == 0)


def test_prior_seeds_initial_accumulated_scores() -> None:
    """A prior (e.g. a language-model log-prior over symbols) should seed
    the grid instead of starting from zero -- pushes then accumulate on
    top of it."""
    prior = np.full((6, 6), 0.5)
    acc = ScoreAccumulator(prior=prior)
    assert np.allclose(acc.accum, 0.5)
    acc.push(flash_id=1, score=1.0)
    assert np.allclose(acc.accum[0, :], 1.5)
    assert np.allclose(acc.accum[1:, :], 0.5)


def test_mismatched_prior_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        ScoreAccumulator(n_rows=6, n_cols=6, prior=np.zeros((5, 6)))


def test_flat_prior_is_reshaped_to_grid() -> None:
    """CharLM.prior() (lm.py) returns a flat (n_symbols,) distribution --
    the accumulator should reshape it rather than require every caller to
    know the grid layout."""
    flat_prior = np.arange(36, dtype=float)
    acc = ScoreAccumulator(prior=flat_prior)
    assert acc.accum.shape == (6, 6)
    assert np.array_equal(acc.accum, flat_prior.reshape(6, 6))


def test_wrong_size_prior_still_raises() -> None:
    with pytest.raises(ValueError, match="elements"):
        ScoreAccumulator(prior=np.zeros(30))


def test_reset_without_prior_goes_to_zero() -> None:
    acc = ScoreAccumulator(prior=np.full((6, 6), 0.5))
    acc.push(flash_id=1, score=5.0)
    acc.reset()
    assert np.all(acc.accum == 0)


def test_reset_accepts_a_fresh_prior_each_call() -> None:
    """A prior is taken fresh per reset() call, not remembered from
    construction -- e.g. a language-model prior for the next symbol
    depends on what's already been typed, so it changes trial to trial."""
    acc = ScoreAccumulator(prior=np.full((6, 6), 0.5))
    acc.push(flash_id=1, score=5.0)

    next_prior = np.full((6, 6), 0.2)
    acc.reset(prior=next_prior)
    assert np.allclose(acc.accum, 0.2)


def test_reset_mismatched_prior_shape_raises() -> None:
    acc = ScoreAccumulator()
    with pytest.raises(ValueError, match="shape"):
        acc.reset(prior=np.zeros((5, 6)))


def test_prior_array_is_copied_not_aliased() -> None:
    """push() must never mutate the caller's original prior array."""
    prior = np.full((6, 6), 0.5)
    acc = ScoreAccumulator(prior=prior)
    acc.push(flash_id=1, score=5.0)
    assert np.allclose(prior, 0.5)
