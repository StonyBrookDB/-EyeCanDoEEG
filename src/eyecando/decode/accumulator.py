"""Row/column score accumulation for the P300 speller grid.

Standalone, no eyecando imports. See decode/README.md for how this
fits into the rest of the decode package.
"""

from __future__ import annotations

import numpy as np


class ScoreAccumulator:
    """Accumulates per-epoch classifier scores across repetitions.

    For a n_rows x n_cols (default: 6x6) grid with row-column flashing, each symbol appears in
    exactly one row and one column. Scores for the row and column
    containing each symbol are summed across rounds to produce a
    per-symbol accumulated score.
    """

    def __init__(self, n_rows: int = 6, n_cols: int = 6, prior: np.ndarray | None = None) -> None:
        """Construct a ScoreAccumulator with a zeroed (or prior-seeded) grid.

        Parameters
        ----------
        n_rows, n_cols : int
            Dimensions of the speller grid this accumulator tracks.
        prior : numpy.ndarray or None
            Initial per-symbol log-scores to seed the grid with (e.g. a
            language-model prior over the next symbol), shape
            ``(n_rows, n_cols)`` or any shape with ``n_rows * n_cols``
            elements (reshaped in row-major order). None (the default)
            starts from an all-zero grid.

        Raises
        ------
        ValueError
            If `prior` is given and its element count doesn't match
            ``n_rows * n_cols``.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.accum = self._build_accum(prior)

    def _build_accum(self, prior: np.ndarray | None) -> np.ndarray:
        if prior is None:
            return np.zeros((self.n_rows, self.n_cols))
        if prior.shape != (self.n_rows, self.n_cols):
            if prior.size != self.n_rows * self.n_cols:
                raise ValueError(
                    f"prior has shape {prior.shape} ({prior.size} elements), expected "
                    f"({self.n_rows}, {self.n_cols}) ({self.n_rows * self.n_cols} elements)"
                )
            prior = prior.reshape((self.n_rows, self.n_cols))
        return prior.copy()  # push() mutates in place -- never the caller's array

    def push(self, flash_id: int, score: float) -> None:
        """Record a classifier score for one flash event.

        Parameters
        ----------
        flash_id : int
            1-based row/column code (data.py's convention): ``1..n_rows``
            identifies a row flash, ``n_rows+1..n_rows+n_cols`` identifies
            a column flash. `score` is added to every cell in that row or
            column.
        score : float
            Classifier decision score for this flash, added to the
            running total of every symbol in the flashed row or column.
        """
        if flash_id > self.n_rows:
            self.accum[:, flash_id - self.n_rows - 1] += score
        else:
            self.accum[flash_id - 1, :] += score

    def scores(self) -> np.ndarray:
        """Return current accumulated scores.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_rows * n_cols,)``, flattened in row-major order --
            index ``i`` corresponds to grid cell ``(i // n_cols, i %
            n_cols)``.
        """
        return self.accum.flatten()

    def reset(self, prior: np.ndarray | None = None) -> None:
        """Reset accumulated scores for the next trial.

        `prior` is taken fresh each call -- e.g. a language-model
        prior over the next symbol depends on what's already
        been typed, so it changes trial to trial rather than
        staying fixed for the whole session.

        Parameters
        ----------
        prior : numpy.ndarray or None
            Same shape/reshape contract as `__init__`'s `prior`. None
            resets to an all-zero grid.

        Raises
        ------
        ValueError
            If `prior` is given and its element count doesn't match
            ``n_rows * n_cols``.
        """
        self.accum = self._build_accum(prior)
