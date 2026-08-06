"""Shared spatial alignment: Euclidean Alignment.

Pure transform, sklearn-compatible -- its only eyecando import is
ea_schedules.py's Schedule protocol/UniformAccumulation; nothing here
couples to any other pipeline stage. Whitens per-subject covariance to
reduce cross-subject variance before spatial filtering. Reference: He &
Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean Space
Data Alignment Approach" (2019).

Uses He & Wu's original *uncentered* second moment (X_i X_i^T / T, no
per-epoch mean subtraction) -- deliberately, not an oversight. See
pipeline/README.md before "fixing" this back to centered.
"""

from __future__ import annotations

import numpy as np

from eyecando.pipeline.ea_schedules import Schedule, UniformAccumulation


def _epoch_covariances(X: np.ndarray) -> np.ndarray:
    """Per-epoch covariance, shape (n_epochs, n_channels, n_channels)."""
    return np.einsum("ijk,ilk->ijl", X, X) / X.shape[-1]


def _whitening_from_cov(mean_cov: np.ndarray) -> np.ndarray:
    """Compute R^(-1/2) from an already-averaged covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    return eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.T


class EuclideanAligner:
    """Per-subject covariance whitening to reduce cross-subject variance.

    Fit on training subjects only. At inference time, a new subject's
    reference is built up over their calibration epochs via update() (see
    below) rather than being recomputed from scratch and discarded each
    calibration cycle.

    `schedule` decides how much weight a new update() batch gets relative
    to everything already accumulated -- see ea_schedules.py. Defaults to
    UniformAccumulation (every epoch ever seen counts equally), which is
    exactly the standard behavior for any offline processing.

    `reference_mean_cov`/`reference_n_epochs` optionally seed this aligner
    with an existing reference covariance (e.g. a population-level EA
    reference computed offline) and the effective epoch count it
    represents. When seeded, `schedule` is consulted starting on the
    *first* update() call (there's already a real running estimate to
    blend against) -- it's only skipped for an unseeded
    aligner's true first batch. transform() also becomes usable
    immediately after construction, before any update()/fit() call.

    Parameters
    ----------
    schedule : Schedule
        Merge-weight policy consulted by `update()` from its second call
        onward. Defaults to a shared `UniformAccumulation()` instance
        (see class docstring above).
    reference_mean_cov : numpy.ndarray or None
        Pre-existing reference covariance to seed this aligner with,
        shape ``(n_channels, n_channels)``. None (default) starts
        unfit -- `transform()` will raise until `fit()` or
        `update()` runs.
    reference_n_epochs : int
        Effective epoch count `reference_mean_cov` represents (used by
        `schedule` to weigh it against future `update()` batches).
        Must be 0 whenever `reference_mean_cov` is None.

    Raises
    ------
    ValueError
        If `reference_n_epochs` is greater than 0 but no
        `reference_mean_cov` was given.
    """

    def __init__(
        self,
        schedule: Schedule = UniformAccumulation(),
        reference_mean_cov: np.ndarray | None = None,
        reference_n_epochs: int = 0,
    ) -> None:
        if reference_n_epochs > 0 and reference_mean_cov is None:
            raise ValueError(
                "reference_n_epochs > 0 requires reference_mean_cov -- an epoch "
                "count with no matrix to attach it to isn't a valid reference"
            )
        self.schedule = schedule
        self._mean_cov = reference_mean_cov.copy() if reference_mean_cov is not None else None
        self._n_epochs = reference_n_epochs
        self.whitening_ = (
            _whitening_from_cov(self._mean_cov) if self._mean_cov is not None else None
        )

    def reference_state(self) -> tuple[np.ndarray | None, int]:
        """Return (mean_cov, n_epochs) -- this aligner's raw, schedule-
        agnostic state, as plain picklable data.

        For persisting alongside a saved model without embedding a whole
        EuclideanAligner object (and whatever `schedule` it happened to be
        constructed with) into that pickle -- reconstruct a fresh aligner
        from these two values plus reference_mean_cov=.../
        reference_n_epochs=... and whichever schedule the caller wants at
        load time.

        Returns
        -------
        tuple[numpy.ndarray or None, int]
            ``(mean_cov, n_epochs)``. ``mean_cov`` is shape
            ``(n_channels, n_channels)``, or None if this aligner has
            never been fit, updated, or seeded. ``n_epochs`` is the
            effective epoch count behind ``mean_cov`` (0 if ``mean_cov``
            is None).
        """
        mean_cov = self._mean_cov.copy() if self._mean_cov is not None else None
        return mean_cov, self._n_epochs

    def fit(self, X: np.ndarray) -> EuclideanAligner:
        """Compute reference covariance from training epochs (unsupervised).

        A direct, one-shot computation -- the plain mean covariance over
        every epoch in `X` at once, with no schedule involved at all (not
        even implicitly): this is the offline/batch case (what
        preprocess_p300 uses per session), where there's no notion of
        "new batch relative to what came before" for a schedule to weigh
        in the first place. Discards any previously accumulated or seeded
        state -- a fresh recomputation from `X` alone. Call
        update() instead of fit() if the goal is to keep building on
        previously seen (or seeded) data incrementally, schedule-weighted.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        EuclideanAligner
            ``self``, with `whitening_` now set from `X`'s mean covariance.

        Raises
        ------
        ValueError
            If `X` has 0 epochs.
        """
        if X.shape[0] == 0:
            raise ValueError(
                "EuclideanAligner.fit() received 0 epochs -- cannot compute a "
                "reference covariance from no data. This usually means upstream "
                "epoch rejection (preprocess_p300's own reject= threshold) "
                "dropped every epoch in this session; check the reject "
                "threshold is calibrated for this recording's actual noise "
                "level rather than assuming it. (Left unguarded, this produces "
                "a NaN mean covariance and fails several calls later with an "
                "opaque 'Eigenvalues did not converge' error instead.)"
            )
        self._mean_cov = _epoch_covariances(X).mean(axis=0)
        self._n_epochs = X.shape[0]
        self.whitening_ = _whitening_from_cov(self._mean_cov)
        return self

    def update(self, X: np.ndarray) -> EuclideanAligner:
        """Fold new epochs into the running mean covariance.

        The reference covariance is a weighted mean over epochs, with a
        per-epoch weight for this batch decided by `self.schedule`
        (UniformAccumulation by default, which makes this mathematically
        exact -- identical to recomputing from every epoch ever seen --
        without needing to keep old epochs in memory). This is what lets a
        live calibration session improve as more data arrives instead of
        discarding everything collected in prior cycles each time a new
        batch comes in.

        General merge rule, for any schedule: given per-epoch weights
        `w` (length n_batch) from `schedule.merge_weights()`,
            new_mean_cov = (1 - w.sum()) * old_mean_cov + sum_i(w[i] * cov_i)
        `1 - w.sum()` (the weight kept on the old running estimate) is a
        general identity, true for any schedule -- see ea_schedules.py's
        module docstring for the derivation.

        The very first batch (this aligner's `_mean_cov` still None) is
        assigned directly -- there's no running estimate yet to blend
        against, so the schedule is never consulted for it.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        EuclideanAligner
            ``self``, with `whitening_` refreshed from the new running
            mean covariance.
        """
        epoch_covs = _epoch_covariances(X)
        n_batch = X.shape[0]
        if self._mean_cov is None:
            self._mean_cov = epoch_covs.mean(axis=0)
            self._n_epochs = n_batch
        else:
            weights = self.schedule.merge_weights(self._n_epochs, n_batch)
            self._mean_cov = (1 - weights.sum()) * self._mean_cov + np.einsum(
                "i,ijk->jk", weights, epoch_covs
            )
            self._n_epochs += n_batch
        self.whitening_ = _whitening_from_cov(self._mean_cov)
        return self

    @staticmethod
    def apply_whitening(matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Apply a whitening matrix to epochs: matrix @ X_i for each epoch.

        Parameters
        ----------
        matrix : numpy.ndarray
            Whitening matrix, shape ``(n_channels, n_channels)``.
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            Whitened epochs, same shape as `X`.
        """
        return np.einsum("ij,njk->nik", matrix, X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored whitening matrix to new epochs.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            Whitened epochs, same shape as `X`.

        Raises
        ------
        RuntimeError
            If this aligner has not been fit, updated, or seeded with a
            reference covariance yet (`whitening_` is still None).
        """
        if self.whitening_ is None:
            raise RuntimeError("EuclideanAligner must be fit before transform")
        return EuclideanAligner.apply_whitening(self.whitening_, X)
