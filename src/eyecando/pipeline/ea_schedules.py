"""Pluggable merge-weight policies for EuclideanAligner's running covariance.

EuclideanAligner.update() blends a batch of new epochs' covariances into
its running estimate -- how much trust each new epoch gets relative to
everything seen so far. Those per-epoch weights are what a Schedule
computes; alignment.py owns everything else.

A Schedule holds only fixed hyperparameters set at construction, with no
running state of its own (the caller tracks epoch counts and passes them
in) -- safe to share as one instance across every EuclideanAligner() call
or concurrent session.

UniformAccumulation (P1), implemented here, is EuclideanAligner's
default and reproduces the fixed weighting it used before this module
existed, so no caller's behavior changes by not specifying a schedule.
P2 (RapidCalibration), P3 (FixedExponential), and P4 (TwoStageLifecycle)
are designed against this same interface but not yet implemented -- see
pipeline/README.md for the merge-weight derivation these all share.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Schedule(Protocol):
    """Structural contract a merge-weight policy must satisfy.

    merge_weights() returns an array of length n_batch_epochs: the weight
    given to each new epoch's covariance when blending it into the
    running estimate. `1 - weights.sum()` is the weight kept on the
    existing running estimate (see module docstring for why that's a
    general identity, not something each schedule computes itself).
    EuclideanAligner only ever calls this from its second update() call
    onward -- the very first batch is assigned directly (there's no
    running estimate yet to blend against).
    """

    def merge_weights(self, n_prior_epochs: int, n_batch_epochs: int) -> np.ndarray:
        """Return this batch's per-epoch blend weights.

        Parameters
        ----------
        n_prior_epochs : int
            Epochs already folded into the running estimate before this
            batch. Always >= 1 -- EuclideanAligner never calls this for a
            first batch (see class docstring).
        n_batch_epochs : int
            Epochs in the batch now being blended in.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_batch_epochs,)``. Weight for each epoch in the
            batch, in order. ``weights.sum()`` must be <= 1 -- the
            remainder, ``1 - weights.sum()``, is what stays on the
            existing running estimate (a general identity applied by the
            caller, not something an implementation computes itself).
        """
        ...


class UniformAccumulation(Schedule):
    """P1: every epoch ever seen counts equally toward the running mean.

    Every epoch in the batch gets weight 1 / (n_prior + n_batch) -- their
    combined share of the total epoch count seen so far -- which is what
    makes the running _mean_cov mathematically identical to recomputing
    the mean covariance from every epoch ever passed in, one batch at a
    time or all at once (see EuclideanAligner.update()'s own docstring).
    """

    def merge_weights(self, n_prior_epochs: int, n_batch_epochs: int) -> np.ndarray:
        """Give every epoch in the batch the same weight.

        Parameters
        ----------
        n_prior_epochs : int
            Epochs already folded into the running estimate.
        n_batch_epochs : int
            Epochs in the batch now being blended in.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_batch_epochs,)``, every entry equal to
            ``1 / (n_prior_epochs + n_batch_epochs)`` -- each epoch's
            share of the total ever seen, which is what makes the
            resulting running mean identical to recomputing over every
            epoch at once (see module docstring).
        """
        return np.full(n_batch_epochs, 1.0 / (n_prior_epochs + n_batch_epochs))
