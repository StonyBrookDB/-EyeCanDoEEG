"""Shared spatial filtering: xDAWN.

Pure transform, sklearn-compatible — no eyecando imports. Composed into
the full model by model.py's P300Model, which fits an `XDawnFilter` on
EA-normalized epochs and flattens its output before handing it to
`LDAClassifier` (lda.py) — this module has no opinion on EA or LDA,
only on the spatial projection itself.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import XdawnTransformer


class XDawnFilter:
    """Learn spatial filters that maximize P300 signal-to-noise ratio.

    Learns channel weightings that maximize target-vs-nontarget SNR. Fit
    on training epochs, applied as a fixed transform in both paths.

    Parameters
    ----------
    n_components : int
        Number of xDAWN spatial components to keep per class. Determines
        the channel dimension of `transform()`'s output.
    """

    def __init__(self, n_components: int = 9) -> None:
        self.n_components = n_components
        self._xdawn = XdawnTransformer(n_components=n_components)
        self._target_block: slice | None = None
        self._target_idx: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XDawnFilter:
        """Learn spatial filter weights from labeled training epochs.

        `y` is 1=target, 0=nontarget. mne's XdawnTransformer fits one
        spatial filter block per class, stacked as (n_components *
        n_classes, n_timepoints); only the target-class block is kept so
        transform() matches the (n_components, n_timepoints)-per-epoch
        contract instead of doubling it across both classes.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.
        y : numpy.ndarray
            Labels, shape ``(n_epochs,)``, 1=target / 0=nontarget.

        Returns
        -------
        XDawnFilter
            ``self``, fitted.
        """
        self._xdawn.fit(X, y)
        self._target_idx = list(self._xdawn.classes_).index(1)
        start = self._target_idx * self.n_components
        self._target_block = slice(start, start + self.n_components)
        return self

    @property
    def evals_(self) -> np.ndarray:
        """Full-rank generalized eigenvalues for the target class, descending.

        These are mne's own values from the underlying generalized
        eigenvalue problem xDAWN solves — not recomputed by hand. Length
        is always n_channels regardless of n_components, since fitting
        only truncates which eigenvectors are kept, not the
        decomposition itself.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_channels,)``, descending.

        Raises
        ------
        RuntimeError
            If this filter has not been fit yet.
        """
        if self._target_idx is None:
            raise RuntimeError("XDawnFilter must be fit before evals_ is available")
        assert self._xdawn.evals_ is not None  # set by _xdawn.fit(), alongside _target_idx
        return self._xdawn.evals_[self._target_idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply stored filter weights.

        Returns (n_epochs, n_components, n_timepoints). Callers that need a
        flat feature vector (e.g. for LDA) reshape it themselves — this
        class only does the spatial projection.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs, shape ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs, n_components, n_timepoints)``.

        Raises
        ------
        RuntimeError
            If this filter has not been fit yet.
        """
        if self._target_block is None:
            raise RuntimeError("XDawnFilter must be fit before transform")
        return self._xdawn.transform(X)[:, self._target_block, :]
