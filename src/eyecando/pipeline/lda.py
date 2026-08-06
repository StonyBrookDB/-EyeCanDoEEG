"""Shared classifier: regularized, balanced-prior LDA.

Pure wrapper around sklearn's LinearDiscriminantAnalysis — no eyecando
imports, mirroring how alignment.py and spatial.py each isolate one
pipeline stage. Composed into the full model by model.py's P300Model,
which pairs an `LDAClassifier` with an `XDawnFilter` (spatial.py) and
feeds it flattened, xDAWN-projected, EA-normalized epochs — this module
has no opinion on any of those upstream steps, only on classifying
whatever flat feature vectors it's handed.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    _class_cov,
    _class_means,
)


class LDAClassifier:
    """LDA tuned for this pipeline's class imbalance and feature dimensionality.

    Uses `priors=[0.5, 0.5]` so the decision boundary reflects learned
    separability rather than the observed ~1:5/6 target:nontarget class
    frequency. `solver="eigen"` with shrinkage handles the rank-deficient
    covariance that results when feature dimension (n_components ×
    n_timepoints) exceeds available target-class trials -- deliberately
    not `solver="lsqr"`; see pipeline/README.md before switching this.

    Parameters
    ----------
    shrinkage : float or str
        Passed directly to sklearn's `LinearDiscriminantAnalysis`.
        ``"auto"`` (default) uses the Ledoit-Wolf lemma to pick shrinkage
        automatically; a float in ``[0, 1]`` fixes it explicitly.
    """

    def __init__(self, shrinkage: float | str = "auto") -> None:
        self.shrinkage = shrinkage
        self.priors = [0.5, 0.5]
        self._lda = LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage=shrinkage,
            priors=self.priors,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAClassifier:
        """Fit on flat (n_epochs, n_features) input. `y` is 1=target, 0=nontarget.

        Parameters
        ----------
        X : numpy.ndarray
            Flattened features, shape ``(n_epochs, n_features)``.
        y : numpy.ndarray
            Labels, shape ``(n_epochs,)``, 1=target / 0=nontarget.

        Returns
        -------
        LDAClassifier
            ``self``, fitted.
        """
        self._lda.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return target-class probability for each epoch.

        Parameters
        ----------
        X : numpy.ndarray
            Flattened features, shape ``(n_epochs, n_features)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``. Probability of the target class
            (label 1), not the full two-column sklearn output.
        """
        proba = self._lda.predict_proba(X)
        target_col = list(self._lda.classes_).index(1)
        return proba[:, target_col]

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return decision function scores. Positive = predict target.

        Parameters
        ----------
        X : numpy.ndarray
            Flattened features, shape ``(n_epochs, n_features)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``. Signed distance from the decision
            boundary -- positive predicts target, negative predicts
            nontarget. Unlike `predict_proba`, this is not bounded to
            [0, 1] and is what accumulation/AUC code should use.
        """
        return self._lda.decision_function(X)

    def class_stats(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-class means and shrunk pooled covariance for arbitrary (X, y).

        Uses this instance's own priors/shrinkage, mirroring exactly what
        sklearn computes internally during fit() -- but exposed standalone
        so a small calibration set's statistics can be computed without
        going through the solver step that requires invertibility (unlike
        the mean/covariance arithmetic here, which is always well-defined,
        the coef_/intercept_ solve is what can fail on tiny sample sizes).

        Parameters
        ----------
        X : numpy.ndarray
            Flattened features, shape ``(n_epochs, n_features)``.
        y : numpy.ndarray
            Labels, shape ``(n_epochs,)``, 1=target / 0=nontarget.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(means, covariance)``. ``means`` is shape
            ``(2, n_features)`` (nontarget row first, target row second,
            sklearn's class-sort order for labels 0/1). ``covariance`` is
            the shrunk pooled covariance, shape
            ``(n_features, n_features)``.
        """
        means = _class_means(X, y)
        covariance = _class_cov(X, y, self.priors, self.shrinkage)
        return means, covariance

    def set_coefficients(self, means: np.ndarray, covariance: np.ndarray) -> None:
        """Set coef_/intercept_ directly from given per-class means and
        shared covariance, bypassing fit() entirely.

        Implements the same Bayes-rule linear discriminant sklearn derives
        internally for binary classification: w = Sigma^-1(mu_1 - mu_0),
        b = -0.5*(mu_1+mu_0)^T w + log(pi_1/pi_0) (the log term is 0 here
        since priors are balanced). Uses a plain square linear solve
        rather than scipy.linalg.lstsq or the eigh-based generalized
        eigenproblem, since this never needs to handle rank deficiency --
        the covariance is expected to already be a valid, invertible
        estimate by the time it reaches here.

        Parameters
        ----------
        means : numpy.ndarray
            Shape ``(2, n_features)``, nontarget row first (index 0),
            target row second (index 1) -- matches `class_stats()`'s
            output order.
        covariance : numpy.ndarray
            Shared pooled covariance, shape ``(n_features, n_features)``.
            Must be invertible -- no rank-deficiency handling here.
        """
        mu0, mu1 = means[0], means[1]
        w = np.linalg.solve(covariance, mu1 - mu0)
        b = -0.5 * (mu0 + mu1) @ w
        self._lda.coef_ = w.reshape(1, -1)
        self._lda.intercept_ = np.array([b])
