"""Tests for eyecando.pipeline.lda.LDAClassifier.

Covers construction (balanced priors, stored shrinkage param), fit()'s
fluent return, predict_proba()/decision_scores() output shape and
range on synthetic class-imbalanced features, that balanced priors
keep predictions from collapsing to the majority class, class_stats()
matching what fit() computes internally, and set_coefficients()
reproducing fit()'s own decision scores when given the same
means/covariance.
"""

import numpy as np

from eyecando.pipeline.lda import LDAClassifier

N_EPOCHS = 100
N_FEATURES = 6


def _synthetic_features(
    n_epochs: int = N_EPOCHS, n_features: int = N_FEATURES, ratio: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (np.arange(n_epochs) % ratio == 0).astype(int)
    X = rng.normal(size=(n_epochs, n_features))
    X[y == 1, 0] += 4.0
    return X, y


def test_fit_returns_self() -> None:
    X, y = _synthetic_features()
    clf = LDAClassifier()
    assert clf.fit(X, y) is clf


def test_priors_are_balanced() -> None:
    clf = LDAClassifier()
    assert clf.priors == [0.5, 0.5]


def test_shrinkage_param_is_stored() -> None:
    clf = LDAClassifier(shrinkage=0.3)
    assert clf.shrinkage == 0.3


def test_predict_proba_shape_and_range() -> None:
    X, y = _synthetic_features()
    clf = LDAClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_EPOCHS,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_class_imbalance_does_not_collapse_to_majority_class() -> None:
    X, y = _synthetic_features()
    clf = LDAClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    predicted_target = proba >= 0.5
    assert predicted_target.any(), "classifier predicted target for zero epochs"
    assert not predicted_target.all(), "classifier predicted target for every epoch"


def test_decision_scores_are_real_valued() -> None:
    X, y = _synthetic_features()
    clf = LDAClassifier().fit(X, y)
    scores = clf.decision_scores(X)
    assert scores.shape == (N_EPOCHS,)
    assert scores.dtype.kind == "f"


def test_class_stats_shapes() -> None:
    X, y = _synthetic_features()
    clf = LDAClassifier(shrinkage=0.5)
    means, covariance = clf.class_stats(X, y)
    assert means.shape == (2, N_FEATURES)
    assert covariance.shape == (N_FEATURES, N_FEATURES)


def test_class_stats_matches_fit_internally() -> None:
    """class_stats() should compute the exact same means_/covariance_ that
    fit() derives internally, since it's the same underlying computation."""
    X, y = _synthetic_features()
    clf = LDAClassifier(shrinkage=0.5).fit(X, y)
    means, covariance = clf.class_stats(X, y)
    assert np.allclose(means, clf._lda.means_)
    assert np.allclose(covariance, clf._lda.covariance_)


def test_set_coefficients_matches_fit() -> None:
    """Directly setting coefficients from a dataset's own means/covariance
    should reproduce the same decision scores fit() would have produced."""
    X, y = _synthetic_features()
    clf_fit = LDAClassifier(shrinkage=0.5).fit(X, y)
    scores_fit = clf_fit.decision_scores(X)

    clf_manual = LDAClassifier(shrinkage=0.5).fit(X, y)  # populates classes_
    means, covariance = clf_manual.class_stats(X, y)
    clf_manual.set_coefficients(means, covariance)
    scores_manual = clf_manual.decision_scores(X)

    assert np.allclose(scores_fit, scores_manual)
