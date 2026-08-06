"""Tests for eyecando.pipeline.spatial.XDawnFilter.

Covers transform() output shape/dimensionality, raising before fit(),
and that transform() slices out the target-class (label 1) filter
block from mne's stacked per-class output rather than the nontarget
block -- checked directly against XdawnTransformer's own filters_
array since which class lands in which stacked position is an mne
implementation detail this wrapper depends on but doesn't control.
"""

import numpy as np
import pytest

from eyecando.pipeline.spatial import XDawnFilter


def _synthetic_epochs(
    n_epochs: int = 60, n_channels: int = 4, n_times: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (np.arange(n_epochs) % 5 == 0).astype(int)
    X = rng.normal(size=(n_epochs, n_channels, n_times))
    bump = 3.0 * np.exp(-((np.arange(n_times) - n_times // 2) ** 2) / 10)
    X[y == 1, 0, :] += bump
    return X, y


def test_transform_output_shape_keeps_components_and_timepoints() -> None:
    n_components = 2
    X, y = _synthetic_epochs(n_channels=4, n_times=20)
    sf = XDawnFilter(n_components=n_components)
    sf.fit(X, y)
    features = sf.transform(X)

    assert features.shape == (X.shape[0], n_components, X.shape[2])


def test_transform_before_fit_raises() -> None:
    X, _ = _synthetic_epochs()
    sf = XDawnFilter(n_components=2)
    with pytest.raises(RuntimeError):
        sf.transform(X)


def test_transform_is_3d() -> None:
    X, y = _synthetic_epochs()
    sf = XDawnFilter(n_components=2)
    sf.fit(X, y)
    features = sf.transform(X)
    assert features.ndim == 3


def test_transform_extracts_target_class_block_not_nontarget() -> None:
    """XDawnFilter.transform() must slice out class label 1's (target's)
    filter block from XdawnTransformer's stacked (n_components * n_classes,
    n_timepoints) output, not label 0's (nontarget's) -- verified directly
    against mne's own filters_ array rather than just trusting the slicing
    arithmetic, since np.unique(y) sorting (and hence which block ends up
    where) is an mne implementation detail this wrapper depends on but
    doesn't control. Guards against a future mne release changing that
    internal stacking order silently breaking this class's target_idx/
    target_block math with no test catching it (see spatial.py review
    2026-08-06)."""
    n_components = 2
    X, y = _synthetic_epochs(n_channels=4, n_times=20)
    sf = XDawnFilter(n_components=n_components)
    sf.fit(X, y)
    features = sf.transform(X)

    target_idx = list(sf._xdawn.classes_).index(1)
    nontarget_idx = list(sf._xdawn.classes_).index(0)
    target_filters = sf._xdawn.filters_[target_idx, :n_components, :]
    nontarget_filters = sf._xdawn.filters_[nontarget_idx, :n_components, :]

    expected_target_features = np.einsum("cf,nft->nct", target_filters, X)
    expected_nontarget_features = np.einsum("cf,nft->nct", nontarget_filters, X)

    assert np.allclose(features, expected_target_features)
    assert not np.allclose(features, expected_nontarget_features)
