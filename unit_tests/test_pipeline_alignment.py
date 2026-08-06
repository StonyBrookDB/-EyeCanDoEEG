"""Tests for eyecando.pipeline.alignment.EuclideanAligner.

Covers the core whitening contract (transform requires fit first;
fit() whitens mean epoch covariance to ~identity); update()'s
incremental-accumulation math (equivalent to fit() from scratch on a
fresh aligner, exactly matching a pooled fit() across batches, and
respecting a custom Schedule rather than hardcoding
UniformAccumulation); fit() discarding prior update()/seeded state;
reference_mean_cov/reference_n_epochs seeding (usable before any
update, blended via the schedule on the first real update, and
round-tripping through reference_state()); and shape preservation.
"""

import numpy as np
import pytest

from eyecando.pipeline.alignment import EuclideanAligner, _whitening_from_cov
from eyecando.pipeline.ea_schedules import UniformAccumulation


def _synthetic_epochs(
    n_epochs: int = 40, n_channels: int = 4, n_times: int = 30, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov_sqrt = rng.normal(size=(n_channels, n_channels))
    white = rng.normal(size=(n_epochs, n_channels, n_times))
    return np.einsum("ij,njk->nik", cov_sqrt, white)


def test_transform_before_fit_raises() -> None:
    aligner = EuclideanAligner()
    with pytest.raises(RuntimeError):
        aligner.transform(_synthetic_epochs())


def test_fit_then_transform_whitens_mean_covariance_to_identity() -> None:
    X = _synthetic_epochs()
    aligner = EuclideanAligner()
    aligner.fit(X)
    X_aligned = aligner.transform(X)

    covs = np.einsum("ijk,ilk->ijl", X_aligned, X_aligned) / X_aligned.shape[-1]
    mean_cov = covs.mean(axis=0)
    assert np.allclose(mean_cov, np.eye(X.shape[1]), atol=0.2)


def test_update_on_fresh_aligner_matches_fit() -> None:
    """update() with no prior state is equivalent to fit() -- the one-shot
    entry point is just update() from scratch."""
    X = _synthetic_epochs()
    via_update = EuclideanAligner().update(X)
    via_fit = EuclideanAligner().fit(X)
    assert np.allclose(via_update.whitening_, via_fit.whitening_)


def test_update_accumulates_equivalent_to_fitting_pooled_data() -> None:
    """Calling update() twice with two batches must give the same result as
    fitting once on the pooled batch -- the running mean covariance is
    exact, not an approximation, so accumulating incrementally shouldn't
    lose any information relative to keeping every epoch and refitting."""
    X1 = _synthetic_epochs(n_epochs=15, seed=1)
    X2 = _synthetic_epochs(n_epochs=25, seed=2)

    incremental = EuclideanAligner()
    incremental.update(X1)
    incremental.update(X2)

    pooled = EuclideanAligner()
    pooled.fit(np.concatenate([X1, X2], axis=0))

    assert np.allclose(incremental.whitening_, pooled.whitening_)


def test_fit_resets_prior_update_state() -> None:
    """fit() discards anything accumulated by earlier update() calls --
    it's the one-shot entry point, not another update()."""
    X1 = _synthetic_epochs(n_epochs=15, seed=1)
    X2 = _synthetic_epochs(n_epochs=25, seed=2)

    aligner = EuclideanAligner()
    aligner.update(X1)
    aligner.fit(X2)

    fresh = EuclideanAligner()
    fresh.fit(X2)

    assert np.allclose(aligner.whitening_, fresh.whitening_)


def test_apply_whitening_matches_transform() -> None:
    X = _synthetic_epochs()
    aligner = EuclideanAligner()
    aligner.fit(X)

    via_transform = aligner.transform(X)
    via_apply_whitening = EuclideanAligner.apply_whitening(aligner.whitening_, X)

    assert np.allclose(via_transform, via_apply_whitening)


def test_custom_schedule_is_actually_used() -> None:
    """A schedule that always returns weights summing to 1.0 makes update()
    fully replace the running covariance with the latest batch every time
    -- confirms EuclideanAligner really delegates to whatever schedule
    it's given, rather than always using UniformAccumulation regardless."""

    class _AlwaysReplace:
        def merge_weights(self, n_prior_epochs: int, n_batch_epochs: int) -> np.ndarray:
            return np.full(n_batch_epochs, 1.0 / n_batch_epochs)

    X1 = _synthetic_epochs(n_epochs=15, seed=1)
    X2 = _synthetic_epochs(n_epochs=25, seed=2)

    replaced = EuclideanAligner(schedule=_AlwaysReplace())
    replaced.update(X1)
    replaced.update(X2)

    only_X2 = EuclideanAligner()
    only_X2.update(X2)

    assert np.allclose(replaced.whitening_, only_X2.whitening_)


def test_seeded_reference_makes_transform_usable_before_any_update() -> None:
    X = _synthetic_epochs()
    reference_cov = np.eye(X.shape[1])
    aligner = EuclideanAligner(reference_mean_cov=reference_cov, reference_n_epochs=1600)

    # No fit()/update() call at all -- must already be usable.
    aligner.transform(X)  # must not raise


def test_seeded_reference_gets_a_real_merge_weight_on_the_first_update() -> None:
    """The whole point of seeding: the first update() call must blend
    against the seeded reference via the schedule, not silently overwrite
    it the way an unseeded aligner's true first batch does."""
    X = _synthetic_epochs(n_epochs=10, seed=3)
    reference_cov = np.eye(X.shape[1]) * 5.0
    reference_n_epochs = 1600

    aligner = EuclideanAligner(
        reference_mean_cov=reference_cov, reference_n_epochs=reference_n_epochs
    )
    aligner.update(X)

    # recompute the same way _epoch_covariances does: uncentered, /T
    epoch_covs = np.einsum("ijk,ilk->ijl", X, X) / X.shape[-1]
    weights = UniformAccumulation().merge_weights(reference_n_epochs, X.shape[0])
    expected_mean_cov = (1 - weights.sum()) * reference_cov + np.einsum(
        "i,ijk->jk", weights, epoch_covs
    )

    assert aligner._n_epochs == reference_n_epochs + X.shape[0]
    assert np.allclose(_whitening_from_cov(expected_mean_cov), aligner.whitening_)


def test_reference_n_epochs_without_reference_mean_cov_raises() -> None:
    with pytest.raises(ValueError):
        EuclideanAligner(reference_n_epochs=1600)


def test_fit_discards_a_seeded_reference() -> None:
    X = _synthetic_epochs(n_epochs=15, seed=4)
    seeded = EuclideanAligner(reference_mean_cov=np.eye(X.shape[1]) * 99.0, reference_n_epochs=1600)
    seeded.fit(X)

    fresh = EuclideanAligner()
    fresh.fit(X)

    assert np.allclose(seeded.whitening_, fresh.whitening_)


def test_reference_state_returns_raw_picklable_fields() -> None:
    X = _synthetic_epochs(n_epochs=15, seed=1)
    aligner = EuclideanAligner()
    aligner.update(X)

    mean_cov, n_epochs = aligner.reference_state()

    assert isinstance(mean_cov, np.ndarray)
    assert n_epochs == 15
    # Must be a copy, not a live view into the aligner's own state.
    mean_cov[0, 0] = 12345.0
    assert aligner._mean_cov[0, 0] != 12345.0


def test_reference_state_on_a_never_updated_aligner() -> None:
    assert EuclideanAligner().reference_state() == (None, 0)


def test_reconstructing_an_aligner_from_reference_state_matches_original() -> None:
    """The whole point of reference_state(): a fresh EuclideanAligner
    seeded from its output must behave identically to the original for
    the next update() call."""
    X1 = _synthetic_epochs(n_epochs=15, seed=1)
    X2 = _synthetic_epochs(n_epochs=10, seed=2)

    original = EuclideanAligner()
    original.update(X1)
    mean_cov, n_epochs = original.reference_state()

    reconstructed = EuclideanAligner(reference_mean_cov=mean_cov, reference_n_epochs=n_epochs)

    original.update(X2)
    reconstructed.update(X2)

    assert np.allclose(original.whitening_, reconstructed.whitening_)


def test_transform_preserves_shape() -> None:
    X = _synthetic_epochs(n_epochs=12, n_channels=3, n_times=20)
    aligner = EuclideanAligner()
    aligner.fit(X)
    X_aligned = aligner.transform(X)
    assert X_aligned.shape == X.shape
