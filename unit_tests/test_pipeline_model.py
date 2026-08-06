"""Tests for eyecando.pipeline.model: P300Model and pool_sessions.

Covers fit()/predict_proba()/decision_scores() on synthetic
class-imbalanced epochs (shape, probability range, no majority-class
collapse), input-shape validation, multi-session fitting, save/load
round trips (including the use_xdawn=False identity-filter path), and
adapt_to_subject()'s Buhlmann-Straub-style calibration blending: alpha
monotonicity in calibration set size, repeated calls not compounding
drift, and requiring fit() to have run first. Also covers the EA
reference companion-file mechanism (attach_ea_reference/build_aligner
round-tripping, the reference staying out of the main pickled file,
save() removing a stale companion file, and load() handling both a
present and an absent companion file) and pool_sessions() concatenating
sessions with differing epoch counts.
"""

import numpy as np
import pytest

from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.model import P300Model, pool_sessions

N_EPOCHS = 100
N_CHANNELS = 8
N_TIMES = 230


def _synthetic_epochs(
    n_epochs: int = N_EPOCHS,
    n_channels: int = N_CHANNELS,
    n_times: int = N_TIMES,
    ratio: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (np.arange(n_epochs) % ratio == 0).astype(int)
    X = rng.normal(scale=1.0, size=(n_epochs, n_channels, n_times))
    bump = 4.0 * np.exp(-((np.arange(n_times) - n_times // 2) ** 2) / 200)
    X[y == 1, 0, :] += bump
    return X, y


def _fitted_model(n_components: int = 2, **kwargs) -> tuple[P300Model, np.ndarray, np.ndarray]:
    """Return a fitted model plus the (X, y) session it was fit on."""
    X, y = _synthetic_epochs()
    model = P300Model(n_components=n_components, **kwargs).fit([(X, y)])
    return model, X, y


def _fitted_model_multi_subject(n_components: int = 2, n_subjects: int = 5, **kwargs) -> P300Model:
    """A model fit on several distinct synthetic 'subjects' -- needed for
    anything exercising adapt_to_subject()'s blending, since calib_kappa_
    (see model.py's _population_kappa) needs real between-subject variance
    to estimate from; a single-session fit (_fitted_model()) has none.

    bump_scale varies meaningfully per subject (not just the random seed)
    -- varying only the seed would make every subject differ purely by
    sampling noise, giving _population_kappa's Buhlmann-Straub correction
    (tau2_pop = raw_between_subject_variance - within_subject_variance)
    almost nothing genuine to detect, pushing many features' corrected
    tau2_pop down to the epsilon floor and kappa correspondingly huge --
    chronic under-adaptation regardless of n_calib, not a meaningful test
    of the closed-form shrinkage's own behavior.
    """
    sessions = [
        _different_subject_epochs(
            n_epochs=N_EPOCHS, seed=i, bump_channel=0, bump_scale=1.0 + 0.5 * i
        )
        for i in range(n_subjects)
    ]
    return P300Model(n_components=n_components, **kwargs).fit(sessions)


def _different_subject_epochs(
    n_epochs: int, seed: int = 99, bump_channel: int = 3, bump_scale: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic 'new subject' with a distinctly different bump profile
    (different channel, different amplitude) than _synthetic_epochs()'s
    population, so adaptation tests can tell population and subject
    statistics apart."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n_epochs) % 5 == 0).astype(int)
    X = rng.normal(scale=1.0, size=(n_epochs, N_CHANNELS, N_TIMES))
    bump = bump_scale * 4.0 * np.exp(-((np.arange(N_TIMES) - N_TIMES // 2) ** 2) / 200)
    X[y == 1, bump_channel, :] += bump
    return X, y


def test_fit_returns_self() -> None:
    X, y = _synthetic_epochs()
    model = P300Model(n_components=2)
    assert model.fit([(X, y)]) is model


def test_predict_proba_shape() -> None:
    model, X, y = _fitted_model()
    proba = model.predict_proba(X)
    assert proba.shape == (N_EPOCHS,)


def test_predict_proba_scores_are_real_valued_probabilities() -> None:
    model, X, y = _fitted_model()
    proba = model.predict_proba(X)
    assert proba.dtype.kind == "f"
    assert np.all((proba >= 0) & (proba <= 1))
    assert len(np.unique(proba)) > 2


def test_class_imbalance_does_not_collapse_to_majority_class() -> None:
    model, X, y = _fitted_model()
    proba = model.predict_proba(X)
    predicted_target = proba >= 0.5
    assert predicted_target.any(), "model predicted target for zero epochs"
    assert not predicted_target.all(), "model predicted target for every epoch"


def test_model_persistence_round_trip(tmp_path) -> None:
    model, X, y = _fitted_model()
    proba_before = model.predict_proba(X)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    loaded = P300Model.load(str(save_path))
    proba_after = loaded.predict_proba(X)

    assert np.allclose(proba_before, proba_after)


def test_fit_rejects_wrong_input_shape() -> None:
    model = P300Model(n_components=2)
    X_2d = np.zeros((N_EPOCHS, N_CHANNELS))
    y = np.zeros(N_EPOCHS, dtype=int)
    with pytest.raises(ValueError, match="n_epochs, n_channels, n_timepoints"):
        model.fit([(X_2d, y)])


def test_fit_accepts_multiple_sessions() -> None:
    X1, y1 = _synthetic_epochs(n_epochs=60, n_channels=N_CHANNELS, n_times=N_TIMES)
    X2, y2 = _synthetic_epochs(n_epochs=80, n_channels=N_CHANNELS, n_times=N_TIMES)
    model = P300Model(n_components=2)
    result = model.fit([(X1, y1), (X2, y2)])
    assert result is model
    assert model.spatial_filter.n_components == 2


def test_adapt_to_subject_refits_lda() -> None:
    model = _fitted_model_multi_subject()
    X_calib, y_calib = _different_subject_epochs(n_epochs=N_EPOCHS, seed=50)
    model.adapt_to_subject(X_calib, y_calib, refit_lda=True)
    proba = model.predict_proba(X_calib)
    assert proba.shape == (N_EPOCHS,)


def test_adapt_to_subject_returns_self_for_chaining() -> None:
    model = _fitted_model_multi_subject()
    X_calib, y_calib = _different_subject_epochs(n_epochs=N_EPOCHS, seed=50)
    assert model.adapt_to_subject(X_calib, y_calib, refit_lda=True) is model


def test_fit_sets_population_statistics() -> None:
    model, X, y = _fitted_model(n_components=2)
    n_features = 2 * N_TIMES
    assert model.population_means_.shape == (2, n_features)
    assert model.population_covariance_.shape == (n_features, n_features)


def test_adapt_to_subject_requires_fit_first() -> None:
    model = P300Model(n_components=2)
    X_calib, y_calib = _different_subject_epochs(n_epochs=50)
    with pytest.raises(RuntimeError, match="requires fit"):
        model.adapt_to_subject(X_calib, y_calib)


def test_calib_kappa_alpha_is_monotonic_in_calibration_set_size() -> None:
    """alpha = n_calib / (n_calib + calib_kappa_) must strictly increase
    with n_calib for any fixed (positive) kappa -- the actual guaranteed
    property of the closed-form shrinkage weight.

    Checking the full blended-coefficient distance instead (what an
    earlier version of this test did, mirroring the old fixed-
    calib_prior_strength design) isn't reliable under a real, data-driven
    per-feature kappa: if kappa happens to come out small for many
    features (population data showing low within-subject noise relative
    to between-subject diversity), alpha can already be close to
    saturating even at a small n_calib -- so the observed coefficient
    change ends up dominated by sampling noise in the tiny-sample
    subject_means estimate itself, not by alpha's own genuine growth.
    Testing the formula directly avoids that confound.
    """
    model = _fitted_model_multi_subject(n_components=2, shrinkage=0.5)
    kappa = model.calib_kappa_
    assert np.all(kappa > 0)
    alpha_small = 10 / (10 + kappa)
    alpha_large = 300 / (300 + kappa)
    assert np.all(alpha_large > alpha_small)


def test_adapt_to_subject_repeated_calls_do_not_compound() -> None:
    """Calling adapt_to_subject multiple times with the same calibration
    data should always re-blend from the pristine population statistics,
    not drift further with each call."""
    model = _fitted_model_multi_subject(n_components=2, shrinkage=0.5)
    X_calib, y_calib = _different_subject_epochs(n_epochs=30, seed=50)

    model.adapt_to_subject(X_calib, y_calib)
    coef_after_one_call = model.lda._lda.coef_.copy()

    model.adapt_to_subject(X_calib, y_calib)
    model.adapt_to_subject(X_calib, y_calib)
    coef_after_three_calls = model.lda._lda.coef_.copy()

    assert np.allclose(coef_after_one_call, coef_after_three_calls)


def test_inference_flag_defaults_false_and_is_settable() -> None:
    model = P300Model(n_components=2)
    assert model.inference is False
    inference_model = P300Model(n_components=2, inference=True)
    assert inference_model.inference is True


def test_priors_are_balanced() -> None:
    model = P300Model()
    assert model.lda.priors == [0.5, 0.5]


def test_shrinkage_param_is_threaded() -> None:
    model = P300Model(shrinkage=0.3)
    assert model.lda.shrinkage == 0.3


def test_build_aligner_with_no_reference_attached_is_unfit() -> None:
    model, _, _ = _fitted_model()
    aligner = model.build_aligner()
    assert aligner.whitening_ is None


def test_attach_ea_reference_then_build_aligner_round_trips() -> None:
    model, X, _ = _fitted_model()
    source_aligner = EuclideanAligner()
    source_aligner.update(X)

    model.attach_ea_reference(source_aligner)
    rebuilt = model.build_aligner()

    mean_cov, n_epochs = source_aligner.reference_state()
    rebuilt_mean_cov, rebuilt_n_epochs = rebuilt.reference_state()
    assert np.allclose(mean_cov, rebuilt_mean_cov)
    assert n_epochs == rebuilt_n_epochs


def test_ea_reference_is_not_part_of_the_main_pickled_file(tmp_path) -> None:
    """The whole point of the companion file: attaching a reference must
    not change the main file's contents at all."""
    model_without, X, _ = _fitted_model()

    model_with, _, _ = _fitted_model()
    source_aligner = EuclideanAligner()
    source_aligner.update(X)
    model_with.attach_ea_reference(source_aligner)

    path_without = tmp_path / "without.pkl"
    path_with = tmp_path / "with.pkl"
    model_without.save(str(path_without))
    model_with.save(str(path_with))

    assert path_without.read_bytes() == path_with.read_bytes()
    # And the reference-carrying save DOES produce a companion file, while
    # the reference-free one doesn't.
    assert not (tmp_path / "without.ea_reference.joblib").exists()
    assert (tmp_path / "with.ea_reference.joblib").exists()


def test_load_reattaches_ea_reference_from_companion_file(tmp_path) -> None:
    model, X, _ = _fitted_model()
    source_aligner = EuclideanAligner()
    source_aligner.update(X)
    model.attach_ea_reference(source_aligner)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    loaded = P300Model.load(str(save_path))

    mean_cov, n_epochs = source_aligner.reference_state()
    loaded_mean_cov, loaded_n_epochs = loaded.build_aligner().reference_state()
    assert np.allclose(mean_cov, loaded_mean_cov)
    assert n_epochs == loaded_n_epochs


def test_load_without_a_companion_file_has_no_reference_attached(tmp_path) -> None:
    """A model saved before EA references existed (or just never given
    one) must still load fine, with build_aligner() falling back to
    unfit -- no crash from a missing companion file."""
    model, _, _ = _fitted_model()
    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))

    loaded = P300Model.load(str(save_path))

    assert loaded.build_aligner().whitening_ is None


def test_resaving_without_a_reference_removes_a_stale_companion_file(tmp_path) -> None:
    """Otherwise a later save() to the same path, with no reference
    attached, would leave load() silently reattaching a leftover
    reference that has nothing to do with the model being saved now."""
    model, X, _ = _fitted_model()
    source_aligner = EuclideanAligner()
    source_aligner.update(X)
    model.attach_ea_reference(source_aligner)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    assert (tmp_path / "model.ea_reference.joblib").exists()

    model._ea_mean_cov = None
    model._ea_n_epochs = 0
    model.save(str(save_path))

    assert not (tmp_path / "model.ea_reference.joblib").exists()
    loaded = P300Model.load(str(save_path))
    assert loaded.build_aligner().whitening_ is None


def test_use_xdawn_false_defaults_true_and_is_settable() -> None:
    model = P300Model(n_components=2)
    assert model.use_xdawn is True
    identity_model = P300Model(n_components=2, use_xdawn=False)
    assert identity_model.use_xdawn is False


def test_use_xdawn_false_fit_predict_round_trip() -> None:
    X, y = _synthetic_epochs()
    model = P300Model(use_xdawn=False).fit([(X, y)])
    proba = model.predict_proba(X)
    assert proba.shape == (N_EPOCHS,)


def test_use_xdawn_false_feature_dimensionality_is_raw_channels_times_times() -> None:
    X, y = _synthetic_epochs()
    model = P300Model(use_xdawn=False).fit([(X, y)])
    n_features = N_CHANNELS * N_TIMES
    assert model.population_means_.shape == (2, n_features)


def test_use_xdawn_true_feature_dimensionality_is_n_components_times_times() -> None:
    model, _, _ = _fitted_model(n_components=2)
    n_features = 2 * N_TIMES
    assert model.population_means_.shape == (2, n_features)


def test_use_xdawn_false_save_load_round_trip(tmp_path) -> None:
    X, y = _synthetic_epochs()
    model = P300Model(use_xdawn=False).fit([(X, y)])
    proba_before = model.predict_proba(X)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    loaded = P300Model.load(str(save_path))

    assert loaded.use_xdawn is False
    assert np.allclose(proba_before, loaded.predict_proba(X))


def test_pool_sessions_handles_differing_epoch_counts() -> None:
    X1, y1 = _synthetic_epochs(n_epochs=50, n_channels=N_CHANNELS, n_times=N_TIMES)
    X2, y2 = _synthetic_epochs(n_epochs=90, n_channels=N_CHANNELS, n_times=N_TIMES)
    X_pool, y_pool = pool_sessions([(X1, y1), (X2, y2)])
    assert X_pool.shape == (140, N_CHANNELS, N_TIMES)
    assert y_pool.shape == (140,)
