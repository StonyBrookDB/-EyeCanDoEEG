"""Shared model container: xDAWN + LDA.

Composes spatial.py (xDAWN) and lda.py (LDA) into a single serializable
P300 model usable by both offline training (classifier.py) and live
inference. Euclidean Alignment is applied upstream in preprocess_p300(),
so all data entering this model is already EA-normalized.

A model can optionally carry an EA reference too (attach_ea_reference()/
build_aligner()) -- e.g. a population-level EuclideanAligner state fit
alongside training, so a live session can seed its own aligner from it
instead of starting cold. That reference lives in its own companion
file (see save()/load()/__getstate__), not the main saved file, so a
model with no reference attached produces exactly the same main file
this class always has.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np

from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.ea_schedules import Schedule, UniformAccumulation
from eyecando.pipeline.lda import LDAClassifier
from eyecando.pipeline.spatial import XDawnFilter


class _IdentitySpatialFilter:
    """Drop-in stand-in for XDawnFilter that does no spatial filtering at
    all -- fit() is a no-op, transform() returns its input unchanged.

    Exists for the component ablation study, to isolate xDawn's own
    contribution: with this in place of XDawnFilter,
    fit()/predict_proba()/decision_scores() need no branching at all --
    self.spatial_filter is always one of the two, satisfying the same
    fit/transform protocol. Trivially joblib-picklable (no state at all).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> _IdentitySpatialFilter:
        """No-op -- there is no spatial filter to learn.

        Parameters
        ----------
        X : numpy.ndarray
            Accepted for interface compatibility with `XDawnFilter.fit`;
            unused.
        y : numpy.ndarray
            Accepted for interface compatibility; unused.

        Returns
        -------
        _IdentitySpatialFilter
            ``self``, matching `XDawnFilter.fit`'s own fluent-return
            convention.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return `X` unchanged.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of any shape.

        Returns
        -------
        numpy.ndarray
            The same array passed in, not a copy.
        """
        return X


def _ea_reference_path(path: str | Path) -> Path:
    """Companion file path for a model's EA reference state, alongside the
    main model file -- e.g. models/p300_classifier.pkl ->
    models/p300_classifier.ea_reference.joblib. A fully separate file, not
    a section of the main pickle -- see module docstring for why.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}.ea_reference.joblib")


def _validate_epochs_shape(X: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(
            f"expected X with shape (n_epochs, n_channels, n_timepoints), got ndim={X.ndim}"
        )


def _flatten(X: np.ndarray) -> np.ndarray:
    """Flatten (n_epochs, n_components, n_timepoints) for LDA input."""
    return X.reshape(X.shape[0], -1)


def _population_kappa(
    sessions: list[tuple[np.ndarray, np.ndarray]],
    spatial_filter,
    lda: LDAClassifier,
) -> np.ndarray:
    """Per-feature closed-form calibration shrinkage constant (Bühlmann-
    Straub / Efron-Morris empirical Bayes): kappa[j] = sigma2_within[j] /
    tau2_pop[j] -- how much to trust a new subject's own calibration
    estimate for feature j versus the population's. Computed once from
    `sessions` (the population's own per-subject training data, one (X, y)
    pair per subject -- the same list fit() receives, before pooling) --
    never the calibration/test subject's own data, since kappa describes a
    property of the population's between/within-subject variance
    structure, not anything about a specific new subject.

    Uses the already-fitted spatial_filter/lda's own class_stats() to get
    each training subject's own per-class means/covariance -- the exact
    same call adapt_to_subject() makes for a new subject, so kappa is
    estimated the same way it's later applied.

    tau2_pop is bias-corrected (Bühlmann-Straub's own defining step):
    sigma2_within is subtracted out of the naive subject-mean variance
    before use, then floored at a small epsilon (the correction can go
    negative). See pipeline/README.md if this is ever revisited for a
    dataset with few training subjects, where the correction matters most.
    """
    subject_means = []
    subject_within_var = []
    for X_s, y_s in sessions:
        X_features = _flatten(spatial_filter.transform(X_s))
        means_s, cov_s = lda.class_stats(X_features, y_s)
        subject_means.append(means_s.mean(axis=0))
        subject_within_var.append(np.diag(cov_s) / len(y_s))

    subject_means_arr = np.array(subject_means)  # (n_subjects, n_features)
    subject_within_var_arr = np.array(subject_within_var)  # (n_subjects, n_features)

    sigma2_within = subject_within_var_arr.mean(axis=0)
    # ddof=1 needs >=2 subjects; with exactly 1, there's no population
    # variance to estimate at all (tau2_raw is undefined, not just noisy) --
    # np.var's own ddof<=0 RuntimeWarning (a warnings.warn() call, not a
    # floating-point trap -- np.errstate doesn't cover it) is expected and
    # harmless here, so it's suppressed rather than left to spam every such
    # fit (real usage always pools multiple subjects; this only ever
    # triggers on a single-session test fixture that never calls
    # adapt_to_subject()).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            tau2_raw = subject_means_arr.var(axis=0, ddof=1)
    tau2_pop = np.clip(tau2_raw - sigma2_within, 1e-8, None)
    return sigma2_within / tau2_pop


def pool_sessions(
    sessions: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (X, y) pairs from multiple sessions/subjects into one pair.

    Every session produces the same timepoint count (verified across all
    subjects/sessions in BNCI2014_009 -- fixed tmin/tmax/sfreq epoching is
    deterministic here), so this is a plain concatenate; only epoch count
    is expected to vary across sessions.

    Parameters
    ----------
    sessions : list[tuple[numpy.ndarray, numpy.ndarray]]
        One ``(X, y)`` pair per session/subject. Each ``X`` is shape
        ``(n_epochs, n_channels, n_timepoints)`` (all sessions must share
        the same ``n_channels``/``n_timepoints``); each ``y`` is shape
        ``(n_epochs,)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(X_pool, y_pool)``, concatenated along axis 0 across every
        session in order.
    """
    X_pool = np.concatenate([X for X, _ in sessions], axis=0)
    y_pool = np.concatenate([y for _, y in sessions], axis=0)
    return X_pool, y_pool


class P300Model:
    """Fitted P300 classifier: XDawnFilter + LDA.

    This is the artifact that training produces and live inference loads.
    Both components travel together — never serialize them separately.

    Expects EA-normalized input (output of preprocess_p300). `y` is 1 for
    target, 0 for non-target. See lda.py for the LDA solver/priors/
    shrinkage choices.

    `inference` controls which pipeline branches activate during a run
    (e.g. whether live-inference-only surfacing is enabled). It is read
    by the live inference path (live/decoder.py); training code leaves
    it False.

    `use_xdawn=False` (for the component ablation study) swaps in
    _IdentitySpatialFilter instead of XDawnFilter -- LDA then sees flattened
    raw EA-normalized epochs directly (n_channels * n_timepoints features,
    not n_components * n_timepoints). `n_components` is ignored in that
    case (documented, not raised on) so a caller flipping this flag doesn't
    also need to change every other argument.

    Parameters
    ----------
    n_components : int
        xDAWN spatial components to keep per epoch. Ignored when
        `use_xdawn=False`.
    shrinkage : float or str
        Passed through to `LDAClassifier` -- see lda.py.
    inference : bool
        Whether this model is being used for live inference vs. training.
        Read by the live path (live/decoder.py); training code leaves it
        False.
    use_xdawn : bool
        True (default) applies xDAWN spatial filtering; False swaps in an
        identity transform for the component ablation study.
    """

    def __init__(
        self,
        n_components: int = 9,
        shrinkage: float | str = "auto",
        inference: bool = False,
        use_xdawn: bool = True,
    ) -> None:
        self.use_xdawn = use_xdawn
        self.spatial_filter = (
            XDawnFilter(n_components=n_components) if use_xdawn else _IdentitySpatialFilter()
        )
        self.lda = LDAClassifier(shrinkage=shrinkage)
        self.inference = inference
        self.population_means_: np.ndarray | None = None
        self.population_covariance_: np.ndarray | None = None
        self.calib_kappa_: np.ndarray | None = None
        # EA reference state (see module docstring) -- never part of
        # __getstate__'s pickled output, only ever written/read via the
        # companion file in save()/load().
        self._ea_mean_cov: np.ndarray | None = None
        self._ea_n_epochs: int = 0

    def fit(self, sessions: list[tuple[np.ndarray, np.ndarray]]) -> P300Model:
        """Fit xDAWN and LDA on a list of EA-normalized (X, y) session pairs.

        Parameters
        ----------
        sessions : list[tuple[numpy.ndarray, numpy.ndarray]]
            One ``(X, y)`` pair per session/subject. Each ``X`` is
            EA-normalized epochs, shape
            ``(n_epochs, n_channels, n_timepoints)``; each ``y`` is shape
            ``(n_epochs,)``, 1=target / 0=nontarget. Also used, before
            pooling, to estimate `calib_kappa_` (see `_population_kappa`).

        Returns
        -------
        P300Model
            ``self``, fitted. `population_means_`, `population_covariance_`,
            and `calib_kappa_` are populated as a side effect for later
            `adapt_to_subject()` calls.

        Raises
        ------
        ValueError
            If any session's ``X`` does not have ndim 3.
        """
        for X_s, _ in sessions:
            _validate_epochs_shape(X_s)
        X_pool, y_pool = pool_sessions(sessions)
        X_features = _flatten(self.spatial_filter.fit(X_pool, y_pool).transform(X_pool))
        self.lda.fit(X_features, y_pool)
        self.population_means_ = self.lda._lda.means_.copy()
        self.population_covariance_ = self.lda._lda.covariance_.copy()
        self.calib_kappa_ = _population_kappa(sessions, self.spatial_filter, self.lda)
        # sklearn's own covariance_/scalings_ are solver-internal artifacts,
        # dead weight past this point -- see pipeline/README.md for why
        # it's safe to clear them and what it saves.
        self.lda._lda.scalings_ = None
        self.lda._lda.covariance_ = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return target-class probability for each EA-normalized epoch.

        Parameters
        ----------
        X : numpy.ndarray
            EA-normalized epochs, shape
            ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``. Target-class probability per epoch.

        Raises
        ------
        ValueError
            If `X` does not have ndim 3.
        """
        _validate_epochs_shape(X)
        X_features = _flatten(self.spatial_filter.transform(X))
        return self.lda.predict_proba(X_features)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Return LDA decision function scores for each EA-normalized epoch.

        Positive = predict target. Use this for accumulation across
        repetitions and AUC evaluation, not predict_proba.

        Parameters
        ----------
        X : numpy.ndarray
            EA-normalized epochs, shape
            ``(n_epochs, n_channels, n_timepoints)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_epochs,)``. Signed decision score per epoch.

        Raises
        ------
        ValueError
            If `X` does not have ndim 3.
        """
        _validate_epochs_shape(X)
        X_features = _flatten(self.spatial_filter.transform(X))
        return self.lda.decision_scores(X_features)

    def adapt_to_subject(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        refit_lda: bool = True,
    ) -> P300Model:
        """Blend population statistics with a new subject's calibration data.

        Rather than discarding the population fit and refitting LDA on the
        (typically small) calibration set alone, blend the subject-specific
        means/covariance toward the population ones this model was trained
        with -- always re-blending from the same pristine population
        statistics captured in fit(), never from a previously-adapted
        state, so repeated calibration calls don't compound drift.

        The shrinkage weight comes from `self.calib_kappa_` (see
        `_population_kappa()`), not a tunable argument passed in here.
        `alpha[j] = n_calib / (n_calib + calib_kappa_[j])` applies
        per-feature to the means blend; the covariance blend uses a
        single pooled `alpha` (`calib_kappa_.mean()`) instead, since a
        per-feature vector doesn't map onto covariance's feature-pair
        off-diagonal entries the same way.

        Parameters
        ----------
        X_calib : numpy.ndarray
            EA-normalized calibration epochs for the new subject, shape
            ``(n_epochs, n_channels, n_timepoints)``.
        y_calib : numpy.ndarray
            Labels, shape ``(n_epochs,)``, 1=target / 0=nontarget.
        refit_lda : bool
            If False, this call is a no-op -- `X_calib`/`y_calib` are
            unused and the model's existing coefficients are left as-is.

        Returns
        -------
        P300Model
            ``self``, with LDA coefficients blended toward the subject's
            calibration data (when `refit_lda` is True).

        Raises
        ------
        RuntimeError
            If `refit_lda` is True but `fit()` has not been called yet
            (no population statistics to blend against).
        ValueError
            If `refit_lda` is True and `X_calib` does not have ndim 3.
        """
        if refit_lda:
            _validate_epochs_shape(X_calib)
            if self.population_means_ is None:
                raise RuntimeError("adapt_to_subject requires fit() to have run first")
            assert self.calib_kappa_ is not None  # set alongside population_means_ by fit()
            X_features = _flatten(self.spatial_filter.transform(X_calib))
            subject_means, subject_covariance = self.lda.class_stats(X_features, y_calib)
            n_calib = len(y_calib)
            alpha = n_calib / (n_calib + self.calib_kappa_)
            blended_means = (1 - alpha) * self.population_means_ + alpha * subject_means
            alpha_cov = n_calib / (n_calib + self.calib_kappa_.mean())
            blended_covariance = (
                1 - alpha_cov
            ) * self.population_covariance_ + alpha_cov * subject_covariance
            self.lda.set_coefficients(blended_means, blended_covariance)
        return self

    def attach_ea_reference(self, aligner: EuclideanAligner) -> None:
        """Extract `aligner`'s raw state (mean_cov, n_epochs -- see
        EuclideanAligner.reference_state()) and hold it on this model, in
        memory. save() writes this to a separate companion file (see
        module docstring); it is never part of the main pickled file.

        Parameters
        ----------
        aligner : EuclideanAligner
            Aligner whose current reference state (`reference_state()`)
            should be captured. Typically a population-level aligner
            fit alongside training.
        """
        self._ea_mean_cov, self._ea_n_epochs = aligner.reference_state()

    def build_aligner(self, schedule: Schedule = UniformAccumulation()) -> EuclideanAligner:
        """Reconstruct a fresh EuclideanAligner from whatever EA reference
        this model currently holds -- attached directly via
        attach_ea_reference(), or loaded from a companion file by load().
        Returns an unfit aligner (schedule only, no reference) if
        none was ever attached -- identical to EuclideanAligner(schedule=schedule).

        Parameters
        ----------
        schedule : Schedule
            Merge-weight policy to construct the new aligner with; see
            ea_schedules.py.

        Returns
        -------
        EuclideanAligner
            A new aligner, seeded from this model's EA reference if one
            is attached, otherwise unfit.
        """
        return EuclideanAligner(
            schedule=schedule,
            reference_mean_cov=self._ea_mean_cov,
            reference_n_epochs=self._ea_n_epochs,
        )

    def __getstate__(self) -> dict:
        """Exclude the EA reference fields from what gets pickled -- see
        module docstring. Whether or not attach_ea_reference() was ever
        called, the main file's contents are identical either way; the
        reference (if any) only ever reaches disk via save()'s separate
        companion file.
        """
        state = self.__dict__.copy()
        state.pop("_ea_mean_cov", None)
        state.pop("_ea_n_epochs", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore pickled state, then default-initialize the EA reference
        fields fresh -- they're never part of `state` (see __getstate__),
        regardless of whether this pickle predates their existence at all
        or was saved by a version of this class that has them. load()
        populates them afterward from the companion file, if one exists.
        """
        self.__dict__.update(state)
        self._ea_mean_cov = None
        self._ea_n_epochs = 0

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk via joblib.

        Writes a separate companion file for the EA reference (if one is
        attached) alongside `path` -- see module docstring/_ea_reference_path().
        If no reference is attached but a stale companion file from an
        earlier save to this same path still exists, it's removed --
        otherwise load() would silently reattach a leftover reference that
        has nothing to do with the model now being saved.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination for the main model file. The EA-reference
            companion file (if any) is written alongside it -- see
            `_ea_reference_path`.
        """
        joblib.dump(self, path)
        reference_path = _ea_reference_path(path)
        if self._ea_mean_cov is not None:
            joblib.dump(
                {"mean_cov": self._ea_mean_cov, "n_epochs": self._ea_n_epochs},
                reference_path,
            )
        elif reference_path.exists():
            reference_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> P300Model:
        """Load a previously serialized model from disk.

        Also loads the companion EA-reference file alongside `path`, if
        one exists (see save()) -- transparent to callers that never used
        attach_ea_reference(): the model just comes back with none
        attached, same as any model saved before this existed.

        Parameters
        ----------
        path : str or pathlib.Path
            Path the main model file was saved to. The EA-reference
            companion file, if any, is derived from this same path (see
            `_ea_reference_path`).

        Returns
        -------
        P300Model
            The loaded model, with its EA reference re-attached if a
            companion file was found.
        """
        model = joblib.load(path)
        reference_path = _ea_reference_path(path)
        if reference_path.exists():
            state = joblib.load(reference_path)
            model._ea_mean_cov = state["mean_cov"]
            model._ea_n_epochs = state["n_epochs"]
        return model
