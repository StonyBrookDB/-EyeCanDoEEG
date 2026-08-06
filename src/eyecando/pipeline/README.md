# eyecando.pipeline — Stage 2: train a model

Filter, epoch, spatially align, spatially filter, classify -- the fit
path from EA-normalized epochs to a saved, deployable model. Entry point:
`scripts/train_offline.py`.

- **`bandpass.py`** — causal Butterworth filtering, streaming-safe
  (shared between the offline batch path here and the live path in
  `eyecando.live.streaming`).
- **`alignment.py`** — `EuclideanAligner`: whitens each session's
  covariance to reduce cross-subject/session variance. Uses He & Wu's
  original *uncentered* second moment -- confirmed directly against the
  paper (arXiv:1808.05464), not assumed; see the module's own docstring
  before "fixing" this back to centered.
- **`ea_schedules.py`** — pluggable merge-weight policies for how much
  trust a new calibration batch gets relative to everything already
  accumulated in `EuclideanAligner`'s running reference.
- **`spatial.py`** — `XDawnFilter`: spatial filtering to concentrate the
  P300 response into a handful of components.
- **`lda.py`** — `LDAClassifier`: regularized, balanced-prior LDA tuned
  for this pipeline's class imbalance and feature dimensionality.
- **`preprocessing.py`** — `preprocess_p300()`: wires the above into one
  offline batch call (filter -> epoch -> EA). `use_ea=False` isolates
  EA's own contribution for tuning experiments.
- **`model.py`** — `P300Model`: the deployable artifact combining xDAWN +
  LDA, plus `adapt_to_subject()`/`attach_ea_reference()`/`build_aligner()`
  for Stage 4 (headset/subject calibration).
- **`classifier.py`** — metric primitives only (`compute_itr`, and the
  private `_classification_metrics`/`_batched_metrics` helpers): the
  hyperparameter *search* that used to live here has moved to
  `eyecando.tuning.model_search`, which imports these rather than
  duplicating them.

## What's not here

Anything about *choosing* hyperparameters (`n_components`, `shrinkage`)
is `eyecando.tuning`, not this package -- this package only knows how to
fit a model given hyperparameters you already have.

## Why these numbers: investigation notes

Module docstrings in this package stick to mechanics. The reasoning
behind a few choices that look arbitrary (or reversible) but aren't
lives here instead.

### EA's uncentered second moment (`alignment.py`)

`EuclideanAligner` whitens by He & Wu's original *uncentered* second
moment (`X_i X_i^T / T`, no per-epoch mean subtraction), which looks at
first glance like a missing centering step. It isn't. Confirmed
directly against the paper (arXiv:1808.05464, Eq. 7 and Eq. 10:
`Sigma_i = X_i X_i^T`, no mean term anywhere), and the wider
Riemannian-BCI literature this paper builds on (Congedo, Barachant &
Andreev 2013, "A New Generation of BCI Based on Riemannian Geometry")
states the assumption explicitly: trial data is assumed zero-mean
*after usual band-pass filtering*. For P300 epochs specifically, a
short epoch's true time-mean is not actually zero -- the P300
deflection **is** a real, non-zero mean shift in the post-stimulus
window, which is exactly the discriminative signal this pipeline wants
classified, not structure to whiten away before EA ever sees it.
Centering here would remove signal, not noise. Don't "fix" this back to
centered without re-deriving why first.

### LDA's `solver="eigen"`, not `"lsqr"` (`lda.py`)

`LDAClassifier` fixes sklearn's `LinearDiscriminantAnalysis` to
`solver="eigen"`. This isn't just "eigen handles rank-deficient
covariance better" -- `solver="lsqr"`'s direct least-squares solve
(`scipy.linalg.lstsq`) was found to **silently return corrupted
coefficients** at higher feature dimensions on some scipy/numpy/OpenBLAS
builds, via a LAPACK `gelsd` defect. Confirmed independent of
shrinkage/data by comparing against the `gelss`/`gelsy` drivers, which
didn't reproduce the corruption. `solver="eigen"` solves a generalized
eigenvalue problem (`scipy.linalg.eigh`) instead and doesn't hit this
path at all. If a future scipy/LAPACK upgrade seems to "fix" `lsqr`,
re-verify against `gelss`/`gelsy` before trusting it, rather than
assuming the defect is gone everywhere.

### Model file size (`model.py`, `P300Model.fit()`)

After `fit()`, `self.lda._lda.scalings_` and `.covariance_` are cleared
(set to `None`). These are sklearn solver-internal artifacts -- needed
to derive `coef_`/`intercept_` during `fit()`, and otherwise only used
by sklearn's own `transform()`, which this pipeline never calls.
`decision_scores()`/`predict_proba()`/`adapt_to_subject()` only ever
use `coef_`/`intercept_` and the `population_means_`/
`population_covariance_` captured separately -- so both cleared
attributes are pure dead weight from that point on. Both are the same
size as `population_covariance_` (~43MB each at `n_components=10` on
this dataset). Clearing them cuts a saved model from ~129MB to ~43MB
with zero change in behavior, verified via a full save/reload round
trip. If a future change starts calling sklearn's own `transform()` on
`self.lda._lda` directly, this optimization would need revisiting.

### The EA merge-weight identity (`ea_schedules.py`, `alignment.py`)

`EuclideanAligner.update()`'s merge rule is
`new_mean_cov = (1 - w.sum()) * old_mean_cov + sum_i(w[i] * cov_i)`,
where `w` are the per-epoch weights a `Schedule` returns. `1 - w.sum()`
(the weight kept on the old running estimate) is a **general identity**,
true for any schedule -- not something each `Schedule` implementation
needs to compute itself. It's the same number you'd get from applying
each epoch's own weight one at a time, recursively, and watching how
much of the running estimate survives every subsequent step's own
discount; the array form in `update()` just computes that directly
instead of via one sequential single-epoch `update()` call per epoch.
For a batch of 1, this collapses to the original scalar form exactly
(`(1-w[0])*old + w[0]*cov`) -- not a special case, the same formula with
one term.

### Why `preprocess_p300()` is batch/offline only (`preprocessing.py`)

`preprocess_p300()` cannot be called per-epoch on live/streaming data,
for two independent reasons:

1. The Butterworth filter (`bandpass.py`'s `build_sos()`/`apply_filter()`)
   is applied to the *continuous* raw signal before epoching, not to
   individual already-windowed epochs. `apply_filter()` does support
   carrying `zi` state across chunks (that's the streaming case -- see
   `bandpass.BandpassFilter`), but `preprocess_p300()` never uses that
   path; it filters the whole continuous recording in one call.
   Filtering an isolated ~0.9s epoch on its own produces a different,
   edge-distorted result: a low `l_freq=0.1Hz` cutoff needs several
   seconds of context to settle, which a single short epoch can't
   supply.
2. `EuclideanAligner.fit()` computes a *mean* covariance across multiple
   epochs -- it needs a representative set to produce a stable whitening
   matrix. A single epoch's own covariance isn't a meaningful substitute.

The live path (`streaming.py`) assembles the equivalent pieces
differently instead: `bandpass.BandpassFilter` for the continuous
stream, and `EuclideanAligner.update()` on each new calibration batch as
it arrives -- not a per-epoch call into `preprocess_p300()`.

### Population calibration shrinkage (`model.py`, `_population_kappa()`)

The Bühlmann-Straub/Efron-Morris shrinkage estimate this function
computes assumes a multi-subject population to estimate between-subject
variance from -- what happens with only one training subject (no
variance to estimate) is covered in the top-level
[README's "Data assumptions & portability"](../../../README.md#data-assumptions--portability)
section, not repeated here.
