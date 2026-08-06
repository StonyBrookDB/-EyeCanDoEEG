"""Offline training pipeline: filtering, epoching, spatial alignment,
spatial filtering, and classification.

Each stage lives in its own dependency-light module, composed by
`model.py`'s `P300Model` into the artifact both training and live
inference load:

- `bandpass.py` -- causal Butterworth bandpass filter primitives, plus
  `BandpassFilter` for callers that want streaming `zi` state carried
  automatically. Shared by the offline path here and by `live/`.
- `preprocessing.py` -- `preprocess_p300()`, the offline/batch
  filter-then-epoch-then-align entry point: turns one session's raw
  recording into EA-normalized `(X, y, stim_ids, n_rejected)`.
- `alignment.py` -- `EuclideanAligner`, per-subject covariance
  whitening (He & Wu, 2019) to reduce cross-subject variance before
  spatial filtering; supports both one-shot `fit()` (offline) and
  incremental `update()` (live calibration).
- `ea_schedules.py` -- pluggable merge-weight policies consulted by
  `EuclideanAligner.update()` when blending a new batch of epochs into
  its running reference covariance.
- `spatial.py` -- `XDawnFilter`, xDAWN spatial filtering that learns
  channel weightings maximizing target-vs-nontarget P300 SNR.
- `lda.py` -- `LDAClassifier`, shrinkage-regularized, balanced-prior
  LDA tuned for this pipeline's class imbalance and feature
  dimensionality.
- `model.py` -- `P300Model`, the fitted xDAWN+LDA container that
  training produces and live inference loads; also handles
  population-to-subject calibration blending (`adapt_to_subject()`)
  and optionally carries a population-level EA reference for seeding a
  live session's aligner.
- `classifier.py` -- metric primitives shared by evaluation and
  tuning: `compute_itr()` (Wolpaw ITR formula) and the
  classification-metrics helpers `eyecando.tuning.model_search` builds
  its hyperparameter search on top of.

`model.py` is the natural starting point for understanding how these
pieces fit together end to end.
"""
