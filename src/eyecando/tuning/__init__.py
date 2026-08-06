"""Offline hyperparameter search and evaluation infrastructure used to
pick the settings the rest of the pipeline ships with.

- `model_search.py` -- Stage 3 nested-LOSO grid search over
  `(n_components, shrinkage)`: `train_loso()` evaluates fixed
  hyperparameters via leave-one-subject-out; `nested_loso()` searches
  the grid fresh inside every outer fold so no held-out subject ever
  influences the hyperparameters used to judge it. Selection criterion
  is ITR at 5 repetitions, not raw single-flash AUC.
- `single_flash.py` -- `single_flash_accuracy()`, per-flash
  classification metrics (accuracy/precision/recall/specificity/F1/
  Brier score) via LOSO, independent of any repetition-level
  accumulation or stopping policy -- just one flash's own
  `P300Model.predict_proba` argmax'd against its true label.
- `policy_scoring.py` -- `selection_score()`, the additive utility
  function (accuracy above a floor, minus time cost) used to pick a
  live decoding policy's threshold/temperature/lm_temperature.
- `trials.py` -- dataset-agnostic reconstruction of character-
  selection trials from flash-level data (`reconstruct_trials()`) and
  scoring them against the true target (`rank_trial()`,
  `decode_trial()`) -- shared by every trial/letter-level analysis
  across datasets.
- `model_cache.py` -- `get_or_fit_model()`, fit-or-load-cached
  `P300Model`, keyed by excluded subjects and hyperparameters, so a
  search that re-evaluates the same combination more than once never
  refits it. Callers must supply their own dataset-specific
  `cache_dir` to avoid cross-dataset cache collisions.

These modules build on `pipeline.classifier`'s metric primitives
(`compute_itr()`, classification metrics) rather than duplicating
them.
"""
