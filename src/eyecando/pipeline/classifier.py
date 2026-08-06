"""Offline training: metric primitives shared by evaluation and tuning.

Imports model.py and preprocessing.py, but not data.py — callers load data
and pass it in.

The nested-LOSO hyperparameter search (train_loso/nested_loso/
_inner_grid_search) lives in eyecando.tuning.model_search, which imports
the constants and compute_itr() below rather than duplicating them --
this module keeps only the metric primitives themselves (used both by
tuning and by any other caller that just wants ITR/classification
metrics without running a search).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

N_SYMBOLS = 36  # 6×6 grid
FLASHES_PER_ROUND = 12  # 6 rows + 6 cols
SECONDS_PER_FLASH = 0.25  # measured from BNCI2014_009's own stim_id event timing


def compute_itr(accuracy: float, n_symbols: int, seconds_per_selection: float) -> float:
    """Compute ITR in bits/min using the Wolpaw formula.

    Uses the online free-spelling convention — `seconds_per_selection`
    must include flash duration, inter-stimulus intervals, and any pauses
    (not just the "cued" classification time).

    Parameters
    ----------
    accuracy : float
        Classification accuracy, in [0, 1]. Clipped away from exactly 0
        or 1 internally to keep the log terms finite.
    n_symbols : int
        Size of the selection alphabet (e.g. `N_SYMBOLS` for this
        project's 6x6 grid).
    seconds_per_selection : float
        Wall-clock time for one selection, including flash duration,
        inter-stimulus intervals, and any pauses.

    Returns
    -------
    float
        Information transfer rate, in bits/minute.
    """
    p = float(np.clip(accuracy, 1e-9, 1 - 1e-9))
    other = (1 - p) / (n_symbols - 1)
    bits_per_selection = np.log2(n_symbols) + p * np.log2(p)
    if other > 0:
        bits_per_selection += (1 - p) * np.log2(other)
    selections_per_min = 60.0 / seconds_per_selection
    return float(bits_per_selection * selections_per_min)


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1, and AUC for one set of scores.

    Recall is "catch all the target flashes" (few missed targets/false
    negatives); precision is "don't cry wolf on nontarget flashes" (few
    false positives). With the ~1:5/6 target:nontarget imbalance here,
    precision is structurally limited even for a good classifier — a
    small false-positive rate on the much larger nontarget pool still
    yields many false positives relative to the smaller true-positive
    pool. F1 (their harmonic mean) exists specifically to catch the
    degenerate case recall alone would miss: a classifier that predicts
    target on every flash gets 100% recall and is useless.
    """
    y_pred = (y_score >= 0).astype(int)  # decision_function threshold is 0, not 0.5
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
    }


def _batched_metrics(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
    n_reps: int,
    rng: np.random.Generator,
) -> dict:
    """Classification metrics when averaging `n_reps` repeated-flash scores.

    Groups scores into batches of `n_reps` and averages each batch, which
    approximates the noise reduction from repeated flashes. This is a
    simplification: it does not simulate the actual row/column grid
    accumulation (that belongs to eyecando.decode.accumulator/eyecando.decode.stopping) — it
    only estimates how averaging affects target/nontarget separability.
    """

    def _batch_means(scores: np.ndarray) -> np.ndarray:
        n_batches = max(len(scores) // n_reps, 1)
        usable = min(len(scores), n_batches * n_reps)
        idx = rng.permutation(len(scores))[:usable].reshape(n_batches, -1)
        return scores[idx].mean(axis=1)

    target_batches = _batch_means(target_scores)
    nontarget_batches = _batch_means(nontarget_scores)
    y_true = np.concatenate([np.ones_like(target_batches), np.zeros_like(nontarget_batches)])
    y_score = np.concatenate([target_batches, nontarget_batches])
    return _classification_metrics(y_true, y_score)
