"""Single-flash classification metrics, LOSO, independent of Stage 2's
trial/repetition-level accumulation entirely -- no ScoreAccumulator/
should_decode/lm involved, just one flash's own P300Model.predict_proba
argmax'd against its true label. Reusable across datasets via
get_or_fit_model (model_cache.py) for the actual model fitting/caching.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from eyecando.tuning.model_cache import get_or_fit_model

SINGLE_FLASH_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "brier")


def single_flash_accuracy(
    subjects_pooled: list[tuple[np.ndarray, np.ndarray]],
    n_subjects: int,
    n_components: int,
    use_ea: bool,
    use_xdawn: bool,
    cache_dir: Path,
    log_fn=None,
) -> dict[str, tuple[float, float, list[float]]]:
    """Single-flash classification metrics for a (use_ea, use_xdawn) pair:
    argmax over one flash's own P300Model.predict_proba, pre-repetition-
    ensembling, LOSO across all subjects.

    Accuracy alone is a weak summary here: target flashes are rare (2 of
    FLASHES_PER_ROUND=12 per round, row+col), so a trivial always-nontarget
    classifier already scores ~83% accuracy without detecting a single
    target. Precision/recall/specificity/F1 (recall is "sensitivity" --
    same quantity, this project's naming already uses "recall" for the
    live accumulator's own semantics; specificity is recall of the
    nontarget class) surface that imbalance instead of hiding it, and cost
    nothing extra to compute -- pred/y_test are already in hand.

    `brier` is the mean squared error between `proba` (the model's own
    P(target) before thresholding to `pred`) and the true 0/1 label --
    the standard Brier score, which is itself the squared-loss Bregman
    divergence for a binary probability forecast (not a separate metric
    from "Bregman score": Brier IS the Bregman score under squared loss,
    the one Bregman divergence that's a proper scoring rule for binary
    probabilities without needing a base measure choice the way, say,
    KL-divergence's Bregman generalization would). Unlike
    accuracy/precision/recall/F1/specificity, which all only look at
    `pred` (proba thresholded at 0.5) and are blind to whether a 0.51 and
    a 0.99 "target" call were equally confident, Brier score rewards
    well-calibrated probabilities and penalizes overconfident
    wrong ones quadratically -- directly relevant here since
    ScoreAccumulator/should_decode accumulate these raw probabilities
    across repetitions, not the thresholded calls this function's other
    metrics summarize. Lower is better (0 = perfect), unlike every other
    metric here.

    Parameters
    ----------
    subjects_pooled : list[tuple[numpy.ndarray, numpy.ndarray]]
        One (X, y) pair per LOSO unit (subject or session, depending on
        the caller's dataset -- see `get_or_fit_model`).
    n_subjects : int
        Number of LOSO folds to run (usually ``len(subjects_pooled)``).
    n_components : int
        Forwarded to `get_or_fit_model`.
    use_ea : bool
        Cache-key-only -- see `get_or_fit_model`/`model_cache_path`.
    use_xdawn : bool
        Forwarded to `get_or_fit_model`.
    cache_dir : pathlib.Path
        Forwarded to `get_or_fit_model` -- callers evaluating a different
        dataset than each other MUST pass a dataset-specific `cache_dir`
        (see `model_cache.py`'s own module docstring for the collision
        this otherwise risks).
    log_fn : callable or None
        If given, called with a one-line progress string after each fold.

    Returns
    -------
    dict[str, tuple[float, float, list[float]]]
        ``{metric_name: (mean, std, per_subject)}`` for each name in
        `SINGLE_FLASH_METRICS` -- `per_subject` (index i = subject i's own
        held-out value) lets a caller pair single-flash accuracy across
        conditions/configs rather than only ever seeing the aggregate.
    """
    per_subject: dict[str, list[float]] = {m: [] for m in SINGLE_FLASH_METRICS}
    for held_out_idx in range(n_subjects):
        start = time.monotonic()
        excluded_ids = frozenset({held_out_idx})
        train_pooled = [
            subjects_pooled[i] for i, _ in enumerate(subjects_pooled) if i not in excluded_ids
        ]
        model = get_or_fit_model(
            excluded_ids,
            n_components,
            "auto",
            train_pooled,
            cache_dir=cache_dir,
            use_ea=use_ea,
            use_xdawn=use_xdawn,
        )
        X_test, y_test = subjects_pooled[held_out_idx]
        proba = model.predict_proba(X_test)
        pred = (proba >= 0.5).astype(int)
        acc = float(np.mean(pred == y_test))
        precision = float(precision_score(y_test, pred, zero_division=0))
        recall = float(recall_score(y_test, pred, zero_division=0))
        # specificity = recall of the nontarget (0) class -- sklearn has no
        # direct helper for this, but flipping both labels reduces it to
        # the same recall_score call.
        specificity = float(recall_score(1 - y_test, 1 - pred, zero_division=0))
        f1 = float(f1_score(y_test, pred, zero_division=0))
        brier = float(np.mean((proba - y_test) ** 2))
        per_subject["accuracy"].append(acc)
        per_subject["precision"].append(precision)
        per_subject["recall"].append(recall)
        per_subject["specificity"].append(specificity)
        per_subject["f1"].append(f1)
        per_subject["brier"].append(brier)
        if log_fn is not None:
            log_fn(
                f"    single-flash metrics {held_out_idx + 1}/{n_subjects}: "
                f"excluding subject {held_out_idx} done ({time.monotonic() - start:.1f}s) "
                f"-- accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, "
                f"specificity={specificity:.4f}, f1={f1:.4f}, brier={brier:.4f}"
            )
    return {m: (float(np.mean(vals)), float(np.std(vals)), vals) for m, vals in per_subject.items()}
