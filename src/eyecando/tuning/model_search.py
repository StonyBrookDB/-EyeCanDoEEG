"""Stage 3 — tune the offline model's hyperparameters (n_components, shrinkage).

Nested-LOSO grid search: run this against your own dataset once it's
loaded and preprocessed (Stages 1-2) to pick n_components/shrinkage before
deploying. Imports its metric primitives (compute_itr,
_classification_metrics, _batched_metrics) from eyecando.pipeline.classifier
rather than duplicating them -- that module keeps only those primitives,
since they're used both by the search loops below and by any other
caller that just wants ITR/classification metrics without running a
search.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from eyecando.pipeline.classifier import (
    FLASHES_PER_ROUND,
    N_SYMBOLS,
    SECONDS_PER_FLASH,
    _batched_metrics,
    _classification_metrics,
    compute_itr,
)
from eyecando.pipeline.model import P300Model, pool_sessions
from eyecando.utils.time_marker import TimeMarker
from eyecando.utils.training_logger import TrainingLogger

REPETITION_COUNTS = (1, 3, 5, 10)


def train_loso(
    subjects: list[tuple[np.ndarray, np.ndarray]],
    n_components: int = 9,
    shrinkage: float | str = "auto",
    logger: TrainingLogger | None = None,
    time_marker: TimeMarker | None = None,
) -> tuple[P300Model, dict]:
    """Run leave-one-subject-out cross-validation with fixed hyperparameters.

    Each held-out subject's X is already EA-normalized by preprocess_p300
    (per-session, unsupervised) before it ever reaches here, so no label
    information from the held-out subject leaks into the model, which is
    fit on the other subjects only.

    Unlike nested_loso(), n_components/shrinkage are fixed by the caller,
    not searched per fold -- use this when you already know (e.g. from a
    prior nested_loso run) which hyperparameters you want, and just need
    a fast, single evaluation with them.

    Parameters
    ----------
    subjects : list of (numpy.ndarray, numpy.ndarray)
        One (X, y) pair per subject. X has shape
        (n_epochs, n_channels, n_timepoints); y is 1 for target / 0 for
        non-target epochs. One fold is run per entry, holding that
        subject's (X, y) out and training on all the others.
    n_components : int
        xDAWN component count passed straight through to every fold's
        ``P300Model`` and to the final, all-subjects model -- not
        searched.
    shrinkage : float or "auto"
        LDA shrinkage passed straight through the same way.
    logger : TrainingLogger or None
        If given, per-fold and summary metrics are logged as they're
        computed.
    time_marker : TimeMarker or None
        If given (together with `logger`), each fold's wall-clock time is
        logged.

    Returns
    -------
    model : P300Model
        Fitted on all subjects (for deployment).
    results : dict
        Per-subject accuracy/precision/recall/F1/AUC, and the same at
        1/3/5/10 repetitions plus ITR, plus mean +/- std across subjects.

    Raises
    ------
    numpy.linalg.LinAlgError
        If `n_components`/`shrinkage` produce a non-positive-definite
        covariance for some fold's training data -- unlike
        ``nested_loso``'s inner grid search, this function does not catch
        or disqualify that case, since the hyperparameters here are fixed
        by the caller rather than being screened by a search.
    """
    rng = np.random.default_rng(0)
    per_subject_results = []

    for held_out_idx in range(len(subjects)):
        train_subjects = [s for i, s in enumerate(subjects) if i != held_out_idx]
        X_test, y_test = subjects[held_out_idx]

        if time_marker is not None:
            time_marker.mark(f"fold_{held_out_idx}")
        if logger is not None:
            logger.log(
                f"[FOLD] {held_out_idx + 1}/{len(subjects)}: holding out subject "
                f"{held_out_idx}, training on {len(train_subjects)} subjects"
            )

        model = P300Model(n_components=n_components, shrinkage=shrinkage)
        model.fit(train_subjects)
        scores = model.decision_scores(X_test)

        target_scores = scores[y_test == 1]
        nontarget_scores = scores[y_test == 0]

        overall = _classification_metrics(y_test, scores)

        per_repetition = {}
        for n_reps in REPETITION_COUNTS:
            metrics = _batched_metrics(target_scores, nontarget_scores, n_reps, rng)
            seconds_per_selection = n_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH
            per_repetition[n_reps] = {
                **metrics,
                "itr_bits_per_min": compute_itr(
                    metrics["accuracy"], N_SYMBOLS, seconds_per_selection
                ),
            }

        per_subject_results.append(
            {
                "subject_index": held_out_idx,
                **overall,
                "per_repetition": per_repetition,
            }
        )

        if logger is not None:
            logger.log(
                f"[FOLD] subject {held_out_idx}: acc={overall['accuracy']:.3f} "
                f"prec={overall['precision']:.3f} rec={overall['recall']:.3f} "
                f"f1={overall['f1']:.3f} auc={overall['auc']:.3f}"
            )
            if time_marker is not None:
                logger.log(
                    f"[FOLD] subject {held_out_idx} total time: "
                    f"{time_marker.elapsed_from(f'fold_{held_out_idx}')}"
                )

    results: dict[str, Any] = {"per_subject": per_subject_results}
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        values = np.array([r[metric] for r in per_subject_results])
        results[f"mean_{metric}"] = float(values.mean())
        results[f"std_{metric}"] = float(values.std())

    if logger is not None:
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            logger.log(
                f"[SUMMARY] mean {metric}: {results[f'mean_{metric}']:.3f} "
                f"+/- {results[f'std_{metric}']:.3f}"
            )

    final_model = P300Model(n_components=n_components, shrinkage=shrinkage).fit(subjects)

    return final_model, results


DEFAULT_N_COMPONENTS_GRID = list(range(1, 7))
DEFAULT_SHRINKAGE_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, "auto"]


def _inner_grid_search(
    train_subjects: list[tuple[np.ndarray, np.ndarray]],
    n_components_grid: list[int],
    shrinkage_grid: list[float | str],
    logger: TrainingLogger | None = None,
    time_marker: TimeMarker | None = None,
) -> tuple[int, float | str, dict, list[dict]]:
    """Joint (n_components, shrinkage) grid search, via leave-one-subject-out
    over `train_subjects` only.

    Uses LeaveOneGroupOut rather than GroupKFold(n_splits=k): GroupKFold
    with k < n_subjects groups multiple subjects together per fold (e.g.
    ~2 subjects per fold with 9 training subjects and n_splits=5), which
    blends subjects together rather than identifying them individually.
    LeaveOneGroupOut always creates exactly one fold per unique subject,
    holding out precisely one at a time -- no blending, matching the
    outer loop's own rigor.

    Both xDAWN and LDA are label-informed transforms, so both are refit
    fresh on each fold's training portion via a full P300Model fit --
    never on data that includes that fold's held-out subject. This is
    called fresh with only the *training* subjects for one outer LOSO
    fold, so no subject ever influences the hyperparameters used to
    judge it.

    Selection criterion is ITR at 5 repetitions, not raw single-flash AUC:
    the deployed system only ever acts on repetition-averaged scores, so a
    combination that maximizes raw AUC isn't necessarily the one that
    maximizes real accuracy/throughput after averaging. Every candidate's
    full metrics (accuracy/precision/recall/f1/auc/itr) are logged if a
    logger is given, not just whichever one was used to pick the winner.

    Returns (best_n_components, best_shrinkage, best_mean_metrics, candidates)
    where best_mean_metrics is the full {accuracy, precision, recall, f1,
    auc, itr} dict achieved by the winning combination, and candidates is
    every (n_components, shrinkage) combination tried -- including
    disqualified ones -- so the full search can be saved as data (e.g. to
    JSON) without re-parsing the text log.
    """
    X_all, y_all = pool_sessions(train_subjects)
    X_all = X_all.astype(np.float64)
    n_per = [len(y) for _, y in train_subjects]
    groups = np.concatenate([np.full(n, i) for i, n in enumerate(n_per)])
    cv = LeaveOneGroupOut()
    folds = list(cv.split(X_all, y_all, groups))
    rng = np.random.default_rng(0)
    seconds_per_selection = 5 * FLASHES_PER_ROUND * SECONDS_PER_FLASH

    best_n, best_shrinkage, best_metrics = n_components_grid[0], shrinkage_grid[0], None
    candidates: list[dict] = []
    for n_comp in n_components_grid:
        for shrink in shrinkage_grid:
            if time_marker is not None:
                time_marker.mark("candidate")
            fold_metrics = []
            singular = False
            for train_idx, test_idx in folds:
                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_te, y_te = X_all[test_idx], y_all[test_idx]
                model = P300Model(n_components=n_comp, shrinkage=shrink)
                try:
                    model.fit([(X_tr, y_tr)])
                    scores = model.decision_scores(X_te)
                except np.linalg.LinAlgError:
                    # This (n_components, shrinkage) pair produced a
                    # non-positive-definite covariance for this fold's
                    # training data -- solver="eigen" refuses to guess
                    # (unlike the old solver="lsqr", which would have
                    # silently returned corrupted coefficients here; see
                    # lda.py). Disqualify the whole combination rather
                    # than average around the failure.
                    singular = True
                    break
                if len(set(y_te)) > 1:
                    metrics = _batched_metrics(scores[y_te == 1], scores[y_te == 0], 5, rng)
                    metrics["itr"] = compute_itr(
                        metrics["accuracy"], N_SYMBOLS, seconds_per_selection
                    )
                    fold_metrics.append(metrics)
            elapsed = (
                time_marker.elapsed_from_pure("candidate") if time_marker is not None else None
            )
            timing_str = f" {time_marker.format_time(elapsed)}" if time_marker is not None else ""

            if singular or not fold_metrics:
                reason = (
                    "non-positive-definite covariance" if singular else "no fold had both classes"
                )
                candidates.append(
                    {
                        "n_components": n_comp,
                        "shrinkage": shrink,
                        "disqualified": True,
                        "reason": reason,
                        "elapsed_seconds": elapsed,
                    }
                )
                if logger is not None:
                    logger.log(
                        f"[SEARCH] disqualified n_components={n_comp}, "
                        f"shrinkage={shrink}: {reason}{timing_str}"
                    )
                continue

            mean_metrics = {
                metric: float(np.mean([fm[metric] for fm in fold_metrics]))
                for metric in ("accuracy", "precision", "recall", "f1", "auc", "itr")
            }
            candidates.append(
                {
                    "n_components": n_comp,
                    "shrinkage": shrink,
                    "disqualified": False,
                    **mean_metrics,
                    "elapsed_seconds": elapsed,
                }
            )
            if logger is not None:
                logger.log(
                    f"[SEARCH] n_components={n_comp}, shrinkage={shrink} | "
                    f"acc={mean_metrics['accuracy']:.3f} prec={mean_metrics['precision']:.3f} "
                    f"rec={mean_metrics['recall']:.3f} f1={mean_metrics['f1']:.3f} "
                    f"auc={mean_metrics['auc']:.3f} itr={mean_metrics['itr']:.2f}{timing_str}"
                )
            if best_metrics is None or mean_metrics["itr"] > best_metrics["itr"]:
                best_n, best_shrinkage, best_metrics = n_comp, shrink, mean_metrics

    if best_metrics is None:
        raise RuntimeError(
            "Every (n_components, shrinkage) combination in the grid produced a "
            "non-positive-definite covariance on at least one fold -- widen the "
            "shrinkage_grid toward larger values or shrink n_components_grid."
        )

    if logger is not None:
        logger.log(
            f"[SEARCH] selected n_components={best_n}, shrinkage={best_shrinkage} "
            f"(itr={best_metrics['itr']:.2f} bits/min)"
        )

    return best_n, best_shrinkage, best_metrics, candidates


def nested_loso(
    subjects: list[tuple[np.ndarray, np.ndarray]],
    n_components_grid: list[int] | None = None,
    shrinkage_grid: list[float | str] | None = None,
    logger: TrainingLogger | None = None,
    time_marker: TimeMarker | None = None,
) -> tuple[P300Model, dict]:
    """Nested LOSO: search (n_components, shrinkage) fresh inside every
    outer fold, using only that fold's training subjects.

    Unlike train_loso() (fixed hyperparameters passed in from an earlier,
    separate, non-nested search over all subjects), this never lets a
    held-out subject influence the hyperparameters used to judge it —
    each of the `len(subjects)` outer folds reruns the joint grid search
    on only its 9 (for 10 subjects) training subjects. That means this
    is ~`len(subjects)`x the compute of a single joint search, since the
    search reruns once per outer fold plus once more for the final model.

    The final deployable model is produced by running the same joint
    search one more time on *all* subjects, then refitting on all of
    them with whatever that search picks — see results["final_n_components"]
    / results["final_shrinkage"]. The per-fold picks in
    results["per_subject"][i]["chosen_n_components"/"chosen_shrinkage"]
    are worth checking for stability: if every fold lands on a similar
    combination, that's reassuring; if they scatter widely, the choice
    is unstable and worth flagging in a write-up.

    If `logger` is given, every candidate the inner search tries (with its
    full accuracy/precision/recall/f1/auc/itr) gets logged, not just the
    winner -- and if `time_marker` is also given, each fold's inner-search
    time and total time are logged too (not per-candidate: that would be
    excessive given the grid can be dozens of combinations).

    Parameters
    ----------
    subjects : list of (numpy.ndarray, numpy.ndarray)
        One (X, y) pair per subject. X has shape
        (n_epochs, n_channels, n_timepoints); y is 1 for target / 0 for
        non-target epochs. One outer fold is run per entry.
    n_components_grid : list of int or None
        Candidate xDAWN component counts for the inner search. Defaults
        to `DEFAULT_N_COMPONENTS_GRID` when None.
    shrinkage_grid : list of (float or "auto") or None
        Candidate LDA shrinkage values for the inner search. Defaults to
        `DEFAULT_SHRINKAGE_GRID` when None.
    logger : TrainingLogger or None
        If given, every inner-search candidate plus per-fold and summary
        metrics are logged as they're computed.
    time_marker : TimeMarker or None
        If given (together with `logger`), each fold's inner-search time
        and total time are logged, plus the final all-subjects search
        time.

    Returns
    -------
    model : P300Model
        Fitted on all subjects with hyperparameters from a final,
        all-subjects joint search (for deployment).
    results : dict
        Per-subject metrics (including which hyperparameters that fold's
        inner search chose, and that fold's full list of search
        candidates), mean +/- std across subjects, and the final chosen
        hyperparameters (``final_n_components``/``final_shrinkage``) plus
        that final search's own candidate list
        (``final_search_candidates``).

    Raises
    ------
    RuntimeError
        Propagated from the inner grid search (for any outer fold, or for
        the final all-subjects search) if every (n_components, shrinkage)
        combination in the grid was disqualified by a non-positive-definite
        covariance on at least one of its folds.
    """
    n_components_grid = n_components_grid or DEFAULT_N_COMPONENTS_GRID
    shrinkage_grid = shrinkage_grid or DEFAULT_SHRINKAGE_GRID
    rng = np.random.default_rng(0)
    per_subject_results = []

    for held_out_idx in range(len(subjects)):
        train_subjects = [s for i, s in enumerate(subjects) if i != held_out_idx]
        X_test, y_test = subjects[held_out_idx]

        if time_marker is not None:
            time_marker.mark(f"fold_{held_out_idx}")
            time_marker.mark(f"fold_{held_out_idx}_search")
        if logger is not None:
            logger.log(
                f"[FOLD] {held_out_idx + 1}/{len(subjects)}: holding out subject "
                f"{held_out_idx}, training on {len(train_subjects)} subjects"
            )

        best_n, best_shrinkage, _, search_candidates = _inner_grid_search(
            train_subjects,
            n_components_grid,
            shrinkage_grid,
            logger=logger,
            time_marker=time_marker,
        )
        if logger is not None and time_marker is not None:
            elapsed = time_marker.elapsed_from(f"fold_{held_out_idx}_search")
            logger.log(f"[FOLD] inner search took {elapsed}")

        model = P300Model(n_components=best_n, shrinkage=best_shrinkage)
        model.fit(train_subjects)
        scores = model.decision_scores(X_test)

        target_scores = scores[y_test == 1]
        nontarget_scores = scores[y_test == 0]

        overall = _classification_metrics(y_test, scores)

        per_repetition = {}
        for n_reps in REPETITION_COUNTS:
            metrics = _batched_metrics(target_scores, nontarget_scores, n_reps, rng)
            seconds_per_selection = n_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH
            per_repetition[n_reps] = {
                **metrics,
                "itr_bits_per_min": compute_itr(
                    metrics["accuracy"], N_SYMBOLS, seconds_per_selection
                ),
            }

        per_subject_results.append(
            {
                "subject_index": held_out_idx,
                "chosen_n_components": best_n,
                "chosen_shrinkage": best_shrinkage,
                **overall,
                "per_repetition": per_repetition,
                "search_candidates": search_candidates,
            }
        )

        if logger is not None:
            logger.log(
                f"[FOLD] subject {held_out_idx}: chosen n_components={best_n}, "
                f"shrinkage={best_shrinkage} | acc={overall['accuracy']:.3f} "
                f"prec={overall['precision']:.3f} rec={overall['recall']:.3f} "
                f"f1={overall['f1']:.3f} auc={overall['auc']:.3f}"
            )
            if time_marker is not None:
                logger.log(
                    f"[FOLD] subject {held_out_idx} total time: "
                    f"{time_marker.elapsed_from(f'fold_{held_out_idx}')}"
                )

    results: dict[str, Any] = {"per_subject": per_subject_results}
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        values = np.array([r[metric] for r in per_subject_results])
        results[f"mean_{metric}"] = float(values.mean())
        results[f"std_{metric}"] = float(values.std())

    if logger is not None:
        chosen_ns = [r["chosen_n_components"] for r in per_subject_results]
        chosen_shrinkages = [r["chosen_shrinkage"] for r in per_subject_results]
        logger.log(f"[SUMMARY] chosen n_components per fold: {chosen_ns}")
        logger.log(f"[SUMMARY] chosen shrinkage per fold: {chosen_shrinkages}")
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            logger.log(
                f"[SUMMARY] mean {metric}: {results[f'mean_{metric}']:.3f} "
                f"+/- {results[f'std_{metric}']:.3f}"
            )

    if time_marker is not None:
        time_marker.mark("final_search")
    final_n, final_shrinkage, _, final_search_candidates = _inner_grid_search(
        subjects, n_components_grid, shrinkage_grid, logger=logger, time_marker=time_marker
    )
    if logger is not None and time_marker is not None:
        logger.log(f"[FINAL] search took {time_marker.elapsed_from('final_search')}")
    results["final_n_components"] = final_n
    results["final_shrinkage"] = final_shrinkage
    results["final_search_candidates"] = final_search_candidates
    final_model = P300Model(n_components=final_n, shrinkage=final_shrinkage).fit(subjects)

    if logger is not None:
        logger.log(
            f"[FINAL] hyperparameters (all-subjects search): "
            f"n_components={final_n}, shrinkage={final_shrinkage}"
        )
        if isinstance(final_shrinkage, (int, float)) and final_shrinkage > 0.85:
            logger.log(
                "[FINAL] WARNING: shrinkage > 0.85 -- LDA is relying almost entirely "
                "on the isotropic fallback. Consider decimating timepoints to reduce "
                "feature dimension before shipping this model."
            )

    return final_model, results
