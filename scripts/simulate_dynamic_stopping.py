"""Stage 3: tune the live decode policy's hyperparameters against the real trained pipeline.

Unlike train_offline.py's ITR@N (a batching approximation that never
runs ScoreAccumulator/should_decode), this script replays each
subject's real row/column-flash trials through them exactly as live
inference would, so reported ITR reflects an actual stopping decision.

Per-subject LOSO, hyperparameter search split into two stages so
temperature/threshold selection never distorts which (n_components,
shrinkage) looks best:

- Stage 1 (select_n_components_shrinkage()): picks (n_components,
  shrinkage) by mean beta_power_score on individual flash predictions,
  with no temperature/threshold involved at all -- see scripts/README.md
  for why beta_power_score over log loss.
- Stage 2 (select_hyperparameters()): with those fixed, searches
  (temperature, threshold) alone via the same nested leave-one-out
  machinery, mirroring classifier.py's _inner_grid_search.

Both also run once more using all 10 subjects to report a single
recommended deployment config alongside the per-fold stability check.
Every subject evaluated is first calibrated via model.adapt_to_subject()
on their first `n_calib_sessions` session(s), so the selected
(temperature, threshold) are correct for the post-adaptation
distribution, not tuned cold.

Usage:
    python scripts/simulate_dynamic_stopping.py

    Defaults run the current best-known config (n_components=9,
    shrinkage=auto, threshold=0.9, temperature=2.0) -- pass wider grids
    to keep exploring, e.g.:
        python scripts/simulate_dynamic_stopping.py \
            [--n-components-grid 9,10,11] [--shrinkage-grid auto] \
            [--thresholds 0.9,0.95,0.99] \
            [--temperatures 1.5,2,3] \
            [--n-calib-sessions 1]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np

from eyecando.decode.accumulator import ScoreAccumulator
from eyecando.decode.lm import FARWELL_DONCHIN_GRID, CharLM
from eyecando.decode.stopping import should_decode
from eyecando.ingestion.data import load_moabb_all_subjects_grouped
from eyecando.pipeline.classifier import (
    FLASHES_PER_ROUND,
    N_SYMBOLS,
    SECONDS_PER_FLASH,
    compute_itr,
)
from eyecando.pipeline.model import P300Model, pool_sessions
from eyecando.pipeline.preprocessing import preprocess_p300
from eyecando.tuning.model_cache import get_or_fit_model
from eyecando.tuning.policy_scoring import selection_score
from eyecando.tuning.trials import Session, Trial, flat_target_index, reconstruct_trials
from eyecando.utils.path_helpers import RESULTS_BASE, run_dir
from eyecando.utils.training_logger import TrainingLogger

mne.set_log_level("WARNING")

MODEL_CACHE_DIR = RESULTS_BASE / "model_cache"


def save_grid(path: Path, values_by_combo: dict[tuple, float], axis_grids: dict[str, list]) -> None:
    """Save a dict keyed by a combo tuple (one value per axis, in the same
    order as axis_grids) as a proper multi-dimensional numpy array, plus a
    JSON sidecar mapping each axis name to the grid values at each index --
    so the full search space can be reloaded and sliced/plotted later
    without rerunning anything. Missing combos (e.g. never evaluated) are
    left as NaN.
    """
    shape = tuple(len(values) for values in axis_grids.values())
    array = np.full(shape, np.nan)
    index_lookups = [
        {value: idx for idx, value in enumerate(values)} for values in axis_grids.values()
    ]
    for combo, value in values_by_combo.items():
        index = tuple(lookup[v] for lookup, v in zip(index_lookups, combo, strict=True))
        array[index] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), array)
    axes_meta = {"axis_order": list(axis_grids.keys()), "axis_values": axis_grids}
    path.with_suffix(".json").write_text(json.dumps(axes_meta, indent=2))


def calibrate_and_score(
    model: P300Model,
    calib_sessions: list[Session],
    test_session: Session,
    as_proba: bool = False,
    use_calib: bool = True,
) -> np.ndarray:
    """Adapt `model` to one subject's calibration sessions, then score one
    of their test sessions.

    adapt_to_subject() always re-blends from the pristine population
    statistics captured at fit() time, so calling this repeatedly on the
    same fitted `model` for different subjects is safe -- nothing
    compounds. Shrinkage weight is a closed-form estimate computed once
    at fit() time, not a parameter this function searches (see model.py's
    adapt_to_subject docstring).

    use_calib=False skips adapt_to_subject() entirely, scoring directly
    off the population-only model -- safe because get_or_fit_model()
    always hands back a fresh fit or cache reload from before any
    adapt_to_subject() call touches it. Used to isolate whether
    calibration earns its complexity given EA's own per-subject
    normalization.

    as_proba : bool
        If True, return calibrated predict_proba() instead of raw
        decision_scores(). Stage 1's log-loss scoring needs calibrated
        probabilities; Stage 2's trial-level accumulation needs raw
        scores, since should_decode depends on magnitude growing with
        evidence, which a bounded probability can't represent.
    """
    if use_calib:
        calib_pairs = [(X_sess, y_sess) for X_sess, y_sess, _, _ in calib_sessions]
        calib_X, calib_y = pool_sessions(calib_pairs)
        model.adapt_to_subject(calib_X, calib_y)
    test_X = test_session[0]
    return model.predict_proba(test_X) if as_proba else model.decision_scores(test_X)


def simulate_trial(
    trial: Trial,
    scores: np.ndarray,
    stim_ids: np.ndarray,
    threshold: float = 0.9,
    temperature: float = 2.0,
    lm: CharLM | None = None,
    context: str = "",
    lm_temperature: float = 1.0,
) -> tuple[int, bool, int]:
    """Replay one trial's repetitions through ScoreAccumulator/should_decode.

    Returns (repetitions_used, correct, decoded_idx) -- decoded_idx lets
    evaluate_stopping_policy() grow `context` across trials within a
    session when an lm is given, matching what a real live session's own
    decoded-so-far context would be (not the ground-truth target, which a
    real system never has). If the trial's real repetitions run out
    before crossing `threshold`, force-decodes at argmax over whatever
    was accumulated -- matching a real system's hard repetition cap.

    `lm`/`context`/`lm_temperature` are forwarded to should_decode() as-is;
    with `lm=None` (the default) should_decode() ignores `context`/
    `lm_temperature` entirely, so this is byte-identical to before these
    parameters existed unless a caller explicitly passes an `lm`.
    """
    # LiveDecoder reuses one persistent ScoreAccumulator across a message,
    # so should_decode()'s post-decode LM-prior seed carries into the next
    # trial there. A fresh ScoreAccumulator per trial here would silently
    # discard that seed -- explicitly seeding from `context` reproduces
    # what the persistent-accumulator live path gets for free.
    prior = lm.prior(context, temperature=lm_temperature) if lm is not None else None
    acc = ScoreAccumulator(prior=prior)
    true_idx = flat_target_index(trial.target_row, trial.target_col)
    decode, idx = False, 0
    for rep_num, (start, end) in enumerate(trial.rep_slices, start=1):
        for i in range(start, end):
            acc.push(int(stim_ids[i]), float(scores[i]))
        decode, idx = should_decode(
            acc,
            threshold=threshold,
            lm=lm,
            context=context,
            temperature=temperature,
            lm_temperature=lm_temperature,
        )
        if decode:
            return rep_num, idx == true_idx, idx
    idx = int(np.argmax(acc.scores()))
    return len(trial.rep_slices), idx == true_idx, idx


def evaluate_stopping_policy(
    trials: list[Trial],
    scores: np.ndarray,
    stim_ids: np.ndarray,
    threshold: float,
    temperature: float,
    lm: CharLM | None = None,
    lm_temperature: float = 1.0,
) -> tuple[float, float, float]:
    """(mean_reps, accuracy, itr) achieved by one (temperature, threshold)
    pair over a set of trials that already have decision scores computed.

    When `lm` is given, `context` grows across `trials` in order (each
    trial's own *decoded* character, via FARWELL_DONCHIN_GRID -- not the
    ground-truth target, matching what a real live session would actually
    have available). `trials` is already in real spelling order (see
    reconstruct_trials()). With `lm=None` (the default), context is never
    read by should_decode() anyway -- byte-identical to before these
    parameters existed.
    """
    reps, correct = [], []
    context = ""
    for trial in trials:
        r, c, idx = simulate_trial(
            trial, scores, stim_ids, threshold, temperature, lm, context, lm_temperature
        )
        reps.append(r)
        correct.append(c)
        if lm is not None:
            context += FARWELL_DONCHIN_GRID[idx]
    mean_reps = float(np.mean(reps))
    accuracy = float(np.mean(correct))
    seconds_per_selection = mean_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH
    itr = compute_itr(accuracy, N_SYMBOLS, seconds_per_selection)
    return mean_reps, accuracy, itr


_LOG_LOSS_EPS = 1e-6


def log_loss(proba: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true label under `proba`.

    A proper scoring rule like Brier, but unbounded rather than capped at
    1.0 -- a confidently wrong prediction (proba near 0 for a true target,
    or near 1 for a true nontarget) is penalized arbitrarily severely as
    proba approaches the wrong extreme, rather than saturating. Uses the
    real calibrated proba (see calibrate_and_score's as_proba=True) rather
    than a rescaled proxy, since the log is only meaningful for genuine
    probabilities. Clipped to [eps, 1-eps] first -- well-separated data
    routinely makes sklearn's predict_proba saturate to an exact 0.0/1.0
    in float64, which would otherwise make this -inf/nan.
    """
    p = np.clip(proba, _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


BETA_POWER_DEFAULT = 4.0


def beta_power_score(proba: np.ndarray, y: np.ndarray, beta: float = BETA_POWER_DEFAULT) -> float:
    """Bregman divergence of the convex generator phi(p)=p^beta -- a proper
    scoring rule (any convex phi gives one via D(y,p) = phi(y) - phi(p) -
    phi'(p)(y-p)) that dials continuously between Brier (beta=2) and
    something closer to log loss's harshness as beta grows, while staying
    bounded in [0,1] like Brier -- no clipping/epsilon needed, since a
    saturated proba=0.0/1.0 is finite here (unlike log loss, where it's
    +-inf). See scripts/README.md for why beta=4 specifically.

    y*(1 - beta*p^(beta-1) + (beta-1)*p^beta) covers the y=1 term;
    (1-y)*(beta-1)*p^beta covers the y=0 term (see the general Bregman
    formula above, specialized to phi(p)=p^beta with y in {0,1}).
    """
    d_target = 1 - beta * proba ** (beta - 1) + (beta - 1) * proba**beta
    d_nontarget = (beta - 1) * proba**beta
    return float(np.mean(np.where(y == 1, d_target, d_nontarget)))


def class_conditional_overlap(
    proba: np.ndarray, y: np.ndarray, n_bins: int = 20, confidently_wrong_margin: float = 0.05
) -> dict:
    """Per-class mean/std of predicted proba, the overlapping coefficient
    (OVL) between the two classes' proba distributions, and a
    confidently-wrong rate.

    OVL is the probability mass shared by both classes' empirical
    histograms (sum of the per-bin minimum, each normalized to sum to 1)
    -- 0 is perfect separation, 1 is zero discrimination. A direct read
    on separability, complementing accuracy/log-loss/Brier.

    confidently_wrong_margin sets how close to the wrong extreme counts
    as "confidently wrong" -- see scripts/README.md for why exact
    0.0/1.0 saturation alone would badly undercount this.
    """
    p_target, p_nontarget = proba[y == 1], proba[y == 0]
    bins = np.linspace(0, 1, n_bins + 1)
    h_t, _ = np.histogram(p_target, bins=bins)
    h_n, _ = np.histogram(p_nontarget, bins=bins)
    h_t_norm = h_t / h_t.sum() if h_t.sum() > 0 else h_t.astype(float)
    h_n_norm = h_n / h_n.sum() if h_n.sum() > 0 else h_n.astype(float)
    n_confidently_wrong = int(
        np.sum(p_target <= confidently_wrong_margin)
        + np.sum(p_nontarget >= 1 - confidently_wrong_margin)
    )
    return {
        "target_mean": float(p_target.mean()),
        "target_std": float(p_target.std()),
        "nontarget_mean": float(p_nontarget.mean()),
        "nontarget_std": float(p_nontarget.std()),
        "overlap_coefficient": float(np.sum(np.minimum(h_t_norm, h_n_norm))),
        "n_confidently_wrong": n_confidently_wrong,
        "confidently_wrong_rate": n_confidently_wrong / len(y),
    }


def select_n_components_shrinkage(
    all_pooled: list[tuple[np.ndarray, np.ndarray]],
    all_sessions: list[list[Session]],
    excluded_ids: frozenset[int],
    n_components_grid: list[int],
    shrinkage_grid: list[float],
    n_calib_sessions: int = 1,
    model_cache_dir: Path | None = None,
    results_path: Path | None = None,
    log_fn: Callable[[str], None] | None = None,
    use_ea: bool = True,
    use_xdawn: bool = True,
    use_calib: bool = True,
) -> tuple[tuple[int, float], dict]:
    """Inner leave-one-out over the subjects not already in `excluded_ids`:
    pick the (n_components, shrinkage) combo minimizing mean
    beta_power_score on individual flash predictions. Returns (best_combo,
    mean_scores_by_combo) -- full per-combo scores, not just the winner,
    so a caller can pull a shortlist via top_k_combos() for a real
    trial-level check rather than trusting this flash-level #1 pick.

    `all_pooled`/`all_sessions` are the full, subject-ID-indexed lists
    (all 10 subjects); `excluded_ids` says which this call can't touch.
    Filtering happens here, by real subject ID, so get_or_fit_model() can
    key its cache by which subjects were excluded -- a caller-pre-filtered
    positional list would lose that identity.

    Deliberately independent of temperature/threshold: scored per-epoch
    via calibrate_and_score(as_proba=True)/beta_power_score, not the
    trial-level stopping simulation, and still calibrated the same way
    (adapt_to_subject on each inner-held-out subject's first
    n_calib_sessions) as deployment, since calibration changes the
    probabilities being judged.

    `results_path`, if given, saves the full grid via save_grid(). `log_fn`,
    if given, logs one line per inner-held-out subject after its full
    (n_components, shrinkage) sweep -- the real unit of work here.
    """
    model_cache_dir = model_cache_dir if model_cache_dir is not None else MODEL_CACHE_DIR
    available_ids = [i for i in range(len(all_pooled)) if i not in excluded_ids]
    stage1_score_by_combo: dict[tuple[int, float], list[float]] = {
        (n_comp, shrink): [] for n_comp in n_components_grid for shrink in shrinkage_grid
    }

    n_combos_per_subject = len(n_components_grid) * len(shrinkage_grid)
    for i, inner_held_out_id in enumerate(available_ids):
        inner_train_ids = [i for i in available_ids if i != inner_held_out_id]
        inner_train = [all_pooled[i] for i in inner_train_ids]
        sessions = all_sessions[inner_held_out_id]
        calib_sessions = sessions[:n_calib_sessions]
        test_sessions = sessions[n_calib_sessions:]
        fit_excluded_ids = excluded_ids | {inner_held_out_id}
        subject_start = time.monotonic()
        subject_score_by_combo: dict[tuple[int, float], list[float]] = {}

        for n_comp in n_components_grid:
            for shrink in shrinkage_grid:
                inner_model = get_or_fit_model(
                    fit_excluded_ids,
                    n_comp,
                    shrink,
                    inner_train,
                    model_cache_dir,
                    use_ea=use_ea,
                    use_xdawn=use_xdawn,
                )

                for test_session in test_sessions:
                    _, y_sess, _, _ = test_session
                    proba = calibrate_and_score(
                        inner_model,
                        calib_sessions,
                        test_session,
                        as_proba=True,
                        use_calib=use_calib,
                    )
                    combo = (n_comp, shrink)
                    score = beta_power_score(proba, y_sess)
                    stage1_score_by_combo[combo].append(score)
                    subject_score_by_combo.setdefault(combo, []).append(score)

        if log_fn is not None:
            elapsed = time.monotonic() - subject_start
            subject_mean_by_combo = {
                combo: float(np.mean(vals)) for combo, vals in subject_score_by_combo.items()
            }
            best_combo_this_subject = min(subject_mean_by_combo, key=subject_mean_by_combo.get)
            log_fn(
                f"    stage 1 inner {i + 1}/{len(available_ids)}: excluding subject "
                f"{inner_held_out_id} done ({n_combos_per_subject} (n_components, shrinkage) "
                f"combos, {elapsed:.1f}s) -- this subject's best: {best_combo_this_subject} "
                f"beta_power_score={subject_mean_by_combo[best_combo_this_subject]:.4f}"
            )

    mean_stage1_scores = {combo: np.mean(vals) for combo, vals in stage1_score_by_combo.items()}
    if results_path is not None:
        save_grid(
            results_path,
            mean_stage1_scores,
            {"n_components": n_components_grid, "shrinkage": shrinkage_grid},
        )
    return min(mean_stage1_scores, key=mean_stage1_scores.get), mean_stage1_scores


def top_k_combos(mean_scores: dict, k: int) -> list:
    """The k combos with the lowest mean_scores value, best first.

    Exists because Stage 1's own cheap flash-level score can rank combos
    in the opposite order from real trial-level performance -- see
    scripts/README.md. Taking only the single best combo means never
    finding out if the runner-up would have been the real winner; this
    lets a caller cheaply carry a shortlist forward into a real
    trial-level check instead.
    """
    return [combo for combo, _ in sorted(mean_scores.items(), key=lambda kv: kv[1])[:k]]


Combo = tuple[
    int, float, float, float, float
]  # (n_components, shrinkage, temperature, threshold, lm_temperature)


def select_hyperparameters(
    all_pooled: list[tuple[np.ndarray, np.ndarray]],
    all_sessions: list[list[Session]],
    excluded_ids: frozenset[int],
    n_components_grid: list[int],
    shrinkage_grid: list[float],
    temperature_grid: list[float],
    threshold_grid: list[float],
    n_calib_sessions: int = 1,
    model_cache_dir: Path | None = None,
    results_path: Path | None = None,
    log_fn: Callable[[str], None] | None = None,
    use_ea: bool = True,
    use_xdawn: bool = True,
    use_calib: bool = True,
    lm: CharLM | None = None,
    lm_temperature_grid: list[float] | None = None,
) -> Combo:
    """Inner leave-one-out over the subjects not already in `excluded_ids`:
    pick the (n_components, shrinkage, temperature, threshold, lm_temperature)
    combo maximizing mean selection_score, without the true held-out
    subject ever influencing the choice -- same nesting rigor as
    classifier.py's _inner_grid_search, extended to the stopping policy.
    Same `all_pooled`/`all_sessions`/`excluded_ids` convention as
    select_n_components_shrinkage; see its docstring for why.

    `lm_temperature_grid` defaults to `[1.0]` (no-op) and is only
    actually swept when `lm` is given. Every inner-held-out subject is
    calibrated the same way as deployment (adapt_to_subject on their
    first `n_calib_sessions`) before evaluation, so the selected
    (temperature, threshold) are correct for the post-adaptation score
    distribution, not tuned cold.

    Only (n_components, shrinkage) cost a classifier refit -- often a
    cache hit already, since Stage 1 fit every one of these combos.
    Calibration and temperature/threshold apply after that fit's scores
    exist, so the whole (temperature, threshold) grid sweeps cheaply per
    fit rather than adding its own multiplicative cost.

    If no combination clears MIN_ACCURACY (every score is -inf), falls
    back to the highest pooled mean accuracy instead of an undefined tie.
    `results_path`, if given, saves the full grid via save_grid().
    """
    model_cache_dir = model_cache_dir if model_cache_dir is not None else MODEL_CACHE_DIR
    lm_temperature_grid = lm_temperature_grid or [1.0]
    available_ids = [i for i in range(len(all_pooled)) if i not in excluded_ids]
    score_by_combo: dict[Combo, list[float]] = {
        (n_comp, shrink, temp, thresh, lm_temp): []
        for n_comp in n_components_grid
        for shrink in shrinkage_grid
        for temp in temperature_grid
        for thresh in threshold_grid
        for lm_temp in lm_temperature_grid
    }
    accuracy_by_combo: dict[Combo, list[float]] = {combo: [] for combo in score_by_combo}

    n_combos_per_subject = len(n_components_grid) * len(shrinkage_grid)
    for i, inner_held_out_id in enumerate(available_ids):
        inner_train_ids = [i for i in available_ids if i != inner_held_out_id]
        inner_train = [all_pooled[i] for i in inner_train_ids]
        sessions = all_sessions[inner_held_out_id]
        calib_sessions = sessions[:n_calib_sessions]
        test_sessions = sessions[n_calib_sessions:]
        fit_excluded_ids = excluded_ids | {inner_held_out_id}
        subject_start = time.monotonic()
        subject_score_by_combo: dict[Combo, list[float]] = {}

        for n_comp in n_components_grid:
            for shrink in shrinkage_grid:
                inner_model = get_or_fit_model(
                    fit_excluded_ids,
                    n_comp,
                    shrink,
                    inner_train,
                    model_cache_dir,
                    use_ea=use_ea,
                    use_xdawn=use_xdawn,
                )

                for test_session in test_sessions:
                    _, _, stim_ids_sess, trials = test_session
                    scores = calibrate_and_score(
                        inner_model, calib_sessions, test_session, use_calib=use_calib
                    )
                    for temp in temperature_grid:
                        for thresh in threshold_grid:
                            for lm_temp in lm_temperature_grid:
                                mean_reps, accuracy, _ = evaluate_stopping_policy(
                                    trials,
                                    scores,
                                    stim_ids_sess,
                                    thresh,
                                    temp,
                                    lm=lm,
                                    lm_temperature=lm_temp,
                                )
                                combo = (n_comp, shrink, temp, thresh, lm_temp)
                                sel_score = selection_score(mean_reps, accuracy)
                                score_by_combo[combo].append(sel_score)
                                accuracy_by_combo[combo].append(accuracy)
                                subject_score_by_combo.setdefault(combo, []).append(sel_score)

        if log_fn is not None:
            elapsed = time.monotonic() - subject_start
            subject_mean_by_combo = {
                combo: float(np.mean(vals)) for combo, vals in subject_score_by_combo.items()
            }
            best_combo_this_subject = max(subject_mean_by_combo, key=subject_mean_by_combo.get)
            log_fn(
                f"    stage 2 inner {i + 1}/{len(available_ids)}: excluding subject "
                f"{inner_held_out_id} done ({n_combos_per_subject} (n_components, shrinkage) "
                f"combos, {elapsed:.1f}s) -- this subject's best: {best_combo_this_subject} "
                f"selection_score={subject_mean_by_combo[best_combo_this_subject]:.2f}"
            )

    mean_scores = {combo: np.mean(vals) for combo, vals in score_by_combo.items()}
    best_combo = max(mean_scores, key=mean_scores.get)
    if not np.isfinite(mean_scores[best_combo]):
        mean_accuracies = {combo: np.mean(vals) for combo, vals in accuracy_by_combo.items()}
        best_combo = max(mean_accuracies, key=mean_accuracies.get)
    if results_path is not None:
        finite_scores = {c: v for c, v in mean_scores.items() if np.isfinite(v)}
        save_grid(
            results_path,
            finite_scores,
            {
                "n_components": n_components_grid,
                "shrinkage": shrinkage_grid,
                "temperature": temperature_grid,
                "threshold": threshold_grid,
                "lm_temperature": lm_temperature_grid,
            },
        )
    return best_combo


def validate_final_config(
    all_pooled: list[tuple[np.ndarray, np.ndarray]],
    all_sessions: list[list[Session]],
    n_components: int,
    shrinkage: float | str,
    temperature: float,
    threshold: float,
    n_calib_sessions: int = 1,
    model_cache_dir: Path | None = None,
    log_fn: Callable[[str], None] | None = None,
    subject_ids: list[int] | None = None,
    always_excluded_ids: frozenset[int] = frozenset(),
    use_ea: bool = True,
    use_xdawn: bool = True,
    use_calib: bool = True,
    lm: CharLM | None = None,
    lm_temperature: float = 1.0,
) -> dict:
    """Real held-out check of one finalized hyperparameter combo: a
    subject-exclusion deploy model per validated subject (often cache-hit
    from earlier per-fold fitting), calibrated the same way as
    deployment, then scored via the actual ScoreAccumulator/should_decode
    trial-level simulation -- see scripts/README.md for why Stage 1/2's
    cheap flash-level proxies alone aren't trustworthy enough for this.

    `subject_ids` restricts which subjects get validated (default: all).
    `always_excluded_ids` is unioned into every validated subject's own
    exclusion set -- lets a caller nest this inside an outer LOSO fold
    (pass the fold's held-out subject here) without leaking that
    subject's own trial-level result into which hyperparameters get
    chosen for it.

    Runs once, on the single combo, across the validated subjects --
    O(n_subjects), not O(grid size), and each fit is often already
    cached from the search that produced this combo.
    """
    model_cache_dir = model_cache_dir if model_cache_dir is not None else MODEL_CACHE_DIR
    n_subjects = len(all_pooled)
    subject_ids = subject_ids if subject_ids is not None else list(range(n_subjects))
    all_reps, all_acc, all_itr = [], [], []
    per_subject: dict[int, dict[str, float]] = {}
    for held_out_idx in subject_ids:
        excluded_ids = always_excluded_ids | {held_out_idx}
        train_pooled = [all_pooled[i] for i in range(n_subjects) if i not in excluded_ids]
        model = get_or_fit_model(
            excluded_ids,
            n_components,
            shrinkage,
            train_pooled,
            model_cache_dir,
            use_ea=use_ea,
            use_xdawn=use_xdawn,
        )
        sessions = all_sessions[held_out_idx]
        calib_sessions = sessions[:n_calib_sessions]
        test_sessions = sessions[n_calib_sessions:]

        subject_reps, subject_acc, subject_itr = [], [], []
        for test_session in test_sessions:
            _, _, stim_ids_sess, trials = test_session
            scores = calibrate_and_score(model, calib_sessions, test_session, use_calib=use_calib)
            mean_reps, accuracy, itr = evaluate_stopping_policy(
                trials,
                scores,
                stim_ids_sess,
                threshold,
                temperature,
                lm=lm,
                lm_temperature=lm_temperature,
            )
            all_reps.append(mean_reps)
            all_acc.append(accuracy)
            all_itr.append(itr)
            subject_reps.append(mean_reps)
            subject_acc.append(accuracy)
            subject_itr.append(itr)

        per_subject[held_out_idx] = {
            "mean_reps": float(np.mean(subject_reps)),
            "accuracy": float(np.mean(subject_acc)),
            "itr": float(np.mean(subject_itr)),
        }
        if log_fn is not None:
            log_fn(
                f"    trial-level validation {subject_ids.index(held_out_idx) + 1}/"
                f"{len(subject_ids)}: subject {held_out_idx} done -- "
                f"accuracy={np.mean(subject_acc):.4f}, mean_reps={np.mean(subject_reps):.3f}, "
                f"itr={np.mean(subject_itr):.2f} bits/min"
            )

    return {
        "mean_reps": float(np.mean(all_reps)),
        "std_reps": float(np.std(all_reps)),
        "mean_accuracy": float(np.mean(all_acc)),
        "std_accuracy": float(np.std(all_acc)),
        "mean_itr": float(np.mean(all_itr)),
        "std_itr": float(np.std(all_itr)),
        # Additive, backward-compatible: existing callers reading only the
        # pooled mean/std keys above are unaffected. held_out_idx-keyed so
        # a future analysis can pair this same subject's deployment-config
        # validation across conditions, the same way per_fold already
        # supports for the fold-loop metrics -- see
        # scripts/analyze_ablation.py's module docstring for why this
        # was missing the first time and had to be added after the fact.
        "per_subject": per_subject,
    }


def _parse_shrinkage_grid(s: str) -> list[float | str]:
    return [t if t == "auto" else float(t) for t in s.split(",")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-components-grid",
        type=lambda s: [int(t) for t in s.split(",")],
        default=[9],
        help=(
            "9 is the best value found by nested-LOSO search so far (see "
            "model.py's P300Model default) -- pass a wider grid to explore further"
        ),
    )
    parser.add_argument(
        "--shrinkage-grid",
        type=_parse_shrinkage_grid,
        default=["auto"],
        help=(
            'comma-separated floats and/or "auto" (sklearn Ledoit-Wolf). "auto" has '
            "beaten every manual value tried in every comparison run so far"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=lambda s: [float(t) for t in s.split(",")],
        default=[0.9],
        help=(
            "0.9 is the best value found by nested-LOSO search so far -- verified as "
            "a genuine local optimum for n_components=9 (accuracy plateaus at 100%% "
            "right at 0.9; higher values only cost more repetitions)."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=lambda s: [float(t) for t in s.split(",")],
        default=[2.0],
        help="2.0 is the best value found alongside threshold=0.9 by nested-LOSO search",
    )
    parser.add_argument(
        "--n-calib-sessions",
        type=int,
        default=1,
        help="how many of a subject's sessions to use as adapt_to_subject calibration data",
    )
    parser.add_argument(
        "--validate-top-k",
        type=int,
        default=5,
        help=(
            "how many of Stage 1's top flash-level-ranked (n_components, shrinkage) "
            "combos get a REAL trial-level check (ScoreAccumulator/should_decode "
            "simulation) before picking the final config. 1 trusts Stage 1's flash-level "
            "#1 pick blindly; higher catches cases where the cheap proxy ranks a combo "
            "highly that performs worse once flashes are actually accumulated into trial "
            "decisions (verified to happen on this pipeline). Cheap to raise -- validation "
            "only adds ~O(n_subjects) work per candidate, not a new fit."
        ),
    )
    parser.add_argument(
        "--use-lm",
        action="store_true",
        help=(
            "construct a real CharLM (lm.py) and pass it into the trial-level "
            "simulation as an LM prior, instead of the default lm=None. Without this "
            "flag, --lm-temperatures is unused and behavior is byte-identical to before "
            "these flags existed."
        ),
    )
    parser.add_argument(
        "--lm-temperatures",
        type=lambda s: [float(t) for t in s.split(",")],
        default=[1.0],
        help=(
            "grid for lm_temperature (scales the LM's log-prior independently of "
            "should_decode's own temperature), searched in Stage 2 alongside "
            "temperature/threshold -- only meaningful when --use-lm is also set. "
            "1.0 is a no-op scale; no prior search data exists for this axis yet"
        ),
    )
    parser.add_argument(
        "--exploration-only",
        action="store_true",
        help=(
            "skip the 10-outer-fold LOSO loop (the expensive, ~10x-cost part) and only "
            "run the cheap final all-subjects search. For coarse-to-fine grid narrowing: "
            "use this to quickly probe a wide (n_components, shrinkage) grid, then narrow "
            "it and run without this flag for the real, trustworthy per-fold evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the two-stage per-subject LOSO hyperparameter search (Stage 1:
    n_components/shrinkage by beta_power_score; Stage 2: temperature/
    threshold via trial-level dynamic-stopping replay), then a final
    all-subjects search for a single recommended deployment config --
    see module docstring for the full per-fold vs. final-config split."""
    args = _parse_args()

    run = run_dir("simulate_dynamic_stopping")
    grids_dir = run / "grids"
    logger = TrainingLogger(run / "run.log")
    logger.log(f"Results (search grids) will be saved under {run}")
    logger.log(f"Fitted models are cached under {MODEL_CACHE_DIR}")

    lm = CharLM() if args.use_lm else None
    if args.use_lm:
        logger.log("Loading CharLM (figmtu/opt-350m-aac) -- --use-lm was passed...")

    logger.log("Loading and preprocessing all subjects...")
    grouped = load_moabb_all_subjects_grouped()

    # per subject: pooled (X, y) for training when this subject is NOT held
    # out, and per-session (X, y, stim_ids, trials) for calibrating/testing
    # when they ARE (y is kept per-session too, needed to build the
    # calibration session's labeled (X, y) for adapt_to_subject).
    subjects_pooled: list[tuple[np.ndarray, np.ndarray]] = []
    subjects_sessions: list[list[Session]] = []
    for session_raws in grouped:
        sessions_xy = []
        sessions: list[Session] = []
        for raw in session_raws:
            X, y, stim_ids, _ = preprocess_p300(raw)
            sessions_xy.append((X, y))
            sessions.append((X, y, stim_ids, reconstruct_trials(y, stim_ids)))
        subjects_pooled.append(pool_sessions(sessions_xy))
        subjects_sessions.append(sessions)

    n_subjects = len(subjects_pooled)
    per_subject_results = []

    if args.exploration_only:
        logger.log(
            "\n--exploration-only: skipping the 10-outer-fold LOSO loop, "
            "running only the cheap final all-subjects search below."
        )

    for held_out_idx in range(n_subjects) if not args.exploration_only else []:
        logger.log(f"Fold {held_out_idx + 1}/{n_subjects}: holding out subject {held_out_idx}")
        excluded_ids = frozenset({held_out_idx})

        _, mean_stage1_scores = select_n_components_shrinkage(
            subjects_pooled,
            subjects_sessions,
            excluded_ids,
            args.n_components_grid,
            args.shrinkage_grid,
            args.n_calib_sessions,
            results_path=grids_dir / f"stage1_fold{held_out_idx}",
            log_fn=logger.log,
        )

        shortlist = top_k_combos(mean_stage1_scores, args.validate_top_k)
        logger.log(f"  stage 1 flash-level shortlist (best first): {shortlist}")

        # Tie-break the shortlist by REAL trial-level performance -- but only
        # using this fold's training subjects (never held_out_idx), so
        # choosing among candidates can't leak the true held-out subject's
        # own outcome into the choice of hyperparameters evaluated on it.
        training_ids = [i for i in range(n_subjects) if i != held_out_idx]
        candidates = []
        for n_comp, shrink in shortlist:
            _, _, temp, thresh, lm_temp = select_hyperparameters(
                subjects_pooled,
                subjects_sessions,
                excluded_ids,
                [n_comp],
                [shrink],
                args.temperatures,
                args.thresholds,
                args.n_calib_sessions,
                lm=lm,
                lm_temperature_grid=args.lm_temperatures if args.use_lm else None,
            )
            nested_validation = validate_final_config(
                subjects_pooled,
                subjects_sessions,
                n_comp,
                shrink,
                temp,
                thresh,
                args.n_calib_sessions,
                subject_ids=training_ids,
                always_excluded_ids=excluded_ids,
                lm=lm,
                lm_temperature=lm_temp,
            )
            real_score = selection_score(
                nested_validation["mean_reps"], nested_validation["mean_accuracy"]
            )
            candidates.append((n_comp, shrink, temp, thresh, lm_temp, real_score))

        (
            best_n_comp,
            best_shrink,
            best_temp,
            best_threshold,
            best_lm_temp,
            _,
        ) = max(candidates, key=lambda c: c[-1])
        if (best_n_comp, best_shrink) != shortlist[0]:
            logger.log(
                "  NOTE: real (training-subjects-only) trial-level tie-break picked a "
                "different combo than the flash-level #1 pick"
            )
        logger.log(
            f"  selected n_components={best_n_comp}, shrinkage={best_shrink}, "
            f"temperature={best_temp}, threshold={best_threshold}, lm_temperature={best_lm_temp}"
        )

        train_pooled = [subjects_pooled[i] for i in range(n_subjects) if i != held_out_idx]
        model = get_or_fit_model(
            excluded_ids, best_n_comp, best_shrink, train_pooled, MODEL_CACHE_DIR
        )

        held_out_sessions = subjects_sessions[held_out_idx]
        calib_sessions = held_out_sessions[: args.n_calib_sessions]
        test_sessions = held_out_sessions[args.n_calib_sessions :]

        session_reps, session_accuracies = [], []
        for test_session in test_sessions:
            _, _, stim_ids_sess, trials = test_session
            scores = calibrate_and_score(model, calib_sessions, test_session)
            mean_reps_sess, accuracy_sess, _ = evaluate_stopping_policy(
                trials,
                scores,
                stim_ids_sess,
                best_threshold,
                best_temp,
                lm=lm,
                lm_temperature=best_lm_temp,
            )
            session_reps.append(mean_reps_sess)
            session_accuracies.append(accuracy_sess)

        # every session has the same trial count here, so averaging per-session
        # means equals averaging over all trials directly -- consistent with how
        # select_hyperparameters/select_n_components_shrinkage aggregate too.
        mean_reps = float(np.mean(session_reps))
        accuracy = float(np.mean(session_accuracies))
        seconds_per_selection = mean_reps * FLASHES_PER_ROUND * SECONDS_PER_FLASH
        itr = compute_itr(accuracy, N_SYMBOLS, seconds_per_selection)
        per_subject_results.append(
            {
                "subject_index": held_out_idx,
                "n_components": best_n_comp,
                "shrinkage": best_shrink,
                "temperature": best_temp,
                "threshold": best_threshold,
                "lm_temperature": best_lm_temp,
                "mean_reps": mean_reps,
                "accuracy": accuracy,
                "itr": itr,
            }
        )

    if per_subject_results:
        header = (
            f"{'subject':>7} | {'n_comp':>6} | {'shrink':>6} | {'temp':>5} | "
            f"{'thresh':>6} | {'lm_temp':>7} | {'mean_reps':>9} | {'accuracy':>8} | "
            f"{'itr (bits/min)':>15}"
        )
        logger.log("")
        logger.log(header)
        logger.log("-" * len(header))
        for r in per_subject_results:
            logger.log(
                f"{r['subject_index']:>7} | {r['n_components']:>6} | {r['shrinkage']:>6} | "
                f"{r['temperature']:>5} | {r['threshold']:>6.2f} | {r['lm_temperature']:>7} | "
                f"{r['mean_reps']:>9.2f} | {r['accuracy']:>8.3f} | {r['itr']:>15.2f}"
            )
        logger.log("-" * len(header))

        itrs = np.array([r["itr"] for r in per_subject_results])
        accs = np.array([r["accuracy"] for r in per_subject_results])
        reps_arr = np.array([r["mean_reps"] for r in per_subject_results])
        logger.log(
            f"chosen n_components per fold: {[r['n_components'] for r in per_subject_results]}"
        )
        logger.log(f"chosen shrinkage per fold: {[r['shrinkage'] for r in per_subject_results]}")
        logger.log(
            f"chosen temperature per fold: {[r['temperature'] for r in per_subject_results]}"
        )
        logger.log(f"chosen threshold per fold: {[r['threshold'] for r in per_subject_results]}")
        logger.log(
            f"chosen lm_temperature per fold: {[r['lm_temperature'] for r in per_subject_results]}"
        )
        logger.log(f"mean accuracy: {accs.mean():.3f} +/- {accs.std():.3f}")
        logger.log(f"mean reps used: {reps_arr.mean():.2f} +/- {reps_arr.std():.2f}")
        logger.log(f"mean ITR: {itrs.mean():.2f} +/- {itrs.std():.2f} bits/min")

    logger.log("\nRunning final search using all subjects (recommended deployment config)...")
    no_exclusions = frozenset()
    _, mean_stage1_scores = select_n_components_shrinkage(
        subjects_pooled,
        subjects_sessions,
        no_exclusions,
        args.n_components_grid,
        args.shrinkage_grid,
        args.n_calib_sessions,
        results_path=grids_dir / "stage1_final",
        log_fn=logger.log,
    )

    shortlist = top_k_combos(mean_stage1_scores, args.validate_top_k)
    logger.log(
        f"\nStage 1's cheap flash-level ranking picked a top-{len(shortlist)} shortlist "
        f"(beta_power_score, best first): {shortlist}"
    )
    logger.log(
        "Running the real trial-level check (ScoreAccumulator/should_decode) on each "
        "shortlisted combo, rather than trusting the flash-level #1 pick blindly -- "
        "flash-level and trial-level rankings can disagree..."
    )

    candidates = []
    for rank, (n_comp, shrink) in enumerate(shortlist, start=1):
        _, _, temp, thresh, lm_temp = select_hyperparameters(
            subjects_pooled,
            subjects_sessions,
            no_exclusions,
            [n_comp],
            [shrink],
            args.temperatures,
            args.thresholds,
            args.n_calib_sessions,
            results_path=grids_dir / f"stage2_shortlist{rank}",
            lm=lm,
            lm_temperature_grid=args.lm_temperatures if args.use_lm else None,
        )
        validation = validate_final_config(
            subjects_pooled,
            subjects_sessions,
            n_comp,
            shrink,
            temp,
            thresh,
            args.n_calib_sessions,
            lm=lm,
            lm_temperature=lm_temp,
        )
        real_score = selection_score(validation["mean_reps"], validation["mean_accuracy"])
        candidates.append(
            {
                "n_components": n_comp,
                "shrinkage": shrink,
                "temperature": temp,
                "threshold": thresh,
                "lm_temperature": lm_temp,
                "real_selection_score": real_score,
                **validation,
            }
        )
        logger.log(
            f"    shortlist {rank}/{len(shortlist)}: n_components={n_comp}, shrinkage={shrink} "
            f"-> temperature={temp}, threshold={thresh}, lm_temperature={lm_temp} | "
            f"REAL trial-level: accuracy={validation['mean_accuracy']:.4f} "
            f"+/- {validation['std_accuracy']:.4f}, mean_reps={validation['mean_reps']:.3f} "
            f"+/- {validation['std_reps']:.3f}, ITR={validation['mean_itr']:.2f} bits/min, "
            f"real_selection_score={real_score:.2f}"
        )

    best = max(candidates, key=lambda c: c["real_selection_score"])
    flash_level_winner = shortlist[0]
    real_winner = (best["n_components"], best["shrinkage"])
    if real_winner != flash_level_winner:
        logger.log(
            "\nNOTE: the real trial-level winner is NOT the flash-level (Stage 1) #1 pick -- "
            f"flash-level ranked {flash_level_winner} first, but "
            f"{real_winner} performs better at the trial level. Using the real winner."
        )

    logger.log(
        f"\nfinal hyperparameters (real trial-level winner among top-{len(shortlist)}): "
        f"n_components={best['n_components']}, shrinkage={best['shrinkage']}, "
        f"temperature={best['temperature']}, "
        f"threshold={best['threshold']}, lm_temperature={best['lm_temperature']}"
    )
    logger.log(
        f"trial-level validation: mean_reps={best['mean_reps']:.3f} "
        f"+/- {best['std_reps']:.3f}, accuracy={best['mean_accuracy']:.4f} "
        f"+/- {best['std_accuracy']:.4f}, ITR={best['mean_itr']:.2f} "
        f"+/- {best['std_itr']:.2f} bits/min"
    )

    logger.log(f"Full log written to {logger.log_path}")


if __name__ == "__main__":
    main()
