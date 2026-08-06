"""Reconstruct character-selection trials from flash-level data, and score
them against the true target -- dataset-agnostic evaluation infrastructure
shared by every trial/letter-level analysis, e.g. BNCI's own Stage 3
tuning in scripts/simulate_dynamic_stopping.py.

Only depends on eyecando.decode.accumulator (to push repetition scores and
read back the accumulated per-symbol ranking) -- no should_decode/CharLM
involved, since dynamic-stopping simulation (simulate_trial(), still in
simulate_dynamic_stopping.py) is a different, policy-specific
concern from plain trial reconstruction/ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eyecando.decode.accumulator import ScoreAccumulator


@dataclass
class Trial:
    """One character-selection trial: a fixed target (row, col) and the
    epoch index ranges, one per repetition, that belong to it.

    Produced by `reconstruct_trials()`, which groups a session's flash-level
    epochs into these trials; consumed by `rank_trial()`/`decode_trial()` to
    score a trial's accumulated classifier output against its true target.
    `target_row`/`target_col` follow data.py's canonical stim_id convention
    (1-6 rows, 7-12 columns) -- the two stim_ids marked target within the
    trial's own repetitions. Each entry in `rep_slices` is a half-open
    ``[start, end)`` index range into the session's flat (per-epoch) `y`/
    `stim_ids`/`scores` arrays, one range per repetition, in recording
    order.
    """

    target_row: int  # 1-6
    target_col: int  # 7-12
    rep_slices: list[tuple[int, int]] = field(default_factory=list)


Session = tuple[np.ndarray, np.ndarray, np.ndarray, list[Trial]]  # (X, y, stim_ids, trials)


def flat_target_index(target_row: int, target_col: int, n_cols: int = 6) -> int:
    """Convert a (row, col) target into a flat index.

    Parameters
    ----------
    target_row : int
        1-6.
    target_col : int
        7-12.
    n_cols : int
        Grid width -- matches ScoreAccumulator's own ``(n_rows, n_cols)``
        flatten order (row-major), so this index always lines up with
        whatever ScoreAccumulator.scores() returns.

    Returns
    -------
    int
        Flat index into a row-major ``(n_rows, n_cols)`` grid.
    """
    return (target_row - 1) * n_cols + (target_col - 7)


def reconstruct_trials(
    y: np.ndarray, stim_ids: np.ndarray, rep_resets: np.ndarray | None = None
) -> list[Trial]:
    """Group a session's epochs into character-selection trials.

    Every repetition is one flash of each stim_id 1-12 (order randomized
    by the recording) -- detected here by watching for a stim_id repeating,
    which marks the start of a new repetition, rather than assuming a fixed
    chunk size of 12: artifact-rejected epochs can drop individual flashes,
    which breaks any fixed-size chunking (verified against real subjects
    with nonzero reject counts).

    The target row/col for a repetition is derived directly from the
    existing target/nontarget label (whichever stim_ids have y=1) -- no
    separate ground truth is needed, since a symbol's row and column are
    exactly the two flashes marked target within one repetition. A change
    in target identity between repetitions marks a new trial. If artifact
    rejection happened to drop a repetition's target flash(es) entirely
    (rare -- reject rates here are all under 1%), that repetition's target
    is assumed unchanged from the trial it's already in, rather than
    losing its evidence.

    `rep_resets`, if given, is an additional signal (same length as `y`)
    marking each epoch's repetition-within-trial number (e.g. Muse 2's own
    `rep` marker column, 1-indexed) -- when a repetition's own rep number
    is 1, that FORCES a new trial regardless of whether the target
    matches the trial in progress. Needed for phrases with
    immediately-repeated characters (e.g. "HELLO"'s double L) -- see
    tuning/README.md for the bug this fixes. Default None preserves every
    existing caller's behavior exactly (no dataset used so far --
    BNCI2014_009 included -- carries this signal).

    A forced-new-trial round that ALSO has no target evidence of its own
    (< 2 target-marked epochs, from rejection) is buffered rather than
    finalized immediately with a guessed target -- see tuning/README.md
    for why the obvious fallback is wrong here. The buffered epochs are
    reattached as the eventual new trial's own first repetition(s) once
    a later round supplies real target evidence for it.

    Parameters
    ----------
    y : numpy.ndarray
        Per-epoch target/nontarget label (1 = target, 0 = nontarget), in
        recording order.
    stim_ids : numpy.ndarray
        Per-epoch stim ID (1-6 row, 7-12 col), parallel to `y`.
    rep_resets : numpy.ndarray or None
        Per-epoch repetition-within-trial number, parallel to `y` -- see
        above. None (the default) disables the forced-new-trial signal
        entirely.

    Returns
    -------
    list[Trial]
        One Trial per detected character selection, in recording order.
    """
    n = len(y)
    trials: list[Trial] = []
    current_target: tuple[int, int] | None = None
    seen_in_rep: set[int] = set()
    rep_start = 0
    pending_slices: list[tuple[int, int]] = []

    def finalize(start: int, end: int) -> None:
        nonlocal current_target
        if end <= start:
            return
        chunk_stim = stim_ids[start:end]
        chunk_y = y[start:end]
        target_stims = chunk_stim[chunk_y == 1]
        has_target_evidence = len(target_stims) >= 2
        forced_new_trial = rep_resets is not None and int(rep_resets[start]) == 1

        if not has_target_evidence and (forced_new_trial or pending_slices):
            # This chunk opened a new trial with no target evidence of its
            # own, or a prior chunk did and we're still waiting for one
            # that resolves it -- buffer rather than fall through to the
            # current_target fallback below (see tuning/README.md).
            pending_slices.append((start, end))
            return

        target = (
            (int(min(target_stims)), int(max(target_stims)))
            if has_target_evidence
            else current_target
        )
        if target is None:
            return  # no identifiable target yet (session opened mid-drop) -- drop this stub
        if target == current_target and trials and not forced_new_trial:
            trials[-1].rep_slices.append((start, end))
        else:
            trials.append(
                Trial(
                    target_row=target[0],
                    target_col=target[1],
                    rep_slices=pending_slices + [(start, end)],
                )
            )
            pending_slices.clear()
            current_target = target

    for i in range(n):
        stim = int(stim_ids[i])
        if stim in seen_in_rep:
            finalize(rep_start, i)
            seen_in_rep = set()
            rep_start = i
        seen_in_rep.add(stim)
    finalize(rep_start, n)
    return trials


def rank_trial(trial: Trial, scores: np.ndarray, stim_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Push every real repetition into a fresh accumulator and rank symbols
    best-to-worst by the result.

    Parameters
    ----------
    trial : Trial
        Which epoch ranges (via `rep_slices`) belong to this trial, and
        its true target.
    scores : numpy.ndarray
        Per-epoch classifier decision score, indexed the same way as
        `stim_ids` (i.e. covering the whole session, not just this
        trial -- `trial.rep_slices` selects the relevant range).
    stim_ids : numpy.ndarray
        Per-epoch stim ID (1-6 row, 7-12 col), parallel to `scores`.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(ranked, true_idx)`` -- `ranked` is every flat symbol index,
        best-to-worst by accumulated score (the full ranking, not just
        argmax, so top-1/top-5/etc. can all be read off the same
        accumulation); `true_idx` is this trial's true target's flat
        index (see `flat_target_index`).
    """
    acc = ScoreAccumulator()
    for start, end in trial.rep_slices:
        for i in range(start, end):
            acc.push(int(stim_ids[i]), float(scores[i]))
    ranked = np.argsort(acc.scores())[::-1]
    true_idx = flat_target_index(trial.target_row, trial.target_col)
    return ranked, true_idx


def decode_trial(trial: Trial, scores: np.ndarray, stim_ids: np.ndarray) -> tuple[int, int]:
    """Push every real repetition (no early stop, no should_decode) and
    return the argmax decode -- the "ran out of repetitions" fallback
    every real stopping policy eventually falls back to, used standalone
    here to score decode accuracy independent of any particular
    threshold/temperature policy.

    Parameters
    ----------
    trial : Trial
        Which epoch ranges (via `rep_slices`) belong to this trial, and
        its true target.
    scores : numpy.ndarray
        Per-epoch classifier decision score, indexed the same way as
        `stim_ids` (i.e. covering the whole session -- `trial.rep_slices`
        selects the relevant range).
    stim_ids : numpy.ndarray
        Per-epoch stim ID (1-6 row, 7-12 col), parallel to `scores`.

    Returns
    -------
    tuple[int, int]
        ``(decoded_idx, true_idx)``.
    """
    acc = ScoreAccumulator()
    for start, end in trial.rep_slices:
        for i in range(start, end):
            acc.push(int(stim_ids[i]), float(scores[i]))
    decoded_idx = int(np.argmax(acc.scores()))
    true_idx = flat_target_index(trial.target_row, trial.target_col)
    return decoded_idx, true_idx
