"""Tests for eyecando.tuning.trials: flat_target_index, reconstruct_trials,
rank_trial, and decode_trial.

Covers flat_target_index's row-major mapping; reconstruct_trials
grouping repeated-flash epochs into a single trial by matching target,
splitting on a target change, and rep_resets forcing a split even when
the target repeats (e.g. back-to-back identical characters); and
rank_trial/decode_trial correctly recovering the true target's flat
index when scores clearly favor it.
"""

import numpy as np

from eyecando.tuning.trials import (
    Trial,
    decode_trial,
    flat_target_index,
    rank_trial,
    reconstruct_trials,
)


def test_flat_target_index_matches_row_major_order() -> None:
    assert flat_target_index(1, 7) == 0
    assert flat_target_index(1, 8) == 1
    assert flat_target_index(2, 7) == 6
    assert flat_target_index(6, 12) == 35


def _one_repetition(target_row: int, target_col: int) -> tuple[np.ndarray, np.ndarray]:
    """12 flashes (stim_ids 1-12, shuffled), y=1 for the two target flashes."""
    stim_ids = np.array([7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6])
    y = (stim_ids == target_row) | (stim_ids == target_col)
    return stim_ids, y.astype(int)


def test_reconstruct_trials_groups_single_trial_by_target() -> None:
    stim_ids_list, y_list = [], []
    for _ in range(3):
        s, y = _one_repetition(2, 8)
        stim_ids_list.append(s)
        y_list.append(y)
    stim_ids = np.concatenate(stim_ids_list)
    y = np.concatenate(y_list)

    trials = reconstruct_trials(y, stim_ids)
    assert len(trials) == 1
    assert trials[0].target_row == 2
    assert trials[0].target_col == 8
    assert len(trials[0].rep_slices) == 3


def test_reconstruct_trials_splits_on_target_change() -> None:
    stim_ids_list, y_list = [], []
    for target in [(2, 8), (2, 8), (3, 9)]:
        s, y = _one_repetition(*target)
        stim_ids_list.append(s)
        y_list.append(y)
    stim_ids = np.concatenate(stim_ids_list)
    y = np.concatenate(y_list)

    trials = reconstruct_trials(y, stim_ids)
    assert len(trials) == 2
    assert (trials[0].target_row, trials[0].target_col) == (2, 8)
    assert len(trials[0].rep_slices) == 2
    assert (trials[1].target_row, trials[1].target_col) == (3, 9)
    assert len(trials[1].rep_slices) == 1


def test_reconstruct_trials_rep_resets_forces_new_trial_on_repeated_target() -> None:
    """Same target twice in a row (e.g. "LL") without rep_resets merges into
    one trial; with rep_resets marking each new character's first rep as 1,
    it correctly splits into two."""
    s1, y1 = _one_repetition(4, 10)
    s2, y2 = _one_repetition(4, 10)
    stim_ids = np.concatenate([s1, s2])
    y = np.concatenate([y1, y2])

    merged = reconstruct_trials(y, stim_ids)
    assert len(merged) == 1

    rep_resets = np.concatenate([np.full(12, 1), np.full(12, 1)])
    split = reconstruct_trials(y, stim_ids, rep_resets=rep_resets)
    assert len(split) == 2


def test_rank_trial_ranks_true_target_first_when_scores_favor_it() -> None:
    trial = Trial(target_row=2, target_col=8, rep_slices=[(0, 12)])
    stim_ids = np.arange(1, 13)
    scores = np.full(12, -1.0)
    scores[stim_ids == 2] = 5.0
    scores[stim_ids == 8] = 5.0

    ranked, true_idx = rank_trial(trial, scores, stim_ids)
    assert true_idx == flat_target_index(2, 8)
    assert ranked[0] == true_idx


def test_decode_trial_returns_argmax_and_true_index() -> None:
    trial = Trial(target_row=1, target_col=7, rep_slices=[(0, 12)])
    stim_ids = np.arange(1, 13)
    scores = np.full(12, -1.0)
    scores[stim_ids == 1] = 5.0
    scores[stim_ids == 7] = 5.0

    decoded_idx, true_idx = decode_trial(trial, scores, stim_ids)
    assert true_idx == flat_target_index(1, 7)
    assert decoded_idx == true_idx
