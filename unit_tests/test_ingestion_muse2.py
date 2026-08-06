"""Tests for eyecando.ingestion.muse2 (the Muse 2 pilot-dataset loader)
and, incidentally, scripts/simulate_dynamic_stopping.py's
reconstruct_trials().

Covers remap_stim_ids()'s row/column code swap (Muse 2's convention is
the opposite of this project's canonical one) and that it's a true
involution; load_session()'s channel dropping (AF7/AF8/Right AUX),
stim_id remapping, and the muse2-specific `rep` channel; and
load_all_sessions()'s directory discovery. The reconstruct_trials
tests (imported via a sys.path hack since that helper lives under
scripts/, not src/) exercise its handling of repeated-character
trials with and without rep_resets, and buffering a repetition round
with insufficient evidence into the next trial rather than
misattributing it.
"""

import sys
from pathlib import Path

import mne
import numpy as np

from eyecando.ingestion.muse2 import (
    DROPPED_CHANNEL_NAMES,
    SAMPLING_RATE_HZ,
    load_all_sessions,
    load_session,
    remap_stim_ids,
)

# reconstruct_trials lives in scripts/, not src/eyecando/ -- no existing
# test file imports from there (no conftest.py path setup either), so
# this mirrors the sys.path pattern used ad hoc elsewhere in this project
# rather than establishing a new convention on its own.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from simulate_dynamic_stopping import reconstruct_trials  # noqa: E402

mne.set_log_level("ERROR")


def _write_csv(path, header: list[str], rows: list[list[float]]) -> None:
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def test_remap_stim_ids_swaps_the_two_halves() -> None:
    stim_id = np.array([1.0, 6.0, 7.0, 12.0, 0.0])
    remapped = remap_stim_ids(stim_id)
    np.testing.assert_array_equal(remapped, [7.0, 12.0, 1.0, 6.0, 0.0])


def test_remap_stim_ids_is_its_own_inverse() -> None:
    # swapping the two halves twice must recover the original -- if it
    # didn't, the remap wouldn't be a pure convention swap
    stim_id = np.arange(0, 13, dtype=float)
    twice = remap_stim_ids(remap_stim_ids(stim_id))
    np.testing.assert_array_equal(twice, stim_id)


def _write_muse_session(tmp_path, n_samples: int = 50, marker_rows=None):
    eeg_path = tmp_path / "eeg.csv"
    marker_path = tmp_path / "markers.csv"
    timestamps = np.arange(n_samples) / SAMPLING_RATE_HZ
    rng = np.random.default_rng(0)
    channels = {
        name: rng.uniform(10, 50, size=n_samples)
        for name in ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
    }
    _write_csv(
        eeg_path,
        ["lsl_timestamp", "TP9", "AF7", "AF8", "TP10", "Right AUX"],
        [
            [timestamps[i], *(channels[c][i] for c in ["TP9", "AF7", "AF8", "TP10", "Right AUX"])]
            for i in range(n_samples)
        ],
    )
    if marker_rows is None:
        marker_rows = [[timestamps[10] + 0.0005, 1, 1, 0, 1]]
    _write_csv(marker_path, ["lsl_timestamp", "code", "is_target", "char_idx", "rep"], marker_rows)
    return eeg_path, marker_path


def test_load_session_drops_aux_and_forehead_channels(tmp_path) -> None:
    eeg_path, marker_path = _write_muse_session(tmp_path)
    raw = load_session(eeg_path, marker_path)
    for ch in DROPPED_CHANNEL_NAMES:
        assert ch not in raw.ch_names
    assert set(raw.ch_names) == {"TP9", "TP10", "stim_id", "target_flag", "rep"}


def test_load_session_rep_channel_matches_markers_csv(tmp_path) -> None:
    eeg_path, marker_path = _write_muse_session(
        tmp_path,
        marker_rows=[
            [np.arange(50)[10] / SAMPLING_RATE_HZ + 0.0005, 1, 1, 0, 1],
            [np.arange(50)[20] / SAMPLING_RATE_HZ + 0.0005, 1, 1, 0, 3],
        ],
    )
    raw = load_session(eeg_path, marker_path)
    rep_data = raw.get_data(picks=["rep"])[0]
    assert rep_data[10] == 1.0
    assert rep_data[20] == 3.0


def test_load_session_remaps_stim_id(tmp_path) -> None:
    # code=1 (Muse: column 1) -> canonical code 7 (this project: column 1)
    eeg_path, marker_path = _write_muse_session(
        tmp_path, marker_rows=[[np.arange(50)[10] / SAMPLING_RATE_HZ + 0.0005, 1, 1, 0, 1]]
    )
    raw = load_session(eeg_path, marker_path)
    stim_id_data = raw.get_data(picks=["stim_id"])[0]
    assert stim_id_data[10] == 7.0


def test_load_session_target_flag_unchanged(tmp_path) -> None:
    eeg_path, marker_path = _write_muse_session(
        tmp_path, marker_rows=[[np.arange(50)[10] / SAMPLING_RATE_HZ + 0.0005, 1, 1, 0, 1]]
    )
    raw = load_session(eeg_path, marker_path)
    target_flag_data = raw.get_data(picks=["target_flag"])[0]
    assert (
        target_flag_data[10] == 2.0
    )  # is_target=1 -> target_flag=2, same as data.py's own convention


def _repetition(
    stim_order: list[int], target_row: int, target_col: int
) -> tuple[list[int], list[int]]:
    """One 12-flash repetition round: stim_ids in `stim_order`, y=1 for
    the two flashes matching (target_row, target_col), else 0."""
    y = [1 if s in (target_row, target_col) else 0 for s in stim_order]
    return stim_order, y


def test_reconstruct_trials_without_rep_resets_merges_repeated_letters() -> None:
    # two back-to-back repetitions with the IDENTICAL target -- without
    # rep_resets, the target-change heuristic alone can't tell "same
    # character repeating" from "a new character that happens to match",
    # e.g. HELLO's double L (see reconstruct_trials's own docstring)
    stim_ids: list[int] = []
    y: list[int] = []
    for _ in range(2):
        s, yy = _repetition([1, 2, 3, 7, 8, 9], target_row=2, target_col=8)
        stim_ids += s
        y += yy
    trials = reconstruct_trials(np.array(y), np.array(stim_ids))
    assert len(trials) == 1
    assert len(trials[0].rep_slices) == 2  # incorrectly merged


def test_reconstruct_trials_with_rep_resets_splits_repeated_letters() -> None:
    # same data as above, but with rep=1 marking each repetition's own
    # start -- rep resets to 1 for BOTH blocks (each is its own trial's
    # first repetition), forcing a real split despite the identical target
    stim_ids: list[int] = []
    y: list[int] = []
    rep_resets: list[int] = []
    for _ in range(2):
        s, yy = _repetition([1, 2, 3, 7, 8, 9], target_row=2, target_col=8)
        stim_ids += s
        y += yy
        rep_resets += [1] * len(s)
    trials = reconstruct_trials(np.array(y), np.array(stim_ids), rep_resets=np.array(rep_resets))
    assert len(trials) == 2
    assert len(trials[0].rep_slices) == 1
    assert len(trials[1].rep_slices) == 1


def test_reconstruct_trials_buffers_forced_new_trial_round_with_insufficient_evidence() -> None:
    # A new trial starts (rep=1), but THIS round is missing its
    # col-target flash (9) -- rejection drops individual flashes, it
    # doesn't duplicate them, so a realistic "lost evidence" round is one
    # stim_id short (row flash 3 present, col flash 9 simply never
    # occurs in this round), not a round with a repeated id. Only 1
    # target-marked epoch survives, not the 2 needed to read off a real
    # target. The next round (rep=2, same character) supplies the
    # missing col flash, resolving the true target (3, 9).
    #
    # Confirmed against real data (session_014, "HELLO") before this fix:
    # trusting the stale current_target fallback here silently created a
    # THIRD, wrongly-labeled "L" trial (target borrowed from the
    # in-progress trial), decoding "HELLLO" instead of "HELLO" -- see
    # reconstruct_trials's own docstring for the full mechanism.
    s0, y0 = _repetition([1, 2, 3, 7, 8, 9], target_row=2, target_col=8)
    s1, y1 = [1, 2, 4, 5, 6, 3], [0, 0, 0, 0, 0, 1]  # col flash 9 missing this round
    s2, y2 = _repetition([1, 2, 4, 5, 9, 3], target_row=3, target_col=9)

    stim_ids = np.array(s0 + s1 + s2)
    y = np.array(y0 + y1 + y2)
    rep_resets = np.array([1] * len(s0) + [1] * len(s1) + [2] * len(s2))

    trials = reconstruct_trials(y, stim_ids, rep_resets=rep_resets)
    assert len(trials) == 2
    assert (trials[0].target_row, trials[0].target_col) == (2, 8)
    assert (trials[1].target_row, trials[1].target_col) == (3, 9)
    # the buffered insufficient-evidence round (s1) is reattached as
    # trial 1's own first (evidence-poor) repetition, not discarded and
    # not misattributed to trial 0
    assert len(trials[1].rep_slices) == 2


def test_load_all_sessions_finds_every_session_dir(tmp_path) -> None:
    for name in ["session_002", "session_001"]:
        d = tmp_path / name
        d.mkdir()
        _write_muse_session(d)

    raws = load_all_sessions(tmp_path)
    assert len(raws) == 2
    assert all(isinstance(r, mne.io.RawArray) for r in raws)
