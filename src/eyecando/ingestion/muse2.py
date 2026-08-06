"""Muse 2 dataset-specific stim-code remapping (data/muse2CleanData/).

Illustrative, not a general-purpose Muse 2 ingestion path -- built
around one specific, already-recorded pilot dataset (fixed directory/CSV
layout, a single subject's sessions, channel-drop choices tuned to that
dataset's own measured noise). Demonstrates the pattern for bringing a
new headset into this project's canonical stim_id/target_flag
convention (same division of labor as bnci2014_009.py); not ready to
point at arbitrary new Muse 2 recordings without review.

Each session directory holds a dual-CSV pair matching data.py's own
load_session_csv() format plus a meta.json describing the recording:
  eeg.csv     -- lsl_timestamp, TP9, AF7, AF8, TP10, Right AUX (5 channels,
                 256 Hz). Only TP9/TP10 are kept -- see ingestion/README.md
                 for the channel-selection rationale.
  markers.csv -- lsl_timestamp, code, is_target, char_idx, rep
                 is_target is already 0/1, matching load_session_csv()'s
                 expected column shape directly -- no remap needed there.
                 `code` does need remapping -- see ingestion/README.md
                 and remap_stim_ids() below.

Kept in its own module, separate from data.py's generic CSV loading, same
division of labor as bnci2014_009.py. Uses data.py's load_session_csv()
unmodified for parsing/uV-to-volts conversion/channel setup -- dropped
channels are removed AFTER loading (via Raw.drop_channels()), not by
passing a shorter ch_names list, since that function's eeg_data array
always has one row per raw CSV column regardless of what ch_names is
given -- passing fewer names raises a channel-count mismatch in
mne.create_info(), it does not select a subset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np

from eyecando.ingestion.data import load_session_csv

DATASET_NAME = "muse2CleanData"
SAMPLING_RATE_HZ = 256.0
# Right AUX is never real EEG; AF7/AF8 are dropped for measured data-quality
# reasons (see module docstring) -- TP9/TP10 are the only channels kept.
DROPPED_CHANNEL_NAMES = ["Right AUX", "AF7", "AF8"]
EEG_CHANNEL_NAMES = ["TP9", "TP10"]


def remap_stim_ids(stim_id: np.ndarray) -> np.ndarray:
    """Swap the two 1-6/7-12 code halves: Muse 2's own convention
    (1-6=columns, 7-12=rows) is the opposite of this project's canonical
    one (1-6=target_row, 7-12=target_col) -- see ingestion/README.md.

    Parameters
    ----------
    stim_id : numpy.ndarray
        Raw `code` values in Muse 2's own convention (0, or 1-12).

    Returns
    -------
    numpy.ndarray
        Same shape, with 1-6 and 7-12 swapped and 0 left as-is.
    """
    return np.where(stim_id == 0, 0.0, np.where(stim_id <= 6, stim_id + 6.0, stim_id - 6.0))


def _read_csv_rows(path: str | Path) -> np.ndarray:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return np.array([[float(v) for v in row] for row in reader])


def _read_rep_channel(eeg_csv_path: str | Path, marker_csv_path: str | Path) -> np.ndarray:
    """Build a `rep` stim channel aligned to eeg.csv's own real
    `lsl_timestamp` values, the same way data.py's load_session_csv()
    aligns stim_id/target_flag -- read as its own pass over both CSVs
    since load_session_csv() only reads markers.csv columns 1-2
    (flash_id/is_target), not column 4 (rep).

    Must use eeg.csv's real timestamps, not `arange(n_samples) / sfreq`
    -- the latter starts at 0, while `lsl_timestamp` starts at whatever
    absolute LSL clock value recording began at, the same frame
    markers.csv's timestamps are in. A 0-based reconstruction would
    silently misalign every marker.

    Needed for reconstruct_trials()'s `rep_resets` parameter -- see that
    function's own docstring for why.
    """
    eeg_array = _read_csv_rows(eeg_csv_path)
    eeg_timestamps = eeg_array[:, 0]

    marker_array = _read_csv_rows(marker_csv_path)
    marker_timestamps = marker_array[:, 0]
    rep = marker_array[:, 4]

    sample_indices = np.array(
        [int(np.argmin(np.abs(eeg_timestamps - t))) for t in marker_timestamps]
    )
    rep_ch = np.zeros(len(eeg_timestamps))
    idx_mask = sample_indices <= len(eeg_timestamps)
    rep_ch[sample_indices[idx_mask]] = rep[idx_mask]
    return rep_ch


def load_session(eeg_csv_path: str | Path, marker_csv_path: str | Path) -> mne.io.RawArray:
    """Load one Muse 2 session (eeg.csv + markers.csv): 2 EEG channels
    (TP9/TP10 only -- Right AUX and AF7/AF8 dropped, see module
    docstring), stim_id remapped to this project's canonical row/col
    convention, plus a `rep` stim channel (Muse-specific, not part of
    data.py's canonical stim_id/target_flag pair) for
    reconstruct_trials()'s `rep_resets` parameter. target_flag needs no
    remap (is_target is already 0/1, matching load_session_csv()'s own
    expectation).

    Parameters
    ----------
    eeg_csv_path : str or pathlib.Path
        Path to that session's eeg.csv.
    marker_csv_path : str or pathlib.Path
        Path to that session's markers.csv.

    Returns
    -------
    mne.io.RawArray
        TP9/TP10 EEG channels plus `stim_id` (remapped), `target_flag`,
        and `rep` stim channels.
    """
    raw = load_session_csv(eeg_csv_path, marker_csv_path, sfreq=SAMPLING_RATE_HZ)
    raw.drop_channels(DROPPED_CHANNEL_NAMES)

    stim_idx = raw.ch_names.index("stim_id")
    assert raw._data is not None  # RawArray is always constructed pre-loaded from an array
    raw._data[stim_idx] = remap_stim_ids(raw._data[stim_idx])

    rep_ch = _read_rep_channel(eeg_csv_path, marker_csv_path)
    rep_info = mne.create_info(["rep"], SAMPLING_RATE_HZ, ch_types=["stim"])
    rep_raw = mne.io.RawArray(rep_ch[np.newaxis, :], rep_info, verbose="ERROR")
    raw.add_channels([rep_raw], force_update_info=True)
    return raw


def load_all_sessions(data_dir: str | Path = "data/muse2CleanData") -> list[mne.io.RawArray]:
    """Load every session under `data_dir` (one Raw per session_* folder,
    sorted by name for a reproducible order). No subject grouping --
    meta.json carries no subject field and everything about this dataset
    (single consistent 4-channel montage, no per-subject directory
    structure) points to these being one subject's sessions, not several
    subjects' -- callers needing leave-one-out validation on this dataset
    should treat each session as the held-out unit, not each subject
    (there's only one).

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing one subdirectory per session, each named
        ``session_*`` and holding that session's eeg.csv/markers.csv pair.

    Returns
    -------
    list[mne.io.RawArray]
        One Raw per session (see `load_session`), in sorted directory-name
        order.
    """
    data_dir = Path(data_dir)
    session_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("session_")
    )
    return [load_session(d / "eeg.csv", d / "markers.csv") for d in session_dirs]
