"""Training-side data ingestion: MOABB, XDF, and CSV sources.

All loaders return a single mne.io.Raw with EEG channels plus two embedded
stim channels:
  - stim_id:     0 = no flash, 1–6 = rows 1–6, 7–12 = columns 1–6
  - target_flag: 0 = no flash, 1 = nontarget, 2 = target

Callers derive events on demand via:
  mne.find_events(raw, stim_channel="target_flag")  # target/nontarget epochs
  mne.find_events(raw, stim_channel="stim_id")      # per-flash-id epochs

Dataset-specific stim-channel remapping (e.g. BNCI2014_009's own "Flash
stim"/"Target stim" convention) lives in its own module -- see
bnci2014_009.py -- not here. That keeps this file itself dataset-agnostic:
adding a different MOABB dataset means writing a new remap module, not
touching any of the loading functions below. load_moabb_subject_raw() is
the fully generic entry point with no remap at all, for a dataset that
doesn't have one yet.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np

from eyecando.ingestion.bnci2014_009 import DATASET_NAME as _DEFAULT_DATASET_NAME
from eyecando.ingestion.bnci2014_009 import raw_from_session as _raw_from_session


def _build_raw_with_stim(
    eeg_data: np.ndarray,
    eeg_ch_names: list[str],
    sfreq: float,
    sample_indices: np.ndarray,
    stim_ids: np.ndarray,
    target_flags: np.ndarray,
) -> mne.io.RawArray:
    """Build a RawArray with EEG plus embedded stim_id and target_flag channels.

    `stim_ids` and `target_flags` are placed as single-sample impulses at
    each index in `sample_indices`; all other samples are zero.

    Parameters
    ----------
    eeg_data : numpy.ndarray
        EEG signal in volts, shape ``(n_channels, n_samples)``.
    eeg_ch_names : list[str]
        EEG channel names, length ``n_channels``.
    sfreq : float
        Sampling rate in Hz.
    sample_indices : numpy.ndarray
        Sample index of each flash event, length ``n_events``. Indices
        equal to ``n_samples`` or beyond are silently dropped (out-of-range
        events from timestamp-alignment rounding).
    stim_ids : numpy.ndarray
        Flash location per event (1-6 rows, 7-12 cols), length ``n_events``.
    target_flags : numpy.ndarray
        Event type per event (1=non-target, 2=target), length ``n_events``.

    Returns
    -------
    mne.io.RawArray
        EEG channels followed by two stim channels named "stim_id" and
        "target_flag".
    """
    n_samples = eeg_data.shape[1]
    stim_id_ch = np.zeros((1, n_samples))
    target_flag_ch = np.zeros((1, n_samples))

    idx_mask = sample_indices <= n_samples
    stim_id_ch[0, sample_indices[idx_mask]] = stim_ids[idx_mask]
    target_flag_ch[0, sample_indices[idx_mask]] = target_flags[idx_mask]

    data = np.vstack([eeg_data, stim_id_ch, target_flag_ch])
    info = mne.create_info(
        eeg_ch_names + ["stim_id", "target_flag"],
        sfreq,
        ch_types=["eeg"] * len(eeg_ch_names) + ["stim", "stim"],
    )
    return mne.io.RawArray(data, info)


def load_moabb_subject(
    subject: int,
    dataset_name: str = _DEFAULT_DATASET_NAME,
) -> list[mne.io.RawArray]:
    """Load all sessions for one subject from a MOABB P300 dataset.

    Downloads the dataset on first call (cached automatically by MOABB, on
    disk, keyed by dataset/subject -- a later call for the same subject
    reuses the cache rather than re-downloading).

    Parameters
    ----------
    subject : int
        MOABB's own subject numbering for `dataset_name` -- not
        necessarily 0-indexed (BNCI2014_009 numbers subjects 1-10).
    dataset_name : str
        Name of a class in `moabb.datasets` (e.g. "BNCI2014_009"). Only
        meaningful for datasets that share BNCI2014_009's "Flash stim"/
        "Target stim" channel names and numeric codes -- `raw_from_session`
        (bnci2014_009.py) hardcodes both. Use `load_moabb_subject_raw` for
        a dataset without a matching remap module.

    Returns
    -------
    list[mne.io.RawArray]
        One Raw per session (each concatenated across that session's
        runs, with `stim_id`/`target_flag` channels embedded -- see module
        docstring), in the order MOABB reports sessions (typically "0",
        "1", "2" for BNCI2014_009).
    """
    import moabb.datasets

    dataset_cls = getattr(moabb.datasets, dataset_name)
    dataset = dataset_cls()
    data = dataset.get_data(subjects=[subject])
    return [_raw_from_session(runs) for runs in data[subject].values()]


def load_moabb_subject_raw(
    subject: int,
    dataset_name: str,
) -> list[mne.io.Raw]:
    """Load all sessions for one subject from an arbitrary MOABB dataset,
    without assuming any particular stim-channel naming or event-code
    convention.

    load_moabb_subject() only works for BNCI2014_009 (or another dataset
    that happens to share its exact "Flash stim"/"Target stim" channel
    names and numeric codes) -- bnci2014_009.remap_raw() hardcodes both. A
    different MOABB P300 dataset will use different channel names and
    different event codes, in a way that can't be known generically in
    advance, so this function does nothing beyond loading and
    concatenating each session's runs as-is: native channel names, native
    stim/marker channels, whatever that dataset happens to provide.

    Inspect the result yourself -- raw.ch_names, raw.annotations,
    mne.find_events(raw) -- to work out the new dataset's own event
    convention before writing a remap module like bnci2014_009.py's for it
    specifically.

    Parameters
    ----------
    subject : int
        MOABB's own subject numbering for `dataset_name` -- see
        `load_moabb_subject`.
    dataset_name : str
        Name of a class in `moabb.datasets`. No default -- unlike
        `load_moabb_subject`, this function has no dataset-specific remap
        to fall back to, so the caller must be explicit.

    Returns
    -------
    list[mne.io.Raw]
        One Raw per session, each session's runs concatenated as-is with
        that dataset's native channels/stim conventions untouched -- no
        `stim_id`/`target_flag` remapping applied.
    """
    import moabb.datasets

    dataset_cls = getattr(moabb.datasets, dataset_name)
    dataset = dataset_cls()
    data = dataset.get_data(subjects=[subject])
    return [mne.concatenate_raws(list(runs.values())) for runs in data[subject].values()]


def load_moabb_session(
    subject: int,
    session: int | str,
    dataset_name: str = _DEFAULT_DATASET_NAME,
) -> mne.io.RawArray:
    """Load a single session for one subject from a MOABB P300 dataset.

    Parameters
    ----------
    subject : int
        MOABB's own subject numbering for `dataset_name` -- see
        `load_moabb_subject`.
    session : int or str
        The session key as reported by MOABB (e.g. "0", "1", "2" for
        BNCI2014_009). An int is accepted and converted to a string before
        lookup (`str(session)`).
    dataset_name : str
        Name of a class in `moabb.datasets` -- see `load_moabb_subject`.

    Returns
    -------
    mne.io.RawArray
        That session's runs concatenated, with `stim_id`/`target_flag`
        channels embedded (see module docstring).

    Raises
    ------
    KeyError
        If `session` (as a string) is not among the subject's available
        session keys.
    """
    import moabb.datasets

    dataset_cls = getattr(moabb.datasets, dataset_name)
    dataset = dataset_cls()
    data = dataset.get_data(subjects=[subject])
    session_key = str(session)
    if session_key not in data[subject]:
        raise KeyError(
            f"Session {session!r} not found for subject {subject}; "
            f"available sessions: {sorted(data[subject])}"
        )
    return _raw_from_session(data[subject][session_key])


def load_moabb_all_subjects(
    dataset_name: str = _DEFAULT_DATASET_NAME,
) -> list[mne.io.RawArray]:
    """Load all subjects and sessions from a MOABB P300 dataset.

    Each session is treated as an independent data unit -- callers that need to
    recover the subject/session identity should use load_moabb_subject() or
    load_moabb_session() directly, or use load_moabb_all_subjects_grouped()
    if what's needed is all subjects with session boundaries intact (e.g.
    for subject-grouped cross-validation, where flattening this list and
    treating each session as its own "subject" would leak a real subject's
    other sessions into whatever fold holds one of their sessions out).

    Parameters
    ----------
    dataset_name : str
        Name of a class in `moabb.datasets` -- see `load_moabb_subject`.

    Returns
    -------
    list[mne.io.RawArray]
        Flat list, one Raw per (subject, session) pair, in subject-then-
        session order (`dataset.subject_list` order, then each subject's
        MOABB-reported session order). Length is
        ``n_subjects * n_sessions_per_subject`` -- subject/session identity
        is not recoverable from this list alone.
    """
    import moabb.datasets

    dataset_cls = getattr(moabb.datasets, dataset_name)
    dataset = dataset_cls()
    data = dataset.get_data(subjects=dataset.subject_list)
    return [
        _raw_from_session(runs)
        for subject in dataset.subject_list
        for runs in data[subject].values()
    ]


def load_moabb_all_subjects_grouped(
    dataset_name: str = _DEFAULT_DATASET_NAME,
) -> list[list[mne.io.RawArray]]:
    """Load all subjects and sessions from a MOABB P300 dataset, grouped by subject.

    Unlike load_moabb_all_subjects() (a flat list that discards subject
    boundaries), this returns one list of session-Raws per real subject --
    outer index is subject order, inner list is that subject's sessions in
    MOABB's reported order. Use this whenever subject identity must be
    preserved, e.g. before pooling each subject's sessions for
    subject-grouped cross-validation.

    Parameters
    ----------
    dataset_name : str
        Name of a class in `moabb.datasets` -- see `load_moabb_subject`.

    Returns
    -------
    list[list[mne.io.RawArray]]
        Outer list in `dataset.subject_list` order; each inner list holds
        that subject's sessions in MOABB's reported order, each session
        already concatenated across its runs with `stim_id`/`target_flag`
        embedded.
    """
    import moabb.datasets

    dataset_cls = getattr(moabb.datasets, dataset_name)
    dataset = dataset_cls()
    data = dataset.get_data(subjects=dataset.subject_list)
    return [
        [_raw_from_session(runs) for runs in data[subject].values()]
        for subject in dataset.subject_list
    ]


def load_session_xdf(
    xdf_path: str | Path,
    eeg_stream_name: str,
    marker_stream_name: str,
) -> mne.io.RawArray:
    """Load a session recording from an XDF file produced by Lab Recorder.

    Locates the EEG and marker streams by name and aligns marker timestamps
    to the nearest EEG sample. Marker payloads are expected to encode
    "target"/"nontarget" (case-insensitive) or the numeric equivalents 1/0.

    stim_id is set to 0 for all flashes (XDF marker labels do not carry
    flash identity); use target_flag to epoch by target/nontarget.

    Parameters
    ----------
    xdf_path : str or pathlib.Path
        Path to the .xdf file.
    eeg_stream_name : str
        Name of the EEG stream within the XDF file, as recorded in that
        stream's ``info["name"]``.
    marker_stream_name : str
        Name of the marker stream within the XDF file, same lookup as
        `eeg_stream_name`.

    Returns
    -------
    mne.io.RawArray
        EEG channels (named from the stream's channel descriptions when
        present, else "ch0", "ch1", ...) plus `stim_id` (always 0) and
        `target_flag` (1=nontarget, 2=target) channels.

    Raises
    ------
    ValueError
        If `eeg_stream_name` or `marker_stream_name` does not match any
        stream in the file.
    KeyError
        If a marker payload is not one of "target"/"nontarget"/"1"/"0"
        (case-insensitive).
    """
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    streams_by_name = {stream["info"]["name"][0]: stream for stream in streams}

    if eeg_stream_name not in streams_by_name:
        raise ValueError(
            f"EEG stream {eeg_stream_name!r} not found in {xdf_path}; "
            f"available streams: {sorted(streams_by_name)}"
        )
    if marker_stream_name not in streams_by_name:
        raise ValueError(
            f"Marker stream {marker_stream_name!r} not found in {xdf_path}; "
            f"available streams: {sorted(streams_by_name)}"
        )

    eeg_stream = streams_by_name[eeg_stream_name]
    marker_stream = streams_by_name[marker_stream_name]

    eeg_data = np.asarray(eeg_stream["time_series"], dtype=float).T
    eeg_timestamps = np.asarray(eeg_stream["time_stamps"], dtype=float)
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    n_channels = eeg_data.shape[0]
    ch_names = [f"ch{i}" for i in range(n_channels)]
    try:
        channel_descs = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = [ch["label"][0] for ch in channel_descs]
        if len(labels) == n_channels:
            ch_names = labels
    except (KeyError, IndexError, TypeError):
        pass

    marker_timestamps = np.asarray(marker_stream["time_stamps"], dtype=float)
    marker_labels = [str(v[0]).strip().lower() for v in marker_stream["time_series"]]

    sample_indices = np.array(
        [int(np.argmin(np.abs(eeg_timestamps - t))) for t in marker_timestamps]
    )
    label_to_flag: dict[str, float] = {"target": 2.0, "1": 2.0, "nontarget": 1.0, "0": 1.0}
    target_flags = np.array([label_to_flag[label] for label in marker_labels])
    stim_ids = np.zeros(len(sample_indices), dtype=float)

    return _build_raw_with_stim(eeg_data, ch_names, sfreq, sample_indices, stim_ids, target_flags)


def _read_csv_rows(path: str | Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, np.array(rows, dtype=float)


def load_session_csv(
    eeg_csv_path: str | Path,
    marker_csv_path: str | Path,
    sfreq: float = 250.0,
    ch_names: list[str] | None = None,
) -> mne.io.RawArray:
    """Load a session recording from Track A's dual-CSV output format.

    EEG CSV format (one row per sample): timestamp, ch0, ch1, ..., chN
    Marker CSV format (one row per flash): timestamp, flash_id, is_target,
    round_number

    Timestamps in both CSVs are LSL local_clock() values in seconds.
    EEG values are divided by 1e6 to convert Track A's microvolt (BrainFlow
    default) convention to the volts MNE expects. flash_id (1–12) is stored
    in stim_id; is_target (0/1) maps to target_flag (1=nontarget, 2=target).

    Parameters
    ----------
    eeg_csv_path : str or pathlib.Path
        Path to the EEG CSV.
    marker_csv_path : str or pathlib.Path
        Path to the marker CSV.
    sfreq : float
        Sampling rate in Hz to assign to the loaded Raw. Not read from
        either CSV -- the EEG CSV's own timestamp column is used only for
        aligning markers, not for deriving sfreq.
    ch_names : list[str] or None
        EEG channel names. None (the default) uses the EEG CSV's header
        row (all columns after the timestamp column).

    Returns
    -------
    mne.io.RawArray
        EEG channels plus `stim_id` (1-12) and `target_flag` (1=nontarget,
        2=target) channels.
    """
    eeg_header, eeg_array = _read_csv_rows(eeg_csv_path)
    eeg_timestamps = eeg_array[:, 0]
    eeg_data_v = eeg_array[:, 1:].T / 1e6

    if ch_names is None:
        ch_names = eeg_header[1:]

    _, marker_array = _read_csv_rows(marker_csv_path)
    marker_timestamps = marker_array[:, 0]
    flash_ids = marker_array[:, 1].astype(int)
    is_target = marker_array[:, 2].astype(int)

    sample_indices = np.array(
        [int(np.argmin(np.abs(eeg_timestamps - t))) for t in marker_timestamps]
    )
    stim_ids = flash_ids.astype(float)
    target_flags = np.where(is_target == 1, 2.0, 1.0)

    return _build_raw_with_stim(eeg_data_v, ch_names, sfreq, sample_indices, stim_ids, target_flags)
