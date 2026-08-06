"""Shared preprocessing: filtering and epoching.

Pure transform -- its only eyecando imports are bandpass.py (build_sos()/
apply_filter(), themselves pure) and alignment.py (EuclideanAligner).

NOTE -- preprocess_p300() as implemented here is batch/offline only; it
cannot be called per-epoch on live/streaming data. See pipeline/README.md
for why, and for how the live path (streaming.py) assembles the
equivalent pieces instead.
"""

from __future__ import annotations

import mne
import numpy as np

from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.bandpass import apply_filter, build_sos

_REJECT_DEFAULT: dict[str, float] = {"eeg": 100e-6}


def preprocess_p300(
    raw: mne.io.Raw,
    tmin: float = -0.1,
    tmax: float = 0.8,
    l_freq: float = 0.1,
    h_freq: float = 20.0,
    baseline: tuple[float | None, float | None] = (None, 0),
    reject: dict[str, float] | None = _REJECT_DEFAULT,
    picks: str = "eeg",
    use_ea: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Filter, epoch, and (optionally) Euclidean-align one session of P300 data.

    Events are derived from the raw's target_flag stim channel
    (1 = nontarget, 2 = target), which all data.py loaders embed.
    Uses the same causal IIR filter (build_sos/apply_filter) as the live
    path so training and inference see identical waveform shapes.
    Euclidean Alignment is applied per-session so cross-session amplitude
    differences are removed before any model sees the data.
    Modifies `raw` in place; pass `raw.copy()` if the original is needed.

    `use_ea=False` skips the EuclideanAligner().fit(X).transform(X) call
    entirely and returns the filtered/epoched data unchanged -- for the
    component ablation study, isolating EA's own contribution independent
    of xDawn/LM. Default True preserves every existing caller's behavior
    exactly.

    Parameters
    ----------
    raw : mne.io.Raw
        One session's continuous recording, with `target_flag` and
        `stim_id` stim channels embedded (see data.py's loaders).
        Modified in place by the bandpass filter step.
    tmin, tmax : float
        Epoch window relative to each stimulus onset, in seconds.
    l_freq, h_freq : float
        Bandpass cutoff frequencies, in Hz, passed to `build_sos`.
    baseline : tuple[float or None, float or None]
        Passed directly to `mne.Epochs` -- the window used to compute
        each epoch's baseline correction.
    reject : dict[str, float] or None
        Passed directly to `mne.Epochs`'s `reject` -- peak-to-peak
        amplitude thresholds per channel type, in volts. None disables
        rejection entirely.
    picks : str
        Channel selection passed to `mne.Epochs`/`get_data`.
    use_ea : bool
        True (default) fits and applies Euclidean Alignment per-session;
        False returns the filtered/epoched data unchanged.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int]
        ``(X_ea, y, stim_ids, n_rejected)``. ``X_ea`` is shape
        ``(n_epochs, n_channels, n_timepoints)`` (EA-whitened unless
        `use_ea=False`). ``y`` is shape ``(n_epochs,)``, 1 for target / 0
        for nontarget. ``stim_ids`` is shape ``(n_epochs,)``, the integer
        stimulus ID (1-12) that flashed on each epoch. ``n_rejected`` is
        the count of epochs dropped by `reject`.
    """
    events = mne.find_events(raw, stim_channel="target_flag", verbose=False)
    event_id = {"nontarget": 1, "target": 2}

    stim_events = mne.find_events(raw, stim_channel="stim_id", verbose=False)
    event_sample_to_stim = dict(zip(stim_events[:, 0], stim_events[:, 2]))

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    sos = build_sos(l_freq, h_freq, raw.info["sfreq"])
    # Every ingestion path in this project (data.py, bnci2014_009.py, muse2.py)
    # constructs `raw` as an mne.io.RawArray, which is always preloaded by
    # construction -- `raw` is typed as the general mne.io.Raw here only to
    # keep this function's interface open to any MNE Raw, not because a
    # lazy/non-preloaded Raw is actually expected to reach it.
    assert raw._data is not None
    raw._data[eeg_picks], _ = apply_filter(sos, raw.get_data(picks=eeg_picks))

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        picks=picks,
        preload=True,
        verbose="WARNING",
    )
    rejects = sum(1 for reasons in epochs.drop_log if reasons)

    stim_ids = np.array([event_sample_to_stim[s] for s in epochs.events[:, 0]], dtype=int)

    X = epochs.get_data(picks="eeg")
    y = (epochs.events[:, 2] == 2).astype(int)
    X_out = EuclideanAligner().fit(X).transform(X) if use_ea else X
    return X_out, y, stim_ids, rejects
