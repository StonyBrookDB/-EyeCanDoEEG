"""BNCI2014_009-specific MOABB stim-channel remapping.

BNCI2014_009 provides:
  "Flash stim"  -- 0 = no flash, 3-14 = which row/col flashed
  "Target stim" -- 0 = no flash, 1 = nontarget, 2 = target

Remapped here into this project's own canonical convention (see data.py's
module docstring): stim_id 1-12, target_flag 0/1/2. Kept in its own
module, separate from data.py's generic MOABB/XDF/CSV loading, so adding
a different MOABB dataset means writing a new remap module like this one
-- not touching the dataset-agnostic loading code. data.py's own
load_moabb_subject_raw() is the documented starting point for that
(inspect raw.ch_names/raw.annotations/mne.find_events(raw) to work out a
new dataset's own event convention before writing its remap module).
"""

from __future__ import annotations

import mne
import numpy as np

DATASET_NAME = "BNCI2014_009"


def remap_raw(raw: mne.io.Raw) -> mne.io.RawArray:
    """Remap one BNCI2014_009 run/session's stim channels to the canonical
    convention.

    Remapping:
      stim_id     = 0 if flash_stim == 0 else flash_stim - 2  (-> 1-12)
      target_flag = target_stim (values 0/1/2 unchanged)

    Parameters
    ----------
    raw : mne.io.Raw
        Must expose channels named "Flash stim" and "Target stim" with
        BNCI2014_009's native codes (0 = no flash, 3-14 = row/col; 0/1/2
        for no-flash/nontarget/target).

    Returns
    -------
    mne.io.RawArray
        EEG channels unchanged, plus canonical `stim_id`/`target_flag`
        channels (see module docstring). Info is rebuilt from scratch --
        montage, subject_info, and meas_date from `raw` are not carried
        over.

    Raises
    ------
    ValueError
        If `raw` has no channel named "Flash stim" or "Target stim".
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]
    eeg_data = raw.get_data(picks=eeg_picks)

    all_data = raw.get_data()
    flash_idx = raw.ch_names.index("Flash stim")
    target_idx = raw.ch_names.index("Target stim")
    flash_stim = all_data[flash_idx]
    target_stim = all_data[target_idx]

    stim_id = np.where(flash_stim == 0, 0.0, flash_stim - 2.0)
    target_flag = target_stim.copy()

    data = np.vstack([eeg_data, stim_id[np.newaxis], target_flag[np.newaxis]])
    info = mne.create_info(
        eeg_ch_names + ["stim_id", "target_flag"],
        raw.info["sfreq"],
        ch_types=["eeg"] * len(eeg_ch_names) + ["stim", "stim"],
    )
    return mne.io.RawArray(data, info)


def raw_from_session(runs: dict[str, mne.io.Raw]) -> mne.io.RawArray:
    """Concatenate all runs for one session and remap stim channels.

    Parameters
    ----------
    runs : dict[str, mne.io.Raw]
        One MOABB session's runs, keyed by run name, each with the same
        native "Flash stim"/"Target stim" channels `remap_raw` expects.
        Concatenated in dict-iteration order (MOABB's own reported run
        order).

    Returns
    -------
    mne.io.RawArray
        The concatenated runs with stim channels remapped to this
        project's canonical `stim_id`/`target_flag` convention.
    """
    raws = list(runs.values())
    raw = mne.concatenate_raws(raws)
    return remap_raw(raw)
