"""Training-side data ingestion: loading EEG recordings into a common
in-memory representation.

Every loader in this subpackage returns one or more `mne.io.Raw`
objects with EEG channels plus two embedded stim channels in one
canonical convention, regardless of source: `stim_id` (0 = no flash,
1-6 = rows, 7-12 = columns) and `target_flag` (0 = no flash,
1 = nontarget, 2 = target). Downstream code (`pipeline.preprocessing`)
derives epochs from these stim channels without caring which loader
produced them.

- `data.py` -- dataset-agnostic loading: MOABB benchmark datasets
  (`load_moabb_subject()` and friends), Lab-Recorder XDF files
  (`load_session_xdf()`), and Track A's dual-CSV format
  (`load_session_csv()`). Also the fully generic
  `load_moabb_subject_raw()`, with no stim-channel remapping applied,
  for a MOABB dataset that doesn't have one yet.
- `bnci2014_009.py` -- stim-channel remapping specific to the
  BNCI2014_009 MOABB dataset, translating its native "Flash stim"/
  "Target stim" codes into the canonical convention above. The
  template to follow when adding a new MOABB dataset.
- `muse2.py` -- stim-code remapping and channel selection for one
  specific, already-recorded Muse 2 pilot dataset (`data/
  muse2CleanData/`) -- illustrative of the pattern for bringing a new
  headset in, not a general-purpose Muse 2 loader.

Adding a new data source means writing a new dataset-specific remap
module alongside `bnci2014_009.py`/`muse2.py`, not modifying `data.py`
itself.
"""
