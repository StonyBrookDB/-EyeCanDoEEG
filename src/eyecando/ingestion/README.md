# eyecando.ingestion — Stage 1: get data in

Dataset-agnostic loading, plus one remap module per real data source.

- **`data.py`** — the generic loaders: MOABB (`load_moabb_subject`,
  `load_moabb_session`, `load_moabb_all_subjects[_grouped]`,
  `load_moabb_subject_raw`), XDF (`load_session_xdf`), and dual-CSV
  (`load_session_csv`). All return `mne.io.Raw`/`RawArray` with two
  embedded stim channels in this project's own canonical convention:
  `stim_id` (0 = no flash, 1-6 = rows, 7-12 = columns) and `target_flag`
  (0 = no flash, 1 = nontarget, 2 = target). This file has no knowledge
  of any specific dataset's native event codes.
- **`bnci2014_009.py`** — remaps MOABB's `BNCI2014_009` (the research-grade
  EEG cap used for training/tuning) from its own `"Flash stim"`/`"Target
  stim"` channels into the canonical convention above.
- **`muse2.py`** — remaps a Muse2 (consumer headset) session's CSV export
  into the same canonical convention.

## Adding a new dataset or headset

1. Load a sample recording and inspect it: `raw.ch_names`,
   `raw.annotations`, `mne.find_events(raw)` (for a MOABB dataset,
   `data.py`'s own `load_moabb_subject_raw()` is the fully generic
   entry point with no remap at all, for exactly this purpose).
2. Work out that source's own event/channel convention.
3. Write a new remap module here, following `bnci2014_009.py` (a MOABB
   dataset) or `muse2.py` (a non-MOABB recording) as the closer template
   for your case. It only needs to produce the two canonical stim
   channels above -- nothing downstream needs to know it exists.

## Why these numbers: investigation notes

### Muse2 channel selection (`muse2.py`)

Only TP9/TP10 (behind-ear) are kept; "Right AUX" is Muse2's reference/
aux channel (never real EEG) and AF7/AF8 (forehead) were dropped
deliberately after directly measuring this dataset's real per-channel
noise: TP9/TP10 are reliably clean in every recorded session (median
peak-to-peak 33-47uV, 95th %ile 46-94uV, everywhere), while AF7/AF8 are
inconsistently and sometimes severely contaminated. This is a property
of the one pilot dataset this module was built against
(`data/muse2CleanData/`), not a general Muse2 hardware limitation --
revisit if the headset or recording setup changes.

### Muse2's stim_id convention is inverted (`muse2.py`, `remap_stim_ids()`)

Muse2's own `code` column uses "1-6 = columns, 7-12 = rows" -- the
**opposite** of this project's canonical convention (1-6 = target_row,
7-12 = target_col; see `data.py`'s module docstring). Left unremapped,
every reconstructed trial's flat grid index would be transposed against
`FARWELL_DONCHIN_GRID`'s non-symmetric 6x6 layout, **silently
corrupting which character each trial actually targets** -- no
exception, no obviously-wrong output, just consistently wrong labels.
`remap_stim_ids()` is the fix; if a new headset's own convention needs
remapping too, check row/col orientation specifically, not just value
range.
