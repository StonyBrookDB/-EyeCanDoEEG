# Emotiv Flex QC Dataset

This directory contains the QC-passing Emotiv Flex P300 epochs from
`emotivFlex-batch1` (EmotivPRO EDF export) and `emotivFlex-batch2` (Cortex
API/LSL). It is intended as a convenient input for downstream training and
analysis. Source recordings are not modified.

## Contents

- `<batch>_session_NNN.npz`: one compressed file per session, directly in
  this directory.
- `<batch>_session_NNN.json`: session metadata and preprocessing provenance,
  directly in this directory.
- `visualizations/`: generated figures for example sessions.
- `manifest.csv`: one row per exported session.
- `dataset.json`: dataset-level counts and provenance.

To inspect a session, from this project directory run:

```powershell
python analysis/visualize_flex_dataset_session.py batch1_session_005
```

The overview figure is saved under `visualizations/`; the script also prints
array shapes, label/code counts, per-character retained epochs, alignment
metadata, and the strongest target-minus-nontarget channels.
It also creates `*_per_character_repetitions.png`: one panel per character
with its 15 Pz target-minus-nontarget ERP curves (one curve per repetition),
plus a character-by-repetition heatmap of the 300–500 ms mean difference.
Each repetition has only about two target and ten non-target flashes, so
individual curves are noisy/descriptive; the heatmap summarizes the trend.

## NPZ arrays

```python
import numpy as np

with np.load("batch1_session_005.npz") as data:
    X = data["epochs_uv"]  # (n_epochs, 32 channels, 57 time samples), float32, microvolts
    y = data["y"]           # 1=target flash, 0=non-target flash
```

Other arrays:

- `stim_code`: canonical 1–6 column / 7–12 row flash code.
- `char_idx`, `repetition`: character and repetition indices from markers.
- `target_row`, `target_col`: zero-based target grid coordinates.
- `target_symbol`: symbol recovered from target row/column marker codes.
- `marker_time_s`: original marker timestamp.
- `aligned_event_time_s`: marker timestamp shifted by the estimated session lag.
- `epoch_sample_times_s`: time coordinate for each epoch sample, relative to
  the aligned event.
- `channel_names`, `bad_channels`: channel order and detected bad channels.

The target labels are taken from the row/column target codes in `markers.csv`,
not solely from `meta.json`'s phrase; this handles known phrase/marker
disagreement in `batch1/session_001`.

## Preprocessing

- Continuous EEG is filtered with a 4th-order 0.5–20 Hz Butterworth SOS
  filter, forward and backward.
- Effective sampling rate is estimated from EEG timestamps. Epochs are
  block-averaged by four samples to approximately 64 Hz.
- Each epoch spans approximately −94 to +797 ms around the aligned event.
- Each epoch/channel has its pre-stimulus baseline subtracted.
- Epoch artifact rejection: reject if at least four channels exceed eight
  robust standard deviations within the epoch.
- `epochs_uv` remain in microvolts. Detected bad channels are retained for
  provenance; consult each sidecar's `bad_channels` list and exclude them
  from model fitting as appropriate.

The QC rule is median channel robust SD ≤ 10 µV. It excludes four noisy
batch1 sessions (sessions 001–004), leaving 26 sessions, 150 characters, and
approximately 27,000 flash epochs. This threshold was chosen after initial
decoding results were inspected; it is post hoc, not preregistered.

## Note

1. Session marker lag was estimated by matching each session's labelled
   target-minus-nontarget difference wave to a template. This uses the
   session's target labels and is supervised alignment, not label-free
   test-time preprocessing. See the sidecar fields `marker_alignment` and
   `lag_estimate_reliable`. Low-confidence estimates use the median lag from
   reliable sessions in the same batch.
2. For unbiased model evaluation, keep all characters from a session in the
   same split. Do not randomly split flash epochs from the same character
   across training and test sets.