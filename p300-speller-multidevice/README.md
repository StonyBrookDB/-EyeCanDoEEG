# p300-speller-multidevice

Offline P300 speller pipeline: collect EEG while a subject copy-spells a
known phrase on a 6x6 Farwell-Donchin grid, then validate and decode the
recording. Four acquisition front-ends are supported — a custom 16-channel
serial headset, a Muse2 (BrainFlow/BLE), and an Emotiv EPOC Flex (two
workflows) — that produce recordings in a compatible format so every
downstream script works with any of them. Also includes a decoder benchmark
(classical xDAWN+Riemannian vs. EEGNet, with Leave-One-Subject-Out
cross-validation) run against the public BNCI2014_009 dataset and against
our own Muse2 recordings.

This is a continuation of the earlier P300 work in [`p300w6by6/`](../p300w6by6/):
it adds the Emotiv EPOC Flex front-ends, session pooling, LOSO validation,
and the classical-vs-EEGNet benchmark.

## Layout

```
p300-speller-multidevice/
├── eeg_serial_collect_16ch.py           acquisition + decode/validate scripts:
├── p300_speller_experiment_16ch.py      kept flat (not in a sub-folder) because
├── p300_speller_experiment_muse2.py     they import each other directly
├── p300_speller_experiment_emotiv.py    (e.g. p300_speller_experiment_16ch.py
├── p300_speller_experiment_emotiv_cortex.py  imports eeg_serial_collect_16ch.py)
├── emotiv_cortex_acquisition.py         and resolve `recordings/` relative to
├── decode_emotivpro.py                  their own file location — splitting
├── visualize_p300.py                    them into sub-folders would break both
├── pool_sessions.py                     the imports and the default paths.
├── train_classical_multi.py
├── train_eegnet_multi.py
└── bnci_benchmark/                      BNCI2014_009 decoder benchmark — no
    ├── train_classical_bnci.py          cross-imports and no dependency on
    ├── train_classical_bnci_loso.py     recordings/, so it's split out on
    ├── train_eegnet_bnci.py             its own.
    ├── train_eegnet_bnci_loso.py
    ├── plot_decoder_comparison.py
    └── results_comparison.md
```

## Pipeline

### Custom 16ch serial / Muse2

```
1. Collect     p300_speller_experiment_16ch.py  or  p300_speller_experiment_muse2.py
                        |
                        v
               recordings/session_NNN/{eeg.csv, markers.csv, meta.json}
                        |
2. Validate     visualize_p300.py, pool_sessions.py, train_classical_multi.py
                        |
3. Decode       train_eegnet_multi.py     (EEGNet, PyTorch)
```

### Emotiv EPOC Flex — EmotivPRO workflow (no Cortex API license required)

```
1. Collect     Open EmotivPRO and click Record
               python p300_speller_experiment_emotiv.py --phrase HELLO --reps 15
               When done: stop EmotivPRO, export recording as EDF
                        |
                        v
               recordings/session_NNN/{markers.csv, meta.json}
               + <exported>.edf  (from EmotivPRO)
                        |
2. Decode      python decode_emotivpro.py recordings/session_NNN/ <exported>.edf
                        |
                        v
               out_erp_emotivpro.png, out_erp_per_channel_emotivpro.png
               + terminal: letter accuracy per rep count
```

### Emotiv EPOC Flex — Cortex API workflow (requires Cortex API EEG license)

```
1. Collect     python p300_speller_experiment_emotiv_cortex.py --phrase HELLO --reps 15
                        |
                        v
               recordings/session_NNN/{eeg.csv, markers.csv, meta.json}
                        |
2. Validate     visualize_p300.py, train_classical_multi.py  (same as above)
```

`eeg_serial_collect_16ch.py` is a standalone driver/debug tool for the
custom headset, used internally by `p300_speller_experiment_16ch.py`.

### Decoder benchmark (`bnci_benchmark/` — BNCI2014_009 + Muse2)

```
train_classical_bnci.py / train_classical_bnci_loso.py    xDAWN+Riemannian on BNCI2014_009
train_eegnet_bnci.py / train_eegnet_bnci_loso.py           EEGNet on BNCI2014_009
plot_decoder_comparison.py                                 figure: Classical vs EEGNet, Muse2 sessions
results_comparison.md                                      written summary of all results above
```

## Scripts

### 1. `eeg_serial_collect_16ch.py`
Low-level serial driver for the custom 16-channel headset. Parses the
`0xAB ... 0xDC 0xBA` frame format (16 x 3-byte signed channel values) off a
COM port. Run directly for a quick standalone capture to a timestamped CSV
(`CurveData_HHMMSS.csv`); otherwise it's imported by
`p300_speller_experiment_16ch.py` as the `EEG_Driver` class.

```
python eeg_serial_collect_16ch.py
```
COM port (`COM3`) and buffer size are hardcoded at the bottom of the file —
edit before running if your setup differs.

### 2. `p300_speller_experiment_16ch.py`
Runs the actual 6x6 P300 copy-spelling experiment on the **custom 16ch
serial headset**. Flashes rows/columns of the grid, streams EEG + stimulus
markers over LSL, and records a fully-labelled session to
`recordings/session_NNN/`.

```
# real experiment, full-screen GUI
python p300_speller_experiment_16ch.py --phrase HELLO --reps 15 --serial-port COM3

# headless dry-run (no window) to check the streaming/recording pipeline
python p300_speller_experiment_16ch.py --phrase AB --reps 3 --headless
```

Key flags: `--phrase` (string to spell, letters must be on the grid),
`--reps` (stimulus repetitions per character), `--serial-port` / `--baud`,
`--outdir` (default `recordings/session_NNN`), `--no-record` (broadcast LSL
only, e.g. if a second machine is recording), `--seed`, and
`--on_ms`/`--off_ms`/`--cue_ms`/`--rest_ms` for flash timing.

On first run it waits for a keypress in the console to open the GUI, then a
click/keypress in the GUI window to actually start streaming — nothing is
recorded before that second gate.

### 3. `p300_speller_experiment_muse2.py`
Identical paradigm, GUI, marker scheme, and output format as the script
above, but acquires from a **Muse2 headset over BrainFlow/BLE** instead of
the serial board. Set `MUSE_NAME` at the top of the file to your device's
BLE name (from `muselsl list`).

```
python p300_speller_experiment_muse2.py --phrase HELLO --reps 15
python p300_speller_experiment_muse2.py --phrase AB --reps 3 --headless
```

Same flags as the 16ch version, minus `--serial-port`/`--baud`.

### 4. `visualize_p300.py` / `pool_sessions.py` / `train_classical_multi.py`
Filter -> epoch -> check for a real P300, at increasing scale: one session,
then pooled sessions with a permutation-test p-value, then a pooled
xDAWN + Riemannian + logistic regression decoder scored by held-out
character accuracy.

```
python visualize_p300.py                                        # latest recordings/session_*
python pool_sessions.py recordings/session_001 recordings/session_002 ...
python train_classical_multi.py                                 # all recordings/session_*
```

Outputs: `out_psd.png` / `out_erp_p300.png` / `out_erp_per_channel.png`
(per session, from `visualize_p300.py`), `out_pooled_erp.png` + printed
p-value (`pool_sessions.py`), printed ROC-AUC + per-repetition-budget
accuracy (`train_classical_multi.py`).

### 5. `p300_speller_experiment_emotiv.py` — EmotivPRO manual workflow

Runs the 6x6 P300 stimulus on the **Emotiv EPOC Flex** without requiring a
Cortex API EEG license. EEG is recorded manually in EmotivPRO; this script
handles only stimulus presentation and saves `markers.csv` with wall-clock
timestamps (`time.time()`, UTC Unix seconds) that align with the timestamps
EmotivPRO embeds in its exported EDF.

**Step-by-step:**

1. Open EmotivPRO, check electrode impedance, then click **Record**.
2. Run the script — it shows a 3-second countdown before the first flash.
3. When the window shows **DONE**, stop the EmotivPRO recording.
4. In EmotivPRO: **File → Export → EDF** and save the `.edf` file.
5. Run `decode_emotivpro.py` to align and decode (see below).

```
python p300_speller_experiment_emotiv.py --phrase HELLO --reps 15
python p300_speller_experiment_emotiv.py --phrase AB --reps 3 --headless
```

Credentials are not needed. Output: `recordings/session_NNN/{markers.csv, meta.json}`.

### 6. `p300_speller_experiment_emotiv_cortex.py` — Cortex API workflow

Same paradigm as above but streams raw EEG directly from the headset via
the **Emotiv Cortex WebSocket API** (`wss://localhost:6868`), so no manual
EmotivPRO recording step is needed. Requires a Cortex API EEG license on
the Emotiv developer account (error `-32232` at runtime means the license
is missing — see the Cortex EEG license note below).

**Prerequisites:**

- Register a free developer app at emotiv.com/my-account/cortex-apps to get
  a Client ID and Client Secret.
- Store them in `.env` at the project root (never commit this file):
  ```
  EMOTIV_CLIENT_ID=your_id_here
  EMOTIV_CLIENT_SECRET=your_secret_here
  ```
- Emotiv Cortex app (or EmotivPRO) must be **running** before launch — it
  hosts the WebSocket server on port 6868.
- `pip install websocket-client python-dotenv`

**First run only:** Cortex shows an approval dialog — click **Allow** in the
EMOTIV Launcher before the script can proceed.

```
python p300_speller_experiment_emotiv_cortex.py --phrase HELLO --reps 15
python p300_speller_experiment_emotiv_cortex.py --phrase AB --reps 3 --headless
```

Output: `recordings/session_NNN/{eeg.csv, markers.csv, meta.json}` — same
format as the 16ch/Muse2 sessions, so `visualize_p300.py` and
`train_classical_multi.py` work unchanged.

The Cortex client lives in `emotiv_cortex_acquisition.py` (imported by this
script); it defines `CortexClient`/`CortexAcquisition` and has no `__main__`
entry point of its own, so it is never run directly.

**Cortex EEG license note:** the license lives on the Emotiv developer
account, not in this repo. There is no local way to query its status —
running the script above and checking whether the `subscribe eeg` step
succeeds is the only check. `.env` only holds the Client ID/Secret used to
request access; it is not the license itself.

### 7. `decode_emotivpro.py`

Aligns an EmotivPRO EDF export with `markers.csv` from
`p300_speller_experiment_emotiv.py`, then plots ERPs and runs the xDAWN +
Riemannian decoder.

**How alignment works:** `markers.csv` stores `wall_time = time.time()` (UTC
Unix seconds) for every flash. The EDF header contains the recording start
time in UTC. The script computes `start_time + sample_index / fs` for every
EEG sample and uses `searchsorted` to match each marker to the nearest
sample. A median clock offset < 50 ms is expected and acceptable for P300.

```
python decode_emotivpro.py recordings/session_001/ path/to/recording.edf

# if EmotivPRO exports extra non-EEG channels, name the ones you want:
python decode_emotivpro.py recordings/session_001/ recording.edf \
    --eeg-channels AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
```

Outputs saved into the session folder:

| File | Content |
|------|---------|
| `out_erp_emotivpro.png` | Grand-average Target vs Non-Target ERP |
| `out_erp_per_channel_emotivpro.png` | Same, one subplot per channel |
| Terminal | Letter accuracy at each repetition count (1 → max reps) |

### 8. `train_eegnet_multi.py`
Same idea as `train_classical_multi.py` but with an EEGNet convolutional
model trained in PyTorch, plus a real-time latency benchmark (causal filter
+ inference time vs. the flash SOA budget) to check the model is fast
enough to run online.

```
python train_eegnet_multi.py                                   # all recordings/session_*
python train_eegnet_multi.py recordings/session_001 recordings/session_002 ...
```

If every session used a `SYNTHETIC_BOARD`, both training scripts print a
note that near-chance accuracy is expected — there's no real P300 in
synthetic data regardless of how much of it you pool.

### 9. `bnci_benchmark/train_classical_bnci.py` / `train_classical_bnci_loso.py`
xDAWN + Riemannian decoder benchmarked on **BNCI2014_009**, a public
research-grade P300 dataset (16 channels, 10 subjects) loaded through
MOABB and auto-downloaded on first run. The plain version trains on
subjects 1–8 and holds out 9–10; the `_loso` version runs full
Leave-One-Subject-Out cross-validation (train on 9, test on 1, repeated
for every subject) and reports per-fold + aggregate metrics.

```
pip install moabb mne numpy scikit-learn pyriemann
cd bnci_benchmark
python train_classical_bnci.py
python train_classical_bnci.py --test-subjects 9 10
python train_classical_bnci_loso.py
python train_classical_bnci_loso.py --subjects 1 2 3 4 5
```

### 10. `bnci_benchmark/train_eegnet_bnci.py` / `train_eegnet_bnci_loso.py`
Same benchmark as above, EEGNet instead of the classical pipeline. Evaluates
single-flash (target vs. non-target) classification — MOABB's BNCI2014_009
epochs don't expose the row/column flash code needed to reconstruct 6x6
character accuracy, so no letter-accuracy number is reported here (unlike
`train_eegnet_multi.py`, which does have that code from our own recordings).
Saves trained weights (`_loso` saves one checkpoint per fold, under
`bnci_benchmark/loso_models/` by default) so a later run doesn't need to
retrain from scratch — these `.pt` files are regenerated by the scripts and
are not checked in.

```
pip install moabb mne torch numpy scikit-learn
cd bnci_benchmark
python train_eegnet_bnci.py
python train_eegnet_bnci_loso.py --epochs 60 --batch 32 --lr 1e-3
```

### 11. `bnci_benchmark/plot_decoder_comparison.py`
Builds the Classical-vs-EEGNet comparison figure from the results of
running the training scripts above against our 8 archived Muse2 sessions
(hardcoded numbers at the top of the file — this script only plots, it
does not retrain). Saves `muse2CleanData/out_decoder_comparison.png`
relative to wherever it's run from, so the archived recordings referenced
in `results_comparison.md` need to be present locally under that path for
the output directory to exist.

```
cd bnci_benchmark
python plot_decoder_comparison.py
```

### 12. `bnci_benchmark/results_comparison.md`
Written summary comparing EEGNet vs. classical xDAWN+Riemannian, on both
BNCI2014_009 and our Muse2 recordings, under both the normal train/test
split and LOSO. Read this first for the headline numbers before diving into
the individual scripts.

## Recording format

Each `recordings/session_NNN/` folder has:

- `eeg.csv` — `lsl_timestamp` + one column per EEG channel.
  *(not present for the EmotivPRO manual workflow — EEG is in the exported EDF)*
- `markers.csv` — one row per flash stimulus.
  - 16ch / Muse2 / Cortex API: `lsl_timestamp, code, is_target, char_idx, rep`
  - EmotivPRO manual: `lsl_timestamp, wall_time, code, is_target, char_idx, rep`
    (`wall_time` = UTC Unix seconds, used to align with the EDF export)
  - `code` 1-6 = column flash, 7-12 = row flash.
- `meta.json` — grid layout, sample rate, channel names, board, timing, and
  the target phrase.

`recordings/` is gitignored and populated by running the collection
scripts above. The archived raw sessions this pipeline was validated
against (dry/wet 16ch electrodes, Muse2, Emotiv EPOC Flex) are kept
locally alongside this folder but are **not committed** here — they're
tens of MB of CSV/EDF data; regenerate them by re-running the collection
scripts, or ask in the project chat for a copy.

## Dependencies

`requirements.txt` pins the core + per-workflow packages actually verified
against this project (`pip install -r requirements.txt`); `torch` and
`moabb` are left unpinned there since they're only needed for
`train_eegnet_multi.py` / `bnci_benchmark/` and weren't installed in the
env those pins were taken from. Per-workflow breakdown:

```
# all workflows
numpy pandas scipy scikit-learn matplotlib mne pyriemann torch

# 16ch serial headset
pyserial pylsl

# Muse2 (muselsl owns the BLE connection and publishes the LSL stream this
# project's script consumes; brainflow is muselsl's own dependency)
pylsl muselsl brainflow

# Emotiv EPOC Flex — EmotivPRO workflow
pylsl

# Emotiv EPOC Flex — Cortex API workflow
pylsl websocket-client python-dotenv

# BNCI2014_009 decoder benchmark
moabb
```

Muse2 acquisition requires the Muse to be paired/visible over BLE (see
`MUSE_NAME` in `p300_speller_experiment_muse2.py`). The custom headset
requires the correct COM port/baud in `eeg_serial_collect_16ch.py` /
`--serial-port` / `--baud`. The Cortex API workflow requires the Emotiv
Cortex app to be running and a valid EEG license on the developer account.
