"""
decode_emotivpro.py

Aligns an EmotivPRO EDF export with the markers.csv from
p300_speller_experiment_emotiv.py, then runs xDAWN + Riemannian decoding
and saves ERP visualisations into the session folder.

How alignment works
-------------------
  markers.csv   wall_time column = time.time() at the moment of each flash
                (UTC Unix seconds, same clock as the OS)
  EDF file      meas_date header = recording start in UTC
                sample N is at:  meas_date + N / fs  seconds
  -> searchsorted(eeg_utc_timestamps, marker_wall_time) gives the EEG sample
     index closest to each flash stimulus onset.

Usage
-----
    python decode_emotivpro.py recordings/session_001/ path/to/recording.edf

    # specify which EDF channels to use as EEG (comma-separated, no spaces):
    python decode_emotivpro.py recordings/session_001/ recording.edf \\
        --eeg-channels AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4

Dependencies
------------
    pip install mne pyriemann scikit-learn matplotlib
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" if you want pop-up windows
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAND        = (1.0, 30.0)       # bandpass Hz
TMIN        = -0.1              # epoch start (s)
TMAX        =  0.8              # epoch end (s)
P300_WINDOW = (250, 500)        # ms — expected P300 peak window
GRID = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789_"]


def code_pair_to_char(row_code, col_code):
    return GRID[row_code - 7][col_code - 1]


# ---------------------------------------------------------------------------
# 0b. Auto-detect UTC offset from EmotivPRO filename + marker wall_times
# ---------------------------------------------------------------------------
def _detect_utc_offset(edf_path, raw_meas_ts, raw_duration_s, wall_times,
                       manual_override=None):
    """
    EmotivPRO embeds the local timezone offset in the filename:
        ...T15.29.46.04.00.edf  →  04 h 00 min offset (absolute value)

    Try +offset and -offset; return whichever makes wall_times land inside
    the EDF recording window.  manual_override takes precedence if supplied.

    Returns offset_hours (float) to ADD to MNE's meas_date.timestamp().
    """
    import re

    if manual_override is not None:
        return manual_override

    edf_start = raw_meas_ts
    edf_end   = raw_meas_ts + raw_duration_s

    # Already aligned — no correction needed
    if edf_start <= wall_times.min() and wall_times.max() <= edf_end:
        return 0.0

    # Parse offset magnitude from filename  (last two numeric groups before .edf)
    m = re.search(r'\.(\d{2})\.(\d{2})(?:\.md)?\.edf$',
                  os.path.basename(edf_path), re.IGNORECASE)
    if not m:
        return None

    offset_h = int(m.group(1)) + int(m.group(2)) / 60.0
    if offset_h == 0:
        return None

    for sign in (+1, -1):
        cs = edf_start + sign * offset_h * 3600
        ce = cs + raw_duration_s
        if cs <= wall_times.min() and wall_times.max() <= ce:
            direction = "ahead of UTC" if sign > 0 else "behind UTC"
            print(f"[align] auto-detected UTC offset: {sign*offset_h:+.2f} h "
                  f"(EmotivPRO stored local time {direction} in EDF header)")
            return sign * offset_h

    return None   # filename has offset but neither sign works


# ---------------------------------------------------------------------------
# 1.  Load EDF and compute absolute UTC timestamps for every sample
# ---------------------------------------------------------------------------
def load_edf(edf_path, eeg_channels=None, utc_offset_hours=0):
    """
    Returns (raw, eeg_utc_timestamps).

    raw                 MNE RawEDF, filtered to requested channels only
    eeg_utc_timestamps  np.ndarray, shape (n_samples,), UTC Unix seconds
                        for every sample in raw
    """
    print(f"[load] reading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs  = raw.info["sfreq"]
    print(f"[load] EDF channels ({len(raw.ch_names)}): {raw.ch_names}")
    print(f"[load] sample rate : {fs} Hz")
    print(f"[load] duration    : {raw.times[-1]:.1f} s  "
          f"({len(raw.times)} samples)")

    # Pick EEG channels
    if eeg_channels:
        missing = [c for c in eeg_channels if c not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Requested channels not found in EDF: {missing}\n"
                f"Available: {raw.ch_names}"
            )
        raw.pick_channels(eeg_channels)
        print(f"[load] kept {len(eeg_channels)} requested EEG channels")
    else:
        # Auto-pick: keep only channels that look like electrode names.
        # EmotivPRO EDFs include many non-EEG columns; drop everything that
        # starts with a known bookkeeping prefix.
        _drop_prefixes = ("TIME_STAMP", "OR_TIME_STAMP", "COUNTER",
                          "INTERPOLATED", "RAW_CQ", "CQ_", "EQ_",
                          "MARKER", "BATTERY", "GYRO", "STATUS", "TRIGGER",
                          "FW")
        keep = [c for c in raw.ch_names
                if not any(c.upper().startswith(p.upper())
                           for p in _drop_prefixes)]
        if not keep:
            keep = raw.ch_names          # nothing to drop — keep all
        raw.pick_channels(keep)
        print(f"[load] auto-selected {len(keep)} EEG channels: {keep}")

    # Absolute UTC timestamps (no correction here — caller applies offset)
    meas_date = raw.info["meas_date"]
    if meas_date is None:
        raise RuntimeError(
            "EDF file has no recording start time in its header.\n"
            "Re-export from EmotivPRO and make sure 'include date/time' is enabled."
        )
    start_utc = meas_date.timestamp() + utc_offset_hours * 3600
    eeg_utc   = start_utc + raw.times
    print(f"[load] EDF start (raw) : {meas_date.timestamp():.3f}  "
          f"({pd.Timestamp(meas_date.timestamp(), unit='s', tz='UTC')})")
    return raw, eeg_utc


# ---------------------------------------------------------------------------
# 2.  Load markers and align to EEG samples
# ---------------------------------------------------------------------------
def load_and_align_markers(session_dir, eeg_utc, raw_n_samples):
    """
    Returns MNE events array and per-row metadata for the marker file.

    Alignment: for each flash we find the EEG sample whose UTC timestamp is
    closest to the marker's wall_time.  The median offset is printed so you
    can spot obvious clock problems (> 500 ms → something is wrong).
    """
    mrk_path = os.path.join(session_dir, "markers.csv")
    mrk = pd.read_csv(mrk_path)
    print(f"\n[align] {len(mrk)} markers loaded from {mrk_path}")

    if "wall_time" not in mrk.columns:
        raise RuntimeError(
            "'wall_time' column not found in markers.csv.\n"
            "This script requires the EmotivPRO workflow output "
            "(p300_speller_experiment_emotiv.py), which saves wall_time "
            "= time.time() for each flash."
        )

    wall_times = mrk["wall_time"].to_numpy()

    # Sanity check: do the wall_times fall within the EDF recording window?
    edf_start, edf_end = eeg_utc[0], eeg_utc[-1]
    in_window = (wall_times >= edf_start) & (wall_times <= edf_end)
    n_out = (~in_window).sum()
    if n_out == len(mrk):
        raise RuntimeError(
            f"None of the {len(mrk)} markers fall within the EDF recording window "
            f"[{edf_start:.1f}, {edf_end:.1f}].\n"
            f"Marker wall_times range: [{wall_times.min():.1f}, {wall_times.max():.1f}].\n"
            "Check that EmotivPRO was already recording before you started the "
            "Python script, and that both files are from the same session."
        )
    if n_out > 0:
        print(f"[align] WARNING: {n_out} markers outside EDF window — they will be dropped")
        mrk = mrk[in_window].reset_index(drop=True)
        wall_times = wall_times[in_window]

    # Find nearest EEG sample for each marker
    sample_idx = np.searchsorted(eeg_utc, wall_times)
    sample_idx = np.clip(sample_idx, 0, raw_n_samples - 1)

    # Report offset
    offsets_ms = (eeg_utc[sample_idx] - wall_times) * 1000
    print(f"[align] clock offset: median={np.median(offsets_ms):.1f} ms  "
          f"max={np.abs(offsets_ms).max():.1f} ms  "
          f"(< 50 ms is good for P300)")

    # Build MNE events array: [sample_idx, 0, event_id]
    # event_id: 2 = Target, 1 = Non-Target  (matches visualize_p300.py)
    event_ids = (mrk["is_target"].to_numpy().astype(int) + 1)   # 0->1, 1->2
    events = np.column_stack([
        sample_idx,
        np.zeros(len(mrk), dtype=int),
        event_ids,
    ]).astype(int)

    return events, mrk


# ---------------------------------------------------------------------------
# 3.  Build MNE Epochs
# ---------------------------------------------------------------------------
def build_epochs(raw, events):
    raw_filt = raw.copy().filter(*BAND, method="iir", verbose=False)
    epochs = mne.Epochs(
        raw_filt, events=events,
        event_id={"Non-Target": 1, "Target": 2},
        tmin=TMIN, tmax=TMAX,
        baseline=(TMIN, 0),       # baseline correct to pre-stimulus interval
        preload=True, verbose=False,
        event_repeated="drop",
    )
    print(f"\n[epoch] {len(epochs)} epochs  "
          f"(Target: {len(epochs['Target'])}  "
          f"Non-Target: {len(epochs['Non-Target'])})")
    return epochs


# ---------------------------------------------------------------------------
# 4.  Bad-channel and noisy-epoch rejection  (same logic as visualize_p300.py)
# ---------------------------------------------------------------------------
def find_bad_channels(raw, thresh=3.0):
    data = raw.get_data()
    stds = data.std(axis=1)
    med  = np.median(stds)
    mad  = np.median(np.abs(stds - med)) or 1e-9
    bad  = [raw.ch_names[i] for i in range(len(stds))
            if abs(stds[i] - med) / mad > thresh]
    return bad


def drop_noisy_epochs(epochs, good_idx, thresh=5.0):
    data = epochs.get_data()[:, good_idx, :]
    ptp  = data.max(axis=2) - data.min(axis=2)
    worst = ptp.max(axis=1)
    med  = np.median(worst)
    mad  = np.median(np.abs(worst - med)) or 1e-9
    bad  = np.where(np.abs(worst - med) / mad > thresh)[0]
    if len(bad):
        epochs.drop(bad, reason="auto-reject")
    return epochs, len(bad)


# ---------------------------------------------------------------------------
# 5.  ERP plots  (saved into session_dir)
# ---------------------------------------------------------------------------
def plot_erp(epochs, session_dir, bad_channels):
    ch_names = epochs.ch_names
    good_idx = [i for i, c in enumerate(ch_names) if c not in bad_channels]
    t = epochs.times * 1000

    tgt  = epochs["Target"].get_data().mean(axis=0)
    ntgt = epochs["Non-Target"].get_data().mean(axis=0)
    tgt_m, ntgt_m = tgt[good_idx].mean(0), ntgt[good_idx].mean(0)
    diff = tgt_m - ntgt_m

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, ntgt_m, label=f"Non-Target (n={len(epochs['Non-Target'])})",
            color="tab:blue")
    ax.plot(t, tgt_m,  label=f"Target (n={len(epochs['Target'])})",
            color="tab:red")
    ax.plot(t, diff, label="Target − Non-Target", color="k", lw=1.5)
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.12,
               label=f"P300 window ({P300_WINDOW[0]}–{P300_WINDOW[1]} ms)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    excl = f", excl. {bad_channels}" if bad_channels else ""
    ax.set_title(f"P300 ERP — EmotivPRO session "
                 f"({len(good_idx)}/{len(ch_names)} ch{excl})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    path = os.path.join(session_dir, "out_erp_emotivpro.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[plot] ERP saved -> {path}")

    win  = (t >= P300_WINDOW[0]) & (t <= P300_WINDOW[1])
    peak = diff[win].max()
    print(f"[result] peak (Target − Non-Target) in "
          f"{P300_WINDOW[0]}–{P300_WINDOW[1]} ms: {peak:.2f} µV"
          + ("  ← P300 detected!" if peak > 0.5 else "  ← low/absent, check signal quality"))
    plt.close(fig)


def plot_per_channel_erp(epochs, session_dir, bad_channels):
    t = epochs.times * 1000
    tgt  = epochs["Target"].get_data().mean(axis=0)
    ntgt = epochs["Non-Target"].get_data().mean(axis=0)
    n_ch = tgt.shape[0]
    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
                             sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= n_ch:
            ax.axis("off")
            continue
        ax.plot(t, ntgt[i], color="tab:blue", lw=1)
        ax.plot(t, tgt[i],  color="tab:red",  lw=1)
        ax.axvspan(*P300_WINDOW, color="orange", alpha=0.10)
        ax.axvline(0, color="gray", lw=0.5)
        ch = epochs.ch_names[i]
        ax.set_title(ch, fontsize=9,
                     color="red" if ch in bad_channels else "black")
    fig.suptitle("Per-channel: Target (red) vs Non-Target (blue)")
    fig.tight_layout()
    path = os.path.join(session_dir, "out_erp_per_channel_emotivpro.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[plot] per-channel ERP saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.  Export aligned EEG + markers as CSV  (same format as Muse2/16ch sessions)
# ---------------------------------------------------------------------------
def export_csvs(raw, eeg_utc, mrk, events, session_dir):
    """
    Write eeg.csv and update markers.csv with lsl_timestamp so this session
    is compatible with visualize_p300.py / train_classical_multi.py.

    eeg.csv columns : lsl_timestamp, <ch1>, <ch2>, ...
    markers.csv gains: lsl_timestamp column (EEG sample time nearest each flash)
    """
    # --- eeg.csv ---
    data = raw.get_data().T          # (n_samples, n_ch)
    eeg_df = pd.DataFrame(data, columns=raw.ch_names)
    eeg_df.insert(0, "lsl_timestamp", eeg_utc)
    eeg_path = os.path.join(session_dir, "eeg.csv")
    eeg_df.to_csv(eeg_path, index=False)
    print(f"[export] eeg.csv  -> {eeg_path}  "
          f"({len(eeg_df)} samples × {len(raw.ch_names)} channels)")

    # --- markers.csv: add lsl_timestamp column (EEG UTC time at each flash) ---
    sample_idx = events[:, 0]                         # already clipped to valid range
    mrk = mrk.copy()
    mrk["lsl_timestamp"] = eeg_utc[sample_idx]
    cols = ["lsl_timestamp"] + [c for c in mrk.columns if c != "lsl_timestamp"]
    mrk = mrk[cols]
    mrk_path = os.path.join(session_dir, "markers.csv")
    mrk.to_csv(mrk_path, index=False)
    print(f"[export] markers.csv updated with lsl_timestamp -> {mrk_path}")

    # --- meta.json: add EEG fields so downstream scripts can read them ---
    meta_path = os.path.join(session_dir, "meta.json")
    meta = json.load(open(meta_path))
    meta["fs"]       = float(raw.info["sfreq"])
    meta["n_eeg"]    = len(raw.ch_names)
    meta["eeg_names"] = raw.ch_names
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] meta.json updated with fs / n_eeg / eeg_names -> {meta_path}")


# ---------------------------------------------------------------------------
# 7.  xDAWN + Riemannian decode — letter accuracy at each rep count
# ---------------------------------------------------------------------------
def decode(epochs, mrk, meta, session_dir):
    phrase = meta["phrase"]
    X      = epochs.get_data()                  # (n_epochs, n_ch, n_times)

    # epochs.selection = indices into the original events/mrk array for
    # every epoch that survived boundary checks + noisy-epoch rejection.
    # Slicing mrk[:len(epochs)] would silently misalign when epochs were
    # dropped from the middle of the sequence.
    sel     = epochs.selection
    mrk_al  = mrk.iloc[sel].reset_index(drop=True)
    labels  = mrk_al["is_target"].to_numpy().astype(int)
    codes   = mrk_al["code"].to_numpy().astype(int)
    gcis    = mrk_al["char_idx"].to_numpy().astype(int)
    reps_   = mrk_al["rep"].to_numpy().astype(int)

    chars     = sorted(set(gcis))
    letter_of = {ci: phrase[ci] for ci in chars}

    if len(chars) < 2:
        print("[decode] need at least 2 characters to train/test; skipping decode")
        return

    rng         = np.random.default_rng(42)
    chars_shuf  = rng.permutation(chars)
    n_train     = max(1, int(round(0.7 * len(chars))))
    train_chars = set(chars_shuf[:n_train].tolist())
    test_chars  = [c for c in chars if c not in train_chars] or chars

    tr = np.isin(gcis, list(train_chars))
    print(f"\n[decode] train: {len(train_chars)} chars  "
          f"test: {len(test_chars)} chars "
          f"({''.join(letter_of[c] for c in test_chars)})")

    n_filters = min(4, X.shape[1])
    clf = make_pipeline(
        XdawnCovariances(nfilter=n_filters, estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(X[tr], labels[tr])

    try:
        auc = roc_auc_score(labels[tr], clf.decision_function(X[tr]))
        print(f"[decode] in-sample single-flash ROC-AUC: {auc:.3f}")
    except ValueError:
        pass

    if len(test_chars) == 1:
        print(f"[decode] NOTE: only 1 test character — accuracy is 0% or 100% by chance. "
              f"Collect more characters (longer phrase) for a meaningful score.")

    max_rep = int(reps_.max())
    print(f"[decode] accuracy at each rep count (max {max_rep} reps):")
    for r in range(1, max_rep + 1):
        correct = 0
        for ci in test_chars:
            sel = (gcis == ci) & (reps_ <= r)
            if not sel.any():
                continue
            sc = clf.decision_function(X[sel])
            cc = codes[sel]
            code_score = np.full(13, -np.inf)
            for k in range(1, 13):
                m = cc == k
                code_score[k] = sc[m].sum() if m.any() else -np.inf
            pred_col = int(np.argmax(code_score[1:7]) + 1)
            pred_row = int(np.argmax(code_score[7:13]) + 7)
            if code_pair_to_char(pred_row, pred_col) == letter_of[ci]:
                correct += 1
        pct = 100 * correct / len(test_chars)
        print(f"  reps={r:2d}  ->  {correct}/{len(test_chars)} chars correct  "
              f"({pct:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Decode P300 session: align EmotivPRO EDF + markers.csv"
    )
    ap.add_argument("session_dir",
                    help="path to recordings/session_NNN/ folder")
    ap.add_argument("edf_file",
                    help="path to the EmotivPRO exported .edf file")
    ap.add_argument("--eeg-channels", default=None,
                    help="comma-separated list of EEG channel names to use "
                         "(default: auto-detect, skip non-EEG channels)")
    ap.add_argument("--edf-utc-offset-hours", type=float, default=None,
                    help="hours to ADD to the EDF start time to reach true UTC. "
                         "Normally auto-detected from the EmotivPRO filename. "
                         "Override only if auto-detection fails "
                         "(e.g. EDT=UTC-4 → pass 4, IST=UTC+5:30 → pass -5.5).")
    args = ap.parse_args()

    session_dir = args.session_dir
    edf_path    = args.edf_file
    eeg_channels = ([c.strip() for c in args.eeg_channels.split(",")]
                    if args.eeg_channels else None)

    if not os.path.isdir(session_dir):
        sys.exit(f"session dir not found: {session_dir}")
    if not os.path.isfile(edf_path):
        sys.exit(f"EDF file not found: {edf_path}")

    meta_path = os.path.join(session_dir, "meta.json")
    if not os.path.isfile(meta_path):
        sys.exit(f"meta.json not found in {session_dir}")
    meta = json.load(open(meta_path))
    print(f"[meta] phrase={meta['phrase']!r}  reps={meta['reps']}  "
          f"board={meta.get('board')}")

    # 1a. Peek at markers to get wall_times for offset detection
    mrk_path = os.path.join(session_dir, "markers.csv")
    if not os.path.isfile(mrk_path):
        sys.exit(f"markers.csv not found in {session_dir}")
    _mrk_peek = pd.read_csv(mrk_path)
    if "wall_time" not in _mrk_peek.columns:
        sys.exit(
            "'wall_time' column missing from markers.csv.\n"
            "This script requires output from p300_speller_experiment_emotiv.py."
        )
    wall_times_peek = _mrk_peek["wall_time"].to_numpy()

    # 1b. Load EDF (no offset applied yet)
    raw, eeg_utc_raw = load_edf(edf_path, eeg_channels, utc_offset_hours=0)
    raw_meas_ts    = eeg_utc_raw[0]
    raw_duration_s = float(raw.times[-1])

    # 1c. Auto-detect timezone offset (or use manual override)
    offset = _detect_utc_offset(
        edf_path, raw_meas_ts, raw_duration_s,
        wall_times_peek,
        manual_override=args.edf_utc_offset_hours,
    )
    if offset is None:
        sys.exit(
            "[error] Cannot auto-align EDF and markers — UTC offset unknown.\n"
            f"  EDF window  : [{raw_meas_ts:.0f}, {raw_meas_ts+raw_duration_s:.0f}]"
            f"  ({pd.Timestamp(raw_meas_ts, unit='s', tz='UTC')} …)\n"
            f"  Marker range: [{wall_times_peek.min():.0f}, {wall_times_peek.max():.0f}]\n"
            f"  Gap         : {wall_times_peek.min() - raw_meas_ts:.0f} s "
            f"({(wall_times_peek.min() - raw_meas_ts)/3600:.2f} h)\n"
            "  Hint: pass --edf-utc-offset-hours N  where N = hours to add to EDF time\n"
            "  to reach UTC (e.g. 4 for EDT/UTC-4, -5.5 for IST/UTC+5:30)."
        )

    # 1d. Apply offset to EEG timestamps
    eeg_utc = eeg_utc_raw + offset * 3600
    if offset:
        corrected_start = pd.Timestamp(raw_meas_ts + offset * 3600, unit='s', tz='UTC')
        print(f"[load] EDF start (corrected {offset:+.2f}h): {corrected_start}")

    # 2. Align markers
    events, mrk = load_and_align_markers(session_dir, eeg_utc, len(raw.times))

    # 3. Epoch
    epochs = build_epochs(raw, events)

    if len(epochs) == 0:
        sys.exit("[error] No epochs remaining after alignment. "
                 "Check that the EDF and session_dir are from the same recording.")

    # 4. Bad channels + noisy epoch rejection
    bad_channels = find_bad_channels(raw)
    if bad_channels:
        print(f"[qc] bad channels flagged (will be excluded from plots): {bad_channels}")
    good_idx = [i for i, c in enumerate(epochs.ch_names) if c not in bad_channels]
    epochs, n_dropped = drop_noisy_epochs(epochs, good_idx)
    if n_dropped:
        print(f"[qc] auto-rejected {n_dropped} noisy epochs")

    # 5. Export aligned CSVs (before any epoch dropping changes the data)
    export_csvs(raw, eeg_utc, mrk, events, session_dir)

    # 6. Plots
    plot_erp(epochs, session_dir, bad_channels)
    plot_per_channel_erp(epochs, session_dir, bad_channels)

    # 7. Decode
    decode(epochs, mrk, meta, session_dir)

    print(f"\n[done] outputs written to {session_dir}")


if __name__ == "__main__":
    main()
