"""
Validate whether a recorded P300 speller session shows a real P300 response.

Same method as eeg-p300-tutorial/01_p300_visualize.py (itself the eeg-notebooks
approach: MNE band-pass filter -> epoch around each stimulus -> average
Target vs Non-Target -> look for the ~250-500 ms positive bump on the
Target curve), adapted to read THIS project's own recording format
(eeg.csv + markers.csv + meta.json from p300_speller_experiment_16ch.py or
p300_speller_experiment_muse2.py) instead
of eegnb's downloaded Muse .csv dataset.

Usage:
    python visualize_p300.py                      # latest recordings/session_*
    python visualize_p300.py recordings/session_001

Output (written into the session folder):
    out_psd.png           power spectrum of the filtered signal
    out_erp_p300.png      Target vs Non-Target averaged ERP (all channels)
    out_erp_per_channel.png   same comparison, one small subplot per channel
                              (helps spot which of the 16 electrodes actually
                              carries the effect -- useful since we don't have
                              a montage/known-good-channel list for this headset)

NOTE on units: the customized headset's raw samples are unscaled ADC counts,
not calibrated volts (we don't know its gain/reference), so the y-axis is
"counts", not uV. That's fine for VALIDATING a P300 shape (relative,
within-recording comparison) -- just not for absolute amplitude claims.
"""
import glob
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BAND = (1.0, 30.0)          # same band as the tutorial
TMIN, TMAX = -0.1, 0.8       # -100 ms to 800 ms, same as the tutorial
P300_WINDOW = (250, 500)     # ms


def latest_session(rec_root):
    sessions = sorted(glob.glob(os.path.join(rec_root, "session_*")))
    if not sessions:
        raise SystemExit(f"no recordings found under {rec_root}")
    return sessions[-1]


def load_session(session_dir):
    meta = json.load(open(os.path.join(session_dir, "meta.json")))
    eeg = pd.read_csv(os.path.join(session_dir, "eeg.csv"))
    mrk = pd.read_csv(os.path.join(session_dir, "markers.csv"))
    return meta, eeg, mrk


def build_epochs(meta, eeg, mrk):
    fs = meta["fs"]
    ch_names = meta["eeg_names"]
    ts = eeg["lsl_timestamp"].to_numpy()
    # select EEG data columns positionally (not by name): older recordings'
    # eeg.csv headers are generic ch0..chN regardless of the real channel
    # names in meta.json, so a name-based lookup can miss (e.g. Muse2's
    # TP9/AF7/AF8/TP10). Column order always matches meta["eeg_names"] order.
    data = eeg.drop(columns="lsl_timestamp").to_numpy().T  # (n_ch, n_samples)

    info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(*BAND, method="iir", verbose=False)

    # map each marker's lsl_timestamp to the nearest EEG sample index
    sample_idx = np.searchsorted(ts, mrk["lsl_timestamp"].to_numpy())
    valid = sample_idx < len(ts)
    events = np.column_stack([
        sample_idx[valid],
        np.zeros(int(valid.sum()), dtype=int),
        mrk["is_target"].to_numpy()[valid] + 1,   # 1=Non-Target, 2=Target
    ]).astype(int)
    event_id = {"Non-Target": 1, "Target": 2}

    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                         tmin=TMIN, tmax=TMAX, baseline=None,
                         preload=True, verbose=False, event_repeated="drop")
    return raw, epochs


def find_bad_channels(raw, thresh=3.0):
    """Flag channels whose overall (band-passed) amplitude is a huge outlier
    relative to the rest, via a robust median+MAD test. Catches the
    recurring "one channel goes haywire" artifact that has shown up on a
    different channel nearly every session (contact/connector issue) --
    those massive isolated spikes otherwise dominate the channel mean.
    Also explicitly flags dead/flat channels (std < 1e-10) which can slip
    through the MAD test when there are few channels."""
    data = raw.get_data()  # (n_ch, n_samples), already band-passed
    stds = data.std(axis=1)
    med = np.median(stds)
    mad = np.median(np.abs(stds - med)) or 1e-9
    bad = []
    for i, s in enumerate(stds):
        if s < 1e-10:                        # dead / flat channel (e.g. AUX all zeros)
            bad.append(raw.ch_names[i])
        elif abs(s - med) / mad > thresh:    # amplitude outlier (too noisy or too quiet)
            bad.append(raw.ch_names[i])
    return bad, dict(zip(raw.ch_names, stds))


def drop_noisy_epochs(epochs, good_idx, thresh=5.0):
    """Auto-reject individual epochs with an abnormally large peak-to-peak
    amplitude on the good channels -- typically a blink or movement during
    that one flash. Lets the subject blink naturally between/around flashes
    instead of having to avoid blinking for the whole session; only the
    handful of contaminated trials get dropped, not the whole recording."""
    data = epochs.get_data()[:, good_idx, :]     # (n_epochs, n_good_ch, n_times)
    ptp = data.max(axis=2) - data.min(axis=2)    # (n_epochs, n_good_ch)
    worst = ptp.max(axis=1)                      # worst channel per epoch
    med = np.median(worst)
    mad = np.median(np.abs(worst - med)) or 1e-9
    bad = np.where(np.abs(worst - med) / mad > thresh)[0]
    if len(bad):
        epochs.drop(bad, reason="auto-reject (large ptp, likely blink/movement)")
    return epochs, len(bad)


def plot_psd(raw, session_dir, bad_channels=None):
    raw_plot = raw.copy()
    if bad_channels:
        keep = [c for c in raw_plot.ch_names if c not in bad_channels]
        raw_plot.pick(keep)
    fig = raw_plot.compute_psd(fmin=1, fmax=30, verbose=False).plot(show=False)
    path = os.path.join(session_dir, "out_psd.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[save] {path}")


def plot_averaged_erp(epochs, session_dir, bad_channels):
    ch_names = epochs.ch_names
    good_idx = [i for i, c in enumerate(ch_names) if c not in bad_channels]

    t = epochs.times * 1000.0
    tgt = epochs["Target"].get_data().mean(axis=0)      # (n_ch, n_times)
    ntgt = epochs["Non-Target"].get_data().mean(axis=0)
    tgt_m, ntgt_m = tgt[good_idx].mean(0), ntgt[good_idx].mean(0)  # mean over GOOD channels only
    diff = tgt_m - ntgt_m

    fig, ax = plt.subplots(figsize=[8, 5])
    ax.plot(t, ntgt_m, label=f"Non-Target (n={len(epochs['Non-Target'])})", color="tab:blue")
    ax.plot(t, tgt_m, label=f"Target (n={len(epochs['Target'])})", color="tab:red")
    ax.plot(t, diff, label="Target - Non-Target", color="k", lw=1.5)
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.12, label="P300 window (250-500 ms)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (raw ADC counts, band-passed 1-30 Hz)")
    excl = f", excluding {bad_channels}" if bad_channels else ""
    ax.set_title(f"P300 ERP -- {os.path.basename(session_dir)} "
                 f"(mean of {len(good_idx)}/{len(ch_names)} channels{excl})")
    ax.legend()
    path = os.path.join(session_dir, "out_erp_p300.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[save] {path}")

    win = (t >= P300_WINDOW[0]) & (t <= P300_WINDOW[1])
    print(f"[result] peak (Target - Non-Target) in "
          f"{P300_WINDOW[0]}-{P300_WINDOW[1]} ms window: {diff[win].max():.1f} counts")


def plot_per_channel_erp(epochs, session_dir, bad_channels):
    t = epochs.times * 1000.0
    tgt = epochs["Target"].get_data().mean(axis=0)
    ntgt = epochs["Non-Target"].get_data().mean(axis=0)
    n_ch = tgt.shape[0]
    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= n_ch:
            ax.axis("off")
            continue
        ax.plot(t, ntgt[i], color="tab:blue", lw=1)
        ax.plot(t, tgt[i], color="tab:red", lw=1)
        ax.axvspan(*P300_WINDOW, color="orange", alpha=0.10)
        ax.axvline(0, color="gray", lw=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        ch = epochs.ch_names[i]
        title = f"{ch} [excluded]" if ch in bad_channels else ch
        ax.set_title(title, fontsize=9, color="red" if ch in bad_channels else "black")
    fig.suptitle("Per-channel Target (red) vs Non-Target (blue)")
    fig.tight_layout()
    path = os.path.join(session_dir, "out_erp_per_channel.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[save] {path}")


def main(session_dir):
    meta, eeg, mrk = load_session(session_dir)
    print(f"[load] {len(eeg)} EEG samples ({meta['n_eeg']} ch) @ "
          f"{meta['fs']:.1f} Hz, {len(mrk)} markers  <- {session_dir}  "
          f"(phrase={meta['phrase']!r})")

    raw, epochs = build_epochs(meta, eeg, mrk)
    print(epochs)

    bad_channels, stds = find_bad_channels(raw)
    if bad_channels:
        print(f"[bad-channels] excluding {bad_channels} from the channel-mean "
              f"(std outliers -- per-channel std: "
              f"{ {k: round(v, 1) for k, v in stds.items()} })")
    else:
        print("[bad-channels] none flagged")

    good_idx = [i for i, c in enumerate(epochs.ch_names) if c not in bad_channels]
    epochs, n_dropped = drop_noisy_epochs(epochs, good_idx)
    if n_dropped:
        print(f"[auto-reject] dropped {n_dropped} epochs with abnormally large "
              f"amplitude (likely blinks/movement): {epochs}")

    plot_psd(raw, session_dir, bad_channels)
    plot_averaged_erp(epochs, session_dir, bad_channels)
    plot_per_channel_erp(epochs, session_dir, bad_channels)
    plt.close("all")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rec_root = os.path.join(here, "recordings")
    session_dir = sys.argv[1] if len(sys.argv) > 1 else latest_session(rec_root)
    main(session_dir)
