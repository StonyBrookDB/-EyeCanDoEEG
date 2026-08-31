"""
Pool multiple recorded P300 sessions together for more statistical power
than any single session gives you (150 target trials/session is often not
enough to tell a real P300 apart from noise -- see the permutation test at
the bottom of this file).

Channels flagged as noisy (find_bad_channels, same heuristic as
visualize_p300.py) in ANY of the pooled sessions are excluded from ALL of
them, since the "one channel goes haywire" issue has moved to a different
channel nearly every session on this headset.

Usage:
    python pool_sessions.py recordings/session_001 recordings/session_002 ...

Output (written next to the recordings/ folder of the first session given):
    out_pooled_erp.png     Target vs Non-Target averaged ERP, pooled trials
    (+ a permutation-test verdict printed to the console)
"""
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np

from visualize_p300 import (BAND, P300_WINDOW, TMAX, TMIN, find_bad_channels,
                             load_session)


def load_and_epoch(session_dir):
    meta, eeg, mrk = load_session(session_dir)
    fs = meta["fs"]
    ch_names = meta["eeg_names"]
    ts = eeg["lsl_timestamp"].to_numpy()
    data = eeg.drop(columns="lsl_timestamp").to_numpy().T

    info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(*BAND, method="iir", verbose=False)
    bad_channels, _ = find_bad_channels(raw)

    sample_idx = np.searchsorted(ts, mrk["lsl_timestamp"].to_numpy())
    valid = sample_idx < len(ts)
    events = np.column_stack([
        sample_idx[valid], np.zeros(int(valid.sum()), dtype=int),
        mrk["is_target"].to_numpy()[valid] + 1,
    ]).astype(int)
    epochs = mne.Epochs(raw, events=events, event_id={"Non-Target": 1, "Target": 2},
                         tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                         verbose=False, event_repeated="drop")
    print(f"  {session_dir}: {len(epochs)} epochs "
          f"({len(epochs['Target'])} target / {len(epochs['Non-Target'])} non-target), "
          f"bad channels: {bad_channels}")
    return epochs, bad_channels, ch_names


def resample_to_grid(epochs, t_common):
    """Interpolate this session's epochs (its own native, possibly-different
    sample rate) onto a shared time grid (seconds) so sessions recorded at
    different effective rates -- this headset has no fixed clock -- can be
    pooled. Returns (n_epochs, n_ch, len(t_common))."""
    data = epochs.get_data()   # (n_epochs, n_ch, n_times_native)
    t_native = epochs.times
    out = np.empty((data.shape[0], data.shape[1], len(t_common)))
    for i in range(data.shape[0]):
        for c in range(data.shape[1]):
            out[i, c] = np.interp(t_common, t_native, data[i, c])
    return out


def permutation_test(data, is_target, window_mask, n_perm=1000, seed=0):
    """Standard label-shuffle permutation test: does the REAL Target vs
    Non-Target split produce a bigger P300-window peak than shuffling which
    epochs count as "target" thousands of times would produce by chance?"""
    real_diff = data[is_target == 1].mean(0).mean(0) - data[is_target == 0].mean(0).mean(0)
    real_peak = real_diff[window_mask].max()

    rng = np.random.default_rng(seed)
    null_peaks = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(is_target)
        d = data[perm == 1].mean(0).mean(0) - data[perm == 0].mean(0).mean(0)
        null_peaks[i] = d[window_mask].max()

    p_value = (null_peaks >= real_peak).mean()
    return real_diff, real_peak, null_peaks, p_value


def main(session_dirs):
    all_epochs, all_bad, all_is_target, ch_names_ref = [], set(), [], None
    t_common = np.linspace(TMIN, TMAX, 200)  # shared grid (seconds); rates differ per session
    for sd in session_dirs:
        epochs, bad, ch_names = load_and_epoch(sd)
        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif ch_names != ch_names_ref:
            raise SystemExit(f"{sd}: channel names differ from other sessions -- "
                              f"can't pool different headsets/boards together")
        all_bad |= set(bad)
        all_is_target.append((epochs.events[:, 2] == 2).astype(int))
        all_epochs.append(resample_to_grid(epochs, t_common))

    print(f"\n[pool] channels excluded (flagged in ANY session): {sorted(all_bad)}")
    good_idx = [i for i, c in enumerate(ch_names_ref) if c not in all_bad]

    data = np.concatenate(all_epochs, axis=0)[:, good_idx, :]
    is_target = np.concatenate(all_is_target)
    print(f"[pool] combined: {len(data)} epochs total "
          f"({is_target.sum()} target / {(is_target == 0).sum()} non-target) "
          f"across {len(session_dirs)} sessions (resampled onto a common "
          f"{len(t_common)}-point grid, {TMIN * 1000:.0f}-{TMAX * 1000:.0f} ms)")

    t = t_common * 1000.0
    window_mask = (t >= P300_WINDOW[0]) & (t <= P300_WINDOW[1])

    real_diff, real_peak, null_peaks, p_value = permutation_test(data, is_target, window_mask)
    tgt_m = data[is_target == 1].mean(0).mean(0)
    ntgt_m = data[is_target == 0].mean(0).mean(0)

    print(f"\n[result] REAL Target-NonTarget peak in P300 window: {real_peak:.1f}")
    print(f"[result] null distribution (1000 label-shuffles): "
          f"mean={null_peaks.mean():.1f}  std={null_peaks.std():.1f}")
    print(f"[result] p-value = {p_value:.4f}  "
          f"({'SIGNIFICANT at p<0.05' if p_value < 0.05 else 'not significant'})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(t, ntgt_m, label=f"Non-Target (n={(is_target == 0).sum()})", color="tab:blue")
    ax.plot(t, tgt_m, label=f"Target (n={is_target.sum()})", color="tab:red")
    ax.plot(t, real_diff, label="Target - Non-Target", color="k", lw=1.5)
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.12, label="P300 window")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (raw ADC counts)")
    ax.set_title(f"Pooled ERP across {len(session_dirs)} sessions "
                 f"(mean of {len(good_idx)}/{len(ch_names_ref)} channels)")
    ax.legend()

    ax = axes[1]
    ax.hist(null_peaks, bins=40, color="gray", alpha=0.7, label="null (label-shuffled)")
    ax.axvline(real_peak, color="red", lw=2, label=f"REAL peak ({real_peak:.0f})")
    ax.set_xlabel("P300-window peak amplitude")
    ax.set_ylabel("count (out of 1000 shuffles)")
    ax.set_title(f"Permutation test  p={p_value:.4f}")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(os.path.dirname(session_dirs[0].rstrip("/\\")), "out_pooled_erp.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python pool_sessions.py <session_dir> [<session_dir> ...]")
    main(sys.argv[1:])
