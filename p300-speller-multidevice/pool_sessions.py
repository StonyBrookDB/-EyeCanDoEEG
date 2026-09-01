"""
Pool multiple recorded P300 sessions together for more statistical power
than any single session gives you (150 target trials/session is often not
enough to tell a real P300 apart from noise -- see the permutation tests at
the bottom of this file).

Channels flagged as noisy (find_bad_channels, same heuristic as
visualize_p300.py) in ANY of the pooled sessions are excluded from ALL of
them, since the "one channel goes haywire" issue has moved to a different
channel nearly every session on this headset.

Two permutation tests are run on the pooled data:
  1. Peak-amplitude test  -- channel-averaged Target-NonTarget peak in the
     P300 window vs. 1000 label-shuffles of the same statistic. Simple, but
     throws away all but 1 timepoint and averages across channels uniformly
     (including channels that carry little discriminative signal).
  2. Decoder-AUC test     -- same xDAWN+Riemannian-tangent-space+logistic
     pipeline used in decode_emotivpro.py's per-session decode, scored by
     cross-validated ROC-AUC, vs. label-shuffles of the same cross-validated
     AUC. Uses all channels' covariance structure instead of a plain average,
     so it is typically far more sensitive to a real P300 than the raw peak.

Usage:
    python pool_sessions.py recordings/session_001 recordings/session_002 ...
    python pool_sessions.py recordings/session_001 recordings/session_002 \\
        --n-perm 1000 --auc-perm 300

Output (written next to the recordings/ folder of the first session given):
    out_pooled_erp.png     Target vs Non-Target averaged ERP + both permutation
                            test histograms
    (+ both permutation-test verdicts printed to the console)
"""
import argparse
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

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


def _cv_auc(X, y, n_splits=5, seed=0):
    """Cross-validated ROC-AUC of the xDAWN + tangent-space + logistic-
    regression pipeline (same architecture as decode_emotivpro.py's decode()),
    scored out-of-fold so the number reflects genuine generalization, not an
    in-sample fit."""
    n_filters = min(4, X.shape[1])
    clf = make_pipeline(
        XdawnCovariances(nfilter=n_filters, estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


def auc_permutation_test(data, is_target, n_perm=200, seed=0, n_splits=5):
    """Label-shuffle permutation test on cross-validated decoder AUC instead
    of raw peak amplitude. Uses every channel's covariance structure (via
    xDAWN) rather than a plain channel average, so it is typically much more
    sensitive to a real but spatially-distributed P300 than the peak test."""
    real_auc = _cv_auc(data, is_target, n_splits=n_splits, seed=seed)

    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(is_target)
        null_aucs[i] = _cv_auc(data, perm, n_splits=n_splits, seed=seed)
        if (i + 1) % max(1, n_perm // 10) == 0:
            print(f"  [auc-perm] {i + 1}/{n_perm} shuffles done", flush=True)

    p_value = (null_aucs >= real_auc).mean()
    return real_auc, null_aucs, p_value


def main(session_dirs, n_perm=1000, auc_perm=200, seed=0):
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

    # Baseline-correct each epoch to its own pre-stimulus mean (TMIN-0ms) --
    # load_and_epoch() builds epochs with baseline=None, so without this the
    # band-passed-but-uncorrected signal can carry a per-epoch DC offset that
    # both the peak test and the xDAWN covariance test are sensitive to.
    baseline_mask = t <= 0
    data = data - data[..., baseline_mask].mean(axis=-1, keepdims=True)

    real_diff, real_peak, null_peaks, p_value = permutation_test(
        data, is_target, window_mask, n_perm=n_perm, seed=seed)
    tgt_m = data[is_target == 1].mean(0).mean(0)
    ntgt_m = data[is_target == 0].mean(0).mean(0)

    print(f"\n[result] REAL Target-NonTarget peak in P300 window: {real_peak:.1f}")
    print(f"[result] null distribution ({n_perm} label-shuffles): "
          f"mean={null_peaks.mean():.1f}  std={null_peaks.std():.1f}")
    print(f"[result] peak-test p-value = {p_value:.4f}  "
          f"({'SIGNIFICANT at p<0.05' if p_value < 0.05 else 'not significant'})")

    print(f"\n[auc-perm] running xDAWN+tangent-space decoder AUC permutation "
          f"test ({auc_perm} shuffles, this is slower than the peak test)...")
    real_auc, null_aucs, auc_p_value = auc_permutation_test(
        data, is_target, n_perm=auc_perm, seed=seed)
    print(f"[result] REAL cross-validated decoder AUC: {real_auc:.3f}")
    print(f"[result] null AUC distribution ({auc_perm} label-shuffles): "
          f"mean={null_aucs.mean():.3f}  std={null_aucs.std():.3f}")
    print(f"[result] AUC-test p-value = {auc_p_value:.4f}  "
          f"({'SIGNIFICANT at p<0.05' if auc_p_value < 0.05 else 'not significant'})")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
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
    ax.set_ylabel(f"count (out of {n_perm} shuffles)")
    ax.set_title(f"Peak permutation test  p={p_value:.4f}")
    ax.legend()

    ax = axes[2]
    ax.hist(null_aucs, bins=30, color="gray", alpha=0.7, label="null (label-shuffled)")
    ax.axvline(real_auc, color="red", lw=2, label=f"REAL AUC ({real_auc:.3f})")
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax.set_xlabel("Cross-validated decoder ROC-AUC")
    ax.set_ylabel(f"count (out of {auc_perm} shuffles)")
    ax.set_title(f"xDAWN decoder AUC permutation test  p={auc_p_value:.4f}")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(os.path.dirname(session_dirs[0].rstrip("/\\")), "out_pooled_erp.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pool P300 sessions and test for a significant P300 "
                     "(peak-amplitude test + xDAWN decoder AUC test)."
    )
    ap.add_argument("session_dirs", nargs="+", help="recordings/session_NNN/ folders to pool")
    ap.add_argument("--n-perm", type=int, default=1000,
                     help="label-shuffles for the peak-amplitude test (default: 1000)")
    ap.add_argument("--auc-perm", type=int, default=200,
                     help="label-shuffles for the AUC test -- slower, each shuffle "
                          "refits a 5-fold cross-validated xDAWN decoder (default: 200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.session_dirs, n_perm=args.n_perm, auc_perm=args.auc_perm, seed=args.seed)
