"""
Pool multiple recorded P300 sessions together for more statistical power
than any single session gives you (150 target trials/session is often not
enough to tell a real P300 apart from noise -- see the permutation tests at
the bottom of this file).

Channels flagged as noisy (find_bad_channels, same heuristic as
visualize_p300.py) in at least --bad-channel-frac (default 25%) of the
pooled sessions are excluded from ALL of them. This is a threshold, not an
"any session" rule: a channel that's only ever flagged bad in one or two
sessions out of many (a one-off contact issue) is kept, while a channel
that's chronically bad across a large fraction of sessions (e.g. Fp1/Fp2
picking up eye-blink artifacts on this headset) is dropped globally so the
pooled test isn't contaminated by it.

Two permutation tests are run on the pooled data:
  1. Peak-amplitude test  -- channel-averaged Target-NonTarget peak in the
     P300 window vs. 1000 label-shuffles of the same statistic. Simple, but
     throws away all but 1 timepoint and averages across channels uniformly
     (including channels that carry little discriminative signal).
  2. Rep-accumulated decoder-AUC test -- the same xDAWN+Riemannian-tangent-
     space+logistic pipeline used in decode_emotivpro.py's per-session
     decode, but scored the way an operational speller actually decides a
     character: each flash's out-of-fold decision score is summed, per
     candidate row/column code, across that character's repetitions
     1..r, and ROC-AUC is computed on whether the true target code scores
     above the other 11 candidates. This is reported as a curve over
     r=1..15 (does accumulating repetitions actually help?) rather than a
     single per-flash number, since the paradigm's whole reason for using
     15 repetitions per character is to accumulate evidence, not to decode
     any one flash in isolation. Significance is a permutation test at
     r=15: the real per-flash scores are kept fixed (already fit once,
     out-of-fold, on the true labels) and only the target-code identity
     used to grade them is re-randomized per character, so no refit is
     needed per shuffle and thousands of shuffles are cheap.

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
from collections import Counter

import matplotlib.pyplot as plt
import mne
import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
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
    mrk = mrk[valid].reset_index(drop=True)
    events = np.column_stack([
        sample_idx[valid], np.zeros(int(valid.sum()), dtype=int),
        mrk["is_target"].to_numpy().astype(int) + 1,
    ]).astype(int)
    epochs = mne.Epochs(raw, events=events, event_id={"Non-Target": 1, "Target": 2},
                         tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                         verbose=False, event_repeated="drop")
    # mrk_al: markers.csv rows for exactly the flashes that survived epoching
    # (boundary clipping + event_repeated="drop"), in the same order as
    # epochs -- needed to recover each epoch's code/rep/character for the
    # rep-accumulated AUC test (is_target alone, kept below for the peak
    # test, doesn't carry that structure).
    mrk_al = mrk.iloc[epochs.selection].reset_index(drop=True)
    print(f"  {session_dir}: {len(epochs)} epochs "
          f"({len(epochs['Target'])} target / {len(epochs['Non-Target'])} non-target), "
          f"bad channels: {bad_channels}")
    return epochs, bad_channels, ch_names, mrk_al


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


def _oof_flash_scores(X, y, groups, n_splits=5, seed=0):
    """Out-of-fold decision_function of the xDAWN + tangent-space +
    logistic-regression pipeline (same architecture as decode_emotivpro.py's
    decode()) for every flash, via group-stratified cross-validation --
    every flash from the same character (`groups`) falls in the same fold,
    so no character's evidence-accumulation score (built downstream from
    these per-flash scores) is ever partly informed by a model that has
    already seen some of that character's own flashes. This gives one
    honest, held-out continuous score per flash, fit once regardless of how
    many permutation shuffles the caller runs afterward."""
    n_filters = min(4, X.shape[1])
    clf = make_pipeline(
        XdawnCovariances(nfilter=n_filters, estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000),
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, groups=groups, method="decision_function")


def rep_accumulated_auc(scores, codes, reps, char_ids, target_row, target_col, max_rep):
    """AUC(r) for r=1..max_rep: sum each character's per-flash scores by
    candidate row/column code across repetitions 1..r, then score whether
    the true target row/column ranks above the other 11 candidates, pooled
    across all characters. `target_row`/`target_col` map each char id to its
    true target codes (7-12 / 1-6) -- kept separate from `scores` so a
    permutation test can re-grade the same fixed scores against a shuffled
    target-code identity without recomputing them."""
    uniq_chars = np.unique(char_ids)
    n_chars = len(uniq_chars)
    char_pos = {c: i for i, c in enumerate(uniq_chars)}
    pos_of_epoch = np.fromiter((char_pos[c] for c in char_ids), dtype=int, count=len(char_ids))

    label = np.zeros((n_chars, 13), dtype=int)
    for c in uniq_chars:
        i = char_pos[c]
        label[i, target_row[c]] = 1
        label[i, target_col[c]] = 1

    running = np.zeros((n_chars, 13))
    auc_curve = np.empty(max_rep)
    for r in range(1, max_rep + 1):
        sel = reps == r
        np.add.at(running, (pos_of_epoch[sel], codes[sel]), scores[sel])
        auc_curve[r - 1] = roc_auc_score(label[:, 1:13].ravel(), running[:, 1:13].ravel())
    return auc_curve


def _code_level_auc_at_max_rep(scores, codes, char_ids, target_row, target_col, max_rep_mask):
    """Same accumulation as rep_accumulated_auc but only the final (all
    repetitions summed) AUC -- used inside the permutation loop, which only
    needs the r=max_rep endpoint, so it skips building the intermediate
    per-r curve (~15x less work per shuffle)."""
    uniq_chars = np.unique(char_ids)
    n_chars = len(uniq_chars)
    char_pos = {c: i for i, c in enumerate(uniq_chars)}
    pos_of_epoch = np.fromiter((char_pos[c] for c in char_ids), dtype=int, count=len(char_ids))

    label = np.zeros((n_chars, 13), dtype=int)
    for c in uniq_chars:
        i = char_pos[c]
        label[i, target_row[c]] = 1
        label[i, target_col[c]] = 1

    running = np.zeros((n_chars, 13))
    sel = max_rep_mask
    np.add.at(running, (pos_of_epoch[sel], codes[sel]), scores[sel])
    return roc_auc_score(label[:, 1:13].ravel(), running[:, 1:13].ravel())


def rep_accumulated_auc_test(scores, codes, reps, char_ids, is_target,
                              max_rep=15, n_perm=1000, seed=0):
    """Permutation test for the rep-accumulated decoder-AUC curve above.
    `scores` are real, honest out-of-fold flash scores (from
    _oof_flash_scores), fit once on the true labels -- they are NOT
    reshuffled per permutation. Instead, each shuffle re-randomizes which
    code counts as each character's "target" (drawing a fresh random
    row 7-12 and column 1-6 per character), so the null asks: would these
    same real per-flash scores have pointed to an arbitrary code just as
    well as they point to the true one? Because nothing is refit, this is
    cheap enough to run thousands of shuffles."""
    codes = np.asarray(codes)
    reps = np.asarray(reps)
    char_ids = np.asarray(char_ids)
    is_target = np.asarray(is_target)

    uniq_chars = np.unique(char_ids)
    target_row, target_col = {}, {}
    for c in uniq_chars:
        m = (char_ids == c) & (is_target == 1)
        tgt_codes = sorted(set(codes[m].tolist()))
        row = next(k for k in tgt_codes if k >= 7)
        col = next(k for k in tgt_codes if k <= 6)
        target_row[c], target_col[c] = row, col

    auc_curve = rep_accumulated_auc(scores, codes, reps, char_ids,
                                     target_row, target_col, max_rep)
    real_auc = auc_curve[-1]

    max_rep_mask = reps <= max_rep
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        fake_row = {c: int(rng.integers(7, 13)) for c in uniq_chars}
        fake_col = {c: int(rng.integers(1, 7)) for c in uniq_chars}
        null_aucs[i] = _code_level_auc_at_max_rep(
            scores, codes, char_ids, fake_row, fake_col, max_rep_mask)

    p_value = (null_aucs >= real_auc).mean()
    return auc_curve, null_aucs, p_value


CHAR_ID_OFFSET = 1000  # max chars/session (<=11 in this dataset) well under this


def main(session_dirs, n_perm=1000, auc_perm=1000, seed=0, bad_channel_frac=0.25, max_rep=15):
    all_epochs, bad_counts, all_is_target, all_codes, all_reps, all_char_ids = \
        [], Counter(), [], [], [], []
    ch_names_ref = None
    t_common = np.linspace(TMIN, TMAX, 200)  # shared grid (seconds); rates differ per session
    for session_i, sd in enumerate(session_dirs):
        epochs, bad, ch_names, mrk_al = load_and_epoch(sd)
        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif ch_names != ch_names_ref:
            raise SystemExit(f"{sd}: channel names differ from other sessions -- "
                              f"can't pool different headsets/boards together")
        bad_counts.update(bad)
        all_is_target.append((epochs.events[:, 2] == 2).astype(int))
        all_epochs.append(resample_to_grid(epochs, t_common))
        all_codes.append(mrk_al["code"].to_numpy().astype(int))
        all_reps.append(mrk_al["rep"].to_numpy().astype(int))
        all_char_ids.append(session_i * CHAR_ID_OFFSET + mrk_al["char_idx"].to_numpy().astype(int))

    n_sessions = len(session_dirs)
    threshold = bad_channel_frac * n_sessions
    all_bad = {c for c, cnt in bad_counts.items() if cnt >= threshold}
    ranked = sorted(bad_counts.items(), key=lambda kv: -kv[1])
    print(f"\n[pool] bad-channel counts across {n_sessions} sessions: "
          f"{', '.join(f'{c}={n}' for c, n in ranked)}")
    print(f"[pool] channels excluded (bad in >= {bad_channel_frac:.0%} of sessions, "
          f"i.e. >= {threshold:.1f} of {n_sessions}): {sorted(all_bad)}")
    good_idx = [i for i, c in enumerate(ch_names_ref) if c not in all_bad]

    data = np.concatenate(all_epochs, axis=0)[:, good_idx, :]
    is_target = np.concatenate(all_is_target)
    codes = np.concatenate(all_codes)
    reps = np.concatenate(all_reps)
    char_ids = np.concatenate(all_char_ids)
    print(f"[pool] combined: {len(data)} epochs total "
          f"({is_target.sum()} target / {(is_target == 0).sum()} non-target), "
          f"{len(np.unique(char_ids))} characters across {len(session_dirs)} sessions "
          f"(resampled onto a common {len(t_common)}-point grid, "
          f"{TMIN * 1000:.0f}-{TMAX * 1000:.0f} ms)")

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

    print(f"\n[auc] fitting group-stratified out-of-fold xDAWN+tangent-space decoder "
          f"(one fit, 5 folds, characters never split across folds)...")
    scores = _oof_flash_scores(data, is_target, char_ids, seed=seed)
    print(f"[auc] running rep-accumulated AUC permutation test "
          f"({auc_perm} shuffles of target-code identity, no refitting needed)...")
    auc_curve, null_aucs, auc_p_value = rep_accumulated_auc_test(
        scores, codes, reps, char_ids, is_target, max_rep=max_rep,
        n_perm=auc_perm, seed=seed)
    real_auc = auc_curve[-1]
    print(f"[result] REAL rep-accumulated decoder AUC at {max_rep} reps: {real_auc:.3f}")
    print(f"[result] AUC by rep count: "
          + ", ".join(f"r={r}:{a:.3f}" for r, a in enumerate(auc_curve, start=1)))
    print(f"[result] null AUC distribution ({auc_perm} target-code shuffles): "
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
    ax.plot(range(1, max_rep + 1), auc_curve, color="k", marker="o", ms=3,
             label="REAL rep-accumulated AUC")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax.axhline(np.quantile(null_aucs, 0.95), color="red", ls=":", lw=1.2,
                label=f"null 95th pct. ({auc_perm} shuffles)")
    ax.set_xlabel("Repetitions accumulated")
    ax.set_ylabel("Cross-validated decoder ROC-AUC")
    ax.set_title(f"Rep-accumulated AUC vs. repetition count  "
                 f"(r={max_rep}: p={auc_p_value:.4f})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(session_dirs[0].rstrip("/\\")), "out_pooled_erp.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pool P300 sessions and test for a significant P300 "
                     "(peak-amplitude test + rep-accumulated xDAWN decoder AUC test)."
    )
    ap.add_argument("session_dirs", nargs="+", help="recordings/session_NNN/ folders to pool")
    ap.add_argument("--n-perm", type=int, default=1000,
                     help="label-shuffles for the peak-amplitude test (default: 1000)")
    ap.add_argument("--auc-perm", type=int, default=1000,
                     help="target-code shuffles for the rep-accumulated AUC test -- cheap "
                          "since the decoder is fit only once (default: 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bad-channel-frac", type=float, default=0.25,
                     help="a channel is excluded from the pool only if it's flagged bad "
                          "in at least this fraction of sessions (default: 0.25, i.e. "
                          "a one-off bad session no longer kills a channel for everyone)")
    ap.add_argument("--max-rep", type=int, default=15,
                     help="max repetitions per character to accumulate evidence over "
                          "(default: 15, matching the paradigm's per-character budget)")
    args = ap.parse_args()
    main(args.session_dirs, n_perm=args.n_perm, auc_perm=args.auc_perm, seed=args.seed,
         bad_channel_frac=args.bad_channel_frac, max_rep=args.max_rep)
