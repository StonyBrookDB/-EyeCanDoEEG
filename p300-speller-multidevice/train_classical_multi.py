"""
Offline decoder for MULTIPLE recorded sessions combined (same idea as
train_eegnet_multi.py, but for the classical xDAWN + Riemannian pipeline
in decode_recording.py).

Why this exists: decode_recording.py splits train/test characters WITHIN
one session, so a short session (e.g. a 5-letter phrase) only leaves 1-2
held-out characters -- too few to trust the reported accuracy. Pooling
epochs across sessions before splitting gives a bigger, less noisy
held-out test set, the same way train_eegnet_multi.py does for EEGNet.

Usage:
    python train_classical_multi.py                          # uses all recordings/session_*
    python train_classical_multi.py recordings/session_001 recordings/session_002 ...
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

GRID = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789_"]
EPOCH_MS = 667
BAND = (0.1, 20.0)
SEED = 1
CHAR_OFFSET = 10_000          # global_char_id = session_index * CHAR_OFFSET + char_idx


def code_pair_to_char(row_code, col_code):
    return GRID[row_code - 7][col_code - 1]


# ---------------------------------------------------------------------------
# 1. Load + epoch ONE session (same recipe as decode_recording.py)
# ---------------------------------------------------------------------------
def load_one_session(session_dir, session_idx, letter_of):
    meta = json.load(open(os.path.join(session_dir, "meta.json")))
    fs = meta["fs"]
    phrase = meta["phrase"]
    epoch_len = int(fs * EPOCH_MS / 1000)
    decim = int(max(1, fs // 50))

    eeg = pd.read_csv(os.path.join(session_dir, "eeg.csv"))
    mrk = pd.read_csv(os.path.join(session_dir, "markers.csv"))
    ts = eeg["lsl_timestamp"].to_numpy()
    X_cont = eeg.drop(columns="lsl_timestamp").to_numpy()

    # Drop flat/dead channels (e.g. Muse2 Right AUX = all zeros) — a singular
    # channel makes the covariance matrix non-positive-definite and crashes xDAWN.
    stds = X_cont.std(axis=0)
    live = stds > 1e-10
    if not live.all():
        dead = [eeg.drop(columns="lsl_timestamp").columns[i]
                for i, ok in enumerate(live) if not ok]
        print(f"  [{os.path.basename(session_dir)}] dropping flat channels: {dead}")
        X_cont = X_cont[:, live]

    b, a = butter(4, list(BAND), btype="band", fs=fs)
    X_cont = filtfilt(b, a, X_cont, axis=0)            # zero-phase -- offline only

    epochs, codes, labels, gchar_idx, reps = [], [], [], [], []
    for _, m in mrk.iterrows():
        on = int(np.searchsorted(ts, m["lsl_timestamp"]))
        if on + epoch_len > X_cont.shape[0]:
            continue
        ep = X_cont[on:on + epoch_len, :][::decim, :].T
        ci = int(m["char_idx"])
        gci = session_idx * CHAR_OFFSET + ci
        letter_of[gci] = phrase[ci]
        epochs.append(ep)
        codes.append(int(m["code"]))
        labels.append(int(m["is_target"]))
        gchar_idx.append(gci)
        reps.append(int(m["rep"]))

    print(f"  [{os.path.basename(session_dir)}] fs={fs}Hz phrase={phrase!r} "
          f"board={meta.get('board')} -> {len(epochs)} flashes")
    return epochs, codes, labels, gchar_idx, reps, fs


def load_multi(session_dirs):
    letter_of = {}                  # global_char_id -> true target letter
    all_ep, all_codes, all_labels, all_gci, all_reps = [], [], [], [], []
    fs_seen = None
    for i, sdir in enumerate(session_dirs):
        ep, codes, labels, gci, reps, fs = load_one_session(sdir, i, letter_of)
        if fs_seen is None:
            fs_seen = fs
        elif fs != fs_seen:
            raise ValueError(f"{sdir}: fs={fs} != {fs_seen} from earlier session "
                              f"-- mixed sampling rates not supported here")
        all_ep += ep; all_codes += codes; all_labels += labels
        all_gci += gci; all_reps += reps

    X = np.array(all_ep)
    print(f"\nCombined: {len(session_dirs)} sessions, {len(set(all_gci))} characters, "
          f"{len(X)} flashes ({sum(all_labels)} target / {len(all_labels) - sum(all_labels)} non-target)")
    return X, np.array(all_codes), np.array(all_labels), np.array(all_gci), np.array(all_reps), letter_of


# ---------------------------------------------------------------------------
# 2. Train xDAWN + Riemannian on pooled train characters, decode the rest
# ---------------------------------------------------------------------------
def main(session_dirs):
    X, codes, labels, gci, reps, letter_of = load_multi(session_dirs)

    # split by (session, character) -- pooled across sessions so held-out
    # test set isn't just 1-2 characters from a single short session
    chars = sorted(set(gci))
    rng0 = np.random.default_rng(SEED)
    chars_shuffled = rng0.permutation(chars)
    n_train = max(1, int(round(0.7 * len(chars))))
    train_chars = set(chars_shuffled[:n_train].tolist())
    test_chars = [c for c in chars if c not in train_chars] or chars
    tr = np.isin(gci, list(train_chars))
    print(f"Train chars: {len(train_chars)}   Test chars: {len(test_chars)}")

    clf = make_pipeline(
        XdawnCovariances(nfilter=min(4, X.shape[1]), estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(X[tr], labels[tr])
    try:
        auc = roc_auc_score(labels[tr], clf.decision_function(X[tr]))
        print(f"In-sample single-flash ROC-AUC (train chars): {auc:.3f}")
    except ValueError:
        print("In-sample AUC undefined (need both classes present)")

    # --- decode each test character at rep budgets r = 1..reps -------------
    max_rep = int(reps.max())
    test_letters = "".join(letter_of[c] for c in test_chars)
    print(f"Decoding {len(test_chars)} held-out characters across all sessions "
          f"(letters: {test_letters}):")
    for r in range(1, max_rep + 1):
        correct = 0
        for c in test_chars:
            sel = (gci == c) & (reps <= r)
            sc = clf.decision_function(X[sel])
            cc = codes[sel]
            code_score = np.full(13, -np.inf)
            for k in range(1, 13):
                m = cc == k
                code_score[k] = sc[m].sum() if m.any() else -np.inf
            pred_col = int(np.argmax(code_score[1:7]) + 1)
            pred_row = int(np.argmax(code_score[7:13]) + 7)
            if code_pair_to_char(pred_row, pred_col) == letter_of[c]:
                correct += 1
        print(f"  reps={r:2d}  ->  char accuracy = {100 * correct / len(test_chars):5.1f}%")

    boards = set()
    for sdir in session_dirs:
        boards.add(json.load(open(os.path.join(sdir, "meta.json"))).get("board"))
    if all(b == "SYNTHETIC_BOARD" for b in boards):
        print("\n[note] ALL sessions are SYNTHETIC_BOARD -> no real P300 exists in any "
              "of them.\n       ~chance accuracy is EXPECTED regardless of how much "
              "data is combined.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        dirs = sorted(glob.glob(os.path.join(here, "recordings", "session_*")) +
                      glob.glob(os.path.join(here, "recordings", "emotiv_*")))
    if not dirs:
        print("no session directories found")
        raise SystemExit(1)
    main(dirs)
