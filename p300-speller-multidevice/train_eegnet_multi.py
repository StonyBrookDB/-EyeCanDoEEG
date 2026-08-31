"""
Train EEGNet on MULTIPLE recorded sessions combined, to see whether more
data (more characters, more flashes) changes accuracy versus the single-
session run in train_eegnet.py.

Same architecture, same causal-filter preprocessing, same row/col decode
logic as train_eegnet.py -- the only difference is that epochs from several
session_NNN/ folders are concatenated before training, and the train/test
split is done by (session, character) instead of just character, so held-out
test characters are still never seen during training regardless of which
session they came from.

IMPORTANT: if all sessions used the SYNTHETIC board, there is still no real
P300 in ANY of them -- more synthetic data does not create a signal that
isn't there. This run answers "does the pipeline behave sensibly with more
data" (e.g. does training loss/behavior stay stable, does per-flash recall
stay near chance as expected), not "is accuracy higher now". Real accuracy
gains only come from real-headset recordings.

Usage:
    python train_eegnet_multi.py                          # uses all recordings/session_*
    python train_eegnet_multi.py recordings/session_001 recordings/session_002 ...
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

GRID = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789_"]
BAND = (0.1, 20.0)
EPOCH_MS = 667
N_EPOCHS = 60
BATCH = 32
LR = 1e-3
SEED = 1
CHAR_OFFSET = 10_000          # global_char_id = session_index * CHAR_OFFSET + char_idx


def code_pair_to_char(row_code, col_code):
    return GRID[row_code - 7][col_code - 1]


# ---------------------------------------------------------------------------
# 1. Load + epoch ONE session (same recipe as train_eegnet.py, causal filter)
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

    # Drop flat/dead channels before filtering and epoching
    stds = X_cont.std(axis=0)
    live = stds > 1e-10
    if not live.all():
        dead = [eeg.drop(columns="lsl_timestamp").columns[i]
                for i, ok in enumerate(live) if not ok]
        print(f"  [{os.path.basename(session_dir)}] dropping flat channels: {dead}")
        X_cont = X_cont[:, live]

    b, a = butter(4, list(BAND), btype="band", fs=fs)
    X_cont = lfilter(b, a, X_cont, axis=0)            # causal -- deployable online

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
    return epochs, codes, labels, gchar_idx, reps, fs, epoch_len, (b, a)


def load_multi(session_dirs):
    letter_of = {}                  # global_char_id -> true target letter
    all_ep, all_codes, all_labels, all_gci, all_reps = [], [], [], [], []
    fs_seen, epoch_len_seen, ba_seen = None, None, None
    for i, sdir in enumerate(session_dirs):
        ep, codes, labels, gci, reps, fs, epoch_len, ba = load_one_session(sdir, i, letter_of)
        if fs_seen is None:
            fs_seen, epoch_len_seen, ba_seen = fs, epoch_len, ba
        elif fs != fs_seen:
            raise ValueError(f"{sdir}: fs={fs} != {fs_seen} from earlier session "
                              f"-- mixed sampling rates not supported here")
        all_ep += ep; all_codes += codes; all_labels += labels
        all_gci += gci; all_reps += reps

    X = np.array(all_ep, dtype=np.float32)
    print(f"\nCombined: {len(session_dirs)} sessions, {len(set(all_gci))} characters, "
          f"{len(X)} flashes ({sum(all_labels)} target / {len(all_labels) - sum(all_labels)} non-target)")
    return (X, np.array(all_codes), np.array(all_labels), np.array(all_gci),
            np.array(all_reps), letter_of, epoch_len_seen, ba_seen)


# ---------------------------------------------------------------------------
# 2. EEGNet (identical to train_eegnet.py)
# ---------------------------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_chan, n_time, F1=8, D=2, F2=16, kern_len=16, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 8), padding=(0, 4), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chan, n_time)
            feat_dim = self.block2(self.block1(dummy)).numel()
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. Real-time latency benchmark (identical to train_eegnet.py)
# ---------------------------------------------------------------------------
def benchmark_latency(model, ba, n_chan, raw_epoch_len, model_n_time, device,
                      soa_ms, n_warmup=20, n_trials=300):
    b, a = ba
    rng = np.random.default_rng(0)

    filt_times = []
    for _ in range(n_trials):
        chunk = rng.normal(size=(raw_epoch_len, n_chan)).astype(np.float64)
        t0 = time.perf_counter()
        lfilter(b, a, chunk, axis=0)
        filt_times.append((time.perf_counter() - t0) * 1000)
    filt_times = np.array(filt_times)

    model.eval()
    x1 = torch.from_numpy(
        rng.normal(size=(1, n_chan, model_n_time)).astype(np.float32)).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x1)
        infer_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x1)
            infer_times.append((time.perf_counter() - t0) * 1000)
    infer_times = np.array(infer_times)

    def stats(arr):
        return (f"mean={arr.mean():.3f}ms  "
                f"median={np.median(arr):.3f}ms  "
                f"p95={np.percentile(arr, 95):.3f}ms")

    total_mean = filt_times.mean() + infer_times.mean()
    print(f"\n=== Real-time latency benchmark (device={device}, n={n_trials} trials) ===")
    print(f"  causal filter (1 window, {n_chan}ch x {raw_epoch_len}smp):  {stats(filt_times)}")
    print(f"  EEGNet inference (batch=1, {n_chan}ch x {model_n_time}smp): {stats(infer_times)}")
    print(f"  ---------------------------------------------------------------")
    print(f"  total compute overhead per flash (mean):  {total_mean:.3f} ms")
    print(f"  SOA budget before next flash:             {soa_ms:.1f} ms")
    print(f"  headroom:                                 {soa_ms - total_mean:.1f} ms "
          f"({100 * total_mean / soa_ms:.1f}% of budget used)")
    print(f"  (separately: must still wait ~{EPOCH_MS} ms after each flash for the\n"
          f"   epoch window to fill -- physiological, same for any classifier)")


# ---------------------------------------------------------------------------
# 4. Train / evaluate across all sessions combined
# ---------------------------------------------------------------------------
def main(session_dirs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Sessions: {[os.path.basename(s) for s in session_dirs]}\n")

    X, codes, labels, gci, reps, letter_of, epoch_len, ba = load_multi(session_dirs)

    # split by (session, character) -- held-out characters unseen regardless of session
    chars = sorted(set(gci))
    rng0 = np.random.default_rng(SEED)
    chars_shuffled = rng0.permutation(chars)
    n_train = max(1, int(round(0.7 * len(chars))))
    train_chars = set(chars_shuffled[:n_train].tolist())
    test_chars = [c for c in chars if c not in train_chars] or chars
    is_train = np.isin(gci, list(train_chars))
    print(f"Train chars: {len(train_chars)}   Test chars: {len(test_chars)}")

    mu = X[is_train].mean(axis=(0, 2), keepdims=True)
    sd = X[is_train].std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - mu) / sd

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(labels.astype(np.float32))

    model = EEGNet(n_chan=X.shape[1], n_time=X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n_pos = labels[is_train].sum()
    n_neg = is_train.sum() - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_idx = np.where(is_train)[0]
    rng = np.random.default_rng(SEED)

    model.train()
    for epoch in range(N_EPOCHS):
        rng.shuffle(tr_idx)
        total = 0.0
        for i in range(0, len(tr_idx), BATCH):
            b_ = tr_idx[i:i + BATCH]
            xb, yb = Xt[b_].to(device), yt[b_].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b_)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{N_EPOCHS}  loss={total / len(tr_idx):.4f}")

    model.eval()
    with torch.no_grad():
        score = model(Xt.to(device)).cpu().numpy()

    try:
        train_auc = roc_auc_score(labels[is_train], score[is_train])
        print(f"In-sample single-flash ROC-AUC (train chars): {train_auc:.3f}")
    except ValueError:
        print("In-sample AUC undefined (need both classes present)")

    te = np.isin(gci, test_chars)
    pred = (score > 0).astype(int)
    for cls, nm in [(1, "target"), (0, "non-target")]:
        m = te & (labels == cls)
        acc = (pred[m] == cls).mean() if m.any() else float("nan")
        print(f"Per-flash recall on {nm:11s}: {acc * 100:5.1f}%  (n={m.sum()})")

    max_rep = int(reps.max())
    test_letters = "".join(letter_of[c] for c in test_chars)
    print(f"\nDecoding {len(test_chars)} held-out characters across all sessions "
          f"({test_letters}):")
    for r in range(1, max_rep + 1):
        correct = 0
        for c in test_chars:
            sel = (gci == c) & (reps <= r)
            cs = np.full(13, -np.inf)
            for k in range(1, 13):
                m = sel & (codes == k)
                cs[k] = score[m].sum() if m.any() else -np.inf
            pred_col = int(np.argmax(cs[1:7]) + 1)
            pred_row = int(np.argmax(cs[7:13]) + 7)
            if code_pair_to_char(pred_row, pred_col) == letter_of[c]:
                correct += 1
        print(f"  reps={r:2d}  ->  char accuracy = {100 * correct / len(test_chars):5.1f}%")

    # --- latency benchmark (use first session's SOA if available) -----------
    first_meta = json.load(open(os.path.join(session_dirs[0], "meta.json")))
    soa_ms = first_meta.get("soa_ms", 175)
    benchmark_latency(model, ba, n_chan=X.shape[1], raw_epoch_len=epoch_len,
                      model_n_time=X.shape[2], device=device, soa_ms=soa_ms)

    boards = set()
    for sdir in session_dirs:
        boards.add(json.load(open(os.path.join(sdir, "meta.json"))).get("board"))
    if all(b == "SYNTHETIC_BOARD" for b in boards):
        print("\n[note] ALL sessions are SYNTHETIC_BOARD -> no real P300 exists in any "
              "of them.\n       ~chance accuracy is EXPECTED regardless of how much "
              "data is combined.\n       This run validates the multi-session "
              "loading/training/decoding pipeline, not skill.")


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
