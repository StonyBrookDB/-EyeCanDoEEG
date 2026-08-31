"""
Train EEGNet on BNCI2014_009 (MOABB P300 Speller).

This script keeps the EEGNet architecture from the user's existing
train_eegnet_multi.py, but replaces the custom CSV/session loader with
MOABB's BNCI2014_009 loader.

Default evaluation:
    Train on subjects 1-8
    Test on held-out subjects 9-10

The split is SUBJECT-WISE, so no EEG epochs from the test subjects are
used during training.

Outputs:
    - training loss
    - held-out target recall
    - held-out non-target recall
    - held-out ROC-AUC
    - saved EEGNet weights + normalization statistics

NOTE:
    BNCI2014_009 is exposed through MOABB as Target vs NonTarget epochs.
    This script therefore evaluates single-flash classification. It does
    NOT claim character accuracy because the MOABB epoch output does not
    directly expose the row/column flash code needed for 6x6 spelling
    reconstruction.

Install:
    pip install moabb mne torch numpy scikit-learn

Run:
    python train_eegnet_bnci.py

Optional:
    python train_eegnet_bnci.py --test-subjects 9 10
    python train_eegnet_bnci.py --test-subjects 10
    python train_eegnet_bnci.py --epochs 60 --batch 32 --lr 1e-3
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from moabb.datasets import BNCI2014_009
from moabb.paradigms import P300


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SEED = 1
EPOCH_MS = 800      # Match the user's existing pipeline
TARGET_FS = 50       # Match its decimation target (~50 Hz)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EEGNet -- same architecture as the user's uploaded script
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(
        self,
        n_chan,
        n_time,
        F1=8,
        D=2,
        F2=16,
        kern_len=16,
        drop=0.5,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1, F1, (1, kern_len),
                padding=(0, kern_len // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                F1, F1 * D, (n_chan, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                F1 * D, F1 * D, (1, 8),
                padding=(0, 4),
                groups=F1 * D,
                bias=False,
            ),
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
# Load BNCI2014_009 through MOABB
# ---------------------------------------------------------------------------

def load_bnci(subjects):
    """
    Returns:
        X:      float32 array [epochs, channels, time]
        y:      int array, 1=Target, 0=NonTarget
        groups: subject ID for each epoch
        fs:     original sampling rate
    """

    print(f"Loading BNCI2014_009 subjects: {subjects}")
    print("MOABB may download the dataset the first time it is used.")

    dataset = BNCI2014_009()

    # P300 paradigm gives 0.8 s epochs around each flash and Target /
    # NonTarget labels.
    paradigm = P300()

    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=subjects,
    )

    # MOABB normally returns:
    #   X = [n_epochs, n_channels, n_times]
    #   y = string labels such as "Target"/"NonTarget"
    X = np.asarray(X, dtype=np.float32)

    if y.dtype.kind in {"U", "S", "O"}:
        y = np.asarray(
            [1 if str(v).lower() == "target" else 0 for v in y],
            dtype=np.int64,
        )
    else:
        y = np.asarray(y, dtype=np.int64)

    # Metadata normally contains subject. Fall back to an even split only
    # if a future MOABB version does not expose it.
    if metadata is not None and "subject" in metadata.columns:
        groups = metadata["subject"].to_numpy()
    elif metadata is not None and "subject_id" in metadata.columns:
        groups = metadata["subject_id"].to_numpy()
    else:
        raise RuntimeError(
            "MOABB did not expose subject IDs in metadata; "
            "cannot safely perform subject-wise evaluation."
        )

    fs = 256.0

    print(f"Loaded X shape: {X.shape}")
    print(
        f"Labels: {int(y.sum())} target / "
        f"{int((y == 0).sum())} non-target"
    )

    return X, y, groups, fs


# ---------------------------------------------------------------------------
# Match the user's existing time representation
# ---------------------------------------------------------------------------

def prepare_epochs(X, fs):
    """
    Match the existing pipeline's approximately:
        667 ms epoch
        decimation to ~50 Hz

    BNCI2014_009 is already documented by MOABB as preprocessed with
    a 0.1-20 Hz bandpass, so we do not apply another causal filter here.
    """

    raw_len = int(round(fs * EPOCH_MS / 1000.0))
    raw_len = min(raw_len, X.shape[-1])

    X = X[:, :, :raw_len]

    decim = max(1, int(round(fs / TARGET_FS)))
    X = X[:, :, ::decim]

    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fit_normalization(X_train):
    """
    Channel-wise normalization using ONLY training subjects.
    """

    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_test, y_test, args, device):

    mu, sd = fit_normalization(X_train)

    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.float32))

    Xte = torch.from_numpy(X_test)
    yte = torch.from_numpy(y_test.astype(np.float32))

    model = EEGNet(
        n_chan=X_train.shape[1],
        n_time=X_train.shape[2],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_idx = np.arange(len(X_train))
    rng = np.random.default_rng(SEED)

    print("\nTraining EEGNet...")
    print(f"Train epochs: {len(train_idx)}")
    print(f"Test epochs:  {len(X_test)}")
    print(f"Input shape:  channels={X_train.shape[1]}, time={X_train.shape[2]}")
    print(f"Device:       {device}")

    model.train()

    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        total_loss = 0.0

        for start in range(0, len(train_idx), args.batch):
            idx = train_idx[start:start + args.batch]

            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_idx)
            print(
                f"  epoch {epoch + 1:3d}/{args.epochs} "
                f"loss={avg_loss:.4f}"
            )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        scores = model(Xte.to(device)).cpu().numpy()

    preds = (scores > 0).astype(int)

    print("\n=== Held-out subject evaluation ===")

    try:
        auc = roc_auc_score(y_test, scores)
        print(f"ROC-AUC:                 {auc:.3f}")
    except ValueError:
        auc = float("nan")
        print("ROC-AUC:                 undefined")

    for cls, name in [(1, "Target"), (0, "NonTarget")]:
        mask = y_test == cls

        if mask.any():
            recall = (preds[mask] == cls).mean()
            print(
                f"{name:10s} recall:        "
                f"{100 * recall:5.1f}%  (n={mask.sum()})"
            )

    overall = (preds == y_test).mean()
    print(f"Single-flash accuracy:   {100 * overall:5.1f}%")

    # Save model + normalization.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "n_channels": X_train.shape[1],
            "n_time": X_train.shape[2],
            "epoch_ms": EPOCH_MS,
            "target_fs": TARGET_FS,
            "train_subjects": args.train_subjects,
            "test_subjects": args.test_subjects,
            "auc": auc,
        },
        output,
    )

    print(f"\nSaved model to: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-subjects",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
    )

    parser.add_argument(
        "--test-subjects",
        nargs="+",
        type=int,
        default=[9, 10],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eegnet_bnci2014_009.pt",
    )

    args = parser.parse_args()

    set_seed(SEED)

    overlap = set(args.train_subjects) & set(args.test_subjects)

    if overlap:
        raise ValueError(
            f"Train/test subject overlap: {sorted(overlap)}"
        )

    if not args.train_subjects:
        raise ValueError("Need at least one training subject.")

    if not args.test_subjects:
        raise ValueError("Need at least one test subject.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==============================================")
    print(" BNCI2014_009 -> EEGNet")
    print("==============================================")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Test subjects:  {args.test_subjects}")
    print(f"Device:         {device}")

    train_subjects = args.train_subjects
    test_subjects = args.test_subjects

    X_train, y_train, g_train, fs = load_bnci(train_subjects)
    X_test, y_test, g_test, _ = load_bnci(test_subjects)

    X_train = prepare_epochs(X_train, fs)
    X_test = prepare_epochs(X_test, fs)

    print(f"\nPrepared train shape: {X_train.shape}")
    print(f"Prepared test shape:  {X_test.shape}")

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        args,
        device,
    )


if __name__ == "__main__":
    main()