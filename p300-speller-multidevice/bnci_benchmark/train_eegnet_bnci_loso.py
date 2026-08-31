"""
Train EEGNet on BNCI2014_009 (MOABB P300 Speller) with LOSO cross-validation.

Leave-One-Subject-Out (LOSO): for each subject i, train on all other subjects
and test on subject i. Reports per-fold and aggregate (mean ± std) metrics.

Outputs:
    - per-fold ROC-AUC, accuracy, target recall, non-target recall
    - aggregate summary across all folds
    - saved EEGNet weights per fold in --output-dir/

Install:
    pip install moabb mne torch numpy scikit-learn

Run:
    python train_eegnet_bnci_loso.py
    python train_eegnet_bnci_loso.py --subjects 1 2 3 4 5
    python train_eegnet_bnci_loso.py --epochs 60 --batch 32 --lr 1e-3
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
EPOCH_MS = 800
TARGET_FS = 50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EEGNet (same architecture as train_eegnet_bnci.py)
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
# Load BNCI2014_009 through MOABB
# ---------------------------------------------------------------------------

def load_bnci(subjects):
    print(f"Loading BNCI2014_009 subjects: {subjects}")
    print("MOABB may download the dataset the first time it is used.")

    dataset = BNCI2014_009()
    paradigm = P300()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    X = np.asarray(X, dtype=np.float32)

    if y.dtype.kind in {"U", "S", "O"}:
        y = np.asarray(
            [1 if str(v).lower() == "target" else 0 for v in y],
            dtype=np.int64,
        )
    else:
        y = np.asarray(y, dtype=np.int64)

    if metadata is not None and "subject" in metadata.columns:
        groups = metadata["subject"].to_numpy()
    elif metadata is not None and "subject_id" in metadata.columns:
        groups = metadata["subject_id"].to_numpy()
    else:
        raise RuntimeError(
            "MOABB did not expose subject IDs in metadata; "
            "cannot safely perform LOSO."
        )

    print(f"Loaded X shape: {X.shape}")
    print(f"Labels: {int(y.sum())} target / {int((y == 0).sum())} non-target")

    return X, y, groups


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_epochs(X, fs=256.0):
    raw_len = int(round(fs * EPOCH_MS / 1000.0))
    raw_len = min(raw_len, X.shape[-1])
    X = X[:, :, :raw_len]
    decim = max(1, int(round(fs / TARGET_FS)))
    X = X[:, :, ::decim]
    return X.astype(np.float32)


def fit_normalization(X_train):
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# ---------------------------------------------------------------------------
# Train one LOSO fold
# ---------------------------------------------------------------------------

def train_fold(X_train, y_train, X_test, y_test, args, device, fold_subject, output_dir):
    mu, sd = fit_normalization(X_train)

    X_train_n = (X_train - mu) / sd
    X_test_n = (X_test - mu) / sd

    Xt = torch.from_numpy(X_train_n)
    yt = torch.from_numpy(y_train.astype(np.float32))
    Xte = torch.from_numpy(X_test_n)

    model = EEGNet(n_chan=X_train.shape[1], n_time=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_idx = np.arange(len(X_train_n))
    rng = np.random.default_rng(SEED)

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
            print(f"    epoch {epoch + 1:3d}/{args.epochs}  loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        scores = model(Xte.to(device)).cpu().numpy()

    preds = (scores > 0).astype(int)

    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = float("nan")

    accuracy = (preds == y_test).mean()

    recalls = {}
    for cls, name in [(1, "Target"), (0, "NonTarget")]:
        mask = y_test == cls
        recalls[name] = (preds[mask] == cls).mean() if mask.any() else float("nan")

    if output_dir:
        out_path = Path(output_dir) / f"fold_subject{fold_subject}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "n_channels": X_train.shape[1],
                "n_time": X_train.shape[2],
                "epoch_ms": EPOCH_MS,
                "target_fs": TARGET_FS,
                "held_out_subject": fold_subject,
                "auc": auc,
                "accuracy": accuracy,
            },
            out_path,
        )
        print(f"    Saved: {out_path}")

    return auc, accuracy, recalls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Subjects to include in LOSO (default: 1-10)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="loso_models",
        help="Directory to save per-fold model weights (empty string to skip)",
    )

    args = parser.parse_args()
    set_seed(SEED)

    if len(args.subjects) < 2:
        raise ValueError("LOSO requires at least 2 subjects.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or None

    print("==============================================")
    print(" BNCI2014_009 -> EEGNet  [LOSO]")
    print("==============================================")
    print(f"Subjects: {args.subjects}  ({len(args.subjects)} folds)")
    print(f"Device:   {device}")

    # Load all subjects at once
    X_all, y_all, groups_all = load_bnci(args.subjects)
    X_all = prepare_epochs(X_all)
    print(f"\nPrepared data shape: {X_all.shape}")

    fold_aucs, fold_accs = [], []
    fold_recalls = {"Target": [], "NonTarget": []}

    for fold_idx, held_out in enumerate(args.subjects):
        print(f"\n--- Fold {fold_idx + 1}/{len(args.subjects)}: held-out subject {held_out} ---")

        train_mask = groups_all != held_out
        test_mask = groups_all == held_out

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        print(f"  Train: {len(X_train)} epochs | Test: {len(X_test)} epochs")

        auc, acc, recalls = train_fold(
            X_train, y_train, X_test, y_test,
            args, device, held_out, output_dir,
        )

        fold_aucs.append(auc)
        fold_accs.append(acc)
        for k in fold_recalls:
            fold_recalls[k].append(recalls[k])

        print(f"  AUC={auc:.3f}  Acc={100*acc:.1f}%  "
              f"Target recall={100*recalls['Target']:.1f}%  "
              f"NonTarget recall={100*recalls['NonTarget']:.1f}%")

    # Summary table
    print("\n==============================================")
    print(" LOSO Summary")
    print("==============================================")
    print(f"{'Subject':<10} {'AUC':>6} {'Acc':>7} {'Tgt%':>7} {'NonTgt%':>9}")
    print("-" * 45)
    for i, subj in enumerate(args.subjects):
        print(f"{subj:<10} {fold_aucs[i]:>6.3f} {100*fold_accs[i]:>6.1f}% "
              f"{100*fold_recalls['Target'][i]:>6.1f}% "
              f"{100*fold_recalls['NonTarget'][i]:>8.1f}%")
    print("-" * 45)

    aucs = np.array(fold_aucs)
    accs = np.array(fold_accs)
    print(f"{'Mean':<10} {np.nanmean(aucs):>6.3f} {100*np.mean(accs):>6.1f}%")
    print(f"{'Std':<10} {np.nanstd(aucs):>6.3f} {100*np.std(accs):>6.1f}%")


if __name__ == "__main__":
    main()
