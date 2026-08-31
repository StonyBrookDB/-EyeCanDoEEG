"""
Train xDAWN + Riemannian (xDAWN Covariances + Tangent Space + Logistic Regression)
on BNCI2014_009 (MOABB P300 Speller) with LOSO cross-validation.

Leave-One-Subject-Out (LOSO): for each subject i, train on all other subjects
and test on subject i. Reports per-fold and aggregate (mean ± std) metrics.

Outputs:
    - per-fold ROC-AUC, accuracy, target recall, non-target recall
    - aggregate summary across all folds

Install:
    pip install moabb mne numpy scikit-learn pyriemann

Run:
    python train_classical_bnci_loso.py
    python train_classical_bnci_loso.py --subjects 1 2 3 4 5
    python train_classical_bnci_loso.py --nfilter 4
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import BNCI2014_009
from moabb.paradigms import P300


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SEED = 1
EPOCH_MS = 800
TARGET_FS = 50


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


# ---------------------------------------------------------------------------
# Train one LOSO fold
# ---------------------------------------------------------------------------

def train_fold(X_train, y_train, X_test, y_test, nfilter):
    nfilter = min(nfilter, X_train.shape[1])
    clf = make_pipeline(
        XdawnCovariances(nfilter=nfilter, estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )

    clf.fit(X_train, y_train)

    scores = clf.decision_function(X_test)
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
    parser.add_argument(
        "--nfilter",
        type=int,
        default=4,
        help="Number of xDAWN spatial filters (default: 4)",
    )

    args = parser.parse_args()

    if len(args.subjects) < 2:
        raise ValueError("LOSO requires at least 2 subjects.")

    print("==============================================")
    print(" BNCI2014_009 -> xDAWN + Riemannian  [LOSO]")
    print("==============================================")
    print(f"Subjects: {args.subjects}  ({len(args.subjects)} folds)")

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

        auc, acc, recalls = train_fold(X_train, y_train, X_test, y_test, args.nfilter)

        fold_aucs.append(auc)
        fold_accs.append(acc)
        for k in fold_recalls:
            fold_recalls[k].append(recalls[k])

        print(f"  AUC={auc:.3f}  Acc={100*acc:.1f}%  "
              f"Target recall={100*recalls['Target']:.1f}%  "
              f"NonTarget recall={100*recalls['NonTarget']:.1f}%")

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
