"""
Train xDAWN + Riemannian (xDAWN Covariances + Tangent Space + Logistic Regression)
on BNCI2014_009 (MOABB P300 Speller).

Default evaluation:
    Train on subjects 1-8
    Test on held-out subjects 9-10

The split is SUBJECT-WISE, so no EEG epochs from the test subjects are used
during training.

Outputs:
    - in-sample ROC-AUC (sanity check)
    - held-out target recall
    - held-out non-target recall
    - held-out ROC-AUC and accuracy

Install:
    pip install moabb mne numpy scikit-learn pyriemann

Run:
    python train_classical_bnci.py
    python train_classical_bnci.py --test-subjects 9 10
    python train_classical_bnci.py --train-subjects 1 2 3 4 5 6 --test-subjects 7 8 9 10
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
            "cannot safely perform subject-wise evaluation."
        )

    fs = 256.0

    print(f"Loaded X shape: {X.shape}")
    print(f"Labels: {int(y.sum())} target / {int((y == 0).sum())} non-target")

    return X, y, groups, fs


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
        "--nfilter",
        type=int,
        default=4,
        help="Number of xDAWN spatial filters (default: 4)",
    )

    args = parser.parse_args()

    overlap = set(args.train_subjects) & set(args.test_subjects)
    if overlap:
        raise ValueError(f"Train/test subject overlap: {sorted(overlap)}")
    if not args.train_subjects:
        raise ValueError("Need at least one training subject.")
    if not args.test_subjects:
        raise ValueError("Need at least one test subject.")

    print("==============================================")
    print(" BNCI2014_009 -> xDAWN + Riemannian")
    print("==============================================")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Test subjects:  {args.test_subjects}")

    X_train, y_train, _, fs = load_bnci(args.train_subjects)
    X_test, y_test, _, _ = load_bnci(args.test_subjects)

    X_train = prepare_epochs(X_train, fs)
    X_test = prepare_epochs(X_test, fs)

    print(f"\nPrepared train shape: {X_train.shape}")
    print(f"Prepared test shape:  {X_test.shape}")

    nfilter = min(args.nfilter, X_train.shape[1])
    clf = make_pipeline(
        XdawnCovariances(nfilter=nfilter, estimator="oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )

    print("\nFitting xDAWN + Riemannian pipeline...")
    clf.fit(X_train, y_train)

    try:
        train_auc = roc_auc_score(y_train, clf.decision_function(X_train))
        print(f"In-sample ROC-AUC (train subjects): {train_auc:.3f}")
    except ValueError:
        print("In-sample AUC undefined (need both classes present)")

    scores = clf.decision_function(X_test)
    preds = (scores > 0).astype(int)

    print("\n=== Held-out subject evaluation ===")

    try:
        auc = roc_auc_score(y_test, scores)
        print(f"ROC-AUC:               {auc:.3f}")
    except ValueError:
        auc = float("nan")
        print("ROC-AUC:               undefined")

    for cls, name in [(1, "Target"), (0, "NonTarget")]:
        mask = y_test == cls
        if mask.any():
            recall = (preds[mask] == cls).mean()
            print(f"{name:10s} recall:      {100 * recall:5.1f}%  (n={mask.sum()})")

    overall = (preds == y_test).mean()
    print(f"Single-flash accuracy: {100 * overall:5.1f}%")


if __name__ == "__main__":
    main()
