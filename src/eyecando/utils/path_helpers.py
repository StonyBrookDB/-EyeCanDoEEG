"""Single source of truth for all result paths. Change RESULTS_BASE to
relocate every run, checkpoint, and model produced by training/calibration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Change this one line to relocate all results
RESULTS_BASE: Path = Path(__file__).resolve().parents[3] / "results"

# Controls whether per-fold models are saved to disk during LOSO.
# Set True when you want to inspect individual fold models; False for normal runs.
SAVE_FOLD_CHECKPOINTS: bool = False


def run_dir(run_type: str) -> Path:
    """Construct a timestamped run directory under RESULTS_BASE, guaranteed
    unique even when multiple processes call this in the same minute.

    Parameters
    ----------
    run_type : str
        One of "training", "calibration", "component_ablation". Becomes the
        directory prefix.

    Returns
    -------
    Path
        e.g. results/training_260625-1430/ (or -1430_2, -1430_3, ... if
        another process already claimed the plain name this minute).
        Directory is created on first call, atomically (`exist_ok=False`)
        so concurrent callers can never share one -- see utils/README.md
        for the bug this fixes.
    """
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    base_name = f"{run_type}_{timestamp}"
    path = RESULTS_BASE / base_name
    suffix = 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            suffix += 1
            path = RESULTS_BASE / f"{base_name}_{suffix}"


def model_dir(run: Path) -> Path:
    """Return the model/ subdirectory for a given run directory.

    Parameters
    ----------
    run : Path
        A run directory, typically one returned by `run_dir()`.

    Returns
    -------
    Path
        `run / "model"`. Created (including any missing parents) if it
        does not already exist.
    """
    path = run / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(run: Path, fold: int, test_subject: int) -> Path:
    """Return path for a per-fold model checkpoint.

    Only called when SAVE_FOLD_CHECKPOINTS is True.

    Parameters
    ----------
    run : Path
        A run directory, typically one returned by `run_dir()`.
    fold : int
        LOSO fold index.
    test_subject : int
        Subject held out as the test subject for this fold.

    Returns
    -------
    Path
        e.g. results/training_260625-1430/model/fold_3_subject_4.pkl
    """
    return model_dir(run) / f"fold_{fold}_subject_{test_subject}.pkl"


def final_model_path(run: Path) -> Path:
    """Return path for the final model refitted on all subjects.

    Parameters
    ----------
    run : Path
        A run directory, typically one returned by `run_dir()`.

    Returns
    -------
    Path
        e.g. results/training_260625-1430/model/p300_classifier_final.pkl
    """
    return model_dir(run) / "p300_classifier_final.pkl"


def calibrated_model_path(run: Path, subject_id: str) -> Path:
    """Return path for a subject-adapted model from a calibration run.

    Parameters
    ----------
    run : Path
        A run directory, typically one returned by `run_dir()`.
    subject_id : str
        Subject identifier as it should appear in the filename (e.g.
        ``"01"``), not a numeric index.

    Returns
    -------
    Path
        e.g. results/calibration_260625-1502/model/p300_calibrated_subj01.pkl
    """
    return model_dir(run) / f"p300_calibrated_subj{subject_id}.pkl"
