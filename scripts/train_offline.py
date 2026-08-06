"""Stage 2: train the offline P300 classifier on MOABB data and save it.

Steps:
1. Load BNCI2014_009 for all subjects, grouped so a subject's sessions
   stay together (a flat per-session list would leak a subject's other
   sessions into the fold holding one of them out).
2. Run preprocess_p300() per session, then pool a subject's sessions
   into one (X, y) pair -- EA is still fit per-session, before pooling.
3. Run LOSO cross-validation. Default: fixed hyperparameters
   (n_components=9, shrinkage="auto" -- the config
   scripts/simulate_dynamic_stopping.py's nested-LOSO search settled on
   across the full calibrate-and-decode pipeline, not just flash-level
   accuracy). --nested-search re-derives them fresh per outer fold
   instead (expensive, blind to calibration/stopping performance --
   mainly a sanity check against the fixed defaults, not how to pick
   them). --n-components/--shrinkage override the fixed values directly.
4. Print a per-subject table: accuracy/precision/recall/F1/AUC/ITR @5reps,
   plus mean +/- std.
5. Save the final model (fit on all subjects) to models/p300_classifier.pkl.

Everything printed here also goes to a timestamped run.log under
results/training_<timestamp>/ via TrainingLogger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from eyecando.ingestion.data import load_moabb_all_subjects_grouped
from eyecando.pipeline.model import pool_sessions
from eyecando.pipeline.preprocessing import preprocess_p300
from eyecando.tuning.model_search import REPETITION_COUNTS, nested_loso, train_loso
from eyecando.utils.path_helpers import run_dir
from eyecando.utils.time_marker import TimeMarker
from eyecando.utils.training_logger import TrainingLogger

# MNE's own INFO-level chatter (e.g. "Creating RawArray", "Estimating
# covariance using EMPIRICAL") is noise at this pipeline's log level --
# only its own warnings/errors are worth seeing here.
mne.set_log_level("WARNING")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "p300_classifier.pkl"
REJECTION_WARNING_THRESHOLD = 0.1


def _parse_shrinkage(value: str) -> float | str:
    return value if value == "auto" else float(value)


DEFAULT_N_COMPONENTS = 9
DEFAULT_SHRINKAGE: float | str = "auto"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            f"Fixed n_components, overriding the default ({DEFAULT_N_COMPONENTS}). "
            "Must be given together with --shrinkage. Incompatible with --nested-search."
        ),
    )
    parser.add_argument(
        "--shrinkage",
        type=_parse_shrinkage,
        default=None,
        help=(
            f'Fixed shrinkage: a float, or "auto", overriding the default '
            f"({DEFAULT_SHRINKAGE!r}). Must be given together with --n-components. "
            "Incompatible with --nested-search."
        ),
    )
    parser.add_argument(
        "--nested-search",
        action="store_true",
        help=(
            "re-derive (n_components, shrinkage) fresh inside every outer fold "
            "(eyecando.pipeline.classifier.nested_loso) instead of using the fixed defaults. "
            "Expensive, and blind to calibration/dynamic-stopping performance -- see "
            "module docstring for why the fixed defaults are preferred."
        ),
    )
    args = parser.parse_args()
    if (args.n_components is None) != (args.shrinkage is None):
        parser.error("--n-components and --shrinkage must be given together, or neither at all")
    if args.nested_search and args.n_components is not None:
        parser.error("--nested-search and --n-components/--shrinkage are mutually exclusive")
    return args


def _subject_to_pooled_arrays(
    session_raws: list, subject_idx: int, prep_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess every session for one real subject, then pool them into
    a single (X, y) pair.

    Each session gets its own per-session EA fit (unsupervised, so no
    label leakage) before pooling -- pooling never touches EA itself,
    only combines the already-normalized results.
    """
    sessions = []
    total_rejected = 0
    for raw in session_raws:
        X, y, _, rejects = preprocess_p300(raw)
        sessions.append((X, y))
        total_rejected += rejects
    X_pool, y_pool = pool_sessions(sessions)
    n_target = int(y_pool.sum())
    prep_rows.append(
        {
            "subject": subject_idx,
            "kept": len(y_pool),
            "rejected": total_rejected,
            "target": n_target,
            "nontarget": len(y_pool) - n_target,
        }
    )
    return X_pool, y_pool


def _log_preprocessing_table(logger: TrainingLogger, prep_rows: list[dict]) -> None:
    """One table for all subjects, then all rejection-rate warnings after --
    not interleaved one subject at a time."""
    header = f"{'subject':>7} | {'kept':>6} | {'rejected':>8} | {'target':>6} | {'nontarget':>9}"
    logger.log(f"[PREP] {header}")
    logger.log(f"[PREP] {'-' * len(header)}")

    warnings = []
    for row in prep_rows:
        logger.log(
            f"[PREP] {row['subject']:>7} | {row['kept']:>6} | {row['rejected']:>8} | "
            f"{row['target']:>6} | {row['nontarget']:>9}"
        )
        total = row["kept"] + row["rejected"]
        rejection_rate = row["rejected"] / total if total > 0 else 0.0
        if rejection_rate > REJECTION_WARNING_THRESHOLD:
            warnings.append(
                f"[PREP] WARNING: subject {row['subject']} rejected {row['rejected']} "
                f"epochs ({rejection_rate:.0%}) -- check for artifacts"
            )
    for warning in warnings:
        logger.log(warning)


def _log_repetition_table(
    logger: TrainingLogger, results: dict, fixed_hyperparams: bool, n_reps: int
) -> None:
    """One table per repetition count -- @1, @3, @5, @10 -- not just @5, so
    the speed/accuracy tradeoff across repetition counts is actually visible
    (and saved to the log) rather than only ever showing the @5 operating
    point."""
    if fixed_hyperparams:
        header = (
            f"{'subject':>7} | {'acc@' + str(n_reps):>6} | {'prec@' + str(n_reps):>6} | "
            f"{'rec@' + str(n_reps):>6} | {'f1@' + str(n_reps):>6} | {'auc@' + str(n_reps):>6} | "
            f"{'itr@' + str(n_reps) + ' (bits/min)':>16}"
        )
    else:
        header = (
            f"{'subject':>7} | {'n_comp':>6} | {'shrink':>6} | {'acc@' + str(n_reps):>6} | "
            f"{'prec@' + str(n_reps):>6} | {'rec@' + str(n_reps):>6} | {'f1@' + str(n_reps):>6} | "
            f"{'auc@' + str(n_reps):>6} | {'itr@' + str(n_reps) + ' (bits/min)':>16}"
        )
    logger.log(f"--- repetitions = {n_reps} ---")
    logger.log(header)
    logger.log("-" * len(header))

    for subject_result in results["per_subject"]:
        rep = subject_result["per_repetition"][n_reps]
        row = f"{subject_result['subject_index']:>7} | "
        if not fixed_hyperparams:
            shrink = subject_result["chosen_shrinkage"]
            shrink_str = shrink if isinstance(shrink, str) else f"{shrink:.2f}"
            row += f"{subject_result['chosen_n_components']:>6} | {shrink_str:>6} | "
        row += (
            f"{rep['accuracy']:>6.3f} | {rep['precision']:>6.3f} | "
            f"{rep['recall']:>6.3f} | {rep['f1']:>6.3f} | {rep['auc']:>6.3f} | "
            f"{rep['itr_bits_per_min']:>16.2f}"
        )
        logger.log(row)

    itrs = [r["per_repetition"][n_reps]["itr_bits_per_min"] for r in results["per_subject"]]
    logger.log("-" * len(header))
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        values = [r["per_repetition"][n_reps][metric] for r in results["per_subject"]]
        logger.log(f"mean {metric}@{n_reps}: {np.mean(values):.3f} +/- {np.std(values):.3f}")
    logger.log(f"mean ITR@{n_reps}reps: {np.mean(itrs):.2f} +/- {np.std(itrs):.2f} bits/min")


def _log_results_table(logger: TrainingLogger, results: dict, fixed_hyperparams: bool) -> None:
    for n_reps in REPETITION_COUNTS:
        _log_repetition_table(logger, results, fixed_hyperparams, n_reps)
        logger.log("")

    logger.log("--- overall, single-flash (no repetition averaging) ---")
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        logger.log(
            f"mean {metric} (overall, single-flash): {results[f'mean_{metric}']:.3f} "
            f"+/- {results[f'std_{metric}']:.3f}"
        )
    logger.log(f"repetition counts evaluated: {REPETITION_COUNTS}")


def main() -> None:
    """Load all subjects, preprocess/pool per subject, run LOSO (fixed
    hyperparameters by default, or --nested-search), print the per-subject
    metrics table, and save the final model to models/p300_classifier.pkl
    (see module docstring for the full 5-step pipeline)."""
    args = _parse_args()
    fixed_hyperparams = not args.nested_search
    n_components = args.n_components if args.n_components is not None else DEFAULT_N_COMPONENTS
    shrinkage = args.shrinkage if args.shrinkage is not None else DEFAULT_SHRINKAGE

    run = run_dir("training")
    logger = TrainingLogger(run / "run.log")
    time_marker = TimeMarker()

    logger.log("[MAIN] Loading all subjects from MOABB BNCI2014_009 (grouped by subject)...")
    grouped_raw_data = load_moabb_all_subjects_grouped()

    logger.log("[MAIN] Preprocessing each session and pooling per subject...")
    prep_rows: list[dict] = []
    subjects = [
        _subject_to_pooled_arrays(session_raws, i, prep_rows)
        for i, session_raws in enumerate(grouped_raw_data)
    ]
    _log_preprocessing_table(logger, prep_rows)

    if fixed_hyperparams:
        logger.log(
            f"[MAIN] Running LOSO across {len(subjects)} subjects with fixed hyperparameters: "
            f"n_components={n_components}, shrinkage={shrinkage}"
        )
        time_marker.mark("loso")
        model, results = train_loso(
            subjects,
            n_components=n_components,
            shrinkage=shrinkage,
            logger=logger,
            time_marker=time_marker,
        )
        logger.log(f"[MAIN] LOSO total time: {time_marker.elapsed_from('loso')}")
    else:
        logger.log(
            f"[MAIN] Running nested LOSO cross-validation across {len(subjects)} subjects..."
        )
        logger.log("[MAIN] (hyperparameters are searched fresh inside every outer fold -- this is")
        logger.log("[MAIN]  ~n_subjects times the compute of a single joint search, by design)")
        time_marker.mark("nested_loso")
        model, results = nested_loso(subjects, logger=logger, time_marker=time_marker)
        logger.log(f"[MAIN] nested LOSO total time: {time_marker.elapsed_from('nested_loso')}")

    _log_results_table(logger, results, fixed_hyperparams)

    results_path = run / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.log(f"[MAIN] Full results written to {results_path}")

    if fixed_hyperparams:
        logger.log(
            f"final model hyperparameters (fixed): "
            f"n_components={n_components}, shrinkage={shrinkage}"
        )
    else:
        logger.log(
            f"final model hyperparameters (from an all-subjects search): "
            f"n_components={results['final_n_components']}, "
            f"shrinkage={results['final_shrinkage']}"
        )

    run_model_path = run / "model.pkl"
    model.save(str(run_model_path))
    logger.log(f"[MAIN] Saved model to {run_model_path}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.log(f"[MAIN] Also updated the latest-deployed model at {MODEL_PATH}")
    logger.log(f"[MAIN] Full log written to {logger.log_path}")


if __name__ == "__main__":
    main()
