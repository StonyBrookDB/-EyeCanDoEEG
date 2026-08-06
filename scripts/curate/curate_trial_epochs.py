"""Curate one subject/session's raw EEG into a single, flat per-session
artifact: every trial's epochs, filtered and extracted once, with a
character label attached to each epoch.

Each session contains several trials (one spelled character each, repeated
flashes of all 12 row/column stim_ids), separated by a real multi-second
pause in the recording. Filtering happens exactly once, on the whole
session, before any trial detection or epoching -- see curate/README.md
for why, and for the trial-gap threshold's derivation.

The row/col -> character mapping uses FARWELL_DONCHIN_GRID's own row-major
convention (row_idx = row - 1, col_idx = col - 7, matching
ScoreAccumulator.push()'s exact indexing), not a verified claim about
BNCI2014_009's original physical grid layout -- chosen for internal
consistency with how CharLM/LiveDecoder read stim_id elsewhere.

This module only ever records each trial's real, physical (unpermuted)
character -- it makes no attempt to cover the full 36-symbol grid itself
(BNCI2014_009's real sessions only ever spell 18 of them). See
curate/README.md for why that coverage gap is handled downstream instead,
in build_curated_word.py.

Usage:
    python scripts/curate/curate_trial_epochs.py
    (curates every subject 1-10, session 0-2 by default -- restrict with
    --subject/--session, e.g. --subject 1 2 --session 0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mne
import numpy as np

from eyecando.decode.lm import FARWELL_DONCHIN_GRID
from eyecando.ingestion.data import load_moabb_session
from eyecando.pipeline.bandpass import apply_filter, build_sos

mne.set_log_level("ERROR")

L_FREQ = 0.1
H_FREQ = 20.0
TMIN = -0.1
TMAX = 0.8
BASELINE = (None, 0)
TRIAL_GAP_THRESHOLD_S = 2.0

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def _detect_trial_boundaries(event_samples: np.ndarray, sfreq: float) -> list[tuple[int, int]]:
    """Group flash-event indices into trials via the real inter-trial gap.

    Returns a list of (start_idx, end_idx) into `event_samples` -- NOT raw
    sample numbers -- one pair per detected trial, end_idx exclusive.
    """
    times = event_samples / sfreq
    gaps = np.diff(times)
    boundary = np.flatnonzero(gaps > TRIAL_GAP_THRESHOLD_S)
    starts = [0] + [int(b) + 1 for b in boundary]
    ends = [int(b) + 1 for b in boundary] + [len(event_samples)]
    return list(zip(starts, ends))


def _trial_character(trial_stim_ids: np.ndarray, trial_target_flags: np.ndarray) -> str:
    """The one (row, col) pair consistently target-flagged across this
    trial's repetitions, mapped to FARWELL_DONCHIN_GRID's row-major layout
    -- this trial's real, physical target character, no relabeling.
    """
    target_stim_ids = [
        sid for sid in range(1, 13) if (trial_target_flags[trial_stim_ids == sid] == 2).any()
    ]
    rows = [s for s in target_stim_ids if s <= 6]
    cols = [s for s in target_stim_ids if s > 6]
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError(
            f"expected exactly one target row and one target col, got rows={rows} cols={cols}"
        )
    row, col = rows[0], cols[0]
    return FARWELL_DONCHIN_GRID[(row - 1) * 6 + (col - 7)]


def curate_session(
    subject: int,
    session: int,
    dataset_name: str = "BNCI2014_009",
) -> dict:
    """Filter the whole session once, epoch every flash at once, then
    attach each trial's real character to its epochs.

    Deliberately diverges from preprocess_p300() in one respect: no
    amplitude-based epoch rejection (reject=None) -- build_curated_word.py's
    simulated calibration counts on every trial contributing exactly the
    same fixed number of epochs (96 for BNCI2014_009), and a partially-
    rejected trial would silently throw that arithmetic off. Filtering
    itself still matches preprocess_p300() exactly (see module docstring).

    Returns a flat payload: `epochs` is one (n_epochs, n_channels,
    n_timepoints) array covering every flash in the session, with
    `stim_ids`/`target_flags`/`trial_index` parallel arrays and a
    `characters` array (each epoch's trial's real character) plus
    `trials_meta` (per-trial character/repetition/rejection counts).
    """
    raw = load_moabb_session(subject=subject, session=session, dataset_name=dataset_name)
    sfreq = raw.info["sfreq"]
    eeg_picks = mne.pick_types(raw.info, eeg=True)

    # Filter once, on the whole continuous session -- see module docstring.
    sos = build_sos(l_freq=L_FREQ, h_freq=H_FREQ, sfreq=sfreq)
    raw._data[eeg_picks], _ = apply_filter(sos, raw.get_data(picks=eeg_picks))

    target_events = mne.find_events(raw, stim_channel="target_flag", verbose=False)
    stim_events = mne.find_events(raw, stim_channel="stim_id", verbose=False)
    stim_id_by_sample = dict(zip(stim_events[:, 0], stim_events[:, 2]))

    epochs = mne.Epochs(
        raw,
        target_events,
        event_id={"nontarget": 1, "target": 2},
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        reject=None,
        picks="eeg",
        preload=True,
        verbose=False,
    )
    X = epochs.get_data(picks="eeg")
    kept_samples = epochs.events[:, 0]
    kept_target_flags = epochs.events[:, 2]
    kept_stim_ids = np.array([stim_id_by_sample[s] for s in kept_samples])
    assert len(kept_samples) == len(target_events), (
        "reject=None must keep every flash -- if this ever fails, something "
        "other than amplitude rejection is dropping epochs"
    )

    # Trial boundaries are detected from the ORIGINAL, full flash sequence
    # (including any rejected epochs) so a rejection near a trial boundary
    # can't shift where the gap is measured -- then mapped onto the KEPT
    # epochs by sample number.
    all_samples = target_events[:, 0]
    trial_ranges = _detect_trial_boundaries(all_samples, sfreq)

    trial_index = np.full(len(kept_samples), -1, dtype=int)
    characters: list[str] = []
    trials_meta: dict[str, list] = {
        "trial_index": [],
        "character": [],
        "n_repetitions": [],
        "n_epochs_kept": [],
        "n_epochs_total": [],
    }
    for t, (start, end) in enumerate(trial_ranges):
        trial_samples = all_samples[start:end]
        lo, hi = trial_samples[0], trial_samples[-1]
        in_trial = (kept_samples >= lo) & (kept_samples <= hi)
        trial_index[in_trial] = t

        # Stim_id/target_flag history is derived from the FULL flash
        # sequence (including any rejected epochs), not just the kept
        # ones -- a trial needs all 12 stim_ids' worth of target-flag
        # history to identify its one real target row/col, and a
        # rejected epoch's target_flag is still known even though its
        # EEG data was dropped.
        this_trial_stim_ids = np.array([stim_id_by_sample[s] for s in trial_samples])
        this_trial_target_flags = target_events[start:end, 2]
        character = _trial_character(this_trial_stim_ids, this_trial_target_flags)

        characters.extend([character] * int(in_trial.sum()))
        trials_meta["trial_index"].append(t)
        trials_meta["character"].append(character)
        trials_meta["n_repetitions"].append(len(trial_samples) // 12)
        trials_meta["n_epochs_kept"].append(int(in_trial.sum()))
        trials_meta["n_epochs_total"].append(len(this_trial_stim_ids))

    assert (trial_index >= 0).all(), "every kept epoch must belong to exactly one detected trial"

    return {
        "dataset_name": dataset_name,
        "subject": subject,
        "session": session,
        "l_freq": L_FREQ,
        "h_freq": H_FREQ,
        "tmin": TMIN,
        "tmax": TMAX,
        "baseline": BASELINE,
        "trial_gap_threshold_s": TRIAL_GAP_THRESHOLD_S,
        "epochs": X,
        "stim_ids": kept_stim_ids,
        "target_flags": kept_target_flags,
        "trial_index": trial_index,
        "characters": np.array(characters),
        "trials_meta": trials_meta,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="One or more subjects to curate. Defaults to all 10 of BNCI2014_009's subjects.",
    )
    parser.add_argument(
        "--session",
        type=int,
        nargs="+",
        default=list(range(3)),
        help="One or more sessions to curate for each subject. Defaults to all 3.",
    )
    parser.add_argument("--dataset-name", default="BNCI2014_009")
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    return parser.parse_args()


def main() -> None:
    """Curate every requested subject/session (default: all of
    BNCI2014_009), saving one joblib artifact per session under
    --out-dir. One bad session logs and continues rather than aborting
    the whole batch (see the try/except below); a summary of
    successes/failures prints at the end."""
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[int, int, Exception]] = []
    for subject in args.subject:
        for session in args.session:
            print(f"--- subject {subject}, session {session} ---")
            try:
                payload = curate_session(subject, session, args.dataset_name)
            except Exception as exc:  # noqa: BLE001 -- one bad session shouldn't abort the whole batch
                print(f"FAILED: {exc}")
                failures.append((subject, session, exc))
                continue

            out_path = out_dir / (
                f"{args.dataset_name}_sub-{subject:02d}_ses-{session:02d}_curated.joblib"
            )
            joblib.dump(payload, out_path)

            spelled = "".join(payload["trials_meta"]["character"])
            print(f"Curated {len(payload['epochs'])} epochs (no rejection)")
            print(f"  real spelled sequence: {spelled!r}")
            print(f"Saved to {out_path}")

    n_total = len(args.subject) * len(args.session)
    print(f"\nDone: {n_total - len(failures)}/{n_total} sessions curated.")
    if failures:
        print("Failures:")
        for subject, session, exc in failures:
            print(f"  subject={subject} session={session}: {exc}")


if __name__ == "__main__":
    main()
