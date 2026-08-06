"""Assemble a synthetic multi-letter "word" from curated per-session
artifacts (scripts/curate/curate_trial_epochs.py's output), simulating a
realistic, progressively-built-up EuclideanAligner per letter -- not a
single aligner fit on a whole session in hindsight.

A word is just a plain string, e.g. "7UQ". Any physical trial can spell
any character: a row/col permutation relabels which CHARACTER a trial's
real physical (row, col) target counts as, without touching the EEG
itself (see `_permutation_for_trial`) -- see curate/README.md for why.
`row_perm`/`col_perm` travel alongside each letter so a later decode can
be checked against the right character.

Every letter gets a real aligner, built from real epochs (earlier
letters' trials in that session) plus borrowed epochs (that session's
other unused trials) up to a fixed target -- see curate/README.md for
`MIN_CALIBRATION_EPOCHS`'s value and the calibration-pool design.

This script never whitens or filters anything itself: it hands back raw
curated epochs plus a freshly-fit EuclideanAligner per letter, for
whatever replays them live (transform() then update(), LiveDecoder's
own ordering) to apply. curate_trial_epochs.py already filtered each
session whole, once -- see curate/README.md for why re-filtering a
reassembled word would be wrong.

Usage:
    python scripts/curate/build_curated_word.py --word 7UQ
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from eyecando.decode.lm import FARWELL_DONCHIN_GRID
from eyecando.pipeline.alignment import EuclideanAligner

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MIN_CALIBRATION_EPOCHS = 192  # two characters' worth, at 96 epochs/trial

_ROWS = list(range(1, 7))
_COLS = list(range(7, 13))


def _grid_position(character: str) -> tuple[int, int]:
    """(row, col) position of `character` in FARWELL_DONCHIN_GRID's own
    row-major convention (row 1-6, col 7-12), matching curate_trial_
    epochs.py's `_trial_character`.
    """
    idx = FARWELL_DONCHIN_GRID.index(character)
    return idx // 6 + 1, idx % 6 + 7


def _pinned_permutation(
    pin: dict[int, int], domain: list[int], rng: np.random.Generator
) -> dict[int, int]:
    """A bijection over `domain` with `pin`'s entries fixed -- every
    other key in `domain` gets a random bijection over whatever values
    `pin` didn't already claim. Used to build a row_perm/col_perm that
    forces one specific real->desired mapping without constraining
    anything else (see module docstring).
    """
    remaining_keys = [k for k in domain if k not in pin]
    remaining_values = [v for v in domain if v not in pin.values()]
    shuffled_values = rng.permutation(remaining_values).tolist()
    perm = dict(pin)
    perm.update(zip(remaining_keys, shuffled_values))
    return perm


def _permutation_for_trial(
    real_row: int, real_col: int, character: str, rng: np.random.Generator
) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    """row_perm/col_perm relabeling this trial's real physical target as
    `character` -- None for either half that's already correct without
    permuting (matching the trial's own real, unpermuted labeling
    whenever possible, purely to keep the common "letter already spells
    what's needed" case simple), a `_pinned_permutation` otherwise.
    """
    desired_row, desired_col = _grid_position(character)
    row_perm = None if real_row == desired_row else _pinned_permutation(
        {real_row: desired_row}, _ROWS, rng
    )
    col_perm = None if real_col == desired_col else _pinned_permutation(
        {real_col: desired_col}, _COLS, rng
    )
    return row_perm, col_perm


def _trial_pool(
    dataset_name: str,
    curated_dir: Path,
    subject: int | None = None,
    session: int | None = None,
    exclude_sessions: frozenset[int] = frozenset(),
) -> list[tuple[int, int, int, str]]:
    """Every curated (subject, session, trial_index, real_character) this
    corpus actually recorded -- one entry per physical trial. Any entry
    is a valid source for ANY letter build_word() needs (see module
    docstring), so this is a flat pool, not a per-character index.

    `session` without `subject` is not a supported combination (a session
    number alone doesn't identify curated files -- they're keyed by
    subject+session) and raises ValueError rather than silently matching
    every subject's session N.

    `exclude_sessions` drops any session whose index is in the set --
    e.g. so a word-level test can be restricted to sessions that weren't
    also used for a subject's own LDA calibration split elsewhere in the
    same evaluation, avoiding the same kind of leakage
    calibrate_and_score's calib/test session split already avoids. Only
    meaningful when `session` itself is None (a single pinned session is
    either excluded entirely -- which would leave no candidates -- or not
    excluded at all, so this only filters within a sweep).
    """
    if session is not None and subject is None:
        raise ValueError("session= requires subject= -- curated files are keyed by both")

    if subject is not None and session is not None:
        pattern = f"{dataset_name}_sub-{subject:02d}_ses-{session:02d}_curated.joblib"
    elif subject is not None:
        pattern = f"{dataset_name}_sub-{subject:02d}_ses-*_curated.joblib"
    else:
        pattern = f"{dataset_name}_sub-*_ses-*_curated.joblib"
    paths = sorted(curated_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"no curated sessions found in {curated_dir} (pattern: {pattern}) "
            "-- run scripts/curate/curate_trial_epochs.py first"
        )
    pool: list[tuple[int, int, int, str]] = []
    for path in paths:
        data = joblib.load(path)
        subj, sess = data["subject"], data["session"]
        if sess in exclude_sessions:
            continue
        trials_meta = data["trials_meta"]
        for trial_index, char in zip(trials_meta["trial_index"], trials_meta["character"]):
            pool.append((subj, sess, trial_index, char))
    return pool


def _assign_trials(
    word: str,
    pool: list[tuple[int, int, int, str]],
    n_sessions: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, str]]:
    """One (subject, session, trial_index, real_character) source trial
    per character in `word`, in word order.

    `n_sessions` distinct sessions are chosen at random from every
    session represented in `pool`, then each letter independently draws
    a fresh random trial from a randomly chosen one of those sessions
    (uniformly at random both times) -- the same physical trial can
    legitimately supply more than one letter (see module docstring), and
    a real subject would never fire the exact same trial every time
    regardless. Raises ValueError if `pool` doesn't have `n_sessions`
    distinct sessions to draw from.
    """
    if not pool:
        raise ValueError(
            "no curated trials available -- run scripts/curate/curate_trial_epochs.py first"
        )
    sessions = sorted({(subj, sess) for subj, sess, _, _ in pool})
    if n_sessions > len(sessions):
        raise ValueError(
            f"requested n_sessions={n_sessions}, but only {len(sessions)} distinct "
            f"session(s) are available in the candidate pool: {sessions}"
        )
    chosen = [sessions[i] for i in rng.choice(len(sessions), size=n_sessions, replace=False)]
    chosen_set = set(chosen)

    trials_by_session: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for subj, sess, trial_index, char in pool:
        if (subj, sess) in chosen_set:
            trials_by_session.setdefault((subj, sess), []).append((trial_index, char))

    assignment = []
    for _character in word:
        subj, sess = chosen[rng.integers(len(chosen))]
        candidates = trials_by_session[(subj, sess)]
        trial_index, real_char = candidates[rng.integers(len(candidates))]
        assignment.append((subj, sess, trial_index, real_char))
    return assignment


def build_word(
    word: str,
    dataset_name: str = "BNCI2014_009",
    curated_dir: Path = PROCESSED_DIR,
    min_calibration_epochs: int = MIN_CALIBRATION_EPOCHS,
    seed: int | None = None,
    subject: int | None = None,
    session: int | None = None,
    exclude_sessions: frozenset[int] = frozenset(),
    n_sessions: int = 1,
) -> list[dict]:
    """Build a word (a plain string, e.g. "7UQ") from curated data alone.

    `seed` (None by default) is forwarded to the trial-assignment and
    permutation rng -- a real subject never fires the same physical
    trial every time, so leave it random; pass a fixed seed only when a
    call needs to be reproducible (e.g. a test).

    `subject`/`session` restrict which curated sessions are candidates
    (see `_trial_pool`) -- `subject` alone for a realistic single-person
    test, both for one pinned session (`n_sessions` then forced to 1).
    `exclude_sessions` additionally drops specific sessions when
    `session` is None (e.g. to stay disjoint from a subject's own LDA
    calibration split elsewhere in the same evaluation).

    Returns one dict per letter, in word order:
    {subject, session, character, trial_index, epochs, stim_ids,
     target_flags, row_perm, col_perm, n_real_epochs,
     n_calibration_epochs, aligner}. `epochs` are raw, unwhitened epochs
    for this letter's trial (this function never calls transform()).
    `stim_ids`/`target_flags` are the parallel per-epoch arrays a live
    replay needs (e.g. via eyecando.live.decoder.PrebuiltEpochStream).
    `row_perm`/`col_perm` (from `_permutation_for_trial`) must be applied
    to a decoded row/col before checking it against `character` -- a
    decoder has no idea a permuted trial was used. `n_real_epochs` is
    how much of the calibration pool came from trials already spelled
    earlier in the word; `n_calibration_epochs` is the pool's total (see
    module docstring). `aligner` is a fresh EuclideanAligner fit on that
    pool (None only if the session had no other trial at all to draw
    from).
    """
    rng = np.random.default_rng(seed)
    pool = _trial_pool(
        dataset_name,
        curated_dir,
        subject=subject,
        session=session,
        exclude_sessions=exclude_sessions,
    )
    effective_n_sessions = 1 if session is not None else n_sessions
    assignment = _assign_trials(word, pool, effective_n_sessions, rng)

    session_cache: dict[tuple[int, int], dict] = {}
    spelled_trials_per_session: dict[tuple[int, int], list[int]] = {}
    result = []

    for character, (subj, sess, trial_index, real_char) in zip(word, assignment):
        key = (subj, sess)
        if key not in session_cache:
            path = curated_dir / f"{dataset_name}_sub-{subj:02d}_ses-{sess:02d}_curated.joblib"
            session_cache[key] = joblib.load(path)
            spelled_trials_per_session[key] = []
        session_data = session_cache[key]
        spelled_so_far = spelled_trials_per_session[key]

        trial_mask = session_data["trial_index"] == trial_index
        this_letter_epochs = session_data["epochs"][trial_mask]

        real_mask = np.isin(session_data["trial_index"], spelled_so_far)
        real_epochs = session_data["epochs"][real_mask]

        # Fixed target, not one that grows with word position -- see
        # curate/README.md. The TOTAL pool (real + borrowed) is capped at
        # exactly `target`, not just the borrowed portion. Once the real
        # pool alone reaches or exceeds the target, borrowing stops and
        # real itself is truncated (order doesn't matter for a covariance
        # fit, so a plain truncation is as good as any other fixed-size
        # subset).
        target = min_calibration_epochs
        if real_epochs.shape[0] > target:
            real_epochs = real_epochs[:target]
        n_real = int(real_epochs.shape[0])

        pool_parts = [real_epochs]
        excluded = set(spelled_so_far) | {trial_index}
        borrow_trials = sorted(
            t for t in session_data["trials_meta"]["trial_index"] if t not in excluded
        )
        collected = n_real  # borrowing tops up TOTAL to target, not another target's worth
        for t in borrow_trials:
            if collected >= target:
                break
            t_epochs = session_data["epochs"][session_data["trial_index"] == t]
            pool_parts.append(t_epochs)
            collected += t_epochs.shape[0]

        calibration_pool = np.concatenate(pool_parts, axis=0)
        aligner = (
            EuclideanAligner().fit(calibration_pool) if calibration_pool.shape[0] > 0 else None
        )
        n_calibration_epochs = int(calibration_pool.shape[0])

        real_row, real_col = _grid_position(real_char)
        row_perm, col_perm = _permutation_for_trial(real_row, real_col, character, rng)
        result.append(
            {
                "subject": subj,
                "session": sess,
                "character": character,
                "trial_index": trial_index,
                "epochs": this_letter_epochs,
                "stim_ids": session_data["stim_ids"][trial_mask],
                "target_flags": session_data["target_flags"][trial_mask],
                "row_perm": row_perm,
                "col_perm": col_perm,
                "n_real_epochs": n_real,
                "n_calibration_epochs": n_calibration_epochs,
                "aligner": aligner,
            }
        )

        spelled_trials_per_session[key].append(trial_index)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", required=True, help='e.g. "7UQ" -- one character per letter.')
    parser.add_argument("--dataset-name", default="BNCI2014_009")
    parser.add_argument("--curated-dir", default=str(PROCESSED_DIR))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix which physical trials get picked, for a reproducible run -- "
        "omit for genuine randomness every run (the default).",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Restrict assembly to one subject's curated sessions -- required for a "
        "realistic single-person live/online decode test (default: any subject).",
    )
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help="Restrict further to one session of --subject (requires --subject).",
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=1,
        help="How many distinct sessions to draw from -- 1 (default) keeps the whole "
        "word within a single session; raise it to deliberately exercise "
        "cross-session splicing.",
    )
    return parser.parse_args()


def main() -> None:
    """Build --word from curated data and print each letter's source
    trial and calibration-pool status (real vs. borrowed epoch counts,
    see module docstring), as a quick smoke test of build_word() itself
    without running a full LiveDecoder replay."""
    args = _parse_args()
    word = build_word(
        args.word,
        args.dataset_name,
        Path(args.curated_dir),
        seed=args.seed,
        subject=args.subject,
        session=args.session,
        n_sessions=args.n_sessions,
    )

    spelled = "".join(letter["character"] for letter in word)
    n_sessions = len({(letter["subject"], letter["session"]) for letter in word})
    print(f"Assembled word: {spelled!r} ({len(word)} letters, {n_sessions} distinct sessions)")
    for i, letter in enumerate(word):
        n_borrowed = letter["n_calibration_epochs"] - letter["n_real_epochs"]
        status = (
            f"aligner ready ({letter['n_calibration_epochs']} epochs: "
            f"{letter['n_real_epochs']} real + {n_borrowed} borrowed)"
            if letter["aligner"] is not None
            else "NO ALIGNER (session has no other trial to draw from)"
        )
        print(
            f"  letter {i}: {letter['character']!r} (subject={letter['subject']}, "
            f"session={letter['session']}, trial={letter['trial_index']}) -- {status}"
        )


if __name__ == "__main__":
    main()
