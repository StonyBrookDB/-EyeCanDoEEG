"""Word-level live-decode testing over curated data, at two levels of
cross-session splicing, each averaged over several repeats since
build_word()'s trial assignment is randomized per call -- a single run
of any mode is one sample, not a reliable estimate.

Modes:

- in_session: subject AND session pinned -- least splicing possible.
- cross_session: subject pinned, session unrestricted, `n_sessions`
  forced to `CROSS_SESSION_N_SESSIONS` -- deliberately spans several of
  that subject's sessions (each with its own real EA/noise
  characteristics) without touching another subject's recordings. See
  curate/README.md for why this is a deliberate choice, not a coverage
  necessity.

Multiple words: --words accepts any number (default: DEFAULT_WORDS --
see curate/README.md for its design and DEFAULT_SENTENCES'). Results
aggregate over every (word, repeat) pair per mode, plus a per-word
breakdown -- repeats average out one word's own random assignment;
several words also spread out any one session's particular noise level.

Usage:
    python scripts/curate/test_curated_words.py --words "HELLO WORLD" \
        --mode in_session --subject 4 --session 0 --repeats 10
    python scripts/curate/test_curated_words.py --words "HELLO WORLD" "CAT" "AGENT 47" \
        --mode cross_session --subject 4 --repeats 10
    # default word list, both modes, for direct comparison:
    python scripts/curate/test_curated_words.py --mode all --subject 4 --session 0 --repeats 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from build_curated_word import build_word

from eyecando.decode.lm import FARWELL_DONCHIN_GRID, CharLM
from eyecando.decode.stopping import should_decode
from eyecando.live.decoder import LiveDecoder, PrebuiltEpochStream
from eyecando.pipeline.classifier import N_SYMBOLS, SECONDS_PER_FLASH, compute_itr
from eyecando.pipeline.model import P300Model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "p300_classifier.pkl"

# See curate/README.md for this list's design (each entry tested
# independently -- see --words) and DEFAULT_SENTENCE below for the
# one-continuous-message case.
DEFAULT_WORDS = [
    "CAT",
    "HELLO",
    "TEST WORD",
    "GOOD JOB",
    "AGENT 47",
    "ROOM 237",
    "HELLO WORLD",
    "GOOD MORNING",
    "DATA SCIENCE",
    "NEURAL SIGNAL",
    "SPEAK YOUR MIND",
    "EYE CAN DO",
    "TYPE FAST TODAY",
    "MISSISSIPPI",  # repeat-heavy stress test
]

# Coherent, real English sentences for testing use_lm as one continuous
# message -- see curate/README.md for the full design (why real sentences
# instead of DEFAULT_WORDS joined together, ordering, corpus sourcing).
DEFAULT_SENTENCE = "DO YOU PREFER CHICKEN OR BEEF"
DEFAULT_SENTENCES = [
    DEFAULT_SENTENCE,
    "I AM SORRY BUT I HAVE TO GO",
    "SARAH NEVER CALLED ME BACK",
    "GOOD MORNING ANDREA",
]

Mode = str  # "in_session" | "cross_session"

# Forced for --mode cross_session -- see module docstring: no longer a
# coverage necessity, just a deliberate choice of how many of a subject's
# sessions to splice together. BNCI2014_009 has 3 sessions/subject; 2
# already exercises real cross-session drift without requiring every
# subject to have a 3rd.
CROSS_SESSION_N_SESSIONS = 2


def _idx_to_char(idx: int, row_perm: dict[int, int] | None, col_perm: dict[int, int] | None) -> str:
    """Undo the standard FARWELL_DONCHIN_GRID row-major mapping, apply
    this letter's own (row_perm, col_perm) if it has one, then re-map --
    the correct way to interpret a decoded index when the trial it came
    from was labeled via an on-demand permutation (see
    build_curated_word.py)."""
    row, col = idx // 6 + 1, idx % 6 + 7
    if row_perm is not None:
        row = row_perm[row]
    if col_perm is not None:
        col = col_perm[col]
    return FARWELL_DONCHIN_GRID[(row - 1) * 6 + (col - 7)]


def run_word_session(
    word: str,
    model: P300Model,
    subject: int | None = None,
    session: int | None = None,
    seed: int | None = None,
    exclude_sessions: frozenset[int] = frozenset(),
    n_sessions: int = 1,
    lm: CharLM | None = None,
    lm_temperature: float = 1.0,
    threshold: float = 0.9,
    temperature: float = 2.0,
) -> dict:
    """Assemble `word` (see build_curated_word.build_word) and replay it
    through a real LiveDecoder, one letter at a time. Shared by this
    script's batch runner and scripts/simulate_live/livetesting.ipynb's
    "session runner" cell, instead of duplicating the replay loop.

    `exclude_sessions`/`n_sessions` forward to build_word() as-is.
    `lm`/`lm_temperature`/`threshold`/`temperature` forward to
    LiveDecoder as-is -- pass the same values a condition selected
    elsewhere, not LiveDecoder's own defaults. `lm.reset_context()` runs
    up front when `lm` is given, since `lm` is typically a shared CharLM
    reused across many calls -- purely a speed optimization (see its own
    docstring), not needed for correctness.

    Returns {"intended": str, "decoded": str, "n_correct": int,
    "n_letters": int, "accuracy": float, "flashes_per_letter": list[int],
    "n_distinct_sessions": int}.
    """
    if lm is not None:
        lm.reset_context()
    word = word.upper()
    letters = build_word(
        word,
        subject=subject,
        session=session,
        seed=seed,
        exclude_sessions=exclude_sessions,
        n_sessions=n_sessions,
    )

    decoded_chars: list[str] = []
    flashes_per_letter: list[int] = []
    decoder: LiveDecoder | None = None

    for letter in letters:
        stream_items = [
            (int(stim_id), int(flag == 2), epoch[None, ...])
            for stim_id, flag, epoch in zip(
                letter["stim_ids"], letter["target_flags"], letter["epochs"]
            )
        ]
        stream = PrebuiltEpochStream(stream_items)
        if decoder is None:
            decoder = LiveDecoder(
                stream=stream,
                model=model,
                aligner=letter["aligner"],
                lm=lm,
                lm_temperature=lm_temperature,
                threshold=threshold,
                temperature=temperature,
            )
        else:
            decoder.stream = stream
            decoder.aligner = letter["aligner"]

        decoded_symbol = None
        epoch_idx = 0
        while True:
            item = stream.get_epoch(timeout=decoder.get_epoch_timeout)
            if item is None:
                break
            stim_id, _is_target, epoch = item
            decoder.score_epoch(stim_id, epoch)
            decode, idx = should_decode(
                decoder.accumulator,
                decoder.threshold,
                lm=decoder.lm,
                context=decoder.context,
                temperature=decoder.temperature,
                lm_temperature=decoder.lm_temperature,
            )
            epoch_idx += 1
            if decode:
                decoded_symbol = idx
                if decoder.lm is not None:
                    decoder.context += decoder.lm.character_set[idx]
                break
        stream.stop()
        flashes_per_letter.append(epoch_idx)

        decoded_char = (
            _idx_to_char(decoded_symbol, letter["row_perm"], letter["col_perm"])
            if decoded_symbol is not None
            else "?"
        )
        decoded_chars.append(decoded_char)

    decoded_text = "".join(decoded_chars)
    n_correct = sum(1 for a, b in zip(decoded_text, word) if a == b)
    n_distinct_sessions = len({(letter["subject"], letter["session"]) for letter in letters})

    return {
        "intended": word,
        "decoded": decoded_text,
        "n_correct": n_correct,
        "n_letters": len(word),
        "accuracy": n_correct / len(word),
        "flashes_per_letter": flashes_per_letter,
        "n_distinct_sessions": n_distinct_sessions,
    }


def _aggregate(records: list[dict]) -> dict:
    """Mean/std across whatever (word, repeat) records are handed in --
    shared by both the overall (all words pooled) and per-word summaries."""
    accs = np.array([r["accuracy"] for r in records])
    flashes_arr = np.array([r["mean_flashes_per_letter"] for r in records])
    itr_arr = np.array([r["itr"] for r in records])
    sessions_arr = np.array([r["n_distinct_sessions"] for r in records])
    return {
        "n": len(records),
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_flashes_per_letter": float(flashes_arr.mean()),
        "std_flashes_per_letter": float(flashes_arr.std()),
        "mean_itr": float(itr_arr.mean()),
        "std_itr": float(itr_arr.std()),
        "mean_distinct_sessions": float(sessions_arr.mean()),
    }


def run_mode(
    mode: Mode,
    words: list[str],
    model: P300Model,
    repeats: int,
    subject: int | None,
    session: int | None,
) -> dict:
    """Repeat run_word_session() `repeats` times per word (fresh randomness
    each call -- seed=None) and aggregate, both overall (every word+repeat
    pooled together) and per-word. mode only controls which subject/session
    restriction gets passed through; the replay/scoring logic is identical
    across modes.
    """
    if mode == "in_session":
        if subject is None or session is None:
            raise ValueError("--mode in_session requires both --subject and --session")
        kwargs = {"subject": subject, "session": session}
    elif mode == "cross_session":
        if subject is None:
            raise ValueError("--mode cross_session requires --subject")
        kwargs = {"subject": subject, "session": None, "n_sessions": CROSS_SESSION_N_SESSIONS}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    per_word_records: dict[str, list[dict]] = {word: [] for word in words}
    for word in words:
        for _ in range(repeats):
            result = run_word_session(word, model, **kwargs)
            mean_flashes_this_run = float(np.mean(result["flashes_per_letter"]))
            seconds_per_selection = mean_flashes_this_run * SECONDS_PER_FLASH
            itr = compute_itr(result["accuracy"], N_SYMBOLS, seconds_per_selection)
            per_word_records[word].append(
                {
                    "accuracy": result["accuracy"],
                    "mean_flashes_per_letter": mean_flashes_this_run,
                    "itr": itr,
                    "n_distinct_sessions": result["n_distinct_sessions"],
                }
            )

    all_records = [r for records in per_word_records.values() for r in records]
    overall = _aggregate(all_records)
    overall["mode"] = mode
    overall["per_word"] = {word: _aggregate(records) for word, records in per_word_records.items()}
    return overall


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        help='One or more words, e.g. --words "HELLO WORLD" CAT. Defaults to DEFAULT_WORDS.',
    )
    parser.add_argument(
        "--mode",
        choices=["in_session", "cross_session", "all"],
        default="all",
    )
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def _format_row(label: str, agg: dict) -> str:
    return (
        f"| {label} "
        f"| {agg['mean_accuracy']:.3f} +/- {agg['std_accuracy']:.3f} "
        f"| {agg['mean_flashes_per_letter']:.2f} +/- {agg['std_flashes_per_letter']:.2f} "
        f"| {agg['mean_itr']:.2f} +/- {agg['std_itr']:.2f} "
        f"| {agg['mean_distinct_sessions']:.2f} |"
    )


def main() -> None:
    """Run every requested word through run_mode() for each requested
    --mode, then print an overall (all words pooled) table and a
    per-word breakdown table -- accuracy/flashes-per-letter/ITR/mean
    distinct sessions, mean +/- std over --repeats (see module
    docstring for why a single run isn't a reliable estimate)."""
    args = _parse_args()
    model = P300Model.load(Path(args.model_path))
    words = args.words if args.words is not None else DEFAULT_WORDS

    modes = ["in_session", "cross_session"] if args.mode == "all" else [args.mode]

    print(
        f"Words ({len(words)}): {[w.upper() for w in words]}, "
        f"{args.repeats} repeat(s) each per mode"
    )

    header = "| mode | accuracy | flashes/letter | ITR (bits/min) | mean distinct sessions |"
    sep = "|---|---|---|---|---|"

    print("\n=== overall (all words pooled) ===")
    print(header)
    print(sep)
    mode_results = {}
    for mode in modes:
        result = run_mode(mode, words, model, args.repeats, args.subject, args.session)
        mode_results[mode] = result
        print(_format_row(mode, result))

    for mode in modes:
        print(f"\n=== per-word breakdown: {mode} ===")
        print(header.replace("| mode |", "| word |"))
        print(sep)
        for word, agg in mode_results[mode]["per_word"].items():
            print(_format_row(word.upper(), agg))


if __name__ == "__main__":
    main()
