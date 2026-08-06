"""Live Test Segment B: marker-only validation for simulate_flash_sequencer.py.

A lighter, marker-only sibling to consume_lsl_stream.py -- no EEG side at
all, no LSLEEGStream. Connects a raw pylsl.StreamInlet to EyeCanDoMarkers
and checks the marker stream is correct and well-formed purely from its own
content: stim_id validity, per-round target/nontarget structure, and ISI
timing -- independent of everything downstream (no EEG, no LiveDecoder).

Trial boundaries are inferred from the marker content itself (which
(row, col) pair each round marks as target), not a --rounds/--word CLI
flag duplicated from the generator -- two independent processes
agreeing only via LSL wire content is the actual proof the encoding
round-trips correctly; matching flags would only prove the flags
matched, not that the stream is self-describing.

round_number is not a wire field -- rounds here means "the next
FLASHES_PER_ROUND (12) consecutive markers," a structural constant, not
anything read off the wire or passed in from the generator.

Usage:
    python scripts/simulate_live/consume_flash_markers.py
    (run scripts/simulate_live/simulate_flash_sequencer.py in another
    terminal first)
"""

from __future__ import annotations

import argparse

import numpy as np
import pylsl

from eyecando.decode.lm import FARWELL_DONCHIN_GRID
from eyecando.pipeline.classifier import FLASHES_PER_ROUND, SECONDS_PER_FLASH

MARKER_STREAM_NAME = "EyeCanDoMarkers"
CONTROL_STREAM_NAME = "EyeCanDoControl"

Record = tuple[float, int, int]  # (timestamp, stim_id, is_target)


def _validate_stim_ids(records: list[Record]) -> list[str]:
    """Every stim_id must be a real row/col code, 1-12."""
    return [
        f"record {i}: stim_id={stim_id} out of range 1-12"
        for i, (_, stim_id, _) in enumerate(records)
        if not (1 <= stim_id <= 12)
    ]


def _chunk_into_rounds(records: list[Record]) -> list[list[Record]]:
    """Consecutive groups of FLASHES_PER_ROUND (12) -- a structural
    constant (6 rows + 6 cols), not the generator's own --rounds, so
    there's no divisibility question here. The stream always ends
    mid-round-or-not (no end-of-stream sentinel exists) -- main() handles
    a trailing short group separately from a truncation anywhere else.
    """
    return [records[i : i + FLASHES_PER_ROUND] for i in range(0, len(records), FLASHES_PER_ROUND)]


def _validate_round(round_records: list[Record]) -> tuple[list[str], tuple[int, int] | None]:
    """Within one full group of 12: exactly {1..12} once each (catches a
    broken shuffle), and exactly one target row (1-6) + one target col
    (7-12). Returns (problems, inferred_target) -- inferred_target is None
    if the round didn't validate cleanly enough to trust one.
    """
    stim_ids = [stim_id for _, stim_id, _ in round_records]
    if sorted(stim_ids) != list(range(1, FLASHES_PER_ROUND + 1)):
        return [f"stim_ids {sorted(stim_ids)} != 1..12 (duplicate or missing)"], None

    target_stims = [stim_id for _, stim_id, is_target in round_records if is_target == 1]
    rows = [s for s in target_stims if s <= 6]
    cols = [s for s in target_stims if s > 6]
    if len(rows) != 1 or len(cols) != 1:
        return [
            f"expected exactly one target row and one target col, got rows={rows} cols={cols}"
        ], None
    return [], (rows[0], cols[0])


def _infer_trials(round_targets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse consecutive rounds sharing the same (row, col) target into
    one trial each -- a new trial starts whenever the target changes.
    `prev` starts at None (never a real (row, col) pair) so the very first
    round is never miscounted."""
    trials: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    for target in round_targets:
        if target != prev:
            trials.append(target)
            prev = target
    return trials


def _row_col_to_char(row: int, col: int) -> str:
    return FARWELL_DONCHIN_GRID[(row - 1) * 6 + (col - 7)]


def _isi_stats(records: list[Record], expected_isi: float, tolerance: float) -> dict:
    if len(records) < 2:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "n_outliers": 0,
            "n_deltas": 0,
        }
    timestamps = np.array([ts for ts, _, _ in records])
    deltas = np.diff(timestamps)
    n_outliers = int(np.sum(np.abs(deltas - expected_isi) > tolerance))
    return {
        "mean": float(deltas.mean()),
        "max": float(deltas.max()),
        "std": float(deltas.std()),
        "n_outliers": n_outliers,
        "n_deltas": len(deltas),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-isi", type=float, default=SECONDS_PER_FLASH)
    parser.add_argument(
        "--isi-tolerance",
        type=float,
        default=0.05,
        help="Seconds of allowed deviation from --expected-isi before a gap counts as "
        "an outlier -- 0.05s covers normal scheduler jitter without masking a real "
        "timing problem.",
    )
    parser.add_argument(
        "--quiet-timeout",
        type=float,
        default=5.0,
        help="How long with no new marker before assuming the run finished.",
    )
    parser.add_argument("--resolve-timeout", type=float, default=15.0)
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Wait for a start signal on EyeCanDoControl before collecting markers -- "
        "only relevant if this script is run standalone against a "
        "--wait-for-trigger generator, not orchestrated by something else.",
    )
    parser.add_argument("--trigger-timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    """Connect to simulate_flash_sequencer.py's Markers stream
    (optionally waiting for its trigger first), collect every marker
    until the stream goes quiet, then validate stim-id/round structure
    and ISI timing purely from the collected wire content (see module
    docstring) and print PASS/FAIL."""
    args = _parse_args()

    print(f"Resolving {MARKER_STREAM_NAME!r} (make sure simulate_flash_sequencer.py is running)...")
    info = pylsl.resolve_byprop("name", MARKER_STREAM_NAME, timeout=args.resolve_timeout)
    if not info:
        raise RuntimeError(
            f"no LSL stream found with name={MARKER_STREAM_NAME!r} within "
            f"{args.resolve_timeout}s -- run scripts/simulate_flash_sequencer.py"
        )
    inlet = pylsl.StreamInlet(info[0])
    print("Resolved.")

    try:
        if args.wait_for_trigger:
            print(
                f"Waiting for the start trigger on {CONTROL_STREAM_NAME!r} "
                "(run scripts/trigger_lsl_start.py once the generator is also waiting)..."
            )
            control_info = pylsl.resolve_byprop(
                "name", CONTROL_STREAM_NAME, timeout=args.trigger_timeout
            )
            if not control_info:
                raise RuntimeError(
                    f"no LSL stream found with name={CONTROL_STREAM_NAME!r} within "
                    f"{args.trigger_timeout}s -- run scripts/trigger_lsl_start.py"
                )
            control_inlet = pylsl.StreamInlet(control_info[0])
            control_inlet.pull_sample(timeout=args.trigger_timeout)
            control_inlet.close_stream()
            print("Trigger observed.")

        print("Waiting for markers...")
        records: list[Record] = []
        while True:
            samples, timestamps = inlet.pull_chunk(timeout=args.quiet_timeout, max_samples=1024)
            if not samples:
                break
            for sample, ts in zip(samples, timestamps):
                records.append((float(ts), int(sample[0]), int(sample[1])))
    finally:
        inlet.close_stream()

    print(f"\nReceived {len(records)} markers.\n")

    stim_id_problems = _validate_stim_ids(records)
    rounds = _chunk_into_rounds(records)

    round_problems: list[str] = []
    round_targets: list[tuple[int, int]] = []
    partial_round_note = None
    for i, round_records in enumerate(rounds):
        is_last = i == len(rounds) - 1
        if len(round_records) < FLASHES_PER_ROUND:
            if is_last:
                partial_round_note = (
                    f"final partial round: {len(round_records)}/{FLASHES_PER_ROUND} "
                    "flashes received (expected at the natural end of a run)"
                )
            else:
                round_problems.append(
                    f"round {i}: truncated to {len(round_records)}/{FLASHES_PER_ROUND} flashes "
                    "-- not the final round, likely a dropped/misdelivered marker"
                )
            continue
        problems, target = _validate_round(round_records)
        if problems:
            round_problems.extend(f"round {i}: {p}" for p in problems)
            continue
        assert target is not None
        round_targets.append(target)

    trials = _infer_trials(round_targets)
    spelled = "".join(_row_col_to_char(row, col) for row, col in trials)
    isi = _isi_stats(records, args.expected_isi, args.isi_tolerance)

    n_full_rounds = len(rounds) - (1 if partial_round_note else 0)
    stim_id_status = "OK" if not stim_id_problems else f"{len(stim_id_problems)} problem(s)"
    print(f"stim-id validity: {stim_id_status}")
    for problem in stim_id_problems:
        print(f"  {problem}")

    print(f"round structure: {len(round_targets)}/{n_full_rounds} full rounds valid")
    for problem in round_problems:
        print(f"  {problem}")
    if partial_round_note:
        print(f"  {partial_round_note}")

    print(
        f"inferred spelled sequence (from marker content alone): {spelled!r} ({len(trials)} trials)"
    )
    print(
        f"ISI: mean={isi['mean']:.3f}s max={isi['max']:.3f}s std={isi['std']:.3f}s "
        f"outliers={isi['n_outliers']}/{isi['n_deltas']} (tolerance={args.isi_tolerance}s)"
    )

    passed = not stim_id_problems and not round_problems
    print()
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
