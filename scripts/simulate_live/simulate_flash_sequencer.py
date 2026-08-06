"""Live Test Segment B: headless timer-driven flash/marker generator.

Unlike simulate_lsl_outlet.py (which replays a precomputed schedule of
markers -- a real recorded session's own flash timing), this generates
a live sequence: rows 1-6 and columns 7-12, freshly shuffled every
round, one LSL marker pushed per flash at real wall-clock time as it's
decided. No rendering, no EEG, no LiveDecoder -- a headless loop's
timing is artificially clean; real screen-flash-to-marker latency is a
separate, later, on-a-real-display check.

`round_number` is tracked as local state for pacing/printing only -- it
is never pushed onto the wire. `is_target` on the wire is binary 0/1
(not the offline 3-state target_flag), matching simulate_lsl_outlet.py's
own convention -- this generator always knows its own designated target,
so it never sends streaming.TARGET_UNKNOWN.

Validate with scripts/simulate_live/consume_flash_markers.py (a lighter,
marker-only sibling to consume_lsl_stream.py -- no EEG side at all).

Usage:
    python scripts/simulate_live/simulate_flash_sequencer.py --word "HI"
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pylsl

from eyecando.decode.lm import FARWELL_DONCHIN_GRID
from eyecando.pipeline.classifier import FLASHES_PER_ROUND, SECONDS_PER_FLASH

MARKER_STREAM_NAME = "EyeCanDoMarkers"
CONTROL_STREAM_NAME = "EyeCanDoControl"


def _char_to_row_col(char: str) -> tuple[int, int]:
    """Inverse of FARWELL_DONCHIN_GRID[(row - 1) * 6 + (col - 7)] -- the
    same row-major convention curate_trial_epochs.py::_trial_character
    already uses, just run backwards: which physical row/col a caller
    should flag as the target so the flash content spells `char`.
    """
    idx = FARWELL_DONCHIN_GRID.index(char)
    row = idx // 6 + 1
    col = idx % 6 + 7
    return row, col


def _round_stim_order(rng: np.random.Generator) -> list[int]:
    """One freshly shuffled permutation of stim_ids 1-12 -- reshuffled
    every round (not a fixed cycle), matching the real Farwell-Donchin
    paradigm's per-round reshuffle."""
    return rng.permutation(np.arange(1, FLASHES_PER_ROUND + 1)).tolist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word",
        required=True,
        help='Characters to spell, e.g. "HI" -- each must be in FARWELL_DONCHIN_GRID.',
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Repetitions per character, matching curate_trial_epochs.py's real "
        "8-repetition convention.",
    )
    parser.add_argument("--isi", type=float, default=SECONDS_PER_FLASH)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the per-round shuffle order, for a reproducible run -- omit for genuine "
        "randomness every run (the default).",
    )
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Wait for a start signal on EyeCanDoControl (sent by trigger_lsl_start.py) "
        "instead of a plain wait_for_consumers().",
    )
    parser.add_argument("--trigger-timeout", type=float, default=300.0)
    parser.add_argument("--source-id", default="eyecando-sim-flash")
    return parser.parse_args()


def main() -> None:
    """Validate --word against FARWELL_DONCHIN_GRID, advertise the Markers
    LSL outlet, wait for a real consumer (or trigger, see
    --wait-for-trigger), then push one freshly shuffled round of 12
    flashes per character x --rounds, each at real wall-clock time (see
    module docstring for the anti-drift scheduling)."""
    args = _parse_args()

    invalid = sorted({char for char in args.word if char not in FARWELL_DONCHIN_GRID})
    if invalid:
        raise ValueError(
            f"--word {args.word!r} contains characters not in FARWELL_DONCHIN_GRID: {invalid} "
            f"-- valid characters are {''.join(FARWELL_DONCHIN_GRID)!r}"
        )

    rng = np.random.default_rng(args.seed)

    marker_info = pylsl.StreamInfo(
        MARKER_STREAM_NAME, "Markers", 2, pylsl.IRREGULAR_RATE, "int32", args.source_id
    )
    marker_outlet = pylsl.StreamOutlet(marker_info)
    print(f"Stream advertised: {MARKER_STREAM_NAME!r}.")

    # Confirm a real consumer has subscribed to THIS stream before ever
    # pushing a flash, regardless of --wait-for-trigger -- the separate
    # EyeCanDoControl handshake below says nothing about whether the
    # consumer's *markers* inlet has actually finished connecting yet.
    # Skipping this silently drops the first several markers (confirmed
    # directly) -- same "resolve isn't connected" gap trigger_lsl_start.py
    # describes for the control stream.
    print("Waiting for a consumer to attach to the Markers stream...")
    marker_outlet.wait_for_consumers(timeout=30.0)

    if args.wait_for_trigger:
        print(
            f"Waiting for a start trigger on {CONTROL_STREAM_NAME!r} "
            "(run scripts/trigger_lsl_start.py once your consumer is connected)..."
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
        print("Trigger received.")

    t0 = pylsl.local_clock()
    flash_index = 0
    print("streaming...")
    for char in args.word:
        target_row, target_col = _char_to_row_col(char)
        for round_number in range(args.rounds):
            for stim_id in _round_stim_order(rng):
                is_target = int(stim_id in (target_row, target_col))
                # Absolute, index-based schedule (t0 + flash_index * isi),
                # not an accumulated sleep(isi) per iteration -- the same
                # anti-drift principle simulate_lsl_outlet.py uses, so a
                # long multi-round run doesn't accumulate scheduler jitter.
                target_time = t0 + flash_index * args.isi
                sleep_for = target_time - pylsl.local_clock()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                marker_outlet.push_sample([stim_id, is_target], timestamp=target_time)
                flash_index += 1
            print(f"  {char!r} round {round_number + 1}/{args.rounds} done")

    elapsed = pylsl.local_clock() - t0
    print(
        f"done streaming ({args.word!r} spelled, {flash_index} flashes pushed, "
        f"{elapsed:.1f}s elapsed)"
    )


if __name__ == "__main__":
    main()
