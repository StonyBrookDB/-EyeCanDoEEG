"""Send the start trigger to a simulate_lsl_outlet.py process running with
--wait-for-trigger, so replay only begins once every consumer (e.g.
run_live_session.py) is already connected and ready -- rather than racing
a fixed delay against however long the consumer takes to start up.

Usage (after simulate_lsl_outlet.py --wait-for-trigger and
run_live_session.py are both already running):
    python scripts/simulate_live/trigger_lsl_start.py
"""

from __future__ import annotations

import argparse
import time

import pylsl

CONTROL_STREAM_NAME = "EyeCanDoControl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consumer-timeout",
        type=float,
        default=30.0,
        help="How long to wait for at least one real listener to attach "
        "before giving up on ever pushing at all.",
    )
    parser.add_argument(
        "--push-duration",
        type=float,
        default=5.0,
        help="How long to keep re-pushing the trigger sample for, once at "
        "least one listener has attached. There can be more than one real "
        "listener (e.g. simulate_lsl_outlet.py's own control_inlet *and* "
        "consume_lsl_stream.py --wait-for-trigger), and pylsl has no API "
        "for a consumer *count* -- only wait_for_consumers()'s 'at least "
        "one'. Re-pushing over a window (confirmed directly: two "
        "listeners connecting 1s and 3.5s apart both received a repeated "
        "push; neither would have caught a single one-shot push sent "
        "before they connected) covers listeners that attach a little "
        "after the first one, without needing to know how many to expect.",
    )
    parser.add_argument("--push-interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Advertise the control stream, wait for at least one real listener
    to attach, then re-push the start signal over --push-duration to
    catch any other listener that attaches slightly later (see module
    docstring for why a single push isn't reliable enough)."""
    args = _parse_args()
    info = pylsl.StreamInfo(
        CONTROL_STREAM_NAME, "Markers", 1, pylsl.IRREGULAR_RATE, "int32", "eyecando-sim-control"
    )
    outlet = pylsl.StreamOutlet(info)
    print(f"Advertising {CONTROL_STREAM_NAME!r}; waiting for a real listener to attach...")
    # wait_for_consumers() blocks until pylsl confirms an inlet has
    # actually subscribed -- not just that the stream was discovered via
    # resolve_byprop(), which says nothing about whether the connection
    # itself has completed. Confirmed directly: a listener that resolves
    # and connects only after a push never receives that push, no matter
    # how long the outlet stays alive afterward -- there's no backfill.
    if not outlet.wait_for_consumers(timeout=args.consumer_timeout):
        raise RuntimeError(
            f"no listener attached to {CONTROL_STREAM_NAME!r} within "
            f"{args.consumer_timeout}s -- start simulate_lsl_outlet.py "
            "--wait-for-trigger (and any other consumer) first"
        )
    print(f"Listener attached. Re-pushing for {args.push_duration}s to catch any stragglers...")
    t0 = time.time()
    n_pushes = 0
    while time.time() - t0 < args.push_duration:
        outlet.push_sample([1])
        n_pushes += 1
        time.sleep(args.push_interval)
    print(f"Start trigger sent ({n_pushes} pushes).")


if __name__ == "__main__":
    main()
