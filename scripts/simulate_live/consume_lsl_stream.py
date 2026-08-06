"""Consume the LSL replay from simulate_lsl_outlet.py through the real
LSLEEGStream and report what it actually produced, compared against the
same session's real flash count -- a soft/manual integration check
(needs a real LSL connection to a running outlet process, so it isn't a
pytest).

Run scripts/simulate_live/simulate_lsl_outlet.py in one terminal first,
then run this in a second terminal while it's replaying. With
--wait-for-trigger there too, pass it here too -- see simulate_live/
README.md for why that signal is re-pushed over a window rather than
sent once, and why --get-epoch-timeout is still worth keeping generous.

Usage:
    python scripts/simulate_live/consume_lsl_stream.py --subject 1 --session 0
"""

from __future__ import annotations

import argparse

import mne
import numpy as np
import pylsl

from eyecando.ingestion.data import load_moabb_session
from eyecando.live.streaming import LSLEEGStream, iter_epochs

mne.set_log_level("WARNING")

EEG_STREAM_NAME = "EyeCanDoEEG"
MARKER_STREAM_NAME = "EyeCanDoMarkers"
CONTROL_STREAM_NAME = "EyeCanDoControl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--dataset-name", default="BNCI2014_009")
    parser.add_argument("--max-marker-lag", type=float, default=2.0)
    parser.add_argument(
        "--get-epoch-timeout",
        type=float,
        default=10.0,
        help="How long to wait for the next epoch before treating the "
        "stream as exhausted, once replay has actually started.",
    )
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Explicitly wait for the start signal on EyeCanDoControl "
        "(sent by trigger_lsl_start.py) before entering the epoch loop -- "
        "use this alongside simulate_lsl_outlet.py --wait-for-trigger.",
    )
    parser.add_argument(
        "--trigger-timeout",
        type=float,
        default=300.0,
        help="How long to wait for the trigger before giving up -- only "
        "used with --wait-for-trigger.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the same session independently for ground-truth flash count,
    connect a real LSLEEGStream to simulate_lsl_outlet.py's outlets
    (optionally waiting for its trigger first), then count every epoch
    iter_epochs() actually produces and report a MATCH/MISMATCH against
    that ground truth."""
    args = _parse_args()

    # Ground truth: how many flashes this session actually contains,
    # loaded independently of whatever simulate_lsl_outlet.py is doing.
    raw = load_moabb_session(
        subject=args.subject, session=args.session, dataset_name=args.dataset_name
    )
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    sfreq = raw.info["sfreq"]
    events = mne.find_events(raw, stim_channel="target_flag", verbose=False)
    expected_n_flashes = len(events)
    print(
        f"Ground truth for subject {args.subject}, session {args.session}: "
        f"{expected_n_flashes} flashes expected"
    )

    stream = LSLEEGStream(
        eeg_stream_name=EEG_STREAM_NAME,
        marker_stream_name=MARKER_STREAM_NAME,
        sfreq=sfreq,
        ch_names=ch_names,
        max_marker_lag=args.max_marker_lag,
        resolve_timeout=15.0,
    )
    print("Resolving streams (make sure simulate_lsl_outlet.py is running)...")
    stream.start()
    print("Resolved.")

    if args.wait_for_trigger:
        print(
            f"Waiting for the start trigger on {CONTROL_STREAM_NAME!r} "
            "(run scripts/trigger_lsl_start.py once the outlet is also waiting)..."
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

    print("Waiting for epochs...")
    received = 0
    for stim_id, is_target, epoch in iter_epochs(stream, timeout=args.get_epoch_timeout):
        received += 1
        if received <= 3 or received % 20 == 0:
            print(
                f"  epoch {received}: stim_id={stim_id}, is_target={is_target}, "
                f"shape={epoch.shape}, finite={np.all(np.isfinite(epoch))}"
            )
    print(f"No epoch within {args.get_epoch_timeout}s -- assuming the replay finished.")

    print(f"\nReceived {received} epochs; expected {expected_n_flashes}.")
    if received == expected_n_flashes:
        print("MATCH -- every flash produced exactly one epoch.")
    else:
        print(
            "MISMATCH -- check for dropped-marker warnings above (see "
            "streaming.py's max_marker_lag), or that both scripts agree on "
            "--subject/--session/--dataset-name."
        )


if __name__ == "__main__":
    main()
