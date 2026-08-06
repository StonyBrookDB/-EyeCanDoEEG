"""Replay a real MOABB P300 session as live LSL EEG + Markers outlets.

A "soft test" fixture for streaming.py's LSLEEGStream: rather than
synthetic signals or monkeypatched pylsl calls (see unit_tests/
test_live_streaming.py), this replays one real recorded session --
actual channel data, actual flash timing -- through genuine LSL
sockets, in real time. Point LSLEEGStream at the same two stream names
and its whole acquire/marker/epoch pipeline runs against ground-truth
data instead of a synthetic stand-in.

Needs a real LSL outlet<->inlet data connection between two processes
on the same machine (or LAN) -- run this locally; some sandboxed/CI
environments block the data socket even when mDNS discovery succeeds.

Usage:
    # terminal 1
    python scripts/simulate_live/simulate_lsl_outlet.py --subject 1 --session 0

    # terminal 2, once terminal 1 prints "streaming..."
    python scripts/simulate_live/consume_lsl_stream.py --subject 1 --session 0

With --wait-for-trigger, replay doesn't start on a fixed delay -- it
waits for a start signal on the EyeCanDoControl stream instead, so a
slower-to-connect consumer (e.g. run_live_session.py, which loads a
model and possibly the language model before it's ready) never misses
the first samples. See scripts/simulate_live/trigger_lsl_start.py and
scripts/simulate_live/run_live_session.py for that workflow.
"""

from __future__ import annotations

import argparse
import time

import mne
import pylsl

from eyecando.ingestion.data import load_moabb_session

mne.set_log_level("WARNING")

EEG_STREAM_NAME = "EyeCanDoEEG"
MARKER_STREAM_NAME = "EyeCanDoMarkers"
CONTROL_STREAM_NAME = "EyeCanDoControl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--dataset-name", default="BNCI2014_009")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (2.0 = twice real time, replaying faster "
        "than the session was recorded).",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=32,
        help="EEG samples per push_chunk() call -- mimics a real amplifier's "
        "polling granularity rather than dumping the whole session at once.",
    )
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Wait for a start signal on the EyeCanDoControl LSL stream "
        "(sent by trigger_lsl_start.py) instead of a fixed 3s delay.",
    )
    parser.add_argument(
        "--trigger-timeout",
        type=float,
        default=300.0,
        help="How long to wait for the trigger before giving up -- a slow "
        "consumer (e.g. run_live_session.py loading a trained model and "
        "the language model) can take a couple minutes to "
        "start up, so this defaults well above that.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested session, advertise the EEG/Markers LSL outlets,
    wait for a real consumer (or trigger, see --wait-for-trigger), then
    replay every sample/flash at real wall-clock time (see module
    docstring for the anti-drift scheduling and --speed)."""
    args = _parse_args()

    raw = load_moabb_session(
        subject=args.subject, session=args.session, dataset_name=args.dataset_name
    )
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    sfreq = raw.info["sfreq"]
    eeg_data = raw.get_data(picks=eeg_picks)  # (n_channels, n_samples), volts

    events = mne.find_events(raw, stim_channel="target_flag", verbose=False)
    flash_samples = events[:, 0]
    flash_labels = events[:, 2]  # 1 = nontarget, 2 = target

    # stim_id shares target_flag's sample indices (both embedded by
    # data.py for the same flashes) -- the 1-12 row/col code the marker
    # channel needs to route a score. The true target/nontarget label
    # rides along too, purely so a consumer can check decoding accuracy;
    # a real, unlabeled deployment would send TARGET_UNKNOWN instead.
    stim_events = mne.find_events(raw, stim_channel="stim_id", verbose=False)
    stim_ids = dict(zip(stim_events[:, 0], stim_events[:, 2]))
    is_targets = dict(zip(events[:, 0], (events[:, 2] == 2).astype(int)))

    print(
        f"Subject {args.subject}, session {args.session}: {eeg_data.shape[1]} samples "
        f"at {sfreq} Hz, {len(ch_names)} channels, {len(flash_samples)} flashes "
        f"({int((flash_labels == 2).sum())} target)"
    )

    eeg_info = pylsl.StreamInfo(
        EEG_STREAM_NAME, "EEG", len(ch_names), sfreq, "float32", "eyecando-sim-eeg"
    )
    eeg_outlet = pylsl.StreamOutlet(eeg_info)
    marker_info = pylsl.StreamInfo(
        MARKER_STREAM_NAME, "Markers", 2, pylsl.IRREGULAR_RATE, "int32", "eyecando-sim-markers"
    )
    marker_outlet = pylsl.StreamOutlet(marker_info)

    print(f"Streams advertised: {EEG_STREAM_NAME!r}, {MARKER_STREAM_NAME!r}.")
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
    else:
        # wait_for_consumers() blocks until a real inlet has subscribed,
        # not just "resolved via mDNS" -- a fixed sleep would be an
        # unverified guess, and a listener connecting after the first
        # real samples are pushed just misses them silently.
        print("Waiting for a consumer to attach to the EEG/Markers streams...")
        eeg_outlet.wait_for_consumers(timeout=30.0)
        marker_outlet.wait_for_consumers(timeout=30.0)

    t0 = pylsl.local_clock()
    chunk_samples = args.chunk_samples
    n_samples = eeg_data.shape[1]
    flash_idx = 0
    print("streaming...")
    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        chunk = eeg_data[:, start:end].T.tolist()  # (n_samples, n_channels)
        chunk_t_end = t0 + (end - 1) / sfreq
        eeg_outlet.push_chunk(chunk, timestamp=chunk_t_end)

        while flash_idx < len(flash_samples) and flash_samples[flash_idx] < end:
            sample_idx = flash_samples[flash_idx]
            marker_ts = t0 + sample_idx / sfreq
            marker_outlet.push_sample(
                [int(stim_ids[sample_idx]), int(is_targets[sample_idx])], timestamp=marker_ts
            )
            flash_idx += 1

        target_wall_time = t0 + (end - 1) / sfreq / args.speed
        sleep_for = target_wall_time - pylsl.local_clock()
        if sleep_for > 0:
            time.sleep(sleep_for)

    print(f"done streaming session ({flash_idx} markers sent)")


if __name__ == "__main__":
    main()
