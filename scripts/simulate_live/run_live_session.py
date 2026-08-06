"""Run LiveDecoder against a replayed LSL session -- an end-to-end 'live'
simulation with no real headset needed, on top of simulate_lsl_outlet.py's
existing replay fixture.

Recommended 3-terminal workflow (lets this script connect, and the model/
language-model load, before any replay data starts flowing, so nothing at
the start of the session is ever missed):

    # terminal 1 -- advertises streams, then waits for a start trigger
    python scripts/simulate_live/simulate_lsl_outlet.py --subject 1 --session 0 --wait-for-trigger

    # terminal 2 -- loads the model, connects, and waits for epochs
    python scripts/simulate_live/run_live_session.py --subject 1 --session 0

    # terminal 3 -- once terminal 2 prints "connected, waiting for epochs...",
    # kick off the actual replay
    python scripts/simulate_live/trigger_lsl_start.py

Loads the already-trained population model from models/p300_classifier.pkl
(train_offline.py's output) rather than running a real per-subject
calibration phase -- the fastest path to watching the whole live
pipeline (LSLEEGStream -> EuclideanAligner -> P300Model ->
ScoreAccumulator -> CharLM -> should_decode, via LiveDecoder) decode
something end to end, not a real deployment protocol. EuclideanAligner
still needs something to fit its whitening matrix on before transform()
works, so `_fit_aligner_from_session()` fits it on the same session
about to be decoded -- a demo-only shortcut; a real deployment fits
from a dedicated calibration block collected before live decoding starts.
"""

from __future__ import annotations

import argparse

import mne

from eyecando.decode.lm import CharLM
from eyecando.ingestion.data import load_moabb_session
from eyecando.live.decoder import LiveDecoder
from eyecando.live.streaming import LSLEEGStream
from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.bandpass import apply_filter, build_sos
from eyecando.pipeline.model import P300Model

mne.set_log_level("WARNING")

EEG_STREAM_NAME = "EyeCanDoEEG"
MARKER_STREAM_NAME = "EyeCanDoMarkers"
DEFAULT_MODEL_PATH = "models/p300_classifier.pkl"


def _fit_aligner_from_session(
    subject: int, session: int, dataset_name: str
) -> tuple[EuclideanAligner, list[str], float]:
    """Fit an EuclideanAligner on this session's own filtered epochs.

    Mirrors preprocess_p300()'s filter+epoch extraction (same
    build_sos()/apply_filter(), same default tmin/tmax/baseline/reject)
    but stops short of applying EA -- preprocess_p300() discards the
    fitted aligner object, which a live session needs to keep calling
    transform() with on each new epoch.

    Also returns this session's ch_names/sfreq (needed to construct
    LSLEEGStream), so main() doesn't reload the session a second time
    just for those -- doubling this script's already-slow MOABB/mne
    startup matters here, since --wait-for-trigger has a fixed budget
    for a consumer to get ready.
    """
    raw = load_moabb_session(subject=subject, session=session, dataset_name=dataset_name)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    sfreq = raw.info["sfreq"]
    sos = build_sos(l_freq=0.1, h_freq=20.0, sfreq=sfreq)
    raw._data[eeg_picks], _ = apply_filter(sos, raw.get_data(picks=eeg_picks))

    events = mne.find_events(raw, stim_channel="target_flag", verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"nontarget": 1, "target": 2},
        tmin=-0.1,
        tmax=0.8,
        baseline=(None, 0),
        reject={"eeg": 100e-6},
        picks="eeg",
        preload=True,
        verbose="WARNING",
    )
    X = epochs.get_data(picks="eeg")
    aligner = EuclideanAligner().fit(X)
    return aligner, ch_names, sfreq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--dataset-name", default="BNCI2014_009")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lm-temperature", type=float, default=1.0)
    parser.add_argument("--no-lm", action="store_true", help="Disable the CharLM prior (lm=None).")
    parser.add_argument("--max-marker-lag", type=float, default=2.0)
    parser.add_argument(
        "--get-epoch-timeout",
        type=float,
        default=120.0,
        help="How long to wait for the next epoch before treating the stream "
        "as exhausted (LiveDecoder.run()'s exit condition). Needs to be well "
        "above whatever gap there is between this script connecting and "
        "simulate_lsl_outlet.py's --wait-for-trigger actually receiving its "
        "trigger, or this script gives up before the trigger ever fires -- "
        "not a sign of anything broken, this class's default (5s) is tuned "
        "for an already-flowing stream instead, not this workflow's startup gap.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the trained model and (optionally) CharLM, fit an
    EuclideanAligner from --subject/--session (see
    _fit_aligner_from_session), connect a real LSLEEGStream, then run
    LiveDecoder end to end and print each decoded character as it
    arrives (see module docstring for the recommended 3-terminal
    workflow)."""
    args = _parse_args()

    print(f"Loading trained model from {args.model_path}...")
    model = P300Model.load(args.model_path)

    print(f"Fitting EuclideanAligner from subject {args.subject}, session {args.session}...")
    aligner, ch_names, sfreq = _fit_aligner_from_session(
        args.subject, args.session, args.dataset_name
    )

    lm = None
    if not args.no_lm:
        print("Loading CharLM (figmtu/opt-350m-aac)...")
        lm = CharLM()

    stream = LSLEEGStream(
        eeg_stream_name=EEG_STREAM_NAME,
        marker_stream_name=MARKER_STREAM_NAME,
        sfreq=sfreq,
        ch_names=ch_names,
        max_marker_lag=args.max_marker_lag,
        resolve_timeout=60.0,
    )
    decoder = LiveDecoder(
        stream,
        model,
        aligner,
        threshold=args.threshold,
        temperature=args.temperature,
        lm=lm,
        lm_temperature=args.lm_temperature,
        get_epoch_timeout=args.get_epoch_timeout,
        # +1 includes the marker's own sample, matching mne's
        # inclusive-of-0 baseline=(None, 0) convention -- see
        # LiveDecoder's own docstring for why this is derived from the
        # stream's n_pre rather than re-computed from tmin/sfreq here.
        n_baseline_samples=stream.n_pre + 1,
    )

    print("Resolving LSL streams (make sure simulate_lsl_outlet.py is running)...")
    stream.start()
    print(
        "connected, waiting for epochs... "
        "(run scripts/simulate_live/trigger_lsl_start.py now if you haven't)"
    )

    decoded_text = ""
    for symbol_idx in decoder.run():
        char = decoder.lm.character_set[symbol_idx] if decoder.lm is not None else str(symbol_idx)
        decoded_text += char
        print(f"\rDecoded so far: {decoded_text}", end="", flush=True)

    print(f"\nDone. Final decoded text: {decoded_text!r}")


if __name__ == "__main__":
    main()
