"""Live inference orchestrator.

Wires LSLEEGStream (streaming.py), EuclideanAligner (alignment.py),
P300Model (model.py), ScoreAccumulator (accumulator.py), CharLM (lm.py),
and should_decode (stopping.py) into one running decode loop. Pure glue
-- no new algorithmic logic of its own. Every piece it calls already has
its own tests, so this module's tests focus on the wiring itself (call
order, per-symbol routing, decode/reset timing, context tracking), not
on re-verifying the algorithms underneath.

`lm` is optional -- passing None keeps should_decode()'s original
no-prior behavior. When set, LiveDecoder is what tracks `self.context`
(the message decoded so far this session) across calls, since
should_decode() itself is stateless and only reads whatever context
it's given.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol

import numpy as np

from eyecando.decode.accumulator import ScoreAccumulator
from eyecando.decode.lm import CharLM
from eyecando.decode.stopping import should_decode
from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.model import P300Model


class _EpochStream(Protocol):
    """Structural contract this module actually needs from a stream --
    matches LSLEEGStream, satisfied by any fake with the same shape."""

    def get_epoch(self, timeout: float = ...) -> tuple[int, int | None, np.ndarray] | None:
        """Return the next (stim_id, is_target, epoch), or None on timeout.

        Parameters
        ----------
        timeout : float
            Seconds to wait before giving up and returning None.

        Returns
        -------
        tuple[int, int | None, numpy.ndarray] or None
            ``stim_id`` is the 1-12 row/column code (data.py's
            convention). ``is_target`` is 0/1 when the true label is
            known, or None for a real, unlabeled deployment. ``epoch`` is
            ``(1, n_channels, n_timepoints)``. None means no epoch arrived
            within `timeout` -- the normal "stream ended" signal `_iter_epochs`
            treats as exhaustion, not a real error.
        """
        ...

    def stop(self) -> None:
        """Release whatever this stream holds open (a connection, a thread).

        Always called exactly once when iteration over this stream ends,
        however it ends (see `_iter_epochs`) -- implementations that hold
        no real resource (e.g. `PrebuiltEpochStream`) can make this a
        no-op.
        """
        ...


class PrebuiltEpochStream:
    """_EpochStream backed by an in-memory list of already-curated epochs
    (e.g. scripts/curate_trial_epochs.py's output, or a session assembled
    by scripts/build_curated_word.py) -- no live LSL connection needed.
    get_epoch() just pops the next (stim_id, is_target, epoch) tuple off
    the list, returning None once exhausted -- run()'s normal "stream
    ended" exit condition, same as a real LSLEEGStream timing out. stop()
    is a no-op since there's no real connection to tear down. Returns
    immediately rather than simulating any inter-epoch pacing --
    sufficient for correctness testing and for replaying a curated
    session; only a dedicated timing test would need paced delivery.
    """

    def __init__(self, epochs: Sequence[tuple[int, int | None, np.ndarray]]) -> None:
        """Store the epochs to replay, in order.

        Parameters
        ----------
        epochs : Sequence[tuple[int, int | None, numpy.ndarray]]
            ``(stim_id, is_target, epoch)`` tuples, copied into an
            internal list and popped from the front by get_epoch(), in
            the order given.
        """
        self._epochs = list(epochs)

    def get_epoch(self, timeout: float = 5.0) -> tuple[int, int | None, np.ndarray] | None:
        """Pop and return the next queued epoch.

        Parameters
        ----------
        timeout : float
            Accepted for interface compatibility with a real
            `_EpochStream`; unused -- popping from an in-memory list never
            actually waits.

        Returns
        -------
        tuple[int, int | None, numpy.ndarray] or None
            The next `(stim_id, is_target, epoch)` this instance was
            constructed with, in order, or None once the list is
            exhausted.
        """
        if self._epochs:
            return self._epochs.pop(0)
        return None

    def stop(self) -> None:
        """No-op -- there is no real connection to tear down."""
        pass


def _iter_epochs(
    stream: _EpochStream, timeout: float
) -> Iterator[tuple[int, int | None, np.ndarray]]:
    # Mirrors streaming.iter_epochs()'s guarantee (stop() always runs,
    # whatever ends iteration) without requiring `stream` to literally be
    # an LSLEEGStream -- LiveDecoder's own tests drive it with a fake that
    # only implements get_epoch()/stop().
    try:
        while True:
            result = stream.get_epoch(timeout=timeout)
            if result is None:
                return
            yield result
    finally:
        stream.stop()


class LiveDecoder:
    """Turns a running epoch stream into a stream of decoded symbols.

    Each epoch is whitened by the aligner's *current* state (built from
    every epoch seen so far this session -- see
    EuclideanAligner.update()) before scoring, and only folded into the
    aligner's running reference afterward -- so alignment keeps
    improving as the session runs without an epoch ever informing its
    own normalization. EA is unsupervised, so every epoch (target or
    non-target) is valid signal for this, not just a dedicated
    calibration subset.

    `model` is expected to already be fit/calibrated when passed in.
    `aligner` may be either already fit (today's usage -- a real session's
    calibration phase happens before live decoding starts, not inside this
    class) or unfit, in which case `run()` starts in
    `"calibrating"` mode: it buffers raw epochs (skipping transform/score/
    decode entirely -- an unfit EuclideanAligner would raise on
    transform() anyway, see alignment.py) until `n_calibration_epochs` have
    arrived, fits `aligner` once on the whole buffer, then switches to
    `"inferring"` starting with the *next* epoch. Calling transform() on
    an unfit EuclideanAligner outside of this class still raises
    immediately, by design -- that's a real precondition this class
    itself now satisfies internally during calibration, not something a
    caller needs to work around.

    Mode is auto-detected from `aligner.whitening_` at construction time:
    `None` means unfit (`EuclideanAligner`'s own state before
    `fit()`/`update()`) -> `"calibrating"`. Anything else, *including a
    duck-typed aligner that doesn't define `whitening_` at all* (e.g. a
    test fake), defaults to `"inferring"` -- preserving every existing
    caller's behavior exactly; only an unfit real
    `EuclideanAligner` opts into calibration mode, never a fake that
    simply doesn't expose fit-state.
    """

    def __init__(
        self,
        stream: _EpochStream,
        model: P300Model,
        aligner: EuclideanAligner,
        n_rows: int = 6,
        n_cols: int = 6,
        threshold: float = 0.9,
        temperature: float = 2.0,
        lm: CharLM | None = None,
        lm_temperature: float = 1.0,
        get_epoch_timeout: float = 5.0,
        n_calibration_epochs: int = 96,
        n_baseline_samples: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        stream : _EpochStream
            Source of epochs to decode -- a real `LSLEEGStream` or any
            duck-typed fake (e.g. `PrebuiltEpochStream`) implementing
            `get_epoch()`/`stop()`.
        model : P300Model
            Already fit/calibrated classifier; scores each aligned epoch
            via `decision_scores()`.
        aligner : EuclideanAligner
            Either already fit (today's usage -- calibration happens
            before live decoding starts) or unfit, in which
            case `run()` starts in `"calibrating"` mode (see class
            docstring).
        n_rows : int
            Speller grid row count, passed through to `ScoreAccumulator`.
        n_cols : int
            Speller grid column count, passed through to
            `ScoreAccumulator`. Also fixes the expected `lm.character_set`
            length, if `lm` is given.
        threshold : float
            Passed through to `should_decode()` -- the posterior
            confidence a symbol must reach before `run()` yields it.
        temperature : float
            Passed through to `should_decode()` -- softmax temperature
            applied to accumulated scores.
        lm : CharLM or None
            Optional character language model providing a prior to
            `should_decode()`. None (the default) keeps `should_decode()`'s
            original no-prior behavior. When set, `LiveDecoder` (not
            `should_decode()`, which is stateless) tracks `self.context`,
            the message decoded so far this session, across calls.
        lm_temperature : float
            Passed through to `should_decode()` -- softmax temperature
            applied to the LM's prior.
        get_epoch_timeout : float
            Seconds `run()` lets `stream.get_epoch()` go without producing
            an epoch before treating the stream as exhausted.
        n_calibration_epochs : int
            Number of raw epochs `run()` buffers in `"calibrating"` mode
            before fitting `aligner` once on the whole buffer and
            switching to `"inferring"`. Unused if `aligner` is already fit
            at construction time.
        n_baseline_samples : int
            How many leading samples of each incoming epoch make up its
            pre-stimulus baseline window (matching preprocess_p300()'s
            baseline=(None, 0) -- the offline path subtracts this
            window's mean from the whole epoch via mne.Epochs before
            EuclideanAligner ever sees it; the live path has no
            equivalent unless a caller supplies this). 0 (the default)
            means no baseline correction -- the pre-existing behavior,
            preserved so this isn't a silent behavior change for any
            caller that doesn't explicitly opt in.

            Use `stream.n_pre + 1` for a real LSLEEGStream (the `+ 1`
            includes the marker's own sample, matching mne's
            inclusive-of-0 baseline convention) -- computed from the same
            tmin/tmax/sfreq the stream itself used to fix its epoch
            window, rather than re-deriving round(-tmin * sfreq) a second
            time here and risking drift between the two.

        Raises
        ------
        ValueError
            If `lm` is given and `len(lm.character_set)` doesn't match
            `n_rows * n_cols` -- the LM's symbol set must line up
            one-to-one with the accumulator grid.
        """
        if lm is not None and len(lm.character_set) != n_rows * n_cols:
            raise ValueError(
                f"lm.character_set has {len(lm.character_set)} symbols, expected "
                f"{n_rows * n_cols} ({n_rows}x{n_cols}) to match the accumulator grid"
            )
        self.stream = stream
        self.model = model
        self.aligner = aligner
        self.accumulator = ScoreAccumulator(n_rows, n_cols)
        self.threshold = threshold
        self.temperature = temperature
        self.lm = lm
        self.lm_temperature = lm_temperature
        self.context = ""
        self.get_epoch_timeout = get_epoch_timeout
        self.n_baseline_samples = n_baseline_samples

        self.n_calibration_epochs = n_calibration_epochs
        self._calibration_epochs: list[tuple[int, int | None, np.ndarray]] = []
        whitening = getattr(aligner, "whitening_", "not_applicable")
        self.mode: Literal["calibrating", "inferring"] = (
            "calibrating" if whitening is None else "inferring"
        )

    def reset_message(self) -> None:
        """Start a new message: clear the tracked context and the LM's cache.

        Call this at the start of each new spelled message/sentence --
        without it, the next decode's context would keep extending
        whatever was typed in the previous message.
        """
        self.context = ""
        if self.lm is not None:
            self.lm.reset_context()

    def score_epoch(self, stim_id: int, epoch: np.ndarray) -> float:
        """Baseline-correct, whiten, score, accumulate, and fold one epoch
        into the aligner.

        Exists separately from run() so the per-epoch wiring (baseline
        correction, transform-then-update ordering, accumulator routing)
        can be tested directly without needing a real or fake stream
        driving a generator.

        Baseline correction (subtracting the first `n_baseline_samples`
        samples' mean from the whole epoch) runs before the aligner sees
        it, same as the offline path -- not because EuclideanAligner's own
        covariance needs a zero-mean input to be well-defined (it uses He
        & Wu's uncentered X X^T directly, matching the Riemannian-BCI
        literature's own convention -- see alignment.py's module
        docstring), but because removing each epoch's own DC offset is
        good preprocessing practice independent of that.
        """
        if self.n_baseline_samples > 0:
            baseline = epoch[..., : self.n_baseline_samples].mean(axis=-1, keepdims=True)
            epoch = epoch - baseline
        epoch_ea = self.aligner.transform(epoch)
        self.aligner.update(epoch)
        score = float(self.model.decision_scores(epoch_ea)[0])
        self.accumulator.push(stim_id, score)
        return score

    def run(self) -> Iterator[int]:
        """Yield the decoded symbol index each time should_decode() fires.

        Runs until the stream stops producing epochs; stream.stop() runs
        automatically when that happens, when the caller breaks out of
        the loop early, or if an exception propagates through (see
        _iter_epochs()).

        While `self.mode == "calibrating"`, epochs are buffered instead of
        scored/decoded (see class docstring) -- transform()/should_decode()
        never run against an unfit aligner. Once the buffer reaches
        `n_calibration_epochs`, `aligner.fit()` runs once and every epoch
        from that point on takes the normal `"inferring"` path below,
        starting with the very next epoch (the calibration epochs
        themselves are never also scored).

        is_target (0/1 when the source knows the true label, None for a
        real deployment) is discarded in the inferring path on purpose --
        should_decode() must stay label-agnostic, since a real session
        never has it. It's kept in the calibration buffer, unused for now
        -- free groundwork for a later adapt_to_subject()/decode-
        correctness step, without needing to revisit this method again.

        Yields
        ------
        int
            `symbol_idx` each time `should_decode()` fires -- the index
            into the accumulator's grid (and, if `lm` is set, into
            `lm.character_set`) of the symbol just decoded.
        """
        for stim_id, is_target, epoch in _iter_epochs(self.stream, self.get_epoch_timeout):
            if self.mode == "calibrating":
                self._calibration_epochs.append((stim_id, is_target, epoch))
                if len(self._calibration_epochs) >= self.n_calibration_epochs:
                    X = np.concatenate([e for _, _, e in self._calibration_epochs], axis=0)
                    self.aligner.fit(X)
                    self.mode = "inferring"
                continue

            self.score_epoch(stim_id, epoch)
            decode, symbol_idx = should_decode(
                self.accumulator,
                self.threshold,
                lm=self.lm,
                context=self.context,
                temperature=self.temperature,
                lm_temperature=self.lm_temperature,
            )
            if decode:
                if self.lm is not None:
                    self.context += self.lm.character_set[symbol_idx]
                yield symbol_idx
