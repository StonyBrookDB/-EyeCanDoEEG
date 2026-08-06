"""Tests for eyecando.live.decoder: LiveDecoder and PrebuiltEpochStream.

Since LiveDecoder is pure wiring over already-tested pieces (aligner,
model, accumulator, should_decode), these tests focus on the wiring
itself rather than re-verifying the underlying algorithms: score_epoch's
per-epoch flow (row/column accumulator routing, transforming with the
aligner's prior state before folding the epoch in, optional baseline
correction); run()'s decode/reset timing and stream.stop() handling,
including one end-to-end pass with a real P300Model/EuclideanAligner;
and the calibration-mode state machine (auto-detecting "calibrating"
vs. "inferring" from the aligner's fit state, buffering epochs without
scoring during calibration, fitting the aligner once on the full
buffer, then transitioning to "inferring" from the next epoch on).
PrebuiltEpochStream's own in-order pop-then-exhaust behavior is
covered directly at the end.
"""

import numpy as np
import pytest

from eyecando.live.decoder import LiveDecoder, PrebuiltEpochStream
from eyecando.pipeline.alignment import EuclideanAligner
from eyecando.pipeline.model import P300Model

N_CHANNELS = 4
N_TIMES = 20


def _epoch(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, N_CHANNELS, N_TIMES))


class _FakeAligner:
    """Identity transform, no-op update -- isolates accumulator/stopping
    wiring from EA's own math, which already has its own dedicated tests
    in test_alignment.py."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def update(self, X: np.ndarray) -> None:
        pass


class _FakeModel:
    """Returns a caller-controlled fixed decision score regardless of
    input, so tests can force should_decode's threshold predictably
    instead of depending on real LDA behavior."""

    def __init__(self, score: float) -> None:
        self.score = score

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.score)


class _FakeStream:
    def __init__(self, items: list[tuple[int, np.ndarray]]) -> None:
        # is_target is discarded by LiveDecoder.run() (see live.py), so
        # fakes here don't need to supply a real one -- None (the
        # unlabeled-deployment case) is fine for every test.
        self._items = [(stim_id, None, epoch) for stim_id, epoch in items]
        self.stop_called = False

    def get_epoch(self, timeout: float = 5.0) -> tuple[int, int | None, np.ndarray] | None:
        if self._items:
            return self._items.pop(0)
        return None

    def stop(self) -> None:
        self.stop_called = True


def test_score_epoch_pushes_score_to_the_right_row_and_column() -> None:
    decoder = LiveDecoder(
        stream=_FakeStream([]),
        model=_FakeModel(score=2.5),
        aligner=_FakeAligner(),
        n_rows=6,
        n_cols=6,
    )
    decoder.score_epoch(stim_id=2, epoch=_epoch())  # row 2 (1-indexed)
    decoder.score_epoch(stim_id=9, epoch=_epoch())  # column 9 -> col index 2

    expected = np.zeros((6, 6))
    expected[1, :] += 2.5  # row 2 -> 0-indexed row 1
    expected[:, 2] += 2.5  # column 9 -> 0-indexed col 2
    np.testing.assert_allclose(decoder.accumulator.accum, expected)


def test_score_epoch_returns_the_computed_score() -> None:
    decoder = LiveDecoder(
        stream=_FakeStream([]), model=_FakeModel(score=1.25), aligner=_FakeAligner()
    )
    score = decoder.score_epoch(stim_id=1, epoch=_epoch())
    assert score == pytest.approx(1.25)


def test_score_epoch_transforms_using_prior_aligner_state_then_updates(monkeypatch) -> None:
    """The aligner must whiten each epoch using the state built from all
    *prior* epochs, then fold this one in afterward -- never the reverse,
    which would let an epoch partially inform its own normalization."""
    aligner = EuclideanAligner()
    aligner.update(_epoch(1))  # seed some state so there's a "before" to compare
    n_epochs_before = aligner._n_epochs

    seen_n_epochs_at_transform = []
    original_transform = EuclideanAligner.transform

    def spy_transform(self: EuclideanAligner, X: np.ndarray) -> np.ndarray:
        seen_n_epochs_at_transform.append(self._n_epochs)
        return original_transform(self, X)

    monkeypatch.setattr(EuclideanAligner, "transform", spy_transform)

    decoder = LiveDecoder(stream=_FakeStream([]), model=_FakeModel(score=0.0), aligner=aligner)
    decoder.score_epoch(stim_id=3, epoch=_epoch(2))

    assert seen_n_epochs_at_transform == [n_epochs_before]
    assert aligner._n_epochs == n_epochs_before + 1


def test_score_epoch_applies_no_baseline_correction_by_default() -> None:
    """n_baseline_samples defaults to 0 -- pre-existing behavior, no
    behavior change for a caller that doesn't opt in."""
    seen = []
    aligner = _FakeAligner()
    monkeypatched_transform = aligner.transform
    aligner.transform = lambda X: (seen.append(X.copy()), monkeypatched_transform(X))[1]

    decoder = LiveDecoder(stream=_FakeStream([]), model=_FakeModel(score=0.0), aligner=aligner)
    epoch = _epoch()
    decoder.score_epoch(stim_id=1, epoch=epoch)

    np.testing.assert_allclose(seen[0], epoch)


def test_score_epoch_subtracts_baseline_mean_before_transform() -> None:
    seen = []
    aligner = _FakeAligner()
    monkeypatched_transform = aligner.transform
    aligner.transform = lambda X: (seen.append(X.copy()), monkeypatched_transform(X))[1]

    n_baseline = 5
    decoder = LiveDecoder(
        stream=_FakeStream([]),
        model=_FakeModel(score=0.0),
        aligner=aligner,
        n_baseline_samples=n_baseline,
    )
    epoch = _epoch()
    decoder.score_epoch(stim_id=1, epoch=epoch)

    expected_baseline = epoch[..., :n_baseline].mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(seen[0], epoch - expected_baseline)
    # The baseline-corrected epoch's own baseline window is now ~zero-mean.
    np.testing.assert_allclose(
        seen[0][..., :n_baseline].mean(axis=-1), np.zeros((1, N_CHANNELS)), atol=1e-10
    )


def test_run_yields_decoded_symbol_and_resets_accumulator_after_decode() -> None:
    # A large fixed score on both a row and a column push makes their
    # intersection cell overwhelmingly dominant -- softmax comfortably
    # crosses threshold=0.9 after just these two epochs.
    items = [(2, _epoch(10)), (9, _epoch(11))]  # row 2, column 9
    stream = _FakeStream(items)
    decoder = LiveDecoder(
        stream=stream,
        model=_FakeModel(score=100.0),
        aligner=_FakeAligner(),
        n_rows=6,
        n_cols=6,
        threshold=0.9,
        temperature=1.0,
    )

    decoded = list(decoder.run())

    expected_symbol_idx = 1 * 6 + 2  # row idx 1, col idx 2, flattened (n_rows, n_cols)
    assert decoded == [expected_symbol_idx]
    assert stream.stop_called
    assert np.all(decoder.accumulator.accum == 0)  # reset after the decode fired


def test_run_stops_stream_when_exhausted_without_decoding() -> None:
    stream = _FakeStream([(2, _epoch(1))])
    decoder = LiveDecoder(stream=stream, model=_FakeModel(score=0.0), aligner=_FakeAligner())

    decoded = list(decoder.run())

    assert decoded == []  # a uniform (all-zero) score never crosses threshold
    assert stream.stop_called


def test_run_with_real_model_and_aligner_end_to_end() -> None:
    """Not a wiring-logic test like the ones above -- just confirms real
    P300Model + real EuclideanAligner plug into LiveDecoder
    (shapes, dtypes) without a live LSL connection, the same way
    test_streaming.py's fakes stand in for real sockets."""
    rng = np.random.default_rng(0)
    n_epochs, n_channels, n_times = 100, 4, 20
    y = (np.arange(n_epochs) % 5 == 0).astype(int)
    X = rng.normal(size=(n_epochs, n_channels, n_times))
    bump = 4.0 * np.exp(-((np.arange(n_times) - n_times // 2) ** 2) / 50)
    X[y == 1, 0, :] += bump

    model = P300Model(n_components=2).fit([(X, y)])
    # A real session calibrates the aligner before live decoding starts --
    # LiveDecoder assumes it's handed an already-fit aligner, the same way
    # it's handed an already-fit model.
    aligner = EuclideanAligner().fit(X[:20])

    items = [(i % 12 + 1, X[i : i + 1]) for i in range(10)]
    stream = _FakeStream(items)
    decoder = LiveDecoder(stream=stream, model=model, aligner=aligner, threshold=0.9)

    decoded = list(decoder.run())

    assert stream.stop_called
    assert all(isinstance(idx, (int, np.integer)) for idx in decoded)


# --- calibration mode ------------------------------------------------------


def test_mode_defaults_to_inferring_for_a_prefit_aligner() -> None:
    aligner = EuclideanAligner().fit(_epoch(0))
    decoder = LiveDecoder(stream=_FakeStream([]), model=_FakeModel(score=0.0), aligner=aligner)
    assert decoder.mode == "inferring"


def test_mode_defaults_to_inferring_for_a_duck_typed_aligner_without_whitening_attr() -> None:
    """_FakeAligner (used throughout this file) has no whitening_ attribute
    at all -- must default to "inferring", exactly matching every test
    above that already relies on immediate scoring/decoding. Only a
    unfit real EuclideanAligner should opt into calibration."""
    decoder = LiveDecoder(
        stream=_FakeStream([]), model=_FakeModel(score=0.0), aligner=_FakeAligner()
    )
    assert decoder.mode == "inferring"


def test_mode_defaults_to_calibrating_for_an_unfit_aligner() -> None:
    aligner = EuclideanAligner()
    assert aligner.whitening_ is None
    decoder = LiveDecoder(stream=_FakeStream([]), model=_FakeModel(score=0.0), aligner=aligner)
    assert decoder.mode == "calibrating"


def test_calibration_buffers_epochs_without_scoring_or_decoding() -> None:
    aligner = EuclideanAligner()
    items = [(i % 12 + 1, None, _epoch(i)) for i in range(3)]
    stream = PrebuiltEpochStream(items)
    decoder = LiveDecoder(
        stream=stream,
        model=_FakeModel(score=100.0),  # would immediately cross threshold if ever scored
        aligner=aligner,
        n_calibration_epochs=5,  # more than the 3 epochs available -- never transitions
    )

    decoded = list(decoder.run())

    assert decoded == []  # should_decode() never ran during calibration
    assert np.all(decoder.accumulator.accum == 0)  # score_epoch() never ran either
    assert decoder.mode == "calibrating"
    assert aligner.whitening_ is None  # never reached n_calibration_epochs, never fit


def test_calibration_transitions_to_inferring_after_target_reached() -> None:
    aligner = EuclideanAligner()
    calib_items = [(i % 12 + 1, None, _epoch(i)) for i in range(4)]
    post_items = [(2, None, _epoch(100)), (9, None, _epoch(101))]  # row 2, column 9
    stream = PrebuiltEpochStream(calib_items + post_items)
    decoder = LiveDecoder(
        stream=stream,
        model=_FakeModel(score=100.0),
        aligner=aligner,
        n_rows=6,
        n_cols=6,
        threshold=0.9,
        temperature=1.0,
        n_calibration_epochs=4,
    )

    decoded = list(decoder.run())

    assert decoder.mode == "inferring"
    assert aligner.whitening_ is not None
    # Only the two post-calibration epochs (row 2, col 9) should ever have
    # been scored -- if a calibration epoch had also been scored, the
    # accumulator wouldn't reset to exactly zero after this single decode.
    expected_symbol_idx = 1 * 6 + 2
    assert decoded == [expected_symbol_idx]
    assert np.all(decoder.accumulator.accum == 0)


def test_calibration_fits_aligner_on_the_whole_buffered_batch() -> None:
    """aligner.fit() must run exactly once, on all n_calibration_epochs
    concatenated together -- not once per epoch (that would be update(),
    a different, incremental accumulation this design deliberately
    doesn't use for the initial calibration fit)."""
    aligner = EuclideanAligner()
    n_calib = 6
    items = [(i % 12 + 1, None, _epoch(i)) for i in range(n_calib)]
    stream = PrebuiltEpochStream(items)
    decoder = LiveDecoder(
        stream=stream, model=_FakeModel(score=0.0), aligner=aligner, n_calibration_epochs=n_calib
    )

    list(decoder.run())

    assert aligner._n_epochs == n_calib


def test_prebuilt_epoch_stream_pops_in_order_then_exhausts() -> None:
    items = [(1, 0, _epoch(0)), (2, 1, _epoch(1))]
    stream = PrebuiltEpochStream(items)

    first = stream.get_epoch()
    second = stream.get_epoch()
    third = stream.get_epoch()

    assert first is not None and first[0] == 1
    assert second is not None and second[0] == 2
    assert third is None
    stream.stop()  # no-op, must not raise
