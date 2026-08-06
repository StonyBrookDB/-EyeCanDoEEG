"""Tests for eyecando.live.streaming: EEGRingBuffer, LSLEEGStream, and
iter_epochs.

EEGRingBuffer coverage: push/get_window slicing, returning None while a
window is still arriving vs. raising once its start has been evicted,
wraparound correctness, and get_window_by_samples's fixed-shape,
nearest-sample-centered contract regardless of sub-sample jitter.

LSLEEGStream coverage: pylsl's resolve/inlet calls are faked (no real
socket in this sandbox) while everything downstream -- acquisition
threading, ring buffer writes, epoch windowing, get_epoch()'s queue --
runs for real. Tests check correct epoch shape/labeling (including the
TARGET_UNKNOWN sentinel decoding to None), the flattened-single-sample
reshape edge case, that the marker-reader thread keeps draining while
epoch building is slow, dropping a too-late marker with a warning
without killing the stream, buffer_seconds validation/derivation from
tmin/tmax/max_marker_lag, and start() raising when a named stream can't
be resolved.

iter_epochs coverage: yields every epoch and always calls stream.stop()
on exhaustion or early break.
"""

import warnings

import numpy as np
import pytest

from eyecando.live.streaming import TARGET_UNKNOWN, EEGRingBuffer, LSLEEGStream, iter_epochs

# --- EEGRingBuffer -----------------------------------------------------


def test_get_window_returns_none_before_any_push() -> None:
    buf = EEGRingBuffer(sfreq=256.0, n_channels=2, buffer_seconds=1.0)
    assert buf.get_window(0.0, 0.5) is None


def test_push_then_get_window_returns_correct_slice() -> None:
    sfreq = 256.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=2, buffer_seconds=2.0)
    n = 512  # 2 seconds
    t = np.arange(n) / sfreq
    samples = np.stack([np.sin(t), np.cos(t)])
    buf.push(samples, t)

    window = buf.get_window(0.5, 1.0)
    assert window is not None
    expected_n = int(np.searchsorted(t, 1.0)) - int(np.searchsorted(t, 0.5))
    assert window.shape == (2, expected_n)
    np.testing.assert_allclose(window[0], np.sin(t[t >= 0.5][: window.shape[1]]))


def test_get_window_returns_none_when_end_is_in_the_future() -> None:
    sfreq = 256.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=2.0)
    t = np.arange(100) / sfreq
    buf.push(np.zeros((1, 100)), t)
    assert buf.get_window(0.0, 5.0) is None


def test_get_window_raises_when_start_predates_evicted_data() -> None:
    sfreq = 256.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=1.0)
    # Push 3 seconds of data into a 1-second buffer, in multiple chunks so
    # wraparound is exercised, not just a single oversized push.
    for start in range(0, 3 * int(sfreq), 100):
        n = min(100, 3 * int(sfreq) - start)
        t = (start + np.arange(n)) / sfreq
        buf.push(np.zeros((1, n)), t)

    with pytest.raises(ValueError):
        buf.get_window(0.0, 0.5)


def test_wraparound_preserves_chronological_order_and_values() -> None:
    sfreq = 100.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=1.0)  # capacity 100
    # Push 250 samples in chunks of 40 -- forces multiple wraps around a
    # 100-sample capacity buffer.
    total = 250
    values = np.arange(total, dtype=np.float64)
    t_all = np.arange(total) / sfreq
    for start in range(0, total, 40):
        end = min(start + 40, total)
        buf.push(values[np.newaxis, start:end], t_all[start:end])
    # get_window()'s end is exclusive and treats a query landing exactly on
    # the newest sample as "might still be arriving" (see its docstring) --
    # push one more sample so the true tail of interest (values[-50:]) is
    # safely in the past relative to the buffer's newest timestamp.
    buf.push(np.array([[9999.0]]), np.array([t_all[-1] + 1.0 / sfreq]))

    # Only the most recent ~1s (100 samples) should still be retrievable.
    window = buf.get_window(t_all[-50], t_all[-1] + 1e-9)
    assert window is not None
    np.testing.assert_allclose(window[0], values[-50:])


def test_push_ignores_empty_chunk() -> None:
    buf = EEGRingBuffer(sfreq=256.0, n_channels=1, buffer_seconds=1.0)
    buf.push(np.zeros((1, 0)), np.zeros(0))
    assert buf.get_window(0.0, 0.1) is None


def test_get_window_by_samples_returns_fixed_shape_regardless_of_offset() -> None:
    """The whole point: for fixed n_pre/n_post, the returned shape must be
    identical no matter where center_ts falls relative to the sample
    grid -- unlike get_window(), which can drift by a sample depending on
    exactly where t_start/t_end land."""
    sfreq = 100.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=2, buffer_seconds=2.0)
    n = 200
    t = np.arange(n) / sfreq
    buf.push(np.stack([t, -t]), t)

    n_pre, n_post = 10, 20
    for offset in (0.0, 0.001, 0.004, 0.006, 0.009):  # sub-sample jitter
        window = buf.get_window_by_samples(1.0 + offset, n_pre, n_post)
        assert window is not None
        assert window.shape == (2, n_pre + n_post + 1)


def test_get_window_by_samples_centers_on_nearest_sample() -> None:
    sfreq = 100.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=2.0)
    n = 200
    t = np.arange(n) / sfreq
    buf.push(t[np.newaxis, :], t)  # channel value == its own timestamp

    # 1.004 is nearer to sample 100 (t=1.00) than sample 101 (t=1.01).
    window = buf.get_window_by_samples(1.004, n_pre=2, n_post=2)
    assert window is not None
    np.testing.assert_allclose(window[0], [0.98, 0.99, 1.00, 1.01, 1.02])


def test_get_window_by_samples_returns_none_when_post_samples_not_yet_arrived() -> None:
    sfreq = 100.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=2.0)
    t = np.arange(100) / sfreq  # up to t=0.99
    buf.push(np.zeros((1, 100)), t)
    assert buf.get_window_by_samples(0.95, n_pre=2, n_post=10) is None


def test_get_window_by_samples_raises_when_pre_samples_evicted() -> None:
    sfreq = 100.0
    buf = EEGRingBuffer(sfreq=sfreq, n_channels=1, buffer_seconds=1.0)  # capacity 100
    for start in range(0, 300, 50):
        t = (start + np.arange(50)) / sfreq
        buf.push(np.zeros((1, 50)), t)
    # Retained tail is roughly t in [2.0, 3.0) -- 2.05 exists, but 200
    # samples of lookback from there reaches back to t < 2.0, long evicted.
    with pytest.raises(ValueError):
        buf.get_window_by_samples(2.05, n_pre=200, n_post=1)


# --- LSLEEGStream --------------------------------------------------------
#
# Real LSL data transfer needs a live socket connection between an outlet
# and inlet, which this sandbox's network policy doesn't allow end-to-end
# (stream discovery succeeds, the data socket does not) -- so these tests
# swap pylsl's inlet/resolve calls for in-memory fakes that satisfy the
# same pull_chunk/close_stream contract. Everything downstream of that --
# threading, continuous-filter zi state, ring buffer writes, epoch
# windowing, the get_epoch() queue -- is the real implementation, exercised
# end to end.


class _FakeInlet:
    """Replays one canned chunk, then empty chunks forever (like a live
    inlet that has no new data yet, not a real end-of-stream)."""

    def __init__(self, samples: list[list[float]], timestamps: list[float]) -> None:
        self._samples = samples
        self._timestamps = timestamps
        self._served = False

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024):
        if not self._served:
            self._served = True
            return self._samples, self._timestamps
        return [], []

    def close_stream(self) -> None:
        pass


def _patch_lsl(monkeypatch, eeg_inlet: _FakeInlet, marker_inlet: _FakeInlet) -> None:
    import eyecando.live.streaming as streaming_mod

    monkeypatch.setattr(
        streaming_mod.pylsl, "resolve_byprop", lambda prop, value, timeout=30.0: [object()]
    )
    inlets = {"EEG": eeg_inlet, "Markers": marker_inlet}
    call_order = iter(["EEG", "Markers"])
    monkeypatch.setattr(streaming_mod.pylsl, "StreamInlet", lambda info: inlets[next(call_order)])


def test_get_epoch_returns_correctly_shaped_epoch(monkeypatch) -> None:
    sfreq = 256.0
    tmin, tmax = -0.1, 0.8
    n_channels = 2

    n_eeg_samples = int(3 * sfreq)
    t = np.arange(n_eeg_samples) / sfreq
    eeg_samples = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1).tolist()
    eeg_inlet = _FakeInlet(eeg_samples, t.tolist())

    marker_time = 1.5  # comfortably inside [0, 3) with margin for tmin/tmax
    marker_inlet = _FakeInlet([[3, 1]], [marker_time])  # stim_id 3, target

    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=sfreq,
        ch_names=["Cz", "Pz"],
        tmin=tmin,
        tmax=tmax,
    )
    stream.start()
    try:
        result = stream.get_epoch(timeout=5.0)
    finally:
        stream.stop()

    # n_pre + n_post + 1, matching LSLEEGStream's own fixed sample counts --
    # NOT round((tmax - tmin) * sfreq) + 1, which can differ by a sample
    # from summing the two rounded halves separately (the whole point of
    # fixing n_pre/n_post up front instead of a single duration).
    expected_n_timepoints = int(round(-tmin * sfreq)) + int(round(tmax * sfreq)) + 1
    assert result is not None
    stim_id, is_target, epoch = result
    assert stim_id == 3
    assert is_target == 1
    assert epoch.shape == (1, n_channels, expected_n_timepoints)


def test_get_epoch_decodes_target_unknown_sentinel_as_none(monkeypatch) -> None:
    """Every marker sample on this stream is a real flash by construction
    (the stimulus only ever sends one with a flash, never in between) --
    there's no zero-filtering to do. is_target's TARGET_UNKNOWN sentinel
    (a real, unlabeled deployment) must come back as None, not the raw
    sentinel int, so callers can `if is_target is not None` rather than
    knowing the wire encoding."""
    sfreq = 100.0
    tmin, tmax = -0.1, 0.8

    n_eeg_samples = 300
    t = np.arange(n_eeg_samples) / sfreq
    eeg_samples = np.stack([t, t], axis=1).tolist()
    eeg_inlet = _FakeInlet(eeg_samples, t.tolist())

    marker_time = 1.5
    marker_inlet = _FakeInlet([[3, TARGET_UNKNOWN]], [marker_time])

    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=sfreq,
        ch_names=["Cz", "Pz"],
        tmin=tmin,
        tmax=tmax,
    )
    stream.start()
    try:
        result = stream.get_epoch(timeout=5.0)
    finally:
        stream.stop()

    assert result is not None
    stim_id, is_target, _epoch = result
    assert stim_id == 3
    assert is_target is None


def test_acquire_loop_reshapes_a_flattened_single_sample_chunk(monkeypatch) -> None:
    """pull_chunk() always documents samples as nested per-sample lists,
    but a single-sample chunk is exactly the shape a flat (unnested) list
    would also satisfy -- if _acquire_loop ever trusted `.T` on whatever
    came back instead of anchoring on the known channel count, a flat
    [c0, c1] chunk would silently transpose into the wrong shape instead
    of (n_channels, 1). Feed it one deliberately flattened sample and
    check the buffer directly (no marker/epoch round trip needed) to
    confirm it landed as one sample across 2 channels, not the reverse."""
    eeg_inlet = _FakeInlet([1.0, 2.0], [0.0])  # flattened: one sample, 2 channels
    marker_inlet = _FakeInlet([], [])  # no marker needed for this check
    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=100.0,
        ch_names=["Cz", "Pz"],
    )
    stream.start()
    try:
        for _ in range(100):  # poll until the acquire thread has processed the chunk
            if stream.buffer._total_written > 0:
                break
            stream._stop_event.wait(0.01)
    finally:
        stream.stop()

    assert stream.buffer._total_written == 1
    data, _ = stream.buffer._valid_view()
    # Shape is the actual point of this test -- (n_channels, n_samples),
    # not (1, 2) from a bad transpose. The values themselves aren't
    # checked against the raw [1.0, 2.0] input: this sample goes through
    # the causal bandpass filter first (see _acquire_loop), which is
    # correctly initialized to the steady-state response for a constant
    # input -- and a bandpass filter's steady state for any constant
    # (0 Hz) signal is 0, by design, regardless of the constant's value.
    assert data.shape == (2, 1)
    assert np.all(np.isfinite(data))


def test_marker_reader_keeps_draining_while_epoch_builder_is_slow(monkeypatch) -> None:
    """The whole reason marker reading and epoch building are two separate
    threads: a P300 speller fires markers far faster than an epoch can
    legitimately be built (building one can block for up to `tmax`
    seconds). If the same loop did both, a burst of markers would sit
    undrained in the marker LSL inlet for as long as building blocks --
    and LSL's own inlet-side buffer is finite, so a long enough stall
    would mean *LSL itself* silently drops markers.

    Simulate "epoch building is slow" by patching _process_marker to
    sleep before doing its real work, then confirm the reader thread
    still pulls and enqueues every real marker promptly regardless --
    i.e. _pending_markers (queued + already popped by the slow builder)
    reaches the full count well before the slow builder could possibly
    have finished processing them one at a time.
    """
    import time

    sfreq = 100.0
    tmin, tmax = -0.1, 0.8

    n_eeg_samples = 400  # 4s, comfortably covers every marker window below
    t = np.arange(n_eeg_samples) / sfreq
    eeg_samples = np.stack([t, t], axis=1).tolist()
    eeg_inlet = _FakeInlet(eeg_samples, t.tolist())

    # 5 real flashes 0.1s apart -- far faster than tmax=0.8s, so building
    # epoch 1 alone would still be "in flight" when flashes 2-5 arrive.
    marker_values = [[3, 1], [4, 0], [5, 1], [6, 0], [7, 1]]
    marker_timestamps = [1.5, 1.6, 1.7, 1.8, 1.9]
    marker_inlet = _FakeInlet(marker_values, marker_timestamps)
    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    popped_count = 0
    original_process_marker = LSLEEGStream._process_marker

    def slow_process_marker(self, marker_ts, stim_id, is_target):
        nonlocal popped_count
        popped_count += 1
        time.sleep(0.3)  # much slower than the 0.1s gap between markers
        return original_process_marker(self, marker_ts, stim_id, is_target)

    monkeypatch.setattr(LSLEEGStream, "_process_marker", slow_process_marker)

    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=sfreq,
        ch_names=["Cz", "Pz"],
        tmin=tmin,
        tmax=tmax,
    )
    stream.start()
    try:
        # Give the reader thread time to drain+filter the whole marker
        # chunk, but nowhere near enough time for the slow builder to have
        # gotten through all 5 (that takes >= 1.2s at 0.3s/marker).
        for _ in range(20):
            total_seen = stream._pending_markers.qsize() + popped_count
            if total_seen >= 5:
                break
            stream._stop_event.wait(0.02)
        total_seen = stream._pending_markers.qsize() + popped_count
    finally:
        stream.stop()

    assert total_seen == 5
    assert popped_count < 5  # the slow builder hadn't gotten through all of them


def test_get_epoch_times_out_when_no_marker_arrives(monkeypatch) -> None:
    eeg_inlet = _FakeInlet([[0.0, 0.0]], [0.0])
    marker_inlet = _FakeInlet([], [])
    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    stream = LSLEEGStream(
        eeg_stream_name="EEG", marker_stream_name="Markers", sfreq=256.0, ch_names=["Cz", "Pz"]
    )
    stream.start()
    try:
        epoch = stream.get_epoch(timeout=0.3)
    finally:
        stream.stop()

    assert epoch is None


def test_constructor_stores_params() -> None:
    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=256.0,
        ch_names=["Cz", "Pz"],
    )
    assert stream.eeg_stream_name == "EEG"
    assert stream.marker_stream_name == "Markers"
    assert stream.sfreq == 256.0
    assert stream.ch_names == ["Cz", "Pz"]
    assert stream.tmin == -0.1
    assert stream.tmax == 0.8


def test_buffer_seconds_defaults_to_epoch_span_plus_max_marker_lag() -> None:
    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=256.0,
        ch_names=["Cz", "Pz"],
        tmin=-0.1,
        tmax=0.8,
        max_marker_lag=2.0,
    )
    assert stream.buffer_seconds == pytest.approx(0.9 + 2.0)


def test_buffer_seconds_too_small_for_epoch_and_lag_raises() -> None:
    with pytest.raises(ValueError):
        LSLEEGStream(
            eeg_stream_name="EEG",
            marker_stream_name="Markers",
            sfreq=256.0,
            ch_names=["Cz", "Pz"],
            tmin=-0.1,
            tmax=0.8,
            max_marker_lag=2.0,
            buffer_seconds=1.0,  # needs >= 2.9
        )


def test_explicit_buffer_seconds_above_minimum_is_accepted() -> None:
    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=256.0,
        ch_names=["Cz", "Pz"],
        tmin=-0.1,
        tmax=0.8,
        max_marker_lag=2.0,
        buffer_seconds=10.0,
    )
    assert stream.buffer_seconds == 10.0


class _SequentialFakeInlet:
    """Serves one canned chunk per call, in order, then empty forever."""

    def __init__(self, chunks: list[tuple[list, list]]) -> None:
        self._chunks = list(chunks)

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024):
        if self._chunks:
            return self._chunks.pop(0)
        return [], []

    def close_stream(self) -> None:
        pass


def test_marker_lagged_past_tolerance_is_dropped_without_killing_the_stream(monkeypatch) -> None:
    """A marker delivered so late its window has already been evicted must
    not crash the marker thread -- it should be dropped (with a warning)
    and later, on-time markers must still produce epochs."""
    sfreq = 100.0
    tmin, tmax, max_marker_lag = -0.1, 0.8, 0.5  # min buffer = 1.4s = 140 samples

    n_eeg_samples = 450  # 4.5s of data, pushed as one chunk -- capacity is
    # 140 samples, so only the last 1.4s (t ~= 3.10-4.49) survives in the
    # buffer once this chunk lands.
    t = np.arange(n_eeg_samples) / sfreq
    eeg_samples = np.tile(t[:, np.newaxis], (1, 2)).tolist()
    eeg_inlet = _SequentialFakeInlet([(eeg_samples, t.tolist())])

    # First marker refers to t=0.5 -- long since evicted (retained data
    # starts at ~3.10). Second marker refers to t=3.5 -- safely inside
    # [3.10, 4.49], with room for its full [tmin, tmax) window.
    marker_inlet = _SequentialFakeInlet([([[4, 0]], [0.5]), ([[7, 1]], [3.5])])

    monkeypatch.setattr(
        "eyecando.live.streaming.pylsl.resolve_byprop", lambda prop, value, timeout=30.0: [object()]
    )
    inlets = {"EEG": eeg_inlet, "Markers": marker_inlet}
    call_order = iter(["EEG", "Markers"])
    monkeypatch.setattr(
        "eyecando.live.streaming.pylsl.StreamInlet", lambda info: inlets[next(call_order)]
    )

    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=sfreq,
        ch_names=["Cz", "Pz"],
        tmin=tmin,
        tmax=tmax,
        max_marker_lag=max_marker_lag,
    )
    # The dropped-marker warning fires from the background marker thread,
    # not the main thread -- pytest.warns() only reliably catches warnings
    # raised synchronously within its own `with` block, so record via
    # warnings.catch_warnings() across the whole start()+poll instead.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream.start()
        try:
            result = None
            for _ in range(200):  # poll until the warning + the good epoch land
                result = stream.get_epoch(timeout=0.05)
                if result is not None:
                    break
        finally:
            stream.stop()

    assert any("too late" in str(w.message) for w in caught)

    # Values aren't checked exactly here -- _acquire_loop runs everything
    # through the causal Butterworth filter before it reaches the buffer
    # (see module docstring), so a raw ramp signal doesn't survive
    # unchanged. Shape + finiteness is what this test is actually about:
    # the lagged marker was dropped (not a crash), and the later, on-time
    # marker still produced a real epoch.
    # n_pre + n_post + 1, matching LSLEEGStream's own fixed sample counts --
    # NOT round((tmax - tmin) * sfreq) + 1, which can differ by a sample
    # from summing the two rounded halves separately (the whole point of
    # fixing n_pre/n_post up front instead of a single duration).
    expected_n_timepoints = int(round(-tmin * sfreq)) + int(round(tmax * sfreq)) + 1
    assert result is not None
    stim_id, is_target, epoch = result
    assert stim_id == 7  # the "ontime" marker's stim_id, not the dropped one's
    assert is_target == 1
    assert epoch.shape == (1, 2, expected_n_timepoints)
    assert np.all(np.isfinite(epoch))


def test_start_raises_when_stream_not_found(monkeypatch) -> None:
    import eyecando.live.streaming as streaming_mod

    monkeypatch.setattr(streaming_mod.pylsl, "resolve_byprop", lambda prop, value, timeout=30.0: [])
    stream = LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=256.0,
        ch_names=["Cz", "Pz"],
        resolve_timeout=0.1,
    )
    with pytest.raises(RuntimeError):
        stream.start()


# --- iter_epochs ----------------------------------------------------------


def _stream_with_n_markers(monkeypatch, n_markers: int) -> LSLEEGStream:
    sfreq = 100.0
    tmin, tmax = -0.1, 0.8

    n_eeg_samples = 300  # 3s, covers markers spaced through [1.0, 1.0+0.1*n)
    t = np.arange(n_eeg_samples) / sfreq
    eeg_samples = np.stack([t, t], axis=1).tolist()
    eeg_inlet = _FakeInlet(eeg_samples, t.tolist())

    marker_values = [[i + 3, i % 2] for i in range(n_markers)]
    marker_timestamps = [1.0 + 0.1 * i for i in range(n_markers)]
    marker_inlet = _FakeInlet(marker_values, marker_timestamps)
    _patch_lsl(monkeypatch, eeg_inlet, marker_inlet)

    return LSLEEGStream(
        eeg_stream_name="EEG",
        marker_stream_name="Markers",
        sfreq=sfreq,
        ch_names=["Cz", "Pz"],
        tmin=tmin,
        tmax=tmax,
    )


def test_iter_epochs_yields_every_epoch_and_stops_the_stream_on_exhaustion(monkeypatch) -> None:
    stream = _stream_with_n_markers(monkeypatch, n_markers=3)
    stream.start()

    results = list(iter_epochs(stream, timeout=2.0))

    assert len(results) == 3
    assert [stim_id for stim_id, _, _ in results] == [3, 4, 5]
    assert [is_target for _, is_target, _ in results] == [0, 1, 0]
    assert all(epoch.shape[0:2] == (1, 2) for _, _, epoch in results)
    # stop() ran (via the generator's finally) without the caller calling it
    # itself -- its cleanup sets these back to None.
    assert stream._acquire_thread is None
    assert stream._eeg_inlet is None


def test_iter_epochs_stops_the_stream_even_on_early_break(monkeypatch) -> None:
    stream = _stream_with_n_markers(monkeypatch, n_markers=3)
    stream.start()

    seen = 0
    for _ in iter_epochs(stream, timeout=2.0):
        seen += 1
        break  # sends GeneratorExit into the suspended generator

    assert seen == 1
    assert stream._acquire_thread is None
    assert stream._eeg_inlet is None
