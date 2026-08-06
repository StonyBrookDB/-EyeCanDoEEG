"""Live ingestion: LSL stream reader with ring buffer.

Imports bandpass.py and pylsl, not classifier.py -- the inference path is
separate from training.

NOTE -- preprocess_p300() (preprocessing.py) cannot be called per-epoch
here; see pipeline/README.md for why. This class assembles its own
pipeline out of BandpassFilter instead. EA/xDAWN/LDA still happen
downstream of get_epoch(), same as the training path.

Pipeline position of EEGRingBuffer: LSL EEG inlet -> causal Butterworth
filter (zi carried across chunks) -> EEGRingBuffer -> marker-triggered
epoch extraction. It sits after filtering and before epoching -- it never
sees raw unfiltered samples, and epoching only ever reads out of it via
get_window(), never straight off the inlet.

Retention is sized from the epoch window and a declared marker-lag
tolerance, not an arbitrary flat duration: see LSLEEGStream's
`max_marker_lag` docstring for why.
"""

from __future__ import annotations

import queue
import threading
import warnings
from collections.abc import Iterator

import numpy as np
import pylsl

from eyecando.pipeline.bandpass import BandpassFilter

# Sentinel for the marker stream's is_target value when the true label
# isn't known -- the real, unlabeled-deployment case. Not 0/1 so it can
# never be confused with a real nontarget/target label.
TARGET_UNKNOWN = -1


class EEGRingBuffer:
    """Fixed-duration rolling buffer of continuous, already-filtered EEG.

    Holds the last `buffer_seconds` of samples plus their timestamps (on
    LSL's own clock), so get_window() can pull out the tmin/tmax slice
    around a marker as soon as enough post-marker data has arrived.

    Filtering happens upstream of push() -- apply_filter() with carried
    `zi` state on each incoming chunk before it's appended here (see
    preprocessing.py's module docstring) -- this buffer only stores
    already-filtered samples, it doesn't filter them itself.

    Backed by a preallocated circular numpy array (not a deque of chunks)
    so memory is fixed and get_window() can slice/searchsorted directly
    instead of concatenating chunks on every call.
    """

    def __init__(self, sfreq: float, n_channels: int, buffer_seconds: float = 5.0) -> None:
        """Allocate the ring buffer's fixed-size backing arrays.

        Parameters
        ----------
        sfreq : float
            Sample rate (Hz) of the already-filtered EEG this buffer will
            receive via push().
        n_channels : int
            Number of EEG channels per sample.
        buffer_seconds : float
            How much history to retain; sets the buffer's fixed capacity
            (``round(sfreq * buffer_seconds)`` samples, minimum 1). Older
            samples are evicted as new ones are pushed once the buffer is
            full.
        """
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.buffer_seconds = buffer_seconds
        self._capacity = max(int(round(sfreq * buffer_seconds)), 1)
        self._data = np.zeros((n_channels, self._capacity), dtype=np.float64)
        self._timestamps = np.zeros(self._capacity, dtype=np.float64)
        self._write_idx = 0
        self._total_written = 0

    def push(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Append a new chunk of already-filtered samples.

        Parameters
        ----------
        samples : numpy.ndarray
            ``(n_channels, n_new_samples)`` already-filtered EEG. If
            ``n_new_samples`` exceeds this buffer's capacity, only the
            trailing ``capacity`` samples are kept.
        timestamps : numpy.ndarray
            ``(n_new_samples,)``, on the EEG inlet's own LSL clock, so
            get_window()/get_window_by_samples() can align directly to
            marker timestamps from the marker stream. Samples older than
            `buffer_seconds` relative to the newest pushed sample are
            evicted here (overwritten by the circular buffer) to keep
            memory bounded.
        """
        n_new = samples.shape[1]
        if n_new == 0:
            return
        if n_new >= self._capacity:
            # Chunk alone exceeds the whole buffer -- only its tail fits.
            samples = samples[:, -self._capacity :]
            timestamps = timestamps[-self._capacity :]
            n_new = samples.shape[1]

        end = self._write_idx + n_new
        if end <= self._capacity:
            self._data[:, self._write_idx : end] = samples
            self._timestamps[self._write_idx : end] = timestamps
        else:
            first_len = self._capacity - self._write_idx
            self._data[:, self._write_idx :] = samples[:, :first_len]
            self._timestamps[self._write_idx :] = timestamps[:first_len]
            wrapped_len = end - self._capacity
            self._data[:, :wrapped_len] = samples[:, first_len:]
            self._timestamps[:wrapped_len] = timestamps[first_len:]

        self._write_idx = end % self._capacity
        self._total_written += n_new

    def _valid_view(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Return (data, timestamps) in chronological order, oldest first.

        Only covers samples actually pushed so far (<= capacity); the
        unwritten tail of a not-yet-full buffer is never included.
        """
        n_valid = min(self._total_written, self._capacity)
        if n_valid == 0:
            return None, None
        if self._total_written <= self._capacity:
            return self._data[:, :n_valid], self._timestamps[:n_valid]
        # Buffer has wrapped -- oldest sample sits at _write_idx.
        data = np.concatenate(
            [self._data[:, self._write_idx :], self._data[:, : self._write_idx]], axis=1
        )
        ts = np.concatenate(
            [self._timestamps[self._write_idx :], self._timestamps[: self._write_idx]]
        )
        return data, ts

    def get_window(self, t_start: float, t_end: float) -> np.ndarray | None:
        """Return the (n_channels, n_samples) slice covering [t_start, t_end).

        Cuts by time, so the returned sample count depends on exactly
        where t_start/t_end land between samples -- it can vary by one
        sample call to call even for a fixed (t_end - t_start) duration.
        For a fixed-length epoch around a marker, use
        get_window_by_samples() instead, which fixes the sample count up
        front instead of deriving it from a time difference.

        Parameters
        ----------
        t_start : float
            Window start, on the same LSL clock as the timestamps passed
            to push().
        t_end : float
            Window end (exclusive), same clock.

        Returns
        -------
        numpy.ndarray or None
            ``(n_channels, n_samples)`` slice, or None if the buffer
            doesn't yet have enough data to cover the full window -- e.g.
            `t_end` is still in the future relative to what's been pushed
            so far, which is the normal case right after a marker arrives
            (callers should keep polling/blocking until this stops
            returning None, not treat it as a real miss).

        Raises
        ------
        ValueError
            If `t_start` predates the oldest sample still in the buffer --
            that data has already been evicted (`buffer_seconds` too
            short for this window), which is a real configuration
            problem, not a transient "not ready yet" case.
        """
        data, ts = self._valid_view()
        if ts is None:
            return None
        assert data is not None  # _valid_view() returns both None or neither
        if t_end > ts[-1]:
            return None
        if t_start < ts[0]:
            raise ValueError(
                f"requested window start {t_start} predates the oldest buffered "
                f"sample {ts[0]} -- buffer_seconds={self.buffer_seconds} is too "
                "short for this epoch window"
            )
        i0 = int(np.searchsorted(ts, t_start, side="left"))
        i1 = int(np.searchsorted(ts, t_end, side="left"))
        return data[:, i0:i1]

    def get_window_by_samples(self, center_ts: float, n_pre: int, n_post: int) -> np.ndarray | None:
        """Return a fixed-length (n_channels, n_pre + n_post + 1) slice
        centered on the sample nearest `center_ts`.

        get_window() cuts by time -- t_start/t_end land wherever they
        land between samples, so the resulting sample count can drift by
        one depending on timing jitter, even for a fixed requested
        duration. This instead resolves `center_ts` to its nearest actual
        sample once, then steps a fixed `n_pre`/`n_post` sample count
        around it -- so for a fixed n_pre/n_post (i.e. fixed tmin/tmax at
        a fixed sfreq), every call returns exactly the same shape, with
        no padding/truncation needed afterward.

        Parameters
        ----------
        center_ts : float
            Marker timestamp to center the window on, same LSL clock as
            the timestamps passed to push(). Resolved to its nearest
            actually-buffered sample before stepping n_pre/n_post.
        n_pre : int
            Number of samples to include before the center sample.
        n_post : int
            Number of samples to include after the center sample.

        Returns
        -------
        numpy.ndarray or None
            ``(n_channels, n_pre + n_post + 1)`` slice, or None if fewer
            than `n_post` samples exist after the center sample yet (the
            normal "still arriving" case -- callers should keep polling).

        Raises
        ------
        ValueError
            If fewer than `n_pre` samples exist before the center sample
            (already evicted -- `buffer_seconds` too short for this
            window).
        """
        data, ts = self._valid_view()
        if ts is None:
            return None
        assert data is not None  # _valid_view() returns both None or neither
        if center_ts > ts[-1]:
            return None
        if center_ts < ts[0]:
            raise ValueError(
                f"requested center {center_ts} predates the oldest buffered "
                f"sample {ts[0]} -- buffer_seconds={self.buffer_seconds} is too "
                "short for this epoch window"
            )

        n = len(ts)
        i_center = int(np.searchsorted(ts, center_ts, side="left"))
        if i_center > 0 and (
            i_center == n or abs(ts[i_center - 1] - center_ts) <= abs(ts[i_center] - center_ts)
        ):
            i_center -= 1

        i_start = i_center - n_pre
        i_end = i_center + n_post + 1
        if i_start < 0:
            raise ValueError(
                f"requested {n_pre} samples of lookback before {center_ts} but only "
                f"{i_center} are buffered -- buffer_seconds={self.buffer_seconds} is "
                "too short for this epoch window"
            )
        if i_end > n:
            return None
        return data[:, i_start:i_end]


class LSLEEGStream:
    """Three-thread LSL acquisition: EEG ring buffer + marker-triggered epoching.

    Produces mne.Epochs-compatible numpy arrays on demand, in the same
    format that data.py produces for training. The downstream pipeline
    (preprocessing, model) does not know whether input came from here
    or from data.py.

    Marker handling is split across two threads, not one, on purpose:
    turning a marker into an epoch can legitimately block for up to
    `tmax` seconds waiting for post-marker buffer data, and a P300
    speller fires markers many times a second. If the same loop that
    drains the marker inlet also did that blocking wait, LSL's own
    finite inlet-side buffer could fill and start silently dropping
    markers we'd never even see -- worse than the `max_marker_lag`-driven
    drop this stream already logs. So `_marker_reader_loop` only ever
    does the cheap, non-blocking part (drain, hand off via
    `_pending_markers`); `_epoch_builder_loop` is the only place that
    ever blocks waiting on buffer data.

    Every marker sample on this stream is a real flash by construction --
    unlike data.py's offline stim_id stim channel (continuous, mostly
    zero), the live marker stream only ever carries a sample when the
    stimulus actually flashes, never in between. Each sample is
    (stim_id, is_target): stim_id is the 1-12 row/column code
    (data.py's convention) needed to route a score via
    ScoreAccumulator.push(); is_target is 0/1 when the source knows the
    true label (replaying labeled offline data, e.g.
    simulate_lsl_outlet.py) or TARGET_UNKNOWN for a real, unlabeled
    deployment. Nothing in the actual decode path (should_decode(),
    LiveDecoder) reads is_target -- it exists purely so a consumer can
    check decoding accuracy against ground truth while testing.
    """

    def __init__(
        self,
        eeg_stream_name: str,
        marker_stream_name: str,
        sfreq: float,
        ch_names: list[str],
        tmin: float = -0.1,
        tmax: float = 0.8,
        max_marker_lag: float = 2.0,
        buffer_seconds: float | None = None,
        l_freq: float = 0.1,
        h_freq: float = 20.0,
        resolve_timeout: float = 30.0,
    ) -> None:
        """
        `max_marker_lag` drives how much EEG history this stream retains.
        The marker and EEG streams are two independent LSL connections
        with no shared latency guarantee, so a marker can arrive after
        the EEG samples it refers to have already streamed past the
        buffer's retention window -- retention is therefore derived, not
        a flat guessed duration: `(tmax - tmin) + max_marker_lag` seconds
        guarantees a marker's full epoch window survives even when that
        marker is delivered up to `max_marker_lag` seconds late.

        If `buffer_seconds` is left as None, it's set to exactly that
        derived minimum. An explicit value smaller than the minimum
        raises immediately at construction, not three minutes into a
        live session when the first slow marker silently loses its
        epoch.

        Parameters
        ----------
        eeg_stream_name : str
            LSL `name` to resolve the continuous EEG stream by, via
            ``pylsl.resolve_byprop`` in start().
        marker_stream_name : str
            LSL `name` to resolve the marker stream by. Every sample on
            this stream is a real flash (see class docstring): two values
            per sample, ``(stim_id, is_target)``.
        sfreq : float
            EEG sample rate (Hz). Used to size the ring buffer and to fix
            this stream's epoch sample counts (`n_pre`/`n_post`) up front.
        ch_names : list[str]
            EEG channel names, in acquisition order. Only its length is
            used (to size the buffer); order is not otherwise validated
            against the actual LSL stream.
        tmin : float
            Epoch start relative to the marker, in seconds (negative =
            before the marker).
        tmax : float
            Epoch end relative to the marker, in seconds.
        max_marker_lag : float
            Worst-case seconds a marker may be delivered behind the EEG
            samples it refers to and still have its epoch recoverable.
            Drives the derived minimum `buffer_seconds` (see above); a
            marker delivered later than this has its epoch dropped with a
            warning (see `_process_marker`).
        buffer_seconds : float or None
            Ring-buffer retention, in seconds. None (the default) uses the
            derived minimum, ``(tmax - tmin) + max_marker_lag``.
        l_freq : float
            Bandpass low cutoff (Hz) for the causal filter applied to
            incoming EEG chunks before they reach the buffer.
        h_freq : float
            Bandpass high cutoff (Hz), same filter.
        resolve_timeout : float
            Seconds start() waits for `pylsl.resolve_byprop` to find each
            named stream before giving up.

        Raises
        ------
        ValueError
            If an explicit `buffer_seconds` is smaller than the derived
            minimum needed to guarantee a well-behaved-but-late marker's
            epoch survives (see above).
        """
        epoch_span = tmax - tmin
        min_buffer_seconds = epoch_span + max_marker_lag
        if buffer_seconds is None:
            buffer_seconds = min_buffer_seconds
        elif buffer_seconds < min_buffer_seconds:
            raise ValueError(
                f"buffer_seconds={buffer_seconds} can't guarantee an epoch survives "
                f"even a well-behaved marker delay: needs >= (tmax - tmin) + "
                f"max_marker_lag = {epoch_span:.3f} + {max_marker_lag:.3f} = "
                f"{min_buffer_seconds:.3f}s. Raise buffer_seconds, lower "
                "max_marker_lag, or shrink the epoch window."
            )

        self.eeg_stream_name = eeg_stream_name
        self.marker_stream_name = marker_stream_name
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.tmin = tmin
        self.tmax = tmax
        self.max_marker_lag = max_marker_lag
        self.buffer_seconds = buffer_seconds
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resolve_timeout = resolve_timeout
        self.buffer = EEGRingBuffer(sfreq, len(ch_names), buffer_seconds)

        self._bandpass = BandpassFilter(l_freq, h_freq, sfreq)
        # Fix the epoch's sample counts once, up front, rather than
        # deriving a sample count later from a time difference -- see
        # EEGRingBuffer.get_window_by_samples() for why that matters:
        # for a fixed tmin/tmax/sfreq, n_pre/n_post/n_timepoints are then
        # fixed too, with no per-epoch padding/truncation needed.
        self._n_pre = int(round(-tmin * sfreq))
        self._n_post = int(round(tmax * sfreq))
        self._n_timepoints = self._n_pre + self._n_post + 1

        # Public so callers building a baseline-correction window (e.g.
        # LiveDecoder) can match this stream's own fixed sample layout
        # exactly, rather than recomputing round(-tmin * sfreq) a second
        # time and risking it drift out of sync with this one.
        self.n_pre = self._n_pre

        self._eeg_inlet: pylsl.StreamInlet | None = None
        self._marker_inlet: pylsl.StreamInlet | None = None
        self._acquire_thread: threading.Thread | None = None
        self._marker_reader_thread: threading.Thread | None = None
        self._epoch_builder_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pending_markers: queue.Queue[tuple[float, int, int | None]] = queue.Queue()
        self._epoch_queue: queue.Queue[tuple[int, int | None, np.ndarray]] = queue.Queue()

    def start(self) -> None:
        """Start the acquire, marker-reader, and epoch-builder threads.

        The acquire thread reads raw chunks from the EEG inlet, filters
        them (apply_filter() with zi carried across chunks), and pushes
        the filtered result into self.buffer. The marker-reader thread
        drains the marker inlet and hands off each real flash's timestamp
        (see class docstring for why it never blocks). The epoch-builder
        thread consumes those timestamps and, for each one, waits for
        self.buffer to cover that marker's window before handing an
        epoch to get_epoch()'s caller.

        Raises
        ------
        RuntimeError
            If either the EEG or the marker stream can't be resolved by
            name within `resolve_timeout` seconds.
        """
        eeg_info = pylsl.resolve_byprop("name", self.eeg_stream_name, timeout=self.resolve_timeout)
        if not eeg_info:
            raise RuntimeError(
                f"no LSL stream found with name={self.eeg_stream_name!r} "
                f"within {self.resolve_timeout}s"
            )
        marker_info = pylsl.resolve_byprop(
            "name", self.marker_stream_name, timeout=self.resolve_timeout
        )
        if not marker_info:
            raise RuntimeError(
                f"no LSL stream found with name={self.marker_stream_name!r} "
                f"within {self.resolve_timeout}s"
            )

        self._eeg_inlet = pylsl.StreamInlet(eeg_info[0])
        self._marker_inlet = pylsl.StreamInlet(marker_info[0])
        self._stop_event.clear()

        self._acquire_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._marker_reader_thread = threading.Thread(target=self._marker_reader_loop, daemon=True)
        self._epoch_builder_thread = threading.Thread(target=self._epoch_builder_loop, daemon=True)
        self._acquire_thread.start()
        self._marker_reader_thread.start()
        self._epoch_builder_thread.start()

    def stop(self) -> None:
        """Stop threads and release LSL inlets."""
        self._stop_event.set()
        if self._acquire_thread is not None:
            self._acquire_thread.join(timeout=2.0)
            self._acquire_thread = None
        if self._marker_reader_thread is not None:
            self._marker_reader_thread.join(timeout=2.0)
            self._marker_reader_thread = None
        if self._epoch_builder_thread is not None:
            self._epoch_builder_thread.join(timeout=2.0)
            self._epoch_builder_thread = None
        if self._eeg_inlet is not None:
            self._eeg_inlet.close_stream()
            self._eeg_inlet = None
        if self._marker_inlet is not None:
            self._marker_inlet.close_stream()
            self._marker_inlet = None

    def get_epoch(self, timeout: float = 2.0) -> tuple[int, int | None, np.ndarray] | None:
        """Block until the next marker-triggered epoch is ready.

        Parameters
        ----------
        timeout : float
            Seconds to wait for the next epoch before giving up and
            returning None.

        Returns
        -------
        tuple[int, int | None, numpy.ndarray] or None
            ``(stim_id, is_target, epoch)``, or None on timeout. `stim_id`
            is the 1-12 row/column code from data.py's convention -- the
            caller needs it to route the epoch's score to the right
            symbol via ``ScoreAccumulator.push(stim_id, score)``.
            `is_target` is 0/1 when the source knows the true label, or
            None (`TARGET_UNKNOWN` on the wire) for a real, unlabeled
            deployment -- see the class docstring. `epoch` is
            ``(1, n_channels, n_timepoints)``.
        """
        try:
            return self._epoch_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _acquire_loop(self) -> None:
        assert self._eeg_inlet is not None  # only started after connect() sets it
        n_channels = len(self.ch_names)
        while not self._stop_event.is_set():
            samples, timestamps = self._eeg_inlet.pull_chunk(timeout=0.1, max_samples=1024)
            if not samples:
                continue
            # pull_chunk() documents samples as a list of per-sample lists
            # (one per channel) -- but a single-sample chunk is exactly the
            # shape numpy would happily also accept as one flat row, and
            # `.T` on that gives (n_samples,) with no channel axis at all
            # rather than raising. Anchor the reshape on the known channel
            # count so a chunk always comes out (n_channels, n_samples),
            # explicitly, instead of trusting whatever shape arrived.
            chunk = np.asarray(samples, dtype=np.float64).reshape(-1, n_channels).T
            filtered = self._bandpass.apply(chunk)
            self.buffer.push(filtered, np.asarray(timestamps, dtype=np.float64))

    def _marker_reader_loop(self) -> None:
        """Pull marker samples and hand them off -- never waits on buffer
        data (that's _epoch_builder_loop's job, via _pending_markers; see
        class docstring for why the split matters). Pulls in chunks like
        _acquire_loop, since a P300 speller flashes many times a second.
        """
        assert self._marker_inlet is not None  # only started after connect() sets it
        while not self._stop_event.is_set():
            marker_samples, marker_timestamps = self._marker_inlet.pull_chunk(
                timeout=0.1, max_samples=1024
            )
            if not marker_samples:
                continue
            for marker_sample, marker_ts in zip(marker_samples, marker_timestamps):
                # int(): the marker channel's own numeric type (float32,
                # int32, or a numeric string) doesn't matter downstream --
                # ScoreAccumulator.push()/data.py's stim_id convention
                # both want a plain 1-12 int.
                stim_id = int(marker_sample[0])
                raw_is_target = int(marker_sample[1])
                is_target = None if raw_is_target == TARGET_UNKNOWN else raw_is_target
                self._pending_markers.put((marker_ts, stim_id, is_target))

    def _epoch_builder_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                marker_ts, stim_id, is_target = self._pending_markers.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self._process_marker(marker_ts, stim_id, is_target):
                return  # stop() was called while waiting for this epoch

    def _process_marker(self, marker_ts: float, stim_id: int, is_target: int | None) -> bool:
        """Turn one marker into a queued (stim_id, is_target, epoch).

        Returns False only when stop() interrupted the wait for data --
        the caller should stop processing further markers in that case.
        """
        window = None
        try:
            while not self._stop_event.is_set():
                window = self.buffer.get_window_by_samples(marker_ts, self._n_pre, self._n_post)
                if window is not None:
                    break
                self._stop_event.wait(0.01)
        except ValueError:
            # This marker was delivered more than max_marker_lag behind
            # the EEG samples it refers to -- its window is already
            # evicted. Drop this one epoch and keep going; a thread
            # crash here would silently kill every epoch after it too.
            warnings.warn(
                f"marker at {marker_ts} arrived too late to recover its epoch "
                f"(exceeded max_marker_lag={self.max_marker_lag}s) -- dropped",
                stacklevel=2,
            )
            return True
        if window is None:
            return False  # stop() was called while waiting for this epoch

        # get_window_by_samples() fixes n_pre/n_post up front, so `window`
        # is already exactly (n_channels, n_timepoints) -- no
        # padding/truncation needed, unlike a time-cut window.
        self._epoch_queue.put((stim_id, is_target, window[np.newaxis, :, :]))
        return True


def iter_epochs(
    stream: LSLEEGStream, timeout: float = 5.0
) -> Iterator[tuple[int, int | None, np.ndarray]]:
    """Yield (stim_id, is_target, epoch) from an already-started
    LSLEEGStream until get_epoch() goes `timeout` seconds without
    producing one.

    Wraps the "pull until nothing arrives, then stop" pattern a one-shot
    consumer script needs (see consume_lsl_stream.py): stream.stop()
    always runs when iteration ends, whether that's this generator
    exhausting itself, the caller breaking out of its `for` loop early, or
    an exception propagating through -- a `break` sends GeneratorExit into
    a suspended generator, which still runs the `finally` below, the same
    as a normal return would.

    Does not call stream.start() -- start the stream (and print whatever
    "resolving streams" message makes sense for the caller) before
    iterating.

    Parameters
    ----------
    stream : LSLEEGStream
        An already-started stream; this function only calls get_epoch()
        and, on exit, stop() -- never start().
    timeout : float
        Seconds get_epoch() may go without producing an epoch before this
        generator treats the stream as exhausted and returns.

    Yields
    ------
    tuple[int, int | None, numpy.ndarray]
        ``(stim_id, is_target, epoch)``, same shape as
        ``LSLEEGStream.get_epoch()``.
    """
    try:
        while True:
            epoch = stream.get_epoch(timeout=timeout)
            if epoch is None:
                return
            yield epoch
    finally:
        stream.stop()
