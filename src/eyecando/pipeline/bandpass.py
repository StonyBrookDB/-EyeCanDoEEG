"""Causal Butterworth bandpass filtering, streaming-safe.

Pure transform -- no eyecando imports.

Two layers here: build_sos()/apply_filter() are the raw primitives --
design a filter once, then apply it to any chunk with an explicit,
caller-managed `zi` (filter state) passed in and a new `zi` returned, so
state can be carried across chunks by whoever calls this repeatedly (a
continuous live stream) or ignored entirely for a single whole-array call
(the offline batch case, e.g. preprocess_p300()). BandpassFilter wraps
the same two functions and holds `zi` internally instead, for a caller
that just wants "filter this chunk, remember the state for me" without
threading a variable through itself -- streaming.py's acquire loop (many
repeated chunked calls) is the motivating case; a single whole-array call
has no need for it.

preprocessing.py imports build_sos()/apply_filter() from here for its own
preprocess_p300() -- this module contains no epoching/EA logic of any
kind, purely the reusable causal-filter primitive shared by both the
offline and live paths.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def build_sos(
    l_freq: float = 0.1,
    h_freq: float = 20.0,
    sfreq: float = 256.0,
    order: int = 4,
) -> np.ndarray:
    """Design a causal Butterworth bandpass SOS filter.

    Call once per session, then pass the result to apply_filter() (or let
    BandpassFilter do this internally).

    Parameters
    ----------
    l_freq, h_freq : float
        Bandpass cutoff frequencies, in Hz.
    sfreq : float
        Sampling rate of the signal this filter will be applied to, in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Second-order-sections representation, shape
        ``(n_sections, 6)`` (scipy's `butter(..., output="sos")`
        convention), for use with `apply_filter`/`BandpassFilter`.
    """
    return butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")


def apply_filter(
    sos: np.ndarray,
    x: np.ndarray,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal SOS bandpass filter to EEG data.

    Parameters
    ----------
    sos : numpy.ndarray
        Second-order-sections filter, from `build_sos`.
    x : numpy.ndarray
        Shape ``(n_channels, n_samples)``.
    zi : numpy.ndarray or None
        Filter state carried across chunks, shape
        ``(n_channels, n_sections, 2)``. If None, initializes from the
        first sample of each channel (steady-state assumption --
        minimises onset transient).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(filtered_x, new_zi)``. `filtered_x` is the same shape as `x`.
        Pass `new_zi` into the next call to maintain continuity across
        chunks, or use BandpassFilter if you'd rather not manage this
        yourself.
    """
    n_channels = x.shape[0]
    zi_template = sosfilt_zi(sos)  # (n_sections, 2)

    if zi is None:
        zi = np.stack([zi_template * x[ch, 0] for ch in range(n_channels)])

    out = np.empty_like(x)
    new_zi = np.empty((n_channels, *zi_template.shape))
    for ch in range(n_channels):
        out[ch], new_zi[ch] = sosfilt(sos, x[ch], zi=zi[ch])
    return out, new_zi


class BandpassFilter:
    """Causal Butterworth bandpass filter with internally-carried zi state.

    Same build_sos()/apply_filter() primitives underneath -- this just
    holds `zi` as instance state instead of making every caller thread it
    through by hand across repeated apply() calls, which is what
    streaming.py's continuous per-chunk filtering actually needs. A
    single whole-array call (preprocess_p300()'s use case) has no need
    for this -- zi never needs to persist past one call there, so that
    path still uses build_sos()/apply_filter() directly.

    Parameters
    ----------
    l_freq, h_freq : float
        Bandpass cutoff frequencies, in Hz.
    sfreq : float
        Sampling rate of the signal this filter will be applied to, in Hz.
    order : int
        Butterworth filter order.
    """

    def __init__(
        self,
        l_freq: float = 0.1,
        h_freq: float = 20.0,
        sfreq: float = 256.0,
        order: int = 4,
    ) -> None:
        self.sos = build_sos(l_freq, h_freq, sfreq, order)
        self._zi: np.ndarray | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Filter one chunk, carrying zi forward from the previous call.

        Parameters
        ----------
        x : numpy.ndarray
            Shape ``(n_channels, n_samples)``.

        Returns
        -------
        numpy.ndarray
            Filtered chunk, same shape as `x`.
        """
        filtered, self._zi = apply_filter(self.sos, x, zi=self._zi)
        return filtered

    def reset(self) -> None:
        """Drop any carried zi state -- the next apply() re-initializes
        from that chunk's own first sample, same as a fresh instance."""
        self._zi = None
