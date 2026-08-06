"""TimeMarker utility for tracking elapsed time between code points.

Shared run bookkeeping used across pipeline stages that need to report how
long a fold, search, or run took (see tuning/model_search.py,
scripts/train_offline.py, scripts/simulate_dynamic_stopping.py). Named
markers (`mark()`/`elapsed_from()`) let a caller time multiple
overlapping/nested spans -- e.g. one outer LOSO fold's total time alongside
that same fold's inner grid-search time -- without juggling raw timestamps
itself.
"""

from __future__ import annotations

import time


class TimeMarker:
    """Simple time marker for tracking elapsed time between code points.

    Attributes
    ----------
    last_time : float
        `time.time()` timestamp of the most recent mark/reset; the
        reference point for `elapsed()`.
    markers : dict[str, float]
        Named timestamps set via `mark()` (or `reset()`), looked up by
        `elapsed_from()`, `elapsed_from_pure()`, and `reset_to()`.
    """

    def __init__(self) -> None:
        self.last_time = time.time()
        self.markers: dict[str, float] = {}

    def mark(self, name: str | None = None) -> None:
        """Set a time marker. If name provided, stores it for later reference.

        Parameters
        ----------
        name : str or None
            If given, the current timestamp is also stored under this key
            in `markers` so it can be referenced later by
            `elapsed_from()`, `elapsed_from_pure()`, or `reset_to()`. If
            omitted, only `last_time` is updated.
        """
        current_time = time.time()
        self.last_time = current_time
        if name:
            self.markers[name] = current_time

    def elapsed(self) -> str:
        """Return formatted elapsed time from last marker.

        Returns
        -------
        str
            Time since `last_time`, formatted by `format_time()` (e.g.
            ``"(1.2s)"``, ``"(3m 4.0s)"``).
        """
        elapsed_seconds = time.time() - self.last_time
        return self._format_time(elapsed_seconds)

    def elapsed_from(self, marker_name: str) -> str:
        """Return formatted elapsed time from a named marker.

        Parameters
        ----------
        marker_name : str
            Key previously stored via `mark()` or `reset()`.

        Returns
        -------
        str
            Time since that marker, formatted by `format_time()`.

        Raises
        ------
        ValueError
            If `marker_name` was never set.
        """
        if marker_name not in self.markers:
            raise ValueError(f"Marker '{marker_name}' not found")
        elapsed_seconds = time.time() - self.markers[marker_name]
        return self._format_time(elapsed_seconds)

    def elapsed_from_pure(self, marker_name: str) -> float:
        """Return raw elapsed seconds since named marker as a float.

        Parameters
        ----------
        marker_name : str
            Key previously stored via `mark()` or `reset()`.

        Returns
        -------
        float
            Unformatted seconds elapsed since that marker, for callers
            that need the raw number rather than `elapsed_from()`'s
            human-readable string.

        Raises
        ------
        ValueError
            If `marker_name` was never set.
        """
        if marker_name not in self.markers:
            raise ValueError(f"Marker '{marker_name}' not found")
        return time.time() - self.markers[marker_name]

    def format_time(self, seconds: float) -> str:
        """Format a raw float seconds value into a readable string.

        Parameters
        ----------
        seconds : float
            Raw duration to format; does not need to come from this
            instance's own markers.

        Returns
        -------
        str
            ``"(Xms)"`` under 1s, ``"(Xs)"`` under 1 minute, ``"(Xm Ys)"``
            under 1 hour, else ``"(Xh Ym Zs)"``.
        """
        return self._format_time(seconds)

    def reset(self, name: str | None = None) -> None:
        """Reset timer to current time, optionally with a new marker name.

        Equivalent to `mark(name)`.

        Parameters
        ----------
        name : str or None
            If given, also stored under this key in `markers`.
        """
        self.mark(name)

    def reset_to(self, marker_name: str) -> None:
        """Reset timer to a previously set marker.

        Sets `last_time` back to `marker_name`'s stored timestamp, so a
        subsequent `elapsed()` measures from that point rather than now.

        Parameters
        ----------
        marker_name : str
            Key previously stored via `mark()` or `reset()`.

        Raises
        ------
        ValueError
            If `marker_name` was never set.
        """
        if marker_name not in self.markers:
            raise ValueError(f"Marker '{marker_name}' not found")
        self.last_time = self.markers[marker_name]

    def clear(self) -> None:
        """Clear all markers and reset to current time."""
        self.markers.clear()
        self.last_time = time.time()

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 1:
            return f"({seconds * 1000:.1f}ms)"
        elif seconds < 60:
            return f"({seconds:.1f}s)"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"({minutes}m {remaining_seconds:.1f}s)"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"({hours}h {minutes}m {remaining_seconds:.1f}s)"
