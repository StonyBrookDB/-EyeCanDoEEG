"""Tests for eyecando.utils.time_marker.TimeMarker.

Covers mark()/elapsed_from()'s formatted-string and elapsed_from_pure()'s
raw-float output, raising on an unknown marker name (elapsed_from and
reset_to), clear() removing all markers, format_time()'s ms/s/m/h
threshold boundaries, and reset() creating a new marker.
"""

import time

import pytest

from eyecando.utils.time_marker import TimeMarker


def test_mark_and_elapsed_from_returns_formatted_string() -> None:
    tm = TimeMarker()
    tm.mark("start")
    time.sleep(0.01)
    assert "ms" in tm.elapsed_from("start") or "s" in tm.elapsed_from("start")


def test_elapsed_from_pure_returns_positive_float() -> None:
    tm = TimeMarker()
    tm.mark("start")
    time.sleep(0.01)
    assert tm.elapsed_from_pure("start") > 0


def test_elapsed_from_unknown_marker_raises() -> None:
    tm = TimeMarker()
    with pytest.raises(ValueError, match="not found"):
        tm.elapsed_from("nonexistent")


def test_reset_to_unknown_marker_raises() -> None:
    tm = TimeMarker()
    with pytest.raises(ValueError, match="not found"):
        tm.reset_to("nonexistent")


def test_clear_removes_all_markers() -> None:
    tm = TimeMarker()
    tm.mark("a")
    tm.mark("b")
    tm.clear()
    assert tm.markers == {}


def test_format_time_thresholds() -> None:
    tm = TimeMarker()
    assert tm.format_time(0.5) == "(500.0ms)"
    assert tm.format_time(5.2) == "(5.2s)"
    assert tm.format_time(125.0) == "(2m 5.0s)"
    assert tm.format_time(3725.0) == "(1h 2m 5.0s)"


def test_reset_creates_new_marker() -> None:
    tm = TimeMarker()
    tm.reset("checkpoint")
    assert "checkpoint" in tm.markers
