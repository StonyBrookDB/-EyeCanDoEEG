"""Tests for eyecando.pipeline.classifier.compute_itr.

Only covers compute_itr's monotonicity: ITR increases with higher
accuracy and with shorter time per selection. Does not exercise
_classification_metrics or _batched_metrics.
"""

from eyecando.pipeline.classifier import compute_itr


def test_itr_increases_with_higher_accuracy() -> None:
    low = compute_itr(accuracy=0.5, n_symbols=36, seconds_per_selection=1.5)
    high = compute_itr(accuracy=0.95, n_symbols=36, seconds_per_selection=1.5)
    assert high > low


def test_compute_itr_increases_with_shorter_selection_time() -> None:
    slow = compute_itr(accuracy=0.9, n_symbols=36, seconds_per_selection=3.0)
    fast = compute_itr(accuracy=0.9, n_symbols=36, seconds_per_selection=1.0)
    assert fast > slow
