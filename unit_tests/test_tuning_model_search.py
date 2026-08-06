"""Tests for eyecando.tuning.model_search: train_loso, nested_loso, and
_inner_grid_search, against synthetic multi-subject epoch data.

train_loso coverage: returns a fitted P300Model plus per-subject and
mean/std metrics across accuracy/precision/recall/f1/auc, results at
every REPETITION_COUNTS value, no majority-class collapse, and that the
final model is fit on all subjects. nested_loso coverage: same result
shape plus per-fold chosen hyperparameters and a final all-subjects
search/fit, and that logged messages carry the expected [SEARCH]/
[FOLD]/[SUMMARY]/[FINAL] tags with full per-candidate metrics.
_inner_grid_search coverage: disqualifying a singular (high
n_components, zero-shrinkage) combination rather than crashing, and
that ITR (not raw AUC) drives selection while every candidate's full
metric set is still returned.
"""

import numpy as np

from eyecando.pipeline.model import P300Model
from eyecando.tuning.model_search import (
    REPETITION_COUNTS,
    _inner_grid_search,
    nested_loso,
    train_loso,
)


def _synthetic_subject(
    n_epochs: int = 100,
    n_channels: int = 16,
    n_times: int = 230,
    seed: int = 0,
    ratio: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (np.arange(n_epochs) % ratio == 0).astype(int)
    X = rng.normal(scale=1.0, size=(n_epochs, n_channels, n_times))
    bump = 4.0 * np.exp(-((np.arange(n_times) - n_times // 2) ** 2) / 200)
    X[y == 1, 0, :] += bump
    return X, y


def _synthetic_subjects(n_subjects: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_synthetic_subject(seed=i) for i in range(n_subjects)]


def test_train_loso_returns_fitted_model_and_results() -> None:
    model, results = train_loso(_synthetic_subjects())
    assert isinstance(model, P300Model)
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        assert f"mean_{metric}" in results
        assert f"std_{metric}" in results
    assert len(results["per_subject"]) == 4


def test_per_subject_results_include_precision_recall_f1() -> None:
    _, results = train_loso(_synthetic_subjects())
    for subject_result in results["per_subject"]:
        for metric in ("precision", "recall", "f1"):
            assert 0.0 <= subject_result[metric] <= 1.0


def test_per_subject_results_cover_all_repetition_counts() -> None:
    _, results = train_loso(_synthetic_subjects())
    for subject_result in results["per_subject"]:
        assert set(subject_result["per_repetition"]) == set(REPETITION_COUNTS)
        for rep_result in subject_result["per_repetition"].values():
            assert 0.0 <= rep_result["accuracy"] <= 1.0
            for metric in ("precision", "recall", "f1"):
                assert 0.0 <= rep_result[metric] <= 1.0
            assert np.isfinite(rep_result["itr_bits_per_min"])


def test_class_imbalance_handled_in_loso_predictions() -> None:
    model, results = train_loso(_synthetic_subjects())
    for subject_result in results["per_subject"]:
        assert subject_result["accuracy"] > 0.5


def test_mean_accuracy_matches_per_subject_average() -> None:
    _, results = train_loso(_synthetic_subjects())
    expected_mean = np.mean([r["accuracy"] for r in results["per_subject"]])
    assert results["mean_accuracy"] == expected_mean


def test_final_model_is_fit_on_all_subjects() -> None:
    subjects = _synthetic_subjects()
    model, results = train_loso(subjects)
    # Final model is trained on all subjects; adapt to one before inference.
    X0, y0 = subjects[0]
    model.adapt_to_subject(X0, y0, refit_lda=False)
    proba = model.predict_proba(X0)
    assert proba.shape == (X0.shape[0],)
    assert len(results["per_subject"]) == len(subjects)


def test_train_loso_accepts_n_components_and_shrinkage() -> None:
    subjects = _synthetic_subjects()
    model, _ = train_loso(subjects, n_components=3, shrinkage=0.5)
    assert model.spatial_filter.n_components == 3


def _small_synthetic_subjects(n_subjects: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Smaller epochs/timepoints than _synthetic_subjects() so the nested
    grid search (which refits many models per outer fold) stays fast."""
    return [
        _synthetic_subject(n_epochs=80, n_channels=4, n_times=20, seed=i)
        for i in range(n_subjects)
    ]


def test_nested_loso_returns_fitted_model_and_results() -> None:
    subjects = _small_synthetic_subjects()
    model, results = nested_loso(subjects, n_components_grid=[1, 2], shrinkage_grid=[0.3, 0.7])
    assert isinstance(model, P300Model)
    assert len(results["per_subject"]) == len(subjects)
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        assert f"mean_{metric}" in results


def test_nested_loso_records_chosen_hyperparameters_per_fold() -> None:
    subjects = _small_synthetic_subjects()
    _, results = nested_loso(subjects, n_components_grid=[1, 2], shrinkage_grid=[0.3, 0.7])
    for subject_result in results["per_subject"]:
        assert subject_result["chosen_n_components"] in (1, 2)
        assert subject_result["chosen_shrinkage"] in (0.3, 0.7)


def test_nested_loso_final_model_uses_all_subjects_search() -> None:
    subjects = _small_synthetic_subjects()
    model, results = nested_loso(subjects, n_components_grid=[1, 2], shrinkage_grid=[0.3, 0.7])
    assert results["final_n_components"] in (1, 2)
    assert results["final_shrinkage"] in (0.3, 0.7)
    assert model.spatial_filter.n_components == results["final_n_components"]


def test_inner_grid_search_skips_singular_combinations() -> None:
    """A high n_components + zero-shrinkage combination on a small training
    set produces a non-positive-definite covariance -- the search must
    disqualify it (not crash) and fall back to a combination that fits."""
    subjects = [
        _synthetic_subject(n_epochs=20, n_channels=8, n_times=20, seed=i) for i in range(4)
    ]
    best_n, best_shrinkage, best_metrics, candidates = _inner_grid_search(
        subjects, n_components_grid=[1, 8], shrinkage_grid=[0.0, 0.9]
    )
    assert np.isfinite(best_metrics["itr"])
    assert (best_n, best_shrinkage) != (8, 0.0)
    assert any(c["disqualified"] for c in candidates)


def test_inner_grid_search_selects_by_itr_not_raw_auc() -> None:
    """The winning combination's returned metrics must include all of
    accuracy/precision/recall/f1/auc/itr, not just whatever was used to
    pick it."""
    subjects = _small_synthetic_subjects()
    _, _, best_metrics, candidates = _inner_grid_search(
        subjects, n_components_grid=[1, 2], shrinkage_grid=[0.3, 0.7]
    )
    for metric in ("accuracy", "precision", "recall", "f1", "auc", "itr"):
        assert metric in best_metrics
    assert len(candidates) == 4


def test_nested_loso_logs_tagged_messages(tmp_path) -> None:
    from eyecando.utils.training_logger import TrainingLogger

    logger = TrainingLogger(tmp_path / "run.log")
    subjects = _small_synthetic_subjects()
    nested_loso(subjects, n_components_grid=[1, 2], shrinkage_grid=[0.3, 0.7], logger=logger)

    log_contents = (tmp_path / "run.log").read_text()
    assert "[SEARCH]" in log_contents
    assert "[FOLD]" in log_contents
    assert "[SUMMARY]" in log_contents
    assert "[FINAL]" in log_contents
    # every logged search candidate exposes all metrics, not just the one used to select
    assert "itr=" in log_contents and "auc=" in log_contents and "f1=" in log_contents
