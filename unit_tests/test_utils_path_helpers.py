"""Tests for eyecando.utils.path_helpers: run_dir, model_dir, and the
checkpoint/final-model/calibrated-model path builders.

Covers run_dir()'s timestamped directory naming/creation (with
RESULTS_BASE monkeypatched to a tmp_path) for both the "training" and
"calibration" prefixes, model_dir()'s subdirectory creation, the exact
filename format of each path builder, and the SAVE_FOLD_CHECKPOINTS
default.
"""

from eyecando.utils import path_helpers


def test_run_dir_creates_timestamped_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(path_helpers, "RESULTS_BASE", tmp_path)
    run = path_helpers.run_dir("training")
    assert run.exists()
    assert run.parent == tmp_path
    assert run.name.startswith("training_")


def test_run_dir_calibration_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(path_helpers, "RESULTS_BASE", tmp_path)
    run = path_helpers.run_dir("calibration")
    assert run.name.startswith("calibration_")


def test_model_dir_creates_subdirectory(tmp_path) -> None:
    model_path = path_helpers.model_dir(tmp_path)
    assert model_path == tmp_path / "model"
    assert model_path.exists()


def test_checkpoint_path_format(tmp_path) -> None:
    path = path_helpers.checkpoint_path(tmp_path, fold=3, test_subject=4)
    assert path == tmp_path / "model" / "fold_3_subject_4.pkl"


def test_final_model_path_format(tmp_path) -> None:
    path = path_helpers.final_model_path(tmp_path)
    assert path == tmp_path / "model" / "p300_classifier_final.pkl"


def test_calibrated_model_path_format(tmp_path) -> None:
    path = path_helpers.calibrated_model_path(tmp_path, subject_id="01")
    assert path == tmp_path / "model" / "p300_calibrated_subj01.pkl"


def test_save_fold_checkpoints_defaults_false() -> None:
    assert path_helpers.SAVE_FOLD_CHECKPOINTS is False
