"""Tests for eyecando.utils.training_logger.TrainingLogger.

Covers log() writing each message to both stdout and the log file,
appending multiple lines in order across calls, and that a log_path
given as a str is normalized to a Path.
"""

from eyecando.utils.training_logger import TrainingLogger


def test_log_writes_to_stdout_and_file(tmp_path, capsys) -> None:
    logger = TrainingLogger(tmp_path / "run.log")
    logger.log("[START] mode=training")

    captured = capsys.readouterr()
    assert "[START] mode=training" in captured.out

    log_contents = (tmp_path / "run.log").read_text()
    assert "[START] mode=training" in log_contents


def test_log_appends_multiple_lines(tmp_path) -> None:
    logger = TrainingLogger(tmp_path / "run.log")
    logger.log("first")
    logger.log("second")
    lines = (tmp_path / "run.log").read_text().splitlines()
    assert lines == ["first", "second"]


def test_log_path_stored_as_path_object(tmp_path) -> None:
    logger = TrainingLogger(str(tmp_path / "run.log"))
    assert logger.log_path == tmp_path / "run.log"
