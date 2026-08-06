"""Minimal logger: prints to stdout and appends to a log file.

Shared, dependency-free run bookkeeping used across pipeline stages that run
long enough to want both a live console view and a permanent record --
offline training (scripts/train_offline.py), Stage 3 hyperparameter search
(tuning/model_search.py), and dynamic-stopping simulation
(scripts/simulate_dynamic_stopping.py). Deliberately has no formatting/level
opinions of its own (unlike Python's `logging` module) -- callers write
their own tagged messages (e.g. "[FOLD] ...", "[SEARCH] ..."), and this
module just makes sure every message reaches both stdout and disk.
"""

from __future__ import annotations

from pathlib import Path


class TrainingLogger:
    """Prints each message to stdout and appends it to a log file.

    No formatting helpers -- callers write their own messages, including
    whatever tag (e.g. "[PREP]", "[SEARCH]", "[FOLD]") identifies where a
    line came from.

    Parameters
    ----------
    log_path : Path
        File to append log lines to. Not created until the first call to
        `log()`; any missing parent directories are not created either.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)

    def log(self, message: str) -> None:
        """Print `message` and append it to `log_path`.

        Parameters
        ----------
        message : str
            Line to record. No timestamp or tag is added -- callers
            include their own (e.g. ``"[FOLD] ..."``).

        Notes
        -----
        Opens and closes `log_path` on every call rather than holding a
        file handle open -- append mode (``"a"``) is safe for multiple
        processes/instances writing to the same path concurrently, each
        call is a single small write.
        """
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")
