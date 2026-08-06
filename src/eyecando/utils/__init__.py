"""Small shared infrastructure with no pipeline-specific logic of its own.

- `path_helpers.py` -- single source of truth for where training/
  calibration runs write their output: `run_dir()` creates a
  timestamped, collision-safe run directory under `RESULTS_BASE`, with
  helpers (`model_dir()`, `checkpoint_path()`, `final_model_path()`,
  `calibrated_model_path()`) for the standard file layout beneath it.
- `training_logger.py` -- `TrainingLogger`, a minimal logger that
  prints each message to stdout and appends it to a log file; callers
  supply their own message tags (e.g. ``"[FOLD]"``).
- `time_marker.py` -- `TimeMarker`, a simple named-marker stopwatch for
  reporting elapsed time between points in a long-running script (used
  throughout `tuning` to time LOSO folds and grid-search candidates).
- `character_grid.py` -- `FARWELL_DONCHIN_GRID`, the classic 6x6
  row-column speller alphabet (26 letters + digits 1-9 + space). A
  plain constant, not run bookkeeping like the rest of this package, but
  dependency-free the same way -- re-exported by `eyecando.decode.lm`
  for callers that already import from there.

Used by `tuning` and by training/calibration scripts to report
progress and lay out results consistently; nothing here depends on any
other eyecando subpackage.
"""
