# eyecando.utils — shared run bookkeeping and constants

Small, dependency-free helpers used across every stage. None of these
import anything else in `eyecando`.

- **`path_helpers.py`** — `run_dir()` (a timestamped, collision-safe run
  directory -- fixed a real race condition where two processes launched
  in the same minute could silently share and clobber one `results.json`),
  plus model/checkpoint path helpers.
- **`time_marker.py`** — `TimeMarker`: named elapsed-time markers for
  logging how long a fold/search/run took.
- **`training_logger.py`** — `TrainingLogger`: prints to stdout and
  appends to a log file, no formatting opinions of its own.
- **`character_grid.py`** — `FARWELL_DONCHIN_GRID`: the speller's 6x6
  symbol alphabet. Not run bookkeeping like the rest of this package,
  but the same dependency-free shape -- moved here from `decode/` since
  it's a plain constant with no LM/accumulator logic attached.
  `eyecando.decode.lm` re-exports it for callers that already import
  `CharLM` from there.

## Why these numbers: investigation notes

### `run_dir()`'s collision fix (`path_helpers.py`)

The original version used `exist_ok=True`, which silently *reused* an
existing same-minute directory instead of erroring -- harmless for a
single process calling this once, but a real bug for concurrent
processes (e.g. a component-ablation study's 4-way parallel-terminal
design -- `scripts/run_component_ablation.py` on `main`, pruned from
this branch): two processes launched in the same minute would share one
`run.log` (interleaved, hard to read) and, worse, one `results.json`
(each process's own end-of-run `write_text` unconditionally overwrites
it, so whichever process finishes last silently destroys the others'
results with no error). Verified directly: launching 3 processes within
the same minute produced exactly this -- one `run.log` with all 3
conditions' log lines interleaved. The current loop makes every call
claim a new directory via `exist_ok=False` (atomic at the filesystem
level), with a numeric suffix if the plain name is already taken,
instead of ever silently sharing one.
