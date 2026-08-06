# eyecando

A P300 event-related-potential brain-computer interface speller: EEG epochs
in, decoded text out. Currently supports offline datasets like BNCI2014_009,
via MOABB and is partially adaptable for a consumer headset (Muse2), and 
covers the full path from raw data through a trained, tuned, calibrated 
model to live decoding.

The package is organized around that path -- see each subpackage's own
README for what it does and why it's split out the way it is:

- [`src/eyecando/ingestion/`](src/eyecando/ingestion/README.md) -- get data in (Stage 1)
- [`src/eyecando/pipeline/`](src/eyecando/pipeline/README.md) -- train a model (Stage 2), and calibrate it to a new subject via `P300Model.adapt_to_subject()`/`build_aligner()` (Stage 4)
- [`src/eyecando/tuning/`](src/eyecando/tuning/README.md) -- tune hyperparameters (Stage 3)
- [`src/eyecando/decode/`](src/eyecando/decode/README.md) -- the decode-time policy, shared by tuning and live
- [`src/eyecando/live/`](src/eyecando/live/README.md) -- real hardware orchestration (Stage 5)
- [`src/eyecando/utils/`](src/eyecando/utils/README.md) -- shared run bookkeeping and constants
- [`scripts/`](scripts/README.md) -- CLI entry points for each stage above

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional: KenLM backend

`eyecando.decode.lm.CharLM(backend="kenlm")` is an alternative to the
default neural language model -- a much cheaper n-gram backend, at the
cost of no digit coverage (see `src/eyecando/decode/README.md`'s
"KenLM tokenization convention" for the details). It's deliberately
**not** installed by `pip install -e ".[dev]"` above: PyPI's `kenlm`
package doesn't build against this project's Python (3.14), and its
default `MAX_ORDER=6` would reject the 12-gram model this project
actually uses regardless. Everything else in this repo works with zero
dependency on `kenlm` ever being installed -- `unit_tests/
test_decode_kenlm_lm.py` skips itself automatically if it isn't.

If you want the KenLM backend, run this once per `.venv` (with the
`.venv` above already activated):

```bash
scripts/setup_kenlm.sh
```

It downloads the model (~231MB) and rebuilds `kenlm`'s Python bindings
from source against your active Python, with the `MAX_ORDER=12`
override this model needs. See the script's own header comment for why
each step is necessary, and re-run it any time `.venv` is recreated.

`results/`, `models/`, and `data/raw/`, `data/processed/` ship empty
(placeholder-only, tracked via `.gitkeep`) -- this is source code, not a
data or trained-model distribution. You'll need to point ingestion at
your own data (see `src/eyecando/ingestion/README.md`) and run the
training/tuning pipeline yourself to populate `models/` and `results/`.

## Running tests

```bash
pytest
```

## Linting / formatting

```bash
ruff check .
ruff format .
```

## Project structure

```
src/eyecando/
  ingestion/    # Stage 1 — dataset-agnostic loaders + per-dataset/headset remap modules
  pipeline/     # Stage 2 — filter, align, spatially filter, classify: the fit path
                # (also Stage 4 — P300Model.adapt_to_subject()/build_aligner()
                #  for headset/subject calibration against a trained model)
  tuning/       # Stage 3 — nested-LOSO hyperparameter search + live-policy scoring
  decode/       # decode-time policy shared by tuning and live: accumulator, stopping, LM backends
  live/         # Stage 5 — real LSL streaming + the live decode loop
  utils/        # run bookkeeping (paths, timing, logging) + shared constants (character_grid)
scripts/
  train_offline.py              # Stage 2 entry point
  simulate_dynamic_stopping.py  # Stage 3 entry point
  curate/                       # synthetic curated-word decode testing
  simulate_live/                # no-hardware LSL integration test harness --
                                 # contains run_live_session.py, the actual Stage 5
                                 # entry point (nested here, not directly under scripts/),
                                 # plus livetesting.ipynb, a scratch notebook for the same path
unit_tests/     # pytest tests, mirroring src/eyecando/ subpackage layout
results/        # placeholder — populated by running tuning/training yourself
models/         # placeholder — populated by running training yourself
data/
  raw/          # placeholder — point this at your own dataset
  processed/    # placeholder — derived/cleaned data lands here
```

## Data assumptions & portability

This pipeline was built and tuned against one dataset first
(**BNCI2014_009**, a public 8-subject P300-speller dataset loaded via
MOABB) and a specific speller layout (the classic **Farwell & Donchin
(1988) 6x6 row-column grid**, 36 symbols). Muse2 support was added
afterward as a second, real-world-hardware ingestion path (see
[`src/eyecando/ingestion/README.md`](src/eyecando/ingestion/README.md)),
but several places downstream of ingestion still carry assumptions that
trace back to that original dataset and grid, rather than being fully
generic. Concretely, if you're bringing new hardware, a different
electrode montage, or a differently-shaped speller grid, here is what you
will actually run into:

- **Timing/grid constants are BNCI-measured, not auto-derived.**
  [`src/eyecando/pipeline/classifier.py`](src/eyecando/pipeline/classifier.py)
  defines `SECONDS_PER_FLASH = 0.25` with the comment "measured from
  BNCI2014_009's own stim_id event timing," alongside `N_SYMBOLS = 36` and
  `FLASHES_PER_ROUND = 12` for the 6x6 grid. These are imported as plain
  module-level constants — not passed through as configurable parameters —
  by `tuning/model_search.py`, `tuning/policy_scoring.py`,
  `scripts/simulate_dynamic_stopping.py`, `scripts/curate/test_curated_words.py`,
  and several `scripts/simulate_live/` scripts. A different flash duration
  or grid size means editing this one file's constants directly (every
  caller picks them up automatically), not passing a new argument at the
  call site.

- **The 6x6 grid is baked into row/col arithmetic in several scripts, not
  just a default parameter.**
  [`src/eyecando/utils/character_grid.py`](src/eyecando/utils/character_grid.py)'s
  `FARWELL_DONCHIN_GRID` is accepted as an overridable `character_set`
  argument in `CharLM` (`decode/lm.py`) — that part is already generic.
  But several scripts under `scripts/curate/` and `scripts/simulate_live/`
  convert between `(row, col)` stim codes and characters with the literal
  expression `FARWELL_DONCHIN_GRID[(row - 1) * 6 + (col - 7)]` — a
  row-major 6x6 assumption written directly into the arithmetic, not
  derived from the grid's actual shape. Swapping in a differently-shaped
  grid (e.g. a smaller pediatric layout, or a non-square one) means
  updating that indexing expression everywhere it appears, not just
  supplying a longer or shorter symbol list.

- **The canonical `stim_id` convention (1-12, row/col) originates from
  this scheme.** Every ingestion loader (`ingestion/data.py`,
  `ingestion/bnci2014_009.py`, `ingestion/muse2.py`) normalizes its
  source data into this project's own `stim_id`/`target_flag`
  convention — 1-6 for target row, 7-12 for target column — which is
  itself the Farwell-Donchin scheme's numbering. `ingestion/muse2.py`'s
  `remap_stim_ids()` is a working example of what a new hardware source
  has to do to fit this convention (in Muse2's case, its native
  1-6=column/7-12=row ordering is the *opposite* of this project's and
  has to be swapped); a different grid topology (not just a
  different code ordering) would need the canonical convention itself
  reconsidered, not just another remap function.

- **Stage 4 (headset/subject calibration)'s population-based approach
  assumes a multi-subject training set.**
  `pipeline.model.P300Model.fit()` takes `sessions: list[tuple[X, y]]` —
  one entry per subject — and uses them to fit a **Bühlmann-Straub /
  Efron-Morris empirical-Bayes shrinkage estimator**
  (`_population_kappa()` in `pipeline/model.py`) that later blends a new
  subject's own calibration data with the population's statistics in
  `adapt_to_subject()`. This is meaningful with a MOABB-scale dataset
  (many subjects' sessions to estimate real between-subject variance
  from) but degenerates with a single subject: the population variance
  term (`tau2_pop`) has nothing to estimate from, gets floored at a tiny
  epsilon, and the resulting shrinkage constant pushes the blend almost
  entirely toward the (nearly meaningless, single-subject) population
  estimate rather than the new subject's own data. The code handles this
  edge case without crashing (see `_population_kappa()`'s own docstring),
  but it's explicitly noted there as exercised only by a single-session
  test fixture — not a configuration a real single-user deployment should
  rely on for meaningful calibration. A from-scratch single-subject
  deployment (no population to train against) would need either a
  substitute prior or a different calibration strategy entirely.

**In short**: the `ingestion/` layer is the part of this codebase
designed to be extended per-hardware (that's its whole purpose —
see its own README), but `pipeline/`, `tuning/`, and several `scripts/`
still carry specific numeric and structural assumptions inherited from
BNCI2014_009 and the classic 6x6 grid. Adding new hardware today is
realistic (Muse2 is a real, working example); adding a fundamentally
different grid shape or running with only a single subject's data (no
population to calibrate against) is not yet a supported, drop-in path —
it would touch the constants and row/col arithmetic named above, and
would need a real design decision about what replaces population-based
shrinkage when there's no population.
