# eyecando.tuning — Stage 3: tune hyperparameters

Two independent things get tuned in this project, and this package holds
the tool for each:

- **`model_search.py`** — `nested_loso()`: nested-LOSO grid search over
  the *offline model's* hyperparameters (`n_components`, `shrinkage`).
  Run this against your own dataset once Stages 1-2 are done, to pick
  hyperparameters before deploying. `train_loso()` evaluates one already-
  chosen (fixed) combination via LOSO, without searching -- use it once
  `nested_loso()` has told you what to use.
- **`policy_scoring.py`** — `selection_score()`: the utility function
  `scripts/simulate_dynamic_stopping.py` uses to pick the *live decode
  policy's* hyperparameters (`threshold`, `temperature`, `lm_temperature`)
  -- an additive accuracy-vs-time tradeoff, not a ratio (see the
  function's own docstring for why a ratio-shaped score is the wrong
  choice here).

Both tools exercise real pipeline (`eyecando.pipeline`) and decode
(`eyecando.decode`) code during the search -- this package doesn't
duplicate any of that logic, it only decides *which* hyperparameters are
worth keeping.

## Why these numbers: investigation notes

### The `cache_dir` collision (`model_cache.py`)

`cache_dir` has no module-level default, deliberately: `model_cache_
path()`'s cache key has no dataset identity in it at all, so two
different datasets' callers sharing one cache directory can produce the
exact same key and silently load each other's incompatible cached
model. This was a real, observed collision, not a hypothetical one --
every caller must supply its own dataset-specific directory; don't
reintroduce a shared default.

### The "HELLO" double-L bug (`trials.py`, `reconstruct_trials()`)

`reconstruct_trials()` groups flash-level epochs into trials by
watching for a target change between repetitions. That heuristic alone
can't tell "still repeating the current character's reps" apart from "a
new character that happens to be the same letter," since the target
doesn't change either way. Confirmed against real data (session_014,
phrase "HELLO"): without the `rep_resets` signal (a repetition-number
marker, e.g. Muse2's own `rep` column), the two L's merged into one
30-repetition trial and the session decoded one character short
("HELO").

The fix (`rep_resets`, forcing a new trial whenever a repetition's own
number resets to 1) introduced a second, subtler bug during
development: a forced-new-trial round that *also* lost its target
flashes to artifact rejection has no target evidence of its own, and
naively falling back to "assume target unchanged from the trial in
progress" is wrong here specifically -- that fallback is only valid
when the round *is* a continuation of the prior trial, which a forced
new trial says it explicitly is not. Trusting the stale fallback here
produced a spurious extra trial mislabeled with the previous target,
confirmed again against session_014's "HELLO": decoding "HELLLO"
instead of "HELLO". The current code buffers such evidence-poor rounds
(`pending_slices`) and reattaches them as the eventual new trial's own
first repetitions once a later round supplies real target evidence,
rather than guessing.
