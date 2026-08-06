# scripts/ — CLI entry points

One entry point per workflow stage, plus two testing subfolders.

- **`train_offline.py`** — Stage 2: fit `P300Model` on MOABB data, save it.
- **`simulate_dynamic_stopping.py`** — Stage 3: tune the live decode
  policy's hyperparameters (threshold, temperature, `lm_temperature`)
  against real trained models. Also where the `MODEL_CACHE_DIR` LOSO
  model cache lives (`get_or_fit_model()`) -- local-only, never committed.
- **`run_live_session.py`** — Stage 5: run `LiveDecoder` against a real or
  replayed LSL session.
- **`curate/`** — synthetic curated-word decode testing: build a synthetic
  multi-letter "word" out of real recorded trials and run it through
  `LiveDecoder`, no live subject needed. See `curate/README.md`.
- **`simulate_live/`** — no-hardware LSL integration test harness for the
  streaming path itself. See `simulate_live/README.md`.

## Why these numbers: investigation notes

### Flash-level proxies disagree with real trial-level accuracy (`simulate_dynamic_stopping.py`)

This is the reason the search is split into cheap flash-level ranking
(Stage 1) plus a real trial-level check (`validate_final_config()`,
`--validate-top-k`) rather than trusting the cheap proxy alone. Verified
directly on this pipeline: a config with the *highest* flash-level
accuracy (89.0%) produced the *worst* trial-level accuracy (85.8%) of
several candidates compared, because its confidently-wrong flash
predictions corrupted trial-level accumulation -- while a config with
*lower* flash accuracy (87.0%) reached 98.3% trial-level accuracy. Only
a real trial-level simulation (`ScoreAccumulator`/`should_decode`)
catches this; no cheap proxy will, which is also why `top_k_combos()`
exists (Stage 1's own #1 pick isn't trustworthy enough to act on
without a shortlist).

### `beta_power_score`'s `beta=4` default (`simulate_dynamic_stopping.py`)

Stage 1 scores candidates with `beta_power_score` rather than plain log
loss because log loss's unbounded tail let a handful of
confidently-wrong predictions dominate the mean on real held-out data,
pushing the search toward a smaller, less accurate model that just
avoided ever committing -- not what "punish confident-wrong" was meant
to produce. `beta=4` was chosen by checking where that tradeoff flips:
on a real held-out subject, the more accurate, more confident model
still won up to `beta=5.5`; only `beta>=6` flipped the preference
toward avoiding commitment. `beta=4` sits on the accurate side with
room to spare, while still punishing confident errors meaningfully more
than plain Brier (`beta=2`).

### `class_conditional_overlap`'s `confidently_wrong_margin` (`simulate_dynamic_stopping.py`)

Checking only exact 0.0/1.0 probability saturation undercounts
"confidently wrong" predictions badly: on real held-out data, one
config had a single exact-saturation error but over 100 more
predictions sitting at `proba<=0.01` while still wrong, which
exact-only counting completely missed. `confidently_wrong_margin=0.05`
is a reasonable default given that, not a value separately tuned
against this data.
