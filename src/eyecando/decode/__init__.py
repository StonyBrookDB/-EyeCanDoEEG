"""Turning per-flash classifier scores into decoded characters.

Sits downstream of `pipeline`'s `P300Model`: this subpackage never
scores an epoch itself, only accumulates and interprets scores that
have already been produced.

- `accumulator.py` -- `ScoreAccumulator`, sums per-flash classifier
  scores into a row/column grid across repetitions, producing a
  per-symbol accumulated score.
- `stopping.py` -- `should_decode()`, the dynamic-stopping rule:
  softmaxes the accumulated scores and decides whether confidence is
  high enough to decode now or another repetition is needed; optional
  language-model reweighting via `request_next()`.
- `lm.py` -- `CharLM`, the single public language-model class this
  project constructs, dispatching to one of two interchangeable
  backends behind one `prior(context, temperature)` contract. Re-exports
  `FARWELL_DONCHIN_GRID` (the speller alphabet itself lives in
  `eyecando.utils.character_grid`) so callers don't need a second import.
- `lm_decoder.py` -- `LMDecoder`, the neural backend: a permanently
  forked, trimmed-down copy of `textslinger`'s causal-subword beam
  search over a 350M-parameter transformer (see this package's
  README.md for the fork's history).
- `kenlm_lm.py` -- the KenLM backend: an alternative character 12-gram
  n-gram model, much cheaper per call than the neural backend but with
  no digit coverage.

Both language-model backends return natural-log probabilities over the
grid alphabet, normalized so `np.exp(prior(...)).sum() == 1` at
temperature 1.0 -- `stopping.py` and `live.decoder.LiveDecoder` never
need to know which backend is active.
"""
