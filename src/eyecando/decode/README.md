# eyecando.decode — the decode-time policy, shared by tuning and live

Given a trained model's per-flash scores, this package decides what to do
with them: accumulate, decide when to stop, and weight by a language
model. None of it is live-hardware-specific -- `scripts/simulate_dynamic_
stopping.py` (Stage 3) runs this exact policy offline against
already-recorded data to tune it, and `eyecando.live.decoder` (Stage 5)
runs the identical code against a real epoch stream. That's why this
isn't inside `eyecando.live`: it's imported on equal footing by both.

- **`accumulator.py`** -- `ScoreAccumulator`: sums per-flash scores into
  per-symbol accumulated scores across repetitions. Standalone, no
  eyecando imports.
- **`stopping.py`** -- `should_decode()`/`request_next()`: the dynamic-
  stopping decision -- decode now, or wait for another round -- optionally
  weighted by an LM prior. Only depends on `accumulator.py`.
- **`lm.py`** -- `CharLM`: the single public language-model class this
  project constructs. `backend="neural"` (default) drives `lm_decoder.py`;
  `backend="kenlm"` drives `kenlm_lm.py`. Callers can't see which backend
  is active -- both implement the same `prior(context, temperature)` /
  `reset_context()` contract. Re-exports `FARWELL_DONCHIN_GRID` -- the
  speller's 6x6 symbol alphabet itself now lives in
  `eyecando.utils.character_grid` (a plain, dependency-free constant, so
  it belongs alongside `utils`'s other shared helpers rather than inside
  `decode`).
- **`lm_decoder.py`** -- `LMDecoder`: the neural backend, a permanently
  forked, trimmed copy of `textslinger`'s causal-subword beam search over
  a 350M-parameter transformer. See "The `lm_decoder.py` fork" below.
- **`kenlm_lm.py`** -- `_KenLMBackend` (private -- construct via
  `CharLM(backend="kenlm")`): the n-gram alternative, an interpolated
  character 12-gram KenLM trie. Requires a special local build
  (`scripts/setup_kenlm.sh`); see "KenLM tokenization convention" below.

Both LM backends return natural-log probabilities over the grid alphabet,
normalized so `np.exp(prior(...)).sum() == 1` at temperature 1.0 --
`stopping.py` and `live.decoder.LiveDecoder` never need to know which
backend is active.

---

## Why these numbers: investigation notes

The individual modules keep their docstrings to mechanics (what a
function does, its parameters, its contract). The reasoning, tuning
history, and bug post-mortems behind those mechanics live here instead,
so they don't have to be re-read every time someone opens the file just
to check a function signature.

### Dynamic-stopping defaults (`stopping.py`, `should_decode()`)

`threshold=0.9` and `temperature=2.0` are not arbitrary -- they're the
result of a nested-LOSO search over `scripts/simulate_dynamic_
stopping.py`'s grid, run against the full calibrate-and-decode pipeline
(not just flash-level accuracy). Verified as a genuine local optimum for
the current model, not just the edge of whatever range was swept:
accuracy plateaus at 100% right at `threshold=0.9`, so higher thresholds
only add repetitions (more flashes per symbol, slower spelling) with no
further accuracy gain. `temperature` softens the softmax used for the
confidence check -- it corrects for the classifier's raw decision scores
being systematically overconfident, and only affects how much evidence
is needed to cross `threshold`, never which symbol wins.

### KenLM tokenization convention (`kenlm_lm.py`)

The model (`lm_dec19_char_large_12gram.kenlm`, from figmtu's "N-gram
language models for AAC" project on OSF, CC BY 4.0 -- see
[osf.io/ajm7t/files/c6mnz](https://osf.io/ajm7t/files/c6mnz)) ships with
no documentation of its exact tokenization convention. The following was
determined empirically by scoring probe strings against the loaded
model:

- **Lowercase only.** `m.score("T H E", ...)` scores ~300 log10-points
  worse than `m.score("t h e", ...)` for identical text -- confirms this
  is the plain (not "mixed case") release from the OSF project page.
- **Space-separated, one character per KenLM "word."** A character
  12-gram model built with SRILM/KenLM tooling is fundamentally a word
  n-gram model where each "word" happens to be a single character. Fed
  an unsegmented string (`"hello"`), the model scores it as one
  incomprehensible 5-character "word" (~-201 log10) versus the correctly
  segmented `"h e l l o"` (~-108).
- **`<sp>` is the space-in-the-original-text token**, not a literal space
  character. Confirmed two ways: (1) `P(<sp> | "the")` scores as the
  single most likely continuation (-0.297 log10, beating any letter) --
  exactly what "word boundary after a common word" should look like; (2)
  using a `<apos>` placeholder the way `<sp>` works for space scores
  catastrophically badly (~-204 log10), while the literal `'` character
  scores fine (~-2.3 log10 for "don't") -- apostrophe is just its own
  character, not a special token; only space needed one.
- **No digits at all** in the training vocabulary (per the OSF page's own
  description: "vocabulary of A-Z, space, and apostrophe"). This
  project's `FARWELL_DONCHIN_GRID` has 9 digit cells (1-9) this model
  can say nothing meaningful about. Checked against every sentence this
  project's own test corpora actually spell (`scripts/curate/
  test_curated_words.py`'s `DEFAULT_SENTENCES`, every Muse2 session's
  `meta.json` phrase) -- none contain a digit, so the gap doesn't bite in
  practice here, but it's a real limitation of the model, not of this
  wrapper. Digit cells get a fixed, deliberately tiny log10 score
  (`_OOV_LOG10`) rather than raising.
- **Log10, not natural log.** Confirmed directly: summing
  `10**(score(context+cand) - score(context))` over the full 27-symbol
  in-vocabulary set landed at 0.9994 (matching a correctly-normalized
  per-context conditional distribution); the same sum using `exp()`
  instead of `10**` was 4.4, nowhere near 1. This is a property of
  KenLM/ARPA in general (ARPA format stores log10 probabilities), not
  specific to this one model.

No probe scripts are checked in for these findings; they were run
ad hoc against the loaded model. Re-verify against `kenlm.Model.score()`
directly if the model file is ever swapped.

### `CharLM`'s raw-prior cache (`lm.py`)

`CharLM._raw_prior_cache` memoizes the neural backend's raw (pre-
temperature) forward pass, keyed on context text alone. This exists
because of a measured, not hypothetical, bottleneck: `scripts/
simulate_dynamic_stopping.py` constructs one `CharLM` and reuses it
across an entire hyperparameter grid (temperature x threshold x
lm_temperature). `temperature`/`threshold` don't affect `LMDecoder.
predict_log_probs()`'s output at all, and `lm_temperature` only rescales
the cached raw result -- so without caching, grid size multiplies real
transformer forward-pass cost for no reason. Measured directly on this
pipeline: Stage 3 with an LM took 30-100x longer per grid combination
than without one, entirely from this redundancy. Only the neural backend
populates this cache -- a KenLM trie lookup is already cheap enough that
memoizing it wouldn't measurably help.

`CharLM.reset_context()` deliberately does *not* clear this cache (it
only clears `LMDecoder`'s own incremental position cache, which is tied
to token positions within one message and must be invalidated at message
boundaries). Clearing `_raw_prior_cache` on every message would be
actively harmful for a caller replaying the same fixed test sentence
across many independent messages (e.g. one per subject in a word-level
test): every message after the first would recompute contexts the first
one already cached. Call `clear_prior_cache()` explicitly instead, if
memory ever needs reclaiming -- rarely worth it, since it's bounded by
the number of distinct contexts ever seen, not by message count.

### The `lm_decoder.py` fork

`lm_decoder.py` is adapted from
[textslinger](https://github.com/kdv123/textslinger) 0.2.4 (MIT
license) -- specifically the character-prediction path of
`causal_subword.py`/`language_model.py`/`exceptions.py`/`helpers.py`,
absorbed into this single file and trimmed to only what `LMDecoder`
needs. This is a **permanent fork**, not something kept in sync with
upstream releases: MIT permits modification, and re-vendoring a future
textslinger release would mean re-deriving this file from scratch, not
a file copy.

Two bugs worth knowing about if this file is ever touched:

- **`HF_HUB_ETAG_TIMEOUT`** is set to 10s at import time because
  `huggingface_hub`'s pre-load HEAD/etag check was observed hanging
  ~19 minutes on an otherwise-healthy network (general connectivity to
  huggingface.co was fine; one stuck request) while running a
  component-ablation study's 4-way parallel smoke test. Falls back to
  the local cache on timeout; doesn't disable the update check, just
  bounds how long a stalled one can block model loading.
- **Output is natural-log, not linear**, even after
  `normalize_logprobs=True`. `LMDecoder.predict_log_probs()` never
  exponentiates. Whether to convert to linear space is a decision for
  whichever accumulation loop consumes the output (currently: nowhere
  does -- `CharLM.prior()` and `stopping.should_decode()` both stay in
  log space) -- guessing wrong in either direction (double-converting,
  or feeding log-probs into linear-space math) is a silent, easy-to-miss
  bug.
