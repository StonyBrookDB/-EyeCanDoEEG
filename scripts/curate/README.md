# scripts/curate/ — test the model in simulated settings

Builds and replays synthetic multi-letter "words" out of real recorded
trials, so decode correctness/ITR can be checked without a live subject.

- **`curate_trial_epochs.py`** — filters and epochs one subject/session
  into a single flat artifact: every trial's epochs, with a character
  label attached to each, plus several row/col permutation variants so
  a single session can cover the full 36-symbol grid (BNCI2014_009's real
  sessions only ever spell 18 of the 36 characters between them).
- **`build_curated_word.py`** — assembles a synthetic cross-session "word"
  from those curated artifacts, simulating a realistic, progressively
  built-up `EuclideanAligner` per letter rather than one aligner fit on a
  whole session in hindsight.
- **`test_curated_words.py`** — replays a built word through a real
  `LiveDecoder` (via `PrebuiltEpochStream`, no LSL connection) at two
  levels of cross-session splicing, and diffs the decoded text against
  what was intended.

Run in that order: curate a session, build a word from curated sessions,
then test decode against the built word.

## Why these numbers: investigation notes

### Trial-gap detection (`curate_trial_epochs.py`)

Trials are separated by a real, multi-second pause in the recording:
observed as ~8s gaps recurring every 96 flashes for BNCI2014_009, versus
the ~0.25s gap between ordinary flashes within a trial --
`TRIAL_GAP_THRESHOLD_S = 2.0` sits comfortably between the two. (This
was originally cited as "confirmed directly, see git history" -- this
repo's history starts at a single squashed initial commit, so that
citation isn't actually checkable here; treat the ~8s/96-flash figures
above as the surviving record instead.)

### Whole-session filtering, not per-trial (`curate_trial_epochs.py`)

Filtering happens once, on the whole continuous session, before any
trial detection -- the same causal Butterworth pass `preprocess_p300()`
uses, with `zi` propagating naturally through the one continuous
recording. An earlier version of this script filtered each trial's own
standalone segment instead, to avoid ever filtering across a splice
boundary -- but a single BNCI2014_009 session *is* one continuous
recording with exactly one run, so there was never a splice to avoid.
That version needed a large, empirically-tuned lead-in margin (~20-30s)
to converge close enough to whole-session filtering to trust; the
current version is correct by construction instead of by convergence,
and needs no margin parameter. The splice concern that motivated the
original design is real for artifacts built from segments that
*aren't* already one continuous recording (e.g. `build_curated_word.py`
reassembling trials into a cross-session synthetic "word") -- it just
doesn't apply to curating a single, already-continuous session.

### Why coverage is handled downstream, not here (`curate_trial_epochs.py`)

Every subject/session in BNCI2014_009 spells the exact same 3 fixed
words -- only 18 of `FARWELL_DONCHIN_GRID`'s 36 characters ever appear
across all 30 real sessions. `curate_trial_epochs.py` only ever records
each trial's real, physical (unpermuted) character; it makes no attempt
to cover the grid's other 18 characters. That gap is closed downstream
in `build_curated_word.py` instead, via on-demand row/col permutations
(see that module's own docstring). This script used to carry its own
large, fixed set of pre-generated permutation variants and a
brute-force seed search to guarantee 36/36 per-session coverage from
them -- unnecessary once permutations can be constructed on demand for
whichever single character is needed.

### The calibration pool's fixed 192-epoch target (`build_curated_word.py`)

`MIN_CALIBRATION_EPOCHS = 192` (two trials' worth, at this dataset's 96
epochs/trial) is a deliberately conservative, round two-trial baseline,
not a precisely optimized cutoff. Chosen because a single trial's 96
epochs is a real but thin baseline: relative error to a full-session
reference measured 3.7% at 96 epochs vs 2.2% at 192, part of a steady
convergence trend rather than a sharp cliff.

An earlier version of this target grew with the letter's position in
the word (192 at position 0, +96 per position after, unbounded) to
model a real subject's calibration set growing over a whole deployment
-- but restricted to one small session (6 trials, 576 epochs total)
that growth saturates to the entire session within only a handful of
letters, defeating the point of testing a realistically *small*
calibration budget. The target is now fixed instead: a real system
recalibrates on a rolling window, not one that grows forever, and this
fixed budget is what that looks like. Once a session runs out of trials
to borrow from, the borrowed contribution is capped at whatever's
actually left rather than erroring -- `aligner` only comes back `None`
in the degenerate case where a session has no other trials at all to
draw from.

### `DEFAULT_WORDS`/`DEFAULT_SENTENCES` design (`test_curated_words.py`)

`DEFAULT_WORDS` is the same list `scripts/simulate_live/livetesting.ipynb`
uses for its own `build_word()` smoke-testing (short/long, repeat-heavy,
digits, spaces). Each entry is tested independently, not joined into one
message -- concatenating unrelated short phrases produces incoherent
word-to-word transitions ("...GOOD JOB AGENT 47 ROOM 237...") that are
exactly as useless for testing a real character LM as BNCI2014_009's own
non-word fixed sequences.

`DEFAULT_SENTENCES` exists separately for testing `use_lm` as one
continuous message (context growing across real word boundaries, not
just within one word) -- coherent, grammatical English, not a
concatenation of test phrases, so the LM has genuine next-character
structure to exploit at every transition. Uppercase-only, no
punctuation (`FARWELL_DONCHIN_GRID` has no lowercase/punctuation cells).
Sourced from a real AAC message-set corpus, picked short (~20-30 chars,
deliberately shorter than an earlier ~76-char version of this list) to
keep any single sentence's LM-context-cascade failure window small -- a
wrong early decode poisons `should_decode()`'s prior for every later
letter in that sentence, since it's fed straight into the accumulating
`context`. `DEFAULT_SENTENCE` (the first entry) is kept first
specifically so any existing results/log line referencing "the first
sentence" still means the same text if the list is ever extended.

### EA is always session-scoped (`build_curated_word.py`)

Mixing sessions into one aligner would defeat the point of correcting
for session-specific drift -- different sessions, even of the same
subject, have real, measurable differences in their EA whitening
matrix.
