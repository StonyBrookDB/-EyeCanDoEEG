# eyecando.live — Stage 5: real hardware orchestration

Just the two pieces that actually talk to real (or replayed) hardware
over LSL and run the live loop -- the decode *policy* itself lives in
`eyecando.decode`, imported here rather than duplicated.

- **`streaming.py`** — `LSLEEGStream`: three-thread LSL acquisition
  (causal bandpass filter -> ring buffer -> marker-triggered epoching)
  producing epochs in the same format `eyecando.ingestion` produces for
  training, so downstream code can't tell the difference. `EEGRingBuffer`
  is the fixed-duration rolling buffer underneath it.
- **`decoder.py`** — `LiveDecoder`: turns a running epoch stream into a
  stream of decoded symbols, wiring together an `EuclideanAligner`
  (`eyecando.pipeline`), a `P300Model` (`eyecando.pipeline`), a
  `ScoreAccumulator` and `should_decode()` (`eyecando.decode`), and
  optionally a `CharLM` (`eyecando.decode.lm`, either backend).
  `PrebuiltEpochStream` (also here) replays an in-memory list of
  already-curated epochs through the same `LiveDecoder` with no real LSL
  connection -- what `scripts/curate/test_curated_words.py` uses.

Entry point: `scripts/simulate_live/run_live_session.py`. For testing this path
without real hardware, see `scripts/simulate_live/`.
