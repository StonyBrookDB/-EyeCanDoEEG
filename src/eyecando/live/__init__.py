"""Real-time inference: streaming EEG in, decoded symbols out.

- `streaming.py` -- `LSLEEGStream`, a three-thread LSL acquisition
  pipeline (continuous EEG inlet -> causal bandpass filter with
  carried filter state -> `EEGRingBuffer` -> marker-triggered epoch
  extraction) producing epochs in the same array format
  `ingestion.data` produces for training, so downstream code can't
  tell the two apart. Marker handling is split across dedicated
  reader/builder threads so a slow epoch build can never stall
  draining the marker inlet.
- `decoder.py` -- `LiveDecoder`, the orchestrator that wires
  `LSLEEGStream` (or any duck-typed epoch source, e.g.
  `PrebuiltEpochStream` for replaying curated data), `pipeline.
  alignment.EuclideanAligner`, `pipeline.model.P300Model`,
  `decode.accumulator.ScoreAccumulator`, `decode.lm.CharLM`, and
  `decode.stopping.should_decode()` into one running decode loop. Pure
  glue -- no new algorithmic logic of its own; also handles an
  optional initial calibration phase that fits the aligner from the
  session's own first epochs before scoring begins.

Everything scored/classified here is produced by `pipeline`/`decode`;
this subpackage's own job is only turning a live stream into the same
kind of epochs and score-accumulation calls those modules already
know how to handle.
"""
