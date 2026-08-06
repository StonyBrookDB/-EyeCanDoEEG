# scripts/simulate_live/ — no-hardware LSL integration test harness

Validates `eyecando.live.streaming`'s actual LSL plumbing (real wall-clock
marker/EEG timing across independently-clocked streams) without a real
headset. This is a *plumbing* check, not a decode-correctness check --
see `scripts/curate/` for that (real recorded target trials, no live
marker timing involved at all).

- **`simulate_lsl_outlet.py`** — replays a real MOABB session as live LSL
  EEG + marker outlets. `--wait-for-trigger` holds replay until
  `trigger_lsl_start.py` fires, so consumers can connect first.
- **`simulate_flash_sequencer.py`** — the live-generator counterpart:
  produces a live sequence of row/col flashes (not a
  precomputed schedule), one real LSL marker pushed per flash as it's
  decided.
- **`consume_lsl_stream.py`** — connects a real `LSLEEGStream` to the
  outlet above and reports what it actually produced vs. the session's
  real flash count -- a manual integration check, not a pytest (needs a
  real LSL connection to a running outlet process).
- **`consume_flash_markers.py`** — the marker-only sibling: validates
  `simulate_flash_sequencer.py`'s output purely from the marker stream's
  own content (stim_id validity, per-round structure, ISI timing), no EEG
  side at all.
- **`trigger_lsl_start.py`** — sends the start signal so every
  `--wait-for-trigger` producer/consumer above begins together, instead
  of racing a fixed delay against however long each side takes to connect.

## Why these numbers: investigation notes

### The `trigger_lsl_start.py` re-push window (`consume_lsl_stream.py`)

`trigger_lsl_start.py`'s `--push-duration` exists because pylsl has no
API for a consumer *count* -- only "at least one"
(`lsl_wait_for_consumers()`/`lsl_have_consumers()`, confirmed by reading
pylsl's own source). A single one-shot push on the `EyeCanDoControl`
stream could reach whichever listener connects first and miss the
other entirely -- confirmed directly, and the reason a single-push
version of this flag was previously removed. Re-pushing over a window
instead of pushing once means a listener that connects a little after
the first one still catches a repeat: confirmed directly with two
listeners connecting 1s and 3.5s apart, both received the repeated
push. This is a mitigation, not a hard guarantee -- a listener that
starts connecting after the whole push window ends would still miss it
-- so `--get-epoch-timeout` on the consumer side is still worth keeping
generous as a backstop.
