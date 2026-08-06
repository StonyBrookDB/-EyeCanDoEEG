"""eyecando: a P300 EEG speller pipeline, from raw recordings to decoded text.

A P300 speller flashes rows/columns of a character grid; a target
flash evokes a P300 event-related potential the classifier learns to
detect, and enough evidence accumulated across repetitions lets the
system decide which character the user is looking at. This package
covers the whole path from recorded/streamed EEG to a spelled message,
split into subpackages by pipeline stage:

- ``ingestion`` -- load EEG into a common in-memory representation
  (MOABB benchmark datasets, XDF/CSV recordings, a Muse 2 pilot
  dataset), each session carrying embedded ``stim_id``/``target_flag``
  stim channels in one canonical convention regardless of source.
- ``pipeline`` -- the offline training path: causal bandpass
  filtering, epoching, Euclidean Alignment (cross-subject covariance
  whitening), xDAWN spatial filtering, and LDA classification, plus
  the ``P300Model`` container that bundles a fitted xDAWN+LDA model
  and the metric primitives (ITR, accuracy/precision/recall/F1/AUC)
  used to evaluate it.
- ``decode`` -- turns per-flash classifier scores into decoded
  characters: row/column score accumulation, a confidence-threshold
  stopping rule, and an optional character-level language model
  (neural transformer or KenLM n-gram) that reweights the accumulator
  and can shorten how many repetitions are needed.
- ``live`` -- real-time inference: an LSL stream reader with a ring
  buffer and marker-triggered epoching, and a decode loop that wires
  streaming, alignment, the model, accumulation, and the language
  model together into a running session.
- ``tuning`` -- offline hyperparameter search and evaluation (nested
  leave-one-subject-out grid search, single-flash metrics, trial
  reconstruction/scoring, model caching) used to pick the settings the
  other stages ship with.
- ``utils`` -- small shared infrastructure (result-path layout, a
  training logger, an elapsed-time marker) with no pipeline-specific
  logic of its own.

Start in ``ingestion`` to load data, ``pipeline`` to see how a model
is trained, and ``live`` to see how a trained model is used in a real
session.
"""
