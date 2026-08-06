"""Dynamic stopping: decide whether accumulated evidence is confident
enough to decode a symbol yet, or another flash round is needed.

Its only eyecando import is accumulator.py's ScoreAccumulator -- pure
decision logic over whatever scores are already accumulated, with no
opinion on how they got there, and no access to ground truth (see
decode/README.md for how live and offline callers both drive this).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from eyecando.decode.accumulator import ScoreAccumulator


class _CharLM(Protocol):
    """Structural contract this module needs from a CharLM -- matches
    eyecando.decode.lm.CharLM, satisfied by any fake with the same shape."""

    character_set: list[str]

    def prior(self, context: str, temperature: float = 1.0) -> np.ndarray:
        """Return a log-prior over `character_set` for what follows `context`.

        Parameters
        ----------
        context : str
            Text already decoded this message, in this project's own
            character-set alphabet.
        temperature : float
            Scales the returned log-probs by ``1/temperature`` (see
            `eyecando.decode.lm.CharLM.prior`'s own docstring for the exact
            contract every real implementation follows).

        Returns
        -------
        numpy.ndarray
            Natural-log values, one per symbol in `character_set`, in
            that same order. Only guaranteed to sum to 1 in linear space
            at ``temperature=1.0``.
        """
        ...


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: exp(x - max(x)) normalized to sum to 1.

    Parameters
    ----------
    x : numpy.ndarray
        Raw scores (e.g. accumulated classifier scores, already divided
        by a temperature). Any shape; normalization is over the whole
        array, not per-row.

    Returns
    -------
    numpy.ndarray
        Same shape as `x`, non-negative, summing to 1.
    """
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def request_next(
    lm: _CharLM | None,
    idx: int,
    context: str,
    lm_temperature: float = 1.0,
) -> np.ndarray | None:
    """Build the next trial's accumulator prior after decoding symbol `idx`.

    `context` must already include the symbol just decoded (should_decode
    handles this); this function only forwards it, it doesn't append.

    Parameters
    ----------
    lm : CharLM or None
        Supplies the prior via `lm.prior()`. None means no LM is wired
        in for this session.
    idx : int
        Index of the symbol just decoded, into `lm.character_set` --
        accepted for interface symmetry with `should_decode`'s own call
        site but not otherwise used here (the caller already folds it
        into `context` before calling).
    context : str
        Text decoded so far, including the symbol just decoded at
        `idx`.
    lm_temperature : float
        Forwarded to `lm.prior()` -- scales the returned log-probs by
        ``1/lm_temperature``. 1.0 is a no-op.

    Returns
    -------
    numpy.ndarray or None
        A log-prior over `lm.character_set`, or None if `lm` is None.
    """
    if lm is None:
        return None
    return lm.prior(context, temperature=lm_temperature)


def should_decode(
    scoresAcc: ScoreAccumulator,
    threshold: float = 0.9,
    lm: _CharLM | None = None,
    context: str = "",
    temperature: float = 2.0,
    lm_temperature: float = 1.0,
) -> tuple[bool, int]:
    """Decide whether to decode or request another round.

    Optionally weights accumulated scores by LM prior before threshold check.

    Parameters
    ----------
    scoresAcc : ScoreAccumulator
        Holds this trial's accumulated classifier scores.
    threshold : float
        Confidence threshold. If softmax(scoresAcc.scores() / temperature)
        exceeds this at its max, decode. Default is a tuned value, not
        arbitrary -- see decode/README.md for the search that produced it.
    lm : CharLM or None
        Supplies a log-prior over the next symbol (via request_next()),
        seeded into the next trial's accumulator when decoding. If None,
        no prior is seeded.
    context : str
        Characters already decoded this message, *before* the symbol
        about to be decoded on this call. Only read when `lm` is set --
        the caller (e.g. LiveDecoder) owns tracking it across calls.
    temperature : float
        Softens (>1) or sharpens (<1) the softmax before the threshold
        check. 1.0 is a no-op. Never changes which symbol wins, only how
        much evidence is needed to cross `threshold`. Default is tuned
        alongside `threshold` -- see decode/README.md.
    lm_temperature : float
        Scales lm's log-prior independently of `temperature` above (which
        only affects the decode/no-decode check, not the prior seeded
        into the next trial). 1.0 is a no-op.

    Returns
    -------
    decode : bool
        True if confidence threshold is met.
    symbol_idx : int
        Index of the highest-scoring symbol (valid only if decode is True).
    """
    softmax_score = softmax(scoresAcc.scores() / temperature)

    decode = np.max(softmax_score) > threshold
    idx = int(np.argmax(softmax_score))
    if decode:
        new_context = context + lm.character_set[idx] if lm is not None else context
        scoresAcc.reset(request_next(lm, idx, new_context, lm_temperature))
    return decode, idx
