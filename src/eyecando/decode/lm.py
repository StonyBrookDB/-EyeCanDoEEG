"""CharLM: the single public language-model class this project's decode
path constructs, over either of two interchangeable backends selected
by `backend=`.

- ``"neural"`` (the default) drives LMDecoder (lm_decoder.py -- the
  detached, permanently-forked wrapper around the real 350M-parameter
  transformer). CharLM maps its beam search onto the speller grid's flat
  symbol order and adds a temperature knob to scale the LM's influence
  independently of should_decode()'s (stopping.py) own softmax
  temperature.
- ``"kenlm"`` drives `_KenLMBackend` (kenlm_lm.py -- a KenLM n-gram
  trie; see that module's own docstring for the model source and
  tokenization convention).

Callers never see which backend is active: both implement the same
`prior(context, temperature)` / `reset_context()` contract -- see
decode/README.md for that contract's exact normalization guarantee and
for why the neural path additionally caches its raw forward pass.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from eyecando.decode.kenlm_lm import _DEFAULT_MODEL_PATH, _KenLMBackend
from eyecando.decode.lm_decoder import LMDecoder
from eyecando.utils.character_grid import FARWELL_DONCHIN_GRID

__all__ = ["FARWELL_DONCHIN_GRID", "CharLM"]

Backend = Literal["neural", "kenlm"]


class CharLM:
    """Character-level language model providing log-priors over the next symbol.

    Used by LiveDecoder (live/decoder.py) to reweight accumulated symbol
    scores before should_decode()'s (stopping.py) threshold check.
    """

    def __init__(
        self,
        character_set: list[str] | None = None,
        model_path: str | None = None,
        backend: Backend = "neural",
    ) -> None:
        """Construct a CharLM.

        Parameters
        ----------
        character_set : list[str] or None
            Symbols to predict over, in the same flat, row-major order
            as the speller grid's accumulator indices. None (the
            default) uses FARWELL_DONCHIN_GRID.
        model_path : str or None
            Optional local checkpoint path -- meaning depends on
            `backend`. For ``"neural"``, forwarded to LMDecoder's
            `lm_path`; None downloads/caches the default hosted model.
            For ``"kenlm"``, the path to the built ``.kenlm`` binary
            (see kenlm_lm.py's module docstring); None defaults to the
            standard location under `models/`.
        backend : {"neural", "kenlm"}
            Which underlying LM this instance drives -- see module
            docstring. ``"neural"`` (the default) preserves every
            existing caller's behavior exactly.

        Raises
        ------
        ImportError
            If `backend="kenlm"` and the `kenlm` package isn't
            installed, or isn't the special `setup_kenlm.sh` build the
            12-gram model needs (see kenlm_lm.py's module docstring).
        """
        self.character_set = (
            list(character_set) if character_set is not None else list(FARWELL_DONCHIN_GRID)
        )
        self.model_path = model_path
        self.backend: Backend = backend

        self._decoder: LMDecoder | None = None
        self._kenlm: _KenLMBackend | None = None
        # Raw (pre-temperature) forward-pass cache, keyed on context alone.
        # Neural backend only -- see decode/README.md for why this exists.
        self._raw_prior_cache: dict[str, np.ndarray] = {}

        if backend == "neural":
            self._decoder = LMDecoder(character_set=self.character_set, lm_path=model_path)
        elif backend == "kenlm":
            self._kenlm = _KenLMBackend(
                self.character_set, model_path if model_path is not None else _DEFAULT_MODEL_PATH
            )
        else:
            raise ValueError(f"backend must be 'neural' or 'kenlm', got {backend!r}")

    def prior(self, context: str, temperature: float = 1.0) -> np.ndarray:
        """Return log P(next_char | context) over self.character_set.

        Natural-log probabilities -- NOT linear. `temperature` divides
        the log-probs before returning; 1.0 is a no-op (leaves
        np.exp(result).sum() == 1), any other value breaks that
        normalization on purpose since the result is meant to be summed
        with unnormalized classifier scores next, not read as a
        standalone distribution.

        On the neural backend, results are memoized on `context` alone
        (see `_raw_prior_cache` and decode/README.md) and always
        returned as a fresh array, so mutating the result in place can't
        corrupt the cache. On the kenlm backend, every call is an
        independent trie lookup.

        Parameters
        ----------
        context : str
            Text already decoded this message (the model's left
            context). On the neural backend, calls with a `context`
            that's a growing extension of a previous call automatically
            reuse LMDecoder's internal cache -- call `reset_context()`
            first if `context` is unrelated to (not an extension of) the
            last call.
        temperature : float
            Scales the returned log-probs by ``1/temperature``.

        Returns
        -------
        numpy.ndarray
            Natural-log probabilities over `self.character_set`, in that
            order.
        """
        if self.backend == "kenlm":
            assert self._kenlm is not None  # set in __init__ whenever backend == "kenlm"
            return self._kenlm.prior(context, temperature)
        assert self._decoder is not None  # set in __init__ whenever backend == "neural"
        if context not in self._raw_prior_cache:
            self._raw_prior_cache[context] = self._decoder.predict_log_probs(context)
        log_probs = self._raw_prior_cache[context]
        if temperature != 1.0:
            return log_probs / temperature
        return log_probs.copy()

    def reset_context(self) -> None:
        """Drop the LM's internal incremental-position cache -- call when
        starting a new message.

        No-op on the kenlm backend, which builds no such cache. On the
        neural backend, safe to skip for correctness (a non-extension
        `context` still triggers a full recompute) but avoids holding
        onto now-irrelevant cache state between messages.

        Deliberately does NOT clear `_raw_prior_cache` -- see decode/
        README.md for why, and use `clear_prior_cache()` instead if that
        cache specifically needs reclaiming.
        """
        if self.backend == "neural":
            assert self._decoder is not None  # set in __init__ whenever backend == "neural"
            self._decoder.reset_cache()

    def clear_prior_cache(self) -> None:
        """Drop _raw_prior_cache. Separate from reset_context() on purpose
        (see decode/README.md) -- only rarely needed, e.g. between
        unrelated long-running test batches, to reclaim memory.
        No-op on the kenlm backend, which never builds this cache."""
        self._raw_prior_cache.clear()
