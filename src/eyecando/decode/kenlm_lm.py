"""The KenLM n-gram backend for lm.py's CharLM (backend="kenlm") --
everything specific to scoring text through a KenLM trie instead of
LMDecoder's neural transformer lives here; CharLM itself decides which
backend to construct and exposes one public `.prior()`/`character_set`
interface regardless of which one is active. `_KenLMBackend` in this
file is a private implementation detail, not meant to be constructed
directly -- go through `CharLM(backend="kenlm")`.

Model: lm_dec19_char_large_12gram.kenlm (OSF: https://osf.io/ajm7t/
files/c6mnz, figmtu's "N-gram language models for AAC" project, CC BY
4.0), a character-level 12-gram KenLM trie. NOT vendored/committed here
(324MB decompressed) -- see scripts/setup_kenlm.sh for how to fetch and build
against it.

Tokenization this backend feeds the model: lowercased, space-joined,
one character per KenLM "word," with a `<sp>` token standing in for a
literal space and no coverage for digits. That convention was reverse-
engineered empirically (the model ships with no tokenization docs) --
see decode/README.md for the probes and evidence behind each part of
it before changing anything here. Scores are log10 internally
(`_LOG10_TO_LN` converts to natural log below) -- also confirmed
empirically, see the README.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_LOG10_TO_LN = math.log(10)
_SPACE_TOKEN = "<sp>"
# Out of vocabulary (digits -- see module docstring): a fixed, very small
# log10 score rather than -inf, so the softmax-style renormalization
# below stays well-defined instead of producing 0/0 for every symbol
# when a context has never been seen with any letter either.
_OOV_LOG10 = -20.0


def _stable_log_normalize(log_probs: np.ndarray) -> np.ndarray:
    """Natural-log-domain softmax normalization: exp(result).sum() == 1.
    Mirrors stopping.py's own softmax() (max-subtraction for numerical
    stability), just staying in log space throughout since the caller
    wants log-probs back, not linear ones."""
    shifted = log_probs - np.max(log_probs)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum


_DEFAULT_MODEL_PATH = "models/lm_dec19_char_large_12gram.kenlm"


class _KenLMBackend:
    """CharLM's private KenLM-backed implementation (backend="kenlm") --
    see module docstring for the tokenization convention this class
    implements. Constructed and owned by CharLM; not a public class.

    Same `prior(context, temperature)` normalization contract as
    CharLM's own neural path (natural-log probabilities over
    `character_set`, `np.exp(prior(...)).sum() == 1` at temperature=1.0)
    so CharLM.prior() can dispatch to either backend without the caller
    ever seeing a difference.
    """

    def __init__(
        self,
        character_set: list[str],
        model_path: str | Path = _DEFAULT_MODEL_PATH,
    ) -> None:
        """Construct a _KenLMBackend.

        Parameters
        ----------
        character_set : list[str]
            Symbols to predict over, in the same flat, row-major order
            as the speller grid's accumulator indices -- forwarded
            as-is from CharLM's own `character_set`.
        model_path : str or pathlib.Path
            Path to the built ``.kenlm`` binary (see module docstring
            for where to fetch it and `setup_kenlm.sh` for the required
            build). Defaults to the standard location under `models/`.

        Raises
        ------
        ImportError
            If the `kenlm` package isn't installed, or isn't the
            special `setup_kenlm.sh` build this 12-gram model needs
            (see module docstring).
        """
        # deferred: only users of this backend need the special build (see setup_kenlm.sh)
        import kenlm

        self.character_set = list(character_set)
        self._model = kenlm.Model(str(model_path))
        # Per-symbol KenLM token, or None for symbols this model's
        # training vocabulary has no representation for at all (digits).
        self._token_for_symbol: dict[str, str | None] = {}
        for sym in self.character_set:
            if sym == " ":
                self._token_for_symbol[sym] = _SPACE_TOKEN
            elif sym.isalpha():
                self._token_for_symbol[sym] = sym.lower()
            else:
                self._token_for_symbol[sym] = None

    def _context_text(self, context: str) -> str:
        """Lowercase, space -> <sp>, space-join into KenLM's own
        one-character-per-word convention."""
        tokens = [_SPACE_TOKEN if ch == " " else ch.lower() for ch in context]
        return " ".join(tokens)

    def prior(self, context: str, temperature: float = 1.0) -> np.ndarray:
        """Return log P(next_char | context) over self.character_set.

        Natural-log, normalized so `np.exp(result).sum() == 1` at
        temperature=1.0 -- same contract as CharLM's neural path. Every
        call is an independent KenLM trie lookup with no state carried
        between calls and no memoization (cheap enough not to need any;
        see decode/README.md) -- also why CharLM.reset_context() is a
        no-op when backend="kenlm".

        Parameters
        ----------
        context : str
            Text already decoded this message, in this project's own
            uppercase/digit/space alphabet -- lowercased and
            re-tokenized internally, never mixed case in what's handed
            to KenLM.
        temperature : float
            Scales the returned log-probs by ``1/temperature``.

        Returns
        -------
        numpy.ndarray
            Natural-log probabilities over `self.character_set`, in
            that order. Digit symbols (out of this model's training
            vocabulary -- see module docstring) get a fixed, small
            score rather than raising.
        """
        context_text = self._context_text(context)
        base_score = self._model.score(context_text, bos=True, eos=False)

        log10_scores = np.empty(len(self.character_set))
        for i, sym in enumerate(self.character_set):
            token = self._token_for_symbol[sym]
            if token is None:
                log10_scores[i] = _OOV_LOG10
                continue
            full_text = f"{context_text} {token}" if context_text else token
            log10_scores[i] = self._model.score(full_text, bos=True, eos=False) - base_score

        log_probs = _stable_log_normalize(log10_scores * _LOG10_TO_LN)
        if temperature != 1.0:
            return log_probs / temperature
        return log_probs
