"""Character-level next-symbol log-probabilities from a causal subword LM.

Adapted from textslinger 0.2.4 (https://github.com/kdv123/textslinger,
MIT license) -- specifically the character-prediction path of
causal_subword.py/language_model.py/exceptions.py/helpers.py, absorbed
into this single file and trimmed to only what LMDecoder needs
(predict_words()/score_item()/score_items() and everything reachable
only from them are gone; see git history for the original four-file,
lightly-trimmed vendoring if a fuller diff against upstream is ever
needed). This is a permanent fork, not something kept in sync with
upstream releases -- MIT permits modification, and re-vendoring a future
textslinger release would mean re-deriving this file from scratch, not
a file copy. See decode/README.md for the fuller history of this
decision, including two bugs worth knowing about before touching this
file.

MIT License

Copyright (c) 2026 Keith Vertanen, Dylan Gaines, Soufia Bahmani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Output is natural-log probabilities, not linear probabilities, even
after normalize_logprobs=True (the default) -- _format_character_predictions
below normalizes strictly within log-space and never exponentiates.
LMDecoder.predict_log_probs() is named and documented accordingly; it
does NOT convert to linear space. That conversion (or not) is a decision
for whichever accumulation loop consumes this output, not this module --
guessing wrong in either direction (double-converting, or feeding
log-probs into linear-space math) is a silent, easy-to-miss bug.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from heapq import heapify, heappush, heappushpop
from typing import Any, cast

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# Bounds the tiny HEAD/etag request huggingface_hub makes before using a
# from_pretrained() model that's already fully cached locally -- observed
# directly hanging ~19 minutes on an otherwise-healthy network (general
# connectivity to huggingface.co was fine; this was one stuck request) while
# running a component-ablation study's 4-way parallel smoke test (main
# branch's scripts/run_component_ablation.py, pruned from this branch).
# huggingface_hub already falls back to the local cache on a timeout here
# (this doesn't disable checking for updates, just bounds how long a stalled
# check can block model loading) -- setdefault so a caller can still widen
# or disable it. Read at from_pretrained() call time (inside LMDecoder's own
# __init__, further down this module), not at import time, so setting it
# here (after these imports, before any instance is constructed) is early
# enough.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")


class TextPredictException(Exception):
    """Base exception for language-model errors raised by this module."""

    def __init__(self, message: str, errors: object | None = None) -> None:
        """Construct a TextPredictException.

        Parameters
        ----------
        message : str
            Human-readable error message.
        errors : object or None
            Optional underlying error details.
        """
        super().__init__(message)
        self.message = message
        self.errors = errors


class InvalidLanguageModelException(TextPredictException):
    """Raised when a language model cannot be loaded from the requested path."""

    ...


class Device(str, Enum):
    """Torch device to load the language model onto.

    ``AUTO`` is resolved by `convert_device` at load time, not stored as
    a literal torch device string itself -- ``CPU``/``CUDA``/``MPS`` are
    passed straight through as-is.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

    @property
    def str(self) -> str:
        """This member's raw string value (e.g. ``"cuda"``).

        Returns
        -------
        str
            Same value `convert_device` would resolve `AUTO` away from --
            for `CPU`/`CUDA`/`MPS` this is just the enum's own value,
            never resolved further.
        """
        return self.value


class Precision(str, Enum):
    """Floating-point precision to load the language model's weights in.

    Passed to `precision_to_dtype` to get the corresponding
    `torch.dtype`. Lower precision trades numerical accuracy for less
    memory and (on supported hardware) faster inference; this project
    does not pick one for the caller -- see `LMDecoder.__init__`'s own
    default.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


PRECISION_TO_TORCH = {
    Precision.FP32: torch.float32,
    Precision.FP16: torch.float16,
    Precision.BF16: torch.bfloat16,
}


def convert_device(device: Device) -> str:
    """Map ``Device.AUTO`` to the best available backend.

    Parameters
    ----------
    device : Device
        Requested device enum. ``AUTO`` resolves to CUDA if available,
        else MPS if available, else CPU. Any other member passes
        through unresolved.

    Returns
    -------
    str
        String name of the resolved backend (e.g. ``"cuda"``,
        ``"mps"``, ``"cpu"``).
    """
    if device == Device.AUTO:
        if torch.cuda.is_available():
            resolved = Device.CUDA
        elif torch.backends.mps.is_available():
            resolved = Device.MPS
        else:
            resolved = Device.CPU
    else:
        resolved = device
    return resolved.str


def precision_to_dtype(precision: Precision) -> torch.dtype:
    """Convert a precision enum to the corresponding PyTorch dtype.

    Parameters
    ----------
    precision : Precision
        Precision enum to convert.

    Returns
    -------
    torch.dtype
        Torch dtype for model loading (``torch.float32``,
        ``torch.float16``, or ``torch.bfloat16``).

    Raises
    ------
    ValueError
        If `precision` isn't a recognized `Precision` member.
    """
    try:
        return PRECISION_TO_TORCH[precision]
    except KeyError:
        raise ValueError(f"Unsupported precision: {precision}")


@dataclass(slots=True)
class PredictCharactersResult:
    """Result returned by ``predict_characters``.

    Attributes
    ----------
    predictions : list[tuple[str, float]]
        ``(character, log_prob)`` pairs, one per configured output
        character. Natural-log, sorted descending by `log_prob` if
        `predict_characters` was called with ``sort_output=True``,
        otherwise in `character_set_lower`'s own order. A character no
        completed search path reached gets ``float("-inf")``.
    cache_state : object or None
        Opaque `SubwordPredictCache`, present only when
        `predict_characters` was called with ``return_cache_state=True``;
        otherwise None. Meant to be passed back in as that call's own
        `cache_state` argument, never inspected directly.
    """

    predictions: list[tuple[str, float]]
    cache_state: object | None = None


@dataclass(slots=True)
class ConfigPredictCharactersSubword:
    """Parameters that control the predict_characters method's beam search.

    Attributes
    ----------
    max_completed_hypotheses : int or None
        Stop the search once this many completed hypotheses (token
        paths that have crossed the end of the supplied context and so
        revealed a predicted character) have been found. None disables
        this limit.
    max_active_hypotheses : int or None
        Maximum number of incomplete hypotheses carried into each next
        round of beam expansion; the weakest are dropped once this
        many are held. None disables this limit (unbounded beam).
    """

    # Stop the search once we hit this many completed hypotheses, None = off
    max_completed_hypotheses: int | None = 32000
    # Maximum number of hypotheses to track during each extension of search, None = off
    max_active_hypotheses: int | None = 8


@dataclass(slots=True)
class SubwordPredictCache:
    """Reusable transformer state for character predictions over growing prefixes.

    Opaque to callers -- obtained from `PredictCharactersResult.cache_state`
    and passed back in as `predict_characters`'s own `cache_state`
    argument on the next call, never constructed or inspected directly.

    Attributes
    ----------
    left_context : str
        The `left_context` text this cache state was built from.
    token_ids : tuple[int, ...]
        Token IDs encoding `left_context` up to the final space,
        including the leading BOS/EOS token.
    past_key_values : Any
        The transformer's own KV cache for `token_ids`, in whatever
        object shape this model's `transformers` version returns.
    sorted_token_ids, sorted_log_probs : numpy.ndarray
        Vocabulary-ranked (descending probability) token IDs and their
        cumulative log-probs, aligned index-for-index.
    model_identity : int
        ``id()`` of the model this cache was built against -- a cache
        built from a different model instance is never reused, even if
        `token_ids` happens to match.
    """

    left_context: str
    token_ids: tuple[int, ...]
    past_key_values: Any
    sorted_token_ids: np.ndarray
    sorted_log_probs: np.ndarray
    model_identity: int


@dataclass(frozen=True, slots=True)
class _PreparedCharacterContext:
    """Context state required to initialize and constrain character search."""

    text: str
    lower_text: str
    target_position: int
    token_ids: list[int]
    decoded_length: int


_CharacterLogProbability = float | np.floating[Any]
_CharacterLogProbabilityArray = np.ndarray[Any, np.dtype[np.floating[Any]]]
_CharacterHypothesis = tuple[_CharacterLogProbability, list[int], int]


def _character_logsumexp(
    values: Sequence[_CharacterLogProbability] | _CharacterLogProbabilityArray,
) -> float:
    """Return a stable log-sum-exp with low overhead for character path scores."""
    if len(values) == 0:
        return float("-inf")

    value_array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(value_array))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(value_array - maximum).sum()))


@dataclass(frozen=True, slots=True)
class _CharacterExpansionResult:
    """State produced by expanding one batch of active hypotheses."""

    next_hypotheses: list[_CharacterHypothesis]
    completed: int
    done: bool


@dataclass(frozen=True, slots=True)
class _CharacterCandidateBatch:
    """Aligned candidate arrays after applying the completed-path cutoff."""

    token_ids: np.ndarray
    log_probs: np.ndarray
    token_lengths: np.ndarray
    completed_mask: np.ndarray
    reached_completed_limit: bool


class CausalSubwordLanguageModel:
    """Character LM based on a pre-trained causal transformer, subword-tokenized.

    Does the actual work `LMDecoder` (this module's public wrapper) exposes
    as a single `predict_log_probs()` call: the model itself only ever
    tokenizes text into *subword* tokens, so getting a probability for the
    single next *character* means running a bounded beam search over token
    paths (`predict_characters()`) until enough of them cross a character
    boundary to marginalize out a per-character distribution -- see
    `_expand_character_hypotheses`/`_character_logsumexp` and the
    `Config`/`_Prepared*`/`_Character*` helper types above for the pieces
    that search is built from. Holds the loaded tokenizer/model plus the
    vocabulary tables (`vocab`, `index_to_word*`, `valid_vocab*`) derived
    from `character_set` once at construction (or at `change_character_set()`),
    and reuses transformer inference state across calls via
    `SubwordPredictCache` so a growing left-context doesn't re-run the whole
    prefix every time.
    """

    def __init__(
        self,
        character_set: list[str],
        lang_model_name: str,
        lm_path: str | None = None,
        device: Device = Device.AUTO,
        precision: Precision = Precision.FP32,
        lora_path: str = "",
        trust_remote_code: bool = False,
    ):
        """Construct an instance of CausalSubwordLanguageModel.

        Loads the tokenizer and model eagerly (from `lm_path` or
        `lora_path` if given, else from the Hugging Face Hub by
        `lang_model_name`) and builds the character-prediction vocabulary
        tables for `character_set` -- this is not a cheap constructor.

        Parameters
        ----------
        character_set : list[str]
            Characters to make predictions over.
        lang_model_name : str
            Name of the Hugging Face causal language model to load.
        lm_path : str or None
            Local directory containing a fine-tuned model, if any. None
            loads `lang_model_name` from the Hugging Face Hub (or its
            local cache) instead.
        device : Device
            Device to use for inference; ``AUTO`` resolves via
            `convert_device`.
        precision : Precision
            Precision to load the model with.
        lora_path : str
            LoRA adapter to load from Hugging Face or a local directory.
            Empty string (the default) loads the base model directly,
            without any adapter.
        trust_remote_code : bool
            Allow trusted model and tokenizer code from Hugging Face.

        Raises
        ------
        InvalidLanguageModelException
            If the tokenizer or model can't be loaded from the given
            name/path, if the model's output vocabulary is smaller than
            the tokenizer's, or if the tokenizer has no integer BOS or
            EOS token to seed the search with.
        """
        self.character_set = character_set
        self.model = None
        # self.tokenizer and self.character_set_lower are not given
        # placeholder values here -- both are always set for real before
        # __init__ returns (tokenizer a few lines down; character_set_lower
        # via change_character_set() at the end of this method) and neither
        # is read in between, so an interim None would only widen their
        # static type for every reader without ever being a real runtime
        # value.
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = defaultdict(list)
        self.index_to_word = []
        self.index_to_word_lower = []
        self.left_context_tokens: list[int] = []
        self.device = convert_device(device)
        self.lora_path = lora_path

        # Hash set versions that we'll create that let us quickly check token IDs against our entire
        # valid set, or in a subset based on a text prefix.
        self.valid_vocab_set = None
        self.vocab_set = defaultdict(set)

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model
        self.model_name = lang_model_name
        self.model_dir = lm_path if lm_path else self.model_name

        # Track how much time spent in different parts of the predict function
        self.predict_total_ns = 0
        self.predict_inference_ns = 0

        try:
            # AutoTokenizer is a dynamic factory whose return type is not
            # understood consistently by static type checkers.
            tokenizer_factory_object = cast(object, AutoTokenizer.from_pretrained)
            tokenizer_factory = cast(
                Callable[..., PreTrainedTokenizerBase], tokenizer_factory_object
            )
            self.tokenizer: PreTrainedTokenizerBase = tokenizer_factory(
                self.model_name,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise InvalidLanguageModelException(
                f"{self.model_name} is not a valid model identifier on HuggingFace."
            ) from e

        # Include added tokens because encode() may return their IDs even though
        # tokenizer.vocab_size reports only the base vocabulary.
        self.vocab_size = len(self.tokenizer)
        self._special_ids_set = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        self._allowed_first_tensor_cache: dict[
            tuple[str, bool, int, str],
            torch.Tensor,
        ] = {}

        try:
            if self.lora_path:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.lora_path,
                    device_map=self.device,
                    dtype=precision_to_dtype(precision),
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    device_map=self.device,
                    dtype=precision_to_dtype(precision),
                    trust_remote_code=trust_remote_code,
                )
        except Exception as e:
            raise InvalidLanguageModelException(
                f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace."
            ) from e

        self.model.eval()

        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            output_vocab_size = output_embeddings.weight.shape[0]
            if self.vocab_size > output_vocab_size:
                raise InvalidLanguageModelException(
                    f"Tokenizer has {self.vocab_size} tokens, but model output "
                    f"vocabulary has only {output_vocab_size}. Resize the model "
                    "embeddings after adding tokenizer tokens."
                )

        self.change_character_set(character_set)

    def change_character_set(self, character_set: list[str]) -> None:
        """Change the active character set of the model.

        Build vocabulary tables mapping token indexes to decoded token strings.

        Parameters
        ----------
        character_set : list[str]
            Character strings to predict over, replacing whatever set
            this instance was constructed or last configured with. Also
            rebuilds the leading BOS/EOS search seed, so this is safe to
            call again later, not just once from `__init__`.

        Raises
        ------
        InvalidLanguageModelException
            If the tokenizer has no integer BOS or EOS token to seed
            the search with.
        """
        self.character_set = character_set

        self.character_set_lower = []
        for ch in self.character_set:
            self.character_set_lower.append(ch.lower())
        self._character_to_index = {
            character: index for index, character in enumerate(self.character_set_lower)
        }

        self.valid_vocab = []
        self.vocab = defaultdict(list)
        self.index_to_word = []
        self.index_to_word_lower = []
        self.valid_vocab_set = None
        self.vocab_set = defaultdict(set)

        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build vocabulary tables mapping token indexes to decoded token strings."""

        # Loop over all the subword tokens in the LLM
        for i in range(self.vocab_size):
            # Map the subword token integer ID to its mixed- and lowercase string
            # decode()'s signature is broadened to `str | list[str]` to also
            # support batch decoding via list[list[int]], but a single-element
            # list[int] argument (as passed here) always returns a plain str.
            word = cast(str, self.tokenizer.decode([i]))
            word_lower = word.lower()
            self.index_to_word.append(word)
            self.index_to_word_lower.append(word_lower)

            # Empty decoded tokens cannot advance a character hypothesis and can
            # otherwise keep the beam search cycling at the same text position.
            if not word_lower:
                continue

            # Check if all the characters in the subword token are in our valid character set
            valid = True
            for ch in word_lower:
                if ch not in self.character_set_lower:
                    valid = False
                    break

            # If the subword token's characters are all valid, keep its token ID
            if valid:
                self.valid_vocab.append(i)
                # Add this token ID to all lists for its valid text prefixes
                for j in range(len(word)):
                    key = word_lower[0 : j + 1]
                    self.vocab[key].append(i)
                    # Construct set for prefix of the word
                    self.vocab_set[key].add(i)

        # Hash set of the vocab indexes for quick testing if token ID is in our entire valid set
        self.valid_vocab_set = set(self.valid_vocab)
        # Most character expansions have consumed the supplied context and may
        # choose any token valid for the configured character set. Build that
        # vocabulary-sized eligibility mask once instead of reconstructing it
        # from roughly the entire vocabulary for every such hypothesis.
        self._valid_vocab_lookup = np.zeros(self.vocab_size, dtype=np.bool_)
        self._valid_vocab_lookup[np.asarray(self.valid_vocab, dtype=np.intp)] = True

        # NumPy views support bulk completion tests during character expansion.
        # Character-index-at-offset arrays are populated lazily because most
        # searches need only a small number of token offsets.
        self._token_text_lengths = np.fromiter(
            (len(text) for text in self.index_to_word_lower),
            dtype=np.int32,
            count=self.vocab_size,
        )
        self._token_character_indices_by_offset: dict[int, np.ndarray] = {}

        # self.vocab maps text to possible following subword tokens, e.g.:
        # self.vocab["cyclo"] = [47495, 49484]
        # self.index_to_word[self.vocab["cyclo"][0]] = cyclop
        # self.index_to_word[self.vocab["cyclo"][1]] = cyclopedia
        # Construct the initial sequence explicitly so tokenizer-specific
        # special-token insertion cannot add zero or multiple BOS tokens.
        start_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if not isinstance(start_token_id, int):
            raise InvalidLanguageModelException(
                f"{self.model_name} tokenizer has no integer BOS or EOS token."
            )

        self.left_context_tokens = [start_token_id]

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _format_character_predictions(
        self,
        char_to_log_probs: Mapping[str, Sequence[_CharacterLogProbabilityArray]],
        *,
        normalize_logprobs: bool,
        sort_output: bool,
    ) -> PredictCharactersResult:
        """Marginalize token paths and construct the character prediction result."""
        # Keep one entry per configured output character, using -inf for
        # characters that no completed token path reached.
        char_probs: np.ndarray[Any, np.dtype[np.float64]] = np.full(
            len(self.character_set_lower),
            float("-inf"),
            dtype=np.float64,
        )
        for index, target_ch in enumerate(self.character_set_lower):
            path_log_prob_chunks = char_to_log_probs.get(target_ch)
            if path_log_prob_chunks:
                # Search stores scores in their original NumPy chunks instead
                # of materializing millions of Python floats. Concatenating in
                # insertion order preserves the previous final reduction order.
                if len(path_log_prob_chunks) == 1:
                    path_log_probs = path_log_prob_chunks[0]
                else:
                    path_log_probs = np.concatenate(path_log_prob_chunks)
                # The specialized reduction avoids repeated SciPy setup while
                # retaining stable log-space marginalization of token paths.
                char_probs[index] = _character_logsumexp(path_log_probs)

        if normalize_logprobs:
            # Normalize in log space. Preserve an empty all--inf support rather
            # than producing NaNs from subtracting -inf from -inf.
            normalizer = _character_logsumexp(char_probs)
            if normalizer != float("-inf"):
                # Update the known float64 array in place. Besides avoiding a
                # temporary, this keeps static type inference from widening the
                # result of NumPy's overloaded subtraction to scalar/array unions.
                char_probs -= normalizer

        predictions = [
            (character, float(log_prob))
            for character, log_prob in zip(self.character_set_lower, char_probs)
        ]
        if sort_output:
            predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
        return PredictCharactersResult(predictions=predictions)

    def _prepare_character_context(self, left_context: str) -> _PreparedCharacterContext:
        """Back off to the final space and prepare the initial search sequence."""
        lower_text = left_context.lower()
        target_position = len(left_context)

        # Encode only the complete context before the final space. Search will
        # reconstruct the remaining partial word from compatible token paths.
        final_space = left_context.rfind(" ")
        token_ids = list(self.left_context_tokens)
        if final_space >= 0:
            token_ids.extend(self._encode(left_context[:final_space]))

        # Search lengths count decoded text after the fixed BOS/EOS prefix.
        decoded_length = 0
        for token_id in token_ids[len(self.left_context_tokens) :]:
            decoded_length += len(self.index_to_word_lower[token_id])

        return _PreparedCharacterContext(
            text=left_context,
            lower_text=lower_text,
            target_position=target_position,
            token_ids=token_ids,
            decoded_length=decoded_length,
        )

    def _generate_character_candidates(
        self,
        prepared_context: _PreparedCharacterContext,
        decoded_length: int,
    ) -> tuple[set[int], set[int]]:
        """Find tokens that can extend one hypothesis through the typed context."""
        vocab_set: set[int] = set()
        extra_vocab_set: set[int] = set()
        remaining_context = prepared_context.lower_text[decoded_length:]

        if not remaining_context:
            # Once the hypothesis has consumed the context, any token valid for
            # the configured character set can supply the predicted character.
            vocab_set = cast(set[int], self.valid_vocab_set)
        else:
            if remaining_context in self.vocab:
                # Include tokens equal to or extending the remaining context.
                vocab_set = self.vocab_set[remaining_context]

            # A token may consume only an initial portion of the remaining
            # context. Tokenize each proper prefix to recover those candidates.
            for prefix_length in range(1, len(remaining_context)):
                tokenization = self._encode(
                    prepared_context.text[decoded_length : decoded_length + prefix_length]
                )
                # Multi-token encodings contain a token already found through a
                # shorter prefix, so only retain single-token encodings here.
                if len(tokenization) == 1:
                    extra_vocab_set.add(tokenization[0])

        return vocab_set, extra_vocab_set

    def _token_character_indices_at_offset(self, offset: int) -> np.ndarray:
        """Return each token's output-character index at a decoded offset."""
        cached = self._token_character_indices_by_offset.get(offset)
        if cached is not None:
            return cached

        character_indices = np.fromiter(
            (
                self._character_to_index.get(text[offset], -1) if len(text) > offset else -1
                for text in self.index_to_word_lower
            ),
            dtype=np.int16,
            count=self.vocab_size,
        )
        self._token_character_indices_by_offset[offset] = character_indices
        return character_indices

    def _infer_character_hypotheses(
        self,
        current_hypotheses: Sequence[_CharacterHypothesis],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one batched inference and return aligned ranked token IDs and scores."""
        token_sequences = [sequence for _, sequence, _ in current_hypotheses]
        tokens_tensor = (
            torch.tensor(token_sequences).reshape(len(current_hypotheses), -1).to(self.device)
        )
        hypothesis_log_probs = [log_prob for log_prob, _, _ in current_hypotheses]
        sorted_token_ids, sorted_log_probs, _ = self._rank_character_tokens(
            tokens_tensor,
            hypothesis_log_probs,
            use_cache=False,
        )
        return sorted_token_ids, sorted_log_probs

    def _rank_character_tokens(
        self,
        tokens_tensor: torch.Tensor,
        hypothesis_log_probs: Sequence[_CharacterLogProbability],
        *,
        use_cache: bool,
        past_key_values: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Any | None]:
        """Infer and rank next tokens, optionally extending an inter-call cache."""
        before_inference_ns = time.time_ns()
        with torch.inference_mode():
            logits, next_past_key_values = self._forward_character_tokens(
                tokens_tensor,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            cumulative_log_probs = self._score_character_tokens(
                logits,
                hypothesis_log_probs,
            )
            sorted_token_ids, sorted_log_probs = self._sort_character_tokens(cumulative_log_probs)
            sorted_token_ids_cpu, sorted_log_probs_cpu = self._copy_character_token_ranking_to_cpu(
                sorted_token_ids,
                sorted_log_probs,
            )

        self.predict_inference_ns += time.time_ns() - before_inference_ns
        return (
            sorted_token_ids_cpu,
            sorted_log_probs_cpu,
            next_past_key_values,
        )

    def _forward_character_tokens(
        self,
        tokens_tensor: torch.Tensor,
        *,
        use_cache: bool,
        past_key_values: Any | None,
    ) -> tuple[torch.Tensor, Any | None]:
        """Run the transformer and return only next-token logits and cache state."""
        model = self.model
        assert model is not None, "language model does not exist!"

        model_arguments: dict[str, Any] = {
            "use_cache": use_cache,
            # Character search needs only the distribution after the final
            # input position, so avoid materializing logits for earlier tokens.
            "logits_to_keep": 1,
        }
        if past_key_values is not None:
            model_arguments["past_key_values"] = past_key_values

        model_output = model(tokens_tensor, **model_arguments)
        next_past_key_values = getattr(model_output, "past_key_values", None) if use_cache else None
        return model_output.logits[:, -1, :], next_past_key_values

    def _score_character_tokens(
        self,
        logits: torch.Tensor,
        hypothesis_log_probs: Sequence[_CharacterLogProbability],
    ) -> torch.Tensor:
        """Convert logits to cumulative path log probabilities."""
        log_probs = torch.log_softmax(logits, dim=1)
        hypothesis_scores = torch.tensor(hypothesis_log_probs).reshape(-1, 1).to(self.device)
        return torch.add(log_probs, hypothesis_scores)

    @staticmethod
    def _sort_character_tokens(
        cumulative_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token IDs and scores aligned in descending probability order."""
        # Sorting both outputs together avoids looking each selected score up by
        # vocabulary ID later during candidate filtering.
        sorted_log_probs, sorted_token_ids = torch.sort(
            cumulative_log_probs,
            descending=True,
            dim=1,
        )
        return sorted_token_ids, sorted_log_probs

    @staticmethod
    def _copy_character_token_ranking_to_cpu(
        sorted_token_ids: torch.Tensor,
        sorted_log_probs: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Copy aligned token rankings to NumPy for CPU-side beam expansion."""
        return (
            sorted_token_ids.detach().cpu().numpy(),
            sorted_log_probs.detach().cpu().numpy(),
        )

    @staticmethod
    def _fork_character_cache(past_key_values: Any) -> Any | None:
        """Fork a cache cheaply by sharing immutable tensors between calls."""
        if isinstance(past_key_values, tuple):
            # Legacy tuple caches are immutable; models return new tuples when
            # appending tokens, so sharing the existing tensor references is safe.
            return past_key_values

        layers = getattr(past_key_values, "layers", None)
        if isinstance(layers, list):
            # DynamicCache updates replace each layer's key/value tensors. Copy
            # the small Python containers while sharing the existing tensors so
            # callers can safely reuse an older SubwordPredictCache state.
            forked = copy(past_key_values)
            forked.layers = [copy(layer) for layer in layers]
            return forked

        return None

    def _infer_character_prefix(
        self,
        prepared_context: _PreparedCharacterContext,
        cache_state: SubwordPredictCache | None,
        *,
        return_cache_state: bool,
        left_context: str,
    ) -> tuple[np.ndarray, np.ndarray, SubwordPredictCache | None]:
        """Infer the fixed context prefix, reusing or extending prior call state."""
        token_ids = tuple(prepared_context.token_ids)
        model_identity = id(self.model)
        compatible_cache = (
            cache_state is not None
            and cache_state.model_identity == model_identity
            and len(cache_state.token_ids) <= len(token_ids)
            and token_ids[: len(cache_state.token_ids)] == cache_state.token_ids
        )

        if compatible_cache:
            assert cache_state is not None
            if token_ids == cache_state.token_ids:
                updated_cache = None
                if return_cache_state:
                    updated_cache = SubwordPredictCache(
                        left_context=left_context,
                        token_ids=token_ids,
                        past_key_values=cache_state.past_key_values,
                        sorted_token_ids=cache_state.sorted_token_ids,
                        sorted_log_probs=cache_state.sorted_log_probs,
                        model_identity=model_identity,
                    )
                return (
                    cache_state.sorted_token_ids,
                    cache_state.sorted_log_probs,
                    updated_cache,
                )

        past_key_values = None
        inference_token_ids = token_ids
        if compatible_cache:
            assert cache_state is not None
            forked_cache = self._fork_character_cache(cache_state.past_key_values)
            if forked_cache is not None:
                past_key_values = forked_cache
                inference_token_ids = token_ids[len(cache_state.token_ids) :]

        use_cache = return_cache_state or past_key_values is not None
        tokens_tensor = torch.tensor(inference_token_ids).reshape(1, -1).to(self.device)
        sorted_token_ids, sorted_log_probs, next_past_key_values = self._rank_character_tokens(
            tokens_tensor,
            [0.0],
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        updated_cache = None
        if return_cache_state:
            if next_past_key_values is None:
                raise TypeError("language model did not return a KV cache")
            updated_cache = SubwordPredictCache(
                left_context=left_context,
                token_ids=token_ids,
                past_key_values=next_past_key_values,
                sorted_token_ids=sorted_token_ids,
                sorted_log_probs=sorted_log_probs,
                model_identity=model_identity,
            )
        return sorted_token_ids, sorted_log_probs, updated_cache

    def _filter_ranked_character_candidates(
        self,
        ranked_token_ids: np.ndarray,
        ranked_log_probs: np.ndarray,
        vocab_set: set[int],
        extra_vocab_set: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retain eligible token IDs and aligned scores in probability order."""
        # The lookup is indexed by token ID. Indexing it with probability-ranked
        # IDs produces an eligibility mask in that same ranking order.
        if vocab_set is self.valid_vocab_set and not extra_vocab_set:
            if len(self._valid_vocab_lookup) != len(ranked_log_probs):
                # Some output heads contain reserved rows beyond len(tokenizer).
                # Pad those rows as ineligible once, then reuse the model-width
                # mask for subsequent unrestricted expansions.
                valid_vocab_lookup = np.zeros(
                    len(ranked_log_probs),
                    dtype=np.bool_,
                )
                valid_vocab_lookup[np.asarray(self.valid_vocab, dtype=np.intp)] = True
                self._valid_vocab_lookup = valid_vocab_lookup
            candidate_lookup = self._valid_vocab_lookup
        else:
            candidate_lookup = np.zeros(len(ranked_log_probs), dtype=np.bool_)
            candidate_lookup[np.fromiter(vocab_set, dtype=np.intp)] = True
            candidate_lookup[np.fromiter(extra_vocab_set, dtype=np.intp)] = True
        ranked_candidate_mask = candidate_lookup[ranked_token_ids]
        return (
            ranked_token_ids[ranked_candidate_mask],
            ranked_log_probs[ranked_candidate_mask],
        )

    def _limit_completed_character_candidates(
        self,
        candidate_token_ids: np.ndarray,
        candidate_log_probs: np.ndarray,
        *,
        current_decoded_length: int,
        context_length: int,
        completed: int,
        max_completed_hypotheses: int | None,
    ) -> _CharacterCandidateBatch:
        """Classify candidates and stop at the global completed-path limit."""
        candidate_lengths = self._token_text_lengths[candidate_token_ids]
        completed_mask = current_decoded_length + candidate_lengths > context_length
        stop_position = len(candidate_token_ids)
        reached_completed_limit = False

        if max_completed_hypotheses is not None:
            remaining_completions = max_completed_hypotheses - completed
            if remaining_completions <= 0:
                stop_position = 0
                reached_completed_limit = True
            else:
                completed_positions = np.flatnonzero(completed_mask)
                if len(completed_positions) >= remaining_completions:
                    stop_position = int(completed_positions[remaining_completions - 1]) + 1
                    reached_completed_limit = True

        return _CharacterCandidateBatch(
            token_ids=candidate_token_ids[:stop_position],
            log_probs=candidate_log_probs[:stop_position],
            token_lengths=candidate_lengths[:stop_position],
            completed_mask=completed_mask[:stop_position],
            reached_completed_limit=reached_completed_limit,
        )

    def _group_completed_character_scores(
        self,
        *,
        prepared_context: _PreparedCharacterContext,
        current_decoded_length: int,
        candidates: _CharacterCandidateBatch,
        char_to_log_probs: dict[str, list[_CharacterLogProbabilityArray]],
    ) -> int:
        """Append completed path scores to groups for their revealed character."""
        if not np.any(candidates.completed_mask):
            return 0

        completed_token_ids = candidates.token_ids[candidates.completed_mask]
        completed_log_probs = candidates.log_probs[candidates.completed_mask]

        # A token starts at current_decoded_length, so this relative offset
        # identifies the character lying at the prediction target position.
        target_offset = prepared_context.target_position - current_decoded_length
        target_character_indices = self._token_character_indices_at_offset(target_offset)[
            completed_token_ids
        ]
        supported_mask = target_character_indices >= 0
        if np.any(supported_mask):
            supported_indices = target_character_indices[supported_mask]
            supported_log_probs = completed_log_probs[supported_mask]

            # Stable integer sorting forms contiguous character groups without
            # changing score order within a group, preserving reference values.
            stable_order = np.argsort(supported_indices, kind="stable")
            grouped_indices = supported_indices[stable_order]
            grouped_log_probs = supported_log_probs[stable_order]
            group_boundaries = np.flatnonzero(grouped_indices[1:] != grouped_indices[:-1]) + 1
            group_start = 0
            group_ends = group_boundaries.tolist()
            group_ends.append(len(grouped_indices))
            for group_end in group_ends:
                character_index = int(grouped_indices[group_start])
                target_character = self.character_set_lower[character_index]
                char_to_log_probs[target_character].append(grouped_log_probs[group_start:group_end])
                group_start = group_end

        return len(completed_token_ids)

    @staticmethod
    def _retain_active_character_candidates(
        *,
        candidates: _CharacterCandidateBatch,
        current_sequence: list[int],
        current_decoded_length: int,
        max_active_hypotheses: int | None,
        pending_hypotheses: list[_CharacterHypothesis],
    ) -> None:
        """Update the bounded next-round heap with incomplete candidate paths."""
        # Active candidates are comparatively rare, and ordered beam replacement
        # is inherently scalar. The min-heap root is the weakest retained path.
        active_positions = np.flatnonzero(~candidates.completed_mask)
        for active_position in active_positions:
            token_id = int(candidates.token_ids[active_position])
            candidate_log_prob = cast(
                _CharacterLogProbability,
                candidates.log_probs[active_position],
            )
            new_hypothesis: _CharacterHypothesis = (
                candidate_log_prob,
                current_sequence + [token_id],
                current_decoded_length + int(candidates.token_lengths[active_position]),
            )
            if max_active_hypotheses is None or len(pending_hypotheses) < max_active_hypotheses:
                heappush(pending_hypotheses, new_hypothesis)
            elif candidate_log_prob > pending_hypotheses[0][0]:
                heappushpop(pending_hypotheses, new_hypothesis)

    def _expand_character_hypotheses(
        self,
        *,
        prepared_context: _PreparedCharacterContext,
        current_hypotheses: Sequence[_CharacterHypothesis],
        sorted_token_ids: np.ndarray,
        sorted_log_probs: np.ndarray,
        config: ConfigPredictCharactersSubword,
        char_to_log_probs: dict[str, list[_CharacterLogProbabilityArray]],
        completed: int,
    ) -> _CharacterExpansionResult:
        """Expand one search round and apply completed and active pruning limits.

        Each input hypothesis has decoded some prefix of ``prepared_context.text``.
        An eligible next token either remains inside that known context (an active
        hypothesis) or crosses its end and therefore reveals the next character
        being predicted (a completed hypothesis). Completed path scores are
        accumulated by revealed character; active paths compete for the next
        round's bounded beam.

        Candidate rows arrive in descending probability order. That ordering is
        part of the pruning semantics and must be retained through filtering and
        the completed-hypothesis cutoff.
        """
        pending_hypotheses: list[_CharacterHypothesis] = []
        done = False

        for current_index, current in enumerate(current_hypotheses):
            _, current_sequence, current_decoded_length = current

            # Phase 1: determine which vocabulary tokens agree with the portion
            # of the supplied context that this hypothesis has not consumed.
            vocab_set, extra_vocab_set = self._generate_character_candidates(
                prepared_context,
                current_decoded_length,
            )

            # Phase 2: retain eligible entries from this hypothesis's ranked
            # vocabulary row while keeping token IDs aligned with their scores.
            candidate_token_ids, candidate_log_probs = self._filter_ranked_character_candidates(
                sorted_token_ids[current_index],
                sorted_log_probs[current_index],
                vocab_set,
                extra_vocab_set,
            )

            # Phase 3: classify paths as completed or active and truncate at the
            # global completed-path limit without disturbing probability order.
            candidates = self._limit_completed_character_candidates(
                candidate_token_ids,
                candidate_log_probs,
                current_decoded_length=current_decoded_length,
                context_length=len(prepared_context.text),
                completed=completed,
                max_completed_hypotheses=config.max_completed_hypotheses,
            )
            done = done or candidates.reached_completed_limit

            # Phase 4: group completed scores by the output character revealed
            # where each token crosses the end of the supplied context.
            completed += self._group_completed_character_scores(
                prepared_context=prepared_context,
                current_decoded_length=current_decoded_length,
                candidates=candidates,
                char_to_log_probs=char_to_log_probs,
            )

            # Phase 5: retain the strongest incomplete paths for the next round.
            self._retain_active_character_candidates(
                candidates=candidates,
                current_sequence=current_sequence,
                current_decoded_length=current_decoded_length,
                max_active_hypotheses=config.max_active_hypotheses,
                pending_hypotheses=pending_hypotheses,
            )

            if (
                config.max_completed_hypotheses is not None
                and completed >= config.max_completed_hypotheses
            ):
                done = True
                break

        return _CharacterExpansionResult(
            next_hypotheses=pending_hypotheses,
            completed=completed,
            done=done,
        )

    def _search_character_hypotheses(
        self,
        prepared_context: _PreparedCharacterContext,
        config: ConfigPredictCharactersSubword,
        initial_ranked_tokens: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, list[_CharacterLogProbabilityArray]]:
        """Run character beam search and collect completed path probabilities."""
        current_hypotheses: list[_CharacterHypothesis] = [
            (
                0.0,
                prepared_context.token_ids,
                prepared_context.decoded_length,
            )
        ]

        # Retain heap semantics for active-beam replacement. Sorting each round
        # ensures completed pruning considers higher-probability paths first.
        heapify(current_hypotheses)
        char_to_log_probs: dict[str, list[_CharacterLogProbabilityArray]] = defaultdict(list)
        completed = 0
        done = False

        while current_hypotheses and not done:
            current_hypotheses.sort(reverse=True)
            if initial_ranked_tokens is None:
                sorted_token_ids, sorted_log_probs = self._infer_character_hypotheses(
                    current_hypotheses,
                )
            else:
                sorted_token_ids, sorted_log_probs = initial_ranked_tokens
                initial_ranked_tokens = None
            expansion = self._expand_character_hypotheses(
                prepared_context=prepared_context,
                current_hypotheses=current_hypotheses,
                sorted_token_ids=sorted_token_ids,
                sorted_log_probs=sorted_log_probs,
                config=config,
                char_to_log_probs=char_to_log_probs,
                completed=completed,
            )
            current_hypotheses = expansion.next_hypotheses
            completed = expansion.completed
            done = expansion.done

        return char_to_log_probs

    def predict_characters(
        self,
        left_context: str,
        *,
        config: ConfigPredictCharactersSubword = ConfigPredictCharactersSubword(),
        normalize_logprobs: bool = True,
        sort_output: bool = True,
        cache_state: object | None = None,
        return_cache_state: bool = False,
    ) -> PredictCharactersResult:
        """Predict the next-character distribution for a text context.

        Runs a beam search over subword token paths that are consistent
        with `left_context`, marginalizing completed paths by the
        character each one reveals just past the end of `left_context`
        (see `_expand_character_hypotheses`'s own docstring for the
        active/completed distinction).

        Parameters
        ----------
        left_context : str
            Text typed so far; the model's left context. Only the
            portion after the final space is searched over character by
            character -- everything up to and including that space is
            encoded directly, on the assumption it's already a complete
            word (or words).
        config : ConfigPredictCharactersSubword
            Beam-search pruning limits. Defaults to
            `ConfigPredictCharactersSubword`'s own defaults.
        normalize_logprobs : bool
            If True (the default), normalize the returned log-probs in
            log-space so ``np.exp(result).sum() == 1`` (skipping
            characters no completed path reached, which stay
            ``-inf``). If False, returns raw unnormalized path sums.
        sort_output : bool
            If True (the default), sort `predictions` descending by
            log-prob. If False, `predictions` preserves
            `character_set_lower`'s own order -- what `LMDecoder` relies
            on to align results positionally with `character_set`.
        cache_state : object or None
            A `SubwordPredictCache` from a previous call (via that
            call's `PredictCharactersResult.cache_state`) to reuse the
            transformer's KV cache for the shared prefix, or None to
            infer the fixed prefix from scratch.
        return_cache_state : bool
            If True, populate the returned result's `cache_state` with a
            `SubwordPredictCache` for reuse on a later, extending call.

        Returns
        -------
        PredictCharactersResult
            `predictions` covers every character in `character_set`;
            `cache_state` is set only if `return_cache_state` is True.

        Raises
        ------
        ValueError
            If `config.max_completed_hypotheses` or
            `config.max_active_hypotheses` is given and not positive.
        TypeError
            If `cache_state` is given and isn't a `SubwordPredictCache`,
            or if `return_cache_state` is True but the underlying model
            doesn't return a KV cache.
        """

        if config.max_completed_hypotheses is not None and config.max_completed_hypotheses <= 0:
            raise ValueError("max_completed_hypotheses must be positive or None")
        if config.max_active_hypotheses is not None and config.max_active_hypotheses <= 0:
            raise ValueError("max_active_hypotheses must be positive or None")
        if cache_state is not None and not isinstance(cache_state, SubwordPredictCache):
            raise TypeError("cache_state must be a SubwordPredictCache")

        assert self.model is not None, "language model does not exist!"
        start_ns = time.time_ns()
        prepared_context = self._prepare_character_context(left_context)
        updated_cache_state = None
        initial_ranked_tokens = None
        if cache_state is not None or return_cache_state:
            sorted_token_ids, sorted_log_probs, updated_cache_state = self._infer_character_prefix(
                prepared_context,
                cache_state,
                return_cache_state=return_cache_state,
                left_context=left_context,
            )
            initial_ranked_tokens = sorted_token_ids, sorted_log_probs
        char_to_log_probs = self._search_character_hypotheses(
            prepared_context,
            config,
            initial_ranked_tokens=initial_ranked_tokens,
        )

        result = self._format_character_predictions(
            char_to_log_probs,
            normalize_logprobs=normalize_logprobs,
            sort_output=sort_output,
        )
        result.cache_state = updated_cache_state

        self.predict_total_ns += time.time_ns() - start_ns
        return result


DEFAULT_MODEL_NAME = "figmtu/opt-350m-aac"


class LMDecoder:
    """Character-level next-symbol log-probabilities from a causal subword LM.

    Thin public wrapper around CausalSubwordLanguageModel: callers get a
    plain numpy array aligned to `character_set`'s own order. Achieved by
    calling predict_characters(..., sort_output=False):
    character_set_lower preserves character_set's own order (built by
    iterating it directly in change_character_set()), so with sorting
    off, predictions[i] already corresponds to self.character_set[i].
    """

    def __init__(
        self,
        character_set: list[str],
        lang_model_name: str = DEFAULT_MODEL_NAME,
        lm_path: str | None = None,
        device: Device = Device.AUTO,
        precision: Precision = Precision.FP32,
        lora_path: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        """Construct an LMDecoder, loading the underlying transformer eagerly.

        Parameters
        ----------
        character_set : list[str]
            Symbols to predict over. `predict_log_probs`'s output is
            aligned to this exact order.
        lang_model_name : str
            Hugging Face model name to load if `lm_path`/`lora_path`
            aren't given. Defaults to the hosted `figmtu/opt-350m-aac`
            model this project uses.
        lm_path, device, precision, lora_path, trust_remote_code
            Forwarded as-is to `CausalSubwordLanguageModel.__init__` --
            see its own docstring for what each controls.

        Raises
        ------
        InvalidLanguageModelException
            If the tokenizer or model can't be loaded -- see
            `CausalSubwordLanguageModel.__init__`.
        """
        self.character_set = list(character_set)
        self._lm = CausalSubwordLanguageModel(
            character_set=self.character_set,
            lang_model_name=lang_model_name,
            lm_path=lm_path,
            device=device,
            precision=precision,
            lora_path=lora_path,
            trust_remote_code=trust_remote_code,
        )
        # Reused across calls so a growing `evidence` (the normal case for
        # a live session -- one character typed at a time) skips re-running
        # the transformer over the whole prefix each time. Safe because
        # _infer_character_prefix() only reuses this cache when the new
        # call's token ids literally start with the cached ones; anything
        # else (a shorter or diverging evidence) falls back to a full,
        # correct recompute rather than reusing stale state.
        self._cache_state: SubwordPredictCache | None = None

    def predict_log_probs(
        self,
        evidence: str,
        config: ConfigPredictCharactersSubword | None = None,
    ) -> np.ndarray:
        """Return normalized log P(next_char | evidence) over
        self.character_set, in that exact order.

        These are natural-log probabilities: np.exp(result) sums to 1;
        `result` itself does not, and is not converted to linear
        probabilities here (see this module's docstring for why).

        Parameters
        ----------
        evidence : str
            Text typed so far in the current message (the model's left
            context).
        config : ConfigPredictCharactersSubword or None
            Search/pruning parameters (max_completed_hypotheses,
            max_active_hypotheses). None uses this module's own defaults.

        Returns
        -------
        numpy.ndarray
            Natural-log probabilities, one per symbol in
            `self.character_set`, in that exact order.
        """
        result = self._lm.predict_characters(
            evidence,
            config=config if config is not None else ConfigPredictCharactersSubword(),
            cache_state=self._cache_state,
            return_cache_state=True,
            sort_output=False,
        )
        # predict_characters() only ever sets cache_state to what this same
        # call passed in as cache_state=..., updated -- always a
        # SubwordPredictCache or None, never some other object.
        self._cache_state = cast(SubwordPredictCache, result.cache_state)
        return np.array([log_prob for _, log_prob in result.predictions], dtype=np.float64)

    def reset_cache(self) -> None:
        """Drop the reused transformer cache -- call when starting a new
        message, so the next predict_log_probs() call doesn't try to
        reuse state left over from a now-irrelevant previous context."""
        self._cache_state = None
