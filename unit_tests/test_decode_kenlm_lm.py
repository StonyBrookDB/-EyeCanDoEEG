"""Tests for CharLM's kenlm backend (backend="kenlm"). Loads the real
lm_dec19_char_large_12gram.kenlm model once per module -- no mocks,
matching test_lm.py's own approach for the neural backend. Skipped
entirely if the model file or the special kenlm build (see
scripts/setup_kenlm.sh) isn't present -- neither is a normal
pyproject.toml dependency, so this is expected on a machine that hasn't
run that script yet, not a failure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eyecando.decode.lm import FARWELL_DONCHIN_GRID, CharLM

_MODEL_PATH = Path("models/lm_dec19_char_large_12gram.kenlm")

kenlm = pytest.importorskip("kenlm", reason="kenlm needs scripts/setup_kenlm.sh's special build")
if not _MODEL_PATH.exists():
    pytest.skip(
        f"{_MODEL_PATH} not present -- run scripts/setup_kenlm.sh", allow_module_level=True
    )


@pytest.fixture(scope="module")
def lm() -> CharLM:
    return CharLM(backend="kenlm")


def test_default_character_set_is_farwell_donchin_grid() -> None:
    assert CharLM(backend="kenlm").character_set == FARWELL_DONCHIN_GRID


def test_custom_character_set_is_stored() -> None:
    custom = list("ABCDEFGHIJ")
    assert CharLM(character_set=custom, backend="kenlm").character_set == custom


def test_prior_returns_valid_log_prob_distribution(lm: CharLM) -> None:
    log_probs = lm.prior(context="THE CA")
    assert log_probs.shape == (len(FARWELL_DONCHIN_GRID),)
    assert np.all(np.isfinite(log_probs))
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)


def test_prior_temperature_scales_log_probs(lm: CharLM) -> None:
    base = lm.prior(context="THE CA", temperature=1.0)
    scaled = lm.prior(context="THE CA", temperature=2.0)
    assert np.allclose(scaled, base / 2.0)


def test_prior_favors_plausible_continuation(lm: CharLM) -> None:
    """After "TH", "E" (-> THE) should heavily dominate implausible
    continuations like "Q" or "Z" -- the concrete behavior this whole
    module's tokenization convention (see kenlm_lm.py's own docstring)
    was reverse engineered to get right."""
    log_probs = lm.prior(context="TH")
    idx = {c: FARWELL_DONCHIN_GRID.index(c) for c in "EQZ"}
    assert log_probs[idx["E"]] > log_probs[idx["Q"]]
    assert log_probs[idx["E"]] > log_probs[idx["Z"]]


def test_prior_favors_car_over_cat_after_the_ca() -> None:
    """Matches the exact probe (kenlm_lm.py's module docstring) that
    validated the tokenization convention: "r" (car) should score above
    "t" (cat) after "the ca"."""
    log_probs = CharLM(backend="kenlm").prior(context="THE CA")
    idx = {c: FARWELL_DONCHIN_GRID.index(c) for c in "RT"}
    assert log_probs[idx["R"]] > log_probs[idx["T"]]


def test_digits_get_a_fixed_low_floor_not_raised(lm: CharLM) -> None:
    """Digits are out of this model's training vocabulary (see
    kenlm_lm.py's module docstring) -- should degrade gracefully to a
    low, finite, uniform score rather than erroring or returning NaN/inf."""
    log_probs = lm.prior(context="THE CA")
    digit_scores = [log_probs[FARWELL_DONCHIN_GRID.index(str(d))] for d in range(1, 10)]
    assert np.all(np.isfinite(digit_scores))
    assert np.allclose(digit_scores, digit_scores[0])
    assert digit_scores[0] < log_probs[FARWELL_DONCHIN_GRID.index("R")]


def test_uppercase_context_and_lowercase_model_do_not_conflict() -> None:
    """This project's own context strings are uppercase
    (FARWELL_DONCHIN_GRID); the model itself is lowercase-only (kenlm_lm.py's
    module docstring) -- prior() must lowercase internally rather than
    silently treating uppercase context as unseen/OOV."""
    log_probs_upper = CharLM(backend="kenlm").prior(context="TH")
    assert np.exp(log_probs_upper).sum() == pytest.approx(1.0, abs=1e-6)
    e_idx = FARWELL_DONCHIN_GRID.index("E")
    assert log_probs_upper[e_idx] > -1.0  # should still be a strong, plausible prediction


def test_empty_context_returns_valid_distribution(lm: CharLM) -> None:
    log_probs = lm.prior(context="")
    assert np.all(np.isfinite(log_probs))
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)


def test_reset_context_is_a_safe_no_op(lm: CharLM) -> None:
    """CharLM.reset_context() is called unconditionally regardless of
    which backend is active (see live/decoder.py's LiveDecoder,
    scripts/curate/test_curated_words.py's run_word_session) -- must be
    a safe no-op on the kenlm backend, not just prior()/character_set."""
    lm.reset_context()
    log_probs = lm.prior(context="TH")
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)


def test_space_in_context_is_handled(lm: CharLM) -> None:
    """Space in a context string must map to the <sp> token (kenlm_lm.py's
    module docstring), not a literal space passed to KenLM's own tokenizer
    (which would just be misinterpreted as a word separator)."""
    log_probs = lm.prior(context="THE ")
    assert np.all(np.isfinite(log_probs))
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)
