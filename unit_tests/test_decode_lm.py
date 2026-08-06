"""Tests for CharLM. Loads the real figmtu/opt-350m-aac model once per
module via LMDecoder -- no mocks, matching test_lm_decoder.py's approach."""

from __future__ import annotations

import numpy as np
import pytest

from eyecando.decode.lm import FARWELL_DONCHIN_GRID, CharLM


@pytest.fixture(scope="module")
def lm() -> CharLM:
    return CharLM()


def test_default_character_set_is_farwell_donchin_grid() -> None:
    assert CharLM().character_set == FARWELL_DONCHIN_GRID


def test_farwell_donchin_grid_is_36_symbols() -> None:
    """Matches ScoreAccumulator's own default 6x6 = 36 grid."""
    assert len(FARWELL_DONCHIN_GRID) == 36


def test_constructor_stores_model_path() -> None:
    assert CharLM(model_path=None).model_path is None


def test_custom_character_set_is_stored() -> None:
    custom = list("ABCDEFGHIJ")
    assert CharLM(character_set=custom).character_set == custom


def test_prior_returns_valid_log_prob_distribution(lm: CharLM) -> None:
    log_probs = lm.prior(context="HELLO WORL")
    assert log_probs.shape == (len(FARWELL_DONCHIN_GRID),)
    assert np.all(np.isfinite(log_probs))
    assert np.all(log_probs <= 1e-9)
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)


def test_prior_temperature_scales_log_probs(lm: CharLM) -> None:
    base = lm.prior(context="HELLO WORL", temperature=1.0)
    scaled = lm.prior(context="HELLO WORL", temperature=2.0)
    assert np.allclose(scaled, base / 2.0)


def test_reset_context_does_not_break_subsequent_calls(lm: CharLM) -> None:
    lm.prior(context="A")
    lm.reset_context()
    log_probs = lm.prior(context="B")
    assert np.all(np.isfinite(log_probs))
    assert np.exp(log_probs).sum() == pytest.approx(1.0, abs=1e-6)
