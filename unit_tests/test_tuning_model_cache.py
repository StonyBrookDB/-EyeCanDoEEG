"""Tests for eyecando.tuning.model_cache: model_cache_path and
get_or_fit_model.

Covers the cache filename encoding (excluded-subject set,
n_components, shrinkage, and that use_ea/use_xdawn only add a filename
suffix when False, preserving pre-flag filenames by default); that
get_or_fit_model actually caches to disk and loads rather than refits
on a repeat call for the same key (verified by feeding a second call
data that would fail if it really refit); and that different
use_ea/use_xdawn keys produce distinct, non-colliding cache files.
"""

from pathlib import Path

import numpy as np

from eyecando.tuning.model_cache import get_or_fit_model, model_cache_path

N_EPOCHS = 60
N_CHANNELS = 4
N_TIMES = 50


def _synthetic_pooled() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    y = (np.arange(N_EPOCHS) % 5 == 0).astype(int)
    X = rng.normal(scale=1.0, size=(N_EPOCHS, N_CHANNELS, N_TIMES))
    bump = 4.0 * np.exp(-((np.arange(N_TIMES) - N_TIMES // 2) ** 2) / 100)
    X[y == 1, 0, :] += bump
    return [(X, y)]


def test_model_cache_path_encodes_key_fields() -> None:
    path = model_cache_path(Path("/tmp/cache"), frozenset({1, 3}), 2, "auto")
    assert path == Path("/tmp/cache/excl_1-3_ncomp2_shrinkauto.joblib")


def test_model_cache_path_defaults_use_no_suffix() -> None:
    """use_ea/use_xdawn both True (the default) must byte-match the
    pre-flag filename convention -- no suffix at all."""
    path = model_cache_path(Path("/tmp/cache"), frozenset(), 2, "auto")
    assert "_noea" not in path.name
    assert "_noxdawn" not in path.name


def test_model_cache_path_flags_change_filename() -> None:
    with_ea = model_cache_path(Path("/tmp/cache"), frozenset(), 2, "auto", use_ea=False)
    with_xdawn = model_cache_path(Path("/tmp/cache"), frozenset(), 2, "auto", use_xdawn=False)
    assert "_noea" in with_ea.name
    assert "_noxdawn" in with_xdawn.name


def test_get_or_fit_model_caches_on_disk(tmp_path: Path) -> None:
    pooled = _synthetic_pooled()
    excluded_ids = frozenset({0})

    get_or_fit_model(excluded_ids, 2, "auto", pooled, cache_dir=tmp_path)
    cache_path = model_cache_path(tmp_path, excluded_ids, 2, "auto")
    assert cache_path.exists()

    # Second call with the same key must load rather than refit -- confirmed
    # by handing it deliberately wrong training data it would fail loudly
    # on if it actually tried to fit again (mismatched channel count).
    bad_pooled = [(pooled[0][0][:, :1, :], pooled[0][1])]
    reloaded = get_or_fit_model(excluded_ids, 2, "auto", bad_pooled, cache_dir=tmp_path)
    assert reloaded.predict_proba(pooled[0][0]).shape[0] == N_EPOCHS


def test_get_or_fit_model_different_keys_do_not_collide(tmp_path: Path) -> None:
    pooled = _synthetic_pooled()
    get_or_fit_model(frozenset({0}), 2, "auto", pooled, cache_dir=tmp_path)
    get_or_fit_model(frozenset({0}), 2, "auto", pooled, cache_dir=tmp_path, use_ea=False)
    path_a = model_cache_path(tmp_path, frozenset({0}), 2, "auto")
    path_b = model_cache_path(tmp_path, frozenset({0}), 2, "auto", use_ea=False)
    assert path_a != path_b
    assert path_a.exists() and path_b.exists()
