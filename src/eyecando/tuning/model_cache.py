"""Fit-or-load-cached P300Model, keyed by which subjects were excluded and
which hyperparameters were used -- shared caching infrastructure for any
LOSO-style search or evaluation that refits the same (excluded subjects,
n_components, shrinkage, use_ea, use_xdawn) combination more than once
across a run (e.g. an outer fold and a later inner search re-evaluating
the same combination Stage 1 already fit).

`cache_dir` is a required argument, deliberately with no module-level
default -- see tuning/README.md for the real collision this prevents.
"""

from __future__ import annotations

from pathlib import Path

from eyecando.pipeline.model import P300Model


def model_cache_path(
    cache_dir: Path,
    excluded_ids: frozenset[int],
    n_components: int,
    shrinkage: float | str,
    use_ea: bool = True,
    use_xdawn: bool = True,
) -> Path:
    """Deterministic cache filename for one (excluded subjects,
    hyperparameters) combination.

    `use_ea`/`use_xdawn` only ever change the filename suffix when
    False -- the default (both True) produces the same filenames as
    before these flags existed, so any already-cached model from before
    they existed still hits. `use_ea` doesn't change how a model is
    *constructed* (that's a preprocessing-stage concern, applied upstream
    of whatever training data a caller fits with) -- it's cache-key-only,
    so a model fit on use_ea=False data is never silently confused with
    one fit on use_ea=True data under the same (excluded_ids,
    n_components, shrinkage) combination.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory this path is under -- see module docstring for why
        every caller must supply its own dataset-specific directory.
    excluded_ids : frozenset[int]
        Which subject indices were excluded from training.
    n_components : int
        xDAWN components, forwarded into the filename.
    shrinkage : float or str
        LDA shrinkage, forwarded into the filename (``.`` replaced with
        ``p`` so it's filesystem-safe).
    use_ea : bool
        Cache-key-only -- see above.
    use_xdawn : bool
        Cache-key-only -- see above.

    Returns
    -------
    pathlib.Path
        ``cache_dir / "excl_<ids>_ncomp<n>_shrink<s>[_noea][_noxdawn].joblib"``.
    """
    excl_str = "-".join(str(i) for i in sorted(excluded_ids)) or "none"
    shrink_str = str(shrinkage).replace(".", "p")
    ea_suffix = "" if use_ea else "_noea"
    xdawn_suffix = "" if use_xdawn else "_noxdawn"
    return (
        cache_dir
        / f"excl_{excl_str}_ncomp{n_components}_shrink{shrink_str}{ea_suffix}{xdawn_suffix}.joblib"
    )


def get_or_fit_model(
    excluded_ids: frozenset[int],
    n_components: int,
    shrinkage: float | str,
    train_pooled: list[tuple],
    cache_dir: Path,
    use_ea: bool = True,
    use_xdawn: bool = True,
) -> P300Model:
    """Fit a P300Model, or load one already cached for this exact
    (excluded subjects, n_components, shrinkage, use_ea, use_xdawn)
    combination.

    Keyed by which real subject IDs were excluded from training, not fold
    position: the same excluded set recurs across different outer folds'
    inner searches (excluding {i, k} is the same training set regardless
    of which one is nominally "outer" vs "inner" held-out), and a later
    search stage commonly re-evaluates the exact same (excluded set,
    n_components, shrinkage) combinations an earlier stage already fit.
    Caching by this key means neither kind of duplicate ever gets refit,
    here or in a future run.

    `use_ea` is cache-key-only (see `model_cache_path`) -- `train_pooled`
    is expected to already reflect whichever use_ea condition is active
    (preprocess_p300's own use_ea flag, applied by the caller before this
    function ever sees the data); this function never touches EA itself.
    `use_xdawn` is threaded into the constructed P300Model directly.

    Parameters
    ----------
    excluded_ids : frozenset[int]
        Which subject indices were excluded from `train_pooled`.
    n_components : int
        Forwarded to `P300Model`.
    shrinkage : float or str
        Forwarded to `P300Model`.
    train_pooled : list[tuple]
        Per-subject (or per-session) pooled ``(X, y)`` training data.
    cache_dir : pathlib.Path
        Where to look for/write the cached model -- see module docstring
        for why this must be dataset-specific, not a shared default.
    use_ea : bool
        Cache-key-only -- see `model_cache_path`.
    use_xdawn : bool
        Forwarded to `P300Model` and folded into the cache key.

    Returns
    -------
    P300Model
        Either loaded from `cache_dir`, or freshly fit and saved there.
    """
    cache_path = model_cache_path(
        cache_dir, excluded_ids, n_components, shrinkage, use_ea, use_xdawn
    )
    if cache_path.exists():
        return P300Model.load(cache_path)
    model = P300Model(n_components=n_components, shrinkage=shrinkage, use_xdawn=use_xdawn)
    model.fit(train_pooled)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(cache_path)
    return model
