"""Visualize one exported Emotiv Flex QC session and print its data composition.

    python analysis/visualize_flex_dataset_session.py
    python analysis/visualize_flex_dataset_session.py batch2_session_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATASET = PROJECT_ROOT / "Dataset"
COLORS = {0: "#4C78A8", 1: "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", default="batch1_session_005")
    return parser.parse_args()


def plot_character_repetitions(
    *,
    session: str,
    phrase: str,
    X: np.ndarray,
    y: np.ndarray,
    char_idx: np.ndarray,
    repetition: np.ndarray,
    symbols: np.ndarray,
    times_ms: np.ndarray,
    pz_idx: int,
    output_dir: Path,
) -> Path:
    """Plot repetition-wise Pz target-minus-nontarget ERP for every character."""
    n_chars = int(char_idx.max()) + 1
    n_reps = int(repetition.max())
    effects = np.full((n_chars, n_reps), np.nan)
    curves = np.full((n_chars, n_reps, X.shape[-1]), np.nan, dtype=np.float32)
    counts = np.zeros((n_chars, n_reps, 2), dtype=np.int16)
    effect_window = (times_ms >= 300) & (times_ms <= 500)
    pz = X[:, pz_idx, :]
    character_symbols = []

    for character in range(n_chars):
        char_mask = char_idx == character
        character_symbols.append(str(symbols[char_mask][0]))
        for rep in range(1, n_reps + 1):
            mask = char_mask & (repetition == rep)
            target_mask = mask & (y == 1)
            non_mask = mask & (y == 0)
            counts[character, rep - 1] = (target_mask.sum(), non_mask.sum())
            if target_mask.any() and non_mask.any():
                curve = pz[target_mask].mean(axis=0) - pz[non_mask].mean(axis=0)
                curves[character, rep - 1] = curve
                effects[character, rep - 1] = curve[effect_window].mean()

    fig = plt.figure(figsize=(17, 21))
    grid = fig.add_gridspec(
        6,
        2,
        height_ratios=[1, 1, 1, 1, 1, 0.8],
        hspace=0.52,
        wspace=0.3,
    )
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_reps)
    line_axis_positions = [(r, c) for r in range(5) for c in range(2)]

    for character, (row, col) in enumerate(line_axis_positions):
        ax = fig.add_subplot(grid[row, col])
        for rep in range(n_reps):
            curve = curves[character, rep]
            if np.isfinite(curve).all():
                ax.plot(
                    times_ms,
                    curve,
                    color=cmap(norm(rep + 1)),
                    alpha=0.58,
                    lw=0.9,
                )
        mean_curve = np.nanmean(curves[character], axis=0)
        if np.isfinite(mean_curve).any():
            ax.plot(times_ms, mean_curve, color="black", lw=1.7, label="mean across reps")
        ax.axvline(0, color="black", lw=0.55)
        ax.axhline(0, color="black", lw=0.45)
        ax.axvspan(300, 500, color="gray", alpha=0.12)
        ax.set_title(
            f"char_idx {character}: {character_symbols[character]} "
            f"({int(np.isfinite(effects[character]).sum())}/{n_reps} reps)"
        )
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.set_xlabel("Time from aligned flash (ms)")
        ax.set_ylabel("Pz target - non-target (uV)")
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    heat_ax = fig.add_subplot(grid[5, :])
    heat = heat_ax.imshow(
        effects,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-np.nanpercentile(np.abs(effects), 95),
        vmax=np.nanpercentile(np.abs(effects), 95),
        extent=[0.5, n_reps + 0.5, -0.5, n_chars - 0.5],
    )
    heat_ax.set_xticks(np.arange(1, n_reps + 1))
    heat_ax.set_yticks(
        np.arange(n_chars),
        labels=[f"{i}: {s}" for i, s in enumerate(character_symbols)],
    )
    heat_ax.set(
        title="Mean Pz target-minus-nontarget voltage in 300-500 ms, by character and repetition",
        xlabel="Repetition",
        ylabel="Target character",
    )
    fig.colorbar(heat, ax=heat_ax, label="Mean difference (uV)", shrink=0.86)

    fig.suptitle(
        f"{session} | {phrase} | each colored line is one repetition",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Line color progresses from repetition 1 (purple) to 15 (yellow).",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.5,
        0.005,
        "Each repetition has at most 2 target and 10 non-target flashes; single-repetition curves are noisy and descriptive.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    output_dir.mkdir(exist_ok=True)
    safe_session = session.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{safe_session}_per_character_repetitions.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("\nPer-character repetition detail (target/non-target flashes retained):")
    for character, symbol in enumerate(character_symbols):
        reps_text = " ".join(
            f"{rep + 1}:{counts[character, rep, 0]}/{counts[character, rep, 1]}"
            for rep in range(n_reps)
        )
        print(f"  char_idx={character} {symbol}: {reps_text}")
    print(f"Repetition figure saved: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    npz_path = DATASET / f"{args.session}.npz"
    json_path = DATASET / f"{args.session}.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Could not find both {npz_path} and {json_path}")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}

    X = data["epochs_uv"]
    y = data["y"]
    code = data["stim_code"]
    char_idx = data["char_idx"]
    repetition = data["repetition"]
    symbols = data["target_symbol"]
    times_ms = data["epoch_sample_times_s"] * 1000
    channels = data["channel_names"].tolist()

    print(f"Session: {metadata['session_id']} | phrase: {metadata['true_text']}")
    print(f"File: {npz_path} ({npz_path.stat().st_size / 1e6:.2f} MB compressed)")
    print(
        f"X = epochs_uv: shape {X.shape} = "
        f"({len(X)} epochs, {X.shape[1]} channels, {X.shape[2]} samples), "
        f"{X.dtype}, unit=uV"
    )
    print(
        f"Time: {times_ms[0]:.1f} to {times_ms[-1]:.1f} ms, "
        f"{1000 / np.median(np.diff(times_ms)):.1f} Hz"
    )
    print(
        f"Labels y: target={int(y.sum())} ({y.mean():.1%}), "
        f"non-target={int((y == 0).sum())} ({(y == 0).mean():.1%})"
    )
    print(
        f"Characters: {len(np.unique(char_idx))}; "
        f"repetitions: {repetition.min()}-{repetition.max()}; "
        f"channels: {', '.join(channels)}"
    )
    print(
        f"QC noise={metadata['median_scale_uv']:.2f} uV; "
        f"marker lag={metadata['lag_s']:+.3f} s "
        f"(template cosine={metadata['lag_peak_cosine']:.3f}, "
        f"reliable={metadata['lag_estimate_reliable']}); "
        f"bad channels={metadata['bad_channels']}"
    )
    print(
        "Lag caution:",
        metadata["marker_alignment"]["warning"],
    )

    print("\nPer-character retained flash counts:")
    print("idx  target  total  target-flashes  missing-of-180")
    per_char = []
    for idx in sorted(np.unique(char_idx)):
        sel = char_idx == idx
        n_total = int(sel.sum())
        n_target = int(np.sum(y[sel] == 1))
        print(
            f"{idx:>3}  {symbols[sel][0]:>6}  {n_total:>5}  "
            f"{n_target:>14}  {180 - n_total:>14}"
        )
        per_char.append((idx, symbols[sel][0], n_total, n_target))

    print("\nFlash code counts (canonical: 1-6=columns, 7-12=rows):")
    print("code  axis     target  non-target  total")
    code_counts = {}
    for k in range(1, 13):
        sel = code == k
        n_target = int(np.sum(y[sel] == 1))
        n_non = int(np.sum(y[sel] == 0))
        code_counts[k] = (n_target, n_non)
        axis = "column" if k <= 6 else "row"
        print(f"{k:>4}  {axis:>6}  {n_target:>7}  {n_non:>10}  {int(sel.sum()):>5}")

    pz_idx = channels.index("Pz")
    pz = X[:, pz_idx, :]
    target = pz[y == 1]
    nontarget = pz[y == 0]
    target_mean = target.mean(axis=0)
    non_mean = nontarget.mean(axis=0)
    target_sem = target.std(axis=0, ddof=1) / np.sqrt(len(target))
    non_sem = nontarget.std(axis=0, ddof=1) / np.sqrt(len(nontarget))
    difference = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    effect_window = (times_ms >= 300) & (times_ms <= 500)
    channel_effect = difference[:, effect_window].mean(axis=1)
    strongest = np.argsort(np.abs(channel_effect))[::-1][:8]

    FIGURES = DATASET / "visualizations"
    FIGURES.mkdir(exist_ok=True)
    figure_path = FIGURES / f"{args.session}_overview.png"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(times_ms, target_mean, color=COLORS[1], label=f"target (n={len(target)})")
    ax.fill_between(
        times_ms,
        target_mean - target_sem,
        target_mean + target_sem,
        color=COLORS[1],
        alpha=0.18,
    )
    ax.plot(
        times_ms,
        non_mean,
        color=COLORS[0],
        label=f"non-target (n={len(nontarget)})",
    )
    ax.fill_between(
        times_ms,
        non_mean - non_sem,
        non_mean + non_sem,
        color=COLORS[0],
        alpha=0.18,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(300, 500, color="gray", alpha=0.12, label="300-500 ms")
    ax.axhline(0, color="black", lw=0.5)
    ax.set(title="Pz ERP: target vs non-target", xlabel="Time from aligned flash (ms)", ylabel="uV")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    im = ax.imshow(
        difference,
        aspect="auto",
        origin="lower",
        extent=[times_ms[0], times_ms[-1], -0.5, len(channels) - 0.5],
        cmap="RdBu_r",
        vmin=-np.percentile(np.abs(difference), 98),
        vmax=np.percentile(np.abs(difference), 98),
    )
    ax.set_yticks(range(len(channels)), labels=channels, fontsize=7)
    ax.axvline(0, color="black", lw=0.7)
    ax.axvspan(300, 500, color="gold", alpha=0.12)
    ax.set(
        title="Target minus non-target by channel",
        xlabel="Time from aligned flash (ms)",
        ylabel="Channel",
    )
    fig.colorbar(im, ax=ax, label="uV", shrink=0.86)

    ax = axes[1, 0]
    codes = np.arange(1, 13)
    target_n = np.array([code_counts[k][0] for k in codes])
    non_n = np.array([code_counts[k][1] for k in codes])
    ax.bar(codes, non_n, color=COLORS[0], label="non-target")
    ax.bar(codes, target_n, bottom=non_n, color=COLORS[1], label="target")
    ax.set_xticks(codes)
    ax.set(
        title="Retained flashes by stimulus code",
        xlabel="Code (1-6 columns; 7-12 rows)",
        ylabel="Epoch count",
    )

    ax = axes[1, 1]
    char_x = np.arange(len(per_char))
    total_n = np.array([r[2] for r in per_char])
    target_n = np.array([r[3] for r in per_char])
    ax.bar(char_x, total_n - target_n, color=COLORS[0], label="non-target")
    ax.bar(char_x, target_n, bottom=total_n - target_n, color=COLORS[1], label="target")
    ax.set_xticks(char_x, labels=[f"{i}: {s}" for i, s, _, _ in per_char])
    ax.set(
        title="Epoch retention by target character",
        xlabel="char_idx: target symbol",
        ylabel="Epoch count",
    )

    fig.suptitle(
        f"{metadata['session_id']} | {metadata['true_text']} | "
        f"{len(X)} clean flashes, {metadata['n_chars']} characters",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.005,
        "ERP ribbons are descriptive epoch-level SEM; flashes within the same character are not independent.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[0], label="non-target"),
            Patch(facecolor=COLORS[1], label="target"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    repetition_figure_path = plot_character_repetitions(
        session=metadata["session_id"],
        phrase=metadata["true_text"],
        X=X,
        y=y,
        char_idx=char_idx,
        repetition=repetition,
        symbols=symbols,
        times_ms=times_ms,
        pz_idx=pz_idx,
        output_dir=DATASET / "visualizations",
    )

    print("\nMean target-minus-nontarget voltage at 300-500 ms, strongest channels:")
    for i in strongest:
        print(f"  {channels[i]:>4}: {channel_effect[i]:+.3f} uV")
    pz_diff = float(difference[pz_idx, effect_window].mean())
    print(f"  Pz difference in 300-500 ms: {pz_diff:+.3f} uV")
    print(f"\nVisualization saved: {figure_path}")
    print(f"Per-character repetitions: {repetition_figure_path}")


if __name__ == "__main__":
    main()
