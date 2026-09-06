"""
Build three separate result figures for the Emotiv Flex 20-session extended
validation (Section IV-C of the paper):

  fig_flex20_persession.png   per-session Target-Non-Target peak amplitude,
                               all 20 sessions (diagnostic heuristic)
  fig_flex20_erp.png          pooled ERP (Target vs Non-Target vs Diff)
  fig_flex20_permtests.png    peak-amplitude + xDAWN decoder AUC permutation
                               test histograms, side by side

The "Key Metrics Summary" panel from the earlier single-dashboard version
was dropped: its three rows duplicate what Table II (any-session vs. >=25%
policy) and Table III (20- vs 17-session sensitivity) already report as
real tables in the paper text, so a fourth, image-rendered copy of the same
numbers added nothing.

Reuses pool_sessions.py's own data-loading and permutation-test functions
(no reimplementation) so every panel is built from the exact arrays that
produced the paper's reported Table II numbers (threshold / >=25%-of-
sessions channel policy). The rep-accumulated AUC test fits the
group-stratified cross-validated xDAWN decoder exactly once, then reuses
those fixed out-of-fold flash scores for every permutation shuffle (each
shuffle only re-randomizes which code counts as each character's "target"
and re-sums, no refitting) -- fast enough for 1000+ shuffles where the old
per-flash-label-shuffle AUC test needed a full refit per shuffle. Results
are still cached to results/flex20/pooled_results.npz so reruns (e.g. to
tweak a figure's styling) load the cache and re-render in seconds. Pass
--force to recompute from scratch.

All the underlying numbers this script draws on -- per-session peaks,
bad-channel counts, and the pooled-test summary table -- live as plain CSVs
in results/flex20/ (session_table.csv, bad_channel_counts.csv,
metrics_summary.csv) rather than hardcoded in this file, so they're
reusable from any future plotting script (or a spreadsheet) without
touching this one. The per-session peak-amplitude bar panel reads
session_table.csv directly (decode_emotivpro.py's per-session heuristic
output) rather than recomputing a third time, per the paper's own footnote
that this is a different, diagnostic-only computation from the
pooled-analysis numbers.

Usage:
    python plot_flex20_dashboard.py recordings/session_001 ... recordings/session_020
    python plot_flex20_dashboard.py recordings/session_001 ... --force   # ignore cache
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from pool_sessions import (CHAR_ID_OFFSET, _oof_flash_scores, load_and_epoch,
                           permutation_test, rep_accumulated_auc_test, resample_to_grid)
from visualize_p300 import P300_WINDOW, TMAX, TMIN

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "flex20")


def load_session_table():
    path = os.path.join(RESULTS_DIR, "session_table.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["session"], r["phrase"], float(r["peak_uv"]), r["detected"] == "True")
            for r in rows]


TEAL = "#1b7a8a"
BRICK = "#c2521a"
BG = "#eef3f5"
GRID = "#d6dee0"


def p_label(p, n_perm, prefix="p"):
    """Format a permutation-test p-value for a plot title. With only
    n_perm shuffles, the smallest value distinguishable from zero is
    1/(n_perm+1), so when no null shuffle reached the observed statistic
    (p == 0, e.g. the 20-shuffle AUC test) this reports that bound instead
    of a literal "p=0.000", which would imply more precision than the
    shuffle count supports -- matching the paper's own text (Section IV-C)."""
    if p == 0:
        return f"{prefix}<{1 / (n_perm + 1):.3f}"
    return f"{prefix}={p:.3f}"


def pool(session_dirs, bad_channel_frac=0.25, n_perm=1000, auc_perm=1000, seed=0, max_rep=15):
    all_epochs, bad_counts, all_is_target, all_codes, all_reps, all_char_ids = \
        [], {}, [], [], [], []
    ch_names_ref = None
    t_common = np.linspace(TMIN, TMAX, 200)
    for session_i, sd in enumerate(session_dirs):
        epochs, bad, ch_names, mrk_al = load_and_epoch(sd)
        if ch_names_ref is None:
            ch_names_ref = ch_names
        for c in bad:
            bad_counts[c] = bad_counts.get(c, 0) + 1
        all_is_target.append((epochs.events[:, 2] == 2).astype(int))
        all_epochs.append(resample_to_grid(epochs, t_common))
        all_codes.append(mrk_al["code"].to_numpy().astype(int))
        all_reps.append(mrk_al["rep"].to_numpy().astype(int))
        all_char_ids.append(session_i * CHAR_ID_OFFSET + mrk_al["char_idx"].to_numpy().astype(int))

    n_sessions = len(session_dirs)
    threshold = bad_channel_frac * n_sessions
    all_bad = {c for c, cnt in bad_counts.items() if cnt >= threshold}
    good_idx = [i for i, c in enumerate(ch_names_ref) if c not in all_bad]

    data = np.concatenate(all_epochs, axis=0)[:, good_idx, :]
    is_target = np.concatenate(all_is_target)
    codes = np.concatenate(all_codes)
    reps = np.concatenate(all_reps)
    char_ids = np.concatenate(all_char_ids)

    t = t_common * 1000.0
    window_mask = (t >= P300_WINDOW[0]) & (t <= P300_WINDOW[1])
    baseline_mask = t <= 0
    data = data - data[..., baseline_mask].mean(axis=-1, keepdims=True)

    real_diff, real_peak, null_peaks, p_value = permutation_test(
        data, is_target, window_mask, n_perm=n_perm, seed=seed)
    tgt_m = data[is_target == 1].mean(0).mean(0)
    ntgt_m = data[is_target == 0].mean(0).mean(0)

    scores = _oof_flash_scores(data, is_target, char_ids, seed=seed)
    auc_curve, null_aucs, auc_p = rep_accumulated_auc_test(
        scores, codes, reps, char_ids, is_target, max_rep=max_rep,
        n_perm=auc_perm, seed=seed)
    real_auc = auc_curve[-1]

    return dict(t=t, tgt_m=tgt_m, ntgt_m=ntgt_m, real_diff=real_diff,
                n_target=int(is_target.sum()), n_nontarget=int((is_target == 0).sum()),
                n_ch_kept=len(good_idx), n_ch_total=len(ch_names_ref),
                real_peak=real_peak, null_peaks=null_peaks, p_value=p_value,
                auc_curve=auc_curve, real_auc=real_auc, null_aucs=null_aucs, auc_p=auc_p,
                n_sessions=n_sessions, n_characters=int(len(np.unique(char_ids))),
                max_rep=max_rep)


def pool_cached(session_dirs, cache_path, force=False, **kwargs):
    if not force and os.path.exists(cache_path):
        print(f"[cache] loading pooled results from {cache_path}")
        with np.load(cache_path) as z:
            return {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files}
    res = pool(session_dirs, **kwargs)
    np.savez(cache_path, **res)
    print(f"[cache] saved pooled results to {cache_path}")
    return res


def plot_persession(out_dir):
    session_table = load_session_table()
    fig, ax = plt.subplots(figsize=(17, 6), facecolor=BG)
    sessions = [s for s, *_ in session_table]
    phrases = [ph for _, ph, *_ in session_table]
    peaks = [p for *_, p, _ in session_table]
    detected = [d for *_, d in session_table]
    colors = [TEAL if d else BRICK for d in detected]
    bars = ax.bar(range(len(peaks)), peaks, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="#666666", lw=1)
    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_xlim(-0.7, len(peaks) - 0.3)
    for i, ph in enumerate(phrases):
        ax.text(i, -1.55, ph, rotation=60, rotation_mode="anchor",
                 ha="right", va="top", fontsize=8.5)
    ax.set_ylabel("Per-session peak (µV)\n(diagnostic heuristic)", fontsize=11)
    ax.set_title("Emotiv Flex, 20 sessions: per-session Target−Non-Target peak amplitude",
                  fontsize=14, fontweight="bold")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.4, bottom=-2.3)
    for i, (b, d) in enumerate(zip(bars, detected)):
        if not d:
            ax.annotate("noise-dominated" if peaks[i] < 0 else "low/absent",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points",
                        xytext=(0, -14 if peaks[i] < 0 else 6),
                        ha="center", va="top" if peaks[i] < 0 else "bottom",
                        fontsize=7.5, color=BRICK, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label="P300 detected (15/20)"),
                        Patch(color=BRICK, label="low/absent (5/20)")],
              loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_position(("data", 0))
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_flex20_persession.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"[save] {out}")


def plot_erp(res, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG)
    ax.plot(res["t"], res["ntgt_m"], color="tab:blue", label=f"Non-Target (n={res['n_nontarget']})")
    ax.plot(res["t"], res["tgt_m"], color="tab:red", label=f"Target (n={res['n_target']})")
    ax.plot(res["t"], res["real_diff"], color="k", lw=1.5, label="Target − Non-Target")
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.15, label="P300 window")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (raw ADC counts)")
    ax.set_title(f"Pooled ERP across {int(res['n_sessions'])} sessions "
                 f"({int(res['n_ch_kept'])}/{int(res['n_ch_total'])} channels)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("white")
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_flex20_erp.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"[save] {out}")


def plot_permtests(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

    ax = axes[0]
    ax.hist(res["null_peaks"], bins=40, color="gray", alpha=0.75, label="null (label-shuffled)")
    ax.axvline(res["real_peak"], color=BRICK, lw=2.2, label=f"REAL peak ({float(res['real_peak']):.1f})")
    ax.set_xlabel("P300-window peak amplitude")
    ax.set_ylabel("count (out of 1000 shuffles)")
    ax.set_title(f"Peak permutation test   {p_label(float(res['p_value']), len(res['null_peaks']))}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("white")

    ax = axes[1]
    max_rep = int(res["max_rep"]) if "max_rep" in res else len(res["auc_curve"])
    ax.plot(range(1, max_rep + 1), res["auc_curve"], color="k", marker="o", ms=4,
             label="REAL rep-accumulated AUC")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax.axhline(np.quantile(res["null_aucs"], 0.95), color=BRICK, ls=":", lw=1.4,
                label=f"null 95th pct. ({len(res['null_aucs'])} shuffles)")
    ax.set_xlabel("Repetitions accumulated")
    ax.set_ylabel("Cross-validated decoder ROC-AUC")
    ax.set_title(f"Rep-accumulated AUC vs. repetitions   "
                 f"(r={max_rep}: {p_label(float(res['auc_p']), len(res['null_aucs']))})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor("white")

    fig.tight_layout()
    out = os.path.join(out_dir, "fig_flex20_permtests.png")
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"[save] {out}")


def plot_combined(res, out_dir):
    """Single, compact 2-row figure combining all three panels above --
    for the page-budget-constrained version of the paper, this replaces
    fig_flex20_persession.png + fig_flex20_erp.png + fig_flex20_permtests.png
    with one figure (one float's worth of caption/spacing overhead instead
    of three)."""
    session_table = load_session_table()
    fig = plt.figure(figsize=(7.2, 6.4), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.42,
                            top=0.94, bottom=0.14, left=0.09, right=0.98)

    ax = fig.add_subplot(gs[0, :])
    sessions = [s for s, *_ in session_table]
    peaks = [p for *_, p, _ in session_table]
    detected = [d for *_, d in session_table]
    colors = [TEAL if d else BRICK for d in detected]
    ax.bar(range(len(peaks)), peaks, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(sessions, fontsize=5.5, rotation=90)
    ax.set_xlim(-0.7, len(peaks) - 0.3)
    ax.set_ylabel("Peak (µV)", fontsize=8)
    ax.set_title("Per-session peak amplitude, all 20 sessions", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label="detected (15/20)"),
                        Patch(color=BRICK, label="low/absent (5/20)")],
              loc="upper right", fontsize=6, framealpha=0.9)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res["t"], res["ntgt_m"], color="tab:blue", lw=1, label="Non-Target")
    ax.plot(res["t"], res["tgt_m"], color="tab:red", lw=1, label="Target")
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.15)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.set_title("Pooled ERP", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(res["null_peaks"], bins=30, color="gray", alpha=0.75)
    ax.axvline(res["real_peak"], color=BRICK, lw=1.8)
    ax.set_xlabel("Peak amplitude", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.set_title(f"Peak perm. test  {p_label(float(res['p_value']), len(res['null_peaks']))}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    ax = fig.add_subplot(gs[1, 2])
    max_rep = int(res["max_rep"]) if "max_rep" in res else len(res["auc_curve"])
    ax.plot(range(1, max_rep + 1), res["auc_curve"], color="k", marker="o", ms=2.5, lw=1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(np.quantile(res["null_aucs"], 0.95), color=BRICK, ls=":", lw=1)
    ax.set_xlabel("Reps accumulated", fontsize=7)
    ax.set_ylabel("Decoder ROC-AUC", fontsize=7)
    ax.set_title(f"AUC vs. reps  {p_label(float(res['auc_p']), len(res['null_aucs']))}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    out = os.path.join(out_dir, "fig_flex20_combined.png")
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    print(f"[save] {out}")


def plot_results_row(res, out_dir):
    """Single-row, 3-panel figure (pooled ERP, peak-perm test, AUC-perm
    test) for the headline >=25%-of-sessions configuration -- the
    page-budget version of the paper drops the per-session bar chart panel
    (that finding is now stated directly in the Section IV-D text instead)
    and keeps only this row, saving vertical space."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), facecolor=BG)

    ax = axes[0]
    ax.plot(res["t"], res["ntgt_m"], color="tab:blue", lw=1, label="Non-Target")
    ax.plot(res["t"], res["tgt_m"], color="tab:red", lw=1, label="Target")
    ax.axvspan(*P300_WINDOW, color="orange", alpha=0.15)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.set_title("Pooled ERP", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    ax = axes[1]
    ax.hist(res["null_peaks"], bins=30, color="gray", alpha=0.75)
    ax.axvline(res["real_peak"], color=BRICK, lw=1.8)
    ax.set_xlabel("Peak amplitude", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.set_title(f"Peak perm. test  {p_label(float(res['p_value']), len(res['null_peaks']))}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    ax = axes[2]
    max_rep = int(res["max_rep"]) if "max_rep" in res else len(res["auc_curve"])
    ax.plot(range(1, max_rep + 1), res["auc_curve"], color="k", marker="o", ms=2.5, lw=1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(np.quantile(res["null_aucs"], 0.95), color=BRICK, ls=":", lw=1)
    ax.set_xlabel("Reps accumulated", fontsize=7)
    ax.set_ylabel("Decoder ROC-AUC", fontsize=7)
    ax.set_title(f"AUC vs. reps  {p_label(float(res['auc_p']), len(res['null_aucs']))}",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("white")

    fig.tight_layout()
    out = os.path.join(out_dir, "fig_flex20_results.png")
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dirs", nargs="+")
    ap.add_argument("--auc-perm", type=int, default=1000)
    ap.add_argument("--out-dir", default=os.path.join(
        "..", "..", "OneDrive", "Documents", "2026-Bella-CSIRE", "paper", "figures"))
    ap.add_argument("--force", action="store_true", help="ignore cache, recompute from scratch")
    ap.add_argument("--combined-only", action="store_true",
                     help="only render the single combined figure, skip the 3 separate ones")
    ap.add_argument("--row-only", action="store_true",
                     help="only render the compact 3-panel results row (no per-session bar chart)")
    args = ap.parse_args()
    cache_path = os.path.join(RESULTS_DIR, "pooled_results.npz")
    res = pool_cached(args.session_dirs, cache_path, force=args.force, auc_perm=args.auc_perm)
    if args.row_only:
        plot_results_row(res, args.out_dir)
    else:
        if not args.combined_only:
            plot_persession(args.out_dir)
            plot_erp(res, args.out_dir)
            plot_permtests(res, args.out_dir)
        plot_combined(res, args.out_dir)
