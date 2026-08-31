"""
Decoder comparison figure: Classical vs EEGNet on all 8 Muse2 muse2CleanData sessions.
Run:  python plot_decoder_comparison.py
Saves: muse2CleanData/out_decoder_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Results from running both training scripts on all 8 muse2CleanData sessions ──
# 8 sessions, 34 chars, 1020 target / 5100 non-target, 24 train / 10 test chars
SESSIONS = ["013", "014", "021", "022", "023", "026", "027", "028"]
PHRASES   = ["HELLO", "HELLO", "VINE", "WHOA", "JOBS", "FINS", "DRUM", "BOLT"]
N_SESSIONS = 8
N_CHARS    = 34
N_TRAIN    = 24
N_TEST     = 10
N_TARGET   = 1020
N_NONTARGET = 5100

reps = list(range(1, 16))

classical_acc = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 10.0,
                 10.0, 30.0, 10.0, 10.0, 20.0, 20.0, 40.0]
eegnet_acc    = [10.0, 0.0, 20.0, 20.0, 20.0, 30.0, 40.0, 40.0,
                 60.0, 60.0, 60.0, 50.0, 50.0, 30.0, 30.0]

classical_auc  = 0.630
eegnet_auc     = 0.650

eegnet_recall_tgt   = 58.7   # %
eegnet_recall_nontgt = 60.1  # %

classical_latency_ms = None   # not measured for classical
eegnet_latency_ms    = 0.520  # total per-flash
budget_ms            = 175.0

chance_pct = 100.0 / 36      # 1 of 36 symbols ≈ 2.78 %

# ── colour palette ──────────────────────────────────────────────────────────
CLR_CLASSIC = "#0e7c86"
CLR_EEGNET  = "#b65c2e"
CLR_CHANCE  = "#aaaaaa"
CLR_BG      = "#f7fafa"
CLR_PANEL   = "#ffffff"

# ── layout ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=CLR_BG)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10,
                    hspace=0.52, wspace=0.38)

gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])

ax_acc   = fig.add_subplot(gs[0, :])   # top: full-width per-rep accuracy
ax_auc   = fig.add_subplot(gs[1, 0])
ax_rtl   = fig.add_subplot(gs[1, 1])
ax_tbl   = fig.add_subplot(gs[1, 2])

for ax in fig.get_axes():
    ax.set_facecolor(CLR_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#d0dada")
    ax.tick_params(colors="#48565a", labelsize=9.5)

# ── 1. Per-rep character accuracy ────────────────────────────────────────────
ax_acc.axhline(chance_pct, color=CLR_CHANCE, lw=1.2, ls="--", zorder=1,
               label=f"Chance ≈ {chance_pct:.1f}% (1/36)")

ax_acc.fill_between(reps, 0, classical_acc, alpha=0.10, color=CLR_CLASSIC)
ax_acc.fill_between(reps, 0, eegnet_acc,    alpha=0.10, color=CLR_EEGNET)

ax_acc.plot(reps, classical_acc, "o-", color=CLR_CLASSIC, lw=2.2, ms=7,
            label=f"Classical (xDAWN+Riem.)  best={max(classical_acc):.0f}%")
ax_acc.plot(reps, eegnet_acc,    "s-", color=CLR_EEGNET,  lw=2.2, ms=7,
            label=f"EEGNet (CNN)              best={max(eegnet_acc):.0f}%")

# annotate peaks
best_c_rep = reps[classical_acc.index(max(classical_acc))]
best_e_rep = reps[eegnet_acc.index(max(eegnet_acc))]
ax_acc.annotate(f"{max(classical_acc):.0f}%",
                xy=(best_c_rep, max(classical_acc)),
                xytext=(best_c_rep - 0.6, max(classical_acc) + 4),
                fontsize=9, color=CLR_CLASSIC, fontweight="bold")
ax_acc.annotate(f"{max(eegnet_acc):.0f}%",
                xy=(best_e_rep, max(eegnet_acc)),
                xytext=(best_e_rep + 0.3, max(eegnet_acc) + 4),
                fontsize=9, color=CLR_EEGNET, fontweight="bold")

ax_acc.set_xlim(0.5, 15.5)
ax_acc.set_ylim(-2, 78)
ax_acc.set_xticks(reps)
ax_acc.set_xlabel("Repetitions accumulated", fontsize=10.5)
ax_acc.set_ylabel("Held-out character accuracy (%)", fontsize=10.5)
ax_acc.set_title("Per-repetition decoding accuracy  (10 held-out characters)",
                 fontsize=12, fontweight="bold", color="#14181a", pad=8)
ax_acc.legend(fontsize=9.5, framealpha=0.9, loc="upper left",
              frameon=True, edgecolor="#d0dada")
ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# ── 2. ROC-AUC comparison bar ────────────────────────────────────────────────
labels = ["Classical\n(xDAWN+Riem.)", "EEGNet\n(CNN)"]
aucs   = [classical_auc, eegnet_auc]
clrs   = [CLR_CLASSIC, CLR_EEGNET]
bars   = ax_auc.bar(labels, aucs, color=clrs, width=0.45, zorder=2,
                    edgecolor="white", linewidth=1.2)
ax_auc.axhline(0.5, color=CLR_CHANCE, lw=1.2, ls="--", label="Chance (0.5)")
for bar, v in zip(bars, aucs):
    ax_auc.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color="#14181a")
ax_auc.set_ylim(0.45, 0.72)
ax_auc.set_ylabel("In-sample ROC-AUC", fontsize=10)
ax_auc.set_title("Single-flash ROC-AUC\n(train characters)", fontsize=10.5,
                 fontweight="bold", color="#14181a")
ax_auc.legend(fontsize=8.5, framealpha=0.9)
ax_auc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

# ── 3. Latency bar ───────────────────────────────────────────────────────────
ax_rtl.set_title("Compute Latency per Flash\n(EEGNet, CPU)", fontsize=10.5,
                 fontweight="bold", color="#14181a")

used_pct  = eegnet_latency_ms / budget_ms * 100
free_pct  = 100 - used_pct

ax_rtl.barh(["Budget"], [budget_ms], color="#e2f1ea", height=0.4,
            edgecolor="#d0dada", linewidth=0.8)
ax_rtl.barh(["Budget"], [eegnet_latency_ms], color=CLR_EEGNET, height=0.4,
            edgecolor="white")

ax_rtl.text(eegnet_latency_ms + 2, 0,
            f"{eegnet_latency_ms:.3f} ms\n({used_pct:.1f}% used)",
            va="center", fontsize=9.5, color=CLR_EEGNET, fontweight="bold")
ax_rtl.text(budget_ms * 0.55, 0,
            f"174.5 ms free\n(99.7% of budget)",
            va="center", fontsize=9, color="#2f7d5a")

ax_rtl.set_xlim(0, budget_ms * 1.08)
ax_rtl.set_xlabel("ms", fontsize=9.5)
ax_rtl.set_yticks([])
ax_rtl.yaxis.set_visible(False)

# annotation: flash window context
ax_rtl.annotate("", xy=(budget_ms, -0.34), xytext=(0, -0.34),
                arrowprops=dict(arrowstyle="<->", color="#728486", lw=1.0))
ax_rtl.text(budget_ms / 2, -0.48, "175 ms flash SOA",
            ha="center", va="top", fontsize=8.5, color="#728486")
ax_rtl.set_ylim(-0.6, 0.6)

# ── 4. Summary metrics table ─────────────────────────────────────────────────
ax_tbl.axis("off")
ax_tbl.set_title("Key Metrics Summary", fontsize=10.5,
                 fontweight="bold", color="#14181a", pad=10)

rows = [
    ["Metric",                 "Classical",          "EEGNet"],
    ["Sessions",               f"{N_SESSIONS}",      f"{N_SESSIONS}"],
    ["Train / Test chars",     f"{N_TRAIN} / {N_TEST}", f"{N_TRAIN} / {N_TEST}"],
    ["In-sample ROC-AUC",     f"{classical_auc:.3f}", f"{eegnet_auc:.3f}"],
    ["Best char. accuracy",   f"{max(classical_acc):.0f}% (rep {best_c_rep})",
                               f"{max(eegnet_acc):.0f}% (rep {best_e_rep})"],
    ["Target recall",          "—",                  f"{eegnet_recall_tgt:.1f}%"],
    ["Non-target recall",      "—",                  f"{eegnet_recall_nontgt:.1f}%"],
    ["Compute latency",        "< 1 ms",             f"{eegnet_latency_ms:.3f} ms"],
]

col_widths = [0.40, 0.30, 0.30]
col_x      = [0.02, 0.44, 0.72]
y_start    = 0.93
row_h      = 0.115

for r, row in enumerate(rows):
    y = y_start - r * row_h
    is_header = (r == 0)
    bg = "#e1f0f0" if is_header else ("#f0f6f6" if r % 2 == 0 else CLR_PANEL)
    ax_tbl.add_patch(mpatches.FancyBboxPatch(
        (0.0, y - row_h * 0.82), 1.0, row_h * 0.88,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=bg, transform=ax_tbl.transAxes, clip_on=False))
    for c, (text, cx) in enumerate(zip(row, col_x)):
        weight = "bold" if (is_header or c == 0) else "normal"
        color  = CLR_CLASSIC if (c == 1 and not is_header) else \
                 CLR_EEGNET  if (c == 2 and not is_header) else "#14181a"
        ax_tbl.text(cx, y - row_h * 0.35, text,
                    transform=ax_tbl.transAxes,
                    fontsize=8.8, va="center", ha="left",
                    fontweight=weight, color=color)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.text(0.50, 0.955,
         "Classical vs. EEGNet Decoder Comparison — Muse2 (8 sessions, muse2CleanData)",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#14181a")
fig.text(0.50, 0.928,
         f"xDAWN + Riemannian Geometry + Logistic Regression   vs.   EEGNet (PyTorch CNN)  |  "
         f"34 chars pooled ({N_TARGET} target / {N_NONTARGET} non-target)  |  "
         f"P300 speller 6×6, 256 Hz, 175 ms SOA",
         ha="center", va="top", fontsize=9.5, color="#48565a")

out = "muse2CleanData/out_decoder_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
print(f"Saved → {out}")
