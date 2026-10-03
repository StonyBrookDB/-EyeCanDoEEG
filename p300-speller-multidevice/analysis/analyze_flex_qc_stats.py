"""
Detailed statistics for the QC subset of the Emotiv Flex batch1+batch2 analysis.

Reads the CSVs written by analyze_flex_batches.py (run that first) and prints / saves
exact p-values, confidence intervals and effect sizes. QC subset = sessions with median
channel noise <= QC_NOISE_UV. The rule uses noise only, but the threshold was chosen after seeing
that the noisiest sessions decoded poorly, so all p-values/CIs below are conditional on that rule.

    python analysis/analyze_flex_qc_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "results"
QC_NOISE_UV = 10.0
CHANCE = 1 / 36
N_BOOT = 10000
RNG = np.random.default_rng(1)
LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.append(line)


def pf(p):
    if p == 0:
        return "p<1e-300"
    return f"p={p:.3f}" if p >= 1e-3 else f"p={p:.1e}"


def t_ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se)
    return m, lo, hi


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_boot(df, col):
    """Resample SESSIONS with replacement, keep each session's characters together."""
    groups = [g[col].to_numpy() for _, g in df.groupby("session")]
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.integers(0, len(groups), len(groups))
        vals[i] = np.concatenate([groups[j] for j in pick]).mean()
    return np.percentile(vals, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def cohens_d_paired(x):
    return float(np.mean(x) / np.std(x, ddof=1))


def mcnemar_exact(a, b):
    """a, b: boolean 'correct' vectors for the same characters. Returns (a_only, b_only, p)."""
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    return a_only, b_only, (1.0 if n == 0 else stats.binomtest(a_only, n, 0.5).pvalue)


def main():
    info = pd.read_csv(OUT / "01_session_quality.csv")
    lag = pd.read_csv(OUT / "00_lag_estimates.csv")
    erp = pd.read_csv(OUT / "02_erp_per_session.csv")
    auc = pd.read_csv(OUT / "03_flash_auc.csv")
    chars = pd.read_csv(OUT / "04_char_level.csv")
    noleak = pd.read_csv(OUT / "06_noleak_char_level.csv")
    noleak_auc = pd.read_csv(OUT / "06_noleak_flash_auc.csv")

    qc = set(info.loc[info.median_scale_uv <= QC_NOISE_UV, "session"])
    not_qc = sorted(set(info.session) - qc)
    if not_qc or len(qc) != 26:
        raise RuntimeError(
            "This statistics script expects results generated from exactly the "
            f"26 QC sessions; found {len(qc)} QC sessions and {len(not_qc)} non-QC "
            "sessions in 01_session_quality.csv. Run analysis/analyze_flex_batches.py first."
        )
    iq = info
    say("QC SUBSET STATISTICS")
    say(f"Analysis input contains only the {len(iq)} sessions passing median channel noise "
        f"<= {QC_NOISE_UV:.0f} uV. The threshold was selected after preliminary decoding, "
        "so this is a post hoc QC cohort.")
    say("All p-values are exact (two-sided unless stated). Sessions, not characters or flashes, are "
        "the unit of replication wherever a test is run on session-level numbers.")

    # ---------------------------------------------------------------- 1. sample
    say("\n[1] SAMPLE")
    for b in ("batch1", "batch2", "all"):
        g = iq if b == "all" else iq[iq.batch == b]
        say(f"  {b:6s}: {len(g)} sessions, {int(g.n_chars.sum())} characters, {int(g.n_flashes.sum())} flashes "
            f"({int(g.n_targets.sum())} target), {g.duration_s.sum() / 60:.1f} min; "
            f"epoch rejection {g.reject_rate.mean():.1%}; noise median {g.median_scale_uv.median():.1f} uV")

    # ---------------------------------------------------------------- 2. lag
    say("\n[2] EEG-MARKER LAG (estimated per session; positive = EEG event later than marker)")
    lq = lag[lag.session.isin(qc)]
    for b in ("batch1", "batch2"):
        g = lq[lq.batch == b]
        say(f"  {b}: mean {g.lag_s.mean():+.3f} s, SD {g.lag_s.std(ddof=1):.3f}, "
            f"range [{g.lag_s.min():+.2f}, {g.lag_s.max():+.2f}]; reliable (template cosine >= 0.75) "
            f"{int(g.reliable.sum())}/{len(g)}")
    unrel = lq[~lq.reliable]
    say("  QC sessions that used the batch-median fallback lag (unreliable estimate): "
        + (", ".join(f"{r.session} (cos {r.peak_cos:.2f})" for _, r in unrel.iterrows()) or "none"))
    lev = stats.levene(lq[lq.batch == "batch1"].lag_s, lq[lq.batch == "batch2"].lag_s)
    say(f"  lag variability batch1 vs batch2 (Levene): W={lev.statistic:.1f}, {pf(lev.pvalue)}")
    say(f"  batch2 lag differs from 0: one-sample t={stats.ttest_1samp(lq[lq.batch == 'batch2'].lag_s, 0).statistic:.1f}, "
        f"{pf(stats.ttest_1samp(lq[lq.batch == 'batch2'].lag_s, 0).pvalue)}")

    # ---------------------------------------------------------------- 3. ERP
    say("\n[3] ERP: Pz mean amplitude 300-500 ms, target minus non-target (each session = one observation)")
    eq = erp[erp.session.isin(qc)]
    for name, g in (("all", eq), ("batch1", eq[eq.batch == "batch1"]), ("batch2", eq[eq.batch == "batch2"])):
        m, lo, hi = t_ci(g.pz_diff_uv)
        t, p = stats.ttest_1samp(g.pz_diff_uv, 0)
        pw = stats.wilcoxon(g.pz_diff_uv).pvalue
        say(f"  {name:6s} n={len(g):2d}: diff = {m:+.2f} uV, 95% CI [{lo:+.2f}, {hi:+.2f}]; "
            f"t({len(g) - 1})={t:.2f}, {pf(p)}; Wilcoxon {pf(pw)}; positive in {int((g.pz_diff_uv > 0).sum())}/{len(g)} sessions; "
            f"mean single-flash d = {g.pz_d.mean():.2f} (95% CI [{t_ci(g.pz_d)[1]:.2f}, {t_ci(g.pz_d)[2]:.2f}])")
    sig = (eq.flash_p < 0.05).sum()
    say(f"  within-session single-flash Welch t-test p<0.05 in {int(sig)}/{len(eq)} sessions "
        f"(p<0.001 in {int((eq.flash_p < 0.001).sum())}); median session p={eq.flash_p.median():.1e}")
    mr, lo_r, hi_r = t_ci(eq.roi_diff_uv)
    say(f"  ROI (Pz,P3,P4,CP1,CP2): {mr:+.2f} uV, 95% CI [{lo_r:+.2f}, {hi_r:+.2f}], "
        f"t={stats.ttest_1samp(eq.roi_diff_uv, 0).statistic:.2f}, {pf(stats.ttest_1samp(eq.roi_diff_uv, 0).pvalue)}")
    for col, lab in (("pz_diff_uv", "Pz diff (uV)"), ("pz_d", "single-flash d"), ("peak_latency_ms", "peak latency (ms)")):
        x1, x2 = eq[eq.batch == "batch1"][col], eq[eq.batch == "batch2"][col]
        w = stats.ttest_ind(x1, x2, equal_var=False)
        u = stats.mannwhitneyu(x1, x2)
        say(f"  batch1 vs batch2 {lab:18s}: {x1.mean():+.2f} vs {x2.mean():+.2f}; Welch t={w.statistic:.2f} {pf(w.pvalue)}; "
            f"Mann-Whitney {pf(u.pvalue)}")

    # ---------------------------------------------------------------- 4. flash AUC
    say("\n[4] FLASH-LEVEL AUC (held-out session; trained on all other sessions; EA unless stated)")
    P = auc[(auc.cond == "pooled") & auc.session.isin(qc)]
    aq = P[P.ea == "EA"].set_index("session").auc
    m, lo, hi = t_ci(aq)
    t, p_t = stats.ttest_1samp(aq, 0.5)
    p_w = stats.wilcoxon(aq - 0.5).pvalue
    say(f"  pooled/EA, n=26 sessions: mean AUC {m:.3f}, SD {aq.std(ddof=1):.3f}, 95% CI [{lo:.3f}, {hi:.3f}], "
        f"min {aq.min():.3f}, max {aq.max():.3f}; AUC>0.5 in {int((aq > 0.5).sum())}/26 sessions, "
        f"AUC>0.9 in {int((aq > 0.9).sum())}/26")
    say(f"    vs chance 0.5: one-sample t(25)={t:.1f}, {pf(p_t)}; Wilcoxon signed-rank {pf(p_w)} "
        f"(smallest possible two-sided exact p with n=26 is {2 / 2**26:.1e})")
    for b in ("batch1", "batch2"):
        v = aq[[s for s in aq.index if s.startswith(b)]]
        m, lo, hi = t_ci(v)
        say(f"    {b}: n={len(v)}, mean {m:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], min {v.min():.3f}")
    v1, v2 = aq[[s for s in aq.index if s.startswith("batch1")]], aq[[s for s in aq.index if s.startswith("batch2")]]
    w = stats.ttest_ind(v1, v2, equal_var=False)
    say(f"    batch1 vs batch2: diff {v1.mean() - v2.mean():+.3f}; Welch t={w.statistic:.2f} {pf(w.pvalue)}; "
        f"Mann-Whitney {pf(stats.mannwhitneyu(v1, v2).pvalue)}")
    raw = P[P.ea == "raw"].set_index("session").auc.loc[aq.index]
    dlt = aq - raw
    say(f"    EA vs raw (paired, 26 sessions): mean diff {dlt.mean():+.4f} AUC, 95% CI "
        f"[{t_ci(dlt)[1]:+.4f}, {t_ci(dlt)[2]:+.4f}]; paired t={stats.ttest_1samp(dlt, 0).statistic:.2f} "
        f"{pf(stats.ttest_1samp(dlt, 0).pvalue)}; Wilcoxon {pf(stats.wilcoxon(dlt).pvalue)}; "
        f"EA better in {int((dlt > 0).sum())}/26; paired d={cohens_d_paired(dlt):.2f}")
    for cond, lab in (("within", "train on same batch only"), ("cross", "train on the OTHER batch only")):
        c = auc[(auc.cond == cond) & (auc.ea == "EA") & auc.session.isin(qc)].set_index("session").auc.loc[aq.index]
        d2 = aq - c
        say(f"    pooled vs {cond} ({lab}): {aq.mean():.3f} vs {c.mean():.3f}; paired diff {d2.mean():+.4f}, "
            f"Wilcoxon {pf(stats.wilcoxon(d2).pvalue)}; {cond} AUC vs 0.5: Wilcoxon {pf(stats.wilcoxon(c - 0.5).pvalue)}, "
            f"min {c.min():.3f}")

    # ---------------------------------------------------------------- 5. characters
    say("\n[5] CHARACTER ACCURACY (pooled LOSO, EA); chance = 1/36 = 2.78%")
    C = chars[(chars.cond == "pooled") & (chars.ea == "EA") & chars.session.isin(qc)].copy()
    R = chars[(chars.cond == "pooled") & (chars.ea == "raw") & chars.session.isin(qc)].copy()
    n = len(C)
    say(f"  n = {n} characters in 26 sessions. Intervals: Wilson (treats characters as independent) and "
        f"cluster bootstrap over sessions ({N_BOOT} resamples; the more honest one).")
    prim = {}
    for r in (1, 2, 3, 4, 5, 6, 8, 15):
        k = int(C[f"c{r}"].sum())
        wl, wh = wilson(k, n)
        bl, bh = cluster_boot(C, f"c{r}")
        pb = stats.binomtest(k, n, CHANCE, alternative="greater").pvalue
        sa = C.groupby("session")[f"c{r}"].mean()
        say(f"  r={r:2d} ({r * 12 * 0.18:5.1f} s of flashes): {k}/{n} = {100 * k / n:5.1f}%  Wilson [{100 * wl:.1f}, {100 * wh:.1f}]  "
            f"cluster-boot [{100 * bl:.1f}, {100 * bh:.1f}]  vs chance: binomial one-sided {pf(pb)}; "
            f"per-session accuracy mean {100 * sa.mean():.1f}%, sessions at 100%: {int((sa == 1).sum())}/26, "
            f"sessions below 80%: {int((sa < 0.8).sum())}")
        prim[r] = pb
    ps = [prim[1], prim[3], prim[5]]
    say(f"  Holm-adjusted over the 3 pre-specified tests (flash AUC>0.5 Wilcoxon, char acc>chance at r=3, r=5): "
        f"{', '.join(pf(x) for x in holm([p_w, prim[3], prim[5]]))}")

    say("\n  Can we say accuracy is at least 90% at 3 / 5 reps?  (one-sided, H0: accuracy <= 90%)")
    for r in (3, 5):
        k = int(C[f"c{r}"].sum())
        pb = stats.binomtest(k, n, 0.90, alternative="greater").pvalue
        bl, _ = cluster_boot(C, f"c{r}")
        say(f"    r={r}: {k}/{n}; binomial {pf(pb)}; cluster-bootstrap lower 95% bound {100 * bl:.1f}%")

    say("\n  EA vs raw, paired on the same 150 characters (exact McNemar):")
    Rr = R.set_index(["session", "char_idx"])
    Cc = C.set_index(["session", "char_idx"]).loc[Rr.index]
    for r in (1, 2, 3, 5):
        a_only, b_only, p = mcnemar_exact(Cc[f"c{r}"].to_numpy().astype(bool), Rr[f"c{r}"].to_numpy().astype(bool))
        say(f"    r={r}: EA {100 * Cc[f'c{r}'].mean():.1f}% vs raw {100 * Rr[f'c{r}'].mean():.1f}%; "
            f"EA-only correct {a_only}, raw-only correct {b_only}, {pf(p)}")

    say("\n  Batch comparison (session-level accuracy; unit = session):")
    for r in (1, 3, 5, 8):
        sa = C.groupby(["session", "batch"])[f"c{r}"].mean().reset_index()
        x1, x2 = sa[sa.batch == "batch1"][f"c{r}"], sa[sa.batch == "batch2"][f"c{r}"]
        u = stats.mannwhitneyu(x1, x2)
        say(f"    r={r}: batch1 {100 * x1.mean():.1f}% (n={len(x1)}) vs batch2 {100 * x2.mean():.1f}% (n={len(x2)}); "
            f"diff {100 * (x1.mean() - x2.mean()):+.1f} pts; Mann-Whitney {pf(u.pvalue)}")

    say("\n  Training scope (r=3 and r=5, same 150 characters):")
    for cond in ("within", "cross"):
        X = chars[(chars.cond == cond) & (chars.ea == "EA") & chars.session.isin(qc)].set_index(["session", "char_idx"]).loc[Cc.index]
        for r in (3, 5):
            a_only, b_only, p = mcnemar_exact(Cc[f"c{r}"].to_numpy().astype(bool), X[f"c{r}"].to_numpy().astype(bool))
            say(f"    {cond:6s} r={r}: {100 * X[f'c{r}'].mean():.1f}% vs pooled {100 * Cc[f'c{r}'].mean():.1f}%; "
                f"pooled-only correct {a_only}, {cond}-only correct {b_only}, McNemar {pf(p)}")

    say("\n  Where errors remain at r=3 (pooled/EA):")
    wrong = C[C.c3 == 0]
    say(f"    {len(wrong)} wrong characters in {wrong.session.nunique()} sessions; error type "
        + str(wrong.e3.value_counts().to_dict()) + "; sessions: " + ", ".join(sorted(wrong.session.unique())))

    # ---------------------------------------------------------------- 6. speed
    say("\n[6] SPEED (flash time only: 12 flashes x 0.18 s measured SOA per repetition)")
    accs = np.array([C[f"c{r}"].mean() for r in range(1, 16)])

    def bits(p):
        return np.log2(36) if p >= 1 else (0 if p <= 0 else np.log2(36) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / 35))

    for r in (1, 2, 3, 4, 5, 6):
        say(f"    r={r}: accuracy {100 * accs[r - 1]:5.1f}%, {r * 12 * 0.18:4.1f} s/char, "
            f"ITR {bits(accs[r - 1]) * 60 / (r * 12 * 0.18):5.1f} bits/min (Wolpaw)")
    say("    (Wolpaw ITR assumes errors are uniformly distributed over the other 35 symbols and ignores cue/rest time; "
        "treat it as an upper-bound comparison across settings, not an achievable rate.)")
    st = []
    for _, row in C.iterrows():
        c = row[[f"c{r}" for r in range(1, 16)]].to_numpy().astype(int)
        w = np.where(c == 0)[0]
        st.append(1 if len(w) == 0 else w.max() + 2)
    st = np.array(st)
    say(f"    reps until the decision stays correct: median {np.median(st):.0f}, mean {st.mean():.2f}; "
        f"1 rep: {np.mean(st <= 1):.0%}, <=2: {np.mean(st <= 2):.0%}, <=3: {np.mean(st <= 3):.0%}, <=5: {np.mean(st <= 5):.0%}, "
        f"max {st.max()}")

    # ---------------------------------------------------------------- 7. no-leak
    say("\n[7] NO-LEAK CHECK (lag estimated from the first half of each session's characters only; "
        "evaluated on the second half)")
    nq = noleak[noleak.session.isin(qc)]
    say(f"  second-half characters in QC sessions: n={len(nq)} in {nq.session.nunique()} sessions")
    rel_half = set(lag.loc[lag.reliable_half, "session"]) & qc
    for lab, d in (("all QC sessions", nq), ("QC sessions where first-half lag was reliable", nq[nq.session.isin(rel_half)])):
        say(f"  {lab}: n={len(d)} chars, {d.session.nunique()} sessions")
        for r in (1, 3, 5, 8):
            k = int(d[f"c{r}"].sum())
            wl, wh = wilson(k, len(d))
            bl, bh = cluster_boot(d, f"c{r}")
            say(f"    r={r}: {k}/{len(d)} = {100 * k / len(d):.1f}%  Wilson [{100 * wl:.0f}, {100 * wh:.0f}]  "
                f"cluster-boot [{100 * bl:.0f}, {100 * bh:.0f}]  vs chance {pf(stats.binomtest(k, len(d), CHANCE, alternative='greater').pvalue)}")
    nauc = noleak_auc[noleak_auc.session.isin(qc)].auc
    say(f"  second-half flash AUC (QC): mean {nauc.mean():.3f}, 95% CI [{t_ci(nauc)[1]:.3f}, {t_ci(nauc)[2]:.3f}], "
        f"Wilcoxon vs 0.5 {pf(stats.wilcoxon(nauc - 0.5).pvalue)}; sessions with first-half lag unreliable: "
        f"{len(qc) - len(rel_half)}/26")

    # ---------------------------------------------------------------- 8. trends
    say("\n[8] TEMPORAL TREND (AUC vs session order, pooled/EA)")
    for b in ("batch1", "batch2"):
        d = aq[[s for s in aq.index if s.startswith(b)]]
        order = [int(s.split("_")[-1]) for s in d.index]
        rho, p = stats.spearmanr(order, d.values)
        say(f"  {b} (n={len(d)}): Spearman rho={rho:+.2f}, {pf(p)}")

    (OUT / "qc_statistics.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nsaved {OUT / 'qc_statistics.txt'}")


if __name__ == "__main__":
    main()
