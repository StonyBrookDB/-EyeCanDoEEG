"""
Pooled statistical analysis of the Emotiv EPOC Flex recordings
(storedRecordings/emotivFlex-batch1 + emotivFlex-batch2).

Each session folder holds eeg.csv / markers.csv / meta.json (32 ch, 256 Hz,
6x6 Farwell-Donchin grid, 15 reps, SOA 175 ms). batch1 = EmotivPRO EDF export,
batch2 = Cortex API stream.

Sections (all printed + saved under analysis/results/):
  0. EEG-marker timing: per-session lag estimation (see note below)
  1. Data quality per session (timestamps, noise, bad channels, rejection)
  2. ERP statistics (target vs non-target, P300 amplitude/latency, batch effect)
  3. Decoding, leave-one-session-out (flash AUC + character accuracy vs reps)
       - pooled LOSO, within-batch LOSO, cross-batch transfer
       - each with and without per-session Euclidean Alignment (EA)
  4. Character-level analysis (accuracy/ITR, settling reps, errors, grid position)
  5. Temporal trends (session order, within-session drift)

No MNE/pyriemann needed: numpy / scipy / scikit-learn only.

    python analysis/analyze_flex_batches.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT = PROJECT_ROOT / "storedRecordings"
OUT = HERE / "results"
BATCHES = {"batch1": ROOT / "emotivFlex-batch1", "batch2": ROOT / "emotivFlex-batch2"}

GRID = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ1234", "56789_"]
N_SYMBOLS = 36
MAX_REPS = 15
SOA_S = 0.18  # measured median SOA is 179.7-181 ms (nominal 175 ms)
FLASHES_PER_REP = 12

FS = 256
BAND = (0.5, 20.0)
BIN = 4  # 256 Hz -> 64 Hz by block-averaging
PRE = 24  # samples before the flash (6 bins = 94 ms baseline)
POST = 204  # samples after (51 bins = 797 ms)
N_BINS = (PRE + POST) // BIN  # 57
BASE_BINS = PRE // BIN  # 6
BIN_T = (np.arange(N_BINS) * BIN - PRE + (BIN - 1) / 2) / FS  # bin-centre times (s)
Z_REJECT = 8.0  # |robust z| beyond this counts as an artifact sample
MIN_ARTIFACT_CH = 4  # epoch rejected if >= this many channels exceed Z_REJECT
P300_WIN = (0.30, 0.50)
PEAK_WIN = (0.25, 0.60)
QC_NOISE_UV = 10.0  # QC: sessions with median channel noise above this are flagged (threshold set post hoc)
ROI = ["Pz", "P3", "P4", "CP1", "CP2"]
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# 1. Loading + epoching
# --------------------------------------------------------------------------
GRID_FLAT = "".join(GRID)


def prepare_session(batch, folder):
    """Load one session and bandpass the continuous EEG; no epoching yet."""
    meta = json.loads((folder / "meta.json").read_text())
    eeg = pd.read_csv(folder / "eeg.csv")
    mrk = pd.read_csv(folder / "markers.csv")
    chans = list(eeg.columns[1:])
    assert chans == meta["eeg_names"], f"{folder}: channel mismatch"

    t = eeg["lsl_timestamp"].to_numpy()
    t0 = t[0]
    n = len(t)
    idx = np.arange(n)
    slope, intercept = np.polyfit(idx, t - t0, 1)  # t = t0 + intercept + slope*i
    resid_ms = ((t - t0) - (intercept + slope * idx)) * 1e3
    fs_eff = 1.0 / slope

    X = eeg[chans].to_numpy(dtype=np.float64).T
    X = X - np.median(X, axis=1, keepdims=True)
    sos = signal.butter(4, BAND, btype="bandpass", fs=fs_eff, output="sos")
    Xf = signal.sosfiltfilt(sos, X, axis=1)

    scale = 1.4826 * np.median(np.abs(Xf - np.median(Xf, axis=1, keepdims=True)), axis=1)
    med = np.median(scale)
    bad = (scale < 0.2 * med) | (scale > 4.0 * med) | ~np.isfinite(scale)

    i0 = np.round((mrk["lsl_timestamp"].to_numpy() - t0 - intercept) / slope).astype(int)

    phrase = meta["phrase"]
    mrk = mrk.copy()
    mrk["batch"] = batch
    mrk["session"] = f"{batch}/{folder.name}"
    # Ground truth comes from the markers' own is_target flags, not meta.json's phrase
    # (batch1/session_001's meta says CSSRE but the stimulus actually targeted CSIRE).
    t_row, t_col, n_phrase_mismatch = {}, {}, 0
    for c, g in mrk[mrk["is_target"] == 1].groupby("char_idx"):
        t_col[c] = int(g.loc[g["code"] <= 6, "code"].mode().iloc[0]) - 1
        t_row[c] = int(g.loc[g["code"] >= 7, "code"].mode().iloc[0]) - 7
        if GRID_FLAT[t_row[c] * 6 + t_col[c]] != phrase[c]:
            n_phrase_mismatch += 1
    mrk["t_row"] = mrk["char_idx"].map(t_row)
    mrk["t_col"] = mrk["char_idx"].map(t_col)
    expected = ((mrk["code"] == mrk["t_col"] + 1) | (mrk["code"] == mrk["t_row"] + 7)).astype(int)
    true_text = "".join(GRID_FLAT[t_row[c] * 6 + t_col[c]] for c in sorted(t_row))

    info = dict(
        session=f"{batch}/{folder.name}",
        batch=batch,
        order=int(folder.name.split("_")[1]),
        phrase=phrase,
        true_text=true_text,
        n_phrase_mismatch=n_phrase_mismatch,
        n_chars=len(phrase),
        n_flashes=len(mrk),
        n_targets=int(mrk["is_target"].sum()),
        eeg_source=meta.get("eeg_source"),
        duration_s=float(t[-1] - t0),
        fs_eff=float(fs_eff),
        ts_jitter_sd_ms=float(resid_ms.std()),
        ts_jitter_max_ms=float(np.abs(resid_ms).max()),
        median_scale_uv=float(med),
        n_bad_ch=int(bad.sum()),
        bad_ch=",".join(np.array(chans)[bad]),
        soa_median_ms=float(np.median(np.diff(mrk["lsl_timestamp"])) * 1e3),
        qc_target_consistent=float((expected == mrk["is_target"]).mean()),
    )
    return dict(info=info, mrk=mrk, Xf=Xf, i0=i0, scale=scale, bad=bad, chans=chans, n=n)


def _gather(p, lag, rows=None):
    """Epochs (n, C, N_BINS) at 64 Hz, baseline-corrected, for markers shifted by `lag` samples."""
    i0 = p["i0"] if rows is None else p["i0"][rows]
    st = i0 + lag - PRE
    ok = (st >= 0) & (st + PRE + POST <= p["n"])
    win = np.clip(st, 0, p["n"] - (PRE + POST))[:, None] + np.arange(PRE + POST)[None, :]
    ep = p["Xf"][:, win].transpose(1, 0, 2)
    ep = ep.reshape(len(ep), p["Xf"].shape[0], N_BINS, BIN).mean(-1).astype(np.float32)
    ep = ep - ep[:, :, :BASE_BINS].mean(-1, keepdims=True)
    return ep, ok


def epoch_session(p, lag):
    """Cut epochs at a given marker-to-EEG lag (samples) and apply artifact rejection."""
    ep, ok = _gather(p, lag)
    safe = np.where(p["bad"], 1.0, p["scale"])
    z = ep / safe[None, :, None]
    z[:, p["bad"], :] = 0.0
    n_art = (np.abs(z).max(axis=2) > Z_REJECT).sum(axis=1)
    kept = ok & (n_art < MIN_ARTIFACT_CH)
    mrk = p["mrk"].copy()
    mrk["kept"] = kept
    info = dict(p["info"], lag_s=lag / FS, reject_rate=float(1 - kept.mean()))
    return dict(info=info, mrk=mrk, ep=ep, scale=p["scale"], bad=p["bad"], chans=p["chans"])


WIDE_PRE, WIDE_POST = 448, 704  # samples around the nominal marker used for lag search
WIDE_NB = (WIDE_PRE + WIDE_POST) // BIN
SEG_START = WIDE_PRE // BIN  # bin index of tau = 0 (nominal marker time)
SEG_BINS = 64  # 1.0 s of difference wave compared against the template
LAG_BINS = np.arange(-100, 113)  # candidate shifts, in 4-sample (15.6 ms) bins -> +/-1.6 s
ANCHOR_S = 0.40  # template's posterior positive peak is placed at this latency
LAG_COS_MIN = 0.75  # lag trusted only if the difference wave matches the template this well
POSTERIOR = ["Pz", "P3", "P4", "CP1", "CP2"]


def wide_diff(p, rows=None):
    """Target-minus-nontarget average over a +/-1.7 s window around the nominal markers,
    in robust-z units (channels x 64 Hz bins), demeaned over time."""
    y = p["mrk"]["is_target"].to_numpy()
    rows = np.arange(len(y)) if rows is None else rows
    st = p["i0"][rows] - WIDE_PRE
    yy = y[rows]
    ok = (st >= 0) & (st + WIDE_PRE + WIDE_POST <= p["n"])
    C = p["Xf"].shape[0]
    safe = np.where(p["bad"], 1.0, p["scale"])
    sums = {0: np.zeros((C, WIDE_NB)), 1: np.zeros((C, WIDE_NB))}
    cnt = {0: 0, 1: 0}
    idx = np.where(ok)[0]
    for chunk in np.array_split(idx, max(1, len(idx) // 150)):
        win = st[chunk][:, None] + np.arange(WIDE_PRE + WIDE_POST)[None, :]
        ep = p["Xf"][:, win].transpose(1, 0, 2).reshape(len(chunk), C, WIDE_NB, BIN).mean(-1)
        z = np.clip(ep / safe[None, :, None], -Z_REJECT, Z_REJECT)
        z[:, p["bad"], :] = 0.0
        for lab in (0, 1):
            sel = yy[chunk] == lab
            sums[lab] += z[sel].sum(0)
            cnt[lab] += int(sel.sum())
    d = sums[1] / cnt[1] - sums[0] / cnt[0]
    return d - d.mean(axis=1, keepdims=True)


def _seg(d, lag_bins):
    j = SEG_START + lag_bins
    return d[:, j : j + SEG_BINS]


def _cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _lag_curve(template, d):
    return np.array([_cos(template, _seg(d, l)) for l in LAG_BINS])


def _robust_z(curve):
    k = int(np.argmax(curve))
    med = np.median(curve)
    sd = 1.4826 * np.median(np.abs(curve - med)) + 1e-9
    return float(curve[k]), float((curve[k] - med) / sd)


def prepare_all():
    prepared = []
    for batch, base in BATCHES.items():
        for folder in sorted(q for q in base.iterdir() if q.is_dir()):
            p = prepare_session(batch, folder)
            print(f"  prepared {p['info']['session']:<22} {p['info']['phrase']}")
            prepared.append(p)
    return prepared


def estimate_all_lags(prepared, n_iter=3):
    """Per-session EEG-marker lag by matching each session's multichannel target-nontarget
    difference wave against a leave-one-session-out template, iterated; then anchored so the
    template's posterior positive peak sits at ANCHOR_S. A second estimate uses only the first
    half of each session's characters (to evaluate on the second half without label leakage)."""
    n = len(prepared)
    batches = np.array([p["info"]["batch"] for p in prepared])
    n_half = [int(np.ceil(p["info"]["n_chars"] / 2)) for p in prepared]
    d_full = [wide_diff(p) for p in prepared]
    d_half = [wide_diff(p, np.where(p["mrk"]["char_idx"].to_numpy() < nh)[0])
              for p, nh in zip(prepared, n_half)]

    lag = np.zeros(n, int)
    rel = np.ones(n, bool)
    peak_cos, peak_z, curves = np.zeros(n), np.zeros(n), [None] * n
    for it in range(n_iter):
        segs = [_seg(d, l) for d, l in zip(d_full, lag)]
        new_lag = lag.copy()
        for i in range(n):
            if it == 0:
                others = [j for j in range(n) if batches[j] == "batch2" and j != i]
            else:
                others = [j for j in range(n) if j != i and rel[j]]
            template = np.mean([segs[j] for j in others], axis=0)
            c = _lag_curve(template, d_full[i])
            new_lag[i] = LAG_BINS[int(np.argmax(c))]
            peak_cos[i], peak_z[i] = _robust_z(c)
            curves[i] = c
        lag = new_lag
        rel = peak_cos >= LAG_COS_MIN

    segs = [_seg(d, l) for d, l in zip(d_full, lag)]
    template_all = np.mean([segs[j] for j in range(n) if rel[j]], axis=0)
    chans = prepared[0]["chans"]
    roi = [chans.index(c) for c in POSTERIOR]
    roi_wave = template_all[roi].mean(0)
    tau = (np.arange(SEG_BINS) * BIN + (BIN - 1) / 2) / FS
    win = (tau >= 0.1) & (tau <= 0.9)
    tau_peak = tau[win][int(np.argmax(roi_wave[win]))]
    offset = int(round((tau_peak - ANCHOR_S) * FS / BIN))  # in bins

    lag_half = np.zeros(n, int)
    peak_z_half, peak_cos_half = np.zeros(n), np.zeros(n)
    for i in range(n):
        template = np.mean([segs[j] for j in range(n) if j != i and rel[j]], axis=0)
        c = _lag_curve(template, d_half[i])
        lag_half[i] = LAG_BINS[int(np.argmax(c))]
        peak_cos_half[i], peak_z_half[i] = _robust_z(c)
    rel_half = peak_cos_half >= LAG_COS_MIN

    df = pd.DataFrame(dict(
        session=[p["info"]["session"] for p in prepared], batch=batches,
        order=[p["info"]["order"] for p in prepared],
        lag_samples=(lag + offset) * BIN, lag_s=(lag + offset) * BIN / FS,
        peak_cos=peak_cos, peak_z=peak_z, reliable=rel,
        lag_half_samples=(lag_half + offset) * BIN, lag_half_s=(lag_half + offset) * BIN / FS,
        peak_cos_half=peak_cos_half, peak_z_half=peak_z_half, reliable_half=rel_half))
    # Sessions with no detectable P300 cannot support their own lag: fall back to the
    # median lag of the reliable sessions of the same batch (flagged, and reported).
    for col, relc, out in (("lag_samples", "reliable", "lag_used"),
                           ("lag_half_samples", "reliable_half", "lag_half_used")):
        df[out] = df[col]
        for b in df.batch.unique():
            m = df.batch == b
            fb = int(np.median(df.loc[m & df[relc], col])) if (m & df[relc]).any() else 0
            df.loc[m & ~df[relc], out] = fb
    lag_axis = (LAG_BINS + offset) * BIN / FS
    curves_d = {p["info"]["session"]: (lag_axis, curves[i]) for i, p in enumerate(prepared)}
    meta = dict(tau_peak_before_anchor=float(tau_peak), template_roi_wave=roi_wave, tau=tau)
    return df, curves_d, meta


# --------------------------------------------------------------------------
# 2. ERP statistics
# --------------------------------------------------------------------------
def erp_stats(sessions):
    chans = sessions[0]["chans"]
    ci = {c: i for i, c in enumerate(chans)}
    win = (BIN_T >= P300_WIN[0]) & (BIN_T <= P300_WIN[1])
    pwin = (BIN_T >= PEAK_WIN[0]) & (BIN_T <= PEAK_WIN[1])
    rows, per_ch_d, wave = [], {}, {}
    for s in sessions:
        m = s["mrk"]
        keep = m["kept"].to_numpy()
        tg = (m["is_target"].to_numpy() == 1) & keep
        nt = (m["is_target"].to_numpy() == 0) & keep
        ep = s["ep"]
        good_roi = [c for c in ROI if not s["bad"][ci[c]]]
        pz = ep[:, ci["Pz"], :]
        roi = ep[:, [ci[c] for c in good_roi], :].mean(1)
        amp_pz = pz[:, win].mean(1)
        amp_roi = roi[:, win].mean(1)
        diff_pz = pz[tg].mean(0) - pz[nt].mean(0)
        sm_diff = np.convolve(diff_pz, np.ones(3) / 3, mode="same")
        peak_lat = BIN_T[pwin][np.argmax(sm_diff[pwin])]
        d_pz = (amp_pz[tg].mean() - amp_pz[nt].mean()) / np.sqrt(
            (amp_pz[tg].var(ddof=1) + amp_pz[nt].var(ddof=1)) / 2
        )
        d_roi = (amp_roi[tg].mean() - amp_roi[nt].mean()) / np.sqrt(
            (amp_roi[tg].var(ddof=1) + amp_roi[nt].var(ddof=1)) / 2
        )
        t_stat, p_val = stats.ttest_ind(amp_pz[tg], amp_pz[nt], equal_var=False)
        rows.append(
            dict(
                session=s["info"]["session"],
                batch=s["info"]["batch"],
                order=s["info"]["order"],
                n_target=int(tg.sum()),
                n_nontarget=int(nt.sum()),
                pz_target_uv=float(amp_pz[tg].mean()),
                pz_nontarget_uv=float(amp_pz[nt].mean()),
                pz_diff_uv=float(amp_pz[tg].mean() - amp_pz[nt].mean()),
                pz_d=float(d_pz),
                roi_diff_uv=float(amp_roi[tg].mean() - amp_roi[nt].mean()),
                roi_d=float(d_roi),
                flash_t=float(t_stat),
                flash_p=float(p_val),
                peak_latency_ms=float(peak_lat * 1e3),
            )
        )
        wave[s["info"]["session"]] = (pz[tg].mean(0), pz[nt].mean(0))
        dch = []
        for c in chans:
            if s["bad"][ci[c]]:
                dch.append(np.nan)
                continue
            a = ep[:, ci[c], :][:, win].mean(1)
            dch.append(
                (a[tg].mean() - a[nt].mean()) / np.sqrt((a[tg].var(ddof=1) + a[nt].var(ddof=1)) / 2)
            )
        per_ch_d[s["info"]["session"]] = dch
    df = pd.DataFrame(rows)
    ch_df = pd.DataFrame(per_ch_d, index=chans).T
    ch_df["batch"] = [s["info"]["batch"] for s in sessions]
    return df, ch_df, wave


# --------------------------------------------------------------------------
# 3. Decoding
# --------------------------------------------------------------------------
def features(s, ea):
    ep = s["ep"]
    safe = np.where(s["bad"], 1.0, s["scale"])
    z = ep / safe[None, :, None]
    z[:, s["bad"], :] = 0.0
    w = z[:, :, BASE_BINS:].astype(np.float64)  # 0..0.8 s, 51 bins
    w = w[:, :, :50]
    if ea:
        kept = s["mrk"]["kept"].to_numpy()
        R = np.einsum("nct,ndt->cd", w[kept], w[kept]) / (kept.sum() * w.shape[2])
        R += 1e-3 * np.trace(R) / R.shape[0] * np.eye(R.shape[0])
        vals, vecs = np.linalg.eigh(R)
        W = (vecs * vals**-0.5) @ vecs.T
        w = np.einsum("cd,ndt->nct", W, w)
    w = w.reshape(w.shape[0], w.shape[1], 25, 2).mean(-1)
    return w.reshape(len(w), -1).astype(np.float32)


def fit_lda(train_sessions, ea, cache):
    Xs, ys = [], []
    for s in train_sessions:
        key = (s["info"]["session"], ea)
        if key not in cache:
            cache[key] = features(s, ea)
        k = s["mrk"]["kept"].to_numpy()
        Xs.append(cache[key][k])
        ys.append(s["mrk"]["is_target"].to_numpy()[k])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", priors=[0.5, 0.5])
    lda.fit(X, y)
    return lda


def score(lda, s, ea, cache):
    key = (s["info"]["session"], ea)
    if key not in cache:
        cache[key] = features(s, ea)
    return lda.decision_function(cache[key])


def decode_chars(s, scores):
    """Per character: correct@r, error type@r for r=1..MAX_REPS."""
    m = s["mrk"]
    kept = m["kept"].to_numpy()
    codes = m["code"].to_numpy()
    reps = m["rep"].to_numpy()
    out = []
    for c in sorted(m["char_idx"].unique()):
        sel = (m["char_idx"].to_numpy() == c) & kept
        if not sel.any():
            continue
        S = np.zeros((13, MAX_REPS + 1))
        N = np.zeros((13, MAX_REPS + 1))
        np.add.at(S, (codes[sel], reps[sel]), scores[sel])
        np.add.at(N, (codes[sel], reps[sel]), 1)
        cS, cN = S.cumsum(1), N.cumsum(1)
        tr = int(m.loc[m["char_idx"] == c, "t_row"].iloc[0])
        tc = int(m.loc[m["char_idx"] == c, "t_col"].iloc[0])
        rec = dict(session=s["info"]["session"], batch=s["info"]["batch"], char_idx=c,
                   t_row=tr, t_col=tc, order=s["info"]["order"],
                   n_chars=s["info"]["n_chars"])
        for r in range(1, MAX_REPS + 1):
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cN[:, r] > 0, cS[:, r] / cN[:, r], -np.inf)
            pc = int(np.argmax(mean[1:7]))
            pr = int(np.argmax(mean[7:13]))
            rec[f"c{r}"] = int(pr == tr and pc == tc)
            rec[f"e{r}"] = "ok" if (pr == tr and pc == tc) else (
                "col" if pr == tr else "row" if pc == tc else "both")
        out.append(rec)
    return out


def run_decoding(sessions):
    cache = {}
    conds = {}
    flash_auc = []
    char_rows = []
    all_scores = {}
    for ea in (False, True):
        tag = "EA" if ea else "raw"
        print(f"  [{tag}] pooled LOSO ...")
        for k, s in enumerate(sessions):
            lda = fit_lda([x for j, x in enumerate(sessions) if j != k], ea, cache)
            sc = score(lda, s, ea, cache)
            all_scores[("pooled", tag, s["info"]["session"])] = sc
        print(f"  [{tag}] within-batch LOSO ...")
        for k, s in enumerate(sessions):
            tr = [x for j, x in enumerate(sessions)
                  if j != k and x["info"]["batch"] == s["info"]["batch"]]
            lda = fit_lda(tr, ea, cache)
            all_scores[("within", tag, s["info"]["session"])] = score(lda, s, ea, cache)
        print(f"  [{tag}] cross-batch ...")
        for src, dst in (("batch1", "batch2"), ("batch2", "batch1")):
            lda = fit_lda([x for x in sessions if x["info"]["batch"] == src], ea, cache)
            for s in sessions:
                if s["info"]["batch"] == dst:
                    all_scores[("cross", tag, s["info"]["session"])] = score(lda, s, ea, cache)

    for (cond, tag, sess), sc in all_scores.items():
        s = next(x for x in sessions if x["info"]["session"] == sess)
        m = s["mrk"]
        k = m["kept"].to_numpy()
        y = m["is_target"].to_numpy()
        auc = roc_auc_score(y[k], sc[k])
        flash_auc.append(dict(cond=cond, ea=tag, session=sess, batch=s["info"]["batch"],
                              order=s["info"]["order"], auc=auc))
        for rec in decode_chars(s, sc):
            rec.update(cond=cond, ea=tag)
            char_rows.append(rec)
    return pd.DataFrame(flash_auc), pd.DataFrame(char_rows), all_scores


# --------------------------------------------------------------------------
# 4. Character-level summaries
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_boot_ci(df, col, n_boot=2000):
    sess = df["session"].unique()
    groups = {s: df.loc[df["session"] == s, col].to_numpy() for s in sess}
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(sess, size=len(sess), replace=True)
        vals.append(np.concatenate([groups[p] for p in pick]).mean())
    return tuple(np.percentile(vals, [2.5, 97.5]))


def wolpaw_bits(p, n=N_SYMBOLS):
    if p <= 0:
        return 0.0
    if p >= 1:
        return float(np.log2(n))
    return float(np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1)))


def itr_bpm(p, r):
    return wolpaw_bits(p) * 60.0 / (r * FLASHES_PER_REP * SOA_S)


def acc_table(chars):
    rows = []
    for (cond, tag), g in chars.groupby(["cond", "ea"]):
        for grp_name, gg in [("all", g), ("batch1", g[g.batch == "batch1"]),
                             ("batch2", g[g.batch == "batch2"])]:
            for r in range(1, MAX_REPS + 1):
                k, n = int(gg[f"c{r}"].sum()), len(gg)
                lo, hi = wilson(k, n)
                clo, chi = cluster_boot_ci(gg, f"c{r}", 500) if r in (1, 3, 5, 8, 10, 15) else (np.nan, np.nan)
                rows.append(dict(cond=cond, ea=tag, group=grp_name, reps=r, n_chars=n, correct=k,
                                 acc=k / n, wilson_lo=lo, wilson_hi=hi, boot_lo=clo, boot_hi=chi,
                                 itr_bpm=itr_bpm(k / n, r),
                                 binom_p_vs_chance=stats.binomtest(k, n, 1 / N_SYMBOLS,
                                                                   alternative="greater").pvalue))
    return pd.DataFrame(rows)


def settle_reps(g):
    c = g[[f"c{r}" for r in range(1, MAX_REPS + 1)]].to_numpy()
    out = []
    for row in c:
        if row[-1] == 0:
            out.append(np.nan)
            continue
        wrong = np.where(row == 0)[0]
        out.append(1 if len(wrong) == 0 else wrong.max() + 2)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
COL = {"batch1": "#1f77b4", "batch2": "#d62728"}


def plot_quality(info, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(info))
    cols = [COL[b] for b in info["batch"]]
    ax[0].bar(x, info["reject_rate"] * 100, color=cols)
    ax[0].set_title("Epochs rejected (%)")
    ax[1].bar(x, info["median_scale_uv"], color=cols)
    ax[1].set_title("Median channel noise (robust SD, uV)")
    ax[1].set_yscale("log")
    ax[2].bar(x, info["ts_jitter_sd_ms"], color=cols)
    ax[2].set_title("EEG timestamp jitter SD (ms)")
    ax[2].set_yscale("log")
    for a in ax:
        a.set_xlabel("session (blue=batch1, red=batch2)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_erp(wave, erp_df, ch_df, path, exclude=()):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for b in ("batch1", "batch2"):
        ss = [s for s in wave if s.startswith(b) and s not in exclude]
        T = np.array([wave[s][0] for s in ss])
        N = np.array([wave[s][1] for s in ss])
        for arr, ls, lab in ((T, "-", "target"), (N, "--", "non-target")):
            mu, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(ss))
            ax[0].plot(BIN_T * 1e3, mu, ls, color=COL[b], label=f"{b} {lab}")
            ax[0].fill_between(BIN_T * 1e3, mu - se, mu + se, color=COL[b], alpha=0.15)
        D = T - N
        mu, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(len(ss))
        ax[1].plot(BIN_T * 1e3, mu, color=COL[b], label=f"{b} (n={len(ss)})")
        ax[1].fill_between(BIN_T * 1e3, mu - se, mu + se, color=COL[b], alpha=0.2)
    ax[0].set_title("Pz grand average (QC sessions, mean +/- SEM)")
    ax[1].set_title("Pz difference wave (target - non-target)")
    for a in ax[:2]:
        a.axvspan(P300_WIN[0] * 1e3, P300_WIN[1] * 1e3, color="gray", alpha=0.1)
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("ms")
        a.set_ylabel("uV")
        a.legend(fontsize=8)
    mean_d = ch_df.drop(columns="batch").drop(index=list(exclude), errors="ignore").mean().sort_values()
    ax[2].barh(mean_d.index, mean_d.values, color="#555")
    ax[2].set_title("Per-channel effect size d (300-500 ms), QC sessions")
    ax[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_auc(auc_df, path):
    d = auc_df[(auc_df.cond == "pooled")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    sess = list(dict.fromkeys(d.session))
    x = np.arange(len(sess))
    w = 0.4
    for off, tag in ((-w / 2, "raw"), (w / 2, "EA")):
        v = d[d.ea == tag].set_index("session").loc[sess]
        ax[0].bar(x + off, v.auc, w, label=tag, color=[COL[b] for b in v.batch],
                  alpha=0.55 if tag == "raw" else 1.0, edgecolor="k", lw=0.3)
    ax[0].axhline(0.5, color="k", lw=0.6)
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Held-out-session flash AUC, pooled LOSO (light=raw, dark=EA)")
    ax[0].set_xlabel("session")
    sub = d[d.ea == "EA"]
    for b in ("batch1", "batch2"):
        v = sub[sub.batch == b]
        ax[1].scatter(v.order, v.auc, color=COL[b], label=b)
        if len(v) > 3:
            sl = stats.linregress(v.order, v.auc)
            xs = np.array([v.order.min(), v.order.max()])
            ax[1].plot(xs, sl.intercept + sl.slope * xs, color=COL[b])
    ax[1].set_xlabel("session order within batch")
    ax[1].set_ylabel("AUC (EA, pooled LOSO)")
    ax[1].legend()
    ax[1].set_title("AUC vs session order")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acc(acc, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    a = acc[(acc.cond == "pooled") & (acc.group == "all")]
    for tag, c in (("raw", "#888"), ("EA", "#000")):
        v = a[a.ea == tag]
        ax[0].plot(v.reps, v.acc * 100, color=c, label=tag)
        ax[0].fill_between(v.reps, v.wilson_lo * 100, v.wilson_hi * 100, color=c, alpha=0.12)
    ax[0].axhline(100 / N_SYMBOLS, color="r", ls=":", label="chance")
    ax[0].set_title("Character accuracy vs reps (pooled LOSO, QC sessions only)")
    b = acc[(acc.cond == "pooled") & (acc.ea == "EA")]
    for g in ("batch1", "batch2"):
        v = b[b.group == g]
        ax[1].plot(v.reps, v.acc * 100, color=COL[g], label=f"{g} (n={int(v.n_chars.iloc[0])})")
        ax[1].fill_between(v.reps, v.wilson_lo * 100, v.wilson_hi * 100, color=COL[g], alpha=0.12)
    ax[1].set_title("By batch (EA, pooled LOSO)")
    cond_style = {"pooled": "-", "within": "--", "cross": ":"}
    for cond, ls in cond_style.items():
        for g in ("batch1", "batch2"):
            v = acc[(acc.cond == cond) & (acc.ea == "EA") & (acc.group == g)]
            if len(v) and not (cond == "cross" and False):
                ax[2].plot(v.reps, v.acc * 100, ls, color=COL[g], label=f"{cond}-{g}")
    ax[2].set_title("Training scope: pooled / within-batch / cross-batch (EA)")
    for a_ in ax:
        a_.set_xlabel("repetitions")
        a_.set_ylabel("% correct")
        a_.set_ylim(0, 102)
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_grid(chars, path, r=8):
    g = chars[(chars.cond == "pooled") & (chars.ea == "EA")]
    M = np.full((6, 6), np.nan)
    Nn = np.zeros((6, 6), int)
    for i in range(6):
        for j in range(6):
            sel = g[(g.t_row == i) & (g.t_col == j)]
            Nn[i, j] = len(sel)
            if len(sel):
                M[i, j] = sel[f"c{r}"].mean() * 100
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(M, vmin=0, vmax=100, cmap="viridis")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{GRID[i][j]}\n{'' if np.isnan(M[i, j]) else int(M[i, j])}\n(n={Nn[i, j]})",
                    ha="center", va="center", color="w", fontsize=7)
    ax.set_title(f"Accuracy at {r} reps by target cell (EA, pooled LOSO)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def plot_lag(curves, lag_df, meta, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for a_, b in zip(ax[:2], ("batch1", "batch2")):
        for _, r in lag_df[lag_df.batch == b].iterrows():
            axis, c = curves[r.session]
            a_.plot(axis, c, color=COL[b], alpha=0.5, lw=1 if r.reliable else 0.6,
                    ls="-" if r.reliable else ":")
            a_.plot(r.lag_s, r.peak_cos, "k.", ms=4)
        a_.axvline(0, color="k", lw=0.6)
        a_.set_title(f"{b}: difference-wave template similarity vs marker shift")
        a_.set_xlabel("estimated EEG-marker lag (s)")
        a_.set_ylabel("cosine similarity (dotted = unreliable)")
    for b in ("batch1", "batch2"):
        v = lag_df[lag_df.batch == b]
        ax[2].scatter(v.order, v.lag_s, color=COL[b], label=b)
        bad = v[~v.reliable]
        ax[2].scatter(bad.order, bad.lag_s, facecolors="none", edgecolors="k", s=120,
                      label="unreliable" if b == "batch1" else None)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlabel("session order within batch")
    ax[2].set_ylabel("estimated lag (s)")
    ax[2].set_title("EEG-marker lag per session")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def p_fmt(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    OUT.mkdir(exist_ok=True)
    log = []

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        log.append(line)

    print("Filtering sessions ...")
    all_prepared = prepare_all()
    excluded = [
        item
        for item in all_prepared
        if item["info"]["median_scale_uv"] > QC_NOISE_UV
    ]
    prepared = [
        item
        for item in all_prepared
        if item["info"]["median_scale_uv"] <= QC_NOISE_UV
    ]
    if not prepared:
        raise RuntimeError("No sessions passed the QC noise threshold.")
    say(
        f"Using QC-passing sessions only: {len(prepared)} sessions; "
        f"excluded {len(excluded)} by median channel noise > {QC_NOISE_UV:g} uV."
    )
    print("Estimating marker lags from QC-passing sessions only ...")
    lag_df, curves, lag_meta = estimate_all_lags(prepared)
    lag_df.to_csv(OUT / "00_lag_estimates.csv", index=False)
    plot_lag(curves, lag_df, lag_meta, OUT / "fig0_lag_curves.png")
    sessions = [epoch_session(p, int(l)) for p, l in zip(prepared, lag_df.lag_used)]
    sessions_half = [epoch_session(p, int(l)) for p, l in zip(prepared, lag_df.lag_half_used)]
    info = pd.DataFrame([s["info"] for s in sessions])
    info.to_csv(OUT / "01_session_quality.csv", index=False)

    say("\n=== 0. EEG-MARKER TIMING (QC sessions only) ===")
    say(f"   template posterior positive peak before anchoring: {lag_meta['tau_peak_before_anchor']:.2f} s "
        f"-> anchored at {ANCHOR_S:.2f} s (all lags are relative to this convention)")
    for b, g in lag_df.groupby("batch"):
        say(f"{b}: lag mean {g.lag_s.mean():+.2f} s, SD {g.lag_s.std(ddof=1):.2f}, "
            f"range [{g.lag_s.min():+.2f}, {g.lag_s.max():+.2f}]; reliable estimate "
            f"(peak cosine >= {LAG_COS_MIN}): {int(g.reliable.sum())}/{len(g)}; "
            f"median peak cosine {g.peak_cos.median():.2f}")
    r1, r2 = (lag_df[(lag_df.batch == b) & lag_df.reliable].lag_s for b in ("batch1", "batch2"))
    lev = stats.levene(r1, r2)
    say(f"   lag spread, reliable sessions only: batch1 SD {r1.std(ddof=1):.2f} s vs batch2 SD "
        f"{r2.std(ddof=1):.2f} s (Levene p={p_fmt(lev.pvalue)}); "
        f"batch2 lag vs 0: t-test p={p_fmt(stats.ttest_1samp(r2, 0).pvalue)}")
    unrel = lag_df[~lag_df.reliable]
    say("   sessions without a reliable lag (fallback = batch median lag): "
        + (", ".join(f"{r.session} (peak cosine {r.peak_cos:.2f})" for _, r in unrel.iterrows()) or "none"))
    say(f"   |lag| > 0.25 s: batch1 {int((r1.abs() > 0.25).sum())}/{len(r1)}, "
        f"batch2 {int((r2.abs() > 0.25).sum())}/{len(r2)}")
    hd = (lag_df.lag_half_used - lag_df.lag_used).abs() / FS
    say(f"   first-half-only lag vs full lag: median |diff| {hd.median():.3f} s, "
        f"90th pct {hd.quantile(0.9):.3f} s; reliable first-half estimates {int(lag_df.reliable_half.sum())}/{len(lag_df)}")
    say("\n=== 1. DATA OVERVIEW / QUALITY ===")
    for b, g in info.groupby("batch"):
        say(f"{b}: {len(g)} sessions, {g.n_chars.sum()} chars, {g.n_flashes.sum()} flashes "
            f"({g.n_targets.sum()} target), source={g.eeg_source.iloc[0]}, "
            f"total {g.duration_s.sum() / 60:.1f} min")
        say(f"   epoch rejection {g.reject_rate.mean():.1%} (range {g.reject_rate.min():.1%}-"
            f"{g.reject_rate.max():.1%}); bad channels/session {g.n_bad_ch.mean():.2f}; "
            f"noise {g.median_scale_uv.median():.1f} uV; fs_eff {g.fs_eff.mean():.3f} Hz; "
            f"timestamp jitter SD {g.ts_jitter_sd_ms.mean():.3f} ms (max {g.ts_jitter_max_ms.max():.1f} ms)")
    say(f"QC total: {info.n_chars.sum()} chars, {info.n_flashes.sum()} flashes, "
        f"marker/target-position QC agreement = {info.qc_target_consistent.min():.4f} (min)")
    u, p = stats.mannwhitneyu(info[info.batch == "batch1"].reject_rate,
                              info[info.batch == "batch2"].reject_rate)
    say(f"   rejection rate batch1 vs batch2: Mann-Whitney p={p_fmt(p)}")
    say("   channels flagged bad (session counts): "
        + str(pd.Series(",".join(info.bad_ch).split(",")).replace("", np.nan).dropna().value_counts().to_dict()))
    plot_quality(info, OUT / "fig1_quality.png")

    say("\n=== 2. ERP STATISTICS (Pz, 300-500 ms mean amplitude) ===")
    erp, ch_df, wave = erp_stats(sessions)
    erp.to_csv(OUT / "02_erp_per_session.csv", index=False)
    ch_df.to_csv(OUT / "02_erp_channel_effect.csv")
    for name, g in [("all", erp), ("batch1", erp[erp.batch == "batch1"]), ("batch2", erp[erp.batch == "batch2"])]:
        t, p = stats.ttest_1samp(g.pz_diff_uv, 0)
        try:
            wp = stats.wilcoxon(g.pz_diff_uv).pvalue
        except ValueError:
            wp = np.nan
        sig = (g.flash_p < 0.05).sum()
        say(f"{name:7s} n={len(g):2d} target-nontarget @Pz = {g.pz_diff_uv.mean():+.2f} uV "
            f"(SD {g.pz_diff_uv.std(ddof=1):.2f}); one-sample t={t:.2f} p={p_fmt(p)}, Wilcoxon p={p_fmt(wp)}; "
            f"d={g.pz_d.mean():.3f}; ROI(Pz,P3,P4,CP1,CP2) diff={g.roi_diff_uv.mean():+.2f} uV d={g.roi_d.mean():.3f}; "
            f"sessions with within-session flash-level p<0.05: {sig}/{len(g)}; "
            f"peak latency {g.peak_latency_ms.mean():.0f} +/- {g.peak_latency_ms.std(ddof=1):.0f} ms")
    b1, b2 = erp[erp.batch == "batch1"], erp[erp.batch == "batch2"]
    for col, lab in (("pz_diff_uv", "Pz diff (uV)"), ("pz_d", "Pz effect size d"),
                     ("peak_latency_ms", "peak latency (ms)")):
        t, p = stats.ttest_ind(b1[col], b2[col], equal_var=False)
        u, pu = stats.mannwhitneyu(b1[col], b2[col])
        say(f"   batch1 vs batch2 {lab}: Welch t={t:.2f} p={p_fmt(p)}, Mann-Whitney p={p_fmt(pu)}")
    top = ch_df.drop(columns="batch").mean().sort_values(ascending=False).head(6)
    say("   top channels by effect size d: " + ", ".join(f"{k} ({v:.2f})" for k, v in top.items()))
    noisy = set(info.loc[info.median_scale_uv > QC_NOISE_UV, "session"])
    plot_erp(wave, erp, ch_df, OUT / "fig2_erp.png", exclude=noisy)

    print("\nDecoding (leave-one-session-out) ...")
    auc_df, chars, all_scores = run_decoding(sessions)
    auc_df.to_csv(OUT / "03_flash_auc.csv", index=False)
    chars.to_csv(OUT / "04_char_level.csv", index=False)

    say("\n=== 3. FLASH-LEVEL AUC (held-out session) ===")
    for (cond, tag), g in auc_df.groupby(["cond", "ea"]):
        parts = []
        for b in ("batch1", "batch2"):
            v = g[g.batch == b].auc
            parts.append(f"{b} {v.mean():.3f} (SD {v.std(ddof=1):.3f}, min {v.min():.3f})")
        wp = stats.wilcoxon(g.auc - 0.5).pvalue
        say(f"{cond:7s} {tag:3s} all mean AUC {g.auc.mean():.3f} (SD {g.auc.std(ddof=1):.3f}); "
            + "; ".join(parts) + f"; vs 0.5 Wilcoxon p={p_fmt(wp)}")
    pooled = auc_df[auc_df.cond == "pooled"].pivot(index="session", columns="ea", values="auc")
    wp = stats.wilcoxon(pooled["EA"], pooled["raw"])
    say(f"   EA vs raw (pooled LOSO, paired over {len(pooled)} sessions): "
        f"mean diff {(pooled.EA - pooled.raw).mean():+.4f}, Wilcoxon p={p_fmt(wp.pvalue)}")
    for ea in ("raw", "EA"):
        a = auc_df[(auc_df.cond == "pooled") & (auc_df.ea == ea)]
        u, p = stats.mannwhitneyu(a[a.batch == "batch1"].auc, a[a.batch == "batch2"].auc)
        say(f"   pooled/{ea}: batch1 vs batch2 AUC Mann-Whitney p={p_fmt(p)}")
    plot_auc(auc_df, OUT / "fig3_auc.png")

    say("\n=== 4. CHARACTER-LEVEL ACCURACY (chance = 1/36 = 2.8%) ===")
    acc = acc_table(chars)
    acc.to_csv(OUT / "04_char_accuracy_by_reps.csv", index=False)
    key_r = (1, 3, 5, 8, 10, 15)
    for (cond, tag), _ in acc.groupby(["cond", "ea"]):
        for grp in ("all", "batch1", "batch2"):
            v = acc[(acc.cond == cond) & (acc.ea == tag) & (acc.group == grp)].set_index("reps")
            cells = []
            for r in key_r:
                x = v.loc[r]
                cells.append(f"r{r}: {x.acc * 100:.0f}% [{x.wilson_lo * 100:.0f}-{x.wilson_hi * 100:.0f}]")
            say(f"{cond:7s} {tag:3s} {grp:7s} n={int(v.n_chars.iloc[0]):3d} | " + " | ".join(cells))
    say("   (brackets = Wilson 95% CI over characters; cluster-bootstrap CIs by session in 04_char_accuracy_by_reps.csv)")

    say("\n--- ITR (Wolpaw, flash time only; 12 flashes x 180 ms measured SOA per rep) -- pooled/EA/all ---")
    v = acc[(acc.cond == "pooled") & (acc.ea == "EA") & (acc.group == "all")].set_index("reps")
    best = v.itr_bpm.idxmax()
    say("  " + " | ".join(f"r{r}: acc {v.loc[r].acc * 100:.0f}% ITR {v.loc[r].itr_bpm:.1f}" for r in key_r))
    say(f"  ITR-optimal fixed reps = {best} ({v.loc[best].itr_bpm:.1f} bits/min at {v.loc[best].acc * 100:.1f}%)")
    for target in (0.8, 0.9, 0.95):
        ok = v[v.acc >= target]
        say(f"  reps needed for >= {int(target * 100)}% accuracy (pooled/EA): "
            f"{int(ok.index.min()) if len(ok) else 'not reached'}")

    say("\n--- Batch effect on per-session character accuracy (pooled/EA) ---")
    for r in (3, 5, 8, 15):
        d = chars[(chars.cond == "pooled") & (chars.ea == "EA")]
        sa = d.groupby(["session", "batch"])[f"c{r}"].mean().reset_index()
        x1, x2 = sa[sa.batch == "batch1"][f"c{r}"].to_numpy(), sa[sa.batch == "batch2"][f"c{r}"].to_numpy()
        diffs = [RNG.choice(x2, len(x2)).mean() - RNG.choice(x1, len(x1)).mean() for _ in range(5000)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        say(f"  reps={r:2d}: batch2 {x2.mean() * 100:.0f}% vs batch1 {x1.mean() * 100:.0f}% "
            f"(session means; diff {100 * (x2.mean() - x1.mean()):+.0f} pts, bootstrap 95% CI "
            f"[{100 * lo:+.0f}, {100 * hi:+.0f}]); Mann-Whitney p={p_fmt(stats.mannwhitneyu(x1, x2).pvalue)}")
    g = chars[(chars.cond == "pooled") & (chars.ea == "EA")]
    st = settle_reps(g)
    say("\n--- Settling repetitions (first rep from which the character stays correct) ---")
    say(f"  pooled/EA: never correct at 15 reps: {np.isnan(st).sum()}/{len(st)} chars; "
        f"median settle {np.nanmedian(st):.0f}, mean {np.nanmean(st):.1f}; "
        f"<=3 reps {np.mean(st <= 3):.0%}, <=5 {np.mean(st <= 5):.0%}, <=8 {np.mean(st <= 8):.0%}, "
        f"<=10 {np.mean(st <= 10):.0%}")
    for b in ("batch1", "batch2"):
        sb = settle_reps(g[g.batch == b])
        say(f"  {b}: median settle {np.nanmedian(sb):.0f}, mean {np.nanmean(sb):.1f}, "
            f"never-correct {np.isnan(sb).sum()}/{len(sb)}")
    u, p = stats.mannwhitneyu(settle_reps(g[g.batch == 'batch1']), settle_reps(g[g.batch == 'batch2']),
                              nan_policy="omit")
    say(f"  settle reps batch1 vs batch2 Mann-Whitney p={p_fmt(p)}")

    say("\n--- Error structure at r=8 / r=15 (pooled/EA) ---")
    for r in (8, 15):
        e = g[f"e{r}"].value_counts()
        say(f"  r={r}: " + ", ".join(f"{k}={v}" for k, v in e.items())
            + "  (row/col = only that coordinate wrong; both = both wrong)")
    g2 = g.copy()
    g2["is_digit_or_blank"] = g2.apply(lambda x: GRID[x.t_row][x.t_col] in "123456789_", axis=1)
    g2["is_edge"] = (g2.t_row.isin([0, 5])) | (g2.t_col.isin([0, 5]))
    for col, lab in (("is_digit_or_blank", "digit/space target"), ("is_edge", "edge cell target")):
        for r in (5, 8):
            a, b = g2[g2[col]][f"c{r}"], g2[~g2[col]][f"c{r}"]
            tab = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
            _, p = stats.fisher_exact(tab)
            say(f"  r={r}: {lab} {a.mean() * 100:.0f}% (n={len(a)}) vs other {b.mean() * 100:.0f}% (n={len(b)}), Fisher p={p_fmt(p)}")
    plot_grid(chars, OUT / "fig5_grid_accuracy.png", r=8)
    plot_acc(acc, OUT / "fig4_char_accuracy.png")

    say("\n=== 5. TEMPORAL TRENDS ===")
    a = auc_df[(auc_df.cond == "pooled") & (auc_df.ea == "EA")]
    for b in ("batch1", "batch2"):
        v = a[a.batch == b]
        rho, p = stats.spearmanr(v.order, v.auc)
        say(f"  {b}: AUC vs session order Spearman rho={rho:+.2f} (n={len(v)}), p={p_fmt(p)}")
    v = erp
    for b in ("batch1", "batch2"):
        vv = v[v.batch == b]
        rho, p = stats.spearmanr(vv.order, vv.pz_diff_uv)
        say(f"  {b}: Pz P300 diff vs session order Spearman rho={rho:+.2f}, p={p_fmt(p)}")
    # within-session drift: first vs second half of each session
    halves = []
    for s in sessions:
        m = s["mrk"]
        k = m["kept"].to_numpy()
        y = m["is_target"].to_numpy()
        half = m["char_idx"].to_numpy() < s["info"]["n_chars"] / 2
        sc = all_scores[("pooled", "EA", s["info"]["session"])]
        if len(np.unique(y[k & half])) > 1 and len(np.unique(y[k & ~half])) > 1:
            halves.append((roc_auc_score(y[k & half], sc[k & half]),
                           roc_auc_score(y[k & ~half], sc[k & ~half])))
    halves = np.array(halves)
    wp = stats.wilcoxon(halves[:, 0], halves[:, 1]).pvalue
    say(f"  within-session drift (pooled/EA, flash AUC first vs second half of characters): "
        f"{halves[:, 0].mean():.3f} -> {halves[:, 1].mean():.3f}, paired Wilcoxon p={p_fmt(wp)} (n={len(halves)})")

    say("\n=== 6. NO-LEAK CHECK: lag estimated from the FIRST half of each session's characters, "
        "evaluated on the SECOND half only ===")
    cache_main, cache_half = {}, {}
    half_auc, half_chars = [], []
    for k, (s, sh) in enumerate(zip(sessions, sessions_half)):
        n_half = int(np.ceil(s["info"]["n_chars"] / 2))
        lda = fit_lda([x for j, x in enumerate(sessions) if j != k], True, cache_main)
        sc = score(lda, sh, True, cache_half)
        m = sh["mrk"].copy()
        m["kept"] = m["kept"] & (m["char_idx"] >= n_half)
        sh2 = dict(sh, mrk=m)
        kk, yy = m["kept"].to_numpy(), m["is_target"].to_numpy()
        half_auc.append(dict(session=s["info"]["session"], batch=s["info"]["batch"],
                             auc=roc_auc_score(yy[kk], sc[kk])))
        for rec in decode_chars(sh2, sc):
            half_chars.append(rec)
    half_auc, half_chars = pd.DataFrame(half_auc), pd.DataFrame(half_chars)
    half_auc.to_csv(OUT / "06_noleak_flash_auc.csv", index=False)
    half_chars.to_csv(OUT / "06_noleak_char_level.csv", index=False)
    ref = chars[(chars.cond == "pooled") & (chars.ea == "EA")].merge(
        half_chars[["session", "char_idx"]], on=["session", "char_idx"])
    for lab, d in (("lag from 1st half (no leak)", half_chars), ("lag from all labels (main)", ref)):
        parts = []
        for r in (1, 3, 5, 8, 10, 15):
            kc, nc = int(d[f"c{r}"].sum()), len(d)
            lo, hi = wilson(kc, nc)
            parts.append(f"r{r}: {kc / nc * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]")
        say(f"  {lab:30s} n={len(d)} chars | " + " | ".join(parts))
    for b in ("batch1", "batch2"):
        v = half_auc[half_auc.batch == b].auc
        say(f"  no-leak flash AUC {b}: mean {v.mean():.3f} (SD {v.std(ddof=1):.3f}, min {v.min():.3f})")
    say(f"  no-leak flash AUC vs 0.5 (all sessions) Wilcoxon p={p_fmt(stats.wilcoxon(half_auc.auc - 0.5).pvalue)}")
    rel_half_sessions = set(lag_df.loc[lag_df.reliable_half, "session"])
    sub = half_chars[half_chars.session.isin(rel_half_sessions)]
    sub_ref = ref[ref.session.isin(rel_half_sessions)]
    for lab, d in (("first-half lag reliable (no leak)", sub), ("   same chars, all-label lag", sub_ref)):
        parts = []
        for r in (1, 3, 5, 8, 10, 15):
            kc, nc = int(d[f"c{r}"].sum()), len(d)
            lo, hi = wilson(kc, nc)
            parts.append(f"r{r}: {kc / nc * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]")
        say(f"  {lab:34s} n={len(d)} chars ({d.session.nunique()} sessions) | " + " | ".join(parts))

    (OUT / "results_summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
