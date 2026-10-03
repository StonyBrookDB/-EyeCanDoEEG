# Emotiv Flex QC Dataset: 26-Session Results

This results directory now contains analysis of **only the 26 QC-passing
sessions** from Emotiv Flex batch1 (EmotivPRO/EDF export) and batch2 (Cortex
API/LSL). All steps reported here—lag-template estimation, epoching,
cross-validation training/testing, and cumulative character predictions—use
only these 26 sessions. The four high-noise sessions remain in the source
recordings but are not included in these results.

## Cohort and QC

- 26 sessions: 16 batch1 and 10 batch2.
- 150 characters and 27,000 marked flashes (4,500 target, 22,500 non-target);
  26,605 epochs remain after boundary/artifact rejection.
- 32 channels, nominally 256 Hz, 6×6 row/column speller, 15 repetitions.
- QC rule: per-session median channel robust SD ≤ 10 µV. The threshold was
  chosen after preliminary decoding was inspected, so the QC cohort is
  **post hoc**, not preregistered.
- Source metadata does not provide participant IDs. These results evaluate
  cross-session generalization only, not generalization to new people.

The full run summary, detailed p-values/confidence intervals, and all
cumulative per-character predictions are integrated later in this report.
Supporting per-session/per-character CSVs and figures are in the adjacent
`results/` directory.

## Marker alignment

Lag is estimated for each QC session by matching its target-minus-nontarget
ERP to a leave-one-session-out template formed from the other QC sessions.
If the template cosine similarity is below 0.75, that session uses the
median reliable lag from its own batch. The lag/template procedure uses the
session's target labels: this is **supervised alignment**, not label-free
test-time preprocessing.

- batch1: mean lag −0.374 s, SD 0.342 s; 16/16 estimates reliable.
- batch2: mean lag +0.236 s, SD 0.019 s; 9/10 reliable; session 010 uses the
  batch-median fallback.
- First-half-only and full-session lag estimates differ by median 0.000 s
  (90th percentile absolute difference 0.016 s); first-half estimates are
  reliable in 20/26 sessions.

The estimated lag is a correction inferred from ERP labels, not a hardware
measurement. It should not replace proper trigger/EEG clock synchronization
in a prospective experiment. See `results/00_lag_estimates.csv` and
`results/fig0_lag_curves.png`.

## Data quality and ERP

| | batch1 | batch2 | all QC |
|---|---:|---:|---:|
| Sessions / characters | 16 / 100 | 10 / 50 | 26 / 150 |
| Marked flashes | 18,000 | 9,000 | 27,000 |
| Mean epoch rejection | 1.7% | 1.0% | 1.5% |
| Median channel noise | 5.9 µV | 6.2 µV | 6.0 µV |

At Pz, the mean target-minus-nontarget voltage in the 300–500 ms window is
+2.08 µV across sessions (95% t interval [+1.77, +2.39], t(25)=13.82,
p=3.3e-13); all 26 session-level differences are positive. The five-channel
ROI (Pz/P3/P4/CP1/CP2) difference is +2.15 µV (95% CI [+1.88, +2.41]).
The average single-flash standardized effect is modest (d=0.30), which is
why evidence is accumulated over repeated flashes.

## Flash-level decoder performance

The analysis uses leave-one-session-out (LOSO): each test session is absent
from its training fold, and folds contain only QC sessions. The baseline is
robust channel scaling plus shrinkage LDA. EA adds a per-session,
unsupervised covariance whitening step. The output metric is flash-level
target/non-target ROC-AUC—not character accuracy.

| Evaluation | Mean AUC | SD | Range |
|---|---:|---:|---:|
| Pooled QC, EA | **0.948** | 0.025 | 0.898–0.984 |
| Pooled QC, no EA | 0.940 | 0.027 | 0.893–0.975 |
| Within-batch QC, EA | 0.946 | 0.026 | 0.893–0.984 |
| Train on other batch, EA | 0.921 | 0.025 | 0.864–0.952 |

All 26 pooled/EA session AUCs exceed 0.5; 25/26 exceed 0.9. Against chance,
the session-level Wilcoxon test gives p=3.0e-8 (the smallest possible exact
two-sided p with 26 paired observations). EA increases pooled AUC by 0.0075
on average (paired Wilcoxon p<0.001), a consistent but numerically modest
improvement. Cross-batch transfer remains above chance, but is lower than
pooled training.

## Character decoding by cumulative repetitions

For each character, flash scores are accumulated by row/column code through
the first *r* repetitions. The highest-scoring row and column intersection
is the decoded symbol. Chance accuracy is 1/36=2.78%.

| Repetitions | Correct | Accuracy | Session-cluster bootstrap 95% interval |
|---:|---:|---:|---:|
| 1 | 85/150 | 56.7% | 47.6–66.2% |
| 2 | 127/150 | 84.7% | 77.5–91.8% |
| 3 | 145/150 | **96.7%** | 94.1–99.3% |
| 4 | 147/150 | 98.0% | 95.9–100% |
| 5 | 148/150 | 98.7% | 96.5–100% |
| 6 | 149/150 | 99.3% | 97.8–100% |
| 8 | 150/150 | 100% observed | Wilson 95% interval 97.5–100% |
| 15 | 150/150 | 100% observed | Wilson 95% interval 97.5–100% |

At three repetitions (about 6.5 s of flash time), 145/150 characters are
correct. The one-sided test of accuracy >90% gives p=0.002; the
session-cluster-bootstrap lower 95% bound is 94.0%. Five repetitions
(10.8 s of flash time) gives 98.7% observed accuracy. These are offline
fixed-repetition estimates, not live adaptive-stopping results.

The maximum observed 150/150 result is not proof that the true error rate is
zero. Characters within a session are correlated; the character-independent
Wilson interval is optimistic, and a bootstrap over only 26 sessions can
become degenerate when every sampled session is perfect. Interpret the
observed perfect scores together with the finite sample size and holdout
checks, rather than as a guarantee for new users.

Wolpaw ITR is maximized at one fixed repetition (54.4 bits/min at 56.7%
accuracy; two repetitions gives 52.3 bits/min at 84.7%). It counts only the
12×SOA flash time and assumes errors are uniformly distributed over the
other 35 symbols; it is a comparison metric, not an observed typing rate.
No language model, user feedback, or online stopping was tested.

All cumulative predictions for the 150 QC characters are included in the
"Cumulative per-character predictions" section below and are also available
as a machine-readable [CSV](results/QC_character_predictions.csv).

## Lag-estimation holdout check

For each QC session, the first half of its characters estimates its lag and
the second half tests it (65 characters total). Accuracy is 89% at 3
repetitions and 94% at 5. In the 53 characters from sessions whose first-half
lag estimate was reliable, accuracy is 96% at 3 and 100% at 5. This supports
the usefulness of the timing correction, but does not make it fully
unsupervised because lag is selected using labeled ERP differences.

## Interpretation limits

1. The noise threshold is post hoc; performance is conditional on this QC
   selection and should not be represented as the average over all collected
   sessions.
2. No participant IDs are available, so 26 sessions cannot be treated as 26
   independent users or as cross-subject validation.
3. Marker-lag estimation uses target labels. Prospective use needs hardware
   synchronization or a lag estimator that does not use held-out target labels.
4. Flashes are nested within characters, and characters within sessions.
   Use session-level resampling/tests when making population claims.
5. These results use the analysis script's robust scaling, optional EA and
   shrinkage LDA. They are not numerically interchangeable with the main
   `eyecando` EA+xDAWN+LDA pipeline.

<!-- CONSOLIDATED QC DETAILS -->

## Full QC analysis run summary

```text
Using QC-passing sessions only: 26 sessions; excluded 4 by median channel noise > 10 uV.

=== 0. EEG-MARKER TIMING (QC sessions only) ===
   template posterior positive peak before anchoring: 0.63 s -> anchored at 0.40 s (all lags are relative to this convention)
batch1: lag mean -0.37 s, SD 0.34, range [-0.84, +0.22]; reliable estimate (peak cosine >= 0.75): 16/16; median peak cosine 0.88
batch2: lag mean +0.24 s, SD 0.02, range [+0.22, +0.27]; reliable estimate (peak cosine >= 0.75): 9/10; median peak cosine 0.81
   lag spread, reliable sessions only: batch1 SD 0.34 s vs batch2 SD 0.02 s (Levene p=<0.001); batch2 lag vs 0: t-test p=<0.001
   sessions without a reliable lag (fallback = batch median lag): batch2/session_010 (peak cosine 0.60)
   |lag| > 0.25 s: batch1 10/16, batch2 1/9
   first-half-only lag vs full lag: median |diff| 0.000 s, 90th pct 0.016 s; reliable first-half estimates 20/26

=== 1. DATA OVERVIEW / QUALITY ===
batch1: 16 sessions, 100 chars, 18000 flashes (3000 target), source=EmotivPRO_manual_export, total 62.7 min
   epoch rejection 1.7% (range 0.1%-4.2%); bad channels/session 0.00; noise 5.9 uV; fs_eff 256.000 Hz; timestamp jitter SD 0.000 ms (max 0.0 ms)
batch2: 10 sessions, 50 chars, 9000 flashes (1500 target), source=cortex_api, total 29.6 min
   epoch rejection 1.0% (range 0.1%-2.8%); bad channels/session 0.80; noise 6.2 uV; fs_eff 256.023 Hz; timestamp jitter SD 5.282 ms (max 83.2 ms)
QC total: 150 chars, 27000 flashes, marker/target-position QC agreement = 1.0000 (min)
   rejection rate batch1 vs batch2: Mann-Whitney p=0.069
   channels flagged bad (session counts): {'FC1': 8}

=== 2. ERP STATISTICS (Pz, 300-500 ms mean amplitude) ===
all     n=26 target-nontarget @Pz = +2.08 uV (SD 0.77); one-sample t=13.82 p=<0.001, Wilcoxon p=<0.001; d=0.302; ROI(Pz,P3,P4,CP1,CP2) diff=+2.15 uV d=0.371; sessions with within-session flash-level p<0.05: 22/26; peak latency 387 +/- 50 ms
batch1  n=16 target-nontarget @Pz = +2.15 uV (SD 0.64); one-sample t=13.33 p=<0.001, Wilcoxon p=<0.001; d=0.345; ROI(Pz,P3,P4,CP1,CP2) diff=+2.32 uV d=0.421; sessions with within-session flash-level p<0.05: 15/16; peak latency 405 +/- 23 ms
batch2  n=10 target-nontarget @Pz = +1.98 uV (SD 0.97); one-sample t=6.50 p=<0.001, Wilcoxon p=0.002; d=0.234; ROI(Pz,P3,P4,CP1,CP2) diff=+1.86 uV d=0.289; sessions with within-session flash-level p<0.05: 7/10; peak latency 359 +/- 69 ms
   batch1 vs batch2 Pz diff (uV): Welch t=0.47 p=0.645, Mann-Whitney p=0.510
   batch1 vs batch2 Pz effect size d: Welch t=2.70 p=0.013, Mann-Whitney p=0.012
   batch1 vs batch2 peak latency (ms): Welch t=2.06 p=0.066, Mann-Whitney p=0.134
   top channels by effect size d: FC1 (0.52), C3 (0.45), Cz (0.44), CP1 (0.43), FC2 (0.39), CP2 (0.36)

=== 3. FLASH-LEVEL AUC (held-out session) ===
cross   EA  all mean AUC 0.921 (SD 0.025); batch1 0.921 (SD 0.026, min 0.864); batch2 0.921 (SD 0.025, min 0.883); vs 0.5 Wilcoxon p=<0.001
cross   raw all mean AUC 0.884 (SD 0.049); batch1 0.915 (SD 0.029, min 0.851); batch2 0.835 (SD 0.031, min 0.777); vs 0.5 Wilcoxon p=<0.001
pooled  EA  all mean AUC 0.948 (SD 0.025); batch1 0.956 (SD 0.023, min 0.903); batch2 0.935 (SD 0.023, min 0.898); vs 0.5 Wilcoxon p=<0.001
pooled  raw all mean AUC 0.940 (SD 0.027); batch1 0.948 (SD 0.025, min 0.893); batch2 0.927 (SD 0.026, min 0.894); vs 0.5 Wilcoxon p=<0.001
within  EA  all mean AUC 0.946 (SD 0.026); batch1 0.956 (SD 0.024, min 0.901); batch2 0.931 (SD 0.021, min 0.893); vs 0.5 Wilcoxon p=<0.001
within  raw all mean AUC 0.940 (SD 0.031); batch1 0.951 (SD 0.027, min 0.890); batch2 0.922 (SD 0.029, min 0.881); vs 0.5 Wilcoxon p=<0.001
   EA vs raw (pooled LOSO, paired over 26 sessions): mean diff +0.0075, Wilcoxon p=<0.001
   pooled/raw: batch1 vs batch2 AUC Mann-Whitney p=0.061
   pooled/EA: batch1 vs batch2 AUC Mann-Whitney p=0.016

=== 4. CHARACTER-LEVEL ACCURACY (chance = 1/36 = 2.8%) ===
cross   EA  all     n=150 | r1: 46% [38-54] | r3: 94% [89-97] | r5: 98% [94-99] | r8: 99% [96-100] | r10: 99% [96-100] | r15: 100% [98-100]
cross   EA  batch1  n=100 | r1: 44% [35-54] | r3: 95% [89-98] | r5: 99% [95-100] | r8: 99% [95-100] | r10: 99% [95-100] | r15: 100% [96-100]
cross   EA  batch2  n= 50 | r1: 50% [37-63] | r3: 92% [81-97] | r5: 96% [87-99] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
cross   raw all     n=150 | r1: 41% [34-49] | r3: 84% [77-89] | r5: 92% [87-95] | r8: 97% [92-99] | r10: 97% [93-99] | r15: 99% [95-100]
cross   raw batch1  n=100 | r1: 47% [38-57] | r3: 91% [84-95] | r5: 99% [95-100] | r8: 100% [96-100] | r10: 100% [96-100] | r15: 99% [95-100]
cross   raw batch2  n= 50 | r1: 30% [19-44] | r3: 70% [56-81] | r5: 78% [65-87] | r8: 90% [79-96] | r10: 92% [81-97] | r15: 98% [90-100]
pooled  EA  all     n=150 | r1: 57% [49-64] | r3: 97% [92-99] | r5: 99% [95-100] | r8: 100% [98-100] | r10: 100% [98-100] | r15: 100% [98-100]
pooled  EA  batch1  n=100 | r1: 59% [49-68] | r3: 98% [93-99] | r5: 100% [96-100] | r8: 100% [96-100] | r10: 100% [96-100] | r15: 100% [96-100]
pooled  EA  batch2  n= 50 | r1: 52% [39-65] | r3: 94% [84-98] | r5: 96% [87-99] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
pooled  raw all     n=150 | r1: 55% [47-62] | r3: 95% [90-97] | r5: 99% [96-100] | r8: 100% [98-100] | r10: 100% [98-100] | r15: 100% [98-100]
pooled  raw batch1  n=100 | r1: 58% [48-67] | r3: 97% [92-99] | r5: 100% [96-100] | r8: 100% [96-100] | r10: 100% [96-100] | r15: 100% [96-100]
pooled  raw batch2  n= 50 | r1: 48% [35-61] | r3: 90% [79-96] | r5: 98% [90-100] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
within  EA  all     n=150 | r1: 54% [46-62] | r3: 94% [89-97] | r5: 99% [95-100] | r8: 100% [98-100] | r10: 100% [98-100] | r15: 100% [98-100]
within  EA  batch1  n=100 | r1: 54% [44-63] | r3: 95% [89-98] | r5: 100% [96-100] | r8: 100% [96-100] | r10: 100% [96-100] | r15: 100% [96-100]
within  EA  batch2  n= 50 | r1: 54% [40-67] | r3: 92% [81-97] | r5: 96% [87-99] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
within  raw all     n=150 | r1: 55% [47-63] | r3: 94% [89-97] | r5: 99% [96-100] | r8: 100% [98-100] | r10: 100% [98-100] | r15: 100% [98-100]
within  raw batch1  n=100 | r1: 58% [48-67] | r3: 96% [90-98] | r5: 100% [96-100] | r8: 100% [96-100] | r10: 100% [96-100] | r15: 100% [96-100]
within  raw batch2  n= 50 | r1: 50% [37-63] | r3: 90% [79-96] | r5: 98% [90-100] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
   (brackets = Wilson 95% CI over characters; cluster-bootstrap CIs by session in 04_char_accuracy_by_reps.csv)

--- ITR (Wolpaw, flash time only; 12 flashes x 180 ms measured SOA per rep) -- pooled/EA/all ---
  r1: acc 57% ITR 54.4 | r3: acc 97% ITR 44.3 | r5: acc 99% ITR 27.8 | r8: acc 100% ITR 18.0 | r10: acc 100% ITR 14.4 | r15: acc 100% ITR 9.6
  ITR-optimal fixed reps = 1 (54.4 bits/min at 56.7%)
  reps needed for >= 80% accuracy (pooled/EA): 2
  reps needed for >= 90% accuracy (pooled/EA): 3
  reps needed for >= 95% accuracy (pooled/EA): 3

--- Batch effect on per-session character accuracy (pooled/EA) ---
  reps= 3: batch2 94% vs batch1 99% (session means; diff -5 pts, bootstrap 95% CI [-11, +1]); Mann-Whitney p=0.205
  reps= 5: batch2 96% vs batch1 100% (session means; diff -4 pts, bootstrap 95% CI [-10, +0]); Mann-Whitney p=0.077
  reps= 8: batch2 100% vs batch1 100% (session means; diff +0 pts, bootstrap 95% CI [+0, +0]); Mann-Whitney p=1.000
  reps=15: batch2 100% vs batch1 100% (session means; diff +0 pts, bootstrap 95% CI [+0, +0]); Mann-Whitney p=1.000

--- Settling repetitions (first rep from which the character stays correct) ---
  pooled/EA: never correct at 15 reps: 0/150 chars; median settle 1, mean 1.7; <=3 reps 95%, <=5 99%, <=8 100%, <=10 100%
  batch1: median settle 1, mean 1.6, never-correct 0/100
  batch2: median settle 2, mean 1.9, never-correct 0/50
  settle reps batch1 vs batch2 Mann-Whitney p=0.334

--- Error structure at r=8 / r=15 (pooled/EA) ---
  r=8: ok=150  (row/col = only that coordinate wrong; both = both wrong)
  r=15: ok=150  (row/col = only that coordinate wrong; both = both wrong)
  r=5: digit/space target 100% (n=4) vs other 99% (n=146), Fisher p=1.000
  r=8: digit/space target 100% (n=4) vs other 100% (n=146), Fisher p=1.000
  r=5: edge cell target 99% (n=80) vs other 99% (n=70), Fisher p=1.000
  r=8: edge cell target 100% (n=80) vs other 100% (n=70), Fisher p=1.000

=== 5. TEMPORAL TRENDS ===
  batch1: AUC vs session order Spearman rho=+0.32 (n=16), p=0.222
  batch2: AUC vs session order Spearman rho=+0.22 (n=10), p=0.533
  batch1: Pz P300 diff vs session order Spearman rho=+0.17, p=0.520
  batch2: Pz P300 diff vs session order Spearman rho=+0.19, p=0.603
  within-session drift (pooled/EA, flash AUC first vs second half of characters): 0.947 -> 0.948, paired Wilcoxon p=0.764 (n=26)

=== 6. NO-LEAK CHECK: lag estimated from the FIRST half of each session's characters, evaluated on the SECOND half only ===
  lag from 1st half (no leak)    n=65 chars | r1: 48% [36-60] | r3: 89% [79-95] | r5: 94% [85-98] | r8: 94% [85-98] | r10: 94% [85-98] | r15: 94% [85-98]
  lag from all labels (main)     n=65 chars | r1: 54% [42-65] | r3: 95% [87-98] | r5: 98% [92-100] | r8: 100% [94-100] | r10: 100% [94-100] | r15: 100% [94-100]
  no-leak flash AUC batch1: mean 0.889 (SD 0.167, min 0.391)
  no-leak flash AUC batch2: mean 0.921 (SD 0.046, min 0.823)
  no-leak flash AUC vs 0.5 (all sessions) Wilcoxon p=<0.001
  first-half lag reliable (no leak)  n=53 chars (20 sessions) | r1: 53% [40-66] | r3: 96% [87-99] | r5: 100% [93-100] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
     same chars, all-label lag       n=53 chars (20 sessions) | r1: 55% [41-67] | r3: 96% [87-99] | r5: 100% [93-100] | r8: 100% [93-100] | r10: 100% [93-100] | r15: 100% [93-100]
```
## Detailed QC statistical tests

```text
QC SUBSET STATISTICS
Analysis input contains only the 26 sessions passing median channel noise <= 10 uV. The threshold was selected after preliminary decoding, so this is a post hoc QC cohort.
All p-values are exact (two-sided unless stated). Sessions, not characters or flashes, are the unit of replication wherever a test is run on session-level numbers.

[1] SAMPLE
  batch1: 16 sessions, 100 characters, 18000 flashes (3000 target), 62.7 min; epoch rejection 1.7%; noise median 5.9 uV
  batch2: 10 sessions, 50 characters, 9000 flashes (1500 target), 29.6 min; epoch rejection 1.0%; noise median 6.2 uV
  all   : 26 sessions, 150 characters, 27000 flashes (4500 target), 92.3 min; epoch rejection 1.5%; noise median 6.0 uV

[2] EEG-MARKER LAG (estimated per session; positive = EEG event later than marker)
  batch1: mean -0.374 s, SD 0.342, range [-0.84, +0.22]; reliable (template cosine >= 0.75) 16/16
  batch2: mean +0.236 s, SD 0.019, range [+0.22, +0.27]; reliable (template cosine >= 0.75) 9/10
  QC sessions that used the batch-median fallback lag (unreliable estimate): batch2/session_010 (cos 0.60)
  lag variability batch1 vs batch2 (Levene): W=17.0, p=3.8e-04
  batch2 lag differs from 0: one-sample t=39.9, p=1.9e-11

[3] ERP: Pz mean amplitude 300-500 ms, target minus non-target (each session = one observation)
  all    n=26: diff = +2.08 uV, 95% CI [+1.77, +2.39]; t(25)=13.82, p=3.3e-13; Wilcoxon p=3.0e-08; positive in 26/26 sessions; mean single-flash d = 0.30 (95% CI [0.25, 0.35])
  batch1 n=16: diff = +2.15 uV, 95% CI [+1.80, +2.49]; t(15)=13.33, p=1.0e-09; Wilcoxon p=3.1e-05; positive in 16/16 sessions; mean single-flash d = 0.35 (95% CI [0.28, 0.41])
  batch2 n=10: diff = +1.98 uV, 95% CI [+1.29, +2.67]; t(9)=6.50, p=1.1e-04; Wilcoxon p=0.002; positive in 10/10 sessions; mean single-flash d = 0.23 (95% CI [0.17, 0.30])
  within-session single-flash Welch t-test p<0.05 in 22/26 sessions (p<0.001 in 15); median session p=1.3e-04
  ROI (Pz,P3,P4,CP1,CP2): +2.15 uV, 95% CI [+1.88, +2.41], t=16.51, p=5.9e-15
  batch1 vs batch2 Pz diff (uV)      : +2.15 vs +1.98; Welch t=0.47 p=0.645; Mann-Whitney p=0.510
  batch1 vs batch2 single-flash d    : +0.35 vs +0.23; Welch t=2.70 p=0.013; Mann-Whitney p=0.012
  batch1 vs batch2 peak latency (ms) : +405.27 vs +358.98; Welch t=2.06 p=0.066; Mann-Whitney p=0.134

[4] FLASH-LEVEL AUC (held-out session; trained on all other sessions; EA unless stated)
  pooled/EA, n=26 sessions: mean AUC 0.948, SD 0.025, 95% CI [0.937, 0.958], min 0.898, max 0.984; AUC>0.5 in 26/26 sessions, AUC>0.9 in 25/26
    vs chance 0.5: one-sample t(25)=90.8, p=5.0e-33; Wilcoxon signed-rank p=3.0e-08 (smallest possible two-sided exact p with n=26 is 3.0e-08)
    batch1: n=16, mean 0.956, 95% CI [0.943, 0.968], min 0.903
    batch2: n=10, mean 0.935, 95% CI [0.918, 0.951], min 0.898
    batch1 vs batch2: diff +0.021; Welch t=2.29 p=0.033; Mann-Whitney p=0.016
    EA vs raw (paired, 26 sessions): mean diff +0.0075 AUC, 95% CI [+0.0049, +0.0101]; paired t=5.99 p=3.0e-06; Wilcoxon p=5.7e-07; EA better in 24/26; paired d=1.17
    pooled vs within (train on same batch only): 0.948 vs 0.946; paired diff +0.0015, Wilcoxon p=0.059; within AUC vs 0.5: Wilcoxon p=3.0e-08, min 0.893
    pooled vs cross (train on the OTHER batch only): 0.948 vs 0.921; paired diff +0.0269, Wilcoxon p=3.0e-08; cross AUC vs 0.5: Wilcoxon p=3.0e-08, min 0.864

[5] CHARACTER ACCURACY (pooled LOSO, EA); chance = 1/36 = 2.78%
  n = 150 characters in 26 sessions. Intervals: Wilson (treats characters as independent) and cluster bootstrap over sessions (10000 resamples; the more honest one).
  r= 1 (  2.2 s of flashes): 85/150 =  56.7%  Wilson [48.7, 64.3]  cluster-boot [47.6, 66.2]  vs chance: binomial one-sided p=2.1e-90; per-session accuracy mean 58.2%, sessions at 100%: 3/26, sessions below 80%: 19
  r= 2 (  4.3 s of flashes): 127/150 =  84.7%  Wilson [78.0, 89.6]  cluster-boot [77.5, 91.8]  vs chance: binomial one-sided p=8.6e-172; per-session accuracy mean 87.0%, sessions at 100%: 14/26, sessions below 80%: 5
  r= 3 (  6.5 s of flashes): 145/150 =  96.7%  Wilson [92.4, 98.6]  cluster-boot [94.1, 99.3]  vs chance: binomial one-sided p=1.1e-217; per-session accuracy mean 96.9%, sessions at 100%: 21/26, sessions below 80%: 0
  r= 4 (  8.6 s of flashes): 147/150 =  98.0%  Wilson [94.3, 99.3]  cluster-boot [95.9, 100.0]  vs chance: binomial one-sided p=8.5e-224; per-session accuracy mean 98.5%, sessions at 100%: 23/26, sessions below 80%: 0
  r= 5 ( 10.8 s of flashes): 148/150 =  98.7%  Wilson [95.3, 99.6]  cluster-boot [96.5, 100.0]  vs chance: binomial one-sided p=4.9e-227; per-session accuracy mean 98.5%, sessions at 100%: 24/26, sessions below 80%: 0
  r= 6 ( 13.0 s of flashes): 149/150 =  99.3%  Wilson [96.3, 99.9]  cluster-boot [97.8, 100.0]  vs chance: binomial one-sided p=1.9e-230; per-session accuracy mean 99.2%, sessions at 100%: 25/26, sessions below 80%: 0
  r= 8 ( 17.3 s of flashes): 150/150 = 100.0%  Wilson [97.5, 100.0]  cluster-boot [100.0, 100.0]  vs chance: binomial one-sided p=3.6e-234; per-session accuracy mean 100.0%, sessions at 100%: 26/26, sessions below 80%: 0
  r=15 ( 32.4 s of flashes): 150/150 = 100.0%  Wilson [97.5, 100.0]  cluster-boot [100.0, 100.0]  vs chance: binomial one-sided p=3.6e-234; per-session accuracy mean 100.0%, sessions at 100%: 26/26, sessions below 80%: 0
  Holm-adjusted over the 3 pre-specified tests (flash AUC>0.5 Wilcoxon, char acc>chance at r=3, r=5): p=3.0e-08, p=2.2e-217, p=1.5e-226

  Can we say accuracy is at least 90% at 3 / 5 reps?  (one-sided, H0: accuracy <= 90%)
    r=3: 145/150; binomial p=0.002; cluster-bootstrap lower 95% bound 94.0%
    r=5: 148/150; binomial p=2.1e-05; cluster-bootstrap lower 95% bound 96.6%

  EA vs raw, paired on the same 150 characters (exact McNemar):
    r=1: EA 56.7% vs raw 54.7%; EA-only correct 10, raw-only correct 7, p=0.629
    r=2: EA 84.7% vs raw 82.0%; EA-only correct 5, raw-only correct 1, p=0.219
    r=3: EA 96.7% vs raw 94.7%; EA-only correct 4, raw-only correct 1, p=0.375
    r=5: EA 98.7% vs raw 99.3%; EA-only correct 0, raw-only correct 1, p=1.000

  Batch comparison (session-level accuracy; unit = session):
    r=1: batch1 62.0% (n=16) vs batch2 52.0% (n=10); diff +10.0 pts; Mann-Whitney p=0.295
    r=3: batch1 98.8% (n=16) vs batch2 94.0% (n=10); diff +4.7 pts; Mann-Whitney p=0.205
    r=5: batch1 100.0% (n=16) vs batch2 96.0% (n=10); diff +4.0 pts; Mann-Whitney p=0.077
    r=8: batch1 100.0% (n=16) vs batch2 100.0% (n=10); diff +0.0 pts; Mann-Whitney p=1.000

  Training scope (r=3 and r=5, same 150 characters):
    within r=3: 94.0% vs pooled 96.7%; pooled-only correct 5, within-only correct 1, McNemar p=0.219
    within r=5: 98.7% vs pooled 98.7%; pooled-only correct 0, within-only correct 0, McNemar p=1.000
    cross  r=3: 94.0% vs pooled 96.7%; pooled-only correct 4, cross-only correct 0, McNemar p=0.125
    cross  r=5: 98.0% vs pooled 98.7%; pooled-only correct 2, cross-only correct 1, McNemar p=1.000

  Where errors remain at r=3 (pooled/EA):
    5 wrong characters in 5 sessions; error type {'row': 3, 'col': 2}; sessions: batch1/session_005, batch1/session_006, batch2/session_002, batch2/session_004, batch2/session_010

[6] SPEED (flash time only: 12 flashes x 0.18 s measured SOA per repetition)
    r=1: accuracy  56.7%,  2.2 s/char, ITR  54.4 bits/min (Wolpaw)
    r=2: accuracy  84.7%,  4.3 s/char, ITR  52.3 bits/min (Wolpaw)
    r=3: accuracy  96.7%,  6.5 s/char, ITR  44.3 bits/min (Wolpaw)
    r=4: accuracy  98.0%,  8.6 s/char, ITR  34.2 bits/min (Wolpaw)
    r=5: accuracy  98.7%, 10.8 s/char, ITR  27.8 bits/min (Wolpaw)
    r=6: accuracy  99.3%, 13.0 s/char, ITR  23.5 bits/min (Wolpaw)
    (Wolpaw ITR assumes errors are uniformly distributed over the other 35 symbols and ignores cue/rest time; treat it as an upper-bound comparison across settings, not an achievable rate.)
    reps until the decision stays correct: median 1, mean 1.72; 1 rep: 55%, <=2: 83%, <=3: 95%, <=5: 99%, max 7

[7] NO-LEAK CHECK (lag estimated from the first half of each session's characters only; evaluated on the second half)
  second-half characters in QC sessions: n=65 in 26 sessions
  all QC sessions: n=65 chars, 26 sessions
    r=1: 31/65 = 47.7%  Wilson [36, 60]  cluster-boot [36, 60]  vs chance p=7.6e-31
    r=3: 58/65 = 89.2%  Wilson [79, 95]  cluster-boot [79, 97]  vs chance p=3.1e-82
    r=5: 61/65 = 93.8%  Wilson [85, 98]  cluster-boot [84, 100]  vs chance p=7.0e-90
    r=8: 61/65 = 93.8%  Wilson [85, 98]  cluster-boot [84, 100]  vs chance p=7.0e-90
  QC sessions where first-half lag was reliable: n=53 chars, 20 sessions
    r=1: 28/53 = 52.8%  Wilson [40, 66]  cluster-boot [40, 67]  vs chance p=1.2e-29
    r=3: 51/53 = 96.2%  Wilson [87, 99]  cluster-boot [91, 100]  vs chance p=5.5e-77
    r=5: 53/53 = 100.0%  Wilson [93, 100]  cluster-boot [100, 100]  vs chance p=3.3e-83
    r=8: 53/53 = 100.0%  Wilson [93, 100]  cluster-boot [100, 100]  vs chance p=3.3e-83
  second-half flash AUC (QC): mean 0.902, 95% CI [0.848, 0.956], Wilcoxon vs 0.5 p=8.9e-08; sessions with first-half lag unreliable: 6/26

[8] TEMPORAL TREND (AUC vs session order, pooled/EA)
  batch1 (n=16): Spearman rho=+0.32, p=0.222
  batch2 (n=10): Spearman rho=+0.22, p=0.533
```
## Cumulative per-character predictions

## Summary

| Session | Phrase | Characters | Accuracy r=1 | r=3 | r=5 | r=8 | r=15 |
|---|---|---:|---:|---:|---:|---:|---:|
| batch1/session_005 | GOOD_FOCUS | 10 | 20% | 90% | 100% | 100% | 100% |
| batch1/session_006 | LOVELY_DAY | 10 | 30% | 90% | 100% | 100% | 100% |
| batch1/session_007 | NEED_WATER | 10 | 60% | 100% | 100% | 100% | 100% |
| batch1/session_008 | OPEN_WINDOW | 11 | 64% | 100% | 100% | 100% | 100% |
| batch1/session_009 | WATER | 5 | 60% | 100% | 100% | 100% | 100% |
| batch1/session_010 | CLOTH | 5 | 100% | 100% | 100% | 100% | 100% |
| batch1/session_011 | STAND | 5 | 100% | 100% | 100% | 100% | 100% |
| batch1/session_012 | SLEPT | 5 | 100% | 100% | 100% | 100% | 100% |
| batch1/session_013 | HUNGRY | 6 | 83% | 100% | 100% | 100% | 100% |
| batch1/session_014 | HAPPY | 5 | 40% | 100% | 100% | 100% | 100% |
| batch1/session_015 | PEACH | 5 | 80% | 100% | 100% | 100% | 100% |
| batch1/session_016 | MUSIC | 5 | 60% | 100% | 100% | 100% | 100% |
| batch1/session_017 | WORSE | 5 | 60% | 100% | 100% | 100% | 100% |
| batch1/session_018 | CANDY | 5 | 60% | 100% | 100% | 100% | 100% |
| batch1/session_019 | RICE | 4 | 50% | 100% | 100% | 100% | 100% |
| batch1/session_020 | NICE | 4 | 25% | 100% | 100% | 100% | 100% |
| batch2/session_001 | HOUSE | 5 | 40% | 100% | 100% | 100% | 100% |
| batch2/session_002 | PLANT | 5 | 40% | 80% | 100% | 100% | 100% |
| batch2/session_003 | QUICK | 5 | 20% | 100% | 100% | 100% | 100% |
| batch2/session_004 | ZEBRA | 5 | 60% | 80% | 100% | 100% | 100% |
| batch2/session_005 | JUMPY | 5 | 40% | 100% | 80% | 100% | 100% |
| batch2/session_006 | VIVID | 5 | 80% | 100% | 100% | 100% | 100% |
| batch2/session_007 | WATER | 5 | 60% | 100% | 100% | 100% | 100% |
| batch2/session_008 | YOUTH | 5 | 80% | 100% | 100% | 100% | 100% |
| batch2/session_009 | KNIFE | 5 | 60% | 100% | 100% | 100% | 100% |
| batch2/session_010 | BRAVE | 5 | 40% | 80% | 80% | 100% | 100% |

## batch1/session_005 — GOOD_FOCUS

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | G (2,1) | 5 | 1 | I | L | I | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** |
| 1 | O (3,3) | 2 | P | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 2 | O (3,3) | 3 | 7 | 7 | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 3 | D (1,4) | 1 | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| 4 | _ (6,6) | 2 | R | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** |
| 5 | F (1,6) | 3 | K | V | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** |
| 6 | O (3,3) | 1 | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 7 | C (1,3) | 2 | 7 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 8 | U (4,3) | 3 | T | T | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 9 | S (4,1) | 2 | K | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |

## batch1/session_006 — LOVELY_DAY

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | L (2,6) | 5 | Q | F | R | R | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** |
| 1 | O (3,3) | 2 | P | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 2 | V (4,4) | 1 | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** |
| 3 | E (1,5) | 2 | K | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 4 | L (2,6) | 1 | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** |
| 5 | Y (5,1) | 3 | 1 | 1 | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| 6 | _ (6,6) | 2 | X | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** |
| 7 | D (1,4) | 2 | J | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| 8 | A (1,1) | 2 | B | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 9 | Y (5,1) | 1 | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## batch1/session_007 — NEED_WATER

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | N (3,2) | 3 | T | T | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 1 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 2 | E (1,5) | 3 | D | D | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 3 | D (1,4) | 1 | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| 4 | _ (6,6) | 3 | 3 | 9 | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** |
| 5 | W (4,5) | 1 | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |
| 6 | A (1,1) | 1 | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 7 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 8 | E (1,5) | 1 | **E** | K | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 9 | R (3,6) | 3 | L | L | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |

## batch1/session_008 — OPEN_WINDOW

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | O (3,3) | 1 | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 1 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 2 | E (1,5) | 2 | 9 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 3 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 4 | _ (6,6) | 2 | 8 | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** | **_** |
| 5 | W (4,5) | 1 | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |
| 6 | I (2,3) | 1 | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 7 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 8 | D (1,4) | 3 | J | J | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| 9 | O (3,3) | 2 | U | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 10 | W (4,5) | 1 | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |

## batch1/session_009 — WATER

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | W (4,5) | 1 | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |
| 1 | A (1,1) | 1 | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 2 | T (4,2) | 2 | B | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 3 | E (1,5) | 2 | 5 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 4 | R (3,6) | 1 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |

## batch1/session_010 — CLOTH

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | C (1,3) | 1 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 1 | L (2,6) | 1 | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** |
| 2 | O (3,3) | 1 | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 3 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 4 | H (2,2) | 1 | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |

## batch1/session_011 — STAND

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | S (4,1) | 1 | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |
| 1 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 2 | A (1,1) | 1 | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 3 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 4 | D (1,4) | 1 | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |

## batch1/session_012 — SLEPT

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | S (4,1) | 1 | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |
| 1 | L (2,6) | 1 | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** |
| 2 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 3 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 4 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |

## batch1/session_013 — HUNGRY

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | H (2,2) | 1 | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |
| 1 | U (4,3) | 1 | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 2 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 3 | G (2,1) | 1 | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** | **G** |
| 4 | R (3,6) | 1 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| 5 | Y (5,1) | 2 | S | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## batch1/session_014 — HAPPY

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | H (2,2) | 3 | I | I | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |
| 1 | A (1,1) | 2 | B | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 2 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 3 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 4 | Y (5,1) | 3 | 4 | Z | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## batch1/session_015 — PEACH

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 1 | E (1,5) | 2 | D | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 2 | A (1,1) | 1 | **A** | B | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 3 | C (1,3) | 1 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 4 | H (2,2) | 1 | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |

## batch1/session_016 — MUSIC

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | M (3,1) | 1 | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** |
| 1 | U (4,3) | 2 | X | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 2 | S (4,1) | 1 | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |
| 3 | I (2,3) | 1 | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 4 | C (1,3) | 3 | F | F | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |

## batch1/session_017 — WORSE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | W (4,5) | 1 | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |
| 1 | O (3,3) | 2 | I | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 2 | R (3,6) | 1 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| 3 | S (4,1) | 1 | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |
| 4 | E (1,5) | 2 | K | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |

## batch1/session_018 — CANDY

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | C (1,3) | 1 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 1 | A (1,1) | 2 | G | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 2 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 3 | D (1,4) | 1 | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| 4 | Y (5,1) | 2 | M | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## batch1/session_019 — RICE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | R (3,6) | 2 | 4 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| 1 | I (2,3) | 2 | U | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 2 | C (1,3) | 1 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 3 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |

## batch1/session_020 — NICE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | N (3,2) | 2 | R | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 1 | I (2,3) | 2 | G | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 2 | C (1,3) | 2 | B | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 3 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |

## batch2/session_001 — HOUSE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | H (2,2) | 1 | **H** | Z | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |
| 1 | O (3,3) | 2 | C | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 2 | U (4,3) | 1 | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 3 | S (4,1) | 2 | A | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** | **S** |
| 4 | E (1,5) | 3 | F | F | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |

## batch2/session_002 — PLANT

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | P (3,4) | 1 | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 1 | L (2,6) | 1 | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** |
| 2 | A (1,1) | 3 | G | G | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 3 | N (3,2) | 2 | Z | **N** | Z | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 4 | T (4,2) | 2 | Z | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |

## batch2/session_003 — QUICK

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Q (3,5) | 2 | D | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** | **Q** |
| 1 | U (4,3) | 2 | C | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 2 | I (2,3) | 2 | O | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 3 | C (1,3) | 1 | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** | **C** |
| 4 | K (2,5) | 2 | G | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** |

## batch2/session_004 — ZEBRA

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Z (5,2) | 1 | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** | **Z** |
| 1 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 2 | B (1,2) | 2 | C | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** |
| 3 | R (3,6) | 1 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| 4 | A (1,1) | 4 | H | H | G | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |

## batch2/session_005 — JUMPY

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | J (2,4) | 2 | B | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** | **J** |
| 1 | U (4,3) | 1 | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 2 | M (3,1) | 2 | S | **M** | **M** | **M** | N | N | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** |
| 3 | P (3,4) | 2 | D | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** | **P** |
| 4 | Y (5,1) | 1 | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

## batch2/session_006 — VIVID

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | V (4,4) | 1 | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** |
| 1 | I (2,3) | 1 | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 2 | V (4,4) | 1 | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** |
| 3 | I (2,3) | 2 | G | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 4 | D (1,4) | 1 | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |

## batch2/session_007 — WATER

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | W (4,5) | 3 | K | K | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** | **W** |
| 1 | A (1,1) | 1 | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 2 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 3 | E (1,5) | 1 | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
| 4 | R (3,6) | 3 | Q | Q | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |

## batch2/session_008 — YOUTH

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Y (5,1) | 1 | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| 1 | O (3,3) | 2 | C | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** | **O** |
| 2 | U (4,3) | 1 | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** | **U** |
| 3 | T (4,2) | 1 | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** | **T** |
| 4 | H (2,2) | 1 | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** | **H** |

## batch2/session_009 — KNIFE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | K (2,5) | 1 | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** | **K** |
| 1 | N (3,2) | 1 | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** | **N** |
| 2 | I (2,3) | 2 | C | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** | **I** |
| 3 | F (1,6) | 1 | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** |
| 4 | E (1,5) | 2 | D | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |

## batch2/session_010 — BRAVE

| idx | Target (row,col) | First correct | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 |
|---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | B (1,2) | 2 | T | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** | **B** |
| 1 | R (3,6) | 1 | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| 2 | A (1,1) | 1 | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** | **A** |
| 3 | V (4,4) | 3 | M | S | **V** | S | S | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** | **V** |
| 4 | E (1,5) | 2 | F | **E** | D | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** | **E** |
