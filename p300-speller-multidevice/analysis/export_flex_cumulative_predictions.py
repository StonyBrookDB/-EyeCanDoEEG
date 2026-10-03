"""Export cumulative character predictions for all QC Flex sessions.

Run from the project directory:
    python analysis/export_flex_cumulative_predictions.py

For each held-out QC session, fit the same EA + shrinkage-LDA model on the
other 29 sessions, then record the predicted character after 1..15
repetitions. Results are saved to results/ as Markdown and CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_flex_batches as analysis


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results"
QC_NOISE_UV = 10.0
N_REPS = 15


def cumulative_predictions(session: dict, scores: np.ndarray) -> list[dict]:
    markers = session["mrk"]
    kept = markers["kept"].to_numpy()
    codes = markers["code"].to_numpy()
    repetitions = markers["rep"].to_numpy()
    char_indices = markers["char_idx"].to_numpy()
    results = []

    for char_idx in sorted(np.unique(char_indices)):
        select = (char_indices == char_idx) & kept
        if not select.any():
            continue

        score_sum = np.zeros((13, N_REPS + 1), dtype=np.float64)
        flash_count = np.zeros((13, N_REPS + 1), dtype=np.int16)
        np.add.at(
            score_sum,
            (codes[select], repetitions[select]),
            scores[select],
        )
        np.add.at(
            flash_count,
            (codes[select], repetitions[select]),
            1,
        )

        cumulative_sum = score_sum.cumsum(axis=1)
        cumulative_count = flash_count.cumsum(axis=1)
        truth_row = int(markers.loc[char_indices == char_idx, "t_row"].iloc[0])
        truth_col = int(markers.loc[char_indices == char_idx, "t_col"].iloc[0])
        truth = analysis.GRID[truth_row][truth_col]

        predictions = []
        for n_repetitions in range(1, N_REPS + 1):
            mean_scores = np.divide(
                cumulative_sum[:, n_repetitions],
                cumulative_count[:, n_repetitions],
                out=np.full(13, -np.inf),
                where=cumulative_count[:, n_repetitions] > 0,
            )
            predicted_col = int(np.argmax(mean_scores[1:7]))
            predicted_row = int(np.argmax(mean_scores[7:13]))
            predictions.append(analysis.GRID[predicted_row][predicted_col])

        first_correct = next(
            (i + 1 for i, prediction in enumerate(predictions) if prediction == truth),
            None,
        )
        results.append(
            {
                "session": session["info"]["session"],
                "batch": session["info"]["batch"],
                "char_idx": int(char_idx),
                "true_symbol": truth,
                "target_row_1based": truth_row + 1,
                "target_col_1based": truth_col + 1,
                "n_retained_flashes": int(select.sum()),
                "first_correct_cumulative_repetition": first_correct,
                **{f"prediction_r{i + 1}": pred for i, pred in enumerate(predictions)},
                **{
                    f"correct_r{i + 1}": int(pred == truth)
                    for i, pred in enumerate(predictions)
                },
            }
        )
    return results


def markdown_cell(symbol: str, truth: str) -> str:
    return f"**{symbol}**" if symbol == truth else symbol


def write_report(predictions: pd.DataFrame, session_meta: dict[str, dict]) -> None:
    lines = [
        "# Emotiv Flex QC sessions: cumulative character decoding",
        "",
        "Each prediction at repetition `r` uses the accumulated scores from "
        "repetitions 1 through `r` (not repetition `r` alone). Bold predictions "
        "match the marker-derived target symbol. Repeated symbols are separate "
        "character positions; `_` denotes the blank-space cell.",
        "",
        "## Evaluation protocol",
        "",
        "- QC subset: session median robust channel SD ≤ 10 µV; 26 sessions, "
        "150 characters.",
        "- Only the 26 QC-passing sessions enter lag-template estimation, "
        "training, and testing. For each held-out session, train on the other "
        "25 QC sessions (pooled leave-one-session-out).",
        "- Per-session Euclidean Alignment followed by shrinkage LDA; row and "
        "column scores are accumulated separately and averaged over available "
        "flashes, then the highest-scoring row/column intersection is decoded.",
        "- The target position comes from the row/column target markers.",
        "- Important limitation: marker-lag estimation used each session's "
        "target labels. This is supervised alignment, not label-free test-time "
        "processing. Session metadata does not identify subjects, so results "
        "show cross-session—not cross-subject—generalization.",
        "- The QC threshold was selected after initial decoding was inspected; "
        "it is post hoc, not preregistered.",
        "",
        "## Summary",
        "",
        "| Session | Phrase | Characters | Accuracy r=1 | r=3 | r=5 | r=8 | r=15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for session_id in sorted(session_meta):
        group = predictions[predictions["session"] == session_id]
        phrase = session_meta[session_id]["true_text"]
        accuracies = {
            r: group[f"correct_r{r}"].mean() for r in (1, 3, 5, 8, 15)
        }
        lines.append(
            f"| {session_id} | {phrase} | {len(group)} | "
            + " | ".join(f"{accuracies[r]:.0%}" for r in (1, 3, 5, 8, 15))
            + " |"
        )

    header = (
        "| idx | Target (row,col) | First correct | "
        + " | ".join(f"r{i}" for i in range(1, N_REPS + 1))
        + " |"
    )
    divider = (
        "|---:|:---:|---:|"
        + "|".join([":---:"] * N_REPS)
        + "|"
    )
    for session_id in sorted(session_meta):
        info = session_meta[session_id]
        group = predictions[predictions["session"] == session_id]
        lines.extend(
            [
                "",
                f"## {session_id} — {info['true_text']}",
                "",
                header,
                divider,
            ]
        )
        for _, row in group.sort_values("char_idx").iterrows():
            first_correct = row["first_correct_cumulative_repetition"]
            first_text = "never" if pd.isna(first_correct) else str(int(first_correct))
            preds = [
                markdown_cell(row[f"prediction_r{r}"], row["true_symbol"])
                for r in range(1, N_REPS + 1)
            ]
            lines.append(
                f"| {int(row['char_idx'])} | {row['true_symbol']} "
                f"({int(row['target_row_1based'])},{int(row['target_col_1based'])}) "
                f"| {first_text} | "
                + " | ".join(preds)
                + " |"
            )

    report = OUTPUT / "QC_character_predictions.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    print("Preparing sessions ...")
    all_prepared = analysis.prepare_all()
    prepared = [
        session
        for session in all_prepared
        if session["info"]["median_scale_uv"] <= QC_NOISE_UV
    ]
    excluded = len(all_prepared) - len(prepared)
    print(
        f"Keeping {len(prepared)} QC sessions; excluding {excluded} sessions "
        f"with median channel noise > {QC_NOISE_UV:g} uV."
    )
    print("Estimating marker lags using QC sessions only ...")
    lags, _, _ = analysis.estimate_all_lags(prepared)
    lag_lookup = lags.set_index("session")

    sessions = [
        analysis.epoch_session(
            session,
            int(lag_lookup.loc[session["info"]["session"], "lag_used"]),
        )
        for session in prepared
    ]
    cache: dict = {}
    rows = []
    metadata = {}

    for index, test_session in enumerate(sessions, start=1):
        test_id = test_session["info"]["session"]
        training_sessions = [s for s in sessions if s is not test_session]
        model = analysis.fit_lda(training_sessions, ea=True, cache=cache)
        test_scores = analysis.score(model, test_session, ea=True, cache=cache)
        rows.extend(cumulative_predictions(test_session, test_scores))
        metadata[test_id] = test_session["info"]
        print(f"[{index:02d}/{len(sessions)}] decoded {test_id}")

    predictions = pd.DataFrame(rows).sort_values(["session", "char_idx"])
    if len(sessions) != 26 or len(predictions) != 150:
        raise RuntimeError(
            f"Expected 26 QC sessions and 150 characters; got "
            f"{len(sessions)} sessions and {len(predictions)} characters."
        )
    csv_path = OUTPUT / "QC_character_predictions.csv"
    predictions.to_csv(csv_path, index=False)
    write_report(predictions, metadata)
    print(
        f"\nSaved {len(predictions)} character rows across "
        f"{len(sessions)} QC test sessions."
    )
    print(f"Markdown: {OUTPUT / 'QC_character_predictions.md'}")
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    main()
