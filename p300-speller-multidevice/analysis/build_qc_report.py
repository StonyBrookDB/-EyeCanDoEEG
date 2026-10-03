"""Consolidate QC analysis outputs into one report.md.

Run the QC-only analysis, statistical summary, and cumulative predictions first:

    python analysis/analyze_flex_batches.py
    python analysis/analyze_flex_qc_stats.py
    python analysis/export_flex_cumulative_predictions.py
    python analysis/build_qc_report.py

The builder appends the full run log, exact statistical output, and per-character
prediction tables to REPORT.md, then removes the three redundant text/Markdown
intermediate files. CSV and figure outputs remain as machine-readable/supporting
data.
"""

from __future__ import annotations

from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
REPORT = RESULTS / "REPORT.md"
INTERMEDIATES = (
    ("results_summary.txt", "Full QC analysis run summary"),
    ("qc_statistics.txt", "Detailed QC statistical tests"),
    ("QC_character_predictions.md", "Cumulative per-character predictions"),
)
MARKER = "<!-- CONSOLIDATED QC DETAILS -->"


def _read_required(filename: str) -> str:
    path = RESULTS / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding QC analysis command first."
        )
    return path.read_text(encoding="utf-8").strip()


def _prediction_detail(markdown: str) -> str:
    heading = "## Summary"
    start = markdown.find(heading)
    if start < 0:
        raise ValueError("Cumulative prediction file does not contain '## Summary'.")
    return markdown[start:].strip()


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError(f"Missing main report: {REPORT}")

    main_report = REPORT.read_text(encoding="utf-8").strip()
    if MARKER in main_report:
        main_report = main_report.split(MARKER, maxsplit=1)[0].rstrip()

    sections = []
    for filename, heading in INTERMEDIATES:
        body = _read_required(filename)
        if filename == "QC_character_predictions.md":
            sections.extend(["## " + heading, "", _prediction_detail(body)])
        else:
            sections.extend(
                [
                    "## " + heading,
                    "",
                    "```text",
                    body,
                    "```",
                ]
            )

    consolidated = (
        main_report
        + "\n\n"
        + MARKER
        + "\n\n"
        + "\n".join(sections).strip()
        + "\n"
    )
    REPORT.write_text(consolidated, encoding="utf-8")

    for filename, _ in INTERMEDIATES:
        (RESULTS / filename).unlink()

    print(f"Consolidated QC report: {REPORT}")
    print(f"Removed {len(INTERMEDIATES)} duplicate intermediate reports.")


if __name__ == "__main__":
    main()
