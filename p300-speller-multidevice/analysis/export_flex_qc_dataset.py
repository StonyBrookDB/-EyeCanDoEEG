"""Export the QC-passing Emotiv Flex sessions as training-ready EEG epochs.

Run from the project directory:
    python analysis/export_flex_qc_dataset.py

The output is stored in ``Dataset``. The source recordings
are never modified. See Dataset/README.md for format, preprocessing, QC, and
the important limitation that marker lag was estimated using each session's
own target labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_flex_batches as analysis


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT = PROJECT_ROOT / "Dataset"
QC_NOISE_UV = 10.0


def _session_metadata(prepared: dict, epoched: dict, kept: np.ndarray) -> dict:
    info = epoched["info"]
    return {
        **info,
        "session_id": info["session"],
        "subject_id": None,
        "subject_id_note": "Subject identifier is not present in source metadata.",
        "n_epochs_exported": int(kept.sum()),
        "n_epochs_rejected": int((~kept).sum()),
        "n_targets_exported": int(epoched["mrk"].loc[kept, "is_target"].sum()),
        "n_nontargets_exported": int(
            (epoched["mrk"].loc[kept, "is_target"] == 0).sum()
        ),
        "qc_rule": f"median channel robust SD <= {QC_NOISE_UV} uV",
        "qc_threshold_note": (
            "Threshold was selected after examining initial decoding results; "
            "this QC subset is post hoc."
        ),
        "bad_channels": [
            channel
            for channel, is_bad in zip(prepared["chans"], prepared["bad"], strict=True)
            if is_bad
        ],
        "channel_robust_sd_uv": {
            channel: float(scale)
            for channel, scale in zip(
                prepared["chans"], prepared["scale"], strict=True
            )
        },
        "epoch_processing": {
            "filter_hz": list(analysis.BAND),
            "filter": "4th-order Butterworth SOS, forward-backward, continuous EEG",
            "resampling": (
                f"block mean, {analysis.BIN} samples per bin "
                f"(nominal {analysis.FS / analysis.BIN:g} Hz)"
            ),
            "stored_epoch_sample_times_s": analysis.BIN_T.tolist(),
            "baseline_interval_s": [
                float(analysis.BIN_T[0]),
                float(analysis.BIN_T[analysis.BASE_BINS - 1]),
            ],
            "baseline_correction": "subtract each epoch/channel baseline mean",
            "artifact_rule": (
                f"reject epoch if >= {analysis.MIN_ARTIFACT_CH} channels exceed "
                f"{analysis.Z_REJECT} robust SD within epoch"
            ),
            "stored_signal_units": "microvolts",
            "bad_channel_handling": (
                "Bad channels are retained in epochs_uv for provenance; use "
                "bad_channels metadata to exclude them during training."
            ),
        },
        "marker_alignment": {
            "method": (
                "session target-minus-nontarget multichannel difference-wave "
                "template matching; low-cosine estimates fall back to the "
                "median lag of reliable sessions in the same batch"
            ),
            "lag_s": float(info["lag_s"]),
            "lag_used_for_epoching_s": float(info["lag_s"]),
            "warning": (
                "Lag estimation uses target labels from this same session. "
                "This is a supervised alignment correction, not label-free "
                "test-time preprocessing."
            ),
        },
    }


def export_session(
    prepared: dict,
    lag_samples: int,
    lag_confidence: dict,
    output: Path,
) -> dict:
    epoched = analysis.epoch_session(prepared, lag_samples)
    markers = epoched["mrk"]
    keep = markers["kept"].to_numpy()
    if not keep.any():
        raise ValueError(f"No artifact-clean epochs remain in {epoched['info']['session']}")

    # Ground-truth symbol is derived from the row/column target codes, which
    # correctly handles sessions whose free-text phrase metadata is inaccurate.
    target_row = markers.loc[keep, "t_row"].to_numpy(dtype=np.int8)
    target_col = markers.loc[keep, "t_col"].to_numpy(dtype=np.int8)
    target_symbol = np.asarray(
        [
            analysis.GRID[r][c]
            for r, c in zip(target_row, target_col, strict=True)
        ],
        dtype="U1",
    )
    marker_t = markers.loc[keep, "lsl_timestamp"].to_numpy(dtype=np.float64)
    lag_s = float(epoched["info"]["lag_s"])

    filename = (
        f"{prepared['info']['batch']}_"
        f"{prepared['info']['session'].split('/')[-1]}.npz"
    )
    output_path = output / filename
    metadata = _session_metadata(prepared, epoched, keep)
    metadata["source_meta"] = json.loads(
        (analysis.BATCHES[prepared["info"]["batch"]]
         / prepared["info"]["session"].split("/")[-1]
         / "meta.json").read_text(encoding="utf-8")
    )
    metadata["output_file"] = filename
    metadata["eeg_source"] = prepared["info"]["eeg_source"]
    metadata["lag_peak_cosine"] = float(lag_confidence["peak_cos"])
    metadata["lag_estimate_reliable"] = bool(lag_confidence["reliable"])
    metadata_path = output_path.with_suffix(".json")

    np.savez_compressed(
        output_path,
        # (n_epochs, n_channels, n_times); filtered, baseline-corrected uV.
        epochs_uv=epoched["ep"][keep].astype(np.float32, copy=False),
        # Canonical binary flash label: 1=target, 0=non-target.
        y=markers.loc[keep, "is_target"].to_numpy(dtype=np.int8),
        stim_code=markers.loc[keep, "code"].to_numpy(dtype=np.int8),
        char_idx=markers.loc[keep, "char_idx"].to_numpy(dtype=np.int16),
        repetition=markers.loc[keep, "rep"].to_numpy(dtype=np.int8),
        target_row=target_row,
        target_col=target_col,
        target_symbol=target_symbol,
        marker_time_s=marker_t,
        aligned_event_time_s=marker_t + lag_s,
        epoch_sample_times_s=analysis.BIN_T.astype(np.float32),
        channel_names=np.asarray(prepared["chans"], dtype="U"),
        bad_channels=np.asarray(metadata["bad_channels"], dtype="U"),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    existing = (
        [
            path
            for path in OUTPUT.iterdir()
            if path.name != "README.md"
            and not (path.is_dir() and not any(path.iterdir()))
        ]
        if OUTPUT.exists()
        else []
    )
    if existing:
        raise FileExistsError(
            f"{OUTPUT} already exists and is not empty. Move it aside or choose "
            "another output directory before exporting; existing files were not changed."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    prepared_sessions = analysis.prepare_all()
    lags, _, _ = analysis.estimate_all_lags(prepared_sessions)
    lag_by_session = lags.set_index("session")

    records = []
    for prepared in prepared_sessions:
        info = prepared["info"]
        if info["median_scale_uv"] > QC_NOISE_UV:
            print(
                f"skip {info['session']}: median channel noise "
                f"{info['median_scale_uv']:.2f} uV > {QC_NOISE_UV:g} uV"
            )
            continue

        session_id = info["session"]
        lag_samples = int(lag_by_session.loc[session_id, "lag_used"])
        record = export_session(
            prepared,
            lag_samples,
            lag_by_session.loc[session_id].to_dict(),
            OUTPUT,
        )
        records.append(record)
        print(
            f"exported {session_id}: {record['n_epochs_exported']} epochs "
            f"({record['n_targets_exported']} target), lag={record['lag_s']:+.3f}s"
        )

    if not records:
        raise RuntimeError("No sessions passed QC; no dataset was exported.")

    manifest = pd.DataFrame(
        [
            {
                key: record.get(key)
                for key in (
                    "session_id",
                    "subject_id",
                    "batch",
                    "eeg_source",
                    "phrase",
                    "true_text",
                    "n_chars",
                    "n_flashes",
                    "n_epochs_exported",
                    "n_epochs_rejected",
                    "n_targets_exported",
                    "n_nontargets_exported",
                    "median_scale_uv",
                    "n_bad_ch",
                    "bad_channels",
                    "lag_s",
                    "peak_cos",
                    "reliable",
                    "output_file",
                )
            }
            for record in records
        ]
    )
    manifest["bad_channels"] = manifest["bad_channels"].map(
        lambda value: ",".join(value) if isinstance(value, list) else value
    )
    manifest.to_csv(OUTPUT / "manifest.csv", index=False)

    dataset_metadata = {
        "dataset_name": "Emotiv Flex P300 QC epochs",
        "version": 1,
        "n_sessions": len(records),
        "n_characters": int(sum(record["n_chars"] for record in records)),
        "n_epochs": int(sum(record["n_epochs_exported"] for record in records)),
        "n_targets": int(sum(record["n_targets_exported"] for record in records)),
        "qc_rule": f"median channel robust SD <= {QC_NOISE_UV} uV",
        "qc_threshold_note": (
            "Post hoc threshold chosen after inspecting initial decoding; "
            "not preregistered."
        ),
        "source_batches": ["emotivFlex-batch1", "emotivFlex-batch2"],
        "subject_id": None,
        "subject_id_note": "Source metadata does not identify subjects; do not claim cross-subject validation.",
        "format": "one compressed NPZ and one JSON sidecar per session at Dataset root; see README.md",
        "generated_by": "export_flex_qc_dataset.py",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (OUTPUT / "dataset.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    print(
        f"\nWrote {len(records)} sessions, {dataset_metadata['n_epochs']} clean epochs, "
        f"{dataset_metadata['n_characters']} characters to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
