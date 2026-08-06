"""Tests for eyecando.ingestion.data: MOABB, XDF, and CSV session loaders.

Covers _raw_from_session's run concatenation and stim-channel
remapping (native "Flash stim"/"Target stim" codes -> stim_id/
target_flag); the load_moabb_* family against a mocked moabb.datasets
(subject/session/all-subjects loading, subject-grouped vs. flattened
results, dataset_name selection, and load_moabb_subject_raw's
no-remap-assumed behavior for datasets that don't share BNCI2014_009's
convention); load_session_csv's unit conversion (microvolts -> volts),
timestamp-to-sample event alignment, and target/nontarget/flash_id
encoding; and load_session_xdf's error handling when a named stream
isn't found.
"""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from eyecando.ingestion.data import (
    _raw_from_session,
    load_moabb_all_subjects,
    load_moabb_all_subjects_grouped,
    load_moabb_session,
    load_moabb_subject,
    load_moabb_subject_raw,
    load_session_csv,
    load_session_xdf,
)

mne.set_log_level("ERROR")


def _synthetic_raw(n_events: int = 6, sfreq: float = 256.0) -> mne.io.RawArray:
    """Synthetic raw with MOABB-style 'Flash stim' and 'Target stim' channels."""
    eeg_ch_names = ["Fz", "Cz", "Pz"]
    n_samples = int(sfreq * (n_events + 1))
    rng = np.random.default_rng(0)
    eeg_data = rng.normal(scale=1e-6, size=(3, n_samples))

    flash_stim = np.zeros((1, n_samples))
    target_stim = np.zeros((1, n_samples))
    onset_samples = (np.arange(1, n_events + 1) * sfreq).astype(int)
    for i, idx in enumerate(onset_samples):
        flash_stim[0, idx] = (i % 12) + 3  # native codes 3–14
        target_stim[0, idx] = 2 if i % 5 == 0 else 1  # 1=nontarget, 2=target

    data = np.vstack([eeg_data, flash_stim, target_stim])
    info = mne.create_info(
        eeg_ch_names + ["Flash stim", "Target stim"],
        sfreq,
        ch_types=["eeg"] * 3 + ["stim", "stim"],
    )
    return mne.io.RawArray(data, info)


def _fake_moabb_data(subjects: list[int]) -> dict[int, dict[str, dict[str, mne.io.RawArray]]]:
    return {subject: {"0": {"0": _synthetic_raw()}} for subject in subjects}


def test_raw_from_session_concatenates_runs() -> None:
    runs = {"0": _synthetic_raw(3), "1": _synthetic_raw(3)}
    raw = _raw_from_session(runs)
    assert raw.n_times == _synthetic_raw(3).n_times * 2
    assert "stim_id" in raw.ch_names
    assert "target_flag" in raw.ch_names


def test_raw_from_session_has_canonical_stim_channels() -> None:
    raw = _raw_from_session({"0": _synthetic_raw()})
    assert "stim_id" in raw.ch_names
    assert "target_flag" in raw.ch_names
    assert isinstance(raw, mne.io.BaseRaw)


def test_raw_from_session_moabb_remapping() -> None:
    """Flash stim codes 3–14 must map to stim_id 1–12; target_flag unchanged."""
    raw = _raw_from_session({"0": _synthetic_raw(n_events=6)})
    stim_id_data = raw.get_data(picks=["stim_id"])[0]
    target_flag_data = raw.get_data(picks=["target_flag"])[0]

    nonzero_stim = stim_id_data[stim_id_data != 0]
    nonzero_target = target_flag_data[target_flag_data != 0]

    assert np.all(nonzero_stim >= 1) and np.all(nonzero_stim <= 12)
    assert set(nonzero_target.astype(int)).issubset({1, 2})


def test_raw_from_session_returns_only_eeg_plus_stim() -> None:
    """Returned raw must not contain 'Flash stim' or 'Target stim'."""
    raw = _raw_from_session({"0": _synthetic_raw()})
    assert "Flash stim" not in raw.ch_names
    assert "Target stim" not in raw.ch_names


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_subject_returns_list_of_raws(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = _fake_moabb_data([1])
    sessions = load_moabb_subject(1)
    mock_dataset.get_data.assert_called_once_with(subjects=[1])
    assert isinstance(sessions, list)
    assert len(sessions) == 1
    assert isinstance(sessions[0], mne.io.BaseRaw)
    assert "stim_id" in sessions[0].ch_names
    assert "target_flag" in sessions[0].ch_names


def _differently_structured_raw() -> mne.io.RawArray:
    """A synthetic raw with NEITHER 'Flash stim' nor 'Target stim' -- a
    stand-in for a MOABB dataset whose channel names/event convention
    doesn't match BNCI2014_009's, proving load_moabb_subject_raw() doesn't
    assume that structure the way _raw_from_session()/bnci2014_009.remap_raw()
    do."""
    sfreq = 128.0
    n_samples = int(sfreq * 4)
    rng = np.random.default_rng(0)
    eeg_data = rng.normal(scale=1e-6, size=(2, n_samples))
    marker_ch = np.zeros((1, n_samples))
    marker_ch[0, [10, 50, 90]] = [1, 2, 1]
    data = np.vstack([eeg_data, marker_ch])
    info = mne.create_info(["O1", "O2", "STI"], sfreq, ch_types=["eeg", "eeg", "stim"])
    return mne.io.RawArray(data, info)


@patch("moabb.datasets.BNCI2014_008")
def test_load_moabb_subject_raw_does_not_assume_bnci2014_009_structure(
    mock_dataset_cls: MagicMock,
) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = {1: {"0": {"0": _differently_structured_raw()}}}
    sessions = load_moabb_subject_raw(1, dataset_name="BNCI2014_008")
    mock_dataset.get_data.assert_called_once_with(subjects=[1])
    assert isinstance(sessions, list)
    assert len(sessions) == 1
    # native channel names survive untouched -- no remapping attempted
    assert sessions[0].ch_names == ["O1", "O2", "STI"]
    assert "stim_id" not in sessions[0].ch_names
    assert "target_flag" not in sessions[0].ch_names


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_subject_raw_concatenates_runs(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    n_times_single = _differently_structured_raw().n_times
    mock_dataset.get_data.return_value = {
        1: {"0": {"0": _differently_structured_raw(), "1": _differently_structured_raw()}}
    }
    sessions = load_moabb_subject_raw(1, dataset_name="BNCI2014_009")
    assert sessions[0].n_times == n_times_single * 2


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_session_returns_single_raw(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = _fake_moabb_data([1])
    raw = load_moabb_session(1, session=0)
    assert isinstance(raw, mne.io.BaseRaw)
    assert "stim_id" in raw.ch_names
    assert "target_flag" in raw.ch_names


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_session_accepts_string_key(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = _fake_moabb_data([1])
    raw = load_moabb_session(1, session="0")
    assert isinstance(raw, mne.io.BaseRaw)


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_session_missing_key_raises(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = _fake_moabb_data([1])
    with pytest.raises(KeyError, match="Session"):
        load_moabb_session(1, session=99)


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_all_subjects_returns_flat_list(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.subject_list = [1, 2, 3]
    # one session per subject → 3 entries total
    mock_dataset.get_data.return_value = _fake_moabb_data([1, 2, 3])
    results = load_moabb_all_subjects()
    mock_dataset.get_data.assert_called_once_with(subjects=[1, 2, 3])
    assert len(results) == 3
    for raw in results:
        assert isinstance(raw, mne.io.BaseRaw)
        assert "stim_id" in raw.ch_names
        assert "target_flag" in raw.ch_names


def _fake_moabb_data_multi_session(
    subjects: list[int], n_sessions: int = 3
) -> dict[int, dict[str, dict[str, mne.io.RawArray]]]:
    return {
        subject: {str(ses): {"0": _synthetic_raw()} for ses in range(n_sessions)}
        for subject in subjects
    }


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_all_subjects_grouped_preserves_subject_boundaries(
    mock_dataset_cls: MagicMock,
) -> None:
    """3 subjects x 3 sessions must come back as 3 groups of 3, not a flat
    list of 9 -- flattening here is exactly what caused subject-session
    leakage in subject-grouped cross-validation."""
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.subject_list = [1, 2, 3]
    mock_dataset.get_data.return_value = _fake_moabb_data_multi_session([1, 2, 3], n_sessions=3)
    results = load_moabb_all_subjects_grouped()
    mock_dataset.get_data.assert_called_once_with(subjects=[1, 2, 3])
    assert len(results) == 3
    for sessions in results:
        assert isinstance(sessions, list)
        assert len(sessions) == 3
        for raw in sessions:
            assert isinstance(raw, mne.io.BaseRaw)
            assert "stim_id" in raw.ch_names
            assert "target_flag" in raw.ch_names


@patch("moabb.datasets.BNCI2014_009")
def test_load_moabb_subject_uses_dataset_name_param(mock_dataset_cls: MagicMock) -> None:
    mock_dataset = mock_dataset_cls.return_value
    mock_dataset.get_data.return_value = _fake_moabb_data([1])
    load_moabb_subject(1, dataset_name="BNCI2014_009")
    mock_dataset_cls.assert_called_once()


def _write_csv(path, header: list[str], rows: list[list[float]]) -> None:
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def test_load_session_csv_units_are_volts(tmp_path) -> None:
    eeg_path = tmp_path / "eeg.csv"
    marker_path = tmp_path / "markers.csv"
    sfreq = 250.0
    n_samples = 50
    timestamps = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(0)
    ch0 = rng.uniform(10, 100, size=n_samples)  # microvolts
    ch1 = rng.uniform(10, 100, size=n_samples)

    _write_csv(
        eeg_path,
        ["timestamp", "ch0", "ch1"],
        [[timestamps[i], ch0[i], ch1[i]] for i in range(n_samples)],
    )
    _write_csv(
        marker_path,
        ["timestamp", "flash_id", "is_target", "round_number"],
        [[timestamps[5], 1, 1, 1]],
    )

    raw = load_session_csv(eeg_path, marker_path, sfreq=sfreq)
    data = raw.get_data(picks="eeg")
    assert (1e-5 <= np.abs(data)).all()
    assert (np.abs(data) <= 1e-4).all()


def test_load_session_csv_event_alignment(tmp_path) -> None:
    eeg_path = tmp_path / "eeg.csv"
    marker_path = tmp_path / "markers.csv"
    sfreq = 250.0
    n_samples = 100
    timestamps = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(0)
    ch0 = rng.uniform(10, 50, size=n_samples)

    _write_csv(
        eeg_path,
        ["timestamp", "ch0"],
        [[timestamps[i], ch0[i]] for i in range(n_samples)],
    )

    known_sample_indices = [10, 47, 90]
    marker_rows = [
        [timestamps[idx] + 0.0005, flash_id, is_target, 1]
        for flash_id, (idx, is_target) in enumerate(
            zip(known_sample_indices, [1, 0, 1], strict=True), start=1
        )
    ]
    _write_csv(marker_path, ["timestamp", "flash_id", "is_target", "round_number"], marker_rows)

    raw = load_session_csv(eeg_path, marker_path, sfreq=sfreq)
    target_flag_data = raw.get_data(picks=["target_flag"])[0]
    nonzero_samples = np.where(target_flag_data != 0)[0].tolist()
    assert nonzero_samples == known_sample_indices


def test_load_session_csv_target_nontarget_encoding(tmp_path) -> None:
    eeg_path = tmp_path / "eeg.csv"
    marker_path = tmp_path / "markers.csv"
    sfreq = 250.0
    n_samples = 50
    timestamps = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(0)
    ch0 = rng.uniform(10, 50, size=n_samples)

    _write_csv(
        eeg_path,
        ["timestamp", "ch0"],
        [[timestamps[i], ch0[i]] for i in range(n_samples)],
    )
    marker_rows = [
        [timestamps[5] + 0.0005, 1, 1, 1],  # is_target=1 → target_flag=2
        [timestamps[15] + 0.0005, 2, 0, 1],  # is_target=0 → target_flag=1
    ]
    _write_csv(marker_path, ["timestamp", "flash_id", "is_target", "round_number"], marker_rows)

    raw = load_session_csv(eeg_path, marker_path, sfreq=sfreq)
    target_flag_data = raw.get_data(picks=["target_flag"])[0]
    assert target_flag_data[5] == 2.0  # target
    assert target_flag_data[15] == 1.0  # nontarget


def test_load_session_csv_flash_id_in_stim_id(tmp_path) -> None:
    eeg_path = tmp_path / "eeg.csv"
    marker_path = tmp_path / "markers.csv"
    sfreq = 250.0
    n_samples = 50
    timestamps = np.arange(n_samples) / sfreq
    rng = np.random.default_rng(0)
    ch0 = rng.uniform(10, 50, size=n_samples)

    _write_csv(eeg_path, ["timestamp", "ch0"], [[timestamps[i], ch0[i]] for i in range(n_samples)])
    _write_csv(
        marker_path,
        ["timestamp", "flash_id", "is_target", "round_number"],
        [[timestamps[10] + 0.0005, 7, 1, 1]],
    )

    raw = load_session_csv(eeg_path, marker_path, sfreq=sfreq)
    stim_id_data = raw.get_data(picks=["stim_id"])[0]
    assert stim_id_data[10] == 7.0


def test_load_session_xdf_stream_not_found_raises() -> None:
    fake_streams = [
        {
            "info": {"name": ["SomeOtherStream"], "nominal_srate": ["250"]},
            "time_series": [[1.0]],
            "time_stamps": [0.0],
        }
    ]
    with patch("pyxdf.load_xdf", return_value=(fake_streams, {})):
        with pytest.raises(ValueError, match="EEG"):
            load_session_xdf("fake.xdf", "EEG", "Markers")
