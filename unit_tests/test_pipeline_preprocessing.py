"""Tests for eyecando.pipeline.preprocessing.preprocess_p300, against
synthetic raws built with embedded stim_id/target_flag channels.

Covers epoch count/shape derived from events, that the bandpass filter
actually attenuates a 50 Hz tone and modifies `raw` in place,
reject-threshold artifact rejection, stim_id alignment/validity, and
the use_ea flag: with EA on, per-epoch covariance is whitened toward
identity; with it off, the output is different (not just
coincidentally so) and covariance is left unwhitened.
"""

import mne
import numpy as np

from eyecando.pipeline.preprocessing import preprocess_p300

mne.set_log_level("ERROR")

SFREQ = 256.0
CH_NAMES = ["Cz", "Pz", "Fz"]


def _make_raw(
    n_events: int = 10,
    seed: int = 0,
    artifact_event: int | None = None,
    artifact_amplitude: float = 500e-6,
    freq_50hz: bool = False,
    spacing_s: float = 4.0,
) -> mne.io.RawArray:
    """Synthetic raw with EEG + stim_id + target_flag channels.

    Events are spaced `spacing_s` apart (default 4s) so the signal is long
    enough for a 0.1 Hz high-pass filter without MNE's filter-length warning.
    target_flag carries 2 (target) for every 5th event and 1 (nontarget)
    otherwise, matching the _synthetic_raw convention in test_data.py.
    """
    rng = np.random.default_rng(seed)
    n_samples = int((n_events * spacing_s + 4) * SFREQ)
    ch_names = [*CH_NAMES, "stim_id", "target_flag"]
    ch_types = ["eeg"] * len(CH_NAMES) + ["stim", "stim"]
    info = mne.create_info(ch_names, SFREQ, ch_types=ch_types)

    eeg_data = rng.normal(scale=5e-6, size=(len(CH_NAMES), n_samples))
    if freq_50hz:
        t = np.arange(n_samples) / SFREQ
        eeg_data += 20e-6 * np.sin(2 * np.pi * 50 * t)

    stim_id_ch = np.zeros((1, n_samples))
    target_flag_ch = np.zeros((1, n_samples))
    onset_samples = (np.arange(1, n_events + 1) * spacing_s * SFREQ).astype(int)
    for i, idx in enumerate(onset_samples):
        stim_id_ch[0, idx] = (i % 12) + 1  # 1–12
        target_flag_ch[0, idx] = 2 if i % 5 == 0 else 1

    data = np.vstack([eeg_data, stim_id_ch, target_flag_ch])
    raw = mne.io.RawArray(data, info)

    if artifact_event is not None:
        idx = onset_samples[artifact_event]
        raw._data[0, idx : idx + 26] += artifact_amplitude

    return raw


def test_epoch_count_matches_events_without_reject() -> None:
    raw = _make_raw(n_events=10)
    X, y, stim_ids, _ = preprocess_p300(raw, reject=None)
    assert len(y) == 10


def test_epoch_shape() -> None:
    raw = _make_raw(n_events=10)
    tmin, tmax = -0.1, 0.8
    X, y, stim_ids, _ = preprocess_p300(raw, tmin=tmin, tmax=tmax)
    expected_times = int((tmax - tmin) * SFREQ) + 1
    assert X.shape[0] == 10
    assert X.shape[1] == len(CH_NAMES)
    assert abs(X.shape[2] - expected_times) <= 1


def test_filter_applied_attenuates_50hz() -> None:
    raw = _make_raw(n_events=10, freq_50hz=True)
    raw_orig_power = np.abs(np.fft.rfft(raw.get_data(picks="eeg")[0])).max()
    preprocess_p300(raw, h_freq=20.0)
    filtered_power = np.abs(np.fft.rfft(raw.get_data(picks="eeg")[0])).max()
    assert filtered_power < 0.1 * raw_orig_power


def test_reject_removes_artifact_epochs() -> None:
    raw = _make_raw(n_events=10, artifact_event=3, artifact_amplitude=500e-6)
    _, y_no_reject, _, _ = preprocess_p300(raw.copy(), reject=None)
    _, y_rejected, _, _ = preprocess_p300(raw.copy(), reject={"eeg": 100e-6})
    assert len(y_rejected) < len(y_no_reject)


def test_picks_eeg_excludes_stim_channels() -> None:
    raw = _make_raw(n_events=10)
    X, _, _, _ = preprocess_p300(raw)
    assert X.shape[1] == len(CH_NAMES)


def test_stim_ids_are_valid_and_aligned() -> None:
    raw = _make_raw(n_events=10)
    X, y, stim_ids, _ = preprocess_p300(raw, reject=None)
    assert stim_ids.shape == y.shape
    assert np.all((stim_ids >= 1) & (stim_ids <= 12))


def test_filter_modifies_raw_data_in_place() -> None:
    raw = _make_raw(n_events=10)
    original = raw.get_data(picks="eeg").copy()
    preprocess_p300(raw, l_freq=0.1, h_freq=20.0)
    assert not np.allclose(raw.get_data(picks="eeg"), original)


def test_use_ea_false_differs_from_use_ea_true() -> None:
    raw_ea = _make_raw(n_events=10)
    raw_no_ea = raw_ea.copy()
    X_ea, _, _, _ = preprocess_p300(raw_ea, reject=None, use_ea=True)
    X_no_ea, _, _, _ = preprocess_p300(raw_no_ea, reject=None, use_ea=False)
    assert not np.allclose(X_ea, X_no_ea)


def test_use_ea_false_is_not_a_silent_no_op() -> None:
    """use_ea=False must skip EA, not just happen to look
    different for some unrelated reason -- confirmed here by checking
    the per-epoch covariance isn't whitened toward identity, which is
    exactly what EA would have done."""
    raw = _make_raw(n_events=10)
    X, _, _, _ = preprocess_p300(raw, reject=None, use_ea=False)
    covs = np.einsum("ijk,ilk->ijl", X, X) / X.shape[-1]
    mean_cov = covs.mean(axis=0)
    assert not np.allclose(mean_cov, np.eye(X.shape[1]), atol=0.2)


def test_use_ea_true_still_defaults_and_whitens() -> None:
    raw = _make_raw(n_events=40, spacing_s=1.0)
    X, _, _, _ = preprocess_p300(raw, reject=None)
    covs = np.einsum("ijk,ilk->ijl", X, X) / X.shape[-1]
    mean_cov = covs.mean(axis=0)
    assert np.allclose(mean_cov, np.eye(X.shape[1]), atol=0.5)
