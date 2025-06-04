import numpy as np
import pytest

from caits.fe import _statistical as stat


class TestStatistical:

    def test_signal_stats_typical_audio(self):
        # Typical 1D audio signal: simple ramp
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        name = "audio"
        fs = 10
        result = stat.signal_stats(arr, name, axis=0, fs=fs, time_mode="time")
        # Mean, median, max, min, std, var, kurtosis, skewness, rms, crest, length
        assert np.isclose(result["audio_mean"], 3.0)
        assert np.isclose(result["audio_median"], 3.0)
        assert np.isclose(result["audio_max"], 5.0)
        assert np.isclose(result["audio_min"], 1.0)
        assert np.isclose(result["audio_std"], np.std(arr))
        assert np.isclose(result["audio_var"], np.var(arr))
        assert np.isclose(result["audio_kurtosis"], 1.7)  # scipy.stats.kurtosis([1,2,3,4,5])
        assert np.isclose(result["audio_skewness"], 0.0)
        assert np.isclose(result["audio_rms"], np.sqrt(np.mean(arr ** 2)))
        assert np.isclose(result["audio_crest_factor"], np.amax(np.abs(arr)) / np.sqrt(np.mean(arr ** 2)))
        assert np.isclose(result["audio_signal_length"], len(arr) / fs)

    def test_rolling_rms_and_zcr_statistics(self, mocker):
        # Prepare a well-formed signal
        signal = np.array([0, 1, -1, 1, -1, 0, 0, 1, -1, 1], dtype=float)
        frame_length = 4
        hop_length = 2

        # Mock rolling_rms and rolling_zcr to return known arrays
        mock_rms = np.array([0.5, 1.0, 1.5])
        mock_zcr = np.array([0.25, 0.5, 0.75])
        mocker.patch("caits.fe._statistical.rolling_rms", return_value=mock_rms)
        mocker.patch("caits.fe._statistical.rolling_zcr", return_value=mock_zcr)

        assert stat.rms_max(signal, frame_length, hop_length) == np.max(mock_rms)
        assert stat.rms_mean(signal, frame_length, hop_length) == np.mean(mock_rms)
        assert stat.rms_min(signal, frame_length, hop_length) == np.min(mock_rms)

        assert stat.zcr_max(signal, frame_length, hop_length) == np.max(mock_zcr)
        assert stat.zcr_mean(signal, frame_length, hop_length) == np.mean(mock_zcr)
        assert stat.zcr_min(signal, frame_length, hop_length) == np.min(mock_zcr)

    def test_envelope_energy_peak_detection_multiband(self):
        # Multi-frequency signal: sum of two sinusoids
        fs = 2000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 400 * t)
        # Use default band settings: 50-1000 Hz, step 50 Hz, fcl_add 50 Hz
        peaks_array = stat.envelope_energy_peak_detection(sig, fs, export="array")
        peaks_dict = stat.envelope_energy_peak_detection(sig, fs, export="dict")
        # There should be at least one band with detected peaks
        assert isinstance(peaks_array, np.ndarray)
        assert peaks_array.shape[0] == len(range(50, 1000, 50))
        assert any(peaks_array > 0)
        assert isinstance(peaks_dict, dict)
        assert set(peaks_dict.keys()) == {f"EEPD{fcl}_{fcl+50}" for fcl in range(50, 1000, 50)}
        assert any(v > 0 for v in peaks_dict.values())

    def test_sample_skewness_short_array_raises(self):
        arr = np.array([1, 2])
        with pytest.raises(ValueError, match="at least 3 elements"):
            stat.sample_skewness(arr)

    def test_central_moments_and_eepd_invalid_inputs(self):
        # Central moments: empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            stat.central_moments(np.array([]))
        # Central moments: unsupported export
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported export=bad"):
            stat.central_moments(arr, export="bad")
        # Envelope energy peak detection: unsupported export
        arr = np.random.randn(100)
        with pytest.raises(ValueError, match="Unsupported export=bad"):
            stat.envelope_energy_peak_detection(arr, fs=1000, export="bad")

    def test_signal_length_invalid_time_mode_raises(self):
        arr = np.arange(10)
        with pytest.raises(ValueError, match="Unsupported export=bad"):
            stat.signal_length(arr, fs=100, time_mode="bad")