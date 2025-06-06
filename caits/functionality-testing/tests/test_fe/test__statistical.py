import numpy as np
import pytest
import caits.fe._statistical as stat

def test_std_value_shape():
    x = np.random.randn(100, 3)
    out = stat.std_value(x)
    assert out.shape == (3,)

def test_variance_value_shape():
    x = np.random.randn(100, 3)
    out = stat.variance_value(x)
    assert out.shape == (3,)

def test_mean_value_shape():
    x = np.random.randn(100, 3)
    out = stat.mean_value(x)
    assert out.shape == (3,)

def test_median_value_shape():
    x = np.random.randn(100, 3)
    out = stat.median_value(x)
    assert out.shape == (3,)

def test_max_value_shape():
    x = np.random.randn(100, 3)
    out = stat.max_value(x)
    assert out.shape == (3,)

def test_min_value_shape():
    x = np.random.randn(100, 3)
    out = stat.min_value(x)
    assert out.shape == (3,)

def test_kurtosis_value_shape():
    x = np.random.randn(100, 3)
    out = stat.kurtosis_value(x)
    assert out.shape == (3,)

def test_sample_skewness_shape():
    x = np.random.randn(100, 3)
    out = stat.sample_skewness(x)
    assert out.shape == (3,)

def test_rms_value_shape():
    x = np.random.randn(100, 3)
    out = stat.rms_value(x)
    assert out.shape == (3,)

def test_rms_max_shape():
    x = np.random.randn(100, 3)
    out = stat.rms_max(x, frame_length=5, hop_length=2, axis=0)
    assert out.shape == (3,)

def test_rms_mean_shape():
    x = np.random.randn(100, 3)
    out = stat.rms_mean(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_rms_min_shape():
    x = np.random.randn(100, 3)
    out = stat.rms_min(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_zcr_value_shape():
    x = np.random.randn(100, 3)
    out = stat.zcr_value(x)
    assert out.shape == (3,)

def test_zcr_max_shape():
    x = np.random.randn(100, 3)
    out = stat.zcr_max(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_zcr_mean_shape():
    x = np.random.randn(100, 3)
    out = stat.zcr_mean(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_zcr_min_shape():
    x = np.random.randn(100, 3)
    out = stat.zcr_min(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_dominant_frequency_shape():
    x = np.sin(2 * np.pi * 1 * np.arange(100) / 100)
    x2d = np.stack([x, x], axis=1)
    out = stat.dominant_frequency(x2d, fs=100)
    assert out.shape == (2,)

def test_central_moments_shape():
    x = np.random.randn(100, 3)
    out = stat.central_moments(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 3)
    out_dict = stat.central_moments(x, export="dict")
    assert isinstance(out_dict, dict)
    assert set(out_dict.keys()) == {"moment0", "moment1", "moment2", "moment3", "moment4"}
    for v in out_dict.values():
        assert v.shape == (3,)

def test_signal_length_shape():
    x = np.random.randn(100, 3)
    out = stat.signal_length(x, fs=10)
    assert out.shape == (3,)
    out_samples = stat.signal_length(x, fs=10, time_mode="samples")
    assert out_samples.shape == (3,)

def test_energy_shape():
    x = np.random.randn(100, 3)
    out = stat.energy(x)
    assert out.shape == (3,)

def test_average_power_shape():
    x = np.random.randn(100, 3)
    out = stat.average_power(x)
    assert out.shape == (3,)

def test_crest_factor_shape():
    x = np.abs(np.random.randn(100, 3)) + 1e-3
    out = stat.crest_factor(x)
    assert out.shape == (3,)

def test_envelope_energy_peak_detection_shape():
    x = np.random.randn(1000, 3)
    out = stat.envelope_energy_peak_detection(x, fs=1000)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 3
    out_dict = stat.envelope_energy_peak_detection(x, fs=1000, export="dict")
    assert isinstance(out_dict, dict)
    for v in out_dict.values():
        assert v.shape == (3,)

def test_mfcc_mean_shape():
    x = np.random.randn(22050, 3)
    out = stat.mfcc_mean(x, sr=22050, n_mfcc=13)
    assert isinstance(out, np.ndarray)
    assert out.shape == (13, 3)

def test_signal_stats_shape():
    x = np.random.randn(100, 3)
    out = stat.signal_stats(x, name="test")
    assert isinstance(out, dict)
    expected_keys = {"test_max", "test_min", "test_mean", "test_median", "test_std", "test_var", "test_kurtosis", "test_skewness", "test_rms", "test_crest_factor", "test_signal_length"}
    assert set(out.keys()) == expected_keys
    for v in out.values():
        assert v.shape == (3,)


def test_std_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.std_value(x, axis=0)
    out1 = stat.std_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_variance_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.variance_value(x, axis=0)
    out1 = stat.variance_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_mean_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.mean_value(x, axis=0)
    out1 = stat.mean_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_median_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.median_value(x, axis=0)
    out1 = stat.median_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_max_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.max_value(x, axis=0)
    out1 = stat.max_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_min_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = stat.min_value(x, axis=0)
    out1 = stat.min_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_kurtosis_value_shape_multi():
    x = np.random.randn(100, 3)
    out = stat.kurtosis_value(x)
    assert out.shape == (3,)

def test_sample_skewness_shape_multi():
    x = np.random.randn(100, 3)
    out = stat.sample_skewness(x)
    assert out.shape == (3,)

def test_rms_value_shape_multi():
    x = np.random.randn(100, 3)
    out = stat.rms_value(x, axis=0)
    assert out.shape == (3,)

def test_signal_stats_shape_multi():
    x = np.random.randn(100, 3)
    out = stat.signal_stats(x, name="test", axis=0)
    assert isinstance(out, dict)
    expected_keys = {"test_max", "test_min", "test_mean", "test_median", "test_std", "test_var", "test_kurtosis", "test_skewness", "test_rms", "test_crest_factor", "test_signal_length"}
    assert set(out.keys()) == expected_keys
