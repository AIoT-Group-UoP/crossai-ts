import numpy as np
import pytest
from caits.fe import (
    std_value,
    variance_value,
    mean_value,
    median_value,
    max_value,
    min_value,
    kurtosis_value,
    sample_skewness,
    rms_value,
    rms_max,
    rms_min,
    rms_mean,
    zcr_value,
    zcr_max,
    zcr_min,
    zcr_mean,
    dominant_frequency,
    central_moments,
    signal_length,
    signal_stats,
    energy,
    average_power,
    crest_factor,
    envelope_energy_peak_detection,mfcc_mean
)

def test_std_value_shape():
    x = np.random.randn(100, 3)
    out = std_value(x, axis=0)
    assert out.shape == (3,)

def test_variance_value_shape():
    x = np.random.randn(100, 3)
    out = variance_value(x, axis=0)
    assert out.shape == (3,)

def test_mean_value_shape():
    x = np.random.randn(100, 3)
    out = mean_value(x, axis=0)
    assert out.shape == (3,)

def test_median_value_shape():
    x = np.random.randn(100, 3)
    out = median_value(x, axis=0)
    assert out.shape == (3,)

def test_max_value_shape():
    x = np.random.randn(100, 3)
    out = max_value(x, axis=0)
    assert out.shape == (3,)

def test_min_value_shape():
    x = np.random.randn(100, 3)
    out = min_value(x, axis=0)
    assert out.shape == (3,)

def test_kurtosis_value_shape():
    x = np.random.randn(100, 3)
    out = kurtosis_value(x, axis=0)
    assert out.shape == (3,)

def test_sample_skewness_shape():
    x = np.random.randn(100, 3)
    out = sample_skewness(x, axis=0)
    assert out.shape == (3,)

def test_rms_value_shape():
    x = np.random.randn(100, 3)
    out = rms_value(x, axis=0)
    assert out.shape == (3,)

def test_rms_max_shape():
    x = np.random.randn(100, 3)
    out = rms_max(x, frame_length=5, hop_length=2, axis=0)
    assert out.shape == (3,)

def test_rms_mean_shape():
    x = np.random.randn(100, 3)
    out = rms_mean(x, frame_length=5, hop_length=2, axis=0)
    assert out.shape == (3,)

def test_rms_min_shape():
    x = np.random.randn(100, 3)
    out = rms_min(x, frame_length=5, hop_length=2, axis=0)
    assert out.shape == (3,)

def test_zcr_value_shape():
    x = np.random.randn(100, 3)
    out = zcr_value(x, axis=0)
    assert out.shape == (3,)

def test_zcr_max_shape():
    x = np.random.randn(100, 3)
    out = zcr_max(x, frame_length=5, hop_length=2, axis=0)
    assert out.shape == (3,)

def test_zcr_mean_shape():
    x = np.random.randn(100, 3)
    out = zcr_mean(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_zcr_min_shape():
    x = np.random.randn(100, 3)
    out = zcr_min(x, frame_length=5, hop_length=2)
    assert out.shape == (3,)

def test_dominant_frequency_shape():
    x = np.sin(2 * np.pi * 1 * np.arange(100) / 100)
    x2d = np.stack([x, x], axis=1)
    out = dominant_frequency(x2d, fs=100, axis=0)
    assert out.shape == (2,)
    x2d = x2d.T
    out = dominant_frequency(x2d, fs=100, axis=1)
    assert out.shape == (2,)

def test_central_moments_shape():
    x = np.random.randn(100, 3)
    out = central_moments(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 3)
    out_dict = central_moments(x, export="dict")
    assert isinstance(out_dict, dict)
    assert set(out_dict.keys()) == {"moment0", "moment1", "moment2", "moment3", "moment4"}
    for v in out_dict.values():
        assert v.shape == (3,)

def test_signal_length_shape():
    x = np.random.randn(100, 3)
    out = signal_length(x, fs=9, axis=0)
    assert out == 100 / 9
    out_samples = signal_length(x, fs=10, time_mode="samples", axis=0)
    assert out_samples == 100

def test_energy_shape():
    x = np.random.randn(100, 3)
    out = energy(x, axis=0)
    assert out.shape == (3,)

def test_average_power_shape():
    x = np.random.randn(100, 3)
    out = average_power(x, axis=0)
    assert out.shape == (3,)

def test_crest_factor_shape():
    x = np.abs(np.random.randn(100, 3)) + 1e-3
    out = crest_factor(x, axis=0)
    assert out.shape == (3,)

def test_envelope_energy_peak_detection_shape():
    x = np.random.randn(1000, 3)
    out = envelope_energy_peak_detection(x, fs=1000, axis=0)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 3
    out_dict = envelope_energy_peak_detection(x, fs=1000, export="dict", axis=0)
    assert isinstance(out_dict, dict)
    for v in out_dict.values():
        assert v.shape == (3,)

def test_mfcc_mean_shape():
    x = np.random.randn(22050, 3)
    out = mfcc_mean(x, sr=22050, n_mfcc=13, axis=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (13, 3)

def test_signal__shape():
    x = np.random.randn(100, 3)
    out = signal_stats(x, name="test", axis=0)
    assert isinstance(out, dict)
    expected_keys = {
        "test_max",
        "test_min",
        "test_mean",
        "test_median",
        "test_std",
        "test_var",
        "test_kurtosis",
        "test_skewness",
        "test_rms",
        "test_crest_factor",
        "test_signal_length",
        "test_dominant_frequency",
    }

    assert set(out.keys()) == expected_keys
    for k, v in out.items():
        if k != "test_signal_length":
            assert v.shape == (3,)


def test_std_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = std_value(x, axis=0)
    out1 = std_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_variance_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = variance_value(x, axis=0)
    out1 = variance_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_mean_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = mean_value(x, axis=0)
    out1 = mean_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_median_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = median_value(x, axis=0)
    out1 = median_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_max_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = max_value(x, axis=0)
    out1 = max_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_min_value_shape_multi():
    x = np.random.randn(100, 3)
    out0 = min_value(x, axis=0)
    out1 = min_value(x, axis=1)
    assert out0.shape == (3,)
    assert out1.shape == (100,)

def test_kurtosis_value_shape_multi():
    x = np.random.randn(100, 3)
    out = kurtosis_value(x, axis=0)
    assert out.shape == (3,)

def test_sample_skewness_shape_multi():
    x = np.random.randn(100, 3)
    out = sample_skewness(x, axis=0)
    assert out.shape == (3,)

def test_rms_value_shape_multi():
    x = np.random.randn(100, 3)
    out = rms_value(x, axis=0)
    assert out.shape == (3,)

