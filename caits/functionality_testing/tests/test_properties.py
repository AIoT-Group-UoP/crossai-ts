import numpy as np
import pytest
import caits.properties as prop


def test_amplitude_envelope_hbt_shape():
    x = np.random.randn(1000)
    env = prop.amplitude_envelope_hbt(x)
    assert env.shape == x.shape
    x2d = np.random.randn(1000, 3)
    env2d = prop.amplitude_envelope_hbt(x2d)
    assert env2d.shape == x2d.shape


def test_instantaneous_frequency_hbt_shape():
    x = np.random.randn(1000)
    fs = 1000
    freq = prop.instantaneous_frequency_hbt(x, fs)
    assert freq.shape == (x.shape[0] - 1,)
    x2d = np.random.randn(1000, 2)
    freq2d = prop.instantaneous_frequency_hbt(x2d, fs)
    assert freq2d.shape == (x2d.shape[0] - 1, x2d.shape[1])


def test_instantaneous_amplitude_hbt_shape():
    x = np.random.randn(1000)
    ia = prop.instantaneous_amplitude_hbt(x)
    assert ia.shape == x.shape
    x2d = np.random.randn(1000, 2)
    ia2d = prop.instantaneous_amplitude_hbt(x2d)
    assert ia2d.shape == x2d.shape


def test_sma_signal_shape():
    x = np.random.randn(100)
    sma = prop.sma_signal(x.reshape(-1, 1))
    assert sma.shape == (100,)
    x2d = np.random.randn(100, 3)
    sma2d = prop.sma_signal(x2d)
    assert sma2d.shape == (100,)


def test_magnitude_signal_shape():
    x = np.random.randn(100)
    mag = prop.magnitude_signal(x.reshape(-1, 1))
    assert mag.shape == (100,)
    x2d = np.random.randn(100, 3)
    mag2d = prop.magnitude_signal(x2d)
    assert mag2d.shape == (100,)


def test_rolling_rms_shape():
    x = np.random.randn(1000)
    frame_length = 100
    hop_length = 50
    rms = prop.rolling_rms(x, frame_length, hop_length)
    expected_frames = 1 + (len(x) + frame_length // 2 * 2 - frame_length) // hop_length
    assert rms.shape == (expected_frames,)
    x2d = np.random.randn(1000, 2)
    rms2d = prop.rolling_rms(x2d, frame_length, hop_length)
    assert rms2d.shape == (expected_frames,)


def test_rolling_zcr_shape():
    x = np.random.randn(1000)
    frame_length = 100
    hop_length = 50
    zcr = prop.rolling_zcr(x, frame_length=frame_length, hop_length=hop_length)
    expected_frames = 1 + (len(x) + frame_length // 2 * 2 - frame_length) // hop_length
    assert zcr.shape == (expected_frames,)
    x2d = np.random.randn(1000, 2)
    zcr2d = prop.rolling_zcr(x2d, frame_length=frame_length, hop_length=hop_length)
    assert zcr2d.shape == (expected_frames,)
