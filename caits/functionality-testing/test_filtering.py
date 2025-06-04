import numpy as np
import pytest
import caits.filtering as filtering

def test_filter_median_simple_basic():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_median_simple(arr, kernel_size=3)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == arr.shape

def test_filter_median_simple_input_type():
    arr = np.random.rand(100, 2)
    assert isinstance(arr, np.ndarray)
    filtering.filter_median_simple(arr, kernel_size=3)

def test_filter_median_simple_shape():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_median_simple(arr, kernel_size=3)
    assert filtered.shape == arr.shape

def test_filter_median_gen_basic():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_median_gen(arr, window_size=3)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == arr.shape

def test_filter_median_gen_input_type():
    arr = np.random.rand(100, 2)
    assert isinstance(arr, np.ndarray)
    filtering.filter_median_gen(arr, window_size=3)

def test_filter_median_gen_shape():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_median_gen(arr, window_size=3)
    assert filtered.shape == arr.shape

def test_filter_butterworth_basic():
    arr = np.random.rand(1000, 2)
    fs = 1000
    filtered = filtering.filter_butterworth(arr, fs=fs, filter_type='lowpass', cutoff_freq=100)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == arr.shape

def test_filter_butterworth_input_type():
    arr = np.random.rand(1000, 2)
    assert isinstance(arr, np.ndarray)
    filtering.filter_butterworth(arr, fs=1000, filter_type='lowpass', cutoff_freq=100)

def test_filter_butterworth_shape():
    arr = np.random.rand(1000, 2)
    filtered = filtering.filter_butterworth(arr, fs=1000, filter_type='lowpass', cutoff_freq=100)
    assert filtered.shape == arr.shape

def test_filter_gaussian_basic():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_gaussian(arr, sigma=1)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == arr.shape

def test_filter_gaussian_input_type():
    arr = np.random.rand(100, 2)
    assert isinstance(arr, np.ndarray)
    filtering.filter_gaussian(arr, sigma=1)

def test_filter_gaussian_shape():
    arr = np.random.rand(100, 2)
    filtered = filtering.filter_gaussian(arr, sigma=1)
    assert filtered.shape == arr.shape
