import numpy as np
import caits.filtering as filtering
import pytest
import itertools

kernel_size = [x for x in range(1, 7, 2)]
window_size = [x for x in range(1, 10)]
fs = [x for x in range(100, 1000, 100)]
fs_cutoff_combs = [
    (f, c)
    for f in fs for c in range(f // 10, f // 2, f // 20)
]
filter_type = ["lowpass", "highpass"]

order = [x for x in range(1, 10)]
zi_enable = [True, False]
method = ['filtfilt', 'sosfilt', 'sosfiltfilt']

sigma = [x for x in range(1, 10)]

filter_median_simple_combs = kernel_size
filter_median_gen_combs = window_size
filter_butterworth_lp_hp_combs = list(itertools.product(fs_cutoff_combs, filter_type))

bp_cutoff_combs = [
    ((f, (l, h)), "bandpass")
    for f in fs
    for h in range(f // 10, f // 2, f // 30)
    for l in range(h // 10, h // 2, h // 10)
]

filter_butterworth_bound_combs = filter_butterworth_lp_hp_combs + bp_cutoff_combs
filter_butterworth_combs = itertools.product(filter_butterworth_bound_combs, order, zi_enable, method)


filter_median_simple_kwargs = [
    {
        "kernel_size": ws
    }
    for ws in filter_median_simple_combs
]

filter_median_gen_kwargs = [
    {
        "window_size": ws
    }
    for ws in filter_median_gen_combs
]

filter_butterworth_kwargs = [
    {
        "fs": c[0][0][0],
        "filter_type": c[0][1],
        "cutoff_freq": c[0][0][1],
        "order": c[1],
        "zi_enable": c[2],
        "method": c[3]
    }
    for c in filter_butterworth_combs
]


simple_median_params = [
    pytest.param(
        kwargs,
        id="-".join([f"{k}={v}" for k, v in kwargs.items()])
    )
    for kwargs in filter_median_simple_kwargs
]

median_gen_params = [
    pytest.param(
        kwargs,
        id="-".join([f"{k}={v}" for k, v in kwargs.items()])
    )
    for kwargs in filter_median_gen_kwargs
]


filter_butterworth_params = [
    pytest.param(
        kwargs,
        id="-".join([f"{k}={v}" for k, v in kwargs.items()])
    )
    for kwargs in filter_butterworth_kwargs
]


@pytest.mark.parametrize(
    "kwargs", simple_median_params
)
def test_median_simple(arr, kwargs):
    assert filtering.filter_median_simple(arr[0], axis=arr[1], **kwargs).shape == arr[0].shape


@pytest.mark.parametrize(
    "kwargs", median_gen_params
)
def test_median_gen(arr, kwargs):
    assert filtering.filter_median_gen(arr[0], axis=arr[1], **kwargs).shape == arr[0].shape


@pytest.mark.parametrize(
    "kwargs", filter_butterworth_params
)
def test_median_gen(arr, kwargs):
    assert filtering.filter_butterworth(arr[0], axis=arr[1], **kwargs).shape == arr[0].shape
