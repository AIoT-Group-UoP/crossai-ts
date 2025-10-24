import numpy as np
import caits.filtering as filtering
import pytest
import itertools

# Median simple
kernel_size = [x for x in range(1, 7, 2)]

# Median gen
window_size = [x for x in range(1, 10)]

# Butterworth
fs = [x for x in range(100, 1000, 100)]
fs_cutoff_combs = [
    (f, c)
    for f in fs for c in range(f // 10, f // 2, f // 20)
]
filter_type = ["lowpass", "highpass"]

order = [x for x in range(1, 10)]
zi_enable = [True, False]
butterworth_method = ['filtfilt', 'sosfilt', 'sosfiltfilt']

# Gaussian
sigma = [x for x in range(1, 4)]
gaussian_mode = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
truncate = [x * 0.5 for x in range(9)]
cval = [x * 0.5 for x in range(9)]

filter_median_simple_combs = kernel_size
filter_median_gen_combs = window_size
filter_butterworth_lp_hp_combs = list(itertools.product(fs_cutoff_combs, filter_type))
filter_gaussian_combs = list(itertools.product(sigma, gaussian_mode, truncate, cval))

bp_cutoff_combs = [
    ((f, (l, h)), "bandpass")
    for f in fs
    for h in range(f // 10, f // 2, f // 30)
    for l in range(h // 10, h // 2, h // 10)
]

filter_butterworth_bound_combs = filter_butterworth_lp_hp_combs + bp_cutoff_combs
filter_butterworth_combs = itertools.product(filter_butterworth_bound_combs, order, zi_enable, butterworth_method)


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

filter_gaussian_kwargs = [
    {
        "sigma": c[0],
        "mode": c[1],
        "truncate": c[2],
        "cval": c[3]
    }
    for c in filter_gaussian_combs
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

filter_gaussian_params = [
    pytest.param(
        kwargs,
        id="-".join([f"{k}-{v}" for k, v in kwargs.items()])
    )
    for kwargs in filter_gaussian_kwargs
]

funs = [
    filtering.filter_median_simple,
    filtering.filter_median_gen,
    filtering.filter_butterworth,
    filtering.filter_gaussian
]

params = [simple_median_params, median_gen_params, filter_butterworth_params, filter_gaussian_params]


def _make_test(fun, params):
    @pytest.mark.parametrize(
        "kwargs", params
    )
    def test_func(arr, kwargs):
        assert fun(arr[0], axis=arr[1], **kwargs).shape == arr[0].shape

    return test_func

for fun, params in zip(funs, params):
    globals()[f"test_{fun.__name__}"] = _make_test(fun, params)