import caits.filtering as filtering
from test.utils import make_test, total_gen


def gen_kernel_size(max_size):
    for x in range(1, max_size+1, 2):
        yield x

def gen_window_size(max_size):
    for x in range(1, max_size+1):
        yield x

def gen_fs(min_fs, max_fs, step):
    for x in range(min_fs, max_fs+step, step):
        yield x

def gen_filter_type():
    for t in ["lowpass", "highpass", "bandpass"]:
        yield t

def gen_cutoff(fs, filter_type):
    if filter_type != "bandpass":
        for f in range(fs // 10, fs // 2, fs // 20):
            yield f
    else:
        for h in range(fs // 10, fs // 2, fs // 30):
            for l in range(h // 10, h // 2, h // 10):
                yield l, h

def gen_order(max_order):
    for x in range(1, max_order):
        yield x

def gen_zi_enable():
    yield True
    yield False

def gen_butterworth_method():
    for x in ["filtfilt", "sosfilt", "sosfiltfilt"]:
        yield x

def gen_sigma():
    for x in range(1, 4):
        yield x

def gen_gaussian_mode():
    for x in ["reflect", "constant", "nearest", "mirror", "wrap"]:
        yield x

def gen_truncate():
    for x in range(9):
        yield 0.5 * x

def gen_cval():
    for x in range(9):
        yield 0.5 * x


filter_median_simple_params = [
    {
        "name": "kernel_size",
        "func": gen_kernel_size,
        "kwargs": {
            "max_size": 7
        }
    }
]

filter_median_gen_params = [
    {
        "name": "window_size",
        "func": gen_window_size,
        "kwargs": {
            "max_size": 10
        }
    }
]

filter_butterworth_params = [
    {
        "name": "fs",
        "func": gen_fs,
        "kwargs": {
            "min_fs": 100,
            "max_fs": 1000,
            "step": 100
        }
    },
    {
        "name": "filter_type",
        "func": gen_filter_type,
        "kwargs": {}
    },
    {
        "name": "cutoff_freq",
        "func": gen_cutoff,
        "kwargs": {
            "fs": "from_fs",
            "filter_type": "from_filter_type"
        }
    },
    {
        "name": "order",
        "func": gen_order,
        "kwargs": {
            "max_order": 10
        }
    },
    {
        "name": "zi_enable",
        "func": gen_zi_enable,
        "kwargs": {}
    },
    {
        "name": "method",
        "func": gen_butterworth_method,
        "kwargs": {}
    }
]


filter_gaussian_params = [
    {
        "name": "sigma",
        "func": gen_sigma,
        "kwargs": {}
    },
    {
        "name": "mode",
        "func": gen_gaussian_mode,
        "kwargs": {}
    },
    {
        "name": "truncate",
        "func": gen_truncate,
        "kwargs": {}
    },
    {
        "name": "cval",
        "func": gen_cval,
        "kwargs": {}
    }
]

funs = [
    filtering.filter_median_simple,
    filtering.filter_median_gen,
    filtering.filter_butterworth,
    filtering.filter_gaussian
]

params = [
    total_gen(filter_median_simple_params),
    total_gen(filter_median_gen_params),
    total_gen(filter_butterworth_params),
    total_gen(filter_gaussian_params)
]


for fun, params in zip(funs, params):
    globals()[f"test_{fun.__name__}"] = make_test(fun, params)