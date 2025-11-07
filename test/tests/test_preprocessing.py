from test.utils import make_test, total_gen
from caits import preprocessing


def gen_native_sr():
    for x in range(100, 1000, 100):
        yield x

def gen_target_sr(native_sr):
   for x in range(native_sr // 10, native_sr, native_sr // 10):
       yield x

def gen_dtype():
    for x in ["float32", "float64", "int32", "int64"]:
        yield x

def gen_epsilon():
    for x in range(3):
        yield 1e-5 * (10 ** x)


normalize_signal_params = []

resample_2d_params = [
    {
        "name": "native_sr",
        "func": gen_native_sr,
        "kwargs": {}
    },
    {
        "name": "target_sr",
        "func": gen_target_sr,
        "kwargs": {
            "native_sr": "from_native_sr"
        }
    },
    {
        "name": "dtype",
        "func": gen_dtype,
        "kwargs": {}
    }
]

trim_signal_params = [
    {
        "name": "epsilon",
        "func": gen_epsilon,
        "kwargs": {}
    }
]



funs = [
    preprocessing.normalize_signal,
    preprocessing.resample_2d,
    preprocessing.trim_signal
]

params = [
    total_gen(normalize_signal_params),
    total_gen(resample_2d_params),
    total_gen(trim_signal_params),
]


for fun, params in zip(funs, params):
    globals()[f"test_{fun.__name__}"] = make_test(fun, params)
