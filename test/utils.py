import pytest

def total_gen(gen_confs, prev=None):
    if not gen_confs:
        return []

    if prev is None:
        prev = {}
    gen_conf = gen_confs[0]
    name = gen_conf["name"]
    gen = gen_conf["func"]
    kwargs = gen_conf["kwargs"].copy()

    for k in kwargs.keys():
        if isinstance(kwargs[k], str) and kwargs[k].startswith("from_"):
            kwargs[k] = prev[kwargs[k][5:]]

    for x in gen(**kwargs):
        res = prev | {name: x}
        if len(gen_confs) > 1:
            for y in total_gen(gen_confs[1:], res):
                yield y
        else:
            yield pytest.param(
                res,
                id="-".join([f"{k}={v}" for k, v in res.items()])
            )

def make_test(fun, params):
    @pytest.mark.parametrize(
        "kwargs", params
    )
    def test_func_with_kwargs(arr, kwargs):
        assert fun(arr[0], axis=arr[1], **kwargs).shape == arr[0].shape

    def test_func_without_kwargs(arr):
        assert fun(arr[0], axis=arr[1]).shape == arr[0].shape

    return test_func_with_kwargs if params else test_func_without_kwargs
