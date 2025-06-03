import numpy as np
from caits.fe._statistical import signal_stats as ss1
from dev.fe._statistical import signal_stats as ss2
from _utils import init_dataset

def test_signal_stats():
    data = init_dataset(uni_dim=False)

    print(data.shape)

    stats1 = ss1(data, name="test", axis=0)
    stats2 = ss2(data.T, name="test", axis=1)

    for k, v in stats1.items():
        print(k, v)
        print(k, stats2[k])
        print(v.shape, stats2[k].shape)
        assert np.array_equal(v, stats2[k], equal_nan=True)