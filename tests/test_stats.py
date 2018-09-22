import pytest
from timspyutils.stats import *

import numpy as np
from scipy import stats

def test_one_scaled_uniform_sample():
    a = 1e-1
    b = 1e1
    lsd_uniform = LogScaledDistribution(stats.uniform, a, b)
    s = lsd_uniform.rvs()
    assert b >= s >= a
    assert s.dtype.type == np.float_

def test_multiple_scaled_uniform_sample():
    a = 1e-1
    b = 1e1
    lsd_uniform = LogScaledDistribution(stats.uniform, a, b)
    s = lsd_uniform.rvs(size = 100)
    assert np.all(b >= s) and np.all(s >= a)
    assert len(s) == 100
    assert s.dtype.type == np.float_