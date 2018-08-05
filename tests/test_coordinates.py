import pytest
from timspyutils.coordinates import *

import numpy as np

def test_cartesian_to_polar():
    x = np.sqrt(2.0)
    y = np.sqrt(2.0)
    r, phi = cartesian2polar(x, y)
    assert np.all(np.allclose([r, phi], [2.0, np.pi/4.0]))
    r, phi = cartesian2polar(-x, -y)
    assert np.all(np.allclose([r, phi], [2.0, (-3.0 * np.pi)/4.0]))