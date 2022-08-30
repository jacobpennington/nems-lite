import numpy as np

from nems import Model
from nems.layers.base import Phi, Parameter


def test_phi_bounds():
    m = Model.from_keywords('wc.18x4.g')  # 8 parameters
    phi = m['wc'].parameters
    assert len(phi.get_bounds_vector()) == 8
    lower1, upper1 = (0, 1)
    lower2, upper2 = (0, 2)
    phi.set_bounds({'mean': (lower1, upper1), 'sd': (lower2, upper2)})

    vector1 = np.full(shape=(4,), fill_value=(upper1-lower1)/2)  # in bounds
    vector2 = np.full(shape=(4,), fill_value=(upper2-lower2)/2)
    vector = np.concatenate([vector1, vector2], axis=0)
    assert phi.within_bounds(vector)

    vector[0] = lower1 - 1                 # less than lower
    assert not phi.within_bounds(vector)  
    vector[0] = lower1                     # closed interval
    assert phi.within_bounds(vector)
    vector[-1] = upper2 + 1                # greater than upper
    assert not phi.within_bounds(vector)
    # Last index should be the only one out of range
    assert sum(phi.get_indices_outof_range(vector)) == 1
    assert phi.get_indices_outof_range(vector)[-1]
