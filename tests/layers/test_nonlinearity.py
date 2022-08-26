import pytest

from nems.layers.nonlinearity import DoubleExponential, RectifiedLinear


def test_constructors():
    # TODO: add require_shape, other tests
    dexp = DoubleExponential(shape=(5,))
    relu = RectifiedLinear(shape=(1,))
