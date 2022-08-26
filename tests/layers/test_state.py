import pytest

from nems.layers.state import StateGain


def test_constructor():
    # TODO: add require_shape, other tests
    sg = StateGain(shape=(2,2))
