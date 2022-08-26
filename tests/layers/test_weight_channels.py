import pytest

from nems.layers.weight_channels import WeightChannels, GaussianWeightChannels


def test_constructor():
    with pytest.raises(TypeError):
        # No shape
        wc = WeightChannels()
    with pytest.raises(TypeError):
        # No shape
        wc = WeightChannels(None)
    with pytest.raises(TypeError):
        # Too few dimensions
        # TODO: may decide to allow as few as 1 dims in the future
        wc = WeightChannels(shape=(2,))

    # These should raise no errors
    wc = WeightChannels(shape=(18,2))
    wc = WeightChannels(shape=(10,3,5))
    wcg = GaussianWeightChannels(shape=(18,2))


# TODO: tests for evaluate shape, others.
