import pytest

from nems.layers.base import Parameter


def test_constructor():
    # Name should be only required argument, shape defaults to ()
    p = Parameter('test')
    assert p.shape == ()

    p = Parameter('test', shape=(2,3))
    assert p.shape == (2,3)
    assert p.prior.shape == (2,3)
    

def test_epsilon_bounds():
    test1 = Parameter('test1', bounds=(0, 1))
    assert test1.bounds == (0, 1)
    # 0 should be replaced by machine epsilon for float32,
    # `np.finfo(np.float32).eps`
    test2 = Parameter('test2', bounds=(0, 1), zero_to_epsilon=True)
    assert test2.bounds[0] > 0

    with pytest.raises(ValueError):
        # Value error b/c 0 should be out of bounds
        test2.update(0)

    with pytest.raises(AttributeError):
        # Attribute error b/c no Phi is assigned, so Parameter
        # doesn't actually have any values
        test2.update(0.5)
