"""Miscellaneous utilities for manipulating Numpy ndarrays."""

import numpy as np


def broadcast_axis_shape(array1, array2, axis=0):
    """Get shape where `array1.shape[axis] == array2.shape[axis]`.
    
    Parameters
    ----------
    array1 : np.ndarray
        Array to be broadcast.
    array2 : np.ndarray
        Array to broadcast to.
    axis : int; default=0.
        Axis of array to broadcast.
    
    Returns
    -------
    shape : tuple
        Broadcasted shape for `array1`.

    """
    shape = list(array1.shape)
    shape[axis] = array2.shape[axis]
    return tuple(shape)


def broadcast_axes(array1, array2, axis=0):
    """Broadcast `array1` to `array2.shape`, but only along specified `axis`.
    
    Parameters
    ----------
    array1 : np.ndarray
        Array to be broadcast.
    array2 : np.ndarray
        Array to broadcast to.
    axis : int; default=0.
        Axis of array to broadcast.

    Returns
    -------
    new_array : np.ndarray
        Broadcasted version of `array1`.

    """
    broadcasted_shape = broadcast_axis_shape(array1, array2, axis=axis)
    new_array = np.broadcast_to(array1, broadcasted_shape)
    return new_array


def broadcast_dicts(d1, d2, axis=0, debug_memory=False):
    """Broadcast axis length of all arrays in one dict against another dict.

    Parameters
    ----------
    d1 : dict of np.ndarray.
        Dictionary containing arrays to be broadcast.
    d2 : dict of np.ndarray.
        Dictionary containing arrays to broadcast to.
    axis : int; default=0.
        Axis to broadcast.
    debug_memory : bool; default=False.
        If True, raise AssertionError if broadcasted arrays do not share memory
        with originals.

    Returns
    -------
    dict of np.ndarray.
    
    """

    new_d = {}
    for k, v in d1.items():
        temp = d2.copy()
        for v2 in temp.values():
            if (v.shape[axis] == 1) and (v2.shape[axis] > 1):
                # Broadcasting is possible.
                new_v = broadcast_axes(v, v2, axis=axis)
                if debug_memory:
                    assert np.shares_memory(new_v, v)
                new_d[k] = new_v
            else:
                # Incompatible shape for broadcasting.
                new_d[k] = v
    
    if len(new_d) == 0:
        # There was nothing to broadcast to
        new_d = d1.copy()

    return new_d
