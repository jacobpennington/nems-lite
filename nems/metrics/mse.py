import numpy as np

# TODO: I don't love having different argument names for mse vs nmse, but I
#       wanted to make it clear that the ordering doesn't matter for the former
#       (and does for the latter). Similar for other modules in this directory.
def mse(x, y):
    """Compute the mean squared error (MSE) between arrays x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays must have the same shape. Most commonly, these will be a
        model output (prediction) and a recorded response (target).

    Returns
    -------
    mse : float

    Examples
    --------
    >>> prediction = model.predict(data)
    >>> target = data['response']
    >>> error = mse(output, target)

    >>> model.score()

    Notes
    -----
    This implementation is compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation.

    """
    squared_errors = (x - y)**2
    return np.mean(squared_errors)


def nmse(prediction, target):
    """Compute MSE normalized by the standard deviation of target.

    Because this metric is more computationally expensive than MSE, but is
    otherwise equivalent, we suggest using `mse` for fitting and `nmse` as a
    post-fit performance metric only.

    Parameters
    ----------
    prediction : ndarray
    target : ndarray
        Shape must match prediction. MSE will be divided by the
        standard deviation of this array.

    Returns
    -------
    normalized_error : float

    See also
    --------
    .mse
    
    """
    std_of_target = np.std(target)
    error = np.sqrt(mse(prediction, target))

    if std_of_target == 0:
        # TODO: ask Stephen about this, not sure why this is coded this way.
        #       This would only pop up if target is a constant vector, which
        #       should be a very easy fit, but in this case the return value
        #       will always be 1 so optimization would fail to converge.

        #       Maybe this should be `std_of_target = 1` instead?
        #       (i.e. just use the un-normalized error).
        #       Alternatively, just raise NotImplementedError to make it clear
        #       that this method can't be used to fit to constant vectors.
        normalized_error = 1
    else:
        normalized_error = error / std_of_target
    return normalized_error
