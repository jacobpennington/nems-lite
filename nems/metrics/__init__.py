"""Collection of performance metrics and related utilities.

Score model outputs by MSE, likelihood, prediction correlation, etc.
Measure equivalence, sparsity, and other properties of model predictions and/or
recorded neural responses.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents
--------
    `mse.py`         : Compute mean squared error between prediction and target.
    `correlation.py` : As above, but compute Pearson correlation coefficient.
    `equivalence.py` : Measure functional similarity of different models.
    `sparseness.py`  : Measure lifetime sparseness of responses and predictions.

"""

from .mse import mse, nmse
from .correlation import correlation, noise_corrected_r


def get_metric(name):
    # TODO: track a dict or something to point string ref from
    #       Model.fit api to appropriate function.

    #       Initial idea: only have a hand-coded dict for a few nicknames,
    #       otherwise require that string is the actual name of the function
    #       (which should be easy to find since they're all in the same place).
    pass
