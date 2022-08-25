"""Collection of performance metrics and related utilities.

Score model outputs by MSE, likelihood, prediction correlation, etc.
Measure equivalence, sparsity, and other properties of model predictions and/or
recorded neural responses. These can also be used as cost functions for the
default SciPy backend.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents
--------
    `mse.py`         : Compute mean squared error between prediction and target.
    `correlation.py` : As above, but compute Pearson correlation coefficient.
    `equivalence.py` : Measure functional similarity of different models.
    `sparseness.py`  : Measure lifetime sparseness of responses and predictions.

"""

from .correlation import correlation, noise_corrected_r
