"""Collection of cost functions & performance evaluation utilities.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents
--------
    `performance.py` : Score model outputs based on prediction correlation, etc
    `equivalence.py` : Measure functional similarity of different models.

"""

from .performance import get_cost_function
