"""A collection of preprocessing utilities.

# TODO: These contents don't all exist yet, sketching this out based on
#       the strfy proposal document.
Contents:
    `spectrogram.py`: Convert sound waveforms to spectrograms.
    `filter.py`: Transform data using smoothing, high-pass filter, etc.
    `normalize.py`: Scale data to a standard dynamic range.
    `rasterize.py`: Convert data to continuous time series (ex: spikes -> PSTH).
    `split.py`: Separate data into estimation & validation sets.
    `mask.py`: Exclude subsets of data based on various criteria.
    `merge.py`: Align and combine datasets for fitting a single large model.

"""

from nems.preprocessing.split import (
    indices_by_fraction,
    split_at_indices,
    get_jackknife_indices,
    get_jackknife
    )