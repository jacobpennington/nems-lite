# nems-lite
WIP refactor of lbhb/NEMS and lbhb/nems_db

# (temporary) installation instructions
```
clone <nems-lite> # using url, ssh, or however you normally clone
conda create -n nems-lite python=3.9  # or use your preferred environment manager
pip install -e nems-lite
```
Note: `mkl` library for `numpy` does not play well with `tensorflow`.
If using `conda` to install dependencies, use `conda-forge`
for `numpy` (which uses `openblas` instead of `mkl`):
`conda install -c conda-forge numpy`
(https://github.com/conda-forge/numpy-feedstock/issues/84)

Coming soon, roughly in order of priority:
* Fix cost function behavior in Model.fit to work with both lists and arrays, and set up (non-TF) batched optimization.
* Convert dev_notebooks to unit tests where appropriate (& automatic testing using Travis, like nems0).
* Finish/clean up docs and set up readthedocs.
* Add more Layers from nems0.
* Add core pre-processing and scoring from nems0.
* Try Numba for Layer.evaluate and cost functions.
* Publish through conda install and pip install (and update readme accordingly).
* Convert scripts and dev_notebooks to tutorials where appropriate.
* Other core features (like jackknifed fits, cross-validation, etc.).
* Backwards-compatibility tools for loading nems0 models.
* Implement Jax back-end.
... (other things on the massive issues list)
