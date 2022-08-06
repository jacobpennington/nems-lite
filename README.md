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
