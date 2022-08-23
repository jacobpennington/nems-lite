"""TODO: docs

Alternative computational frameworks for fitting models.
To include: SciPy (default), TensorFlow, ... (others)

NOTE: No public API here so that backend dependencies are only imported if used.

"""

# TODO: better way to do this. For now, new backends just have to add on
#       this if/else chain.
def get_backend(name):
    name = name.lower()
    if name == 'scipy':
        from .scipy import SciPyBackend
        backend = SciPyBackend
    elif (name == 'tf') or (name == 'tensorflow'):
        from .tf import TensorFlowBackend
        backend = TensorFlowBackend
    else:
        raise NotImplementedError(f"Unrecognized backend: {name}.")

    return backend
