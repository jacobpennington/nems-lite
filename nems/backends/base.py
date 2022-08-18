

class Backend:
    """TODO: docs"""

    def __init__(self, nems_model, input, fitter_options=None):
        self.nems_model = nems_model
        self._input = input
        self._model = None
        if fitter_options is None:
            fitter_options = {}

    @property
    def model(self):
        """Get backend equivalent of NEMS Model."""
        model = getattr(self, '_model', None)
        if model is None:
            # Build model using most recent input
            model = self.build_model(self._input)

    def build(self, input, *args, **kwargs):
        """Return whatever object the backend uses for fitting/predicting."""
        raise NotImplementedError

    def fit(self, input, *args, **kwargs):
        """Call _model.fit() or equivalent."""
        raise NotImplementedError

    def predict(self, input, *args, **kwargs):
        """Call _model.predict() or equivalent."""
        raise NotImplementedError
