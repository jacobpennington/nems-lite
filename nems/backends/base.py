

class Backend:
    """TODO: docs"""

    def __init__(self, nems_model, input, eval_kwargs=None, **backend_options):
        self.nems_model = nems_model
        if eval_kwargs is None: eval_kwargs = {}
        self.model = self.build(
            input, eval_kwargs=eval_kwargs, **backend_options
            )

    # Must accept input data and eval_kwargs (dict of kwargs for Model.evaluate)
    def build(self, input, eval_kwargs=None, **backend_options):
        """Return whatever object the backend uses for fitting/predicting."""
        raise NotImplementedError
        
    def fit(self, input, *args, eval_kwargs=None, **kwargs):
        """Call _model.fit() or equivalent."""
        raise NotImplementedError

    def predict(self, input, *args, eval_kwargs=None, **kwargs):
        """Call _model.predict() or equivalent."""
        raise NotImplementedError
