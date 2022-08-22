

class Backend:
    """TODO: docs"""

    def __init__(self, nems_model, data, eval_kwargs=None, **backend_options):
        # TODO: store copy of nems_model instead.
        self.nems_model = nems_model
        if eval_kwargs is None: eval_kwargs = {}
        self.model = self._build(
            data, eval_kwargs=eval_kwargs, **backend_options
            )

    # Must accept input data and eval_kwargs (dict of kwargs for Model.evaluate)
    def _build(self, data, eval_kwargs=None, **backend_options):
        """Return whatever object the backend uses for fitting/predicting.
        
        This method should expect `data` to be a DataSet.
        TODO: rewrite doc

        """
        raise NotImplementedError
        
    def _fit(self, data, *args, eval_kwargs=None, **kwargs):
        """Call _model.fit() or equivalent.
        
        This method should expect `data` to be a DataSet.
        TODO: rewrite doc
        
        """
        raise NotImplementedError

    def predict(self, input, *args, eval_kwargs=None, **kwargs):
        """Call _model.predict() or equivalent.
        
        This should expect `input` in the same format as
        `nems.models.base.Model.evaluate`, i.e. ndarray or dict.
        TODO: rewrite doc
        
        """
        raise NotImplementedError
