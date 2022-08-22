import numpy as np


class Backend:
    """TODO: docs"""

    def __init__(self, nems_model, data, eval_kwargs=None, **backend_options):
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


class FitResults:
    """TODO: docs"""

    def __init__(self, initial_parameters, final_parameters, initial_error,
                 final_error, backend_name, **misc):
        self.initial_error = initial_error
        self.final_error = final_error
        self.n_parameters = np.array(initial_parameters).size
        self.n_parameters_changed = sum(initial_parameters != final_parameters)
        self.backend = backend_name
        self.misc = misc

    def __repr__(self):
        attrs = self.__dict__
        misc = attrs.pop('misc')

        string = "Fit Results:\n"
        string += "="*11 + "\n"
        for k, v in attrs.items():
            string += f"{k}: {v}\n"
        string += '-'*11 + "\n"
        string += f"Misc:\n"
        string += f"{misc}\n"
        string += "="*11

        return string
