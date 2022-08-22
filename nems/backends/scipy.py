import scipy.optimize

from nems.metrics import get_metric
from .base import Backend, FitResults


class SciPyBackend(Backend):
    """Default backend: just passthrough NEMS model methods except fit."""
    def _build(self, data=None, eval_kwargs=None, **backend_options):
        return self.nems_model

    def _fit(self, data, eval_kwargs=None, cost_function='mse',
             epochs=1, log_spacing=5, **fitter_options):
        """TODO: docs.

        See `nems.models.base.Model.fit`.

        Parameters
        ----------
        cost_function : str or func; default='mse'
            Specifies which metric to use for computing error while fitting.
            If str  : Invoke `nems.metrics.get_metric(str)`.
            If func : Use this function to compute errors. Should accept two
                      array arguments and return float. 
        log_spacing : int; default=5.
            Log progress of fitter every `log_spacing` iterations.
            Note: this is the number of iterations, not the number of cost
            function evaluations (of which there may be many per iteration).

        Notes
        -----
        For this Backend, additional **fitter_options are captured because they
        need to be passed directly to `scipy.optimize.minimize`.

        """

        if eval_kwargs is None: eval_kwargs = {}
        eval_kwargs['use_existing_maps'] = True
        eval_kwargs['as_dataset'] = True
        batch_size = eval_kwargs.get('batch_size', 0)

        if isinstance(cost_function, str):
            # Convert string reference to metric function
            cost_function = get_metric(cost_function)
        wrapper = FitWrapper(
            cost_function, self.nems_model, eval_kwargs, log_spacing
            )
        
        for ep in range(epochs):
            print(f"Epoch {ep}")
            print("="*30)
            if batch_size == 0:
                # Don't need to mess with batching.
                _data = data.copy()
                fit_result = wrapper.get_fit_result(_data, **fitter_options)
            else:
                batch_generator = self.nems_model.generate_batch_data(
                    data, **eval_kwargs
                    )
                for i, _data in enumerate(batch_generator):
                    print(" "*4 + f"Batch {i}")
                    print(" "*4 + "-"*26)
                    fit_result = wrapper.get_fit_result(_data, **fitter_options)

        print(
            f"Fit successful: {fit_result.success}\n"
            f"Status: {fit_result.status}\n"
            f"Message: {fit_result.message}"
        )
        self.model.set_parameter_vector(fit_result.x)
        
        initial_parameters = wrapper.initial_parameters
        final_parameters = wrapper.model.get_parameter_vector()
        initial_error = wrapper(initial_parameters)
        final_error = wrapper(final_parameters)
        nems_fit_results = FitResults(
            initial_parameters, final_parameters, initial_error, final_error,
            backend_name='scipy', scipy_fit_result=fit_result
        )

        return nems_fit_results

    def predict(self, input, eval_kwargs=None):
        if eval_kwargs is None: eval_kwargs = {}
        return self.nems_model.predict(input, **eval_kwargs)


class FitWrapper:
    def __init__(self, fn, model, eval_kwargs, log_spacing):
        self.fn = fn
        try:
            self.name = fn.__name__
        except AttributeError:
            self.name = 'none'
        self.model = model
        self.initial_parameters = model.get_parameter_vector(as_list=True)
        self.bounds = model.get_bounds_vector(none_for_inf=True)
        self.eval_kwargs = eval_kwargs
        self.log_spacing = log_spacing
        self.data = None
        self.iteration = 0

    def __call__(self, vector):
        self.model.set_parameter_vector(vector, ignore_checks=True)
        evaluated_data = self.model.evaluate(self.data, **self.eval_kwargs)
        self.data = evaluated_data
        cost = self.compute_cost()
        return cost

    # TODO: better callback scheme, use logging
    def callback(self, vector):
        if self.iteration % self.log_spacing == 0:
            # Shouldn't need to set parameter vector, that should have
            # been done by the optimization iteration.
            cost = self.__call__(vector)
            print(" "*8 + f"Iteration {self.iteration},"
                    f" error is: {cost:.8f}...")
        self.iteration += 1

    def compute_cost(self):
        prediction_list, target_list = self._get_arrays()
        if (len(prediction_list) == 1) and (len(target_list) == 1):
            cost = self.fn(prediction_list[0], target_list[0])
        else:
            # Dict keys are not stored in a guaranteed order, so can't expect
            # .values() to match up even if the lengths are the same. Need to
            # provide a separate mapping of {'pred_key' -> 'target_key'}
            # (and do something different instead of just getting lists).
            raise NotImplementedError(
                "TODO: SciPy cost function for multiple predictions/targets."
                )

        return cost

    def _get_arrays(self):
        prediction = self.data.outputs
        target = self.data.targets

        if len(prediction) == 0:
            # No predictions, error
            raise ValueError(f"{self.name}: No predictions found in data.")
        else:
            predictions = list(prediction.values())
        
        if len(target) == 0:
            # No target, error
            raise ValueError(f"{self.name}: No targets found in data.")
        else:
            targets = list(target.values())

        return predictions, targets

    def get_fit_result(self, data, **fitter_options):
        """TODO: docs."""
        self.data = data
        self.iteration = 0

        if 'method' not in fitter_options:
            fitter_options['method'] = 'L-BFGS-B'
        fit_result = scipy.optimize.minimize(
            self, self.initial_parameters, bounds=self.bounds,
            callback=self.callback, **fitter_options
            )

        return fit_result
