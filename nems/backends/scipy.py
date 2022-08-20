import scipy

from .base import Backend


class SciPyBackend(Backend):
    """Default backend: just passthrough NEMS model methods except fit."""
    def build(self, input=None, eval_kwargs=None, **backend_options):
        return self.nems_model
    def predict(self, input, eval_kwargs=None):
        if eval_kwargs is None: eval_kwargs = {}
        return self.nems_model.predict(input, **eval_kwargs)
    def fit(self, input, target=None, eval_kwargs=None, cost_function='mse',
            log_spacing=5, **fitter_options):
        """TODO: docs.

        See `nems.models.base.Model.fit`.

        Parameters
        ----------
        log_spacing : int; default=5.
            Log progress of fitter every `log_spacing` iterations.
            Note: this is the number of iterations, not the number of cost
            function evaluations (of which there may be many per iteration).

        """
        if eval_kwargs is None: eval_kwargs = {}
        if 'use_existing_maps' not in eval_kwargs:
            eval_kwargs['use_existing_maps'] = True
        if 'undo_reorder' not in eval_kwargs:
            eval_kwargs['undo_reorder'] = False
        if eval_kwargs.get('skip_initialization'):
            data = input
        else:
            data = DataSet


        # TODO: possibly scrap this and put scipy .fit back in Model.fit?
        #       and just not have a separate Backend. Getting confusing/frustrating
        #       with duplicating the same options over and over for no reason other
        #       than organizing the code. Can still return some
        #       TBD FitResult object for all backends. 

        # TODO: need something like generate_batch_data so that cost can be
        #       computed one batch at a time.


        def _scipy_cost_wrapper(_vector, _model, _input, _target,
                                _cost_function, _eval_kwargs):
            _model.set_parameter_vector(_vector, ignore_checks=True)
            _prediction = _model.predict(_input, **_eval_kwargs)
            cost = _cost_function(_prediction, _target)
            return cost

        # TODO: better callback scheme, use logging
        log_info = {'iteration': 0}
        def _scipy_callback(_vector):
            nonlocal log_info, log_spacing, input, target
            if log_info['iteration'] % log_spacing == 0:
                # Shouldn't need to set parameter vector, that should have
                # been done by the optimization iteration.
                prediction = self.nems_model.predict(input, **eval_kwargs)
                cost = cost_function(prediction, target)
                print(f"iteration {log_info['iteration']},"
                        f" error is: {cost:.8f}...")
            log_info['iteration'] += 1
        
        initial_parameters = self.nems_model.get_parameter_vector(as_list=True)
        bounds = self.nems_model.get_bounds_vector(none_for_inf=True)
        cost_args = (self.nems_model, input, target, cost_function, eval_kwargs)
        fit_result = scipy.optimize.minimize(
            _scipy_cost_wrapper, initial_parameters, cost_args,
            bounds=bounds,
            method='L-BFGS-B', callback=_scipy_callback, **fitter_options
        )
        print(
            f"Fit successful: {fit_result.success}\n"
            f"Status: {fit_result.status}\n"
            f"Message: {fit_result.message}"
        )
        improved_parameters = fit_result.x
        self.nems_model.set_parameter_vector(improved_parameters)
