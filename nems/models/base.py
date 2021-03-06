from operator import itemgetter
from functools import partial

import numpy as np
import scipy.optimize

from nems.registry import keyword_lib
from nems.metrics import get_metric
# temporarily import layers to make sure they're registered in keyword_lib
import nems.layers  
del nems.layers


class Model:
    """TODO: docstring."""
    # NOTE: there are hard-coded references to the names of these class attrs
    # in `Model.from_json`. Be sure to update that method if these change.
    default_input = 'input'
    default_output = 'output'
    default_state = 'state'
    default_backend = 'scipy'

    def __init__(self, layers=None, name=None):
        """TODO: docstring"""
        self._layers = {}  #  layer.name : layer obj, increment on clashes
        if layers is not None:
            self.add_layers(*layers)
        self.name = name if name is not None else 'unnamed Model'

    @property
    def layers(self):
        """Get all Model Layers. Supports integer or string indexing."""
        return _LayerDict(self._layers)

    @property
    def bounds(self):
        """Get all Model bounds as a dict. See `Layer.bounds`."""
        return {k: v.bounds for k, v in self.layers.items()}

    @property
    def priors(self):
        """Get all Model priors as a dict. See `Layer.priors`."""
        return {k: v.priors for k, v in self.layers.items()}

    def add_layers(self, *layers):
        """Add Layers to this Model, stored in `Model._layers`.

        Parameters
        ----------
        layers : N-tuple of Layers

        See also
        --------
        nems.layers.base.Layer
        
        """
        for layer in layers:
            layer.model = self  # each layer gets a reference to parent Model
            key = name = layer.name
            i = 0
            while key in self._layers:
                # Avoid name clashes
                key = f'{name}.{i}'
                i += 1
            self._layers[key] = layer

    def get_layer(self, key):
        """Get one Layer. Key can be int or string (`Layer.name`)."""
        return self.layers[key]

    def insert_layer(self, index, name=None):
        """TODO: add layer at specific integer index."""
        raise NotImplementedError

    def evaluate(self, input, state=None, input_name=None, state_name=None,
                 output_name=None, return_full_data=True, n=None, time_axis=0,
                 channel_axis=1, undo_reorder=True):
        """Transform input(s) by invoking `Layer.evaluate` for each Layer.

        TODO: add more context, examples

        Parameters
        ----------
        input : ndarray or dict
            If ndarray, use this as the input to the first Layer. Otherwise,
            use keys specified by `input_name` or `Layer.input` to determine
            the first input.
        state : ndarray; optional.
            If ndarray, add to intermediate data dictionary. This option can
            only be used in conjunction with an array `input`. If other data is
            needed, use a dictionary input containing all data.
        input_name : str; optional.
            Specifies which data stream should be provided as input to the
            first layer. Note that priority will be given to `Layer.input` in
            case of a clash. I.e. if `input_name is not None`, but also
            `Layer.input is not None` for the first layer, `Layer.input` will
            be used.
            If None : use `Layer.input` of first layer if specified,
                      otherwise `Model.default_input`.
            If str  : a single input array at key `input_name`.
        state_name : str; optional.
            If str, and `state is not None`, then `state_name` will be the key
            for `state`. Otherwise, `Model.default_state` will be used.
        output_name : str; optional.
            Specifies name(s) for array output the final layer if
            `Layer.output is None`.
            If None : use `Model.default_output`.
            If str  : Use this name for each of the Layer's outputs
                      (incremented if multiple).
        return_full_data : bool; default=True
            If True, return a dictionary containing all input data and all
            uniquely-keyed Layer outputs.
        n : int; optional.
            Evaluate the first `n` Layers (all by defualt).
        time_axis : int; default=0.
            Axis along which time points are stored. Data arrays will be
            re-ordered such that `array.shape = (time_axis, channel_axis, ...)`.
        channel_axis : int; default=1.
            Axis along which stimulus channels, neural channels, or other such
            information is stored (i.e. variables rather than observations).
            Data arrays will be re-ordered such that
            `array.shape = (time_axis, channel_axis, ...)`. Note that an empty
            axis will be added to data with only one dimension to represent a
            single channel (i.e. `shape=(T,)` --> `shape=(T,N)`). This includes
            intermediate outputs generated by `Layer.evaluate`.
        undo_reorder : bool; default=True.
            If True, and data axes were re-ordered, revert to the original
            ordering after evaluating. Set False to return the re-ordered data.

        Returns
        -------
        data : ndarray, list, or dict
            Format depends on `return_full_data` and the `evaluate` method of
            the final layer in the model.

        See also
        --------
        nems.layers.base.Layer.evaluate
        
        """
        if input_name is None:
            input_name = self.default_input
        if output_name is None:
            output_name = self.default_output
        if state_name is None:
            state_name = self.default_state

        if isinstance(input, np.ndarray):
            data = {input_name: input}
            if state is not None:
                data[state_name] = state
        else:
            # Arrays in shallow copy will share memory, but the new data
            # dictionary will end up with additional keys.
            data = input.copy()

        # Rearrange and/or add axes if needed.
        for k, v in data.items():
            # TODO: This seems safest. Any Layer that expects a 1-dim input
            #       should be able to handle (T,1) pretty easily, and most
            #       Layers already expect 2-dim inputs (even if 1 channel).
            #       Ask Stephen if any good cases for not doing this.
            if data[k].ndim == 1:
                data[k] = data[k][..., np.newaxis]
            if (time_axis != 0) or (channel_axis != 1):
                # Move time_axis to axis=0, channel_axis to axis=1
                data[k] = np.moveaxis(v, [time_axis, channel_axis], [0, 1])

        layers = self.layers[:n]
        # Set first input if None
        all_inputs = [layer.input for layer in layers]
        if all_inputs[0] is None:
            all_inputs[0] = input_name
        # Set last output if None
        all_outputs = [layer.output for layer in layers]
        if all_outputs[-1] is None:
            all_outputs[-1] = output_name

        # Loop through transformations
        iter_zip = zip(layers, all_inputs, all_outputs)
        for layer, next_input, next_output in iter_zip:
            if next_input is None:
                # Use previous output
                data_in = data_out
                state_in = {}
                if (layer.state_name is not None):
                    state_in[layer.state_name] = data[state_name]
                data_out = layer.evaluate(*data_in, **state_in)
            elif isinstance(next_input, str):
                # Get data from one key
                data_in = data[next_input]
                data_out = layer.evaluate(data_in)
            elif isinstance(next_input, list):
                # Get data from all keys, in list order
                data_in = [data[k] for k in next_input]
                data_out = layer.evaluate(*data_in)
            elif isinstance(next_input, dict):
                # Map evaluate arguments to specific data keys
                data_in = {k: data[v] for k, v in next_input.items()}
                data_out = layer.evaluate(**data_in)
            else:
                raise TypeError(
                    f"Unrecognized type for {layer.name}.input:"
                    f"{type(next_input)}"
                    )

            # Cast output as list if only one array, for fewer checks
            if not isinstance(data_out, (list, tuple)):
                data_out = [data_out]

            if next_output is not None:
                if isinstance(next_output, str):
                    output_keys = [
                        f"{next_output}.{i}" if i != 0 else f"{next_output}"
                        for i in range(len(next_output))
                        ]
                elif isinstance(next_output, list):
                    output_keys = next_output
                else:
                    raise TypeError(
                        f"Unrecognized type for {layer.name}.output:"
                        f"{type(next_output)}"
                        )
                
                for k, v in zip(output_keys, data_out):
                    # Add singleton channel axis if needed.
                    if v.ndim == 1:
                        v = v[..., np.newaxis]
                    data[k] = v

        if undo_reorder:
            # Rearrange axes if needed (reverse of above).
            if (time_axis != 0) or (channel_axis != 1):
                for k, v in data.items():
                    # Move axis=0 to time_axis, axis=1 to channel_axis.
                    data[k] = np.moveaxis(v, [0, 1], [time_axis, channel_axis])

        if not return_full_data:
            data = data_out
        return data


    def predict(self, input, return_full_data=False, **eval_kwargs):
        """As `Model.evaluate`, but return only the last output by default."""
        return self.evaluate(input, return_full_data=return_full_data,
                             **eval_kwargs)

    def fit(self, input, target=None, target_name=None, backend=None,
            cost_function='mse', fitter_options=None, log_spacing=5,
            sanitize_data=False, **eval_kwargs):
        """Optimize model parameters to match `Model.predict(input)` to target.
        
        TODO: where do jackknife indices fit in? possibly use segmentor idea
              from old NEMS that never really got implemented, as an alternative
              to requiring jackknifing to be set up ahead of time.

              simplest solution: don't deal with it here at all. can have a
              separate method/function that just loops through calls to .fit
              and sets indices in between.

        TODO: want to add explicit support for multiple targets?
              E.g. fit to two types of data simultaneously.

        Parameters
        ----------
        input : np.ndarray or dict
            If ndarray, use this as the input to the first Layer. Otherwise,
            use keys specified by `input_name` or `Layer.input` to determine
            the first input.
        target : np.ndarray or None; optional.
            If ndarray, this will be the fitter's target data (i.e. try to
            match the model prediction to this). This option can only be used
            in conjunction with ndarray `input`. If other data is needed, use a
            dictionary input containing all data and specify `target_name` to
            indicate the key of the target data.
        target_name : str or None; optional.
            If str, and `target is None`, then `target_name` should be the key
            for the target data in `input`.
        backend : str or None; optional.
            Determines how Model will be fit.
            If None    : Refer to `Model.default_backend`.
            If 'scipy' : Use `scipy.optimize.minimize(method='L-BFGS-B')`.
            If 'tf'    : Use TensorFlow. Also aliased as 'tensorflow'.
            # TODO: any other options we want to support?
        cost_function : str or func; default='mse'
            Specifies which metric to use for computing error while fitting.
            If str  : Invoke `nems.metrics.get_metric(str)`.
            If func : Use this function to compute errors. Should accept two
                      array arguments and return float. 
        fitter_options : dict or None
            Keyword arguments to pass on to the fitter. For a list of valid
            options, see documentation for `scipy.optimize.minimize`
            and TODO: tensorflow.
        log_spacing : int; default=5.
            Log progress of fitter every `log_spacing` iterations.
            Note: this is the number of iterations, not the number of cost
            function evaluations (of which there may be many per iteration).
        sanitize_data : bool; default=False.
            If True, pass input through `Model.evaluate(undo_reorder=False)`
            one time prior to fitting, so that `np.moveaxis` and `newaxis`
            operations are not repeated on every evaluation while fitting.


        """
        if target is None:
            # Must specify either target or target_name
            target = input[target_name]
        if fitter_options is None:
            # TODO: maybe set Model.default_fitter_options for this instead?
            #       
            fitter_options = {}
        if backend is None:
            backend = self.default_backend
        if isinstance(cost_function, str):
            # Convert string reference to metric function
            cost_function = get_metric(cost_function)

        if sanitize_data:
            # Evaluate once to set data-ordering and dimension padding, so that
            # those operations are not repeated for each evaluation.
            # TODO: this is very kludgy and not comparmentalized well... work on a
            #       better option, or maybe just skip this altogether and inform
            #       users that they should either re-order prior to fitting or
            #       accept the minor performance hit.
            # TODO: After some rough testing, this doesn't seem to be worth it.
            #       About a 5% increase (or less) for fit duration on 3-dim
            #       random data. Leaving it here for now to do some more testing
            #       when larger models are implemented, but can likely be axed.
            data = self.evaluate(input, undo_reorder=False, **eval_kwargs)
            if isinstance(input, np.ndarray):
                key = eval_kwargs.get('input_name', self.default_input)
                input = data[key]
            else:
                input = data
            eval_kwargs['time_axis'] = 0
            eval_kwargs['channel_axis'] = 1

        if backend == 'scipy':
            # TODO: probably move this to a subroutine? will decide after
            #       sketching out more
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
                    prediction = self.predict(input, **eval_kwargs)
                    cost = cost_function(prediction, target)
                    print(f"iteration {log_info['iteration']},"
                          f" error is: {cost:.4f}")
                log_info['iteration'] += 1
            
            initial_parameters = self.get_parameter_vector(as_list=True)
            bounds = self.get_bounds_vector(none_for_inf=True)
            cost_args = (self, input, target, cost_function, eval_kwargs)
            fit_result = scipy.optimize.minimize(
                _scipy_cost_wrapper, initial_parameters, cost_args,
                bounds=bounds,
                method='L-BFGS-B', callback=_scipy_callback, **fitter_options
            )
            improved_parameters = fit_result.x
            self.set_parameter_vector(improved_parameters)

        elif (backend == 'tf') or (backend == 'tensorflow'):
            # TODO: similar to scipy, last step should set new model parameters
            #       in-place
            raise NotImplementedError


    def score(self, prediction, target):
        # TODO: this should point to an independent utility function, but
        #       placed here for convenience (and also to provide model defaults).
        pass

    def get_bounds_vector(self, none_for_inf=True):
        """Get all parameter bounds, formatted as a list of 2-tuples.

        Parameters
        ----------
        none_for_inf : bool
            If True, +/- np.inf is replaced with None
            (for scipy-compatible bounds).

        Returns
        -------
        model_bounds : list of 2-tuple

        See also
        --------
        nems.layers.base.Layer.get_bounds_vector

        """
        # collect all bounds, flatten the intermediate bounds lists
        bounds = [b for layer in self.layers for b in
                  layer.get_bounds_vector(none_for_inf=none_for_inf)]
        return bounds

    def get_parameter_vector(self, as_list=True):
        """Get all parameter values, formatted as a single vector.
        
        Parameters
        ----------
        as_list : bool
            If True, returns a list instead of ndarray
            (for scipy compatibility)

        Returns
        -------
        model_vector : list or ndarray

        See also
        --------
        nems.layers.base.Layer.get_parameter_vector

        """
        # collect all layer vectors
        vectors = []
        for layer in self.layers:
            vector = layer.get_parameter_vector(as_list=as_list)
            vectors.append(vector)
        # flatten list
        if as_list:
            model_vector = [v for vector in vectors for v in vector]
        else:
            model_vector = np.concatenate(vectors)
        
        return model_vector

    def set_parameter_vector(self, vector, ignore_checks=False):
        """Set all parameter values with a single vector.

        Parameters
        ----------
        vector : ndarray or list
            New parameter vector. Size must match the total number of flattened
            parameter values.
        ignore_checks : bool
            If True, set new values without checking size or bounds.
            (intended as a minor optimization for the scipy fitter).

        See also
        --------
        nems.layers.base.Layer.set_parameter_vector
        
        """
        first_index = 0
        for layer in self.layers:
            parameter_size = layer.parameters.size
            last_index = first_index + parameter_size
            layer.set_parameter_vector(vector[first_index:last_index],
                                       ignore_checks=ignore_checks)
            first_index = last_index

    def get_parameter_values(self, *layer_keys):
        """Get all parameter values, formatted as a dict.
        
        Parameters
        ----------
        layer_keys : N-tuple of strings
            Keys indicating which Layers to get parameter values for. If no keys
            are specified, get values for all layers.

        Returns
        -------
        all_values : dict

        See also
        --------
        nems.layers.base.Layer.get_parameter_values
        
        """
        if layer_keys == ():
            layer_keys = self._layers.keys()
        all_values = {}
        for k in layer_keys:
            values = self.layers[k].get_parameter_values(as_dict=True)
            all_values[k] = values
        return all_values

    def set_parameter_values(self, layer_dict):
        """Set new parameter values from key, value pairs.
        
        See also
        --------
        nems.layers.base.Layer.set_parameter_values

        """
        for k, v in layer_dict.items():
            self.layers[k].set_parameter_values(v)

    def sample_from_priors(self, inplace=True, as_vector=False):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, sampled values will be used to update each Parameter
            inplace. Otherwise, the sampled values will be returned without
            changing current values.
        as_vector : bool, default=False
            If True, return sampled values as a flattened vector instead of a
            list of arrays.

        Returns
        -------
        samples : ndarray or list of ndarray

        See also
        --------
        nems.layers.base.Layer.sample_from_priors

        """
        samples = [l.sample_from_priors(inplace=inplace, as_vector=as_vector)
                   for l in self.layers]
        if as_vector:
            samples = np.concatenate(samples)
        return samples

    def mean_of_priors(self, inplace=True, as_vector=False):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, mean values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, means
            will be returned without changing current values.
        as_vector : bool, default=False
            If True, return means as a flattened vector instead of a
            list of arrays.

        Returns
        -------
        means : ndarray or list of ndarray

        See also
        --------
        nems.layers.base.Phi.mean

        """
        means = [l.mean_of_priors(inplace=inplace, as_vector=as_vector)
                 for l in self.layers]
        if as_vector:
            means = np.concatenate(means)
        return means

    def set_index(self, index, new_index='initial'):
        """Change which set of parameter values is referenced.

        Intended for use with jackknifing or other procedures that fit multiple
        iterations of a single model. Rather than using many copies of a full
        Model object, each layer tracks copies of its parameter values.

        Parameters
        ----------
        i : int
            New index for parameter copies. If `i-1` exceeds the number of
            existing copies, then new copies will be added until `i` is a
            valid index.
        new_index : str or None, default='initial'
            Determines how new values are generated if `i` is out of range.
            If `'sample'`   : sample from priors.
            Elif `'mean'`   : mean of priors.
            Elif `'initial'`: initial value of each parameter.
            Elif `'copy'`   : copy of current values.
            Elif `None`     : raise IndexError instead of adding new vectors.

        See also
        --------
        nems.layers.base.Layer.set_index

        """
        for layer in self.layers:
            layer.set_index(index, new_index=new_index)

    def freeze_layers(self, *layer_keys):
        """Invoke `Layer.freeze_parameters()` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.freeze_parameters

        """
        for layer in self.layers.get(*layer_keys):
            layer.freeze_parameters()

    def unfreeze_layers(self, *layer_keys):
        """Invoke `Layer.unfreeze_parameters()` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.unfreeze_parameters

        """
        for layer in self.layers.get(*layer_keys):
            layer.unfreeze_parameters()

    def __repr__(self):
        # Get important args/kwargs and string-format as call to constructor.
        # (also attrs, TODO)
        args = []    # TODO  --  what should be here?
        kwargs = {}  # TODO  --  what should be here?
        args_string = ", ".join([f"{a}" for a in args])
        kwargs_string = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        attr_string = ""
        for attr in ['input', 'output', 'state', 'backend']:
            default = f'default_{attr}'
            self_attr = getattr(self, default)
            base_attr = getattr(Model, default)
            if self_attr != base_attr:
                # 7 max length, add spaces to keep values aligned
                space_gap = " "*(7-len(attr))  
                attr_string += f".{default}:  " + space_gap + f"{self_attr}\n"
        header = f"{type(self).__name__}({args_string}{kwargs_string})\n"
        tilde_break = "~"*64 + "\n"

        string = header
        string += tilde_break
        string += attr_string
        string += ".layers:\n\n"
        # Extra equal_breka above first layer, so that its heading looks the
        # same as subsequent layers.
        string += "="*32 + "\n"
        for i, layer in enumerate(self.layers):
            if i != 0:
                # Add blank line between layers if more than one
                string += '\n'
            string += layer.__repr__()
        string += "\n" + tilde_break

        return string

    @classmethod
    def from_keywords(cls, *keywords):
        """Construct a Model from a list of keywords or a model string.
        
        Parameters
        ----------
        keywords : N-tuple of strings
            If first string contains hyphens, it will be interpreted as a
            "model string" where each hyphen separates two keywords.

        Returns
        -------
        Model

        See also
        --------
        nems.layers.base.Layer.from_keyword
        nems.scripts.keywords
        
        """
        # Check for kw1-kw2-... (mode; string) format.
        # If so, split into list of keywords
        split = keywords[0].split('-')
        if len(split) > 1:
            keywords = split
        # Get Layer instances by invoking `Layer.from_keyword` through registry.
        layers = [keyword_lib[kw] for kw in keywords]
        return cls(layers=layers)

    # Add compatibility for saving to json
    def to_json(self):
        """Encode a Model as a dictionary.

        TODO: after specifying some built-in Models (e.g. subclasses), determine
              if `Model.<to/from>_json` need to be updated to support those.
              As long as they're just adding specific Layers the base versions
              should work, but not sure exactly how that's going to work yet.

        Returns
        -------
        data : dict

        See also
        --------
        `nems.tools.json`

        """
        # TODO: any other metadata?
        data = {
            'layers': list(self._layers.values()),
            'name': self.name,
            'defaults': {
                'input': self.default_input, 'output': self.default_output, 
                'state': self.default_state, 'backend': self.default_backend
                }
            }
        
        return data

    @classmethod
    def from_json(cls, json):
        """Decode a Model from a dictionary.

        Returns
        -------
        Model

        See also
        --------
        `nems.tools.json`

        """
        # TODO: any other metadata?
        model = cls(layers=json['layers'], name=json['name'])
        for k, v in json['defaults'].items():
            setattr(model, f"default_{k}", v)
        return model

    # Placed this code next to `_LayerDict` for easier cross-checking of code
    def __getitem__(self, key):
        return self.layers[key]

    def __len__(self):
        """Define `len(Model) = <number of layers in the Model>`."""
        return len(self.layers)

    def __iter__(self):
        """Reroute iteration functions to Model.layers.__iter__."""
        return self.layers.__iter__()


class _LayerDict:
    """Wrapper for Layer._layers to enable fancy-indexed gets.

    Supports: integer and string indexing, multi-indexing (one type at a time).
    Note that index assignment is not supported. To change a Model's Layers,
    use `Model.add_layers`, `Model.remove_layers`, etc.

    Examples
    --------
    >>> layers = _LayerDict({'one': 1, 'two': 2, 'three': 3})
    >>> layers
    {'one': 1, 'two': 2, 'three': 3}
    >>> layers[0]
    1
    >>> layers['one']
    1
    >>> layers['one', 'three']
    1, 3
    >>> layers['one', 0]
    KeyError: 0
    >>> layers['two'] = 22
    TypeError: '_LayerDict' object does not support item assignment

    """
    def __init__(self, _dict):
        self._dict = _dict
        self._values = list(_dict.values())

    def _container_for(self, key):
        if isinstance(key, int):
            container = self._values
        else:
            # Should be a string key
            container = self._dict
        return container

    def __getitem__(self, keys):
        if isinstance(keys, (str, int, slice)):
            keys = tuple([keys])
        value = self.get(*keys, default=None)
        if value is None:
            raise KeyError(keys)
        return value

    def get(self, *keys, default=None):
        # no keys, get all layers
        if keys == ():
            layers = self._values
        elif isinstance(keys[0], slice):
            layers = self._values[keys[0]]
        else:
            container = self._container_for(keys[0])
            layers = []
            for key in keys:
                try:
                    layer = container[key]
                except (IndexError, KeyError):
                    layer = default
                layers.append(layer)
        
        # Tuple wrapper (to replace list) is just for output consistency with
        # __getitem__, should be no practical difference in most cases.
        # Unwrap instead if it's a singleton list.
        if isinstance(layers, list):
            if len(layers) == 1:
                layers = layers[0]
            else:
                layers = tuple(layers)

        return layers

    def __iter__(self):
        """Iterate over Layers (not keys)."""
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._values

    def __repr__(self):
        return self._dict.__repr__()
