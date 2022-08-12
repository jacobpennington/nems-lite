import copy
import textwrap
import itertools

import numpy as np
import scipy.optimize

from nems.registry import keyword_lib
from nems.metrics import get_metric
from nems.visualization import plot_model
# Temporarily import layers to make sure they're registered in keyword_lib
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
        self.name = name if name is not None else 'UnnamedModel'

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
            key = layer.name
            i = 0
            while key in self._layers:
                # Avoid name clashes
                key = f'{layer.name}{i}'
                i += 1
            self._layers[key] = layer
            # Also update `Layer.name` so that there's no mismatch between
            # a Layer's name and its key in the Model.
            layer._name = key

    def get_layer_index(self, name):
        """Get integer index for Layer with `.name == name`."""
        return list(self.layers.keys()).index(name)

    def get_layer(self, key):
        """Get one Layer. Key can be int or string (`Layer.name`)."""
        return self.layers[key]

    def insert_layer(self, index, name=None):
        """TODO: add layer at specific integer index."""
        raise NotImplementedError

    def evaluate(self, input, state=None, input_name=None, state_name=None,
                 output_name=None, n=None, time_axis=0, channel_axis=1,
                 undo_reorder=True, return_full_data=True,
                 use_existing_maps=False, batch_size=None):
        """Transform input(s) by invoking `Layer.evaluate` for each Layer.

        Evaluation encapsulates three steps:
            1) Package data and metadata in a single container, possibly after
               some reformatting depending on which options are specified.
            2) Loop over `Model.layers`, invoking `Layer._evaluate` to
               transform the data.
            3) Clean up no-longer-needed data, and possibly undo re-formatting.
        See `Model.generate_layer_data` (and subroutines) for implementation.

        During the evaluation process, `input` (and `state` if provided) will
        be packaged into a `data` dictionary, with the following structure:
            array_name : ndarray. 
                If `input` is a dict, each array in `input` will be
                shallow-copied to `data` with the same key. Otherwise, arrays
                `input` and `state` will be added to `data` with keys
                `input_name` and `state_name` respectively (or the relevant
                `Model.default_<name>` attribute).
            _last_output : ndarray, list, or None
                Return value of the most recently evaluated Layer. This key is
                removed after evaluation is complete.
            _state_name : str
                Specifies key in `data` (either `state_name` or model default)
                that should be included as an additional input for Layers that
                have not specified `Layer.input` but have defined 
                `Layer.state_arg`. This key is removed after evaluation.

        Parameters
        ----------
        input : ndarray, list of ndarray, or dict
            If ndarray or list, use this as the input to the first Layer.
            Otherwise, use keys specified by `input_name` or `Layer.input` to
            determine the first input.
        state : ndarray or list of ndarray; optional.
            Add this to intermediate data dictionary. This option can
            only be used in conjunction with an array or list `input`. If other
            data is needed, use a dictionary input containing all data.
        input_name : str; optional.
            Specifies which array should be provided as input to the
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
        return_full_data : bool; default=True
            If True, return a dictionary containing all input data and all
            uniquely-keyed Layer outputs.
        use_existing_maps : bool; default=False.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        batch_size : int; optional.
            TODO, still WIP

        Returns
        -------
        data : ndarray, list, or dict
            Type depends on `return_full_data` option and the return type of
            the `evaluate` method of the final layer in the model.

        See also
        --------
        nems.layers.base.Layer._evaluate
        Model.generate_layer_data

        Warnings
        --------
        Since arrays in `data` share memory with `input`, modifying shared
        arrays in-place is strongly discouraged.
        
        """
        if batch_size is not None:
            current_args = locals()
            current_args['batch_size'] = None
            current_args.pop('self')  # otherwise self will be passed twice
            # TODO: split data into batches along first axis, assumes
            #       user has already separated data by trial or something else
            #       and has shape (S, T, N) instead of (T,N) where S represents
            #       number of samples/trials/whatever.
            # NOTE: per SVD request, also support passing in list directly
            #       (e.g. if data is already a list, assume it has been split
            #        and that each array in the list is one batch).
            batches = []
            for batch in batches:
                batch_data = self.evaluate(**current_args)
                # TODO: then what? I guess anything else would be handled
                #       by the fitter.
                # TODO: doing this loop here is a bit wasteful, since data_map
                #       should be the same for every batch. If some of those
                #       details were moved into Layers, that would help.
                #       Ex: on first loop, Model gets Layer.input_map from
                #       each Layer, evaluates them, combines them, stores that
                #       as the map for every batch. Then eval just has to
                #       pass in new data each time.

        current_args = locals()
        current_args.pop('self')
        data_generator = self.generate_layer_data(**current_args)
        if n is not None: n -= 1
        layer_data = self.get_layer_data(data_generator, n, n)[0]

        if not return_full_data:
            data = layer_data['out']  # output of final Layer._evaluate
        else:
            data = layer_data['data']
        return data

    def _initialize_data(self, input, state=None, input_name=None,
                         state_name=None, time_axis=0, channel_axis=1,
                         **eval_kwargs):
        """Package `input` and `state` into a `data` dictionary.
        
        Internal for `Model.generate_layer_data`. See `Model.evaluate` for
        parameter documentation.

        Returns
        -------
        data : dict
        
        """

        if input_name is None: input_name = self.default_input
        if state_name is None: state_name = self.default_state

        # Initialize `data` dictionary.
        if isinstance(input, (np.ndarray, list)):
            data = {input_name: input}
            if state is not None:
                data[state_name] = state
        else:
            # Arrays in shallow copy will share memory, but the new data
            # dictionary will end up with additional keys after evaluation.
            data = input.copy()

        # Rearrange and/or add axes if needed.
        for k, v in data.items():
            if data[k].ndim == 1:
                # TODO: raise warning that most Layers (and some optional code
                #       in Model.evaluate) assume 2 dimensions. Don't actually
                #       pad axis here b/c that will duplicate memory, i.e.
                #       data[k] will point to a new copy instead of the original
                #       array.
                pass
            if (time_axis != 0) or (channel_axis != 1):
                # Move time_axis to axis=0, channel_axis to axis=1
                data[k] = np.moveaxis(v, [time_axis, channel_axis], [0, 1])

        reserved_keys = ['_last_output', '_state_name']
        for rk in reserved_keys:
            if rk in data:
                # TODO: raise warning that this is a reserved key, data will
                #       be overwritten in evaluate loop
                pass

        # First Layer will use this key if Layer.input is None.
        data['_input_name'] = input_name
        # Layers with a `state_arg` that don't specify any inputs will get
        # the array at this key, if it exists, in addition to last_output.
        data['_state_name'] = state_name
        # No outputs yet
        data['_last_output'] = None

        return data


    def _evaluate_layer(self, layer, data):
        """Evaluates one Layer. Internal for `Model.generate_layer_data`.
        
        Returns
        -------
        args : list of ndarray
            Positional arguments for `Layer.evaluate`.
        kwargs : dict of ndarray
            Keyword arguments for `Layer.evaluate`.
        output : ndarray or list of ndarray
            Return value of `Layer.evaluate(*args, **kwargs)`.
        
        """
        
        # Get input & output arrays
        args, kwargs, output = layer._evaluate(data)

        # Save output (or don't) based on Layer.DataMap.
        # data_keys is always a list, but output might be a list or one array.
        data_keys = layer.data_map.out
        if isinstance(output, (list, tuple)):
            data.update({k: v for k, v in zip(data_keys, output)
                        if k is not None})
        elif data_keys[0] is not None:
            data[data_keys[0]] = output
        # Always save output to _last_output, for use by subsequent Layers.
        data['_last_output'] = output

        return args, kwargs, output


    def _finalize_data(self, final_layer, data, output_name=None,
                       undo_reorder=True, time_axis=0, channel_axis=1,
                       **eval_kwargs):
        """Remove metadata and undo re-formatting if needed.

        Internal for `Model.generate_layer_data`. See `Model.evaluate` for
        parameter documentation.

        Returns
        -------
        None
        
        """

        if output_name is None: output_name = self.default_output
        # Remove metadata, no longer needed.
        _ = data.pop('_input_name')
        _ = data.pop('_state_name')

        # Re-name last output if keys not specified by Layer
        final_output = data.pop('_last_output')
        if final_layer.output is None:
            data[output_name] = final_output
            # Re-map `data_map.out` so that it reflects `output_name`.
            final_layer.data_map.map_outputs(final_output, output_name)

        if undo_reorder:
            # Rearrange axes if needed (reverse of above).
            if (time_axis != 0) or (channel_axis != 1):
                for k, v in data.items():
                    # Move axis=0 to time_axis, axis=1 to channel_axis.
                    data[k] = np.moveaxis(v, [0, 1], [time_axis, channel_axis])


    def generate_layer_data(self, input, copy_data=False,
                            use_existing_maps=False, **eval_kwargs):
        """Generate input and output arrays for each Layer in Model.
        
        This method serves as the core loop for `Model.evaluate`, but is exposed
        separately for use in plotting, debugging, etc. The loop is implemented
        as a generator to reduce memory overhead when only one Layer at a time
        is needed.

        Parameters
        ----------
        input : dict or ndarray
            See `Model.evaluate`.
        copy_data : bool; default=False.
            If True, a deep copy of data will be stored in `layer_data['data']`
            after each `Layer._evaluate`. This will be very memory intensive
            for large data, and is generally not recommended.
        use_existing_maps : bool; default=False.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        eval_kwargs : dict
            Additional keyword arguments for `Model._initialize_data` and
            `Model._finalize_data`. See `Model.evaluate` for documentation.

        Yields
        ------
        layer_data : dict
            `layer_data` has the following structure: {
                'index': int, index of Layer within Model.
                'layer': str, Layer.name.
                'args': list of ndarray, positional arguments
                    for `Layer._evaluate`.
                'kwargs': dict of ndarray, keyword arguments
                    for `Layer._evaluate`
                'out': ndarray or list of ndarray, return value of
                    `Layer._evaluate(*args, **kwargs)`.
                'data' : dict
                    See `Model.evaluate` for details.
                }

        Warnings
        --------
        layer_data['data'], is a reference to a data structure that is
        iteratively updated in-place during evaluation. Modifying this
        structure in-place is strongly discouraged, as it can violate the state
        expectations of not-yet-evaluated Layers. To make modifications safely,
        use `copy_data=True`.

        See also
        --------
        Model.get_layer_data
        Model.print_layer_data

        Examples
        --------
        Get a list of all outputs in memory simultaneously:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> data_list = [d['out'] for d, _ in generator]

        Get positional arguments for the first layer:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> args = next(generator)['args']

        """

        data = self._initialize_data(input, **eval_kwargs)

        max_n = len(self.layers)
        for n, layer in enumerate(self.layers):
            if not use_existing_maps:
                layer.reset_map()
            a, k, o = self._evaluate_layer(layer, data)
            layer_data = {
                'index': n, 'layer': layer.name,
                'args': a, 'kwargs': k, 'out': o
                }

            if n < (max_n - 1):
                if copy_data:
                    layer_data['data'] = copy.deepcopy(data)
                else:
                    layer_data['data'] = data
                yield layer_data

        # On final layer, only update data after evaluation
        self._finalize_data(layer, data, last_output=o, **eval_kwargs)
        if copy_data:
            layer_data['data'] = copy.deepcopy(data)
        else:
            layer_data['data'] = data

        yield layer_data


    # TODO: maybe remove the data_generator arg and just have this wrap
    #       generate_layer_data? 
    def get_layer_data(self, data_generator, first_index=None, last_index=None):
        """Return data for layers between specified indices (inclusive).
        
        Parameters
        ----------
        data_generator : generator
            Return value of `Model.generate_layer_data`.
        first_index : int; optional.
            Index within Model of the first Layer to get data for.
        last_index : int; optional.
            Index within Model of the last Layer to get data for.

        Returns
        -------
        list of (dict, dict)
            See `Model.generate_layer_data`.

        Examples
        --------
        Get a list of all inputs & outputs in memory simultaneously:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> data_list = get_layer_data(generator)

        Get the keyword arguments for the 3rd Layer:
        >>> generator = generate_layer_data(input, **eval_kwargs)
        >>> kwargs3 = get_layer_data(generator, 3, 3)['kwargs']
        
        """
        if last_index is not None: last_index += 1
        subset = itertools.islice(data_generator, first_index, last_index)
        return [d for d in subset]

    def print_layer_data(self, input, max_char=79, max_array_length=20,
                         show_full_data=False, **eval_kwargs):
        """Pretty-print the return value of `Model.generate_layer_data`.

        Parameters
        ----------
        input : ndarray or dict
            See `Model.evaluate`.
        max_char : int; default=79.
            Maximum number of characters to display on each line.
            TODO: separators currently ignore this.
        max_array_length : int; default=20.
            Show truncated arrays if they contain more than this many entries.
            Equivalent to `np.set_printoptions(threshold=max_array_length)`,
            but the previous threshold will be reset after printing.
            TODO: add precision option?
        show_full_data : bool; default=False.
            If True print the entire `data` dictionary for each Layer.

        TODO: option to return string instead of printing?
        
        """
        def wrap(k, v):
            return textwrap.fill(f'{k}: {str(v)}', max_char) + '\n' + '-'*16

        current_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=max_array_length)

        for d in self.generate_layer_data(input, **eval_kwargs):
            _data = d.pop('data')

            # Input/output info
            print('_'*36 + f'in/out:' + '_'*36)
            for k, v in d.items():
                if isinstance(v, list):
                    print(f'{k}:')
                    i = 0
                    for val in v:
                        print(wrap(i, val))
                        i += 1
                    if i == 0:
                        print('-'*16)
                elif isinstance(v, dict):
                    print(f'{k}:')
                    j = 0
                    for key, value in v.items():
                        print(wrap(key, value))
                        j += 1
                    if j == 0:
                        print('-'*16)
                else:
                    print(wrap(k, v))
            print('\u203E'*79)

            if show_full_data:
                # Data dictionary
                print('_'*38 + 'data' + '_'*37)
                for k, v in _data.items():
                    print(wrap(k, v))
                print('\u203E'*79 + '\n\n')

        np.set_printoptions(threshold=current_threshold)


    def predict(self, input, return_full_data=False, **eval_kwargs):
        """As `Model.evaluate`, but return only the last output by default."""
        return self.evaluate(input, return_full_data=return_full_data,
                             **eval_kwargs)

    def fit(self, input, target=None, target_name=None, backend=None,
            cost_function='mse', fitter_options=None, log_spacing=5,
            undo_reorder=False, use_existing_maps=True, **eval_kwargs):
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
        target : np.ndarray or list of np.ndarray; optional.
            TODO: support list
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
            TODO: any other options we want to support?
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
        undo_reorder : bool; default=False.
            If True, and data axes were re-ordered, revert to the original
            ordering after evaluating. Set False to return the re-ordered data.
            This is disabled by default during fitting to reduce overhead.
        use_existing_maps : bool; default=True.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        eval_kwargs : dict
            Other keyword arguments will be supplied to `Model.evaluate`.

        """

        # Must specify either target or target_name
        if target is None: target = input[target_name]
        # TODO: maybe set Model.default_fitter_options for this instead?
        if fitter_options is None: fitter_options = {}
        if backend is None: backend = self.default_backend
        if isinstance(cost_function, str):
            # Convert string reference to metric function
            cost_function = get_metric(cost_function)

        # Set data-ordering prior to fitting so that those operations are
        # not repeated for each evaluation.
        data = self._initialize_data(
            input, undo_reorder=undo_reorder, 
            use_existing_maps=use_existing_maps,
            **eval_kwargs
            )
        if not undo_reorder:
            # Reset these to default values, otherwise arrays will
            # be improperly reordered if these kwargs had non-default values.
            _ = eval_kwargs.pop('time_axis', None)
            _ = eval_kwargs.pop('channel_axis', None)


        # TODO: prediction & target currently assumed to be arrays, but they can
        #       also be lists. Need to do some extra checks to align those
        #       correctly for the cost function. Only working so far by accident,
        #       will break if more than one array in prediction list.
        #
        #       Need to decide what to do in this case: iterate through the lists
        #       and take the average? (allows different lengths)
        #       Concatenate and compute all at once? (all same lengths)
        #       Other? (e.g. multiple predicts vs one target, take average)
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
                          f" error is: {cost:.8f}...")
                log_info['iteration'] += 1
            
            initial_parameters = self.get_parameter_vector(as_list=True)
            bounds = self.get_bounds_vector(none_for_inf=True)
            cost_args = (self, input, target, cost_function, eval_kwargs)
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

    def plot(self, input, **kwargs):
        """Alias for `nems.visualization.model.plot_model`.
        
        By default, the result of each `Layer.evaluate` will be shown.
        
        """
        return plot_model(self, input, **kwargs)

    # added .summary() to mirror tensorflow models, for intuitive comparisons.
    def summary(self):
        """Prints long-form model description (alias for `print(Model)`)."""
        print(self)

    def __str__(self):
        header, tilde_break, attr_string = self._repr_helper()
        string = header
        string += tilde_break
        string += attr_string
        string += ".layers:\n\n"
        # Extra equal_break above first layer, so that its heading looks the
        # same as subsequent layers.
        string += "="*32 + "\n"
        for i, layer in enumerate(self.layers):
            if i != 0:
                # Add blank line between layers if more than one
                string += '\n'
            string += str(layer)
        string += "\n" + tilde_break

        return string

    def __repr__(self):
        header, tilde_break, _ = self._repr_helper()
        string = header
        string += tilde_break
        for i, layer in enumerate(self.layers):
            if i != 0:
                string += "\n\n" # break between layers
            string += layer.__repr__()
        string += "\n" + tilde_break

        return string

    def _repr_helper(self):
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

        return header, tilde_break, attr_string


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

    def __getitem__(self, keys):
        # tuple([]) wrapper to enable multiple keys with Model.layers[] syntax.
        if isinstance(keys, (str, int, slice)):
            keys = tuple([keys])
        value = self.get(*keys, default=None)

        # Raise KeyError if any keys returned None
        if value is None:
            raise KeyError(keys)
        elif isinstance(value, list):
            none_in_list = [x is None for x in value]
            if np.any(none_in_list):
                raise KeyError(keys)

        return value

    def get(self, *keys, default=None):
        # no keys, get all layers
        if keys == ():
            layers = self._values
        elif isinstance(keys[0], slice):
            layers = self._values[keys[0]]
        else:
            # Require all keys str or all int, mixing not allowed.
            # This is meant to discourage hard-to-read code.
            if isinstance(keys[0], int):
                container = self._values
            else:
                container = self._dict
            layers = []
            for key in keys:
                try:
                    layer = container[key]
                except (IndexError, KeyError):
                    layer = default
                layers.append(layer)
        
        # List wrapper (to replace tuple) is just for output consistency should
        # be no practical difference in most cases.
        # Unwrap instead if it's a singleton list, *unless* keys was slice.
        if isinstance(layers, (tuple, list)):
            if (len(layers) == 1) and not isinstance(keys[0], slice):
                layers = layers[0]
            elif isinstance(layers, tuple):
                layers = list(layers)

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


# TODO: may not end up doing this, but sketching out the idea here.
#       Essentially a watered-down Recording that handles data sanitation
#       but otherwise acts as a dict, no epochs/splitting/etc. Possibly
#       to/from json but would just be a wrapper for saving the dict, goal is
#       for the class to be stateless except for ._data.
class DataSet(dict):
    def __init__(self, input, state=None, target=None, input_name=None,
                 state_name=None, output_name=None, target_name=None,
                 time_axis=0, channel_axis=1, undo_reorder=True):
        # TODO: may not actually need all these kwargs, or may not need to
        #       save them all as attrs
        self.initialize_data(input, state, target)  # other kwargs too

    def initialize_data(self, input, state, target):  # other kwargs too
        # TODO: do stuff from Model._initialize_data to generate data
        # TODO: 3 options for parsing default names:
        #       1) do it in the model before creating DataSet, make input and
        #          output names required.
        #       2) do it here with reference to Model
        #          (breaks statelessness, Model.default_<name> could change.
        #           also breaks compartmentalization)
        #       3) do it here with extra kwargs for defaults.
        #       4) do it here and move defaults to be DataSet class attrs
        #          (less convenient for users)
        data = input
        super().__init__(**data)  # enables __getitem__, __setitem__, etc.

    def finalize_data(self):
        # TODO: port Model._finalize_data() here.
        pass

    def passthrough_data_operations(self):
        # TODO: (not a real method name) convenience wrapper for
        #       `for array in data, get fn(array)`
        #       and/or `for array in data, set array = fn(array)` (i.e. inplace)
        #       For example, to apply the same jacknife indices to all arrays
        pass

    def other_sanitation(self):
        # TODO: (not a real method name) what else would be useful?
        pass

    
    