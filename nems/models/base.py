import copy
import textwrap
import itertools

import numpy as np
import scipy.optimize

from nems.registry import keyword_lib
from nems.metrics import get_metric
from nems.backends import SciPyBackend, TensorFlowBackend
from nems.visualization import plot_model
# Temporarily import layers to make sure they're registered in keyword_lib
import nems.layers  
del nems.layers


class Model:
    """TODO: docstring."""

    def __init__(self, layers=None, name=None):
        """TODO: docstring"""
        self._layers = {}  #  layer.name : layer obj, increment on clashes
        if layers is not None:
            self.add_layers(*layers)
        self.name = name if name is not None else 'UnnamedModel'
        # TODO: store optional metadata from kwarg
        # TODO: save metadata to json (and load)
        self.meta = {}

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
                 output_name=None, n=None, return_full_data=True,
                 skip_initialization=False, use_existing_maps=False,
                 batch_size=0, permute_batches=False):
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
            If None : use `Layer.input` of first layer if specified.
            If str  : a single input array at key `input_name`.
        state_name : str; optional.
            If str, and `state is not None`, then `state_name` will be the key
            for `state`.
        output_name : str; optional.
            Specifies name(s) for array output the final layer if
            `Layer.output is None`.
            If str  : Use this name for each of the Layer's outputs
                      (incremented if multiple).
        n : int; optional.
            Evaluate the first `n` Layers (all by defualt).
        return_full_data : bool; default=True
            If True, return a dictionary containing all input data and all
            uniquely-keyed Layer outputs.
        skip_initialization : bool; default=False
            TODO, still WIP
            If True, don't package input, state etc. into a DataSet.
            Instead, `input` should already be a DataSet object.
        use_existing_maps : bool; default=False.
            If True, existing DataMaps will be used rather than generating a
            new one for each Layer. This is disabled by default for direct
            evaluation, but enabled by default when called from `Model.fit` to
            eliminate unnecessary overhead for repeated evaluations of the same
            data.
        batch_size : int; default=0.
            TODO, still WIP
            -1: No sample dimension, prepend one and use 1 batch of size 1.
            None: Use 1 batch which is the entire sample dimension.
        permute_batches : bool; default=False.
            TODO, still WIP
            If True, randomly shuffle batches prior to evaluation. Typically
            used during fitting, to shuffle between epochs.

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

        if not skip_initialization:
            # Package arrays and/or dicts in a DataSet
            data = DataSet(
                input=input, state=state, input_name=input_name,
                state_name=state_name, output_name=output_name,
            )
        else:
            # Input should already be a properly formatted DataSet
            # (for example, passed by Model.fit)
            data = input
        if batch_size == 0:
            data = data.prepend_samples()
            _batch_size = 1
        else:
            _batch_size = batch_size

        batch_out = []
        batches = data.as_batches(_batch_size, permute=permute_batches)
        for batch in batches:
            samples = batch.as_samples()
            sample_out = []
            for sample in samples:
                data_generator = self.generate_layer_data(
                    sample, use_existing_maps=use_existing_maps,
                    skip_initialization=True
                )
                if n is not None: n -= 1
                else: n = len(self.layers)-1
                # Get data for the final layer only, to reduce memory use.
                layer_data = self.get_layer_data(data_generator, n, n)[-1]
                sample_out.append(layer_data['data'].prepend_samples())
            batch_out.extend(sample_out)
        all_outputs = DataSet.concatenate_sample_outputs(batch_out)
        # Inputs (and any targets) should not have changed
        data.outputs = all_outputs
        if batch_size == 0:
            # Remove prepended sample dimension
            data = data.squeeze_samples()

        if not return_full_data:
            out = data.outputs
            if len(out) == 1:
                out = list(out.values())[0]
        else:
            out = data.as_dict()

        return out

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
        data.save_output(data_keys, output)

        return args, kwargs, output


    # TODO: possibly move this method and any related subroutines to a separate
    #       module (inside a new `base` directory), with simple wrappers in
    #       Model as the public-facing API.
    def generate_layer_data(self, input, copy_data=False,
                            use_existing_maps=False, skip_initialization=False,
                            **eval_kwargs):
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
        skip_initialization : bool; default=False.
            If True, don't create a new DataSet. Input should already be a
            properly formatted DataSet.
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

        # TODO: don't do this here, need to refactor to expect DataSet
        #       as input
        if not skip_initialization:
            data = DataSet(input, **eval_kwargs)
        else:
            data = input

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
        #self._finalize_data(layer, data, last_output=o, **eval_kwargs)
        data.finalize_data(final_layer=layer)
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

    def get_data_maps(self):
        """Get a dictionary of {Layer.name: Layer.DataMap} for all Layers.

        Similar to `Model.layers`, this dictionary is wrapped so that indexing
        with integers, slices, or multiple keys is also possible.
        
        Returns
        -------
        dict

        See also
        --------
        nems.layers.base.map.DataMap
        
        """
        return _LayerDict({layer.name: layer.data_map for layer in self.layers})

    def predict(self, input, return_full_data=False, **eval_kwargs):
        """As `Model.evaluate`, but return only the last output by default.
        
        TODO: This only works for models where the final layer produces the
              only output. Need to think about how to make it work for models
              that produce multiple outputs at different stages.

              Rough idea: return all arrays from data that were not present
              in input.
        
        """
        return self.evaluate(input, return_full_data=return_full_data,
                             **eval_kwargs)


    # TODO: Move the backend implementations and any related subroutines to a
    #       separate module (inside a new `base` directory), with simple
    #       wrappers in Model as the public-facing API.
    def fit(self, input, target=None, target_name=None, backend='scipy',
            cost_function='mse', fitter_options=None, backend_options=None,
            **eval_kwargs):
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
        backend : str; default='scipy'.
            Determines how Model will be fit.
            If 'scipy' : Use `scipy.optimize.minimize(method='L-BFGS-B')`.
            If 'tf'    : Use TensorFlow. Also aliased as 'tensorflow'.
            TODO: any other options we want to support?
        cost_function : str or func; default='mse'
            Specifies which metric to use for computing error while fitting.
            If str  : Invoke `nems.metrics.get_metric(str)`.
            If func : Use this function to compute errors. Should accept two
                      array arguments and return float. 
        fitter_options : dict; optional.
            Keyword arguments to pass on to the fitter. For a list of valid
            options, see documentation for `scipy.optimize.minimize`
            and TODO: tensorflow.
        backend_options : dict; optional.
            Keyword arguments to pass to the Backend constructor.
        eval_kwargs : dict
            Keyword arguments to supply to `Model.evaluate`.

        """

        if fitter_options is None: fitter_options = {}
        if backend_options is None: backend_options = {}
        if isinstance(cost_function, str):
            # Convert string reference to metric function
            cost_function = get_metric(cost_function)

        # TODO: Answer: enforce always having the batch/sample dimension, but
        #       expect data w/o it by default. I.e. prepend a singleton dim
        #       unless user specifies batch=something

        # Evaluate once prior to fetching backend, to ensure all DataMaps are
        # up to date and include outputs.
        _ = self.evaluate(
            input, use_existing_maps=False, **eval_kwargs
            )

        # TODO: replace with DataSet method?
        input = self._initialize_data(input, **eval_kwargs)
        # Move target out of input
        if target is None: target = input.pop(target_name, None)
        # Remove meta keys added by initialize
        # TODO: After migrating to DataSet class, set this as attributes instead
        #       so that this isn't an issue.
        for k, v in input.items():
            if isinstance(v, str):
                _ = input.pop(k)

        # TODO: prediction & target currently assumed to be arrays, but they can
        #       also be lists. Need to do some extra checks to align those
        #       correctly for the cost function. Only working so far by accident,
        #       will break if more than one array in prediction list.
        #
        #       Need to decide what to do in this case: iterate through the lists
        #       and take the average? (allows different lengths)
        #       Concatenate and compute all at once? (all same lengths)
        #       Other? (e.g. multiple predicts vs one target, take average)
        #
        #       wish list:  matching lists (compare, add/average on concat.)
        #                   matching arrays
        #                   one array -> list
        #                   list -> one array
        #                   (roughly in order of priority)
        #       *also think about best way to compare cost for multiple neurons etc

        # TODO: convert this to something like
        #       backend = _lookup_backend(backend)
        #       backend.fit(...)
        #       return backend
        #       (i.e. no if/else loop, implement elsewhere)

        # TODO: return a copy of some kind instead of updating model parameters
        #       in-place. Should be as simple as replacing self with self.copy()
        #       (after implementing a .copy method).
        if backend == 'scipy':
            scipy_backend = SciPyBackend(
                self, input, eval_kwargs=eval_kwargs, **backend_options
                )
            scipy_backend.fit(
                input, target, eval_kwargs=eval_kwargs, **fitter_options
                )
            return scipy_backend

        elif (backend == 'tf') or (backend == 'tensorflow'):
            tf_backend = TensorFlowBackend(
                self, input, eval_kwargs=eval_kwargs, **backend_options
                )
            tf_backend.fit(
                input, target, eval_kwargs=eval_kwargs, **fitter_options
                )
            return tf_backend

        else:
            raise NotImplementedError(f"Unrecognized backend: {backend}.")
            

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
        header, tilde_break  = self._repr_helper()
        string = header
        string += tilde_break
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
        header, tilde_break = self._repr_helper()
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
        header = f"{type(self).__name__}({args_string}{kwargs_string})\n"
        tilde_break = "~"*64 + "\n"

        return header, tilde_break


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


# TODO: update docs elsewhere to list correct type for `data`.
# TODO: revisit assumptions about data type. I think we need to just only
#       allow arrays in the data dictionaries to make things simpler. For lists,
#       store them at separate keys and use Layer.input to supply them as a list
#       if needed. Pretty sure Layer.output already gets treated like this anyway,
#       I just sort of lost track.
class DataSet:
    default_input = 'input'
    default_output = 'output'
    default_target = 'target'
    default_state = 'state'

    def __init__(self, input, state=None, target=None, dtype=np.float64,
                 input_name=None, state_name=None, output_name=None,
                 target_name=None):
        """TODO: docs."""
        # Set self.<attr>_name to default if <attr>_name is None, otherwise
        # save self.<attr>_name.
        names = zip(['input', 'state', 'output', 'target'],
                    [input_name, state_name, output_name, target_name])
        for attr, name in names:
            if name is None: name = getattr(self, f'default_{attr}')
            setattr(self, f'{attr}_name', name)
        self.dtype = dtype
        # TODO: how to use dtype
        self.initialize_data(input, state, target)  # other kwargs too

    def initialize_data(self, input, state, target, **eval_kwargs):
        """TODO: docs"""
        # NOTE: additional kwargs ignored for convenience, since initialize
        #       and finalize are both related to Model.evaluate.

        # Initialize inputs
        if isinstance(input, (np.ndarray, list)):
            input_dict = {self.input_name: input}
            if state is not None:
                input_dict[self.state_name] = state
        else:
            # Arrays in shallow copy will share memory, but the new data
            # dictionary will end up with additional keys after evaluation.
            input_dict = input.copy()

        # Initialize outputs
        output_dict = {}

        # Initialize targets
        if target is None:
            target_dict = {}
        elif isinstance(target, (np.ndarray, list)):
            target_dict = {self.target_name: target}
        else:
            target_dict = target.copy()

        self.inputs = input_dict
        self.outputs = output_dict
        self.targets = target_dict

    def save_output(self, keys, output):
        """TODO: docs"""
        # data_keys is always a list, but output might be a list or one array.
        if isinstance(output, (list, tuple)):
            self.outputs.update({k: v for k, v in zip(keys, output)
                        if k is not None})
        elif keys[0] is not None:
            self.outputs[keys[0]] = output
        # Always save output to _last_output for use by Model.evaluate
        self.outputs['_last_output'] = output

    def finalize_data(self, final_layer, **eval_kwargs):
        """TODO: docs"""
        # NOTE: additional kwargs ignored for convenience, since initialize
        #       and finalize are both related to Model.evaluate.
        # Re-name last output if keys not specified by Layer
        final_output = self.outputs.pop('_last_output')
        if final_layer.output is None:
            self.outputs[self.output_name] = final_output
            # Re-map `data_map.out` so that it reflects `output_name`.
            final_layer.data_map.map_outputs(final_output, self.output_name)

    def as_broadcasted_samples(self):
        """TODO: docs"""
        # TODO: Iterate through all inputs, outputs, and targets and
        #       broadcast them against each other (on first dimension only)
        #       so that, for example:
        #       input (1, 100, 5), input2 (10, 100, 5), target (10, 100, 5)
        #       changes to
        #       input(10, 100, 5) ... (rest same)
        #       without duplicating memory.
        # NOTE: Broadcasting only happens *within* each dict, i.e. inputs
        #       will not be broadcast to match targets or vise-versa.
        #       That shouldn't be necessary, since cost functions will handle
        #       broadcasting between inputs & targets separately.

        inputs = self._broadcast_dict(self.inputs)
        outputs = self._broadcast_dict(self.outputs)
        targets = self._broadcast_dict(self.targets)

        return self.modified_copy(inputs, outputs, targets)

    @staticmethod
    def _broadcast_dict(d):
        """TODO: docs, internal for broadcast_samples."""
        if len(d) < 2:
            # Nothing to broadcast to
            new_d = d.copy()
        else:
            new_d = {}
            for k, v in d.items():
                temp = d.copy()
                temp.pop(k)  # compare against all other arrays
                for k2, v2 in temp.items():
                    try:
                        # Only try to broadcast to the other array's first dim
                        # (i.e. number of samples). If v.shape = (1, ...) and
                        # v2.shape = (N, ...), new_v.shape = (N, ...).
                        new_v = np.broadcast_to(v, v2.shape[:1] + v.shape[1:])
                        assert np.shares_memory(new_v, v)
                        new_d[k] = new_v
                    except ValueError:
                        # Incompatible shape (either both arrays have multiple
                        # samples or v has multiple and v2 has 1).
                        new_d[k] = v
        
        return new_d

    def as_batches(self, batch_size=None, permute=True):
        """TODO: docs"""
        # TODO: per SVD request, also support passing in list directly
        #       (e.g. if data is already a list, assume it has been split
        #        and that each array in the list is one batch).

        # Split data into batches along first axis. Should end up with a list
        # of arrays with shape (B, T, N), where B is `batch_size` (i.e. number
        # of samples per batch).
        # NOTE: This implementation results in a list of views into the
        #       original data (i.e. memory is shared). If changes are made,
        #       make sure the new version doesn't result in copies (which
        #       could increase memory usage dramatically).
        # NOTE: The last batch will be smaller than the others if the number
        #       of samples doesn't divide evenly. Need to set up cost to
        #       account for that (i.e. can't average across batches on arrays,
        #       but could average across batches on already-computed costs).
        d = self.as_broadcasted_samples()

        batched_inputs, batched_outputs, batched_targets = [
            d._arrays_to_batches(_dict, batch_size)
            for _dict in [d.inputs, d.outputs, d.targets]
        ]

        # NOTE: Every array must have the same number of samples
        # (and the same number of batches as a result), otherwise this will
        # not work as intended.
        n_batches = len(list(batched_inputs.values())[0])

        # Index into batched_data instead of collecting a list of batches,
        # to ensure memory is still shared. Also makes permutations easier.
        indices = np.arange(n_batches)
        if permute:
            # Randomly shuffle indices
            np.random.shuffle(indices)

        for i in indices:
            inputs = {k: v[i] for k, v in batched_inputs.items()}
            outputs = {k: v[i] for k, v in batched_outputs.items()}
            targets = {k: v[i] for k, v in batched_targets.items()}
            d.assert_no_copies(inputs, outputs, targets)
            yield d.modified_copy(inputs, outputs, targets)

    def _arrays_to_batches(self, data, batch_size):
        """TODO: docs, internal for as_batches."""
        if (batch_size is None) and (len(data) > 0):
            # Assume sample dimension exists, set batch_size to force 1 batch
            batch_size = list(data.values())[0].shape[0]
        batched_data = {
            k: np.split(v, np.arange(batch_size, len(v), batch_size))
            for k, v in data.items()
            }

        return batched_data

    def as_samples(self):
        """TODO: docs"""
        # NOTE: must alread have a sample dimension, use as_batches first
        #       if not.
        # NOTE: Every array must have the same number of samples
        #       otherwise this will not work as intended.
        n_samples = len(list(self.inputs.values())[0])
        s_inputs, s_outputs, s_targets = [
            {k: np.split(v, n_samples) for k, v in d.items()}
            for d in [self.inputs, self.outputs, self.targets]
            ]

        # TODO: want to be able to permute samples within batches as well?
        for i in range(n_samples):
            inputs = {k: v[i].squeeze(axis=0) for k, v in s_inputs.items()}
            outputs = {k: v[i].squeeze(axis=0) for k, v in s_outputs.items()}
            targets = {k: v[i].squeeze(axis=0) for k, v in s_targets.items()}
            self.assert_no_copies(inputs, outputs, targets)
            yield self.modified_copy(inputs, outputs, targets)


    def as_dict(self):
        return {**self.inputs, **self.outputs, **self.targets}

    # Pass dict get (but not set) operations to self.inputs, outputs, targets
    def __getitem__(self, key):
        return self.as_dict()[key]
    def get(self, key, default):
        return self.as_dict().get(key, default)
    def items(self):
        return self.as_dict().items()
    def __iter__(self):
        return self.as_dict().__iter__()
    def __len__(self):
        return len(self.as_dict())

    def modified_copy(self, inputs, outputs, targets):
        # TODO: make sure this still shares memory
        data = DataSet(
            inputs, state=None, target=targets, dtype=self.dtype,
            input_name=self.input_name, state_name=self.state_name,
            output_name=self.output_name, target_name=self.target_name
            )
        data.outputs = outputs
        return data

    def apply(self, fn, allow_copies=False):
        """TODO: docs. Maps {k: v} -> {k: fn(v)} for all k, v."""
        inputs, outputs, targets = [
            self._apply_to_dict(fn, d, allow_copies=allow_copies)
            for d in [self.inputs, self.outputs, self.targets]
            ]
        return self.modified_copy(inputs, outputs, targets)
    
    def _apply_to_dict(self, fn, d, allow_copies=False):
        new_d = d.copy()
        for k, v in new_d.items():
            new_v = fn(v)
            if not allow_copies:
                assert np.shares_memory(new_v, v)
            new_d[k] = new_v
        return new_d

    def assert_no_copies(self, inputs, outputs, targets):
        """TODO: docs. For debugging, check if arrays share memory with self."""
        for k in inputs.keys():
            assert np.shares_memory(inputs[k], self.inputs[k])
        for k in outputs.keys():
            assert np.shares_memory(outputs[k], self.outputs[k])
        for k in targets.keys():
            assert np.shares_memory(targets[k], self.targets[k])

    def prepend_samples(self):
        """Prepend a singleton sample dimension."""
        return self.apply(lambda v: v[np.newaxis,...], allow_copies=False)

    def squeeze_samples(self):
        """Remove singleton sample dimension from all arrays."""
        return self.apply(lambda v: np.squeeze(v, axis=0), allow_copies=False)

    # TODO: Looks like there's no way to concatenate numpy views without
    #       creating copies, since views can be non-continguous. So instead,
    #       only concatenating outputs (which have to be new arrays anyway
    #       by definition). But if we can figure out a way to concatenate
    #       inputs and targets without duplicating memory, this should just
    #       return a modified copy with all the inputs, outputs and targets
    #       concatenated instead.
    @staticmethod
    def concatenate_sample_outputs(data_sets):
        outputs = {}
        for d in data_sets:
            for k, v in d.outputs.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)
        concatenated = {k: np.concatenate(v, axis=0)
                        for k, v in outputs.items()}

        return concatenated
