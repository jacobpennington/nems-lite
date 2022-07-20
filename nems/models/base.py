from operator import itemgetter

import numpy as np


class Model:
    default_input = 'stimulus'
    default_output = 'prediction'
    default_target = 'response'
    default_state = 'state'
    default_backend = 'scipy'

    def __init__(self, layers=None):
        # TODO
        self._layers = {}  #  layer.name : layer obj, increment on clashes
        pass

    @property
    def layers(self):
        """Get all Model Layers. Supports integer or string indexing."""
        return _LayerDict(self._layers)

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

    def evaluate(self, input, input_name=None, state=None, state_name=None,
                 output_name=None, return_full_data=True):
        """Transform input(s) by invoking `Layer.evaluate` for each Layer.

        TODO: add more context, examples

        Parameters
        ----------
        input : ndarray or dict
            If ndarray, use this as the input to the first Layer. Otherwise,
            use keys specified by `input_name` or `Layer.input` to determine
            the first input.
        input_name : str or None; optional.
            Specifies which data stream should be provided as input to the
            first layer. Note that priority will be given to `Layer.input` in
            case of a clash. I.e. if `input_name is not None`, but also
            `Layer.input is not None` for the first layer, `Layer.input` will
            be used.
            If None : use `Layer.input` of first layer if specified,
                      otherwise `Model.default_input`.
            If str  : a single input array at key `input_name`.
        state : ndarray or None; optional.
            If ndarray, add to intermediate data dictionary. This option can
            only be used in conjunction with an array `input`. If other data is
            needed, use a dictionary input containing all data.
        state_name : str or None; optional.
            If str, and `state is not None`, then `state_name` will be the key
            for `state`. Otherwise, `Model.default_state` will be used.
        output_name : str or None; optional.
            Specifies name(s) for array output the final layer if
            `Layer.output is None`.
            If None : use `Model.default_output`.
            If str  : Use this name for each of the Layer's outputs
                      (incremented if multiple).
        return_full_data : bool; default=True
            If True, return a dictionary containing all input data and all
            uniquely-keyed Layer outputs.

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

        # Set first input if None
        all_inputs = [layer.input for layer in self.layers]
        if all_inputs[0] is None:
            all_inputs[0] = input_name
        # Set last output if None
        all_outputs = [layer.output for layer in self.layers]
        if all_outputs[-1] is None:
            all_outputs[-1] = output_name

        # Loop through transformations
        iter_zip = zip(self.layers, all_inputs, all_outputs)
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
                    data[k] = v

        if not return_full_data:
            data = data_out
        return data

    def predict(self, input, return_full_data=False, **eval_kwargs):
        """As `Model.evaluate`, but return only the last output by default."""
        return self.evaluate(input, return_full_data=return_full_data,
                             **eval_kwargs)

    def fit(self, input, target, state=None, fitter_options=None, **eval_kwargs):
        # TODO: see `scripts/simple_fit.py` for ideas on how to format this.
        #       But overall, should be a straightforward wrapper/router that
        #       picks an optimization (scipy, tensorflow, etc), calls the
        #       relevant fn, and then updates parameters.
        #
        #       More complicated fits (like LBHB's three-stage initialization
        #       for LN models) should *not* be included here in a huge if/else
        #       statement. Instead, they should be separate functions that use
        #       calls to this method as building blocks
        #       (see `scripts/freeze_parameters.py`)
        pass

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
        # collect all bounds
        boundses = []
        for layer in self.layers:
            bounds = layer.get_bounds_vector(none_for_inf=none_for_inf)
            boundses.append(bounds)
        # flatten list
        model_bounds = [t for bounds in boundses for t in bounds]

        return model_bounds

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
        for layer in self.layers:
            layer.set_parameter_vector(vector, ignore_checks=ignore_checks)

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
        for attr in ['input', 'output', 'target', 'state', 'backend']:
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
        string += ".layers:\n\n\n\n"
        for i, layer in enumerate(self.layers):
            if i != 0:
                # Add blank line between layers if more than one
                string += '\n\n\n'
            string += layer.__repr__()
        string += "\n" + tilde_break

        return string

    # Add compatibility for saving to json
    def to_json(self):
        # TODO
        # iterate layer.to_json + add metadata
        pass

    def from_json(json):
        # TODO
        # store metadata, iterate layer.from_json and invoke add_layers
        pass


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
        # TODO: is this worth it vs a for loop? test timing
        if not isinstance(keys, tuple):
            # only one argument, wrap it
            keys = [keys]
        layers = itemgetter(*keys)(self._container_for(keys[0]))
        return layers

    def get(self, *keys, default=None):
        # no keys, get all layers
        if keys == ():
            layers = self._values
        else:
            container = self._container_for(keys[0])
            layers = []
            for key in keys:
                try:
                    layer = container[key]
                except (IndexError, KeyError):
                    layer = default
                layers.append(layer)
        
        # Tuple wrapper just for output consistency with __getitem__,
        # not really necessary.
        return tuple(layers)

    def __iter__(self):
        """Iterate over Layers (not keys)."""
        return iter(self._values)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._values

    def __repr__(self):
        return self._dict.__repr__()
