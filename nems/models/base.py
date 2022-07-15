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

    def evaluate(self, input):
        # TODO: need to check for None input/output and specify 
        #       the default behavior in those cases.
        # inputs = [Recording[name].values for name in Layer.inputs]
        # outputs = Layer.evaluate(inputs)
        # Recording.update(
        #     {name: array_to_signal(array)
        #      for name, array in zip(Layer.output, outputs)}
        # )
        pass

    def fit(self):
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

    def get_parameter_vector(self, as_list=True):
        # TODO: for scipy fitter
        pass

    def get_bounds_vector(self):
        # TODO: for scipy fitter
        pass

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
        for layer in self._layers.values():
            layer.set_index(index, new_index=new_index)

    def freeze_parameters(self, *layer_keys):
        """Invoke `Layer.freeze_parameters` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.freeze_parameters
            (The layer-level method)

        """
        for key in layer_keys:
            self.layers[key].freeze_parameters()

    def predict(self, input):
        # TODO: I guess this would really just be a wrapper for evaluate?
        pass

    def score(self, prediction, target):
        # TODO: this should point to an independent utility function, but
        #       placed here for convenience (and also to provide model defaults).
        pass

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
    """Simple wrapper for Layer._layers to enable int- or string-indexed gets.
    
    Note that index assignment is not supported. To change a Model's Layers,
    use `Model.add_layers`, `Model.remove_layers`, etc.

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

    def __getitem__(self, key):
        return self._container_for(key)[key]

    def get(self, key, default=None):
        container = self._container_for(key)
        try:
            layer = container[key]
        except IndexError:
            layer = default
        return layer
