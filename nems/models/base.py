class Model:
    default_input = 'stimulus'
    default_output = 'prediction'
    default_target = 'response'
    default_state = 'state'
    default_backend = 'scipy'

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

    def predict(self, input):
        # TODO: I guess this would really just be a wrapper for evaluate?
        pass

    def score(self, prediction, target):
        # TODO: this should point to an independent utility function, but
        #       placed here for convenience (and also to provide model defaults).
        pass

    def add_layers(self, *layers):
        """Invokes `self.add_layer` for each layer in arguments."""
        for m in layers:
            self.add_layer(m)

    def add_layer(self):
        # TODO: in addition to adding to a .layers list (or dict or whatever),
        #       set `layer.model = self` so that each layer has a pointer
        #       to its parent Modelspec.
        pass

    def freeze_parameters(self, *layer_keys):
        """Invoke `Layer.freeze_parameters` for each keyed layer.
        
        See also
        --------
        nems.layers.base.Layer.freeze_parameters
            (The layer-level method)

        """
        for key in layer_keys:
            self.layers[key].freeze_parameters()

    def to_json(self):
        # TODO
        # iterate layer.to_json + add metadata 
        pass

    def from_json(json):
        # TODO
        # store metadata, iterate layer.from_json and invoke add_layers
        pass

    # TODO: not real names, just to remind me what I mean. Added methods
    # for phi->vector->phi and bounds->vector conversion to layers. So need
    # to add a model-level version that just collects all the pieces for fit.
    #
    # Considering keeping this functionality separate from these classes, adds
    # extra complexity for not much benefit.
    def to_vector():
        pass
    def from_vector():
        pass
    def bounds():
        pass
