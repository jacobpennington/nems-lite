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

    def evaluate(self, recording):
        # TODO: need to check for None input/output and specify 
        #       the default behavior in those cases.
        # inputs = [Recording[name].values for name in Module.inputs]
        # outputs = Module.evaluate(inputs)
        # Recording.update(
        #     {name: array_to_signal(array)
        #      for name, array in zip(Module.output, outputs)}
        # )
        pass

    def add_modules(self, *modules):
        """Invokes `self.add_module` for each module in arguments."""
        for m in modules:
            self.add_module(m)

    def add_module(self):
        # TODO: in addition to adding to a .modules list (or dict or whatever),
        #       set `module.model = self` so that each module has a pointer
        #       to its parent Modelspec.
        pass

    def freeze_parameters(self, *module_keys):
        """Invoke `Module.freeze_parameters` for each keyed module.
        
        See also
        --------
        nems.modules.base.Module.freeze_parameters
            The Module-level method 

        """
        for key in module_keys:
            self.modules[key].freeze_parameters()

    def to_json(self):
        # TODO
        # iterate module.to_json + add metadata 
        pass

    def from_json(json):
        # TODO
        # store metadata, iterate module.from_json and invoke add_modules
        pass

    # TODO: not real names, just to remind me what I mean. Added methods
    # for phi->vector->phi and bounds->vector conversion to modules. So need
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
