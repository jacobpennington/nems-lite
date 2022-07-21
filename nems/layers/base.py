"""Base structures for representing data transformation steps of Models.

Layer     : Top-level component. Implements an `evaluate` method, which carries
            out the transformation.
Phi       : Stores and manages fittable parameters for the transformation.
Parameter : Low-level representation of individual parameters.

"""

import json

import numpy as np

from nems.registry import layer
from nems.distributions import Normal


# TODO: add examples and tests
class Layer:
    """Encapsulates one data-transformation step of a NEMS Model.

    Base class for NEMS Layers.

    """

    # Any subclass of Layer will be registered here, for use by
    # `Layer.from_json`
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, input=None, output=None, parameters=None,
                 priors=None, bounds=None, name=None):
        """Encapsulates one data-transformation step of a NEMS ModelSpec.

        Layers are intended to exist as components of a parent Model by
        invoking `Model.add_layer` or `Model.__init__(layers=...)`.

        Parameters
        ----------
        input : str, list, dict, or None; optional
            Specifies which data streams should be provided as inputs by
            parent Model during evaluation, where strings refer to keys for a
            dict of arrays provided to `Model.fit`.
            If None : output of previous Layer.
            If str  : a single input array.
            If list : many input arrays.
            If dict : many input arrays, with keys specifying which parameter
                      of `Layer.evaluate` each array is associated with.
            (see examples below)
        output : str, list, or None; optional
            Specifies name(s) for array output(s) of `Layer.evaluate`.
            If None : use default name specified by parent Model.
            If str  : same name for every output (incremented if multiple).
            If list : one name per output (length must match).
            (see examples below)
        parameters : nems.layers.base.Phi or None; optional
            Specifies values for fittable parameters used by `Layer.evaluate`.
            If None : Phi returned by `Layer.initial_parameters`.
        priors : dict of Distributions or None; optional
            Determines prior that each Layer parameter will sample values from.
            Keys must correspond to names of parameters, such that each
            Parameter utilizes `Parameter(name, prior=priors[name])`.
            If `None` : all parameters default to Normal(mean=zero, sd=one),
            where zero and one are appropriately shaped arrays of 0 and 1.
            Individual `None` entries in a `priors` dict result in the same
            behavior for those parameters.
        bounds : dict of 2-tuples or None; optional
            Determines minimum and maximum values for fittable parameters. Keys
            must correspond to names of parameters, such that each Parameter
            utilizes `Parameter(name, bounds=bounds[name])`.
            If None : use defaults defined in `Layer.initial_parameters`.
        name : str or None; optional
            A name for the Layer so that it can be referenced through the
            parent Model, in addition to integer indexing.

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Phi
        nems.layers.base.Parameter

        Examples
        --------
        Subclasses that need to overwrite `__init__()` should specify new
        arguments (if any) in the method definition, followed by **kwargs, and
        invoke super().__init__(**kwargs) to ensure all required attributes are
        set correctly. While not strictly required, this is the easiest way to
        ensure Layers function properly within a Model.

        If `initial_parameters` needs access to new attributes, they should be
        set prior to invoking `super().__init__()`. New options that interact
        with base attributes (like `Layer.parameters` or `Layer.priors`) should
        be coded after invoking `super().__init__()`, to ensure the relevant
        attributes have been set.

        For example:
        >>> def __init__(self, new_arg1, new_kwarg2=None, **kwargs):
        ...     self.something_new = new_arg1
        ...     super().__init__(**kwargs)
        ...     self.do_something_to_priors(new_kwarg2)


        When specifying input for Layers, use:
        `None` to retrieve output of previous Model Layer (default).
        `'data_key'` to retrieve a single specific array.
        `['data_key1', ...]` to retrieve many arrays. This is preferred if
            order of arguments for `Layer.evaluate` is not important
        `{'arg1': 'data_key1', ...}` to map many arrays to specific arguments
            for `Layer.evaluate`.

        >>> data = {'stimulus': stim, 'pupil': pupil, 'state1': state1,
        ...         'state2': state2}
        >>> layers = [
        ...     WeightChannels(shape=(18,4), input='stimulus'),
        ...     FIR(shape=(4, 25)),
        ...     DoubleExponential(output='LN_output'),
        ...     Sum(input=['LN_output', 'state', 'pupil'],
        ...                output='summed_output'),
        ...     LinearWeighting(
        ...         input={
        ...             'pred': 'summed_output',
        ...             'pupil': 'pupil',
        ...             'other_states': ['state1', 'state2']
        ...             }
        ...         )
        ...     ]

        """

        # input/output should be string, list of strings,
        # dict of strings, or None
        self.input = input
        self.output = output

        self.priors = priors
        self.bounds = bounds
        self.name = name if name is not None else 'unnamed module'
        self.model = None  # pointer to parent ModelSpec

        if parameters is None:
            parameters = self.initial_parameters()
        self.parameters = parameters

        # If string, `state_name` will be interpreted as the name of an argument
        # for `Layer.evaluate`. During Model evaluation, if `Layer.input` is
        # None and a state array is provided to `Model.evaluate`, then state
        # will be added to other inputs as a keyword argument, i.e.:
        # `layer.evaluate(*inputs, **state)`.
        self.state_name = None

    @layer('baseclass')
    @staticmethod
    def from_keyword(keyword):
        """Construct a Layer corresponding to a registered keyword.

        Each Layer subclass can (optionally) overwrite `from_keyword` to
        enable compatibility with the NEMS keyword system. This is a
        string-based shortcut system for quickly specifying a `Model` without
        needing to import individual Layer subclasses.

        To work correctly within this system, a `from_keyword` method must
        follow these requirements:
        1) The `@layer` decorator, imported from `nems.registry`, must be added
           above the method. The decorator must receive a single string as
           its argument, which serves as the keyword "head": the identifier for
           the relevant Layer subclass. This string must contain only letters
           (i.e. alphabetical characters - no numbers, punctuation, special
           characters, etc). Keyword heads are typically short and all
           lowercase, but this is not enforced.
        2) `from_keyword` must be a static method (i.e. it receives neither
           `cls` nor `self` as an implicit argument). The `@staticmethod`
           decorator is also necessary since the method will normally be
           invoked without a reference to the relevant Layer.
        3) `from_keyword` must accept a single argument. Within the keyword
           system, this argument will be a string of the form:
           `'{head}.{option1}.{option2}...'`
           where any number of options can be specified, separated by periods.
           Options can contain any* character other than hyphens or underscores,
           which are reserved for composing many keywords at once.
           Options for builtin NEMS layers mostly follow certain formatting
           norms for consistency, but these are not enforced:
           a) Use a lowercase 'x' between dimensions for shape:
                `shape=(3,4)`     -> '3x4'
           # TODO: replace examples for below with examples for layers
           #       (but they illustrate the format in the meantime)
           b) Scientific notation refers to negative exponents:
                `tolerance=0.001` -> 't1e3'
                `max_iter=1000`   -> 'i1000' or 'i1K'
           c) Boolean options use a single lowercase letter if possible:
                `normalize=True`  -> 'n'
           # TODO: any other formatting norms I missed?
           * Users should still be aware that escapes and other characters with
           python- or system-specific meanings should be avoided/used with care.
           Generally, sticking to alpha-numeric characters is safest.
        4) Return an instance of a Layer subclass.
        
        See also
        --------
        nems.registry
        nems.layers.weight_channels.WeightChannels.from_keyword

        Examples
        --------
        A minimal version of `from_keyword` for WeightChannels can be defined
        as follows:
        >>> class WeightChannels(Layer):
        >>>     def __init__(self, shape, **kwargs):
        >>>         self.shape = shape
        >>>         super().__init__(**kwargs)
        
        >>>     @layer('wtchans')
        >>>     @staticmethod
        >>>     def from_keyword(kw):
        ...         options = kw.split('.')
        ...         for op in options:
        ...             if ('x' in op) and (op[0].isdigit()):
        ...                 dims = op.split('x')
        ...                 shape = tuple([int(d) for d in dims])
        ...         # Raises UnboundLocalError if shape is never defined.
        ...         return WeightChannels(shape)

        >>> wtchans = WeightChannels.from_keyword('wtchans.18x2')
        >>> wtchans.shape
        (18, 2)
        
        Note: the actual definition referenced above uses `'wc'` as the keyword
        head, not `'wtchans'`. The key is changed for this example to avoid
        a name clash when testing.

        """
        return Layer()

    def initial_parameters(self):
        """Get initial values for `Layer.parameters`.
        
        Default usage is that `Layer.initial_parameters` will be invoked during
        construction to set `Layer.parameters` if `parameters is None`. Each
        Layer subclass should overwrite this method to initialize appropriate
        values for its parameters, and document the Layer's parameters in the
        overwritten docstring.

        Returns
        -------
        parameters : Phi

        See also
        --------
        nems.layers.weight_channels.WeightChannels.initial_parameters

        Examples
        --------
        A minimal version of `initial_parameters` for WeightChannels can be
        defined as follows:
        >>> class WeightChannels(Layer):
        >>>     def __init__(self, shape, **kwargs):
        >>>         self.shape = shape
        >>>         super().__init__(**kwargs)
        
        >>>     def initial_parameters(self):
        ...         coeffs = Parameter(name='coefficients', shape=self.shape)
        ...         return Phi(coeffs)

        >>> wc = WeightChannels(shape=(2, 1))
        >>> wc.parameters
        Parameter(name=coefficients, shape=(2, 1))
        .values:
        [[0.]
         [0.]]
        
        """
        return Phi()

    def evaluate(self, *args, **kwargs):  
        """Applies some mathematical operation to the argument(s).
        
        Each Layer subclass must overwrite this method. Any number of arguments
        is acceptable, but each should correspond to one name in `self.input`
        at runtime. An arbitrary number of return values is also allowed, and
        each should correspond to one name in `self.output`.
        
        Input and output names will be associated with arguments and return
        values, respectively, in list-order. If `self.input` is a dictionary,
        inputs will instead be mapped to specific arguments (where each key is
        an argument, and each value is a data array).

        Warnings
        --------
        Evaluate should never modify inputs in-place, as this could change the
        input to other Layers that expect the original data. Intermediate
        operations should always return copies. If a modified copy of the input
        is needed as standalone data, then the evaluate method of a separate
        layer should produce that output instead.
        
        Returns
        -------
        N-tuple of numpy.ndarray

        See also
        --------
        Layer.__init__

        Examples
        --------

        >>> class DummyLayer(Layer):
        >>>     def evaluate(self, x, y, z):
        ...         a = x + y
        ...         b = 2*z
        ...         return a, b
        
        >>> x = DummyLayer(input=['one', 'two', 'three'], output=['two', 'new'])
        >>> data = {'one': x, 'two': y, 'three': z}
        
        During fitting, `x.evaluate` would receive `(x, y, z)` as arguments
        (in that order) and return `(x+y, 2*z)`, resulting in:
        >>> data
        {'one': x, 'two': x+y, 'three': z, 'new': 2*z}

        """
        raise NotImplementedError(f'{self.__class__} has not defined evaluate.')

    def description(self):
        """Short description of Layer's function.
        
        Defaults to `Layer.evaluate.__doc__` if not overwritten.

        Example
        -------
        def evaluate(self):
            '''A really long docstring with citations and notes other stuff.'''
            return a + np.exp(b-c)

        def description(self):
            return '''Implements a simple exponential: $a + e^{(b-c)}$'''

        """
        return self.evaluate.__doc__

    def tensorflow_layer(self):
        """Build an equivalent `Tensorflow.keras.layers.Layer` representation.

        TODO: How would this fit into the arbitrary input/output scheme that
              .evaluate() uses for scipy optimization? Maybe it can't, since
              Tensorflow already has its own way of managing the data flow.

        TODO: should layers be instantiated at this point, or just return
              the layer class? (probably with some frozen kwargs). Depends
              on how the higher-level model builder is formated.

        """
        raise NotImplementedError(
            f'{self.__class__} has not defined a Tensorflow implementation.'
            )

    def set_parameter_values(self, parameter_dict):
        """Set new parameter values from key, value pairs.
        
        Alias for `Layer.update`.

        """
        self.parameters.update(parameter_dict)

    def get_parameter_values(self, *parameter_keys):
        """Get all parameter values, formatted as a list of arrays.
        
        See also
        --------
        Phi.get_values
        
        """
        return self.parameters.get_values(*parameter_keys)

    def get_parameter_vector(self, as_list=True):
        """Get all parameter values, formatted as a single vector.
        
        Parameters
        ----------
        as_list : bool
            If True, returns a list instead of ndarray.

        Returns
        -------
        list or ndarray

        See also
        --------
        Phi.get_vector

        """
        return self.parameters.get_vector(as_list=as_list)

    def set_parameter_vector(self, vector, ignore_checks=False):
        """Set parameter values with a single vector.

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
        Phi.set_vector
        
        """
        self.parameters.set_vector(vector, ignore_checks=ignore_checks)

    def get_bounds_vector(self, none_for_inf=True):
        """Get all parameter bounds, formatted as a list of 2-tuples.

        Parameters
        ----------
        none_for_inf : bool
            If True, +/- np.inf is replaced with None
            (for scipy-compatible bounds).

        Returns
        -------
        list of 2-tuple

        See also
        --------
        Phi.get_bounds

        """
        return self.parameters.get_bounds(none_for_inf=none_for_inf)

    def sample_from_priors(self, inplace=True, as_vector=False):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, sampled values will be used to update each Parameter
            (and, in turn, Phi._array) inplace. Otherwise, the
            sampled values will be returned without changing current values.
        as_vector : bool, default=False
            If True, return sampled values as a flattened vector instead of a
            list of arrays.

        Returns
        -------
        samples : ndarray or list of ndarray

        See also
        --------
        nems.layers.base.Phi.sample

        """
        return self.parameters.sample(inplace=inplace, as_vector=as_vector)

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
        return self.parameters.mean(inplace=inplace, as_vector=as_vector)

    def freeze_parameters(self, *parameter_keys):
        """Use parameter values for evaluation only, do not optimize.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Layer.parameters`.
            If no keys are specified, all parameters will be frozen.
        
        See also
        --------
        Phi.freeze_parameters
        Parameter.freeze
        
        """
        self.parameters.freeze_parameters(*parameter_keys)

    def unfreeze_parameters(self, *parameter_keys):
        """Make parameter values optimizable.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Layer.parameters`.
            If no keys are specified, all parameters will be unfrozen.
        
        See also
        --------
        Phi.unfreeze_parameters
        Parameter.unfreeze

        """
        self.parameters.unfreeze_parameters(*parameter_keys)

    def set_index(self, i, new_index='initial'):
        """Change which set of parameter values is referenced.

        Intended for use with jackknifing or other procedures that fit multiple
        iterations of a single model. Rather than using many copies of a full
        Model object, each Phi object tracks copies of the underlying vector of
        parameter values.

        Parameters
        ----------
        i : int
            New index for `Phi._array`. If `i >= len(Phi._array)`, then new
            vectors will be appended until `Phi._array` is sufficiently large.
        new_index : str or None, default='initial'
            Determines how new vectors are generated if `i` is out of range.
            If `'sample'`   : invoke `Phi.sample(inplace=False)`.
            Elif `'mean'`   : invoke `Phi.mean(inplace=False)`.
            Elif `'initial'`: set to `[p.initial_value for p in <Parameters>]`.
            Elif `'copy'`   : copy current `Phi.get_vector()`.
            Elif `None`     : raise IndexError instead of adding new vectors.

        See also
        --------
        nems.models.base.Model.set_index

        """
        self.parameters.set_index(i, new_index=new_index)

    # Passthrough dict-like interface for Layer.parameters
    # NOTE: 'get' operations through this interface return Parameter instances,
    #       not arrays. For array format, reference Parameter.values or use
    #       `Layer.get_parameter_values`.
    def __getitem__(self, key):
        """Get Parameter (not Parameter.values)."""
        return self.parameters[key]
    
    def get(self, key, default=None):
        """Get Parameter (not Parameter.values)."""
        return self.parameters.get(key, default=default)

    def __setitem__(self, key, val):
        """Set Parameter.values (not Parameter itself)."""
        self.parameters[key] = val

    def update(self, dct):
        """Update Parameter values (not the Parameters themselves).
        
        Alias for `Layer.set_parameter_values`.

        """
        self.parameters.update(dct)

    def __iter__(self):
        return self.parameters.__iter__()

    def keys(self):
        return self.parameters.keys()

    def items(self):
        return self.parameters.items()

    def values(self):
        return self.parameters.values()

    def __repr__(self):
        layer_dict = Layer().__dict__
        self_dict = self.__dict__
        # Get attributes that are not part of base class
        # e.g. shape for WeightChannels
        # then convert to string with "k=v" format
        self_only = ", ".join([f"{k}={v}" for k, v in self_dict.items()
                               if k not in layer_dict])
        header = f"{type(self).__name__}({self_only})\n"
        equal_break = "="*32 + "\n"
        string = header + equal_break
        string += ".parameters:\n\n\n"
        string += self.parameters.__repr__() + "\n"
        string += equal_break
        return string

    # Add compatibility for saving to .json
    def to_json(self):
        """Encode a Layer as a dictionary.

        This (base class) method encodes all attributes that are common to
        all Layers. Subclasses that need to save additional kwargs or attributes
        should overwrite `to_json`, but invoke `Layer.to_json` within the new
        method as a starting point (see examples below).
        
        See also
        --------
        'Layer.from_json`
        `nems.tools.json`
        `nems.layers.weight_channels.WeightChannels.to_json`

        Examples
        --------
        >>> class WeightChannels(Layer):
        ...     # ...
        >>>     def to_json(self):
        ...         data = Layer.to_json(self)
        ...         data['kwargs'].update(shape=self.shape)
        ...         return data

        >>> class DummyLayer(Layer):
        >>>     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        ...         important_attr = None
        >>>     def update_important_attr(self, *args):
        ...         important_attr = do_stuff(args)
        ...     # ...
        ...     # Want to preserve the state of `important_attr` in encoding.
        >>>     def to_json(self):
        ...         data = Layer.to_json(self)
        ...         data['attributes'].update(
        ...             important_attr=self.important_attr
        ...             )
        ...         return data

        """
        data = {
            'kwargs': {
                'input': self.input, 'output': self.output,
                'parameters': self.parameters, 'priors': self.priors,
                'bounds': self.bounds, 'name': self.name, 
            },
            'attributes': {},
            'class_name': type(self).__name__
            }
        return data

    @classmethod
    def from_json(cls, json):
        """Decode a Layer from a dictionary.

        Parameters
        ----------
        json : dict
            json data encoded by `Layer.to_json`.

        Returns
        -------
        layer : Layer or subclass
        
        See also
        --------
        `Layer.to_json`
        `nems.tools.json`.

        """
        subclass = cls.subclasses[json['class_name']]
        if subclass.from_json.__qualname__ != Layer.from_json.__qualname__:
            # Subclass has overwritten `from_json`, use that method instead.
            layer = subclass.from_json()
        else:
            layer = Layer(**json['kwargs'])
            for k, v in json['attributes'].items():
                setattr(layer, k, v)

        return layer



###############################################################################
#####                             Phi                                     #####
###############################################################################

# TODO: add examples and tests
#       (for both Phi and Parameter)
# TODO: TF compatibility functions? during fitting, values would have to exist
#       as TF primitives, but need to be able to translate to/from
#       TF and NEMS representations.

class Phi:
    """Stores, and manages updates to, Parameters for one Layer."""

    def __init__(self, *parameters):
        """Stores, and manages updates to, Parameters for one Layer.

        In general, Phi instances should not need to be interacted with directly
        unless implementing a new Layer subclass or a related function. Instead,
        parameters should be accessed through Model- or Layer-level methods.

        Additionally, the set of Parameters assigned to a Phi object is meant
        to be fixed at construction. To get a Phi object with some parameters
        either added or removed, construct a new Phi instance or use
        `Phi.modified_copy`.

        Parameters
        ----------
        parameters : N-tuple of Parameter; optional

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Parameter
        
        """
        self._array = [[]]
        self._index = 0
        self._dict = {}
        self.size = 0
        # Add parameter values to nested list
        for p in parameters:
            self._add_parameter(p)
        # Convert to ndarray
        self._array = np.array(self._array)
        # Cached indexing into `_array` for `get_vector` and `set_vector`.
        self._vector_mask = None
        self._update_vector_mask()

    # TODO: if it becomes necessary, we can relax this restriction and allow
    #       for adding/removing Parameters. But I can't think of a case where
    #       it would really be needed since Layer parameters aren't meant to
    #       change, and this keeps the implementation of other methods simpler.
    def _add_parameter(self, parameter):
        """Add a new parameter to `Phi._dict` and update `Phi._array`.
        
        Sets `parameter.phi`, `parameter.first_index`, and
        `parameter.last_index` to comply with `Phi.ge_vector()` formatting.
        
        This method should only be invoked during construction, since
        `Phi._array` will be converted to ndarray afterward. This limitation
        is intentional: a Phi object is meant to represent a fixed set of
        model parameters. If different parameters are needed after construction,
        create a new Phi object.
        
        """
        # Start at end of existing vector, track size for next parameter
        parameter.first_index = self.size
        self.size += parameter.size
        parameter.last_index = self.size-1
        # Always only one "row" during construction, so use 0 index.
        self._array[0].extend(parameter.initial_value)
        self._dict[parameter.name] = parameter
        parameter.phi = self

    def _get_mask(self, *index_ranges):
        """Get index mask into current vector within `Phi._array`.

        Parameters
        ----------
        index_ranges : N-tuple of 2-tuples
            First tuple entry = first index, second tuple entry = last index.

        Returns
        -------
        mask : boolean ndarray, shape=Phi._array.shape

        Notes
        -----
        Using `mask` for selection will result in a copy of `Phi._array`.
        Using `mask` for assignment will change values of `Phi._array` itself.
        
        """
        mask = np.full(self._array.shape, False)
        for first, last in index_ranges:
            mask[self._index][first:last+1] = True
        return mask

    def _update_vector_mask(self):
        """Update cached copy of current mask for `Phi.<get/set>_vector`.
        
        This method must be invoked any time there is a change to the indices
        within `Phi._array` to which `Phi.<get/set>_vector` would refer
        (i.e. parameters are frozen/unfrozen, `Phi._index` changes, etc).
        
        """
        parameter_ranges = [
            (p.first_index, p.last_index) for p in self._dict.values()
            if not p.is_frozen
            ]
        self._vector_mask = self._get_mask(*parameter_ranges)

    def get_vector(self, as_list=False):
        """Get a copy of `Phi._array` sliced at `Phi._index`.
        
        Parameters
        ----------
        as_list : bool
            If True, return `vector.tolist()` instead of `vector`.

        Returns
        -------
        vector : ndarray or list
        
        """
        vector = self._array[self._vector_mask]
        if as_list:
            vector = vector.tolist()
        return vector

    def get_bounds(self, none_for_inf=True):
        """Return a list of bounds from each parameter in `Phi._dict`.
        
        Parameters
        ----------
        none_for_inf : bool, default=True
            If True, replace (+/-)`np.inf` with None for compatibility with
            `scipy.optimize.minimize`.
        
        Returns
        -------
        bounds : list of 2-tuples
        
        """
        bounds = [p.bounds for p in self._dict.values()]
        if none_for_inf:
            subbed_bounds = []
            for b in bounds:
                lower, upper = b
                if np.isinf(lower):
                    lower = None
                if np.isinf(upper):
                    upper = None
                subbed_bounds.append((lower, upper))
            bounds = subbed_bounds
        
        return bounds

    def within_bounds(self, vector):
        """False if anywhere `vector < bounds[0]` or `vector > bounds[1]`."""
        passed = True
        bounds = self.get_bounds(none_for_inf=False)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        if np.any(vector < lower) or np.any(vector > upper):
            passed = False
        return passed

    def get_indices_outof_range(self, vector, as_bool=True):
        """Get indices where `vector < bounds[0]` or `vector > bounds[1]`."""
        bounds = self.get_bounds(none_for_inf=False)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        if as_bool:
            indices = np.logical_or(vector < lower, vector > upper)
        else:
            check_low = np.argwhere(vector < lower)
            check_high = np.argwhere(vector > upper)
            indices = np.vstack([check_low, check_high]).flatten()
        
        return indices

    def set_vector(self, vector, ignore_checks=False):
        """Set values of `Phi._array` sliced at `Phi._index` to a new vector.

        Parameters
        ----------
        vector : ndarray or list
            New parameter values. Size must match `Phi.get_vector`.
        ignore_checks : bool
            If True, set new values without checking size or bounds.
            (intended as a minor optimization for the scipy fitter).
        
        """
        if not ignore_checks:
            if np.array(vector).size != self._vector_mask.size:
                raise ValueError(f"Size of new vector != Phi.get_vector.")
            if not self.within_bounds(vector):
                bad_indices = self.get_indices_outof_range(vector, as_bool=False)
                raise ValueError("Vector out of bounds at indices:\n"
                                 f"{bad_indices}.")

        self._array[self._vector_mask] = vector

    def _get_parameter_mask(self, p):
        """Get an index mask as in `Phi._get_mask`, but for one Parameter."""
        return self._get_mask((p.first_index, p.last_index))

    def _get_parameter_vector(self, p):
        """Get a sliced copy of `Phi._array` corresponding to one Parameter."""
        mask = self._get_parameter_mask(p)
        return self._array[mask]
        
    def freeze_parameters(self, *parameter_keys):
        """Use parameter values for evaluation only, do not optimize.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Phi._dict`.
            If no keys are specified, all parameters will be frozen.
        
        See also
        --------
        Layer.freeze_parameters
        Parameter.freeze
        
        """
        if parameter_keys == ():
            # no keys given, freeze all parameters
            parameter_keys = list(self._dict.keys())
        for pk in parameter_keys:
            p = self._dict[pk]
            if p.is_frozen:
                # Already frozen, nothing to do
                pass
            else:
                p.freeze()
        self._update_vector_mask()

    def unfreeze_parameters(self, *parameter_keys):
        """Make parameter values optimizable.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        parameter_keys : N-tuple of str
            Each key must match the name of a Parameter in `Phi._dict`.
            If no keys are specified, all parameters will be unfrozen.
        
        See also
        --------
        Layer.unfreeze_parameters
        Parameter.unfreeze

        """
        if parameter_keys == ():
            # no keys given, freeze all parameters
            parameter_keys = list(self._dict.keys())
        for pk in parameter_keys:
            p = self._dict[pk]
            if not p.is_frozen:
                # Already unfrozen, nothing to do
                pass
            else:
                p.unfreeze()
        self._update_vector_mask()

    def sample(self, inplace=False, as_vector=True):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, sampled values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, the sampled values
            will be returned without changing current values.
        as_vector : bool, default=True
            If True, return sampled values as a flattened vector instead of a
            list of arrays.

        Return
        ------
        samples : ndarray or list of ndarray

        """
        samples = [p.sample(inplace=inplace) for p in self._dict.values()]
        if as_vector:
            unravelled = [np.ravel(s) for s in samples]
            samples = np.concatenate(unravelled)
        return samples

    def mean(self, inplace=False, as_vector=True):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, mean values will be used to update each Parameter
            (and, in turn, `Phi._array`) inplace. Otherwise, means will be
            returned without changing current values.
        as_vector : bool, default=True
            If True, return means as a flattened list instead of a
            list of arrays.

        Return
        ------
        means : list

        """
        means = [p.mean(inplace=inplace) for p in self._dict.values()]
        if as_vector:
            unravelled = [np.ravel(m) for m in means]
            means = np.concatenate(unravelled)
        return means


    def set_index(self, i, new_index='initial'):
        """Change which vector to reference within `Phi._array`.

        Updates `Phi._vector_mask`.

        Parameters
        ----------
        i : int
            New index for `Phi._array`. If `i >= len(Phi._array)`, then new
            vectors will be appended until `Phi._array` is sufficiently large.
        new_index : str or None, default='initial'.
            Determines how new vectors are generated if `i` is out of range.
            If `'sample'`   : invoke `Phi.sample()`.
            Elif `'mean'`   : invoke `Phi.mean()`.
            Elif `'initial'`: set to `[p.initial_value for p in <Parameters>]`.
            Elif `'copy'`   : copy current `Phi.get_vector()`.
            Elif `None`     : raise IndexError instead of adding new vectors.

        """
        array_length = len(self._array)
        if i >= array_length:
            # Array isn't that big yet, so add new vector(s).
            new_indices = range(array_length, i+1)
            if new_index == 'sample':
                new_vectors = [self.sample() for j in new_indices]
            elif new_index == 'mean':
                new_vectors = [self.mean() for j in new_indices]
            elif new_index == 'initial':
                new_vectors = [
                    np.concatenate([p.initial_value
                                    for p in self._dict.values()])
                    for j in new_indices
                    ]
            elif new_index == 'copy':
                new_vectors = [self.get_vector() for j in new_indices]
            else:
                # Should be None. Don't add new vectors, raise an error
                # instead. May be useful for testing.
                raise IndexError(f'list index {i} out of range for Phi.')
            # Convert to 2-dim vectors and concatenate after existing vectors
            new_rows = [v[np.newaxis, ...] for v in new_vectors]
            self._array = np.concatenate([self._array] + new_rows)

        self._index = i
        self._update_vector_mask()

    # Provide dict-like interface into Phi._dict
    def __getitem__(self, key):
        return self._dict[key]
    
    def get(self, key, default=None):
        return self._dict.get(key, default)

    def get_values(self, *keys):
        return [self._dict[k].values for k in keys]

    def update(self, dct):
        """Update Parameter values (not the Parameters themselves)."""
        for k, v in dct.items():
            self._dict[k].update(v)   

    def __setitem__(self, key, val):
        """Update Parameter value (not the Parameters itself)."""
        self._dict[key].update(val)

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        footer = f"Index: {self._index}\n" + "-"*16 + "\n"
        string = ""
        for i, p in enumerate(self._dict.values()):
            if i != 0:
                # Add blank line between parameters if more than one
                string += "\n"
            string += p.__repr__()
            string += footer
        string += "\n"

        return string

    # Add compatibility for saving to .json
    def to_json(self):
        """Encode Phi object as json. See `nems.tools.json`."""
        p = list(self._dict.values())
        frozen_parameters = [k for k, v in self._dict.items() if v.is_frozen]
        data = {
            'args': p,
            'attributes': {
                '_array': self._array,
                '_index': self._index
            },
            'frozen_parameters': frozen_parameters
        }
        return data

    def from_json(json):
        """Decode Phi object from json. See `nems.tools.json`."""
        phi = Phi(*json['args'])
        for k, v in json['attributes'].items():
            setattr(phi, k, v)
        phi.freeze_parameters(json['frozen_parameters'])
        return phi

    def from_dict(dct, default_bounds='infinite'):
        """Construct Phi from a specially formatted dictionary.
        
        Parameters
        ----------
        dct : dict
            Must contain three nested dictionaries at keys 'values', 'prior',
            and 'bounds'. Each dictionary must have the same keys, which will
            be the parameter names. Values within each dictionary will be used
            as arguments for `initial_value`, `prior`, and `bounds`,
            respectively.
        default_bounds : string, default='infinite'
            Determines behavior when `bounds=None`.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))

        See also
        --------
        Parameter.__init__
        
        """
        parameters = []
        for name, value in dct['values'].items():
            value = np.array(value)
            prior = dct['prior'][name]
            bounds = dct['bounds'][name]
            p = Parameter(name, shape=value.shape, prior=prior, bounds=bounds,
                          default_bounds=default_bounds, initial_value=value)
            parameters.append(p)
        phi = Phi(*parameters)
        return phi

    def modified_copy(self, keys_to_keep, parameters_to_add):
        """TODO."""
        #       ref `keys_to_keep` to store Parameter objects,
        #       combine with parameters_to_add,
        #       build new phi,
        #       overwrite part of new array with copy of old array
        #       copy old index
        raise NotImplementedError


###############################################################################
#####                           Parameter                                 #####
###############################################################################

class Parameter:
    """Stores and manages updates to values for one parameter of one Layer."""

    def __init__(self, name, shape=(), prior=None, bounds=None,
                 default_bounds='infinite', initial_value='mean'):
        """Stores and manages updates to values for one parameter of one Layer.

        Parameters are intended to exist as components of a parent Phi instance,
        by invoking `Phi.add_parameter`. Without establishing this relationship,
        most Parameter methods will not work.

        As with Phi, Parameters should generally not be interacted with
        directly unless implementing new Layer subclasses or other core
        functionality. Wherever possible, users should interact with fittable
        parameters using Model- or Layer-level methods.

        Parameters
        ----------
        name : str
            This will also be the Parameter's key in `Phi._dict`.
        shape : N-tuple, default=()
            Shape of `Parameter.values`.
        prior : nems.distributions.Distribution or None; optional
            Prior distribution for this parameter, with matching shape.
        bounds : 2-tuple or None; optional
            (minimum, maximum) values for the entries of `Parameter.values`.
        default_bounds : str, default='infinite'
            Determines behavior when `bounds=None`.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))
        initial_value : str, scalar, or ndarray, default='mean'
            Determines initial entries of `Parameter.values`.
            If `'mean'`   : set values to `Parameter.prior.mean()`.
            If `'sample'` : set values to `Parameter.prior.sample()`.
            If scalar     : set values to `np.full(Parameter.shape, scalar)`.
            If ndarray    : set values to array (must match `Parameter.shape`).

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Phi

        """
        self.name = name
        self.shape = shape
        self.size = 1
        for axis in shape:
            self.size *= axis

        # default to multivariate normal
        if prior is None:
            zero = np.zeros(shape=self.shape)
            one = np.ones(shape=self.shape)
            prior = Normal(mean=zero, sd=one)  
        self.prior = prior
        if prior.shape != self.shape:
            raise ValueError(
                "Parameter.shape != Parameter.prior.shape for...\n"
                f"Parameter:       {self.name}\n"
                f"Parameter.shape: {self.shape}\n"
                f"prior.shape:     {prior.shape}"
                )

        # set default based on `default_bounds`
        if bounds is None:
            if default_bounds == 'percentile':
                bounds = (prior.percentile(0.0001), prior.percentile(0.9999))
            elif default_bounds == 'infinite':
                bounds = (-np.inf, np.inf)
            else:
                raise ValueError(
                    "Unrecognized default_bounds for...\n"
                    f"Parameter:      {self.name}\n"
                    f"default_bounds: {default_bounds}\n"
                    "Accepted values are 'percentile' or 'infinite'."
                    )
        self.bounds = bounds

        if isinstance(initial_value, str) and (initial_value == 'mean'):
            value = self.prior.mean()
        elif isinstance(initial_value, str) and (initial_value) == 'sample':
            value = self.prior.sample(bounds=self.bounds)
        elif np.isscalar(initial_value):
            value = np.full(self.shape, initial_value)
        elif isinstance(initial_value, np.ndarray):
            value = initial_value
        else:
            raise ValueError(
                "Unrecognized initial_value for...\n"
                f"Parameter:     {self.name}\n"
                f"initial_value: {initial_value}\n"
                "Accepted values are 'mean', 'sample', scalar, or ndarray."
                )

        self.initial_value = np.ravel(value)

        # Must be set by `Phi.add_parameter` for `Parameter.values` to work.
        self.phi = None  # Pointer to parent Phi instance.
        self.first_index = None  # Location of data within Phi.get_vector()
        self.last_index = None

        self.is_frozen = False
    
    # TODO: any other tracking/upkeep that needs to happen with
    #       freezing/unfreezing, or is the flag sufficient?
    def freeze(self):
        """Use parameter values for evaluation only, do not optimize."""
        self.is_frozen = True

    def unfreeze(self):
        """Make parameter values optimizable."""
        self.is_frozen = False

    @property
    def is_fittable(self):
        """Alias property for negation of `Parameter.is_frozen`."""
        return not self.is_frozen

    @property
    def values(self):
        """Get corresponding parameter values stored by parent Phi."""
        values = self.phi._get_parameter_vector(self)
        return np.reshape(values, self.shape)

    def sample(self, n=1, inplace=False):
        """Get or set new values by sampling from `Parameter.prior`.

        Re-sampling is used to ensure that the final sample falls within
        `Parameter.bounds`.
        
        Parameters
        ----------
        n : int
            Number of samples to get.
        inplace : bool
            If True, sampled values will be used to update `Parameter.values`
            inplace. Otherwise, the sampled values will be returned without
            changing current values.

        See also
        --------
        nems.layers.base.Phi.sample
        nems.distributions.base.Distribution.sample

        Return
        ------
        sample : ndarray

        """

        sample = self.prior.sample(n=n, bounds=self.bounds)
        if inplace:
            self.update(sample)
        return sample

    def mean(self, inplace=False):
        """Get, or set parameter values to, mean of `Parameter.prior`.
        
        Note that `Parameter.mean()` may return a value outside of
        `Parameter.bounds`. In that event, either priors or bounds should be
        changed.

        Parameters
        ----------
        inplace : bool, default=False
            If True, mean value will be used to update `Parameter.values`
            inplace Otherwise, mean will be returned without changing current
            values.

        Return
        ------
        mean : ndarray

        """
        mean = self.prior.mean()
        if inplace:
            self.update(mean)

        return mean

    def update(self, value, ignore_bounds=False):
        """Set `Parameters.values` to `value` by updating `Phi._array`.
        
        Parameters
        ----------
        value : scalar or array-like
            New value for `Parameter.values`. Must match `Parameter.shape`.
        ignore_bounds : bool
            If True, ignore `Parameter.bounds` when updating. Otherwise,
            new values will be rejected if they are less than `bounds[0]` or
            greater than `bounds[1]`.
        
        """
        value = np.asarray(value)
        if not ignore_bounds:
            lower, upper = self.bounds
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError(
                    f"value out-of-bounds for...\n"
                    f"Parameter: {self.name}\n"
                    f"Bounds:    {self.bounds}\n"
                    f"Value:     {value}"
                )

        if np.shape(value) != self.shape:
            raise ValueError(
                f"Parameter {self.name} requires shape {self.shape}, but "
                f"{value} has shape {np.shape(value)}"
            )
        else:
            flat_value = np.ravel(value)
            mask = self.phi._get_parameter_mask(self)
            self.phi._array[mask] = flat_value


    # Add compatibility for saving to .json    
    def to_json(self):
        """Encode Parameter object as json. See `nems.tools.json`."""
        data = {
            'kwargs': {
                'name': self.name,
                'shape': self.shape,
                'prior': self.prior,
                'bounds': self.bounds
            },
            'attributes': {
                'is_frozen': self.is_frozen
            }
        }
        return data

    def from_json(json):
        """Decode Parameter object from json. See `nems.tools.json`."""
        p = Parameter(**json['kwargs'])
        for k, v in json['attributes'].items():
            setattr(p, k, v)
        return p

    def __repr__(self):
        dash_break = "-"*16 + "\n"
        string = f"Parameter(name={self.name}, shape={self.shape})\n"
        string += dash_break
        string += f".prior:     {self.prior}\n"
        string += f".bounds:    {self.bounds}\n"
        string += f".is_frozen: {self.is_frozen}\n"
        string += ".values:\n"
        string += f"{self.values}\n"
        string += dash_break
        return string

    # Add compatibility with numpy ufuncs, len(), and other methods that
    # should point to `Parameter.values` instead of `Parameter`.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Propagate numpy ufunc operations to `Parameter.values`.
        
        Convenience method to eliminate excessive references to
        `Parameter.values`.
        
        Notes
        -----
        Works with `@` but not `np.dot`.

        See also
        --------
        https://numpy.org/doc/stable/reference/ufuncs.html
        
        """
        f = getattr(ufunc, method)
        # replace Parameter objects with Parameter.values
        subbed_inputs = [
            x.values if isinstance(x, Parameter) else x
            for x in inputs
            ]
        output = f(*subbed_inputs, **kwargs)

        return output

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return self.values.__iter__()
