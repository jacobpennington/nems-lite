"""Implements DataSet container for use by Model.evaluate.

TODO: May be a better place to put this?

"""

import numpy as np


# TODO: remove/make optional all the asser(shares_memory) calls, to speed up
#       evaluation. still want something in place for debugging, but they don't 
#       need to be checked every time.
class DataSet:
    # Use these names if `input_name`, `state_name`, etc. are not specified
    # in DataSet.__init__.
    default_input = 'input'
    default_output = 'output'
    default_target = 'target'
    default_state = 'state'

    def __init__(self, input, state=None, target=None, input_name=None,
                 state_name=None, output_name=None, target_name=None,
                 dtype=np.float64, debug_memory=False, **kwargs):
        """Container for tracking dictionaries of data arrays.
        
        See `Model.evaluate` and `Model.fit` for detailed documentation of
        parameters.

        Parameters
        ----------
        input : ndarray or dict.
        state : ndarray; optional.
        target : ndarray or dict; optional.
        input_name : str; optional.
        state_name : str; optional.
        output_name : str; optional.
        target_name : str; optional.
        dtype : type; default=np.float64.
            TODO: WIP. Want to specify a consistent datatype to cast all
                  arrays to.
        debug_memory : bool; default=False.
            TODO: Not actually implemented yet.
            If True, check `np.shares_memory(array, copied_array)` in several
            places to ensure that memory is not duplicated unintentionally.
        kwargs : dict; optional.
            Extra kwargs are silently ignored for convenience, so that other
            code can use `DataSet(input, **evaluate_kwargs)`.
        
        Attributes
        ----------
        inputs : dict.
            All model inputs, stored as `{key: ndarray}`.
        outputs : dict.
            All saved Layer outputs, stored in the same format. Special key
            '_last_output' will contain the most recent Layer output during
            evaluation, but this key is removed by `DataSet.finalize_data`.
            Will be empty if `save_output` has never been called.
        targets : dict.
            All optimization targets, stored in the same format. Will be empty
            if `target is None`.

        """
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

    def initialize_data(self, input, state=None, target=None):
        """Package data into dictionaries, store in attributes.
        
        Assigns data to `DataSet.inputs, DataSet.outputs, DataSet.targets`.

        Parameters
        ----------
        input : ndarray or dict.
        state : ndarray; optional.
        target : ndarray or dict; optional.
        
        """

        # Initialize inputs
        if isinstance(input, np.ndarray):
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
        elif isinstance(target, np.ndarray):
            target_dict = {self.target_name: target}
        else:
            target_dict = target.copy()

        self.inputs = input_dict
        self.outputs = output_dict
        self.targets = target_dict

    def save_output(self, keys, output):
        """Save `output` in `DataSet.outputs` for keys that are not `None`.
        
        Parameters
        ----------
        keys : list of str.
            Indicates keys in `DataSet.outputs` where array(s) in `output`
            should be stored. If a key is None, the output is not saved
            (except at special key `_last_output`).
        output : ndarray or list of ndarray.
            Output of a call to Layer._evaluate.
        
        """
        # keys is always a list, but output might be a list or one array.
        if isinstance(output, (list, tuple)):
            self.outputs.update({k: v for k, v in zip(keys, output)
                        if k is not None})
        elif keys[0] is not None:
            self.outputs[keys[0]] = output
        # Always save output to _last_output for use by Model.evaluate
        self.outputs['_last_output'] = output

    def finalize_data(self, final_layer):
        """Remove special keys from `DataSet.outputs` and update final_layer.
        
        If `final_layer.output is None`, save the Layer's output to
        `DataSet.output_name` instead and update `Layer.data_map` to reflect
        the new key(s).
        
        """
        # Re-name last output if keys not specified by Layer
        final_output = self.outputs['_last_output']
        if final_layer.output is None:
            # Re-map `data_map.out` so that it reflects `output_name`.
            final_layer.data_map.map_outputs(final_output, self.output_name)
            self.save_output(final_layer.data_map.out, final_output)
        _ = self.outputs.pop('_last_output')

    def as_broadcasted_samples(self):
        """Broadcasts all data arrays against each other for number of samples.
        
        Arrays with shape (1, T, ..., N), where T is the number of time bins
        and N is the number of output channels, will broadcast to shape
        `(S, T, ..., N)`, where S is the number of samples in other data arrays.
        First all inputs are broadcast against other inputs (and outputs against
        other outputs, etc) and then inputs are broadcast against outputs and
        targets, outputs against inputs and targets, etc.

        Examples
        --------
        >>> stimulus = np.random.rand(1, 1000, 18)  # one stimulus sample
        >>> response = np.random.rand(10, 1000, 18) # multiple trials
        >>> data = DataSet(input=stimulus, target=response)
        >>> broadcasted = data.as_broadcasted_samples()
        >>> broadcasted.inputs['stimulus'].shape
        (10, 1000, 18)
        >>> np.shares_memory(broadcasted.inputs['stimulus'], stimulus)
        True
        
        """

        # In case inputs/outputs and targets have different numbers of samples,
        # broadcast within each category first. 
        inputs = self._broadcast_dict(self.inputs, self.inputs, same=True)
        outputs = self._broadcast_dict(self.outputs, self.outputs, same=True)
        targets = self._broadcast_dict(self.targets, self.targets, same=True)

        # Then broadcast each category to the others.
        inputs = self._broadcast_dict(inputs, {**outputs, **targets})
        outputs = self._broadcast_dict(outputs, {**inputs, **targets})
        targets = self._broadcast_dict(targets, {**inputs, **outputs})

        return self.modified_copy(inputs, outputs, targets)

    # TODO: maybe document this as a public method, or move to general utilities?
    #       Could be useful elsewhere.
    @staticmethod
    def _broadcast_dict(d1, d2, same=False):
        """Internal for broadcast_samples."""
        if (len(d1) == 0) or (len(d1) == 1 and same) or (len(d2) == 0):
            # Nothing to broadcast to
            new_d = d1.copy()
        else:
            new_d = {}
            for k, v in d1.items():
                temp = d2.copy()
                if k in temp:
                    temp.pop(k)  # don't need to broadcast to self
                for v2 in temp.values():
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

    def as_batches(self, batch_size=None, permute=False):
        """Generate copies of DataSet containing single batches.
        
        Parameters
        ----------
        batch_size : int; optional.
            Number of samples to include in each batch. If the total number of
            samples is not evenly divisble by `batch_size`, then one batch will
            have fewer samples. If `batch_size` is None, `batch_size` will be
            set equal to the total number of samples.
        permute : bool; default=False.
            TODO: Not implemented yet.
            If True, shuffle yield order of batches.

        Yields
        ------
        DataSet

        Notes
        -----
        This implementation results in a list of views into the original data
        (i.e. memory is shared). If changes are made, make sure the new version
        doesn't result in copies (which could increase memory usage
        dramatically). 
        
        """

        # TODO: This handles cases of 1 sample -> many samples or vise-versa,
        #       but do we want to support n samples -> m samples? Would have
        #       to do some kind of tiling that may have side-effects that I'm
        #       not thinking of, and I'm not sure if this would be useful.
        d = self.as_broadcasted_samples()
        
        # Split data into batches along first axis. Should end up with a list
        # of arrays with shape (B, T, N), where B is `batch_size` (i.e. number
        # of samples per batch), stored at each key.
        batched_inputs, batched_outputs, batched_targets = [
            d._arrays_to_batches(_dict, batch_size)
            for _dict in [d.inputs, d.outputs, d.targets]
        ]

        n_batches = len(list(batched_inputs.values())[0])

        # Index into batched_data instead of collecting a list of batches,
        # to ensure memory is still shared. Also makes permutations easier.
        indices = np.arange(n_batches)
        if permute:
            # Randomly shuffle indices
            np.random.shuffle(indices)
            # TODO: Not quite this simple, have to be able to put the concatenated
            #       outputs back in the right order. So need to store the shuffled
            #       indices somehow.
            raise NotImplementedError("Shuffling batches not implemented yet")

        for i in indices:
            inputs = {k: v[i] for k, v in batched_inputs.items()}
            outputs = {k: v[i] for k, v in batched_outputs.items()}
            targets = {k: v[i] for k, v in batched_targets.items()}
            d.assert_no_copies(inputs, outputs, targets)
            yield d.modified_copy(inputs, outputs, targets)

    def _arrays_to_batches(self, data, batch_size):
        """Internal for `as_batches`.
        
        Parameters
        ----------
        data : dict
        batch_size : int

        Returns
        -------
        dict
        
        """

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

    def copy(self):
        return self.modified_copy(self.inputs, self.outputs, self.targets)

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
