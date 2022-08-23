"""Implements DataSet container for use by Model.evaluate.

TODO: May be a better place to put this?

"""

import numpy as np


# TODO: move this to a new module
# TODO: update docs elsewhere to list correct type for `data`.
# TODO: revisit assumptions about data type. I think we need to just only
#       allow arrays in the data dictionaries to make things simpler. For lists,
#       store them at separate keys and use Layer.input to supply them as a list
#       if needed. Pretty sure Layer.output already gets treated like this anyway,
#       I just sort of lost track.
# TODO: remove/make optional all the asser(shares_memory) calls, to speed up
#       evaluation. still want something in place for debugging, but they don't 
#       need to be checked every time.
class DataSet:
    default_input = 'input'
    default_output = 'output'
    default_target = 'target'
    default_state = 'state'

    def __init__(self, input, state=None, target=None, dtype=np.float64,
                 input_name=None, state_name=None, output_name=None,
                 target_name=None, **kwargs):
        """TODO: docs.
        
        NOTE: extra kwargs are silently ignored for convenience.
        
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

    def initialize_data(self, input, state, target):
        """TODO: docs"""

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

    def finalize_data(self, final_layer):
        """TODO: docs"""
        # Re-name last output if keys not specified by Layer
        final_output = self.outputs['_last_output']
        if final_layer.output is None:
            # Re-map `data_map.out` so that it reflects `output_name`.
            final_layer.data_map.map_outputs(final_output, self.output_name)
            self.save_output(final_layer.data_map.out, final_output)
        _ = self.outputs.pop('_last_output')

    def as_broadcasted_samples(self):
        """TODO: docs"""
        # TODO: Iterate through all inputs, outputs, and targets and
        #       broadcast them against each other (on first dimension only)
        #       so that, for example:
        #       input (1, 100, 5), input2 (10, 100, 5), target (10, 100, 5)
        #       changes to
        #       input(10, 100, 5) ... (rest same)
        #       without duplicating memory.

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
        """TODO: docs, internal for broadcast_samples."""
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

    def as_batches(self, batch_size=None, permute=True):
        """TODO: docs"""

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
        # TODO: This handles cases of 1 sample -> many samples or vise-versa,
        #       but do we want to support n samples -> m samples? Would have
        #       to do some kind of tiling that may have side-effects that I'm
        #       not thinking of, and I'm not sure if this would be useful.
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
