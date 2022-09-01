import pytest
import numpy as np

from nems import Model
from nems.models.dataset import DataSet
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems.layers.base import Layer


def test_constructor():
    # No required args
    model = Model()
    assert len(model.layers) == 0

    # Should be able to specify name, meta, layers
    layer = WeightChannels(shape=(5,1))
    name = 'test'
    meta = {'testing': 123}
    model2 = Model(layers=[layer], name=name, meta=meta)
    assert model2.layers[0] == layer
    assert model2.name == name
    assert model2.meta == meta


def test_add_layers():
    layers = [WeightChannels(shape=(10,2)), FiniteImpulseResponse(shape=(5,2))]
    model1 = Model(layers=layers)
    model2 = Model()
    model2.add_layers(*layers)
    # same as constructor?
    assert model1 == model2
    # all layers?
    assert len(model2.layers) == 2
    # correct order?
    assert model2.layers[0] == layers[0]
    assert model2.layers[1] == layers[1]


def test_indexing():
    model = Model()
    model.add_layers(
        Layer(name='zero', input='zero'), Layer(name='one', input='one'),
        Layer(name='two', input='two')
    )

    # Index int
    assert model.layers[0].name == 'zero'
    assert model.layers[-1].name == 'two'
    # Multiple ints
    assert model.layers[0, 1][1].name == 'one'
    # String
    assert model.layers['zero'].input == 'zero'
    assert len(model.layers['one', 'two']) == 2
    # Slice
    assert len(model.layers[0:None]) == 3


def test_get_set():
    wc_zero = np.zeros(shape=(5, 3))
    fir_one = np.ones(shape=(10, 3))
    wc = WeightChannels(shape=(5, 3), name='wc')
    fir = FiniteImpulseResponse(shape=(10, 3), name='fir')

    model = Model(layers=[wc, fir]).sample_from_priors()
    values = model.get_parameter_values()
    # Fetching the correct values?
    assert np.allclose(values['wc']['coefficients'],
                       model.layers['wc'].get_parameter_values())
    assert np.allclose(values['fir']['coefficients'],
                       model.layers['fir'].get_parameter_values())

    model.set_parameter_values(
        {'wc': {'coefficients': wc_zero}, 'fir': {'coefficients': fir_one}}
        )
    # Setting correct values?
    assert np.allclose(model.layers['wc'].get_parameter_values(), wc_zero)
    assert np.allclose(model.layers['fir'].get_parameter_values(), fir_one)


class TestEvaluate:

    def test_evaluate(self, spectrogram, state):
        time, spectral = spectrogram.shape
        _, n_states = state.shape
        # Basic sequential model, each layer should pass output to the next one
        # and state should be automatically added to inputs for stategain.
        model = Model.from_keywords(
            f'wc.{spectral}x1-fir.15x1-dexp.1-stategain.{n_states}x1'
            )
        out = model.evaluate(spectrogram, state=state)
        # Should be one output, same number of time bins
        assert out[DataSet.default_output].shape == (time, 1)
        # state kwarg gets mapped automatically since StateGain.input is None.
        assert model.layers[-1].data_map.kwargs == \
            {'state': DataSet.default_state}

    def test_evaluate_batches(self, spectrogram, state_with_samples):
        # Prepend singleton sample axis.
        # This should broadcast to number of state samples.
        spectrogram = spectrogram[np.newaxis, ...]
        spectral = spectrogram.shape[-1]
        samples, time, n_states = state_with_samples.shape
        model = Model.from_keywords(
            f'wc.{spectral}x1-fir.15x1-dexp.1-stategain.{n_states}x1'
            )

        with pytest.raises(Exception):
            # default batch_size=0 not compatible with sample dimension.
            model.evaluate(spectrogram, state=state_with_samples)

        out1 = model.evaluate(spectrogram, state=state_with_samples,
                              batch_size=None)
        assert out1[DataSet.default_output].shape == (samples, time, 1)
        # Even though spectrogram broadcasts, the original shape should be
        # preserved in the output.
        assert out1[DataSet.default_input].shape == (1, time, spectral)


def test_fit(spectrogram, response, spectrogram_with_samples,
             response_with_samples):
    pass  # TODO


def test_freeze_layers(spectrogram):
    time, spectral = spectrogram.shape
    wc = WeightChannels(shape=(spectral, 3))
    fir = FiniteImpulseResponse(shape=(15, 3, 1))
    dexp = DoubleExponential(shape=(1,))

    model = Model(layers=[wc, fir, dexp])
    p_count = model.parameter_count
    assert p_count == (wc.parameter_count + fir.parameter_count
                       + dexp.parameter_count)
    assert p_count == spectral*3 + 15*3 + 4
    p_info = model.parameter_info
    assert p_info['model']['frozen'] == 0
    assert p_info['model']['unfrozen'] == p_count

    out = model.predict(spectrogram)
    model.freeze_layers()
    frozen_out = model.predict(spectrogram)
    # Total count shouldn't change.
    # Frozen/unfrozen counts should.
    assert model.parameter_count == p_count
    assert model.parameter_info['model']['frozen'] == p_count
    assert model.parameter_info['model']['unfrozen'] == 0
    # Parameter vector should not include frozen parameters.
    assert len(model.get_parameter_vector(as_list=True)) == 0
    # Evaluation shouldn't change.
    assert np.allclose(out, frozen_out)

    model.unfreeze_layers()
    unfrozen_out = model.predict(spectrogram)
    # Total count shouldn't change.
    # Frozen/unfrozen counts should.
    assert model.parameter_count == p_count
    assert model.parameter_info['model']['frozen'] == 0
    assert model.parameter_info['model']['unfrozen'] == p_count
    # Parameter vector should include everything again.
    assert len(model.get_parameter_vector(as_list=True)) == p_count
    # Evaluation shouldn't change.
    assert np.allclose(out, unfrozen_out)

