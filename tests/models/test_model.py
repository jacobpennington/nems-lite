import numpy as np

from nems import Model
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
