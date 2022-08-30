import pytest

from nems import Model
from nems.tools.json import nems_to_json, nems_from_json, NEMSEncoder


def test_save_load():
    model = Model.from_keywords('wc.18x4', 'fir.4x15')
    encoded_model = nems_to_json(model)
    decoded_model = nems_from_json(encoded_model)
    assert decoded_model == model
