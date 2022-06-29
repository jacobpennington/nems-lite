import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Serialize Numpy arrays safely as JSONs. 
    
    References
    ----------
    .. [1] Stack overflow answer by user 'tlausch' at:
       https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array

    """

    def default(self, obj):
        """If input object is an ndarray, convert it to a dict.

        Data, shape, and dtype will be preserved. Data is encoded as a list,
        which makes it text-readable.

        """

        # TODO: special cases for distributions, layers, etc

        if isinstance(obj, np.ndarray):
            data = obj.tolist()
            encoded_obj = dict(
                __ndarray__=data,
                dtype=str(obj.dtype),
                shape=obj.shape,
                encoding='list'
                )

        to_json_exists = getattr(obj, "to_json", None)
        if callable(to_json_exists):
            encoded_obj = obj.to_json()
        else:
            # Let the base class default method raise the TypeError
            encoded_obj = json.JSONEncoder.default(self, obj)

        return encoded_obj

def json_numpy_obj_hook(dct):
    """Decode a previously encoded numpy.ndarray with proper shape and dtype."""

    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = dct['__ndarray__']
        dct = np.asarray(data, dct['dtype']).reshape(dct['shape'])

    return dct
