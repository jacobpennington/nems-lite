"""A collection of miscellaneous utility functions.

TODO: Anything that used to be in nems.utils.py should go in this directory,
      with functions of similar purpose grouped in modules. Even if there's just
      one function that belongs in a solo group, it should *not* just be dumped
      in 'misc' or 'etc' or a similarly named file with other one-offs.

      Alternatively, if all utilites can be reasonably assigned to specific
      libraries, we can just get rid of this directory altogether.


Contents
--------
    `json.py` : Ensure proper json-ification of Models, Layers, etc.
    # TODO: may end up with a separate plotting library, in which case these
    #       tools should be moved there.
    `plotting.py` : Miscellaneous tools for assisting with visualization.

"""

from .json import nems_to_json, nems_from_json, NEMSEncoder, NEMSDecoder
