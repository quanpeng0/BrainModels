# -*- coding: utf-8 -*-

__version__ = "1.0.1"

try:
    from . import numba_backend
except ModuleNotFoundError:
    pass

from . import tensor_backend
from .tensor_backend import neurons
from .tensor_backend import synapses
from .utils import ops_buffer


def set_backend(backend):
    global neurons
    global synapses

    if backend in ['tensor', 'numpy', 'pytorch', 'tensorflow', 'jax']:
        neurons = tensor_backend.neurons
        synapses = tensor_backend.synapses

    elif backend in ['numba', 'numba-parallel', 'numba-cuda']:
        neurons = numba_backend.neurons
        synapses = numba_backend.synapses

    else:
        raise ValueError(f'Unknown backend "{backend}".')
