# -*- coding: utf-8 -*-

import brainpy as bp

# NumPy
import numpy as np
bp.ops.set_buffer('numpy', {'clip': np.clip})
bp.ops.set_buffer('numpy', {'mean': np.mean})

# PyTorch
try:
    import torch

    bp.ops.set_buffer('pytorch', {'clip': torch.clamp})
    bp.ops.set_buffer('pytorch', {'mean': torch.mean})

except ModuleNotFoundError:
    pass


# TensorFlow
try:
    import tensorflow as tf

    bp.ops.set_buffer('tensorflow', {'clip': tf.clip_by_value})
    bp.ops.set_buffer('tensorflow', {'mean': tf.mean})

except ModuleNotFoundError:
    pass


# Numba
try:
    import numba as nb

    @nb.njit
    def nb_clip(x, x_min, x_max):
        x = np.maximum(x, x_min)
        x = np.minimum(x, x_max)
        return x

    bp.ops.set_buffer('numba', {'clip': nb_clip})
    bp.ops.set_buffer('numba-parallel', {'clip': nb_clip})
    bp.ops.set_buffer('numba', {'mean': np.mean})

except ModuleNotFoundError:
    pass

