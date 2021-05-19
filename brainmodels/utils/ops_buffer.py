# -*- coding: utf-8 -*-

import brainpy as bp

# NumPy
import numpy as np
bp.ops.set_buffer('numpy', {'clip': np.clip})
bp.ops.set_buffer('numpy', {'mean': np.mean})

# PyTorch
try:
    import torch

    try:
        bp.ops.set_buffer('pytorch', clip=torch.clamp, mean=torch.mean)
    except AttributeError:
        pass

except ModuleNotFoundError:
    pass


# TensorFlow
try:
    import tensorflow as tf

    try:
        bp.ops.set_buffer('tensorflow', clip=tf.clip_by_value, mean=tf.mean)
    except AttributeError:
        pass

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

    bp.ops.set_buffer('numba', clip=nb_clip, mean=np.mean)
    bp.ops.set_buffer('numba-parallel', clip=nb_clip, mean=np.mean)

except ModuleNotFoundError:
    pass

