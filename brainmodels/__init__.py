# -*- coding: utf-8 -*-

__version__ = "1.0.2"

import brainpy as bp
import numpy as np

from . import on_numba
from . import on_tensor

# NumPy

bp.ops.set_buffer('numpy', clip=np.clip, mean=np.mean)

# PyTorch
try:
    import torch

    try:
        all_ops = dict(
            clip=torch.clamp,
            mean=torch.mean
        )
    except AttributeError:
        all_ops = dict()

    bp.ops.set_buffer('pytorch', **all_ops)

except ModuleNotFoundError:
    pass

# TensorFlow
try:
    import tensorflow as tf

    try:
        all_ops = dict(
            clip=tf.clip_by_value,
            mean=tf.mean
        )
    except AttributeError:
        all_ops = dict()
    bp.ops.set_buffer('tensorflow', **all_ops)

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


    all_ops = dict(
        clip=nb_clip,
        mean=np.mean,
    )

    bp.ops.set_buffer('numba', **all_ops)
    bp.ops.set_buffer('numba-parallel', **all_ops)

except ModuleNotFoundError:
    pass
