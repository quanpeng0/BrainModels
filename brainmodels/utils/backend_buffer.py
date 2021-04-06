# -*- coding: utf-8 -*-


import numpy as np
np_clip = np.clip

try:
    import torch
    torch_clip = torch.clamp
except ModuleNotFoundError:
    torch_clip = None

try:
    import tensorflow as tf
    tf_clip = tf.clip_by_value
except ModuleNotFoundError:
    tf_clip = None

try:
    import numba as nb

    @nb.njit
    def nb_clip(x, x_min, x_max):
        x = np.maximum(x, x_min)
        x = np.minimum(x, x_max)
        return x
except ModuleNotFoundError:
    nb_clip = None


import brainpy as bp

bp.backend.set_buffer('numpy', {'clip': np_clip})
bp.backend.set_buffer('numba', {'clip': nb_clip})
bp.backend.set_buffer('pytorch', {'clip': torch_clip})
bp.backend.set_buffer('tensorflow', {'clip': tf_clip})
