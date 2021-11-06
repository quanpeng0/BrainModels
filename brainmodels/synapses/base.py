# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from brainmodels.neurons.base import Neuron

__all__ = [
  'Synapse'
]


class Synapse(bp.TwoEndConn):
  def __init__(self, pre, post, conn, method='euler', **kwargs):
    super(Synapse, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    if not isinstance(pre, Neuron):
      raise bp.errors.BrainPyError(f'"pre" must be an instance of {Neuron}.')
    if not isinstance(post, Neuron):
      raise bp.errors.BrainPyError(f'"post" must be an instance of {Neuron}.')
    self.pre = pre
    self.post = post

    # connections
    self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
    self.num = len(self.pre_ids)

    # integrals
    self.integral = bp.odeint(method=method, f=self.derivative)

    # functions
    if bm.is_numpy_backend():
      self.steps.replace('update', self.numpy_update)
      self.target_backend = 'numpy'

    elif bm.is_jax_backend():
      self.steps.replace('update', self.jax_update)
      self.target_backend = 'jax'

    else:
      raise bp.errors.UnsupportedError(
        f'Do not support {bm.get_backend_name()} backend '
        f'for synapse model {self}.')

  def derivative(self, *args, **kwargs):
    raise NotImplementedError

  def numpy_update(self, _t, _dt):
    raise NotImplementedError

  def jax_update(self, _t, _dt):
    raise NotImplementedError
