# -*- coding: utf-8 -*-

import brainpy.math as bm
from .base import Synapse

__all__ = [
  'VoltageJump'
]


class VoltageJump(Synapse):
  """Voltage jump synapse model.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} w \delta(t-t_j-D)

  where :math:`w` denotes the chemical synaptic strength, :math:`t_j` the spiking
  moment of the presynaptic neuron :math:`j`, :math:`C` the set of neurons connected
  to the post-synaptic neuron, and :math:`D` the transmission delay of chemical
  synapses. For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.

  **Model Examples**

  - `Simple illustrated example <../synapses/voltage_jump.ipynb>`_
  - `(Bouchacourt & Buschman, 2019) Flexible Working Memory Model <../../examples/working_memory/Bouchacourt_2019_Flexible_working_memory.ipynb>`_

  **Model Parameters**

  ============= ============== ======== ===========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------
  w             1              mV       The synaptic strength.
  ============= ============== ======== ===========================================

  """

  def __init__(self, pre, post, conn, delay=0., post_has_ref=False, w=1.,
               post_key='V', **kwargs):
    super(VoltageJump, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # checking
    if post_has_ref:
      assert hasattr(post, 'refractory'), 'Post-synaptic group must has "refractory" variable.'
    if bm.is_numpy_backend():
      assert post_key == 'V', f'"NumPy" backend only supports "V"'

    # parameters
    self.delay = delay
    self.post_key = post_key
    self.post_has_ref = post_has_ref

    # variables
    self.w = w
    assert bm.size(w) == 1, 'This implementation only support scalar weight. '
    self.pre_spike = self.register_constant_delay('pre_spike', size=self.pre.num, delay=delay)

  def derivative(self, a, t):
    pass

  def numpy_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    for i in range(self.num):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      if delayed_pre_spike[pre_id]:
        if self.post_has_ref:
          self.post.V[post_id] += self.w * (1. - self.post.refractory[post_id])
        else:
          self.post.V[post_id] += self.w

  def jax_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    fsp = bm.vmap(lambda pre_i, pre_sp: pre_sp[pre_i], in_axes=(0, None))
    spikes = fsp(self.pre_ids.value, delayed_pre_spike)
    post_sp = bm.segment_sum(spikes, self.post_ids, self.post.num)
    target = getattr(self.post, self.post_key)
    if self.post_has_ref:
      target += self.w * post_sp * (1. - self.post.refractory)
    else:
      target += self.w * post_sp
