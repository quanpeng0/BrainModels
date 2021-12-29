# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

from .base import Synapse

__all__ = [
  'VoltageJump',
  'DeltaSynapse',
]


class DeltaSynapse(Synapse):
  """Voltage Jump Synapse Model, or alias of Delta Synapse Model.

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

  def __init__(self, pre, post, conn, delay=0., post_has_ref=False, w=1., post_key='V', name=None):
    super(DeltaSynapse, self).__init__(pre=pre, post=post, conn=conn, build_integral=False, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs(post_key)
    if post_has_ref:
      self.check_post_attrs('refractory')

    # parameters
    self.delay = delay
    self.post_key = post_key
    self.post_has_ref = post_has_ref

    # connections
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.w = w
    self.pre_spike = bp.ConstantDelay(pre.num, delay, dtype=pre.spike.dtype)

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    post_vs = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.w)
    target = getattr(self.post, self.post_key)
    if self.post_has_ref:
      target += post_vs * (1. - self.post.refractory)
    else:
      target += post_vs


class VoltageJump(DeltaSynapse):
  def __init__(self, pre, post, conn, delay=0., post_has_ref=False, w=1., name=None):
    super(VoltageJump, self).__init__(pre=pre, post=post, conn=conn, delay=delay,
                                      post_has_ref=post_has_ref, w=w, name=name)


