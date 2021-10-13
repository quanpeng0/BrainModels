# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'VoltageJump'
]


class VoltageJump(bp.TwoEndConn):
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

  def __init__(self, pre, post, conn, delay=0., post_has_ref=False,
               w=1., update_type='sparse', **kwargs):
    super(VoltageJump, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # checking
    assert hasattr(pre, 'spike'), 'Pre-synaptic group must has "spike" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'
    if post_has_ref:
      assert hasattr(post, 'refractory'), 'Post-synaptic group must has "refractory" variable.'

    # parameters
    self.delay = delay
    self.post_has_ref = post_has_ref

    # connections
    if update_type == 'sparse':
      self.pre_slice, self.post_ids = self.conn.requires('pre_slice', 'post_ids')
      self.steps.replace('update', self.sparse_update)
      self.size = self.post.num
      self.target_backend = 'numpy'

    elif update_type == 'loop':
      self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
      self.steps.replace('update', self._loop_update)
      self.size = len(self.pre_ids)
      self.target_backend = 'numpy'

    elif update_type == 'dense':
      self.conn_mat = self.conn.requires('conn_mat')
      self.steps.replace('update', self.dense_update)
      self.size = self.conn_mat.shape

    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # variables
    self.w = w
    assert bm.size(w) == 1, 'This implementation only support scalar weight. '
    self.pre_spike = self.register_constant_delay('pre_spike', size=self.pre.shape, delay=delay)

  def sparse_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()

    spike_pres = bm.where(pre_spike)[0]
    for pre_id in spike_pres:
      start, end = self.pre_slice[pre_id]
      post_ids = self.post_ids[start: end]
      if self.post_has_ref:
        self.post.V[post_ids] += self.w * (1. - self.post.refractory[post_ids])
      else:
        self.post.V[post_ids] += self.w

  def _loop_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()
    for i in range(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      if pre_spike[pre_id]:
        if self.post_has_ref:
          self.post.V[post_id] += self.w * (1. - self.post.refractory[post_id])
        else:
          self.post.V[post_id] += self.w

  def dense_update(self, _t, _dt):
    raise NotImplemented
