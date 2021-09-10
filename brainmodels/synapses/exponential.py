# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'ExponentialCUBA', 'ExponentialCOBA'
]


class ExponentialCUBA(bp.TwoEndConn):
  r"""Current-based single exponential decay synapse model.

  .. math::

       \frac{d s}{d t} = -\frac{s}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k})

  For conductance-based (co-base=True):

  .. math::

      I_{syn}(t) = g_{syn} (t) (V(t)-E_{syn})


  For current-based (co-base=False):

  .. math::

      I(t) = \bar{g} s (t)


  **Synapse Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  tau_decay     8.             ms       The time constant of decay.

  ============= ============== ======== ===================================================================================

  **Synapse State**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  s                  0                  Gating variable.

  weight           0                  Synaptic weights.

  ================ ================== =========================================================

  References
  ----------

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0,
               update_type='loop_slice', **kwargs):
    super(ExponentialCUBA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # checking
    assert hasattr(pre, 'spike'), 'Pre-synaptic group must has "spike" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

    # parameters
    self.tau = tau
    self.delay = delay

    # connections
    if update_type == 'loop_slice':
      self.pre_slice = self.conn.requires('pre_slice')
      self.update = self._loop_slice_update
      self.size = self.post.num
      self.target_backend = 'numpy'

    elif update_type == 'loop':
      self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
      self.update = self._loop_update
      self.size = len(self.pre_ids)
      self.target_backend = 'numpy'

    elif update_type == 'matrix':
      self.conn_mat = self.conn.requires('conn_mat')
      self.update = self._matrix_update
      self.size = self.conn_mat.shape

    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # variables
    self.g_max = g_max
    assert bp.math.size(g_max) == 1, 'This implementation only support scalar "g_max". '
    self.g = bp.math.Variable(bp.math.zeros(self.size))
    self.pre_spike = self.register_constant_delay('pre_spike', self.size, delay)

  @bp.odeint(method='exponential_euler')
  def integral(self, s, t):
    ds = -s / self.tau
    return ds

  def output_current(self, g):
    return g

  def _loop_slice_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()

    self.g[:] = self.integral(self.g, _t, dt=_dt)
    for pre_id in range(self.pre.num):
      if pre_spike[pre_id]:
        start, end = self.pre_slice[pre_id]
        for post_id in self.post_ids[start: end]:
          self.g[post_id] += self.g_max

    self.post.input[:] += self.output_current(self.g)

  def _loop_update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    pre_spike = self.pre_spike.pull()

    self.g[:] = self.integral(self.g, _t, dt=_dt)
    for pre_i in range(self.pre.num):
      if pre_spike[pre_i]:
        start, end = self.pre_slice[pre_i]
        for post_i in self.post_ids[start: end]:
          self.g[post_i] += self.g_max

    self.post.input[:] += self.output_current(self.g)

  def _matrix_update(self, _t, _dt):
    raise NotImplementedError


class ExponentialCOBA(ExponentialCUBA):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               update_type='loop_slice', **kwargs):
    super(ExponentialCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                          g_max=g_max, delay=delay, tau=tau,
                                          update_type=update_type, **kwargs)

    self.E = E
    assert hasattr(self.post, 'V'), 'Post-synaptic group must has "V" variable.'

  def output_current(self, g):
    return g * (self.E - self.post.V)
