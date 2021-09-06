# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'AMPA',
]


class AMPA(bp.TwoEndConn):
  """AMPA conductance-based synapse.
  
  .. math::

      I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

      \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

  **Synapse Parameters**
  
  ============= ============== ======== ================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------
  g_max         .42            µmho(µS) Maximum conductance.

  E             0.             mV       The reversal potential for the synaptic current.

  alpha         .98            \        Binding constant.

  beta          .18            \        Unbinding constant.

  T             .5             mM       Neurotransmitter concentration.

  T_duration    .5             ms       Duration of the neurotransmitter concentration.
  ============= ============== ======== ================================================

  **Synapse State**

  ================ ================== =========================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ -------------------------
  s                 0                 Gating variable.
  
  g                 0                 Synapse conductance.
  ================ ================== =========================

  References
  ----------
  .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
          and implications for stimulus processing[J]. Proceedings of the
          National Academy of Sciences, 2012, 109(45): 18553-18558.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0., alpha=0.98,
               beta=0.18, T=0.5, T_duration=0.5, update_type='loop', **kwargs):
    super(AMPA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # checking
    assert hasattr(pre, 't_last_spike'), 'Pre-synaptic group must has "t_last_spike" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

    # parameters
    self.delay = delay
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

    # connections
    if update_type == 'loop':
      self.pre_ids, self.post_ids = self.conn.requires('pre_ids', 'post_ids')
      self.update = self._loop_update
      self.size = len(self.pre_ids)
      self.target_backend = 'numpy'

    elif update_type == 'loop_slice':
      raise NotImplementedError

    elif update_type == 'matrix':
      raise NotImplementedError

    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # variables
    self.s = bp.math.Variable(bp.math.zeros(self.size))
    self.g = self.register_constant_delay('g', self.size, delay)

  @bp.odeint(method='exponential_euler')
  def int_s(self, s, t, TT):
    ds = self.alpha * TT * (1 - s) - self.beta * s
    return ds

  def _loop_update(self, _t, _dt):
    g_delayed = self.s.pull()
    for i in range(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      # output
      self.post.input[post_id] -= g_delayed[i] * (self.post.V[post_id] - self.E)
      # update
      TT = ((_t - self.pre.t_last_spike[pre_id]) < self.T_duration) * self.T
      self.s[i] = self.int_s(self.s[i], _t, TT, dt=_dt)
    self.g.push(self.g_max * self.s)
