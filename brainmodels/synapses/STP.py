# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'STP'
]


class STP(bp.TwoEndConn):
  r"""Short-term plasticity proposed by Tsodyks and Markram (Tsodyks 98) [1]_.

  The model is given by

  .. math::

      \begin{aligned}
      \frac{du}{dt} & = & -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}),\nonumber \\
      \frac{dx}{dt} & = & \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
      \frac{dI}{dt} & = & -\frac{I}{\tau_s} + Au^+x^-\delta(t-t_{sp}),
      \end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike.

  The synaptic current generated at the synapse by the spike arriving
  at :math:`t_{sp}` is then given by

  .. math::

      \Delta I(t_{spike}) = Au^+x^-

  where :math:`A` denotes the response amplitude that would be produced
  by total release of all the neurotransmitter (:math:`u=x=1`), called
  absolute synaptic efficacy of the connections.


  **Synapse Parameters**

  ============= ============== ======== ===========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------
  tau_d         200.           ms       Time constant of short-term depression.

  tau_f         1500.          ms       Time constant of short-term facilitation.

  U             .15            \        The increment of :math:`u` produced by a spike.
  ============= ============== ======== ===========================================

  **Synapse State**

  ST refers to the synapse state, items in ST are listed below:

  =============== ================== =====================================================================
  **Member name** **Initial values** **Explanation**
  --------------- ------------------ ---------------------------------------------------------------------
  u                 0                 Release probability of the neurotransmitters.

  x                 1                 A Normalized variable denoting the fraction of remain neurotransmitters.

  w                 1                 Synapse weight.

  g                 0                 Synapse conductance.
  =============== ================== =====================================================================

  References:
  .. [1] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
          with dynamic synapses." Neural computation 10.4 (1998): 821-835.
  """

  def __init__(self, pre, post, conn, delay=0., U=0.15, tau_f=1500., tau_d=200.,
               tau=8., w=0.1, update_type='loop', **kwargs):
    super(STP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    assert hasattr(pre, 'spike'), 'Pre-synaptic group must has "spike" variable.'
    assert hasattr(post, 'V'), 'Post-synaptic group must has "V" variable.'
    assert hasattr(post, 'input'), 'Post-synaptic group must has "input" variable.'

    # parameters
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.delay = delay
    self.w = w

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
    self.I = bp.math.Variable(bp.math.zeros(self.size))
    self.x = bp.math.Variable(bp.math.ones(self.size))
    self.u = bp.math.Variable(bp.math.zeros(self.size))

  @bp.odeint(method='exponential_euler')
  def integral(self, I, u, x, t):
    dudt = - u / self.tau_f
    dxdt = (1 - x) / self.tau_d
    dIdt = -I / self.tau
    return dIdt, dudt, dxdt

  def _loop_update(self, _t, _dt):
    self.I[:], u, x = self.integral(self.I, self.u, self.x, _t, dt=_dt)
    for i in range(self.size):
      pre_id, post_id = self.pre_ids[i], self.post_ids[i]
      if self.pre.spike[pre_id] > 0:
        u[i] += self.U * (1 - self.u[i])
        x[i] -= u[i] * self.x[i]
        self.I[i] += self.w * u[i] * self.x[i]
      self.post.input[post_id] += self.I[i]
    self.u[:] = u
    self.x[:] = x
