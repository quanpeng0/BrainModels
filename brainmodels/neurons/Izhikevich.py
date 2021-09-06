# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'Izhikevich'
]


class Izhikevich(bp.NeuGroup):
  """The Izhikevich neuron model [1]_ [2]_.

  The dynamics are given by:

  .. math ::

      \\frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

      \\frac{d u}{d t} &=a(b V-u)

  .. math ::

      \\text{if}  v \\geq 30  \\text{mV}, \\text{then}
      \\begin{cases} v \\leftarrow c \\\\ u \\leftarrow u+d \\end{cases}

  **Neuron Parameters**

  ============= ============== ======== ================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------------------
  type          None           \        The neuron spiking type.

  a             0.02           \        It determines the time scale of

                                        the recovery variable :math:`u`.

  b             0.2            \        It describes the sensitivity of the

                                        recovery variable :math:`u` to

                                        the sub-threshold fluctuations of the

                                        membrane potential :math:`v`.

  c             -65.           \        It describes the after-spike reset value

                                        of the membrane potential :math:`v` caused by

                                        the fast high-threshold :math:`K^{+}`

                                        conductance.

  d             8.             \        It describes after-spike reset of the

                                        recovery variable :math:`u`

                                        caused by slow high-threshold

                                        :math:`Na^{+}` and :math:`K^{+}` conductance.

  t_refractory  0.             ms       Refractory period length. [ms]

  V_th          30.            mV       The membrane potential threshold.
  ============= ============== ======== ================================================================================

  **Neuron Variables**

  An object of neuron class record those variables for each neuron:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                          -65        Membrane potential.

  u                          1          Recovery variable.

  input                      0          External and synaptic input current.

  spike                      0          Flag to mark whether the neuron is spiking.

  t_last_spike               -1e7       Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
         Transactions on neural networks 14.6 (2003): 1569-1572.

  .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
         IEEE transactions on neural networks 15.5 (2004): 1063-1070.
  """

  def __init__(self, size, a=0.02, b=0.20, c=-65., d=8., t_refractory=0.,
               V_th=30., update_type='vector', **kwargs):
    super(Izhikevich, self).__init__(size=size, **kwargs)

    # params
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.t_refractory = t_refractory
    self.V_th = V_th

    # vars
    self.V = bp.math.Variable(bp.math.ones(self.num) * -65.)
    self.u = bp.math.Variable(bp.math.ones(self.num))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.refractory = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

    # update method
    self.update_type = update_type
    if update_type == 'forloop':
      self.update = self._forloop_update
      self.target_backend = 'numpy'
    elif update_type == 'vector':
      self.update = self._vector_update
      self.target_backend = 'general'
    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

  @bp.odeint
  def int_V(self, V, t, u, Iext):
    dVdt = 0.04 * V * V + 5 * V + 140 - u + Iext
    return dVdt

  @bp.odeint(method='exponential_euler')
  def int_u(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt

  def _forloop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
      u = self.int_u(self.u[i], _t, self.V[i], dt=_dt)
      if not refractory:
        V = self.int_V(self.V[i], _t, self.u[i], self.input[i], dt=_dt)
        if V >= self.V_th:
          V = self.c
          u += self.d
          self.t_last_spike[i] = _t
          refractory = True
          spike = True
        else:
          spike = False
        self.V[i] = V
        self.u[i] = u
      self.spike[i] = spike
      self.refractory[i] = refractory
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    V = self.int_V(self.V, _t, self.u, self.input, dt=_dt)
    u = self.int_u(self.u, _t, self.V, dt=_dt)
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.c, V)
    self.u[:] = bp.math.where(spike, u + self.d, u)
    self.refractory[:] = bp.math.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike
