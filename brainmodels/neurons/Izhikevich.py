# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'Izhikevich'
]


class Izhikevich(bp.NeuGroup):
  r"""The Izhikevich neuron model.

  **Model Descriptions**

  The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

  .. math ::

      \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

      \frac{d u}{d t} &=a(b V-u)

  .. math ::

      \text{if}  v \geq 30  \text{mV}, \text{then}
      \begin{cases} v \leftarrow c \\
      u \leftarrow u+d \end{cases}

  **Model Examples**

  - `Detailed examples to reproduce different firing patterns <../../examples/neurons/Izhikevich_2003_Izhikevich_model.ipynb>`_

  **Model Parameters**

  ============= ============== ======== ================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------------------
  a             0.02           \        It determines the time scale of
                                        the recovery variable :math:`u`.
  b             0.2            \        It describes the sensitivity of the
                                        recovery variable :math:`u` to
                                        the sub-threshold fluctuations of the
                                        membrane potential :math:`v`.
  c             -65            \        It describes the after-spike reset value
                                        of the membrane potential :math:`v` caused by
                                        the fast high-threshold :math:`K^{+}`
                                        conductance.
  d             8              \        It describes after-spike reset of the
                                        recovery variable :math:`u`
                                        caused by slow high-threshold
                                        :math:`Na^{+}` and :math:`K^{+}` conductance.
  tau_ref       0              ms       Refractory period length. [ms]
  V_th          30             mV       The membrane potential threshold.
  ============= ============== ======== ================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                          -65        Membrane potential.
  u                          1          Recovery variable.
  input                      0          External and synaptic input current.
  spike                      False      Flag to mark whether the neuron is spiking.
  refractory                False       Flag to mark whether the neuron is in refractory period.
  t_last_spike               -1e7       Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
         Transactions on neural networks 14.6 (2003): 1569-1572.

  .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
         IEEE transactions on neural networks 15.5 (2004): 1063-1070.
  """

  def __init__(self, size, a=0.02, b=0.20, c=-65., d=8., tau_ref=0.,
               V_th=30., update_type='vector', num_batch=None, **kwargs):
    # initialization
    super(Izhikevich, self).__init__(size=size, num_batch=num_batch, **kwargs)

    # params
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.V_th = V_th
    self.tau_ref = tau_ref

    # update method
    self.update_type = update_type
    if update_type == 'loop':
      self.steps.replace('update', self._loop_update)
      self.target_backend = 'numpy'
    elif update_type == 'vector':
      self.steps.replace('update', self._vector_update)
      self.target_backend = 'general'
    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # vars
    self.V = bm.Variable(bm.ones(self.shape) * -65.)
    self.u = bm.Variable(bm.ones(self.shape))
    self.input = bm.Variable(bm.zeros(self.shape))
    self.spike = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.shape) * -1e7)

  @bp.odeint
  def int_V(self, V, t, u, Iext):
    dVdt = 0.04 * V * V + 5 * V + 140 - u + Iext
    return dVdt

  @bp.odeint
  def int_u(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.tau_ref)
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
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.c, V)
    self.u[:] = bm.where(spike, u + self.d, u)
    self.refractory[:] = bm.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike
