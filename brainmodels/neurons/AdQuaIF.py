# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'AdQuaIF'
]


class AdQuaIF(bp.NeuGroup):
  r"""Adaptive quadratic integrate-and-fire neuron model.

  The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

  .. math::

      \begin{aligned}
      \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
      \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  **Examples**

  - `Simple example <../neurons/AdQuaIF.ipynb>`_

  **Model Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  V_rest         -65            mV       Resting potential.
  V_reset        -68            mV       Reset potential after spike.
  V_th           -30            mV       Threshold potential of spike and reset.
  V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                         than :math:`V_{rest}`.
  a               1              \       The sensitivity of the recovery variable :math:`u` to
                                         the sub-threshold fluctuations of the membrane
                                         potential :math:`v`
  b              .1             \        The increment of :math:`w` produced by a spike.
  c              .07             \       Coefficient describes membrane potential update.
                                         Larger than 0.
  tau            10             ms       Membrane time constant.
  tau_w          10             ms       Time constant of the adaptation current.
  ============= ============== ======== =======================================================

  **Model Variables**

  ================== ================= ==========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ----------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= ==========================================================

  References
  ----------

  .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
         neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
  .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
         nonlinear integrate-and-fire neurons." SIAM Journal on Applied
         Mathematics 68, no. 4 (2008): 1045-1079.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0,
               a=1., b=.1, c=.07, tau=10., tau_w=10., update_type='vector', **kwargs):
    # update method
    self.update_type = update_type
    if update_type == 'loop':
      self.update = self._loop_update
      self.target_backend = 'numpy'
    elif update_type == 'vector':
      self.update = self._vector_update
      self.target_backend = 'general'
    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

    # initialize
    super(AdQuaIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.c = c
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w

    # variables
    self.V = bm.Variable(bm.ones(self.num) * V_reset)
    self.w = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  @bp.odeint
  def int_V(self, V, t, w, Iext):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + Iext) / self.tau
    return dVdt

  @bp.odeint(method='exponential_euler')
  def int_w(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      w = self.int_w(self.w[i], _t, self.V[i], dt=_dt)
      V = self.int_V(self.V[i], _t, self.w[i], self.input[i], dt=_dt)
      spike = (V >= self.V_th)
      if spike:
        V = self.V_rest
        w += self.b
        self.t_last_spike[i] = _t
      self.V[i] = V
      self.w[i] = w
      self.spike[i] = spike
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    w = self.int_w(self.w, _t, self.V, dt=_dt)
    V = self.int_V(self.V, _t, self.w, self.input, dt=_dt)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.w[:] = bm.where(spike, w + self.b, w)
    self.spike[:] = spike
    self.input[:] = 0.
