# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'AdQuaIF'
]


class AdQuaIF(bp.NeuGroup):
  r"""Adaptive Quadratic Integrate-and-Fire neuron model [1]_.

  .. math::

      \tau_m \frac{d V}{d t}=a_0(V-V_{rest})(V-V_c) - R w + RI(t)

      \tau_w \frac{d w}{d t}=a(V-V_{rest}) - w + b \tau_w \sum \delta (t-t^f)


  **Neuron Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65.           mV       Resting potential.

  V_reset       -68.           mV       Reset potential after spike.

  V_th          -30.           mV       Threshold potential of spike and reset.

  a_0           .07            \        Coefficient describes membrane potential update. Larger than 0.

  V_c           -50.           mV       Critical voltage for spike initiation. Must be larger than V_rest.

  a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`

  b             .1             \        The increment of :math:`w` produced by a spike.

  R             1              \        Membrane resistance.

  tau           10             ms       Membrane time constant. Compute by R * C.

  tau_w         10             ms       Time constant of the adaptation current.

  t_refractory  0              ms       Refractory period length.

  noise         0.             \        the noise fluctuation.

  mode          'scalar'       \        Data structure of ST members.
  ============= ============== ======== ========================================================================================================================

  **Neuron Variables**

  An object of neuron class record those variables:

  ================== ================= ==========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ----------------------------------------------------------
  V                   0.                Membrane potential.

  w                   0.                Adaptation current.

  input               0.                External and synaptic input current.

  spike               0.                Flag to mark whether the neuron is spiking.

  refractory          0.                Flag to mark whether the neuron is in refractory period.

  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= ==========================================================

  References
  ----------

  .. [1] Zheng G, Tonnelier A. Chaotic solutions in the quadratic integrate-and-fire 
         neuron with adaptation[J]. Cognitive neurodynamics, 2009, 3(3): 197-204.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0,
               a_0=.07, a=1., b=.1, R=1., tau=10., tau_w=10., t_refractory=0.,
               update_type='vector', **kwargs):
    super(AdQuaIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.a_0 = a_0
    self.a = a
    self.b = b
    self.R = R
    self.tau = tau
    self.tau_w = tau_w
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * V_reset)
    self.w = bp.math.Variable(bp.math.zeros(self.num))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.refractory = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

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

  @bp.odeint
  def int_V(self, V, t, w, Iext):
    dVdt = (self.a_0 * (V - self.V_rest) * (V - self.V_c) - 
            self.R * w + self.R * Iext) / self.tau
    return dVdt

  @bp.odeint(method='exponential_euler')
  def int_w(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
      w = self.int_w(self.w[i], _t, self.V[i], dt=_dt)
      if not refractory:
        V = self.int_V(self.V[i], _t, self.w[i], self.input[i], dt=_dt)
        spike = (V >= self.V_th)
        if spike:
          V = self.V_rest
          w += self.b
          self.t_last_spike[i] = _t
          refractory = True
        self.V[i] = V
      self.w[i] = w
      self.spike[i] = spike
      self.refractory[i] = refractory
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    w = self.int_w(self.w, _t, self.V, dt=_dt)
    V = self.int_V(self.V, _t, self.w, self.input, dt=_dt)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.w[:] = bp.math.where(spike, w + self.b, w)
    self.refractory[:] = bp.math.logical_or(refractory, spike)
    self.spike[:] = spike
    self.input[:] = 0.
