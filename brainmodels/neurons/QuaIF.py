# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'QuaIF'
]


class QuaIF(bp.NeuGroup):
  """Quadratic Integrate-and-Fire neuron model [1]_.
      
  .. math::

      \\tau \\frac{d V}{d t}=a_0(V-V_{rest})(V-V_c) + RI(t)
  
  where the parameters are taken to be :math:`a_0` =0.07, and
  :math:`V_c = -50 mV` (Latham et al., 2000 [1]_).

  **Neuron Parameters**
  
  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65.           mV       Resting potential.

  V_reset       -68.           mV       Reset potential after spike.

  V_th          -30.           mV       Threshold potential of spike and reset.

  V_c           -50.           mV       Critical voltage for spike initiation. Must be larger than V_rest.

  a_0           .07            \        Coefficient describes membrane potential update. Larger than 0.

  R             1              \        Membrane resistance.

  tau           10             ms       Membrane time constant. Compute by R * C.

  t_refractory  0              ms       Refractory period length.

  noise         0.             \        the noise fluctuation.
  ============= ============== ======== ========================================================================================================================

  **Neuron Variables**

  An object of neuron class record those variables:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0.                Membrane potential.

  input               0.                External and synaptic input current.

  spike               0.                Flag to mark whether the neuron is spiking.

                                        Can be seen as bool.

  refractory          0.                Flag to mark whether the neuron is in refractory period.

                                        Can be seen as bool.

  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
          (2000) Intrinsic dynamics in neuronal networks. I. Theory.
          J. Neurophysiology 83, pp. 808–827.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0, a_0=.07,
               R=1., tau=10., t_refractory=0., update_type='vector', **kwargs):
    super(QuaIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.a_0 = a_0
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * V_reset)
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
  def integral(self, V, t, Iext):
    dVdt = (self.a_0 * (V - self.V_rest) * (V - self.V_c) + self.R * Iext) / self.tau
    return dVdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
      if not refractory:
        V = self.integral(self.V[i], _t, self.input[i], dt=_dt)
        spike = (V >= self.V_th)
        if spike:
          V = self.V_rest
          self.t_last_spike[i] = _t
          refractory = True
        self.V[i] = V
      self.spike[i] = spike
      self.refractory[i] = refractory
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.refractory[:] = bp.math.logical_or(refractory, spike)
    self.spike[:] = spike
    self.input[:] = 0.
