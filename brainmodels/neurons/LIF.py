# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'LIF'
]


class LIF(bp.NeuGroup):
  r"""Leaky Integrate-and-Fire neuron model [1]_.

  .. math::

      \tau \frac{d V}{d t}=-(V-V_{rest}) + RI(t)

  **Neuron Parameters**

  ============= ============== ======== =========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------
  V_rest        0.             mV       Resting potential.

  V_reset       -5.            mV       Reset potential after spike.

  V_th          20.            mV       Threshold potential of spike.

  R             1.             \        Membrane resistance.

  tau           10.            ms       Membrane time constant. Compute by R * C.

  t_refractory  5.             ms       Refractory period length.(ms)
  ============= ============== ======== =========================================

  **Neuron Variables**

  An object of neuron class record those variables for each neuron:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  0.                Membrane potential.

  input              0.                External and synaptic input current.

  spike              0.                Flag to mark whether the neuron is spiking.

  refractory         0.                Flag to mark whether the neuron is in refractory period.

                                       Can be seen as bool.

  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.
  """

  def __init__(self, size, t_refractory=1., V_rest=0., V_reset=-5.,
               V_th=20., R=1., tau=10., update_type='vector', **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * V_rest)
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.refractory = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
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

  @bp.odeint(method='exponential_euler')
  def integral(self, V, t, Iext):
    dvdt = (-V + self.V_rest + self.R * Iext) / self.tau
    return dvdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
      if not refractory:
        V = self.integral(self.V[i], _t, self.input[i], dt=_dt)
        spike = (V >= self.V_th)
        if spike:
          V = self.V_reset
          self.t_last_spike[i] = _t
          refractory = True
        self.V[i] = V
      self.spike[i] = spike
      self.refractory[i] = refractory
    self.input[:] = 0.

  def _vector_update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.refractory[:] = bp.math.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike
