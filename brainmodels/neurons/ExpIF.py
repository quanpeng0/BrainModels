# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
'ExpIF'
]


class ExpIF(bp.NeuGroup):
  """Exponential Integrate-and-Fire neuron model [1]_.

  .. math::

      \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t)

  **Neuron Parameters**

  ============= ============== ======== ===================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------------------
  V_rest        -65.           mV       Resting potential.

  V_reset       -68.           mV       Reset potential after spike.

  V_th          -30.           mV       Threshold potential of spike.

  V_T           -59.9          mV       Threshold potential of generating action potential.

  delta_T       3.48           \        Spike slope factor.

  R             10.            \        Membrane resistance.

  C             1.             \        Membrane capacitance.

  tau           10.            \        Membrane time constant. Compute by R * C.

  t_refractory  1.7            \        Refractory period length.
  ============= ============== ======== ===================================================

  **Neuron Variables**

  An object of neuron class record those variables for each neuron:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  0.                Membrane potential.

  input              0.                External and synaptic input current.

  spike              0.                Flag to mark whether the neuron is spiking.

                                       Can be seen as bool.

  refractory         0.                Flag to mark whether the neuron is in refractory period.

                                       Can be seen as bool.

  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48,
               R=10., C=1., tau=10., t_refractory=1.7, update_type='vector', **kwargs):
    super(ExpIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.R = R
    self.C = C
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * V_rest)
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
    exp_v = self.delta_T * bp.math.exp((V - self.V_T) / self.delta_T)
    dvdt = (- (V - self.V_rest) + exp_v + self.R * Iext) / self.tau
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
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.refractory[:] = refractory | spike
    self.input[:] = 0.
    self.spike[:] = spike
