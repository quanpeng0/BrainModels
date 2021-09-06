# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'FHN'
]


class FHN(bp.NeuGroup):
  """FitzHugh-Nagumo neuron model [1]_.

  The FHN Model is an example of a relaxation oscillator
  because, if the external stimulus :math:`I_{\\text{ext}}`
  exceeds a certain threshold value, the system will exhibit
  a characteristic excursion in phase space, before the
  variables :math:`v` and :math:`w` relax back to their rest values.

  This behaviour is typical for spike generations (a short,
  nonlinear elevation of membrane voltage :math:`v`,
  diminished over time by a slower, linear recovery variable
  :math:`w`) in a neuron after stimulation by an external
  input current.

  .. math::

      {\\dot {v}}=v-{\\frac {v^{3}}{3}}-w+RI_{\\rm {ext}}

      \\tau {\\dot  {w}}=v+a-bw


  **Neuron Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  a             1              \        positive constant

  b             1              \         positive constant

  tau           10             ms       Membrane time constant.

  V_th          1.             mV       Threshold potential of spike.
  ============= ============== ======== ========================


  **Neuron Variables**

  An object of neuron class record those variables for each synapse:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0.                Membrane potential.

  w                   0.                A recovery variable which represents
                                        the combined effects of sodium channel
                                        de-inactivation and potassium channel
                                        deactivation.

  input               0.                External and synaptic input current.

  spike               0.                Flag to mark whether the neuron is spiking.

  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] FitzHugh, Richard. "Impulses and physiological states in theoretical models of nerve membrane." Biophysical journal 1.6 (1961): 445-466.


  """

  def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.8, update_type='vector', **kwargs):
    super(FHN, self).__init__(size=size, **kwargs)

    # parameters
    self.a = a
    self.b = b
    self.tau = tau
    self.Vth = Vth

    # variables
    self.V = bp.math.Variable(bp.math.zeros(self.num))
    self.w = bp.math.Variable(bp.math.zeros(self.num))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
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

  @bp.odeint
  def integral(self, V, w, t, Iext):
    dw = (V + self.a - self.b * w) / self.tau
    dV = V - V * V * V / 3 - w + Iext
    return dV, dw

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      V, w = self.integral(self.V[i], self.w[i], _t, self.input[i], dt=_dt)
      spike = (V >= self.Vth) and (self.V[i] < self.Vth)
      self.spike[i] = spike
      if spike:
        self.t_last_spike[i] = _t
      self.V[i] = V
      self.w[i] = w
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    V, self.w[:] = self.integral(self.V, self.w, _t, self.input, dt=_dt)
    self.spike[:] = bp.math.logical_and(V >= self.Vth, self.V < self.Vth)
    self.V[:] = V
    self.input[:] = 0.
