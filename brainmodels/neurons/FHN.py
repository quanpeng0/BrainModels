# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'FHN'
]


class FHN(bp.NeuGroup):
  r"""FitzHugh-Nagumo neuron model.

  **Model Descriptions**

  The FitzHugh–Nagumo model (FHN), named after Richard FitzHugh (1922–2007)
  who suggested the system in 1961 [1]_ and J. Nagumo et al. who created the
  equivalent circuit the following year, describes a prototype of an excitable
  system (e.g., a neuron).

  The motivation for the FitzHugh-Nagumo model was to isolate conceptually
  the essentially mathematical properties of excitation and propagation from
  the electrochemical properties of sodium and potassium ion flow. The model
  consists of

  - a *voltage-like variable* having cubic nonlinearity that allows regenerative
    self-excitation via a positive feedback, and
  - a *recovery variable* having a linear dynamics that provides a slower negative feedback.

  .. math::

     \begin{aligned}
     {\dot {v}} &=v-{\frac {v^{3}}{3}}-w+RI_{\rm {ext}},  \\
     \tau {\dot  {w}}&=v+a-bw.
     \end{aligned}

  The FHN Model is an example of a relaxation oscillator
  because, if the external stimulus :math:`I_{\text{ext}}`
  exceeds a certain threshold value, the system will exhibit
  a characteristic excursion in phase space, before the
  variables :math:`v` and :math:`w` relax back to their rest values.
  This behaviour is typical for spike generations (a short,
  nonlinear elevation of membrane voltage :math:`v`,
  diminished over time by a slower, linear recovery variable
  :math:`w`) in a neuron after stimulation by an external
  input current.

  **Model Examples**

  - `Illustrated example <../neurons/FHN.ipynb>`_

  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  a             1              \        Positive constant
  b             1              \        Positive constant
  tau           10             ms       Membrane time constant.
  V_th          1.8            mV       Threshold potential of spike.
  ============= ============== ======== ========================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 A recovery variable which represents
                                        the combined effects of sodium channel
                                        de-inactivation and potassium channel
                                        deactivation.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] FitzHugh, Richard. "Impulses and physiological states in theoretical models of nerve membrane." Biophysical journal 1.6 (1961): 445-466.
  .. [2] https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
  .. [3] http://www.scholarpedia.org/article/FitzHugh-Nagumo_model

  """

  def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.8,
               update_type='vector', num_batch=None, **kwargs):
    # initialization
    super(FHN, self).__init__(size=size, num_batch=num_batch, **kwargs)

    # parameters
    self.a = a
    self.b = b
    self.tau = tau
    self.Vth = Vth

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

    # variables
    self.V = bm.Variable(bm.zeros(self.shape))
    self.w = bm.Variable(bm.zeros(self.shape))
    self.input = bm.Variable(bm.zeros(self.shape))
    self.spike = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.shape) * -1e7)

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
    self.spike[:] = bm.logical_and(V >= self.Vth, self.V < self.Vth)
    self.t_last_spike[:] = bm.where(self.spike, _t, self.t_last_spike)
    self.V[:] = V
    self.input[:] = 0.
