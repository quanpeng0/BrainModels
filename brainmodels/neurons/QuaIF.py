# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'QuaIF'
]


class QuaIF(bp.NeuGroup):
  r"""Quadratic Integrate-and-Fire neuron model.


  In contrast to physiologically accurate but computationally expensive
  neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
  to produce **action potential-like patterns** and ignores subtleties
  like gating variables, which play an important role in generating action
  potentials in a real neuron. However, the QIF model is incredibly easy
  to implement and compute, and relatively straightforward to study and
  understand, thus has found ubiquitous use in computational neuroscience.

  .. math::

      \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)
  
  where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

  **Examples**

  - `Illustrated example <../neurons/QuaIF.ipynb>`_

  **Model Parameters**
  
  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
  c             .07            \        Coefficient describes membrane potential update. Larger than 0.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_ref       0              ms       Refractory period length.
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  refractory          False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
          (2000) Intrinsic dynamics in neuronal networks. I. Theory.
          J. Neurophysiology 83, pp. 808–827.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0, c=.07,
               R=1., tau=10., tau_ref=0., update_type='vector', **kwargs):
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

    # initialization
    super(QuaIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.c = c
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.V = bm.Variable(bm.ones(self.num) * V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  @bp.odeint
  def integral(self, V, t, Iext):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * Iext) / self.tau
    return dVdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.tau_ref)
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
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.refractory[:] = bm.logical_or(refractory, spike)
    self.spike[:] = spike
    self.input[:] = 0.
