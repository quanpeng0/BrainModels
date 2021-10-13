# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'LIF'
]


class LIF(bp.NeuGroup):
  r"""Leaky integrate-and-fire neuron model.

  **Model Descriptions**

  The formal equations of a LIF model [1]_ is given by:

  .. math::

      \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + I(t) \\
      \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset} \quad
      \text{last} \quad \tau_{ref} \quad  \text{ms}

  where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
  membrane potential, :math:`V_{reset}` is the reset membrane potential,
  :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
  :math:`\tau_{ref}` is the refractory time period,
  and :math:`I` is the time-variant synaptic inputs.

  **Model Examples**

  - `Illustrated example <../neurons/LIF.ipynb>`_
  - `(Brette, Romain. 2004) LIF phase locking <../../examples/neurons/Romain_2004_LIF_phase_locking.ipynb>`_

  **Model Parameters**

  ============= ============== ======== =========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------
  V_rest         0              mV       Resting membrane potential.
  V_reset        -5             mV       Reset potential after spike.
  V_th           20             mV       Threshold potential of spike.
  tau            10             ms       Membrane time constant. Compute by R * C.
  tau_ref       5              ms       Refractory period length.(ms)
  ============= ============== ======== =========================================

  **Neuron Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                    0                Membrane potential.
  input                0                External and synaptic input current.
  spike                False             Flag to mark whether the neuron is spiking.
  refractory           False             Flag to mark whether the neuron is in refractory period.
  t_last_spike         -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
         neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.
  """

  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., tau=10.,
               tau_ref=1., update_type='vector', num_batch=None, **kwargs):
    # initialization
    super(LIF, self).__init__(size=size, num_batch=num_batch, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.tau = tau
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

    # variables
    self.V = bm.Variable(bm.ones(self.shape) * V_rest)
    self.input = bm.Variable(bm.zeros(self.shape))
    self.refractory = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.spike = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.shape) * -1e7)

  @bp.odeint(method='exponential_euler')
  def integral(self, V, t, Iext):
    dvdt = (-V + self.V_rest + Iext) / self.tau
    return dvdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.tau_ref)
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
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.refractory[:] = bm.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike
