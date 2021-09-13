# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

__all__ = [
  'ExpIF'
]


class ExpIF(bp.NeuGroup):
  r"""Exponential integrate-and-fire neuron model.

  In the exponential integrate-and-fire model [1]_, the differential
  equation for the membrane potential is given by

  .. math::

      \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
      \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

  This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
  and "threshold" :math:`\vartheta_{rh}`.

  The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
  defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
  :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
  where :math:`\tau_{\rm ref}` is an absolute refractory time.
  If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
  its exact value does not play any role. The reason is that the upswing of the action
  potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
  an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
  convenience. For a formal mathematical analysis of the model, the threshold can be pushed
  to infinity.

  The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
  and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
  It is one of the prominent examples of a precise theoretical prediction in computational
  neuroscience that was later confirmed by experimental neuroscience.

  Two important remarks:

  - (i) The right-hand side of the above equation contains a nonlinearity
    that can be directly extracted from experimental data [3]_. In this sense the exponential
    nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
  - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
    rate for constant input, and the linear response to fluctuations, even in the presence
    of input noise [4]_.

  Illustrated example please see `this notebook <../neurons/ExpIF.ipynb>`_.

  **Model Parameters**

  ============= ============== ======== ===================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  R             1              \        Membrane resistance.
  tau           10             \        Membrane time constant. Compute by R * C.
  tau_ref       1.7            \        Refractory period length.
  ============= ============== ======== ===================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  0                 Membrane potential.
  input              0                 External and synaptic input current.
  spike              False             Flag to mark whether the neuron is spiking.
  refractory         False             Flag to mark whether the neuron is in refractory period.
  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
         Neuronal dynamics: From single neurons to networks and models
         of cognition. Cambridge University Press.
  .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
         Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
         are reliable predictors of naturalistic pyramidal-neuron voltage
         traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
  .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
         integrate-and-fire neurons to modulated current-based and
         conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
  .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48,
               R=1., tau=10., tau_ref=1.7, update_type='vector', **kwargs):
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
    super(ExpIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.V = bm.Variable(bm.ones(self.num) * V_rest)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  @bp.odeint
  def integral(self, V, t, Iext):
    exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dvdt = (- (V - self.V_rest) + exp_v + self.R * Iext) / self.tau
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
    self.spike[:] = spike
    self.input[:] = 0.
