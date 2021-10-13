# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

__all__ = [
  'AdExIF'
]


class AdExIF(bp.NeuGroup):
  r"""Adaptive exponential integrate-and-fire neuron model.

  **Model Descriptions**

  The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
  spiking neuron model with two variables [1]_ [2]_.

  .. math::

      \begin{aligned}
      \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
      \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  The first equation describes the dynamics of the membrane potential and includes
  an activation term with an exponential voltage dependence. Voltage is coupled to
  a second equation which describes adaptation. Both variables are reset if an action
  potential has been triggered. The combination of adaptation and exponential voltage
  dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

  The adaptive exponential integrate-and-fire model is capable of describing known
  neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
  initial bursting, fast spiking, and regular spiking.

  **Model Examples**

  - `Examples for different firing patterns <../neurons/AdExIF_model.ipynb>`_

  **Model Parameters**

  ============= ============== ======== ========================================================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
  V_rest        -65            mV       Resting potential.
  V_reset       -68            mV       Reset potential after spike.
  V_th          -30            mV       Threshold potential of spike and reset.
  V_T           -59.9          mV       Threshold potential of generating action potential.
  delta_T       3.48           \        Spike slope factor.
  a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
  b             1              \        The increment of :math:`w` produced by a spike.
  R             1              \        Membrane resistance.
  tau           10             ms       Membrane time constant. Compute by R * C.
  tau_w         30             ms       Time constant of the adaptation current.
  ============= ============== ======== ========================================================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
         mechanisms determine the neuronal response to fluctuating
         inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
  .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48, a=1.,
               b=1., tau=10., tau_w=30., R=1., num_batch=None, **kwargs):
    super(AdExIF, self).__init__(size=size, num_batch=num_batch, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w
    self.R = R

    # variables
    self.V = bm.Variable(bm.ones(self.shape) * V_reset)
    self.w = bm.Variable(bm.zeros(self.shape))
    self.input = bm.Variable(bm.zeros(self.shape))
    self.spike = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.shape, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.shape) * -1e7)

  @bp.odeint
  def int_V(self, V, t, w, Iext):
    dVdt = (- V + self.V_rest + self.delta_T * bm.exp((V - self.V_T) / self.delta_T) -
            self.R * w + self.R * Iext) / self.tau
    return dVdt

  @bp.odeint(method='exponential_euler')
  def int_w(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  def update(self, _t, _dt):
    w = self.int_w(self.w, _t, self.V, dt=_dt)
    V = self.int_V(self.V, _t, self.w, self.input, dt=_dt)
    spike = V >= self.V_th
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.w[:] = bm.where(spike, w + self.b, w)
    self.input[:] = 0.
    self.spike[:] = spike
