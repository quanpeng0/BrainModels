# -*- coding: utf-8 -*-

import brainpy.math as bm
from .base import Neuron

__all__ = [
  'AdQuaIF'
]


class AdQuaIF(Neuron):
  r"""Adaptive quadratic integrate-and-fire neuron model.

  **Model Descriptions**

  The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

  .. math::

      \begin{aligned}
      \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
      \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
      \end{aligned}

  once the membrane potential reaches the spike threshold,

  .. math::

      V \rightarrow V_{reset}, \\
      w \rightarrow w+b.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainmodels
    >>> group = brainmodels.neurons.AdQuaIF(1, monitors=['V', 'w'])
    >>> group.run(300, inputs=('input', 30.))
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> bp.visualize.line_plot(group.mon.ts, group.mon.V, ylabel='V')
    >>> fig.add_subplot(gs[1, 0])
    >>> bp.visualize.line_plot(group.mon.ts, group.mon.w, ylabel='w', show=True)

  **Model Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  V_rest         -65            mV       Resting potential.
  V_reset        -68            mV       Reset potential after spike.
  V_th           -30            mV       Threshold potential of spike and reset.
  V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                         than :math:`V_{rest}`.
  a               1              \       The sensitivity of the recovery variable :math:`u` to
                                         the sub-threshold fluctuations of the membrane
                                         potential :math:`v`
  b              .1             \        The increment of :math:`w` produced by a spike.
  c              .07             \       Coefficient describes membrane potential update.
                                         Larger than 0.
  tau            10             ms       Membrane time constant.
  tau_w          10             ms       Time constant of the adaptation current.
  ============= ============== ======== =======================================================

  **Model Variables**

  ================== ================= ==========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ----------------------------------------------------------
  V                   0                 Membrane potential.
  w                   0                 Adaptation current.
  input               0                 External and synaptic input current.
  spike               False             Flag to mark whether the neuron is spiking.
  t_last_spike        -1e7              Last spike time stamp.
  ================== ================= ==========================================================

  **References**

  .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
         neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
  .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
         nonlinear integrate-and-fire neurons." SIAM Journal on Applied
         Mathematics 68, no. 4 (2008): 1045-1079.
  """

  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_c=-50.0, a=1., b=.1,
               c=.07, tau=10., tau_w=10., method='euler', **kwargs):
    super(AdQuaIF, self).__init__(size=size, method=method, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.c = c
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w

    # variables
    self.w = bm.Variable(bm.zeros(self.num))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

  def derivative(self, V, w, t, Iext):
    dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + Iext) / self.tau
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dVdt, dwdt

  def update(self, _t, _dt):
    V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.w[:] = bm.where(spike, w + self.b, w)
    self.spike[:] = spike
    self.input[:] = 0.
