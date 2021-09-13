# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

__all__ = [
  'HH'
]


class HH(bp.NeuGroup):
  r"""Hodgkin–Huxley neuron model.

  **Model Descriptions**

  The Hodgkin-Huxley (HH; Hodgkin & Huxley, 1952) model [1]_ for the generation of
  the nerve action potential is one of the most successful mathematical models of
  a complex biological process that has ever been formulated. The basic concepts
  expressed in the model have proved a valid approach to the study of bio-electrical
  activity from the most primitive single-celled organisms such as *Paramecium*,
  right through to the neurons within our own brains.

  Mathematically, the model is given by,

  .. math::

      C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V &-E_{Na})
      + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

      \frac {dx} {dt} &= \alpha_x (1-x)  - \beta_x, \quad x\in {\rm{\{m, h, n\}}}

      &\alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}

      &\beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})

      &\alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})

      &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}

      &\alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}

      &\beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})

  The illustrated example of HH neuron model please see `this notebook <../neurons/HH_model.ipynb>`_.

  The Hodgkin–Huxley model can be thought of as a differential equation system with
  four state variables, :math:`V_{m}(t),n(t),m(t)`, and :math:`h(t)`, that change
  with respect to time :math:`t`. The system is difficult to study because it is a
  nonlinear system and cannot be solved analytically. However, there are many numeric
  methods available to analyze the system. Certain properties and general behaviors,
  such as limit cycles, can be proven to exist.

  *1. Center manifold*

  Because there are four state variables, visualizing the path in phase space can
  be difficult. Usually two variables are chosen, voltage :math:`V_{m}(t)` and the
  potassium gating variable :math:`n(t)`, allowing one to visualize the limit cycle.
  However, one must be careful because this is an ad-hoc method of visualizing the
  4-dimensional system. This does not prove the existence of the limit cycle.

  .. image:: ../../images/Hodgkin_Huxley_Limit_Cycle.png
      :align: center

  A better projection can be constructed from a careful analysis of the Jacobian of
  the system, evaluated at the equilibrium point. Specifically, the eigenvalues of
  the Jacobian are indicative of the center manifold's existence. Likewise, the
  eigenvectors of the Jacobian reveal the center manifold's orientation. The
  Hodgkin–Huxley model has two negative eigenvalues and two complex eigenvalues
  with slightly positive real parts. The eigenvectors associated with the two
  negative eigenvalues will reduce to zero as time :math:`t` increases. The remaining
  two complex eigenvectors define the center manifold. In other words, the
  4-dimensional system collapses onto a 2-dimensional plane. Any solution
  starting off the center manifold will decay towards the *center manifold*.
  Furthermore, the limit cycle is contained on the center manifold.

  *2. Bifurcations*

  If the injected current :math:`I` were used as a bifurcation parameter, then the
  Hodgkin–Huxley model undergoes a Hopf bifurcation. As with most neuronal models,
  increasing the injected current will increase the firing rate of the neuron.
  One consequence of the Hopf bifurcation is that there is a minimum firing rate.
  This means that either the neuron is not firing at all (corresponding to zero
  frequency), or firing at the minimum firing rate. Because of the all-or-none
  principle, there is no smooth increase in action potential amplitude, but
  rather there is a sudden "jump" in amplitude. The resulting transition is
  known as a `canard <http://www.scholarpedia.org/article/Canards>`_.

  .. image:: ../../images/Hodgkins_Huxley_bifurcation_by_I.gif
     :align: center

  The following image shows the bifurcation diagram of the Hodgkin–Huxley model
  as a function of the external drive :math:`I` [3]_. The green lines show the amplitude
  of a stable limit cycle and the blue lines indicate unstable limit-cycle behaviour,
  both born from Hopf bifurcations. The solid red line shows the stable fixed point
  and the black line shows the unstable fixed point.

  .. image:: ../../images/Hodgkin_Huxley_bifurcation.png
     :align: center

  **Model Examples**

  - `An illustrated example <../neurons/HH_model.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ====================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------
  V_th          20.            mV       the spike threshold.
  C             1.             ufarad   capacitance.
  E_Na          50.            mV       reversal potential of sodium.
  E_K           -77.           mV       reversal potential of potassium.
  E_leak        54.387         mV       reversal potential of unspecific.
  g_Na          120.           msiemens conductance of sodium channel.
  g_K           36.            msiemens conductance of potassium channel.
  g_leak        .03            msiemens conductance of unspecific channels.
  ============= ============== ======== ====================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                        -65         Membrane potential.
  m                        0.05        gating variable of the sodium ion channel.
  n                        0.32        gating variable of the potassium ion channel.
  h                        0.60        gating variable of the sodium ion channel.
  input                     0          External and synaptic input current.
  spike                    False       Flag to mark whether the neuron is spiking.
  t_last_spike       -1e7               Last spike time stamp.
  ================== ================= =========================================================

  **References**

  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
         of membrane current and its application to conduction and excitation
         in nerve." The Journal of physiology 117.4 (1952): 500.
  .. [2] https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
  .. [3] Ashwin, Peter, Stephen Coombes, and Rachel Nicks. "Mathematical
         frameworks for oscillatory network dynamics in neuroscience."
         The Journal of Mathematical Neuroscience 6, no. 1 (2016): 1-92.
  """

  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
               gL=0.03, V_th=20., C=1.0, update_type='vector', **kwargs):
    # initialization
    super(HH, self).__init__(size=size, **kwargs)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

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
    self.V = bm.Variable(-65. * bm.ones(self.num))
    self.m = bm.Variable(0.5 * bm.ones(self.num))
    self.h = bm.Variable(0.6 * bm.ones(self.num))
    self.n = bm.Variable(0.32 * bm.ones(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  @bp.odeint(method='exponential_euler')
  def integral(self, V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C

    return dVdt, dmdt, dhdt, dndt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      V, m, h, n = self.integral(self.V[i], self.m[i], self.h[i], self.n[i],
                                 _t, self.input[i], dt=_dt)
      spike = (self.V[i] < self.V_th) and (V >= self.V_th)
      self.spike[i] = spike
      if spike:
        self.t_last_spike[i] = _t
      self.V[i] = V
      self.m[i] = m
      self.h[i] = h
      self.n[i] = n
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input, dt=_dt)
    self.spike[:] = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike[:] = bm.where(self.spike, _t, self.t_last_spike)
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.
