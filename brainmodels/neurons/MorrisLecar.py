# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'MorrisLecar'
]


class MorrisLecar(bp.NeuGroup):
  """
  The Morris-Lecar neuron model [1]_. (Also known as :math:`I_{Ca}+I_K`-model.)

  .. math::

      C\\frac{dV}{dt} = -  g_{Ca} M_{\\infty} (V - & V_{Ca})- g_{K} W(V - V_{K}) - g_{Leak} (V - V_{Leak}) + I_{ext}

      & \\frac{dW}{dt} = \\frac{W_{\\infty}(V) - W}{ \\tau_W(V)}

  **Neuron Parameters**

  ============= ============== ======== =======================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------------------
  noise         0.             \        The noise fluctuation.
  V_Ca          130.           mV       Equilibrium potentials of Ca+.(mV)
  g_Ca          4.4            \        Maximum conductance of corresponding Ca+.(mS/cm2)
  V_K           -84.           mV       Equilibrium potentials of K+.(mV)
  g_K           8.             \        Maximum conductance of corresponding K+.(mS/cm2)
  V_Leak        -60.           mV       Equilibrium potentials of leak current.(mV)
  g_Leak        2.             \        Maximum conductance of leak current.(mS/cm2)
  C             20.            \        Membrane capacitance.(uF/cm2)
  V1            -1.2           \        Potential at which M_inf = 0.5.(mV)
  V2            18.            \        Reciprocal of slope of voltage dependence of M_inf.(mV)
  V3            2.             \        Potential at which W_inf = 0.5.(mV)
  V4            30.            \        Reciprocal of slope of voltage dependence of W_inf.(mV)
  phi           0.04           \        A temperature factor.(1/s)
  V_th          10.            mV       the spike threshold.
  ============= ============== ======== =======================================================

  **Neuron Variables**

  An object of neuron class record those variables for each neuron:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  -20.              Membrane potential.

  W                  0.02              Gating variable, refers to the fraction of
                                       opened K+ channels.

  input              0.                External and synaptic input current.

  spike              0.                Flag to mark whether the neuron is spiking.

  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Meier, Stephen R., Jarrett L. Lancaster, and Joseph M. Starobin.
         "Bursting regimes in a reaction-diffusion system with action
         potential-dependent equilibrium." PloS one 10.3 (2015):
         e0122401.
  """

  def __init__(self, size, V_Ca=130., g_Ca=4.4, V_K=-84., g_K=8., V_leak=-60.,
               g_leak=2., C=20., V1=-1.2, V2=18., V3=2., V4=30., phi=0.04,
               V_th=10., update_type='vector', **kwargs):
    super(MorrisLecar, self).__init__(size=size, **kwargs)

    # params
    self.V_Ca = V_Ca
    self.g_Ca = g_Ca
    self.V_K = V_K
    self.g_K = g_K
    self.V_leak = V_leak
    self.g_leak = g_leak
    self.C = C
    self.V1 = V1
    self.V2 = V2
    self.V3 = V3
    self.V4 = V4
    self.phi = phi
    self.V_th = V_th

    # vars
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.V = bp.math.Variable(bp.math.ones(self.num) * -20.)
    self.W = bp.math.Variable(bp.math.ones(self.num) * 0.02)
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
  def integral(self, V, W, t, I_ext):
    M_inf = (1 / 2) * (1 + bp.math.tanh((V - self.V1) / self.V2))
    I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
    I_K = self.g_K * W * (V - self.V_K)
    I_Leak = self.g_leak * (V - self.V_leak)
    dVdt = (- I_Ca - I_K - I_Leak + I_ext) / self.C

    tau_W = 1 / (self.phi * bp.math.cosh((V - self.V3) / (2 * self.V4)))
    W_inf = (1 / 2) * (1 + bp.math.tanh((V - self.V3) / self.V4))
    dWdt = (W_inf - W) / tau_W
    return dVdt, dWdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      V, W = self.integral(self.V[i], self.W[i], _t, self.input[i], dt=_dt)
      spike = bp.math.logical_and(self.V[i] < self.V_th, V >= self.V_th)
      self.V[i] = V
      self.W[i] = W
      self.spike[i] = spike
      if spike:
        self.t_last_spike[i] = _t
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    V, self.W[:] = self.integral(self.V, self.W, _t, self.input, dt=_dt)
    spike = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = V
    self.spike[:] = spike
    self.input[:] = 0.
