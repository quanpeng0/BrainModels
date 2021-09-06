# -*- coding: utf-8 -*-


import brainpy as bp

__all__ = [
  'HH'
]


class HH(bp.NeuGroup):
  """Hodgkin–Huxley neuron model [1]_.

  Alan Hodgkin and Andrew Huxley described the model in 1952 [1]_ to explain
  the ionic mechanisms underlying the initiation and propagation of action
  potentials in the squid giant axon.

  .. math::

      C \\frac {dV} {dt} = -(\\bar{g}_{Na} m^3 h (V &-E_{Na})
      + \\bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

      \\frac {dx} {dt} &= \\alpha_x (1-x)  - \\beta_x, \\quad x\\in {\\rm{\\{m, h, n\\}}}

      &\\alpha_m(V) = \\frac {0.1(V+40)}{1-exp(\\frac{-(V + 40)} {10})}

      &\\beta_m(V) = 4.0 exp(\\frac{-(V + 65)} {18})

      &\\alpha_h(V) = 0.07 exp(\\frac{-(V+65)}{20})

      &\\beta_h(V) = \\frac 1 {1 + exp(\\frac{-(V + 35)} {10})}

      &\\alpha_n(V) = \\frac {0.01(V+55)}{1-exp(-(V+55)/10)}

      &\\beta_n(V) = 0.125 exp(\\frac{-(V + 65)} {80})


  **Neuron Parameters**

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

  noise         0.             \        the noise fluctuation.
  ============= ============== ======== ====================================

  **Neuron Variables**

  An object of neuron class record those variables:

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

  References
  ----------

  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
         of membrane current and its application to conduction and excitation
         in nerve." The Journal of physiology 117.4 (1952): 500.

  """

  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
               gL=0.03, V_th=20., C=1.0, update_type='vector', **kwargs):
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

    # variables
    self.V = bp.math.Variable(-65. * bp.math.ones(self.num))
    self.m = bp.math.Variable(0.5 * bp.math.ones(self.num))
    self.h = bp.math.Variable(0.6 * bp.math.ones(self.num))
    self.n = bp.math.Variable(0.32 * bp.math.ones(self.num))
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

  @bp.odeint(method='exponential_euler')
  def integral(self, V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
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
      spike = bp.math.logical_and(self.V[i] < self.V_th, V >= self.V_th)
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
    self.spike[:] = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.
