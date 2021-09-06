# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'GIF'
]


class GIF(bp.NeuGroup):
  """Generalized Integrate-and-Fire model (GIF model) [1]_.

  .. math::

      &\\frac{d I_j}{d t} = - k_j I_j

      &\\frac{d V}{d t} = ( - (V - V_{rest}) + R\\sum_{j}I_j + RI) / \\tau

      &\\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\\infty})

  When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

  .. math::

      &I_j \\leftarrow R_j I_j + A_j

      &V \\leftarrow V_{reset}

      &V_{th} \\leftarrow max(V_{th_{reset}}, V_{th})

  Note that :math:`I_j` refers to arbitrary number of internal currents.

  **Neuron Parameters**

  ============= ============== ======== ====================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- --------------------------------------------------------------------
  V_rest        -70.           mV       Resting potential.

  V_reset       -70.           mV       Reset potential after spike.

  V_th_inf      -50.           mV       Target value of threshold potential :math:`V_{th}` updating.

  V_th_reset    -60.           mV       Free parameter, should be larger than :math:`V_{reset}`.

  R             20.            \        Membrane resistance.

  tau           20.            ms       Membrane time constant. Compute by :math:`R * C`.

  a             0.             \        Coefficient describes the dependence of 

                                        :math:`V_{th}` on membrane potential.

  b             0.01           \        Coefficient describes :math:`V_{th}` update.

  k1            0.2            \        Constant pf :math:`I1`.

  k2            0.02           \        Constant of :math:`I2`.

  R1            0.             \        Free parameter. 

                                        Describes dependence of :math:`I_1` reset value on

                                        :math:`I_1` value before spiking.

  R2            1.             \        Free parameter. 

                                        Describes dependence of :math:`I_2` reset value on

                                        :math:`I_2` value before spiking.

  A1            0.             \        Free parameter.

  A2            0.             \        Free parameter.

  noise         0.             \        noise.
  ============= ============== ======== ====================================================================

  **Neuron Variables**    

  An object of neuron class record those variables for each neuron:

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  V                  -70.              Membrane potential.

  input              0.                External and synaptic input current.

  spike              0.                Flag to mark whether the neuron is spiking. 

                                       Can be seen as bool.

  V_th               -50.              Spiking threshold potential.

  I1                 0.                Internal current 1.

  I2                 0.                Internal current 2.

  t_last_spike       -1e7              Last spike time stamp.
  ================== ================= =========================================================

  References
  ----------

  .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
         integrate-and-fire neural model produces diverse spiking
         behaviors." Neural computation 21.3 (2009): 704-718.
  """

  def __init__(self, size, V_rest=-70., V_reset=-70., V_th_inf=-50., V_th_reset=-60.,
               R=20., tau=20., a=0., b=0.01, k1=0.2, k2=0.02, R1=0., R2=1., A1=0.,
               A2=0., update_type='vector', **kwargs):
    super(GIF, self).__init__(size=size, **kwargs)

    # params
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th_inf = V_th_inf
    self.V_th_reset = V_th_reset
    self.R = R
    self.tau = tau
    self.a = a
    self.b = b
    self.k1 = k1
    self.k2 = k2
    self.R1 = R1
    self.R2 = R2
    self.A1 = A1
    self.A2 = A2

    # vars
    self.I1 = bp.math.Variable(bp.math.zeros(self.num))
    self.I2 = bp.math.Variable(bp.math.zeros(self.num))
    self.V = bp.math.Variable(bp.math.ones(self.num) * -70.)
    self.V_th = bp.math.Variable(bp.math.ones(self.num) * -50.)
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
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
  def integral(self, I1, I2, V_th, V, t, Iext):
    dI1dt = - self.k1 * I1
    dI2dt = - self.k2 * I2
    dVthdt = self.a * (V - self.V_rest) - self.b * (V_th - self.V_th_inf)
    dVdt = (- (V - self.V_rest) + self.R * Iext + self.R * I1 + self.R * I2) / self.tau
    return dI1dt, dI2dt, dVthdt, dVdt

  def _loop_update(self, _t, _dt):
    for i in range(self.num):
      I1, I2, V_th, V = self.integral(self.I1[i], self.I2[i], self.V_th[i],
                                      self.V[i], _t, self.input[i], dt=_dt)
      self.spike[i] = self.V_th[i] < V
      if self.spike[i]:
        V = self.V_reset
        I1 = self.R1 * I1 + self.A1
        I2 = self.R2 * I2 + self.A2
        self.t_last_spike[i] = _t
        if V_th < self.V_th_reset:
          V_th = self.V_th_reset
      self.I1[i] = I1
      self.I2[i] = I2
      self.V_th[i] = V_th
      self.V[i] = V
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    I1, I2, V_th, V = self.integral(self.I1, self.I2, self.V_th,
                                    self.V, _t, self.input, dt=_dt)
    spike = self.V_th <= V
    V = bp.math.where(spike, self.V_reset, V)
    I1 = bp.math.where(spike, self.R1 * I1[spike] + self.A1, I1)
    I2 = bp.math.where(spike, self.R2 * I2[spike] + self.A2, I2)
    reset_th = bp.math.logical_and(V_th < self.V_th_reset, spike)
    V_th = bp.math.where(reset_th, self.V_th_reset, V_th)
    self.spike[:] = spike
    self.I1[:] = I1
    self.I2[:] = I2
    self.V_th[:] = V_th
    self.V[:] = V
    self.input[:] = 0.
