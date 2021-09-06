# -*- coding: utf-8 -*-

import brainpy as bp

__all__ = [
  'HindmarshRose'
]


class HindmarshRose(bp.NeuGroup):
  """Hindmarsh-Rose neuron model [1]_ [2]_.

  The Hindmarsh–Rose model of neuronal activity
  is aimed to study the spiking-bursting behavior
  of the membrane potential observed in experiments
  made with a single neuron.

  The model has the mathematical form of a system of
  three nonlinear ordinary differential equations on
  the dimensionless dynamical variables :math:`x(t)`,
  :math:`y(t)`, and :math:`z(t)`. They read:

     .. math::
         &\\frac{d V}{d t} = y - a V^3 + b V^2 - z + I

         &\\frac{d y}{d t} = c - d V^2 - y

         &\\frac{d z}{d t} = r (s (V - V_{rest}) - z)

  where :math:`a, b, c, d` model the working of the fast ion channels,
  :math:`I` models the slow ion channels.

  **Neuron Parameters**

  ============= ============== ========= ============================================================
  **Parameter** **Init Value** **Unit**  **Explanation**
  ------------- -------------- --------- ------------------------------------------------------------
  a             1.             \         Model parameter.

                                        Fixed to a value best fit neuron activity.

  b             3.             \         Model parameter.

                                        Allows the model to switch between bursting

                                        and spiking, controls the spiking frequency.

  c             1.             \         Model parameter.

                                        Fixed to a value best fit neuron activity.

  d             5.             \         Model parameter.

                                        Fixed to a value best fit neuron activity.

  r             0.01           \         Model parameter.

                                        Controls slow variable z's variation speed.

                                        Governs spiking frequency when spiking, and affects the

                                        number of spikes per burst when bursting.

  s             4.             \         Model parameter. Governs adaption.

  noise         0.             \         noise.
  ============= ============== ========= ============================================================

  **Neuron State**

  =============== ================= =====================================
  **Member name** **Initial Value** **Explanation**
  --------------- ----------------- -------------------------------------
  V               -1.6              Membrane potential.

  y               -10.              Gating variable.

  z               0.                Gating variable.

  spike           False             Whether generate the spikes.

  input           0.                External and synaptic input current.

  t_last_spike       -1e7               Last spike time stamp.
  =============== ================= =====================================

  References
  ----------

  .. [1] Hindmarsh, James L., and R. M. Rose. "A model of neuronal bursting using
        three coupled first order differential equations." Proceedings of the
        Royal society of London. Series B. Biological sciences 221.1222 (1984):
        87-102.
  .. [2] Storace, Marco, Daniele Linaro, and Enno de Lange. "The Hindmarsh–Rose
        neuron model: bifurcation analysis and piecewise-linear approximations."
        Chaos: An Interdisciplinary Journal of Nonlinear Science 18.3 (2008):
        033128.
  """

  def __init__(self, size, a=1., b=3., c=1., d=5., r=0.01, s=4.,
               V_rest=-1.6, V_th=1.0, update_type='vector', **kwargs):
    super(HindmarshRose, self).__init__(size=size, **kwargs)

    # parameters
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.r = r
    self.s = s
    self.V_th = V_th
    self.V_rest = V_rest

    # variables
    self.z = bp.math.Variable(bp.math.zeros(self.num))
    self.V = bp.math.Variable(bp.math.ones(self.num) * -1.6)
    self.y = bp.math.Variable(bp.math.ones(self.num) * -10.)
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

    # update method
    self.update_type = update_type
    if update_type == 'forloop':
      self.update = self._forloop_update
      self.target_backend = 'numpy'
    elif update_type == 'vector':
      self.update = self._vector_update
      self.target_backend = 'general'
    else:
      raise bp.errors.UnsupportedError(f'Do not support {update_type} method.')

  @bp.odeint
  def integral(self, V, y, z, t, Iext):
    dVdt = y - self.a * V * V * V + self.b * V * V - z + Iext
    dydt = self.c - self.d * V * V - y
    dzdt = self.r * (self.s * (V - self.V_rest) - z)
    return dVdt, dydt, dzdt

  def _forloop_update(self, _t, _dt):
    for i in range(self.num):
      V, self.y[i], self.z[i] = self.integral(self.V[i], self.y[i], self.z[i], _t, self.input[i], dt=_dt)
      spike = bp.math.logical_and(V > self.V_th, V <= self.V_th)
      self.spike[i] = spike
      if spike:
        self.t_last_spike[i] = _t
      self.V[i] = V
      self.input[i] = 0.

  def _vector_update(self, _t, _dt):
    self.V[:], self.y[:], self.z[:] = self.integral(self.V, self.y, self.z, _t, self.input, dt=_dt)
    self.input[:] = 0.
