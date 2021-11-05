# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from .base import Channel

__all__ = [
  'IDR',
  'IK2',
]


class IDR(Channel):
  r"""The delayed rectifier potassium channel current.

  The potassium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

      \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4
      \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p)
      \\
      \alpha_{p} &=\frac{0.032\left(V-V_{sh}-15\right)}{1-\exp \left(-\left(V-V_{sh}-15\right) / 5\right)}
      \\
      \beta_p &= 0.5 \exp \left(-\left(V-V_{sh}-10\right) / 40\right)
      \end{aligned}

  where :math:`\phi` is a temperature-dependent factor, which is given by
  :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

  **Model Examples**

  - `(Brette, et, al., 2007) COBAHH <../../examples/ei_nets/Brette_2007_COBAHH.ipynb>`_


  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  T : float
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : float
    The shift of the membrane potential to spike.

  References
  ----------
  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  """

  def __init__(self, E=-90., g_max=10., T=36., T_base=3., V_sh=-50., **kwargs):
    super(IDR, self).__init__(**kwargs)

    self.T = T
    self.T_base = T_base
    self.E = E
    self.g_max = g_max
    self.V_sh = V_sh

  def init(self, host, **kwargs):
    super(IDR, self).init(host)
    self.p = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))

  @bp.odeint(method='exponential_euler')
  def integral(self, p, t, V):
    phi = self.T_base ** ((self.T - 36) / 10)
    alpha_p = 0.032 * (V - self.V_sh - 15.) / (1. - bm.exp(-(V - self.V_sh - 15.) / 5.))
    beta_p = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    dpdt = phi * (alpha_p * (1. - p) - beta_p * p)
    return dpdt

  def update(self, _t, _dt, **kwargs):
    self.p[:] = self.integral(self.p, _t, self.host.V, dt=_dt)
    g = self.g_max * self.p ** 4
    self.host.I_ion += g * (self.E - self.host.V)
    self.host.V_linear -= g


class IK2(Channel):
  def __init__(self, E, g_max, **kwargs):
    super(IK2, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max
    self.integral = bp.ode.ExponentialEuler(self.derivative)

  def init(self, host, **kwargs):
    super(IK2, self).init(host)
    self.n = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))

  def derivative(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt

  def update(self, _t, _dt, **kwargs):
    self.n[:] = self.integral(self.n, _t, self.host.V, dt=_dt)
    g = self.g_max * self.n ** 4
    self.host.I_ion += g * (self.E - self.host.V)
    self.host.V_linear -= g
