# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from .base import Channel

__all__ = [
  'INa',
  'INa2',
]


class INa(Channel):
  r"""The sodium current model.

  The sodium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

    \begin{aligned}
    I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q
    \\
    \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p)
    \\
    \alpha_{p} &=\frac{0.32\left(V-V_{sh}-13\right)}{1-\exp \left(-\left(V-V_{sh}-13\right) / 4\right)}
    \\
    \beta_{p} &=\frac{-0.28\left(V-V_{sh}-40\right)}{1-\exp \left(\left(V-V_{sh}-40\right) / 5\right)}
    \\
    \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h)
    \\
    \alpha_q &=0.128 \exp \left(-\left(V-V_{sh}-17\right) / 18\right)
    \\
    \beta_q &= \frac{4}{1+\exp \left(-\left(V-V_{sh}-40\right) / 5\right)}
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

  def __init__(self, E=50., g_max=90., T=36., V_sh=-50., **kwargs):
    super(INa, self).__init__(**kwargs)

    self.T = T
    self.E = E
    self.g_max = g_max
    self.V_sh = V_sh
    self.integral = bp.ode.ExponentialEuler(self.derivative)

  def init(self, host, **kwargs):
    super(INa, self).init(host)
    self.p = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))
    self.q = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))

  def derivative(self, p, q, t, V):
    phi = 3 ** ((self.T - 36) / 10)
    alpha_p = 0.32 * (V - self.V_sh - 13.) / (1. - bm.exp(-(V - self.V_sh - 13.) / 4.))
    beta_p = -0.28 * (V - self.V_sh - 40.) / (1. - bm.exp((V - self.V_sh - 40.) / 5.))
    dpdt = phi * (alpha_p * (1. - p) - beta_p * p)

    alpha_q = 0.128 * bm.exp(-(V - self.V_sh - 17.) / 18.)
    beta_q = 4. / (1. + bm.exp(-(V - self.V_sh - 40.) / 5.))
    dqdt = phi * (alpha_q * (1. - q) - beta_q * q)
    return dpdt, dqdt

  def update(self, _t, _dt, **kwargs):
    self.p[:], self.q[:] = self.integral(self.p, self.q, _t, self.host.V, dt=_dt)
    g = self.g_max * self.p ** 3 * self.q
    self.host.I_ion += g * (self.E - self.host.V)
    self.host.V_linear -= g


class INa2(bp.Channel):
  def __init__(self, E=50., g_max=120., **kwargs):
    super(INa2, self).__init__(**kwargs)

    self.E = E
    self.g_max = g_max
    self.integral = bp.ode.ExponentialEuler(self.derivative)

  def init(self, host, **kwargs):
    super(INa2, self).init(host)
    self.m = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))
    self.h = bp.math.Variable(bp.math.zeros(host.shape, dtype=bp.math.float_))

  def derivative(self, m, h, t, V):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    return dmdt, dhdt

  def update(self, _t, _dt, **kwargs):
    self.m[:], self.h[:] = self.integral(self.m, self.h, _t, self.host.V, dt=_dt)
    g = self.g_max * self.m ** 3 * self.h
    self.host.I_ion += g * (self.E - self.host.V)
    self.host.V_linear -= g
